"""ANDES simulation environment wrapper.

Provides a `GridEnv` class that loads an ANDES case, runs power flow, and
exposes time-domain simulation in *segments* (``advance_until(t)``) so that
an external controller (LLM agent) can observe, intervene, and continue.

Key observation for M1: ANDES ``TDS`` solver stores the final time as
``ss.TDS.config.tf``. Setting ``config.tf`` to a later value and calling
``ss.TDS.run()`` again continues from the current state. This is the
'segmentation' mechanism that makes interactive red/blue loops possible.

Usage (smoke test):
    env = GridEnv("kundur_full.xlsx")
    env.setup()
    env.advance_until(2.0)
    obs = env.observe()
    env.advance_until(5.0)
"""

from __future__ import annotations

# Bootstrap FIRST — caps BLAS threads to 1 before numpy gets imported.
from llmad import _bootstrap  # noqa: F401,E402  side-effect import

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Observation:
    """Snapshot of exposed state at a given time."""

    t: float
    bus_voltage: dict[int, float]       # bus_idx -> |V| (pu)
    bus_angle: dict[int, float]         # bus_idx -> angle (rad)
    gen_omega: dict[Any, float]         # gen idx -> omega (pu)
    gen_delta: dict[Any, float]         # gen idx -> rotor angle (rad)
    line_flow: dict[Any, float]         # line idx -> |P| (pu, optional)

    def summary(self) -> str:
        def _stats(d):
            if not d:
                return "n/a"
            v = np.array(list(d.values()))
            return f"min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}"
        return (
            f"t={self.t:.3f}s | V: {_stats(self.bus_voltage)} | "
            f"omega: {_stats(self.gen_omega)} | delta: {_stats(self.gen_delta)}"
        )


class GridEnv:
    """Thin wrapper around an ANDES ``System`` with segmented TDS execution."""

    #: How many disabled Fault slots to reserve at setup time.
    #:
    #: Design note (why pre-allocation): ANDES' compiled DAE + Jacobian
    #: sparsity pattern is frozen during ``TDS.init``. Adding a ``Fault``
    #: device afterwards leaves its ExtAlgeb addresses unresolved *and*
    #: omits its ``tf``/``tc`` from the TDS switch table, so the callback
    #: never fires (silent no-op on IEEE 39). Attempting to rewire
    #: post-init hits Jacobian triplet errors. The clean fix is to reserve
    #: N inert Fault rows (``u=0``, ``tf=1e9``) before ``TDS.init``, and
    #: have ``apply_fault`` parametrise one of them at simulation time.
    _DEFAULT_FAULT_SLOTS: int = 8

    def __init__(
        self,
        case: str | Path,
        verbose: bool = False,
        fault_slots: int = _DEFAULT_FAULT_SLOTS,
    ) -> None:
        self.case_path = str(case)
        self.verbose = verbose
        self.fault_slots = int(fault_slots)
        self.ss: Any = None  # ANDES System instance
        self._pflow_done = False
        self._tds_initialized = False
        self.pending_faults: list[dict[str, Any]] = []
        self.last_tds_error: str | None = None
        # Idx strings of the reserved Fault rows (ordered, consumed in turn).
        self._fault_slot_idx: list[str] = []
        self._next_slot: int = 0

    # -------------------------------------------------------- fault queueing
    def queue_fault(self, req: dict[str, Any]) -> None:
        """Queue a fault to be flushed to ``ss.Fault`` before the next run."""
        self.pending_faults.append(dict(req))

    def flush_pending_faults(self) -> None:
        """Parametrise reserved ``Fault`` slots from queued requests.

        For each queued request, grab the next idle pre-allocated slot,
        assign its bus/tf/tc, flip ``u`` from 0 to 1, and refresh the TDS
        switch-time table so the tf/tc callbacks fire at the right time.
        Raises ``RuntimeError`` if the slot pool is exhausted — at which
        point the caller should either raise ``fault_slots`` in the
        constructor or drop the extra request.
        """
        if not self.pending_faults:
            return
        ss = self.ss
        for req in self.pending_faults:
            if self._next_slot >= len(self._fault_slot_idx):
                raise RuntimeError(
                    f"Fault slot pool exhausted ({self.fault_slots} slots); "
                    f"raise GridEnv(fault_slots=...) to accommodate more "
                    f"scheduled faults."
                )
            slot = self._fault_slot_idx[self._next_slot]
            self._next_slot += 1
            # Parametrise the reserved slot.
            ss.Fault.set("bus", slot, value=req["bus_idx"], base="device")
            ss.Fault.set("tf",  slot, value=float(req["tf"]), base="device")
            ss.Fault.set("tc",  slot, value=float(req["tc"]), base="device")
            ss.Fault.set("xf",  slot, value=float(req.get("xf", 1e-4)),
                         base="device")
            ss.Fault.set("rf",  slot, value=float(req.get("rf", 0.0)),
                         base="device")
            ss.Fault.set_status(slot, 1)
        # Rebuild the switch-time table from scratch so the newly-real
        # tf/tc values enter (the 1e9 dummies remain for untouched slots
        # but sit safely beyond any realistic horizon).
        ss.switch_dict.clear()
        ss.switch_times = np.array([])
        ss.n_switches = 0
        ss.store_switch_times(ss.exist.pflow_tds)
        self.pending_faults.clear()

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        """Load case, pre-allocate Fault slots, run PFlow, init TDS."""
        import andes

        if not self.verbose:
            andes.config_logger(stream_level=40)  # 40 == logging.ERROR
        # Load WITHOUT setup so we can inject reserved Fault rows first.
        self.ss = andes.load(self.case_path, setup=False, no_output=True)
        self._preallocate_fault_slots()
        self.ss.setup()
        self.ss.PFlow.run()
        self._pflow_done = True
        # Initialize TDS (does not run yet); tf controls segment end.
        self.ss.TDS.init()
        self._tds_initialized = True
        if self.verbose:
            print(f"[GridEnv] loaded {self.case_path}; "
                  f"PF converged={self.ss.PFlow.converged}; "
                  f"Fault slots reserved={len(self._fault_slot_idx)}")

    def _preallocate_fault_slots(self) -> None:
        """Reserve ``self.fault_slots`` disabled Fault rows with dummy
        tf/tc far beyond any realistic horizon. Must run *before*
        ``ss.setup()`` so the reserved devices participate in address
        assignment and Jacobian pattern generation.
        """
        if self.fault_slots <= 0:
            return
        ss = self.ss
        # Any existing Bus idx works — the bus will be overwritten when a
        # slot is used. Pick the first.
        default_bus = ss.Bus.idx.v[0]
        for i in range(self.fault_slots):
            idx = f"_llmad_fault_slot_{i}"
            ss.Fault.add(
                idx=idx,
                bus=default_bus,
                tf=1e9 + float(i),
                tc=1e9 + float(i) + 0.1,
                u=0,
                xf=1e-4,
                rf=0.0,
            )
            self._fault_slot_idx.append(idx)

    # -------------------------------------------------------------- stepping
    def advance_until(self, t_end: float) -> float:
        """Run TDS from current sim time to ``t_end`` (seconds)."""
        if not self._tds_initialized:
            raise RuntimeError("Call setup() first.")
        self.ss.TDS.config.tf = float(t_end)
        self.last_tds_error = None
        try:
            self.ss.TDS.run()
        except Exception as exc:
            self.last_tds_error = f"{type(exc).__name__}: {exc}"
        return float(self.ss.dae.t)

    @property
    def t(self) -> float:
        """Current simulation time."""
        try:
            return float(self.ss.dae.t)
        except Exception:
            return 0.0

    # ------------------------------------------------------------ observation
    def observe(self, buses: list[int] | None = None) -> Observation:
        """Return a structured snapshot of key states at current time."""
        ss = self.ss
        bus_idx_all = list(ss.Bus.idx.v)
        sel = buses if buses is not None else bus_idx_all

        # Voltages and angles
        v_map: dict[int, float] = {}
        a_map: dict[int, float] = {}
        for b in sel:
            try:
                v_map[b] = float(ss.Bus.v.v[ss.Bus.idx2uid(b)])
                a_map[b] = float(ss.Bus.a.v[ss.Bus.idx2uid(b)])
            except Exception:
                pass

        # Generator rotor states (prefer GENROU, fallback GENCLS)
        omega_map: dict[Any, float] = {}
        delta_map: dict[Any, float] = {}
        for model_name in ("GENROU", "GENCLS"):
            model = getattr(ss, model_name, None)
            if model is None or model.n == 0:
                continue
            for i, idx in enumerate(model.idx.v):
                try:
                    omega_map[idx] = float(model.omega.v[i])
                    delta_map[idx] = float(model.delta.v[i])
                except Exception:
                    pass

        # Line flows (optional)
        line_map: dict[Any, float] = {}
        if hasattr(ss, "Line") and ss.Line.n > 0:
            try:
                for i, idx in enumerate(ss.Line.idx.v):
                    line_map[idx] = float(abs(ss.Line.a1.v[i] - ss.Line.a2.v[i]))
            except Exception:
                pass

        return Observation(
            t=self.t,
            bus_voltage=v_map,
            bus_angle=a_map,
            gen_omega=omega_map,
            gen_delta=delta_map,
            line_flow=line_map,
        )

    # --------------------------------------------------------------- utility
    def list_devices(self, model_name: str) -> list[Any]:
        """Return idx list for a given ANDES model (e.g., 'Line', 'GENROU')."""
        model = getattr(self.ss, model_name, None)
        if model is None:
            raise ValueError(f"Unknown model '{model_name}'.")
        return list(model.idx.v)
