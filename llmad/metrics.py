"""Stability metrics — post-run analysis of a GridEnv's time-series.

The red-team driver needs a *single* verdict per run: did the grid go
unstable within the horizon, and if so, when? This module reads
``ss.dae.ts`` (the full TDS history buffer) and computes the three
thresholds used in the red-team prompt:

    * rotor-speed deviation :  max |ω - 1| > ω_tol        (default 0.02 pu)
    * low-voltage event     :  any bus V  < v_floor        (default 0.85 pu)
    * angle drift           :  max_i |δ_i(t) - δ_i(0)| > δ_drift
                               (default 1.0 rad ≈ 57°)

The δ criterion uses deviation from the **initial** steady-state rotor
angle of each machine, not the absolute δ-spread across machines. In
IEEE 39 at rest the 10-machine spread is already ~1.05 rad, so a flat
absolute threshold would register as "failed" at t=0. Drift-from-initial
is zero at t=0 by construction and grows only when machines slip out of
synchronism — the physically meaningful instability signal.

``t_fail`` is the earliest time any threshold is crossed; if none are,
``t_fail`` is ``None`` and ``failed`` is ``False``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


# ---------------------------------------------------------------- thresholds

@dataclass(frozen=True)
class Thresholds:
    omega_tol: float = 0.02     # |ω - 1| pu
    v_floor: float = 0.85       # pu
    delta_drift: float = 1.0    # rad, max_i |δ_i(t) - δ_i(0)|


# ------------------------------------------------------------------- report

@dataclass
class StabilityReport:
    t_start: float
    t_end: float
    n_samples: int

    omega_min: float
    omega_max: float
    omega_deviation_max: float   # max |ω - 1| over the run
    v_min: float
    v_max: float
    delta_drift_max: float       # max over (t, machine) of |δ_i(t) - δ_i(0)|
    delta_spread_max: float      # max over time of (max_i δ_i - min_i δ_i)

    # First-crossing times for each threshold (None if not crossed).
    t_omega_fail: float | None
    t_v_fail: float | None
    t_delta_fail: float | None

    # The earliest of the three (the headline metric).
    t_fail: float | None
    failed: bool
    failure_reason: str | None   # "omega" | "voltage" | "delta" | None

    thresholds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Replace None with a JSON-friendly null-equivalent only if caller
        # wants that; dataclass asdict already emits None -> None which
        # json.dumps renders as null. So no transform needed.
        return d

    def summary(self) -> str:
        tail = f"t_fail={self.t_fail:.3f}s ({self.failure_reason})" if self.failed \
            else "stable"
        return (
            f"[{self.t_start:.1f}->{self.t_end:.1f}s n={self.n_samples}] "
            f"|dw|_max={self.omega_deviation_max:.4f} "
            f"V_min={self.v_min:.4f} "
            f"d_drift_max={self.delta_drift_max:.4f} rad | {tail}"
        )


# -------------------------------------------------------------- column index

def _find_columns(xy_name: list[str], pattern: str) -> list[int]:
    """Return column indices whose xy_name matches the given regex."""
    rx = re.compile(pattern)
    return [i for i, nm in enumerate(xy_name) if rx.search(nm)]


def _omega_cols(xy_name: list[str]) -> list[int]:
    """Generator rotor-speed columns.

    ANDES labels GENROU speed as ``omega GENROU <idx>`` and GENCLS as
    ``omega GENCLS <idx>``. Pattern intentionally anchors on the leading
    ``omega `` (with trailing space) to avoid matching e.g. ``omega_ref``.
    """
    return _find_columns(xy_name, r"^omega (GENROU|GENCLS) ")


def _delta_cols(xy_name: list[str]) -> list[int]:
    return _find_columns(xy_name, r"^delta (GENROU|GENCLS) ")


def _v_cols(xy_name: list[str]) -> list[int]:
    """Bus voltage magnitude columns (labelled ``v Bus <idx>``)."""
    return _find_columns(xy_name, r"^v Bus ")


# --------------------------------------------------------------- computation

def compute_stability(
    env: Any,
    thresholds: Thresholds | None = None,
    t_window: tuple[float, float] | None = None,
) -> StabilityReport:
    """Scan ``env.ss.dae.ts`` and produce a StabilityReport.

    Requires that the env has been run with a non-empty TDS history.
    If the buffer is empty (no ``advance_until`` call yet) we return a
    zero-sample report marked as not-failed.

    ``t_window`` optionally restricts the scan to ``[t0, t1]`` seconds.
    δ-drift is still measured relative to the first sample *inside the
    window* — so a window covering the end of the horizon measures
    end-state drift, not cumulative drift from t=0. This is what the
    blue-team driver uses to score "recovered at the end" vs. "in-bounds
    for the whole trajectory".
    """
    thr = thresholds or Thresholds()
    ss = env.ss
    ts = ss.dae.ts
    xy_name = list(ss.dae.xy_name)

    t = np.asarray(getattr(ts, "t", []), dtype=float)
    xy = np.asarray(getattr(ts, "xy", np.empty((0, len(xy_name)))), dtype=float)
    if t_window is not None and t.size > 0:
        t0, t1 = t_window
        mask = (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
        t = t[mask]
        xy = xy[mask]
    if t.size == 0 or xy.size == 0:
        return StabilityReport(
            t_start=float(env.t), t_end=float(env.t), n_samples=0,
            omega_min=1.0, omega_max=1.0, omega_deviation_max=0.0,
            v_min=1.0, v_max=1.0, delta_drift_max=0.0, delta_spread_max=0.0,
            t_omega_fail=None, t_v_fail=None, t_delta_fail=None,
            t_fail=None, failed=False, failure_reason=None,
            thresholds={"omega_tol": thr.omega_tol, "v_floor": thr.v_floor,
                        "delta_drift": thr.delta_drift},
        )

    omega_idx = _omega_cols(xy_name)
    delta_idx = _delta_cols(xy_name)
    v_idx = _v_cols(xy_name)

    omega = xy[:, omega_idx] if omega_idx else np.ones((t.size, 1))
    delta = xy[:, delta_idx] if delta_idx else np.zeros((t.size, 1))
    v_mat = xy[:, v_idx]     if v_idx     else np.ones((t.size, 1))

    # Per-sample aggregates.
    omega_dev_t = np.max(np.abs(omega - 1.0), axis=1)       # (T,)
    v_min_t = np.min(v_mat, axis=1)                          # (T,)
    delta_spread_t = np.max(delta, axis=1) - np.min(delta, axis=1)  # (T,)
    # Per-machine drift from initial steady state; invariant to cross-machine
    # offset, zero at t=0 by construction.
    delta0 = delta[0:1, :]                                   # (1, G)
    delta_drift_t = np.max(np.abs(delta - delta0), axis=1)   # (T,)

    # First-crossing helpers.
    def _first_cross(mask: np.ndarray) -> float | None:
        hits = np.nonzero(mask)[0]
        return float(t[hits[0]]) if hits.size else None

    t_omega_fail = _first_cross(omega_dev_t > thr.omega_tol)
    t_v_fail = _first_cross(v_min_t < thr.v_floor)
    t_delta_fail = _first_cross(delta_drift_t > thr.delta_drift)

    # Overall verdict.
    candidates = [
        ("omega", t_omega_fail),
        ("voltage", t_v_fail),
        ("delta", t_delta_fail),
    ]
    active = [(r, tf) for (r, tf) in candidates if tf is not None]
    if active:
        reason, t_fail = min(active, key=lambda x: x[1])
        failed = True
    else:
        reason, t_fail, failed = None, None, False

    return StabilityReport(
        t_start=float(t[0]),
        t_end=float(t[-1]),
        n_samples=int(t.size),
        omega_min=float(np.min(omega)),
        omega_max=float(np.max(omega)),
        omega_deviation_max=float(np.max(omega_dev_t)),
        v_min=float(np.min(v_mat)),
        v_max=float(np.max(v_mat)),
        delta_drift_max=float(np.max(delta_drift_t)),
        delta_spread_max=float(np.max(delta_spread_t)),
        t_omega_fail=t_omega_fail,
        t_v_fail=t_v_fail,
        t_delta_fail=t_delta_fail,
        t_fail=t_fail,
        failed=failed,
        failure_reason=reason,
        thresholds={"omega_tol": thr.omega_tol, "v_floor": thr.v_floor,
                    "delta_drift": thr.delta_drift},
    )
