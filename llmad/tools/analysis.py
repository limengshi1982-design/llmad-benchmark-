"""Blue-team Phase I diagnostic tools — read-only, PMU-scoped.

The defender sees only a configured fraction of buses (see
``llmad.observability.PMU_COVERAGE``). These tools wrap ``ss.dae.ts`` so
the LLM can inspect the recent trajectory *within the PMU footprint*, and
derive a few features that are useful for classifying A1/A2/A3/A4:

    * ``get_pmu_window``          — raw time-series window of PMU signals.
    * ``compute_rocof``           — ROCOF over a window (A2 signature).
    * ``compute_angle_diff_matrix`` — rotor-angle spread (A1 signature).
    * ``detect_dominant_mode``    — dominant FFT mode of omega (A3 signature).

All four are read-only and never mutate ``ss``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Tool
from ..observability import PMU_COVERAGE


# ------------------------------------------------------------------ helpers

def _case_name(env: Any) -> str:
    """Best-effort case-name extraction for PMU lookup.

    GridEnv stores the raw case file path. We map common substrings to the
    PMU_COVERAGE keys; if nothing matches we fall back to exposing *all*
    buses (100% PMU) so the tools remain usable in ad-hoc tests.
    """
    path = (getattr(env, "case_path", "") or "").lower()
    for key in PMU_COVERAGE:
        if key in path:
            return key
    # heuristic fallbacks
    if "39" in path:
        return "ieee39"
    if "kundur" in path:
        return "kundur"
    return ""


def _pmu_bus_set(env: Any) -> set:
    name = _case_name(env)
    if name and name in PMU_COVERAGE:
        return set(PMU_COVERAGE[name]["buses"])
    # No coverage configured -> treat every bus as covered.
    return set(env.ss.Bus.idx.v)


def _gen_bus_map(env: Any) -> dict[Any, Any]:
    """Map generator idx -> terminal bus idx, across GENROU + GENCLS."""
    out: dict[Any, Any] = {}
    ss = env.ss
    for model_name in ("GENROU", "GENCLS"):
        m = getattr(ss, model_name, None)
        if m is None or m.n == 0:
            continue
        for gidx, bidx in zip(m.idx.v, m.bus.v):
            out[gidx] = bidx
    return out


def _ts_arrays(env: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ss = env.ss
    t = np.asarray(ss.dae.ts.t, dtype=float)
    xy = np.asarray(ss.dae.ts.xy, dtype=float)
    names = list(ss.dae.xy_name)
    return t, xy, names


def _window_mask(t: np.ndarray, duration: float, t_now: float) -> np.ndarray:
    t0 = max(0.0, t_now - float(duration))
    return (t >= t0 - 1e-9) & (t <= t_now + 1e-9)


# --------------------------------------------------------- get_pmu_window

def get_pmu_window(
    env: Any,
    duration: float = 5.0,
    signals: list[str] | None = None,
    downsample: int = 1,
) -> dict[str, Any]:
    """Return the most-recent ``duration`` seconds of PMU-visible signals.

    ``signals`` accepts any subset of ``['voltage', 'omega', 'delta']``.
    Only buses in the PMU coverage set (and generators whose terminal bus
    is in that set) are returned, mirroring the blue observer's real view.
    """
    signals = signals or ["voltage", "omega"]
    t, xy, names = _ts_arrays(env)
    if t.size == 0:
        return {"ok": False, "error": "no trajectory yet (advance the sim first)"}
    mask = _window_mask(t, duration, env.t)
    t_w = t[mask]
    xy_w = xy[mask]
    if downsample > 1:
        t_w = t_w[::downsample]
        xy_w = xy_w[::downsample]

    pmu_buses = _pmu_bus_set(env)
    g2b = _gen_bus_map(env)

    out: dict[str, Any] = {
        "t_start": round(float(t_w[0]), 4) if t_w.size else None,
        "t_end": round(float(t_w[-1]), 4) if t_w.size else None,
        "n_samples": int(t_w.size),
        "pmu_buses": sorted(pmu_buses, key=lambda x: str(x)),
        "t": [round(float(x), 4) for x in t_w.tolist()],
    }

    if "voltage" in signals:
        v_series: dict[str, list[float]] = {}
        for i, n in enumerate(names):
            if not n.startswith("v Bus "):
                continue
            try:
                bus_num = int(n.split()[-1])
            except ValueError:
                continue
            if bus_num not in pmu_buses:
                continue
            v_series[str(bus_num)] = [round(float(v), 5) for v in xy_w[:, i].tolist()]
        out["voltage"] = v_series

    if "omega" in signals:
        w_series: dict[str, list[float]] = {}
        for i, n in enumerate(names):
            if not (n.startswith("omega GENROU") or n.startswith("omega GENCLS")):
                continue
            gidx = n.split()[-1]
            try:
                gidx_int = int(gidx)
                gkey = gidx_int
            except ValueError:
                gkey = gidx
            b = g2b.get(gkey, g2b.get(gidx))
            if b not in pmu_buses:
                continue
            w_series[str(gkey)] = [round(float(v), 6) for v in xy_w[:, i].tolist()]
        out["omega"] = w_series

    if "delta" in signals:
        d_series: dict[str, list[float]] = {}
        for i, n in enumerate(names):
            if not (n.startswith("delta GENROU") or n.startswith("delta GENCLS")):
                continue
            gidx = n.split()[-1]
            try:
                gidx_int = int(gidx)
                gkey = gidx_int
            except ValueError:
                gkey = gidx
            b = g2b.get(gkey, g2b.get(gidx))
            if b not in pmu_buses:
                continue
            d_series[str(gkey)] = [round(float(v), 5) for v in xy_w[:, i].tolist()]
        out["delta"] = d_series

    return out


GET_PMU_WINDOW = Tool(
    name="get_pmu_window",
    description=(
        "Return the most-recent N seconds of PMU-visible trajectory "
        "(voltage, omega, delta) — restricted to the buses in the blue "
        "observer's coverage set. Use this as the primary data-pull for "
        "Phase I diagnosis."
    ),
    parameters={
        "type": "object",
        "properties": {
            "duration": {
                "type": "number",
                "default": 5.0,
                "description": "Window length in seconds (back from current t).",
            },
            "signals": {
                "type": "array",
                "items": {"type": "string", "enum": ["voltage", "omega", "delta"]},
                "description": "Which signal families to include.",
            },
            "downsample": {
                "type": "integer",
                "default": 1,
                "description": "Return every k-th sample to cap token usage.",
            },
        },
    },
    fn=get_pmu_window,
    read_only=True,
)


# ----------------------------------------------------------- compute_rocof

def compute_rocof(env: Any, duration: float = 2.0) -> dict[str, Any]:
    """Rate-of-change-of-frequency across PMU-visible generators.

    ROCOF is reported per generator in Hz/s (assuming f_base=60 Hz and the
    standard ANDES omega pu convention). Large |ROCOF| and a coherent sign
    across the fleet is a MadIoT / load-step (A2) signature; persistent
    oscillatory sign-flips hint at A3.
    """
    t, xy, names = _ts_arrays(env)
    if t.size < 3:
        return {"ok": False, "error": "need at least 3 samples in the window"}
    mask = _window_mask(t, duration, env.t)
    t_w = t[mask]
    xy_w = xy[mask]
    if t_w.size < 3:
        return {"ok": False, "error": "window too short for ROCOF"}

    pmu_buses = _pmu_bus_set(env)
    g2b = _gen_bus_map(env)

    rocof: dict[str, float] = {}
    peak: dict[str, float] = {}
    for i, n in enumerate(names):
        if not (n.startswith("omega GENROU") or n.startswith("omega GENCLS")):
            continue
        gidx = n.split()[-1]
        try:
            gkey: Any = int(gidx)
        except ValueError:
            gkey = gidx
        b = g2b.get(gkey, g2b.get(gidx))
        if b not in pmu_buses:
            continue
        w = xy_w[:, i]
        f_hz = 60.0 * w  # pu -> Hz
        df = np.gradient(f_hz, t_w)
        rocof[str(gkey)] = round(float(df[-1]), 4)
        peak[str(gkey)] = round(float(df[np.argmax(np.abs(df))]), 4)

    if not rocof:
        return {"ok": False, "error": "no PMU-covered generators found"}

    vals = np.array(list(peak.values()))
    return {
        "t_window": [round(float(t_w[0]), 4), round(float(t_w[-1]), 4)],
        "rocof_final_hz_per_s": rocof,
        "rocof_peak_hz_per_s": peak,
        "fleet_peak_abs": round(float(np.max(np.abs(vals))), 4),
        "fleet_peak_signed_mean": round(float(np.mean(vals)), 4),
    }


COMPUTE_ROCOF = Tool(
    name="compute_rocof",
    description=(
        "Compute per-generator ROCOF (Hz/s) over the most-recent window, "
        "restricted to PMU-covered machines. Large coherent negative ROCOF "
        "suggests A2 (load step / MadIoT); divergent signs suggest A3 "
        "oscillatory instability."
    ),
    parameters={
        "type": "object",
        "properties": {
            "duration": {
                "type": "number",
                "default": 2.0,
                "description": "Window length in seconds.",
            },
        },
    },
    fn=compute_rocof,
    read_only=True,
)


# ----------------------------------------------- compute_angle_diff_matrix

def compute_angle_diff_matrix(env: Any, top_k: int = 3) -> dict[str, Any]:
    """Rotor-angle spread across PMU-visible generators.

    Returns the ``top_k`` largest pairwise |Δδ| at the current time, plus a
    fleet-wide drift summary relative to the first stored sample. Large and
    growing Δδ_max is the canonical A1 (cascade / separation) signature.
    """
    t, xy, names = _ts_arrays(env)
    if t.size == 0:
        return {"ok": False, "error": "no trajectory yet"}

    pmu_buses = _pmu_bus_set(env)
    g2b = _gen_bus_map(env)

    cols: list[int] = []
    labels: list[str] = []
    for i, n in enumerate(names):
        if not (n.startswith("delta GENROU") or n.startswith("delta GENCLS")):
            continue
        gidx = n.split()[-1]
        try:
            gkey: Any = int(gidx)
        except ValueError:
            gkey = gidx
        b = g2b.get(gkey, g2b.get(gidx))
        if b not in pmu_buses:
            continue
        cols.append(i)
        labels.append(str(gkey))

    if len(cols) < 2:
        return {"ok": False, "error": "need >=2 PMU-covered generators for Δδ"}

    d_now = xy[-1, cols]
    d_init = xy[0, cols]
    drift = d_now - d_init
    # pairwise |Δδ| at t_now
    diff = d_now[:, None] - d_now[None, :]
    adiff = np.abs(diff)
    # take upper triangle, exclude diagonal
    iu = np.triu_indices(len(cols), k=1)
    pair_vals = adiff[iu]
    order = np.argsort(-pair_vals)[:top_k]
    top_pairs = []
    for k in order:
        i_, j_ = iu[0][k], iu[1][k]
        top_pairs.append({
            "pair": [labels[i_], labels[j_]],
            "abs_delta_rad": round(float(pair_vals[k]), 4),
        })

    return {
        "t": round(env.t, 4),
        "n_gens": len(cols),
        "delta_spread_rad": round(float(adiff.max()), 4),
        "delta_drift_max_rad": round(float(np.max(np.abs(drift))), 4),
        "delta_drift_per_gen_rad": {
            lbl: round(float(v), 4) for lbl, v in zip(labels, drift)
        },
        "top_pairs": top_pairs,
    }


COMPUTE_ANGLE_DIFF = Tool(
    name="compute_angle_diff_matrix",
    description=(
        "Current-time pairwise |Δδ| across PMU-covered generators, plus "
        "per-generator drift from the first recorded sample. A large, "
        "growing spread is the hallmark of A1 cascade separation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "top_k": {
                "type": "integer",
                "default": 3,
                "description": "How many largest |Δδ| pairs to return.",
            },
        },
    },
    fn=compute_angle_diff_matrix,
    read_only=True,
)


# -------------------------------------------------------- detect_dominant_mode

def detect_dominant_mode(
    env: Any,
    duration: float = 6.0,
    signal: str = "omega",
) -> dict[str, Any]:
    """FFT-based dominant oscillatory mode across PMU-covered generators.

    Returns the frequency (Hz) of the strongest sub-3 Hz spectral peak
    averaged across PMU-covered generators, plus an energy estimate.
    Useful for flagging A3 (PSS/controller tamper): inter-area modes in
    Kundur typically sit around 0.5–0.8 Hz, local modes 1–2 Hz.
    """
    t, xy, names = _ts_arrays(env)
    if t.size < 16:
        return {"ok": False, "error": "need >=16 samples for FFT"}
    mask = _window_mask(t, duration, env.t)
    t_w = t[mask]
    xy_w = xy[mask]
    if t_w.size < 16:
        return {"ok": False, "error": "window too short for FFT"}

    pmu_buses = _pmu_bus_set(env)
    g2b = _gen_bus_map(env)

    if signal == "omega":
        prefix = ("omega GENROU", "omega GENCLS")
    elif signal == "delta":
        prefix = ("delta GENROU", "delta GENCLS")
    else:
        return {"ok": False, "error": f"signal must be 'omega' or 'delta', got '{signal}'"}

    cols: list[int] = []
    for i, n in enumerate(names):
        if not n.startswith(prefix):
            continue
        gidx = n.split()[-1]
        try:
            gkey: Any = int(gidx)
        except ValueError:
            gkey = gidx
        b = g2b.get(gkey, g2b.get(gidx))
        if b not in pmu_buses:
            continue
        cols.append(i)

    if not cols:
        return {"ok": False, "error": "no PMU-covered signal columns"}

    # uniform resample onto a regular grid (ts.t can be non-uniform)
    n = t_w.size
    t_uniform = np.linspace(t_w[0], t_w[-1], n)
    sig = np.zeros((n, len(cols)))
    for j, c in enumerate(cols):
        sig[:, j] = np.interp(t_uniform, t_w, xy_w[:, c])
    # detrend
    sig = sig - sig.mean(axis=0, keepdims=True)

    dt = float(t_uniform[-1] - t_uniform[0]) / max(1, n - 1)
    if dt <= 0:
        return {"ok": False, "error": "non-positive sample interval"}
    freqs = np.fft.rfftfreq(n, d=dt)
    spec = np.abs(np.fft.rfft(sig, axis=0))
    # average across generators, limit to sub-3 Hz band
    spec_mean = spec.mean(axis=1)
    band = (freqs > 0.05) & (freqs < 3.0)
    if not band.any():
        return {"ok": False, "error": "no in-band frequency bins"}
    band_freqs = freqs[band]
    band_spec = spec_mean[band]
    k = int(np.argmax(band_spec))
    f_peak = float(band_freqs[k])
    energy = float(band_spec[k])
    total = float(band_spec.sum()) + 1e-12
    concentration = energy / total

    return {
        "t_window": [round(float(t_uniform[0]), 4), round(float(t_uniform[-1]), 4)],
        "n_samples": int(n),
        "signal": signal,
        "f_peak_hz": round(f_peak, 4),
        "peak_energy": round(energy, 4),
        "peak_concentration": round(concentration, 4),
    }


DETECT_DOMINANT_MODE = Tool(
    name="detect_dominant_mode",
    description=(
        "FFT-based dominant oscillatory mode (Hz) across PMU-covered "
        "generators. Inter-area modes ~0.5-0.8 Hz and local modes 1-2 Hz "
        "are expected; abnormal energy concentration hints at A3 (PSS / "
        "controller tamper) driving weakly-damped oscillations."
    ),
    parameters={
        "type": "object",
        "properties": {
            "duration": {
                "type": "number",
                "default": 6.0,
                "description": "FFT window length (seconds).",
            },
            "signal": {
                "type": "string",
                "enum": ["omega", "delta"],
                "default": "omega",
                "description": "Which state to spectrally analyse.",
            },
        },
    },
    fn=detect_dominant_mode,
    read_only=True,
)
