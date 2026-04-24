"""Simulation-control tools (advance time, checkpoint/rollback).

These are *not* attack or defense actions — they let the agent control the
simulator clock. A red agent may schedule events, then call
``advance_until`` to let the dynamics unfold.
"""

from __future__ import annotations

from typing import Any

from .base import Tool


def advance_until(env: Any, t_end: float) -> dict[str, Any]:
    if t_end <= env.t:
        return {"ok": False, "error": f"t_end={t_end} must exceed current t={env.t}"}
    env.flush_pending_faults()
    t_before = env.t
    env.advance_until(t_end)
    t_after = env.t
    nonconverge = (float(t_end) - t_after) > 1e-3
    obs = env.observe()

    # Trajectory-wide extrema over [t_before, t_after] — so transient dips
    # during faults surface to the LLM even if the final sample has settled.
    import numpy as np  # local import keeps module top lean
    ss = env.ss
    out = {
        "action": "advance_until",
        "t_before": round(t_before, 4),
        "t_after": round(t_after, 4),
        "t_requested": round(float(t_end), 4),
        "converged": not nonconverge,
    }
    if nonconverge:
        out["ok"] = False
        out["error"] = "tds_nonconverge"
        detail = (
            f"TDS stopped at t={t_after:.4f} before reaching "
            f"t_end={float(t_end):.4f} (gap={float(t_end) - t_after:.4f}s). "
            f"Numerical divergence or CCT exceeded."
        )
        tds_error = getattr(env, "last_tds_error", None)
        if tds_error:
            detail += f" Solver error: {tds_error}"
        out["error_detail"] = detail
    try:
        ts_t = np.asarray(ss.dae.ts.t, dtype=float)
        ts_xy = np.asarray(ss.dae.ts.xy, dtype=float)
        names = list(ss.dae.xy_name)
        mask = (ts_t >= t_before - 1e-9) & (ts_t <= obs.t + 1e-9)
        if mask.any() and ts_xy.size:
            seg = ts_xy[mask]
            omega_cols = [i for i, n in enumerate(names)
                          if n.startswith("omega GENROU") or n.startswith("omega GENCLS")]
            v_cols = [i for i, n in enumerate(names) if n.startswith("v Bus ")]
            if omega_cols:
                w = seg[:, omega_cols]
                out["omega_min"] = round(float(w.min()), 6)
                out["omega_max"] = round(float(w.max()), 6)
            if v_cols:
                v = seg[:, v_cols]
                out["v_min"] = round(float(v.min()), 5)
                out["v_max"] = round(float(v.max()), 5)
    except Exception:
        pass

    # Final-sample values (what the current state is right now).
    omega_vals = list(obs.gen_omega.values())
    if omega_vals:
        out["omega_final_min"] = round(min(omega_vals), 6)
        out["omega_final_max"] = round(max(omega_vals), 6)
    if obs.bus_voltage:
        out["v_final_min"] = round(min(obs.bus_voltage.values()), 5)
        out["v_final_max"] = round(max(obs.bus_voltage.values()), 5)
    return out


ADVANCE_UNTIL = Tool(
    name="advance_until",
    description=(
        "Advance the TDS simulation to `t_end` seconds. Returns trajectory-"
        "wide extrema over [t_before, t_after] — omega_min/omega_max and "
        "v_min/v_max cover the whole segment (so fault transients surface) — "
        "plus omega_final_min/max and v_final_min/max for the final sample. "
        "Returns converged=false and error='tds_nonconverge' if ANDES stops "
        "before the requested t_end. "
        "Any pending scheduled events (e.g., apply_fault) are flushed before "
        "the run."
    ),
    parameters={
        "type": "object",
        "properties": {
            "t_end": {
                "type": "number",
                "description": "Target simulation time in seconds (must exceed current).",
            },
        },
        "required": ["t_end"],
    },
    fn=advance_until,
)
