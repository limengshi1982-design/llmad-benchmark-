"""Blue-team recovery tools — state-changing mitigations.

These are used by the defender agent during Phase II to restore the grid
after an attack. Signatures mirror red-team style (take ``env`` and
named args, return dict).

Coverage in M2:
    * ``shed_load``        — fractional PQ reduction (UFLS proxy)
    * ``reclose_line``     — bring a previously-tripped line back online
    * ``trip_gen_defense`` — trip a compromised synchronous generator
                             (blue-side alias; semantics identical to
                             attack.TRIP_GEN but lives in the blue
                             registry for clarity)

Deferred to M3:
    * ``redispatch``       — governor/ref mutation (A4 AGC-FDI scenario)
    * ``retune_pss``       — PSS gain restore (A3 controller tamper)
"""

from __future__ import annotations

from typing import Any

from .base import Tool
from .attack import trip_gen as _trip_gen


# --------------------------------------------------------------- shed_load
def shed_load(env: Any, load_idx: str, fraction: float) -> dict[str, Any]:
    """Proportionally reduce active & reactive demand on a PQ load.

    ``fraction=0.3`` sheds 30% of the load setpoint (new = old * 0.7).
    Implemented with the same device-base PQ.set hook used by the red
    scale_load tool — runtime-effective after the next TDS step.
    """
    ss = env.ss
    if not (0.0 < fraction <= 1.0):
        return {"ok": False, "error": f"fraction={fraction} outside (0, 1]"}
    if load_idx not in set(ss.PQ.idx.v):
        return {"ok": False, "error": f"unknown load '{load_idx}'",
                "available": list(ss.PQ.idx.v)}
    uid = ss.PQ.idx2uid(load_idx)
    p_before = float(ss.PQ.p0.v[uid])
    q_before = float(ss.PQ.q0.v[uid])
    keep = 1.0 - fraction
    ss.PQ.set("p0", load_idx, value=p_before * keep, base="device")
    ss.PQ.set("q0", load_idx, value=q_before * keep, base="device")
    return {
        "action": "shed_load",
        "load_idx": load_idx,
        "fraction_shed": fraction,
        "p0_before": p_before,
        "p0_after": float(ss.PQ.p0.v[uid]),
        "delta_p_pu": p_before * fraction,
        "t": round(env.t, 4),
    }


SHED_LOAD = Tool(
    name="shed_load",
    description=(
        "Shed a fraction of a PQ load (both P and Q) by multiplying its "
        "setpoint by (1 - fraction). fraction=0.3 means shed 30%. "
        "Use for under-frequency load shedding (UFLS) after A1/A2 attacks."
    ),
    parameters={
        "type": "object",
        "properties": {
            "load_idx": {"type": "string", "description": "PQ load idx."},
            "fraction": {
                "type": "number",
                "description": "Shed fraction in (0, 1]; e.g. 0.3 = 30% shed.",
            },
        },
        "required": ["load_idx", "fraction"],
    },
    fn=shed_load,
)


# ------------------------------------------------------------ reclose_line
def reclose_line(env: Any, line_idx: str) -> dict[str, Any]:
    """Bring a previously-offline line back online (set u=1)."""
    ss = env.ss
    if line_idx not in set(ss.Line.idx.v):
        return {"ok": False, "error": f"unknown line '{line_idx}'",
                "available": list(ss.Line.idx.v)}
    before = int(ss.Line.u.v[ss.Line.idx2uid(line_idx)])
    ss.Line.set_status(line_idx, 1)
    after = int(ss.Line.u.v[ss.Line.idx2uid(line_idx)])
    return {
        "action": "reclose_line",
        "line_idx": line_idx,
        "status_before": before,
        "status_after": after,
        "t": round(env.t, 4),
    }


RECLOSE_LINE = Tool(
    name="reclose_line",
    description=(
        "Reclose a transmission line that was previously tripped (set u=1). "
        "Does nothing if the line is already online."
    ),
    parameters={
        "type": "object",
        "properties": {
            "line_idx": {"type": "string", "description": "Line idx to reclose."},
        },
        "required": ["line_idx"],
    },
    fn=reclose_line,
)


# -------------------------------------------------------- trip_gen_defense
TRIP_GEN_DEFENSE = Tool(
    name="trip_gen_defense",
    description=(
        "Trip (disconnect) a synchronous generator that is suspected to be "
        "compromised — e.g. its controller has been tampered with (A3) or "
        "it has become an AGC-FDI injection point (A4). Trips GENROU/GENCLS "
        "by setting u=0."
    ),
    parameters={
        "type": "object",
        "properties": {
            "gen_idx": {"type": "string", "description": "Generator idx."},
            "model": {
                "type": "string",
                "default": "GENROU",
                "enum": ["GENROU", "GENCLS"],
            },
        },
        "required": ["gen_idx"],
    },
    fn=_trip_gen,
)
