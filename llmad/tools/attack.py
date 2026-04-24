"""Red-team attack tools — perturbation injection into a live GridEnv.

Design notes
------------
* ``trip_line`` / ``trip_gen`` use ANDES' ``set_status(idx, 0)``. This is the
  *same* mechanism the built-in ``Toggle`` event uses; the device is marked
  offline, its power contribution zeroed on the next step, and dependent
  models propagate accordingly.
* ``scale_load`` mutates ``PQ.p0`` / ``PQ.q0`` via ``set('...', base='device')``
  — i.e. device-base values written to both ``vin`` and ``v``. The next TDS
  step picks them up.
* ``apply_fault`` is **pre-schedule only**: the caller decides the fault
  window ``[t_on, t_off]`` and this tool stores the request; the wrapper
  flushes it to ``ss.Fault`` before the next ``advance_until``. Run-time
  injection of new ``Fault`` devices would require a TDS re-init, which is
  heavier than M2-1 needs.
"""

from __future__ import annotations

from typing import Any

from .base import Tool


# ---------------------------------------------------------------- trip_line

def trip_line(env: Any, line_idx: str) -> dict[str, Any]:
    ss = env.ss
    if line_idx not in set(ss.Line.idx.v):
        return {"ok": False, "error": f"unknown line '{line_idx}'",
                "available": list(ss.Line.idx.v)}
    before = int(ss.Line.u.v[ss.Line.idx2uid(line_idx)])
    ss.Line.set_status(line_idx, 0)
    after = int(ss.Line.u.v[ss.Line.idx2uid(line_idx)])
    return {
        "action": "trip_line",
        "line_idx": line_idx,
        "status_before": before,
        "status_after": after,
        "t": round(env.t, 4),
    }


TRIP_LINE = Tool(
    name="trip_line",
    description="Take a transmission line offline immediately (set its u=0).",
    parameters={
        "type": "object",
        "properties": {
            "line_idx": {
                "type": "string",
                "description": "Line idx (discover via list_devices(model='Line')).",
            },
        },
        "required": ["line_idx"],
    },
    fn=trip_line,
)


# ---------------------------------------------------------------- trip_gen

def trip_gen(env: Any, gen_idx: str, model: str = "GENROU") -> dict[str, Any]:
    ss = env.ss
    m = getattr(ss, model, None)
    if m is None or m.n == 0:
        return {"ok": False, "error": f"model '{model}' not present or empty"}
    if gen_idx not in set(m.idx.v):
        return {"ok": False, "error": f"unknown {model} '{gen_idx}'",
                "available": list(m.idx.v)}
    m.set_status(gen_idx, 0)
    return {
        "action": "trip_gen",
        "model": model,
        "gen_idx": gen_idx,
        "t": round(env.t, 4),
    }


TRIP_GEN = Tool(
    name="trip_gen",
    description="Disconnect a synchronous generator immediately.",
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
    fn=trip_gen,
)


# --------------------------------------------------------------- scale_load

def scale_load(env: Any, load_idx: str, scale: float) -> dict[str, Any]:
    """Multiply the active & reactive load setpoints by `scale` (device base).

    A scale > 1 is a demand spike (MadIoT-style); 0 < scale < 1 is a drop.

    Caveat (M2-1): direct mutation of ``PQ.p0`` after ``TDS.init`` flows
    into the algebraic equations on the next step but does **not**
    recompute the constants of any downstream dynamic-load breakdown (ZIP
    coefficients, FLoad reference). For faithful MadIoT magnitudes
    replace PQ with FLoad/ZIP and bind to ``Alter`` events; tracked for
    M2-2 calibration.
    """
    ss = env.ss
    if scale <= 0 or scale > 10:
        return {"ok": False, "error": f"scale={scale} outside (0, 10]"}
    if load_idx not in set(ss.PQ.idx.v):
        return {"ok": False, "error": f"unknown load '{load_idx}'",
                "available": list(ss.PQ.idx.v)}
    uid = ss.PQ.idx2uid(load_idx)
    p_before = float(ss.PQ.p0.v[uid])
    q_before = float(ss.PQ.q0.v[uid])
    ss.PQ.set("p0", load_idx, value=p_before * scale, base="device")
    ss.PQ.set("q0", load_idx, value=q_before * scale, base="device")
    return {
        "action": "scale_load",
        "load_idx": load_idx,
        "scale": scale,
        "p0_before": p_before,
        "p0_after": float(ss.PQ.p0.v[uid]),
        "q0_before": q_before,
        "q0_after": float(ss.PQ.q0.v[uid]),
        "t": round(env.t, 4),
    }


SCALE_LOAD = Tool(
    name="scale_load",
    description=(
        "Multiplicatively perturb the active/reactive load setpoint of a PQ "
        "load. Use scale>1 for MadIoT-style demand spikes, <1 for drops."
    ),
    parameters={
        "type": "object",
        "properties": {
            "load_idx": {"type": "string", "description": "PQ load idx."},
            "scale": {
                "type": "number",
                "description": "Multiplicative factor; must be in (0, 10].",
            },
        },
        "required": ["load_idx", "scale"],
    },
    fn=scale_load,
)


# --------------------------------------------------------------- apply_fault

def apply_fault(
    env: Any,
    bus_idx,
    t_on: float,
    t_off: float,
    xf: float = 1e-4,
    rf: float = 0.0,
) -> dict[str, Any]:
    """Schedule a bus fault. Must be called *before* ``advance_until(t>=t_on)``.

    This implementation appends a pre-scheduled request to
    ``env.pending_faults`` which the GridEnv flushes to ``ss.Fault`` the next
    time it advances. Done this way because adding a ``Fault`` device after
    ``TDS.init`` requires a reinit that is expensive and error-prone.
    """
    if t_off <= t_on:
        return {"ok": False, "error": "t_off must be > t_on"}
    if t_on < env.t:
        return {"ok": False, "error": f"t_on={t_on} is earlier than current t={env.t}"}
    req = {
        "bus_idx": bus_idx,
        "tf": float(t_on),
        "tc": float(t_off),
        "xf": float(xf),
        "rf": float(rf),
    }
    env.queue_fault(req)
    return {"action": "schedule_fault", **req, "scheduled_at": round(env.t, 4)}


APPLY_FAULT = Tool(
    name="apply_fault",
    description=(
        "Schedule a bus three-phase fault with specified impedance between "
        "t_on and t_off. Must be called before advancing past t_on."
    ),
    parameters={
        "type": "object",
        "properties": {
            "bus_idx": {"type": ["string", "integer"], "description": "Target bus idx."},
            "t_on": {"type": "number", "description": "Fault-on time (seconds)."},
            "t_off": {"type": "number", "description": "Fault-off time (seconds)."},
            "xf": {"type": "number", "default": 1e-4, "description": "Fault reactance (pu)."},
            "rf": {"type": "number", "default": 0.0, "description": "Fault resistance (pu)."},
        },
        "required": ["bus_idx", "t_on", "t_off"],
    },
    fn=apply_fault,
)
