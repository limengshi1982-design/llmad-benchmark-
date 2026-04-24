"""Read-only observation tools exposed to the LLM agents."""

from __future__ import annotations

from typing import Any

from .base import Tool


# --------------------------------------------------------------------- fn's

def list_devices(env: Any, model: str) -> dict[str, Any]:
    """Return the idx list of devices for a given ANDES model.

    Typical models: 'Line', 'Bus', 'PQ', 'PV', 'Slack', 'GENROU', 'GENCLS'.
    """
    ss = env.ss
    m = getattr(ss, model, None)
    if m is None:
        return {"ok": False, "error": f"unknown model '{model}'"}
    return {
        "model": model,
        "count": int(m.n),
        "idx": list(m.idx.v),
    }


def get_measurements(
    env: Any,
    signals: list[str] | None = None,
    buses: list | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Return a snapshot of current measurements for the requested signals.

    Signals: any subset of ``['voltage', 'angle', 'omega', 'delta', 'line_flow']``.
    When ``buses`` is None, all buses are included (truncated to ``limit``).
    """
    signals = signals or ["voltage", "omega", "delta"]
    obs = env.observe(buses=buses)

    out: dict[str, Any] = {"t": round(obs.t, 4)}
    if "voltage" in signals:
        out["voltage"] = {str(k): round(v, 5) for k, v in list(obs.bus_voltage.items())[:limit]}
    if "angle" in signals:
        out["angle_rad"] = {str(k): round(v, 5) for k, v in list(obs.bus_angle.items())[:limit]}
    if "omega" in signals:
        out["omega_pu"] = {str(k): round(v, 6) for k, v in obs.gen_omega.items()}
    if "delta" in signals:
        out["delta_rad"] = {str(k): round(v, 5) for k, v in obs.gen_delta.items()}
    if "line_flow" in signals:
        out["line_flow_pu"] = {str(k): round(v, 5) for k, v in list(obs.line_flow.items())[:limit]}
    return out


# ------------------------------------------------------------- tool specs

LIST_DEVICES = Tool(
    name="list_devices",
    description=(
        "List device idx for an ANDES model. Use this before issuing an "
        "action to discover legal device identifiers. Typical models: "
        "'Line', 'Bus', 'PQ', 'GENROU'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": "ANDES model name (e.g., 'Line', 'PQ', 'GENROU').",
            },
        },
        "required": ["model"],
    },
    fn=list_devices,
    read_only=True,
)

GET_MEASUREMENTS = Tool(
    name="get_measurements",
    description=(
        "Return the current simulation time and measurement snapshot for "
        "the requested signals. Use this to decide attacks or diagnose "
        "defender-visible state."
    ),
    parameters={
        "type": "object",
        "properties": {
            "signals": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["voltage", "angle", "omega", "delta", "line_flow"],
                },
                "description": "Which signal families to include.",
            },
            "buses": {
                "type": "array",
                "items": {"type": ["string", "integer"]},
                "description": "Optional bus idx subset. If omitted, all buses up to `limit`.",
            },
            "limit": {
                "type": "integer",
                "default": 20,
                "description": "Maximum buses/lines reported when `buses` is omitted.",
            },
        },
    },
    fn=get_measurements,
    read_only=True,
)
