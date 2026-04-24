"""Default tool registries.

* ``build_red_registry()`` — tools available to the red-team (attacker) agent.
* ``build_blue_registry()`` — tools available to the blue-team (defender)
  agent. Composed of (i) PMU-scoped observation/diagnostic tools used in
  Phase I, and (ii) mitigation tools used in Phase II.
* ``build_observer_registry()`` — minimal read-only subset for sanity tests
  and the oracle-diagnosis ablation.
"""

from .base import Registry, Tool, jsonify
from .observation import GET_MEASUREMENTS, LIST_DEVICES
from .attack import APPLY_FAULT, SCALE_LOAD, TRIP_GEN, TRIP_LINE
from .runtime import ADVANCE_UNTIL
from .analysis import (
    COMPUTE_ANGLE_DIFF,
    COMPUTE_ROCOF,
    DETECT_DOMINANT_MODE,
    GET_PMU_WINDOW,
)
from .defense import RECLOSE_LINE, SHED_LOAD, TRIP_GEN_DEFENSE

__all__ = [
    "Registry",
    "Tool",
    "jsonify",
    "build_red_registry",
    "build_blue_registry",
    "build_observer_registry",
]


def build_red_registry() -> Registry:
    """Tools available to the red-team (attacker) agent."""
    reg = Registry()
    reg.extend(
        [
            LIST_DEVICES,
            GET_MEASUREMENTS,
            TRIP_LINE,
            TRIP_GEN,
            SCALE_LOAD,
            APPLY_FAULT,
            ADVANCE_UNTIL,
        ]
    )
    return reg


def build_blue_registry() -> Registry:
    """Tools available to the blue-team (defender) agent.

    Phase I (diagnosis) tools are read-only and PMU-scoped; Phase II
    (mitigation) tools mutate grid state but are limited to safe operator
    actions (load shedding, line reclosing, tripping of a generator the
    defender believes is compromised).
    """
    reg = Registry()
    reg.extend(
        [
            LIST_DEVICES,
            GET_PMU_WINDOW,
            COMPUTE_ROCOF,
            COMPUTE_ANGLE_DIFF,
            DETECT_DOMINANT_MODE,
            SHED_LOAD,
            RECLOSE_LINE,
            TRIP_GEN_DEFENSE,
            ADVANCE_UNTIL,
        ]
    )
    return reg


def build_observer_registry() -> Registry:
    """Read-only subset — useful for diagnostic-only agents."""
    reg = Registry()
    reg.extend([LIST_DEVICES, GET_MEASUREMENTS])
    return reg
