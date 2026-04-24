"""Scripted attack scenarios — ground-truth sequences for blue-team eval.

Phase I of the blue agent is evaluated against *known* attacks so we can
score diagnosis accuracy. These scenarios are deterministic Python
functions that use the same tool layer as the red agent would — so the
post-attack trajectory is identical to what a successful LLM attacker
would produce.

Each scenario returns a ``ScenarioResult`` with:
    * ``label``     — canonical class name ('A1', 'A2', 'A3', 'A4')
    * ``case``      — catalog key ('kundur', 'ieee39')
    * ``actions``   — list of dicts (what was actually done, in order)
    * ``t_attack``  — first-injection timestamp (for reporting)
    * ``params``    — echo of the parameters the scenario used

Design (D5): hardcoded programmatic functions first. A later pass may
replace these with the red-LLM output for a fully-LLM-vs-LLM benchmark,
but that introduces noise we don't want in the diagnosis metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .tools.attack import apply_fault, scale_load, trip_line


# --------------------------------------------------------------- data model

@dataclass
class ScenarioResult:
    label: str
    case: str
    t_attack: float
    actions: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------- device pickers

def _pick_line(env: Any, preferred: list[str] | None = None) -> str:
    """Return the first preferred line idx that exists, else the first line."""
    lines = list(env.ss.Line.idx.v)
    if preferred:
        for p in preferred:
            if p in lines:
                return p
    return lines[0]


def _pick_load(env: Any, preferred: list[str] | None = None) -> str:
    loads = list(env.ss.PQ.idx.v)
    if preferred:
        for p in preferred:
            if p in loads:
                return p
    # pick the largest load — most impactful for A2
    largest = None
    best = -1.0
    for uid, idx in enumerate(loads):
        p0 = float(env.ss.PQ.p0.v[uid])
        if p0 > best:
            best = p0
            largest = idx
    return largest or loads[0]


def _pick_bus(env: Any, preferred: list[int] | None = None) -> Any:
    buses = list(env.ss.Bus.idx.v)
    if preferred:
        for b in preferred:
            if b in buses:
                return b
    return buses[0]


# -------------------------------------------------------------- A1 cascade

def scripted_kundur_a1(env: Any, t_attack: float = 2.0) -> ScenarioResult:
    """Kundur A1 (severe) — trip one of the 8-9 tie lines between areas.

    The bus-8-to-bus-9 corridor has only two parallel circuits (Line_7,
    Line_8); tripping one forces the survivor to carry ~400 MW of
    inter-area flow that exceeds its rating, driving angle separation
    and (at default loading) complete loss of synchronism. Use the
    ``kundur_a1_mild`` variant for a recoverable A1 benchmark.
    """
    line = _pick_line(env, preferred=["Line_7", "Line_8"])
    res = trip_line(env, line_idx=line)
    return ScenarioResult(
        label="A1",
        case="kundur",
        t_attack=float(env.t),
        actions=[res],
        params={"target_t": t_attack, "line_idx": line, "severity": "severe"},
    )


def scripted_kundur_a1_mild(env: Any, t_attack: float = 2.0) -> ScenarioResult:
    """Kundur A1 (mild) — trip one of the three 7-8 tie lines.

    The bus-7-to-bus-8 corridor has three parallel circuits (Line_4,
    Line_5, Line_6); tripping one leaves 2/3 of the tie capacity and
    redistributes ~130 MW onto each surviving line — below their ratings.
    A competent defender can stabilise with modest load shedding or no
    action. Semantically still an A1 cascade (transmission-line trip).
    """
    line = _pick_line(env, preferred=["Line_4", "Line_5", "Line_6"])
    res = trip_line(env, line_idx=line)
    return ScenarioResult(
        label="A1",
        case="kundur",
        t_attack=float(env.t),
        actions=[res],
        params={"target_t": t_attack, "line_idx": line, "severity": "mild"},
    )


def scripted_ieee39_a1(env: Any, t_attack: float = 2.0) -> ScenarioResult:
    """IEEE 39 A1 (baseline) — trip a peripheral transmission line.

    Line_26 (17-27) connects a load-pocket to the central network.
    Tripping it causes a localised flow redistribution that IEEE 39
    absorbs without loss of synchronism — useful as a "near-nominal"
    A1 example. For a stressing A1 use ``ieee39_a1_severe``.
    """
    line = _pick_line(env, preferred=["Line_26", "Line_29", "Line_34"])
    res = trip_line(env, line_idx=line)
    return ScenarioResult(
        label="A1",
        case="ieee39",
        t_attack=float(env.t),
        actions=[res],
        params={"target_t": t_attack, "line_idx": line, "severity": "light"},
    )


def scripted_ieee39_a1_severe(env: Any, t_attack: float = 2.0) -> ScenarioResult:
    """IEEE 39 A1 (severe) — trip a central-corridor transmission line.

    Line_28 (22-23) sits on the main east-west transfer path between
    the eastern generator cluster (gens 6,7) and the load centres.
    Tripping it causes substantial angle swing across the corridor
    with partial voltage dips — a meaningful recovery test.
    """
    line = _pick_line(env, preferred=["Line_28", "Line_21", "Line_23"])
    res = trip_line(env, line_idx=line)
    return ScenarioResult(
        label="A1",
        case="ieee39",
        t_attack=float(env.t),
        actions=[res],
        params={"target_t": t_attack, "line_idx": line, "severity": "severe"},
    )


# ---------------------------------------------------------------- A2 load

def scripted_kundur_a2(
    env: Any,
    t_attack: float = 2.0,
    scale: float = 1.8,
) -> ScenarioResult:
    """Kundur A2 — MadIoT-style step increase on the largest load.

    A sudden ~80% jump in demand drives a coherent frequency sag across
    all four machines; the signature is a large, coherent-sign ROCOF.
    """
    load = _pick_load(env, preferred=["PQ_1", "PQ_2"])
    res = scale_load(env, load_idx=load, scale=scale)
    return ScenarioResult(
        label="A2",
        case="kundur",
        t_attack=float(env.t),
        actions=[res],
        params={"target_t": t_attack, "load_idx": load, "scale": scale},
    )


def scripted_ieee39_a2(
    env: Any,
    t_attack: float = 2.0,
    scale: float = 1.6,
) -> ScenarioResult:
    """IEEE 39 A2 — MadIoT step on a large load."""
    load = _pick_load(env, preferred=["PQ_8", "PQ_3", "PQ_15"])
    res = scale_load(env, load_idx=load, scale=scale)
    return ScenarioResult(
        label="A2",
        case="ieee39",
        t_attack=float(env.t),
        actions=[res],
        params={"target_t": t_attack, "load_idx": load, "scale": scale},
    )


# --------------------------------------- ambient primary-fault scenarios
#
# These are *natural* disturbances (bolted bus fault, load step) used as the
# primary event in the piggyback threat model (see
# ``docs/piggyback_threat_model_spec.md``). They return ``label="ambient"``
# so that the run orchestrator does not confuse them with attacker actions;
# the ``truth_label`` in a piggyback episode is still inferred from the red
# agent's tool trace.
#
# Convention: bus-fault primaries are *schedulable* — they queue a Fault
# event at ``t_on`` (kwarg; strictly in the future relative to ``env.t``)
# and do not advance the simulation themselves. The load-step primary is
# an immediate PQ mutation and therefore requires ``env.t == t_fault`` on
# entry (the caller must pre-advance). In both cases the caller
# subsequently calls ``advance_until(t_a)`` to propagate the disturbance
# through red's observation window.
#
# NB: ANDES refuses to integrate a switch that coincides with the current
# time ("Current step size is zero"), so bus-fault primaries enforce
# ``t_on > env.t`` by inserting a small 1e-3 s buffer when ``t_on`` is not
# given explicitly.

_FAULT_SCHEDULING_BUFFER_S = 1e-3


def _resolve_fault_t_on(env: Any, t_on: float | None) -> float:
    """Return a ``t_on`` that is strictly greater than ``env.t``.

    When the caller provides an explicit ``t_on`` we honour it (with a
    tiny buffer if it would collide with ``env.t``); when omitted, we
    default to ``env.t + 1e-3``.
    """
    now = float(env.t)
    if t_on is None:
        return now + _FAULT_SCHEDULING_BUFFER_S
    if float(t_on) <= now:
        return now + _FAULT_SCHEDULING_BUFFER_S
    return float(t_on)


def primary_ieee39_bf_b16_80ms(
    env: Any, duration: float = 0.08, t_on: float | None = None,
) -> ScenarioResult:
    """IEEE 39 primary — 80 ms bolted 3-phase fault at bus 16.

    Bus 16 is a mid-system load bus between the area ties. The voltage
    sag propagates across Lines 15/16/17 and is visible at the 50% PMU
    subset without destabilising the system on its own — the viable
    "piggyback candidate" bucket (network-adjacent fault).
    """
    bus = _pick_bus(env, preferred=[16])
    t_on = _resolve_fault_t_on(env, t_on)
    t_off = t_on + duration
    res = apply_fault(env, bus_idx=bus, t_on=t_on, t_off=t_off)
    return ScenarioResult(
        label="ambient",
        case="ieee39",
        t_attack=t_on,
        actions=[res],
        params={"bus_idx": bus, "t_on": t_on, "t_off": t_off,
                "duration_s": duration, "class": "bus_fault"},
    )


def primary_ieee39_bf_b22_80ms(
    env: Any, duration: float = 0.08, t_on: float | None = None,
) -> ScenarioResult:
    """IEEE 39 primary — 80 ms bolted fault at bus 22 (near gen 35).

    Closer to a generator than bus 16; exercises stator flux decay + AVR
    response. Expect a visible gen-35 ω excursion and local voltage dip
    but recoverable via primary-damping controls.
    """
    bus = _pick_bus(env, preferred=[22])
    t_on = _resolve_fault_t_on(env, t_on)
    t_off = t_on + duration
    res = apply_fault(env, bus_idx=bus, t_on=t_on, t_off=t_off)
    return ScenarioResult(
        label="ambient",
        case="ieee39",
        t_attack=t_on,
        actions=[res],
        params={"bus_idx": bus, "t_on": t_on, "t_off": t_off,
                "duration_s": duration, "class": "bus_fault"},
    )


def primary_ieee39_bf_b29_100ms(
    env: Any, duration: float = 0.10, t_on: float | None = None,
) -> ScenarioResult:
    """IEEE 39 primary — 100 ms bolted fault at bus 29 (peripheral load).

    Bus 29 is electrically further from the central network so the
    fault needs longer duration (100 ms) to produce a PMU-visible
    transient. Reaches gens 38/39 via Line 34.
    """
    bus = _pick_bus(env, preferred=[29])
    t_on = _resolve_fault_t_on(env, t_on)
    t_off = t_on + duration
    res = apply_fault(env, bus_idx=bus, t_on=t_on, t_off=t_off)
    return ScenarioResult(
        label="ambient",
        case="ieee39",
        t_attack=t_on,
        actions=[res],
        params={"bus_idx": bus, "t_on": t_on, "t_off": t_off,
                "duration_s": duration, "class": "bus_fault"},
    )


def primary_ieee39_bf_b02_60ms(
    env: Any, duration: float = 0.06, t_on: float | None = None,
) -> ScenarioResult:
    """IEEE 39 primary — short 60 ms fault at bus 2 (near swing gen 30).

    Tight clearance near the swing machine. Stresses the AVR + governor
    of gen 30. May be too mild to be perceptible — the screening sweep
    will confirm.
    """
    bus = _pick_bus(env, preferred=[2])
    t_on = _resolve_fault_t_on(env, t_on)
    t_off = t_on + duration
    res = apply_fault(env, bus_idx=bus, t_on=t_on, t_off=t_off)
    return ScenarioResult(
        label="ambient",
        case="ieee39",
        t_attack=t_on,
        actions=[res],
        params={"bus_idx": bus, "t_on": t_on, "t_off": t_off,
                "duration_s": duration, "class": "bus_fault"},
    )


# P5 (trip_line + auto-reclose 300 ms later) is skipped in v1 because the
# env only supports pre-scheduled *faults* (``pending_faults``); there is
# no ``pending_line_ops`` queue. Implementing P5 would require adding
# scheduled line-toggle support at the env layer. Bucket 3 is still
# covered by the load-step primary below.


def primary_ieee39_load_step_pq08_plus25(
    env: Any,
    scale: float = 1.25,
    t_on: float | None = None,
) -> ScenarioResult:
    """IEEE 39 primary — permanent +25% step on PQ_8 (large load bus).

    Mimics a large industrial-load pickup. Pulls system frequency down
    and exercises governor response. Unlike the bolted faults, this is
    *permanent* — the post-transient steady state differs from nominal.
    Useful as a non-fault primary so piggyback diversity covers both
    short-circuit and load-step physics.

    ``t_on`` is accepted for signature uniformity with bus-fault primaries
    but ignored: ``scale_load`` is an immediate PQ mutation, so the caller
    must already have advanced the env to the intended t_fault.
    """
    del t_on  # load step is a synchronous mutation; caller owns timing
    load = _pick_load(env, preferred=["PQ_8", "PQ_3", "PQ_15"])
    t_on = float(env.t)
    res = scale_load(env, load_idx=load, scale=scale)
    return ScenarioResult(
        label="ambient",
        case="ieee39",
        t_attack=t_on,
        actions=[res],
        params={"load_idx": load, "scale": scale, "t_on": t_on,
                "class": "load_step", "permanent": True},
    )


# -------------------------------------- Kundur 2-area primary scenarios

def primary_kundur_bf_b6_60ms(
    env: Any, duration: float = 0.06, t_on: float | None = None,
) -> ScenarioResult:
    """Kundur primary — 60 ms bolted fault at bus 6 (area-1 gen-side HV bus).

    Bus 6 sits at the HV terminal of the gen-1/gen-2 step-up transformer
    group in Area 1. A short clearance fault exercises gen-1/gen-2 AVR
    response and leaves a visible voltage-sag signature on the 7-8 tie
    corridor. Analogue of IEEE-39 b22 (near-generator, mid-duration).
    """
    bus = _pick_bus(env, preferred=[6])
    t_on = _resolve_fault_t_on(env, t_on)
    t_off = t_on + duration
    res = apply_fault(env, bus_idx=bus, t_on=t_on, t_off=t_off)
    return ScenarioResult(
        label="ambient",
        case="kundur",
        t_attack=t_on,
        actions=[res],
        params={"bus_idx": bus, "t_on": t_on, "t_off": t_off,
                "duration_s": duration, "class": "bus_fault"},
    )


def primary_kundur_bf_b8_60ms(
    env: Any, duration: float = 0.06, t_on: float | None = None,
) -> ScenarioResult:
    """Kundur primary — 60 ms bolted fault at bus 8 (tie-corridor centre).

    Bus 8 is the midpoint of the 7-8 / 8-9 double tie corridor between
    areas. A fault here stresses the 400 MW inter-area flow and produces
    a clean angle-swing signature across both area-1 and area-2 PMUs.
    Analogue of IEEE-39 b16 (mid-network).
    """
    bus = _pick_bus(env, preferred=[8])
    t_on = _resolve_fault_t_on(env, t_on)
    t_off = t_on + duration
    res = apply_fault(env, bus_idx=bus, t_on=t_on, t_off=t_off)
    return ScenarioResult(
        label="ambient",
        case="kundur",
        t_attack=t_on,
        actions=[res],
        params={"bus_idx": bus, "t_on": t_on, "t_off": t_off,
                "duration_s": duration, "class": "bus_fault"},
    )


def primary_kundur_bf_b9_40ms(
    env: Any, duration: float = 0.04, t_on: float | None = None,
) -> ScenarioResult:
    """Kundur primary — 40 ms bolted fault at bus 9 (area-2 tie-end bus).

    Bus 9 is the area-2 end of the 8-9 tie corridor adjacent to the
    area-2 load centre (PQ_7). Duration tightened to 40 ms after
    screening revealed 80 ms exceeds CCT on this bus (P_fail|alone =
    5/5 nonconv). Analogue of IEEE-39 b29 (peripheral) but with
    shorter clearance reflecting Kundur's thinner electrical margin.
    """
    bus = _pick_bus(env, preferred=[9])
    t_on = _resolve_fault_t_on(env, t_on)
    t_off = t_on + duration
    res = apply_fault(env, bus_idx=bus, t_on=t_on, t_off=t_off)
    return ScenarioResult(
        label="ambient",
        case="kundur",
        t_attack=t_on,
        actions=[res],
        params={"bus_idx": bus, "t_on": t_on, "t_off": t_off,
                "duration_s": duration, "class": "bus_fault"},
    )


# -------------------------------------------------- scenario registry

SCENARIOS: dict[str, Callable[..., ScenarioResult]] = {
    "kundur_a1":        scripted_kundur_a1,
    "kundur_a1_mild":   scripted_kundur_a1_mild,
    "kundur_a2":        scripted_kundur_a2,
    "ieee39_a1":        scripted_ieee39_a1,
    "ieee39_a1_severe": scripted_ieee39_a1_severe,
    "ieee39_a2":        scripted_ieee39_a2,
}


PRIMARY_SCENARIOS: dict[str, Callable[..., ScenarioResult]] = {
    "primary_ieee39_bf_b16_80ms":            primary_ieee39_bf_b16_80ms,
    "primary_ieee39_bf_b22_80ms":            primary_ieee39_bf_b22_80ms,
    "primary_ieee39_bf_b29_100ms":           primary_ieee39_bf_b29_100ms,
    "primary_ieee39_bf_b02_60ms":            primary_ieee39_bf_b02_60ms,
    "primary_ieee39_load_step_pq08_plus25":  primary_ieee39_load_step_pq08_plus25,
    "primary_kundur_bf_b6_60ms":             primary_kundur_bf_b6_60ms,
    "primary_kundur_bf_b8_60ms":             primary_kundur_bf_b8_60ms,
    "primary_kundur_bf_b9_40ms":             primary_kundur_bf_b9_40ms,
}


def run_scenario(env: Any, name: str, **kwargs: Any) -> ScenarioResult:
    """Resolve a scenario short-name and execute it against ``env``."""
    fn = SCENARIOS.get(name)
    if fn is None:
        raise KeyError(
            f"Unknown scenario '{name}'. Known: {', '.join(sorted(SCENARIOS))}"
        )
    return fn(env, **kwargs)


def run_primary_scenario(env: Any, name: str, **kwargs: Any) -> ScenarioResult:
    """Resolve a primary (ambient) scenario and execute it against ``env``.

    Separated from ``run_scenario`` so that a caller cannot accidentally
    fire an attack scenario where an ambient primary is expected, and
    vice versa. Primary functions assume ``env.t == t_fault`` on entry.
    """
    fn = PRIMARY_SCENARIOS.get(name)
    if fn is None:
        raise KeyError(
            f"Unknown primary scenario '{name}'. "
            f"Known: {', '.join(sorted(PRIMARY_SCENARIOS))}"
        )
    return fn(env, **kwargs)


def list_scenarios() -> list[str]:
    return sorted(SCENARIOS)


def list_primary_scenarios() -> list[str]:
    return sorted(PRIMARY_SCENARIOS)
