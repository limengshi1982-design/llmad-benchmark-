# Piggyback primary-fault candidates — IEEE 39

Implementation checklist for `llmad/scenarios.py::primary_ieee39_*` family.
Each candidate below is a self-contained scenario function returning
`ScenarioResult(label="ambient", ...)`. Screening sweep per spec §2.2
validates `P(failed | p alone) ∈ [0.0, 0.2]` across 10 seeds before a
candidate is adopted for the main batch.

## IEEE 39 topology reminders (for picking faulted elements)

- 10 generators at buses 30–39; gen at bus 30 is the swing.
- Load buses concentrated on 1-29; largest PQ at buses 3, 4, 8, 15, 16, 20,
  21, 23, 25, 27, 29.
- Area structure (typical study partition): Gens 30/37/38/39 roughly form one
  electrical cluster; 31/32/33/34/35/36 another. Ties include Line_14 (bus
  14–15), Line_22 (bus 21–22), Line_27 (bus 26–27).
- IEEE 39 at nominal loading clears most single-line outages via governor +
  PSS damping within ~10 s → exactly the "viable" region we want for piggyback.

## Candidate list (6 to screen)

| ID | Scenario name | Primary op | Parameters | Targeted bus/line | Physics rationale | Expected transient strength |
|---:|---|---|---|---|---|---|
| P1 | `primary_ieee39_bf_b16_80ms` | `apply_fault` | `bus=16, t_on=t_fault, t_off=t_fault+0.08` | Bus 16 (mid-system load bus between area ties) | 80 ms bolted 3-phase fault simulating insulator flashover; causes voltage sag propagating across Lines 15/16/17 | Medium (v_min ≈ 0.85–0.90 near fault, local δ drift, but self-clears) |
| P2 | `primary_ieee39_bf_b22_80ms` | `apply_fault` | `bus=22, t_on=t_fault, t_off=t_fault+0.08` | Bus 22 (near gen 35, heavy transit) | Same fault type, closer to a generator; exercises stator flux decay + AVR response rather than purely network response | Medium-high (gen 35 ω excursion visible; local v dip) |
| P3 | `primary_ieee39_bf_b29_100ms` | `apply_fault` | `bus=29, t_on=t_fault, t_off=t_fault+0.10` | Bus 29 (edge load, near gen 38/39 via Line_34) | Longer duration (100 ms) because peripheral bus needs more severity to be visible at 50% PMU set | Medium (longer duration compensates for electrical distance) |
| P4 | `primary_ieee39_bf_b02_60ms` | `apply_fault` | `bus=2, t_on=t_fault, t_off=t_fault+0.06` | Bus 2 (generation-adjacent, between gens 30/37) | Short 60 ms fault near swing generator; stresses swing machine + AVR | Low-medium (tight clearance → fast recovery, tests observability) |
| P5 | `primary_ieee39_trip_reclose_L15_300ms` | `trip_line` + `reclose_line` | `line="Line_15", t_trip=t_fault, t_reclose=t_fault+0.30` | Line_15 (bus 15–16) | Topology transient: brief outage on a tie-proximate line, auto-reclose representative of distance relay + autoreclose scheme | High immediately after trip, then recovers at reclose |
| P6 | `primary_ieee39_load_step_pq08_plus25_500ms` | `scale_load` | `load="PQ_08", scale=1.25, ramp=0.5` | PQ at bus 8 | Permanent step up of 25% on a large load; pulls frequency down, exercises governors. Note: this one is *permanent* so it's a harder null baseline (frequency may settle below nominal but above thresholds) | Low-medium on ω_dev; no voltage violation expected |

## Ordered by screening priority

P1, P2, P3 are the most likely to land in `[0.0, 0.2]` viable band. P4 may be
too mild (drop candidate if `ω_dev_max < 1e-3` AND `v_min > 0.98`). P5 is the
"topological" primary we need for diversity — keep unless it's degenerate-easy.
P6 is the load-step primary (different physics from faults); keep unless
completely imperceptible at PMU.

Target outcome: ≥3 viable candidates spanning (a) network-adjacent fault,
(b) generation-adjacent fault, (c) topological or load-step primary.

## Implementation notes

- `ScenarioResult.label = "ambient"` for every primary. The convention is:
  primary is a natural disturbance, not an attacker label. `truth_label` is
  still inferred from red tool-trace post-hoc.
- Each `primary_*` function receives `env` and the current `env.t`; the
  `t_fault` inside the scenario must be absolute time, not relative. Use
  `env.t + Δ` to schedule within the current scenario, or require caller to
  `advance_until(t_fault)` first — pick one convention and enforce it.
- `apply_fault` in the existing skill takes `(bus, t_on, t_off, fault_type)`;
  set `fault_type="3ph"` (default). If ANDES requires resistance, use
  `r_fault=1e-6` for a bolted fault.
- For `P5` (line + reclose), remember to register both events. If the tool
  layer only exposes `trip_line(line, t)` and `reclose_line(line, t)`
  separately, the scenario function should call both inside a single
  transaction so they are scheduled together.
- Screening-sweep invocation (per candidate):
  ```
  python 10_red_vs_blue_batch.py \
    --red-mode none \
    --blue-profiles none \     # if supported; else a no-op blue
    --cases ieee39 \
    --primary-faults primary_ieee39_bf_b16_80ms \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --tag screen_P1
  ```
  Use `--red-mode none` with a no-op blue so the recorded `failed` is the
  primary-only baseline.

## What to record in `analysis/piggyback_screening_ieee39.md`

Per candidate, 10 seeds:

| column | meaning |
|---|---|
| `failed_rate` | `sum(stability_full.failed) / 10` |
| `ω_dev_max_mean / std` | transient severity proxy |
| `v_min_mean / std` | voltage-side severity |
| `δ_drift_max_mean` | angle-side severity |
| `t_peak_ω / t_peak_v` | when the transient peaks — determines red's `t_a` window |
| viable? | `failed_rate ∈ [0.0, 0.2]` AND signal visible at 50% PMU |

Then a short paragraph selecting the 3 primaries for the main batch, with
one-sentence physics justification each. That markdown becomes a figure/table
source for the paper.

## Acceptance criterion before moving to main batch

At least one candidate from each of these three buckets must survive
screening:

1. Network-adjacent fault (P1 or P3).
2. Generation-adjacent fault (P2 or P4).
3. Topological or load-step primary (P5 or P6).

If a bucket is empty, the fallback is:

- Bucket 1 empty: try longer fault (100→120 ms) on the same buses.
- Bucket 2 empty: try a different gen-adjacent bus.
- Bucket 3 empty: document as limitation, use only buckets 1+2 in main batch.

Do not silently extend the viable band above 0.2; that would invalidate the
"piggyback amplifies a recoverable primary" framing.
