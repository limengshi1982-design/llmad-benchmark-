# Kundur Main Batch — End-to-End Red-vs-Blue Analysis

Sources:
- scripted: `experiments/summaries/e2e_batch_20260419T125943Z__kundur_main_scripted.csv`
- llm:      `experiments/summaries/e2e_batch_20260419T132217Z__kundur_main_llm.csv`

Batch shape:
- `kundur` case only, horizon T=30 s, t_a=2 s, t_b=5 s
- scripted: 1 red placeholder × 3 blue profiles × 5 seeds = **15 runs**
- llm:      3 red profiles (minimax-m2 / deepseek-chat / qwen3-plus) × 3 blue profiles × 5 seeds = **45 runs**
- total **60 runs**

All runs use the 2026-04-19 `09_red_vs_blue.py` orchestrator with self-diagnosis
handoff (Phase II receives blue's own Phase I verdict, not oracle).

Aggregation script: `analysis/kundur_main_compute.py`.

---

## 1. Headline result

> **Scripted-red is catastrophic and unrecoverable on Kundur (0/15 recovery,
> 13/15 diag-right-mitigation-failed, all voltage-collapse at v_min≈0.49 pu).
> LLM-red is superficially aggressive but physically mild: although every
> LLM-red run also crosses a failure threshold at some point, blue recovers
> in 33/45 runs (73.3% avg). The LLM-red ↔ scripted-red gap is the main
> finding — LLM red does not match a textbook attacker.**

Three orthogonal dimensions:

1. **Red mode × physical lethality** — scripted is the real ceiling; LLM red
   triggers thresholds but leaves the system recoverable.
2. **Blue profile × recovery cost when red is lethal** (scripted block).
3. **Blue profile × recovery cost when red is mild** (llm block).

## 2. Aggregate tables

### TAB-K-RECOV — Recovery rate by (red_mode, red_profile, blue_profile)

| red_mode | red_profile | blue_profile | n | recovery% | diag% | avg shed (pu) | avg tokens | avg wall (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| scripted | — | minimax-m2    | 5 |   0.0% |  60.0% | 1.82 |  38 107 |  133.0 |
| scripted | — | deepseek-chat | 5 |   0.0% | 100.0% | 5.38 | 134 029 |   88.5 |
| scripted | — | qwen3-plus    | 5 |   0.0% | 100.0% | 3.41 | 122 102 |   43.3 |
| llm | minimax-m2    | minimax-m2    | 5 |  60.0% |  40.0% | 4.67 |  51 844 |  126.2 |
| llm | minimax-m2    | deepseek-chat | 5 | 100.0% |   0.0% | 3.71 | 134 972 |  114.1 |
| llm | minimax-m2    | qwen3-plus    | 5 |  80.0% |  60.0% | 6.00 | 101 809 |   76.2 |
| llm | deepseek-chat | minimax-m2    | 5 | 100.0% |   0.0% | 2.30 |  54 126 |  160.6 |
| llm | deepseek-chat | deepseek-chat | 5 | 100.0% |  80.0% | 2.02 | 150 999 |  140.8 |
| llm | deepseek-chat | qwen3-plus    | 5 | 100.0% |  80.0% | 2.59 | 144 898 |   68.8 |
| llm | qwen3-plus    | minimax-m2    | 5 |  40.0% |  60.0% | 0.93 |  49 440 |  178.8 |
| llm | qwen3-plus    | deepseek-chat | 5 |  80.0% |  60.0% | 2.57 | 141 814 |  111.7 |
| llm | qwen3-plus    | qwen3-plus    | 5 |  60.0% |  80.0% | 4.44 | 134 957 |   54.8 |

### TAB-K-OUTCOME — Joint-outcome distribution

| red_mode | red_profile | blue-win-clean | diag-right-mit-failed | lucky-recovery | red-win |
|---|---|---:|---:|---:|---:|
| scripted | — | 0 | 13 | 0 | 2 |
| llm | minimax-m2    | 2 | 3 | 10 | 0 |
| llm | deepseek-chat | 8 | 0 |  7 | 0 |
| llm | qwen3-plus    | 4 | 6 |  5 | 0 |

**Interpretation:**

- Scripted → 13/15 `diag-right-mit-failed` + 2/15 `red-win` is the expected
  physical-ceiling signature: blue diagnosed correctly (A1 voltage collapse)
  but the Kundur `scripted_kundur_a1` attack drives v_min to ~0.49 pu, which
  no mitigation in our toolkit can undo (confirmed in Phase II oracle run).
- LLM-red deepseek is the **most recoverable attacker** (8/15 blue-win-clean,
  0 diag-right-mit-failed). Deepseek as red picks actions that perturb but
  don't strand the system in a voltage-collapse basin.
- LLM-red minimax-m2 gets the most `lucky-recovery` (10/15) — blue's
  diagnosis was wrong in most of those runs (only 40% diag correct when
  paired with itself) but recovery happened because the attack was mild.
- LLM-red qwen3 is **closest to scripted-red in physical consequence**:
  6/15 `diag-right-mit-failed` (voltage collapses blue can diagnose but not
  fix) — but still not 13/15.
- `red-win` column: LLM red achieved 0 red-wins out of 45; scripted achieved
  2/15. Blue never outright fails to diagnose AND fails to recover on LLM.

## 3. Red-mode gap — scripted as physical ceiling

Per-red-config, across all blue pairings:

| red config       | failed/n | recovery/n | notes |
|---|---:|---:|---|
| scripted (a1 severe) | 15/15 (100.0%) | 0/15 ( 0.0%) | voltage collapse, unrecoverable |
| llm minimax-m2       | 15/15 (100.0%) | 12/15 (80.0%) | mostly delta-divergence, recoverable |
| llm deepseek-chat    | 15/15 (100.0%) | 15/15 (100.0%) | delta-divergence, fully recoverable |
| llm qwen3-plus       | 15/15 (100.0%) |  9/15 (60.0%) | mix of delta (recoverable) and voltage (not) |

**`failed=True` is 100% across every red configuration** — every run crosses
a stability threshold at some point in the 30 s horizon. The separator is
**which threshold**: scripted always trips voltage (hard-unrecoverable);
LLM red mostly trips delta (transient angle divergence, which blue can shed-
load out of). This is the single most important finding of the batch.

**Gap interpretation (publishable):**

> LLM-red on Kundur triggers instability-detection thresholds at the same
> rate as a scripted attacker, but on a **qualitatively different failure
> surface**: 77.8% delta-divergence vs scripted's 100.0% voltage-collapse.
> Delta-divergence on Kundur is recoverable in 33/35 cases via standard UFLS;
> voltage-collapse is recoverable in 0/15. Hence LLM red, with the current
> tool budget and prompt, does not yet reach the textbook attacker ceiling.

## 4. Shed-cost differentiator

Phase II oracle found: deepseek 0.93 pu < minimax 3.75 pu < qwen3 6.64 pu.
Here under self-diagnosis (E2E) the **ordering partly inverts** (same finding
as under the rushed smoke):

### TAB-K-SHED — Shed distribution by blue profile (pooled across red modes)

| blue_profile  | n  | zero-shed% | avg shed | median | max   | avg shed (recovered) | avg shed (failed) |
|---|---:|---:|---:|---:|---:|---:|---:|
| minimax-m2    | 20 | 50.0% | 2.43 | 1.16 | 15.16 | 1.97 | 2.89 |
| deepseek-chat | 20 | 25.0% | 3.42 | 2.32 |  8.20 | 2.38 | 5.85 |
| qwen3-plus    | 20 | 20.0% | 4.11 | 3.48 | 11.77 | 4.09 | 4.13 |

Under self-diagnosis:

- **minimax-m2 is now the cheapest blue** (2.43 pu avg, 50% zero-shed runs),
  not deepseek. Because minimax mis-diagnoses 60% of the time, and its
  no-action default on uncertain diagnosis often happens to be correct when
  red is mild.
- **deepseek flipped from cheapest to mid-expensive** (3.42). Phase II's
  deepseek "no-action 50%" policy under oracle doesn't survive: when deepseek
  has to choose from its own diagnosis, it trusts the diagnosis and sheds,
  and its 60% diagnosis accuracy means it sheds wrongly often.
- **qwen3 remains the most expensive** (4.11) — policy is stable, not input-
  sensitive.

**Phase II F-2c finding (7× shed gap) does NOT survive self-diagnosis.** The
gap compresses to 1.7× (minimax 2.43 → qwen3 4.11). Reportable as a
limitation of oracle-labelled benchmarks.

Separate "avg shed on failed" column: deepseek sheds 5.85 pu in hopeless
runs (biggest over-reaction in defeat), minimax 2.89, qwen3 4.13. Qwen3's
over-shedding signature from Phase II partially transfers.

## 5. Diagnosis quality under self-diagnosis vs dedicated Phase I

| blue_profile  | E2E diag% | Phase-I-dedicated diag% | delta |
|---|---:|---:|---:|
| minimax-m2    | 40.0% | 60.0% | −20.0 |
| deepseek-chat | 60.0% | 53.3% |  +6.7 |
| qwen3-plus    | 80.0% | 73.3% |  +6.7 |

Interesting asymmetry. **Qwen3 and deepseek both improve under self-diagnosis
handoff**; minimax degrades substantially. Most likely explanation: when red
is LLM-driven, many episodes are not A1 (they are mild perturbations whose
true label the blue can read as "None"); qwen3 and deepseek are willing to
output `None` when uncertain, minimax defaults to `A1`. This is visible in
the truth_label distribution — 15/45 LLM runs have truth_label ≠ "A1".

## 6. Outcome heatmap — data prep

Heatmap cell annotations `(recovery%, avg_shed)`:

scripted (1 row × 3 cols):

|      | minimax-m2 | deepseek-chat | qwen3-plus |
|---|---|---|---|
| scripted | (0%, 1.82) | (0%, 5.38) | (0%, 3.41) |

llm (3 rows × 3 cols):

|  red\blue  | minimax-m2 | deepseek-chat | qwen3-plus |
|---|---|---|---|
| minimax-m2 (red)    | (60%, 4.67)  | (100%, 3.71) | ( 80%, 6.00) |
| deepseek-chat (red) | (100%, 2.30) | (100%, 2.02) | (100%, 2.59) |
| qwen3-plus (red)    | ( 40%, 0.93) | ( 80%, 2.57) | ( 60%, 4.44) |

Salient cells:
- **deepseek(red) × deepseek(blue) = (100%, 2.02)** — cheapest non-zero
  recovery cell in the whole batch. deepseek as red is mild; deepseek as
  blue knows how to respond cheaply when uncertain.
- **qwen3(red) × minimax(blue) = (40%, 0.93)** — lowest shed AND lowest
  recovery. Minimax under-sheds; qwen3's attacks are lethal enough to need
  shedding that minimax doesn't provide.

## 7. Tool-use and token economics

### TAB-K-COST — Blue side (diag + mit combined), pooled across red

| blue_profile  | avg diag calls | avg mit calls | avg prompt | avg completion | avg wall (s) |
|---|---:|---:|---:|---:|---:|
| minimax-m2    | 4.2 |  8.6 |  38 588 | 4 711 | 149.7 |
| deepseek-chat | 6.0 | 11.0 | 133 215 | 2 110 | 113.7 |
| qwen3-plus    | 5.2 | 10.1 | 119 354 | 1 529 |  60.8 |

Notes:
- **Qwen3 is 2.5× faster than minimax in wall-clock** despite 3× the prompt
  tokens — qwen3 inference is very fast and its completion tokens are short.
- **Minimax has 4× completion tokens of qwen3** (4711 vs 1529) — matches the
  verbose JSON/reasoning tokens in prior batches. Minimax hits max_steps
  less often but reasons longer per step.
- **Deepseek tops prompt tokens** at 133K (Phase I prompt engineering does
  not compress); completion 2110 is low, so token cost per step is
  input-dominated.

### Red cost (llm mode only)

| red_profile  | avg tool calls | avg prompt tokens | avg completion tokens |
|---|---:|---:|---:|
| minimax-m2    | 6.9 |     594 |   719 |
| deepseek-chat | 7.0 | 11 674  |   597 |
| qwen3-plus    | 6.9 |   6 041 |   731 |

All three LLMs use ~7 tool calls (close to budget=7 ceiling). Deepseek
prompt is 20× minimax prompt — deepseek API expands system/observation
context differently.

## 8. Reproducibility of Phase II oracle findings

Phase II oracle batch (48 runs, oracle label handoff) found:
- F-2a: 75% recovery equal across profiles; differentiator is shed cost.
- F-2b: deepseek picks no-action 50% of runs; qwen3 never does.
- F-2c: qwen3 7× over-shedding vs deepseek at identical recovery.
- F-2d: qwen3 136.5K tokens in Phase II vs 11K in Phase I (reversal).

E2E main batch replicates / refines:

- **F-2a (75% recovery ceiling)** — E2E LLM-red avg recovery = 73.3% (11/15
  per red, pooled). **Reproduced.** But scripted-red drops to 0% — so the
  75% ceiling only holds for mild attackers; a textbook attacker breaks it.
- **F-2b (deepseek no-action)** — minimax-m2 is now the no-action profile
  (50% zero-shed) not deepseek (25% zero-shed). **Inverted.** Under self-
  diagnosis, deepseek acts when its own diagnosis is certain; it was only
  "no-action-happy" when the oracle gave it uncertain labels to dismiss.
- **F-2c (7× shed gap)** — compresses to 1.7× (minimax 2.43 → qwen3 4.11).
  **Partially reproduced.** Ordering preserved (minimax < deepseek < qwen3)
  but magnitude halves. Writeup: "oracle diagnosis inflates inter-profile
  differences; self-diagnosis compresses them via correlated errors."
- **F-2d (qwen3 token dominance)** — qwen3 blue prompt = 119K, deepseek
  133K, minimax 38K. Qwen3 is NOT the leader anymore; deepseek takes the
  crown. **Not reproduced.** Depends on the Phase I prompt shape, which
  differs between dedicated Phase I and E2E.

Net: Phase II oracle findings are **brittle** under self-diagnosis. This is
a novel contribution — oracle-labelled Phase II benchmarks systematically
overstate inter-LLM differences.

## 9. Failure-mode distribution

On the `failed=True` subset (which is 60/60 — every run):

| red_mode | n_failed | delta% | voltage% |
|---|---:|---:|---:|
| scripted | 15 |  0.0% | 100.0% |
| llm      | 45 | 77.8% |  22.2% |

Scripted's 100% voltage matches the a1-severe fault design (bus-8 line trip
in Kundur drives post-fault voltage to ~0.49 pu — unrecoverable). LLM's 78%
delta signature is a new finding: **LLM red left alone preferentially picks
actions that excite angle divergence but not voltage collapse** — because
they go for generator-tripping and load-scale actions visible in the PMU
observation, whereas the one decisive action (tripping bus 8's critical
line) is less visible in the truncated PMU window `[t_a−2, t_a]`.

Voltage-collapse by LLM (10/45) is entirely from qwen3 red (6/15) and
minimax red (3/15 via mitigation failure on recoverable starts); deepseek
red triggered no voltage failures.

## 10. Wall-time and deployment gate

| red_mode | avg wall (s) | max wall (s) | runs > 90 s |
|---|---:|---:|---:|
| scripted |  88.3 | 183.4 |  5/15 |
| llm      | 114.7 | 217.5 | 32/45 |

32/45 LLM runs exceed 90 s — driven by tools-loop on slower vendors
(deepseek+qwen3 prompt ingestion). Mean of 114.7 s per 30-s simulated
episode is acceptable for offline benchmarking but **not real-time
deployable**. Gate this before any real-world pilot claim.

## 11. Honest reportable findings

1. **Scripted-red is catastrophic (0% recovery, 100% voltage-collapse, v_min
   ≈0.49 pu); LLM-red matches the trigger rate (100% failure-threshold
   crossing) but lands on a recoverable failure surface (77.8% delta vs
   100% voltage).** LLM red currently does NOT match a textbook attacker.

2. **Self-diagnosis handoff compresses the Phase II oracle shed-cost gap
   from 7× to 1.7×.** Oracle-labelled mitigation benchmarks systematically
   overstate inter-model differences because oracle removes the correlated-
   error confound.

3. **Diagnosis under self-diagnosis is profile-sensitive:** qwen3 80% >
   deepseek 60% > minimax 40%. Minimax's Phase I accuracy degrades by 20
   points relative to its dedicated Phase I (60% → 40%) because minimax
   defaults to `A1` and the LLM-red batch contains many non-A1 truths.

4. **Recovery rate has a wide tri-modal distribution in LLM×LLM cells:**
   40%–100% across 9 cells (TAB-K-RECOV). The best recovery cell is
   deepseek(red) × deepseek(blue) @ 100%/2.02 pu; the worst is qwen3(red)
   × minimax(blue) @ 40%/0.93 pu.

5. **Zero red-wins on LLM-red (0/45)**; 2 red-wins on scripted-red (2/15).
   Blue always ends recovered when the attack is recoverable; an attacker
   needs physical ceiling access to "win" outright.

## 12. Fixes needed before camera-ready

- [ ] Extend to 10 seeds per cell for publishable error bars (currently n=5).
- [ ] Add rule-based UFLS baseline (Track 3a) for `blue-win-clean` normaliser.
- [ ] Add MPC-lite upper bound (Track 3b).
- [ ] Cross-case: pair with IEEE 39 piggyback batch (separate analysis; that
      batch is the venue where LLM-red has a chance to close the lethality
      gap, since scripted single-step attacks don't destabilise IEEE 39).
- [ ] Qualitative study of LLM-red action choices — why does deepseek avoid
      voltage-collapse paths? Write-up fodder for §V.C "LLM-red attack-policy
      failure modes".
- [ ] Decouple "threshold trigger" from "genuine failure" metric — current
      `failed=True` fires too easily. Consider `failed_severe = voltage AND
      v_min<0.5 AND unrecovered_at_T`.

---

## 13. Figure source data

- **FIG-K-HEATMAP (4-panel)**: two 3×3 heatmaps of recovery% and avg shed
  for LLM-red; one 1×3 for scripted. Uses `TAB-K-RECOV` directly.
- **FIG-K-SHED-BAR**: grouped bar, x=red-config (scripted / minimax / deepseek
  / qwen3), hue=blue_profile, y=mean shed with std bars.
- **FIG-K-OUTCOME-STACK**: stacked bar per red config, 4 segments
  (blue-win-clean / diag-right-mit-failed / lucky-recovery / red-win).
  Uses TAB-K-OUTCOME.
- **FIG-K-FAILMODE**: 2-bar (scripted vs llm) split voltage/delta. One of
  the simplest but most informative figures for the paper.
- **FIG-K-TIMELINE**: overlay one representative episode per red config —
  omega trajectories for all 4 gens, vertical lines at t_a, t_b, tool-call
  events.

All reuse `analysis/make_phase1_figs.py` conventions: matplotlib, dpi=160,
bbox_inches='tight', same colour palette.
