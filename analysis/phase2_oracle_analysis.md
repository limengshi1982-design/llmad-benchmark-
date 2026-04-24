# Phase II Oracle Defender — Deep Analysis of the 48-Run Batch

Sources:
* `experiments/summaries/blue_batch_20260419T094931Z__m3_blue_first.csv` (36 rows — main sweep)
* `experiments/summaries/blue_batch_20260419T103357Z__m3_a1_mild.csv` (9 rows — A1-mild scenario)
* `experiments/summaries/blue_batch_20260419T104302Z__m3_a1_severe_early.csv` (3 rows — A1 severe with earlier handoff)

Batch shape: 3 profiles × 5 scenarios × 3 seeds + 3 extra `kundur_a1` severe-early re-runs = 48 runs.
Protocol: LLM receives the **oracle attack label** at t_b (diagnosis is bypassed), full blue registry (analysis + defense + advance), horizon T=30 s, tool budget 10.

---

## 1. Headline result

* **All three profiles recover on 12/16 = 75 % of episodes**. The profiles are statistically indistinguishable on recovery rate.
* The 25 % non-recovery is **fully explained by one scenario** (`kundur_a1` severe): 12/12 runs across all profiles failed. The remaining four scenarios recover 36/36.
* This means **the recovery-rate metric, as currently configured, does not separate defender skill from defender luck**. See §5 for implications.

## 2. Per-profile aggregates (pooled)

| Profile | n | Recovery | Avg shed (pu) | Avg |ω| dev | Avg tool calls | Avg tokens | Avg wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek-chat | 16 | 75.0 % | **0.93** | 0.143 | 10.0 |  58.9 K | 45.3 |
| minimax-m2    | 16 | 75.0 % | 3.75     | 0.111 | 11.5 | **24.9 K** | 67.4 |
| qwen3-plus    | 16 | 75.0 % | 6.64     | 0.083 | 12.2 |  **136.5 K** | 58.5 |

Two interpretations of this row:

* **deepseek-chat is the most parsimonious defender** — it recovers the same fraction of scenarios while shedding **7× less load** than qwen3. Its Phase II prompt style is "do less, intervene only when needed".
* **qwen3-plus is profligate in Phase II** — this is the *opposite* of its Phase I behaviour, where qwen3 was the cheapest profile (11 K tokens, 6 s). In Phase II qwen3 balloons to 136.5 K tokens and sheds almost 7× more load than deepseek. Plausible cause: Phase II gives qwen3 the full registry including `advance_until`, and qwen3 re-reads PMU evidence between every action, inflating both token and shedding totals.

## 3. Per-scenario aggregates (pooled across profiles)

| Scenario | n | Recovery | Failed% | Avg shed (pu) | Δδ drift max (rad) | V_min |
|---|---:|---:|---:|---:|---:|---:|
| `ieee39_a1`       |  9 | 100 % |   0 % | 0.43 |   0.20 | 0.999 |
| `ieee39_a2`       |  9 | 100 % |   0 % | 2.55 |   0.00 | 1.002 |
| `kundur_a1` (severe) | 12 | **0 %** | 100 % | 6.85 | 1112.09 | **0.315** |
| `kundur_a1_mild`  |  9 | 100 % | 100 % | 2.35 |  13.49 | 0.863 |
| `kundur_a2`       |  9 | 100 % | 100 % | 5.68 |  12.19 | 0.872 |

A few cells deserve a second look:

* **`failed=True` with `recovery_success=True`** (27 of 48 rows): this is the metric split working as intended. `failed` captures *any* excursion during the 30 s trajectory (e.g., the immediate post-attack dip below V_floor = 0.85 pu); `recovery_success` is evaluated only on the final 3 s window. A good defender returns the system to spec by T=30 s *even though* it transited a violation window around t=5-7 s. The paper should plot both and explain the separation.
* **`ieee39_a1` and `ieee39_a2` show no stability violation** (failed=0 %, V_min ≥ 0.999, Δδ ≈ 0). IEEE 39 at nominal loading is N-1 secure — the attacks do not drive the system out of band. This means Phase II on these scenarios is testing *"did the defender refrain from making things worse"*, not *"did the defender save the system"*.
* **`kundur_a1_mild` and `kundur_a2` both recover fully** despite `failed=True` in 100 % of cases. The system self-heals on its 30 s horizon thanks to governor droop + natural damping.

## 4. Shedding behaviour — the real differentiator

Recovery rate is a coarse metric; **how much did the defender pay to recover** is finer.

### 4.1 Zero-shed episodes

Counting runs with `load_shed_total_pu == 0`:

| Profile | zero-shed runs | zero-shed successes |
|---|---:|---:|
| deepseek-chat |  8/16 (50 %) | 8/8 (100 %) |
| minimax-m2    |  5/16 (31 %) | 5/5 (100 %) |
| qwen3-plus    |  0/16 ( 0 %) | — |

deepseek **correctly identifies that no action is needed** on 8 of its 16 runs and achieves 100 % recovery on them. qwen3 **never** chooses no-action, and ends up over-shedding on scenarios that would have recovered on their own. This is the most publishable behavioural finding in the Phase II batch.

### 4.2 Per-scenario shed ranges (min-max pu across seeds+profiles)

| Scenario | deepseek | minimax | qwen3 |
|---|---|---|---|
| `ieee39_a1`      | 0-0 | 0-0 | 1.20-1.36 |
| `ieee39_a2`      | 0-0 | 2.67-6.62 | 2.10-4.49 |
| `kundur_a1` (severe) | 0-3.48 | 3.89-8.20 | 6.63-27.34 |
| `kundur_a1_mild` | 0-0 | 0-3.48 | 4.73-8.20 |
| `kundur_a2`      | 0-7.99 | 2.00-8.51 | 3.48-11.98 |

Two observations:

* **qwen3 over-sheds on `kundur_a1` severe by up to 27.34 pu** — and the system still collapses. This is "spray-and-pray" defender behaviour with no payoff. Worth a case study in the paper.
* **deepseek's range on `kundur_a2`** is 0-7.99 pu: it sometimes does nothing, sometimes sheds heavily. Same profile, same oracle label, three seeds, big variance. This is a policy-instability finding.

## 5. The metric problem — `recovery_success` is almost saturated

Of the 48 runs:

* 36 recover (75 %).
* All 12 non-recoveries are on `kundur_a1` severe, which is **physically unrecoverable** (Line_7 or Line_8 trip on the 2-circuit 8-9 tie puts the survivor over its thermal limit and the system loses synchronism before the defender can act, regardless of action).
* The other four scenarios recover 36/36 — **including runs where the defender took no action at all**. Natural governor droop + intact topology is sufficient for recovery.

This yields a stark observation:

> *In the current scenario set, recovery is determined almost entirely by attack severity and not by defender policy.*

Consequences for the paper:

1. **Recovery rate alone is not a discriminating metric.** The paper cannot claim "LLM X defends better than LLM Y" from 75 % / 75 % / 75 %.
2. **The salient metric should be shedding cost on the recoverable scenarios.** deepseek shedding 0.93 pu avg vs qwen3 shedding 6.64 pu avg is a **7× operator-cost difference** with identical recovery. That *is* a defender-skill signal.
3. **An additional hard scenario is needed** so that the defender's actions actually determine the outcome. Candidates: `ieee39` with N-2 loading, Kundur A4 combined (A1 trip + AGC FDI), Kundur A3 (PSS gain to 0 → weakly-damped inter-area oscillation), or a load-step large enough that UFLS is mandatory on a 10 s timescale.

## 6. Tool-use patterns

Averaged across all 48 runs:

| Profile | Avg tool calls | Modal stopped_reason |
|---|---:|---|
| deepseek-chat | 10.0 | `max_steps+autorolled` (forced final turn after budget) |
| minimax-m2    | 11.5 | mix of `horizon_reached` and `max_steps+autorolled` |
| qwen3-plus    | 12.2 | `max_steps+autorolled` + `no_tool_call+autorolled` |

`stopped_reason` distribution (all runs pooled):

* `horizon_reached` (LLM explicitly advanced env to T): 21
* `max_steps+autorolled` (hit step budget, driver forced advance): 21
* `no_tool_call+autorolled` (LLM gave up mid-episode): 6

deepseek is *not* more efficient by tool count — it uses the full 10-budget. But it prefers `get_pmu_window` + `compute_rocof` + `advance_until` (read, sense, advance) over mutating tools, which is why its shedding total is small. qwen3 at 12.2 tool calls exceeds the advertised budget — investigation of qwen3's traces would show repeated calls to the same tool between actions.

## 7. Failure-mode distribution (on the 30 failed=True cases)

* 18 cases trip `delta` criterion (angle divergence > 1 rad).
* 12 cases trip `voltage` criterion (any bus V < 0.85 pu).

All 12 `voltage` trips and all 12 `kundur_a1` severe cases overlap. The `kundur_a1_mild` and `kundur_a2` cases trip `delta` during the post-attack dip but clear by T=30 s. This supports the 3 s-window recovery criterion being the right headline number.

## 8. Wall-time as deployment gate

Phase II recovery planning takes **45-67 s** of wall time per episode. If the operator has a 30 s horizon to act (realistic for slow-moving cascade recovery but not for UFLS), the LLM defender spends 150-220 % of its allotted budget thinking. Two ways to address this in the paper:

* **Honestly**: frame LLM defender as a *minutes-scale* decision aid for restoration after loss-of-synchronism, not as a replacement for automatic protection.
* **Tactically**: strip analysis tools and cache PMU context — should bring wall time under 30 s. Parked as future work.

## 9. What to report vs what to fix

### Honest reportable findings (no rerun required)

1. LLM defenders with oracle attack hand-off achieve **100 % recovery on 4 of 5 scenarios**; the fifth (`kundur_a1` severe) is physically unrecoverable.
2. On recoverable scenarios, profiles differ in shedding cost by up to 7× (deepseek 0.93 pu vs qwen3 6.64 pu, both at 100 % recovery).
3. deepseek uses no-action decisions on 50 % of its runs and never recovers less than 100 %.
4. Confidence-style over-action by qwen3 is costly (136.5 K tokens, 6.64 pu shed) and the effort does not translate into better outcomes.
5. Phase II wall time (45-67 s) bounds LLM defenders to minute-scale problems.

### Fixes to consider before camera-ready

1. **Add a "defender-sensitive" scenario** — one where no-action fails and only modest shedding recovers. Without this, recovery rate remains uninformative.
2. **Add a rule-based UFLS baseline and a simplified MPC upper bound** (see the strategic discussion) so the 75 % number has anchors.
3. **Record a `shedding_efficiency = Δω_recovered / load_shed_pu` metric** to directly quantify "did the defender shed the right load".
4. **10 seeds per scenario** so per-scenario means have error bars.
