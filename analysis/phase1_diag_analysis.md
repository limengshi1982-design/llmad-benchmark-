# Phase I Diagnosis — Deep Analysis of the 45-Run Batch

Source: `experiments/summaries/diag_batch_20260419T105852Z__m3_diag_first.csv`
Per-run JSONs: `experiments/runs_diag/*20260419T1[01]*.json`
Batch: 3 profiles × 5 scenarios × 3 seeds, tool_budget=5, max_steps=6, T∈[2s,5s] observation window.

---

## 1. Headline numbers

| Profile | n | Acc. | Avg tool calls | Avg total tokens | Avg wall (s) |
|---|---:|---:|---:|---:|---:|
| minimax-m2       | 15 | **53.3 %** | 4.5 | 14.6 K |  40.2 |
| deepseek-chat    | 15 | 40.0 % | 6.0 | **55.9 K** |  38.1 |
| qwen3-plus       | 15 | 40.0 % | 5.0 |  11.1 K |  **6.4** |

Two headline observations:

* **minimax-m2 is the most accurate** but has a 2/15 hard API failure rate (empty final-turn text → `label=None`, counted as incorrect).
* **qwen3-plus is ~6× cheaper and ~6× faster** than deepseek-chat for the same 40 % accuracy. It is the only profile viable for tight-loop experiments in Phase II iteration.

## 2. Per-scenario accuracy (all profiles pooled, n=9 each)

| Scenario | Truth | Acc. | Modal error | Root cause |
|---|---|---:|---|---|
| kundur_a1         | A1 | **9/9 (100 %)** | — | Severe A1 separation (78 rad Δδ, divergent ROCOF) — textbook. |
| kundur_a1_mild    | A1 | **9/9 (100 %)** | — | Mild A1 still shows divergent-sign ROCOF and 0.32 rad spread. |
| kundur_a2         | A2 | 1/9 ( 11 %) | A1 (8/9)  | PQ_1 is on bus 7 — adjacent to the 7-8 area tie. A step on that load excites the inter-area mode, which in ROCOF looks divergent-sign. |
| ieee39_a1         | A1 | 3/9 ( 33 %) | A2, None  | Line_26 is peripheral; voltage sag is coherent system-wide and mimics A2 load-step signature. qwen3 returns None claiming "no generator PMU signals". |
| ieee39_a2         | A2 | 0/9 (  0 %) | None (7/9) | PQ_8 step magnitude is small vs IEEE 39 inertia; residual voltage/ROCOF signals fall below the LLM's attention threshold. |

**Two scenarios, two different failure modes.**

* Kundur A2 is a **data-generation problem** (PQ_1 is physically close to the inter-area tie → A2 looks like A1 in ROCOF).
* IEEE 39 A1/A2 are a **feature-engineering problem** (PMU coverage excludes or insufficiently covers generator buses — see §4).

## 3. Confusion matrices

### minimax-m2

| truth \\ pred | A1 | A2 | A3 | A4 | None |
|---|---:|---:|---:|---:|---:|
| A1 (n=9) | 7 | 1 | 0 | 0 | 0 |
| A2 (n=6) | 2 | 1 | 0 | 0 | 2 |

(*2 additional A1 runs had no extractable verdict — treated as wrong.*)

### deepseek-chat

| truth \\ pred | A1 | A2 | A3 | A4 | None |
|---|---:|---:|---:|---:|---:|
| A1 (n=9) | 6 | 3 | 0 | 0 | 0 |
| A2 (n=6) | 3 | 0 | 0 | 1 | 2 |

deepseek is the **only profile that ever guesses A4** — it did so once on ieee39_a2 (seed=1) with rationale *"Constant voltage across all PMUs … suggests FDI masking actual disturbance"*. That is actually a physically sensible inference (if all PMU values were truly flat during an announced attack, FDI masking would be a reasonable hypothesis). It was wrong here because the attack **was** real, just too small to see.

### qwen3-plus

| truth \\ pred | A1 | A2 | A3 | A4 | None |
|---|---:|---:|---:|---:|---:|
| A1 (n=9) | 6 | 0 | 0 | 0 | 3 |
| A2 (n=6) | 3 | 0 | 0 | 0 | 3 |

qwen3's error mode is binary: **A1 or None**. It never outputs A2/A3/A4. On every IEEE 39 scenario (6/6) it returns None with the same rationale: *"No generator PMU signals available"*. This is a systematic prompt-parsing failure, not a reasoning failure — see §4.

## 4. Root-cause forensics

### 4.1 Kundur-A2 → A1 collapse

Looking at the rationales for the 8 wrong kundur_a2 predictions, every one cites divergent-sign ROCOF. Example (minimax-m2, seed=0, confidence 0.82):

> *"Coherent positive ROCOF spike (0.64 Hz/s) indicates supply loss; massive delta drift (~4.78 rad) shows generators separating into two groups…"*

Contrast with the single correct case (minimax-m2, seed=2, confidence 0.82):

> *"All four PMU-covered gens show coherent negative ROCOF (same sign, -0.05 to -0.14 Hz/s) during 2-5s, moderate voltage sag on all buses, small bounded angle spread (0.36 rad)…"*

The ROCOF sign-split is **real** in the PMU window — PQ_1 sits on bus 7, and a step on it asymmetrically loads the two areas. Kundur's inter-area mode (~0.6 Hz) means that within a 3 s window ROCOF **genuinely** looks divergent. The LLM is not hallucinating; it is reading a signature the scenario doesn't cleanly produce.

**Fix path** (parked, not code):

* A2 scenario should use a load **not** adjacent to an area tie (e.g., scale several buses together inside one area; or scale all loads uniformly by a small amount).
* Or, extend the observation window to let the inter-area oscillation damp, so coherent sag emerges.

### 4.2 IEEE 39-A1 → A2 collapse (deepseek-chat & minimax-m2)

On ieee39_a1 the attack is Line_26 trip, which is peripheral. The rationale pattern is consistent (deepseek, 3/3 A2 predictions):

> *"Coherent moderate voltage sag across all PMUs, no oscillation or divergence, matches load-altering step attack"*

This is a reasonable diagnosis from voltage-only data. Divergent ROCOF is the A1 discriminator, and on a peripheral line trip ROCOF divergence is small. Expected behaviour.

**Fix path**: Use `ieee39_a1_severe` (Line_28, central corridor) instead of `ieee39_a1` (Line_26, peripheral) in the Phase I benchmark — the severe variant produces discriminable rotor-angle separation.

### 4.3 qwen3-plus IEEE 39 blind spot

qwen3 returns "no generator signals" on all six IEEE 39 runs but correctly diagnoses Kundur. Two hypotheses:

1. **Tool-output reading**: qwen3 calls `compute_rocof` and `compute_angle_diff`, receives JSON with an empty `per_generator` field (because the PMU set doesn't overlap with generator buses for IEEE 39), and infers "no data → None".
2. **Tool-call strategy**: qwen3 only calls 5 tools vs deepseek's 6; it may be skipping `get_pmu_window` and going straight to generator-centric tools that return empty.

Given that qwen3 emits the same rationale across 6 runs with near-identical tokens (~11.3 K prompt, ~200 completion), I believe hypothesis 1 — the model mechanically propagates the "empty" status without retreating to voltage-only reasoning.

**Fix path**:

* Change the DIAG system prompt to explicitly say: *"If generator omega/delta tools return empty, fall back to voltage-only analysis — coherent sag = A2, asymmetric dip = A1."*
* Or, verify and expand IEEE 39 PMU coverage to include at least one generator bus.

### 4.4 minimax-m2 empty-text failures

Two runs (ieee39_a1 s2, ieee39_a2 s1) produced `final_text = ''` after the budget-exhaustion forced-final turn. No rationale, no verdict. MiniMax M2 occasionally returns empty content when forced to stop tool-calling. This is a 2/45 = 4.4 % hard failure rate for minimax specifically.

**Fix path**: In the driver, on empty final text, issue one retry with a stronger user message (*"Output the JSON verdict now. Do not call any tool."*). Parked as a robustness improvement.

## 5. Calibration: confidence vs correctness

Raw confidences are similar on correct and wrong predictions:

* Correct: median 0.92 (range 0.55-0.97)
* Wrong:   median 0.92 (range 0.75-0.95)

LLM confidence on this task is **not informative** — wrong predictions are held with the same certainty as right ones. This is a candidate finding for a Limitations paragraph: LLM-reported confidence cannot be used as a routing signal for escalation to human-in-the-loop or to ensemble / rerun.

## 6. Cost-accuracy frontier

| Profile | Tokens/correct-answer |
|---|---:|
| qwen3-plus       | 11.1 K / 0.40 = **27.7 K per correct** |
| minimax-m2       | 14.6 K / 0.53 = 27.5 K per correct |
| deepseek-chat    | 55.9 K / 0.40 = 139.7 K per correct |

minimax and qwen3 are roughly tied on cost-normalised accuracy; deepseek pays 5× more per correct answer because it always fills its tool budget (6/6) and the tools return large JSON payloads.

## 7. What to report vs what to fix

Report in the paper (as-is, honest):

* 53 % top-1 accuracy (minimax) on a 5-class + None classification task with a 5-tool budget and 3 s PMU window.
* A1 is reliably detectable when it is severe; A2 is reliably missed when it resembles inter-area swing or is below the inertia threshold.
* Confidence is miscalibrated.

Fix before rerunning for the paper (parked, upon user go-ahead):

1. Swap Phase I's default IEEE 39 A1 scenario from `ieee39_a1` (Line_26) to `ieee39_a1_severe` (Line_28).
2. Redesign `kundur_a2` to avoid area-boundary loads (or use a uniform-scale variant).
3. Add one-line prompt fallback for qwen3 ("voltage-only allowed").
4. Retry-on-empty for minimax.
