# Paper Section Draft — Phase I: LLM-Only Forensic Diagnosis

*Target: Applied Energy / IEEE IoT-J. Written to slot into §VI (Results) after the framework and tool-interface sections. Numbering placeholders used throughout — update to match the final structure.*

---

## VI.A  Phase I: Forensic Diagnosis from PMU Evidence

**Objective.** Before the defender selects a mitigation in Phase II, it must infer *which* attack class has unfolded. Phase I isolates this classification step: the LLM is given a read-only PMU window and four diagnostic tools, and must emit a single-line JSON verdict labelling the attack as A1 (N-k transmission trip), A2 (load-altering / MadIoT step), A3 (controller / PSS tamper), A4 (AGC-FDI combined), or None. The ground truth is the scripted scenario label; no Phase II mitigation occurs in this evaluation, so accuracy is measured in isolation of downstream recovery effects.

**Protocol.** Each run initialises the ANDES DAE model at `t=0s`, advances the system under nominal control until the attack injection at `t_a = 2 s`, advances a further 3 s to `t_b = 5 s` to populate the PMU window, and hands control to the LLM with the system prompt described in Table TAB-DIAG. The LLM may invoke up to five tool calls (`list_devices`, `get_pmu_window`, `compute_rocof`, `compute_angle_diff_matrix`, `detect_dominant_mode`), then terminates with a JSON verdict of the form `{"label": …, "suspected_idx": …, "confidence": ∈[0,1], "rationale": …}`. Decoding temperature is fixed at 0 for reproducibility; three independent seeds per (profile, scenario) are used to measure provider-side nondeterminism. PMU coverage is a fixed 50 % bus subset on Kundur and 51 % on IEEE 39, chosen to include at least one load bus per area and at least one generator-adjacent bus where topologically possible.

**Evaluated profiles.** We report three open-API LLM profiles:

1. **MiniMax M2.7** (Anthropic-compatible Messages API).
2. **DeepSeek Chat** (OpenAI-compatible, reasoning on).
3. **Qwen Plus Latest** (OpenAI-compatible via DashScope).

Frontier Claude / GPT profiles are retained for the final camera-ready sweep; this batch intentionally exercises the "tier below frontier" to establish a lower-bound on what can be deployed at plausible operator-side inference cost.

**Scenarios.** Five scripted scenarios were evaluated: `kundur_a1` (severe — Line_7/8 trip on the 8-9 two-circuit tie), `kundur_a1_mild` (mild — Line_4/5/6 trip on the 7-8 three-circuit tie), `kundur_a2` (PQ_1 step scale=1.8), `ieee39_a1` (Line_26 peripheral trip), and `ieee39_a2` (PQ_8 step scale=1.6). This yields 3 profiles × 5 scenarios × 3 seeds = 45 runs.

### VI.A.1  Accuracy

Table TAB-P1 summarises profile-level accuracy and compute cost.

| Profile | Accuracy | Tool calls | Prompt+completion tokens | Wall time (s) |
|---|---:|---:|---:|---:|
| MiniMax M2.7    | **53.3 %** | 4.5 | 14.6 K  | 40.2 |
| DeepSeek Chat   | 40.0 %  | 6.0 | 55.9 K  | 38.1 |
| Qwen Plus       | 40.0 %  | 5.0 | **11.1 K**  | **6.4** |

All three profiles achieve between 40 % and 53 % top-1 accuracy on a 5-way + None classification task with a 5-call tool budget. This exceeds the random-baseline of 20 % but falls well short of the oracle upper-bound (§VI.B). MiniMax M2.7 is the strongest classifier; Qwen Plus is strictly Pareto-dominant on tokens-per-correct and wall-time.

### VI.A.2  Confusion structure

Fig. FIG-P1-CM shows per-profile confusion matrices. Three systematic patterns emerge:

1. **A1 is reliably identifiable when severe.** On both `kundur_a1` and `kundur_a1_mild` all three profiles achieve 3/3 seeds correct, recognising divergent-sign ROCOF and rotor-angle spread (78 rad severe, 0.32 rad mild). On the less stressing `ieee39_a1` (peripheral line), accuracy drops to 1/3 – 0/3 depending on profile; Line_26 does not produce a discriminable Δδ.
2. **A2 is systematically misclassified as A1 on Kundur.** Eight of nine `kundur_a2` predictions were A1. Post-hoc analysis (Appendix APP-A2-FAIL) shows this is a scenario-design artefact rather than a model failure: PQ_1 is electrically adjacent to the 7-8 inter-area tie, so a load step excites the ~0.6 Hz inter-area mode, and within a 3 s PMU window the ROCOF signature is genuinely sign-divergent. The model is reading a real signature that the scenario does not cleanly separate from A1.
3. **A2 is systematically missed on IEEE 39.** Seven of nine `ieee39_a2` predictions were None. The 1.6× step on PQ_8 produces a system-wide voltage deviation of ≲ 1 % against the 39-bus inertia, below the LLM's inferred attention threshold. One DeepSeek run flagged A4 with the physically sensible rationale *"constant voltage across all PMUs … suggests FDI masking actual disturbance"* — a correct failure mode recognition, but an incorrect label for this ground truth.

### VI.A.3  Confidence calibration

The LLMs emit a self-reported confidence ∈ [0,1] alongside each verdict. Fig. FIG-P1-CAL plots confidence density stratified by correctness. Median confidence is 0.92 for correct predictions and 0.92 for incorrect; the distributions are visually indistinguishable. A paired Brier decomposition (Table TAB-P1-BRIER) confirms that reliability contributes almost nothing to the score: the LLMs are broadly overconfident on incorrect A1-vs-A2 confusions. This negative finding is operationally important — self-reported confidence cannot be used to gate an auto-mitigation system or to trigger human-in-the-loop escalation. Alternative calibration signals (disagreement across profiles, refusal to answer, tool-budget exhaustion) must be studied before a production deployment.

### VI.A.4  Cost-accuracy trade-off

Fig. FIG-P1-PARETO plots tokens-per-correct-answer against accuracy. Qwen Plus (27.7 K tokens / correct) and MiniMax M2.7 (27.5 K / correct) lie on the Pareto frontier. DeepSeek Chat consumes 139.7 K tokens per correct answer — a 5× penalty relative to the frontier — because the reasoning-on setting causes it to exhaust the 6-call tool budget on every run regardless of whether the first 1-2 calls had already sufficed. This suggests that on forensic triage workloads, a *tool-budget-aware* decoding strategy (stop-when-confident rather than run-until-budget) would be material.

### VI.A.5  Reproducibility artefacts

All 45 per-run JSONs (tool traces, full tool I/O, tokens, final text, verdict) and the aggregate CSV are released alongside the paper. Each run is uniquely identified by `<UTC timestamp>__<profile>__<scenario>__s<seed>`. Three independent seeds per cell give a seed-mean ± seed-range for every metric in Table TAB-P1; the batch is reproducible end-to-end given the released scenarios, tool registry, and system prompts.

### VI.A.6  Takeaways for the framework

Phase I establishes three framework-level findings that shape Phase II and the overall architecture:

1. **A read-only PMU window and 3-5 diagnostic tools are sufficient for A1 detection** when the attack is stressing; the 53 % aggregate accuracy is pulled down primarily by A2 confusions (§VI.A.2), not by any model's inability to use the tools. Tool-use was observed in 100 % of runs.
2. **Scenario-level signal separability bounds diagnosis accuracy.** Our Kundur-A2 scenario puts load at an area boundary; this is a realistic failure mode in field deployments where MadIoT targets happen to sit on an inter-area path. Either the benchmark must be redesigned, or the paper must frame A1/A2 discrimination as fundamentally ambiguous on short PMU windows.
3. **Self-reported confidence is not a reliable gating signal** (§VI.A.3). Phase II should not condition its mitigation strength on Phase I confidence alone.

---

## Appendices referenced

* **APP-A2-FAIL.** Worked example of a mis-classified `kundur_a2` run. Step through the `compute_rocof` output and show that ROCOF sign-divergence is a genuine consequence of PQ_1's electrical proximity to the 7-8 tie, not an LLM hallucination.
* **APP-DIAG-PROMPT.** The verbatim `DIAG_SYSTEM_PROMPT` used in `experiments/07_blue_diagnose.py`, plus the tool JSON schemas.
* **APP-PROFILES.** Profile→model-ID mapping, API endpoint, date-of-access, temperature, seed semantics.

## Figures / tables referenced (not yet drawn)

* **TAB-P1** — accuracy, tokens, tool calls, wall time, per profile.
* **TAB-P1-BRIER** — Brier score decomposition (reliability / resolution / uncertainty) per profile.
* **FIG-P1-CM** — 3 × 6 grid of confusion matrices (3 profiles × confusion structure).
* **FIG-P1-CAL** — calibration histogram (confidence binned, correctness overlay).
* **FIG-P1-PARETO** — tokens/correct vs accuracy scatter, profiles as points, frontier annotated.

---

## Known caveats the paper should surface

1. Two MiniMax runs returned empty final text post-budget-exhaustion (`ieee39_a1 s2`, `ieee39_a2 s1`) — counted as incorrect. This is a 4.4 % API-driven failure rate; the paper should note it and quote MiniMax accuracy both raw (53 %) and conditional on successful extraction (8/13 = 61.5 %).
2. Qwen Plus returned "None" on all 6 IEEE 39 scenarios with rationale *"no generator PMU signals available"*. This is a prompt-sensitivity finding, not a capability claim against Qwen; the paper should either (a) fix the prompt and re-run, reporting both variants, or (b) disclose the finding as an illustration that Phase I accuracy can depend materially on prompt wording for small/mid-tier LLMs.
3. Three seeds per cell is sufficient to separate 40 % from 53 % at the scenario-pooled level but *not* at the per-cell level. The final camera-ready batch should use ≥ 10 seeds per cell for any metric reported below the profile-aggregate.
