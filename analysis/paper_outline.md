# Paper Outline — LLM Red/Blue Team in Power-Grid Transient Stability

**Working title (long)**: *When the Operator is a Language Model: Adversarial Evaluation of LLM-Driven Power-Grid Defence Under Stealth and Piggyback Threats*

**Working title (short)**: *LLMs vs. LLMs on the Grid: A Red/Blue Evaluation of Language-Model Operators under Piggyback Attack*

**Target venues (primary → fallback)**:
1. Applied Energy (IF ≈ 11) — applied systems energy, accepts empirical studies
2. IEEE Internet of Things Journal (IF ≈ 8) — cyber-physical angle fits
3. IEEE Trans. Smart Grid — classical fit but space tighter
4. (arXiv pre-print in parallel regardless)

---

## 1. Hook & Positioning (for Introduction)

### The shift worth writing about

Grid control centres and bulk-power SOC tools are being augmented with LLM-based agents — for alarm triage, incident summarisation, and increasingly for first-line operational suggestions. The literature on this trend is bifurcated:
- **Optimistic work** showing LLMs can correctly interpret alarms, classify faults, and propose textbook mitigations on curated benchmarks.
- **Cyber-physical security work** on classical adversarial ML (MadIoT, FDIA, measurement poisoning) that treats defenders as fixed policies.

Almost nothing empirically evaluates LLM *defenders* under LLM *attackers* on a realistic transient-stability horizon. This is the gap.

### Why now

Two threads converge:
1. LLMs with tool-use capability (function calling) can actually operate simulators, read measurements, and issue control actions — not just describe them. This makes closed-loop evaluation possible.
2. Attackers in the real world rarely cause raw instability out of nothing; they exploit **legitimate disturbances** as cover. Any defender — human or LLM — must distinguish "benign post-fault dynamics" from "attacker exploiting post-fault dynamics." This is the *piggyback* threat pattern, and it is precisely where rule-based classifiers fail.

### Our contributions (three bullets)

- **C1 — Closed-loop LLM-vs-LLM evaluation harness on ANDES**, with per-phase tool layers for attacker, diagnostician, and mitigator, producing standardised `ScenarioResult` / `stability_*` artefacts per run. Public release on acceptance.
- **C2 — A piggyback threat-model formalisation** with a primary-fault library whose members are individually recoverable (validated empirically, `P(fail | p) = 0`) but jointly unrecoverable with LLM attackers overlaid. This isolates the marginal lethality of *attack craft* from primary severity.
- **C3 — Empirical results on two canonical cases (Kundur 2-area, IEEE-39)** showing that (a) a classical rule-based UFLS defender recovers **0/15** piggyback runs — it either never reaches defender-handoff or polls omega and finds no frequency sag because the piggyback failure mode is voltage/angle-driven — so every LLM defender strictly dominates the rule-based floor by ≥27 pp; (b) within the LLM band, defender-model choice shifts grid survival by **13 pp pooled** (40 % dsk/qwen vs 27 % minimax) and up to 70 pp per primary-fault cell under identical adversary; (c) piggyback-style prompts evade rule-based diagnosis tautologically (**0/90 labellable** at n=30 per blue) while degrading recovery versus the naive baseline; and (d) an **"engagement-is-destiny" pattern**: across 90 piggyback runs, defenders that invoked ≥2 diagnosis/mitigation tools recovered **32/33 (97.0 %)**, those that did not recovered **0/57**. MiniMax-m2's 26.7 % recovery is almost fully explained by a 73 % zero-engagement rate, not by reasoning quality once engaged. An engagement-floor check is a cheap deployment-readiness screen.

---

## 2. Threat Models

### 2.1 Baseline attacker (A1/A2/A3/A4)

Direct physical disturbances delivered via tool calls: line trip (A1), load ramp / MadIoT step (A2), generator trip (A3), bus fault (A4). These are the usual N-k events; diagnosis heuristics pattern-match red's tool trace to derive `truth_label`.

### 2.2 Piggyback attacker

- A **primary event** *p* fires at *t = t_fault*: a bolted bus fault or load step drawn from a library pre-screened for recoverability.
- The **LLM red agent** begins acting at *t_a* (≥ *t_fault*), observing the post-fault transient and injecting its own disturbance that is intended to (i) exploit the primary's dynamics and (ii) appear consistent with continued natural recovery.
- Red's system prompt explicitly rewards stealth: it is told the defender is rule-based and pattern-matching, and instructed to combine actions so that simple trace-pattern classification fails.
- Blue agent takes over at *t_b*; horizon extends to *T*. Outcome is scored on `stability_final` (last 3 s of *T*) against fixed thresholds (|ω| ≤ 0.02 pu, V ≥ 0.85 pu).

### 2.3 Why piggyback is the paper's centrepiece

- It is *the* operationally realistic threat. Real attackers do not crash the grid out of thin air; they wait for a line trip, a storm, a load spike, and piggyback.
- It breaks the diagnostic paradigm most operators still rely on (rule-based plus intuition). Our empirical finding (3/15 baseline-attack runs triggered classification vs 0/45 piggyback runs) is the punchline.
- It requires LLMs on both sides. Rule-based attackers cannot compose behaviour; rule-based defenders cannot reason about intent. So the threat model *needs* the medium the paper studies.

---

## 3. System Architecture (Methods)

### 3.1 Simulator — ANDES TDS

- Full DAE time-domain integration. IEEE-39 and Kundur 2-area cases, default dynamic data, default controllers (exciters, governors). Fixed TDS step size.
- `GridEnv` wrapper exposes `advance_until(t)`, pre-allocated fault slot bank (8 slots, dummy far-future `tf`), `queue_fault` / `flush_pending_faults` for strictly-future event scheduling, and `observe()` returning a bounded PMU window.

### 3.2 Tool layer (per phase)

- **Red tools**: `list_devices`, `get_pmu_window`, `advance_until`, `trip_line`, `scale_load`, `apply_fault`, `trip_generator` (subset depending on attack space).
- **Diagnosis tools** (Phase I): analysis primitives only — `compute_rocof`, `compute_angle_diff`, `detect_dominant_mode`, `get_pmu_window`. No actuation.
- **Mitigation tools** (Phase II): `load_shed_step`, `fault_clear`, plus analysis tools.

Each phase has a per-run token budget and a max-tool-call budget to prevent runaway loops.

### 3.3 Scoring

- `recovery_success`: stability_final stable under thresholds.
- `diag_correct`: heuristic label from red's tool trace vs blue's Phase-I verdict. **(For piggyback we note that this metric is degenerate and substitute the `piggyback_effective` metric defined below.)**
- `piggyback_effective` *(new, this paper)*: `red_mode == llm AND red_tool_calls > 0 AND recovery_success == False` — a direct "did the attacker accomplish lasting harm" measure that does not depend on heuristic pattern-matching.
- `blue_engaged`: `diag_tool_calls + mit_tool_calls > 1` (i.e., the defender actually called analysis or mitigation tools beyond the forced advance).
- `marginal_lethality(p, config)`: `P(fail | p, config) − P(fail | p, null)`.

### 3.4 Experimental design

- Cases: Kundur 2-area (11 buses), IEEE-39 (39 buses).
- Models: DeepSeek-Chat (baseline reasoner), Qwen3-Plus, MiniMax-M2.
- Kundur batch (full 3 × 3 LLM × 5 seeds + 15 scripted): measures diagnostic capability and recovery under A1/A2 direct attacks.
- IEEE-39 piggyback batch (L-design, 60 LLM runs + 15 null baseline): measures piggyback lethality and defender heterogeneity.

---

## 4. Results

### 4.1 Primary-library validation (screening)

One figure: P(fail | p alone) for each candidate primary. Expose which primaries enter the "viable" band `[0, 0.2]` — *bf_b16_80ms*, *bf_b22_80ms*, *bf_b29_100ms*. Discuss the rejected ones (*bf_b02_60ms*, *load_step_pq08_+25%*) as too severe alone. Acknowledge: load-step family is not yet represented in the viable library; camera-ready should add a milder step.

### 4.2 Kundur — LLM-vs-LLM direct attacks (result #1)

Table: TAB-K-RECOV with recovery%, diag%, shed, outcome distribution over 3 × 3 red/blue × 5 seeds. Commentary: defender-model choice already shows a 40-pp swing here, setting up IEEE-39 piggyback as the harder test.

### 4.3 IEEE-39 piggyback — LLM defenders strictly dominate the rule-based anchor (result #2)

- **Headline table**: TAB-K-RECOV for {B0 null (100 %, n=15), **BU piggyback × scripted-UFLS (0.0 %, n=15)**, C1 baseline-red × dsk (50.0 %, n=30), C2 piggyback-red × dsk (40.0 %, n=30), piggyback-red × qwen3 (40.0 %, n=30), piggyback-red × minimax (26.7 %, n=30), C3 qwen-red × {dsk, qwen3, minimax} (66.7 % / 100.0 % / 93.3 %, n=15 each)}.
- **Marginal lethality heatmap**: primary × config cells, 3 × 8 matrix (incl. UFLS column and C3 sub-columns).
- **Key observation 1 — UFLS floor**: every LLM defender, even the weakest, beats the classical rule-based defender by ≥27 pp. Red pre-empts UFLS in 9/15 runs (grid dies before handoff); in the remaining 6/15, UFLS polls omega but **never fires a shed stage** because the piggyback failure mode is voltage/angle-driven, not frequency-driven. Classical UFLS is structurally blind to the instability mode.
- **Key observation 2 — LLM band (strong-attacker, dsk-red)**: fixing red (deepseek + piggyback), fixing primary, fixing seed — varying only the blue defender gives 10 %–80 % recovery per cell, 26.7 %–40.0 % pooled. The two engaging defenders (dsk, qwen) tie at 40 %; the disengaging defender (minimax) sits 13 pp below. *Note: earlier n=5 numbers placed qwen at 53.3 %; the seed bump to n=10 revealed this as sampling noise.*
- **Key observation 3 — attacker capability matters**: swapping red from `deepseek-chat` to `qwen3-plus` under the identical piggyback prompt collapses the attack: pooled blue recovery jumps from 35.6 % (C2) to 86.7 % (C3, 39/45). Only `bf_b29 × dsk-blue` remains red-dominant. The piggyback prompt is **necessary but not sufficient** for lethality; attacker instruction-following capability is the second requirement.

### 4.4 Piggyback prompt vs. baseline prompt (result #3, matched-N)

Holding red-model = deepseek-chat and blue = deepseek fixed, both arms at n=30:
- Piggyback reduces truth-labelable attacks from 5/30 → 0/30 (stealth claim).
- Piggyback reduces recovery by **10.0 pp** (50.0 % → 40.0 %; lethality claim) — within matched-N CI.
- **Per-primary effect is heterogeneous**: the b16/b22 drop (7-8 → 2-3 per 10) is a clean attacker-capability signal — piggyback moves recovery decisively on recoverable-looking primaries. The b29 column runs the other way (0/10 under baseline vs 7/10 under piggyback) for a different reason: L3-B log forensics (§appendix) show the b29 column under baseline-red is driven by a simulator/step-boundary artefact — a singular-Jacobian stiffness at t≈10.033 s interacting with dsk-blue's split mit advance pattern. The b29 column should therefore be annotated as "numerical-integration-sensitive" rather than reported as a symmetric inversion.

→ Stealth does not come at a cost to lethality. The prompt's real action is to shift the attack surface toward recoverable-looking primaries (b16/b22) where the defender would otherwise succeed.

### 4.5 The engagement-is-destiny result (result #4)

- **Pooled across all 90 C2 runs + 45 C3 runs (n=135)**: engaged (≥2 diag/mit calls, excluding forced advance) recovers **71/77 (92.2 %)**; disengaged (≤1 call) recovers **0/58 (0 %)**. Pattern is invariant to attacker model.
- Per-blue under C2 (dsk-red, n=30): dsk 12/13 vs 0/17 (engaged-zero is 17/30 = 57 %); qwen 12/12 vs 0/18 (18/30 = 60 % zero); minimax 8/8 vs 0/22 (22/30 = 73 % zero). MiniMax-m2's lowest pooled recovery (26.7 %) is almost fully mediated by this propensity — conditional on engaging, it recovers 100 %.
- Per-blue under C3 (qwen-red, n=15): dsk 10/15 vs 0/0; qwen 15/15 vs 0/0; minimax 14/14 vs 0/1. The weaker attacker forces engagement itself to go up (because there is more to observe), which mechanistically explains the jump in headline recovery.
- **Cross-platform reproduction on Kundur (L3-A, 2026-04-20, n=45)**: all 45 engaged, 45/45 recovered (100 % across all three blues). The IEEE-39 disengagement pathology does not manifest on the smaller system — disengagement is triggered by information density, not a fixed model trait. The *engaged → recov* side of the rule is reinforced across platforms (77/78 = 98.7 %); the *disengaged → fail* side is only testable on IEEE-39.
- Token-vs-wall scatter (fig 3) shows the disengaged cluster visibly at the origin (short wall, low completion tokens, zero tool calls).
- Prescription: a simple engagement-floor test is a cheap deployment-readiness check; blues that never call any diagnostic tool on a real-world transient are a de-facto zero-defence policy regardless of nominal capability.

### 4.6 Qualitative analysis

- Representative transcript excerpts (one per blue model) showing characteristic reasoning patterns — qwen3 willing to shed, dsk over-cautious, minimax exiting with "system appears nominal" under a collapsing grid.
- Brief discussion of how piggyback red framed its attack in natural language; where the text disguised the tool call.

### 4.7 Oracle ceiling — perfect-attribution upper anchor (result #5)

- **Oracle defender**: t=0 state snapshot; at t_b=5 s, reverse every red-induced delta (`Line.u`, `PQ.p0/q0`, `GENROU.u`, `GENCLS.u`) and execute a single `advance_until(30)` — no LLM reasoning, no tool budget. This is the L3-C upper anchor, analogous to MPC-with-full-state but without the multi-step optimisation.
- **Ceilings**: ORACLE-C1 (baseline-dsk × oracle, n=15) = **40.0 %**; ORACLE-C2 (piggyback-dsk × oracle, n=30) = **70.0 %**; ORACLE-C3 (piggyback-qwen × oracle, n=15) = **93.3 %**.
- **Headroom**: best LLM blue on pb-dsk = 40 %; best LLM blue on pb-qwen = 100 % / 93.3 % on dsk-blue × qwen-red, pooled 86.7 %. Gap to ceiling on dsk-red = **+30 pp**; gap on qwen-red = **+7–53 pp** (depending on blue). This is the recoverable headroom a better diagnosis / mitigation pipeline could plausibly capture.
- **Non-monotonic on baseline-red**: on ORACLE-C1, oracle (40 %) is *below* LLM-dsk-blue (50 %). Mechanism: baseline-red occasionally trips a single line; oracle's `reclose_line` reversal at t=5 collides with the post-primary transient (seeds with `reversed 1 reclose_line → NONCONV at t=5.03`). LLM blue's pattern-matched mitigation dodges this numerical cliff. **Oracle is a ceiling in the asymptotic "no-reversal-needed" regime but not a universal upper bound.** This is an honest limitation to report: perfect attribution is not strictly better than conservative heuristics when the reversal operator is itself numerically fragile.
- **Per-primary**: on dsk-red, oracle recovers 9/10 on b29 (vs LLM best 8/10), consistent with the b29 stiffness being dodged by a single-shot advance rather than a split advance. On b16/b22, oracle recovers 12/20 vs LLM best 5/20 — the clean capability headroom lives here.
- **Interpretation for §2 thesis**: LLM blue is materially above the classical-rule floor (+27 pp over UFLS) and materially below a perfect-attribution ceiling (−30 pp on the hard attacker, −7 to −53 pp on the weak attacker depending on blue). The recovery-band story now has all three anchors.

---

## 5. Limitations

- All headline arms now matched-N: B0/BU=15, C1=30, C2=30 per blue, C3=15 per blue. Matched-N dsk-red × dsk-blue: piggyback is 10 pp more lethal than baseline; the b16/b22 component is a clean capability signal, the b29 column is confounded by a numerical-integration artefact at t≈10.033 s (camera-ready should either drop b29 from the matched-N headline or fix the API to smooth the stiff boundary).
- Attacker-model effect: C3 (qwen3-red × piggyback) pooled recovery is 86.7 % vs C2's 35.6 %. The paper must condition the "piggyback is dangerous" claim on attacker capability. A third red model (e.g. Claude Sonnet) would strengthen the 3-point attacker-capability curve, but the dsk↔qwen swing is already large enough to state the dependency.
- Primary library contains only bolted faults in the viable band; load step is missing. A scale = 1.15 variant is in the next pass.
- Rule-based UFLS anchor (T2d) landed at 0/15 recovery and reveals the piggyback failure mode is not frequency-driven (UFLS polled but never fired in the 6/15 runs that reached handoff). Perfect-attribution oracle anchor (L3-C) landed at 70.0 % / 93.3 % against dsk-red / qwen-red piggyback, establishing a +30 to +53 pp headroom above the best LLM blue. The recovery-band floor-to-ceiling spread on IEEE-39 piggyback is now 0 % → 40 % → 70 % (UFLS → LLM-dsk-blue → oracle on dsk-red).
- MiniMax result should be cross-validated with the minimax-m2 blue on Kundur piggyback before being promoted to a general claim; the engagement-floor recommendation already holds cross-model on IEEE-39.

---

## 6. Related Work (skeleton)

- LLMs for power-grid decision support (alarm triage, textbook QA) — cite recent workshop papers, IEEE SmartGridComm 2024–25 tracks.
- Cyber-physical adversarial ML on grids: MadIoT (Soltan et al. 2018), FDIA (Liu/Ning/Reiter), measurement poisoning.
- Agentic evaluations in other cyber-physical domains — autonomous driving red-teaming, robotic manipulation evals. Motivate the transferable methodology.
- Transient-stability simulation frameworks — ANDES (Cui, Zhang 2020), PSS/E, PowerWorld. Differentiate our tool-layer wrapping.

---

## 7. Artefact release (for the paper, reviewers, and future work)

- `llmad/` Python package with `GridEnv`, tool layers, scenario library, metrics.
- Reproducibility: seeded runs, fixed ANDES version, pinned LLM profile configs, raw CSV + JSON per run.
- Supplementary: transcripts, figure generation scripts, primary-screening CSVs.

---

## Suggested figure budget (for a ~10-page submission)

| # | Figure | Purpose | Status |
|---|---|---|---|
| 1 | Architecture schematic | Explain red/diag/mit pipeline, handoffs | not started |
| 2 | Primary-screening scatter | P(fail | p alone) per candidate | pending (data exists) |
| 3 | TAB-K-RECOV bar chart | Headline recovery comparison | in progress |
| 4 | TAB-MLET heatmap | Primary × config marginal lethality | in progress |
| 5 | Minimax disengagement scatter | Token vs wall, coloured by blue | in progress |
| 6 | Representative PMU trace with annotations | One piggyback run, show primary + red intervention | not started |
| 7 | Kundur diag-confusion matrix (supplementary) | 3 × 3 red/blue diag% | data exists |

---

## Open questions before submission (tracked in TODO)

1. Which target venue? (AE vs IEEE-IoT-J vs Trans Smart Grid — scope & format differ).
2. Is the minimax "disengagement" a general LLM risk or model-specific? Needs Kundur-piggyback cross-validation.
3. How rigorous a rule-based UFLS / oracle baseline do reviewers expect? Minimum viable is a 3-step frequency-based UFLS at fixed thresholds; fancier is MPC-lite.
4. Public transcripts — OK with leaking token-level prompts, or paraphrase? (operational-security framing.)
5. Ethics / dual-use statement — what does the venue require? (AE is lighter than IEEE security venues.)
