# B4 — End-to-End Red-vs-Blue Orchestrator: Implementation Spec

*This is a design document only. The file `experiments/09_red_vs_blue.py` is to be written by the project author; this spec defines structure, data contracts, invariants, and test plan so the implementation is straightforward and consistent with the existing 03/05/07 codebase.*

---

## 1. Purpose

Tie the three existing single-agent loops into one self-contained episode per seed:

```
t=0  ─── env.setup() ───→ t=t_a  ─── RED plans + executes ───→ t=t_b  ─── BLUE Phase I diagnoses ───→ BLUE Phase II mitigates ───→ t=T  ─── score
```

Each stage already has a working in-process entry point:

| Stage | Existing function | File |
|---|---|---|
| Red attack loop            | `run(...)` | `experiments/03_red_attack_llm.py` |
| Blue Phase I diagnosis     | `run(...)` | `experiments/07_blue_diagnose.py` |
| Blue Phase II oracle defender | `run(...)` | `experiments/05_blue_defend_oracle.py` |

**However** — each of those three entry points **owns its own `GridEnv`**. B4 cannot call them directly; it must run a *single persistent* `GridEnv` across all three stages so that the defender sees the grid that the red agent actually damaged. That means B4 is a new driver, not a three-way `run()` chain.

---

## 2. Control flow

```text
build env, env.setup(), env.advance_until(t_a)

# --- RED phase ---
red_client  = LLMClient(profile=red_profile)
red_reg     = build_red_registry()
red_trace, red_actions = chat_loop(
    client=red_client, reg=red_reg, env=env,
    system_prompt=RED_SYS.format(horizon=t_b, budget=red_budget),
    user_prompt=RED_USER,
    max_steps=red_max_steps, tool_budget=red_budget,
    temperature=T, hard_stop_t=t_b,
)

# --- BLUE Phase I ---
diag_client = LLMClient(profile=blue_profile)
diag_reg    = build_diagnosis_registry()     # from 07_blue_diagnose.py
verdict, diag_trace = chat_loop_readonly(
    client=diag_client, reg=diag_reg, env=env,
    system_prompt=DIAG_SYS.format(pmu_buses=..., budget=diag_budget),
    user_prompt=DIAG_USER.format(t_a=t_a, t_b=t_b, t_now=env.t),
    max_steps=diag_max_steps, tool_budget=diag_budget,
    force_final_on_exhaust=True,
)

# --- BLUE Phase II ---
# NOTE: blue Phase II gets the *LLM's own* diagnosis, not the oracle label.
mit_client = LLMClient(profile=blue_profile)           # or blue_profile_phase2 if split
mit_reg    = build_blue_registry()                     # full defender registry
mit_trace  = chat_loop(
    client=mit_client, reg=mit_reg, env=env,
    system_prompt=MIT_SYS.format(
        diagnosis=verdict,                             # ← feed self-diagnosis
        horizon=T, budget=mit_budget,
    ),
    user_prompt=MIT_USER,
    max_steps=mit_max_steps, tool_budget=mit_budget,
    temperature=T, hard_stop_t=T,
)

# --- scoring ---
report_full  = compute_stability(env, thr)             # whole trajectory
report_final = compute_stability(env, thr, t_window=(max(env.t-3.0, t_b), env.t))

record = {  ...see §5...  }
```

**Invariants to preserve**:

* `env` is constructed once per episode and advanced forward monotonically. Never re-`setup()` mid-episode.
* The red phase **must end** at `t >= t_b` even if the LLM doesn't call `advance_until`. Force a `dispatch("advance_until", {"t_end": t_b})` after the red loop exits if needed.
* Phase I runs **read-only** tools. Implementation note: Phase I's registry doesn't include `advance_until` — so time does not move during diagnosis, which is what we want.
* Phase II ends when `env.t >= T` or the mitigation tool budget is exhausted. Same forced-advance guard.

---

## 3. Data contracts

### 3.1 Red-team ground truth

The orchestrator must record what the red agent *actually* did (for Phase I accuracy scoring). Recover this from `red_trace`:

```python
def infer_attack_label_from_trace(trace: list[dict]) -> str | None:
    """First mutation-class tool call defines the attack label.
    trip_line -> A1; scale_load -> A2; alter_param -> A3; inject_fdi -> A4.
    """
    MAP = {"trip_line": "A1", "trip_gen": "A1",
           "scale_load": "A2",
           "apply_fault": "A1"}      # fault is a 3-phase short = cascade trigger
    for rec in trace:
        lbl = MAP.get(rec["name"])
        if lbl:
            return lbl
    return None                      # red never attacked -> "None"
```

*A3 and A4 aren't in the current `build_red_registry` — if they're added later, extend the map.*

### 3.2 Phase I verdict

Reuse `_extract_verdict` from `07_blue_diagnose.py`. Returns `{"label": "A1"|"A2"|"A3"|"A4"|"None"|None, ...}`. Keep `label=None` cases so the CSV can distinguish *model refused to answer* from *model answered "None"*.

### 3.3 Phase II success criterion

Reuse `05_blue_defend_oracle.py`'s recovery definition:

```python
recovery_success = bool(
    report_final.omega_deviation_max <= thr.omega_tol
    and report_final.v_min >= thr.v_floor
)
```

Where `report_final` is the stability report over the last 3 s before the horizon (see §2). Do **not** use the whole-trajectory `failed` flag for recovery — that was the false-positive issue fixed in an earlier iteration.

---

## 4. Scoring — episode-level

For each episode emit:

| Field | Meaning |
|---|---|
| `truth_label`           | Inferred from red_trace (§3.1). |
| `diagnosed_label`       | From Phase I verdict. |
| `diag_correct`          | `truth_label == diagnosed_label`. |
| `recovery_success`      | From §3.3. |
| `load_shed_total_pu`    | Sum of `shed_load` fractions × initial P in red+blue traces. |
| `omega_deviation_max`, `v_min`, `delta_spread_max` | From `report_final`. |
| `t_fail`                | From `report_full` if system failed. |
| `red_tool_calls`, `diag_tool_calls`, `mit_tool_calls` | Per-phase counts. |
| `red_tokens`, `diag_tokens`, `mit_tokens` (prompt+completion split) | Per-phase cost. |
| `red_wall_s`, `diag_wall_s`, `mit_wall_s` | Per-phase latency. |

The **joint outcome** of an episode belongs to a 2×2:

|  | recovery_success=T | recovery_success=F |
|---|---|---|
| diag_correct=T | **blue-win-clean** | diag-right-mitigation-failed |
| diag_correct=F | lucky-recovery     | **red-win** |

The 2×2 is the paper's headline table.

---

## 5. Per-run JSON artefact

Write one JSON per episode to `experiments/runs_e2e/<run_id>.json` with the full three-phase trace. Schema:

```jsonc
{
  "run_id": "20260419T113000Z__minimax-m2__qwen3-plus__kundur__s0",
  "red_profile": "minimax-m2",
  "blue_profile": "qwen3-plus",
  "case": "kundur",
  "t_a": 2.0, "t_b": 5.0, "horizon_s": 30.0,
  "temperature": 0.0, "seed": 0,

  "red":  {"model": "...", "tokens": {...}, "tool_trace": [...], "final_text": "..."},
  "diag": {"model": "...", "tokens": {...}, "tool_trace": [...], "verdict": {...}},
  "mit":  {"model": "...", "tokens": {...}, "tool_trace": [...], "final_text": "..."},

  "truth_label": "A1",
  "diagnosed_label": "A1",
  "diag_correct": true,

  "stability_full":  {...},
  "stability_final": {...},

  "recovery_success": true,
  "load_shed_total_pu": 0.12,

  "pmu_coverage": {"fraction": 0.5, "buses": [...]},
  "n_tool_calls": {"red": 4, "diag": 3, "mit": 6}
}
```

Reuse serialisation patterns from `05_blue_defend_oracle.py`.

---

## 6. Batch driver (`10_red_vs_blue_batch.py`)

Parallel to `06_blue_defend_batch.py`. Arguments:

```
--red-profiles      nargs='+'   default ['minimax-m2']
--blue-profiles     nargs='+'   default ['qwen3-plus']
--cases             nargs='+'   default ['kundur', 'ieee39']
--seeds             nargs='+' type=int  default [0,1,2]
--t-a / --t-b / --horizon / --*-budget / --*-max-steps
--temperature       0.0
--tag
```

Sweep is 4-D: `len(red) * len(blue) * len(cases) * len(seeds)`. Write incremental CSV rows to `summaries/e2e_batch_<stamp>_<tag>.csv`; print a grouped summary at the end broken out by `(red_profile, blue_profile, case)` → (recovery%, diag_acc%, shed_pu, tokens, wall).

---

## 7. Prompts — placeholders to fill in

Keep these strings small; both Kundur and IEEE 39 ideally share the same prompt.

* `RED_SYS`: Adapt from `03_red_attack_llm.py::RED_SYSTEM_PROMPT`. Critical: *"When finished attacking, do not call `advance_until` past `t_b={t_b}`. The defender will act after `t_b`."* The existing 03 prompt tells the red to advance to horizon; B4 must cap it at `t_b`.
* `DIAG_SYS`: Copy verbatim from `07_blue_diagnose.py::DIAG_SYSTEM_PROMPT`.
* `MIT_SYS`: Adapt from `05_blue_defend_oracle.py`'s system prompt but substitute the **diagnosed** label (not the oracle label) into the description. Critical: *"Your Phase I diagnosis believes the attack is {diagnosed_label} (confidence {confidence:.2f}, rationale: {rationale}). Phase I may be wrong — cross-check with fresh PMU evidence before committing to a mitigation."*

---

## 8. Edge cases the orchestrator must handle

1. **Red refuses to attack.** `red_trace` contains no mutating call. Set `truth_label = "None"`. Phase I should correctly say None; Phase II should correctly be a no-op.
2. **Red crashes ANDES** (e.g., severe fault with no recovery possible). `env.advance_until(t_b)` raises or `env.failed=True`. Record stability, short-circuit Phase I/II with null verdicts and recovery=False.
3. **Phase I returns `label=None` with low confidence.** Phase II proceeds with `diagnosed_label=None`. The mitigation LLM should treat this as "uncertain — take conservative actions (modest shed, no trip)". Include this instruction in `MIT_SYS`.
4. **Phase II runs out of budget before horizon.** Force-advance to `T` with `tool_choice="none"`; stability is reported at `T` regardless. Mark `stopped_reason = "phase2_budget_exhausted"`.
5. **Empty final-text** on any LLM (seen on MiniMax). On Phase I, verdict is `{label: None}` — OK. On Phase II, it means the model emitted no summary; that's fine, stability is measured from env state not from text.
6. **Red profile == Blue profile.** Allowed — the paper may want a same-LLM red-vs-blue comparison to isolate asymmetry of information (white-box vs PMU) from asymmetry of capability. No code change needed; just document.

---

## 9. Test plan (run before the batch)

1. **Smoke** (5 min): `kundur`, `seed=0`, `red=minimax-m2`, `blue=qwen3-plus`. Expect: red trips a line, Phase I says A1, Phase II sheds some load, recovery_success depends on severity. Eyeball the JSON.
2. **Null-attack smoke**: force red to return no tool call (send it a system prompt like "Observe only; do not attack."). Verify `truth_label=None`, Phase I says None, Phase II no-ops, `recovery_success=True`.
3. **Red-crashes smoke**: force red to `trip_line(Line_7)` (severe Kundur A1). Expect env crash or very negative recovery; verify driver doesn't hang.
4. **Consistency with stand-alone drivers**: run `07_blue_diagnose.py --scenario kundur_a1 --seed 0` and compare to a B4 run where red fires `kundur_a1` via a scripted-red shortcut (§10). Phase I accuracy should be statistically identical.

---

## 10. Scripted-red shortcut (optional but recommended)

To compare "LLM-vs-LLM" against "oracle-attack-vs-LLM", accept a `--red-mode={llm,scripted}` flag. In `scripted` mode, skip the red LLM loop and call `run_scenario(env, scenario_name)` directly. This gives a controlled A/B to isolate defender performance from attacker variability. This is the same `run_scenario` imported in 07 — zero new tools required.

---

## 11. What *not* to do (keep scope tight)

* **Do not** multi-turn red and blue interleaved. B4 is sequential per the paper's framework figure; interleaved control is a separate paper.
* **Do not** implement a "validator" agent to police LLM tool calls. The existing tool dispatch layer already refuses illegal idx / out-of-range args. A validator is on the future-work list.
* **Do not** change any file under `llmad/tools/`. B4 is pure driver composition.
* **Do not** add new scenarios, new prompts beyond §7, or new metrics beyond §4 in this PR. Those belong in follow-ups.

---

## 12. Acceptance criteria

B4 is done when all of these hold:

* `python experiments/09_red_vs_blue.py --red-profile minimax-m2 --blue-profile qwen3-plus --case kundur --seed 0` runs end-to-end, prints the 2×2 outcome, and writes `runs_e2e/<run_id>.json`.
* `python experiments/10_red_vs_blue_batch.py --red-profiles minimax-m2 --blue-profiles qwen3-plus --cases kundur --seeds 0 1 2` runs 3 episodes and prints a grouped summary.
* All four §9 smoke tests pass manually.
* No changes to `llmad/tools/*` or `llmad/scenarios.py`.

---

## 13. Estimated code size

* `09_red_vs_blue.py`: ~200-250 LOC (three chat-loop helpers, a scoring section, argparse).
* `10_red_vs_blue_batch.py`: ~150 LOC (follows `06_blue_defend_batch.py` 1:1).
* Shared chat-loop helpers (`_chat_until_horizon`, `_chat_readonly`): could live inline in `09` for the first cut; factor out once `09` and `10` both want them.

Most of the code will be assembly / glue. The hard design decisions — tool registries, scenario labels, recovery criteria, PMU coverage, prompt structure — have already been made in 03/05/07.
