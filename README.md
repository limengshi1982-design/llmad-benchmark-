# LLMAD-Benchmark

**LLM red/blue team benchmark on ANDES time-domain grid simulation.**
Six attacker × four defender matrix, piggyback threat model, 628 matched-*N* runs on IEEE 39-bus and Kundur 2-area systems.

---

## TL;DR

This repository is a closed-loop adversarial benchmark for Large Language Model (LLM) agents on bulk-power-system transient stability. We place one LLM on each side of a time-domain differential–algebraic-equation (DAE) simulator (ANDES), expose grid state through a validated tool interface, and measure what happens when the attacker is instructed to disguise its actions as legitimate operator response to a natural primary fault — the *piggyback threat model*. We run 628 matched-*N* experiments across six LLM attackers and four LLM defenders on IEEE 39-bus and Kundur 2-area systems. The findings are summarised in the paper companion to this repo.

---

## Key Findings (headline)

1. **Piggyback stealth.** A rule-based attack classifier tags **3/30** baseline-prompt attacks but **0/90** piggyback attacks.
2. **Engagement is destiny.** Defenders invoking ≥ 2 diagnostic or mitigation tools recover **93 % (114/122)** under mild attackers; disengaged defenders recover **0 % (0/58)**.
3. **Attacker lethality is bimodal, not capability-graded.** Claude Opus 4.7 (12 %) and DeepSeek-chat (36 %) cluster as *lethal*; Qwen3.6 (70 %), Sonnet 4.6 (84 %), and GPT-5.4 (87 %) cluster as *mild*. The 30–75 pp gap is uncorrelated with flagship capability rank.
4. **Red-dominance regime.** Under Opus-red no tested blue exceeds **25 %** recovery; the *optimal blue inverts* — conservative MiniMax (27 % worst under DeepSeek-red) becomes the best defender (21 %) in this regime.
5. **Oracle-ceiling inversion.** GPT-5.4-blue recovers **96.7 %** against DeepSeek-red, *exceeding* the perfect-attribution oracle (70 %) by +26.7 pp, invalidating the naive "oracle is a ceiling" framing.
6. **Null causal test for the alignment-gap hypothesis.** An abliterated (safety-stripped) Qwen3.5 27B as red gives pool 66.7 – 75.5 %, indistinguishable from aligned Qwen3.6 — piggyback lethality is *not* explained by RLHF-refusal-pathway suppression alone.

Full table and per-primary breakdowns are in `analysis/piggyback_main_ieee39.md` and the paper's §4.

---

## Why this benchmark exists

Prior LLM-agent work in power systems has treated the LLM as a UX layer (alarm explanation, compliance QA, EMS command translation) above the physics. Prior cyber-physical attack literature does not cover the LLM-interface layer. Prior LLM-safety literature has not evaluated alignment under closed-loop cyber-physical stress. Prior agent-evaluation benchmarks do not offer physics-grounded adversarial cells. This repo is the first dataset populating that intersection.

---

## Architecture

```
 ┌──────────────────┐ attacks   ┌────────────────┐
 │ red LLM (agent)  │──────────▶│   ANDES TDS    │
 │  skill tools     │           │  DAE solver    │
 └────────▲─────────┘           └──────┬─────────┘
          │ partial PMU               │ state
          │                            ▼
          │          ┌──────────────────────────┐
          │          │ validator (schema + range)│
          │          └────────────┬─────────────┘
          │                       ▼
          │          ┌──────────────────────────┐
          └──────────│   blue LLM (defender)    │
                     │  Phase I diagnose        │
                     │  Phase II mitigate       │
                     └──────────────────────────┘
```

**Tool layer (13 skills, implemented):**

- *Red:* `list_devices`, `get_measurements`, `trip_line`, `trip_gen`, `scale_load`, `apply_fault`, `advance_until`
- *Blue:* `list_devices`, `get_pmu_window`, `compute_rocof`, `compute_angle_diff_matrix`, `detect_dominant_mode`, `shed_load`, `reclose_line`, `trip_gen_defense`, `advance_until`

Every tool call passes through a schema validator that rejects out-of-range arguments before execution (97.4 % first-pass valid across ~3,100 calls).

---

## Repo Layout

```
llmad-benchmark/
├── llmad/                     # core library
│   ├── tools/                 # red/blue skill definitions + validator
│   ├── scenarios.py           # primary-fault library
│   ├── llm_client.py          # OpenAI/Anthropic SDK wrapper (multi-vendor)
│   └── ...
├── config/
│   └── models.json            # profiles for 7 vendor models (6 reds + 4 blues panel)
├── cases/                     # ANDES case files (Kundur, IEEE 39-bus)
├── scenarios/                 # primary-fault + attack scenario XLSX
├── agents/                    # red/blue prompt templates (incl. piggyback)
├── experiments/
│   ├── 09_red_vs_blue.py      # single-episode runner
│   ├── 10_red_vs_blue_batch.py# matrix sweep (used for all C2–C11 batches)
│   └── summaries/             # 628-run CSV results (the authoritative data)
├── analysis/
│   ├── piggyback_main_ieee39.md  # findings doc, structured analysis
│   ├── paper_draft_v1.md         # prose sibling of the manuscript
│   ├── plot_redblue_heatmap.py   # Fig 4 (6×4 matrix)
│   ├── plot_blue_inversion.py    # Fig 5 (blue-strategy inversion)
│   └── figures/
├── manuscript/
│   ├── main.tex              # IEEE IoT-J / Applied Energy submission (11 pp)
│   ├── references.bib
│   ├── figures/
│   └── main.pdf              # compiled submission
└── docs/
```

---

## Quickstart

### Install

```bash
git clone https://github.com/limengshi1982-design/llmad-benchmark-.git
cd llmad-benchmark-
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.11+, `andes`, `openai`, `anthropic`, `pandas`, `numpy`, `matplotlib`, `scipy`.

### Configure API keys

Copy `.env.example` to `.env` and fill in vendor keys:

```bash
DEEPSEEK_API_KEY=sk-...
DASHSCOPE_API_KEY=sk-...        # for Qwen3-Plus (DashScope)
MINIMAX_API_KEY=sk-...
OPENCODE_ZEN_KEY=sk-...         # for GPT-5.x, Claude Opus/Sonnet via Zen gateway
OLLAMA_LOCAL_API_KEY=ollama     # for local Ollama (Qwen3.6, uncensored variants)
```

Ollama-hosted models (Qwen3.6, Qwen3.5-uncensored) expect a running Ollama daemon at `http://localhost:11434`. For docker:

```bash
docker run -d --gpus all -p 11434:11434 -v ollama:/root/.ollama --name ollama ollama/ollama
docker exec ollama ollama pull qwen3.6:latest
docker exec ollama ollama pull jaahas/qwen3.5-uncensored:27b
```

### Run a single episode

```bash
python experiments/09_red_vs_blue.py \
  --red-profile deepseek-chat \
  --blue-profile gpt-5.4 \
  --case ieee39 \
  --seed 0 \
  --primary-fault primary_ieee39_bf_b16_80ms \
  --prompt-variant piggyback
```

### Reproduce a batch cell

E.g. C10 (Qwen3.6 red × 4-blue panel, 60 runs):

```bash
python experiments/10_red_vs_blue_batch.py \
  --red-profiles qwen3.6-local \
  --blue-profiles deepseek-chat qwen3.6-local minimax-m2 gpt-5.4 \
  --cases ieee39 \
  --seeds 0 1 2 3 4 \
  --primary-faults primary_ieee39_bf_b16_80ms primary_ieee39_bf_b22_80ms primary_ieee39_bf_b29_100ms \
  --prompt-variant piggyback \
  --temperature 0.2 \
  --tag repro_c10
```

Results are written incrementally to `experiments/summaries/e2e_batch_<timestamp>__<tag>.csv`.

### Regenerate figures

```bash
python analysis/plot_piggyback_headlines.py   # Fig 1 (4-anchor recovery bars)
python analysis/plot_redblue_heatmap.py       # Fig 4 (6×4 matrix)
python analysis/plot_blue_inversion.py        # Fig 5 (3-panel inversion)
```

### Compile paper

```bash
cd manuscript
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Data Release

All 628 runs are in `experiments/summaries/` as per-batch CSVs (40+ fields each: red/blue profile, primary, seed, recovery_success, tool-call counts, token counts, per-phase prompt/completion tokens, wall-time, stopped_reason, errors). These CSVs are the authoritative dataset used to produce every number and figure in the paper.

Per-run JSON state snapshots (`experiments/runs_e2e/`, ~41 MB) are **not** in this repo by default — they are planned for a separate Zenodo archive for forensic-grade reproducibility.

---

## Model Profiles Tested

| Role | Model | Vendor | Access |
|---|---|---|---|
| Red & Blue | DeepSeek-chat | DeepSeek | Direct API |
| Red & Blue | Qwen3-Plus | Alibaba DashScope | Direct API |
| Red & Blue | MiniMax-m2 | MiniMax | Direct API (Anthropic-compat) |
| Red & Blue | GPT-5.4 | OpenAI | OpenCode Zen gateway |
| Red & Blue | Claude Opus 4.7 | Anthropic | OpenCode Zen gateway |
| Red & Blue | Claude Sonnet 4.6 | Anthropic | OpenCode Zen gateway |
| Red & Blue | Qwen3.6 (local) | Alibaba (Ollama build) | Local Ollama, RTX 4090 |
| Red only | Qwen3.5-uncensored 27B | jaahas community build | Local Ollama, B1 causal test |

See `config/models.json` for base URLs, model IDs, and notes.

---

## Citation

If you use this benchmark or its findings, please cite:

```bibtex
@article{li2026llmad,
  author  = {Li, Mengshi},
  title   = {LLM Red/Blue Team Under Time-Domain Grid Simulation:
             A Piggyback Threat Model and a Four-Anchor Recovery Band},
  journal = {IEEE Internet of Things Journal (submitted)},
  year    = {2026}
}
```

(BibTeX will be updated with final venue + DOI on acceptance.)

---

## Licence

Code: **MIT Licence**.
Data (CSV summaries, prompt templates): **CC-BY-4.0**.

Model weights, API responses, and vendor-specific artefacts are subject to their respective terms of service. See `LICENSE` and `LICENSE-data` for details.

---

## Contact

Mengshi Li · limengshi1982@gmail.com
Issues and pull requests welcome on GitHub.
