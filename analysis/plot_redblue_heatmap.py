"""Fig 4: 5-red × 4-blue recovery heatmap.

Rows = red attackers (sorted by pooled lethality, top = most lethal).
Cols = blue defenders.
Cell = recov %, annotated with (n).
Divergent palette centred at 50 %.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUMM = ROOT / "experiments" / "summaries"


def _newest(pat: str) -> Path | None:
    matches = sorted(SUMM.glob(pat))
    return matches[-1] if matches else None


def _load(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ---- load every red×blue source CSV we have ----
C2 = _load(_newest("*piggyback_main_c2*.csv"))
C2_N10 = _load(_newest("*piggyback_main_c2_n10*.csv"))
C2_MMX_RERUN = _load(_newest("*piggyback_minimax_rerun*.csv"))
C2_B29_MMX_RETRY = _load(_newest("*piggyback_b29_mmx_retry*.csv"))
C3 = _load(_newest("*piggyback_main_c3*.csv"))
C3_QWEN = _load(_newest("*piggyback_c3_qwen_red*.csv"))

GPT_C2 = _load(_newest("*gpt54_C2_pbdsk*.csv"))
GPT_C3 = _load(_newest("*gpt54_C3_pbqwen*.csv"))
GPT_C4 = _load(_newest("*gpt54_C4_redgpt*.csv"))
GPT_C5 = _load(_newest("*gpt54_C5_self*.csv"))

OPUS_PILOT = _load(_newest("*claude47opus_C6_redopus*.csv"))
OPUS_SCOPED = _load(_newest("*claude47opus_C6_27*.csv"))
OPUS_GPT = _load(_newest("*claude47opus_C8_gptblue*.csv"))

SONNET = _load(_newest("*claude46sonnet_C7_redsonnet*.csv"))
SONNET_GPT = _load(_newest("*claude46sonnet_C9_gptblue*.csv"))
QWEN36_C10 = _load(_newest("*qwen36_C10_replace_c3*.csv"))
QWEN35UNC_C11 = _load(_newest("*qwen35unc_C11_B1causal*.csv"))


def cell(rows: list[pd.DataFrame], red: str | None = None, blue: str | None = None) -> tuple[float, int]:
    """Return (recov_pct, n) for the pooled rows matching red/blue filters."""
    parts = []
    for df in rows:
        if df.empty:
            continue
        sub = df
        if red is not None:
            sub = sub[sub["red_profile"] == red]
        if blue is not None:
            sub = sub[sub["blue_profile"] == blue]
        if len(sub):
            parts.append(sub)
    if not parts:
        return (float("nan"), 0)
    cat = pd.concat(parts, ignore_index=True)
    n = len(cat)
    rec = int(cat["recovery_success"].astype(bool).sum())
    return (100.0 * rec / n if n else float("nan"), n)


# ---- define the 5×4 matrix ----
reds = [
    ("Opus 4.7",        [OPUS_PILOT, OPUS_SCOPED, OPUS_GPT], "claude-opus-4-7"),
    ("DeepSeek-chat",   [C2, C2_N10, C2_MMX_RERUN, C2_B29_MMX_RETRY, GPT_C2], "deepseek-chat"),
    ("Qwen3.6",         [QWEN36_C10], "qwen3.6-local"),
    ("Qwen3.5-unc",     [QWEN35UNC_C11], "qwen3.5-uncensored-local"),
    ("Sonnet 4.6",      [SONNET, SONNET_GPT], "claude-sonnet-4-6"),
    ("GPT-5.4",         [GPT_C4, GPT_C5], "gpt-5.4"),
]
# Blue columns: qwen column aliases qwen3-plus for rows 1-2/4-5 and qwen3.6-local for row 3 (diagonal)
blues = ["deepseek-chat", "qwen3-plus", "minimax-m2", "gpt-5.4"]
blue_display = ["DeepSeek", "Qwen*", "MiniMax", "GPT-5.4"]

M = np.full((len(reds), len(blues)), np.nan)
N = np.zeros((len(reds), len(blues)), dtype=int)
for i, (rname, rsources, rprof) in enumerate(reds):
    for j, bp in enumerate(blues):
        # For the Qwen column on Qwen3.6 / Qwen3.5-unc rows, use qwen3.6-local as blue profile (C11 used qwen3.6-blue)
        effective_bp = "qwen3.6-local" if (bp == "qwen3-plus" and rprof in ("qwen3.6-local", "qwen3.5-uncensored-local")) else bp
        pct, n = cell(rsources, red=rprof, blue=effective_bp)
        M[i, j] = pct
        N[i, j] = n

# ---- plot ----
fig, ax = plt.subplots(figsize=(8, 4.8))
cmap = plt.get_cmap("RdYlGn")
im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")

ax.set_xticks(range(len(blues)))
ax.set_xticklabels(blue_display, fontsize=10)
ax.set_yticks(range(len(reds)))
ax.set_yticklabels([r[0] for r in reds], fontsize=10)
ax.set_xlabel("Blue defender", fontsize=11)
ax.set_ylabel("Red attacker (sorted by pool lethality, top = most lethal)", fontsize=11)

for i in range(len(reds)):
    for j in range(len(blues)):
        v = M[i, j]
        n = N[i, j]
        if np.isnan(v) or n == 0:
            text = "—"
            color = "#888"
        else:
            text = f"{v:.0f}%\nn={n}"
            color = "black" if 25 < v < 85 else "white"
        ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

# pooled-row column
rsum = np.nanmean(M, axis=1)
# pooled-col row
csum = np.nanmean(M, axis=0)
# annotate in title
title = "Recovery rate on IEEE-39 piggyback: 5-red × 4-blue matrix (%, with per-cell n)"
ax.set_title(title, fontsize=11, pad=8)

cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("blue recovery %", fontsize=10)

plt.tight_layout()
outdir = ROOT / "analysis" / "figures"
outdir.mkdir(parents=True, exist_ok=True)
out = outdir / "fig4_redblue_heatmap.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"[plot] wrote {out}")

# also print matrix for console eyeball
print()
print("MATRIX (recov%, n):")
print(f"{'Red':<16s}" + " | ".join(f"{b:>12s}" for b in blue_display))
for i, (rname, _, _) in enumerate(reds):
    row = []
    for j in range(len(blues)):
        v = M[i, j]
        n = N[i, j]
        if np.isnan(v) or n == 0:
            row.append("         — ")
        else:
            row.append(f"{v:5.1f}% n={n:>2d}")
    print(f"{rname:<16s}" + " | ".join(row))
