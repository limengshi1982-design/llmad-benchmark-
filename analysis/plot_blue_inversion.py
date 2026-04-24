"""Fig 5: Blue-strategy inversion under mild vs lethal red attackers.

Two side-by-side bar charts:
- Left: per-blue recov % under a MILD red (GPT-5.4) — engagement-rewarded regime
- Right: per-blue recov % under the HARDEST red (Opus 4.7) — engagement-punished regime

The bar ordering shows that the "best blue" label flips: under GPT-5.4-red,
MiniMax-blue is high (100%) because the attack is easy; under Opus-red,
the ordering compresses into a low band (< 25%) but MiniMax *still* holds up
better than the diagnose-and-act defenders — a documented inversion.
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
    m = sorted(SUMM.glob(pat))
    return m[-1] if m else None


def _load(p: Path | None) -> pd.DataFrame:
    return pd.read_csv(p) if p and p.exists() else pd.DataFrame()


# Under Opus-red
OPUS_PILOT = _load(_newest("*claude47opus_C6_redopus*.csv"))
OPUS_SCOPED = _load(_newest("*claude47opus_C6_27*.csv"))
OPUS_GPT = _load(_newest("*claude47opus_C8_gptblue*.csv"))

# Under GPT-5.4-red (C4) for reference (a mild-red baseline)
GPT_C4 = _load(_newest("*gpt54_C4_redgpt*.csv"))
GPT_C5 = _load(_newest("*gpt54_C5_self*.csv"))

# Under DeepSeek-red (the other lethal tier) for optional third panel
C2 = _load(_newest("*piggyback_main_c2*.csv"))
C2_N10 = _load(_newest("*piggyback_main_c2_n10*.csv"))
C2_MMX = _load(_newest("*piggyback_minimax_rerun*.csv"))
C2_B29 = _load(_newest("*piggyback_b29_mmx_retry*.csv"))
GPT_C2 = _load(_newest("*gpt54_C2_pbdsk*.csv"))


def recov(rows_pooled: list[pd.DataFrame], blue: str) -> tuple[float, int]:
    parts = []
    for df in rows_pooled:
        if df.empty:
            continue
        s = df[df["blue_profile"] == blue]
        if len(s):
            parts.append(s)
    if not parts:
        return (float("nan"), 0)
    cat = pd.concat(parts, ignore_index=True)
    n = len(cat)
    r = int(cat["recovery_success"].astype(bool).sum())
    return (100.0 * r / n if n else float("nan"), n)


blues = ["deepseek-chat", "qwen3-plus", "minimax-m2", "gpt-5.4"]
disp = ["DeepSeek", "Qwen3", "MiniMax", "GPT-5.4"]

panels = [
    ("GPT-5.4-red (mild)",       [GPT_C4, GPT_C5],                    "#43a047"),
    ("DeepSeek-red (lethal)",    [C2, C2_N10, C2_MMX, C2_B29, GPT_C2], "#ef6c00"),
    ("Opus-red (most lethal)",   [OPUS_PILOT, OPUS_SCOPED, OPUS_GPT], "#c62828"),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
for ax, (title, srcs, bar_color) in zip(axes, panels):
    vals, ns = [], []
    for bp in blues:
        v, n = recov(srcs, bp)
        vals.append(v)
        ns.append(n)
    x = np.arange(len(blues))
    bars = ax.bar(x, vals, color=bar_color, edgecolor="black", linewidth=0.6)
    for xi, v, n in zip(x, vals, ns):
        label = "—" if np.isnan(v) else f"{v:.0f}%\nn={n}"
        y = 5 if np.isnan(v) else max(v + 2, 5)
        ax.text(xi, y, label, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(disp, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Blue recovery %" if ax is axes[0] else "")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.axhline(y=50, color="gray", linestyle=":", linewidth=0.6)

fig.suptitle(
    "Blue-strategy inversion: the best blue flips with attacker tier",
    fontsize=12, y=1.02,
)
plt.tight_layout()
out = ROOT / "analysis" / "figures" / "fig5_blue_inversion.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"[plot] wrote {out}")
