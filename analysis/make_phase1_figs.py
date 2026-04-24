"""Generate 3 Phase I figures from the diagnosis batch CSV.

Writes:
    analysis/fig_p1_cm.png      — 3 confusion matrices (one per profile)
    analysis/fig_p1_cal.png     — confidence vs correctness calibration
    analysis/fig_p1_pareto.png  — tokens-per-correct vs accuracy Pareto frontier

Run:
    F:/Research/LLMAD/.venv/Scripts/python.exe analysis/make_phase1_figs.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path(__file__).resolve().parent.parent / "experiments" / "summaries" / "diag_batch_20260419T105852Z__m3_diag_first.csv"
OUT_DIR = Path(__file__).resolve().parent
LABELS = ["A1", "A2", "A3", "A4", "None"]
PROFILE_ORDER = ["minimax-m2", "deepseek-chat", "qwen3-plus"]


def load_rows() -> list[dict]:
    out: list[dict] = []
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["correct"] = r["correct"].strip().lower() == "true"
            try:
                r["confidence"] = float(r["confidence"]) if r["confidence"] else None
            except ValueError:
                r["confidence"] = None
            for k in ("prompt_tokens", "completion_tokens"):
                try:
                    r[k] = float(r[k]) if r[k] else 0.0
                except ValueError:
                    r[k] = 0.0
            if not r["predicted_label"]:
                r["predicted_label"] = "NULL"
            out.append(r)
    return out


def fig_cm(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    cols = LABELS + ["NULL"]
    cmap = plt.get_cmap("Blues")
    for ax, prof in zip(axes, PROFILE_ORDER):
        sub = [r for r in rows if r["profile"] == prof]
        M = np.zeros((len(LABELS), len(cols)), dtype=int)
        for r in sub:
            try:
                i = LABELS.index(r["truth_label"])
                j = cols.index(r["predicted_label"])
            except ValueError:
                continue
            M[i, j] += 1
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=max(M.max(), 1))
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=0)
        ax.set_yticks(range(len(LABELS)))
        ax.set_yticklabels(LABELS)
        ax.set_xlabel("Predicted")
        if prof == PROFILE_ORDER[0]:
            ax.set_ylabel("Truth")
        n = len(sub)
        correct = sum(1 for r in sub if r["correct"])
        ax.set_title(f"{prof}\nacc={100*correct/n:.1f}% (n={n})", fontsize=10)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if v == 0:
                    continue
                col = "white" if v > M.max() * 0.5 else "black"
                ax.text(j, i, str(v), ha="center", va="center", color=col, fontsize=10)
    fig.suptitle("Phase I confusion matrices (rows=truth, cols=predicted)", fontsize=11)
    fig.tight_layout()
    p = OUT_DIR / "fig_p1_cm.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


def fig_cal(rows: list[dict]) -> None:
    conf_c = [r["confidence"] for r in rows if r["correct"] and r["confidence"] is not None]
    conf_w = [r["confidence"] for r in rows if not r["correct"] and r["confidence"] is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    # Left: KDE-like histogram overlay
    bins = np.linspace(0.5, 1.0, 11)
    ax1.hist(conf_c, bins=bins, alpha=0.55, label=f"correct (n={len(conf_c)})",
             color="#2b8cbe", edgecolor="black", linewidth=0.4)
    ax1.hist(conf_w, bins=bins, alpha=0.55, label=f"wrong (n={len(conf_w)})",
             color="#d95f02", edgecolor="black", linewidth=0.4)
    ax1.axvline(np.median(conf_c), color="#2b8cbe", linestyle="--", linewidth=1,
                label=f"median correct = {np.median(conf_c):.2f}")
    ax1.axvline(np.median(conf_w), color="#d95f02", linestyle="--", linewidth=1,
                label=f"median wrong = {np.median(conf_w):.2f}")
    ax1.set_xlabel("Self-reported confidence")
    ax1.set_ylabel("Count")
    ax1.set_title("Confidence distribution by correctness")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Right: reliability diagram (binned accuracy vs confidence)
    confs = np.array([r["confidence"] for r in rows if r["confidence"] is not None])
    corrs = np.array([r["correct"] for r in rows if r["confidence"] is not None], dtype=float)
    edges = np.linspace(0.5, 1.0, 6)
    bin_ids = np.digitize(confs, edges) - 1
    bin_ids = np.clip(bin_ids, 0, len(edges) - 2)
    centers, accs, ns = [], [], []
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        centers.append((edges[b] + edges[b + 1]) / 2)
        accs.append(corrs[mask].mean())
        ns.append(int(mask.sum()))
    ax2.plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.5, label="perfect calibration")
    ax2.scatter(centers, accs, s=[50 + 10 * n for n in ns], color="#2b8cbe",
                edgecolor="black", zorder=3, label="observed (bin size ∝ n)")
    for x, y, n in zip(centers, accs, ns):
        ax2.text(x, y + 0.04, f"n={n}", ha="center", fontsize=8)
    ax2.set_xlim(0.4, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Confidence bin center")
    ax2.set_ylabel("Accuracy in bin")
    ax2.set_title("Reliability diagram")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Phase I confidence calibration", fontsize=11)
    fig.tight_layout()
    p = OUT_DIR / "fig_p1_cal.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


def fig_pareto(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {"minimax-m2": "o", "deepseek-chat": "s", "qwen3-plus": "^"}
    colors = {"minimax-m2": "#1b9e77", "deepseek-chat": "#d95f02", "qwen3-plus": "#7570b3"}

    # Per-profile aggregate point
    for prof in PROFILE_ORDER:
        sub = [r for r in rows if r["profile"] == prof]
        n = len(sub)
        correct = sum(1 for r in sub if r["correct"])
        acc = correct / n
        tokens = np.mean([r["prompt_tokens"] + r["completion_tokens"] for r in sub])
        wall = np.mean([float(r.get("wall_time_s", 0) or 0) for r in sub])
        tpc = tokens / acc if acc > 0 else np.inf
        ax.scatter([tokens], [100 * acc], s=260, c=colors[prof], marker=markers[prof],
                   edgecolor="black", linewidth=1.2, label=f"{prof} ({100*acc:.0f}%, {tpc/1000:.0f}K/correct)",
                   zorder=4)
        # Annotate
        ax.annotate(f"{prof}\nwall={wall:.1f}s", (tokens, 100 * acc),
                    textcoords="offset points", xytext=(12, 6), fontsize=8,
                    color=colors[prof])

    # Per-scenario markers within profile (small dots)
    by_ps = defaultdict(list)
    for r in rows:
        by_ps[(r["profile"], r["scenario"])].append(r)
    for (prof, scen), sub in by_ps.items():
        n = len(sub)
        correct = sum(1 for r in sub if r["correct"])
        acc = correct / n
        tokens = np.mean([r["prompt_tokens"] + r["completion_tokens"] for r in sub])
        ax.scatter([tokens], [100 * acc], s=40, c=colors[prof], marker=markers[prof],
                   alpha=0.35, edgecolor="none", zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Avg tokens per run (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(-5, 100)
    ax.set_title("Phase I: accuracy vs token cost\n(big = profile agg; small = per-scenario)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    p = OUT_DIR / "fig_p1_pareto.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


def main() -> None:
    rows = load_rows()
    print(f"loaded {len(rows)} rows from {CSV_PATH.name}")
    fig_cm(rows)
    fig_cal(rows)
    fig_pareto(rows)


if __name__ == "__main__":
    main()
