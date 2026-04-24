"""Three headline figures for the piggyback main batch.

Figures emitted to ``analysis/figures/``:
    fig1_recovery_bars.png         TAB-K-RECOV across {B0, C1, C2-dsk, C2-qwen, C2-mmx}
    fig2_marglit_heatmap.png       Primary x config marginal-lethality matrix
    fig3_blue_engagement.png       Token-vs-wall scatter revealing minimax cluster

All three read CSVs under experiments/summaries/ and write PNGs. Rerun after
any batch update to refresh.

Usage (from repo root)::

    python analysis/plot_piggyback_headlines.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SUMM = ROOT / "experiments" / "summaries"
FIGS = ROOT / "analysis" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# --- CSV paths (update to latest minimax rerun if present) -----------------
C1_CSV = SUMM / "e2e_batch_20260420T002251Z__piggyback_main_c1.csv"
C1_N10_CSV = SUMM / "e2e_batch_20260420T044610Z__piggyback_c1_seedbump_v2.csv"
C2_CSV = SUMM / "e2e_batch_20260420T004145Z__piggyback_main_c2.csv"
C2_N10_CSV = SUMM / "e2e_batch_20260420T030543Z__piggyback_main_c2_n10.csv"
SCR_CSV = SUMM / "e2e_batch_20260419T160657Z__piggyback_screen_v2.csv"
UFLS_CSV = SUMM / "e2e_batch_20260420T034739Z__piggyback_ufls_baseline.csv"

# If a newer minimax rerun CSV exists, prefer it for the minimax subset
def _newest_matching(pattern: str) -> Path | None:
    matches = sorted(SUMM.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None

MMX_CSV = _newest_matching("*piggyback_minimax_rerun*.csv")
ORACLE_C1_CSV = _newest_matching("*piggyback_oracle_c1*.csv")
ORACLE_C2_CSV = _newest_matching("*piggyback_oracle_c2*.csv")
ORACLE_C3_CSV = _newest_matching("*piggyback_oracle_c3*.csv")
GPT54_C2_CSV = _newest_matching("*gpt54_C2_pbdsk*.csv")
GPT54_C3_CSV = _newest_matching("*gpt54_C3_pbqwen*.csv")
GPT54_C4_CSV = _newest_matching("*gpt54_C4_redgpt*.csv")
GPT54_C5_CSV = _newest_matching("*gpt54_C5_self*.csv")


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _as_bool(s: str) -> bool:
    return str(s).strip().lower() == "true"


def _as_int(s: str) -> int:
    try:
        return int(str(s).strip())
    except Exception:
        return 0


def _as_float(s: str, d: float = 0.0) -> float:
    try:
        return float(str(s).strip())
    except Exception:
        return d


VIABLE = {
    "primary_ieee39_bf_b16_80ms",
    "primary_ieee39_bf_b22_80ms",
    "primary_ieee39_bf_b29_100ms",
}


def load_all() -> tuple[list, list, list]:
    c1 = load(C1_CSV)
    if C1_N10_CSV.exists():
        c1 += load(C1_N10_CSV)
        print(f"[plot] appended C1 seed-bump CSV: {C1_N10_CSV.name}")
    c2 = load(C2_CSV)
    # Merge minimax rerun FIRST (it supersedes the broken minimax seeds 0-4
    # from C2_CSV). Must happen BEFORE the seed-bump append, otherwise the
    # `blue_profile != minimax-m2` filter wipes minimax seeds 5-9.
    if MMX_CSV is not None:
        mmx = load(MMX_CSV)
        c2 = [r for r in c2 if r["blue_profile"] != "minimax-m2"] + mmx
        print(f"[plot] using minimax rerun CSV: {MMX_CSV.name}")
    if C2_N10_CSV.exists():
        c2 += load(C2_N10_CSV)
    scr = [r for r in load(SCR_CSV) if r["primary_fault"] in VIABLE]
    b29_retry = _newest_matching("*piggyback_b29_mmx_retry*.csv")
    if b29_retry is not None:
        retry_rows = load(b29_retry)
        keys = {(r["primary_fault"], r["seed"], r["blue_profile"]) for r in retry_rows}
        c2 = [r for r in c2
              if (r["primary_fault"], r["seed"], r["blue_profile"]) not in keys] + retry_rows
        print(f"[plot] using minimax b29 retry CSV: {b29_retry.name}")
    return c1, c2, scr


def fig1_recovery_bars() -> None:
    c1, c2, scr = load_all()
    ufls = load(UFLS_CSV) if UFLS_CSV.exists() else []
    oracle_c1 = load(ORACLE_C1_CSV) if ORACLE_C1_CSV is not None else []
    oracle_c2 = load(ORACLE_C2_CSV) if ORACLE_C2_CSV is not None else []
    oracle_c3 = load(ORACLE_C3_CSV) if ORACLE_C3_CSV is not None else []
    gpt54_c2 = load(GPT54_C2_CSV) if GPT54_C2_CSV is not None else []
    gpt54_c3 = load(GPT54_C3_CSV) if GPT54_C3_CSV is not None else []
    gpt54_c4 = load(GPT54_C4_CSV) if GPT54_C4_CSV is not None else []
    gpt54_c5 = load(GPT54_C5_CSV) if GPT54_C5_CSV is not None else []
    configs = [
        ("B0\nnull × null",       scr,                                            "#9e9e9e"),
        ("BU\npb × UFLS",         ufls,                                           "#616161"),
        ("C1\nbase × dsk",        c1,                                             "#42a5f5"),
        ("C2\npb × dsk",          [r for r in c2 if r["blue_profile"] == "deepseek-chat"], "#1976d2"),
        ("C2\npb × qwen3",        [r for r in c2 if r["blue_profile"] == "qwen3-plus"],    "#43a047"),
        ("C2\npb × minimax",      [r for r in c2 if r["blue_profile"] == "minimax-m2"],    "#e53935"),
        ("C2\npb × gpt-5.4",      gpt54_c2,                                       "#fb8c00"),
        ("C3\nqwen × gpt-5.4",    gpt54_c3,                                       "#ef6c00"),
        ("C4\ngpt × dsk",         [r for r in gpt54_c4 if r["blue_profile"] == "deepseek-chat"], "#64b5f6"),
        ("C4\ngpt × qwen3",       [r for r in gpt54_c4 if r["blue_profile"] == "qwen3-plus"],    "#81c784"),
        ("C4\ngpt × minimax",     [r for r in gpt54_c4 if r["blue_profile"] == "minimax-m2"],    "#ef9a9a"),
        ("C5\ngpt × gpt-5.4",     gpt54_c5,                                       "#e65100"),
        ("ORACLE\nbase × oracle", oracle_c1,                                      "#ab47bc"),
        ("ORACLE\ndsk × oracle",  oracle_c2,                                      "#8e24aa"),
        ("ORACLE\nqwen × oracle", oracle_c3,                                      "#6a1b9a"),
    ]
    labels = [c[0] for c in configs]
    recov = [100 * sum(_as_bool(r["recovery_success"]) for r in rows) / max(1, len(rows))
             for _, rows, _ in configs]
    nonconv = [sum(1 for r in rows if r.get("stopped_reason") == "tds_nonconverge")
               for _, rows, _ in configs]
    exc = [sum(1 for r in rows if r.get("stopped_reason") == "exception")
           for _, rows, _ in configs]
    ns = [len(rows) for _, rows, _ in configs]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(configs))
    bars = ax.bar(x, recov, color=[c[2] for c in configs], edgecolor="black", linewidth=0.6)
    for xi, v, n, nc, ec in zip(x, recov, ns, nonconv, exc):
        label = f"{v:.0f}%\n(n={n})"
        if nc or ec:
            label += f"\n[{nc}nc{'/' + str(ec) + 'ex' if ec else ''}]"
        ax.text(xi, max(v + 2, 3), label, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Recovery rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Recovery under piggyback attack — IEEE-39\n(B0/BU n=15, C1/C2 n=30, ORACLE-C1/C3 n=15, ORACLE-C2 n=30; 5-10 seeds × 3 primaries)")
    ax.axhline(80, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.text(len(configs) - 0.5, 81, "80 % target", fontsize=7, color="gray", ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = FIGS / "fig1_recovery_bars.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[plot] wrote {out}")


def fig2_marglit_heatmap() -> None:
    c1, c2, _ = load_all()
    all_rows = c1 + c2
    configs = [
        ("base × dsk",   ("baseline", "deepseek-chat")),
        ("pb × dsk",     ("piggyback", "deepseek-chat")),
        ("pb × qwen3",   ("piggyback", "qwen3-plus")),
        ("pb × minimax", ("piggyback", "minimax-m2")),
    ]
    primaries = [
        ("bf_b16_80ms",  "primary_ieee39_bf_b16_80ms"),
        ("bf_b22_80ms",  "primary_ieee39_bf_b22_80ms"),
        ("bf_b29_100ms", "primary_ieee39_bf_b29_100ms"),
    ]

    M = np.zeros((len(primaries), len(configs)))
    for i, (_, pri_full) in enumerate(primaries):
        for j, (_, (pv, bp)) in enumerate(configs):
            rows = [r for r in all_rows
                    if r["prompt_variant"] == pv
                    and r["blue_profile"] == bp
                    and r["primary_fault"] == pri_full]
            if rows:
                fail = sum(1 for r in rows if not _as_bool(r["recovery_success"]))
                M[i, j] = 100 * fail / len(rows)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([c[0] for c in configs], fontsize=9)
    ax.set_yticks(range(len(primaries)))
    ax.set_yticklabels([p[0] for p in primaries], fontsize=9)
    ax.set_xlabel("Red config × Blue model")
    ax.set_ylabel("Primary fault")
    ax.set_title("Marginal lethality: P(fail | primary + attack) − P(fail | primary alone)")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt_color = "white" if M[i, j] > 55 else "black"
            ax.text(j, i, f"{M[i, j]:.0f}%", ha="center", va="center",
                    color=txt_color, fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Failure rate (%)", rotation=270, labelpad=14)
    plt.tight_layout()
    out = FIGS / "fig2_marglit_heatmap.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[plot] wrote {out}")


def fig3_blue_engagement() -> None:
    _, c2, _ = load_all()
    blue_colors = {
        "deepseek-chat": "#1976d2",
        "qwen3-plus":    "#43a047",
        "minimax-m2":    "#e53935",
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for blue, color in blue_colors.items():
        rows = [r for r in c2 if r["blue_profile"] == blue]
        if not rows:
            continue
        tok = [_as_int(r["prompt_tokens"]) + _as_int(r["completion_tokens"]) for r in rows]
        wall = [_as_float(r["wall_time_s"]) for r in rows]
        recov = [_as_bool(r["recovery_success"]) for r in rows]
        engaged = [(_as_int(r["diag_tool_calls"]) + _as_int(r["mit_tool_calls"])) > 1
                   for r in rows]
        # marker by engagement, fill by recovery
        for t, w, rec, eng in zip(tok, wall, recov, engaged):
            if eng:
                ax.scatter(t / 1000.0, w, marker="o", s=80,
                           facecolor=(color if rec else "none"),
                           edgecolor=color, linewidth=1.4)
            else:
                # disengaged: render as x in the model's color
                ax.scatter(t / 1000.0, w, marker="x", s=80,
                           c=color, linewidth=1.4)

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = []
    for blue, color in blue_colors.items():
        legend_elems.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                                   markeredgecolor=color, label=blue, markersize=9))
    legend_elems.append(Line2D([0], [0], marker="o", color="w",
                               markerfacecolor="black", markeredgecolor="black",
                               label="engaged × recov", markersize=9))
    legend_elems.append(Line2D([0], [0], marker="o", color="w",
                               markerfacecolor="none", markeredgecolor="black",
                               label="engaged × fail", markersize=9))
    legend_elems.append(Line2D([0], [0], marker="x", color="black",
                               label="not engaged", markersize=9, linestyle=""))
    ax.legend(handles=legend_elems, fontsize=8, loc="upper left", framealpha=0.9)

    ax.set_xlabel("Total tokens used (thousands)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Blue-side engagement under piggyback — MiniMax cluster at origin is\n"
                 "the disengagement failure mode (no diagnosis / mitigation tools called)")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = FIGS / "fig3_blue_engagement.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[plot] wrote {out}")


def main() -> None:
    fig1_recovery_bars()
    fig2_marglit_heatmap()
    fig3_blue_engagement()
    print(f"[plot] all figures written to {FIGS}")


if __name__ == "__main__":
    main()
