"""Post-hoc piggyback-aware metrics.

The `diag_correct` column in the e2e CSVs is degenerate under the piggyback
threat model: the truth-label heuristic labels only naive (A1/A2) attacks,
so piggyback attacks end up with `truth_label=None`, which tautologically
matches `diagnosed_label=None`. This script derives metrics that are
meaningful for piggyback analysis *without* requiring new runs.

Metrics:
    piggyback_effective  red acted AND the grid did not recover
                         -> a clean "did the attacker accomplish something?"
    blue_engaged         diag or mitigation phase actually called tools
                         -> distinguishes Minimax's early-exit from real defense
    attack_reached_blue  sim stayed solvable past t_a (blue got a chance)
                         -> separates "red got lucky" from "red did work"

Usage (from repo root)::

    python analysis/piggyback_metrics.py
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _as_bool(s: str) -> bool:
    return str(s).strip().lower() == "true"


def _as_int(s: str) -> int:
    try:
        return int(str(s).strip())
    except Exception:
        return 0


def _as_float(s: str, default: float = 0.0) -> float:
    try:
        return float(str(s).strip())
    except Exception:
        return default


def derive_metrics(row: dict[str, Any]) -> dict[str, Any]:
    """Derive piggyback-aware metrics from one e2e CSV row.

    All inputs are the string values coming out of csv.DictReader.

    Note on call counts: the orchestrator records a "forced advance_until"
    as a red tool-call even for ``red_mode=none`` scripted-null runs. To
    keep ``red_acted`` semantically meaningful ("the attacker actually
    tried something") we additionally require ``red_mode=='llm'``.
    """
    red_tool_calls = _as_int(row.get("red_tool_calls", 0))
    diag_tool_calls = _as_int(row.get("diag_tool_calls", 0))
    mit_tool_calls = _as_int(row.get("mit_tool_calls", 0))
    t_a = _as_float(row.get("t_a", 2.0))
    t_fail = row.get("t_fail", "")
    stopped_reason = row.get("stopped_reason", "")
    red_mode = str(row.get("red_mode", "")).strip()
    blue_profile = str(row.get("blue_profile", "")).strip()

    recovered = _as_bool(row.get("recovery_success", "False"))
    red_acted = red_mode == "llm" and red_tool_calls > 0
    blue_engaged = blue_profile != "scripted-null" and (
        diag_tool_calls + mit_tool_calls
    ) > 1  # >1 excludes the forced-advance accounting

    # attack_reached_blue = the simulation advanced past t_a so blue had a
    # chance to react. If nonconverge happened strictly before t_a the red
    # effectively KO'd the grid before hand-off.
    if stopped_reason == "tds_nonconverge" and t_fail:
        try:
            attack_reached_blue = float(t_fail) >= t_a
        except ValueError:
            attack_reached_blue = True
    else:
        attack_reached_blue = True

    return {
        "red_acted": red_acted,
        "blue_engaged": blue_engaged,
        "attack_reached_blue": attack_reached_blue,
        "piggyback_effective": red_acted and not recovered,
        # sharper variant: red acted, grid failed, AND blue got to try
        "piggyback_effective_strict": (
            red_acted and not recovered and attack_reached_blue
        ),
    }


def summarize(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"label": label, "n": 0}
    drs = [derive_metrics(r) for r in rows]
    out = {
        "label": label,
        "n": n,
        "red_acted%": 100 * sum(d["red_acted"] for d in drs) / n,
        "blue_engaged%": 100 * sum(d["blue_engaged"] for d in drs) / n,
        "attack_reached_blue%": 100 * sum(d["attack_reached_blue"] for d in drs) / n,
        "piggyback_effective%": 100 * sum(d["piggyback_effective"] for d in drs) / n,
        "piggyback_effective_strict%": 100 * sum(d["piggyback_effective_strict"] for d in drs) / n,
        "recovery%": 100 * sum(_as_bool(r["recovery_success"]) for r in rows) / n,
    }
    return out


def pretty_print_row(d: dict[str, Any]) -> None:
    if d.get("n", 0) == 0:
        print(f'  {d["label"]:<40s}  (no rows)')
        return
    print(
        f'  {d["label"]:<40s} '
        f' n={d["n"]:3d}'
        f'  red_acted={d["red_acted%"]:5.1f}%'
        f'  blue_eng={d["blue_engaged%"]:5.1f}%'
        f'  reach_blue={d["attack_reached_blue%"]:5.1f}%'
        f'  pb_eff={d["piggyback_effective%"]:5.1f}%'
        f'  pb_eff_strict={d["piggyback_effective_strict%"]:5.1f}%'
        f'  recov={d["recovery%"]:5.1f}%'
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    summaries = root / "experiments" / "summaries"
    c1 = list(csv.DictReader(
        open(summaries / "e2e_batch_20260420T002251Z__piggyback_main_c1.csv",
             encoding="utf-8-sig")))
    # Append the C1 seed-bump (seeds 5-9, baseline-red × dsk, 3 primaries).
    c1_seedbump = summaries / "e2e_batch_20260420T044610Z__piggyback_c1_seedbump_v2.csv"
    if c1_seedbump.exists():
        c1 += list(csv.DictReader(open(c1_seedbump, encoding="utf-8-sig")))
        print(f"[metrics] appended C1 seed-bump CSV: {c1_seedbump.name}")
    c2 = list(csv.DictReader(
        open(summaries / "e2e_batch_20260420T004145Z__piggyback_main_c2.csv",
             encoding="utf-8-sig")))
    # Merge the latest minimax-rerun CSV first (supersedes the broken
    # minimax seeds 0-4 from the original c2 CSV). This must come BEFORE
    # the seed-bump append so it does not wipe minimax seeds 5-9.
    mmx_matches = sorted(summaries.glob("*piggyback_minimax_rerun*.csv"),
                         key=lambda p: p.stat().st_mtime)
    if mmx_matches:
        mmx_rerun = list(csv.DictReader(
            open(mmx_matches[-1], encoding="utf-8-sig")))
        c2 = [r for r in c2 if r["blue_profile"] != "minimax-m2"] + mmx_rerun
        print(f"[metrics] using minimax rerun CSV: {mmx_matches[-1].name}")
    # Append the n=5→10 seed-bump rows (seeds 5-9 for all three blues).
    seed_bump = summaries / "e2e_batch_20260420T030543Z__piggyback_main_c2_n10.csv"
    if seed_bump.exists():
        c2 += list(csv.DictReader(open(seed_bump, encoding="utf-8-sig")))
        print(f"[metrics] appended seed-bump CSV: {seed_bump.name}")
    # Supersede any (primary, seed, blue) rows covered by the b29 retry,
    # which closes the 2 anthropic-529 exception holes with clean nonconv rows.
    b29_retry = sorted(summaries.glob("*piggyback_b29_mmx_retry*.csv"),
                       key=lambda p: p.stat().st_mtime)
    if b29_retry:
        retry_rows = list(csv.DictReader(
            open(b29_retry[-1], encoding="utf-8-sig")))
        keys = {(r["primary_fault"], r["seed"], r["blue_profile"]) for r in retry_rows}
        c2 = [r for r in c2
              if (r["primary_fault"], r["seed"], r["blue_profile"]) not in keys] + retry_rows
        print(f"[metrics] using minimax b29 retry CSV: {b29_retry[-1].name}")
    scr = list(csv.DictReader(
        open(summaries / "e2e_batch_20260419T160657Z__piggyback_screen_v2.csv",
             encoding="utf-8-sig")))
    viable = {
        "primary_ieee39_bf_b16_80ms",
        "primary_ieee39_bf_b22_80ms",
        "primary_ieee39_bf_b29_100ms",
    }
    scr_viable = [r for r in scr if r["primary_fault"] in viable]

    # ----- Oracle ceiling batches (L3-C). Each CSV contains N runs of
    # scripted-oracle blue, which reverses red-induced deltas and rides
    # through to horizon. These give the upper bound for the recovery band.
    def _load_oracle(tag: str) -> list[dict[str, Any]]:
        matches = sorted(summaries.glob(f"*{tag}*.csv"),
                         key=lambda p: p.stat().st_mtime)
        if not matches:
            return []
        rows = list(csv.DictReader(open(matches[-1], encoding="utf-8-sig")))
        print(f"[metrics] loaded oracle CSV: {matches[-1].name} ({len(rows)} rows)")
        return rows

    oracle_c1 = _load_oracle("piggyback_oracle_c1")
    oracle_c2 = _load_oracle("piggyback_oracle_c2")
    oracle_c3 = _load_oracle("piggyback_oracle_c3")

    print("=== POOLED CONFIGURATION METRICS ===")
    pretty_print_row(summarize(scr_viable, "B0 null × null"))
    pretty_print_row(summarize(c1, "C1 baseline-red × dsk"))
    for blue in ["deepseek-chat", "qwen3-plus", "minimax-m2"]:
        rows = [r for r in c2 if r["blue_profile"] == blue]
        pretty_print_row(summarize(rows, f"C2 piggyback-red × {blue}"))
    if oracle_c1:
        pretty_print_row(summarize(oracle_c1, "ORACLE-C1 baseline-dsk × oracle"))
    if oracle_c2:
        pretty_print_row(summarize(oracle_c2, "ORACLE-C2 piggyback-dsk × oracle"))
    if oracle_c3:
        pretty_print_row(summarize(oracle_c3, "ORACLE-C3 piggyback-qwen × oracle"))

    print()
    print("=== PER-PRIMARY × BLUE (piggyback-red arm only) ===")
    for pri in sorted(viable):
        pri_short = pri.replace("primary_ieee39_", "")
        print(f"  Primary: {pri_short}")
        for blue in ["deepseek-chat", "qwen3-plus", "minimax-m2"]:
            rows = [r for r in c2
                    if r["blue_profile"] == blue and r["primary_fault"] == pri]
            pretty_print_row(summarize(rows, f"  v {blue}"))

    print()
    print("=== BLUE-ENGAGEMENT DIAGNOSTIC ===")
    print("(blue_engaged = diag_tool_calls + mit_tool_calls > 1)  # excludes forced-advance")
    print("(counts over C2 only, where we can attribute to blue choice)")
    by_blue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in c2:
        by_blue[r["blue_profile"]].append(r)
    pooled_eng_recov = 0
    pooled_eng_total = 0
    pooled_dis_recov = 0
    pooled_dis_total = 0
    for blue, rows in sorted(by_blue.items()):
        n = len(rows)
        calls = [_as_int(r.get("diag_tool_calls", 0)) + _as_int(r.get("mit_tool_calls", 0))
                 for r in rows]
        zero = sum(1 for c in calls if c == 0)
        engaged_rows = [r for r, c in zip(rows, calls) if c > 1]
        disengaged_rows = [r for r, c in zip(rows, calls) if c <= 1]
        eng_recov = sum(1 for r in engaged_rows if _as_bool(r["recovery_success"]))
        dis_recov = sum(1 for r in disengaged_rows if _as_bool(r["recovery_success"]))
        pooled_eng_recov += eng_recov; pooled_eng_total += len(engaged_rows)
        pooled_dis_recov += dis_recov; pooled_dis_total += len(disengaged_rows)
        print(
            f"  {blue:<20s}  n={n:3d}"
            f"  zero_eng={zero}/{n}"
            f"  mean_calls={sum(calls) / n:.1f}"
            f"  engaged×recov={eng_recov}/{len(engaged_rows)}"
            f"  disengaged×recov={dis_recov}/{len(disengaged_rows)}"
        )
    if pooled_eng_total + pooled_dis_total > 0:
        print(
            f"  {'pooled C2':<20s}  n={pooled_eng_total + pooled_dis_total:3d}"
            f"  engaged×recov={pooled_eng_recov}/{pooled_eng_total}"
            f"  disengaged×recov={pooled_dis_recov}/{pooled_dis_total}"
        )

    # --------------------------------------------------- Oracle per-primary
    if oracle_c1 or oracle_c2 or oracle_c3:
        print()
        print("=== ORACLE CEILING × PER-PRIMARY ===")
        for label, rows in [
            ("ORACLE-C1 baseline-dsk", oracle_c1),
            ("ORACLE-C2 piggyback-dsk", oracle_c2),
            ("ORACLE-C3 piggyback-qwen", oracle_c3),
        ]:
            if not rows:
                continue
            print(f"  {label}:")
            for pri in sorted(viable):
                pri_short = pri.replace("primary_ieee39_", "")
                pri_rows = [r for r in rows if r["primary_fault"] == pri]
                n = len(pri_rows)
                if n == 0:
                    print(f"    {pri_short:<20s}  (no rows)")
                    continue
                recov = sum(1 for r in pri_rows if _as_bool(r["recovery_success"]))
                print(f"    {pri_short:<20s}  recov={recov}/{n}")


if __name__ == "__main__":
    main()
