"""Compute aggregate stats for the Kundur main batch analysis.

Reads the two CSVs and prints every number needed to fill the
`{{...}}` placeholders in `analysis/kundur_main_analysis.md`.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
SCRIPTED_CSV = ROOT / "experiments/summaries/e2e_batch_20260419T125943Z__kundur_main_scripted.csv"
LLM_CSV = ROOT / "experiments/summaries/e2e_batch_20260419T132217Z__kundur_main_llm.csv"


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def fnum(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    if v == "" or v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def inum(row: dict, key: str, default: int = 0) -> int:
    return int(fnum(row, key, default))


def fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def pct(num: int, den: int, nd: int = 1) -> str:
    if den == 0:
        return "—"
    return f"{100.0 * num / den:.{nd}f}"


def summarize(rows: list[dict], tag: str) -> None:
    print(f"\n==== {tag}: n={len(rows)} ====")
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["red_mode"], r["red_profile"], r["blue_profile"])
        by_cell[key].append(r)

    print("\n--- TAB-K-RECOV (recovery%, diag%, shed, tokens, wall) ---")
    for key in sorted(by_cell):
        cell = by_cell[key]
        n = len(cell)
        recov = sum(1 for r in cell if r["recovery_success"].lower() == "true")
        diag = sum(1 for r in cell if r["diag_correct"].lower() == "true")
        shed = mean(fnum(r, "load_shed_total_pu") for r in cell)
        toks = mean(fnum(r, "prompt_tokens") + fnum(r, "completion_tokens") for r in cell)
        wall = mean(fnum(r, "wall_time_s") for r in cell)
        print(
            f"  {key}: n={n} recov={pct(recov,n)}% diag={pct(diag,n)}% "
            f"shed={fmt(shed,2)} tokens={fmt(toks,0)} wall={fmt(wall,1)}"
        )

    print("\n--- TAB-K-OUTCOME (joint outcome distribution) ---")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        # one row per red-config (red_mode, red_profile or '-')
        rp = r["red_profile"] if r["red_mode"] == "llm" else "-"
        groups[(r["red_mode"], rp)].append(r)
    for key in sorted(groups):
        cell = groups[key]
        n = len(cell)
        buckets = defaultdict(int)
        for r in cell:
            buckets[r["joint_outcome"]] += 1
        print(f"  {key}: n={n}  " + "  ".join(f"{k}={v}" for k, v in sorted(buckets.items())))


def lethality(rows: list[dict]) -> None:
    print("\n--- Lethality per red config ---")
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        rp = r["red_profile"] if r["red_mode"] == "llm" else "scripted"
        buckets[(r["red_mode"], rp)].append(r)
    for key, cell in sorted(buckets.items()):
        n = len(cell)
        failed = sum(1 for r in cell if r["failed"].lower() == "true")
        print(f"  {key}: failed={failed}/{n}  ({pct(failed,n)}%)")


def shed_by_blue(rows: list[dict]) -> None:
    print("\n--- TAB-K-SHED by blue_profile ---")
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r["blue_profile"]].append(r)
    for bp in sorted(buckets):
        cell = buckets[bp]
        n = len(cell)
        sheds = [fnum(r, "load_shed_total_pu") for r in cell]
        zero = sum(1 for s in sheds if s == 0.0)
        recov_sheds = [fnum(r, "load_shed_total_pu") for r in cell if r["recovery_success"].lower() == "true"]
        fail_sheds = [fnum(r, "load_shed_total_pu") for r in cell if r["recovery_success"].lower() != "true"]
        print(
            f"  {bp}: n={n} zero%={pct(zero,n)} avg={fmt(mean(sheds))} "
            f"median={fmt(median(sheds))} max={fmt(max(sheds))} "
            f"avg_recov={fmt(mean(recov_sheds)) if recov_sheds else '—'} "
            f"avg_fail={fmt(mean(fail_sheds)) if fail_sheds else '—'}"
        )


def diag_by_blue(rows: list[dict]) -> None:
    print("\n--- Phase I diagnosis accuracy by blue profile (E2E) ---")
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r["blue_profile"]].append(r)
    for bp in sorted(buckets):
        cell = buckets[bp]
        n = len(cell)
        diag = sum(1 for r in cell if r["diag_correct"].lower() == "true")
        print(f"  {bp}: n={n} diag={diag}/{n} ({pct(diag,n)}%)")


def failure_modes(rows: list[dict]) -> None:
    print("\n--- Failure-mode distribution (failed=True only) ---")
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["failed"].lower() != "true":
            continue
        rmode = r["red_mode"]
        buckets[rmode].append(r)
    for rm in sorted(buckets):
        cell = buckets[rm]
        n = len(cell)
        counts = defaultdict(int)
        for r in cell:
            counts[r["failure_reason"]] += 1
        print(f"  {rm}: n_failed={n}  " + "  ".join(f"{k}={v} ({pct(v,n)}%)" for k, v in sorted(counts.items())))


def cost_by_blue(rows: list[dict]) -> None:
    print("\n--- TAB-K-COST by blue profile ---")
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r["blue_profile"]].append(r)
    for bp in sorted(buckets):
        cell = buckets[bp]
        diag_calls = mean(fnum(r, "diag_tool_calls") for r in cell)
        mit_calls = mean(fnum(r, "mit_tool_calls") for r in cell)
        pt = mean(fnum(r, "diag_prompt_tokens") + fnum(r, "mit_prompt_tokens") for r in cell)
        ct = mean(fnum(r, "diag_completion_tokens") + fnum(r, "mit_completion_tokens") for r in cell)
        wall = mean(fnum(r, "wall_time_s") for r in cell)
        print(f"  {bp}: diag_calls={fmt(diag_calls,1)} mit_calls={fmt(mit_calls,1)} pt={fmt(pt,0)} ct={fmt(ct,0)} wall={fmt(wall,1)}")


def cost_by_red(rows: list[dict]) -> None:
    print("\n--- Red cost by red profile (llm mode only) ---")
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["red_mode"] != "llm":
            continue
        buckets[r["red_profile"]].append(r)
    for rp in sorted(buckets):
        cell = buckets[rp]
        calls = mean(fnum(r, "red_tool_calls") for r in cell)
        pt = mean(fnum(r, "red_prompt_tokens") for r in cell)
        ct = mean(fnum(r, "red_completion_tokens") for r in cell)
        print(f"  {rp}: tool_calls={fmt(calls,1)} prompt={fmt(pt,0)} completion={fmt(ct,0)}")


def wall_summary(rows: list[dict]) -> None:
    print("\n--- Wall-time summary ---")
    for rm in ["scripted", "llm"]:
        cell = [r for r in rows if r["red_mode"] == rm]
        if not cell:
            continue
        walls = [fnum(r, "wall_time_s") for r in cell]
        over90 = sum(1 for w in walls if w > 90.0)
        print(f"  {rm}: avg={fmt(mean(walls),1)} max={fmt(max(walls),1)} n>90s={over90}/{len(walls)}")


def main() -> None:
    scripted = load(SCRIPTED_CSV)
    llm = load(LLM_CSV)
    all_rows = scripted + llm

    print(f"scripted rows: {len(scripted)}")
    print(f"llm rows:      {len(llm)}")
    print(f"total:         {len(all_rows)}")

    summarize(all_rows, "ALL")
    lethality(all_rows)
    shed_by_blue(all_rows)
    diag_by_blue(all_rows)
    failure_modes(all_rows)
    cost_by_blue(all_rows)
    cost_by_red(all_rows)
    wall_summary(all_rows)


if __name__ == "__main__":
    main()
