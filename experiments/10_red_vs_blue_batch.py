"""End-to-end red-vs-blue batch harness.

Sweeps red profile x blue profile x case x seed, calls
09_red_vs_blue.run(...) in-process, writes incremental CSV rows, and
prints grouped recovery/diagnosis summaries.
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "BLIS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))

from importlib import import_module  # noqa: E402

single_run = import_module("09_red_vs_blue").run  # type: ignore[attr-defined]


CSV_FIELDS = [
    "run_id", "red_profile", "blue_profile", "case", "red_mode",
    "prompt_variant", "primary_fault", "t_fault",
    "seed", "temperature", "t_a", "t_b", "horizon_s", "stopped_reason",
    "truth_label", "diagnosed_label", "diag_correct", "recovery_success",
    "joint_outcome", "load_shed_total_pu",
    "failed", "failure_reason", "t_fail",
    "omega_deviation_max", "v_min", "delta_drift_max", "delta_spread_max",
    "red_tool_calls", "diag_tool_calls", "mit_tool_calls",
    "red_prompt_tokens", "red_completion_tokens",
    "diag_prompt_tokens", "diag_completion_tokens",
    "mit_prompt_tokens", "mit_completion_tokens",
    "prompt_tokens", "completion_tokens", "wall_time_s",
    "error",
]


def _summaries_dir() -> Path:
    d = Path(__file__).resolve().parent / "summaries"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record_to_row(rec: dict) -> dict:
    st = rec.get("stability_final") or {}
    st_full = rec.get("stability_full") or {}
    red_tokens = ((rec.get("red") or {}).get("tokens") or {})
    diag_tokens = ((rec.get("diag") or {}).get("tokens") or {})
    mit_tokens = ((rec.get("mit") or {}).get("tokens") or {})
    calls = rec.get("n_tool_calls") or {}
    return {
        "run_id": rec.get("run_id"),
        "red_profile": rec.get("red_profile"),
        "blue_profile": rec.get("blue_profile"),
        "case": rec.get("case"),
        "red_mode": rec.get("red_mode"),
        "prompt_variant": rec.get("prompt_variant"),
        "primary_fault": rec.get("primary_fault"),
        "t_fault": rec.get("t_fault"),
        "seed": rec.get("seed"),
        "temperature": rec.get("temperature"),
        "t_a": rec.get("t_a"),
        "t_b": rec.get("t_b"),
        "horizon_s": rec.get("horizon_s"),
        "stopped_reason": rec.get("stopped_reason"),
        "truth_label": rec.get("truth_label"),
        "diagnosed_label": rec.get("diagnosed_label"),
        "diag_correct": rec.get("diag_correct"),
        "recovery_success": rec.get("recovery_success"),
        "joint_outcome": rec.get("joint_outcome"),
        "load_shed_total_pu": rec.get("load_shed_total_pu"),
        "failed": st_full.get("failed"),
        "failure_reason": st_full.get("failure_reason"),
        "t_fail": st_full.get("t_fail"),
        "omega_deviation_max": st.get("omega_deviation_max"),
        "v_min": st.get("v_min"),
        "delta_drift_max": st.get("delta_drift_max"),
        "delta_spread_max": st.get("delta_spread_max"),
        "red_tool_calls": calls.get("red"),
        "diag_tool_calls": calls.get("diag"),
        "mit_tool_calls": calls.get("mit"),
        "red_prompt_tokens": red_tokens.get("prompt_tokens"),
        "red_completion_tokens": red_tokens.get("completion_tokens"),
        "diag_prompt_tokens": diag_tokens.get("prompt_tokens"),
        "diag_completion_tokens": diag_tokens.get("completion_tokens"),
        "mit_prompt_tokens": mit_tokens.get("prompt_tokens"),
        "mit_completion_tokens": mit_tokens.get("completion_tokens"),
        "prompt_tokens": rec.get("prompt_tokens"),
        "completion_tokens": rec.get("completion_tokens"),
        "wall_time_s": rec.get("wall_time_s"),
        "error": rec.get("error"),
    }


def _avg(items: list[dict], key: str) -> float:
    vals = [r[key] for r in items if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else float("nan")


def _primary_fault_value(value: str | None) -> str | None:
    if value is None:
        return None
    if str(value).lower() in {"none", "null", "-"}:
        return None
    return value


def _print_summary(rows: list[dict]) -> None:
    buckets: dict[tuple[str, str, str, str], list[dict]] = {}
    for r in rows:
        key = (
            r.get("primary_fault") or "None",
            r.get("red_profile") or "?",
            r.get("blue_profile") or "?",
            r.get("case") or "?",
        )
        buckets.setdefault(key, []).append(r)

    print("\n=== E2E-BATCH SUMMARY ===")
    print(f"{'primary':<32} {'red':<16} {'blue':<16} {'case':<10} {'n':>3} "
          f"{'recov%':>7} {'diag%':>7} {'shed_pu':>8} {'tokens':>10} "
          f"{'wall':>7} {'nonconv':>7}")
    for (primary, red, blue, case), items in sorted(buckets.items()):
        n = len(items)
        recov_pct = 100.0 * sum(1 for r in items if r.get("recovery_success")) / n
        diag_pct = 100.0 * sum(1 for r in items if r.get("diag_correct")) / n
        print(
            f"{primary:<32} {red:<16} {blue:<16} {case:<10} {n:>3} "
            f"{recov_pct:>6.1f}% {diag_pct:>6.1f}% "
            f"{_avg(items, 'load_shed_total_pu'):>8.4f} "
            f"{_avg(items, 'prompt_tokens') + _avg(items, 'completion_tokens'):>10.0f} "
            f"{_avg(items, 'wall_time_s'):>7.1f} "
            f"{sum(1 for r in items if r.get('failure_reason') == 'nonconverge'):>7}"
        )

    print("\n=== E2E-BATCH OUTCOMES ===")
    print(f"{'primary':<32} {'red':<16} {'blue':<16} {'case':<10} "
          f"{'blue-win-clean':>15} {'diag-right-fail':>15} "
          f"{'lucky-recovery':>15} {'red-win':>8}")
    for (primary, red, blue, case), items in sorted(buckets.items()):
        cnt = {r.get("joint_outcome"): 0 for r in items}
        for r in items:
            cnt[r.get("joint_outcome")] = cnt.get(r.get("joint_outcome"), 0) + 1
        print(
            f"{primary:<32} {red:<16} {blue:<16} {case:<10} "
            f"{cnt.get('blue-win-clean', 0):>15} "
            f"{cnt.get('diag-right-mitigation-failed', 0):>15} "
            f"{cnt.get('lucky-recovery', 0):>15} "
            f"{cnt.get('red-win', 0):>8}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--red-profiles", nargs="+", default=["minimax-m2"])
    ap.add_argument("--blue-profiles", nargs="+", default=["qwen3-plus"])
    ap.add_argument("--cases", nargs="+", default=["kundur", "ieee39"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--t-a", type=float, default=2.0, dest="t_a")
    ap.add_argument("--t-b", type=float, default=5.0, dest="t_b")
    ap.add_argument("--horizon", type=float, default=30.0)
    ap.add_argument("--red-max-steps", type=int, default=12)
    ap.add_argument("--diag-max-steps", type=int, default=8)
    ap.add_argument("--mit-max-steps", type=int, default=14)
    ap.add_argument("--red-budget", type=int, default=6)
    ap.add_argument("--diag-budget", type=int, default=6)
    ap.add_argument("--mit-budget", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--red-mode", choices=["llm", "scripted", "none"], default="llm")
    ap.add_argument("--scripted-scenario", default=None)
    ap.add_argument("--primary-faults", nargs="+", default=[None])
    ap.add_argument("--prompt-variant", choices=["baseline", "piggyback"],
                    default="baseline")
    ap.add_argument("--t-fault", type=float, default=None, dest="t_fault")
    ap.add_argument("--no-per-run-log", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"__{args.tag}" if args.tag else ""
    csv_path = _summaries_dir() / f"e2e_batch_{stamp}{tag}.csv"

    rows: list[dict] = []
    total = (
        len(args.primary_faults)
        * len(args.red_profiles)
        * len(args.blue_profiles)
        * len(args.cases)
        * len(args.seeds)
    )
    print(f"[e2e-batch] plan: {len(args.primary_faults)} primary x "
          f"{len(args.red_profiles)} red x "
          f"{len(args.blue_profiles)} blue x {len(args.cases)} cases x "
          f"{len(args.seeds)} seeds = {total} runs")
    print(f"[e2e-batch] csv -> {csv_path}")

    idx = 0
    for primary_fault_raw in args.primary_faults:
        primary_fault = _primary_fault_value(primary_fault_raw)
        for red_profile in args.red_profiles:
            for blue_profile in args.blue_profiles:
                for case in args.cases:
                    for seed in args.seeds:
                        idx += 1
                        print(f"\n------ [e2e-batch {idx}/{total}] "
                              f"primary={primary_fault} "
                              f"red={red_profile} blue={blue_profile} "
                              f"case={case} seed={seed} ------")
                        try:
                            rec = single_run(
                                red_profile=red_profile,
                                blue_profile=blue_profile,
                                case=case,
                                t_a=args.t_a,
                                t_b=args.t_b,
                                horizon=args.horizon,
                                red_max_steps=args.red_max_steps,
                                diag_max_steps=args.diag_max_steps,
                                mit_max_steps=args.mit_max_steps,
                                red_budget=args.red_budget,
                                diag_budget=args.diag_budget,
                                mit_budget=args.mit_budget,
                                temperature=args.temperature,
                                seed=seed,
                                red_mode=args.red_mode,
                                scripted_scenario=args.scripted_scenario,
                                primary_fault=primary_fault,
                                t_fault=args.t_fault,
                                prompt_variant=args.prompt_variant,
                                log=not args.no_per_run_log,
                            )
                            row = _record_to_row(rec)
                        except Exception as exc:
                            traceback.print_exc()
                            row = {
                                "run_id": f"FAIL__{red_profile}__{blue_profile}__{case}__s{seed}",
                                "red_profile": red_profile,
                                "blue_profile": blue_profile,
                                "case": case,
                                "red_mode": args.red_mode,
                                "prompt_variant": args.prompt_variant,
                                "primary_fault": primary_fault,
                                "t_fault": args.t_fault,
                                "seed": seed,
                                "temperature": args.temperature,
                                "t_a": args.t_a,
                                "t_b": args.t_b,
                                "horizon_s": args.horizon,
                                "stopped_reason": "exception",
                                "truth_label": None,
                                "diagnosed_label": None,
                                "diag_correct": False,
                                "recovery_success": False,
                                "joint_outcome": "red-win",
                                "load_shed_total_pu": None,
                                "failed": None,
                                "failure_reason": None,
                                "t_fail": None,
                                "omega_deviation_max": None,
                                "v_min": None,
                                "delta_drift_max": None,
                                "delta_spread_max": None,
                                "red_tool_calls": 0,
                                "diag_tool_calls": 0,
                                "mit_tool_calls": 0,
                                "red_prompt_tokens": 0,
                                "red_completion_tokens": 0,
                                "diag_prompt_tokens": 0,
                                "diag_completion_tokens": 0,
                                "mit_prompt_tokens": 0,
                                "mit_completion_tokens": 0,
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "wall_time_s": 0.0,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        rows.append(row)

                        with open(csv_path, "w", encoding="utf-8", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                            w.writeheader()
                            for r in rows:
                                w.writerow({k: r.get(k) for k in CSV_FIELDS})

    _print_summary(rows)
    print(f"\n[e2e-batch] done. wrote {csv_path}")


if __name__ == "__main__":
    main()
