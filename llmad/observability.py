"""PMU coverage registry — which buses the blue observer can see.

Per design decision D1, coverage is **fixed per case** (not randomised
per episode) so every run is reproducible and the appendix can list the
exact PMU name set. Coverage ratio is used as an ablation axis later;
for M2 we ship a single 50% layout per case.

The lists were chosen with the following heuristics:
    * include every generator-terminal bus (operators always have these)
    * fill in load-centre and tie-line buses to spread sensing across
      the geographic/electrical footprint
    * round up to the ceil(0.5 * n_bus) target so the resulting
      fraction is close to 0.5 on both grids
"""

from __future__ import annotations


# bus-idx lists. ANDES bus idx is an int in both Kundur & IEEE 39.
PMU_COVERAGE: dict[str, dict] = {
    "kundur": {
        "fraction": 0.5,
        # 10-bus Kundur 2-area. Gens sit on 1-4; load centres are 7/8/9.
        "buses": [1, 2, 3, 4, 7, 9],
    },
    "ieee39": {
        # 10-gen generator terminals (30-39) + interface/load buses.
        # 20 of 39 buses -> 51%.
        "fraction": 0.51,
        "buses": [
            2, 10, 16, 17, 18, 19, 20, 21, 22, 23,
            25, 26, 29, 30, 31, 33, 36, 38, 39, 4,
        ],
    },
}


def pmu_buses(case_name: str) -> list[int]:
    """Return the PMU-covered bus idx list for a case."""
    if case_name not in PMU_COVERAGE:
        raise KeyError(
            f"No PMU coverage defined for case '{case_name}'. "
            f"Known: {', '.join(PMU_COVERAGE)}. Add a row to PMU_COVERAGE."
        )
    return list(PMU_COVERAGE[case_name]["buses"])


def pmu_coverage_info(case_name: str) -> dict:
    """Return a descriptor dict used in logs/prompts."""
    e = PMU_COVERAGE[case_name]
    return {
        "case": case_name,
        "fraction": e["fraction"],
        "n_buses": len(e["buses"]),
        "buses": list(e["buses"]),
    }
