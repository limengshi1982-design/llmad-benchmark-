"""Case catalog — short-name resolver for the test grids we plan to use.

Keys are the canonical names we reference throughout the paper
(``kundur``, ``ieee39``, etc.). Each resolves to the ``.xlsx`` file bundled
with ANDES. Extension cases (e.g. high-renewable IEEE39) can be added
later by dropping files under ``cases/`` of the repo and registering them
here with a custom path.
"""

from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXTRA_CASES_DIR = _REPO_ROOT / "cases"


# Relative paths inside the ANDES package's ``cases/`` directory.
_BUNDLED: dict[str, str] = {
    "kundur":   "kundur/kundur_full.xlsx",
    "kundur_freq": "kundur/kundur_freq.xlsx",
    "ieee14":   "ieee14/ieee14_full.xlsx",
    "ieee39":   "ieee39/ieee39_full.xlsx",
    "ieee39_lite": "ieee39/ieee39.xlsx",
    "npcc":     "npcc/npcc.xlsx",
}


def list_cases() -> list[str]:
    """Return all case short-names known to the catalog."""
    extras = [p.stem for p in _EXTRA_CASES_DIR.glob("*.xlsx")] if _EXTRA_CASES_DIR.exists() else []
    return sorted(set(_BUNDLED) | set(extras))


def find_case(name: str) -> str:
    """Resolve a short-name (or a raw path) to an absolute .xlsx path."""
    # Raw path?
    p = Path(name)
    if p.is_absolute() and p.exists():
        return str(p)
    if p.exists():
        return str(p.resolve())

    # Repo-local override (cases/<name>.xlsx) takes precedence.
    local = _EXTRA_CASES_DIR / f"{name}.xlsx"
    if local.exists():
        return str(local)

    # Bundled case.
    if name in _BUNDLED:
        import andes
        bundled = Path(andes.__file__).resolve().parent / "cases" / _BUNDLED[name]
        if not bundled.exists():
            raise FileNotFoundError(f"Bundled case '{name}' expected at {bundled}")
        return str(bundled)

    raise KeyError(
        f"Unknown case '{name}'. Known: {', '.join(list_cases())}. "
        f"You can also pass an absolute .xlsx path."
    )
