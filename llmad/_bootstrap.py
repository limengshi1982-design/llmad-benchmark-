"""Process-level bootstrap — MUST be imported before numpy / scipy / andes.

Why this file exists
--------------------
On multi-core Windows/Linux boxes (this dev machine has 32 cores) the default
BLAS backends (OpenBLAS, MKL) spawn one thread per core for every small matrix
op. ANDES' TDS loop performs many small sparse ops per step, so the thread
count explodes, the kernel context-switches itself to death, and the process
eventually OOMs. We cap threading to 1 before any numerical library is
imported. The LLM red/blue orchestration does not need BLAS parallelism —
fine-grained concurrency lives at the *agent* level, not inside each sparse
solve.

Usage
-----
Make this the **first** import in any entry point that will touch ANDES::

    from llmad import _bootstrap  # noqa: F401 — side-effect import
    import andes
    ...

Functions
---------
- ``apply()``: idempotent; re-runs env-var setup.
- ``report()``: returns a dict describing effective settings.
- ``rss_mb()``: current resident-set size in MiB (for probes).
"""

from __future__ import annotations

import os
import sys

# --- 1. Cap numerical threading ------------------------------------------------
# These variables are read *at import time* of numpy/scipy, so set them first.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def apply() -> None:
    """Idempotently apply the thread caps."""
    for var in _THREAD_VARS:
        # Only set if the user has not explicitly overridden.
        os.environ.setdefault(var, "1")
    # Numba, if ever turned on in ANDES runtime config, also respects this.
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    _force_utf8_streams()


def _force_utf8_streams() -> None:
    """Reconfigure stdout/stderr to UTF-8 with replacement fallback.

    Why: Windows' default console codec is ``cp936`` / ``gbk`` on CN-locale
    boxes. Frontier LLMs frequently emit emoji (🚨, ✅) and CJK-extension
    characters (≫, 。) in their narration. A plain ``print`` of those will
    raise ``UnicodeEncodeError`` mid-run and kill the whole experiment.
    We reconfigure to UTF-8 ``errors='replace'`` so printing is best-effort
    but never fatal. Tee'd log files are unaffected.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass  # best-effort; never block startup on cosmetic issue


def report() -> dict[str, str | int | None]:
    """Return a snapshot of effective settings."""
    info: dict[str, str | int | None] = {v: os.environ.get(v) for v in _THREAD_VARS}
    info["cpu_count"] = os.cpu_count()
    info["python"] = sys.version.split()[0]
    return info


def rss_mb() -> float | None:
    """Return current process RSS in MiB, or None if not available."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


# Side-effect: run on first import.
apply()
