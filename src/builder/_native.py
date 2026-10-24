# src/builder/_native.py
"""
Optional C++ kernels (the `qdb_native` package built from native/).

KERNELS is the imported module, or None when it is not installed or
QDB_NATIVE=0 is set; every caller keeps a pure-Python path for that case.
Kernels leave values within EPS of a threshold undecided and the callers
settle them with the reference (Python/scipy) expression, so results are
identical with and without the kernels.
"""
from __future__ import annotations

import os

EPS = 1e-9


def _load():
    if os.environ.get("QDB_NATIVE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    try:
        import qdb_native
    except Exception:
        return None
    return qdb_native


KERNELS = _load()
