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


def _disabled() -> bool:
    return os.environ.get("QDB_NATIVE", "1").strip().lower() in {"0", "false", "no", "off"}


def _load():
    if _disabled():
        return None
    try:
        import qdb_native
    except Exception:
        return None
    return qdb_native


KERNELS = _load()
# Version string of the kernels in use, or None (recorded in every manifest).
VERSION = getattr(KERNELS, "__version__", "unknown") if KERNELS is not None else None


def describe() -> str:
    """One line for the log: which implementation the hot loops run on."""
    if KERNELS is not None:
        return f"C++ kernels in use (qdb_native {VERSION})"
    if _disabled():
        return "C++ kernels switched off (QDB_NATIVE=0); using Python"
    return "C++ kernels not installed (pip install ./native); using Python"
