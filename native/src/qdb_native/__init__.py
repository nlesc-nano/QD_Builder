"""Optional C++ kernels for nanocrystal-builder.

The builder imports this package when it is installed and falls back to its
pure-Python code otherwise (or when QDB_NATIVE=0 is set).
"""
from ._core import pair_counts  # noqa: F401

__version__ = "0.1.0"
