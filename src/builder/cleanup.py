# nanocrystal_builder/cleanup.py
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
from numpy.typing import NDArray

from .analysis import coord_numbers

def prune_low_coord_sites(
    symbols: List[str],
    pts: NDArray[np.float64],
    min_cn: int = 2,
    max_passes: int = 10,
    verbose: bool = False,
) -> tuple[List[str], NDArray[np.float64], int, int]:
    """
    Iteratively remove atoms with coordination number < min_cn.
    Default min_cn=2 removes all monocoordinated (CN=1) atoms.
    Recomputes CN after each pass until no removals or max_passes reached.

    Returns: (symbols_pruned, pts_pruned, total_removed, passes)
    """
    syms = list(symbols)
    arr = np.asarray(pts, float)
    total_removed = 0
    passes = 0

    while passes < max_passes and len(syms) > 0:
        cn = coord_numbers(syms, arr)
        keep_mask = cn >= min_cn
        removed = int((~keep_mask).sum())
        if removed == 0:
            break

        if verbose:
            rm_idx = np.where(~keep_mask)[0]
            by_el: Dict[str, int] = {}
            for i in rm_idx:
                by_el[syms[i]] = by_el.get(syms[i], 0) + 1
            print(f"  prune pass {passes+1}: removed {removed} atoms with CN<{min_cn}  {by_el}")

        syms = [s for s, k in zip(syms, keep_mask) if k]
        arr = arr[keep_mask]
        total_removed += removed
        passes += 1

    return syms, arr, total_removed, passes

