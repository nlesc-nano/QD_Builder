# nanocrystal_builder/chemistry.py
from __future__ import annotations
import random
from typing import Dict, List, Tuple, Set
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .nc_types import Plane, Facet
from .analysis import coord_numbers, bulk_cn_by_interior, _pair_cut

def facet_surface_charge(symbols, pts, planes, charges, surf_tol):
    surf_Q = {}
    for fid, (n, d) in enumerate(planes):
        shell = np.where((d - pts @ n) < surf_tol)[0]
        surf_Q[fid] = int(sum(charges[symbols[i]] for i in shell))
    return surf_Q

def find_dangling_cations(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    facets: List[Facet],
    charges: Dict[str, int],
    surf_tol: float,
    allowed_facets: Set[int] | None = None,
    verbose: bool = False,
) -> List[Tuple[int, NDArray[np.float64], float]]:
    """
    Return sorted list of (idx, facet_normal, depth) for under-coordinated cations
    that belong to exactly one facet shell and (optionally) to allowed facets.
    """
    cn = coord_numbers(symbols, pts)
    bulk_cn = bulk_cn_by_interior(symbols, pts, planes, surf_tol)
    cations = {el for el, q in charges.items() if q > 0}

    hits = {i: [] for i in range(len(symbols))}
    for fid, (n, d) in enumerate(planes):
        shell = np.where((d - pts @ n) < surf_tol)[0]
        for i in shell:
            hits[i].append(fid)

    candidates = []
    for i, fids in hits.items():
        if len(fids) != 1:
            continue
        fid = fids[0]
        if allowed_facets and fid not in allowed_facets:
            continue
        s = symbols[i]
        if s not in cations:
            continue
        deficit = bulk_cn[s] - cn[i]
        if deficit <= 0:
            continue
        n, d = planes[fid]
        depth = d - pts[i] @ n
        candidates.append((deficit, depth, i, n))
        if verbose:
            hkl = facets[fid]
            print(f"{s}#{i:4d} | facet ({hkl.h}{hkl.k}{hkl.l}) | CN={cn[i]} bulk={bulk_cn[s]} def={deficit} | depth={depth:.2f} Å")

    candidates.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    return [(i, n, depth) for (deficit, depth, i, n) in candidates]

def place_ligand(
    symbols: List[str],
    pts: NDArray[np.float64],
    idx_cat: int,
    normal: NDArray[np.float64],
    ligand: str,
    planes: List[Plane],
) -> tuple[list[str], NDArray[np.float64]]:
    """
    Add ligand along +normal at bond cutoff distance from cation, ensure:
      (a) it’s outside the half-space (n·x > d + δ)
      (b) no clashes to existing atoms closer than 0.8 * pair_cut
    """
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    cat_sym = symbols[idx_cat]
    from .analysis import _pair_cut as pc
    bond_len = pc(cat_sym, ligand)

    new = pts[idx_cat] + bond_len * normal
    # (a) outside hull
    n0, d0 = max(planes, key=lambda pl: np.dot(new, pl[0]) - pl[1])
    if (new @ n0) <= (d0 + 0.05):  # let it protrude a bit
        new = new + 0.3 * bond_len * normal

    # (b) clash check
    tree = cKDTree(pts)
    idxs = tree.query_ball_point(new, r=2.5)  # generous
    for j in idxs:
        if np.linalg.norm(pts[j] - new) < 0.8 * pc(symbols[j], ligand):
            return symbols, pts  # skip placement

    symbols = list(symbols)
    symbols.append(ligand)
    pts = np.vstack([pts, new])
    return symbols, pts

