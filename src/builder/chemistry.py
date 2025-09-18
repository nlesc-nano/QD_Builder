# nanocrystal_builder/chemistry.py
from __future__ import annotations
import random
from typing import Dict, List, Tuple, Set
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .nc_types import Plane, Facet
from .analysis import coord_numbers, bulk_cn_by_interior
from .analysis import _pair_cut as pc 

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
    from .analysis import coord_numbers_bipartite, bulk_cn_opposite_by_interior
    cn = coord_numbers_bipartite(symbols, pts, charges)
    bulk_cn = bulk_cn_opposite_by_interior(symbols, pts, planes, surf_tol, charges)
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


# Plane is (n, d) with half-space n·x >= d
Plane = Tuple[NDArray[np.float64], float]

def place_ligand(
    symbols: List[str],
    pts: NDArray[np.float64],
    idx_cat: int,
    normal: NDArray[np.float64],
    ligand: str,
    planes: List[Plane],
    charges: Dict[str, int] | None = None,   # NEW (optional)
) -> Tuple[List[str], NDArray[np.float64], int | None]:
    """
    Place `ligand` bonded to cation at `idx_cat` along `normal`.
    Hard guarantees:
      • candidate lies outside all planes (n·x >= d + eps_out)
      • nearest neighbor to ligand is the TARGET CATION at ~bond_len
      • reject if too close to any anion or existing ligand anion (like-charge spacing)
      • general clash check vs all atoms

    Returns (new_symbols, new_pts, new_index) or (symbols, pts, None) if failed.
    """
    from .analysis import _pair_cut as pc
    from scipy.spatial import cKDTree

    eps_out = 0.05
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    origin = pts[idx_cat]
    cat_sym = symbols[idx_cat]
    bond_len = pc(cat_sym, ligand)

    tree = cKDTree(pts)
    uniq_syms = set(symbols)
    r_query = max(pc(s, ligand) for s in uniq_syms) * 1.6 if uniq_syms else 3.0

    # quick partitions (if charges known)
    anion_idx: list[int] = []
    lig_idx: list[int] = []
    if charges is not None:
        for j, s in enumerate(symbols):
            if charges.get(s, 0) < 0:
                anion_idx.append(j)
            if s == ligand:
                lig_idx.append(j)

    def ensure_outside(p: NDArray[np.float64]) -> NDArray[np.float64]:
        min_slack = float("inf")
        for (n_i, d_i) in planes:
            slack = float(np.dot(n_i, p) - d_i)
            if slack < min_slack:
                min_slack = slack
        if min_slack < eps_out:
            p = p + (eps_out - min_slack) * normal
        return p

    def ok_bond_to_target(p: NDArray[np.float64]) -> bool:
        d_cat = np.linalg.norm(p - origin)
        if not (0.92 * bond_len <= d_cat <= 1.08 * bond_len):
            return False
        # make sure the closest atom is the target cation
        dists, idxs = tree.query(p, k=min(6, len(symbols)))
        if np.isscalar(dists):
            # only one neighbor
            return int(idxs) == idx_cat
        # find closest neighbor
        kmin = int(idxs[0]); dmin = float(dists[0])
        if kmin != idx_cat:
            return False
        # margin vs the next neighbor
        if len(dists) > 1 and float(dists[1]) < dmin + 0.12:
            return False
        return True

    def clear_of_anions_and_ligands(p: NDArray[np.float64]) -> bool:
        # stronger repulsion to like-charge anions/ligands
        if anion_idx:
            for j in anion_idx:
                if j == idx_cat:
                    continue
                if np.linalg.norm(p - pts[j]) < 1.10 * pc(symbols[j], ligand):
                    return False
        if lig_idx:
            for j in lig_idx:
                if np.linalg.norm(p - pts[j]) < 1.10 * pc(ligand, symbols[j]):
                    return False
        return True

    def no_general_clash(p: NDArray[np.float64]) -> bool:
        idxs = tree.query_ball_point(p, r_query)
        for j in idxs:
            if j == idx_cat:
                # allow bonded cation at ~bond_len
                if np.linalg.norm(pts[j] - p) < 0.90 * bond_len:
                    return False
                continue
            if np.linalg.norm(pts[j] - p) < 0.80 * pc(symbols[j], ligand):
                return False
        return True

    # candidate directions: normal + lateral fan
    # (keep small angles to favor true bonding geometry)
    angs = [0, 12, -12, 24, -24, 36, -36, 48, -48]
    # small bond stretch if needed
    scales = [1.00, 1.05]

    # lateral axis
    u = np.cross(normal, np.array([1.0, 0.0, 0.0], dtype=float))
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(normal, np.array([0.0, 1.0, 0.0], dtype=float))
    u /= (np.linalg.norm(u) + 1e-12)

    best_p = None
    best_sep = -1.0  # maximize min separation to anions/ligands
    for s in scales:
        for a in angs:
            ang = np.deg2rad(a)
            n_try = (np.cos(ang) * normal + np.sin(ang) * u)
            n_try /= (np.linalg.norm(n_try) + 1e-12)
            p = ensure_outside(origin + s * bond_len * n_try)
            if not ok_bond_to_target(p):
                continue
            if not no_general_clash(p):
                continue
            if not clear_of_anions_and_ligands(p):
                continue
            # score: min distance to (anions ∪ ligand anions)
            sep = float("inf")
            if anion_idx:
                sep = min(sep, *(np.linalg.norm(p - pts[j]) for j in anion_idx if j != idx_cat))
            if lig_idx:
                sep = min(sep, *(np.linalg.norm(p - pts[j]) for j in lig_idx))
            if sep > best_sep:
                best_sep = sep
                best_p = p

    if best_p is None:
        return symbols, pts, None

    new_symbols = list(symbols)
    new_symbols.append(ligand)
    new_pts = np.vstack([pts, best_p])
    return new_symbols, new_pts, len(new_symbols) - 1

