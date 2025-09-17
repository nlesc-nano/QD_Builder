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
) -> Tuple[List[str], NDArray[np.float64]]:
    """
    Place ligand near cation idx_cat, along 'normal', just beyond the *matched* facet.
    - Enforce n_face·p >= d_face + eps for one plane (not all).
    - Avoid clashes; try small lateral fan and 5% bond stretch if needed.
    """
    eps = 0.05
    n_vec = np.asarray(normal, float)
    n_vec /= (np.linalg.norm(n_vec) + 1e-12)

    origin = pts[idx_cat]
    bond_len = pc(symbols[idx_cat], ligand)

    # --- pick the facet whose normal best aligns with 'normal' ---
    if not planes:
        n0, d0 = n_vec, float(np.dot(n_vec, origin))
    else:
        best = None; best_cos = -1.0
        for (n, d) in planes:
            n = np.asarray(n, float)
            ln = np.linalg.norm(n) + 1e-12
            nu, du = n / ln, float(d) / ln      # normalize (n,d)
            cos = float(np.dot(nu, n_vec))
            if cos > best_cos:
                best_cos, best = cos, (nu, du)
        n0, d0 = best

    def ensure_outside_face(p: NDArray[np.float64]) -> NDArray[np.float64]:
        slack = float(np.dot(n0, p) - d0)
        if slack < eps:
            p = p + (eps - slack) * n0
        return p

    tree = cKDTree(pts)
    r_query = 1.5 * max(pc(s, ligand) for s in set(symbols)) if symbols else 2.5

    def clashes(p: NDArray[np.float64]) -> bool:
        for j in tree.query_ball_point(p, r_query):
            if j == idx_cat:
                if np.linalg.norm(pts[j] - p) < 0.9 * bond_len:
                    return True
                continue
            if np.linalg.norm(pts[j] - p) < 0.8 * pc(symbols[j], ligand):
                return True
        return False

    # candidate 0: straight along 'normal'
    cand = ensure_outside_face(origin + bond_len * n_vec)
    if clashes(cand):
        # try a small lateral fan
        u = np.cross(n_vec, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(n_vec, np.array([0.0, 1.0, 0.0]))
        u /= (np.linalg.norm(u) + 1e-12)

        for ang_deg in (15, -15, 25, -25):
            ang = np.deg2rad(ang_deg)
            n_try = (np.cos(ang) * n_vec + np.sin(ang) * u)
            n_try /= (np.linalg.norm(n_try) + 1e-12)
            p_try = ensure_outside_face(origin + bond_len * n_try)
            if not clashes(p_try):
                cand = p_try
                break
        else:
            # last resort: slightly longer bond
            p_try = ensure_outside_face(origin + 1.05 * bond_len * n_vec)
            if clashes(p_try):
                return symbols, pts
            cand = p_try

    new_symbols = list(symbols)
    new_symbols.append(ligand)
    new_pts = np.vstack([pts, cand])
    return new_symbols, new_pts

