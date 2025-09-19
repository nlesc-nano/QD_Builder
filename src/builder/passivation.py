# nanocrystal_builder/passivation.py
from __future__ import annotations
import random
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import numpy.linalg as LA
from numpy.typing import NDArray

from .nc_types import Plane, Facet
from .analysis import coord_numbers, bulk_cn_by_interior, coord_numbers_bipartite, bulk_cn_opposite_by_interior, PairCuts
from .analysis import _pair_cut as pc
from .analysis import get_true_bulk_cn_from_cif
from .chemistry import facet_surface_charge, place_ligand

__all__ = ["collect_anion_candidates", "charge_balance"]

# --------------------------------------
# Internal helpers
# --------------------------------------
# Add this entire helper function to your passivation.py file.

def _select_n_farthest_candidates(
    candidates: List[dict],
    pts: NDArray[np.float64],
    n_to_select: int
) -> List[dict]:
    """
    Selects N candidates from a list that are maximally separated in 3D space using FPS.
    """
    if not candidates or n_to_select <= 0:
        return []

    if len(candidates) <= n_to_select:
        return candidates

    pool = list(candidates)
    pool_indices = np.array([c['idx'] for c in pool])
    pool_pts = pts[pool_indices]

    selected_candidates = []

    # Seed the selection with the first candidate
    current_selection_idx_in_pool = 0
    selected_candidates.append(pool.pop(current_selection_idx_in_pool))

    last_selected_pt = pool_pts[current_selection_idx_in_pool][np.newaxis, :]
    pool_pts = np.delete(pool_pts, current_selection_idx_in_pool, axis=0)

    min_distances = np.linalg.norm(pool_pts - last_selected_pt, axis=1)

    # Iteratively select the remaining N-1 points
    for _ in range(n_to_select - 1):
        if not pool: break

        farthest_idx_in_pool = np.argmax(min_distances)
        
        selected_candidates.append(pool.pop(farthest_idx_in_pool))
        last_selected_pt = pool_pts[farthest_idx_in_pool][np.newaxis, :]
        
        pool_pts = np.delete(pool_pts, farthest_idx_in_pool, axis=0)
        min_distances = np.delete(min_distances, farthest_idx_in_pool)

        if pool_pts.size == 0: break

        new_distances_to_last = np.linalg.norm(pool_pts - last_selected_pt, axis=1)
        min_distances = np.minimum(min_distances, new_distances_to_last)

    return selected_candidates

def _build_facet_frames(planes: List[Plane]):
    """Return list of tuples (n̂, d̂, u, v, x0) with normalized plane normals."""
    frames = []
    for (n, d) in planes:
        n = np.asarray(n, float); ln = LA.norm(n) + 1e-12
        n = n / ln; d = float(d) / ln
        a = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, n)) > 0.9: a = np.array([0.0, 1.0, 0.0])
        u = np.cross(n, a); u /= (LA.norm(u) + 1e-12)
        v = np.cross(n, u); v /= (LA.norm(v) + 1e-12)
        x0 = d * n
        frames.append((n, d, u, v, x0))
    return frames

def _unit_planes_from_frames(frames: List[tuple]) -> List[Tuple[np.ndarray, float]]:
    """Return (n̂, d̂) list from frames."""
    return [(n, d) for (n, d, *_rest) in frames]

def _plane_uv(frames, fid: int, x: NDArray[np.float64]) -> Tuple[float, float]:
    """Project 3D point x to facet fid's (u,v) coordinates."""
    n, d, u, v, x0 = frames[fid]
    xproj = x - (np.dot(x, n) - d) * n  # orthogonal projection
    return float(np.dot(xproj - x0, u)), float(np.dot(xproj - x0, v))

def _incident_facets(idx: int, pts: NDArray[np.float64], frames: List[tuple], surf_tol: float) -> List[int]:
    out = []
    xi = pts[idx]
    for fid, (n, d, *_rest) in enumerate(frames):
        depth = d - float(np.dot(xi, n))  # positive if inside
        if depth < surf_tol:
            out.append(fid)
    return out

def _record_uv_allfacets(idx: int,
                         pts: NDArray[np.float64],
                         frames: List[tuple],
                         surf_tol: float,
                         uv_taken: Dict[int, List[Tuple[float, float]]],
                         edit_count_facet: Dict[int, int]) -> None:
    for fid in _incident_facets(idx, pts, frames, surf_tol):
        uv = _plane_uv(frames, fid, pts[idx])
        uv_taken.setdefault(fid, []).append(uv)
        edit_count_facet[fid] = edit_count_facet.get(fid, 0) + 1

def _facet_memberships(pts: NDArray[np.float64], planes: List[Plane], surf_tol: float) -> List[List[int]]:
    """List facet-IDs each atom belongs to (within surf_tol)."""
    mem = [[] for _ in range(len(pts))]
    for fid, (n, d) in enumerate(planes):
        shell = np.where((d - pts @ n) < surf_tol)[0]
        for i in shell:
            mem[i].append(fid)
    return mem

# ---------- Geometry-driven role classification ----------

def _intersections_geometry(frames: List[tuple]):
    """
    From unit planes (frames), build:
      - edges_by_facet[fid] -> list of lines (p0, u) on facet fid (u unit)
      - verts_by_facet[fid] -> list of vertex points on facet fid
    """
    planes = _unit_planes_from_frames(frames)
    N = len(planes)
    edges_by_facet: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {k: [] for k in range(N)}
    verts_by_facet: Dict[int, List[np.ndarray]] = {k: [] for k in range(N)}

    def _line_ij(i: int, j: int):
        ni, di = planes[i]; nj, dj = planes[j]
        u = np.cross(ni, nj)
        lu = LA.norm(u)
        if lu < 1e-9:
            return None
        u /= lu
        A = np.vstack([ni, nj])       # 2x3
        b = np.array([di, dj], float) # 2
        p0 = A.T @ LA.pinv(A @ A.T) @ b
        return p0, u

    for i in range(N):
        for j in range(i+1, N):
            out = _line_ij(i, j)
            if out is None: 
                continue
            p0, u = out
            edges_by_facet[i].append((p0, u))
            edges_by_facet[j].append((p0, u))

    # vertices (triple intersections)
    for i in range(N):
        ni, di = planes[i]
        for j in range(i+1, N):
            nj, dj = planes[j]
            for k in range(j+1, N):
                nk, dk = planes[k]
                M = np.vstack([ni, nj, nk])  # 3x3
                if abs(LA.det(M)) < 1e-9:
                    continue
                x = LA.solve(M, np.array([di, dj, dk], float))
                verts_by_facet[i].append(x)
                verts_by_facet[j].append(x)
                verts_by_facet[k].append(x)

    return edges_by_facet, verts_by_facet

def _point_line_distance(p: np.ndarray, p0: np.ndarray, u: np.ndarray) -> float:
    v = p - p0
    return LA.norm(v - (v @ u) * u)

def _role_by_geometry(i: int, fid: int,
                      pts: NDArray[np.float64],
                      frames: List[tuple],
                      edges_by_facet: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
                      verts_by_facet: Dict[int, List[np.ndarray]],
                      edge_tol: float,
                      vertex_tol: float) -> Tuple[str, int]:
    """
    Decide 'vertex'/'edge'/'unique' for atom i on facet fid by geometric proximity.
      - vertex if within vertex_tol of any vertex point of fid
      - else edge if within edge_tol of any (fid, *) edge line
      - else unique
    Returns (role_name, role_rank) with ranks: unique=0, edge=1, vertex=2.
    """
    x = pts[i]
    vlist = verts_by_facet.get(fid, [])
    if vlist:
        dv = min(float(LA.norm(x - v)) for v in vlist)
        if dv < vertex_tol:
            return "vertex", 2
    elist = edges_by_facet.get(fid, [])
    if elist:
        de = min(_point_line_distance(x, p0, u) for (p0, u) in elist)
        if de < edge_tol:
            return "edge", 1
    return "unique", 0

# ---------- Placement fallback along facet normal ----------

def _fallback_place_on_facet(
    symbols: List[str],
    pts: NDArray[np.float64],
    idx_cat: int,
    fid: int,
    ligand: str,
    frames: List[tuple],
    charges: Dict[str, int],
    *,
    eps_out: float = 0.05,
) -> Tuple[List[str], NDArray[np.float64], Optional[int]]:
    """
    Place ligand along facet normal of fid (using unit planes for outside constraint).
    Short line/tilt search; extra spacing vs anions. Returns (symbols2, pts2, new_index or None).
    """
    import math
    n_unit, d_unit, *_ = frames[fid]
    origin = pts[idx_cat]
    cat_sym = symbols[idx_cat]
    bond0 = pc(cat_sym, ligand)

    def _ensure_outside(p: NDArray[np.float64]) -> NDArray[np.float64]:
        slack = float(np.dot(n_unit, p) - d_unit)
        if slack < eps_out:
            p = p + (eps_out - slack) * n_unit
        return p

    def _ok(p: NDArray[np.float64]) -> bool:
        for j, sj in enumerate(symbols):
            if j == idx_cat:
                if np.linalg.norm(pts[j] - p) < 0.90 * bond0:
                    return False
                continue
            minsep = 0.80 * pc(sj, ligand)
            if charges.get(sj, 0) < 0:
                minsep = max(minsep, 0.90 * pc(sj, ligand))
            if np.linalg.norm(pts[j] - p) < minsep:
                return False
        return True

    a = np.array([1.0, 0.0, 0.0], float)
    if abs(np.dot(a, n_unit)) > 0.9:
        a = np.array([0.0, 1.0, 0.0], float)
    u = np.cross(n_unit, a); u /= (LA.norm(u) + 1e-12)

    scales = (1.00, 1.05, 1.10, 1.15, 1.20)
    tilts  = (0.0, math.radians(8.0), -math.radians(8.0), math.radians(15.0), -math.radians(15.0))

    for sc in scales:
        for ang in tilts:
            dirv = n_unit if ang == 0.0 else (math.cos(ang) * n_unit + math.sin(ang) * u)
            dirv /= (LA.norm(dirv) + 1e-12)
            p = origin + sc * bond0 * dirv
            p = _ensure_outside(p)
            if _ok(p):
                new_symbols = list(symbols); new_symbols.append(ligand)
                new_pts = np.vstack([pts, p])
                return new_symbols, new_pts, len(new_symbols) - 1

    return symbols, pts, None

# --------------------------------------
# Candidate collection (no mutation)
# --------------------------------------

def prepass_surface_cleanup(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    charges: Dict[str, int],
    anion_ligand: str,
    surf_tol: float,
    pair_cuts: PairCuts | None = None,
    *,
    verbose: bool = False,
) -> Tuple[List[str], NDArray[np.float64], Dict[int, List[Tuple[float, float]]], Dict[int, int]]:
    """
    Always run before charge balancing:
      • swap all surface anions with CN<=2 to the anion ligand (e.g., Se2- -> Cl-)
      • remove all surface cations with CN==1 (spacing-aware)
    Returns (symbols, pts, uv_taken, edit_count_facet) to seed spacing state.
    """
    from .analysis import coord_numbers_bipartite

    frames = _build_facet_frames(planes)
    uv_taken: Dict[int, List[Tuple[float, float]]] = {}
    edit_count_facet: Dict[int, int] = {}

    def _facet_members() -> List[List[int]]:
        return _facet_memberships(pts, planes, surf_tol)

    def _is_surface(i: int, mem: List[List[int]]) -> bool:
        return len(mem[i]) > 0

    mem = _facet_members()
    cn  = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
    changed = True
    while changed:
        changed = False
        # A. swap anions CN<=2
        for i, s in enumerate(symbols):
            if not _is_surface(i, mem):
                continue
            if charges.get(s, 0) >= 0:
                continue
            if s == anion_ligand:
                continue
            if cn[i] <= 2:
                if verbose:
                    print(f"prepass: swap anion {s}#{i} (CN={int(cn[i])}) → {anion_ligand}")
                symbols[i] = anion_ligand
                _record_uv_allfacets(i, pts, frames, surf_tol, uv_taken, edit_count_facet)
                changed = True
        if changed:
            mem = _facet_members()
            cn  = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)

        # B. remove all cation CN==1 (spacing-aware)
        while True:
            mem = _facet_members()
            cn  = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
            cands = []
            for i, s in enumerate(symbols):
                if charges.get(s, 0) <= 0: continue
                if not _is_surface(i, mem): continue
                if int(round(cn[i])) != 1: continue
                best = None
                for fid, (n, d, *_rest) in enumerate(frames):
                    depth = d - float(np.dot(pts[i], n))
                    if depth < surf_tol:
                        if best is None or depth < best[0]:
                            best = (depth, fid)
                if best is None: continue
                depth, fid = best
                uv = _plane_uv(frames, fid, pts[i])
                from math import hypot
                taken = uv_taken.get(fid, [])
                dmin = float("inf") if not taken else min(hypot(uv[0]-x, uv[1]-y) for (x,y) in taken)
                cands.append((dmin, -depth, -i, i, s, fid))
            if not cands: break
            cands.sort(reverse=True)
            _, _, _, i, s, _fid = cands[0]
            if verbose:
                print(f"prepass: remove cation {s}#{i} (CN=1)")
            _record_uv_allfacets(i, pts, frames, surf_tol, uv_taken, edit_count_facet)
            symbols.pop(i)
            pts = np.delete(pts, i, axis=0)
            changed = True

    return symbols, pts, uv_taken, edit_count_facet

def collect_anion_candidates(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    charges: Dict[str, int],
    ligand: str,
    surf_tol: float,
    verbose: bool,
    pair_cuts: Optional[PairCuts] = None,
    *,
    ignore_deficit: bool = False,
    true_bulk_cn: Dict[str, int] | None = None,
) -> Tuple[List[dict], List[dict]]:
    """
    Builds lists of anion candidates for swapping. Does NOT mutate 'symbols'.

    This function identifies surface anions that are potential candidates for being
    swapped with a ligand. The key filtering logic is based on a coordination
    "deficit," which can be optionally ignored for more aggressive cleanup steps.

    Args:
        symbols: List of atomic symbols.
        pts: Numpy array of atomic coordinates.
        planes: List of facet planes defining the nanocrystal shape.
        charges: Dictionary mapping symbols to integer charges.
        ligand: The symbol of the anion ligand (e.g., 'Cl').
        surf_tol: Thickness (in Angstroms) of the surface shell.
        verbose: If True, prints detailed candidate counts.
        ignore_deficit: If True, the coordination deficit check is skipped,
            and all under-coordinated native anions are collected.
        true_bulk_cn: An optional dictionary of ideal coordination numbers
            calculated from the pristine bulk CIF file.

    Returns:
        A tuple containing two lists of dictionaries: (outer_candidates, sublayer_candidates).
        Each dictionary represents a candidate atom and its properties.
    """
    pts = np.asarray(pts, float)
    from .analysis import coord_numbers_bipartite, bulk_cn_opposite_by_interior
    
    # Calculate current coordination numbers for all atoms.
    cn = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
    
    # Determine the target bulk coordination numbers. If true_bulk_cn is provided,
    # it uses those ideal values; otherwise, it estimates from the nanocrystal interior.
    bulk_cn = bulk_cn_opposite_by_interior(
        symbols, pts, planes, surf_tol, charges, true_bulk_cn=true_bulk_cn
    )

    # Define which elements are native anions (negative charge, not the ligand).
    anions = {el for el, q in charges.items() if q < 0 and el != ligand}

    memberships = _facet_memberships(pts, planes, surf_tol)

    # Define depth thresholds for classifying atoms as "outer" or "sublayer".
    outer_thr = 0.35 * surf_tol
    subl_thr  = 1.20 * surf_tol

    outer: List[dict] = []
    subl:  List[dict] = []

    # Iterate through each facet plane to find surface atoms.
    for fid, (n, d) in enumerate(planes):
        shell_indices = np.where((d - pts @ n) < surf_tol)[0]
        for i in shell_indices:
            s = symbols[i]
            # Skip atoms that are not native anions.
            if s not in anions:
                continue
            
            # The deficit check is now optional. For post-polish rebalancing,
            # we ignore it to find all structurally unstable anions.
            deficit = max(0, int(bulk_cn.get(s, cn[i])) - int(cn[i]))
            if not ignore_deficit:
                if deficit <= 0:
                    continue

            depth = d - float(np.dot(pts[i], n))
            
            # Determine the atom's role based on how many facets it belongs to.
            m = len(memberships[i])
            role = "unique" if m == 1 else ("edge" if m == 2 else "vertex")
            role_rank = 0 if m == 1 else (1 if m == 2 else 2)
            
            # Compile a record for the candidate atom.
            rec = {
                "idx": i, 
                "elem": s, 
                "cn": int(cn[i]), 
                "bulk_cn": int(bulk_cn.get(s, cn[i])),
                "deficit": int(deficit), 
                "depth": depth,
                "role": role, 
                "role_rank": role_rank, 
                "fid": fid
            }
            
            # Classify the candidate as outer or sublayer based on its depth.
            if depth < outer_thr:
                outer.append(rec)
            elif depth < subl_thr:
                subl.append(rec)

    if verbose:
        print(f"    - Outer anion candidates: {len(outer)}")
        if outer:
            d_hist: Dict[int,int] = {}
            for r in outer: d_hist[r["deficit"]] = d_hist.get(r["deficit"], 0) + 1
            print(f"      • deficit counts (outer): {dict(sorted(d_hist.items(), reverse=True))}")
        print(f"    - Sublayer anion candidates: {len(subl)}")

    return outer, subl

def collect_anion_candidates_old(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    charges: Dict[str, int],
    ligand: str,
    surf_tol: float,
    verbose: bool,
) -> Tuple[List[dict], List[dict]]:
    """
    Build lists of ANION candidates. Does NOT mutate 'symbols'.
    Returns (outer_candidates, sublayer_candidates): dict(idx, elem, cn, bulk_cn, deficit, depth, role, role_rank, fid)
    """
    pts = np.asarray(pts, float)
    from .analysis import coord_numbers_bipartite, bulk_cn_opposite_by_interior
    cn = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
    bulk_cn = bulk_cn_opposite_by_interior(symbols, pts, planes, surf_tol, charges, true_bulk_cn=true_bulk_cn, pair_cuts=pair_cuts)
    anions = {el for el, q in charges.items() if q < 0 and el != ligand}

    memberships = _facet_memberships(pts, planes, surf_tol)

    outer_thr = 0.35 * surf_tol
    subl_thr  = 1.20 * surf_tol

    outer: List[dict] = []
    subl:  List[dict] = []

    for fid, (n, d) in enumerate(planes):
        shell = np.where((d - pts @ n) < surf_tol)[0]
        for i in shell:
            s = symbols[i]
            if s not in anions:
                continue
            deficit = max(0, bulk_cn[s] - cn[i])
            if deficit <= 0:
                continue
            depth = d - float(np.dot(pts[i], n))
            # fallback role (collector is used only when Q<0 swaps): membership-based is fine here
            m = len(memberships[i])
            role = "unique" if m == 1 else ("edge" if m == 2 else "vertex")
            role_rank = 0 if m == 1 else (1 if m == 2 else 2)
            rec = {
                "idx": i, "elem": s, "cn": int(cn[i]), "bulk_cn": int(bulk_cn[s]),
                "deficit": int(deficit), "depth": depth,
                "role": role, "role_rank": role_rank, "fid": fid
            }
            if depth < outer_thr:
                outer.append(rec)
            elif depth < subl_thr:
                subl.append(rec)

    if verbose:
        print(f"    - Outer anion candidates: {len(outer)}")
        if outer:
            d_hist: Dict[int,int] = {}
            for r in outer: d_hist[r["deficit"]] = d_hist.get(r["deficit"], 0) + 1
            print(f"      • deficit counts (outer): {dict(sorted(d_hist.items(), reverse=True))}")
        print(f"    - Sublayer anion candidates: {len(subl)}")

    return outer, subl

# --------------------------------------
# Cation-site collection for additions (geometry-aware)
# --------------------------------------

def _collect_cation_sites(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    charges: Dict[str, int],
    surf_tol: float,
    pair_cuts: Optional[PairCuts] = None,
    *,
    outer_only: bool = True,
    allow_shared: bool = True,         # ignored if geometry classifier used; we still gate by role_rank==0
    include_sublayer: bool = False,
    allowed_facets: Optional[Set[int]] = None,
    # geometry classifiers (optional; pass from charge_balance if available)
    frames: Optional[List[tuple]] = None,
    edges_by_facet: Optional[Dict[int, List[Tuple[np.ndarray, np.ndarray]]]] = None,
    verts_by_facet: Optional[Dict[int, List[np.ndarray]]] = None,
    edge_tol: Optional[float] = None,
    vertex_tol: Optional[float] = None,
) -> List[Tuple[int, NDArray[np.float64], float, int, int, int]]:
    """
    One candidate per cation atom:
      (idx, n_out, depth_min, deficit, role_rank, fid_star)

    - deficit uses bipartite CNs: bulk_cn_opposite_by_interior(...) - coord_numbers_bipartite(..., pair_cuts=pair_cuts)
    - n_out is weighted average of incident facet normals
    - role_rank uses geometry if frames/edges/verts given, else membership count
    """
    from .analysis import coord_numbers_bipartite, bulk_cn_opposite_by_interior

    ptsA = np.asarray(pts, float)
    N = len(symbols)
    cations = {el for el, q in charges.items() if q > 0}

    # normalized facet normals + d (unit planes)
    n_unit, d_unit = [], []
    for (n, d) in planes:
        n = np.asarray(n, float)
        ln = LA.norm(n) + 1e-12
        n_unit.append(n / ln)
        d_unit.append(float(d) / ln)
    n_unit = np.stack(n_unit, axis=0) if planes else np.zeros((0, 3))
    d_unit = np.array(d_unit, float) if planes else np.zeros((0,))

    # incident facets + depths
    mem = [[] for _ in range(N)]
    depths = [[] for _ in range(N)]
    for fid, (nu, du) in enumerate(zip(n_unit, d_unit)):
        if allowed_facets is not None and fid not in allowed_facets:
            continue
        t = du - ptsA @ nu
        shell = np.where(t < surf_tol)[0]
        for i in shell:
            mem[i].append(fid)
            depths[i].append(float(t[i]))

    # bipartite CNs and bulk targets
    cn_bi   = coord_numbers_bipartite(symbols, ptsA, charges, pair_cuts=pair_cuts)
    bulk_cn = bulk_cn_opposite_by_interior(symbols, ptsA, planes, surf_tol, charges)

    outer_thr = 0.35 * surf_tol
    subl_thr  = 1.20 * surf_tol

    out: List[Tuple[int, NDArray[np.float64], float, int, int, int]] = []
    use_geom = (frames is not None and edges_by_facet is not None and
                verts_by_facet is not None and edge_tol is not None and vertex_tol is not None)

    for i, s in enumerate(symbols):
        if s not in cations: continue
        if not mem[i]: continue

        # shallowest facet
        dlist = depths[i]
        depth_min = min(dlist)
        fid_star  = mem[i][int(np.argmin(dlist))]

        # outer/sublayer gating
        if outer_only:
            if not (depth_min < outer_thr): continue
        else:
            if include_sublayer:
                if not (depth_min < subl_thr): continue
            else:
                if depth_min >= outer_thr: continue

        # deficit (bipartite)
        tgt = int(bulk_cn.get(s, int(round(cn_bi[i]))))
        dft = int(max(0, tgt - int(round(cn_bi[i]))))
        if dft <= 0: continue

        # role by geometry if available, else by membership count
        if use_geom:
            _role, role_rank = _role_by_geometry(i, fid_star, ptsA, frames,
                                                 edges_by_facet, verts_by_facet,
                                                 edge_tol, vertex_tol)
        else:
            m = len(mem[i])
            role_rank = 0 if m == 1 else (1 if m == 2 else 2)

        if not allow_shared and role_rank != 0:
            continue

        # averaged outward normal (weights: shallower facets more)
        vec = np.zeros(3, float)
        for fid, dep in zip(mem[i], dlist):
            w = 1.0 / (dep + 1e-6)
            vec += w * n_unit[fid]
        ln = LA.norm(vec)
        n_out = n_unit[fid_star] if ln < 1e-8 else (vec / ln)

        out.append((i, n_out, depth_min, dft, role_rank, fid_star))

    out.sort(key=lambda t: (-t[3], t[2], t[4], t[0]))  # deficit desc, shallower, role, idx
    return out

def _collect_cation_remove_candidates(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    charges: Dict[str, int],
    surf_tol: float,
    allowed_facets: Optional[Set[int]] = None,
    *,
    frames: Optional[List[tuple]] = None,
    edges_by_facet: Optional[Dict[int, List[Tuple[np.ndarray, np.ndarray]]]] = None,
    verts_by_facet: Optional[Dict[int, List[np.ndarray]]] = None,
    edge_tol: Optional[float] = None,
    vertex_tol: Optional[float] = None,
) -> List[Tuple[int, str, int, float, int, int]]:
    """
    Return surface cation-removal candidates:
      (idx, elem, q_cation, depth, role_rank, fid)
    Ranking preference (coarse): (q asc, role_rank desc, depth asc, idx)
    """
    ptsA = np.asarray(pts, float)
    cations = {el for el, q in charges.items() if q > 0}
    use_geom = (frames is not None and edges_by_facet is not None and
                verts_by_facet is not None and edge_tol is not None and vertex_tol is not None)

    memberships = [[] for _ in range(len(symbols))]
    for fid, (n, d) in enumerate(planes):
        if allowed_facets is not None and fid not in allowed_facets:
            continue
        shell = np.where((d - ptsA @ n) < surf_tol)[0]
        for i in shell:
            memberships[i].append(fid)

    best: Dict[int, Tuple[int, str, int, float, int, int]] = {}
    for fid, (n, d) in enumerate(planes):
        if allowed_facets is not None and fid not in allowed_facets:
            continue
        n_unit = n / (LA.norm(n) + 1e-12)
        shell = np.where((d - ptsA @ n_unit) < surf_tol)[0]
        for i in shell:
            s = symbols[i]
            if s not in cations:
                continue
            depth = float(d - np.dot(ptsA[i], n_unit))
            if use_geom:
                _role, role_rank = _role_by_geometry(i, fid, ptsA, frames,
                                                     edges_by_facet, verts_by_facet,
                                                     edge_tol, vertex_tol)
            else:
                m = len(memberships[i])
                role_rank = 0 if m == 1 else (1 if m == 2 else 2)
            q = int(charges.get(s, 0))
            rec = (i, s, q, depth, role_rank, fid)
            if i not in best or depth < best[i][3]:
                best[i] = rec

    out = list(best.values())
    out.sort(key=lambda r: (r[2], -r[4], r[3], r[0]))
    return out

def _unique_center_candidates(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    charges: Dict[str, int],
    surf_tol: float,
    *,
    frames: List[tuple],
    edges_by_facet: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
    verts_by_facet: Dict[int, List[np.ndarray]],
    edge_tol: float,
    vertex_tol: float,
    outer_thr_scale: float = 0.35,   # outer = depth < 0.35*surf_tol
) -> List[Tuple[float,int,int,float,int]]:
    """
    Return list of center-biased UNIQUE outer candidates with deficit==1:
      items: (center_score, idx, fid, depth_min, deficit)
    center_score = distance to nearest edge line on that facet (bigger = more central).
    """
    from .analysis import coord_numbers_bipartite, bulk_cn_opposite_by_interior

    ptsA = np.asarray(pts, float)
    cations = {el for el, q in charges.items() if q > 0}
    n_unit, d_unit = [], []
    for (n, d) in planes:
        n = np.asarray(n, float); ln = np.linalg.norm(n) + 1e-12
        n_unit.append(n / ln); d_unit.append(float(d) / ln)
    n_unit = np.stack(n_unit, axis=0) if planes else np.zeros((0,3))
    d_unit = np.array(d_unit, float) if planes else np.zeros((0,))

    # surface membership + shallowest facet per atom
    mem = [[] for _ in range(len(symbols))]
    depths = [[] for _ in range(len(symbols))]
    for fid, (nu, du) in enumerate(zip(n_unit, d_unit)):
        t = du - ptsA @ nu
        shell = np.where(t < surf_tol)[0]
        for i in shell:
            mem[i].append(fid); depths[i].append(float(t[i]))

    cn_bi   = coord_numbers_bipartite(symbols, ptsA, charges, pair_cuts=pair_cuts)
    from .analysis import bulk_cn_opposite_by_interior
    bulk_cn = bulk_cn_opposite_by_interior(symbols, ptsA, planes, surf_tol, charges)

    outer_thr = outer_thr_scale * surf_tol
    out = []
    for i, s in enumerate(symbols):
        if s not in cations: continue
        if not mem[i]:      continue
        dlist = depths[i]
        depth_min = min(dlist)
        fid_star  = mem[i][int(np.argmin(dlist))]
        if depth_min >= outer_thr:   # outer only
            continue

        # deficit == 1 (using bipartite/bulk target)
        tgt = int(bulk_cn.get(s, int(round(cn_bi[i]))))
        dft = int(max(0, tgt - int(round(cn_bi[i]))))
        if dft != 1:
            continue

        # role by geometry: require UNIQUE on this facet
        role_name, role_rank = _role_by_geometry(i, fid_star, ptsA, frames,
                                                 edges_by_facet, verts_by_facet,
                                                 edge_tol, vertex_tol)
        if role_rank != 0:
            continue

        # center score = distance to nearest edge line on this facet
        elist = edges_by_facet.get(fid_star, [])
        if elist:
            de = min(_point_line_distance(ptsA[i], p0, u) for (p0,u) in elist)
        else:
            # no explicit edges: use distance to plane origin (proxy)
            n, d, *_ = frames[fid_star]
            x0 = d * n
            de = float(np.linalg.norm(ptsA[i] - x0))

        out.append((de, i, fid_star, depth_min, dft))

    # prefer more central (de desc), then shallower, then idx
    out.sort(key=lambda t: (-t[0], t[3], t[1]))
    return out

# --------------------------------------
# Charge balance (stepwise, facet-aware swaps & removals)
# --------------------------------------

def charge_balance(
    symbols: List[str],
    pts: NDArray[np.float64],
    _outer_candidates: List[dict],
    _sublayer_candidates: List[dict],
    charges: Dict[str, int],
    ligand: str,
    verbose: bool,
    planes: List[Plane],
    facets: List[Facet],
    surf_tol: float,
    rng: random.Random,
    cif_path: str,
    *,
    prefer_remove_parity: bool = False,
    positive_q_strategy: str = "remove", 
    cation_ligand: str | None = None,  # kept for compatibility; not used in polish anymore
):
    from .analysis import coord_numbers_bipartite

    def total_Q() -> int:
        return int(sum(charges.get(s, 0) for s in symbols))

    # frames (unit planes & UV bases)
    true_bulk_cn = get_true_bulk_cn_from_cif(cif_path, charges)
    frames = _build_facet_frames(planes)
    planes_unit = _unit_planes_from_frames(frames)
    com = np.mean(pts, axis=0)

    # ---- Adaptive tolerances from lattice (opp. NN distances on surface) ----
    charges_arr = np.array([charges.get(s, 0) for s in symbols], int)
    surf_mask = np.zeros(len(symbols), dtype=bool)
    for (n, d, *_rest) in frames:
        surf_mask |= (d - pts @ n) < surf_tol
    opp_nn = []
    for i in np.where(surf_mask)[0]:
        qi = charges_arr[i]
        if qi == 0:
            continue
        diffs = pts - pts[i]
        mask = (charges_arr * qi) < 0
        if not np.any(mask):
            continue
        dists = np.linalg.norm(diffs[mask], axis=1)
        if dists.size:
            opp_nn.append(float(np.min(dists)))
    b_nn = float(np.median(opp_nn)) if opp_nn else 2.5
    edge_tol   = min(0.50 * b_nn, 0.30 * surf_tol)
    vertex_tol = min(0.35 * b_nn, 0.20 * surf_tol)

    edges_by_facet, verts_by_facet = _intersections_geometry(frames)

    # Ensure the ligand is negative for charge accounting
    if ligand not in charges or int(charges.get(ligand, 0)) >= 0:
        if verbose:
            print(f"[warning] ligand '{ligand}' missing or non-negative in charges map; forcing −1 for balancing.")
        charges = dict(charges)  # avoid mutating caller’s dict
        charges[ligand] = -1

    # --- Step 0: pre-pass surface cleanup (always) ---
    symbols, pts, uv_taken, edit_count_facet = prepass_surface_cleanup(
        symbols, pts, planes, charges, ligand, surf_tol, verbose=verbose
    )

    from collections import defaultdict
    add_count_facet = defaultdict(int)   # counts only anion additions per facet

    # helpers that use current uv_taken/edit_count_facet
    from math import hypot
    def _min_uv_dist(fid: int, uv: Tuple[float, float]) -> float:
        lst = uv_taken.get(fid, [])
        if not lst:
            return float("inf")
        x, y = uv
        return min(hypot(x - u, y - v) for (u, v) in lst)

    # ---- guarded stabilization: avoid creating new CN=2 cations
    def _fix_undercoord_anions_once(log_each: bool = True) -> bool:
        """
        One pass: swap any surface anion with CN<=2 to ligand (X−),
        BUT skip swaps that would reduce any neighbor cation below CN=3.
        """
        nonlocal symbols, pts
        changed = False
        cn = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)

        # quick neighbor check using pair cutoffs
        def _neighbors(i):
            si = symbols[i]; xi = pts[i]
            out = []
            for j, sj in enumerate(symbols):
                if j == i:
                    continue
                rc = pc(si, sj)
                if rc <= 0:
                    continue
                if np.linalg.norm(pts[j] - xi) < rc + 1e-9:
                    out.append(j)
            return out

        for i, s in enumerate(symbols):
            if charges.get(s, 0) >= 0:   # skip cations
                continue

            # surface?
            incident = _incident_facets(i, pts, frames, surf_tol)
            if not incident:
                continue

            if s != ligand and int(round(cn[i])) <= 2:
                # would removing this anion drop any neighbor cation below 3?
                harmful = False
                for j in _neighbors(i):
                    sj = symbols[j]
                    if charges.get(sj, 0) <= 0:
                        continue  # only worry about cations
                    # j currently counts i as a neighbor; removing i -> cn_j' = cn[j] - 1
                    if int(round(cn[j])) <= 3:
                        harmful = True
                        break
                if harmful:
                    continue  # skip this anion; it would create a CN=2 cation

                before = total_Q()
                old = symbols[i]
                symbols[i] = ligand
                _record_uv_allfacets(i, pts, frames, surf_tol, uv_taken, edit_count_facet)
                after = total_Q()
                if verbose and log_each:
                    print(f"stabilize: swap anion {old}#{i} (CN={int(cn[i])}) → {ligand}  | Q:{before:+d}→{after:+d}")
                changed = True

        return changed

    def _role_by_membership_soft(i: int, *, gamma: float = 1.05) -> int:
        """
        Robust role from plane memberships:
          count how many planes this site is within gamma*surf_tol of.
          return rank: unique=0, edge=1, vertex=2
        """
        x = pts[i]
        k = 0
        thr = gamma * surf_tol
        for (n, d, *_rest) in frames:  # frames are normalized (n̂, d̂, ...)
            depth = d - float(np.dot(x, n))  # positive if inside
            if depth < thr:
                k += 1
        if k >= 3:
            return 2  # vertex
        elif k == 2:
            return 1  # edge
        else:
            return 0  # unique


    # --- keep Q in sync whenever stabilization mutates symbols ---
    def _stabilize_and_update_Q() -> bool:
        changed = _fix_undercoord_anions_once(log_each=True)
        return changed

    def _add_one_anion(allowed_facets: Optional[Set[int]] = None) -> bool:
        """Add ONE anion on a unique, outer, deficit=1 cation; min-max round-robin across facets."""
        nonlocal symbols, pts

        # candidates (center-biased unique, deficit=1)
        cand = _unique_center_candidates(
            symbols, pts, planes, charges, surf_tol,
            frames=frames,
            edges_by_facet=edges_by_facet, verts_by_facet=verts_by_facet,
            edge_tol=edge_tol, vertex_tol=vertex_tol,
        )
        if not cand:
            return False

        # group by facet and filter by allowed_facets (if provided)
        by_facet: Dict[int, List[Tuple[float,int,int,float,int]]] = {}
        for rec in cand:
            de, idx, fid, depth, dft = rec
            if allowed_facets is not None and fid not in allowed_facets:
                continue
            by_facet.setdefault(fid, []).append(rec)
        if not by_facet:
            return False

        # min-max round-robin: pick facets with the fewest additions so far
        minc = min(add_count_facet[fid] for fid in by_facet.keys())
        f_pool = [fid for fid in sorted(by_facet.keys()) if add_count_facet[fid] == minc]

        # pick the facet whose top candidate is most central
        best = None; best_key = None
        for fid in f_pool:
            recs = by_facet[fid]
            recs.sort(key=lambda t: (-t[0], t[3], t[1]))  # center desc, shallow, idx
            top = recs[0]
            center_score = top[0]
            facet_edits  = edit_count_facet.get(fid, 0)
            key = (center_score, -facet_edits, -fid)
            if best_key is None or key > best_key:
                best_key, best = key, top

        if best is None:
            return False

        center_score, idx, fid, depth, dft = best

        # Place along COM→cation radial, flipped outward wrt chosen facet
        radial = pts[idx] - com
        nr = np.linalg.norm(radial) + 1e-12
        radial /= nr
        n_f, d_f, *_ = frames[fid]
        if np.dot(radial, n_f) < 0.0:
            radial = -radial

        sym2, pts2, j_new = place_ligand(
            symbols, pts, idx, radial, ligand, planes_unit, charges=charges
        )
        if j_new is None:
            # try facet-normal fallback on the same site/facet
            sym2, pts2, j_new = _fallback_place_on_facet(
                symbols, pts, idx, fid, ligand, frames, charges
            )
            if j_new is None:
                return False

        # commit + log with fresh totals
        before = total_Q()
        symbols, pts = sym2, pts2
        _record_uv_allfacets(j_new, pts, frames, surf_tol, uv_taken, edit_count_facet)
        add_count_facet[fid] += 1

        after = total_Q()
        if verbose:
            print(f"add {ligand} near {sym2[idx]}#{idx} "
                  f"(def=1, unique, center_score={center_score:.2f}, depth={depth:.2f} Å, facet {fid})  | "
                  f"Q:{before:+d}→{after:+d}")

        _stabilize_and_update_Q()
        return True


    def _role_rank_strict(i: int) -> int:
        """
        Match the facet table: role by *exact* plane membership at surf_tol.
        unique=0, edge=1, vertex=2.
        """
        k = len(_incident_facets(i, pts, frames, surf_tol))
        return 2 if k >= 3 else (1 if k == 2 else 0)
    
    # Convenience: a reusable driver for the Q<0 pipeline
    # Uses global RR + FPS and treats vertex/edge/unique together.
    # Convenience: a reusable driver for the Q<0 pipeline
    # Swaps ONLY CN==3 anions → ligand using global RR + FPS across ALL roles.

    def _drive_Q_negative_batch(
        symbols: List[str],
        pts: NDArray[np.float64],
        charges: Dict[str, int],
        planes: List[Plane],
        frames: List[tuple],
        surf_tol: float,
        ligand: str,
        uv_taken: Dict[int, List[Tuple[float, float]]],
        edit_count_facet: Dict[int, int],
        verbose: bool,
        prefer_remove_parity: bool,
        target_cn: int,  # <-- NEW ARGUMENT
    ):
        """
        Batch-swaps anions of a specific coordination number (target_cn)
        to bring negative charge towards zero efficiently.
        """
        q = int(sum(charges.get(s, 0) for s in symbols))
        if q >= 0:
            return symbols, pts
    
        # 1. Find a native anion to determine the charge change per swap.
        native_anion_sym = next((s for s in symbols if charges.get(s, 0) < 0 and s != ligand), None)
        if not native_anion_sym:
            return symbols, pts
    
        q_ligand = charges.get(ligand, -1)
        q_anion = charges.get(native_anion_sym)
        delta_q_per_swap = q_ligand - q_anion
    
        if delta_q_per_swap <= 0:
            return symbols, pts
        
        # 2. Calculate how many swaps are needed.
        n_to_swap = abs(q) // delta_q_per_swap
        
        if q == -1 and delta_q_per_swap == 1:
            n_to_swap = 1
        elif q == -1 and prefer_remove_parity:
            return symbols, pts
    
        if n_to_swap == 0:
            return symbols, pts
    
        # 3. Collect candidates matching the target_cn.
        outer, _ = collect_anion_candidates(symbols, pts, planes, charges, ligand, surf_tol, verbose=False, pair_cuts=pair_cuts)
        swap_pool = [r for r in outer if int(r.get("cn", 99)) == target_cn and 0 <= r['idx'] < len(symbols) and symbols[r['idx']] != ligand]
    
        if not swap_pool:
            return symbols, pts
    
        # 4. Select N best-distributed candidates using FPS.
        selected_to_swap = _select_n_farthest_candidates(swap_pool, pts, n_to_swap)
    
        # 5. Execute all swaps.
        if selected_to_swap:
            if verbose: print(f"[Balance] Batch swapping {len(selected_to_swap)} CN={target_cn} anions for best distribution...")
            for cand in selected_to_swap:
                i = cand['idx']
                old_sym = symbols[i]
                if verbose:
                    before_swap = int(sum(charges.get(s, 0) for s in symbols))
                    symbols[i] = ligand
                    after_swap = int(sum(charges.get(s, 0) for s in symbols))
                    print(f" -> swap {old_sym}#{i} -> {ligand} | Q:{before_swap:+d}→{after_swap:+d}")
                else:
                    symbols[i] = ligand
                
                _record_uv_allfacets(i, pts, frames, surf_tol, uv_taken, edit_count_facet)
            
            while _fix_undercoord_anions_once(log_each=False):
                pass
        
        return symbols, pts
    
    
    
    def _drive_Q_negative():
        nonlocal symbols, pts
        from math import hypot
    
        if total_Q() >= 0:
            return
    
        stop_swaps = False
        while total_Q() < 0 and not stop_swaps:
            # Refresh candidates to reflect current topology
            outer_candidates, _ = collect_anion_candidates(symbols, pts, planes, charges, ligand, surf_tol, verbose=False, pair_cuts=pair_cuts
            )
            if not outer_candidates:
                break
    
            # Work highest deficit first, but only CN==3 anions
            deficits = sorted({r["deficit"] for r in outer_candidates}, reverse=True)
            made_progress = False
    
            for deficit in deficits:
                if stop_swaps or total_Q() >= 0:
                    break
    
                # Pool ALL roles; keep only CN==3 and not already ligand
                cand = [r for r in outer_candidates
                        if r["deficit"] == deficit
                        and int(r["cn"]) == 3
                        and 0 <= r["idx"] < len(symbols)
                        and symbols[r["idx"]] != ligand]
                if not cand:
                    continue
    
                # Per-facet best candidate using GLOBAL FPS (vs uv_taken)
                # score_in_facet = (dmin_vs_uv_taken, -depth, -idx)
                by_facet: Dict[int, Tuple[Tuple[float, float, float], dict]] = {}
                for r in cand:
                    fid = r["fid"]
                    ux, vy = _plane_uv(frames, fid, pts[r["idx"]])
                    taken = uv_taken.get(fid, [])
                    dmin = float("inf") if not taken else min(hypot(ux - x, vy - y) for (x, y) in taken)
                    score_in_facet = (dmin, -float(r["depth"]), -int(r["idx"]))
                    best = by_facet.get(fid)
                    if best is None or score_in_facet > best[0]:
                        by_facet[fid] = (score_in_facet, r)
    
                if not by_facet:
                    continue
    
                # Global round-robin across facets with FEWEST prior edits
                min_edits = min(edit_count_facet.get(fid, 0) for fid in by_facet.keys())
                facet_pool = [fid for fid in by_facet.keys() if edit_count_facet.get(fid, 0) == min_edits]
    
                # Optional parity stop at Q = -1, unless a perfect swap to Q=0 is possible.
                if total_Q() == -1:
                    # Check if the swap is ideal (delta_q = +1, e.g., Se(-2) -> Cl(-1))
                    q_ligand = charges.get(ligand, -1)
                    q_anion = charges.get(picked["elem"], -2) # 'picked' is the candidate anion
                    delta_q = q_ligand - q_anion

                    if delta_q == 1:
                        # This is a perfect swap to reach Q=0, so we should proceed.
                        pass
                    elif prefer_remove_parity:
                        # The swap would NOT result in Q=0, and the user prefers to stop.
                        if verbose:
                            print(f"(parity) stopping at Q=-1; swap would not result in Q=0 (delta_q={delta_q}).")
                        stop_swaps = True
                        break
 
                # Pick facet by its best FPS candidate (larger dmin first)
                def facet_key(fid: int):
                    return by_facet[fid][0]
    
                fid_pick = max(facet_pool, key=facet_key)
                picked = by_facet[fid_pick][1]
    
                # Execute swap and record into GLOBAL spacing/edit trackers
                i = picked["idx"]
                before = total_Q()
                old = symbols[i]
                symbols[i] = ligand  # e.g., Se → Cl
                after = total_Q()
    
                _record_uv_allfacets(i, pts, frames, surf_tol, uv_taken, edit_count_facet)
    
                if verbose:
                    role = ["unique", "edge", "vertex"][picked["role_rank"]]
                    print(
                        f"swap {old}#{i} (CN {picked['cn']}/{picked['bulk_cn']}, {role}, depth={picked['depth']:.2f} Å) "
                        f"→ {ligand}  | Q:{before:+d}→{after:+d}"
                    )
    
                _stabilize_and_update_Q()
                made_progress = True
                break  # one swap per while-iteration; refresh candidates
    
            if not made_progress:
                # No suitable CN=3 anion to swap this round; stop.
                break
        
    
    Q = total_Q()
    if verbose:
        print(f"# Q before (after prepass) = {Q:+d}")

    # --- A) Drive Q<0 if needed ---
    if Q < 0:
        # Iteratively swap anions, starting from the lowest coordination number.
        # Get all unique CNs of potential native anion swap candidates
        outer_cands, _ = collect_anion_candidates(symbols, pts, planes, charges, ligand, surf_tol, verbose=False, pair_cuts=pair_cuts)
        candidate_cns = sorted(list(set(r['cn'] for r in outer_cands if r['elem'] != ligand)))

        for target_cn in candidate_cns:
            if total_Q() >= 0:
                break
            symbols, pts = _drive_Q_negative_batch(
                symbols, pts, charges, planes, frames, surf_tol, ligand,
                uv_taken, edit_count_facet, verbose, prefer_remove_parity,
                target_cn=target_cn
            )
        Q = total_Q()

    # --- B) If Q > 0: prefer removing CN=2 cations, then re-stabilize anions ---
    if Q > 0:
        progressed = True
        while total_Q() > 0 and progressed:
            progressed = False
            cn = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
            mem = _facet_memberships(pts, planes, surf_tol)
            cand = []
            for i, s in enumerate(symbols):
                q_site = int(charges.get(s, 0))
                if q_site <= 0:
                    continue
                if len(mem[i]) == 0:  # not on surface
                    continue
                if int(round(cn[i])) != 2:
                    continue
                if q_site > total_Q():        # avoid overshoot; we’ll have later strict removals
                    continue

                # choose the shallowest incident facet for spacing and logging
                best = None
                for fid, (n, d, *_rest) in enumerate(frames):
                    depth = d - float(np.dot(pts[i], n))
                    if depth < surf_tol:
                        if best is None or depth < best[0]:
                            best = (depth, fid)
                if best is None:
                    continue
                depth, fid = best

                # role by TOPOLOGICAL membership (robust V/E/U ordering)
                role_rank = _role_rank_strict(i)
 
                # facet spacing (bigger is better)
                uv = _plane_uv(frames, fid, pts[i])
                dmin = _min_uv_dist(fid, uv)
                fcount = edit_count_facet.get(fid, 0)

                # score: prefer vertex(2) > edge(1) > unique(0), then spacing, shallower, fewer prior edits, smaller idx
                score = (role_rank, dmin, -depth, -fcount, -i)
                cand.append((score, i, s, q_site, depth, role_rank, fid))

            if not cand:
                break

            cand.sort(reverse=True)
            (_, i, s, q_site, depth, role_rank, fid) = cand[0]

            before = total_Q()
            # record UVs BEFORE deletion for cross-manifold FPS
            _record_uv_allfacets(i, pts, frames, surf_tol, uv_taken, edit_count_facet)
            # delete the CN=2 cation
            symbols.pop(i)
            pts = np.delete(pts, i, axis=0)
            after = total_Q()
            if verbose:
                role_name = ["unique", "edge", "vertex"][role_rank]
                print(f"remove {s}#{i} (CN=2, q=+{q_site}, {role_name}, depth={depth:.2f} Å, facet {fid})  | "
                      f"Q:{before:+d}→{after:+d}")

            # re-stabilize: repeatedly convert any surface anions with CN<=2 → ligand (guarded)
            changed = True
            while changed:
                changed = _fix_undercoord_anions_once()

            progressed = True

        # (IMPORTANT) We do NOT do the old 'downgrade-to-L+' fallback anymore.
    if verbose:
        def _role_hist(tag: str):
            U=E=V=0
            for i,s in enumerate(symbols):
                if charges.get(s,0) <= 0: continue
                if not _incident_facets(i, pts, frames, surf_tol): continue
                r = _role_rank_strict(i)
                U += (r==0); E += (r==1); V += (r==2)
            print(f"[roles {tag}] U={U} E={E} V={V}")
        _role_hist("init")

    if verbose: _role_hist("pre-SectionC")    

    # --- C) If still Q > 0: Branch based on user strategy ---
    if total_Q() > 0:
        # PATH A: Cation Removal (cation-deficient surface)
        if positive_q_strategy == "remove":
            if verbose: print(f"[Balance] Using 'remove' strategy for Q > 0...")
            outer_thr = 0.35 * surf_tol
            ROLE_ORDER = (2, 1, 0) # V, E, U
            progressed = True
            while total_Q() > 0 and progressed:
                progressed = False
                cn_now = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
                cand_by_cn_role: Dict[int, Dict[int, List[tuple]]] = {}
                for i, s in enumerate(symbols):
                    q_site = int(charges.get(s, 0))
                    if q_site <= 0: continue
                    best = None
                    for fid, (n, d, *_rest) in enumerate(frames):
                        depth = d - float(np.dot(pts[i], n))
                        if depth < surf_tol:
                            if best is None or depth < best[0]: best = (depth, fid)
                    if best is None or best[0] >= outer_thr: continue
                    depth, fid = best
                    cn_i = int(round(cn_now[i]))
                    if cn_i < 3 or q_site > total_Q(): continue
                    role_rank = _role_rank_strict(i)
                    uv = _plane_uv(frames, fid, pts[i])
                    dmin = _min_uv_dist(fid, uv)
                    fcount = edit_count_facet.get(fid, 0)
                    rec = (dmin, -depth, -fcount, -i, i, s, fid, depth, role_rank, q_site, cn_i)
                    cand_by_cn_role.setdefault(cn_i, {0: [], 1: [], 2: []})[role_rank].append(rec)
                if not cand_by_cn_role: break
                for cn_target in sorted(cand_by_cn_role.keys()):
                    if total_Q() <= 0: break
                    for role in ROLE_ORDER:
                        role_list = cand_by_cn_role[cn_target].get(role, [])
                        if not role_list: continue
                        fids = {r[6] for r in role_list}
                        minc = min(edit_count_facet.get(fid, 0) for fid in fids)
                        fpool = [fid for fid in sorted(fids) if edit_count_facet.get(fid, 0) == minc]
                        best_per_facet: Dict[int, Tuple[Tuple[float, float, float, float], tuple]] = {}
                        for r in role_list:
                            if r[6] not in fpool: continue
                            key = (r[0], r[1], r[2], r[3])
                            if r[6] not in best_per_facet or key > best_per_facet[r[6]][0]:
                                best_per_facet[r[6]] = (key, r)
                        if not best_per_facet: continue
                        chosen = max(best_per_facet.values(), key=lambda kv: kv[0])[1]
                        _, _, _, _, i, s, fid, depth, role_rank, q_site, cn_i = chosen
                        before = total_Q()
                        _record_uv_allfacets(i, pts, frames, surf_tol, uv_taken, edit_count_facet)
                        symbols.pop(i)
                        pts = np.delete(pts, i, axis=0)
                        after = total_Q()
                        if verbose:
                            role_name = ["unique", "edge", "vertex"][role_rank]
                            print(f"remove {s}#{i} (CN={cn_i}, q=+{q_site}, {role_name}) | Q:{before:+d}→{after:+d}")
                        progressed = True
                        _stabilize_and_update_Q()
                        break
                    if progressed: break
        
        # PATH B: Anion Addition (ligand-rich surface)
        elif positive_q_strategy == "add":
            if verbose: print(f"[Balance] Using 'add' strategy for Q > 0...")
            progressed = True
            while total_Q() > 0 and progressed:
                progressed = _add_one_anion()

### Steps 3 & 4: Structural and Electrical Polish ###

# REPLACE the end of your charge_balance function (from the comment "D) Final parity guard")
# with this new, comprehensive polishing block.


    # ==============================================================================
    # --- HELPER FUNCTIONS FOR POLISHING STEPS ---
    # ==============================================================================

    def _remove_one_ligand(
        symbols: List[str],
        pts: NDArray[np.float64],
        charges: Dict[str, int],
        ligand: str,
        verbose: bool,
    ) -> Tuple[List[str], NDArray[np.float64], bool]:
        """
        Finds and removes the single most unstable (lowest CN) ligand.
        Used to correct a Q=-1 charge to Q=0.
        """
        nonlocal total_Q
        from .analysis import coord_numbers_bipartite

        ligand_indices = [i for i, s in enumerate(symbols) if s == ligand]
        if not ligand_indices: return symbols, pts, False

        cn = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
        
        ligands_with_cn = [(cn[i], i) for i in ligand_indices]
        ligands_with_cn.sort() # Sorts by CN, then index
        
        idx_to_remove = ligands_with_cn[0][1]
        min_cn = ligands_with_cn[0][0]
        
        before = total_Q()
        s = symbols[idx_to_remove]
        
        if verbose:
            print(f"polish: REMOVE LIGAND {s}#{idx_to_remove} (CN={min_cn}) to correct parity | Q:{before:+d}→{before - charges.get(s, 0):+d}")
        
        symbols.pop(idx_to_remove)
        pts = np.delete(pts, idx_to_remove, axis=0)
        return symbols, pts, True

    def _remove_one_native_anion(
        symbols: List[str],
        pts: NDArray[np.float64],
        charges: Dict[str, int],
        target_anion_charge: int,
        ligand: str,
        verbose: bool
    ) -> Tuple[List[str], NDArray[np.float64], bool]:
        """
        Finds and removes a native anion with a specific charge, but only if it's safe.
        """
        nonlocal total_Q
        from .analysis import coord_numbers_bipartite, _pair_cut as pc

        def _get_neighbors(i):
            neighbors = []
            for j in range(len(symbols)):
                if i != j and np.linalg.norm(pts[i] - pts[j]) <= pc(symbols[i], symbols[j]):
                    neighbors.append(j)
            return neighbors

        cn = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
        
        candidate_indices = [i for i, s in enumerate(symbols) if s != ligand and charges.get(s, 0) == target_anion_charge]
        
        safe_candidates = []
        for i in candidate_indices:
            is_safe = all(not (charges.get(symbols[j], 0) > 0 and cn[j] <= 3) for j in _get_neighbors(i))
            if is_safe:
                safe_candidates.append({'idx': i, 'cn': cn[i]})
        
        if not safe_candidates: return symbols, pts, False

        safe_candidates.sort(key=lambda c: c['cn'])
        idx_to_remove = safe_candidates[0]['idx']
        
        before = total_Q()
        s = symbols[idx_to_remove]
        
        if verbose:
            print(f"polish: REMOVE NATIVE ANION {s}#{idx_to_remove} (CN={safe_candidates[0]['cn']}) to balance charge | Q:{before:+d}→{before - charges.get(s, 0):+d}")
            
        symbols.pop(idx_to_remove)
        pts = np.delete(pts, idx_to_remove, axis=0)
        return symbols, pts, True

    # ==============================================================================
    # --- D) STRUCTURAL POLISH  ---
    # ==============================================================================
    # This unified loop iteratively removes one unstable cation at a time and
    # immediately rebalances the structure and charge. This "remove-one-fix-one"
    # approach is more stable than separate, aggressive polishing steps.
    if verbose: print(f"\n[Balance] Entering iterative polish-and-rebalance loop...")
    
    while True:
        cn = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
        mem = _facet_memberships(pts, planes, surf_tol)
        
        # Step 1: Find the single best CN=2 cation to remove.
        best_cation_to_remove_idx = -1
        highest_score = None
        
        for i, s in enumerate(symbols):
            if charges.get(s, 0) > 0 and mem[i] and int(round(cn[i])) == 2:
                # Prioritize removing cations that are more exposed (less facet membership)
                # and have a lower index as a tie-breaker.
                score = (-len(mem[i]), i) 
                if highest_score is None or score > highest_score:
                    highest_score = score
                    best_cation_to_remove_idx = i
        
        # If no CN=2 cations are left, the structure is stable. Exit the loop.
        if best_cation_to_remove_idx == -1:
            if verbose: print("[Balance] No more unstable CN=2 cations found. Polish complete.")
            break
            
        # Step 2: Remove the selected cation.
        i = best_cation_to_remove_idx
        s = symbols[i]
        before_q = total_Q()
        
        _record_uv_allfacets(i, pts, frames, surf_tol, uv_taken, edit_count_facet)
        symbols.pop(i)
        pts = np.delete(pts, i, axis=0)
        
        if verbose:
            print(f"polish: REMOVE {s}#{i} (CN=2) | Q:{before_q:+d}→{total_Q():+d}")
            
        # Step 3: Immediately run the full rebalancing process to fix the charge
        # and clean up any newly created unstable anions.
        if total_Q() < 0:
            # Collect all swappable anions, ignoring the deficit check to find ALL CN=1,2,3...
            outer_cands, _ = collect_anion_candidates(symbols, pts, planes, charges, ligand, surf_tol, verbose=False, pair_cuts=pair_cuts, ignore_deficit=True, true_bulk_cn=true_bulk_cn)
            candidate_cns = sorted(list(set(r['cn'] for r in outer_cands if r['elem'] != ligand)))

            # Loop from lowest CN to highest to swap anions and restore neutrality.
            for target_cn in candidate_cns:
                if total_Q() >= 0:
                    break
                symbols, pts = _drive_Q_negative_batch(
                    symbols, pts, charges, planes, frames, surf_tol, ligand,
                    uv_taken, edit_count_facet, verbose, prefer_remove_parity,
                    target_cn=target_cn
                )

 
    # ==============================================================================
    # --- E) FINAL ELECTRICAL POLISH (Endgame Logic) ---
    # ==============================================================================
    
    q = total_Q()
    if verbose and q != 0:
        print(f"[Balance] Entering final electrical polish with Q={q:+d}...")

    # Case 1: Handle negative charge
    if q < 0:
        success = False
        # Option A: First, try a "perfect" one-shot removal of a native anion
        if not success:
            symbols, pts, success = _remove_one_native_anion(symbols, pts, charges, q, ligand, verbose)
        
        # Option B: If not, and Q=-1, try removing a ligand
        if not success and q == -1:
            symbols, pts, success = _remove_one_ligand(symbols, pts, charges, ligand, verbose)

    # Case 2: Handle positive charge
    elif q > 0:
        # Continuously add anions until charge is neutral or no sites are left
        while total_Q() > 0:
            surf_Q_end = facet_surface_charge(symbols, pts, planes, charges, surf_tol)
            allowed_final = {fid for fid, q_surf in surf_Q_end.items() if q_surf > 0} or None
            
            # Attempt to add one anion
            success = _add_one_anion(allowed_final) or _add_one_anion(None)
            
            # If adding a ligand failed (no more sites), we must stop.
            if not success:
                if verbose:
                    print("[polish] Halting ligand addition: no placeable sites remain.")
                break

    # ==============================================================================
    # --- FINAL REPORT ---
    # ==============================================================================
    Q = total_Q()
    if verbose:
        print(f"\n# Q after  = {Q:+d}")
    if Q != 0 and verbose:
        print("WARNING: neutrality not reached. Consider deeper edits or charges.")

    return symbols, pts
