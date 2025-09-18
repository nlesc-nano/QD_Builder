# nanocrystal_builder/passivation.py
from __future__ import annotations
import random
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import numpy.linalg as LA
from numpy.typing import NDArray

from .nc_types import Plane, Facet
from .analysis import coord_numbers, bulk_cn_by_interior
from .analysis import _pair_cut as pc
from .chemistry import facet_surface_charge, place_ligand

__all__ = ["collect_anion_candidates", "charge_balance"]

# --------------------------------------
# Internal helpers
# --------------------------------------

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
    cn  = coord_numbers_bipartite(symbols, pts, charges)
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
            cn  = coord_numbers_bipartite(symbols, pts, charges)

        # B. remove all cation CN==1 (spacing-aware)
        while True:
            mem = _facet_members()
            cn  = coord_numbers_bipartite(symbols, pts, charges)
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
) -> Tuple[List[dict], List[dict]]:
    """
    Build lists of ANION candidates. Does NOT mutate 'symbols'.
    Returns (outer_candidates, sublayer_candidates): dict(idx, elem, cn, bulk_cn, deficit, depth, role, role_rank, fid)
    """
    pts = np.asarray(pts, float)
    from .analysis import coord_numbers_bipartite, bulk_cn_opposite_by_interior
    cn = coord_numbers_bipartite(symbols, pts, charges)
    bulk_cn = bulk_cn_opposite_by_interior(symbols, pts, planes, surf_tol, charges)
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

    - deficit uses bipartite CNs: bulk_cn_opposite_by_interior(...) - coord_numbers_bipartite(...)
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
    cn_bi   = coord_numbers_bipartite(symbols, ptsA, charges)
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

    cn_bi   = coord_numbers_bipartite(symbols, ptsA, charges)
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
    *,
    prefer_remove_parity: bool = False,
    cation_ligand: str | None = None,  # kept for compatibility; not used in polish anymore
):
    from .analysis import coord_numbers_bipartite, bulk_cn_opposite_by_interior

    def total_Q() -> int:
        return int(sum(charges.get(s, 0) for s in symbols))

    # frames (unit planes & UV bases)
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
        cn = coord_numbers_bipartite(symbols, pts, charges)

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
        symbols, pts = sym2, pts2
        _record_uv_allfacets(j_new, pts, frames, surf_tol, uv_taken, edit_count_facet)
        add_count_facet[fid] += 1

        before = total_Q()
        after = total_Q()
        if verbose:
            print(f"add {ligand} near {symbols[idx]}#{idx} "
                  f"(def=1, unique, center_score={center_score:.2f}, depth={depth:.2f} Å, facet {fid})  | "
                  f"Q:{before:+d}→{after:+d}")

        _stabilize_and_update_Q()
        return True

    # Convenience: a reusable driver for the Q<0 pipeline
    # after pre-pass, recompute Q
    # Convenience: a reusable driver for the Q<0 pipeline (global RR + FPS; no ligand removal at Q=-1)
    def _drive_Q_negative():
        nonlocal symbols, pts
        from math import hypot
    
        if total_Q() >= 0:
            return
    
        stop_swaps = False
        while total_Q() < 0 and not stop_swaps:
            # Always refresh candidates to reflect current topology
            outer_candidates, _ = collect_anion_candidates(
                symbols, pts, planes, charges, ligand, surf_tol, verbose=False
            )
            if not outer_candidates:
                break
            deficits = sorted({r["deficit"] for r in outer_candidates}, reverse=True)
    
            made_progress = False
            for deficit in deficits:
                if stop_swaps or total_Q() >= 0:
                    break
                for role_rank in (0, 1, 2):  # unique → edge → vertex
                    if stop_swaps or total_Q() >= 0:
                        break
    
                    while total_Q() < 0:
                        if prefer_remove_parity and total_Q() == -1:
                            if verbose:
                                print("(parity) stopping swaps at Q=-1 (no ligand removal).")
                            stop_swaps = True
                            break
    
                        # Candidates for this (deficit, role_rank)
                        cand = [r for r in outer_candidates
                                if r["deficit"] == deficit and r["role_rank"] == role_rank
                                and 0 <= r["idx"] < len(symbols) and symbols[r["idx"]] != ligand]
                        if not cand:
                            break
    
                        # For each facet, keep only its best candidate by global FPS score:
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
                            break
    
                        # Round-robin across facets with fewest edits so far (global RR),
                        # tie-break by the per-facet best FPS score.
                        min_edits = min(edit_count_facet.get(fid, 0) for fid in by_facet.keys())
                        facet_pool = [fid for fid in by_facet.keys() if edit_count_facet.get(fid, 0) == min_edits]
    
                        def facet_key(fid: int):
                            # Prefer larger dmin, then shallower (-depth), then smaller idx
                            return by_facet[fid][0]
    
                        fid_pick = max(facet_pool, key=facet_key)
                        picked = by_facet[fid_pick][1]
    
                        # Execute swap and RECORD in global spacing/edit trackers
                        i = picked["idx"]
                        before = total_Q()
                        old = symbols[i]
                        symbols[i] = ligand  # e.g., Se→Cl (−2→−1 increases Q by +1)
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
    
                        if total_Q() >= 0:
                            stop_swaps = True
                            break  # exit while total_Q()<0
    
            if not made_progress:
                # Try a single −1 addition on a facet that is still cation-rich
                surf_Q_now = facet_surface_charge(symbols, pts, planes, charges, surf_tol)
                allowed_now = {fid for fid, q in surf_Q_now.items() if q > 0} or None
                if not _add_one_anion(allowed_now):
                    break  # cannot progress further
    
    Q = total_Q()
    if verbose:
        print(f"# Q before (after prepass) = {Q:+d}")

    # --- A) Drive Q<0 if needed ---
    if Q < 0:
        _drive_Q_negative()
        Q = total_Q()

    # --- B) If Q > 0: prefer removing CN=2 cations, then re-stabilize anions ---
    if Q > 0:
        progressed = True
        while total_Q() > 0 and progressed:
            progressed = False
            cn = coord_numbers_bipartite(symbols, pts, charges)
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

                # role via geometry (preferred) else membership count
                if 'edges_by_facet' in locals():
                    _role_name, role_rank = _role_by_geometry(
                        i, fid, pts, frames, edges_by_facet, verts_by_facet,
                        edge_tol, vertex_tol
                    )
                else:
                    m = len(mem[i])
                    role_rank = 0 if m == 1 else (1 if m == 2 else 2)

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

    # --- C) If still Q > 0: strict V→E→U removals with global RR+FPS ---
    if total_Q() > 0:
        surf_Q = facet_surface_charge(symbols, pts, planes, charges, surf_tol)
        rich = {fid for fid, q in surf_Q.items() if q > 0}
        allowed = rich if rich else None

        ROLE_ORDER = (2, 1, 0)  # vertex, edge, unique
        progressed = True
        while total_Q() > 0 and progressed:
            progressed = False
            cand_all = _collect_cation_remove_candidates(
                symbols, pts, planes, charges, surf_tol, allowed_facets=allowed,
                frames=frames, edges_by_facet=edges_by_facet, verts_by_facet=verts_by_facet,
                edge_tol=edge_tol, vertex_tol=vertex_tol
            )
            if not cand_all:
                break
            for role in ROLE_ORDER:
                role_cand = [r for r in cand_all if r[4] == role and r[2] <= total_Q()]
                if not role_cand:
                    continue
                fids = {r[5] for r in role_cand}
                minc = min(edit_count_facet.get(fid, 0) for fid in fids)
                fids_rr = [fid for fid in sorted(fids) if edit_count_facet.get(fid, 0) == minc]

                best = None; best_score = None
                for fid in fids_rr:
                    on_f = [r for r in role_cand if r[5] == fid]
                    for (i, s, q_site, depth, role_rank, _fid) in on_f:
                        uv = _plane_uv(frames, fid, pts[i])
                        dmin = _min_uv_dist(fid, uv)
                        score = (dmin, -q_site, -depth, -i)
                        if best_score is None or score > best_score:
                            best_score, best = score, (i, s, q_site, depth, role_rank, fid)
                if best is None:
                    continue

                i, s, q_site, depth, role_rank, fid = best
                before = total_Q()
                _record_uv_allfacets(i, pts, frames, surf_tol, uv_taken, edit_count_facet)
                symbols.pop(i)
                pts = np.delete(pts, i, axis=0)
                after = total_Q()
                if verbose:
                    role_name = ["unique", "edge", "vertex"][role_rank]
                    print(f"remove {s}#{i} (q=+{q_site}, {role_name}, depth={depth:.2f} Å, facet {fid})  | Q:{before:+d}→{after:+d}")
                progressed = True
                _stabilize_and_update_Q()
                break

            if not progressed:
                # recompute facet charges and try one −1 add
                surf_Q_now = facet_surface_charge(symbols, pts, planes, charges, surf_tol)
                allowed_now = {fid for fid, q in surf_Q_now.items() if q > 0} or None
                if _add_one_anion(allowed_now):
                    progressed = True

    # --- D) Final parity guard (e.g., Q = +1): try one anion add
    if total_Q() > 0:
        surf_Q_end = facet_surface_charge(symbols, pts, planes, charges, surf_tol)
        allowed_final = {fid for fid, q in surf_Q_end.items() if q > 0} or None
        ok = _add_one_anion(allowed_final) or _add_one_anion(None)
        if verbose and not ok:
            print("[parity] add-one-anion failed: no placeable site (likely plane/clash constraint).")

    # --- E) CN polish: delete any leftover surface CN=2 cations; rebalance via Q<0 each time
    def _polish_cn2_by_deletion_and_rebalance():
        nonlocal symbols, pts
        progressed = False
        while True:
            cn = coord_numbers_bipartite(symbols, pts, charges)
            mem = _facet_memberships(pts, planes, surf_tol)

            cand = []
            for i, s in enumerate(symbols):
                if charges.get(s, 0) <= 0:
                    continue
                if not mem[i]:
                    continue
                if int(round(cn[i])) != 2:
                    continue

                # shallowest incident facet (for logging/placement)
                best = None
                for fid, (n, d, *_rest) in enumerate(frames):
                    depth = d - float(np.dot(pts[i], n))
                    if depth < surf_tol:
                        if best is None or depth < best[0]:
                            best = (depth, fid)
                if best is None:
                    continue
                depth, fid = best
                uv = _plane_uv(frames, fid, pts[i])
                dmin = _min_uv_dist(fid, uv)
                fcount = edit_count_facet.get(fid, 0)
                score = (dmin, -fcount, -depth, -i)
                cand.append((score, i, s, depth, fid))

            if not cand:
                break

            cand.sort(reverse=True)
            _score, i, s, depth, fid = cand[0]

            # record UVs BEFORE deletion for cross-manifold spacing
            _record_uv_allfacets(i, pts, frames, surf_tol, uv_taken, edit_count_facet)
            before = total_Q()
            q_site = int(charges.get(s, 0))

            # delete the cation
            symbols.pop(i)
            pts = np.delete(pts, i, axis=0)
            after = total_Q()

            if verbose:
                # approximate role just for log (based on incident count after deletion; harmless if off by 1)
                m = len(_incident_facets(min(i, len(symbols)-1), pts, frames, surf_tol)) if len(symbols) else 1
                role_name = ["unique","edge","vertex"][min(max(m,1)-1,2)]
                print(f"polish: REMOVE {s}#{i} (CN=2, q=+{q_site}, {role_name}, depth={depth:.2f} Å, facet {fid})  "
                      f"| Q:{before:+d}→{after:+d}")

            progressed = True

            # immediately rebalance toward Q→0 using the standard Q<0 branch
            _drive_Q_negative()

        return progressed

    _polish_cn2_by_deletion_and_rebalance()

    if total_Q() > 0:
        # try to add one anion on a cation-rich facet; then anywhere
        surf_Q_end2 = facet_surface_charge(symbols, pts, planes, charges, surf_tol)
        allowed_final2 = {fid for fid, q in surf_Q_end2.items() if q > 0} or None
        ok = _add_one_anion(allowed_final2) or _add_one_anion(None)
        if verbose and not ok:
            print("[final] add-one-anion failed: no placeable site.")
    elif total_Q() < 0:
        # if polish somehow ended negative, drive it back up with your Q<0 routine
        _drive_Q_negative()
    
    Q = total_Q()
    if verbose:
        print(f"# Q after  = {Q:+d}")
    if Q != 0 and verbose:
        print("WARNING: neutrality not reached. Consider deeper edits or charges.")

    return symbols, pts

