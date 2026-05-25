# nanocrystal_builder/passivation.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import numpy.linalg as LA
from numpy.typing import NDArray

from .nc_types import Plane, Facet
from .analysis import coord_numbers_bipartite, bulk_cn_opposite_by_interior, PairCuts

__all__ = ["collect_anion_candidates"]

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

# --------------------------------------
# Candidate collection (no mutation)
# --------------------------------------

# --------------------------------------
# Region / charge-neutral bundle helpers
# --------------------------------------

def _atom_region_index(
    i: int,
    region_masks: Optional[List[NDArray[np.bool_]]] = None,
    atom_regions: Optional[List[int]] = None,
) -> int:
    if atom_regions is not None and 0 <= i < len(atom_regions):
        reg = int(atom_regions[i])
        if reg >= 0:
            return reg
    if region_masks is None:
        return 0
    for k, mask in enumerate(region_masks):
        if i < len(mask) and bool(mask[i]):
            return k
    return max(0, len(region_masks) - 1)


def atom_regions_from_masks(
    n_atoms: int,
    region_masks: List[NDArray[np.bool_]],
) -> List[int]:
    out = [-1] * n_atoms
    for k, mask in enumerate(region_masks):
        for i in range(n_atoms):
            if i < len(mask) and bool(mask[i]):
                out[i] = k
    return out


def _native_anion_elements(charges: Dict[str, int], ligand: str) -> Set[str]:
    return {el for el, q in charges.items() if q < 0 and el != ligand}


def _bond_neighbor_indices(
    i: int,
    symbols: List[str],
    pts: NDArray[np.float64],
    pair_cuts: PairCuts | None = None,
) -> List[int]:
    from .analysis import _pair_cut, _pair_cut_calibrated
    xi = pts[i]
    out: List[int] = []
    for j, sj in enumerate(symbols):
        if j == i:
            continue
        rc = _pair_cut_calibrated(symbols[i], sj, pair_cuts) if pair_cuts is not None else _pair_cut(symbols[i], sj)
        if rc <= 0:
            continue
        if float(np.linalg.norm(pts[j] - xi)) <= rc + 1e-12:
            out.append(j)
    return out


def _pick_bundle_anion_indices(
    symbols: List[str],
    pts: NDArray[np.float64],
    cn: NDArray[np.int_],
    charges: Dict[str, int],
    ligand: str,
    mem: List[List[int]],
    cation_idx: int,
    *,
    need: int = 2,
    pair_cuts: PairCuts | None = None,
) -> List[int]:
    """Pick surface native anions whose swap to ligand balances removing one +2 cation."""
    native = _native_anion_elements(charges, ligand)
    picked: List[int] = []
    seen: set[int] = set()

    def _try_add(idx: int, priority: int, bucket: List[Tuple[int, int, int]]) -> None:
        if idx in seen or idx == cation_idx:
            return
        if symbols[idx] not in native:
            return
        if int(cn[idx]) > 3:
            return
        if not mem[idx]:
            return
        bucket.append((priority, int(cn[idx]), idx))

    bucket: List[Tuple[int, int, int]] = []
    for j in _bond_neighbor_indices(cation_idx, symbols, pts, pair_cuts):
        _try_add(j, 0, bucket)
    for i, s in enumerate(symbols):
        _try_add(i, 1, bucket)

    bucket.sort(key=lambda t: (t[0], t[1], t[2]))
    for _prio, _cnv, idx in bucket:
        if len(picked) >= need:
            break
        picked.append(idx)
        seen.add(idx)
    return picked


def _try_q_neutral_cation_removal_bundle(
    symbols: List[str],
    pts: NDArray[np.float64],
    frames: List[tuple],
    charges: Dict[str, int],
    ligand: str,
    surf_tol: float,
    pair_cuts: PairCuts | None,
    uv_taken: Dict[int, List[Tuple[float, float]]],
    edit_count_facet: Dict[int, int],
    mem: List[List[int]],
    cn: NDArray[np.int_],
    cation_idx: int,
    *,
    region_masks: Optional[List[NDArray[np.bool_]]] = None,
    region_removals: Optional[Dict[int, int]] = None,
    atom_regions: Optional[List[int]] = None,
    verbose: bool = False,
) -> Tuple[bool, NDArray[np.float64]]:
    """
    Remove one surface CN<=2 cation and swap two surface anions to ligand (net Q=0).
    """
    q_cat = int(charges.get(symbols[cation_idx], 0))
    if q_cat <= 0:
        return False, pts
    # Two anion swaps (+2 for -2→-1) balance one +2 cation removal.
    anion_swaps = _pick_bundle_anion_indices(
        symbols, pts, cn, charges, ligand, mem, cation_idx, need=2, pair_cuts=pair_cuts,
    )
    if len(anion_swaps) < 2:
        return False, pts

    reg_removed = _atom_region_index(cation_idx, region_masks, atom_regions)
    before = int(sum(int(charges.get(s, 0)) for s in symbols))
    for ai in anion_swaps:
        old = symbols[ai]
        if verbose:
            print(
                f"bundle: swap anion {old}#{ai} (CN={int(cn[ai])}) → {ligand} "
                f"(companion to cation removal #{cation_idx})"
            )
        symbols[ai] = ligand
        _record_uv_allfacets(ai, pts, frames, surf_tol, uv_taken, edit_count_facet)

    cat_el = symbols[cation_idx]
    if verbose:
        print(f"bundle: remove cation {cat_el}#{cation_idx} (CN={int(cn[cation_idx])})")
    _record_uv_allfacets(cation_idx, pts, frames, surf_tol, uv_taken, edit_count_facet)
    symbols.pop(cation_idx)
    pts = np.delete(pts, cation_idx, axis=0)
    if atom_regions is not None:
        atom_regions.pop(cation_idx)

    after = int(sum(int(charges.get(s, 0)) for s in symbols))
    if verbose and after != before:
        print(f"   ↳ bundle Q:{before:+d}→{after:+d} (expected unchanged)")

    if region_removals is not None:
        region_removals[reg_removed] = region_removals.get(reg_removed, 0) + 1
    return True, pts


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
    stack_passivation: bool = False,
    region_masks: Optional[List[NDArray[np.bool_]]] = None,
    region_removals: Optional[Dict[int, int]] = None,
    atom_regions: Optional[List[int]] = None,
) -> Tuple[List[str], NDArray[np.float64], Dict[int, List[Tuple[float, float]]], Dict[int, int]]:
    """
    Always run before charge balancing:
      • swap all surface anions with CN<=2 to the anion ligand (e.g., Se2- -> Cl-)
      • remove all surface cations with CN<=2 (spacing-aware)
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
        mem = _facet_members()
        cn  = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
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

        # B. remove all cation CN<=2 (spacing-aware)
        while True:
            mem = _facet_members()
            cn  = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
            cands = []
            for i, s in enumerate(symbols):
                if charges.get(s, 0) <= 0: continue
                if not _is_surface(i, mem): continue
                if int(round(cn[i])) > 2: continue
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
            if stack_passivation:
                cands.sort(key=lambda t: (
                    region_removals.get(_atom_region_index(t[3], region_masks, atom_regions), 0) if region_removals is not None else 0,
                    -t[0], -t[1],
                ))
            else:
                cands.sort(reverse=True)
            _, _, _, i, s, _fid = cands[0]
            q_cat = int(charges.get(s, 0))
            q_now = int(sum(int(charges.get(sym, 0)) for sym in symbols))
            if stack_passivation and q_cat > 0 and q_now - q_cat < 0:
                ok, pts = _try_q_neutral_cation_removal_bundle(
                    symbols, pts, frames, charges, anion_ligand, surf_tol, pair_cuts,
                    uv_taken, edit_count_facet, mem, cn, i,
                    region_masks=region_masks, region_removals=region_removals,
                    atom_regions=atom_regions, verbose=verbose,
                )
                if ok:
                    changed = True
                    break
                # try next candidate in another region
                progressed_bundle = False
                for _, _, _, alt_i, alt_s, _ in cands[1:]:
                    ok, pts = _try_q_neutral_cation_removal_bundle(
                        symbols, pts, frames, charges, anion_ligand, surf_tol, pair_cuts,
                        uv_taken, edit_count_facet, mem, cn, alt_i,
                        region_masks=region_masks, region_removals=region_removals,
                        atom_regions=atom_regions, verbose=verbose,
                    )
                    if ok:
                        changed = True
                        progressed_bundle = True
                        break
                if progressed_bundle:
                    break
                break
            if verbose:
                print(f"prepass: remove cation {s}#{i} (CN={int(round(cn[i]))})")
            _record_uv_allfacets(i, pts, frames, surf_tol, uv_taken, edit_count_facet)
            reg_removed = _atom_region_index(i, region_masks, atom_regions)
            symbols.pop(i)
            pts = np.delete(pts, i, axis=0)
            if atom_regions is not None:
                atom_regions.pop(i)
            if stack_passivation and region_removals is not None:
                region_removals[reg_removed] = region_removals.get(reg_removed, 0) + 1
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
        symbols, pts, planes, surf_tol, charges, true_bulk_cn=true_bulk_cn, pair_cuts=pair_cuts
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
    cn_bi: Optional[NDArray[np.int_]] = None,
    bulk_cn: Optional[Dict[str, int]] = None,
    # geometry classifiers (optional; pass from a higher-level passivation policy if available)
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
    from .analysis import coord_numbers_bipartite

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
    if cn_bi is None:
        cn_bi = coord_numbers_bipartite(symbols, ptsA, charges, pair_cuts=pair_cuts)
    if bulk_cn is None:
        interior = np.ones(N, dtype=bool)
        for n, d in planes:
            interior &= ~((float(d) - ptsA @ np.asarray(n, float)) < surf_tol)
        arr_sym = np.array(symbols, dtype=object)
        bulk_cn = {}
        for el in set(symbols):
            vals = cn_bi[(arr_sym == el) & interior]
            if vals.size == 0:
                vals = cn_bi[arr_sym == el]
            bulk_cn[el] = int(np.bincount(vals.astype(int)).argmax()) if vals.size else 0

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
