from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .nc_types import Plane
from .analysis import (
    PairCuts,
    derive_pair_cuts_from_cif,
    pretty_print_pair_cuts,
    coord_numbers_bipartite,
    bulk_cn_opposite_by_interior,
    compute_strict_missing_bond_vectors,
    compute_cif_virtual_sites,
)
from .analysis import _pair_cut as pc
from .analysis import _pair_cut_calibrated
from .passivation import (
    prepass_surface_cleanup,
    collect_anion_candidates,
    _build_facet_frames,
    _plane_uv,
    _incident_facets,
    _record_uv_allfacets,
    _facet_memberships,
    _collect_cation_sites,
    _intersections_geometry,
    _role_by_geometry,
    _atom_region_index,
    _try_q_neutral_cation_removal_bundle,
    atom_regions_from_masks,
)
from .io_utils import write_xyz, center_coords

# Role ranks: 0=unique, 1=edge, 2=vertex
ROLE_ORDER_VEU = ["unique", "edge", "vertex"]
UVCache = Dict[Tuple[int, int], Tuple[float, float]]  # (facet_id, atom_idx) -> (u,v)


def derive_true_bulk_cn_from_cif(
    cif_path: str,
    charges: Dict[str, int],
    pair_cuts: Optional[PairCuts] = None,
) -> Dict[str, int]:
    import re
    from pymatgen.core import Structure
    try:
        struct = Structure.from_file(cif_path)
        bulk_cn = {}
        for site in struct.sites:
            el = re.sub(r'[^a-zA-Z]', '', site.species_string)
            if el not in bulk_cn:
                neighs = struct.get_neighbors(site, r=4.5)
                cn_count = 0
                for n_site in neighs:
                    n_el = re.sub(r'[^a-zA-Z]', '', n_site.species_string)
                    if charges.get(el, 0) * charges.get(n_el, 0) < 0:
                        rc = _pair_cut_calibrated(el, n_el, pair_cuts)
                        if n_site.nn_distance <= rc + 0.15:
                            cn_count += 1
                bulk_cn[el] = max(bulk_cn.get(el, 0), cn_count)
        return bulk_cn
    except Exception:
        return {}


def _total_Q(symbols: List[str], charges: Dict[str, int]) -> int:
    return int(sum(int(charges.get(s, 0)) for s in symbols))


def _surface_low_cn_cations_remain(
    symbols: List[str],
    pts: NDArray[np.float64],
    frames: List[Plane],
    cn_bi: NDArray[np.int_],
    charges: Dict[str, int],
    surf_tol: float,
    *,
    prepass_mode: str = "standard",
    prepass_min_cn_terrace: int = 2,
    prepass_min_cn_edge: int = 2,
    prepass_min_cn_vertex: int = 2,
) -> bool:
    outer_thr = 0.35 * surf_tol
    edges_by_facet = verts_by_facet = None
    edge_tol = max(0.25 * surf_tol, 0.35)
    vertex_tol = max(0.75 * surf_tol, 0.75)
    if prepass_mode == "role-aware":
        try:
            edges_by_facet, verts_by_facet = _intersections_geometry(frames)
        except Exception:
            edges_by_facet = verts_by_facet = None
    for i, s in enumerate(symbols):
        if charges.get(s, 0) <= 0:
            continue
        inc = _incident_facets(i, pts, frames, surf_tol)
        if not inc:
            continue

        cation_cn = int(cn_bi[i])
        role = 0
        if prepass_mode == "role-aware":
            if edges_by_facet is not None and verts_by_facet is not None:
                fid_min = min(inc, key=lambda fid: frames[fid][1] - float(np.dot(pts[i], frames[fid][0])))
                _role_name, role = _role_by_geometry(
                    i, fid_min, pts, frames, edges_by_facet, verts_by_facet, edge_tol, vertex_tol
                )
            else:
                m_count = len(inc)
                role = 0 if m_count == 1 else (1 if m_count == 2 else 2)
            if role == 0:
                min_cn = prepass_min_cn_terrace
            elif role == 1:
                min_cn = prepass_min_cn_edge
            else:
                min_cn = prepass_min_cn_vertex
        else:
            min_cn = 3

        if cation_cn >= min_cn:
            continue

        depths = [frames[fid][1] - float(np.dot(pts[i], frames[fid][0])) for fid in inc]
        if min(depths) >= outer_thr:
            continue
        return True
    return False


def _min_ligand_ligand_spacing(
    symbols: List[str],
    pts: NDArray[np.float64],
    ligand: str,
) -> float:
    idx = [i for i, s in enumerate(symbols) if s == ligand]
    if len(idx) < 2:
        return 2.5
    
    ligand_pts = np.asarray(pts[idx], float)
    tree = cKDTree(ligand_pts)
    dists, _ = tree.query(ligand_pts, k=2)
    dmin = np.min(dists[:, 1])
    return 1.05 * dmin if np.isfinite(dmin) else 2.5


def _ligand_position_allowed(
    new_pos: NDArray[np.float64],
    symbols: List[str],
    pts: NDArray[np.float64],
    ligand: str,
) -> bool:
    idx = [i for i, s in enumerate(symbols) if s == ligand]
    if not idx:
        return True
    min_dist = _min_ligand_ligand_spacing(symbols, pts, ligand)
    return all(float(np.linalg.norm(new_pos - pts[i])) >= min_dist - 1e-6 for i in idx)


def _get_vacant_directions_for_site(
    j: int,
    symbols: List[str],
    pts: NDArray[np.float64],
    ref_struct,
    pt_tree: cKDTree,
) -> List[np.ndarray]:
    if ref_struct is None:
        return []
    try:
        f_coord = ref_struct.lattice.get_fractional_coords(pts[j])
        site_idx = int(np.argmin([
            float(np.linalg.norm((site.frac_coords - f_coord + 0.5) % 1 - 0.5))
            for site in ref_struct.sites
        ]))
        bulk_site = ref_struct.sites[site_idx]
        all_neigh = ref_struct.get_neighbors(bulk_site, r=4.5)
        if not all_neigh:
            return []
        min_dist = min(neigh.nn_distance for neigh in all_neigh)
        bulk_neighbors = [neigh for neigh in all_neigh if neigh.nn_distance <= 1.15 * min_dist]
        ideal_vectors = []
        for neigh in bulk_neighbors:
            vec = neigh.coords - bulk_site.coords
            ln = float(np.linalg.norm(vec))
            if ln > 1e-8:
                ideal_vectors.append(vec / ln)
        actual_vectors = []
        neighbor_ids = pt_tree.query_ball_point(pts[j], r=4.5)
        for act_idx in neighbor_ids:
            if act_idx == j:
                continue
            rc = pc(symbols[j], symbols[act_idx])
            if rc <= 0:
                continue
            if float(np.linalg.norm(pts[act_idx] - pts[j])) <= rc + 1e-12:
                vec = pts[act_idx] - pts[j]
                ln = float(np.linalg.norm(vec))
                if ln > 1e-8:
                    actual_vectors.append(vec / ln)
        vacant_vectors = []
        for t_k in ideal_vectors:
            is_occupied = False
            for v_a in actual_vectors:
                if float(np.dot(v_a, t_k)) > 0.90:
                    is_occupied = True
                    break
            if not is_occupied:
                vacant_vectors.append(t_k)
        return vacant_vectors
    except Exception:
        return []


def _trial_ligand_addition_score(
    symbols: List[str],
    pts: NDArray[np.float64],
    charges: Dict[str, int],
    ligand: str,
    pair_cuts: Optional[PairCuts],
    new_pos: NDArray[np.float64],
    *,
    host_idx: int,
    cn_before: NDArray[np.int_],
    pt_tree: cKDTree,
    bulk_map: Optional[Dict[str, int]] = None,
    max_search_cut: float = 5.0,
    ref_struct = None,
) -> Tuple[int, int, int, int]:
    """
    Score a proposed ligand placement by the local CN gain it creates on nearby
    positive cations. Returns:
      (score, host_gain, touched_cations, overcoord_penalty)
    """
    host_pos = pts[host_idx]
    local_cut = max(4.5, float(max_search_cut))

    score = 0
    host_gain = 0
    touched = 0
    over_penalty = 0
    neighbor_ids = pt_tree.query_ball_point(np.asarray(new_pos, float), r=local_cut)
    for j in neighbor_ids:
        sj = symbols[j]
        if charges.get(sj, 0) <= 0:
            continue
        dist = float(np.linalg.norm(pts[j] - new_pos))
        rc = _pair_cut_calibrated(sj, ligand, pair_cuts)
        if dist > rc:
            continue

        # Cone directional filter to prevent spillover/false overcoordination penalties on adjacent fully-coordinated cations
        if j != host_idx and ref_struct is not None:
            vacant_dirs = _get_vacant_directions_for_site(j, symbols, pts, ref_struct, pt_tree)
            u_j = (new_pos - pts[j]) / (dist + 1e-12)
            if not any(float(np.dot(u_j, t_k)) > 0.80 for t_k in vacant_dirs):
                continue

        before = int(cn_before[j])
        after = before + 1
        if after != before:
            touched += 1
        bulk_target = int(bulk_map.get(sj, after) if bulk_map is not None else after)
        before_clamped = min(before, bulk_target)
        after_clamped = min(after, bulk_target)
        gain = max(0, after_clamped - before_clamped)
        if j == host_idx:
            host_gain = gain
        score += 6 * gain
        if after > bulk_target:
            over = after - bulk_target
            over_penalty += over
            score -= 8 * over

    # Prefer placements that touch multiple nearby cations and do not just
    # rebalance the host atom in isolation.
    score += 2 * max(0, touched - 1)
    # Tiny bias toward staying close to the host when scores are otherwise equal.
    score += int(round(10.0 / max(1.0, float(np.linalg.norm(np.asarray(new_pos, float) - host_pos)))))
    return score, host_gain, touched, over_penalty


def _neighbors(i: int, symbols: List[str], pts: NDArray[np.float64], pt_tree: Optional[cKDTree] = None) -> List[int]:
    if pt_tree is None:
        pt_tree = cKDTree(pts)
    
    uniq = set(symbols)
    max_rc = max(pc(symbols[i], s) for s in uniq) if uniq else 0.0
    if max_rc <= 0:
        return []
        
    idxs = pt_tree.query_ball_point(pts[i], r=max_rc)
    out: List[int] = []
    for j in idxs:
        if j == i:
            continue
        rc = pc(symbols[i], symbols[j])
        if rc <= 0:
            continue
        if np.linalg.norm(pts[j] - pts[i]) <= rc + 1e-12:
            out.append(j)
    return out


def _ligand_protected_cation_mask(
    symbols: List[str],
    pts: NDArray[np.float64],
    charges: Dict[str, int],
    ligand: str,
    pair_cuts: Optional[PairCuts],
) -> NDArray[np.bool_]:
    protected = np.zeros(len(symbols), dtype=bool)
    lig_idx = [i for i, s in enumerate(symbols) if s == ligand]
    cat_idx = [i for i, s in enumerate(symbols) if charges.get(s, 0) > 0]
    if not lig_idx or not cat_idx:
        return protected

    max_rc = max(_pair_cut_calibrated(symbols[i], ligand, pair_cuts) for i in cat_idx)
    lig_tree = cKDTree(pts[lig_idx])
    for i in cat_idx:
        hits = lig_tree.query_ball_point(pts[i], r=max_rc + 1e-6)
        if not hits:
            continue
        rc = _pair_cut_calibrated(symbols[i], ligand, pair_cuts)
        xi = pts[i]
        for h in hits:
            if np.linalg.norm(pts[lig_idx[h]] - xi) <= rc + 1e-6:
                protected[i] = True
                break
    return protected


def _prune_orphan_ligands(
    symbols: List[str],
    pts: NDArray[np.float64],
    charges: Dict[str, int],
    ligand: str,
    pair_cuts: Optional[PairCuts],
    *,
    verbose: bool = True,
    atom_regions: Optional[List[int]] = None,
) -> Tuple[List[str], NDArray[np.float64], int]:
    lig_idx = [i for i, s in enumerate(symbols) if s == ligand]
    cat_idx = [i for i, s in enumerate(symbols) if charges.get(s, 0) > 0]
    if not lig_idx or not cat_idx:
        orphan_idx = lig_idx
    else:
        max_rc = max(_pair_cut_calibrated(ligand, symbols[i], pair_cuts) for i in cat_idx)
        cat_tree = cKDTree(pts[cat_idx])
        orphan_idx = []
        for i in lig_idx:
            hits = cat_tree.query_ball_point(pts[i], r=max_rc + 1e-6)
            has_host = False
            xi = pts[i]
            for h in hits:
                j = cat_idx[h]
                rc = _pair_cut_calibrated(ligand, symbols[j], pair_cuts)
                if np.linalg.norm(pts[j] - xi) <= rc + 1e-6:
                    has_host = True
                    break
            if not has_host:
                orphan_idx.append(i)
    if not orphan_idx:
        return symbols, pts, 0

    keep = np.ones(len(symbols), dtype=bool)
    keep[orphan_idx] = False
    before = _total_Q(symbols, charges)
    symbols = [s for s, k in zip(symbols, keep) if k]
    pts = pts[keep]
    if atom_regions is not None:
        atom_regions[:] = [r for r, k in zip(atom_regions, keep) if k]
    after = _total_Q(symbols, charges)
    if verbose:
        print(f"prune orphan {ligand}: removed {len(orphan_idx)} unbound ligand(s) | Q:{before:+d}→{after:+d}")
    return symbols, pts, len(orphan_idx)


def _get_uv(frames: List[Plane], fid: int, idx: int, pts: NDArray[np.float64], cache: UVCache) -> Tuple[float, float]:
    key = (fid, int(idx))
    if key in cache:
        return cache[key]
    uv = _plane_uv(frames, fid, pts[int(idx)])
    cache[key] = uv
    return uv


# ---------- PRIORITY 1: swap native surface anions with CN<=2 to ligand X- ----------
def _priority1_swap_undercoord_anions_once(
    symbols: List[str],
    pts: NDArray[np.float64],
    frames: List[Plane],
    mem: List[List[int]],
    cn_bi: NDArray[np.int_],
    charges: Dict[str, int],
    ligand: str,
    surf_tol: float,
    uv_cache: UVCache,
    *, verbose: bool = True,
    pt_tree: Optional[cKDTree] = None,
) -> Tuple[bool, List[str], NDArray[np.float64]]:
    for i, s in enumerate(list(symbols)):
        if charges.get(s, 0) >= 0:
            continue
        if s == ligand:
            continue
        if int(cn_bi[i]) > 2:
            continue
        if not mem[i]:
            continue
        # avoid creating new CN<3 cations
        harmful = False
        for j in _neighbors(i, symbols, pts, pt_tree=pt_tree):
            if charges.get(symbols[j], 0) <= 0:
                continue
            if int(cn_bi[j]) <= 3:
                harmful = True
                break
        if harmful:
            continue
        before = _total_Q(symbols, charges)
        old = symbols[i]
        symbols[i] = ligand
        after = _total_Q(symbols, charges)
        if verbose:
            print(f"swap {old}#{i} (CN<=2) → {ligand} | Q:{before:+d}→{after:+d}")
        return True, symbols, pts  # single action per iteration
    return False, symbols, pts


# ---------- PRIORITY 2: remove surface cations with CN<=2 ----------
def _priority2_remove_low_cn_cation_once(
    symbols: List[str],
    pts: NDArray[np.float64],
    frames: List[Plane],
    mem: List[List[int]],
    cn_bi: NDArray[np.int_],
    charges: Dict[str, int],
    surf_tol: float,
    uv_taken: Dict[int, List[Tuple[float, float]]],
    edit_count_facet: Dict[int, int],
    uv_cache: UVCache,
    ligand: str,
    pair_cuts: Optional[PairCuts],
    *,
    verbose: bool = True,
    stack_passivation: bool = False,
    region_masks: Optional[List[NDArray[np.bool_]]] = None,
    region_removals: Optional[Dict[int, int]] = None,
    atom_regions: Optional[List[int]] = None,
    prepass_mode: str = "standard",
    prepass_min_cn_terrace: int = 2,
    prepass_min_cn_edge: int = 2,
    prepass_min_cn_vertex: int = 2,
) -> Tuple[bool, List[str], NDArray[np.float64]]:
    use_stack = stack_passivation
    cands: List[Tuple] = []
    edges_by_facet = verts_by_facet = None
    edge_tol = max(0.25 * surf_tol, 0.35)
    vertex_tol = max(0.75 * surf_tol, 0.75)
    if prepass_mode == "role-aware":
        try:
            edges_by_facet, verts_by_facet = _intersections_geometry(frames)
        except Exception:
            edges_by_facet = verts_by_facet = None
    for i, s in enumerate(symbols):
        if charges.get(s, 0) <= 0:
            continue

        inc = _incident_facets(i, pts, frames, surf_tol)
        if not inc:
            continue

        cation_cn = int(cn_bi[i])
        role = 0
        if prepass_mode == "role-aware":
            fid_for_role = min(inc, key=lambda fid: frames[fid][1] - float(np.dot(pts[i], frames[fid][0])))
            if edges_by_facet is not None and verts_by_facet is not None:
                _role_name, role = _role_by_geometry(
                    i, fid_for_role, pts, frames, edges_by_facet, verts_by_facet, edge_tol, vertex_tol
                )
            else:
                m_count = len(inc)
                role = 0 if m_count == 1 else (1 if m_count == 2 else 2)
            if role == 0:
                min_cn = prepass_min_cn_terrace
            elif role == 1:
                min_cn = prepass_min_cn_edge
            else:
                min_cn = prepass_min_cn_vertex
        else:
            min_cn = 3

        if cation_cn >= min_cn:
            continue

        depths = [(fid, frames[fid][1] - float(np.dot(pts[i], frames[fid][0]))) for fid in inc]
        fid, depth = min(depths, key=lambda t: t[1])
        inc_count = len(inc)
        role_rank = 2 if inc_count >= 3 else (1 if inc_count == 2 else 0)
        uv = _get_uv(frames, fid, i, pts, uv_cache)
        taken = uv_taken.get(fid, [])
        if taken:
            from math import hypot
            dmin = min(hypot(uv[0]-u, uv[1]-v) for (u, v) in taken)
        else:
            dmin = float("inf")
        if use_stack:
            reg = _atom_region_index(i, region_masks, atom_regions)
            reg_count = region_removals.get(reg, 0) if region_removals is not None else 0
            cands.append((reg_count, role_rank, -dmin, depth, fid, i))
        else:
            cands.append((role_rank, -dmin, depth, fid, i))

    if not cands:
        return False, symbols, pts

    if use_stack:
        cands.sort(key=lambda t: (t[0], -t[1], t[2], t[3], t[5]))
        _reg, role_rank, _negdmin, depth, fid, i = cands[0]
    else:
        cands.sort(key=lambda t: (-t[0], t[1], t[2], t[4]))
        role_rank, _negdmin, depth, fid, i = cands[0]

    q_cat = int(charges.get(symbols[i], 0))
    q_now = _total_Q(symbols, charges)
    if use_stack and q_cat > 0 and q_now - q_cat < 0:
        cn_now = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
        for _r, _rr, _nd, _dp, _f, alt_i in cands:
            ok, pts = _try_q_neutral_cation_removal_bundle(
                symbols, pts, frames, charges, ligand, surf_tol, pair_cuts,
                uv_taken, edit_count_facet, mem, cn_now, alt_i,
                region_masks=region_masks, region_removals=region_removals,
                atom_regions=atom_regions, verbose=verbose,
            )
            if ok:
                cn2 = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
                native_anions = {el for el, v in charges.items() if v < 0 and el != ligand}
                for j, sj in enumerate(list(symbols)):
                    if charges.get(sj, 0) >= 0 or sj == ligand or sj not in native_anions:
                        continue
                    if len(_incident_facets(j, pts, frames, surf_tol)) == 0:
                        continue
                    if int(cn2[j]) < 3:
                        old = symbols[j]
                        symbols[j] = ligand
                        if verbose:
                            print(f"stabilize: swap {old}#{j} → {ligand} (neighbor CN<3)")
                return True, symbols, pts
        return False, symbols, pts

    before = _total_Q(symbols, charges)
    if verbose:
        print(f"remove {symbols[i]}#{i} (CN={int(cn_bi[i])}, role={ROLE_ORDER_VEU[role_rank]}, facet={fid}, depth={depth:.2f} Å)")
    _record_uv_allfacets(i, pts, frames, surf_tol, uv_taken, edit_count_facet)
    reg_removed = _atom_region_index(i, region_masks, atom_regions)
    removed_elem = symbols.pop(i)
    pts = np.delete(pts, i, axis=0)
    if atom_regions is not None:
        atom_regions.pop(i)
    symbols, pts, _ = _prune_orphan_ligands(
        symbols, pts, charges, ligand, pair_cuts, verbose=verbose, atom_regions=atom_regions
    )
    if use_stack and region_removals is not None:
        region_removals[reg_removed] = region_removals.get(reg_removed, 0) + 1
    after = _total_Q(symbols, charges)
    if verbose:
        print(f"   ↳ Q:{before:+d}→{after:+d} (removed {removed_elem})")

    # stabilization: convert native anions (not ligand) that dropped to CN<3 → ligand
    cn2 = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
    native_anions = {el for el, v in charges.items() if v < 0 and el != ligand}
    for j, sj in enumerate(list(symbols)):
        if charges.get(sj, 0) >= 0:
            continue
        if sj == ligand:
            continue
        if sj not in native_anions:
            continue
        if len(_incident_facets(j, pts, frames, surf_tol)) == 0:
            continue
        if int(cn2[j]) < 3:
            old = symbols[j]
            bQ = _total_Q(symbols, charges)
            symbols[j] = ligand
            aQ = _total_Q(symbols, charges)
            if verbose:
                print(f"stabilize: swap {old}#{j} → {ligand} (neighbor CN<3) | Q:{bQ:+d}→{aQ:+d}")
    return True, symbols, pts


# ---------- PRIORITY 3A: Q<0 — swap stable anions (CN>=3) using V→E→U + FPS ----------
def _collect_anion_candidates_flat(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    charges: Dict[str, int],
    ligand: str,
    surf_tol: float,
    pair_cuts: Optional[PairCuts],
    *,
    ignore_deficit: bool = False,
) -> List[dict]:
    outer, subl = collect_anion_candidates(
        symbols, pts, planes, charges, ligand, surf_tol, verbose=False,
        pair_cuts=pair_cuts, ignore_deficit=ignore_deficit,
    )
    return list(outer) if len(outer) > 0 else list(outer) + list(subl)


def _priority3_balance_negative_q(
    symbols: List[str],
    pts: NDArray[np.float64],
    frames: List[Plane],
    planes_raw: List[Plane],
    mem: List[List[int]],
    cn_bi: NDArray[np.int_],
    charges: Dict[str, int],
    ligand: str,
    surf_tol: float,
    uv_taken: Dict[int, List[Tuple[float, float]]],
    edit_count_facet: Dict[int, int],
    uv_cache: UVCache,
    pair_cuts: Optional[PairCuts],
    *, verbose: bool = True,
    stack_passivation: bool = False,
) -> Tuple[bool, List[str], NDArray[np.float64]]:
    cand = _collect_anion_candidates_flat(
        symbols, pts, planes_raw, charges, ligand, surf_tol, pair_cuts,
        ignore_deficit=stack_passivation,
    )
    if not cand:
        return False, symbols, pts

    # prefer CN=3, then 4, then 5+
    for cn_target in (3, 4, 5, 6):
        subset = [c for c in cand if int(c.get("cn", 0)) == cn_target]
        if not subset:
            continue
        # refresh spacing against global uv_taken
        for r in subset:
            fid = int(r.get("fid", -1))
            idx = int(r.get("idx", -1))
            if fid < 0 or idx < 0:
                continue
            uv = _get_uv(frames, fid, idx, pts, uv_cache)
            taken = uv_taken.get(fid, [])
            if taken:
                from math import hypot
                r["dmin_uv"] = min(hypot(uv[0]-u, uv[1]-v) for (u, v) in taken)
            else:
                r["dmin_uv"] = float("inf")
        # role V->E->U, then spacing, then shallower
        subset.sort(key=lambda r: (
            int(r.get("role_rank", 0)),           # higher is better: vertex>edge>unique
            float(r.get("dmin_uv", 0.0)),
            -float(r.get("depth", 0.0)),
        ), reverse=True)
        picked = subset[0]
        i = int(picked.get("idx", -1))
        fid = int(picked.get("fid", -1))
        if i < 0 or fid < 0:
            continue
        old = symbols[i]
        if old == ligand:
            continue
        before = _total_Q(symbols, charges)
        symbols[i] = ligand
        after = _total_Q(symbols, charges)
        _record_uv_allfacets(i, pts, frames, surf_tol, uv_taken, edit_count_facet)
        if verbose:
            role_idx = int(picked.get("role_rank", 0))
            role_name = ROLE_ORDER_VEU[role_idx]
            depth = float(picked.get("depth", 0.0))
            print(f"swap {old}#{i} (CN={cn_target}, {role_name}, facet={fid}, depth={depth:.2f} Å) → {ligand} | Q:{before:+d}→{after:+d}")
        # single action per call
        return True, symbols, pts
    return False, symbols, pts


# ---------- PRIORITY 3B: Q>0 — remove cations (outer-only; V>E>U; CN tiers) ----------
def _priority3_balance_positive_q_remove(
    symbols: List[str],
    pts: NDArray[np.float64],
    frames: List[Plane],
    planes_raw: List[Plane],
    mem: List[List[int]],
    cn_bi: NDArray[np.int_],
    charges: Dict[str, int],
    surf_tol: float,
    uv_taken: Dict[int, List[Tuple[float, float]]],
    edit_count_facet: Dict[int, int],
    uv_cache: UVCache,
    ligand: str,
    pair_cuts: Optional[PairCuts],
    positive_q_strategy_selector: Optional[Callable[[int, str, NDArray[np.float64]], Optional[str]]] = None,
    *, verbose: bool = True,
    atom_regions: Optional[List[int]] = None,
    cif_path: Optional[str] = None,
) -> Tuple[bool, List[str], NDArray[np.float64]]:
    outer_thr = 0.35 * surf_tol
    true_bulk_cn = derive_true_bulk_cn_from_cif(cif_path, charges, pair_cuts=pair_cuts) if cif_path else None
    bulk_map = bulk_cn_opposite_by_interior(symbols, pts, planes_raw, surf_tol, charges, pair_cuts=pair_cuts, true_bulk_cn=true_bulk_cn)

    cands_by_cn: Dict[int, List[Tuple[int, float, float, int, int, int]]] = {}
    protected = _ligand_protected_cation_mask(symbols, pts, charges, ligand, pair_cuts)
    for i, s in enumerate(symbols):
        if charges.get(s, 0) <= 0:
            continue
        if positive_q_strategy_selector is not None:
            if positive_q_strategy_selector(i, s, pts[i]) != "remove":
                continue
        if protected[i]:
            continue
        inc = _incident_facets(i, pts, frames, surf_tol)
        if not inc:
            continue
        depths = [(fid, frames[fid][1] - float(np.dot(pts[i], frames[fid][0]))) for fid in inc]
        fid, depth = min(depths, key=lambda t: t[1])
        shell = "outer" if depth < outer_thr else "sublayer"
        if shell != "outer":
            continue
        ci = int(cn_bi[i])
        tgt = int(bulk_map.get(s, ci))
        deficit = tgt - ci
        inc_count = len(inc)
        role_rank = 2 if inc_count >= 3 else (1 if inc_count == 2 else 0)  # V>E>U for removal
        uv = _get_uv(frames, fid, i, pts, uv_cache)
        taken = uv_taken.get(fid, [])
        if taken:
            from math import hypot
            dmin = min(hypot(uv[0]-u, uv[1]-v) for (u, v) in taken)
        else:
            dmin = float("inf")
        cands_by_cn.setdefault(ci, []).append((role_rank, -dmin, depth, fid, i, deficit))

    if not cands_by_cn:
        return False, symbols, pts

    for cn_tier in sorted(cands_by_cn.keys()):  # 1 -> 2 -> 3 -> ...
        lst = cands_by_cn[cn_tier]
        lst.sort(key=lambda t: (-t[5], -t[0], t[1], t[2], t[4]))
        role_rank, neg_dmin, depth, fid, i, deficit = lst[0]

        before = _total_Q(symbols, charges)
        role_name = ROLE_ORDER_VEU[role_rank]
        if verbose:
            print(f"remove {symbols[i]}#{i} (CN={cn_tier}, role={role_name}, facet={fid}, shell=outer, depth={depth:.2f} Å, deficit={deficit})")
        _record_uv_allfacets(i, pts, frames, surf_tol, uv_taken, edit_count_facet)
        removed_elem = symbols.pop(i)
        pts = np.delete(pts, i, axis=0)
        if atom_regions is not None:
            atom_regions.pop(i)
        symbols, pts, _ = _prune_orphan_ligands(
            symbols, pts, charges, ligand, pair_cuts, verbose=verbose, atom_regions=atom_regions
        )
        after = _total_Q(symbols, charges)
        if verbose:
            print(f"   ↳ Q:{before:+d}→{after:+d} (removed {removed_elem})")

        # Post-stabilization: native anions (not ligand) that dropped to CN<3 → ligand
        cn2 = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
        native_anions = {el for el, v in charges.items() if v < 0 and el != ligand}
        for j, sj in enumerate(list(symbols)):
            if charges.get(sj, 0) >= 0:
                continue
            if sj == ligand:
                continue
            if sj not in native_anions:
                continue
            if len(_incident_facets(j, pts, frames, surf_tol)) == 0:
                continue
            if int(cn2[j]) < 3:
                old = symbols[j]
                bQ = _total_Q(symbols, charges)
                symbols[j] = ligand
                aQ = _total_Q(symbols, charges)
                if verbose:
                    print(f"stabilize: swap {old}#{j} → {ligand} (neighbor CN<3) | Q:{bQ:+d}→{aQ:+d}")
        return True, symbols, pts

    return False, symbols, pts


# ---------- PRIORITY 3C: Q>0 — add anion ligands (outer-only; UNIQUE>EDGE>VERTEX) ----------
def _priority3_balance_positive_q_add(
    symbols: List[str],
    pts: NDArray[np.float64],
    frames: List[Plane],
    planes_raw: List[Plane],
    mem: List[List[int]],
    cn_bi: NDArray[np.int_],
    charges: Dict[str, int],
    ligand: str,
    surf_tol: float,
    uv_taken: Dict[int, List[Tuple[float, float]]],
    edit_count_facet: Dict[int, int],
    add_count_facet: Dict[int, int],
    host_taken: Dict[int, int],
    uv_cache: UVCache,
    pair_cuts: Optional[PairCuts],
    positive_q_strategy_selector: Optional[Callable[[int, str, NDArray[np.float64]], Optional[str]]] = None,
    *, verbose: bool = True,
    include_sublayer: bool = False,
    stack_passivation: bool = False,
    atom_regions: Optional[List[int]] = None,
    region_masks: Optional[List[NDArray[np.bool_]]] = None,
    cif_path: Optional[str] = None,
) -> Tuple[bool, List[str], NDArray[np.float64]]:
    # gather candidates; include_sublayer=True expands search to sublayer atoms
    true_bulk_cn = derive_true_bulk_cn_from_cif(cif_path, charges, pair_cuts=pair_cuts) if cif_path else None
    bulk_map = bulk_cn_opposite_by_interior(symbols, pts, planes_raw, surf_tol, charges, pair_cuts=pair_cuts, true_bulk_cn=true_bulk_cn)
    pt_tree = cKDTree(np.asarray(pts, float))
    ref_struct = None
    if cif_path:
        try:
            from pymatgen.core import Structure
            ref_struct = Structure.from_file(cif_path)
        except Exception as e:
            if verbose:
                print(f"[debug:add] Reference CIF load failed for template directions: {e}. Using outward normals only.")
    edges_by_facet = verts_by_facet = None
    edge_tol = max(0.25 * surf_tol, 0.35)
    vertex_tol = max(0.75 * surf_tol, 0.75)
    try:
        edges_by_facet, verts_by_facet = _intersections_geometry(frames)
    except Exception:
        edges_by_facet = verts_by_facet = None
    sites = _collect_cation_sites(
        symbols, pts, planes_raw, charges, surf_tol, pair_cuts=pair_cuts,
        outer_only=not include_sublayer, allow_shared=True,
        include_sublayer=include_sublayer,
        cn_bi=cn_bi,
        bulk_cn=bulk_map,
        frames=frames if edges_by_facet is not None and verts_by_facet is not None else None,
        edges_by_facet=edges_by_facet,
        verts_by_facet=verts_by_facet,
        edge_tol=edge_tol,
        vertex_tol=vertex_tol,
    )
    if positive_q_strategy_selector is not None:
        sites = [
            rec for rec in sites
            if positive_q_strategy_selector(rec[0], symbols[rec[0]], pts[rec[0]]) == "add"
        ]
    total_sites = len(sites)
    if verbose:
        d1 = sum(1 for rec in sites if rec[3] == 1)
        u = sum(1 for rec in sites if rec[4] == 0)
        e = sum(1 for rec in sites if rec[4] == 1)
        v = sum(1 for rec in sites if rec[4] == 2)
        facets = len(set(rec[5] for rec in sites))
        print(f"[summary:add] sites={total_sites} | def1={d1}, def>1={total_sites-d1} | U/E/V={u}/{e}/{v} | facets={facets}")
    if not sites:
        return False, symbols, pts

    # Partition by deficit (support any positive integer). Build ordered deficit tiers: [1] + sorted(>1)
    by_def: Dict[int, List[Tuple[int, np.ndarray, float, int, int, int]]] = defaultdict(list)
    for (i, n_out, depth_min, dft, role_rank, fid) in sites:
        if dft >= 1:
            by_def[dft].append((i, n_out, depth_min, dft, role_rank, fid))
    if not by_def:
        return False, symbols, pts
    # Prioritize largest deficit first for physical structural stability
    ordered_defs = sorted(by_def.keys(), reverse=True)

    # Try deficits in descending order; within each, try roles VERTEX(2)→EDGE(1)→UNIQUE(0)
    for def_tier in ordered_defs:
        pool = by_def.get(def_tier, [])
        if not pool:
            continue
        if verbose:
            print(f"[summary:add:pick] chosen_deficit={def_tier} | pool={len(pool)}")

        # Eligibility must follow the current CN deficit. A CN=2 edge/vertex
        # that already received one ligand can still be CN=3 and legitimately
        # need one more ligand; comparing previous additions against the
        # recomputed deficit incorrectly suppresses that second repair.
        sub = list(pool)
        if not sub:
            continue

        # Score each host by its best physically allowed ligand placement.
        # 1. Pre-compute strict missing bond vectors for hosts in this tier
        sub_mask = np.zeros(len(symbols), dtype=bool)
        for (i, n_out, depth_min, dft, role_rank, fid) in sub:
            sub_mask[i] = True

        strict_missing = {}
        if ref_struct is not None:
            try:
                strict_missing = compute_strict_missing_bond_vectors(
                    symbols,
                    pts,
                    charges,
                    pair_cuts,
                    ref_struct,
                    sub_mask,
                    planes_raw,
                    surf_tol,
                )
            except Exception as e:
                if verbose:
                    print(f"[debug:add] compute_strict_missing_bond_vectors failed for host tier: {e}. Falling back to outward normal.")

        # 2. Pre-compute unified CIF-Intersection virtual sites for this tier
        sub_records = {rec[0]: rec for rec in sub}
        host_candidates_map = defaultdict(list)

        merged_cif_sites = []
        if ref_struct is not None:
            try:
                merged_cif_sites = compute_cif_virtual_sites(
                    symbols,
                    pts,
                    charges,
                    pair_cuts,
                    ref_struct,
                    sub_mask,
                    planes_raw,
                    surf_tol,
                )
            except Exception as e:
                if verbose:
                    print(f"[debug:add] compute_cif_virtual_sites failed for host tier: {e}. Falling back to standard normal.")

        if merged_cif_sites:
            # We have exact crystallographic virtual sites (single and bridges)
            for site in merged_cif_sites:
                new_pos = site["pos"]
                hosts = site["hosts"]
                multiplicity = site["multiplicity"]
                # Use the first host as the primary scoring host
                primary_host = hosts[0]
                rec = sub_records.get(primary_host)
                if rec is None:
                    continue
                u_vec = site["u_vecs"][primary_host]
                
                host_candidates_map[primary_host].append({
                    "pos": new_pos,
                    "u_vec": u_vec,
                    "placed_via_bulk": True,
                    "is_bridge": (multiplicity >= 2),
                    "record": rec,
                })
        else:
            # Fallback to standard host-centered candidates along outward normal
            for (i, n_out, depth_min, dft, role_rank, fid) in sub:
                try:
                    rc = pc(symbols[i], ligand)
                    offset = (0.95 * rc) if rc > 0 else 2.5
                except Exception:
                    offset = 2.5
                
                u_vec = n_out / (float(np.linalg.norm(n_out)) + 1e-12)
                pos = pts[i] + u_vec * offset
                
                host_candidates_map[i].append({
                    "pos": pos,
                    "u_vec": u_vec,
                    "placed_via_bulk": False,
                    "is_bridge": False,
                    "record": (i, n_out, depth_min, dft, role_rank, fid),
                    "offset": offset,
                })

        # 3. Evaluate all candidates from all hosts in sub
        all_valid_candidates = []
        for primary_host, cands in host_candidates_map.items():
            for cand in cands:
                new_pos = cand["pos"]
                u_vec = cand["u_vec"]
                placed_via_bulk = cand["placed_via_bulk"]
                is_bridge = cand["is_bridge"]
                (i_rec, n_out, depth_min, dft, role_rank, fid) = cand["record"]
                offset = cand.get("offset", float(np.linalg.norm(new_pos - pts[primary_host])))

                if stack_passivation and not include_sublayer and not _ligand_position_allowed(new_pos, symbols, pts, ligand):
                    continue

                score, host_gain, touched, over_penalty = _trial_ligand_addition_score(
                    symbols,
                    pts,
                    charges,
                    ligand,
                    pair_cuts,
                    new_pos,
                    host_idx=primary_host,
                    cn_before=cn_bi,
                    pt_tree=pt_tree,
                    bulk_map=bulk_map,
                    max_search_cut=max(4.5, abs(_pair_cut_calibrated(symbols[primary_host], ligand, pair_cuts)) + 1.0),
                    ref_struct=ref_struct,
                )

                # Multi-cation repair bonus
                cn3_repaired = 0
                local_cut = max(4.5, float(abs(_pair_cut_calibrated(symbols[primary_host], ligand, pair_cuts)) + 1.0))
                neighbor_ids = pt_tree.query_ball_point(np.asarray(new_pos, float), r=local_cut)
                for j in neighbor_ids:
                    sj = symbols[j]
                    if charges.get(sj, 0) <= 0:
                        continue
                    dist = float(np.linalg.norm(pts[j] - new_pos))
                    rc = _pair_cut_calibrated(sj, ligand, pair_cuts)
                    if dist > rc:
                        continue
                    before = int(cn_bi[j])
                    bulk_target = int(bulk_map.get(sj, before + 1) if bulk_map is not None else before + 1)
                    if before == bulk_target - 1:
                        cn3_repaired += 1
                
                if cn3_repaired >= 2:
                    score += 20 * (cn3_repaired - 1)

                inc_facets = _incident_facets(primary_host, pts, frames, surf_tol)
                add_side_count = (
                    float(np.mean([add_count_facet.get(f, 0) for f in inc_facets]))
                    if inc_facets else 0.0
                )
                edit_side_count = (
                    float(np.mean([edit_count_facet.get(f, 0) for f in inc_facets]))
                    if inc_facets else 0.0
                )
                side_balance = -(add_side_count + 0.25 * edit_side_count)

                uv = _get_uv(frames, fid, primary_host, pts, uv_cache)
                repair_rank = max(int(cn3_repaired), int(touched))
                cand_rec = {
                    "repair_rank": repair_rank,
                    "score": score,
                    "host_gain": host_gain,
                    "touched": touched,
                    "role_rank": role_rank,
                    "side_balance": side_balance,
                    "fid": fid,
                    "pos": new_pos,
                    "u_vec": u_vec,
                    "placed_via_bulk": placed_via_bulk,
                    "inc_facets_key": tuple(sorted(inc_facets)),
                    "add_side_count": add_side_count,
                    "edit_side_count": edit_side_count,
                    "offset": offset,
                    "dft_val": dft,
                    "primary_host": primary_host,
                    "depth_min": depth_min
                }
                all_valid_candidates.append(cand_rec)

        if not all_valid_candidates:
            continue

        # Dynamic 3D Farthest Point Sampling (FPS) Min-Max Batch Selection
        q_initial = _total_Q(symbols, charges)
        ligand_q_change = abs(charges.get(ligand, 1))
        max_to_add = max(1, int(q_initial // ligand_q_change))

        # 1. Identify pre-existing ligand positions to calculate distance to
        ligand_indices = [idx for idx, s in enumerate(symbols) if s == ligand]
        existing_ligand_pts = pts[ligand_indices] if ligand_indices else np.zeros((0, 3))

        committed_positions = []
        committed_hosts = set()
        batch_to_place = []

        d_exclude = 1.8 if include_sublayer else 3.0

        # Group candidates by (repair_rank, score) to prioritize structural deficits and chemical scores,
        # allowing different roles (vertex, edge, unique) with the same score to be evaluated together on equal footing
        tiers = defaultdict(list)
        for c in all_valid_candidates:
            tiers[(c["repair_rank"], c["score"])].append(c)

        # Process tiers in descending order (highest repair_rank, highest score first)
        sorted_tier_keys = sorted(tiers.keys(), reverse=True)

        for rep_rank, scr in sorted_tier_keys:
            candidates = tiers[(rep_rank, scr)]

            # Seeded shuffle of candidates in this tier to break absolute coordination ties uniformly
            import random
            random.Random(42).shuffle(candidates)

            # Initialize dynamic 3D dmin for each candidate relative to existing and already committed ligands
            for c in candidates:
                pos = c["pos"]
                dmin_val = float("inf")
                if len(existing_ligand_pts) > 0:
                    dmin_val = min(dmin_val, float(np.min(np.linalg.norm(existing_ligand_pts - pos, axis=1))))
                if committed_positions:
                    dmin_val = min(dmin_val, float(np.min(np.linalg.norm(np.asarray(committed_positions) - pos, axis=1))))
                c["dmin_3d"] = dmin_val

            # Greedy Min-Max Farthest Point Selection Loop
            while candidates:
                if len(batch_to_place) >= max_to_add:
                    break

                # Pick candidate: if it is the very first one in the batch, pick a random safe candidate
                # to avoid absolute center clustering bias, otherwise pick the one with maximum dmin_3d
                if len(committed_positions) == 0:
                    best_idx = None
                    for idx, c in enumerate(candidates):
                        if c["dmin_3d"] >= d_exclude:
                            best_idx = idx
                            break
                    if best_idx is None:
                        # No candidate is safe from steric exclusion
                        break
                else:
                    best_idx = max(range(len(candidates)), key=lambda idx: candidates[idx]["dmin_3d"])
                
                best_cand = candidates[best_idx]

                # Steric exclusion limit: stop if the best available site is too close to placed ligands
                if best_cand["dmin_3d"] < d_exclude:
                    break

                # Commit!
                new_pos = best_cand["pos"]
                primary_host = best_cand["primary_host"]

                committed_positions.append(new_pos)
                committed_hosts.add(primary_host)
                
                # Store candidate along with its global batch order rank
                batch_to_place.append((len(batch_to_place) + 1, best_cand))
                candidates.pop(best_idx)

                # Dynamically update dmin_3d for all remaining candidates in this tier
                for c in candidates:
                    dist_to_new = float(np.linalg.norm(c["pos"] - new_pos))
                    c["dmin_3d"] = min(c["dmin_3d"], dist_to_new)

        if not batch_to_place:
            continue

        q_initial = _total_Q(symbols, charges)
        num_added = len(batch_to_place)

        for rank_num, best_cand in batch_to_place:
            primary_host = best_cand["primary_host"]
            new_pos = best_cand["pos"]
            fid = best_cand["fid"]
            inc_facets_key = best_cand["inc_facets_key"]
            role_rank = best_cand["role_rank"]
            dft_val = best_cand["dft_val"]
            score = best_cand["score"]
            offset = best_cand["offset"]
            dmin_3d = best_cand["dmin_3d"]
            depth_min = best_cand["depth_min"]

            host_elem = symbols[primary_host]
            cn_before = int(cn_bi[primary_host])

            symbols.append(ligand)
            pts = np.vstack([pts, new_pos])
            if atom_regions is not None:
                atom_regions.append(_atom_region_index(primary_host, region_masks, atom_regions))

            _record_uv_allfacets(primary_host, pts, frames, surf_tol, uv_taken, edit_count_facet)
            for add_fid in inc_facets_key:
                add_count_facet[add_fid] = add_count_facet.get(add_fid, 0) + 1
            if not inc_facets_key:
                add_count_facet[fid] = add_count_facet.get(fid, 0) + 1
            host_taken[primary_host] = host_taken.get(primary_host, 0) + 1

            if verbose:
                role_name = ROLE_ORDER_VEU[role_rank]
                print(
                    f"  [batch:add] rank #{rank_num}: place {ligand} on {host_elem}#{primary_host} "
                    f"(CN={cn_before}, role={role_name}, facet={fid}, deficit={dft_val}, score={score:.1f}, dmin_3d={dmin_3d:.2f}) at +{offset:.2f} Å"
                )

        q_final = _total_Q(symbols, charges)
        print(f"[batch:add] Added {num_added} '{ligand}' ligand(s) in this pass | Q: {q_initial:+d} → {q_final:+d}")
        return True, symbols, pts

    # If we reach here, we found no acceptable candidate in any tier
    return False, symbols, pts


def _experimental_exhausted_positive_q_fallback_once(
    symbols: List[str],
    pts: NDArray[np.float64],
    frames: List[Plane],
    planes_raw: List[Plane],
    cn_bi: NDArray[np.int_],
    charges: Dict[str, int],
    ligand: str,
    surf_tol: float,
    pair_cuts: Optional[PairCuts],
    *,
    verbose: bool = True,
) -> Tuple[bool, List[str], NDArray[np.float64]]:
    """
    Last-resort experimental Q>0 move. It is intentionally separate from the
    normal priority stack and should only be called after existing remove/add
    candidates are exhausted.
    """
    q_before = _total_Q(symbols, charges)
    positive_charges = sorted({int(v) for v in charges.values() if int(v) > 0})
    max_cation_q = max(positive_charges) if positive_charges else 1
    if q_before > max_cation_q:
        if verbose:
            print(
                "[experimental] skip fallback: residual positive charge is too large "
                f"for a single cation/orphan-ligand cleanup (Q={q_before:+d})."
            )
        return False, symbols, pts

    def simulate_remove(idx: int):
        trial_symbols = list(symbols)
        trial_pts = pts.copy()
        removed = trial_symbols.pop(idx)
        trial_pts = np.delete(trial_pts, idx, axis=0)
        trial_symbols, trial_pts, n_orphan = _prune_orphan_ligands(
            trial_symbols, trial_pts, charges, ligand, pair_cuts, verbose=False
        )
        native_anions = {el for el, val in charges.items() if val < 0 and el != ligand}
        cn_after = coord_numbers_bipartite(trial_symbols, trial_pts, charges, pair_cuts=pair_cuts)
        swapped = 0
        swapped_logs: List[Tuple[int, str, int]] = []
        for j, sj in enumerate(list(trial_symbols)):
            if sj not in native_anions:
                continue
            cnj = int(cn_after[j])
            if cnj >= 3:
                continue
            old = trial_symbols[j]
            trial_symbols[j] = ligand
            swapped += 1
            swapped_logs.append((j, old, cnj))
        return removed, trial_symbols, trial_pts, n_orphan, swapped, swapped_logs

    cands = []
    edges_by_facet = verts_by_facet = None
    edge_tol = max(0.25 * surf_tol, 0.35)
    vertex_tol = max(0.75 * surf_tol, 0.75)
    try:
        edges_by_facet, verts_by_facet = _intersections_geometry(frames)
    except Exception:
        edges_by_facet = verts_by_facet = None
    for i, s in enumerate(symbols):
        if charges.get(s, 0) <= 0:
            continue
        inc = _incident_facets(i, pts, frames, surf_tol)
        if not inc:
            continue
        ci = int(cn_bi[i])
        if ci not in (3, 4):
            continue
        depths = [(fid, frames[fid][1] - float(np.dot(pts[i], frames[fid][0]))) for fid in inc]
        fid_min, depth = min(depths, key=lambda t: t[1])
        inc_count = len(inc)
        role_rank = 2 if inc_count >= 3 else (1 if inc_count == 2 else 0)
        if edges_by_facet is not None and verts_by_facet is not None:
            _role_name, role_rank = _role_by_geometry(
                i, fid_min, pts, frames, edges_by_facet, verts_by_facet, edge_tol, vertex_tol
            )
        removed, trial_symbols, trial_pts, n_orphan, swapped, swapped_logs = simulate_remove(i)
        after = _total_Q(trial_symbols, charges)
        if after < 0:
            continue
        if abs(after) >= abs(q_before):
            continue
        cands.append((
            after,
            abs(after),
            -n_orphan,
            ci,
            -role_rank,
            depth,
            i,
            fid_min,
            tuple(sorted(inc)),
            removed,
            trial_symbols,
            trial_pts,
            n_orphan,
            swapped,
            swapped_logs,
        ))

    if not cands:
        return False, symbols, pts

    before = q_before
    cands.sort(key=lambda t: (t[1], t[0], t[3], t[4], t[5], t[6]))
    (
        after,
        _abs_after,
        _neg_n_orphan,
        cn_removed,
        _neg_role,
        depth,
        idx,
        fid_min,
        inc_key,
        removed,
        trial_symbols,
        trial_pts,
        n_orphan,
        swapped,
        swapped_logs,
    ) = cands[0]
    if verbose:
        print(
            "[experimental] Existing Q>0 balancing exhausted; "
            f"remove surface {removed}#{idx} "
            f"(CN={cn_removed}, role={ROLE_ORDER_VEU[-_neg_role]}, facet={fid_min}, "
            f"inc={list(inc_key)}, depth={depth:.2f} Å), "
            f"then remove {n_orphan} orphan {ligand} ligand(s)."
        )

    for j, old, cnj in swapped_logs:
        if verbose:
            print(f"[experimental] stabilize: swap {old}#{j} (CN={cnj}) → {ligand}")

    if verbose:
        print(
            "[experimental] fallback move complete "
            f"| Q:{before:+d}→{after:+d}, orphan_ligands_removed={n_orphan}, "
            f"native_anions_swapped={swapped}"
        )
    return True, trial_symbols, trial_pts


def _optimize_ligand_distribution_at_neutrality(
    symbols: List[str],
    pts: NDArray[np.float64],
    frames: List[Plane],
    planes: List[Plane],
    cn_bi: NDArray[np.int_],
    charges: Dict[str, int],
    ligand: str,
    surf_tol: float,
    pair_cuts: Optional[PairCuts],
    *,
    verbose: bool = True,
    cif_path: Optional[str] = None,
    atom_regions: Optional[List[int]] = None,
) -> Tuple[List[str], NDArray[np.float64]]:
    """
    Safely migrates existing ligands from fully-coordinated donor cations (CN >= T - 1 after removal)
    to highly undercoordinated acceptor cations (CN <= T - 2 before migration) to optimize overall surface stability.
    """
    true_bulk_cn = derive_true_bulk_cn_from_cif(cif_path, charges, pair_cuts=pair_cuts) if cif_path else None
    bulk_map = bulk_cn_opposite_by_interior(symbols, pts, planes, surf_tol, charges, pair_cuts=pair_cuts, true_bulk_cn=true_bulk_cn)
    
    ref_struct = None
    if cif_path:
        try:
            from pymatgen.core import Structure
            ref_struct = Structure.from_file(cif_path)
        except Exception as e:
            if verbose:
                print(f"[debug:migrate] Reference CIF load failed: {e}. Using outward normals.")

    def calculate_fitness(cn_current: NDArray[np.int_]) -> int:
        fit = 0
        for i, s in enumerate(symbols):
            if charges.get(s, 0) <= 0:
                continue
            T_i = int(bulk_map.get(s, 4))
            c = int(cn_current[i])
            if c >= T_i:
                fit += 0
            elif c == T_i - 1:
                fit += -1
            else:
                fit += -10
        return fit

    migration_count = 0

    while True:
        cn_current = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
        pt_tree = cKDTree(np.asarray(pts, float))
        min_spacing = _min_ligand_ligand_spacing(symbols, pts, ligand)
        
        acceptors = []
        for i, s in enumerate(symbols):
            if charges.get(s, 0) <= 0:
                continue
            T_i = int(bulk_map.get(s, 4))
            if int(cn_current[i]) <= T_i - 2:
                acceptors.append(i)
        
        if not acceptors:
            break

        ligand_indices = [idx for idx, s in enumerate(symbols) if s == ligand]
        if not ligand_indices:
            break

        best_migration = None
        best_delta_fit = 0

        for l_idx in ligand_indices:
            hosts = []
            for i, s in enumerate(symbols):
                if charges.get(s, 0) <= 0:
                    continue
                rc = _pair_cut_calibrated(s, ligand, pair_cuts)
                if float(np.linalg.norm(pts[i] - pts[l_idx])) <= rc + 1e-12:
                    hosts.append(i)
            
            if not hosts:
                continue

            safe_donor = True
            for h in hosts:
                T_h = int(bulk_map.get(symbols[h], 4))
                if int(cn_current[h]) - 1 < T_h - 1:
                    safe_donor = False
                    break
            
            if not safe_donor:
                continue

            cn_temp = cn_current.copy()
            for h in hosts:
                cn_temp[h] -= 1

            candidate_positions = []
            for i in acceptors:
                try:
                    rc = pc(symbols[i], ligand)
                    offset = (0.95 * rc) if rc > 0 else 2.5
                except Exception:
                    offset = 2.5

                inc = _incident_facets(i, pts, frames, surf_tol)
                
                n_unit = []
                for (n_f, d_f) in planes:
                    ln_f = np.linalg.norm(n_f) + 1e-12
                    n_unit.append(np.asarray(n_f, float) / ln_f)
                
                vec = np.zeros(3, float)
                for fid in inc:
                    vec += n_unit[fid]
                ln = np.linalg.norm(vec)
                n_out = n_unit[inc[0]] if ln < 1e-8 else (vec / ln)

                placed_via_bulk = False
                direction_pool = [n_out]
                if ref_struct is not None:
                    try:
                        f_coord = ref_struct.lattice.get_fractional_coords(pts[i])
                        site_idx = int(np.argmin([
                            float(np.linalg.norm((site.frac_coords - f_coord + 0.5) % 1 - 0.5))
                            for site in ref_struct.sites
                        ]))
                        bulk_site = ref_struct.sites[site_idx]
                        all_neigh = ref_struct.get_neighbors(bulk_site, r=4.5)
                        if all_neigh:
                            min_dist = min(neigh.nn_distance for neigh in all_neigh)
                            bulk_neighbors = [neigh for neigh in all_neigh if neigh.nn_distance <= 1.15 * min_dist]
                            ideal_vectors = []
                            for neigh in bulk_neighbors:
                                vec_ideal = neigh.coords - bulk_site.coords
                                ln_ideal = float(np.linalg.norm(vec_ideal))
                                if ln_ideal > 1e-8:
                                    ideal_vectors.append(vec_ideal / ln_ideal)
                            
                            actual_vectors = []
                            neighbor_ids = pt_tree.query_ball_point(pts[i], r=4.5)
                            for act_idx in neighbor_ids:
                                if act_idx == i or act_idx == l_idx:
                                    continue
                                rc_neigh = pc(symbols[i], symbols[act_idx])
                                if rc_neigh <= 0:
                                    continue
                                if float(np.linalg.norm(pts[act_idx] - pts[i])) <= rc_neigh + 1e-12:
                                    vec_act = pts[act_idx] - pts[i]
                                    ln_act = float(np.linalg.norm(vec_act))
                                    if ln_act > 1e-8:
                                        actual_vectors.append(vec_act / ln_act)
                            
                            vacant_vectors = []
                            for t_k in ideal_vectors:
                                is_occupied = False
                                for v_a in actual_vectors:
                                    if float(np.dot(v_a, t_k)) > 0.90:
                                        is_occupied = True
                                        break
                                if not is_occupied:
                                    vacant_vectors.append(t_k)
                            
                            if vacant_vectors:
                                direction_pool = vacant_vectors
                                placed_via_bulk = True
                    except Exception:
                        pass
                
                for vec in direction_pool:
                    ln_v = float(np.linalg.norm(vec))
                    if ln_v > 1e-8:
                        candidate_positions.append({
                            "pos": pts[i] + (vec / ln_v) * offset,
                            "placed_via_bulk": placed_via_bulk,
                            "hosts": [i]
                        })

            for idx_a1, i1 in enumerate(acceptors):
                for i2 in acceptors[idx_a1 + 1:]:
                    if float(np.linalg.norm(pts[i1] - pts[i2])) > 4.5:
                        continue
                    
                    rc1 = pc(symbols[i1], ligand)
                    rc2 = pc(symbols[i2], ligand)
                    
                    cands1 = [c for c in candidate_positions if c["hosts"] == [i1]]
                    cands2 = [c for c in candidate_positions if c["hosts"] == [i2]]
                    for c1 in cands1:
                        for c2 in cands2:
                            if float(np.linalg.norm(c1["pos"] - c2["pos"])) < 1.8:
                                p_bridge = (c1["pos"] + c2["pos"]) / 2.0
                                if float(np.linalg.norm(p_bridge - pts[i1])) <= rc1 + 0.15 and \
                                   float(np.linalg.norm(p_bridge - pts[i2])) <= rc2 + 0.15:
                                    candidate_positions.append({
                                        "pos": p_bridge,
                                        "placed_via_bulk": c1["placed_via_bulk"] or c2["placed_via_bulk"],
                                        "hosts": [i1, i2]
                                    })

            for cand in candidate_positions:
                new_pos = cand["pos"]
                
                too_close = False
                for other_l in ligand_indices:
                    if other_l == l_idx:
                        continue
                    if float(np.linalg.norm(new_pos - pts[other_l])) < min_spacing - 1e-6:
                        too_close = True
                        break
                
                if too_close:
                    continue

                cn_after = cn_temp.copy()
                local_cut = 5.0
                neighbor_ids = pt_tree.query_ball_point(np.asarray(new_pos, float), r=local_cut)
                for j in neighbor_ids:
                    sj = symbols[j]
                    if charges.get(sj, 0) <= 0:
                        continue
                    dist = float(np.linalg.norm(pts[j] - new_pos))
                    rc = _pair_cut_calibrated(sj, ligand, pair_cuts)
                    if dist <= rc:
                        if ref_struct is not None:
                            vacant_dirs = _get_vacant_directions_for_site(j, symbols, pts, ref_struct, pt_tree)
                            u_j = (new_pos - pts[j]) / (dist + 1e-12)
                            if not any(float(np.dot(u_j, t_k)) > 0.80 for t_k in vacant_dirs):
                                continue
                        cn_after[j] += 1

                delta_fit = calculate_fitness(cn_after) - calculate_fitness(cn_current)
                if delta_fit > best_delta_fit:
                    best_delta_fit = delta_fit
                    best_migration = (delta_fit, l_idx, new_pos, cand["hosts"], hosts)

        if best_migration is not None:
            delta_fit, l_idx, new_pos, host_list, donor_hosts = best_migration
            if verbose:
                donors_str = ", ".join(f"{symbols[d]}#{d} (CN={int(cn_current[d])})" for d in donor_hosts)
                acceptors_str = ", ".join(f"{symbols[a]}#{a} (CN={int(cn_current[a])})" for a in host_list)
                print(f"[passivation:migrate] Migrated ligand {ligand}#{l_idx} binding {donors_str} → {acceptors_str} (Delta Fitness: +{delta_fit})")
            pts[l_idx] = new_pos
            migration_count += 1
        else:
            break

    if verbose and migration_count > 0:
        print(f"[passivation:migrate] Completed {migration_count} ligand migrations successfully.")
    
    return symbols, pts


# ---------- MASTER CONTROLLER ----------
def charge_balance_iterative(
    symbols: List[str],
    pts: NDArray[np.float64],
    charges: Dict[str, int],
    ligand: str,
    verbose: bool,
    planes: List[Plane],
    surf_tol: float,
    cif_path: str,
    *,
    positive_q_strategy: str = "remove",
    write_all: bool = False,
    prefix: str = "nc",
    include_sublayer: bool = False,
    experimental_exhausted_positive_q_fallback: bool = False,
    pair_cuts_override: Optional[PairCuts] = None,
    positive_q_strategy_by_z: Optional[Tuple[float, str, str]] = None,
    region_masks: Optional[List[NDArray[np.bool_]]] = None,
    stack_passivation: bool = False,
    prepass_mode: str = "standard",
    prepass_min_cn_terrace: int = 2,
    prepass_min_cn_edge: int = 2,
    prepass_min_cn_vertex: int = 2,
) -> Tuple[List[str], NDArray[np.float64]]:
    # Calibrate pair cuts from CIF once (robust bipartite ruler everywhere)
    pair_cuts = (
        pair_cuts_override
        if pair_cuts_override is not None
        else derive_pair_cuts_from_cif(cif_path, charges, safety=1.00)
    )
    if verbose:
        # print a few key pairs if possible
        elems = sorted(set(symbols))
        hints = []
        for a in elems:
            for b in elems:
                if charges.get(a, 0) * charges.get(b, 0) < 0:
                    hints.append((a, b))
                    if len(hints) >= 4:
                        break
            if len(hints) >= 4:
                break
        pretty_print_pair_cuts(pair_cuts, pairs_hint=hints)
    # SAVE 1: Before Pre-Pass
    if write_all:
        write_xyz(f"{prefix}_01_before_prepass.xyz", symbols, center_coords(pts))

    region_removals: Dict[int, int] = defaultdict(int)
    atom_regions: Optional[List[int]] = None
    if stack_passivation and region_masks is not None:
        atom_regions = atom_regions_from_masks(len(symbols), region_masks)

    # Prepass cleanup (prints its own actions)
    symbols, pts, uv_taken, edit_count_facet = prepass_surface_cleanup(
        symbols, pts, planes, charges, ligand, surf_tol, pair_cuts=pair_cuts, verbose=verbose,
        stack_passivation=stack_passivation,
        region_masks=region_masks if stack_passivation else None,
        region_removals=region_removals if stack_passivation else None,
        atom_regions=atom_regions,
        prepass_mode=prepass_mode,
        prepass_min_cn_terrace=prepass_min_cn_terrace,
        prepass_min_cn_edge=prepass_min_cn_edge,
        prepass_min_cn_vertex=prepass_min_cn_vertex,
    )
    symbols, pts, _ = _prune_orphan_ligands(
        symbols, pts, charges, ligand, pair_cuts, verbose=verbose, atom_regions=atom_regions
    )
    # Update planes to reflect the new shrunk boundary of the nanocrystal after prepass
    # In rebalancing mode (prepass_mode == "none"), exclude passivating ligand atoms
    # from determining the boundary because they lie outside the scaffold.
    if prepass_mode == "none":
        scaffold_mask = np.array(symbols) != ligand
        scaffold_pts = pts[scaffold_mask] if np.any(scaffold_mask) else pts
    else:
        scaffold_pts = pts

    new_planes = []
    for n, d in planes:
        d_new = float(np.max(scaffold_pts @ n) + 1e-5)
        new_planes.append((n, d_new))
    planes = new_planes

    if write_all:
        write_xyz(f"{prefix}_02_after_prepass.xyz", symbols, center_coords(pts))

    add_count_facet: Dict[int, int] = defaultdict(int)
    host_taken: Dict[int, int] = {}
    positive_q_strategy_selector = None
    if positive_q_strategy_by_z is not None:
        z_cut, core_strategy, shell_strategy = positive_q_strategy_by_z
        core_strategy = str(core_strategy).strip().lower()
        shell_strategy = str(shell_strategy).strip().lower()
        valid = {"remove", "add", "skip", "none"}
        if core_strategy not in valid or shell_strategy not in valid:
            raise ValueError("positive_q_strategy_by_z strategies must be remove, add, skip, or none")

        def _selector(_idx: int, _sym: str, xyz: NDArray[np.float64]) -> Optional[str]:
            mode = core_strategy if float(xyz[2]) <= float(z_cut) else shell_strategy
            return None if mode in {"skip", "none"} else mode

        positive_q_strategy_selector = _selector

    has_saved_stabilized = False

    def _finish(
        symbols: List[str],
        pts: NDArray[np.float64],
    ) -> Tuple[List[str], NDArray[np.float64]]:
        symbols, pts, _ = _prune_orphan_ligands(
            symbols, pts, charges, ligand, pair_cuts, verbose=verbose, atom_regions=atom_regions
        )
        return symbols, pts

    while True:
        # Shared per-iteration state (with robust bipartite CN)
        frames = _build_facet_frames(planes)
        mem = _facet_memberships(pts, planes, surf_tol)
        cn_bi = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
        pt_tree = cKDTree(np.asarray(pts, float))
        uv_cache: UVCache = {}

        Q = _total_Q(symbols, charges)
        if verbose:
            print(f"\n[loop] Q={Q:+d} — reassessing priorities...")

        # Priority 1
        progressed, symbols, pts = _priority1_swap_undercoord_anions_once(
            symbols, pts, frames, mem, cn_bi, charges, ligand, surf_tol, uv_cache, verbose=verbose,
            pt_tree=pt_tree
        )
        if progressed:
            continue

        # Priority 2
        progressed, symbols, pts = _priority2_remove_low_cn_cation_once(
            symbols, pts, frames, mem, cn_bi, charges, surf_tol, uv_taken, edit_count_facet, uv_cache, ligand, pair_cuts, verbose=verbose,
            stack_passivation=stack_passivation,
            region_masks=region_masks if stack_passivation else None,
            region_removals=region_removals if stack_passivation else None,
            atom_regions=atom_regions,
            prepass_mode=prepass_mode,
            prepass_min_cn_terrace=prepass_min_cn_terrace,
            prepass_min_cn_edge=prepass_min_cn_edge,
            prepass_min_cn_vertex=prepass_min_cn_vertex,
        )
        if progressed:
            continue

        # ---> SAVE 3: Structurally Stabilized, Before Electrical Balancing <---
        if write_all and not has_saved_stabilized:
            write_xyz(f"{prefix}_03_stabilized_pre_Q.xyz", symbols, center_coords(pts))
            has_saved_stabilized = True

        # Electrical
        Q = _total_Q(symbols, charges)
        if Q == 0:
            symbols, pts, n_orphan = _prune_orphan_ligands(
                symbols, pts, charges, ligand, pair_cuts, verbose=verbose, atom_regions=atom_regions
            )
            if n_orphan:
                continue

            # Post-passivation ligand migration/redistribution to resolve highly undercoordinated sites
            symbols, pts = _optimize_ligand_distribution_at_neutrality(
                symbols, pts, frames, planes, cn_bi, charges, ligand, surf_tol, pair_cuts,
                verbose=verbose, cif_path=cif_path, atom_regions=atom_regions
            )

            if stack_passivation:
                cn_check = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
                if _surface_low_cn_cations_remain(
                    symbols, pts, frames, cn_check, charges, surf_tol,
                    prepass_mode=prepass_mode,
                    prepass_min_cn_terrace=prepass_min_cn_terrace,
                    prepass_min_cn_edge=prepass_min_cn_edge,
                    prepass_min_cn_vertex=prepass_min_cn_vertex,
                ):
                    if verbose:
                        print("[loop] Q=0 but surface low CN cations remain — continuing structural cleanup...")
                    continue
            if verbose:
                print("[done] Structural + electrical stability reached (Q=0, no CN≤2 on surface).")
            return _finish(symbols, pts)

        if Q < 0:
            if verbose:
                print("[strategy] Q<0 → swap stable anions (CN=3→4→5), V→E→U + FPS")
            progressed, symbols, pts = _priority3_balance_negative_q(
                symbols, pts, frames, planes, mem, cn_bi, charges, ligand, surf_tol, uv_taken, edit_count_facet, uv_cache,
                pair_cuts, verbose=verbose, stack_passivation=stack_passivation,
            )
            if progressed:
                continue
            # Fallback additions are only valid if the ligand charge moves Q
            # toward zero. For the common anion-ligand case, adding ligand
            # under Q<0 would make the structure more negative.
            if charges.get(ligand, 0) > 0:
                progressed, symbols, pts = _priority3_balance_positive_q_add(
                    symbols, pts, frames, planes, mem, cn_bi, charges, ligand, surf_tol,
                    uv_taken, edit_count_facet, add_count_facet, host_taken, uv_cache, pair_cuts, verbose=verbose,
                    include_sublayer=include_sublayer,
                    stack_passivation=stack_passivation,
                    atom_regions=atom_regions,
                    region_masks=region_masks if stack_passivation else None,
                    cif_path=cif_path,
                )
                if progressed:
                    continue
            if verbose:
                print("[halt] Q<0 but no valid swaps/additions remain. Consider revising facet energies or inputs.")
            return _finish(symbols, pts)

        # Q > 0
        if positive_q_strategy_selector is not None:
            if verbose:
                print("[strategy] Q>0 → region-aware positive_q_strategy (core/shell by interface z)")
            progressed, symbols, pts = _priority3_balance_positive_q_remove(
                symbols, pts, frames, planes, mem, cn_bi, charges, surf_tol,
                uv_taken, edit_count_facet, uv_cache, ligand, pair_cuts,
                positive_q_strategy_selector=positive_q_strategy_selector,
                verbose=verbose,
                cif_path=cif_path,
            )
            if progressed:
                continue
            progressed, symbols, pts = _priority3_balance_positive_q_add(
                symbols, pts, frames, planes, mem, cn_bi, charges, ligand, surf_tol,
                uv_taken, edit_count_facet, add_count_facet, host_taken, uv_cache, pair_cuts,
                positive_q_strategy_selector=positive_q_strategy_selector,
                verbose=verbose,
                include_sublayer=include_sublayer,
                stack_passivation=stack_passivation,
                atom_regions=atom_regions,
                region_masks=region_masks if stack_passivation else None,
                cif_path=cif_path,
            )
            if progressed:
                continue
            if experimental_exhausted_positive_q_fallback:
                progressed, symbols, pts = _experimental_exhausted_positive_q_fallback_once(
                    symbols,
                    pts,
                    frames,
                    planes,
                    cn_bi,
                    charges,
                    ligand,
                    surf_tol,
                    pair_cuts,
                    verbose=verbose,
                )
                if progressed:
                    continue
            if verbose:
                print("[halt] Q>0 and no region-allowed remove/add candidates remain.")
            return _finish(symbols, pts)

        if verbose:
            print(f"[strategy] Q>0 → positive_q_strategy='{positive_q_strategy}' (remove cations or add anions)")
        if positive_q_strategy == "remove":
            # --- transactional remove with rollback if we overshoot below zero ---
            _sym0, _pts0 = list(symbols), pts.copy()
            _Q_before = _total_Q(symbols, charges)
            progressed, symbols, pts = _priority3_balance_positive_q_remove(
                symbols, pts, frames, planes, mem, cn_bi, charges, surf_tol,
                uv_taken, edit_count_facet, uv_cache, ligand, pair_cuts, verbose=verbose,
                cif_path=cif_path,
            )
            if progressed:
                _Q_after = _total_Q(symbols, charges)
                if _Q_after < 0:
                    # Revert the last cation removal and finish by adding ligands to reach Q=0
                    if verbose:
                        print("[rollback] Removal would flip Q positive→negative. Reverting and adding ligands to hit Q=0…")
                    symbols, pts = _sym0, _pts0
                    q_lig = abs(int(charges.get(ligand, -1))) or 1
                    need = (_Q_before + q_lig - 1) // q_lig
                    _added_any = False
                    for _ in range(int(need)):
                        ok, symbols, pts = _priority3_balance_positive_q_add(
                            symbols, pts, frames, planes, mem, cn_bi, charges, ligand, surf_tol,
                            uv_taken, edit_count_facet, add_count_facet, host_taken, uv_cache, pair_cuts, verbose=verbose,
                            include_sublayer=include_sublayer,
                            atom_regions=atom_regions,
                            region_masks=region_masks,
                            cif_path=cif_path,
                        )
                        if not ok:
                            break
                        _added_any = True
                        if _total_Q(symbols, charges) == 0:
                            break
                    if _total_Q(symbols, charges) == 0:
                        # Achieved neutrality; continue outer loop
                        continue
                    # If we couldn't add enough ligands, try one final add move below.
                    if _added_any:
                        continue
                else:
                    continue
            # Final add-ligand fallback.
            progressed, symbols, pts = _priority3_balance_positive_q_add(
                symbols, pts, frames, planes, mem, cn_bi, charges, ligand, surf_tol,
                uv_taken, edit_count_facet, add_count_facet, host_taken, uv_cache, pair_cuts, verbose=verbose,
                include_sublayer=include_sublayer,
                stack_passivation=stack_passivation,
                atom_regions=atom_regions,
                region_masks=region_masks if stack_passivation else None,
                cif_path=cif_path,
            )
            if progressed:
                continue
            if experimental_exhausted_positive_q_fallback:
                progressed, symbols, pts = _experimental_exhausted_positive_q_fallback_once(
                    symbols,
                    pts,
                    frames,
                    planes,
                    cn_bi,
                    charges,
                    ligand,
                    surf_tol,
                    pair_cuts,
                    verbose=verbose,
                )
                if progressed:
                    continue
            if verbose:
                print("[halt] Q>0 but no removable cations/additions available. Consider revising facet energies.")
            return _finish(symbols, pts)
        else:
            # positive_q_strategy == "add"
            progressed, symbols, pts = _priority3_balance_positive_q_add(
                symbols, pts, frames, planes, mem, cn_bi, charges, ligand, surf_tol,
                uv_taken, edit_count_facet, add_count_facet, host_taken, uv_cache, pair_cuts, verbose=verbose,
                include_sublayer=include_sublayer,
                stack_passivation=stack_passivation,
                atom_regions=atom_regions,
                region_masks=region_masks if stack_passivation else None,
                cif_path=cif_path,
            )
            if progressed:
                continue
            if experimental_exhausted_positive_q_fallback:
                progressed, symbols, pts = _experimental_exhausted_positive_q_fallback_once(
                    symbols,
                    pts,
                    frames,
                    planes,
                    cn_bi,
                    charges,
                    ligand,
                    surf_tol,
                    pair_cuts,
                    verbose=verbose,
                )
                if progressed:
                    continue
            if verbose:
                print("[halt] Q>0 and cannot add more ligands — likely surface fully saturated.")
            return _finish(symbols, pts)
