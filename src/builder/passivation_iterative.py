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
    _atom_region_index,
    _try_q_neutral_cation_removal_bundle,
    atom_regions_from_masks,
)
from .io_utils import write_xyz, center_coords

# Role ranks: 0=unique, 1=edge, 2=vertex
ROLE_ORDER_VEU = ["unique", "edge", "vertex"]
UVCache = Dict[Tuple[int, int], Tuple[float, float]]  # (facet_id, atom_idx) -> (u,v)

def _total_Q(symbols: List[str], charges: Dict[str, int]) -> int:
    return int(sum(int(charges.get(s, 0)) for s in symbols))


def _surface_low_cn_cations_remain(
    symbols: List[str],
    pts: NDArray[np.float64],
    frames: List[Plane],
    cn_bi: NDArray[np.int_],
    charges: Dict[str, int],
    surf_tol: float,
) -> bool:
    outer_thr = 0.35 * surf_tol
    for i, s in enumerate(symbols):
        if charges.get(s, 0) <= 0:
            continue
        if int(cn_bi[i]) > 2:
            continue
        inc = _incident_facets(i, pts, frames, surf_tol)
        if not inc:
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
    dmin = float("inf")
    for a_pos, i in enumerate(idx):
        for j in idx[a_pos + 1:]:
            d = float(np.linalg.norm(pts[i] - pts[j]))
            if 1e-9 < d < dmin:
                dmin = d
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


def _neighbors(i: int, symbols: List[str], pts: NDArray[np.float64]) -> List[int]:
    xi = pts[i]
    out: List[int] = []
    for j, sj in enumerate(symbols):
        if j == i:
            continue
        rc = pc(symbols[i], sj)
        if rc <= 0:
            continue
        if np.linalg.norm(pts[j] - xi) <= rc + 1e-12:
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
    *, verbose: bool = True
) -> Tuple[bool, List[str], NDArray[np.float64]]:
    for i, s in enumerate(list(symbols)):
        if charges.get(s, 0) >= 0:
            continue
        if s == ligand:
            continue
        if int(cn_bi[i]) > 2:
            continue
        if len(_incident_facets(i, pts, frames, surf_tol)) == 0:
            continue
        # avoid creating new CN<3 cations
        harmful = False
        for j in _neighbors(i, symbols, pts):
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
) -> Tuple[bool, List[str], NDArray[np.float64]]:
    use_stack = stack_passivation
    cands: List[Tuple] = []
    for i, s in enumerate(symbols):
        if charges.get(s, 0) <= 0:
            continue
        if int(cn_bi[i]) > 2:
            continue
        inc = _incident_facets(i, pts, frames, surf_tol)
        if not inc:
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
) -> Tuple[bool, List[str], NDArray[np.float64]]:
    outer_thr = 0.35 * surf_tol
    bulk_map = bulk_cn_opposite_by_interior(symbols, pts, planes_raw, surf_tol, charges, pair_cuts=pair_cuts)

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
        deficit = max(0, tgt - ci)
        if deficit <= 0:
            continue  # bulk-like; don't remove
        if ci < 3:
            continue  # handled by Priority 2
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

    for cn_tier in sorted(cands_by_cn.keys()):  # 3 -> 4 -> 5 -> ...
        if cn_tier < 3:
            continue
        lst = cands_by_cn[cn_tier]
        lst.sort(key=lambda t: (-t[0], t[1], t[2], t[4]))
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
) -> Tuple[bool, List[str], NDArray[np.float64]]:
    # gather candidates; include_sublayer=True expands search to sublayer atoms
    sites = _collect_cation_sites(
        symbols, pts, planes_raw, charges, surf_tol, pair_cuts=pair_cuts,
        outer_only=not include_sublayer, allow_shared=True,
        include_sublayer=include_sublayer,
        cn_bi=cn_bi,
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
    ordered_defs = ([1] if 1 in by_def and by_def[1] else []) + sorted([k for k in by_def.keys() if k >= 2])

    # Try deficits in order; within each, try roles UNIQUE(0)→EDGE(1)→VERTEX(2)
    for def_tier in ordered_defs:
        pool = by_def.get(def_tier, [])
        if not pool:
            continue
        if verbose:
            print(f"[summary:add:pick] chosen_deficit={def_tier} | pool={len(pool)}")

        for role in (0, 1, 2):
            sub = [rec for rec in pool if rec[4] == role and host_taken.get(rec[0], 0) < 1]
            if not sub:
                continue

            # Round-robin across facets: fewest additions so far; within facet, FPS + shallow
            from math import hypot
            min_count = min((add_count_facet.get(fid, 0) for (_, _, _, _, _, fid) in sub), default=0)
            candidate_best = None
            for fid in sorted(set(rec[5] for rec in sub)):
                if add_count_facet.get(fid, 0) != min_count:
                    continue
                facet_pool = [rec for rec in sub if rec[5] == fid]
                best = None
                for (i, n_out, depth_min, dft, role_rank, fid2) in facet_pool:
                    uv = _get_uv(frames, fid, i, pts, uv_cache)
                    taken = uv_taken.get(fid, [])
                    dmin = min(hypot(uv[0]-u, uv[1]-v) for (u, v) in taken) if taken else float("inf")
                    key = (dmin, -depth_min, -i)
                    cand = (key, (i, n_out, depth_min, dft, role_rank, fid))
                    if best is None or cand[0] > best[0]:
                        best = cand
                if best is not None and (candidate_best is None or best[0] > candidate_best[0]):
                    candidate_best = best

            if candidate_best is None:
                continue  # try next role or deficit

            _, (i, n_out, depth_min, dft, role_rank, fid) = candidate_best

            # Validate deficit from the current loop's CN and candidate record.
            cn_before = int(cn_bi[i])
            deficit_chk = int(dft)
            if deficit_chk <= 0:
                if verbose:
                    print(f"[debug:add] veto: deficit_chk={deficit_chk} (cn_before={cn_before}) for host {symbols[i]}#{i}")
                continue  # try next candidate

            # place ligand along outward vector
            n_vec = n_out
            ln = float(np.linalg.norm(n_vec))
            if ln < 1e-8:
                n_vec = frames[fid][0]
                ln = float(np.linalg.norm(n_vec))
            if ln > 1e-8:
                n_vec = n_vec / ln
            try:
                rc = pc(symbols[i], ligand)
                offset = (0.95 * rc) if rc > 0 else 2.5
            except Exception:
                offset = 2.5
            new_pos = pts[i] + n_vec * offset
            if stack_passivation and not _ligand_position_allowed(new_pos, symbols, pts, ligand):
                if verbose:
                    print(f"[debug:add] veto: ligand spacing for host {symbols[i]}#{i}")
                continue

            host_elem = symbols[i]
            before = _total_Q(symbols, charges)
            symbols.append(ligand)
            pts = np.vstack([pts, new_pos])
            if atom_regions is not None:
                atom_regions.append(_atom_region_index(i, region_masks, atom_regions))
            after = _total_Q(symbols, charges)

            _record_uv_allfacets(i, pts, frames, surf_tol, uv_taken, edit_count_facet)
            add_count_facet[fid] = add_count_facet.get(fid, 0) + 1
            host_taken[i] = host_taken.get(i, 0) + 1

            cn_after_arr = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
            cn_after = int(cn_after_arr[i])
            role_name = ROLE_ORDER_VEU[role_rank]
            shell_label = "sublayer" if depth_min >= 0.35 * surf_tol else "outer"
            if verbose:
                print(
                    f"add ligand {ligand} to {host_elem}#{i} "
                    f"(CN_before={cn_before}, CN_after={cn_after}, role={role_name}, facet={fid}, shell={shell_label}, deficit={deficit_chk}) "
                    f"at +{offset:.2f} Å | Q:{before:+d}→{after:+d}"
                )
            return True, symbols, pts  # single successful action

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
        _fid, depth = min(depths, key=lambda t: t[1])
        inc_count = len(inc)
        role_rank = 2 if inc_count >= 3 else (1 if inc_count == 2 else 0)
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
            f"remove surface {removed}#{idx} (CN={cn_removed}, depth={depth:.2f} Å), "
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
    )
    symbols, pts, _ = _prune_orphan_ligands(
        symbols, pts, charges, ligand, pair_cuts, verbose=verbose, atom_regions=atom_regions
    )
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
        uv_cache: UVCache = {}

        Q = _total_Q(symbols, charges)
        if verbose:
            print(f"\n[loop] Q={Q:+d} — reassessing priorities...")

        # Priority 1
        progressed, symbols, pts = _priority1_swap_undercoord_anions_once(
            symbols, pts, frames, mem, cn_bi, charges, ligand, surf_tol, uv_cache, verbose=verbose
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
            if stack_passivation:
                cn_check = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
                if _surface_low_cn_cations_remain(symbols, pts, frames, cn_check, charges, surf_tol):
                    if verbose:
                        print("[loop] Q=0 but surface CN≤2 cations remain — continuing structural cleanup...")
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
                uv_taken, edit_count_facet, uv_cache, ligand, pair_cuts, verbose=verbose
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
