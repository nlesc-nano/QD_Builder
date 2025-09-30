from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from .nc_types import Plane, Facet
from .analysis import (
    PairCuts,
    derive_pair_cuts_from_cif,
    pretty_print_pair_cuts,
    coord_numbers_bipartite,
    bulk_cn_opposite_by_interior,
)
from .analysis import _pair_cut as pc
from .passivation import (
    prepass_surface_cleanup,
    collect_anion_candidates,
    _build_facet_frames,
    _plane_uv,
    _incident_facets,
    _record_uv_allfacets,
    _facet_memberships,
    _collect_cation_sites,
)

# Role ranks: 0=unique, 1=edge, 2=vertex
ROLE_ORDER_VEU = ["unique", "edge", "vertex"]
UVCache = Dict[Tuple[int, int], Tuple[float, float]]  # (facet_id, atom_idx) -> (u,v)

# Global calibrated pair-cuts (set once in controller)
_PAIR_CUTS_GLOBAL: Optional[PairCuts] = None


def _total_Q(symbols: List[str], charges: Dict[str, int]) -> int:
    return int(sum(int(charges.get(s, 0)) for s in symbols))


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
    *, verbose: bool = True
) -> Tuple[bool, List[str], NDArray[np.float64]]:
    cands: List[Tuple[int, float, float, int, int]] = []  # (role_rank, -dmin, depth, fid, idx)
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
        cands.append((role_rank, -dmin, depth, fid, i))

    if not cands:
        return False, symbols, pts

    cands.sort(key=lambda t: (-t[0], t[1], t[2], t[4]))
    role_rank, _negdmin, depth, fid, i = cands[0]
    before = _total_Q(symbols, charges)
    if verbose:
        print(f"remove {symbols[i]}#{i} (CN={int(cn_bi[i])}, role={ROLE_ORDER_VEU[role_rank]}, facet={fid}, depth={depth:.2f} Å)")
    _record_uv_allfacets(i, pts, frames, surf_tol, uv_taken, edit_count_facet)
    removed_elem = symbols.pop(i)
    pts = np.delete(pts, i, axis=0)
    after = _total_Q(symbols, charges)
    if verbose:
        print(f"   ↳ Q:{before:+d}→{after:+d} (removed {removed_elem})")

    # stabilization: convert native anions (not ligand) that dropped to CN<3 → ligand
    cn2 = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=_PAIR_CUTS_GLOBAL)
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
) -> List[dict]:
    outer, subl = collect_anion_candidates(
        symbols, pts, planes, charges, ligand, surf_tol, verbose=False, pair_cuts=_PAIR_CUTS_GLOBAL
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
    *, verbose: bool = True,
) -> Tuple[bool, List[str], NDArray[np.float64]]:
    cand = _collect_anion_candidates_flat(
        symbols, pts, planes_raw, charges, ligand, surf_tol
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
    *, verbose: bool = True
) -> Tuple[bool, List[str], NDArray[np.float64]]:
    outer_thr = 0.35 * surf_tol
    bulk_map = bulk_cn_opposite_by_interior(symbols, pts, planes_raw, surf_tol, charges, pair_cuts=_PAIR_CUTS_GLOBAL)

    cands_by_cn: Dict[int, List[Tuple[int, float, float, int, int, int]]] = {}
    for i, s in enumerate(symbols):
        if charges.get(s, 0) <= 0:
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
        after = _total_Q(symbols, charges)
        if verbose:
            print(f"   ↳ Q:{before:+d}→{after:+d} (removed {removed_elem})")

        # Post-stabilization: native anions (not ligand) that dropped to CN<3 → ligand
        cn2 = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=_PAIR_CUTS_GLOBAL)
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
    *, verbose: bool = True
) -> Tuple[bool, List[str], NDArray[np.float64]]:
    # gather candidates (outer-only)
    sites = _collect_cation_sites(
        symbols, pts, planes_raw, charges, surf_tol, pair_cuts=_PAIR_CUTS_GLOBAL,
        outer_only=True, allow_shared=True, include_sublayer=False
    )
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

            # Validate deficit using robust bipartite bulk target
            bulk_map = bulk_cn_opposite_by_interior(symbols, pts, planes_raw, surf_tol, charges, pair_cuts=_PAIR_CUTS_GLOBAL)
            cn_before_arr = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=_PAIR_CUTS_GLOBAL)
            cn_before = int(cn_before_arr[i])
            tgt = int(bulk_map.get(symbols[i], cn_before))
            deficit_chk = max(0, tgt - cn_before)
            if deficit_chk <= 0:
                if verbose:
                    print(f"[debug:add] veto: deficit_chk={deficit_chk} (cn_before={cn_before}, tgt={tgt}) for host {symbols[i]}#{i}")
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

            host_elem = symbols[i]
            before = _total_Q(symbols, charges)
            symbols.append(ligand)
            pts = np.vstack([pts, new_pos])
            after = _total_Q(symbols, charges)

            _record_uv_allfacets(i, pts, frames, surf_tol, uv_taken, edit_count_facet)
            add_count_facet[fid] = add_count_facet.get(fid, 0) + 1
            host_taken[i] = host_taken.get(i, 0) + 1

            cn_after_arr = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=_PAIR_CUTS_GLOBAL)
            cn_after = int(cn_after_arr[i])
            role_name = ROLE_ORDER_VEU[role_rank]
            if verbose:
                print(
                    f"add ligand {ligand} to {host_elem}#{i} "
                    f"(CN_before={cn_before}, CN_after={cn_after}, role={role_name}, facet={fid}, shell=outer, deficit={deficit_chk}) "
                    f"at +{offset:.2f} Å | Q:{before:+d}→{after:+d}"
                )
            return True, symbols, pts  # single successful action

    # If we reach here, we found no acceptable candidate in any tier
    return False, symbols, pts


# ---------- MASTER CONTROLLER ----------
def charge_balance_iterative(
    symbols: List[str],
    pts: NDArray[np.float64],
    outer_candidates: List[dict],
    sublayer_candidates: List[dict],
    charges: Dict[str, int],
    ligand: str,
    verbose: bool,
    planes: List[Plane],
    facets: List[Facet],
    surf_tol: float,
    rng: random.Random,
    cif_path: str,
    *,
    positive_q_strategy: str = "remove",
    prefer_remove_parity: bool = False,
) -> Tuple[List[str], NDArray[np.float64]]:
    global _PAIR_CUTS_GLOBAL

    # Calibrate pair cuts from CIF once (robust bipartite ruler everywhere)
    _PAIR_CUTS_GLOBAL = derive_pair_cuts_from_cif(cif_path, charges, safety=1.00)
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
        pretty_print_pair_cuts(_PAIR_CUTS_GLOBAL, pairs_hint=hints)

    # Prepass cleanup (prints its own actions)
    symbols, pts, uv_taken, edit_count_facet = prepass_surface_cleanup(
        symbols, pts, planes, charges, ligand, surf_tol, pair_cuts=_PAIR_CUTS_GLOBAL, verbose=verbose
    )
    add_count_facet: Dict[int, int] = defaultdict(int)
    host_taken: Dict[int, int] = {}

    while True:
        # Shared per-iteration state (with robust bipartite CN)
        frames = _build_facet_frames(planes)
        mem = _facet_memberships(pts, planes, surf_tol)
        cn_bi = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=_PAIR_CUTS_GLOBAL)
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
            symbols, pts, frames, mem, cn_bi, charges, surf_tol, uv_taken, edit_count_facet, uv_cache, ligand, verbose=verbose
        )
        if progressed:
            continue

        # Electrical
        Q = _total_Q(symbols, charges)
        if Q == 0:
            if verbose:
                print("[done] Structural + electrical stability reached (Q=0, no CN≤2 on surface).")
            return symbols, pts

        if Q < 0:
            if verbose:
                print("[strategy] Q<0 → swap stable anions (CN=3→4→5), V→E→U + FPS")
            progressed, symbols, pts = _priority3_balance_negative_q(
                symbols, pts, frames, planes, mem, cn_bi, charges, ligand, surf_tol, uv_taken, edit_count_facet, uv_cache,
                verbose=verbose
            )
            if progressed:
                continue
            # fallback: add anions on cation sites
            progressed, symbols, pts = _priority3_balance_positive_q_add(
                symbols, pts, frames, planes, mem, cn_bi, charges, ligand, surf_tol,
                uv_taken, edit_count_facet, add_count_facet, host_taken, uv_cache, verbose=verbose
            )
            if progressed:
                continue
            if verbose:
                print("[halt] Q<0 but no valid swaps/additions remain. Consider revising facet energies or inputs.")
            return symbols, pts

        # Q > 0
        if verbose:
            print(f"[strategy] Q>0 → positive_q_strategy='{positive_q_strategy}' (remove cations or add anions)")
        if positive_q_strategy == "remove":
            # --- transactional remove with rollback if we overshoot below zero ---
            _sym0, _pts0 = list(symbols), pts.copy()
            _Q_before = _total_Q(symbols, charges)
            progressed, symbols, pts = _priority3_balance_positive_q_remove(
                symbols, pts, frames, planes, mem, cn_bi, charges, surf_tol,
                uv_taken, edit_count_facet, uv_cache, ligand, verbose=verbose
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
                            uv_taken, edit_count_facet, add_count_facet, host_taken, uv_cache, verbose=verbose
                        )
                        if not ok:
                            break
                        _added_any = True
                        if _total_Q(symbols, charges) == 0:
                            break
                    if _total_Q(symbols, charges) == 0:
                        # Achieved neutrality; continue outer loop
                        continue
                    # If we couldn't add enough ligands, fall through to parity finisher below
                    if _added_any:
                        continue
                else:
                    continue
            # parity finisher
            progressed, symbols, pts = _priority3_balance_positive_q_add(
                symbols, pts, frames, planes, mem, cn_bi, charges, ligand, surf_tol,
                uv_taken, edit_count_facet, add_count_facet, host_taken, uv_cache, verbose=verbose
            )
            if progressed:
                continue
            if verbose:
                print("[halt] Q>0 but no removable cations/additions available. Consider revising facet energies.")
            return symbols, pts
        else:
            # positive_q_strategy == "add"
            progressed, symbols, pts = _priority3_balance_positive_q_add(
                symbols, pts, frames, planes, mem, cn_bi, charges, ligand, surf_tol,
                uv_taken, edit_count_facet, add_count_facet, host_taken, uv_cache, verbose=verbose
            )
            if progressed:
                continue
            if verbose:
                print("[halt] Q>0 and cannot add more ligands — likely surface fully saturated.")
            return symbols, pts
