# nanocrystal_builder/passivation.py
from __future__ import annotations
import random
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from numpy.typing import NDArray

from .nc_types import Plane, Facet
from .analysis import coord_numbers, bulk_cn_by_interior
from .chemistry import facet_surface_charge, place_ligand

__all__ = ["collect_anion_candidates", "charge_balance"]

# --------------------------------------
# Internal helpers
# --------------------------------------
def _facet_memberships(pts: NDArray[np.float64], planes: List[Plane], surf_tol: float) -> List[List[int]]:
    """List of facet-IDs each atom belongs to (within surf_tol)."""
    mem = [[] for _ in range(len(pts))]
    for fid, (n, d) in enumerate(planes):
        shell = np.where((d - pts @ n) < surf_tol)[0]
        for i in shell:
            mem[i].append(fid)
    return mem

def _role_and_rank(m: int) -> Tuple[str, int]:
    """
    Map membership count to role + rank:
      unique (m=1) -> rank 0 (highest priority for swaps)
      edge   (m=2) -> rank 1
      vertex (m>=3)-> rank 2
    """
    if m == 1:  return "unique", 0
    if m == 2:  return "edge",   1
    return "vertex",             2

def _build_facet_frames(planes: List[Plane]):
    """
    For each facet plane, build an orthonormal in-plane frame (u, v) and a point x0 on the plane.
    Returns list of (n, d, u, v, x0).
    """
    frames = []
    for (n, d) in planes:
        n = n / (np.linalg.norm(n) + 1e-12)
        a = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, n)) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        u = np.cross(n, a); u /= (np.linalg.norm(u) + 1e-12)
        v = np.cross(n, u); v /= (np.linalg.norm(v) + 1e-12)
        x0 = d * n
        frames.append((n, d, u, v, x0))
    return frames

def _plane_uv(frames, fid: int, x: NDArray[np.float64]) -> Tuple[float, float]:
    """Project 3D point x to facet fid's (u,v) coordinates."""
    n, d, u, v, x0 = frames[fid]
    xproj = x - (np.dot(x, n) - d) * n      # orthogonal projection onto plane
    return float(np.dot(xproj - x0, u)), float(np.dot(xproj - x0, v))

# --------------------------------------
# Candidate collection (no mutation)
# --------------------------------------
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
    Returns (outer_candidates, sublayer_candidates), where each candidate is:
      dict(idx, elem, cn, bulk_cn, deficit, depth, role, role_rank, fid)

    Intended ranking (applied later during swaps):
      (-deficit, role_rank, depth)  # larger deficit first (e.g., 1/4 before 2/4 before 3/4), then unique>edge>vertex, then shallower
    """
    pts = np.asarray(pts, float)
    cn = coord_numbers(symbols, pts)
    bulk_cn = bulk_cn_by_interior(symbols, pts, planes, surf_tol)
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
            role, role_rank = _role_and_rank(len(memberships[i]))
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
# Cation-site collection for additions
# --------------------------------------
def _collect_cation_sites(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    charges: Dict[str, int],
    surf_tol: float,
    *,
    outer_only: bool = True,
    allow_shared: bool = True,         # allow edge/vertex
    include_sublayer: bool = False,
    allowed_facets: Optional[Set[int]] = None,
) -> List[Tuple[int, NDArray[np.float64], float, int, int, int]]:
    """
    Return candidate cation sites to attach one ligand:
      (idx, normal, depth, deficit, role_rank, fid)

    - outer_only: restrict to outer layer (depth < 0.35*surf_tol)
    - allow_shared: include edge/vertex (shared) atoms
    - include_sublayer: allow sublayer cations (depth < 1.2*surf_tol)
    - allowed_facets: optional whitelist of facet ids

    Results are later ranked by: higher deficit first, shallower depth first,
    unique>edge>vertex, then idx.
    """
    pts = np.asarray(pts, float)
    cn = coord_numbers(symbols, pts)
    bulk = bulk_cn_by_interior(symbols, pts, planes, surf_tol)
    cations = {el for el, q in charges.items() if q > 0}

    # memberships
    memberships = [[] for _ in range(len(symbols))]
    for fid, (n, d) in enumerate(planes):
        shell = np.where((d - pts @ n) < surf_tol)[0]
        for i in shell:
            memberships[i].append(fid)

    outer_thr = 0.35 * surf_tol
    subl_thr  = 1.20 * surf_tol

    out: List[Tuple[int, NDArray[np.float64], float, int, int, int]] = []
    for fid, (n, d) in enumerate(planes):
        if allowed_facets is not None and fid not in allowed_facets:
            continue
        n_unit = n / (np.linalg.norm(n) + 1e-12)
        shell = np.where((d - pts @ n_unit) < surf_tol)[0]
        for i in shell:
            s = symbols[i]
            if s not in cations:
                continue
            depth = d - float(np.dot(pts[i], n_unit))
            if outer_only and not (depth < outer_thr):
                continue
            if not include_sublayer and depth >= outer_thr:
                continue
            if include_sublayer and not (depth < subl_thr):
                continue

            m = len(memberships[i])
            role_rank = 0 if m == 1 else (1 if m == 2 else 2)
            if not allow_shared and role_rank != 0:
                continue

            deficit = max(0, bulk[s] - cn[i])
            if deficit <= 0:
                continue

            out.append((i, n_unit, depth, int(deficit), role_rank, fid))

    # keep only shallowest record per atom (dedupe potential multiple shell hits)
    best: Dict[int, Tuple[int, NDArray[np.float64], float, int, int, int]] = {}
    for rec in out:
        i = rec[0]
        if i not in best or rec[2] < best[i][2]:
            best[i] = rec
    return list(best.values())

# --------------------------------------
# Charge balance (stepwise, facet-aware swaps & removals)
# --------------------------------------
def charge_balance(
    symbols: List[str],
    pts: NDArray[np.float64],
    outer_candidates: List[dict],
    sublayer_candidates: List[dict],   # reserved for future use
    charges: Dict[str, int],
    ligand: str,
    verbose: bool,
    planes: List[Plane],
    facets: List[Facet],
    surf_tol: float,
    rng: random.Random,
    *,
    prefer_remove_parity: bool = False,
):
    """
    Stepwise neutrality:

    1) While Q < 0: Se→Cl on OUTER anions using facet-aware FPS **within categories**:
         Categories prioritized by deficit (larger first: 1/4 → 2/4 → 3/4), then role (unique → edge → vertex).
         For each (deficit, role):
           A) Seed: take one per facet (shallower first).
           B) Space: choose candidate that maximizes min in-plane distance to existing swaps on that facet (FPS).

       Parity policy (odd Q after +2 steps per swap):
         - prefer_remove_parity=False (default): allow overshoot to Q=+1; later add one ligand (−1) if needed.
         - prefer_remove_parity=True: stop at Q=-1 and resolve by removing one ligand (+1).

    2) If still Q < 0 after swaps: REMOVE some Cl to raise Q (facet-aware & spaced).
         CN 3/4 first, then 2/4; within CN: vertex > edge > unique; deeper first.
         Seed one removal per facet before spacing.

    3) If Q > 0: add Cl near under-coordinated cations (outer first; relax as needed).

    Returns updated (symbols, pts).
    """
    def total_Q() -> int:
        return int(sum(charges.get(s, 0) for s in symbols))

    Q = total_Q()
    if verbose:
        print(f"# Q before = {Q:+d}")

    frames = _build_facet_frames(planes)

    # --- 1) Facet-aware Se->Cl swaps with FPS within categories ---
    deficits = sorted({r["deficit"] for r in outer_candidates}, reverse=True) if outer_candidates else []

    swapped_log: List[dict] = []   # {idx, cn, bulk_cn, role, role_rank, depth, fid}
    # UV positions of swapped Cl per (deficit, role_rank, facet)
    swap_uv: Dict[Tuple[int, int, int], List[Tuple[float, float]]] = {}

    def _available_in_category(deficit: int, role_rank: int):
        return [r for r in outer_candidates
                if r["deficit"] == deficit and r["role_rank"] == role_rank
                and 0 <= r["idx"] < len(symbols) and symbols[r["idx"]] != ligand]

    def _seeded_facets_in_category(deficit: int, role_rank: int) -> set[int]:
        return {fid for (d, rr, fid), uv in swap_uv.items() if d == deficit and rr == role_rank and uv}

    def _min_uv_dist_for_swap(r: dict) -> float:
        key = (r["deficit"], r["role_rank"], r["fid"])
        uv_list = swap_uv.get(key, [])
        if not uv_list:
            return 0.0
        ux, vy = _plane_uv(frames, r["fid"], pts[r["idx"]])
        return float(min(np.hypot(ux - x, vy - y) for (x, y) in uv_list))

    # Loop categories
    stop_swaps = False
    for deficit in deficits:
        if stop_swaps: break
        for role_rank in (0, 1, 2):  # unique -> edge -> vertex
            if stop_swaps: break
            while Q < 0:
                # Parity preference: if we prefer resolving by removal, stop at -1.
                if prefer_remove_parity and Q == -1:
                    if verbose:
                        print("(parity) stopping swaps at Q=-1 to resolve by removing one ligand.")
                    stop_swaps = True
                    break

                cand = _available_in_category(deficit, role_rank)
                if not cand:
                    break

                # Stage A: seed one per facet if possible
                seeded_facets = _seeded_facets_in_category(deficit, role_rank)
                unseeded = {r["fid"] for r in cand if r["fid"] not in seeded_facets}
                if unseeded:
                    fid_pick = min(unseeded)  # deterministic facet selection
                    c_facet = [r for r in cand if r["fid"] == fid_pick]
                    c_facet.sort(key=lambda r: (r["depth"], r["idx"]))  # shallower first
                    picked = c_facet[0]
                else:
                    # Stage B: spacing (maximize min UV distance to existing swaps on same facet)
                    scored: List[Tuple[float, float, int, int]] = []
                    # tuple: (dmin, -depth, idx, idx_in_cand)
                    for k, r in enumerate(cand):
                        dmin = _min_uv_dist_for_swap(r)
                        scored.append((dmin, -float(r["depth"]), int(r["idx"]), k))
                    scored.sort(key=lambda t: (t[0], t[1], -t[2]), reverse=True)
                    picked = cand[scored[0][3]]

                # Apply swap
                i = picked["idx"]
                before = Q
                old = symbols[i]
                symbols[i] = ligand
                Q = total_Q()
                swapped_log.append({
                    "idx": i,
                    "cn": int(picked["cn"]),
                    "bulk_cn": int(picked["bulk_cn"]),
                    "role": ["unique", "edge", "vertex"][role_rank],
                    "role_rank": role_rank,
                    "depth": float(picked["depth"]),
                    "fid": int(picked["fid"]),
                })
                # record UV for FPS
                key = (deficit, role_rank, picked["fid"])
                uv = _plane_uv(frames, picked["fid"], pts[i])
                swap_uv.setdefault(key, []).append(uv)

                if verbose:
                    print(f"swap {old}#{i} (CN {picked['cn']}/{picked['bulk_cn']}, "
                          f"{['unique','edge','vertex'][role_rank]}, depth={picked['depth']:.2f} Å) "
                          f"→ {ligand}  | Q:{before:+d}→{Q:+d}")

                if Q >= 0:
                    stop_swaps = True
                    break

    # --- 2) If still negative, facet-aware Cl removals (seed + spacing) ---
    if Q < 0 and swapped_log:
        vac_uv: Dict[int, List[Tuple[float, float]]] = {}
        seeded_remove: set[int] = set()

        def _reindex_after_delete(rem: int):
            for r in swapped_log:
                if r["idx"] > rem:
                    r["idx"] -= 1

        def _removable_now() -> List[dict]:
            return [r for r in swapped_log if 0 <= r["idx"] < len(symbols) and symbols[r["idx"]] == ligand]

        def _min_uv_dist(fid: int, uv: Tuple[float, float]) -> float:
            pts_uv = vac_uv.get(fid, [])
            if not pts_uv:
                return 0.0
            ux, vy = uv
            return float(min(np.hypot(ux - x, vy - y) for (x, y) in pts_uv))

        cn_groups = (3, 2)

        while Q < 0:
            removable = _removable_now()
            if not removable:
                if verbose:
                    print("WARNING: No removable Cl left from swapped sites; still negative.")
                break

            # pick highest CN group available
            group = None
            for g in cn_groups:
                if any(r["cn"] == g for r in removable):
                    group = g
                    break
            cand = [r for r in removable if (group is None or r["cn"] == group)]
            if not cand:
                break

            # Stage A: one removal per facet (prefer vertex>edge>unique; deeper first)
            unseeded = {r["fid"] for r in cand if r["fid"] not in seeded_remove}
            if unseeded:
                fid_pick = min(unseeded)
                c_facet = [r for r in cand if r["fid"] == fid_pick]
                c_facet.sort(key=lambda r: (r["role_rank"], r["depth"], -int(r["idx"])), reverse=True)
                picked = c_facet[0]
            else:
                # Stage B: spacing within facet(s) with existing removals
                scored: List[Tuple[float, float, int, int]] = []
                # tuple: (dmin, depth, -idx, idx_in_cand)
                for k, r in enumerate(cand):
                    uv = _plane_uv(frames, r["fid"], pts[r["idx"]])
                    dmin = _min_uv_dist(r["fid"], uv)
                    scored.append((dmin, float(r["depth"]), -int(r["idx"]), k))
                scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
                picked = cand[scored[0][3]]

            # Apply removal
            i = picked["idx"]
            fid = picked["fid"]
            before = Q

            # record vacancy position
            uv = _plane_uv(frames, fid, pts[i])
            vac_uv.setdefault(fid, []).append(uv)
            seeded_remove.add(fid)

            # remove atom i
            symbols.pop(i)
            pts = np.delete(pts, i, axis=0)
            _reindex_after_delete(i)

            Q = total_Q()
            if verbose:
                print(f"remove {ligand}#{i} (from Se, orig CN {picked['cn']}/{picked['bulk_cn']}, "
                      f"{picked['role']}, depth={picked['depth']:.2f} Å, facet {fid})  | Q:{before:+d}→{Q:+d}")

    elif Q < 0 and not swapped_log:
        # Rare: Q<0 but no swapped Cl to remove (e.g., initial structure already negative and no anion swaps available)
        if verbose:
            print("NOTE: Q<0 but no swapped Cl available to remove; attempting cation additions as fallback.")

    # --- 3) If positive, add ligands near dangling/under-coordinated cations ---
    if Q > 0 or (Q < 0 and not swapped_log):
        # Strategy: progressively relax constraints to guarantee a solution.
        def _try_add(sites: List[Tuple[int, NDArray[np.float64], float, int, int, int]]) -> None:
            nonlocal symbols, pts, Q
            # sort: higher deficit first, shallower first, unique>edge>vertex, then idx
            sites.sort(key=lambda t: (-t[3], t[2], t[4], t[0]))
            for idx, n, depth, deficit, role_rank, fid in sites:
                if Q <= 0:
                    break
                before = Q
                symbols, pts = place_ligand(symbols, pts, idx, n, ligand, planes)
                Q = total_Q()
                if verbose:
                    role = ["unique", "edge", "vertex"][role_rank]
                    print(f"add {ligand} near {symbols[idx]}#{idx} "
                          f"(CN deficit={deficit}, {role}, depth={depth:.2f} Å, facet {fid})  | "
                          f"Q:{before:+d}→{Q:+d}")

        # 3.1: cation-rich facets only, outer, unique only
        surf_Q = facet_surface_charge(symbols, pts, planes, charges, surf_tol)
        rich = {fid for fid, q in surf_Q.items() if q > 0} or None
        if Q > 0:
            sites = _collect_cation_sites(
                symbols, pts, planes, charges, surf_tol,
                outer_only=True, allow_shared=False, include_sublayer=False, allowed_facets=rich
            )
            _try_add(sites)

        # 3.2: allow shared (edge/vertex), still outer
        if Q > 0:
            sites = _collect_cation_sites(
                symbols, pts, planes, charges, surf_tol,
                outer_only=True, allow_shared=True, include_sublayer=False, allowed_facets=rich
            )
            _try_add(sites)

        # 3.3: include shallow sublayer
        if Q > 0:
            sites = _collect_cation_sites(
                symbols, pts, planes, charges, surf_tol,
                outer_only=False, allow_shared=True, include_sublayer=True, allowed_facets=rich
            )
            _try_add(sites)

        # 3.4: absolute fallback — globally most under-coordinated outer cation
        if Q > 0:
            sites = _collect_cation_sites(
                symbols, pts, planes, charges, surf_tol,
                outer_only=True, allow_shared=True, include_sublayer=False, allowed_facets=None
            )
            if sites:
                # already sorted in _try_add, but we need one pick
                sites.sort(key=lambda t: (-t[3], t[2], t[4], t[0]))
                idx, n, depth, deficit, role_rank, fid = sites[0]
                before = Q
                symbols, pts = place_ligand(symbols, pts, idx, n, ligand, planes)
                Q = total_Q()
                if verbose:
                    role = ["unique", "edge", "vertex"][role_rank]
                    print(f"add {ligand} near {symbols[idx]}#{idx} "
                          f"(CN deficit={deficit}, {role}, depth={depth:.2f} Å, facet {fid})  | "
                          f"Q:{before:+d}→{Q:+d}")

    # If prefer_remove_parity=False and we're exactly at Q=+1 (common in +3/−3 with −1 ligand),
    # the addition loop above will add exactly one ligand and finish at Q=0.
    # If prefer_remove_parity=True and we stopped at Q=-1, the removal loop above
    # removes exactly one Cl (from swapped_log) and finishes at Q=0.

    if verbose:
        print(f"# Q after  = {Q:+d}")
    if Q != 0 and verbose:
        print("WARNING: neutrality not reached. Consider enabling deeper swaps or revisiting charges.")

    return symbols, pts

