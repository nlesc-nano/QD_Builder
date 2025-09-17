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

def _build_facet_frames(planes):
    frames = []
    for (n, d) in planes:
        n = np.asarray(n, float); ln = np.linalg.norm(n) + 1e-12
        n = n / ln; d = float(d) / ln
        a = np.array([1.0, 0.0, 0.0]); 
        if abs(np.dot(a, n)) > 0.9: a = np.array([0.0, 1.0, 0.0])
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

def _collect_cation_remove_candidates(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    charges: Dict[str, int],
    surf_tol: float,
    allowed_facets: Optional[Set[int]] = None,
) -> List[Tuple[int, str, int, float, int, int]]:
    """
    Return surface cation-removal candidates:
      (idx, elem, q_cation, depth, role_rank, fid)

    Ranking preference (applied later):
      q_cation asc  → remove lowest-charged species first (e.g., Cu+ before In3+)
      role vertex>edge>unique  → role_rank desc (2,1,0)
      depth asc (shallower first)
    """
    pts = np.asarray(pts, float)
    cations = {el for el, q in charges.items() if q > 0}

    # memberships per atom
    memberships = [[] for _ in range(len(symbols))]
    for fid, (n, d) in enumerate(planes):
        if allowed_facets is not None and fid not in allowed_facets:
            continue
        shell = np.where((d - pts @ n) < surf_tol)[0]
        for i in shell:
            memberships[i].append(fid)

    # candidates, keeping the SHALLOWEST facet hit per atom
    best: Dict[int, Tuple[int, str, int, float, int, int]] = {}
    for fid, (n, d) in enumerate(planes):
        if allowed_facets is not None and fid not in allowed_facets:
            continue
        n_unit = n / (np.linalg.norm(n) + 1e-12)
        shell = np.where((d - pts @ n_unit) < surf_tol)[0]
        for i in shell:
            s = symbols[i]
            if s not in cations:
                continue
            depth = float(d - np.dot(pts[i], n_unit))
            role_rank = 0 if len(memberships[i]) == 1 else (1 if len(memberships[i]) == 2 else 2)
            q = int(charges.get(s, 0))
            rec = (i, s, q, depth, role_rank, fid)
            if i not in best or depth < best[i][3]:
                best[i] = rec

    out = list(best.values())
    # primary coarse ranking (we’ll still do facet seeding + FPS when picking)
    out.sort(key=lambda r: (r[2], -r[4], r[3], r[0]))  # (q asc, role_rank desc, depth asc, idx)
    return out


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
    cation_ligand: str | None = None,  # <-- NEW (kw-only, backward compatible)
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
    # --- 3) Prefer removing surface cations (default) with strict V→E→U gating and global spacing ---
    if Q > 0 or (Q < 0 and not swapped_log):
        frames = _build_facet_frames(planes)  # frames[fid] = (n_norm, d_norm, u, v, x0)

        from math import hypot
        # UV positions taken on each facet by ANY prior edit (removal/addition)
        uv_taken: Dict[int, List[Tuple[float, float]]] = {}
        # per-facet removal counts (for facet round-robin)
        rem_count_facet: Dict[int, int] = {}

        def _min_uv_dist(fid: int, uv: Tuple[float, float]) -> float:
            pts_uv = uv_taken.get(fid, [])
            if not pts_uv:
                return float("inf")
            x, y = uv
            return min(hypot(x - u, y - v) for (u, v) in pts_uv)

        def _incident_fids(idx: int) -> List[int]:
            """Which facets this atom belongs to (within surf_tol), using normalized (n,d) in frames."""
            out = []
            for fid, (n, d, *_rest) in enumerate(frames):
                depth = d - float(np.dot(pts[idx], n))
                if depth < surf_tol:
                    out.append(fid)
            return out

        def _record_uv_allfacets(idx: int) -> None:
            """Record UV of site idx on *all* incident facets to enforce cross-manifold spacing."""
            for fid in _incident_fids(idx):
                uv = _plane_uv(frames, fid, pts[idx])
                uv_taken.setdefault(fid, []).append(uv)
                rem_count_facet[fid] = rem_count_facet.get(fid, 0) + 1

        # prefer cation-rich facets for removals; if none rich, allow all
        surf_Q = facet_surface_charge(symbols, pts, planes, charges, surf_tol)
        rich = {fid for fid, q in surf_Q.items() if q > 0}
        allowed = rich if rich else None

        def _collect_cation_remove_candidates(
            symbols: List[str],
            pts: NDArray[np.float64],
            planes: List[Plane],
            charges: Dict[str, int],
            surf_tol: float,
            allowed_facets: Optional[Set[int]] = None,
        ) -> List[Tuple[int, str, int, float, int, int]]:
            """
            Return surface cation-removal candidates as (idx, elem, q_cation, depth, role_rank, fid),
            where role_rank: unique=0, edge=1, vertex=2. fid is the shallowest-hit facet for that atom.
            """
            pts = np.asarray(pts, float)
            cations = {el for el, q in charges.items() if q > 0}

            # memberships by half-space test
            memberships = [[] for _ in range(len(symbols))]
            for fid, (n, d) in enumerate(planes):
                if allowed_facets is not None and fid not in allowed_facets:
                    continue
                shell = np.where((d - pts @ n) < surf_tol)[0]
                for i in shell:
                    memberships[i].append(fid)

            best: Dict[int, Tuple[int, str, int, float, int, int]] = {}
            for fid, (n, d) in enumerate(planes):
                if allowed_facets is not None and fid not in allowed_facets:
                    continue
                n_unit = n / (np.linalg.norm(n) + 1e-12)
                shell = np.where((d - pts @ n_unit) < surf_tol)[0]
                for i in shell:
                    s = symbols[i]
                    if s not in cations:
                        continue
                    depth = float(d - np.dot(pts[i], n_unit))
                    m = len(memberships[i])
                    role_rank = 0 if m == 1 else (1 if m == 2 else 2)  # unique, edge, vertex
                    q = int(charges.get(s, 0))
                    rec = (i, s, q, depth, role_rank, fid)
                    if i not in best or depth < best[i][3]:
                        best[i] = rec

            out = list(best.values())
            # Coarse sort (used only for tie-breaking later)
            out.sort(key=lambda r: (r[2], -r[4], r[3], r[0]))  # q asc, role vertex>edge>unique, shallow first
            return out

        def _gather_removals() -> List[Tuple[int, str, int, float, int, int]]:
            return _collect_cation_remove_candidates(symbols, pts, planes, charges, surf_tol, allowed_facets=allowed)

        def _pick_from_role(cand: List[Tuple[int, str, int, float, int, int]], need_role: int) -> Optional[Tuple[int, str, int, float, int, int]]:
            """Pick best candidate within a fixed role: facet RR, then max spacing on that facet."""
            cand = [r for r in cand if r[4] == need_role]  # keep only that manifold
            if not cand:
                return None
            # facet RR: facets with fewest removals so far
            fids = {r[5] for r in cand}
            minc_f = min(rem_count_facet.get(fid, 0) for fid in fids)
            fids_rr = [fid for fid in sorted(fids) if rem_count_facet.get(fid, 0) == minc_f]

            best = None
            best_score = None
            for fid in fids_rr:
                on_f = [r for r in cand if r[5] == fid]
                for (i, s, q, depth, role_rank, _fid) in on_f:
                    uv = _plane_uv(frames, fid, pts[i])
                    dmin = _min_uv_dist(fid, uv)
                    # spacing first; then prefer lower q (Cu+ before In3+), then shallower, then older idx
                    score = (dmin, -q, -depth, -i)
                    if best_score is None or score > best_score:
                        best_score, best = score, (i, s, q, depth, role_rank, fid)
            return best

        # optional fine-step: add one anion (−1) if we cannot remove within current gating without overshoot
        def _add_one_anion() -> bool:
            nonlocal symbols, pts, Q
            sites = _collect_cation_sites(
                symbols, pts, planes, charges, surf_tol,
                outer_only=True, allow_shared=False, include_sublayer=False, allowed_facets=allowed
            )
            if not sites:
                sites = _collect_cation_sites(
                    symbols, pts, planes, charges, surf_tol,
                    outer_only=True, allow_shared=True, include_sublayer=False, allowed_facets=allowed
                )
            if not sites:
                sites = _collect_cation_sites(
                    symbols, pts, planes, charges, surf_tol,
                    outer_only=False, allow_shared=True, include_sublayer=True, allowed_facets=allowed
                )
            if not sites:
                return False

            # choose site maximizing spacing wrt ALL prior edits on that facet
            sites.sort(key=lambda t: (-t[3], t[2], t[4], t[0]))  # deficit desc, depth asc, role, idx
            best = None; best_key = None
            for (idx, n, depth, deficit, role_rank, fid) in sites:
                uv = _plane_uv(frames, fid, pts[idx])
                dmin = _min_uv_dist(fid, uv)
                key = (rem_count_facet.get(fid, 0), -dmin, depth, -deficit, role_rank, idx)
                if best_key is None or key < best_key:
                    best_key, best = key, (idx, n, depth, deficit, role_rank, fid)

            if best is None:
                return False

            idx, n, depth, deficit, role_rank, fid = best
            before = Q
            symbols, pts = place_ligand(symbols, pts, idx, n, ligand, planes)
            Q = int(sum(charges.get(x, 0) for x in symbols))
            # record UV for additions on all incident facets of the host site (spacing awareness)
            _record_uv_allfacets(idx)

            if verbose:
                role = ["unique", "edge", "vertex"][role_rank]
                print(f"add {ligand} near {symbols[idx]}#{idx} "
                      f"(def={deficit}, {role}, depth={depth:.2f} Å, facet {fid})  | Q:{before:+d}→{Q:+d}")
            return True

        # -------- main loop with strict manifold gating (vertex → edge → unique) --------
        ROLE_ORDER = (2, 1, 0)  # 2=vertex, 1=edge, 0=unique/center
        while Q > 0:
            cand_all = _gather_removals()
            if not cand_all:
                # nothing removable; try one -1 addition as last resort and re-loop
                if not _add_one_anion():
                    if verbose:
                        print("WARNING: no removable cations available and no suitable anion-add site found.")
                    break
                continue

            progressed = False
            for role in ROLE_ORDER:
                # only consider this role if there exists at least one candidate in it
                if not any(r[4] == role for r in cand_all):
                    continue
                # avoid overshoot: restrict to q <= Q within the role
                role_cand = [r for r in cand_all if r[4] == role and r[2] <= Q]
                if not role_cand:
                    # cannot progress within this role without overshoot; try next role
                    continue
                # pick best within role (facet RR + farthest spacing)
                pick = _pick_from_role(role_cand, role)
                if pick is None:
                    continue

                i, s, q, depth, role_rank, fid = pick
                before = Q

                # record UV on all incident facets BEFORE deletion (cross-manifold spacing)
                _record_uv_allfacets(i)

                # delete atom i
                symbols.pop(i)
                pts = np.delete(pts, i, axis=0)

                Q = int(sum(charges.get(x, 0) for x in symbols))
                if verbose:
                    role_name = ["unique", "edge", "vertex"][role_rank]
                    print(f"remove {s}#{i} (q=+{q}, {role_name}, depth={depth:.2f} Å, facet {fid})  | Q:{before:+d}→{Q:+d}")
                progressed = True
                break  # restart from highest role (keep V→E→U gating)

            if progressed:
                continue

            # Reached here: no role could progress without overshoot → try a single -1 anion add
            if not _add_one_anion():
                if verbose:
                    print("WARNING: stuck by overshoot constraints and cannot add an anion; stopping.")
                break


    if verbose:
        print(f"# Q after  = {Q:+d}")
    if Q != 0 and verbose:
        print("WARNING: neutrality not reached. Consider enabling deeper swaps or revisiting charges.")

    return symbols, pts

