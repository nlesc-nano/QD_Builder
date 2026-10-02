from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .analysis import _pair_cut_calibrated
from .nc_types import AlloyingPostTreatSpec, Config, Plane
from .neutral_ligand_posttreat import _subsample_sites


def _surface_mask(pts: NDArray[np.float64], planes: List[Plane], surf_tol: float) -> NDArray[np.bool_]:
    pts = np.asarray(pts, float)
    mask = np.zeros(len(pts), bool)
    for normal, d in planes or []:
        normal = np.asarray(normal, float)
        mask |= ((float(d) - pts @ normal) < float(surf_tol))
    return mask


def _native_species(bulk_struct, cfg: Config) -> set[str]:
    if bulk_struct is not None and hasattr(bulk_struct, "sites"):
        species = {str(site.specie.symbol) for site in bulk_struct.sites}
    else:
        species = {s for s, q in cfg.charges.items() if int(q) != 0}
    species.discard(cfg.passivation.ligand)
    if cfg.passivation.cation_ligand:
        species.discard(cfg.passivation.cation_ligand)
    return species


def detect_alloying_options(
    syms: List[str],
    pts: NDArray[np.float64],
    cfg: Config,
    bulk_struct,
    planes: List[Plane],
) -> List[dict]:
    native = _native_species(bulk_struct, cfg)
    surface = _surface_mask(np.asarray(pts, float), planes, getattr(cfg.passivation, "surf_tol", 2.0))
    out = []
    for sym in sorted(native):
        idxs = [i for i, s in enumerate(syms) if s == sym]
        if not idxs:
            continue
        surface_count = sum(1 for i in idxs if i < len(surface) and bool(surface[i]))
        total_count = len(idxs)
        q = int(cfg.charges.get(sym, 0))
        out.append({
            "element": sym,
            "charge": q,
            "site_type": "cation" if q > 0 else ("anion" if q < 0 else "neutral"),
            "surface_count": int(surface_count),
            "core_count": int(total_count - surface_count),
            "total_count": int(total_count),
        })
    return out


def _candidate_indices(
    syms: List[str],
    pts: NDArray[np.float64],
    replace: str,
    region: str,
    surface: NDArray[np.bool_],
) -> List[int]:
    candidates = []
    for i, sym in enumerate(syms):
        if sym != replace:
            continue
        is_surface = i < len(surface) and bool(surface[i])
        if region == "surface" and not is_surface:
            continue
        if region == "core" and is_surface:
            continue
        candidates.append(i)
    return candidates


def _select_indices(
    indices: List[int],
    pts: NDArray[np.float64],
    ratio: float,
    target_count: int,
    distribution: str,
    seed: int,
) -> List[int]:
    if not indices:
        return []
    n = len(indices)
    if int(target_count or 0) > 0:
        k = min(int(target_count), n)
    else:
        k = int(round(float(ratio) * n))
    if k <= 0:
        return []
    local_positions = np.asarray([pts[i] for i in indices], float)
    selected_local = _subsample_sites(local_positions, min(1.0, k / max(1, n)), distribution, seed)
    return [indices[int(i)] for i in selected_local[:k]]


def _strip_ligands_for_lower_valence(
    syms: List[str],
    pts: NDArray[np.float64],
    charges: Dict[str, int],
    ligand: str,
    substituted: Set[int],
    n_strip: int,
    min_host_cn: int = 2,
) -> Tuple[List[str], NDArray[np.float64], int]:
    """Remove up to ``n_strip`` anionic X-type ligands after a lower-valent substitution.

    A Zn2+ on an In3+ site carries one positive charge less, so the surface
    holds one Cl- too many.  Strip ligands bound to the substituted cations
    first (terminal before bridging), then the ligands closest to them, never
    leaving a host cation with fewer than ``min_host_cn`` bonds.
    """
    pts = np.asarray(pts, float)
    q_lig = int(charges.get(ligand, 0))
    if n_strip <= 0 or q_lig >= 0:
        return syms, pts, 0
    cation_idx = np.array([i for i, s in enumerate(syms) if int(charges.get(s, 0)) > 0], int)
    if len(cation_idx) == 0:
        return syms, pts, 0
    tree_all = cKDTree(pts)
    anion_set = {s for s, q in charges.items() if int(q) < 0}
    max_cut = max(_pair_cut_calibrated(str(syms[c]), a, None) for c in set(cation_idx.tolist()) for a in anion_set)

    def neighbours(i: int, opposite_positive: bool) -> List[int]:
        out = []
        for j in tree_all.query_ball_point(pts[i], max_cut):
            if j == i or not alive[j]:
                continue
            qj = int(charges.get(syms[j], 0))
            if (qj > 0) != opposite_positive or qj == 0:
                continue
            if np.linalg.norm(pts[j] - pts[i]) <= _pair_cut_calibrated(syms[i], syms[j], None):
                out.append(j)
        return out

    alive = np.ones(len(syms), bool)
    sub_pts = pts[sorted(substituted)] if substituted else np.zeros((0, 3))
    sub_tree = cKDTree(sub_pts) if len(sub_pts) else None
    removed = 0
    while removed < n_strip:
        best = None
        for li in (i for i, s in enumerate(syms) if s == ligand and alive[i]):
            hosts = neighbours(li, opposite_positive=True)
            if not hosts:
                continue
            host_cn_after = min(len(neighbours(h, opposite_positive=False)) - 1 for h in hosts)
            if host_cn_after < min_host_cn:
                continue
            on_sub = any(h in substituted for h in hosts)
            d_sub = float(sub_tree.query(pts[li])[0]) if sub_tree is not None else 0.0
            key = (not on_sub, len(hosts), d_sub, -host_cn_after, li)
            if best is None or key < best[0]:
                best = (key, li)
        if best is None:
            break
        alive[best[1]] = False
        removed += 1
    keep = np.where(alive)[0]
    return [syms[i] for i in keep], pts[keep], removed


def run_alloying_posttreatment(
    syms: List[str],
    pts: NDArray[np.float64],
    cfg: Config,
    bulk_struct,
    planes: List[Plane],
) -> Tuple[List[str], NDArray[np.float64], List[dict]]:
    spec: AlloyingPostTreatSpec = getattr(
        getattr(cfg, "post_treatment", None), "alloying", AlloyingPostTreatSpec()
    )
    if not spec.enabled or not spec.passes:
        return syms, pts, []

    print("\n[post-treatment] ── Inorganic alloying ───────────────────────────────")
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    work_syms = list(syms)
    work_pts = np.asarray(pts, float).copy()
    ledger = []

    for pass_idx, pass_spec in enumerate(spec.passes):
        surface = _surface_mask(work_pts, planes, getattr(cfg.passivation, "surf_tol", 2.0))
        candidates = _candidate_indices(work_syms, work_pts, pass_spec.replace, pass_spec.region, surface)
        selected = _select_indices(
            candidates,
            work_pts,
            pass_spec.ratio,
            pass_spec.target_count,
            pass_spec.distribution,
            spec.seed + pass_idx,
        )
        print(
            f"\n[alloying:pass-{pass_idx + 1}] {pass_spec.replace}->{pass_spec.replacement} "
            f"region={pass_spec.region} ratio={pass_spec.ratio:.2f} target={pass_spec.target_count} "
            f"dist={pass_spec.distribution}"
        )
        print(f"  → Selected {len(selected)} / {len(candidates)} eligible atoms")
        if not selected:
            continue
        for idx in selected:
            work_syms[idx] = pass_spec.replacement
        q_old = int(cfg.charges.get(pass_spec.replace, 0))
        q_new = int(pass_spec.replacement_charge)
        ledger.append({
            "replace": pass_spec.replace,
            "replacement": pass_spec.replacement,
            "replacement_charge": q_new,
            "region": pass_spec.region,
            "count": len(selected),
            "charge_delta": len(selected) * (q_new - q_old),
        })

    total = sum(int(entry.get("count", 0)) for entry in ledger)
    print(f"[alloying:done] Total atoms substituted: {total}")

    # A lower-valent cation (Zn2+ for In3+) leaves the shell with surplus
    # X-type ligand; strip exactly that surplus instead of letting the generic
    # rebalance swap native anions for ligand.
    ligand = cfg.passivation.ligand
    charges = dict(cfg.charges)
    for entry in ledger:
        charges.setdefault(entry["replacement"], int(entry["replacement_charge"]))
    q_now = int(sum(int(charges.get(s, 0)) for s in work_syms))
    delta = int(sum(int(entry["charge_delta"]) for entry in ledger))
    q_lig = int(charges.get(ligand, 0)) if ligand else 0
    if ledger and q_now < 0 and delta < 0 and q_lig < 0:
        n_strip = min(-q_now, -delta) // -q_lig
        substituted = {
            i for i, s in enumerate(work_syms)
            if any(s == entry["replacement"] for entry in ledger)
        }
        # Keep hosts at CN>=3 while possible: a CN-2 cation is pruned by the
        # rebalance that follows, undoing the substitution.
        stripped = 0
        for min_host_cn in (3, 2):
            if stripped >= n_strip:
                break
            work_syms, work_pts, n_done = _strip_ligands_for_lower_valence(
                work_syms, work_pts, charges, ligand, substituted, n_strip - stripped,
                min_host_cn=min_host_cn,
            )
            stripped += n_done
            substituted = {
                i for i, s in enumerate(work_syms)
                if any(s == entry["replacement"] for entry in ledger)
            }
        ledger[-1]["ligands_stripped"] = stripped
        q_after = int(sum(int(charges.get(s, 0)) for s in work_syms))
        print(f"[alloying:charge] stripped {stripped} {ligand} ligand(s) | Q:{q_now:+d}→{q_after:+d}")
    return work_syms, work_pts, ledger
