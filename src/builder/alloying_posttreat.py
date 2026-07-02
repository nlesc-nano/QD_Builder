from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

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
    return work_syms, work_pts, ledger
