from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .analysis import (
    PairCuts,
    _pair_cut_calibrated,
    compute_cif_virtual_sites,
    derive_pair_cuts_from_cif,
)
from .nc_types import Config, Plane, ZTypeDisplacementPostTreatSpec
from .neutral_ligand_posttreat import _subsample_sites


def _surface_mask(pts: NDArray[np.float64], planes: List[Plane], surf_tol: float) -> NDArray[np.bool_]:
    pts = np.asarray(pts, float)
    mask = np.zeros(len(pts), bool)
    for normal, d in planes or []:
        normal = np.asarray(normal, float)
        mask |= ((float(d) - pts @ normal) < float(surf_tol))
    return mask


def _native_species(bulk_struct) -> set[str]:
    if bulk_struct is None or not hasattr(bulk_struct, "sites"):
        return set()
    return {str(site.specie.symbol) for site in bulk_struct.sites}


def _formula(cation: str, anion: str, anion_count: int) -> str:
    suffix = "" if int(anion_count) == 1 else str(int(anion_count))
    return f"{cation}{anion}{suffix}"


def _derive_anion_count(cation: str, anion: str, charges: Dict[str, int]) -> Optional[int]:
    q_cat = int(charges.get(cation, 0))
    q_an = int(charges.get(anion, 0))
    if q_cat <= 0 or q_an >= 0:
        return None
    denom = abs(q_an)
    if denom <= 0 or q_cat % denom != 0:
        return None
    return q_cat // denom


def _eligible_indices(
    syms: List[str],
    pts: NDArray[np.float64],
    symbol: str,
    *,
    require_surface: bool,
    surface: NDArray[np.bool_],
) -> List[int]:
    out = []
    for i, sym in enumerate(syms):
        if sym != symbol:
            continue
        if require_surface and (i >= len(surface) or not bool(surface[i])):
            continue
        out.append(i)
    return out


def _anion_search_radius(cation: str, anion: str, cuts: Optional[PairCuts]) -> float:
    try:
        return max(6.0, 2.5 * float(_pair_cut_calibrated(cation, anion, cuts)))
    except Exception:
        return 8.0


def _count_possible_groups(
    cation_indices: List[int],
    anion_indices: List[int],
    anion_count: int,
) -> int:
    if anion_count <= 0:
        return 0
    return min(len(cation_indices), len(anion_indices) // int(anion_count))


def _count_bound_groups(
    syms: List[str],
    pts: NDArray[np.float64],
    cation_indices: List[int],
    anion_indices: List[int],
    anion_count: int,
    cuts: Optional[PairCuts],
    *,
    allow_unbound_completion: bool = False,
) -> int:
    needed = max(1, int(anion_count))
    count = 0
    used_anions = set()
    for ci in cation_indices:
        cation = syms[int(ci)]
        c_pos = pts[int(ci)]
        bound = []
        for ai in anion_indices:
            if ai in used_anions:
                continue
            anion = syms[int(ai)]
            try:
                cutoff = _pair_cut_calibrated(cation, anion, cuts)
            except Exception:
                cutoff = 3.2
            dist = float(np.linalg.norm(pts[int(ai)] - c_pos))
            if dist <= max(3.2, 1.15 * cutoff):
                bound.append((dist, int(ai)))
        if not bound:
            continue
        bound.sort()
        chosen = [ai for _dist, ai in bound[:needed]]
        if allow_unbound_completion and len(chosen) < needed:
            chosen_set = set(chosen)
            extras = sorted(
                (
                    (float(np.linalg.norm(pts[int(ai)] - c_pos)), int(ai))
                    for ai in anion_indices
                    if int(ai) not in used_anions and int(ai) not in chosen_set
                ),
                key=lambda item: item[0],
            )
            for _dist, ai in extras:
                chosen.append(ai)
                if len(chosen) == needed:
                    break
        if len(chosen) < needed:
            continue
        used_anions.update(chosen)
        count += 1
    return count


def _bound_hosts(
    syms: List[str],
    pts: NDArray[np.float64],
    ligand_idx: int,
    charges: Dict[str, int],
    cuts: Optional[PairCuts],
) -> List[int]:
    ligand = syms[ligand_idx]
    q_lig = int(charges.get(ligand, 0))
    if q_lig == 0:
        return []
    hosts: List[Tuple[float, int]] = []
    for i, sym in enumerate(syms):
        if i == ligand_idx:
            continue
        if int(charges.get(sym, 0)) * q_lig >= 0:
            continue
        dist = float(np.linalg.norm(pts[i] - pts[ligand_idx]))
        if dist <= _pair_cut_calibrated(sym, ligand, cuts):
            hosts.append((dist, i))
    hosts.sort()
    return [i for _dist, i in hosts]


def _relocate_orphan_ligands(
    syms: List[str],
    pts: NDArray[np.float64],
    cfg: Config,
    bulk_struct,
    planes: List[Plane],
    cuts: Optional[PairCuts],
    ligand_symbols: set[str],
) -> Tuple[List[str], NDArray[np.float64], Dict[str, int]]:
    """Move unbound passivation ligands to available CIF virtual sites, greedily."""
    pts = np.asarray(pts, float)
    ligand_symbols = {s for s in ligand_symbols if int(cfg.charges.get(s, 0)) != 0}
    if not ligand_symbols:
        return syms, pts, {"detected": 0, "relocated": 0, "removed": 0}

    orphan_records: List[Tuple[str, np.ndarray]] = []
    orphan_indices = set()
    for i, sym in enumerate(syms):
        if sym not in ligand_symbols:
            continue
        if _bound_hosts(syms, pts, i, cfg.charges, cuts):
            continue
        orphan_indices.add(i)
        orphan_records.append((sym, pts[i].copy()))

    if not orphan_records:
        return syms, pts, {"detected": 0, "relocated": 0, "removed": 0}

    keep = [i for i in range(len(syms)) if i not in orphan_indices]
    base_syms = [syms[i] for i in keep]
    base_pts = pts[keep] if keep else np.zeros((0, 3), float)

    if len(base_syms) == 0:
        return base_syms, base_pts, {
            "detected": len(orphan_records),
            "relocated": 0,
            "removed": len(orphan_records),
        }

    surf_tol = getattr(cfg.passivation, "surf_tol", 2.0)
    surface = _surface_mask(base_pts, planes, surf_tol)
    try:
        virtual_sites = compute_cif_virtual_sites(
            base_syms,
            base_pts,
            cfg.charges,
            cuts,
            bulk_struct,
            surface,
            planes,
            surf_tol,
        )
    except Exception:
        virtual_sites = []

    used_sites: set[int] = set()
    placed: List[Tuple[str, np.ndarray]] = []
    existing_ligand_pts = [base_pts[i] for i, sym in enumerate(base_syms) if sym in ligand_symbols]
    min_ligand_spacing = 1.5

    for lig_sym, old_pos in orphan_records:
        q_lig = int(cfg.charges.get(lig_sym, 0))
        candidates = []
        for site_idx, site in enumerate(virtual_sites):
            if site_idx in used_sites:
                continue
            hosts = [int(h) for h in site.get("hosts", []) if 0 <= int(h) < len(base_syms)]
            if not hosts:
                continue
            if not any(int(cfg.charges.get(base_syms[h], 0)) * q_lig < 0 for h in hosts):
                continue
            raw_pos = site.get("pos")
            if raw_pos is None:
                continue
            pos = np.asarray(raw_pos, float)
            if pos.shape != (3,) or not np.all(np.isfinite(pos)):
                continue
            if existing_ligand_pts:
                dmin = float(np.min(np.linalg.norm(np.asarray(existing_ligand_pts) - pos, axis=1)))
                if dmin < min_ligand_spacing:
                    continue
            candidates.append((float(np.linalg.norm(pos - old_pos)), -int(site.get("multiplicity", 1)), site_idx, pos))
        if not candidates:
            continue
        _dist, _rank, site_idx, pos = min(candidates)
        used_sites.add(site_idx)
        placed.append((lig_sym, pos.copy()))
        existing_ligand_pts.append(pos.copy())

    if placed:
        base_syms = list(base_syms) + [sym for sym, _pos in placed]
        base_pts = np.vstack([base_pts, np.asarray([pos for _sym, pos in placed], float)])

    relocated = len(placed)
    return base_syms, np.asarray(base_pts, float), {
        "detected": len(orphan_records),
        "relocated": relocated,
        "removed": len(orphan_records) - relocated,
    }


def _filter_passivated_indices(
    indices: List[int],
    syms: List[str],
    pts: NDArray[np.float64],
    native_species: set[str],
    cutoff: float = 3.5
) -> List[int]:
    non_native_coords = [pts[j] for j, sym in enumerate(syms) if sym not in native_species]
    if not non_native_coords:
        return indices
    non_native_arr = np.asarray(non_native_coords, float)
    filtered = []
    for idx in indices:
        pos = pts[idx]
        dists = np.linalg.norm(non_native_arr - pos, axis=1)
        if np.min(dists) >= cutoff:
            filtered.append(idx)
    return filtered


def detect_z_type_displacement_options(
    syms: List[str],
    pts: NDArray[np.float64],
    cfg: Config,
    bulk_struct,
    planes: List[Plane],
    cif_path: Optional[str] = None,
) -> List[dict]:
    """Return charge-neutral Z-type options available on the current surface."""
    charges = cfg.charges
    pts = np.asarray(pts, float)
    surf_tol = getattr(cfg.passivation, "surf_tol", 2.0)
    surface = _surface_mask(pts, planes, surf_tol)
    native = _native_species(bulk_struct)
    passivation_ligands = {cfg.passivation.ligand}
    if cfg.passivation.cation_ligand:
        passivation_ligands.add(cfg.passivation.cation_ligand)

    species_present = sorted(set(syms))
    cations = [s for s in species_present if int(charges.get(s, 0)) > 0]
    anions = [s for s in species_present if int(charges.get(s, 0)) < 0]
    for lig in passivation_ligands:
        if lig and lig not in anions and int(charges.get(lig, 0)) < 0:
            anions.append(lig)

    options = []
    for cation in cations:
        for anion in anions:
            anion_count = _derive_anion_count(cation, anion, charges)
            if not anion_count:
                continue

            is_core_ion_pair = (cation in native) and (anion in native)

            cation_indices = _eligible_indices(
                syms, pts, cation, require_surface=True, surface=surface
            )
            if is_core_ion_pair:
                cation_indices = _filter_passivated_indices(cation_indices, syms, pts, native)
            if not cation_indices:
                continue

            require_anion_surface = anion in native and anion not in passivation_ligands
            anion_indices = _eligible_indices(
                syms, pts, anion, require_surface=require_anion_surface, surface=surface
            )
            if is_core_ion_pair:
                anion_indices = _filter_passivated_indices(anion_indices, syms, pts, native)

            if anion in passivation_ligands and anion_indices:
                try:
                    cuts = derive_pair_cuts_from_cif(cif_path, charges, safety=1.00) if cif_path else None
                except Exception:
                    cuts = None
                count = _count_bound_groups(
                    syms,
                    pts,
                    cation_indices,
                    anion_indices,
                    anion_count,
                    cuts,
                    allow_unbound_completion=True,
                )
            elif anion not in species_present and anion in passivation_ligands:
                count = len(cation_indices)
            else:
                count = _count_possible_groups(cation_indices, anion_indices, anion_count)
            if count <= 0:
                continue
            options.append({
                "label": _formula(cation, anion, anion_count),
                "formula": _formula(cation, anion, anion_count),
                "cation": cation,
                "anion": anion,
                "anion_count": int(anion_count),
                "candidate_count": int(count),
            })
    options.sort(key=lambda x: (x["cation"], x["anion"], x["anion_count"]))
    return options


def _select_groups(
    syms: List[str],
    pts: NDArray[np.float64],
    cation_indices: List[int],
    anion_indices: List[int],
    anion_count: int,
    ratio: float,
    target_count: int,
    distribution: str,
    seed: int,
    search_radius: float,
) -> List[Tuple[int, List[int]]]:
    if not cation_indices or len(anion_indices) < anion_count:
        return []

    cation_positions = np.asarray([pts[i] for i in cation_indices], float)
    possible = _count_possible_groups(cation_indices, anion_indices, anion_count)
    if int(target_count or 0) > 0:
        target = min(int(target_count), possible)
    else:
        target = int(round(float(ratio) * possible))
    target = min(target, possible)
    if target <= 0:
        return []

    selected_local = _subsample_sites(cation_positions, min(1.0, target / max(1, len(cation_indices))), distribution, seed)
    if len(selected_local) < len(cation_indices):
        remaining = [i for i in range(len(cation_indices)) if i not in set(selected_local)]
        selected_local = list(selected_local) + remaining

    anion_positions = np.asarray([pts[i] for i in anion_indices], float)
    from scipy.spatial import cKDTree

    tree = cKDTree(anion_positions)
    available_anions = set(anion_indices)
    groups: List[Tuple[int, List[int]]] = []
    for local_idx in selected_local:
        if len(groups) >= target:
            break
        ci = cation_indices[int(local_idx)]

        chosen: List[int] = []
        k = min(len(anion_indices), max(int(anion_count) * 4, 16))
        while k <= len(anion_indices):
            dists, local_anion_ids = tree.query(pts[ci], k=k)
            dists = np.atleast_1d(dists)
            local_anion_ids = np.atleast_1d(local_anion_ids)
            candidates = []
            for dist, local_ai in zip(dists, local_anion_ids):
                if not np.isfinite(dist):
                    continue
                ai = anion_indices[int(local_ai)]
                if ai not in available_anions:
                    continue
                candidates.append((float(dist), ai))
            close = [ai for dist, ai in candidates if dist <= search_radius]
            pool = close if len(close) >= anion_count else [ai for _dist, ai in candidates]
            chosen = pool[:anion_count]
            if len(chosen) >= anion_count or k == len(anion_indices):
                break
            k = min(len(anion_indices), k * 2)
        if len(chosen) < anion_count:
            continue
        groups.append((ci, list(chosen)))
        available_anions.difference_update(chosen)
    return groups


def run_z_type_displacement_posttreatment(
    syms: List[str],
    pts: NDArray[np.float64],
    cfg: Config,
    bulk_struct,
    planes: List[Plane],
    cif_path: str,
) -> Tuple[List[str], NDArray[np.float64], List[dict]]:
    spec: ZTypeDisplacementPostTreatSpec = getattr(
        getattr(cfg, "post_treatment", None),
        "z_type_displacement",
        ZTypeDisplacementPostTreatSpec(),
    )
    if not spec.enabled or not spec.passes:
        return syms, pts, []

    print("\n[post-treatment] ── Z-type displacement ─────────────────────────────")
    random.seed(spec.seed)
    np.random.seed(spec.seed)

    cuts = None
    try:
        cuts = derive_pair_cuts_from_cif(cif_path, cfg.charges, safety=1.00)
    except Exception:
        cuts = None

    work_syms = list(syms)
    work_pts = np.asarray(pts, float).copy()
    native = _native_species(bulk_struct)
    passivation_ligands = {cfg.passivation.ligand}
    if cfg.passivation.cation_ligand:
        passivation_ligands.add(cfg.passivation.cation_ligand)
    ledger: List[dict] = []

    for pass_idx, pass_spec in enumerate(spec.passes):
        cation = pass_spec.cation
        anion = pass_spec.anion
        anion_count = int(pass_spec.anion_count or 0)
        if anion_count <= 0:
            derived = _derive_anion_count(cation, anion, cfg.charges)
            if not derived:
                print(f"  [warning] Cannot derive neutral formula for {cation}/{anion}; skipping.")
                continue
            anion_count = derived
        formula = _formula(cation, anion, anion_count)
        print(
            f"\n[z-type:pass-{pass_idx + 1}] formula={formula} ratio={pass_spec.ratio:.2f} "
            f"dist={pass_spec.distribution}"
        )

        is_core_ion_pair = (cation in native) and (anion in native)

        surface = _surface_mask(work_pts, planes, getattr(cfg.passivation, "surf_tol", 2.0))
        cation_indices = _eligible_indices(
            work_syms, work_pts, cation, require_surface=True, surface=surface
        )
        if is_core_ion_pair:
            cation_indices = _filter_passivated_indices(cation_indices, work_syms, work_pts, native)
        require_anion_surface = anion in native and anion not in passivation_ligands
        anion_indices = _eligible_indices(
            work_syms, work_pts, anion, require_surface=require_anion_surface, surface=surface
        )
        if is_core_ion_pair:
            anion_indices = _filter_passivated_indices(anion_indices, work_syms, work_pts, native)
        possible = _count_possible_groups(cation_indices, anion_indices, anion_count)
        print(
            f"  → Candidates: {possible} groups from {len(cation_indices)} surface {cation} "
            f"and {len(anion_indices)} eligible {anion}"
        )
        if possible <= 0:
            continue

        groups = _select_groups(
            work_syms,
            work_pts,
            cation_indices,
            anion_indices,
            anion_count,
            pass_spec.ratio,
            pass_spec.target_count,
            pass_spec.distribution,
            spec.seed + pass_idx,
            _anion_search_radius(cation, anion, cuts),
        )
        if not groups:
            print("  → No non-conflicting groups selected.")
            continue

        remove_set = set()
        for ci, ais in groups:
            remove_set.add(ci)
            remove_set.update(ais)

        keep = [i for i in range(len(work_syms)) if i not in remove_set]
        work_syms = [work_syms[i] for i in keep]
        work_pts = work_pts[keep]
        work_syms, work_pts, orphan_stats = _relocate_orphan_ligands(
            work_syms,
            work_pts,
            cfg,
            bulk_struct,
            planes,
            cuts,
            passivation_ligands,
        )
        ledger.append({
            "formula": formula,
            "cation": cation,
            "anion": anion,
            "anion_count": anion_count,
            "removed": len(groups),
            "removed_atoms": len(remove_set),
            "orphan_ligands_detected": orphan_stats["detected"],
            "orphan_ligands_relocated": orphan_stats["relocated"],
            "orphan_ligands_removed": orphan_stats["removed"],
        })
        print(
            f"  → Removed {len(groups)} {formula} group(s): "
            f"{cation}={len(groups)}, {anion}={len(groups) * anion_count}"
        )
        if orphan_stats["detected"]:
            print(
                "  → Orphan ligand cleanup: "
                f"detected={orphan_stats['detected']}, "
                f"relocated={orphan_stats['relocated']}, "
                f"removed={orphan_stats['removed']}"
            )

    total = sum(int(entry.get("removed", 0)) for entry in ledger)
    print(f"[z-type:done] Total Z-type groups removed: {total}")
    return work_syms, np.asarray(work_pts, float), ledger
