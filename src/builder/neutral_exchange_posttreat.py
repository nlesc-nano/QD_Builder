from __future__ import annotations

import dataclasses
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
from .nc_types import Config, Plane, NeutralExchangePostTreatSpec, NeutralExchangePass
from .neutral_ligand_posttreat import _get_vdw, _rdconf_to_numpy, _smiles_to_3d_mol, _subsample_sites


_COMMON_FORMAL_CHARGES = {
    "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Ag": 1,
    "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Zn": 2, "Cd": 2, "Hg": 2, "Pb": 2,
    "Al": 3, "Ga": 3, "In": 3, "Bi": 3,
    "F": -1, "Cl": -1, "Br": -1, "I": -1,
    "O": -2, "S": -2, "Se": -2, "Te": -2,
    "N": -3, "P": -3, "As": -3, "Sb": -3,
}


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
    random.seed(seed)
    np.random.seed(seed)

    candidates = []
    for ci in cation_indices:
        c_pos = pts[ci]
        dists = np.linalg.norm(pts[anion_indices] - c_pos, axis=1)
        neighbors = [anion_indices[idx] for idx in np.argsort(dists) if dists[idx] <= search_radius]
        if len(neighbors) >= anion_count:
            candidates.append((ci, neighbors))

    if not candidates:
        return []

    limit = int(target_count) if int(target_count) > 0 else int(round(float(ratio) * len(candidates)))
    limit = max(0, min(limit, len(candidates)))
    if limit <= 0:
        return []

    ratio_eff = min(1.0, limit / max(1, len(candidates)))
    ordered_indices = _subsample_sites(
        np.asarray([pts[ci] for ci, _neighbors in candidates], float),
        ratio_eff,
        distribution,
        seed,
    )
    ordered_set = set(int(i) for i in ordered_indices)
    ordered_indices = list(ordered_indices[:limit])
    if len(ordered_indices) < len(candidates):
        ordered_indices.extend(i for i in range(len(candidates)) if i not in ordered_set)

    groups = []
    available_anions = set(anion_indices)
    for idx in ordered_indices:
        if len(groups) >= limit:
            break
        ci, neighbors = candidates[idx]
        chosen = []
        for ai in neighbors:
            if ai in available_anions:
                chosen.append(ai)
                if len(chosen) == anion_count:
                    break
        if len(chosen) < anion_count:
            continue
        groups.append((ci, chosen))
        available_anions.difference_update(chosen)
    return groups


def _bound_anion_indices(
    syms: List[str],
    pts: NDArray[np.float64],
    ci: int,
    anion_indices: List[int],
    cuts: Optional[PairCuts],
) -> List[int]:
    cation = syms[int(ci)]
    c_pos = pts[int(ci)]
    neighbors = []
    for ai in anion_indices:
        anion = syms[int(ai)]
        try:
            cutoff = _pair_cut_calibrated(cation, anion, cuts)
        except Exception:
            cutoff = 3.2
        dist = float(np.linalg.norm(pts[int(ai)] - c_pos))
        if dist <= max(3.2, 1.15 * cutoff):
            neighbors.append((dist, int(ai)))
    neighbors.sort()
    return [idx for _dist, idx in neighbors]


def _select_bound_groups(
    syms: List[str],
    pts: NDArray[np.float64],
    cation_indices: List[int],
    anion_indices: List[int],
    anion_count: int,
    ratio: float,
    target_count: int,
    distribution: str,
    seed: int,
    cuts: Optional[PairCuts],
    *,
    exact_bound_count: bool,
    min_bound_count: int = 1,
    allow_unbound_completion: bool = False,
) -> List[Tuple[int, List[int]]]:
    candidates = []
    requested = max(1, int(anion_count))
    minimum = max(1, min(int(min_bound_count), requested))
    for ci in cation_indices:
        bound = _bound_anion_indices(syms, pts, ci, anion_indices, cuts)
        if exact_bound_count and len(bound) != requested:
            continue
        if len(bound) >= minimum:
            chosen = list(bound[:min(len(bound), requested)])
            if allow_unbound_completion and len(chosen) < requested:
                chosen_set = set(chosen)
                c_pos = pts[int(ci)]
                extras = sorted(
                    (
                        (float(np.linalg.norm(pts[int(ai)] - c_pos)), int(ai))
                        for ai in anion_indices
                        if int(ai) not in chosen_set
                    ),
                    key=lambda item: item[0],
                )
                for _dist, ai in extras:
                    chosen.append(ai)
                    if len(chosen) == requested:
                        break
            if allow_unbound_completion and len(chosen) < requested:
                continue
            candidates.append((ci, chosen))
    if not candidates:
        return []

    limit = int(target_count) if int(target_count) > 0 else int(round(float(ratio) * len(candidates)))
    limit = max(0, min(limit, len(candidates)))
    if limit <= 0:
        return []

    ratio_eff = min(1.0, limit / max(1, len(candidates)))
    ordered_indices = _subsample_sites(
        np.asarray([pts[ci] for ci, _bound in candidates], float),
        ratio_eff,
        distribution,
        seed,
    )[:limit]

    groups = []
    used_anions = set()
    for idx in ordered_indices:
        ci, bound = candidates[int(idx)]
        if any(ai in used_anions for ai in bound):
            continue
        groups.append((ci, list(bound)))
        used_anions.update(bound)
        if len(groups) >= limit:
            break
    return groups


def _rotation_about_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        return np.eye(3)
    ux, uy, uz = axis / norm
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    K = np.array([
        [0.0, -uz, uy],
        [uz, 0.0, -ux],
        [-uy, ux, 0.0],
    ])
    return np.eye(3) + sin_t * K + (1.0 - cos_t) * (K @ K)


def _hard_clash_count(
    symbols: List[str],
    coords: np.ndarray,
    env_syms: List[str],
    env_pts: np.ndarray,
    *,
    margin: float = 0.65,
) -> int:
    if len(coords) == 0 or len(env_pts) == 0:
        return 0
    from pymatgen.core import Element
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(env_pts, float))
    clashes = 0
    for sym, pos in zip(symbols, coords):
        try:
            z = Element(sym).Z
        except Exception:
            z = 6
        r_i = _get_vdw(z)
        hits = tree.query_ball_point(np.asarray(pos, float), r=r_i + 2.2)
        for h in hits:
            env_sym = env_syms[int(h)]
            try:
                z_j = Element(env_sym).Z
            except Exception:
                z_j = 6
            threshold = max(0.75, r_i + _get_vdw(z_j) - margin)
            if float(np.linalg.norm(np.asarray(pos, float) - env_pts[int(h)])) < threshold:
                clashes += 1
    return clashes


def _rotation_matrix_aligning_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the rotation matrix that rotates vector a to align with vector b."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    a = a / norm_a if norm_a > 1e-8 else np.array([0.0, 0.0, 1.0])
    b = b / norm_b if norm_b > 1e-8 else np.array([0.0, 0.0, 1.0])
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        else:
            orth = np.array([1.0, 0.0, 0.0])
            if abs(a[0]) > 0.9:
                orth = np.array([0.0, 1.0, 0.0])
            orth = np.cross(a, orth)
            norm_orth = np.linalg.norm(orth)
            orth = orth / norm_orth if norm_orth > 1e-8 else np.array([0.0, 1.0, 0.0])
            K = np.array([
                [0, -orth[2], orth[1]],
                [orth[2], 0, -orth[0]],
                [-orth[1], orth[0], 0]
            ])
            return np.eye(3) + 2 * (K @ K)
    K = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return np.eye(3) + K + (K @ K) * ((1 - c) / (s ** 2))


def _prepare_anion_fragment(frag_mol, seed: int, ff: str) -> dict:
    from rdkit import Chem
    from .ligand_exchange_posttreat import _prepare_charged_ligand
    rw = Chem.RWMol(frag_mol)
    for atom in rw.GetAtoms():
        if atom.GetFormalCharge() < 0:
            atom.SetFormalCharge(0)
            if atom.GetAtomicNum() in {8, 16}: # O or S
                h_idx = rw.AddAtom(Chem.Atom(1))
                rw.AddBond(atom.GetIdx(), h_idx, Chem.BondType.SINGLE)
    neutral_mol = rw.GetMol()
    Chem.SanitizeMol(neutral_mol)
    neutral_smiles = Chem.MolToSmiles(neutral_mol)
    return _prepare_charged_ligand(neutral_smiles, -1, seed, ff)


def _place_single_anion_ligand(lig_prep: dict, host_pos: np.ndarray, ai_pos: np.ndarray, n_surf: np.ndarray) -> Tuple[List[str], np.ndarray]:
    from pymatgen.core import Element
    coords = np.asarray(lig_prep["coords"], float).copy()
    d1 = int(lig_prep["d1"])
    v_tail = np.asarray(lig_prep["v_tail"], float)
    
    v_target = ai_pos - host_pos
    norm = np.linalg.norm(v_target)
    n0 = v_target / norm if norm > 1e-8 else n_surf
    
    R = _rotation_matrix_aligning_vectors(v_tail, n0)
    coords = coords @ R.T
    coords = coords - coords[d1] + ai_pos
    
    symbols = [Element.from_Z(int(z)).symbol for z in lig_prep["numbers"]]
    return symbols, coords


def _detect_zwitterion_anchors(mol) -> Tuple[int, int]:
    cat_idx = None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() > 0 and atom.GetAtomicNum() in {7, 15}:
            cat_idx = int(atom.GetIdx())
            break
    if cat_idx is None:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in {7, 15}:
                cat_idx = int(atom.GetIdx())
                break
    if cat_idx is None:
        raise ValueError("No cationic anchor (N or P) found")

    an_idx = None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() < 0 and atom.GetAtomicNum() in {8, 16}:
            an_idx = int(atom.GetIdx())
            break
    if an_idx is None:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in {8, 16}:
                an_idx = int(atom.GetIdx())
                break
    if an_idx is None:
        raise ValueError("No anionic anchor (O or S) found")
    return cat_idx, an_idx


def _tail_vector_from_cation(mol, coords: np.ndarray, cat_idx: int, an_idx: int) -> np.ndarray:
    """Return a vector toward the hydrophobic branch attached to the cationic head."""
    seen_an = {int(cat_idx)}
    queue = [int(an_idx)]
    while queue:
        cur = queue.pop(0)
        seen_an.add(cur)
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            ni = int(nb.GetIdx())
            if ni not in seen_an:
                queue.append(ni)

    branches = []
    for nb in mol.GetAtomWithIdx(int(cat_idx)).GetNeighbors():
        ni = int(nb.GetIdx())
        if ni == an_idx or ni in seen_an:
            continue
        branch_atoms = []
        seen = {int(cat_idx), ni}
        queue = [ni]
        while queue:
            cur = queue.pop(0)
            if mol.GetAtomWithIdx(cur).GetAtomicNum() > 1:
                branch_atoms.append(cur)
            for nb2 in mol.GetAtomWithIdx(cur).GetNeighbors():
                n2 = int(nb2.GetIdx())
                if n2 not in seen:
                    seen.add(n2)
                    queue.append(n2)
        if branch_atoms:
            carbon_count = sum(1 for idx in branch_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6)
            branches.append((carbon_count, len(branch_atoms), branch_atoms))

    if branches:
        _c_count, _size, atoms = max(branches, key=lambda t: (t[0], t[1]))
        vec = coords[atoms].mean(axis=0) - coords[int(cat_idx)]
    else:
        heavy = [
            i for i, atom in enumerate(mol.GetAtoms())
            if atom.GetAtomicNum() > 1 and i not in {int(cat_idx), int(an_idx)}
        ]
        vec = coords[heavy].mean(axis=0) - coords[int(cat_idx)] if heavy else np.array([0.0, 0.0, 1.0])
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])


def _place_zwitterion(
    smiles: str,
    ci_pos: np.ndarray,
    ai_pos: np.ndarray,
    n_surf: np.ndarray,
    seed: int,
    env_syms: Optional[List[str]] = None,
    env_pts: Optional[np.ndarray] = None,
) -> Tuple[List[str], np.ndarray]:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from pymatgen.core import Element

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid zwitterion SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) < 0:
        raise RuntimeError(f"3-D embedding failed for zwitterion SMILES: {smiles!r}")
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=300)
    except Exception:
        pass

    cat_idx, an_idx = _detect_zwitterion_anchors(mol)
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    symbols = [Element.from_Z(atom.GetAtomicNum()).symbol for atom in mol.GetAtoms()]

    source_axis = coords[an_idx] - coords[cat_idx]
    target_axis = np.asarray(ai_pos, float) - np.asarray(ci_pos, float)
    if float(np.linalg.norm(target_axis)) < 1e-8:
        target_axis = np.asarray(n_surf, float)
    R = _rotation_matrix_aligning_vectors(source_axis, target_axis)
    base = (coords - coords[cat_idx]) @ R.T + np.asarray(ci_pos, float)

    axis = target_axis / (float(np.linalg.norm(target_axis)) + 1e-12)
    tail_vec = _tail_vector_from_cation(mol, coords, cat_idx, an_idx) @ R.T
    n_surf = np.asarray(n_surf, float)
    n_surf = n_surf / (float(np.linalg.norm(n_surf)) + 1e-12)
    target_tail = n_surf - np.dot(n_surf, axis) * axis
    source_tail = tail_vec - np.dot(tail_vec, axis) * axis
    if float(np.linalg.norm(target_tail)) < 1e-8:
        target_tail = n_surf
    if float(np.linalg.norm(source_tail)) < 1e-8:
        source_tail = tail_vec

    best_coords = base
    best_score = float("inf")
    env_syms = env_syms or []
    env_pts = np.asarray(env_pts, float) if env_pts is not None else np.zeros((0, 3), float)
    for deg in range(0, 360, 10):
        R_ax = _rotation_about_axis(axis, np.deg2rad(deg))
        test = (base - ci_pos) @ R_ax.T + ci_pos
        test_tail = source_tail @ R_ax.T
        align_penalty = -float(np.dot(
            test_tail / (float(np.linalg.norm(test_tail)) + 1e-12),
            target_tail / (float(np.linalg.norm(target_tail)) + 1e-12),
        ))
        clash_penalty = 100.0 * _hard_clash_count(symbols, test, env_syms, env_pts)
        score = clash_penalty + align_penalty
        if score < best_score:
            best_score = score
            best_coords = test

    final_coords = best_coords
    return symbols, final_coords


def _surface_outward_direction(pos: np.ndarray, planes: List[Plane]) -> np.ndarray:
    if not planes:
        return np.array([0.0, 0.0, 1.0])
    best_n = np.array([0.0, 0.0, 1.0])
    min_d = float("inf")
    for normal, d in planes:
        dist = abs(float(d) - np.dot(pos, normal))
        if dist < min_d:
            min_d = dist
            best_n = np.asarray(normal, float)
    return best_n


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


def _charge_for_symbol(sym: str, charges: Dict[str, int]) -> int:
    return int(charges.get(sym, _COMMON_FORMAL_CHARGES.get(sym, 0)))


def _parse_mxn_replacement(spec: str, charges: Dict[str, int]) -> dict:
    """Return fragment-aware MXn replacement data from a formula or ionic SMILES."""
    import re

    text = str(spec or "").strip()
    if not text:
        raise ValueError("MXn exchange requires a replacement formula or ionic SMILES")

    if "." not in text and not any(ch in text for ch in "[]+-"):
        parts = re.findall(r"([A-Z][a-z]?)(\d*)", text)
        if len(parts) < 2:
            raise ValueError(f"Cannot parse MXn formula: {text!r}")
        cat_sym, cat_count_raw = parts[0]
        cat_count = int(cat_count_raw or "1")
        if cat_count != 1:
            raise ValueError(f"MXn formula currently expects one cation, got {text!r}")
        anion_frags: List[dict] = []
        for sym, count_raw in parts[1:]:
            count = int(count_raw or "1")
            for _ in range(count):
                anion_frags.append({
                    "symbol": sym,
                    "charge": _charge_for_symbol(sym, charges),
                    "is_atomic": True,
                    "smiles": sym,
                    "mol": None,
                })
        cat_charge = _charge_for_symbol(cat_sym, charges)
        return {
            "cation_symbol": cat_sym,
            "cation_charge": cat_charge,
            "anion_fragments": anion_frags,
        }

    def _normalize_ionic_mxn_smiles(value: str) -> str:
        normalized = re.sub(r"\[([A-Z][a-z]?)(\d+)\+\]", r"[\1+\2]", value)

        def fix_bracketed_fragment(match: re.Match) -> str:
            frag = match.group(1)
            if re.fullmatch(r"[A-Z][a-z]?[+-]\d*", frag):
                return f"[{frag}]"
            if "O-" in frag and "[O-]" not in frag:
                frag = frag.replace("O-", "[O-]")
            if "S-" in frag and "[S-]" not in frag:
                frag = frag.replace("S-", "[S-]")
            return frag

        return re.sub(r"\[([A-Za-z0-9@+\-=#$()\\/]+)\]", fix_bracketed_fragment, normalized)

    from rdkit import Chem

    text = _normalize_ionic_mxn_smiles(text)

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"Invalid MXn replacement SMILES: {text!r}")
    frags = Chem.GetMolFrags(mol, asMols=True)
    cations = []
    anions = []
    for frag in frags:
        q = int(Chem.GetFormalCharge(frag))
        if q > 0:
            heavy = [a for a in frag.GetAtoms() if a.GetAtomicNum() > 1]
            if len(heavy) != 1 or frag.GetNumAtoms() != 1:
                raise ValueError(
                    "MXn exchange currently requires a single atomic cation fragment; "
                    "molecular cation fragments are not supported"
                )
            cations.append((heavy[0].GetSymbol(), q))
        elif q < 0:
            heavy = [a for a in frag.GetAtoms() if a.GetAtomicNum() > 1]
            if not heavy:
                raise ValueError(f"Anion fragment in MXn replacement has no heavy atom: {text!r}")
            anions.append({
                "symbol": heavy[0].GetSymbol(),
                "charge": q,
                "is_atomic": frag.GetNumAtoms() == 1,
                "smiles": Chem.MolToSmiles(frag),
                "mol": frag,
            })
    if not cations:
        raise ValueError(f"No cation fragment found in MXn replacement: {text!r}")
    if len(cations) != 1:
        raise ValueError(f"MXn exchange requires exactly one cation fragment, got {len(cations)}")
    cat_sym, cat_charge = cations[0]
    if not anions:
        raise ValueError(f"No anion fragments found in MXn replacement: {text!r}")
    return {
        "cation_symbol": cat_sym,
        "cation_charge": cat_charge,
        "anion_fragments": anions,
    }


def _total_formal_q(symbols: List[str], charges: Dict[str, int]) -> int:
    return int(sum(_charge_for_symbol(sym, charges) for sym in symbols))


def _rebalance_with_passivation_ligands(
    syms: List[str],
    pts: NDArray[np.float64],
    charges: Dict[str, int],
    ligand: str,
    q_target: int,
    *,
    distribution: str,
    seed: int,
) -> Tuple[List[str], NDArray[np.float64], int]:
    q_now = _total_formal_q(syms, charges)
    delta = int(q_now - q_target)
    q_lig = _charge_for_symbol(ligand, charges)
    if delta == 0 or q_lig == 0:
        return syms, pts, 0

    if delta < 0 and q_lig < 0:
        ligand_idx = [i for i, sym in enumerate(syms) if sym == ligand]
        if not ligand_idx:
            return syms, pts, 0
        needed = int((-delta + abs(q_lig) - 1) // abs(q_lig))
        remove_count = min(needed, len(ligand_idx))
        selected = _subsample_sites(
            np.asarray([pts[i] for i in ligand_idx], float),
            min(1.0, remove_count / max(1, len(ligand_idx))),
            distribution,
            seed,
        )[:remove_count]
        remove_set = {ligand_idx[i] for i in selected}
        keep = np.ones(len(syms), dtype=bool)
        for idx in remove_set:
            keep[idx] = False
        return [s for s, k in zip(syms, keep) if k], pts[keep], -len(remove_set)

    # Adding passivation ligands needs surface virtual-site machinery; leave to
    # the dedicated X-type rebalancer when a full passivation context is needed.
    return syms, pts, 0


def _remove_anions_for_charge(
    syms: List[str],
    pts: NDArray[np.float64],
    charges: Dict[str, int],
    needed_charge: int,
    *,
    preferred_symbols: List[str],
    fallback_symbols: List[str],
    distribution: str,
    seed: int,
) -> Tuple[List[str], NDArray[np.float64], int]:
    if needed_charge <= 0:
        return syms, pts, 0
    work_syms = list(syms)
    work_pts = np.asarray(pts, float).copy()
    removed = 0
    remaining = int(needed_charge)

    for sym in list(dict.fromkeys(preferred_symbols + fallback_symbols)):
        q = _charge_for_symbol(sym, charges)
        if q >= 0 or remaining <= 0:
            continue
        idxs = [i for i, s in enumerate(work_syms) if s == sym]
        if not idxs:
            continue
        count = min(len(idxs), int((remaining + abs(q) - 1) // abs(q)))
        selected_local = _subsample_sites(
            np.asarray([work_pts[i] for i in idxs], float),
            min(1.0, count / max(1, len(idxs))),
            distribution,
            seed + removed,
        )[:count]
        remove_set = {idxs[int(i)] for i in selected_local}
        keep = np.ones(len(work_syms), dtype=bool)
        for i in remove_set:
            keep[int(i)] = False
        removed += len(remove_set)
        remaining -= len(remove_set) * abs(q)
        work_syms = [s for s, k in zip(work_syms, keep) if k]
        work_pts = work_pts[keep]
        if remaining <= 0:
            break
    return work_syms, work_pts, removed


def _atomic_rebalance_with_mxn_ligand(
    syms: List[str],
    pts: NDArray[np.float64],
    cfg: Config,
    bulk_struct,
    planes: List[Plane],
    cif_path: str,
    ligand: str,
    residual_q: int,
    distribution: str,
) -> Tuple[List[str], NDArray[np.float64]]:
    from .ligand_exchange_posttreat import rebalance_ligand_exchange_charge

    passivation = dataclasses.replace(cfg.passivation, ligand=ligand)
    cfg_tmp = dataclasses.replace(cfg, passivation=passivation)
    ledger = [{
        "charge": 0,
        "ignored_element_charge": _total_formal_q(syms, cfg.charges) - int(residual_q),
        "distribution": distribution,
    }]
    return rebalance_ligand_exchange_charge(
        syms,
        pts,
        cfg_tmp,
        bulk_struct,
        planes,
        cif_path,
        ledger,
        verbose=False,
    )


def _add_atomic_anions_at_virtual_sites(
    syms: List[str],
    pts: NDArray[np.float64],
    cfg: Config,
    bulk_struct,
    planes: List[Plane],
    cif_path: str,
    ligand: str,
    residual_q: int,
    *,
    seed: int,
) -> Tuple[List[str], NDArray[np.float64], int]:
    q_lig = _charge_for_symbol(ligand, cfg.charges)
    if residual_q <= 0 or q_lig >= 0:
        return syms, pts, 0
    needed = int((residual_q + abs(q_lig) - 1) // abs(q_lig))
    if needed <= 0:
        return syms, pts, 0
    try:
        pair_cuts = derive_pair_cuts_from_cif(cif_path, cfg.charges, safety=1.00)
    except Exception:
        pair_cuts = None
    surf_tol = getattr(cfg.passivation, "surf_tol", 2.0)
    surface = _surface_mask(np.asarray(pts, float), planes, surf_tol)
    virtual_sites = compute_cif_virtual_sites(
        list(syms),
        np.asarray(pts, float),
        cfg.charges,
        pair_cuts,
        bulk_struct,
        surface,
        planes,
        surf_tol,
    )
    if not virtual_sites:
        return syms, pts, 0
    rng = random.Random(int(seed))
    ordered = sorted(
        list(virtual_sites),
        key=lambda rec: (-int(rec.get("multiplicity", 1)), rng.random()),
    )
    work_syms = list(syms)
    work_pts_list = [np.asarray(p, float) for p in np.asarray(pts, float)]
    added = 0
    for rec in ordered:
        pos = np.asarray(rec.get("pos"), float)
        if pos.shape != (3,):
            continue
        if work_pts_list:
            dmin = min(float(np.linalg.norm(pos - p)) for p in work_pts_list)
            if dmin < 0.8:
                continue
        work_syms.append(ligand)
        work_pts_list.append(pos)
        added += 1
        if added >= needed:
            break
    return work_syms, np.asarray(work_pts_list, float), added


def _detect_exchange_anchor(mol) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Detect the donor atom (S, O, N, P) as the anchor for neutral exchange.
    Returns: anchor_idx, anchor_pos, anchor_vec_to_body.
    """
    coords = _rdconf_to_numpy(mol)
    from rdkit import Chem

    def unit_v(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-8 else v

    patterns = [
        ("carboxylic_acid_o", Chem.MolFromSmarts("[CX3](=O)[OX2H1]"), 2),
        ("phosphonic_acid_o", Chem.MolFromSmarts("[PX4](=O)([OX2H1])[OX2H1,O-]"), 2),
        ("sulfonic_acid_o", Chem.MolFromSmarts("[#16X6](=O)(=O)[OX2H1]"), 3),
        ("thiol_s", Chem.MolFromSmarts("[SX2H1]"), 0),
        ("alcohol_o", Chem.MolFromSmarts("[OX2H1][#6]"), 0),
        ("quaternary_ammonium", Chem.MolFromSmarts("[#7X4+]"), 0),
        ("amine_n", Chem.MolFromSmarts("[NX3;H2,H1,H0;!$([NX3](=O))]"), 0),
        ("phosphine_p", Chem.MolFromSmarts("[PX3]"), 0),
        ("thioether_s", Chem.MolFromSmarts("[SX2;H0]"), 0),
    ]

    for name, patt, donor_match_idx in patterns:
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            continue

        match = matches[0]
        donor_idx = int(match[donor_match_idx])
        donor_pos = coords[donor_idx]
        numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], int)
        heavy_idx = [i for i, z in enumerate(numbers) if z > 1 and i != donor_idx]
        if heavy_idx:
            body_vec = unit_v(coords[heavy_idx].mean(axis=0) - donor_pos)
        else:
            neigh = [nb.GetIdx() for nb in mol.GetAtomWithIdx(donor_idx).GetNeighbors() if nb.GetAtomicNum() > 1]
            if neigh:
                body_vec = unit_v(coords[neigh[0]] - donor_pos)
            else:
                body_vec = np.array([0.0, 0.0, 1.0])

        return donor_idx, donor_pos, body_vec

    # Fallback to the first heavy atom, or atom 0
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], int)
    heavy = [i for i, z in enumerate(numbers) if z > 1]
    d_idx = heavy[0] if heavy else 0
    d_pos = coords[d_idx]
    heavy_rest = [i for i in heavy if i != d_idx]
    if heavy_rest:
        body_vec = unit_v(coords[heavy_rest].mean(axis=0) - d_pos)
    else:
        body_vec = np.array([0.0, 0.0, 1.0])
    return d_idx, d_pos, body_vec


def run_neutral_exchange_posttreatment(
    syms: List[str],
    pts: NDArray[np.float64],
    cfg: Config,
    bulk_struct,
    planes: List[Plane],
    cif_path: str,
) -> Tuple[List[str], NDArray[np.float64], List[dict]]:
    spec: NeutralExchangePostTreatSpec = getattr(
        getattr(cfg, "post_treatment", None),
        "neutral_exchange",
        NeutralExchangePostTreatSpec(),
    )
    if not spec.enabled or not spec.passes:
        return syms, pts, []

    print("\n[post-treatment] ── Neutral exchange ──────────────────────────────────")
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
        exchange_type = "mxn" if pass_spec.exchange_type == "salt" else pass_spec.exchange_type
        print(
            f"\n[neutral-exchange:pass-{pass_idx + 1}] formula={formula} ratio={pass_spec.ratio:.2f} "
            f"type={exchange_type} smiles={pass_spec.smiles!r}"
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
        bound_exchange = anion in passivation_ligands
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

        q_before_pass = _total_formal_q(work_syms, cfg.charges)
        remove_set = set()
        for ci, ais in groups:
            remove_set.add(ci)
            remove_set.update(ais)

        removed_coords = {}
        for old_idx in remove_set:
            removed_coords[old_idx] = work_pts[old_idx].copy()

        keep = [i for i in range(len(work_syms)) if i not in remove_set]
        new_syms = [work_syms[i] for i in keep]
        new_pts = list(work_pts[keep])

        smiles = pass_spec.smiles
        branch_success = False
        placed_charge_correction = 0
        preferred_removal_anions: List[str] = []
        mxn_atomic_surplus_ligand: Optional[str] = None

        if exchange_type == "mxn":
            try:
                mxn = _parse_mxn_replacement(smiles, cfg.charges)
                cat_sym = mxn["cation_symbol"]
                cat_charge = int(mxn["cation_charge"])
                anion_fragments = list(mxn["anion_fragments"])
                cfg.charges.setdefault(cat_sym, int(cat_charge))
                for frag in anion_fragments:
                    cfg.charges.setdefault(str(frag["symbol"]), int(frag["charge"]))
                preferred_removal_anions = [str(frag["symbol"]) for frag in anion_fragments]
                atomic_anions = [frag for frag in anion_fragments if bool(frag["is_atomic"])]
                if atomic_anions:
                    mxn_atomic_surplus_ligand = str(atomic_anions[0]["symbol"])
                prepared_molecular: Dict[str, dict] = {}

                def place_fragment(frag: dict, host_pos: np.ndarray, site_pos: np.ndarray, surf_norm: np.ndarray):
                    if bool(frag["is_atomic"]):
                        return [str(frag["symbol"])], np.asarray([site_pos], float)
                    key = str(frag["smiles"])
                    if key not in prepared_molecular:
                        prepared_molecular[key] = _prepare_anion_fragment(
                            frag["mol"],
                            seed=spec.seed + pass_idx,
                            ff="uff",
                        )
                    return _place_single_anion_ligand(
                        prepared_molecular[key],
                        np.asarray(host_pos, float),
                        np.asarray(site_pos, float),
                        np.asarray(surf_norm, float),
                    )

                for ci, ais in groups:
                    new_syms.append(cat_sym)
                    new_pts.append(removed_coords[ci])
                    surf_norm = _surface_outward_direction(removed_coords[ci], planes)
                    for j, ai in enumerate(ais[:len(anion_fragments)]):
                        frag = anion_fragments[j]
                        syms_frag, pts_frag = place_fragment(
                            frag,
                            removed_coords[ci],
                            removed_coords[ai],
                            surf_norm,
                        )
                        for sym_f, pos_f in zip(syms_frag, pts_frag):
                            new_syms.append(sym_f)
                            new_pts.append(np.asarray(pos_f, float))
                        if not bool(frag["is_atomic"]):
                            placed_charge_correction += (
                                _total_formal_q(syms_frag, cfg.charges)
                                - int(frag["charge"])
                            )
                print(
                    f"  → MXn exchange successful: placed {len(groups)} {cat_sym} "
                    f"cation(s) with {len(anion_fragments)} replacement anion fragment(s)"
                )
                branch_success = True
            except Exception as exc:
                print(f"  [error] MXn exchange failed: {exc}")
        
        elif exchange_type == "zwitterion":
            try:
                zw_placed = 0
                for ci, ais in groups:
                    if not ais:
                        continue
                    ci_pos = removed_coords[ci]
                    ai_pos = removed_coords[ais[0]]
                    surf_norm = _surface_outward_direction(ci_pos, planes)
                    syms_zw, coords_zw = _place_zwitterion(
                        smiles,
                        ci_pos,
                        ai_pos,
                        surf_norm,
                        seed=spec.seed + pass_idx,
                        env_syms=new_syms,
                        env_pts=np.asarray(new_pts, float),
                    )
                    for sym_z, pos_z in zip(syms_zw, coords_zw):
                        new_syms.append(sym_z)
                        new_pts.append(pos_z)
                    placed_charge_correction += _total_formal_q(syms_zw, cfg.charges)
                    zw_placed += 1
                print(f"  → Zwitterion exchange successful: placed {zw_placed} zwitterion molecules")
                branch_success = zw_placed > 0
            except Exception as exc:
                print(f"  [error] Zwitterion exchange failed: {exc}")
        
        elif exchange_type == "l_type":
            try:
                from pymatgen.core import Element
                
                l_placed = 0
                
                # Determine negative indices for closest anion reference
                negative_indices = [
                    j for j, sym in enumerate(work_syms)
                    if int(cfg.charges.get(sym, 0)) < 0
                ]
                
                # Filter out the displaced anions so we don't align to them!
                negative_indices = [j for j in negative_indices if j not in remove_set]
                
                for ci, ais in groups:
                    if not ais:
                        continue
                    # Target position is the position of the closest displaced Cl anion
                    target_pos = removed_coords[ais[0]]
                    ci_pos = removed_coords[ci]
                    surf_norm = _surface_outward_direction(ci_pos, planes)
                    
                    # Prepare ligand molecule
                    mol = _smiles_to_3d_mol(smiles, seed=spec.seed + pass_idx, ff="uff")
                    anchor_idx, anchor_pos, anchor_vec_to_body = _detect_exchange_anchor(mol)
                    
                    coords = _rdconf_to_numpy(mol).copy()
                    
                    # Align anchor_vec_to_body with surf_norm
                    R = _rotation_matrix_aligning_vectors(anchor_vec_to_body, surf_norm)
                    coords = coords @ R.T
                    coords = coords - coords[anchor_idx] + target_pos
                    
                    # Identify polar hydrogen(s) bound to anchor
                    anchor_atom = mol.GetAtomWithIdx(anchor_idx)
                    h_indices = []
                    for nb in anchor_atom.GetNeighbors():
                        if nb.GetAtomicNum() == 1:
                            h_indices.append(int(nb.GetIdx()))
                            
                    # Find closest remaining surface anion
                    closest_anion_pos = None
                    if negative_indices:
                        dists = [np.linalg.norm(work_pts[j] - target_pos) for j in negative_indices]
                        closest_idx = negative_indices[np.argmin(dists)]
                        closest_anion_pos = work_pts[closest_idx]
                        
                    # Optimize rotation around surf_norm to direct hydrogen to closest surface anion
                    if h_indices and closest_anion_pos is not None:
                        best_theta = 0.0
                        min_d = float("inf")
                        for deg in range(0, 360, 5):
                            rad = np.radians(deg)
                            cos_t = np.cos(rad)
                            sin_t = np.sin(rad)
                            ux, uy, uz = surf_norm
                            K = np.array([
                                [0.0, -uz, uy],
                                [uz, 0.0, -ux],
                                [-uy, ux, 0.0]
                            ])
                            R_rot = np.eye(3) + sin_t * K + (1.0 - cos_t) * (K @ K)
                            
                            test_coords = (coords - target_pos) @ R_rot.T + target_pos
                            h_pos = test_coords[h_indices[0]]
                            d = float(np.linalg.norm(h_pos - closest_anion_pos))
                            if d < min_d:
                                min_d = d
                                best_theta = rad
                                
                        cos_t = np.cos(best_theta)
                        sin_t = np.sin(best_theta)
                        ux, uy, uz = surf_norm
                        K = np.array([
                            [0.0, -uz, uy],
                            [uz, 0.0, -ux],
                            [-uy, ux, 0.0]
                        ])
                        R_best = np.eye(3) + sin_t * K + (1.0 - cos_t) * (K @ K)
                        coords = (coords - target_pos) @ R_best.T + target_pos
                        
                    symbols = [Element.from_Z(atom.GetAtomicNum()).symbol for atom in mol.GetAtoms()]
                    for sym_l, pos_l in zip(symbols, coords):
                        new_syms.append(sym_l)
                        new_pts.append(pos_l)
                    placed_charge_correction += _total_formal_q(symbols, cfg.charges)
                    l_placed += 1
                print(f"  → L-type exchange successful: placed {l_placed} L-type molecules directly on displaced sites")
                branch_success = l_placed > 0
            except Exception as exc:
                print(f"  [error] L-type exchange failed: {exc}")

        if not branch_success:
            print("  → Neutral exchange placement failed; leaving this pass unchanged.")
            continue

        work_syms = new_syms
        work_pts = np.asarray(new_pts, float)
        q_after_pass = _total_formal_q(work_syms, cfg.charges)
        if exchange_type in {"zwitterion", "l_type"}:
            residual_q = 0
        else:
            residual_q = int(q_after_pass - q_before_pass - placed_charge_correction)
        if residual_q != 0:
            try:
                before_rebalance = q_after_pass - placed_charge_correction
                if exchange_type == "mxn" and residual_q < 0:
                    work_syms, work_pts, removed = _remove_anions_for_charge(
                        work_syms,
                        work_pts,
                        cfg.charges,
                        -residual_q,
                        preferred_symbols=preferred_removal_anions,
                        fallback_symbols=list(passivation_ligands),
                        distribution=pass_spec.distribution,
                        seed=spec.seed + pass_idx + 4441,
                    )
                    if removed <= 0:
                        raise RuntimeError("no eligible anions available for negative residual compensation")
                elif exchange_type == "mxn" and preferred_removal_anions:
                    if not mxn_atomic_surplus_ligand:
                        print(
                            "  [warning] Positive MXn residual requires molecular anion passivation-site "
                            "placement, which is not available for this pass; skipping compensation."
                        )
                    else:
                        work_syms, work_pts = _atomic_rebalance_with_mxn_ligand(
                            work_syms,
                            work_pts,
                            cfg,
                            bulk_struct,
                            planes,
                            cif_path,
                            mxn_atomic_surplus_ligand,
                            residual_q,
                            pass_spec.distribution,
                        )
                        residual_after_iter = (
                            _total_formal_q(work_syms, cfg.charges)
                            - q_before_pass
                            - placed_charge_correction
                        )
                        if residual_after_iter > 0:
                            work_syms, work_pts, added = _add_atomic_anions_at_virtual_sites(
                                work_syms,
                                work_pts,
                                cfg,
                                bulk_struct,
                                planes,
                                cif_path,
                                mxn_atomic_surplus_ligand,
                                residual_after_iter,
                                seed=spec.seed + pass_idx + 8803,
                            )
                            if added <= 0:
                                raise RuntimeError(
                                    f"could not place surplus {mxn_atomic_surplus_ligand} at an eligible passivation site"
                                )
                else:
                    from .ligand_exchange_posttreat import rebalance_ligand_exchange_charge

                    charge_ledger = [{
                        "charge": 0,
                        "ignored_element_charge": q_before_pass + placed_charge_correction,
                        "distribution": pass_spec.distribution,
                    }]
                    work_syms, work_pts = rebalance_ligand_exchange_charge(
                        work_syms,
                        work_pts,
                        cfg,
                        bulk_struct,
                        planes,
                        cif_path,
                        charge_ledger,
                        verbose=False,
                    )
                after_rebalance = _total_formal_q(work_syms, cfg.charges) - placed_charge_correction
                print(
                    f"  → Charge compensation: "
                    f"Q {before_rebalance:+d}->{after_rebalance:+d} (target {q_before_pass:+d})"
                )
            except Exception as exc:
                print(f"  [warning] Neutral exchange charge compensation failed: {exc}")
        
        ledger.append({
            "formula": formula,
            "cation": cation,
            "anion": anion,
            "anion_count": anion_count,
            "exchange_type": exchange_type,
            "exchanged": len(groups),
            "removed_atoms": len(remove_set),
        })

    print(f"[neutral-exchange:done] Total neutral exchange passes completed.")
    return work_syms, np.asarray(work_pts, float), ledger
