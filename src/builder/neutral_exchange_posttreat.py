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
from .neutral_ligand_posttreat import (
    _atomic_number,
    _branch_heavy_count,
    _donor_antilone_pair_vec,
    _get_vdw,
    _prepare_neutral_ligand,
    _rdconf_to_numpy,
    _refine_one_ligand,
    _rotation_matrix_from_vectors,
    _rotations_about,
    _set_backbone_trans,
    _subsample_sites,
    _z_to_symbol,
)


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


def _group_candidates(
    pts: NDArray[np.float64],
    cation_indices: List[int],
    anion_indices: List[int],
    anion_count: int,
    search_radius: float,
) -> List[Tuple[int, List[int]]]:
    candidates = []
    for ci in cation_indices:
        c_pos = pts[ci]
        dists = np.linalg.norm(pts[anion_indices] - c_pos, axis=1)
        neighbors = [anion_indices[idx] for idx in np.argsort(dists) if dists[idx] <= search_radius]
        if len(neighbors) >= anion_count:
            candidates.append((ci, neighbors))
    return candidates


def _group_limit(n_candidates: int, ratio: float, target_count: int) -> int:
    limit = int(target_count) if int(target_count) > 0 else int(round(float(ratio) * n_candidates))
    return max(0, min(limit, n_candidates))


def _vacancy_candidates(
    pts: NDArray[np.float64],
    cation_indices: List[int],
    anion_indices: List[int],
    anion_count: int,
    ratio: float,
    target_count: int,
    distribution: str,
    seed: int,
    search_radius: float,
) -> Tuple[List[Tuple[int, List[int]]], int]:
    """
    Candidate groups in fill order, and how many to fill.

    The groups ``_select_groups`` would pick come first; the others follow
    in distribution order and replace groups that cannot be filled.
    """
    random.seed(seed)
    np.random.seed(seed)
    candidates = _group_candidates(pts, cation_indices, anion_indices, anion_count, search_radius)
    limit = _group_limit(len(candidates), ratio, target_count)
    if not candidates or limit <= 0:
        return [], 0
    site_pts = np.asarray([pts[ci] for ci, _neighbors in candidates], float)
    first = [int(i) for i in _subsample_sites(site_pts, min(1.0, limit / len(candidates)), distribution, seed)[:limit]]
    chosen = set(first)
    rest = [int(i) for i in _subsample_sites(site_pts, 1.0, distribution, seed) if int(i) not in chosen]
    return [candidates[i] for i in first + rest], limit


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

    candidates = _group_candidates(pts, cation_indices, anion_indices, anion_count, search_radius)
    if not candidates:
        return []

    limit = _group_limit(len(candidates), ratio, target_count)
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


_PHOSPHONATE_SHORTHAND = "COP(=O)OCC[NH3+]"
_PHOSPHONATE_ZWITTERION = "COP(=O)([O-])OCC[NH3+]"


def _zwitterion_validation_error(smiles: str, detail: str) -> ValueError:
    suggestion = ""
    if "".join(str(smiles).split()) == _PHOSPHONATE_SHORTHAND:
        suggestion = f" Did you mean {_PHOSPHONATE_ZWITTERION!r}?"
    return ValueError(f"Invalid zwitterion SMILES {smiles!r}: {detail}.{suggestion}")


def _validate_zwitterion_smiles(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise _zwitterion_validation_error(smiles, "RDKit could not parse the molecule")

    formal_charge = int(Chem.GetFormalCharge(mol))
    positive = [
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if atom.GetFormalCharge() > 0 and atom.GetAtomicNum() in {7, 15}
    ]
    negative = [
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if atom.GetFormalCharge() < 0 and atom.GetAtomicNum() in {8, 16}
    ]
    if formal_charge != 0:
        raise _zwitterion_validation_error(
            smiles,
            f"formal charge is {formal_charge:+d}; zwitterions require net charge 0",
        )
    if not positive or not negative:
        raise _zwitterion_validation_error(
            smiles,
            "zwitterions require explicit positive N/P and negative O/S centers",
        )
    return mol


def _detect_zwitterion_anchors(mol) -> Tuple[int, int]:
    positive = [
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if atom.GetFormalCharge() > 0 and atom.GetAtomicNum() in {7, 15}
    ]
    negative = [
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if atom.GetFormalCharge() < 0 and atom.GetAtomicNum() in {8, 16}
    ]
    if not positive:
        raise ValueError("No explicit cationic anchor (N+ or P+) found")
    if not negative:
        raise ValueError("No explicit anionic anchor (O- or S-) found")
    return positive[0], negative[0]


def _zwitterion_side_branch(mol, cat_idx: int, an_idx: int) -> Tuple[Optional[int], List[int]]:
    """Find the largest carbon-containing branch off the anchor-to-anchor spacer."""
    from rdkit import Chem

    path = list(Chem.rdmolops.GetShortestPath(mol, int(cat_idx), int(an_idx)))
    if len(path) < 2:
        return None, []
    path_set = set(path)
    center_idx = int(path[-2])
    candidates = []
    for nb in mol.GetAtomWithIdx(center_idx).GetNeighbors():
        start = int(nb.GetIdx())
        if start in path_set:
            continue
        seen = {center_idx, *path_set}
        queue = [start]
        atoms = []
        while queue:
            cur = queue.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            atom = mol.GetAtomWithIdx(cur)
            if atom.GetAtomicNum() > 1:
                atoms.append(cur)
            for next_atom in atom.GetNeighbors():
                next_idx = int(next_atom.GetIdx())
                if next_idx not in seen:
                    queue.append(next_idx)
        carbon_atoms = [
            idx for idx in atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
        ]
        if carbon_atoms:
            candidates.append((len(carbon_atoms), len(atoms), carbon_atoms))
    if not candidates:
        return None, []
    _carbon_count, _size, branch_atoms = max(candidates, key=lambda row: (row[0], row[1]))
    return center_idx, branch_atoms


def _cation_tail_atoms(mol, cat_idx: int, an_idx: int) -> List[int]:
    """Heavy atoms of the largest carbon branch on the cationic head, off the spacer."""
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
    if not branches:
        return []
    _c_count, _size, atoms = max(branches, key=lambda t: (t[0], t[1]))
    return atoms


def _tail_vector_from_cation(mol, coords: np.ndarray, cat_idx: int, an_idx: int) -> np.ndarray:
    """Return a vector toward the hydrophobic branch attached to the cationic head."""
    atoms = _cation_tail_atoms(mol, cat_idx, an_idx)
    if atoms:
        vec = coords[atoms].mean(axis=0) - coords[int(cat_idx)]
    else:
        heavy = [
            i for i, atom in enumerate(mol.GetAtoms())
            if atom.GetAtomicNum() > 1 and i not in {int(cat_idx), int(an_idx)}
        ]
        vec = coords[heavy].mean(axis=0) - coords[int(cat_idx)] if heavy else np.array([0.0, 0.0, 1.0])
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])


# Pose search for the molecules placed into displaced MX groups.  A pose is
# clean when no heavy pair is inside the vdW sum - _CLASH_MARGIN and no pair
# with H inside the vdW sum - _CLASH_H_MARGIN, the anchor group's contacts
# with the cations it binds excepted; tails stay within _MAX_TAIL_DEG of the
# local outward normal.
_CLASH_MARGIN = 0.4
_CLASH_H_MARGIN = 0.6
_MAX_TAIL_DEG = 60.0
_MAX_CATION_SHIFT = 1.5    # zwitterion N+/P+ within this of the displaced cation site
_ZW_ELEVATION_DEG = (0.0, 10.0, 20.0, 30.0)   # spacer axis lifted off the anion-to-cation line
_ZW_LATERAL_DEG = (-15.0, 0.0, 15.0)          # and swung sideways
_N_REFINE = 256


def _pose_sterics(
    poses: np.ndarray,
    numbers: np.ndarray,
    env_pos: np.ndarray,
    env_z: np.ndarray,
    atoms: Optional[np.ndarray] = None,
    exempt_lig=(),
    exempt_env=(),
    chunk: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hard contacts and clearance of each pose (n_pose, n_atom, 3).

    Clearance is the smallest distance - (vdW sum - margin) over the pairs in
    vdW reach (inf when none); a contact is a pair with negative clearance.
    Pairs of an ``exempt_lig`` atom with an ``exempt_env`` atom are bonds and
    are not counted.  ``atoms`` restricts the ligand atoms scored.
    """
    from scipy.spatial import cKDTree

    poses = np.asarray(poses, float)
    numbers = np.asarray(numbers, int)
    n_pose = poses.shape[0]
    atoms = np.arange(poses.shape[1]) if atoms is None else np.asarray(atoms, int)
    contacts = np.zeros(n_pose, int)
    clearance = np.full(n_pose, np.inf)
    env_pos = np.asarray(env_pos, float).reshape(-1, 3)
    env_z = np.asarray(env_z, int)
    if not len(env_pos) or not len(atoms) or not n_pose:
        return contacts, clearance
    lig_z = numbers[atoms]
    lig_vdw = np.array([_get_vdw(int(z)) for z in lig_z], float)
    env_vdw = np.array([_get_vdw(int(z)) for z in env_z], float)
    ex_lig = np.isin(atoms, np.asarray(list(exempt_lig), int))
    ex_env = np.zeros(len(env_pos), bool)
    ex_env[np.asarray(list(exempt_env), int)] = True
    reach = float(lig_vdw.max() + env_vdw.max())
    tree = cKDTree(env_pos)
    k = min(24, len(env_pos))
    for start in range(0, n_pose, chunk):
        sub = poses[start:start + chunk][:, atoms]
        n_p = sub.shape[0]
        dists, idx = tree.query(sub.reshape(-1, 3), k=k, distance_upper_bound=reach)
        dists = dists.reshape(n_p, len(atoms), k)
        idx = idx.reshape(n_p, len(atoms), k)
        found = idx < len(env_pos)
        safe = np.where(found, idx, 0)
        found &= ~(ex_lig[None, :, None] & ex_env[safe])
        with_h = (lig_z[None, :, None] == 1) | (env_z[safe] == 1)
        allowed = np.where(with_h, _CLASH_H_MARGIN, _CLASH_MARGIN)
        gap = np.where(found, dists - (lig_vdw[None, :, None] + env_vdw[safe]) + allowed, np.inf)
        contacts[start:start + n_p] = np.sum(gap < 0.0, axis=(1, 2))
        clearance[start:start + n_p] = gap.min(axis=(1, 2))
    return contacts, clearance


def _spacer_torsion_variants(mol, xyz: np.ndarray, path: List[int], max_torsions: int = 6) -> List[np.ndarray]:
    """
    Copies of ``xyz`` with the spacer torsions set to every gauche+/anti/gauche- combination.

    MMFF minima of a zwitterion are salt-bridged or anti about the bond to
    the cationic head; bridging two surface sites with the tail up needs
    the gauche ones too.  Bond lengths and angles are kept; self-clashing
    variants are dropped.
    """
    import itertools
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms

    torsions = []
    for k in range(len(path) - 1):
        bond = mol.GetBondBetweenAtoms(int(path[k]), int(path[k + 1]))
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        before = [a.GetIdx() for a in mol.GetAtomWithIdx(int(path[k])).GetNeighbors()
                  if a.GetAtomicNum() > 1 and a.GetIdx() != path[k + 1]]
        after = [a.GetIdx() for a in mol.GetAtomWithIdx(int(path[k + 1])).GetNeighbors()
                 if a.GetAtomicNum() > 1 and a.GetIdx() != path[k]]
        if not before or not after:
            continue
        a = int(path[k - 1]) if k > 0 else max(before, key=lambda i: _branch_heavy_count(mol, i, path[k]))
        d = int(path[k + 2]) if k + 2 < len(path) else max(after, key=lambda i: _branch_heavy_count(mol, i, path[k + 1]))
        torsions.append((a, int(path[k]), int(path[k + 1]), d))
    torsions = torsions[:int(max_torsions)]
    if not torsions:
        return []

    hops = Chem.GetDistanceMatrix(mol)
    numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], int)
    far = np.triu(hops >= 4, 1)
    heavy_pair = (numbers[:, None] > 1) & (numbers[None, :] > 1)
    min_d = np.where(heavy_pair, 2.9, 2.0)
    work = Chem.Mol(mol)
    conf = work.GetConformer()
    out = []
    for angles in itertools.product((60.0, 180.0, 300.0), repeat=len(torsions)):
        for i, pos in enumerate(xyz):
            conf.SetAtomPosition(i, pos.tolist())
        for (a, b, c, d), ang in zip(torsions, angles):
            rdMolTransforms.SetDihedralDeg(conf, a, b, c, d, ang)
        v = _rdconf_to_numpy(work)
        dist = np.linalg.norm(v[:, None] - v[None, :], axis=2)
        if not np.any(far & (dist < min_d)):
            out.append(v)
    return out


def _prepare_zwitterion(smiles: str, seed: int, n_confs: int = 30) -> dict:
    """Embed a zwitterion: ETKDG conformers, each also with its branches all-trans,
    plus the gauche/anti spacer-torsion variants of the first one.

    The branches off the cation-anion spacer are extended so tails stand
    up; the spacer torsions set the anchor distance and the tail direction.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from pymatgen.core import Element

    mol = Chem.AddHs(_validate_zwitterion_smiles(smiles))
    cat_idx, an_idx = _detect_zwitterion_anchors(mol)
    branch_center_idx, branch_atoms = _zwitterion_side_branch(mol, cat_idx, an_idx)
    path = [int(i) for i in Chem.rdmolops.GetShortestPath(mol, int(cat_idx), int(an_idx))]

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs), params=params))
    if not conf_ids:
        raise RuntimeError(f"3-D embedding failed for zwitterion SMILES: {smiles!r}")

    def optimize(m, cid=-1):
        try:
            if AllChem.MMFFHasAllMoleculeParams(m):
                AllChem.MMFFOptimizeMolecule(m, confId=int(cid), maxIters=300)
            else:
                AllChem.UFFOptimizeMolecule(m, confId=int(cid), maxIters=300)
        except Exception:
            pass

    confs = []
    for cid in conf_ids:
        optimize(mol, cid)
        confs.append(_rdconf_to_numpy(mol, cid))
        extended = Chem.Mol(mol, confId=int(cid))
        _set_backbone_trans(extended, -1, fixed=path)
        optimize(extended)
        confs.append(_rdconf_to_numpy(extended))
    confs.extend(_spacer_torsion_variants(mol, confs[1], path))

    tail_atoms = _cation_tail_atoms(mol, cat_idx, an_idx)
    tail_end = None
    if tail_atoms:
        hops = Chem.GetDistanceMatrix(mol)[int(cat_idx)]
        tail_end = int(max(tail_atoms, key=lambda i: hops[i]))

    return {
        "symbols": [Element.from_Z(atom.GetAtomicNum()).symbol for atom in mol.GetAtoms()],
        "numbers": np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], int),
        "cat_idx": int(cat_idx),
        "an_idx": int(an_idx),
        "an_group": _anchor_group(mol, an_idx),
        # Equivalent anionic atoms (carboxylate, phosphonate O): each may take the site.
        "an_anchors": [
            i for i in _anchor_group(mol, an_idx)
            if mol.GetAtomWithIdx(i).GetAtomicNum() == mol.GetAtomWithIdx(int(an_idx)).GetAtomicNum()
        ],
        "branch_center_idx": branch_center_idx,
        "branch_atoms": list(branch_atoms),
        "tail_atoms": tail_atoms,
        "tail_end": tail_end,
        "confs": np.asarray(confs, float),
    }


def _search_zwitterion_pose(
    prep: dict,
    ci_pos: np.ndarray,
    ai_pos: np.ndarray,
    n_surf: np.ndarray,
    env_pos: Optional[np.ndarray] = None,
    env_z: Optional[np.ndarray] = None,
    hosts=(),
) -> Tuple[Optional[np.ndarray], str]:
    """
    Best clean pose of a zwitterion bridging a displaced cation-anion pair.

    An anionic anchor atom sits on the displaced anion site and the cationic
    anchor lies along the spacer axis toward the displaced cation site,
    with the axis lifted up to 30 deg off the surface and swung +-15 deg
    sideways.  Spinning about that axis keeps both anchors in place.  Poses
    are kept only when the cationic anchor is within _MAX_CATION_SHIFT of
    its site, the cation's tail (its mean and its far end) stays
    within _MAX_TAIL_DEG of the outward normal with no atom below the
    cationic head, a side branch faces outward, and no atom clashes with
    the environment (the anionic group's contacts with ``hosts``
    excepted).  Among those, the
    upright, outward, roomy pose with the cation nearest its site wins.

    Returns (coords, "") or (None, reason).
    """
    ci_pos = np.asarray(ci_pos, float)
    ai_pos = np.asarray(ai_pos, float)
    n_surf = np.asarray(n_surf, float)
    n_surf = n_surf / (float(np.linalg.norm(n_surf)) + 1e-12)
    confs = np.asarray(prep["confs"], float)
    numbers = np.asarray(prep["numbers"], int)
    cat_idx, an_idx = int(prep["cat_idx"]), int(prep["an_idx"])
    tail_atoms = np.asarray(prep["tail_atoms"], int)
    branch_atoms = np.asarray(prep["branch_atoms"], int)
    branch_center = prep["branch_center_idx"]

    axis0 = ci_pos - ai_pos
    if float(np.linalg.norm(axis0)) < 1e-8:
        axis0 = n_surf
    axis0 = axis0 / float(np.linalg.norm(axis0))
    up = n_surf - np.dot(n_surf, axis0) * axis0
    if float(np.linalg.norm(up)) < 1e-6:
        up = np.cross(axis0, [1.0, 0.0, 0.0] if abs(axis0[0]) < 0.9 else [0.0, 1.0, 0.0])
    up = up / float(np.linalg.norm(up))
    side = np.cross(axis0, up)
    axes = []
    for elev in np.deg2rad(_ZW_ELEVATION_DEG):
        for lat in np.deg2rad(_ZW_LATERAL_DEG):
            axes.append(np.cos(elev) * (np.cos(lat) * axis0 + np.sin(lat) * side) + np.sin(elev) * up)
    spins = np.deg2rad(np.arange(0.0, 360.0, 15.0))

    # Conformers whose anchor distance lets the cation reach its site, with
    # each equivalent anionic atom on the anion site.
    site_d = float(np.linalg.norm(ci_pos - ai_pos))
    local = []
    for xyz in confs:
        for anchor in prep.get("an_anchors") or [an_idx]:
            src = xyz[cat_idx] - xyz[int(anchor)]
            if abs(float(np.linalg.norm(src)) - site_d) <= _MAX_CATION_SHIFT:
                local.append(xyz - xyz[int(anchor)])
    if not local:
        return None, "no conformer bridges both sites"
    local = np.asarray(local, float)
    rots = []
    for xyz in local:
        src = xyz[cat_idx]
        for ax in axes:
            rots.append(_rotations_about(ax, spins) @ _rotation_matrix_from_vectors(src, ax))
    rots = np.concatenate(rots)
    conf_of = np.repeat(np.arange(len(local)), len(axes) * len(spins))
    poses = np.einsum("pij,paj->pai", rots, local[conf_of]) + ai_pos

    # Geometric filters and soft preferences.
    shift = np.linalg.norm(poses[:, cat_idx] - ci_pos, axis=1)
    keep = shift <= _MAX_CATION_SHIFT
    soft = -0.5 * shift
    if len(tail_atoms):
        rel = poses[:, tail_atoms] - poses[:, cat_idx:cat_idx + 1]
        tail = rel.mean(axis=1)
        tail_cos = (tail @ n_surf) / (np.linalg.norm(tail, axis=1) + 1e-12)
        end = poses[:, int(prep["tail_end"])] - poses[:, cat_idx]
        end_cos = (end @ n_surf) / (np.linalg.norm(end, axis=1) + 1e-12)
        keep &= np.minimum(tail_cos, end_cos) >= np.cos(np.deg2rad(_MAX_TAIL_DEG))
        keep &= (rel @ n_surf).min(axis=1) >= -0.3
        soft += tail_cos
    if branch_center is not None and len(branch_atoms):
        bvec = poses[:, branch_atoms].mean(axis=1) - poses[:, int(branch_center)]
        branch_proj = (bvec @ n_surf) / (np.linalg.norm(bvec, axis=1) + 1e-12)
        keep &= branch_proj > 0.05
        soft += branch_proj
    if not keep.any():
        return None, "no pose with the tail along the outward normal"
    cand = np.flatnonzero(keep)

    env_pos = np.zeros((0, 3)) if env_pos is None else np.asarray(env_pos, float).reshape(-1, 3)
    env_z = np.zeros(0, int) if env_z is None else np.asarray(env_z, int)
    if len(env_pos):
        # Only the environment within reach of the molecule matters.
        radius = float(np.max(np.linalg.norm(local, axis=2))) + 7.0
        near = np.flatnonzero(np.sum((env_pos - ai_pos) ** 2, axis=1) <= radius ** 2)
        remap = {int(j): i for i, j in enumerate(near)}
        hosts = [remap[int(h)] for h in hosts if int(h) in remap]
        env_pos, env_z = env_pos[near], env_z[near]
    exempt = prep["an_group"]

    heavy = np.flatnonzero(numbers > 1)
    if len(cand) > _N_REFINE and len(heavy) < len(numbers):
        contacts, clear = _pose_sterics(poses[cand], numbers, env_pos, env_z, heavy, exempt, hosts)
        ok = contacts == 0
        if not ok.any():
            return None, "no clash-free pose"
        cand = cand[ok]
        rank = soft[cand] + np.minimum(clear[ok], 1.0)
        cand = cand[np.argsort(-rank, kind="stable")[:_N_REFINE]]
    contacts, clear = _pose_sterics(poses[cand], numbers, env_pos, env_z, None, exempt, hosts)
    ok = contacts == 0
    if not ok.any():
        return None, "no clash-free pose"
    cand, clear = cand[ok], clear[ok]
    best = int(cand[int(np.argmax(soft[cand] + np.minimum(clear, 1.0)))])
    return poses[best], ""


def _place_zwitterion(
    smiles: str,
    ci_pos: np.ndarray,
    ai_pos: np.ndarray,
    n_surf: np.ndarray,
    seed: int,
    env_syms: Optional[List[str]] = None,
    env_pts: Optional[np.ndarray] = None,
) -> Tuple[List[str], np.ndarray]:
    prep = _prepare_zwitterion(smiles, seed)
    env_z = np.array([_atomic_number(s) for s in (env_syms or [])], int)
    coords, reason = _search_zwitterion_pose(prep, ci_pos, ai_pos, n_surf, env_pts, env_z)
    if coords is None:
        raise RuntimeError(f"Could not place zwitterion {smiles!r}: {reason}")
    return list(prep["symbols"]), np.asarray(coords, float)


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

    # A plain formula (ZnCl2, CdBr2): element symbols and counts only. Anything
    # else - including neutral SMILES such as CC(=O)O - goes to RDKit, which
    # requires charged fragments, instead of being read letter by letter.
    if re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", text):
        parts = re.findall(r"([A-Z][a-z]?)(\d*)", text)
        if len(parts) < 2:
            raise ValueError(f"Cannot parse MXn formula: {text!r}")
        cat_sym, cat_count_raw = parts[0]
        cat_count = int(cat_count_raw or "1")
        if cat_count != 1:
            raise ValueError(f"MXn formula currently expects one cation, got {text!r}")
        if _charge_for_symbol(cat_sym, charges) <= 0 or any(_charge_for_symbol(sym, charges) >= 0 for sym, _n in parts[1:]):
            raise ValueError(
                f"MXn formula {text!r} must be a cation followed by anions with charges set "
                f"(e.g. ZnCl2); for a molecular salt give an ionic SMILES such as "
                f"[Zn+2].CC(=O)[O-].CC(=O)[O-]"
            )
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


def _detect_exchange_anchor(mol, coords: Optional[np.ndarray] = None) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Detect the donor atom (S, O, N, P) as the anchor for neutral exchange.
    coords (default: the molecule's conformer) gives the geometry.
    Returns: anchor_idx, anchor_pos, anchor_vec_to_body.
    """
    coords = _rdconf_to_numpy(mol) if coords is None else np.asarray(coords, float)
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


def _exchange_anchor_frame(mol, coords: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, str]:
    """Anchor frame of an L-type exchange ligand: the donor atom and its anti-lone-pair axis."""
    donor_idx, _pos, _body = _detect_exchange_anchor(mol, coords)
    return donor_idx, coords[donor_idx], _donor_antilone_pair_vec(mol, coords, donor_idx), "exchange_donor"


def _anchor_group(mol, anchor_idx: int) -> List[int]:
    """The anchor and the terminal O/S atoms on its heavy neighbour (carboxyl, phosphonate, sulfonate O)."""
    group = {int(anchor_idx)}
    for center in mol.GetAtomWithIdx(int(anchor_idx)).GetNeighbors():
        if center.GetAtomicNum() > 1:
            for nb in center.GetNeighbors():
                if nb.GetAtomicNum() in {8, 16} and nb.GetDegree() == 1:
                    group.add(int(nb.GetIdx()))
    return sorted(group)


class _LTypePoseArgs:
    """Pose grid and margins for _refine_one_ligand (as the neutral-ligand pass)."""
    sterics_mode = "vdw"
    coarse_step_deg = 15.0
    tilt_deg = (15.0, 30.0, 45.0)        # tails stay within _MAX_TAIL_DEG of the normal
    azimuth_step_deg = 30.0
    adaptive_offset_steps = 1
    adaptive_offset_step = 0.3
    sterics_margin = _CLASH_MARGIN
    h_margin = _CLASH_H_MARGIN
    donor_dev_free_deg = 25.0
    max_donor_dev_deg = 60.0
    donor_weight = 0.02
    tilt_weight = 0.01
    neighbor_repulsion = 0.5
    neighbor_radius = 10.0
    n_refine = _N_REFINE


def _local_outward_normal(pos: np.ndarray, core_pts: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Unit vector from the centroid of the core atoms within 8 A to ``pos``."""
    pos = np.asarray(pos, float)
    if len(core_pts):
        near = core_pts[np.sum((core_pts - pos) ** 2, axis=1) <= 64.0]
        if len(near):
            v = pos - near.mean(axis=0)
            if float(np.linalg.norm(v)) > 0.5:
                return v / float(np.linalg.norm(v))
    fallback = np.asarray(fallback, float)
    return fallback / (float(np.linalg.norm(fallback)) + 1e-12)


def _place_into_vacancies(
    syms: List[str],
    pts: NDArray[np.float64],
    candidates: List[Tuple[int, List[int]]],
    anion_count: int,
    limit: int,
    native: set,
    charges: Dict[str, int],
    cuts: Optional[PairCuts],
    place_one,
    env_radius: float,
    normal_at_anion: bool = False,
) -> Tuple[List[Tuple[int, List[int]]], List[str], List[np.ndarray], Dict[str, int]]:
    """
    Displace groups one at a time, in candidate order, and fill each vacancy.

    ``place_one(ci, ais, env_pos, env_z, hosts, n_surf)`` gets the atoms
    within ``env_radius`` of the displaced anion, with the group removed, plus
    every molecule placed so far (``hosts``: env indices of the cations
    bonded to the displaced anions; ``n_surf``: local outward normal at the
    displaced cation, or anion with ``normal_at_anion``) and returns
    (symbols, coords, "") or (None, None, reason).  A group without a clean
    pose keeps its atoms and the next candidate takes its place, until
    ``limit`` groups are filled.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(pts, float)
    z_all = np.array([_atomic_number(s) for s in syms], int)
    is_cation = np.array([_charge_for_symbol(s, charges) > 0 for s in syms], bool)
    is_native = np.array([(not native) or s in native for s in syms], bool)
    alive = np.ones(len(syms), bool)
    tree = cKDTree(pts)
    lig_syms: List[str] = []
    lig_pts: List[np.ndarray] = []
    lig_z: List[int] = []
    accepted: List[Tuple[int, List[int]]] = []
    skipped: Dict[str, int] = {}
    used_anions: set = set()
    for ci, neighbors in candidates:
        if len(accepted) >= limit:
            break
        ais = [int(ai) for ai in neighbors if int(ai) not in used_anions][:int(anion_count)]
        if len(ais) < int(anion_count):
            continue
        group = [int(ci), *ais]
        alive[group] = False
        site = pts[ais[0]]
        near = np.asarray(sorted(j for j in tree.query_ball_point(site, float(env_radius)) if alive[j]), int)
        env_pos = pts[near]
        env_z = z_all[near]
        env_index = {int(j): k for k, j in enumerate(near)}
        hosts = set()
        for ai in ais:
            for j in tree.query_ball_point(pts[ai], 6.0):
                if int(j) not in env_index or not is_cation[j]:
                    continue
                try:
                    cut = float(_pair_cut_calibrated(syms[j], syms[ai], cuts))
                except Exception:
                    cut = 3.2
                if float(np.linalg.norm(pts[j] - pts[ai])) <= max(3.2, 1.15 * cut):
                    hosts.add(env_index[int(j)])
        hosts = sorted(hosts)
        core_near = pts[near[is_native[near]]]
        at = pts[ais[0]] if normal_at_anion else pts[int(ci)]
        n_surf = _local_outward_normal(at, core_near, at - pts.mean(axis=0))
        if lig_pts:
            env_pos = np.vstack([env_pos, np.asarray(lig_pts, float)])
            env_z = np.concatenate([env_z, np.asarray(lig_z, int)])
        mol_syms, mol_pts, reason = place_one(int(ci), ais, env_pos, env_z, hosts, n_surf)
        if mol_syms is None:
            alive[group] = True
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        accepted.append((int(ci), ais))
        used_anions.update(ais)
        for sym, pos in zip(mol_syms, mol_pts):
            lig_syms.append(sym)
            lig_pts.append(np.asarray(pos, float))
            lig_z.append(_atomic_number(sym))
    return accepted, lig_syms, lig_pts, skipped


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
        smiles = pass_spec.smiles
        print(
            f"\n[neutral-exchange:pass-{pass_idx + 1}] formula={formula} ratio={pass_spec.ratio:.2f} "
            f"type={exchange_type} smiles={smiles!r}"
        )
        if exchange_type == "zwitterion":
            _validate_zwitterion_smiles(smiles)

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

        if exchange_type in {"zwitterion", "l_type"}:
            # One molecule per vacancy, placed against the dot and the
            # molecules already placed; a group with no clean pose is left in
            # place, so the charge stays balanced, and the next one is tried.
            candidates, limit = _vacancy_candidates(
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
            if not candidates:
                print("  → No non-conflicting groups selected.")
                continue
            q_before_pass = _total_formal_q(work_syms, cfg.charges)
            label = "Zwitterion" if exchange_type == "zwitterion" else "L-type"
            try:
                if exchange_type == "zwitterion":
                    zw_prep = _prepare_zwitterion(smiles, spec.seed + pass_idx)
                    an_idx = zw_prep["an_idx"]
                    env_radius = float(np.max(np.linalg.norm(
                        zw_prep["confs"] - zw_prep["confs"][:, an_idx:an_idx + 1], axis=2
                    ))) + 7.0

                    def place_one(ci, ais, env_pos, env_z, hosts, n_surf):
                        coords, reason = _search_zwitterion_pose(
                            zw_prep, work_pts[ci], work_pts[ais[0]], n_surf, env_pos, env_z, hosts
                        )
                        if coords is None:
                            return None, None, reason
                        return list(zw_prep["symbols"]), coords, ""
                else:
                    lig = _prepare_neutral_ligand(
                        smiles, seed=spec.seed + pass_idx, ff="uff", anchor_frame=_exchange_anchor_frame
                    )
                    lig_args = _LTypePoseArgs()
                    from rdkit import Chem
                    lig_group = _anchor_group(Chem.AddHs(Chem.MolFromSmiles(smiles)), lig["anchor_idx"])
                    lig_heavy = np.flatnonzero(np.asarray(lig["numbers"]) > 1)
                    env_radius = float(np.max(np.linalg.norm(lig["coords"] - lig["anchor_center"], axis=1))) + 7.0
                    intended = np.asarray([work_pts[ais[0]] for _ci, ais in candidates[:limit]], float)

                    def place_one(ci, ais, env_pos, env_z, hosts, n_surf):
                        site = work_pts[ais[0]]
                        n0 = n_surf
                        if hosts:
                            bonded = min(hosts, key=lambda h: float(np.linalg.norm(env_pos[h] - site)))
                            n0 = site - env_pos[bonded]
                        numbers, coords, contacts = _refine_one_ligand(
                            {
                                "dpos": site,
                                "n0": n0,
                                "n_surf": n_surf,
                                "numbers": lig["numbers"],
                                "coords": lig["coords"],
                                "anchor_idx": lig["anchor_idx"],
                                "anchor_center": lig["anchor_center"],
                                "anchor_vec": lig["anchor_vec"],
                                "tail_vec": lig["tail_vec"],
                                "hosts": hosts,
                                "anchor_group": lig_group,
                                "other_site_positions": intended,
                            },
                            env_pos,
                            env_z,
                            lig_args,
                        )
                        if contacts:
                            return None, None, "no clash-free pose"
                        far = coords[lig_heavy] - coords[lig["anchor_idx"]]
                        far = far[int(np.argmax(np.linalg.norm(far, axis=1)))]
                        if float(far @ n_surf) < np.cos(np.deg2rad(_MAX_TAIL_DEG)) * float(np.linalg.norm(far)):
                            return None, None, "no clash-free pose with the tail along the outward normal"
                        return [_z_to_symbol(int(z)) for z in numbers], coords, ""

                groups, lig_syms, lig_pts, skipped = _place_into_vacancies(
                    work_syms,
                    work_pts,
                    candidates,
                    anion_count,
                    limit,
                    native,
                    cfg.charges,
                    cuts,
                    place_one,
                    env_radius,
                    normal_at_anion=exchange_type == "l_type",
                )
            except Exception as exc:
                if exchange_type == "zwitterion":
                    raise RuntimeError(f"Zwitterion exchange failed: {exc}") from exc
                print(f"  [error] L-type exchange failed: {exc}")
                print("  → Neutral exchange placement failed; leaving this pass unchanged.")
                continue
            remove_set = {i for ci, ais in groups for i in (ci, *ais)}
            keep = [i for i in range(len(work_syms)) if i not in remove_set]
            new_syms = [work_syms[i] for i in keep] + lig_syms
            new_pts = list(work_pts[keep]) + lig_pts
            placed_charge_correction = _total_formal_q(lig_syms, cfg.charges)
            print(f"  → {label} exchange: placed {len(groups)} of {limit} molecules on displaced sites")
            for reason, count in skipped.items():
                print(f"  → Left {count} group(s) in place: {reason}")
            if len(groups) < limit:
                print(
                    f"  [warning] Placed {len(groups)} of {limit} requested {label} molecules: "
                    "the remaining groups have no clash-free pose"
                )
            branch_success = len(groups) > 0
            preferred_removal_anions: List[str] = []
            mxn_atomic_surplus_ligand: Optional[str] = None
        else:
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

            branch_success = False
            placed_charge_correction = 0
            preferred_removal_anions = []
            mxn_atomic_surplus_ligand = None

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
