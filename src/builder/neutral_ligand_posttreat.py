# src/builder/neutral_ligand_posttreat.py
"""
Neutral-ligand post-treatment for QD nanocrystals.

After charge-balance (Q=0) is reached, surface atoms may still be structurally
undercoordinated relative to their bulk coordination number (bulk_cn).  This
module attaches *neutral* organic/inorganic ligands (supplied as SMILES strings)
to those remaining open coordination sites.

The key novelty over miniCAT is that anchor positions are computed from the
crystal-geometry "missing bond" direction of each undercoordinated surface atom,
rather than from an explicit dummy atom in the structure.

Functions copied / adapted from miniCAT (minicat/main.py, minicat/functional_groups_class.py):
  – smiles_to_3d_mol, _refine_one_ligand_vectorized, rotation_matrix_from_vectors,
    unit, outward_normal, build_sterics_tree, vdw_radii_table
Origin: https://github.com/nlesc-nano/miniCAT (MIT licence).
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.distance import pdist, squareform

from .nc_types import Config, NeutralLigandPostTreatSpec, NeutralLigandPass
from .analysis import (
    coord_numbers_bipartite,
    bulk_cn_opposite_by_interior,
    _pair_cut_calibrated,
    PairCuts,
    _unit,
    _bulk_ideal_direction_sets,
    _actual_opposite_bond_vectors,
    _match_missing_ideal_dirs,
    _surface_outward_direction,
    compute_strict_missing_bond_vectors,
    _match_actual_to_ideal,
    compute_cif_virtual_sites,
)
from .nc_types import Plane

# Basic geometry helpers (from miniCAT)
# ──────────────────────────────────────────────────────────────────────────────


def _unit_rows(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=1)
    n[n < 1e-12] = 1.0
    return v / n[:, None]


def _rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3) if np.dot(a, b) > 0 else -np.eye(3)
    c = np.dot(a, b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


# VdW radii lookup (from miniCAT; fallback 2.0 Å)
_VDW_RADII: Dict[int, float] = {
    1: 1.2, 5: 1.8, 6: 1.7, 7: 1.55, 8: 1.52,
    9: 1.47, 14: 2.1, 15: 1.8, 16: 1.8, 17: 1.75,
    35: 1.85, 53: 1.98,
}
try:
    from rdkit.Chem import GetPeriodicTable as _rpt
    _pt = _rpt()
    for _z in range(1, 119):
        try:
            _VDW_RADII.setdefault(_z, float(_pt.GetRvdw(_z)))
        except Exception:
            pass
    del _pt, _rpt, _z
except Exception:
    pass


def _get_vdw(z: int) -> float:
    return _VDW_RADII.get(int(z), 2.0)


def _build_sterics_tree(atoms_pos: np.ndarray, atoms_z: np.ndarray,
                        exclude_mask: np.ndarray, mode: str):
    """Build a KDTree for steric clashes (adapted from miniCAT)."""
    keep = ~exclude_mask
    if mode == "heavy":
        keep &= (atoms_z > 1)
    coords = atoms_pos[keep]
    radii = np.array([_get_vdw(int(z)) for z in atoms_z[keep]])
    if len(coords) == 0:
        return cKDTree(np.zeros((1, 3))), np.array([2.0])
    return cKDTree(coords), radii


# ──────────────────────────────────────────────────────────────────────────────
# RDKit-based ligand preparation (adapted from miniCAT)
# ──────────────────────────────────────────────────────────────────────────────

def _smiles_to_3d_mol(smiles: str, seed: int = 1337, ff: str = "uff"):
    """Embed a SMILES into 3-D coordinates using RDKit (neutral form)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError(
            "Neutral ligand post-treatment requires RDKit. "
            "Install it with: conda install -c conda-forge rdkit"
        )
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) < 0:
        raise RuntimeError(f"3-D embedding failed for SMILES: {smiles!r}")
    if ff == "mmff" and AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    return mol


def _rdconf_to_numpy(mol) -> np.ndarray:
    """Extract conformer coordinates as (N, 3) float array."""
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    xyz = np.zeros((n, 3), float)
    for i in range(n):
        p = conf.GetAtomPosition(i)
        xyz[i] = [p.x, p.y, p.z]
    return xyz


def _find_attached_h(mol, heavy_idx: int) -> Optional[int]:
    atom = mol.GetAtomWithIdx(int(heavy_idx))
    for nb in atom.GetNeighbors():
        if nb.GetAtomicNum() == 1:
            return int(nb.GetIdx())
    return None


def _anchor_vec_to_body(mol, coords: np.ndarray, anchor_idx: int) -> np.ndarray:
    """Return a neutral-ligand vector from anchor toward the molecular body."""
    anchor_idx = int(anchor_idx)
    anchor = coords[anchor_idx]
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], int)
    heavy_idx = [i for i, z in enumerate(numbers) if z > 1 and i != anchor_idx]
    if heavy_idx:
        return _unit(coords[heavy_idx].mean(axis=0) - anchor)
    neigh = [nb.GetIdx() for nb in mol.GetAtomWithIdx(anchor_idx).GetNeighbors()]
    if neigh:
        return _unit(coords[neigh[0]] - anchor)
    return np.array([0.0, 0.0, 1.0])


def _detect_neutral_anchor(mol) -> Tuple[int, np.ndarray, np.ndarray, str]:
    """
    Detect a neutral functional group and return its anchor frame.

    The molecule is not ionized or deprotonated.  Acidic groups anchor through
    their neutral X-H proton; neutral donors anchor through the donor atom.
    """
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("Neutral ligand post-treatment requires RDKit.")

    coords = _rdconf_to_numpy(mol)

    patterns = [
        ("carboxylic_acid_h", Chem.MolFromSmarts("[CX3](=O)[OX2H1]")),
        ("phosphonic_acid_h", Chem.MolFromSmarts("[PX4](=O)([OX2H1])[OX2H1,O-]")),
        ("sulfonic_acid_h", Chem.MolFromSmarts("[#16X6](=O)(=O)[OX2H1]")),
        ("thiol_h", Chem.MolFromSmarts("[SX2H1]")),
        ("alcohol_h", Chem.MolFromSmarts("[OX2H1][#6]")),
        ("quaternary_ammonium", Chem.MolFromSmarts("[#7X4+]")),
        ("amine_n", Chem.MolFromSmarts("[NX3;H2,H1,H0;!$([NX3](=O))]")),
        ("phosphine_p", Chem.MolFromSmarts("[PX3]")),
        ("thioether_s", Chem.MolFromSmarts("[SX2;H0]")),
    ]

    for name, patt in patterns:
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            continue

        match = matches[0]
        if name in {"carboxylic_acid_h"}:
            o_idx = int(match[2])
            h_idx = _find_attached_h(mol, o_idx)
            if h_idx is not None:
                return h_idx, coords[h_idx], _unit(coords[o_idx] - coords[h_idx]), name
        if name in {"phosphonic_acid_h", "sulfonic_acid_h"}:
            for idx in match:
                atom = mol.GetAtomWithIdx(int(idx))
                if atom.GetAtomicNum() == 8:
                    h_idx = _find_attached_h(mol, int(idx))
                    if h_idx is not None:
                        return h_idx, coords[h_idx], _unit(coords[int(idx)] - coords[h_idx]), name
        if name == "thiol_h":
            s_idx = int(match[0])
            h_idx = _find_attached_h(mol, s_idx)
            if h_idx is not None:
                return h_idx, coords[h_idx], _unit(coords[s_idx] - coords[h_idx]), name
        if name == "alcohol_h":
            o_idx = int(match[0])
            h_idx = _find_attached_h(mol, o_idx)
            if h_idx is not None:
                return h_idx, coords[h_idx], _unit(coords[o_idx] - coords[h_idx]), name

        anchor_idx = int(match[0])
        return anchor_idx, coords[anchor_idx], _anchor_vec_to_body(mol, coords, anchor_idx), name

    anchor_idx = _detect_anchor_atom(mol)
    return anchor_idx, coords[anchor_idx], _anchor_vec_to_body(mol, coords, anchor_idx), "electronegative_atom"


def _detect_anchor_atom(mol, direction_hint: Optional[np.ndarray] = None) -> int:
    """
    Find the best anchor atom in a neutral ligand.

    Strategy (in order):
      1. Most electronegative heavy atom: O > N > S > P > halogens > C.
      2. If tied, pick the one whose position is most "inward" along
         direction_hint (i.e. closest to the surface atom).

    Returns the atom index in the RDKit mol (with Hs).
    """
    ELECTRONEGATIVITY_ORDER = {8: 0, 7: 1, 16: 2, 15: 3, 9: 4, 17: 5, 35: 6, 53: 7}
    coords = _rdconf_to_numpy(mol)
    atoms = list(mol.GetAtoms())
    best_idx = None
    best_score = (999, 0.0)
    for i, atom in enumerate(atoms):
        z = atom.GetAtomicNum()
        if z <= 1:
            continue  # skip H and dummy
        en_rank = ELECTRONEGATIVITY_ORDER.get(z, 8)
        # secondary: if direction_hint given, prefer atom closer to surface
        if direction_hint is not None:
            proj = float(np.dot(coords[i], _unit(direction_hint)))
        else:
            proj = 0.0
        score = (en_rank, -proj)   # lower en_rank = better; lower proj = closer to surface
        if score < best_score:
            best_score = score
            best_idx = i
    if best_idx is None:
        # All atoms are H (degenerate case) — just pick atom 0
        best_idx = 0
    return best_idx


# ──────────────────────────────────────────────────────────────────────────────
# Spatial distribution assignment (adapted from miniCAT)
# ──────────────────────────────────────────────────────────────────────────────

def _get_spatially_ordered_indices(dummy_pos: np.ndarray, maximize_dist: bool) -> List[int]:
    num_sites = dummy_pos.shape[0]
    if num_sites == 0:
        return []
    proximity_matrix = np.exp(-squareform(pdist(dummy_pos)))
    np.fill_diagonal(proximity_matrix, 0)
    available_indices = set(range(num_sites))
    ordered_indices = []
    total_proximities = proximity_matrix.sum(axis=1)
    op = np.argmin if maximize_dist else np.argmax
    first_site_idx = int(op(total_proximities))
    ordered_indices.append(first_site_idx)
    available_indices.remove(first_site_idx)
    for _ in range(num_sites - 1):
        avail_list = list(available_indices)
        scores = proximity_matrix[avail_list][:, ordered_indices].sum(axis=1)
        best_local_idx = int(op(scores))
        next_site_idx = avail_list[best_local_idx]
        ordered_indices.append(next_site_idx)
        available_indices.remove(next_site_idx)
    return ordered_indices


def _subsample_sites(site_positions: np.ndarray, ratio: float,
                     distribution: str, seed: int) -> List[int]:
    """Return list of integer indices into site_positions to actually passivate."""
    n = len(site_positions)
    k = max(1, int(round(ratio * n)))
    k = min(k, n)
    if distribution == "random":
        rng = random.Random(seed)
        return rng.sample(range(n), k)
    # spatial ordering (segmented = clustered, uniform = spread)
    maximize_dist = (distribution == "uniform")
    ordered = _get_spatially_ordered_indices(site_positions, maximize_dist)
    return ordered[:k]


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised ligand placement (adapted from miniCAT; works on virtual anchors)
# ──────────────────────────────────────────────────────────────────────────────

def _refine_one_ligand(
    site_config: dict,
    env_pos: np.ndarray,
    env_z: np.ndarray,
    args_ns,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Place and orientate one neutral ligand at a virtual anchor site.

    Parameters
    ----------
    site_config : dict with keys:
        dpos      – virtual anchor position (Å), i.e. pts[i] + offset * n0
        n0        – outward normal (missing bond direction)
        numbers   – atomic numbers of ligand atoms
        coords    – ligand coords in its own frame (from RDKit conformer)
        anchor_idx– index of anchor atom in the ligand
        metal_pos – position of the surface atom (for bond-length reference)
        bond_len  – target M–L bond length (Å)
    env_pos : (M, 3) float — positions of already-placed atoms in environment
    env_z   : (M,)   int  — atomic numbers of environment atoms
    args_ns : namespace with fields: sterics_mode, coarse_step_deg,
              adaptive_offset_steps, adaptive_offset_step, neighbor_repulsion
    Returns
    -------
    numbers, coords : arrays for the placed ligand
    """
    numbers    = np.asarray(site_config["numbers"], int)
    lig_coords = np.asarray(site_config["coords"], float)
    anchor_idx = int(site_config["anchor_idx"])
    anchor_center = np.asarray(
        site_config.get("anchor_center", lig_coords[anchor_idx]), float
    )
    anchor_vec = _unit(np.asarray(site_config.get("anchor_vec", [0.0, 0.0, 1.0]), float))
    n0         = _unit(site_config["n0"])
    dpos       = np.asarray(site_config["dpos"], float)
    metal_pos  = np.asarray(site_config["metal_pos"], float)
    bond_len   = float(site_config["bond_len"])
    host_env_idx = site_config.get("host_env_idx")
    other_site_positions = np.asarray(site_config.get("other_site_positions", []), float)

    # ---------- Align ligand: anchor atom → surface atom (inward) ----------
    # Step 1: translate so the neutral anchor point is at origin.
    lig0 = lig_coords - anchor_center

    # Step 2: align anchor→body to point AWAY from surface (along n0)
    #   → so anchor atom ends up closest to the surface
    aligned = (_rotation_matrix_from_vectors(anchor_vec, n0) @ lig0.T).T

    # ---------- Steric scan: rotate around n0 ----------
    # Do not count the intended host-anchor contact as a steric clash.
    exclude_mask = np.zeros(len(env_z), bool)
    hosts_to_exclude = site_config.get("hosts", [host_env_idx])
    for h in hosts_to_exclude:
        if h is not None and 0 <= int(h) < len(exclude_mask):
            exclude_mask[int(h)] = True

    extra_base = bond_len - float(np.linalg.norm(metal_pos - dpos))

    phis = np.deg2rad(np.arange(0, 360, float(getattr(args_ns, "coarse_step_deg", 20.0))))
    n_adaptive = int(getattr(args_ns, "adaptive_offset_steps", 4))
    adaptive_step = float(getattr(args_ns, "adaptive_offset_step", 0.15))
    extras = np.arange(n_adaptive + 1) * adaptive_step

    phi_grid, extra_grid = np.meshgrid(phis, extras)
    phi_flat = phi_grid.flatten()
    extra_flat = extra_grid.flatten()
    num_poses = len(phi_flat)

    ux, uy, uz = n0
    c, s = np.cos(phi_flat), np.sin(phi_flat)
    R = np.array([
        [c + ux*ux*(1-c),   ux*uy*(1-c) - uz*s,  ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c),    uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)  ],
    ]).transpose(2, 0, 1)  # (num_poses, 3, 3)

    all_rotated = np.einsum("pij,aj->pai", R, aligned)   # (num_poses, N_lig, 3)
    anchor_pts  = dpos + n0 * (extra_base + extra_flat[:, None])
    all_coords  = all_rotated + anchor_pts[:, None, :]   # (num_poses, N_lig, 3)

    sterics_mode = str(getattr(args_ns, "sterics_mode", "vdw"))
    if len(env_pos) > 0:
        cand_idx = (
            np.where(numbers > 1)[0] if sterics_mode == "heavy"
            else np.arange(len(numbers))
        )
        tree, env_radii = _build_sterics_tree(env_pos, env_z, exclude_mask, sterics_mode)
        query_pts = all_coords[:, cand_idx, :].reshape(-1, 3)
        dist_vals, neigh_idx = tree.query(query_pts, k=1)
        dist_vals = dist_vals.reshape(num_poses, -1)
        if sterics_mode == "vdw":
            cand_radii  = np.array([_get_vdw(int(numbers[i])) for i in cand_idx])
            env_r_match = env_radii[neigh_idx].reshape(num_poses, -1)
            clearances  = dist_vals - (cand_radii[None, :] + env_r_match)
        else:
            clearances = dist_vals
        steric_scores = np.min(clearances, axis=1)
    else:
        steric_scores = np.zeros(num_poses)

    if other_site_positions.size and float(getattr(args_ns, "neighbor_repulsion", 0.0)) > 0.0:
        if other_site_positions.ndim == 1:
            other_site_positions = other_site_positions.reshape(1, 3)
        neighbor_dirs = _unit_rows(other_site_positions - dpos)
        lig_coms = np.mean(all_coords, axis=1)
        tail_vecs = _unit_rows(lig_coms - anchor_pts)
        alignments = np.einsum("pi,ki->pk", tail_vecs, neighbor_dirs)
        alignments[alignments < 0] = 0.0
        penalties = float(getattr(args_ns, "neighbor_repulsion", 0.0)) * np.sum(alignments ** 2, axis=1)
    else:
        penalties = np.zeros(num_poses)

    best = int(np.argmax(steric_scores - penalties))
    return numbers, all_coords[best]


# ──────────────────────────────────────────────────────────────────────────────
# Bond length estimation from covalent radii (replaces miniCAT DEFAULT_MX table)
# ──────────────────────────────────────────────────────────────────────────────

_COV_RADII_ANGSTROM: Dict[int, float] = {
    # common elements (pm → Å ÷ 100)
    1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 30: 1.22, 31: 1.22,
    32: 1.20, 33: 1.19, 34: 1.20, 35: 1.20, 46: 1.24, 47: 1.45,
    48: 1.44, 49: 1.42, 50: 1.39, 51: 1.39, 52: 1.38, 53: 1.39,
    78: 1.36, 79: 1.36, 80: 1.32, 81: 1.45, 82: 1.46,
}


def _cov_radius_z(z: int) -> float:
    if z in _COV_RADII_ANGSTROM:
        return _COV_RADII_ANGSTROM[z]
    try:
        from pymatgen.core.periodic_table import Element
        el = Element.from_Z(z)
        r = el.covalent_radius
        if r is not None:
            return float(r)
    except Exception:
        pass
    return 1.2


def _cov_radius_sym(sym: str) -> float:
    try:
        from pymatgen.core.periodic_table import Element
        el = Element(sym)
        r = el.covalent_radius
        if r is not None:
            return float(r)
    except Exception:
        pass
    try:
        from rdkit.Chem import GetPeriodicTable
        pt = GetPeriodicTable()
        return float(pt.GetRcovalent(sym))
    except Exception:
        pass
    return 1.2


def _bond_length_from_cov_radii(metal_sym: str, donor_z: int) -> float:
    """Estimate host-anchor distance from covalent radii with a small safety margin."""
    r_metal = _cov_radius_sym(metal_sym)
    r_donor = _cov_radius_z(donor_z)
    return 1.05 * (r_metal + r_donor)


# ──────────────────────────────────────────────────────────────────────────────
# Missing-bond vector computation
# ──────────────────────────────────────────────────────────────────────────────

def _ideal_bulk_directions(site_sym: str, bulk_struct, charges: Dict[str, int]) -> List[np.ndarray]:
    """
    Return the ideal coordination directions for a site of type `site_sym`
    in the bulk unit cell (all vectors unit-normalised, Cartesian).

    Strategy:
      1. Find atoms of opposite charge in the unit cell nearest to any site of
         type `site_sym`.
      2. Return those bond vectors (expressed relative to the site) as the
         ideal coordination directions.
    """
    from pymatgen.core import Structure

    site_q = charges.get(site_sym, 0)
    sites_of_type = [s for s in bulk_struct.sites if s.specie.symbol == site_sym]
    if not sites_of_type:
        return []

    # Use the first site as reference
    ref_site = sites_of_type[0]
    ref_cart = np.array(ref_site.coords)

    # Collect opposite-charge neighbors (in a 3×3×3 supercell search)
    opp_sites = [s for s in bulk_struct.sites if charges.get(s.specie.symbol, 0) * site_q < 0]
    if not opp_sites:
        return []

    # Enumerate supercell images to get near neighbors
    latt = bulk_struct.lattice
    candidate_vecs = []
    for s in opp_sites:
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    shift = i * latt.matrix[0] + j * latt.matrix[1] + k * latt.matrix[2]
                    v = np.array(s.coords) + shift - ref_cart
                    dist = float(np.linalg.norm(v))
                    if dist > 0.1:
                        candidate_vecs.append((dist, v))

    if not candidate_vecs:
        return []

    candidate_vecs.sort(key=lambda x: x[0])
    # First-shell: all within 1.2 × nearest distance
    d_min = candidate_vecs[0][0]
    first_shell = [_unit(v) for d, v in candidate_vecs if d < 1.2 * d_min]
    # Deduplicate (in case of symmetry)
    unique_dirs: List[np.ndarray] = []
    for v in first_shell:
        if all(float(np.dot(v, u)) < 0.99 for u in unique_dirs):
            unique_dirs.append(v)
    return unique_dirs



def _bulk_cn_refs_from_struct(
    bulk_struct,
    charges: Dict[str, int],
    species: set[str],
    fallback: Dict[str, int],
) -> Dict[str, int]:
    refs: Dict[str, int] = {}
    for sym in species:
        sets = _bulk_ideal_direction_sets(sym, bulk_struct, charges)
        refs[sym] = max((len(dirs) for dirs in sets), default=int(fallback.get(sym, 0)))
    return refs



def compute_missing_bond_vectors(
    syms: List[str],
    pts: NDArray,
    charges: Dict[str, int],
    cuts: Optional[PairCuts],
    bulk_struct,
    surf_mask: NDArray,  # bool array: True = surface atom
) -> Dict[int, List[np.ndarray]]:
    """
    For each surface atom i, compute the list of missing bond direction unit
    vectors (the directions of the ideal coordination polyhedron that have no
    actual neighbor).

    Returns a dict mapping atom index → list of unit vectors.
    If the ideal geometry cannot be determined, falls back to a radial-outward
    direction (COM-based).
    """
    pts = np.asarray(pts, float)
    com = pts.mean(axis=0)
    tree = cKDTree(pts)

    # Cache ideal directions per species (same in whole crystal)
    ideal_cache: Dict[str, List[np.ndarray]] = {}

    result: Dict[int, List[np.ndarray]] = {}

    for i in np.where(surf_mask)[0]:
        sym = syms[i]
        q_i = charges.get(sym, 0)
        if q_i == 0:
            continue   # skip neutral atoms (ligands etc.)

        # Get ideal directions from bulk unit cell
        if sym not in ideal_cache:
            ideal_cache[sym] = _ideal_bulk_directions(sym, bulk_struct, charges)
        ideal_dirs = ideal_cache[sym]

        if not ideal_dirs:
            # Fallback: radial outward
            result[i] = [_unit(pts[i] - com)]
            continue

        # Find actual neighbors
        max_rcut = max(
            _pair_cut_calibrated(sym, s2, cuts)
            for s2 in set(syms)
            if charges.get(s2, 0) * q_i < 0
        ) if any(charges.get(s2, 0) * q_i < 0 for s2 in set(syms)) else 4.0

        neigh_idxs = tree.query_ball_point(pts[i], r=max_rcut)
        actual_vecs = []
        for j in neigh_idxs:
            if j == i:
                continue
            if charges.get(syms[j], 0) * q_i >= 0:
                continue  # same-charge or neutral: skip
            d = pts[j] - pts[i]
            dist = float(np.linalg.norm(d))
            if dist > 0.1 and dist <= max_rcut:
                actual_vecs.append(_unit(d))

        # Rotate ideal directions to match the local frame of atom i
        # We align the "mean of ideal dirs" to the "mean of actual bonds" as
        # a quick local-frame registration.
        if actual_vecs and ideal_dirs:
            mean_actual = _unit(np.mean(actual_vecs, axis=0))
            mean_ideal  = _unit(np.mean(ideal_dirs, axis=0))
            R = _rotation_matrix_from_vectors(mean_ideal, mean_actual)
            rotated_ideal = [_unit(R @ d) for d in ideal_dirs]
        else:
            rotated_ideal = ideal_dirs

        missing = _match_actual_to_ideal(actual_vecs, rotated_ideal)

        if not missing:
            result[i] = []
        else:
            # Sanity check: missing directions should point outward
            outward_missing = []
            for m in missing:
                if float(np.dot(m, pts[i] - com)) < 0:
                    m = -m   # flip if pointing inward
                outward_missing.append(m)
            result[i] = outward_missing

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Surface site classification
# ──────────────────────────────────────────────────────────────────────────────

def _find_undercoordinated_surface_sites(
    syms: List[str],
    pts: NDArray,
    charges: Dict[str, int],
    planes: List[Plane],
    surf_tol: float,
    cuts: Optional[PairCuts],
) -> Tuple[NDArray, Dict[str, int]]:
    """
    Find surface atoms still undercoordinated after charge balance.

    Returns
    -------
    surf_mask : bool array (True = surface atom, at least 1 Wulff plane within surf_tol)
    bulk_cn   : dict mapping species → bulk CN
    """
    pts = np.asarray(pts, float)
    n = len(syms)

    # Build surface mask
    surf_mask = np.zeros(n, bool)
    for (normal, d) in planes:
        normal = np.asarray(normal, float)
        surf_mask |= ((d - pts @ normal) < surf_tol)

    # Bulk CN
    bulk_cn = bulk_cn_opposite_by_interior(
        syms, pts, planes, surf_tol, charges, pair_cuts=cuts
    )

    return surf_mask, bulk_cn


def _native_core_species(cfg: Config, bulk_struct) -> set[str]:
    """Species eligible as inorganic neutral-ligand hosts."""
    if bulk_struct is not None and hasattr(bulk_struct, "sites"):
        species = {str(site.specie.symbol) for site in bulk_struct.sites}
    else:
        excluded = {cfg.passivation.ligand}
        if cfg.passivation.cation_ligand:
            excluded.add(cfg.passivation.cation_ligand)
        organic = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
        species = {
            s for s, q in cfg.charges.items()
            if int(q) != 0 and s not in excluded and s not in organic
        }
    species.discard(cfg.passivation.ligand)
    if cfg.passivation.cation_ligand:
        species.discard(cfg.passivation.cation_ligand)
    return species


def _virtual_site_occupied_by_ligand(
    site_pos: np.ndarray,
    syms: List[str],
    pts: NDArray,
    native_species: set[str],
    tol: float = 0.85,
) -> bool:
    """Return True if a non-native atom already occupies this virtual site."""
    site_pos = np.asarray(site_pos, float)
    pts = np.asarray(pts, float)
    tol2 = float(tol) ** 2
    for sym, xyz in zip(syms, pts):
        if sym in native_species:
            continue
        if float(np.sum((np.asarray(xyz, float) - site_pos) ** 2)) <= tol2:
            return True
    return False


def _neutral_ligand_surface_mask(
    pts: NDArray,
    planes: List[Plane],
    surf_tol: float,
    native_species: set[str],
    charges: Dict[str, int],
    cuts: Optional[PairCuts],
) -> Tuple[np.ndarray, float]:
    """
    Surface shell used only for neutral-ligand site discovery.

    Wulff planes can sit on one sublattice.  On a cation-rich facet, the
    undercoordinated cations may therefore be one native bond-length below the
    outermost mathematical plane and fail the stricter construction surf_tol.
    The CN deficit check still decides whether an atom is truly passivatable,
    so this wider shell does not expose bulk-saturated interior atoms.
    """
    native = [s for s in native_species if charges.get(s, 0) != 0]
    native_pair_cut = 0.0
    for i, s1 in enumerate(native):
        for s2 in native[i + 1:]:
            if charges.get(s1, 0) * charges.get(s2, 0) < 0:
                native_pair_cut = max(native_pair_cut, _pair_cut_calibrated(s1, s2, cuts))
    shell_tol = max(float(surf_tol), float(surf_tol) + native_pair_cut)

    mask = np.zeros(len(pts), bool)
    pts = np.asarray(pts, float)
    for normal, d in planes:
        n = _unit(np.asarray(normal, float))
        mask |= ((float(d) - pts @ n) < shell_tol)
    return mask, shell_tol



def _choose_outward_vectors(
    vectors: List[np.ndarray],
    outward: np.ndarray,
    n_slots: int,
) -> List[np.ndarray]:
    outward = _unit(outward)
    cleaned = []
    for vec in vectors:
        v = _unit(vec)
        if float(np.dot(v, outward)) < 0.0:
            v = -v
        cleaned.append(v)
    cleaned.sort(key=lambda v: float(np.dot(v, outward)), reverse=True)
    selected = [v for v in cleaned if float(np.dot(v, outward)) >= 0.35]
    if not selected:
        selected = [outward]
    while len(selected) < n_slots:
        selected.append(outward)
    return selected[:n_slots]


# ──────────────────────────────────────────────────────────────────────────────
# Top-level orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_neutral_ligand_posttreatment(
    syms: List[str],
    pts: NDArray,
    cfg: Config,
    bulk_struct,        # pymatgen Structure (CIF-loaded)
    planes: List[Plane],
) -> Tuple[List[str], NDArray]:
    """
    Run the neutral-ligand post-treatment and return the updated (syms, pts).

    Called from main.py after charge_balance_iterative() returns Q=0.
    """
    spec: NeutralLigandPostTreatSpec = getattr(
        getattr(cfg, "post_treatment", None),
        "neutral_ligands",
        cfg.passivation.neutral_ligands,
    )
    if not spec.enabled or not spec.passes:
        return syms, pts

    charges    = cfg.charges
    surf_tol   = cfg.passivation.surf_tol
    cuts: Optional[PairCuts] = None  # will be computed lazily if needed

    # Derive pair cuts (same logic as main.py)
    try:
        from .analysis import derive_pair_cuts_from_cif
        # bulk_struct may be None in some tests; guard
        if bulk_struct is not None and hasattr(bulk_struct, "sites"):
            # We need the CIF path to derive cuts; use struct directly
            # Approximate: use supercell approach inline
            pass  # cuts remain None → covalent fallback
    except Exception:
        cuts = None

    print("\n[post-treatment] ── Neutral-ligand passivation ──────────────────────────")

    # Setup RNG
    random.seed(spec.seed)
    np.random.seed(spec.seed)

    # Setup args namespace for _refine_one_ligand
    class _Args:
        sterics_mode        = spec.sterics_mode
        coarse_step_deg     = 20.0
        adaptive_offset_steps = 4
        adaptive_offset_step  = 0.15
        neighbor_repulsion  = 0.5

    args_ns = _Args()

    # Working copies (lists for fast append)
    work_syms = list(syms)
    work_pts  = list(pts)
    native_species = _native_core_species(cfg, bulk_struct)
    if native_species:
        print(f"  Native inorganic host species: {sorted(native_species)}")

    # ── Pass loop ──────────────────────────────────────────────────────────────
    total_placed = 0
    for pass_idx, pass_spec in enumerate(spec.passes):
        print(f"\n[neutral-ligand:pass-{pass_idx+1}] "
              f"target={pass_spec.target!r}  smiles={pass_spec.smiles!r}  "
              f"ratio={pass_spec.ratio:.2f}  dist={pass_spec.distribution}")

        # Recompute surface mask and bulk CN on current structure
        cur_syms = work_syms
        cur_pts  = np.asarray(work_pts, float)

        surf_mask, bulk_cn = _find_undercoordinated_surface_sites(
            cur_syms, cur_pts, charges, planes, surf_tol, cuts
        )
        neutral_surf_mask, neutral_shell_tol = _neutral_ligand_surface_mask(
            cur_pts, planes, surf_tol, native_species, charges, cuts
        )
        cn_refs = _bulk_cn_refs_from_struct(bulk_struct, charges, native_species, bulk_cn)

        # Compute actual CN
        cn = coord_numbers_bipartite(cur_syms, cur_pts, charges, pair_cuts=cuts)

        # Prepare ligand molecule once.  It remains neutral: no ionic transform.
        try:
            mol = _smiles_to_3d_mol(pass_spec.smiles, seed=spec.seed, ff=spec.ff)
        except Exception as exc:
            print(f"  [warning] Could not prepare SMILES {pass_spec.smiles!r}: {exc}. Skipping pass.")
            continue

        lig_coords = _rdconf_to_numpy(mol)
        lig_numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], int)
        anchor_atom_idx, anchor_center, anchor_vec, anchor_name = _detect_neutral_anchor(mol)
        print(f"  → Neutral anchor: {anchor_name} atom_index={anchor_atom_idx}")

        # --- Identify eligible native inorganic sites for this pass ---
        eligible_indices: List[int] = []
        for i in np.where(neutral_surf_mask)[0]:
            sym = cur_syms[i]
            if sym not in native_species:
                continue
            q_i = charges.get(sym, 0)
            if q_i == 0:
                continue
            deficit = int(cn_refs.get(sym, bulk_cn.get(sym, 0))) - int(cn[i])
            if deficit <= 0:
                continue
            # Filter by target
            site_type = "cation" if q_i > 0 else "anion"
            if pass_spec.target not in (site_type, "both"):
                continue
            # Each deficit slot → one "virtual site" entry
            for _ in range(deficit):
                eligible_indices.append(i)

        if not eligible_indices:
            print(f"  → No eligible sites found. Skipping.")
            continue

        # eligible_indices may have duplicates (one per deficit slot)
        # Build a mapping: atom_idx → how many times it appears
        from collections import Counter
        slot_counts = Counter(eligible_indices)
        unique_eligible = list(slot_counts.keys())
        print(f"  → {len(unique_eligible)} eligible atoms  "
              f"({sum(slot_counts.values())} total slots; neutral shell={neutral_shell_tol:.2f} Å)")

        # Compute unified CIF-Intersection virtual sites
        eligible_mask = np.zeros(len(cur_syms), bool)
        for i in unique_eligible:
            eligible_mask[i] = True

        merged_cif_sites = []
        if bulk_struct is not None:
            try:
                merged_cif_sites = compute_cif_virtual_sites(
                    cur_syms,
                    cur_pts,
                    charges,
                    cuts,
                    bulk_struct,
                    eligible_mask,
                    planes,
                    neutral_shell_tol,
                )
            except Exception as e:
                print(f"  [warning] compute_cif_virtual_sites failed: {e}. Skipping pass.")
                continue

        if not merged_cif_sites:
            print("  → No strict virtual sites available. Skipping.")
            continue

        before_occupied_filter = len(merged_cif_sites)
        merged_cif_sites = [
            site for site in merged_cif_sites
            if not _virtual_site_occupied_by_ligand(
                site["pos"], cur_syms, cur_pts, native_species
            )
        ]
        n_filtered = before_occupied_filter - len(merged_cif_sites)
        if n_filtered:
            print(
                f"  → Skipped {n_filtered} virtual sites already occupied by "
                "post-treatment/native ligands"
            )

        if not merged_cif_sites:
            print("  → No unoccupied strict virtual sites available. Skipping.")
            continue

        # Option 1: Strictly Limit to One Ligand Per Surface Atom (Maximum Steric Protection)
        # Ensure that each surface host atom is associated with at most one selected virtual site.
        unique_host_sites = []
        passivated_hosts = set()
        for site in merged_cif_sites:
            if any(h in passivated_hosts for h in site["hosts"]):
                continue
            unique_host_sites.append(site)
            for h in site["hosts"]:
                passivated_hosts.add(h)

        if not unique_host_sites:
            print("  → No independent virtual sites available after steric host filtering. Skipping.")
            continue

        total_unique = len(unique_host_sites)
        k = max(1, int(round(pass_spec.ratio * total_unique)))
        k = min(k, total_unique)

        # Subsample/order the unique_host_sites based on the distribution to treat facets on same ground
        site_positions = np.asarray([s["pos"] for s in unique_host_sites], float)
        if pass_spec.distribution == "random":
            rng = random.Random(spec.seed + pass_idx)
            ordered_indices = list(range(total_unique))
            rng.shuffle(ordered_indices)
        elif pass_spec.distribution in ("uniform", "segmented"):
            maximize_dist = (pass_spec.distribution == "uniform")
            ordered_indices = _get_spatially_ordered_indices(site_positions, maximize_dist)
        else:
            # Fallback: keep the original multiplicity-based order (descending)
            ordered_indices = list(range(total_unique))

        selected_cif_sites = []
        for idx in ordered_indices:
            if len(selected_cif_sites) >= k:
                break
            selected_cif_sites.append(unique_host_sites[idx])

        # satisfied_slots is the sum of multiplicities (deficits satisfied)
        satisfied_slots = sum(site["multiplicity"] for site in selected_cif_sites)
        total_slots = sum(site["multiplicity"] for site in unique_host_sites)

        print(f"  → {len(selected_cif_sites)} sites selected for passivation (satisfying {satisfied_slots}/{total_slots} deficit slots; distribution={pass_spec.distribution})")

        if not selected_cif_sites:
            continue

        # Pre-compute site configs and refine ligands against each other, miniCAT-style.
        base_syms = list(work_syms)
        base_pts = np.asarray(work_pts, float)
        site_configs = []
        site_dpos = []
        for site in selected_cif_sites:
            new_pos = site["pos"]
            hosts = site["hosts"]
            primary_host = hosts[0]
            sym_i = cur_syms[primary_host]
            metal_pos = cur_pts[primary_host]
            n0 = site["u_vecs"][primary_host]
            bond_len = float(np.linalg.norm(new_pos - metal_pos))
            site_dpos.append(new_pos)

            site_configs.append({
                "dpos":       new_pos,
                "n0":         n0,
                "numbers":    lig_numbers,
                "coords":     lig_coords,
                "anchor_idx": anchor_atom_idx,
                "anchor_center": anchor_center,
                "anchor_vec":  anchor_vec,
                "metal_pos":  metal_pos,
                "bond_len":   bond_len,
                "host_env_idx": primary_host,
                "hosts":      hosts,
            })

        for i, sc in enumerate(site_configs):
            sc["other_site_positions"] = [
                p for j, p in enumerate(site_dpos) if j != i
            ]

        # Partition sites into independent, spatially isolated batches
        batches = []
        site_positions = np.asarray(site_dpos, float)
        for idx in range(len(site_positions)):
            pos = site_positions[idx]
            placed_in_batch = False
            for batch in batches:
                clash = False
                for other_idx in batch:
                    dist = float(np.linalg.norm(pos - site_positions[other_idx]))
                    if dist < 10.0:
                        clash = True
                        break
                if not clash:
                    batch.append(idx)
                    placed_in_batch = True
                    break
            if not placed_in_batch:
                batches.append([idx])

        placed_ligands: List[Tuple[np.ndarray, np.ndarray]] = [
            (np.array([], dtype=int), np.zeros((0, 3), float))
            for _ in site_configs
        ]
        for batch in batches:
            # Build env_pos containing base nanocrystal plus all ligands placed in previous batches
            env_pos_parts = [base_pts]
            env_z_parts = [np.array([_atomic_number(s) for s in base_syms], int)]
            for j, (nums_j, coords_j) in enumerate(placed_ligands):
                if len(nums_j) > 0:
                    env_pos_parts.append(np.asarray(coords_j, float))
                    env_z_parts.append(np.asarray(nums_j, int))
            env_pos = np.vstack(env_pos_parts)
            env_z = np.concatenate(env_z_parts)
            
            # Place and optimize all ligands in the current batch simultaneously
            for i in batch:
                site_config = site_configs[i]
                placed_ligands[i] = _refine_one_ligand(
                    site_config, env_pos, env_z, args_ns
                )

        placed_count = 0
        for placed_numbers, placed_coords in placed_ligands:
            for at_z, at_pos in zip(placed_numbers, placed_coords):
                sym_lig = _z_to_symbol(int(at_z))
                work_syms.append(sym_lig)
                work_pts.append(at_pos)
            placed_count += 1
            total_placed += 1

        print(f"  → Placed ligand on {placed_count} sites")

    print(f"[neutral-ligand:done] Total neutral ligand atoms added: "
          f"{len(work_syms) - len(syms)}")

    return work_syms, np.asarray(work_pts, float)


# ──────────────────────────────────────────────────────────────────────────────
# Small utilities: Z ↔ symbol conversion
# ──────────────────────────────────────────────────────────────────────────────

def _atomic_number(sym: str) -> int:
    try:
        from pymatgen.core.periodic_table import Element
        return int(Element(sym).Z)
    except Exception:
        pass
    try:
        from rdkit.Chem import GetPeriodicTable
        pt = GetPeriodicTable()
        return int(pt.GetAtomicNumber(sym))
    except Exception:
        pass
    # Minimal fallback table
    _FALLBACK = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
                 "Cl": 17, "Br": 35, "I": 53, "Cd": 48, "Se": 34, "In": 49,
                 "Zn": 30, "Ga": 31, "As": 33, "Pb": 82}
    return _FALLBACK.get(sym, 0)


def _z_to_symbol(z: int) -> str:
    try:
        from pymatgen.core.periodic_table import Element
        return str(Element.from_Z(z).symbol)
    except Exception:
        pass
    try:
        from rdkit.Chem import GetPeriodicTable
        pt = GetPeriodicTable()
        return str(pt.GetElementSymbol(z))
    except Exception:
        pass
    _FALLBACK = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S",
                 17: "Cl", 35: "Br", 53: "I", 48: "Cd", 34: "Se", 49: "In",
                 30: "Zn", 31: "Ga", 33: "As", 82: "Pb"}
    return _FALLBACK.get(z, "X")
