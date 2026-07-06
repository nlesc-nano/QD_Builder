from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .analysis import (
    PairCuts,
    _pair_cut_calibrated,
    bulk_cn_opposite_by_interior,
    coord_numbers_bipartite,
    compute_cif_virtual_sites,
    derive_pair_cuts_from_cif,
    _surface_outward_direction,
)
from .nc_types import Config, LigandExchangePostTreatSpec, Plane
from .neutral_ligand_posttreat import (
    _anchor_vec_to_body,
    _atomic_number,
    _bond_length_from_cov_radii,
    _build_sterics_tree,
    _detect_anchor_atom,
    _get_vdw,
    _rdconf_to_numpy,
    _rotation_matrix_from_vectors,
    _smiles_to_3d_mol,
    _subsample_sites,
    _unit,
    _z_to_symbol,
)


def _find_attached_h(mol, heavy_idx: int) -> Optional[int]:
    atom = mol.GetAtomWithIdx(int(heavy_idx))
    for nb in atom.GetNeighbors():
        if nb.GetAtomicNum() == 1:
            return int(nb.GetIdx())
    return None


def _map_index(idx: Optional[int], h_idx: int) -> Optional[int]:
    if idx is None:
        return None
    if idx == h_idx:
        return None
    return idx - 1 if idx > h_idx else idx


def _compute_v_tail(mol, coords: np.ndarray, d1: int, d2: Optional[int], c_center: Optional[int]) -> np.ndarray:
    numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], int)
    ignore_set = {d1}
    if d2 is not None:
        ignore_set.add(d2)
    if c_center is not None:
        ignore_set.add(c_center)
        
    heavy_body = [i for i, z in enumerate(numbers) if z > 1 and i not in ignore_set]
    if heavy_body:
        anchor_ref = coords[c_center] if c_center is not None else coords[d1]
        return _unit(coords[heavy_body].mean(axis=0) - anchor_ref)
        
    neighbors = [int(nb.GetIdx()) for nb in mol.GetAtomWithIdx(d1).GetNeighbors()]
    if neighbors:
        return _unit(coords[d1] - coords[neighbors[0]])
    return np.array([0.0, 0.0, 1.0])


def _embed_and_optimize(mol, seed: int, ff: str):
    from rdkit.Chem import AllChem

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) < 0:
        raise RuntimeError("3-D embedding failed for charged ligand")
    if ff == "mmff" and AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    return mol


def _branch_size(mol, start_idx: int, parent_idx: int) -> int:
    seen = {int(parent_idx), int(start_idx)}
    queue = [int(start_idx)]
    count = 0
    while queue:
        cur = queue.pop(0)
        count += 1
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            ni = int(nb.GetIdx())
            if ni in seen:
                continue
            seen.add(ni)
            queue.append(ni)
    return count


def _smallest_branch_anchor_vec(mol, coords: np.ndarray, anchor_idx: int) -> np.ndarray:
    anchor_idx = int(anchor_idx)
    anchor = coords[anchor_idx]
    neighs = [int(nb.GetIdx()) for nb in mol.GetAtomWithIdx(anchor_idx).GetNeighbors()]
    if not neighs:
        return np.array([0.0, 0.0, 1.0])
    best_nb = min(neighs, key=lambda nb: _branch_size(mol, nb, anchor_idx))
    return _unit(anchor - coords[best_nb])


def _cation_anchor_vec(mol, coords: np.ndarray, anchor_idx: int, anchor_name: str) -> np.ndarray:
    if anchor_name in {
        "quat_ammonium",
        "quat_phosphonium",
        "methyl_ammonium",
        "methyl_phosphonium",
    }:
        return _smallest_branch_anchor_vec(mol, coords, anchor_idx)
    return _anchor_vec_to_body(mol, coords, anchor_idx)


def _add_methyl_cation(rw, anchor_idx: int):
    from rdkit import Chem

    c_idx = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(anchor_idx, c_idx, Chem.BondType.SINGLE)
    for _ in range(3):
        h_idx = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(c_idx, h_idx, Chem.BondType.SINGLE)
    rw.GetAtomWithIdx(anchor_idx).SetFormalCharge(+1)


def _prepare_anion_ligand(smiles: str, seed: int, ff: str):
    from rdkit import Chem

    mol = _smiles_to_3d_mol(smiles, seed=seed, ff=ff)
    patterns = [
        ("carboxylate", Chem.MolFromSmarts("[CX3](=O)[OX2H1]")),
        ("phosphonate", Chem.MolFromSmarts("[PX4](=O)([OX2H1])[OX2H1,O-]")),
        ("sulfonate", Chem.MolFromSmarts("[#16X6](=O)(=O)[OX2H1]")),
        ("thiolate", Chem.MolFromSmarts("[SX2H1]")),
        ("alkoxide", Chem.MolFromSmarts("[OX2H1][#6]")),
    ]

    neutral_d1 = None
    neutral_d2 = None
    neutral_c_center = None
    anchor_name = ""

    for name, patt in patterns:
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            continue
        
        match = matches[0]
        if name == "carboxylate":
            # [C]=0, (=O)=1, [OH]=2
            neutral_d1 = int(match[2])
            neutral_c_center = int(match[0])
            neutral_d2 = int(match[1])
        elif name in {"phosphonate", "sulfonate"}:
            neutral_c_center = int(match[0])
            candidates = [int(idx) for idx in match if mol.GetAtomWithIdx(int(idx)).GetAtomicNum() == 8]
            for c in candidates:
                h = _find_attached_h(mol, c)
                if h is not None:
                    neutral_d1 = c
                    break
            # d2 is another O neighbor of c_center
            for nb in mol.GetAtomWithIdx(neutral_c_center).GetNeighbors():
                nbi = int(nb.GetIdx())
                if nb.GetAtomicNum() == 8 and nbi != neutral_d1:
                    neutral_d2 = nbi
                    break
        elif name == "thiolate":
            neutral_d1 = int(match[0])
        elif name == "alkoxide":
            neutral_d1 = int(match[0])
            
        anchor_name = name
        break

    if neutral_d1 is None:
        raise ValueError(f"No recognizable functional group found in {smiles!r}")

    h_idx = _find_attached_h(mol, neutral_d1)
    if h_idx is None:
        raise ValueError(f"No removable acidic proton found on the anchor atom in {smiles!r}")

    # Deprotonate
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(neutral_d1).SetFormalCharge(-1)
    rw.RemoveAtom(h_idx)
    charged = rw.GetMol()
    Chem.SanitizeMol(charged)
    
    # Re-embed and optimize charged form
    charged = _embed_and_optimize(charged, seed=seed, ff=ff)
    coords = _rdconf_to_numpy(charged)

    # Map indices to deprotonated form
    d1 = _map_index(neutral_d1, h_idx)
    d2 = _map_index(neutral_d2, h_idx)
    c_center = _map_index(neutral_c_center, h_idx)

    return charged, d1, d2, c_center, coords, anchor_name


def _prepare_cation_ligand(smiles: str, seed: int, ff: str):
    from rdkit import Chem

    mol = _smiles_to_3d_mol(smiles, seed=seed, ff=ff)
    charged_patterns = [
        ("quat_ammonium", Chem.MolFromSmarts("[#7X4+]")),
        ("quat_phosphonium", Chem.MolFromSmarts("[#15X4+]")),
    ]
    for name, patt in charged_patterns:
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            anchor_idx = int(matches[0][0])
            coords = _rdconf_to_numpy(mol)
            return mol, anchor_idx, None, None, coords, name

    patterns = [
        ("ammonium", Chem.MolFromSmarts("[NX3;!$([NX3](=O))]")),
        ("phosphonium", Chem.MolFromSmarts("[PX3]")),
        ("sulfonium", Chem.MolFromSmarts("[SX2;H0]")),
    ]

    anchor_idx = None
    anchor_name = ""
    for name, patt in patterns:
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            anchor_idx = int(matches[0][0])
            anchor_name = name
            break
    if anchor_idx is None:
        anchor_idx = _detect_anchor_atom(mol)
        if mol.GetAtomWithIdx(anchor_idx).GetAtomicNum() not in {7, 15, 16}:
            raise ValueError(f"No protonatable donor found in {smiles!r}")
        anchor_name = "protonated_donor"

    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(anchor_idx)
    if atom.GetAtomicNum() in {7, 15} and _find_attached_h(mol, anchor_idx) is None:
        anchor_name = "methyl_ammonium" if atom.GetAtomicNum() == 7 else "methyl_phosphonium"
        _add_methyl_cation(rw, anchor_idx)
    else:
        atom.SetFormalCharge(+1)
        h_idx = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(anchor_idx, h_idx, Chem.BondType.SINGLE)
    charged = rw.GetMol()
    Chem.SanitizeMol(charged)
    charged = _embed_and_optimize(charged, seed=seed, ff=ff)
    coords = _rdconf_to_numpy(charged)
    return charged, int(anchor_idx), None, None, coords, anchor_name


def _detect_ligand_charge(smiles: str) -> int:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")

    # Check if matches any anion patterns
    anion_patterns = [
        Chem.MolFromSmarts("[CX3](=O)[OX2H1]"),
        Chem.MolFromSmarts("[PX4](=O)([OX2H1])[OX2H1,O-]"),
        Chem.MolFromSmarts("[#16X6](=O)(=O)[OX2H1]"),
        Chem.MolFromSmarts("[SX2H1]"),
        Chem.MolFromSmarts("[OX2H1][#6]"),
    ]
    for patt in anion_patterns:
        if patt is not None and mol.HasSubstructMatch(patt):
            return -1

    # Check if matches any cation patterns
    cation_patterns = [
        Chem.MolFromSmarts("[#7X4+]"),
        Chem.MolFromSmarts("[#15X4+]"),
        Chem.MolFromSmarts("[NX3;!$([NX3](=O))]"),
        Chem.MolFromSmarts("[PX3]"),
        Chem.MolFromSmarts("[SX2;H0]"),
    ]
    for patt in cation_patterns:
        if patt is not None and mol.HasSubstructMatch(patt):
            return 1

    raise ValueError(f"Could not automatically determine charge for SMILES: {smiles!r}")


def _prepare_charged_ligand(smiles: str, charge: Optional[int], seed: int, ff: str):
    if charge is None or charge == 0:
        charge = _detect_ligand_charge(smiles)

    if charge < 0:
        mol, d1, d2, c_center, coords, anchor_name = _prepare_anion_ligand(smiles, seed, ff)
    elif charge > 0:
        mol, d1, d2, c_center, coords, anchor_name = _prepare_cation_ligand(smiles, seed, ff)
    else:
        raise ValueError("Ligand exchange requires a non-zero charged ligand")
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], int)
    
    # Calculate v_tail in conformer space
    v_tail = _compute_v_tail(mol, coords, d1, d2, c_center)
    
    return {
        "smiles": smiles,
        "name": anchor_name,
        "numbers": numbers,
        "coords": coords,
        "d1": d1,
        "d2": d2,
        "c_center": c_center,
        "v_tail": v_tail,
        "charge": charge,
    }


def _bound_hosts(
    symbols: List[str],
    pts: NDArray[np.float64],
    ligand_idx: int,
    charges: Dict[str, int],
    cuts: Optional[PairCuts],
) -> List[int]:
    lig = symbols[ligand_idx]
    q_lig = int(charges.get(lig, 0))
    if q_lig == 0:
        return []
    hosts: List[Tuple[float, int]] = []
    for i, sym in enumerate(symbols):
        if i == ligand_idx:
            continue
        if int(charges.get(sym, 0)) * q_lig >= 0:
            continue
        dist = float(np.linalg.norm(pts[i] - pts[ligand_idx]))
        if dist <= _pair_cut_calibrated(sym, lig, cuts):
            hosts.append((dist, i))
    hosts.sort()
    return [i for _dist, i in hosts]


def _smiles_assignments(selected_local: List[int], smiles_count: int) -> Dict[int, int]:
    return {local_idx: pos % smiles_count for pos, local_idx in enumerate(selected_local)}


def _place_exchange_ligand(
    site_config: dict,
    env_pos: np.ndarray,
    env_z: np.ndarray,
    args_ns,
) -> Tuple[np.ndarray, np.ndarray]:
    numbers = np.asarray(site_config["numbers"], int)
    lig_coords = np.asarray(site_config["coords"], float)
    d1 = int(site_config["d1"])
    d2 = site_config["d2"]
    c_center = site_config["c_center"]
    v_tail = np.asarray(site_config["v_tail"], float)
    n0 = _unit(np.asarray(site_config["n0"], float))
    n_surf = _unit(np.asarray(site_config.get("n_surf", n0), float))
    dpos = np.asarray(site_config["dpos"], float) # original displaced Cl position
    r_search = float(site_config.get("r_search", 5.5))
    cfg_charges = site_config.get("cfg_charges", {})
    primary_host_idx_in_env = site_config.get("primary_host_idx_in_env")
    other_site_positions = np.asarray(site_config.get("other_site_positions", []), float)
    sterics_margin = float(getattr(args_ns, "sterics_margin", 0.4))
    neighbor_repulsion = float(getattr(args_ns, "neighbor_repulsion", 0.5))

    # Translate conformer so d1 is at origin
    coords_translated = lig_coords - lig_coords[d1]

    # Build local conformer basis
    e_y = v_tail
    v_arb = np.array([1.0, 0.0, 0.0]) if abs(e_y[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e_x = _unit(v_arb - np.dot(v_arb, e_y) * e_y)
    e_z = np.cross(e_x, e_y)
    E = np.column_stack([e_x, e_y, e_z])

    # Target frame in QD space
    f_y = n_surf
    f_arb = np.array([1.0, 0.0, 0.0]) if abs(f_y[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    f_x_0 = _unit(f_arb - np.dot(f_arb, f_y) * f_y)
    f_z_0 = np.cross(f_x_0, f_y)

    # Search for neighboring cations for bidentate coordination reward
    env_symbols = [_z_to_symbol(z) for z in env_z]
    cation_metals = {"Pb", "Cd", "Zn", "In", "Ga", "Cs", "Na", "K", "Ba", "Sr", "Ca", "Mg"}
    cation_indices = [
        idx for idx, sym in enumerate(env_symbols)
        if int(cfg_charges.get(sym, 0)) > 0 or sym in cation_metals
    ]
    neighbor_cation_pts = []
    for idx in cation_indices:
        if idx == primary_host_idx_in_env:
            continue
        dist = float(np.linalg.norm(env_pos[idx] - dpos))
        if 1.5 <= dist <= r_search:
            neighbor_cation_pts.append(env_pos[idx])

    # 360-degree scan
    phis = np.deg2rad(np.arange(0, 360, 10.0))
    candidate_poses = []
    for phi in phis:
        f_x = np.cos(phi) * f_x_0 + np.sin(phi) * f_z_0
        f_z = np.cross(f_x, f_y)
        F = np.column_stack([f_x, f_y, f_z])
        R = F @ E.T
        placed_coords = (R @ coords_translated.T).T + dpos
        candidate_poses.append(placed_coords)

    # Exclude hosts from environment for steric checks
    exclude_mask = np.zeros(len(env_z), bool)
    hosts_to_exclude = site_config.get("hosts", [])
    for h in hosts_to_exclude:
        if h is not None and 0 <= int(h) < len(exclude_mask):
            exclude_mask[int(h)] = True

    # Track valid coordination partners of this site
    hosts_to_exclude = set(site_config.get("hosts", []))
    neighbor_cation_indices = set()
    for idx in cation_indices:
        if idx == primary_host_idx_in_env:
            continue
        dist = float(np.linalg.norm(env_pos[idx] - dpos))
        if dist <= r_search:
            neighbor_cation_indices.add(idx)
    coord_partners = hosts_to_exclude.union(neighbor_cation_indices)

    dents = {d1}
    if d2 is not None:
        dents.add(d2)

    base_size = int(site_config.get("base_size", len(env_pos)))
    original_indices = np.arange(len(env_z))
    from scipy.spatial import cKDTree
    
    pose_scores = []
    
    tree = None
    if len(env_pos) > 0:
        tree = cKDTree(env_pos)

    for pose_coords in candidate_poses:
        score = 0.0
        collision_penalty = 0.0
        
        if tree is not None:
            # Query the closest k neighbors to avoid shadowing/occlusion of colliding atoms by the coordination partner
            k_val = min(8, len(env_pos))
            dists, indices = tree.query(pose_coords, k=k_val)
            if k_val == 1:
                dists = dists[:, np.newaxis]
                indices = indices[:, np.newaxis]
                
            clearance = []
            for a in range(len(numbers)):
                r_vdw_lig = _get_vdw(int(numbers[a]))
                min_clearance_for_atom = 999.0
                
                for step in range(k_val):
                    env_idx = int(indices[a, step])
                    dist_val = dists[a, step]
                    
                    env_atomic_num = env_z[env_idx]
                    r_vdw_env = _get_vdw(int(env_atomic_num))
                    d_threshold = r_vdw_lig + r_vdw_env - sterics_margin
                    
                    overlap = d_threshold - dist_val
                    if overlap > 0.0:
                        # Ignore coordination bonds between dents and their coordination metal partners
                        if env_idx in coord_partners and a in dents:
                            pass
                        else:
                            if env_idx < base_size:
                                collision_penalty += 1000.0 * (overlap ** 2)
                            else:
                                collision_penalty += 10.0 * (overlap ** 2)
                                
                    is_coordination = (env_idx in coord_partners and a in dents)
                    if not is_coordination:
                        min_clearance_for_atom = min(min_clearance_for_atom, dist_val - (r_vdw_lig + r_vdw_env))
                        
                clearance.append(min_clearance_for_atom)
                
            score = float(np.min(clearance))

        # Coordination reward for bidentate
        if d2 is not None and neighbor_cation_pts:
            d2_coord = pose_coords[d2]
            min_c_dist = min(float(np.linalg.norm(d2_coord - c_pt)) for c_pt in neighbor_cation_pts)
            coordination_reward = 2.0 * np.exp(-((min_c_dist - 2.4) ** 2) / (2 * (0.6 ** 2)))
            score += coordination_reward

        # Neighbor tail repulsion penalty
        if other_site_positions.size and neighbor_repulsion > 0.0:
            if other_site_positions.ndim == 1:
                other_site_positions = other_site_positions.reshape(1, 3)
            other_dirs = other_site_positions - dpos
            other_norms = np.linalg.norm(other_dirs, axis=1)
            other_dirs = other_dirs[other_norms > 1e-12] / other_norms[other_norms > 1e-12, None]
            if len(other_dirs):
                tail_vec = _unit(np.mean(pose_coords, axis=0) - dpos)
                alignments = np.dot(other_dirs, tail_vec)
                alignments[alignments < 0.0] = 0.0
                score -= neighbor_repulsion * np.sum(alignments ** 2)

        score -= collision_penalty
        pose_scores.append(score)

    best_idx = int(np.argmax(pose_scores))
    return numbers, candidate_poses[best_idx]


def _is_lattice_site(pos: np.ndarray, bulk_struct, syms: List[str], pts: np.ndarray) -> bool:
    if bulk_struct is None:
        return False
        
    f_coords_all = bulk_struct.lattice.get_fractional_coords(pts)
    bulk_species = {s.specie.symbol for s in bulk_struct.sites}
    ref_idx = -1
    for idx in range(len(syms)):
        if syms[idx] in bulk_species:
            ref_idx = idx
            break
            
    if ref_idx < 0:
        return False
        
    ref_sym = syms[ref_idx]
    ref_sites = [s for s in bulk_struct.sites if s.specie.symbol == ref_sym]
    ref_f = f_coords_all[ref_idx]
    
    best_f_shift = np.zeros(3)
    found_shift = False
    for site in ref_sites:
        candidate_shift = (site.frac_coords - ref_f) % 1
        
        test_indices = []
        for j in range(min(50, len(syms))):
            if syms[j] in bulk_species:
                test_indices.append(j)
                
        if not test_indices:
            best_f_shift = candidate_shift
            found_shift = True
            break
            
        all_match = True
        for j in test_indices:
            shifted_f = (f_coords_all[j] + candidate_shift + 0.5) % 1 - 0.5
            j_sites = [s for s in bulk_struct.sites if s.specie.symbol == syms[j]]
            if not j_sites:
                all_match = False
                break
            min_diff = min(
                float(np.linalg.norm((s.frac_coords - shifted_f + 0.5) % 1 - 0.5))
                for s in j_sites
            )
            if min_diff > 0.05:
                all_match = False
                break
                
        if all_match:
            best_f_shift = candidate_shift
            found_shift = True
            break
            
    if not found_shift:
        return False
        
    pos_f = bulk_struct.lattice.get_fractional_coords(pos)
    pos_f_unshifted = (pos_f + best_f_shift + 0.5) % 1 - 0.5
    
    for site in bulk_struct.sites:
        frac_diff = (site.frac_coords - pos_f_unshifted + 0.5) % 1 - 0.5
        cart_diff = bulk_struct.lattice.get_cartesian_coords(frac_diff)
        dist = float(np.linalg.norm(cart_diff))
        if dist < 0.15:
            return True
            
    return False


def _native_species(bulk_struct) -> set[str]:
    if bulk_struct is None or not hasattr(bulk_struct, "sites"):
        return set()
    return {str(site.specie.symbol) for site in bulk_struct.sites}


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


def _effective_charge(symbols: List[str], charges: Dict[str, int], ledger: List[dict]) -> int:
    q_element = int(sum(int(charges.get(sym, 0)) for sym in symbols))
    q_ignored = int(sum(int(entry.get("ignored_element_charge", 0)) for entry in ledger))
    q_exchange = int(sum(int(entry.get("charge", 0)) for entry in ledger))
    return q_element - q_ignored + q_exchange


def _remove_passivation_ligands_for_charge(
    syms: List[str],
    pts: NDArray[np.float64],
    cfg: Config,
    ledger: List[dict],
    needed: int,
    *,
    distribution: str,
    seed: int,
) -> Tuple[List[str], NDArray[np.float64], int]:
    ligand = cfg.passivation.ligand
    ligand_idx = [i for i, sym in enumerate(syms) if sym == ligand]
    if needed <= 0 or not ligand_idx:
        return syms, pts, 0
    remove_count = min(int(needed), len(ligand_idx))
    selected_local = _subsample_sites(
        np.asarray([pts[i] for i in ligand_idx], float),
        min(1.0, remove_count / max(1, len(ligand_idx))),
        distribution,
        seed,
    )[:remove_count]
    remove_set = {ligand_idx[i] for i in selected_local}
    keep = np.ones(len(syms), dtype=bool)
    for i in remove_set:
        keep[i] = False
    return [s for s, k in zip(syms, keep) if k], pts[keep], len(remove_set)


def rebalance_ligand_exchange_charge(
    syms: List[str],
    pts: NDArray[np.float64],
    cfg: Config,
    bulk_struct,
    planes: List[Plane],
    cif_path: str,
    ledger: List[dict],
    *,
    verbose: bool = True,
) -> Tuple[List[str], NDArray[np.float64]]:
    """
    Correct only the formal charge introduced by charged X-type exchange.

    This deliberately does not run the full structural passivation loop: native
    cations/anions must not be removed or swapped as a side effect of ligand
    exchange compensation.
    """
    if not ledger:
        return syms, pts

    ligand = cfg.passivation.ligand
    q_lig = int(cfg.charges.get(ligand, 0))
    q_total = _effective_charge(syms, cfg.charges, ledger)
    if q_total == 0:
        return syms, pts

    work_syms = list(syms)
    work_pts = np.asarray(pts, float).copy()
    surf_tol = getattr(cfg.passivation, "surf_tol", 2.0)

    if q_total < 0:
        if q_lig >= 0:
            if verbose:
                print(
                    f"[ligand-exchange:charge-balance] residual Q={q_total:+d}; "
                    f"cannot remove non-anionic passivation ligand {ligand!r}"
                )
            return work_syms, work_pts
        needed = int((-q_total + abs(q_lig) - 1) // abs(q_lig))
        if not any(sym == ligand for sym in work_syms):
            if verbose:
                print(
                    f"[ligand-exchange:charge-balance] residual Q={q_total:+d}; "
                    f"no {ligand} ligands available to remove"
                )
            return work_syms, work_pts
        distribution = str(ledger[-1].get("distribution", "uniform"))
        before = q_total
        work_syms, work_pts, removed = _remove_passivation_ligands_for_charge(
            work_syms,
            work_pts,
            cfg,
            ledger,
            needed,
            distribution=distribution,
            seed=int(getattr(cfg.post_treatment.ligand_exchange, "seed", 1337)) + 7919,
        )
        after = _effective_charge(work_syms, cfg.charges, ledger)
        if verbose:
            print(
                f"[ligand-exchange:charge-balance] removed {removed} {ligand} "
                f"ligand(s) | Q:{before:+d}->{after:+d}"
            )
        return work_syms, work_pts

    if q_lig >= 0:
        if verbose:
            print(
                f"[ligand-exchange:charge-balance] residual Q={q_total:+d}; "
                f"cannot add non-anionic passivation ligand {ligand!r}"
            )
        return work_syms, work_pts

    from .passivation import _build_facet_frames, _facet_memberships
    from .passivation_iterative import _priority3_balance_positive_q_add

    pair_cuts = derive_pair_cuts_from_cif(cif_path, cfg.charges, safety=1.00)
    add_count_facet: Dict[int, int] = defaultdict(int)
    edit_count_facet: Dict[int, int] = defaultdict(int)
    uv_taken: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    host_taken: Dict[int, int] = {}
    target = int((q_total + abs(q_lig) - 1) // abs(q_lig))
    added = 0
    before = q_total

    for _ in range(target):
        frames = _build_facet_frames(planes)
        mem = _facet_memberships(work_pts, planes, surf_tol)
        cn_bi = coord_numbers_bipartite(work_syms, work_pts, cfg.charges, pair_cuts=pair_cuts)
        progressed, work_syms, work_pts = _priority3_balance_positive_q_add(
            work_syms,
            work_pts,
            frames,
            planes,
            mem,
            cn_bi,
            cfg.charges,
            ligand,
            surf_tol,
            uv_taken,
            edit_count_facet,
            add_count_facet,
            host_taken,
            {},
            pair_cuts,
            verbose=verbose,
            include_sublayer=False,
            cif_path=cif_path,
        )
        if not progressed:
            break
        added += 1
        if _effective_charge(work_syms, cfg.charges, ledger) <= 0:
            break

    after = _effective_charge(work_syms, cfg.charges, ledger)
    trimmed = 0
    if after < 0:
        needed_remove = int((-after + abs(q_lig) - 1) // abs(q_lig))
        work_syms, work_pts, trimmed = _remove_passivation_ligands_for_charge(
            work_syms,
            work_pts,
            cfg,
            ledger,
            needed_remove,
            distribution=str(ledger[-1].get("distribution", "uniform")),
            seed=int(getattr(cfg.post_treatment.ligand_exchange, "seed", 1337)) + 1543,
        )
        after = _effective_charge(work_syms, cfg.charges, ledger)
    if verbose:
        print(
            f"[ligand-exchange:charge-balance] added {added} {ligand} ligand(s) "
            f"and trimmed {trimmed} excess {ligand} ligand(s) | Q:{before:+d}->{after:+d}"
        )
    return work_syms, work_pts


def run_ligand_exchange_posttreatment(
    syms: List[str],
    pts: NDArray[np.float64],
    cfg: Config,
    bulk_struct,
    planes: List[Plane],
    cif_path: str,
) -> Tuple[List[str], NDArray[np.float64], List[dict]]:
    spec: LigandExchangePostTreatSpec = getattr(
        getattr(cfg, "post_treatment", None),
        "ligand_exchange",
        LigandExchangePostTreatSpec(),
    )
    if not spec.enabled or not spec.passes:
        return syms, pts, []

    cuts = derive_pair_cuts_from_cif(cif_path, cfg.charges, safety=1.00)

    print("\n[post-treatment] ── Charged ligand exchange ───────────────────────────")
    random.seed(spec.seed)
    np.random.seed(spec.seed)

    # 1. Dynamic Calibration from bulk_struct
    d_cc = 4.3 # default fallback
    if bulk_struct is not None:
        cation_syms = {sym for sym, q in cfg.charges.items() if q > 0}
        if not cation_syms:
            cation_syms = {"Pb", "Cd", "Zn", "In", "Ga", "Cs", "Na", "K", "Ba", "Sr", "Ca", "Mg"}
        
        cation_indices = []
        for idx, site in enumerate(bulk_struct):
            if site.species_string in cation_syms:
                cation_indices.append(idx)
        
        if len(cation_indices) >= 2:
            sub_matrix = bulk_struct.distance_matrix[cation_indices][:, cation_indices]
            pos_dists = sub_matrix[sub_matrix > 0.05]
            if len(pos_dists) > 0:
                d_cc = float(np.min(pos_dists))
        elif len(cation_indices) == 1:
            neighbors = bulk_struct.get_neighbors(bulk_struct[cation_indices[0]], r=8.0)
            cation_neigh_dists = [
                float(n.distance) for n in neighbors
                if n.species_string in cation_syms and n.distance > 0.05
            ]
            if cation_neigh_dists:
                d_cc = float(np.min(cation_neigh_dists))
                
    r_search = 1.3 * d_cc
    print(f"  → Calibrated bulk cation-cation nearest distance: {d_cc:.3f} Å")
    print(f"  → Set neighboring bridging search radius: {r_search:.3f} Å")

    class _Args:
        sterics_mode = spec.sterics_mode
        coarse_step_deg = 10.0
        neighbor_repulsion = 0.5
        sterics_margin = 0.4

    work_syms = list(syms)
    work_pts = np.asarray(pts, float).copy()
    total_exchanged = 0
    charge_ledger: List[dict] = []

    for pass_idx, pass_spec in enumerate(spec.passes):
        replace_charge = pass_spec.replace_charge
        if replace_charge is None:
            replace_charge = int(cfg.charges.get(pass_spec.replace, 0))

        molecular_charge = pass_spec.charge
        if molecular_charge is None:
            try:
                molecular_charge = _detect_ligand_charge(pass_spec.smiles[0])
            except Exception as exc:
                print(f"  [warning] Could not determine ligand charge from {pass_spec.smiles[0]!r}: {exc}")
                molecular_charge = -1 if replace_charge < 0 else 1

        print(
            f"\n[ligand-exchange:pass-{pass_idx + 1}] replace={pass_spec.replace!r} "
            f"replace_charge={replace_charge:+d} ligand_charge={molecular_charge:+d} ratio={pass_spec.ratio:.2f} "
            f"dist={pass_spec.distribution} smiles={list(pass_spec.smiles)!r}"
        )

        ligands = []
        for sidx, smiles in enumerate(pass_spec.smiles):
            try:
                prepared = _prepare_charged_ligand(
                    smiles,
                    molecular_charge,
                    seed=spec.seed + 101 * (pass_idx + 1) + sidx,
                    ff=spec.ff,
                )
                ligands.append(prepared)
                print(
                    f"  → Prepared {smiles!r}: {prepared['name']} "
                    f"d1={prepared['d1']} d2={prepared['d2']} c_center={prepared['c_center']}"
                )
            except Exception as exc:
                print(f"  [warning] Could not prepare {smiles!r}: {exc}. Skipping this SMILES.")
        if not ligands:
            print("  → No valid charged ligands prepared. Skipping pass.")
            continue

        # Compute surface mask dynamically on work_pts
        surf_tol = getattr(cfg.passivation, "surf_tol", 2.0)
        surf_mask = np.zeros(len(work_syms), bool)
        if planes:
            for (normal, d) in planes:
                normal = np.asarray(normal, float)
                surf_mask |= ((d - work_pts @ normal) < surf_tol)

        native = _native_species(bulk_struct)
        native_replace = pass_spec.replace in native
        bulk_cn = {}
        cn = None
        if native_replace:
            cn = coord_numbers_bipartite(work_syms, work_pts, cfg.charges, pair_cuts=cuts)
            bulk_cn = bulk_cn_opposite_by_interior(
                work_syms, work_pts, planes, surf_tol, cfg.charges, pair_cuts=cuts
            )
        candidate_indices = [
            i for i, sym in enumerate(work_syms)
            if sym == pass_spec.replace and int(cfg.charges.get(sym, 0)) == replace_charge
        ]
        if native_replace:
            candidate_indices = _filter_passivated_indices(candidate_indices, work_syms, work_pts, native)
        candidates = []
        for li in candidate_indices:
            is_on_surface = surf_mask[li] if li < len(surf_mask) else False
            if native_replace:
                if not is_on_surface:
                    continue
                target_cn = int(bulk_cn.get(work_syms[li], 0))
                if cn is not None and target_cn > 0 and int(cn[li]) >= target_cn:
                    continue
            hosts = _bound_hosts(work_syms, work_pts, li, cfg.charges, cuts)
            if not hosts:
                continue
            # Exclude in-place swapped core/sublayer anions (which sit in bulk coordination pockets with 3 or more neighbors)
            # unless they are physically on the surface (within surf_tol).
            if len(hosts) >= 3 and not is_on_surface:
                continue
            primary = hosts[0]
            vec = work_pts[li] - work_pts[primary]
            norm = float(np.linalg.norm(vec))
            if norm < 1e-8:
                continue
            surf_norm = _surface_outward_direction(li, work_pts, planes, surf_tol)
            candidates.append({
                "ligand_idx": li,
                "hosts": hosts,
                "primary_host": primary,
                "pos": work_pts[li].copy(),
                "n0": vec / norm,
                "n_surf": surf_norm,
                "bond_len": norm,
            })

        if not candidates:
            print("  → No bound native ligands found for exchange. Skipping pass.")
            continue

        site_positions = np.asarray([c["pos"] for c in candidates], float)
        if int(getattr(pass_spec, "target_count", 0) or 0) > 0:
            ratio_eff = min(1.0, int(pass_spec.target_count) / max(1, len(candidates)))
        else:
            ratio_eff = pass_spec.ratio
        selected_local = _subsample_sites(site_positions, ratio_eff, pass_spec.distribution, spec.seed + pass_idx)
        if int(getattr(pass_spec, "target_count", 0) or 0) > 0:
            selected_local = selected_local[:int(pass_spec.target_count)]
        assignments = _smiles_assignments(selected_local, len(ligands))
        selected = [candidates[i] for i in selected_local]
        print(f"  → Selected {len(selected)} / {len(candidates)} native ligands for exchange")

        remove_set = {c["ligand_idx"] for c in selected}
        old_to_base: Dict[int, int] = {}
        base_syms: List[str] = []
        base_pts_list: List[np.ndarray] = []
        for old_idx, (sym, xyz) in enumerate(zip(work_syms, work_pts)):
            if old_idx in remove_set:
                continue
            old_to_base[old_idx] = len(base_syms)
            base_syms.append(sym)
            base_pts_list.append(np.asarray(xyz, float))
        base_pts = np.asarray(base_pts_list, float)
        base_z = np.array([_atomic_number(s) for s in base_syms], int)

        site_configs = []
        site_dpos = []
        for local_idx, c in zip(selected_local, selected):
            lig = ligands[assignments[local_idx]]
            primary = c["primary_host"]
            host_env_idx = old_to_base.get(primary)
            mapped_hosts = [old_to_base[h] for h in c["hosts"] if h in old_to_base]
            
            # Place Dent 1 exactly at the displaced native ligand's coordinate
            anchor_pos = c["pos"].copy()
            site_dpos.append(anchor_pos)
            
            site_configs.append({
                "dpos": anchor_pos,
                "n0": c["n0"],
                "n_surf": c["n_surf"],
                "numbers": lig["numbers"],
                "coords": lig["coords"],
                "d1": lig["d1"],
                "d2": lig["d2"],
                "c_center": lig["c_center"],
                "v_tail": lig["v_tail"],
                "primary_host_idx_in_env": host_env_idx,
                "hosts": mapped_hosts or [host_env_idx],
                "r_search": r_search,
                "cfg_charges": cfg.charges,
                "base_size": len(base_syms),
                "replacement_charge": molecular_charge,
                "removed_charge": replace_charge,
                "replacement_smiles": lig["smiles"],
                "replacement_name": lig["name"],
            })

        for i, sc in enumerate(site_configs):
            sc["other_site_positions"] = np.asarray([p for j, p in enumerate(site_dpos) if j != i], float)

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

        placed = [(np.array([], dtype=int), np.zeros((0, 3), float)) for _ in site_configs]
        for batch in batches:
            # Build env_pos containing base nanocrystal plus all ligands placed in previous batches
            env_pos_parts = [base_pts]
            env_z_parts = [base_z]
            for j, (nums_j, coords_j) in enumerate(placed):
                if len(nums_j) > 0:
                    env_pos_parts.append(np.asarray(coords_j, float))
                    env_z_parts.append(np.asarray(nums_j, int))
            env_pos = np.vstack(env_pos_parts)
            env_z = np.concatenate(env_z_parts)
            
            # Place and optimize all ligands in the current batch simultaneously
            for i in batch:
                sc = site_configs[i]
                placed[i] = _place_exchange_ligand(sc, env_pos, env_z, _Args)

        new_syms = list(base_syms)
        new_pts = base_pts.copy()
        exchanged = 0
        for sc, (nums, coords) in zip(site_configs, placed):
            if len(nums) == 0:
                continue
            atom_syms = [_z_to_symbol(int(z)) for z in nums]
            for z, xyz in zip(nums, coords):
                new_syms.append(_z_to_symbol(int(z)))
                new_pts = np.vstack([new_pts, np.asarray(xyz, float)])
            molecular_charge = int(sc["replacement_charge"])
            charge_ledger.append({
                "smiles": sc["replacement_smiles"],
                "kind": sc["replacement_name"],
                "charge": molecular_charge,
                "removed_charge": int(sc.get("removed_charge", molecular_charge)),
                "removed_symbol": pass_spec.replace,
                "distribution": pass_spec.distribution,
                "ignored_element_charge": int(
                    sum(int(cfg.charges.get(sym, 0)) for sym in atom_syms)
                ),
            })
            exchanged += 1

        work_syms = new_syms
        work_pts = new_pts
        total_exchanged += exchanged
        print(f"  → Exchanged {exchanged} native ligands")

    print(f"[ligand-exchange:done] Total native ligands exchanged: {total_exchanged}")
    return work_syms, np.asarray(work_pts, float), charge_ledger
