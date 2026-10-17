# src/builder/z_type_ligand_posttreat.py
"""
Z-type ligand addition: bind neutral MXn complexes to free surface anions.

A Z-type ligand is a neutral two-electron acceptor, a metal complex such as
CdCl2 or Cd(O2CR)2 that binds through its metal to a surface anion.  The
complex is built on the lattice:

  * the metal M takes the empty cation site above an undercoordinated core
    anion, i.e. the CIF virtual site an L-type donor would use on that anion;
  * its n anions X take the free anion sites around M, preferring sites that
    also lie on the missing bonds of core surface cations, so that X bridges
    M and the surface.

Each unit is neutral, so the particle charge is unchanged.  This is the
reverse of ``z_type_displacement``.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .analysis import _surface_outward_direction, _unit, nc_bulk_frac_shift
from .nc_types import Config, Plane, ZTypeLigandPostTreatSpec
from .neutral_exchange_posttreat import (
    _parse_mxn_replacement,
    _place_single_anion_ligand,
    _prepare_anion_fragment,
    _rotation_about_axis,
)
from .neutral_ligand_posttreat import (
    _get_spatially_ordered_indices,
    _native_core_species,
    _z_to_symbol,
    neutral_ligand_host_sites,
)

# A new atom may not come closer than this fraction of the bulk M-X bond to
# any atom already present (bonded partners sit at 1.0).
_CLEARANCE_FRAC = 0.8
# Same no-ligand-within radius the native-anion X-type exchange uses.
_FREE_ANION_CUTOFF = 3.5


def _bulk_site_neighbour_vectors(
    pos: np.ndarray,
    bulk_struct,
    charges: Dict[str, int],
    f_shift: np.ndarray,
) -> List[np.ndarray]:
    """First-shell opposite-charge neighbour vectors of the lattice site at ``pos``."""
    f = bulk_struct.lattice.get_fractional_coords(np.asarray(pos, float))
    f = (f + f_shift + 0.5) % 1 - 0.5
    site_idx = int(np.argmin([
        float(np.linalg.norm((site.frac_coords - f + 0.5) % 1 - 0.5))
        for site in bulk_struct.sites
    ]))
    site = bulk_struct.sites[site_idx]
    q = int(charges.get(str(site.specie.symbol), 0))
    neigh = bulk_struct.get_neighbors(site, r=4.5)
    if not neigh or q == 0:
        return []
    d_min = min(n.nn_distance for n in neigh)
    return [
        np.asarray(n.coords - site.coords, float)
        for n in neigh
        if n.nn_distance <= 1.15 * d_min and int(charges.get(str(n.specie.symbol), 0)) * q < 0
    ]


def _free_anion_hosts(
    syms: List[str],
    pts: NDArray[np.float64],
    native_species: set[str],
    hosts: List[int],
) -> bool:
    """True if no non-native atom lies within the exchange cutoff of any host."""
    non_native = np.asarray([p for s, p in zip(syms, pts) if s not in native_species], float)
    if non_native.size == 0:
        return True
    for h in hosts:
        if float(np.min(np.linalg.norm(non_native - pts[h], axis=1))) < _FREE_ANION_CUTOFF:
            return False
    return True


def z_type_ligand_host_sites(
    syms: List[str],
    pts: NDArray[np.float64],
    cfg: Config,
    bulk_struct,
    planes: List[Plane],
    native_species: Optional[set[str]] = None,
    *,
    target_symbol: Optional[str] = None,
    cuts=None,
) -> List[dict]:
    """
    Empty cation sites above free core anions, one per anion.

    The anion must be a native surface anion below its bulk coordination
    (the L-type anion sites), with no ligand atom within 3.5 Å (the check the
    native-anion X-type exchange uses), and the cation site must be clear of
    every atom.  The builder UI reports ``len()`` as the available count.
    """
    pts = np.asarray(pts, float)
    if native_species is None:
        native_species = _native_core_species(cfg, bulk_struct)
    sites = neutral_ligand_host_sites(
        syms, pts, cfg, bulk_struct, planes, native_species,
        target="anion", target_symbol=target_symbol, cuts=cuts,
    )
    if not sites:
        return []
    tree = cKDTree(pts)
    out = []
    for site in sites:
        hosts = list(site["hosts"])
        if not _free_anion_hosts(syms, pts, native_species, hosts):
            continue
        bond = float(np.linalg.norm(site["pos"] - pts[hosts[0]]))
        if tree.query_ball_point(site["pos"], r=_CLEARANCE_FRAC * bond):
            continue
        out.append(site)
    return out


def _parse_complex(text: str, charges: Dict[str, int]) -> dict:
    parsed = _parse_mxn_replacement(text, charges)
    q = int(parsed["cation_charge"]) + sum(int(f["charge"]) for f in parsed["anion_fragments"])
    if q != 0:
        raise ValueError(f"Z-type complex {text!r} is not neutral (charge {q:+d})")
    return parsed


def _complex_formula(metal: str, frags: List[dict]) -> str:
    parts = []
    for name, n in Counter(f["symbol"] if f["is_atomic"] else f["smiles"] for f in frags).items():
        label = name if len(name) <= 2 else f"({name})"
        parts.append(label + (str(n) if n > 1 else ""))
    return metal + "".join(parts)


def _place_molecular_anion(
    prep: dict,
    metal_pos: np.ndarray,
    site_pos: np.ndarray,
    env: cKDTree,
) -> Tuple[List[str], np.ndarray]:
    """Anchor at the anion site, tail along M->site, rotated for clearance."""
    syms, coords = _place_single_anion_ligand(prep, metal_pos, site_pos, _unit(site_pos - metal_pos))
    axis = _unit(site_pos - metal_pos)
    best, best_score = coords, -np.inf
    for theta in np.deg2rad(np.arange(0.0, 360.0, 20.0)):
        rot = (coords - site_pos) @ _rotation_about_axis(axis, theta).T + site_pos
        score = float(np.min(env.query(rot)[0]))
        if score > best_score:
            best, best_score = rot, score
    return syms, best


def run_z_type_ligand_posttreatment(
    syms: List[str],
    pts: NDArray[np.float64],
    cfg: Config,
    bulk_struct,
    planes: List[Plane],
) -> Tuple[List[str], NDArray[np.float64], List[dict]]:
    """
    Add neutral MXn complexes on free surface anions.

    Returns the updated structure and one ledger entry per pass.  Passes with
    molecular anions also carry ``charge`` / ``ignored_element_charge`` so the
    caller can count those anions by their molecular charge.
    """
    spec: ZTypeLigandPostTreatSpec = cfg.post_treatment.z_type_ligands
    if not spec.enabled or not spec.passes:
        return syms, pts, []

    print("\n[post-treatment] ── Z-type ligands ─────────────────────────────────────")
    work_syms = list(syms)
    work_pts = [np.asarray(p, float) for p in np.asarray(pts, float)]
    native_species = _native_core_species(cfg, bulk_struct)
    surf_tol = cfg.passivation.surf_tol
    f_shift = nc_bulk_frac_shift(work_syms, np.asarray(work_pts), bulk_struct)
    if f_shift is None:
        print("  [warning] Cannot map the particle onto the CIF lattice. Skipping.")
        return syms, pts, []
    ledger: List[dict] = []

    for pass_idx, pass_spec in enumerate(spec.passes):
        print(
            f"\n[z-type-ligand:pass-{pass_idx + 1}] complex={pass_spec.complex!r} "
            f"target_symbol={pass_spec.target_symbol!r} ratio={pass_spec.ratio:.2f} "
            f"dist={pass_spec.distribution}"
        )
        try:
            parsed = _parse_complex(pass_spec.complex, cfg.charges)
        except Exception as exc:
            print(f"  [warning] {exc}. Skipping pass.")
            continue
        metal = parsed["cation_symbol"]
        frags = parsed["anion_fragments"]
        formula = _complex_formula(metal, frags)
        cfg.charges.setdefault(metal, int(parsed["cation_charge"]))
        preps = []
        for k, frag in enumerate(frags):
            if frag["is_atomic"]:
                cfg.charges.setdefault(frag["symbol"], int(frag["charge"]))
                preps.append(None)
                continue
            preps.append(_prepare_anion_fragment(frag["mol"], spec.seed + 101 * (pass_idx + 1) + k, spec.ff))

        cur_pts = np.asarray(work_pts, float)
        sites = z_type_ligand_host_sites(
            work_syms, cur_pts, cfg, bulk_struct, planes, native_species,
            target_symbol=pass_spec.target_symbol,
        )
        total = len(sites)
        k_req = int(pass_spec.target_count) if pass_spec.target_count > 0 else int(round(pass_spec.ratio * total))
        k_req = min(k_req, total)
        print(f"  → {total} free core anion sites; placing {k_req} {formula} units")
        if k_req <= 0:
            continue

        positions = np.asarray([s["pos"] for s in sites], float)
        if pass_spec.distribution == "random":
            order = list(range(total))
            random.Random(spec.seed + pass_idx).shuffle(order)
        else:
            order = _get_spatially_ordered_indices(positions, pass_spec.distribution == "uniform")

        placed = 0
        n_clash = 0
        n_no_anion_site = 0
        n_bridging = 0
        n_added_atoms = 0
        for idx in order:
            if placed >= k_req:
                break
            site = sites[idx]
            m_pos = np.asarray(site["pos"], float)
            host = site["hosts"][0]
            cur_pts = np.asarray(work_pts, float)
            tree = cKDTree(cur_pts)
            bond = float(np.linalg.norm(m_pos - cur_pts[host]))
            if tree.query_ball_point(m_pos, r=_CLEARANCE_FRAC * bond):
                n_clash += 1
                continue

            # Free anion sites around M: not occupied and clear of every atom.
            outward = _surface_outward_direction(host, cur_pts, planes, surf_tol)
            candidates = []
            for vec in _bulk_site_neighbour_vectors(m_pos, bulk_struct, cfg.charges, f_shift):
                x_pos = m_pos + vec
                d_x = float(np.linalg.norm(vec))
                if tree.query_ball_point(x_pos, r=_CLEARANCE_FRAC * d_x):
                    continue
                cation_partners = sum(
                    1 for j in tree.query_ball_point(x_pos, r=1.15 * d_x)
                    if int(cfg.charges.get(work_syms[j], 0)) > 0
                )
                candidates.append((cation_partners, float(np.dot(_unit(vec), outward)), x_pos))
            if len(candidates) < len(frags):
                n_no_anion_site += 1
                continue
            candidates.sort(key=lambda c: (-c[0], -c[1]))
            chosen = candidates[:len(frags)]

            new_syms = [metal]
            new_pts = [m_pos]
            for frag, prep, (partners, _, x_pos) in zip(frags, preps, chosen):
                n_bridging += int(partners > 0)
                if prep is None:
                    new_syms.append(frag["symbol"])
                    new_pts.append(x_pos)
                else:
                    env = cKDTree(np.vstack([cur_pts, np.asarray(new_pts)]))
                    f_syms, f_pts = _place_molecular_anion(prep, m_pos, x_pos, env)
                    new_syms.extend(f_syms)
                    new_pts.extend(f_pts)
            work_syms.extend(new_syms)
            work_pts.extend(new_pts)
            n_added_atoms += len(new_syms)
            placed += 1

        print(
            f"  → Placed {placed} {formula} units ({n_bridging}/{placed * len(frags)} anions "
            f"bridging to surface cations); skipped {n_clash} crowded sites, "
            f"{n_no_anion_site} without {len(frags)} free anion sites"
        )
        entry = {
            "complex": pass_spec.complex,
            "formula": formula,
            "metal": metal,
            "anions": [f["smiles"] for f in frags],
            "available_sites": total,
            "requested": k_req,
            "placed": placed,
            "bridging_anions": n_bridging,
            "skipped_crowded": n_clash,
            "skipped_no_anion_site": n_no_anion_site,
            "atoms_added": n_added_atoms,
        }
        molecular = [f for f in frags if not f["is_atomic"]]
        if molecular and placed:
            ignored = sum(
                int(cfg.charges.get(_z_to_symbol(int(z)), 0))
                for prep in preps if prep is not None for z in prep["numbers"]
            )
            entry["charge"] = placed * sum(int(f["charge"]) for f in molecular)
            entry["ignored_element_charge"] = placed * ignored
        ledger.append(entry)

    print(f"[z-type-ligand:done] Total atoms added: {len(work_syms) - len(syms)}")
    return work_syms, np.asarray(work_pts, float), ledger
