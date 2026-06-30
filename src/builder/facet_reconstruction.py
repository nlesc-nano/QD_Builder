# src/builder/facet_reconstruction.py
"""
Polar-facet Lannoo reconstruction.

Works directly on the post-passivation structure (no stripping). Ligand bonds count
toward CN in the Lannoo formula, so atoms bonded to ligands appear fully coordinated.
The reconstruction targets residual dangling-bond character that global passivation
could not resolve due to global charge-neutrality constraints.

Algorithm:
  Phase 1 — compute Lannoo facet charges on the passivated structure
  Phase 2 — reconstruct each selected facet greedily, most-charged first:
               anion-rich  (Q<0): remove a cation from the facet sublayer,
                                  then cleanup newly undercoordinated anions
               cation-rich (Q>0): strip cation-bound ligands, then remove a
                                  low-CN surface cation
             after every move: locally passivate only anions bonded to the
                               removed cation
             stop when |Q_facet_Lannoo| stops decreasing
  Phase 3 — one final global charge-balance pass
  Phase 4 — report before/after per facet

Lannoo formula (Harrison 1980, zinc-blende CN_bulk=4):
  cation (formal>0): q_i = formal * (1 - CN/4)    [empty dangling bonds]
  anion  (formal<0): q_i = (8+formal)*(1-CN/4) - (4-CN)*(2 - (8+formal)/4)
                     which simplifies to: q_i = formal * (1 - CN/4)
  → both species: q_i = formal * (1 - min(CN, 4) / 4)
  → fully coordinated (CN≥4): q_i = 0 for any species
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .analysis import PairCuts, _pair_cut_calibrated, coord_numbers_bipartite, derive_pair_cuts_from_cif
from .facets import detect_facets_from_nc
from .nc_types import Facet, FacetReconstructionSpec, SurfaceReconstructionSpec, Plane

CN_BULK = 4
DEFAULT_LIGAND_CN_REF = 3


# --------------------------------------------------------------------------
# Result dataclass
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FacetCharge:
    fid: int
    hkl: Tuple[int, int, int]
    n_surface: int    # atoms in this facet shell, including ligands
    n_active: int     # atoms with non-zero Lannoo charge
    q_formal: float   # split-weighted formal charge on this facet shell
    q_lannoo: float   # split-weighted sum of Lannoo charges
    q_per_active: float
    termination: str  # "anion-rich" | "cation-rich" | "balanced"


@dataclass(frozen=True)
class PolarFacetReport:
    fid: int
    hkl: Tuple[int, int, int]
    n_surface: int
    n_cation_def: int
    n_anion_def: int
    q_cation: float
    q_anion: float
    q_net: float

    @property
    def polarity(self) -> str:
        if self.q_net > 1e-9:
            return "positive"
        if self.q_net < -1e-9:
            return "negative"
        return "balanced"


# --------------------------------------------------------------------------
# Lannoo math
# --------------------------------------------------------------------------

def _cn_ref_for_symbol(sym: str, ligand: str) -> int:
    return DEFAULT_LIGAND_CN_REF if sym == ligand else CN_BULK


def _lannoo_atom_q(formal: int, cn: int, cn_ref: int) -> float:
    """
    Lannoo charge for a surface atom.
    Ligand bonds count toward CN, so ligand-bonded atoms appear fully coordinated.
    Generalized formula: q_i = formal * (1 - min(CN, CN_ref) / CN_ref)
    """
    cn_ref = max(1, int(cn_ref))
    cn_eff = min(cn, cn_ref)
    return float(formal) * (1.0 - cn_eff / cn_ref)


def _lannoo_all_atoms(
    symbols: List[str],
    pts: NDArray[np.float64],
    charges: Dict[str, int],
    ligand: str,
    pair_cuts: Optional[PairCuts],
) -> NDArray[np.float64]:
    """
    Per-atom Lannoo charge array.
    Ligands are included directly with their own CN reference. CN counts bonds
    to ALL opposite-charge neighbors (native + ligands).
    """
    cn = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
    q = np.zeros(len(symbols), dtype=float)
    for i, sym in enumerate(symbols):
        formal = int(charges.get(sym, 0))
        if formal == 0:
            continue
        q[i] = _lannoo_atom_q(formal, int(cn[i]), _cn_ref_for_symbol(sym, ligand))
    return q


def _compute_facet_charges(
    symbols: List[str],
    pts: NDArray[np.float64],
    facets: List[Facet],
    planes: List[Plane],
    charges: Dict[str, int],
    ligand: str,
    surf_tol: float,
    atom_q: NDArray[np.float64],
) -> List[FacetCharge]:
    """Aggregate formal and Lannoo facet charges with split edge weighting."""
    memberships: List[List[int]] = [[] for _ in range(len(pts))]
    for fid, (n, d) in enumerate(planes):
        for i in np.where((d - pts @ n) < surf_tol)[0]:
            memberships[int(i)].append(fid)

    rows: List[FacetCharge] = []
    for fid, (facet, (n, d)) in enumerate(zip(facets, planes)):
        shell_all = [int(i) for i in np.where((d - pts @ n) < surf_tol)[0]]
        if not shell_all:
            continue

        q_formal = 0.0
        for i in shell_all:
            w = 1.0 / max(1, len(memberships[i]))
            q_formal += w * float(charges.get(symbols[i], 0))

        q_total = 0.0
        n_active = 0
        for i in shell_all:
            qi = float(atom_q[i])
            if abs(qi) < 1e-12:
                continue
            n_active += 1
            w = 1.0 / max(1, len(memberships[i]))  # split: edge atoms count once globally
            q_total += w * qi

        q_per_active = q_total / n_active if n_active else 0.0
        term = ("anion-rich" if q_total < -1e-9
                else ("cation-rich" if q_total > 1e-9 else "balanced"))
        rows.append(FacetCharge(
            fid=fid,
            hkl=(facet.h, facet.k, facet.l),
            n_surface=len(shell_all),
            n_active=n_active,
            q_formal=q_formal,
            q_lannoo=q_total,
            q_per_active=q_per_active,
            termination=term,
        ))
    return rows


def _print_lannoo_table(
    rows: List[FacetCharge],
    header: str,
    target_hkls: Set[Tuple[int, int, int]],
) -> None:
    print(f"\n=== LANNOO FACET CHARGES — {header} ===")
    print("  fid      hkl    term          Nsurf Nact  Q_formal  Q_Lannoo  Q/act")
    for r in rows:
        sel = "  <-- selected" if r.hkl in target_hkls else ""
        hkl_str = f"({r.hkl[0]} {r.hkl[1]} {r.hkl[2]})"
        print(
            f"  {r.fid:3d}  {hkl_str:>11s}  {r.termination:<12s}"
            f"  {r.n_surface:4d}  {r.n_active:3d}"
            f"  {r.q_formal:+8.3f}  {r.q_lannoo:+8.3f}  {r.q_per_active:+6.3f}{sel}"
        )


def _print_reconstruction_summary(
    rows_before: List[FacetCharge],
    rows_stripped: List[FacetCharge],
    rows_after: List[FacetCharge],
    rows_final: List[FacetCharge],
    target_hkls: Set[Tuple[int, int, int]],
    strip_log: Dict[Tuple[int, int, int], List[str]],
    move_log: Dict[Tuple[int, int, int], List[str]],
) -> None:
    before_map = {r.hkl: r for r in rows_before}
    stripped_map = {r.hkl: r for r in rows_stripped}
    after_map = {r.hkl: r for r in rows_after}
    final_map = {r.hkl: r for r in rows_final}

    print("\n=== RECONSTRUCTION SUMMARY BY FACET ===")
    print(
        "        hkl    treatment        "
        "Qf_pre Qf_strip Qf_treat Qf_final  "
        "QL_pre QL_strip QL_treat QL_final  stripCl moves"
    )
    for hkl in sorted(target_hkls):
        rb = before_map.get(hkl)
        rs = stripped_map.get(hkl)
        ra = after_map.get(hkl)
        rf = final_map.get(hkl)
        if rb is None:
            continue

        qf_b = rb.q_formal
        qf_s = rs.q_formal if rs else float("nan")
        qf_a = ra.q_formal if ra else float("nan")
        qf_f = rf.q_formal if rf else float("nan")
        ql_b = rb.q_lannoo
        ql_s = rs.q_lannoo if rs else float("nan")
        ql_a = ra.q_lannoo if ra else float("nan")
        ql_f = rf.q_lannoo if rf else float("nan")
        treatment = "cation-vacancy" if ql_b < 0 else "cation-removal"
        hkl_str = f"({hkl[0]} {hkl[1]} {hkl[2]})"
        print(
            f"  {hkl_str:>11s}  {treatment:<15s}"
            f"  {qf_b:+7.3f} {qf_s:+8.3f} {qf_a:+8.3f} {qf_f:+8.3f}"
            f"  {ql_b:+7.3f} {ql_s:+8.3f} {ql_a:+8.3f} {ql_f:+8.3f}"
            f"  {len(strip_log.get(hkl, [])):7d}"
            f"  {len(move_log.get(hkl, [])):5d}"
        )


# --------------------------------------------------------------------------
# Native scaffold utilities
# --------------------------------------------------------------------------

def _native_view(
    symbols: List[str],
    pts: NDArray[np.float64],
    ligand: str,
) -> Tuple[List[str], NDArray[np.float64], List[int]]:
    idx = [i for i, s in enumerate(symbols) if s != ligand]
    return [symbols[i] for i in idx], pts[idx], idx


def _native_facets_and_planes(
    symbols: List[str],
    pts: NDArray[np.float64],
    struct,
    charges: Dict[str, int],
    facet_seeds: List[Facet],
    ligand: str,
    surf_tol: float,
) -> Tuple[List[Facet], List[Plane]]:
    """Detect facets from native scaffold only (no ligands), for stable plane directions."""
    nat_syms, nat_pts, _ = _native_view(symbols, pts, ligand)
    if not nat_syms:
        return [], []
    return detect_facets_from_nc(nat_syms, nat_pts, struct.lattice, charges, facet_seeds, surf_tol)


def _total_q(symbols: List[str], charges: Dict[str, int]) -> int:
    return int(sum(int(charges.get(s, 0)) for s in symbols))


def _hkl_family(hkl: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return tuple(sorted((abs(int(hkl[0])), abs(int(hkl[1])), abs(int(hkl[2])))))


def _native_core_species_from_struct(struct, charges: Dict[str, int], ligand: str) -> Set[str]:
    if struct is not None and hasattr(struct, "sites"):
        species = {str(site.specie.symbol) for site in struct.sites}
    else:
        organic = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
        species = {s for s, q in charges.items() if int(q) != 0 and s not in organic}
    species.discard(ligand)
    return species


def _bulk_ideal_direction_sets(
    site_sym: str,
    bulk_struct,
    charges: Dict[str, int],
) -> List[List[np.ndarray]]:
    """
    Return all distinct first-shell opposite-charge direction sets for a species.

    Do not assume tetrahedral coordination or one crystallographic site per
    element.  If the CIF contains multiple local environments for the same
    species, each environment contributes one candidate direction set.
    """
    if bulk_struct is None or not hasattr(bulk_struct, "sites") or not hasattr(bulk_struct, "lattice"):
        return []

    site_q = int(charges.get(site_sym, 0))
    if site_q == 0:
        return []

    lattice = bulk_struct.lattice
    opp_sites = [
        s for s in bulk_struct.sites
        if int(charges.get(str(s.specie.symbol), 0)) * site_q < 0
    ]
    if not opp_sites:
        return []

    direction_sets: List[List[np.ndarray]] = []
    seen_keys: Set[Tuple[Tuple[float, float, float], ...]] = set()
    for ref_site in bulk_struct.sites:
        if str(ref_site.specie.symbol) != site_sym:
            continue

        ref_cart = np.asarray(ref_site.coords, float)
        candidates: List[Tuple[float, np.ndarray]] = []
        for opp in opp_sites:
            opp_cart = np.asarray(opp.coords, float)
            for ia in range(-1, 2):
                for ib in range(-1, 2):
                    for ic in range(-1, 2):
                        shift = ia * lattice.matrix[0] + ib * lattice.matrix[1] + ic * lattice.matrix[2]
                        vec = opp_cart + shift - ref_cart
                        dist = float(np.linalg.norm(vec))
                        if dist > 0.1:
                            candidates.append((dist, vec))

        if not candidates:
            continue
        candidates.sort(key=lambda rec: rec[0])
        d_min = candidates[0][0]
        dirs: List[np.ndarray] = []
        for dist, vec in candidates:
            if dist >= 1.2 * d_min:
                break
            unit = vec / np.linalg.norm(vec)
            if all(float(np.dot(unit, old)) < 0.99 for old in dirs):
                dirs.append(unit)

        if not dirs:
            continue
        key = tuple(sorted(tuple(np.round(v, 6)) for v in dirs))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        direction_sets.append(dirs)

    return direction_sets


def _bulk_cn_refs_from_struct(
    bulk_struct,
    charges: Dict[str, int],
    species: Set[str],
) -> Dict[str, int]:
    refs: Dict[str, int] = {}
    for sym in species:
        sets = _bulk_ideal_direction_sets(sym, bulk_struct, charges)
        if sets:
            refs[sym] = max(len(dirs) for dirs in sets)
        else:
            refs[sym] = CN_BULK
    return refs


def _surface_recon_atom_q(
    symbols: List[str],
    pts: NDArray[np.float64],
    charges: Dict[str, int],
    pair_cuts: Optional[PairCuts],
    active_species: Set[str],
    cn_refs: Optional[Dict[str, int]] = None,
) -> NDArray[np.float64]:
    """Lannoo-style q_i = formal * (1 - min(CN,CNref)/CNref) for selected core species."""
    cn = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
    q = np.zeros(len(symbols), dtype=float)
    for i, sym in enumerate(symbols):
        if sym not in active_species:
            continue
        formal = int(charges.get(sym, 0))
        if formal == 0:
            continue
        cn_ref = max(1, int((cn_refs or {}).get(sym, CN_BULK)))
        q[i] = float(formal) * (1.0 - min(int(cn[i]), cn_ref) / cn_ref)
    return q


def _facet_memberships(pts: NDArray[np.float64], planes: List[Plane], surf_tol: float) -> List[List[int]]:
    memberships: List[List[int]] = [[] for _ in range(len(pts))]
    for fid, (n, d) in enumerate(planes):
        n = np.asarray(n, float)
        for i in np.where((float(d) - pts @ n) < surf_tol)[0]:
            memberships[int(i)].append(fid)
    return memberships


def _surface_recon_facet_rows(
    symbols: List[str],
    pts: NDArray[np.float64],
    facets: List[Facet],
    planes: List[Plane],
    charges: Dict[str, int],
    surf_tol: float,
    atom_q: NDArray[np.float64],
) -> List[FacetCharge]:
    memberships = _facet_memberships(pts, planes, surf_tol)
    rows: List[FacetCharge] = []
    for fid, (facet, (n, d)) in enumerate(zip(facets, planes)):
        shell = [int(i) for i in np.where((float(d) - pts @ n) < surf_tol)[0]]
        if not shell:
            continue
        q_total = 0.0
        n_active = 0
        q_formal = 0.0
        for i in shell:
            w = 1.0 / max(1, len(memberships[i]))
            q_formal += w * float(charges.get(symbols[i], 0))
            qi = float(atom_q[i])
            if abs(qi) > 1e-12:
                n_active += 1
                q_total += w * qi
        term = "anion-rich" if q_total < -1e-9 else ("cation-rich" if q_total > 1e-9 else "balanced")
        rows.append(FacetCharge(
            fid=fid,
            hkl=(facet.h, facet.k, facet.l),
            n_surface=len(shell),
            n_active=n_active,
            q_formal=q_formal,
            q_lannoo=q_total,
            q_per_active=q_total / n_active if n_active else 0.0,
            termination=term,
        ))
    return rows


def _surface_recon_reports(
    symbols: List[str],
    pts: NDArray[np.float64],
    facets: List[Facet],
    planes: List[Plane],
    charges: Dict[str, int],
    surf_tol: float,
    atom_q: NDArray[np.float64],
) -> List[PolarFacetReport]:
    memberships = _facet_memberships(pts, planes, surf_tol)
    reports: List[PolarFacetReport] = []
    for fid, (facet, (n, d)) in enumerate(zip(facets, planes)):
        shell = [int(i) for i in np.where((float(d) - pts @ n) < surf_tol)[0]]
        if not shell:
            continue

        n_cat = 0
        n_an = 0
        q_cat = 0.0
        q_an = 0.0
        for i in shell:
            qi = float(atom_q[i])
            if abs(qi) < 1e-12:
                continue
            if int(charges.get(symbols[i], 0)) > 0:
                n_cat += 1
            elif int(charges.get(symbols[i], 0)) < 0:
                n_an += 1
            w = 1.0 / max(1, len(memberships[i]))
            if qi > 0:
                q_cat += w * qi
            else:
                q_an += w * qi

        reports.append(PolarFacetReport(
            fid=fid,
            hkl=(facet.h, facet.k, facet.l),
            n_surface=len(shell),
            n_cation_def=n_cat,
            n_anion_def=n_an,
            q_cation=q_cat,
            q_anion=q_an,
            q_net=q_cat + q_an,
        ))
    return reports


def _print_polar_report(
    reports: List[PolarFacetReport],
    header: str,
    target_hkls: Set[Tuple[int, int, int]],
    before: Optional[Dict[Tuple[int, int, int], PolarFacetReport]] = None,
) -> None:
    print(f"\n=== POLAR FACET RESIDUAL CHARGE — {header} ===")
    delta_col = "  ΔQ_net" if before is not None else ""
    print(
        "  fid      hkl    polarity    Nsurf  Cat_def  An_def"
        "    Q_cat    Q_anion    Q_net" + delta_col + "   action"
    )
    for r in reports:
        if r.hkl not in target_hkls and abs(r.q_net) < 1e-9:
            continue
        hkl_str = f"({r.hkl[0]} {r.hkl[1]} {r.hkl[2]})"
        if r.q_net < -1e-9:
            action = "swap anions"
        elif r.q_net > 1e-9:
            action = "add ligands"
        else:
            action = "-"
        sel = " *" if r.hkl in target_hkls else ""
        delta = ""
        if before is not None:
            rb = before.get(r.hkl)
            dq = r.q_net - (rb.q_net if rb is not None else 0.0)
            delta = f"  {dq:+7.3f}"
        print(
            f"  {r.fid:3d}  {hkl_str:>11s}  {r.polarity:<9s}"
            f"  {r.n_surface:5d}  {r.n_cation_def:7d}  {r.n_anion_def:6d}"
            f"  {r.q_cation:+7.3f}  {r.q_anion:+9.3f}  {r.q_net:+7.3f}"
            f"{delta}   {action}{sel}"
        )


def _native_pair_cut(native_species: Set[str], charges: Dict[str, int], pair_cuts: Optional[PairCuts]) -> float:
    native = sorted(s for s in native_species if charges.get(s, 0) != 0)
    best = 0.0
    for i, s1 in enumerate(native):
        for s2 in native[i + 1:]:
            if charges.get(s1, 0) * charges.get(s2, 0) < 0:
                best = max(best, _pair_cut_calibrated(s1, s2, pair_cuts))
    return best if best > 0 else 3.0


def _auto_sublattice_min_separation(
    points: NDArray[np.float64],
    native_bond_cut: float,
) -> float:
    """
    Minimum spacing for reconstruction swaps on one ionic sublattice.

    The native cation-anion cutoff is too short for this purpose: nearest
    anion-anion surface neighbors are second-neighbor distances in the crystal.
    Use the candidate same-sublattice nearest-neighbor distance when available
    and keep a conservative bond-cut based floor for small candidate sets.
    """
    pts = np.asarray(points, float)
    floor = 1.75 * float(native_bond_cut)
    if len(pts) < 2:
        return floor
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    d[d < 1e-8] = np.inf
    nearest = d.min(axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if len(nearest) == 0:
        return floor
    return max(floor, 1.05 * float(np.median(nearest)))


def _fps_indices(
    points: NDArray[np.float64],
    n_pick: int,
    min_separation: float,
    seed: int,
) -> List[int]:
    if n_pick <= 0 or len(points) == 0:
        return []
    rng = np.random.default_rng(seed)
    pts_arr = np.asarray(points, float)
    centroid = pts_arr.mean(axis=0)
    first_pool = np.where(np.linalg.norm(pts_arr - centroid, axis=1) >= 0.0)[0]
    first = int(first_pool[np.argmax(np.linalg.norm(pts_arr[first_pool] - centroid, axis=1))])
    selected = [first]
    remaining = set(range(len(points)))
    remaining.remove(first)
    while remaining and len(selected) < n_pick:
        rem = np.array(sorted(remaining), dtype=int)
        dmin = np.min(np.linalg.norm(pts_arr[rem, None, :] - pts_arr[np.array(selected)][None, :, :], axis=2), axis=1)
        allowed = rem[dmin >= min_separation]
        if len(allowed) == 0:
            break
        allowed_dmin = np.array([dmin[np.where(rem == idx)[0][0]] for idx in allowed])
        max_d = float(np.max(allowed_dmin))
        tied = allowed[np.where(np.abs(allowed_dmin - max_d) < 1e-9)[0]]
        nxt = int(rng.choice(tied))
        selected.append(nxt)
        remaining.remove(nxt)
    return selected


def _greedy_independent_indices(points: NDArray[np.float64], min_separation: float) -> List[int]:
    pts = np.asarray(points, float)
    if len(pts) == 0:
        return []
    centroid = pts.mean(axis=0)
    order = sorted(
        range(len(pts)),
        key=lambda i: (-float(np.linalg.norm(pts[i] - centroid)), i),
    )
    selected: List[int] = []
    for i in order:
        if all(float(np.linalg.norm(pts[i] - pts[j])) >= min_separation for j in selected):
            selected.append(i)
    return selected


def _maximum_independent_indices(
    points: NDArray[np.float64],
    min_separation: float,
) -> List[int]:
    """
    Return the largest non-adjacent subset under the distance constraint.

    For the small per-facet candidate sets typical here, use exact branch and
    bound.  For very large facets, fall back to a deterministic maximal set so
    runtime stays bounded.
    """
    pts = np.asarray(points, float)
    n = len(pts)
    if n <= 1:
        return list(range(n))
    if n > 64:
        return _greedy_independent_indices(pts, min_separation)

    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    adj = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and d[i, j] < min_separation:
                mask |= 1 << j
        adj[i] = mask

    greedy = _greedy_independent_indices(pts, min_separation)
    best_mask = 0
    for i in greedy:
        best_mask |= 1 << i
    best_count = len(greedy)

    def branch(chosen_mask: int, remaining_mask: int) -> None:
        nonlocal best_mask, best_count
        if remaining_mask == 0:
            count = chosen_mask.bit_count()
            if count > best_count:
                best_count = count
                best_mask = chosen_mask
            return
        if chosen_mask.bit_count() + remaining_mask.bit_count() <= best_count:
            return

        rem_indices = [i for i in range(n) if (remaining_mask >> i) & 1]
        v = max(rem_indices, key=lambda i: (adj[i] & remaining_mask).bit_count())

        branch(chosen_mask | (1 << v), remaining_mask & ~(1 << v) & ~adj[v])
        branch(chosen_mask, remaining_mask & ~(1 << v))

    branch(0, (1 << n) - 1)
    return [i for i in range(n) if (best_mask >> i) & 1]


def _surface_outward_direction(idx: int, pts: NDArray[np.float64], planes: List[Plane], surf_tol: float) -> np.ndarray:
    nearest: Optional[Tuple[float, np.ndarray]] = None
    incident: List[np.ndarray] = []
    for n, d in planes:
        n = np.asarray(n, float)
        nn = np.linalg.norm(n)
        if nn > 1e-12:
            n = n / nn
        depth = float(d) - float(np.dot(pts[idx], n))
        if nearest is None or depth < nearest[0]:
            nearest = (depth, n)
        if depth < surf_tol:
            incident.append(n)
    if incident:
        vec = np.sum(incident, axis=0)
        nv = np.linalg.norm(vec)
        if nv > 1e-12:
            return vec / nv
    return nearest[1] if nearest is not None else np.array([0.0, 0.0, 1.0])


def _missing_vectors_for_hosts(
    symbols: List[str],
    pts: NDArray[np.float64],
    host_indices: List[int],
    charges: Dict[str, int],
    pair_cuts: Optional[PairCuts],
    bulk_struct,
) -> Dict[int, List[np.ndarray]]:
    if not host_indices:
        return {}
    try:
        from .neutral_ligand_posttreat import compute_missing_bond_vectors
        mask = np.zeros(len(symbols), dtype=bool)
        for i in host_indices:
            mask[i] = True
        return compute_missing_bond_vectors(symbols, pts, charges, pair_cuts, bulk_struct, mask)
    except Exception:
        return {}


def _actual_opposite_bond_vectors(
    symbols: List[str],
    pts: NDArray[np.float64],
    host_idx: int,
    charges: Dict[str, int],
    pair_cuts: Optional[PairCuts],
) -> List[np.ndarray]:
    host_sym = symbols[host_idx]
    host_q = int(charges.get(host_sym, 0))
    if host_q == 0:
        return []
    vecs: List[np.ndarray] = []
    for j, sym_j in enumerate(symbols):
        if j == host_idx:
            continue
        if int(charges.get(sym_j, 0)) * host_q >= 0:
            continue
        cutoff = _pair_cut_calibrated(host_sym, sym_j, pair_cuts)
        vec = np.asarray(pts[j], float) - np.asarray(pts[host_idx], float)
        dist = float(np.linalg.norm(vec))
        if 0.1 < dist <= cutoff:
            vecs.append(vec / dist)
    return vecs


def _match_missing_ideal_dirs(
    actual_vecs: List[np.ndarray],
    ideal_dirs: List[np.ndarray],
    *,
    min_dot: float = 0.70,
) -> Tuple[List[np.ndarray], float]:
    assigned: Set[int] = set()
    score = 0.0
    for actual in actual_vecs:
        actual = np.asarray(actual, float)
        if np.linalg.norm(actual) < 1e-12:
            continue
        actual = actual / np.linalg.norm(actual)
        best_idx = -1
        best_dot = -2.0
        for k, ideal in enumerate(ideal_dirs):
            if k in assigned:
                continue
            dot = float(np.dot(actual, ideal))
            if dot > best_dot:
                best_idx = k
                best_dot = dot
        if best_idx >= 0 and best_dot >= min_dot:
            assigned.add(best_idx)
            score += best_dot
        else:
            score -= 1.0
    missing = [np.asarray(ideal_dirs[k], float) for k in range(len(ideal_dirs)) if k not in assigned]
    score -= 0.25 * abs(len(missing) - max(0, len(ideal_dirs) - len(actual_vecs)))
    return missing, score


def _strict_missing_vectors_for_hosts(
    symbols: List[str],
    pts: NDArray[np.float64],
    host_indices: List[int],
    charges: Dict[str, int],
    pair_cuts: Optional[PairCuts],
    bulk_struct,
    planes: List[Plane],
    surf_tol: float,
) -> Dict[int, List[np.ndarray]]:
    """
    Missing first-shell directions from the bulk coordination polyhedron.

    This intentionally has no radial/outward fallback and never flips a vector:
    if a crystallographic missing slot cannot be identified, the host is not
    used for reconstruction ligand compensation.
    """
    direction_cache: Dict[str, List[List[np.ndarray]]] = {}
    result: Dict[int, List[np.ndarray]] = {}
    for host_idx in host_indices:
        host_sym = symbols[host_idx]
        if host_sym not in direction_cache:
            direction_cache[host_sym] = _bulk_ideal_direction_sets(host_sym, bulk_struct, charges)
        direction_sets = direction_cache[host_sym]
        if not direction_sets:
            continue

        actual = _actual_opposite_bond_vectors(symbols, pts, host_idx, charges, pair_cuts)
        if not actual:
            continue

        best_missing: List[np.ndarray] = []
        best_score = -float("inf")
        for ideal_dirs in direction_sets:
            missing, score = _match_missing_ideal_dirs(actual, ideal_dirs)
            if score > best_score:
                best_score = score
                best_missing = missing

        if not best_missing:
            continue

        outward = _surface_outward_direction(host_idx, pts, planes, surf_tol)
        outward_slots = []
        for vec in best_missing:
            vec = np.asarray(vec, float)
            norm = np.linalg.norm(vec)
            if norm < 1e-12:
                continue
            vec = vec / norm
            if float(np.dot(vec, outward)) > 0.05:
                outward_slots.append(vec)
        if outward_slots:
            outward_slots.sort(key=lambda v: float(np.dot(v, outward)), reverse=True)
            result[host_idx] = outward_slots
    return result


def _ligand_add_positions_for_slots(
    symbols: List[str],
    pts: NDArray[np.float64],
    slots: List[Tuple[int, np.ndarray]],
    ligand: str,
    pair_cuts: Optional[PairCuts],
) -> List[np.ndarray]:
    positions: List[np.ndarray] = []
    for host_idx, vec in slots:
        vec = np.asarray(vec, float)
        if np.linalg.norm(vec) < 1e-12:
            continue
        vec = vec / np.linalg.norm(vec)
        host = symbols[host_idx]
        bond_len = 0.84 * _pair_cut_calibrated(host, ligand, pair_cuts)
        positions.append(np.asarray(pts[host_idx], float) + bond_len * vec)
    return positions


def _slot_points(slots: List[Tuple[int, np.ndarray]], pts: NDArray[np.float64]) -> NDArray[np.float64]:
    if not slots:
        return np.zeros((0, 3), float)
    return np.asarray([pts[host_idx] for host_idx, _ in slots], float)


def _choose_missing_vectors_for_host(
    host_idx: int,
    n_slots: int,
    missing: Dict[int, List[np.ndarray]],
    pts: NDArray[np.float64],
    planes: List[Plane],
    surf_tol: float,
) -> List[np.ndarray]:
    outward = _surface_outward_direction(host_idx, pts, planes, surf_tol)
    vecs = missing.get(host_idx) or [outward]
    cleaned = []
    for v in vecs:
        v = np.asarray(v, float)
        if np.linalg.norm(v) < 1e-12:
            continue
        v = v / np.linalg.norm(v)
        if float(np.dot(v, outward)) < 0.0:
            v = -v
        cleaned.append(v)
    if not cleaned:
        cleaned = [outward]
    cleaned.sort(key=lambda v: float(np.dot(v, outward)), reverse=True)
    while len(cleaned) < n_slots:
        cleaned.append(outward)
    return cleaned[:n_slots]


def _build_compensation_slots(
    symbols: List[str],
    pts: NDArray[np.float64],
    host_indices: List[int],
    cn: NDArray[np.int_],
    cn_refs: Dict[str, int],
    missing: Dict[int, List[np.ndarray]],
    planes: List[Plane],
    surf_tol: float,
) -> List[Tuple[int, np.ndarray]]:
    slots: List[Tuple[int, np.ndarray]] = []
    for host_idx in host_indices:
        cn_ref = max(1, int(cn_refs.get(symbols[host_idx], CN_BULK)))
        deficit = max(0, cn_ref - int(cn[host_idx]))
        if deficit <= 0:
            continue
        vectors = missing.get(host_idx, [])
        for vec in vectors[:deficit]:
            slots.append((host_idx, vec))
    return slots


def _select_compensation_slots(
    slots: List[Tuple[int, np.ndarray]],
    pts: NDArray[np.float64],
    n_needed: int,
    min_separation: float,
    seed: int,
) -> List[Tuple[int, np.ndarray]]:
    if n_needed <= 0:
        return []
    rng = np.random.default_rng(seed)
    available = list(slots)
    selected: List[Tuple[int, np.ndarray]] = []
    centroid = pts[[host for host, _ in available]].mean(axis=0) if available else np.zeros(3)

    while available and len(selected) < n_needed:
        allowed: List[Tuple[int, np.ndarray]] = []
        for slot in available:
            host_idx = slot[0]
            host_pos = pts[host_idx]
            ok = True
            for selected_host, _ in selected:
                if host_idx == selected_host:
                    continue
                if float(np.linalg.norm(host_pos - pts[selected_host])) < min_separation:
                    ok = False
                    break
            if ok:
                allowed.append(slot)
        if not allowed:
            break

        if selected:
            selected_hosts = np.asarray([pts[host_idx] for host_idx, _ in selected], float)
            scored = []
            for slot in allowed:
                host_idx, vec = slot
                dist_to_selected = float(np.min(np.linalg.norm(selected_hosts - pts[host_idx], axis=1)))
                radial = float(np.linalg.norm(pts[host_idx] - centroid))
                outward = float(np.linalg.norm(np.asarray(vec, float)))
                scored.append((dist_to_selected, radial, outward, -host_idx, slot))
            best_dist = max(s[0] for s in scored)
            tied = [s for s in scored if abs(s[0] - best_dist) < 1e-9]
            chosen = tied[int(rng.integers(len(tied)))][-1]
        else:
            scored = [
                (float(np.linalg.norm(pts[slot[0]] - centroid)), float(np.linalg.norm(np.asarray(slot[1], float))), -slot[0], slot)
                for slot in allowed
            ]
            best_radial = max(s[0] for s in scored)
            tied = [s for s in scored if abs(s[0] - best_radial) < 1e-9]
            chosen = tied[int(rng.integers(len(tied)))][-1]

        selected.append(chosen)
        available.remove(chosen)

    return selected


def _ligand_add_positions_for_cations(
    symbols: List[str],
    pts: NDArray[np.float64],
    cation_indices: List[int],
    charges: Dict[str, int],
    ligand: str,
    pair_cuts: Optional[PairCuts],
    planes: List[Plane],
    surf_tol: float,
    bulk_struct,
) -> List[np.ndarray]:
    if not cation_indices:
        return []
    missing = _missing_vectors_for_hosts(symbols, pts, cation_indices, charges, pair_cuts, bulk_struct)

    positions: List[np.ndarray] = []
    for i in cation_indices:
        outward = _surface_outward_direction(i, pts, planes, surf_tol)
        vecs = missing.get(i) or [outward]
        vecs = [v / np.linalg.norm(v) for v in vecs if np.linalg.norm(v) > 1e-12]
        if not vecs:
            vecs = [outward]
        vecs.sort(key=lambda v: float(np.dot(v, outward)), reverse=True)
        vec = vecs[0]
        if float(np.dot(vec, outward)) < 0:
            vec = -vec
        host = symbols[i]
        bond_len = 0.84 * _pair_cut_calibrated(host, ligand, pair_cuts)
        positions.append(np.asarray(pts[i], float) + bond_len * vec)
    return positions


# --------------------------------------------------------------------------
# Phase 2: greedy per-facet reconstruction
# --------------------------------------------------------------------------

def _facet_q_lannoo_full(
    symbols: List[str],
    pts: NDArray[np.float64],
    n: NDArray[np.float64],
    d: float,
    atom_q: NDArray[np.float64],
    ligand: str,
    surf_tol: float,
) -> float:
    """Full-weight (no split) Lannoo Q for one facet — used as greedy stopping criterion."""
    shell = np.where((d - pts @ n) < surf_tol)[0]
    return float(sum(atom_q[int(i)] for i in shell))


def _remove_cation_and_cleanup_bonded_anions(
    symbols: List[str],
    pts: NDArray[np.float64],
    idx: int,
    cn_value: int,
    charges: Dict[str, int],
    ligand: str,
    pair_cuts: Optional[PairCuts],
) -> Tuple[List[str], NDArray[np.float64], List[str]]:
    """
    Remove one cation vacancy, then only passivate native anions that were
    bonded to that removed cation and became undercoordinated.
    """
    old = symbols[idx]
    xyz = pts[idx].copy()

    bonded_anions = []
    for j, sym in enumerate(symbols):
        if j == idx or sym == ligand or int(charges.get(sym, 0)) >= 0:
            continue
        cutoff = _pair_cut_calibrated(old, sym, pair_cuts)
        if float(np.linalg.norm(pts[j] - xyz)) <= cutoff:
            bonded_anions.append(j)

    symbols.pop(idx)
    pts = np.delete(pts, idx, axis=0)

    shifted = [j if j < idx else j - 1 for j in bonded_anions]
    logs = [f"remove {old}#{idx}(CN={cn_value}, xyz=[{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}])"]

    for j in shifted:
        if j < 0 or j >= len(symbols) or symbols[j] == ligand:
            continue
        cn_after = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
        if int(charges.get(symbols[j], 0)) < 0 and int(cn_after[j]) < CN_BULK:
            old_anion = symbols[j]
            symbols[j] = ligand
            logs.append(f"local cleanup: {old_anion}#{j}(CN={int(cn_after[j])})→{ligand}")

    return symbols, pts, logs


def _strip_ligands_from_cation_rich_facets(
    symbols: List[str],
    pts: NDArray[np.float64],
    rows: List[FacetCharge],
    planes: List[Plane],
    charges: Dict[str, int],
    ligand: str,
    target_hkls: Set[Tuple[int, int, int]],
    surf_tol: float,
    pair_cuts: Optional[PairCuts],
    verbose: bool,
) -> Tuple[List[str], NDArray[np.float64], Dict[Tuple[int, int, int], List[str]]]:
    """
    Remove all anion ligands attached to cations on selected cation-rich facets
    before any vacancy reconstruction is attempted.
    """
    cation_rows = [r for r in rows if r.hkl in target_hkls and r.q_lannoo > 1e-9]
    strip_log: Dict[Tuple[int, int, int], List[str]] = {r.hkl: [] for r in cation_rows}
    if not cation_rows:
        return symbols, pts, strip_log

    ligand_indices = [i for i, s in enumerate(symbols) if s == ligand]
    if not ligand_indices:
        return symbols, pts, strip_log

    remove_to_hkl: Dict[int, Tuple[int, int, int]] = {}
    for row in cation_rows:
        n, d = planes[row.fid]
        facet_cations = [
            int(i)
            for i in np.where((d - pts @ n) < surf_tol)[0]
            if symbols[int(i)] != ligand and int(charges.get(symbols[int(i)], 0)) > 0
        ]
        for li in ligand_indices:
            if li in remove_to_hkl:
                continue
            for ci in facet_cations:
                cutoff = _pair_cut_calibrated(ligand, symbols[ci], pair_cuts)
                if float(np.linalg.norm(pts[li] - pts[ci])) <= cutoff:
                    remove_to_hkl[li] = row.hkl
                    strip_log[row.hkl].append(
                        f"strip {ligand}#{li} attached to {symbols[ci]}#{ci}"
                    )
                    break

    if not remove_to_hkl:
        if verbose:
            print("[recon-strip] no cation-rich facet ligands found to strip.")
        return symbols, pts, strip_log

    for li in sorted(remove_to_hkl, reverse=True):
        if verbose:
            hkl = remove_to_hkl[li]
            print(f"[recon-strip] {hkl}: remove {ligand}#{li} before vacancy treatment")
        symbols.pop(li)
        pts = np.delete(pts, li, axis=0)

    return symbols, pts, strip_log


def _candidate_spacing(candidates: List[Tuple[int, float, int]], pts: NDArray[np.float64]) -> float:
    """Nearest Cd-Cd spacing among facet candidates, expanded slightly to forbid adjacency."""
    if len(candidates) < 2:
        return 0.0
    idx = [rec[2] for rec in candidates]
    dmin = float("inf")
    for a_pos, i in enumerate(idx):
        for j in idx[a_pos + 1:]:
            d = float(np.linalg.norm(pts[i] - pts[j]))
            if 1e-9 < d < dmin:
                dmin = d
    return 1.05 * dmin if np.isfinite(dmin) else 0.0


def _allowed_by_vacancy_spacing(
    x: NDArray[np.float64],
    vacancies: List[NDArray[np.float64]],
    min_dist: float,
) -> Tuple[bool, float]:
    if not vacancies:
        return True, float("inf")
    nearest = min(float(np.linalg.norm(x - v)) for v in vacancies)
    return nearest >= min_dist, nearest


def _facet_cation_candidates(
    symbols: List[str],
    pts: NDArray[np.float64],
    cn: NDArray[np.int_],
    n: NDArray[np.float64],
    d_fixed: float,
    charges: Dict[str, int],
    ligand: str,
    surf_tol: float,
    original_q_lannoo: float,
) -> List[Tuple[int, float, int]]:
    shell = [int(i) for i in np.where((d_fixed - pts @ n) < surf_tol)[0]]
    if original_q_lannoo < 0:
        outer_thr = 0.35 * surf_tol
        cands = [
            (int(cn[i]), float(d_fixed - np.dot(pts[i], n)), i)
            for i in shell
            if symbols[i] != ligand
            and int(charges.get(symbols[i], 0)) > 0
            and float(d_fixed - np.dot(pts[i], n)) >= outer_thr
            and int(cn[i]) <= CN_BULK
        ]
        if cands:
            return cands
        return [
            (int(cn[i]), float(d_fixed - np.dot(pts[i], n)), i)
            for i in shell
            if symbols[i] != ligand
            and int(charges.get(symbols[i], 0)) > 0
            and int(cn[i]) <= CN_BULK
        ]

    return [
        (int(cn[i]), float(d_fixed - np.dot(pts[i], n)), i)
        for i in shell
        if symbols[i] != ligand
        and int(charges.get(symbols[i], 0)) > 0
        and int(cn[i]) < CN_BULK
    ]


def _reconstruct_facet_spaced(
    symbols: List[str],
    pts: NDArray[np.float64],
    hkl: Tuple[int, int, int],
    n: NDArray[np.float64],
    d: float,
    planes: List[Plane],
    charges: Dict[str, int],
    ligand: str,
    surf_tol: float,
    pair_cuts: Optional[PairCuts],
    verbose: bool,
    original_q_lannoo: float,
) -> Tuple[List[str], NDArray[np.float64], List[str]]:
    """
    Apply spaced local cation-vacancy events on one facet. Neighboring Cd
    vacancies are forbidden, and each accepted event must reduce |Q_Lannoo|.
    """
    moves: List[str] = []
    d_fixed = float(d)
    vacancy_coords: List[NDArray[np.float64]] = []

    while True:
        cn = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
        atom_q = _lannoo_all_atoms(symbols, pts, charges, ligand, pair_cuts)
        q_facet = _facet_q_lannoo_full(symbols, pts, n, d_fixed, atom_q, ligand, surf_tol)
        cands = _facet_cation_candidates(
            symbols, pts, cn, n, d_fixed, charges, ligand, surf_tol, original_q_lannoo
        )
        if not cands:
            if verbose:
                print(f"[recon] {hkl}: no Cd candidates remain.")
            return symbols, pts, moves

        min_spacing = _candidate_spacing(cands, pts)
        trials = []
        for cn_i, depth_i, idx_i in cands:
            allowed, nearest = _allowed_by_vacancy_spacing(pts[idx_i], vacancy_coords, min_spacing)
            if not allowed:
                continue
            trial_symbols = list(symbols)
            trial_pts = pts.copy()
            trial_symbols, trial_pts, trial_logs = _remove_cation_and_cleanup_bonded_anions(
                trial_symbols, trial_pts, idx_i, cn_i, charges, ligand, pair_cuts
            )
            trial_q = _facet_q_lannoo_full(
                trial_symbols,
                trial_pts,
                n,
                d_fixed,
                _lannoo_all_atoms(trial_symbols, trial_pts, charges, ligand, pair_cuts),
                ligand,
                surf_tol,
            )
            improvement = abs(q_facet) - abs(trial_q)
            if improvement <= 1e-9:
                continue
            trials.append((nearest, improvement, -cn_i, -depth_i, idx_i, cn_i, depth_i, trial_q, trial_logs))

        if not trials:
            if verbose:
                print(
                    f"[recon] {hkl}: stop; no non-adjacent Cd vacancy improves "
                    f"|Q_Lannoo|={abs(q_facet):.3f}."
                )
            return symbols, pts, moves

        if vacancy_coords:
            trials.sort(reverse=True)  # farthest first, then best improvement
        else:
            trials.sort(key=lambda rec: (rec[1], rec[2], rec[3]), reverse=True)

        nearest, improvement, _neg_cn, _neg_depth, best_i, best_cn, best_depth, q_new, _logs = trials[0]
        vacancy_xyz = pts[best_i].copy()
        symbols, pts, move_logs = _remove_cation_and_cleanup_bonded_anions(
            symbols, pts, best_i, best_cn, charges, ligand, pair_cuts
        )
        vacancy_coords.append(vacancy_xyz)
        nearest_txt = "none" if not np.isfinite(nearest) else f"{nearest:.2f} Å"
        move_logs[0] += (
            f" | vacancy_spacing_min={min_spacing:.2f} Å nearest_prev={nearest_txt}"
            f" | Q_Lannoo {q_facet:+.3f}→{q_new:+.3f}"
        )
        if original_q_lannoo < 0:
            move_logs[0] += f" | anion-rich sublayer depth={best_depth:.2f} Å"

        for move_str in move_logs:
            if verbose:
                print(f"[recon] {hkl}: {move_str}")
            moves.append(move_str)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def reconstruct_polar_facets(
    symbols: List[str],
    pts: NDArray[np.float64],
    *,
    struct,
    facet_seeds: List[Facet],
    charges: Dict[str, int],
    ligand: str,
    surf_tol: float,
    cif_path: str,
    spec: FacetReconstructionSpec | SurfaceReconstructionSpec,
    charge_balance_fn,
    verbose: bool,
    write_all: bool,
    prefix: str,
) -> Tuple[List[str], NDArray[np.float64]]:
    """
    Simplified polar-facet reconstruction post-treatment.

    The reconstruction runs after ordinary charge-balance passivation.  It
    computes residual polar-facet charge from CN-deficient native ions, swaps a
    sparse subset of native anions on negative polar facets to the reconstruction
    ligand, and compensates each swap by adding one ligand to an available
    positive polar surface site.
    """
    if not spec.enabled:
        return symbols, pts

    recon_ligand = getattr(spec, "ligand", None) or ligand
    configured_hkls: Set[Tuple[int, int, int]] = set(getattr(spec, "facets", ()) or ())
    auto_facets = bool(getattr(spec, "auto_facets", False) or not configured_hkls)
    target_reduction = float(getattr(spec, "target_reduction", 0.5))
    seed = int(getattr(spec, "seed", 1337))
    pair_cuts = derive_pair_cuts_from_cif(cif_path, charges, safety=1.00)

    print(f"\n{'='*60}")
    print("[post-treatment:surface-reconstruction] Simplified polar reconstruction")
    print(f"[recon] ligand={recon_ligand!r}  facets={'auto' if auto_facets else sorted(configured_hkls)}  "
          f"target_reduction={target_reduction:.2f}")
    print(f"[recon] Q_total = {_total_q(symbols, charges):+d}  (expected 0 after charge-balance passivation)")

    native_species = _native_core_species_from_struct(struct, charges, recon_ligand)
    if not native_species:
        print("[recon] No native inorganic species found; skipping.")
        print('='*60)
        return symbols, pts

    # Detect native scaffold planes (geometry only; defines atom membership).
    all_facets, all_planes = _native_facets_and_planes(
        symbols, pts, struct, charges, facet_seeds, recon_ligand, surf_tol
    )
    if not all_planes:
        print("[recon] No facets detected on native scaffold; skipping.")
        print('='*60)
        return symbols, pts

    active_species = set(native_species)
    cn_refs = _bulk_cn_refs_from_struct(struct, charges, active_species)
    atom_q = _surface_recon_atom_q(symbols, pts, charges, pair_cuts, active_species, cn_refs)
    reports_before = _surface_recon_reports(
        symbols, pts, all_facets, all_planes, charges, surf_tol, atom_q
    )
    configured_families = {_hkl_family(hkl) for hkl in configured_hkls} if configured_hkls else set()
    target_hkls = {
        r.hkl
        for r in reports_before
        if abs(r.q_net) > 1e-9 and (auto_facets or _hkl_family(r.hkl) in configured_families)
    }
    if not target_hkls:
        print("[recon] No polar facets with residual CN-deficit charge found; skipping.")
        print('='*60)
        return symbols, pts
    print(f"[recon] Treating polar facets: {sorted(target_hkls)}")
    _print_polar_report(reports_before, "BEFORE reconstruction", target_hkls)

    memberships = _facet_memberships(pts, all_planes, surf_tol)
    cn = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)

    neg_fids = {
        r.fid for r in reports_before
        if r.hkl in target_hkls and r.q_net < -1e-9
    }
    pos_fids = {
        r.fid for r in reports_before
        if r.hkl in target_hkls and r.q_net > +1e-9
    }

    anion_candidates: List[int] = []
    cation_candidates: List[int] = []
    for i, sym in enumerate(symbols):
        if sym not in native_species:
            continue
        q_formal = int(charges.get(sym, 0))
        cn_ref = max(1, int(cn_refs.get(sym, CN_BULK)))
        if q_formal == 0 or int(cn[i]) >= cn_ref:
            continue
        fids = set(memberships[i])
        if q_formal < 0 and fids & neg_fids:
            anion_candidates.append(i)
        elif q_formal > 0 and fids & pos_fids:
            cation_candidates.append(i)

    if not anion_candidates:
        print("[recon] No CN-deficient native anions found on negative polar facets; skipping.")
        print('='*60)
        return symbols, pts
    if not cation_candidates:
        print("[recon] No available cation-rich sites found for compensating ligand addition; skipping.")
        print('='*60)
        return symbols, pts

    ligand_charge = int(charges.get(recon_ligand, -1))
    if ligand_charge >= 0:
        print(
            f"[recon] Reconstruction ligand {recon_ligand!r} has charge {ligand_charge:+d}; "
            "negative ligand charge is required for anion-swap compensation. Skipping."
        )
        print('='*60)
        return symbols, pts
    add_charge_unit = abs(ligand_charge)
    replacement_delta: Dict[int, float] = {}
    swap_charge_delta: Dict[int, int] = {}
    for i in anion_candidates:
        formal_delta = ligand_charge - int(charges.get(symbols[i], 0))
        if formal_delta <= 0:
            continue
        old_q = float(atom_q[i])
        cn_ref = max(1, int(cn_refs.get(symbols[i], CN_BULK)))
        lig_q = float(ligand_charge) * (1.0 - min(int(cn[i]), cn_ref) / cn_ref)
        replacement_delta[i] = max(0.05, lig_q - old_q)
        swap_charge_delta[i] = int(formal_delta)
    anion_candidates = [i for i in anion_candidates if i in swap_charge_delta]
    if not anion_candidates:
        print(
            f"[recon] No anion swaps produce positive compensation charge with ligand {recon_ligand!r}; skipping."
        )
        print('='*60)
        return symbols, pts

    native_cut = _native_pair_cut(native_species, charges, pair_cuts)
    min_separation = getattr(spec, "min_separation", None)
    if min_separation is None:
        min_separation = _auto_sublattice_min_separation(pts[anion_candidates], native_cut)
    min_separation = float(min_separation)

    cation_missing = _strict_missing_vectors_for_hosts(
        symbols,
        pts,
        cation_candidates,
        charges,
        pair_cuts,
        struct,
        all_planes,
        surf_tol,
    )
    compensation_slots = _build_compensation_slots(
        symbols,
        pts,
        cation_candidates,
        cn,
        cn_refs,
        cation_missing,
        all_planes,
        surf_tol,
    )
    capacity_slots = compensation_slots
    compensation_capacity = len(capacity_slots) * add_charge_unit
    if compensation_capacity <= 0:
        print("[recon] No positive-facet missing coordination slots available for compensation; skipping.")
        print('='*60)
        return symbols, pts

    reports_by_fid = {r.fid: r for r in reports_before}
    selected_anions: List[int] = []
    plan_rows: List[Tuple[Tuple[int, int, int], float, int, int, int, int, int]] = []
    neg_reports = sorted(
        [reports_by_fid[fid] for fid in neg_fids],
        key=lambda r: abs(r.q_net),
        reverse=True,
    )

    # 1. Pre-calculate candidates symmetrically for all negative facets
    facet_data = {}
    for report in neg_reports:
        local_candidates = [
            i for i in anion_candidates
            if report.fid in memberships[i]
        ]
        if not local_candidates:
            facet_data[report.fid] = {
                "local_candidates": [],
                "max_nonadjacent": [],
            }
            continue
        
        max_local_idx = _maximum_independent_indices(pts[local_candidates], min_separation)
        max_nonadjacent = [local_candidates[k] for k in max_local_idx]
        
        facet_data[report.fid] = {
            "local_candidates": local_candidates,
            "max_nonadjacent": max_nonadjacent,
        }

    # Group the negative facets by their HKL family
    from collections import defaultdict
    family_groups = defaultdict(list)
    for report in neg_reports:
        fam = _hkl_family(report.hkl)
        family_groups[fam].append(report)

    # Define capacity C
    C = len(compensation_slots)

    # Find the slot requirement ratio per swap to adjust S_max
    first_idx = anion_candidates[0] if anion_candidates else None
    charge_ratio = (swap_charge_delta[first_idx] // add_charge_unit) if first_idx is not None else 1
    charge_ratio = max(1, charge_ratio)

    # Save target swaps for each facet using the capacity-bounded formula
    final_target_swaps = {}
    for fam, group in family_groups.items():
        n_facets = len(group)
        # Find the minimum candidate count among the equivalent facets in this family
        A_min = len(facet_data[group[0].fid]["local_candidates"])
        for r in group:
            A_min = min(A_min, len(facet_data[r.fid]["local_candidates"]))
        
        # S_max is the absolute capacity-limited maximum swaps per negative polar facet
        S_max = min(A_min, C // (n_facets * charge_ratio))
        
        # S_target is scaled by the target ratio
        S_target = int(np.round(S_max * target_reduction))
        S_target = max(0, min(S_max, S_target))
        
        for r in group:
            final_target_swaps[r.fid] = S_target

    # 2. Dynamic Iterative Batch Passivation & Reconstruction Loop
    applied_swaps = {report.fid: [] for report in neg_reports}
    picked_slots = []
    
    while True:
        # Check active facets that still need swaps
        active_facets = [
            report for report in neg_reports
            if len(applied_swaps[report.fid]) < final_target_swaps[report.fid]
        ]
        if not active_facets:
            break

        # Symmetrically select 1 swap candidate per active facet in each family
        batch_swaps = []
        batch_selected_anions = set()
        for report in active_facets:
            fdata = facet_data[report.fid]
            target = final_target_swaps[report.fid]
            max_nonadjacent = fdata["max_nonadjacent"]
            local_candidates = fdata["local_candidates"]
            
            # Pool: if we are within non-adjacent limits, use max_nonadjacent; otherwise, relax to local_candidates
            if target <= len(max_nonadjacent):
                pool = max_nonadjacent
            else:
                pool = local_candidates
                
            remaining_pool = [i for i in pool if i not in selected_anions and i not in batch_selected_anions]
            if not remaining_pool:
                continue
                
            # Farthest Point Selection relative to already swapped sites on this facet
            swapped_on_facet = applied_swaps[report.fid]
            if not swapped_on_facet:
                # Pick point farthest from centroid of the facet
                facet_pts = pts[local_candidates]
                centroid = facet_pts.mean(axis=0) if len(facet_pts) > 0 else pts.mean(axis=0)
                best_cand = max(remaining_pool, key=lambda idx: float(np.linalg.norm(pts[idx] - centroid)))
            else:
                # Farthest point from already swapped points
                best_cand = max(
                    remaining_pool,
                    key=lambda idx: min(float(np.linalg.norm(pts[idx] - pts[s])) for s in swapped_on_facet)
                )
            
            batch_swaps.append((report.fid, best_cand))
            batch_selected_anions.add(best_cand)

        if not batch_swaps:
            break

        # Calculate compensating ligand additions for this batch
        total_batch_charge = sum(swap_charge_delta[idx] for _, idx in batch_swaps)
        needed_added_ligands = total_batch_charge // add_charge_unit
        
        # Select from remaining capacity slots
        available_slots = [
            slot for slot in capacity_slots
            if slot not in picked_slots
        ]
        if len(available_slots) < needed_added_ligands:
            break
            
        # Select slots using 3D FPS relative to already picked slots
        batch_slots = []
        for _ in range(needed_added_ligands):
            rem_slots = [s for s in available_slots if s not in batch_slots]
            if not rem_slots:
                break
            if not picked_slots and not batch_slots:
                # First point: pick one with highest outwards radial distance
                centroid = pts.mean(axis=0)
                best_slot = max(rem_slots, key=lambda s: float(np.linalg.norm(pts[s[0]] - centroid)))
            else:
                # Pick slot farthest from already selected slots
                ref_pts = np.asarray([pts[s[0]] for s in (picked_slots + batch_slots)], float)
                best_slot = max(
                    rem_slots,
                    key=lambda s: float(np.min(np.linalg.norm(ref_pts - pts[s[0]], axis=1)))
                )
            batch_slots.append(best_slot)

        if len(batch_slots) < needed_added_ligands:
            break

        # Commit batch!
        for fid, idx in batch_swaps:
            applied_swaps[fid].append(idx)
            selected_anions.append(idx)
        picked_slots.extend(batch_slots)

    # Populate final plan_rows for reporting
    for report in neg_reports:
        fdata = facet_data[report.fid]
        applied = len(applied_swaps[report.fid])
        chosen_charge = applied * add_charge_unit
        max_charge = len(fdata["max_nonadjacent"]) * add_charge_unit
        plan_rows.append((
            report.hkl,
            report.q_net,
            final_target_swaps[report.fid],
            len(fdata["max_nonadjacent"]),
            max_charge,
            chosen_charge,
            applied,
        ))

    selected_charge_delta = sum(swap_charge_delta[idx] for idx in selected_anions)
    picked_anions = selected_anions
    needed_added_ligands = selected_charge_delta // add_charge_unit

    if not picked_anions:
        print(
            f"[recon] Spacing constraint rejected all anion swaps "
            f"(min_separation={min_separation:.2f} Å); skipping."
        )
        print('='*60)
        return symbols, pts

    print(f"\n=== SURFACE RECONSTRUCTION PLAN ===")
    print(
        "        hkl    Q_net_before  requested  max_nonadjacent"
        "  max_ΔQ  selected_ΔQ  applied"
    )
    for hkl, q_before, requested, max_allowed, max_charge, chosen_charge, applied in plan_rows:
        hkl_str = f"({hkl[0]} {hkl[1]} {hkl[2]})"
        print(
            f"  {hkl_str:>11s}  {q_before:+12.3f}"
            f"  {requested:9d}  {max_allowed:15d}"
            f"  {max_charge:6d}  {chosen_charge:11d}  {applied:7d}"
        )
    print(
        f"[recon] compensation capacity: {len(capacity_slots)} missing-coordination "
        f"{recon_ligand} slots "
        f"= {-compensation_capacity:+d} charge"
    )
    print(
        f"[recon] selected swaps: +{selected_charge_delta:d} charge; "
        f"adding {needed_added_ligands:d} {recon_ligand} ligands "
        f"({needed_added_ligands * ligand_charge:+d} charge)"
    )
    print(
        f"[recon] applied total swaps={len(picked_anions)}, "
        f"min_separation={min_separation:.2f} Å"
    )

    new_symbols = list(symbols)
    new_pts = np.asarray(pts, float).copy()
    for idx in picked_anions:
        old = new_symbols[idx]
        new_symbols[idx] = recon_ligand
        if verbose:
            print(f"[recon] swap {old}#{idx} CN={int(cn[idx])} -> {recon_ligand}")

    add_positions = _ligand_add_positions_for_slots(
        new_symbols, new_pts, picked_slots, recon_ligand, pair_cuts
    )
    if len(add_positions) != len(picked_slots):
        print(
            f"[recon] Internal slot geometry issue: generated {len(add_positions)} ligand positions "
            f"for {len(picked_slots)} selected slots; skipping."
        )
        print('='*60)
        return symbols, pts
    for (host_idx, _vec), pos in zip(picked_slots, add_positions):
        new_symbols.append(recon_ligand)
        new_pts = np.vstack([new_pts, np.asarray(pos, float)])
        if verbose:
            print(f"[recon] add {recon_ligand} on {symbols[host_idx]}#{host_idx}")

    symbols, pts = new_symbols, new_pts

    post_facets, post_planes = _native_facets_and_planes(
        symbols, pts, struct, charges, facet_seeds, recon_ligand, surf_tol
    )
    post_atom_q = _surface_recon_atom_q(symbols, pts, charges, pair_cuts, native_species, cn_refs)
    reports_after = _surface_recon_reports(
        symbols, pts, post_facets, post_planes, charges, surf_tol, post_atom_q
    )
    before_by_hkl = {r.hkl: r for r in reports_before}
    _print_polar_report(reports_after, "AFTER reconstruction", target_hkls, before=before_by_hkl)

    print(
        f"[recon] Done. swapped={len(picked_anions)} added={len(add_positions)} "
        f"Q_total={_total_q(symbols, charges):+d}"
    )
    print('='*60)

    return symbols, pts
