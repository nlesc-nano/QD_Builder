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
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .analysis import PairCuts, _pair_cut_calibrated, coord_numbers_bipartite, derive_pair_cuts_from_cif
from .facets import detect_facets_from_nc
from .nc_types import Facet, FacetReconstructionSpec, Plane

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
    spec: FacetReconstructionSpec,
    charge_balance_fn,
    verbose: bool,
    write_all: bool,
    prefix: str,
) -> Tuple[List[str], NDArray[np.float64]]:
    """
    Lannoo polar-facet reconstruction.

    Works on the post-passivation structure directly — no stripping.
    Ligand bonds count toward CN, so the Lannoo charge reflects only residual
    dangling bonds that global passivation could not neutralize.
    """
    if not spec.enabled or not spec.facets:
        return symbols, pts

    configured_hkls: Set[Tuple[int, int, int]] = set(spec.facets)
    target_hkls: Set[Tuple[int, int, int]] = set(configured_hkls)
    pair_cuts = derive_pair_cuts_from_cif(cif_path, charges, safety=1.00)

    print(f"\n{'='*60}")
    print(f"[recon] Polar-facet reconstruction configured from: {sorted(configured_hkls)}")
    print(f"[recon] Q_total = {_total_q(symbols, charges):+d}  (should be 0 after global passivation)")

    # Detect native scaffold planes (geometry only; defines atom membership)
    all_facets, all_planes = _native_facets_and_planes(
        symbols, pts, struct, charges, facet_seeds, ligand, surf_tol
    )
    detected_hkls = {(f.h, f.k, f.l) for f in all_facets}
    missing = configured_hkls - detected_hkls
    if missing:
        print(f"[recon] WARNING: hkl(s) not detected in native scaffold: {missing}")

    # Phase 1: Lannoo charges on the post-passivation structure
    # CN includes bonds to both native atoms and ligands (so passivated atoms show q=0)
    atom_q = _lannoo_all_atoms(symbols, pts, charges, ligand, pair_cuts)
    rows_before = _compute_facet_charges(
        symbols, pts, all_facets, all_planes, charges, ligand, surf_tol, atom_q
    )
    configured_families = {_hkl_family(hkl) for hkl in configured_hkls}
    target_hkls = {
        r.hkl
        for r in rows_before
        if abs(r.q_lannoo) > 1e-9 and _hkl_family(r.hkl) in configured_families
    }
    print(f"[recon] Treating detected configured polar-family facets: {sorted(target_hkls)}")
    if not target_hkls:
        print("[recon] No polar facets with non-zero Lannoo charge found; skipping.")
        return symbols, pts
    _print_lannoo_table(rows_before, "BEFORE reconstruction", target_hkls)

    # Phase 1b: strip all ligands from selected cation-rich facets before any
    # facet-by-facet vacancy treatment. This prevents old global passivation
    # ligands from defining the polar reconstruction chemistry.
    symbols, pts, strip_log = _strip_ligands_from_cation_rich_facets(
        symbols,
        pts,
        rows_before,
        all_planes,
        charges,
        ligand,
        target_hkls,
        surf_tol,
        pair_cuts,
        verbose,
    )
    stripped_facets, stripped_planes = _native_facets_and_planes(
        symbols, pts, struct, charges, facet_seeds, ligand, surf_tol
    )
    atom_q_stripped = _lannoo_all_atoms(symbols, pts, charges, ligand, pair_cuts)
    rows_stripped = _compute_facet_charges(
        symbols, pts, stripped_facets, stripped_planes, charges, ligand, surf_tol, atom_q_stripped
    )
    _print_lannoo_table(rows_stripped, "AFTER cation-rich ligand strip / before vacancy treatment", target_hkls)

    # Phase 2: one local reconstruction event per polar facet, then total
    # charge balance is applied after every facet has been visited.
    before_by_hkl = {r.hkl: r for r in rows_before}
    target_rows = [
        (before_by_hkl[r.hkl], r, stripped_planes[r.fid])
        for r in rows_stripped
        if r.hkl in target_hkls and r.hkl in before_by_hkl
    ]
    target_rows.sort(key=lambda t: abs(t[0].q_lannoo), reverse=True)

    move_log: Dict[Tuple[int, int, int], List[str]] = {}

    for original_row, row, (n_vec, d_val) in target_rows:
        hkl = original_row.hkl
        print(f"\n[recon] Facet {hkl}: original {original_row.termination}, "
              f"Q_Lannoo={original_row.q_lannoo:+.3f}, N_surface={row.n_surface}")

        # Re-detect planes after each facet's reconstruction (sequential dependency)
        _cur_facets, cur_planes = _native_facets_and_planes(
            symbols, pts, struct, charges, facet_seeds, ligand, surf_tol
        )

        symbols, pts, moves = _reconstruct_facet_spaced(
            symbols, pts, hkl, n_vec, d_val, cur_planes,
            charges, ligand, surf_tol, pair_cuts, verbose,
            original_q_lannoo=original_row.q_lannoo,
        )
        move_log[hkl] = moves

    # Lannoo report after reconstruction (before global rebalance)
    post_facets, post_planes = _native_facets_and_planes(
        symbols, pts, struct, charges, facet_seeds, ligand, surf_tol
    )
    atom_q_post = _lannoo_all_atoms(symbols, pts, charges, ligand, pair_cuts)
    rows_after = _compute_facet_charges(
        symbols, pts, post_facets, post_planes, charges, ligand, surf_tol, atom_q_post
    )
    _print_lannoo_table(rows_after, "AFTER reconstruction / before global rebalance", target_hkls)

    # Phase 3: one final global charge-balance pass.
    # include_sublayer=True because reconstruction may leave undercoordinated cations
    # in the sublayer (As→Cl swaps shorten the effective bond, reducing nearby cation CN).
    Q_now = _total_q(symbols, charges)
    print(f"\n[recon] Final global charge balance (Q_total={Q_now:+d})...")
    symbols, pts = charge_balance_fn(
        symbols, pts, charges, ligand,
        verbose=verbose,
        planes=post_planes,
        surf_tol=surf_tol,
        cif_path=cif_path,
        positive_q_strategy="remove",
        write_all=write_all,
        prefix=f"{prefix}_recon",
        include_sublayer=True,
    )

    # Final Lannoo report
    final_facets, final_planes = _native_facets_and_planes(
        symbols, pts, struct, charges, facet_seeds, ligand, surf_tol
    )
    atom_q_final = _lannoo_all_atoms(symbols, pts, charges, ligand, pair_cuts)
    rows_final = _compute_facet_charges(
        symbols, pts, final_facets, final_planes, charges, ligand, surf_tol, atom_q_final
    )
    _print_lannoo_table(rows_final, "FINAL (after global rebalance)", target_hkls)
    _print_reconstruction_summary(
        rows_before, rows_stripped, rows_after, rows_final, target_hkls, strip_log, move_log
    )
    print(f"[recon] Done. Q_total = {_total_q(symbols, charges):+d}")
    print('='*60)

    return symbols, pts
