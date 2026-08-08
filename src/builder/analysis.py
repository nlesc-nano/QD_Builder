# nanocrystal_builder/analysis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple, Optional
from math import inf
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .constants import COV_RAD
from .nc_types import Plane, Facet

# ===============================
# Pair-cut calibration machinery
# ===============================

@dataclass(frozen=True)
class PairCuts:
    """Stores per-(element, element) calibrated distance cutoffs (Å)."""
    rc: Dict[Tuple[str, str], float]  # unordered pair key (min(a,b), max(a,b)) → cutoff

def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)

def _cov_radius(sym: str) -> float:
    # Prefer pymatgen covalent radii if available; fall back to table constant
    try:
        from pymatgen.core.periodic_table import Element
        r = Element(sym).covalent_radius
        if r is not None:
            return float(r)
    except Exception:
        pass
    return COV_RAD.get(sym, 1.20)

def _pair_cut(a: str, b: str) -> float:
    # Legacy generic cutoff: modest expansion of covalent-sum
    return 1.25 * (_cov_radius(a) + _cov_radius(b))

def _pair_cut_calibrated(a: str, b: str, cuts: Optional[PairCuts]) -> float:
    if cuts is not None:
        key = _pair_key(a, b)
        if key in cuts.rc:
            return cuts.rc[key]
    return _pair_cut(a, b)

def _supercell_from_cif(cif_path: str, mult=(3, 3, 3)):
    """Return (symbols, coords) from a CIF-expanded supercell, or (None, None) if pymatgen missing."""
    try:
        from pymatgen.core import Structure
    except ImportError:
        print("[warn] pymatgen missing; skipping CIF-based calibration. (pip install pymatgen)")
        return None, None
    s = Structure.from_file(cif_path)
    s.make_supercell(list(mult))
    symbols = [str(x.specie.symbol) for x in s.sites]
    xyz = s.cart_coords
    return symbols, xyz

def derive_pair_cuts_from_cif(
    cif_path: str,
    charges: Dict[str, int],
    *,
    q_only_opposite: bool = True,
    safety: float = 1.00,
    min_samples: int = 5,
) -> Optional[PairCuts]:
    """
    Calibrate per-pair distance cutoffs from the pristine bulk CIF by separating
    the 1st and 2nd neighbor shells. This prevents CN inflation at surfaces.

    - q_only_opposite: only calibrate for opposite-charge pairs (typical for bipartite CN).
    - safety: multiplicative safety factor applied to the final separator (≥1.0 recommended).
    - min_samples: minimum number of first/second distances required per pair; else fallback.
    """
    syms, xyz = _supercell_from_cif(cif_path)
    if syms is None:
        return None

    syms = np.array(syms, dtype=object)
    xyz = np.asarray(xyz, float)
    tree = cKDTree(xyz)
    elems = sorted(set(syms.tolist()))
    # generous upper bound for neighbor search
    cov_upper = 1.25 * max(_cov_radius(a) + _cov_radius(b) for a in elems for b in elems)

    rc: Dict[Tuple[str, str], float] = {}

    for a in elems:
        qa = charges.get(a, 0)
        idx_a = np.where(syms == a)[0]
        if idx_a.size == 0:
            continue

        for b in elems:
            qb = charges.get(b, 0)
            if q_only_opposite and qa * qb >= 0:
                continue
            key = _pair_key(a, b)
            if key in rc:
                continue

            first_shell: List[float] = []
            second_shell: List[float] = []

            # Sample all a-sites (fine for modest supercells)
            for i in idx_a:
                cand = tree.query_ball_point(xyz[i], r=cov_upper)
                # distances to b-type only
                dists = []
                for j in cand:
                    if j == i:
                        continue
                    if syms[j] != b:
                        continue
                    d = float(np.linalg.norm(xyz[j] - xyz[i]))
                    dists.append(d)
                if not dists:
                    continue
                dists.sort()
                first_shell.append(dists[0])
                if len(dists) >= 2:
                    second_shell.append(dists[1])

            if len(first_shell) < min_samples or len(second_shell) < min_samples:
                # fallback to covalent-based cutoff if too few samples
                rc[key] = _pair_cut(a, b)
                continue

            # Robust separator: high quantile of 1st, low quantile of 2nd
            r1_max = float(np.percentile(first_shell, 99.0))
            r2_min = float(np.percentile(second_shell, 1.0))
            if r2_min <= r1_max:
                cutoff = r1_max * 1.05  # slight expansion if overlap
            else:
                cutoff = 0.5 * (r1_max + r2_min)

            rc[key] = cutoff * safety

    return PairCuts(rc=rc)


def merge_pair_cuts_from_cifs(
    cif_paths: Iterable[str],
    charges: Dict[str, int],
    *,
    safety: float = 1.00,
) -> Optional[PairCuts]:
    """
    Build one coordination cutoff table from several material CIFs.

    Single-material charge balancing calibrates from the same CIF used to build
    the cluster.  Core/shell particles contain element pairs from multiple CIFs,
    so stack mode needs the union of those per-material calibrations.

    Isovalent species (e.g. Zn2+ and Cd2+) share the same cutoff for each
    opposite-charge partner so identical geometry yields identical CNs.
    """
    merged: Dict[Tuple[str, str], float] = {}
    any_cut = False
    for cif_path in cif_paths:
        cuts = derive_pair_cuts_from_cif(cif_path, charges, safety=safety)
        if cuts is None:
            continue
        any_cut = True
        for key, value in cuts.rc.items():
            if key in merged:
                merged[key] = max(merged[key], value)
            else:
                merged[key] = value
    if not any_cut:
        return None
    return harmonize_pair_cuts_by_charge(PairCuts(rc=merged), charges)


def harmonize_pair_cuts_by_charge(cuts: PairCuts, charges: Dict[str, int]) -> PairCuts:
    """
    Unify cutoffs among isovalent element pairs so coordination on shared
    geometry does not depend on chemical label (e.g. Zn2+ vs Cd2+).
    """
    rc = dict(cuts.rc)
    elems = set(charges.keys())
    for a, b in rc:
        elems.add(a)
        elems.add(b)

    by_charge: Dict[int, set[str]] = {}
    for el in elems:
        by_charge.setdefault(int(charges.get(el, 0)), set()).add(el)

    charge_vals = sorted(by_charge.keys())
    for qa in charge_vals:
        for qb in charge_vals:
            if qa * qb >= 0:
                continue
            max_cut = 0.0
            for a in by_charge[qa]:
                for b in by_charge[qb]:
                    key = _pair_key(a, b)
                    val = rc.get(key, _pair_cut(a, b))
                    max_cut = max(max_cut, val)
            for a in by_charge[qa]:
                for b in by_charge[qb]:
                    rc[_pair_key(a, b)] = max_cut
    return PairCuts(rc=rc)

def pretty_print_pair_cuts(cuts: Optional[PairCuts], pairs_hint: Optional[List[Tuple[str,str]]] = None):
    """One-line dump of key pair cutoffs (calibrated vs covalent fallback)."""
    if cuts is None:
        print("[pair-cuts] using covalent fallback for all pairs (no CIF calibration).")
        return
    print("[pair-cuts] calibrated cutoffs (Å):")
    shown = set()
    if pairs_hint:
        for a,b in pairs_hint:
            key = _pair_key(a,b)
            val = cuts.rc.get(key, None)
            base = _pair_cut(a,b)
            if val is not None:
                print(f"  {key[0]}–{key[1]}: {val:.3f}  (fallback {base:.3f})")
                shown.add(key)
    # Also print a few of the remaining
    rest = [k for k in cuts.rc.keys() if k not in shown]
    for key in sorted(rest)[:8]:
        a,b = key
        base = _pair_cut(a,b)
        print(f"  {a}–{b}: {cuts.rc[key]:.3f}  (fallback {base:.3f})")
    if len(rest) > 8:
        print(f"  ... (+{len(rest)-8} more)")


# ===============================
# Coordination-number functions
# ===============================

def coord_numbers_bipartite(
    symbols: List[str],
    pts: NDArray[np.float64],
    charges: Dict[str, int],
    pair_cuts: Optional[PairCuts] = None,
) -> NDArray[np.int_]:
    """
    Bipartite CN: counts only opposite-charge neighbors within calibrated pair cutoffs.
    If pair_cuts is None, falls back to covalent-based thresholds.
    """
    N = len(symbols)
    if N == 0:
        return np.zeros(0, dtype=int)

    pts = np.asarray(pts, float)
    tree = cKDTree(pts)

    uniq_elems = sorted(list(set(symbols)))
    max_rcut = 0.0 if not uniq_elems else max(_pair_cut_calibrated(a, b, pair_cuts) for a in uniq_elems for b in uniq_elems)

    elem_to_idx = {el: idx for idx, el in enumerate(uniq_elems)}
    sym_indices = np.array([elem_to_idx[s] for s in symbols], dtype=int)

    cutoff_matrix = np.zeros((len(uniq_elems), len(uniq_elems)), dtype=float)
    for u, el_u in enumerate(uniq_elems):
        for v, el_v in enumerate(uniq_elems):
            cutoff_matrix[u, v] = _pair_cut_calibrated(el_u, el_v, pair_cuts)

    coo = tree.sparse_distance_matrix(tree, max_rcut, output_type='coo_matrix')
    row, col, dists = coo.row, coo.col, coo.data

    charge_arr = np.array([charges.get(s, 0) for s in symbols], dtype=float)
    bipartite_mask = (charge_arr[row] * charge_arr[col] < 0) | (charge_arr[row] == 0)
    cutoff_mask = dists <= cutoff_matrix[sym_indices[row], sym_indices[col]]
    valid_mask = (row != col) & bipartite_mask & cutoff_mask

    cn = np.bincount(row[valid_mask], minlength=N)
    return cn

def coord_numbers(
    symbols: List[str],
    pts: NDArray[np.float64],
    pair_cuts: Optional[PairCuts] = None,
) -> NDArray[np.int_]:
    """
    Non-bipartite CN: counts all neighbors within calibrated (or covalent) pair cutoffs.
    Signature kept backward-compatible: older code may call coord_numbers(symbols, pts).
    """
    N = len(symbols)
    if N == 0:
        return np.zeros(0, dtype=int)

    pts = np.asarray(pts, float)
    tree = cKDTree(pts)

    uniq_elems = sorted(list(set(symbols)))
    max_rcut = 0.0 if not uniq_elems else max(_pair_cut_calibrated(a, b, pair_cuts) for a in uniq_elems for b in uniq_elems)

    elem_to_idx = {el: idx for idx, el in enumerate(uniq_elems)}
    sym_indices = np.array([elem_to_idx[s] for s in symbols], dtype=int)

    cutoff_matrix = np.zeros((len(uniq_elems), len(uniq_elems)), dtype=float)
    for u, el_u in enumerate(uniq_elems):
        for v, el_v in enumerate(uniq_elems):
            cutoff_matrix[u, v] = _pair_cut_calibrated(el_u, el_v, pair_cuts)

    coo = tree.sparse_distance_matrix(tree, max_rcut, output_type='coo_matrix')
    row, col, dists = coo.row, coo.col, coo.data

    cutoff_mask = dists <= cutoff_matrix[sym_indices[row], sym_indices[col]]
    valid_mask = (row != col) & cutoff_mask

    cn = np.bincount(row[valid_mask], minlength=N)
    return cn

def mode(vals: Iterable[int]) -> int:
    arr = np.fromiter(vals, dtype=int, count=-1)
    if arr.size == 0:
        return 0
    # ensure ints for bincount
    return int(np.bincount(arr.astype(int)).argmax())

# ===============================
# Bulk CN estimators
# ===============================

def _atoms_in_any_shell(pts: NDArray[np.float64], planes: List[Plane], surf_tol: float) -> NDArray[np.bool_]:
    hits = np.zeros(len(pts), dtype=bool)
    for (n, d) in planes:
        hits |= ((d - pts @ n) < surf_tol)
    return hits

def bulk_cn_opposite_by_interior(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    surf_tol: float,
    charges: Dict[str, int],
    *,
    true_bulk_cn: Dict[str, int] | None = None,
    pair_cuts: Optional[PairCuts] = None,
) -> Dict[str, int]:
    """
    Mode CN per element using interior atoms (opposite-charge neighbors).
    If true_bulk_cn is provided (from CIF), it is used directly.
    Otherwise, CNs are computed with calibrated (or covalent) pair cuts.
    """
    if true_bulk_cn:
        return {s: int(true_bulk_cn.get(s, 0)) for s in set(symbols)}

    cn = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
    interior = ~_atoms_in_any_shell(pts, planes, surf_tol)
    bulk: Dict[str, int] = {}
    arr_sym = np.array(symbols, dtype=object)
    for el in set(symbols):
        vals = cn[(arr_sym == el) & interior]
        if vals.size == 0:
            vals = cn[(arr_sym == el)]
        bulk[el] = int(np.bincount(vals.astype(int)).argmax()) if vals.size else 0

    by_charge: Dict[int, int] = {}
    for el, val in bulk.items():
        q = int(charges.get(el, 0))
        by_charge[q] = max(by_charge.get(q, 0), int(val))
    for el in bulk:
        bulk[el] = by_charge[int(charges.get(el, 0))]
    return bulk

def bulk_cn_by_interior(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    surf_tol: float,
    pair_cuts: Optional[PairCuts] = None,
) -> Dict[str, int]:
    """
    Non-bipartite bulk estimator (all neighbors), using calibrated cuts if provided.
    """
    cn = coord_numbers(symbols, pts, pair_cuts=pair_cuts)
    interior = ~_atoms_in_any_shell(pts, planes, surf_tol)
    bulk: Dict[str, int] = {}
    arr_sym = np.array(symbols, dtype=object)
    for el in set(symbols):
        vals = cn[(arr_sym == el) & interior]
        if vals.size == 0:
            vals = cn[(arr_sym == el)]
        bulk[el] = int(np.bincount(vals.astype(int)).argmax()) if vals.size else 0
    return bulk

# ===============================
# CIF-derived true bulk CN
# ===============================

def get_true_bulk_cn_from_cif(cif_path: str, charges: Dict[str, int]) -> Dict[str, int]:
    """
    Calculates the ideal, true bulk coordination number for each element by
    reading the original CIF file, creating a supercell, and finding the
    most common (mode) **bipartite** CN for each element type.
    """
    try:
        from pymatgen.core import Structure
    except ImportError:
        print("[warn] pymatgen missing; cannot compute true bulk CN. (pip install pymatgen)")
        return {}

    structure = Structure.from_file(cif_path)
    structure.make_supercell([3, 3, 3])
    supercell_symbols = [str(s.specie.symbol) for s in structure.sites]
    supercell_pts = structure.cart_coords

    # Use covalent fallback here (initial) — or compute pair_cuts first and feed them
    # but to avoid recursion we do a first pass without cuts.
    cn_values = coord_numbers_bipartite(supercell_symbols, supercell_pts, charges, pair_cuts=None)

    true_bulk_cn: Dict[str, int] = {}
    arr_sym = np.array(supercell_symbols, dtype=object)
    for el in set(supercell_symbols):
        vals = cn_values[arr_sym == el].astype(int)
        if vals.size > 0:
            true_bulk_cn[el] = int(np.bincount(vals).argmax())
        else:
            true_bulk_cn[el] = 0
    print(f"[info] true bulk CN (from CIF): {true_bulk_cn}")
    return true_bulk_cn

# ===============================
# Reports & summaries
# ===============================

def facet_cn_summary(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    facets: List[Facet],
    surf_tol: float,
    pair_cuts: Optional[PairCuts] = None,
):
    cn = coord_numbers(symbols, pts, pair_cuts=pair_cuts)
    bulk = bulk_cn_by_interior(symbols, pts, planes, surf_tol, pair_cuts=pair_cuts)
    max_cn = int(cn.max(initial=0))

    print("\n=== PER-FACET CN SUMMARY (COMPACT) ===")
    for fid, (n, d) in enumerate(planes):
        shell = np.where((d - pts @ n) < surf_tol)[0]
        if not shell.size:
            continue
        depth = d - pts[shell] @ n
        outer = shell[depth < 0.35 * surf_tol]
        subl  = shell[(depth >= 0.35 * surf_tol) & (depth < 1.2 * surf_tol)]

        stats = []
        for label, group in (("outer", outer), ("sublayer", subl)):
            if group.size == 0:
                continue
            hist: Dict[str, List[int]] = {}
            for i in group:
                el = symbols[i]
                hist.setdefault(el, [0] * (max_cn + 1))
                hist[el][cn[i]] += 1
            for el, vec in hist.items():
                row = " ".join(f"{vec[c]:5d}" for c in range(max_cn + 1))
                stats.append(f"  {el:>2s} | {row}    {label:8s} (bulk {bulk[el]})")

        if stats:
            f = facets[fid]
            print(f"\nFacet ({f.h}{f.k}{f.l})  #atoms={len(shell)}")
            hdr = "  El | " + " ".join(f"CN{c:>3d}" for c in range(max_cn + 1))
            print(hdr)
            print("  ---+" + "-" * (len(hdr) - 5))
            for line in stats:
                print(line)

def _facet_memberships(pts: NDArray[np.float64], planes: List[Plane], surf_tol: float) -> List[List[int]]:
    """Return list of facet-id memberships for each atom (within surf_tol)."""
    mem = [[] for _ in range(len(pts))]
    for fid, (n, d) in enumerate(planes):
        shell = np.where((d - pts @ n) < surf_tol)[0]
        for i in shell:
            mem[i].append(fid)
    return mem

def surface_report(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    facets: List[Facet],
    surf_tol: float,
    charges: Dict[str,int],
    pair_cuts: Optional[PairCuts] = None,
):
    cn = coord_numbers(symbols, pts, pair_cuts=pair_cuts)
    bulk = bulk_cn_by_interior(symbols, pts, planes, surf_tol, pair_cuts=pair_cuts)
    hits = {i: [] for i in range(len(symbols))}
    for fid, (n, d) in enumerate(planes):
        shell = np.where((d - pts @ n) < surf_tol)[0]
        for i in shell:
            if fid not in hits[i]:
                hits[i].append(fid)

    print("\n=== PER-ATOM SURFACE LIST ===")
    for i, facet_ids in hits.items():
        if not facet_ids:
            continue
        facet_str = "edge" if len(facet_ids) > 1 else f"({facets[facet_ids[0]].h}{facets[facet_ids[0]].k}{facets[facet_ids[0]].l})"
        print(f"{i:4d}  {symbols[i]:>2s}  {facet_str:>8s}  {cn[i]}/{bulk[symbols[i]]}")

def facet_atom_report(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    facets: List[Facet],
    surf_tol: float,
    charges: Dict[str, int],
    pair_cuts: Optional[PairCuts] = None,
):
    """
    Per-facet detailed table with labels:
      role:  unique (1 facet) | edge (2 facets) | vertex (>=3 facets)
      layer: outer | sublayer | (blank if deeper than sublayer threshold)
      deficit: bulk_CN(el) - CN(i)  (>=0) using calibrated (or covalent) cuts
      target: '*' if (anion & outer & deficit>0)
    """
    cn = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
    bulk = bulk_cn_opposite_by_interior(symbols, pts, planes, surf_tol, charges, pair_cuts=pair_cuts)
    memberships = _facet_memberships(pts, planes, surf_tol)

    outer_thr = 0.35 * surf_tol
    subl_thr  = 1.20 * surf_tol

    print("\n=== PER-FACET SURFACE ATOMS (DETAILED) ===")
    print("Legend: role={unique|edge|vertex}  layer={outer|sublayer}  target='*' if (anion & outer & deficit>0)")

    for fid, (n, d) in enumerate(planes):
        shell = np.where((d - pts @ n) < surf_tol)[0]
        if not shell.size:
            continue

        f = facets[fid]
        print(f"\nFacet ({f.h}{f.k}{f.l})  #atoms={len(shell)}")
        print(" idx  el         x(Å)        y(Å)        z(Å)   CN/bulk  role     layer      deficit  type    tgt")
        for i in sorted(shell.tolist(), key=lambda j: (symbols[j], j)):
            x, y, z = pts[i]
            s = symbols[i]
            depth = d - float(np.dot(pts[i], n))

            m = len(memberships[i])
            role = "unique" if m == 1 else ("edge" if m == 2 else "vertex")

            layer = "outer" if depth < outer_thr else ("sublayer" if depth < subl_thr else "")

            deficit = max(0, int(bulk[s]) - int(cn[i]))
            q = int(charges.get(s, 0))
            etype = "anion" if q < 0 else ("cation" if q > 0 else "neutral")

            target = "*" if (layer == "outer" and q < 0 and deficit > 0) else ""

            print(f"{i:4d}  {s:>2s}  {x:10.4f}  {y:10.4f}  {z:10.4f}   {int(cn[i])}/{int(bulk[s])}    {role:6s}  {layer:8s}   {deficit:7d}  {etype:7s}  {target:3s}")


def facet_families_overview(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    facets: List[Facet],
    surf_tol: float,
    charges: Dict[str,int],
):
    """
    Print how many planes per facet family (|h|,|k|,|l|) and per-facet surface charge.
    """
    def fam_key(h,k,l): return tuple(sorted((abs(h),abs(k),abs(l))))
    families: Dict[Tuple[int,int,int], List[int]] = {}
    for fid, f in enumerate(facets):
        families.setdefault(fam_key(f.h,f.k,f.l), []).append(fid)

    print("\n=== FACET FAMILIES OVERVIEW ===")
    for fam, ids in sorted(families.items()):
        label = "".join(str(x) for x in fam)
        print(f"Family {label}: {len(ids)} faces")
    # also print each facet's surface charge sign
    print("\nFacet charges (surface shell only):")
    for fid, (n,d) in enumerate(planes):
        shell = np.where((d - pts @ n) < surf_tol)[0]
        Q = int(sum(charges.get(symbols[i],0) for i in shell))
        f = facets[fid]
        label = f"({f.h}{f.k}{f.l})"
        richness = "cation-rich" if Q>0 else ("anion-rich" if Q<0 else "neutral")
        print(f"  {label:>8s}  #atoms={len(shell):3d}  Q={Q:+d}  {richness}")


# ──────────────────────────────────────────────────────────────────────────────
# Virtual Sites and Strict Outward-Pointing Coordination Matching
# ──────────────────────────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    ln = float(np.linalg.norm(v))
    return v / ln if ln > 1e-12 else np.zeros_like(v)


def cif_first_shell_vector_sets(
    bulk_struct,
    site_sym: str,
    neighbor_symbols: Iterable[str],
) -> List[List[np.ndarray]]:
    """
    Return distinct CIF Cartesian first-shell vector sets for one site species.

    The vectors retain their physical bond lengths and crystallographic
    orientation.  This is the common low-level source for passivation virtual
    directions and nucleation lattice occupancy.
    """
    if bulk_struct is None or not hasattr(bulk_struct, "sites") or not hasattr(bulk_struct, "lattice"):
        return []

    neighbor_set = set(neighbor_symbols)
    if not neighbor_set:
        return []
    neighbor_sites = [
        site
        for site in bulk_struct.sites
        if str(site.specie.symbol) in neighbor_set
    ]
    if not neighbor_sites:
        return []

    latt = bulk_struct.lattice
    vector_sets: List[List[np.ndarray]] = []
    seen: Set[Tuple[Tuple[float, float, float], ...]] = set()
    for ref_site in bulk_struct.sites:
        if str(ref_site.specie.symbol) != site_sym:
            continue
        ref_cart = np.asarray(ref_site.coords, float)
        candidates: List[Tuple[float, np.ndarray]] = []
        for neighbor in neighbor_sites:
            neighbor_cart = np.asarray(neighbor.coords, float)
            for ia in range(-1, 2):
                for ib in range(-1, 2):
                    for ic in range(-1, 2):
                        shift = ia * latt.matrix[0] + ib * latt.matrix[1] + ic * latt.matrix[2]
                        vec = neighbor_cart + shift - ref_cart
                        dist = float(np.linalg.norm(vec))
                        if dist > 0.1:
                            candidates.append((dist, vec))
        if not candidates:
            continue
        candidates.sort(key=lambda rec: rec[0])
        d_min = candidates[0][0]
        vectors: List[np.ndarray] = []
        for dist, vec in candidates:
            if dist >= 1.2 * d_min:
                break
            unit = _unit(vec)
            if all(float(np.dot(unit, _unit(old))) < 0.99 for old in vectors):
                vectors.append(np.asarray(vec, float))
        if not vectors:
            continue
        key = tuple(sorted(tuple(np.round(v, 6)) for v in vectors))
        if key in seen:
            continue
        seen.add(key)
        vector_sets.append(vectors)
    return vector_sets


def _bulk_ideal_direction_sets(
    site_sym: str,
    bulk_struct,
    charges: Dict[str, int],
) -> List[List[np.ndarray]]:
    """
    Return all distinct first-shell opposite-charge direction sets for `site_sym`.

    This supports lattices with non-tetrahedral coordination and multiple
    crystallographic environments for the same species. Direction sets stay in
    the CIF Cartesian frame; they are not reoriented from facet normals.
    """
    if bulk_struct is None or not hasattr(bulk_struct, "sites") or not hasattr(bulk_struct, "lattice"):
        return []

    site_q = int(charges.get(site_sym, 0))
    if site_q == 0:
        return []

    opposite_symbols = {
        str(s.specie.symbol)
        for s in bulk_struct.sites
        if int(charges.get(str(s.specie.symbol), 0)) * site_q < 0
    }
    if not opposite_symbols:
        return []

    return [
        [_unit(vector) for vector in vectors]
        for vectors in cif_first_shell_vector_sets(
            bulk_struct,
            site_sym,
            opposite_symbols,
        )
    ]


def _actual_opposite_bond_vectors(
    syms: List[str],
    pts: NDArray,
    idx: int,
    charges: Dict[str, int],
    cuts: Optional[PairCuts],
) -> List[np.ndarray]:
    sym_i = syms[idx]
    q_i = int(charges.get(sym_i, 0))
    if q_i == 0:
        return []
    vectors: List[np.ndarray] = []
    for j, sym_j in enumerate(syms):
        if j == idx:
            continue
        if int(charges.get(sym_j, 0)) * q_i >= 0:
            continue
        vec = np.asarray(pts[j], float) - np.asarray(pts[idx], float)
        dist = float(np.linalg.norm(vec))
        if 0.1 < dist <= _pair_cut_calibrated(sym_i, sym_j, cuts):
            vectors.append(vec / dist)
    return vectors


def _match_missing_ideal_dirs(
    actual_vecs: List[np.ndarray],
    ideal_dirs: List[np.ndarray],
    min_dot: float = 0.70,
) -> Tuple[List[np.ndarray], float]:
    assigned: Set[int] = set()
    score = 0.0
    for actual in actual_vecs:
        actual = _unit(actual)
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


def _surface_outward_direction(
    idx: int,
    pts: NDArray,
    planes: List[Plane],
    surf_tol: float,
) -> np.ndarray:
    """Use the local facet normal, or summed incident normals on edges/vertices."""
    pts = np.asarray(pts, float)
    incident = []
    nearest: Optional[Tuple[float, np.ndarray]] = None
    for normal, d in planes:
        n = _unit(np.asarray(normal, float))
        depth = float(d) - float(np.dot(pts[idx], n))
        if nearest is None or depth < nearest[0]:
            nearest = (depth, n)
        if depth < surf_tol:
            incident.append((depth, n))
    if incident:
        outer = [_n for depth, _n in incident if depth < 0.35 * surf_tol]
        normals = outer if outer else [n for _, n in incident]
        summed = _unit(np.sum(normals, axis=0))
        if np.linalg.norm(summed) > 1e-12:
            return summed
    if nearest is not None:
        return nearest[1]
    return _unit(pts[idx] - pts.mean(axis=0))


def compute_strict_missing_bond_vectors(
    syms: List[str],
    pts: NDArray,
    charges: Dict[str, int],
    cuts: Optional[PairCuts],
    bulk_struct,
    surf_mask: NDArray,
    planes: List[Plane],
    surf_tol: float,
) -> Dict[int, List[np.ndarray]]:
    """
    Compute missing first-shell polyhedron slots without outward fallback.

    The returned vectors are unmatched bulk bond directions in the CIF Cartesian
    frame. Vectors are never flipped; inward or ambiguous slots are rejected.
    """
    pts = np.asarray(pts, float)
    direction_cache: Dict[str, List[List[np.ndarray]]] = {}
    result: Dict[int, List[np.ndarray]] = {}

    for idx in np.where(surf_mask)[0]:
        sym = syms[int(idx)]
        if int(charges.get(sym, 0)) == 0:
            continue
        if sym not in direction_cache:
            direction_cache[sym] = _bulk_ideal_direction_sets(sym, bulk_struct, charges)
        direction_sets = direction_cache[sym]
        if not direction_sets:
            continue

        actual_vecs = _actual_opposite_bond_vectors(syms, pts, int(idx), charges, cuts)
        if not actual_vecs:
            continue

        best_missing: List[np.ndarray] = []
        best_score = -float("inf")
        for ideal_dirs in direction_sets:
            missing, score = _match_missing_ideal_dirs(actual_vecs, ideal_dirs)
            if score > best_score:
                best_score = score
                best_missing = missing

        outward = _surface_outward_direction(int(idx), pts, planes, surf_tol)
        strict = []
        for vec in best_missing:
            vec = _unit(vec)
            if np.linalg.norm(vec) < 1e-12:
                continue
            if float(np.dot(vec, outward)) > 0.05:
                strict.append(vec)
        strict.sort(key=lambda v: float(np.dot(v, outward)), reverse=True)
        if strict:
            result[int(idx)] = strict

    return result


def _match_actual_to_ideal(
    actual_bond_vecs: List[np.ndarray],
    ideal_dirs: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Greedily assign each actual bond to the closest ideal direction.
    Returns the list of ideal directions that had NO actual bond assigned to them
    (the "missing bond" directions).
    """
    if not ideal_dirs:
        return []
    assigned_ideal: Set[int] = set()
    remaining_actual = list(range(len(actual_bond_vecs)))

    for i_actual in remaining_actual:
        a_vec = actual_bond_vecs[i_actual]
        best_ideal = -1
        best_dot = -2.0
        for i_ideal, i_dir in enumerate(ideal_dirs):
            if i_ideal in assigned_ideal:
                continue
            dot = float(np.dot(a_vec, i_dir))
            if dot > best_dot:
                best_dot = dot
                best_ideal = i_ideal
        if best_ideal >= 0 and best_dot > 0.3:   # 0.3 ≈ 72.5°, generous threshold
            assigned_ideal.add(best_ideal)

    missing = [ideal_dirs[i] for i in range(len(ideal_dirs)) if i not in assigned_ideal]
    return missing


def compute_cif_virtual_sites(
    syms: List[str],
    pts: NDArray,
    charges: Dict[str, int],
    cuts: Optional[PairCuts],
    bulk_struct,
    surf_mask: NDArray,
    planes: List[Plane],
    surf_tol: float,
) -> List[Dict]:
    """
    Finds strict missing neighbors, projects their exact ideal bulk Cartesian 
    positions in the nanocrystal frame using CIF translation vectors, clusters 
    overlapping vacancies (<0.2 Å), and returns a list of merged virtual sites 
    sorted by multiplicity (descending).
    
    Each merged site dict contains:
      - 'pos': Cartesian intersection position (average of projected positions)
      - 'hosts': List of host cation indices coordinating this site
      - 'multiplicity': Number of host cations
      - 'u_vecs': Dict mapping host_idx -> unit vector pointing from host to pos
    """
    if bulk_struct is None or not hasattr(bulk_struct, "sites") or not hasattr(bulk_struct, "lattice"):
        return []

    pts = np.asarray(pts, float)
    tree = cKDTree(pts)
    raw_virtual_sites = []

    # 1. Calibrate constant fractional shift between NC and bulk reference structure
    f_coords = bulk_struct.lattice.get_fractional_coords(pts)
    best_f_shift = np.zeros(3)
    found_shift = False

    bulk_species = {s.specie.symbol for s in bulk_struct.sites}
    ref_idx = -1
    for idx in range(len(syms)):
        if syms[idx] in bulk_species:
            ref_idx = idx
            break

    if ref_idx >= 0:
        ref_sym = syms[ref_idx]
        ref_sites = [s for s in bulk_struct.sites if s.specie.symbol == ref_sym]
        ref_f = f_coords[ref_idx]
        for site in ref_sites:
            candidate_shift = (site.frac_coords - ref_f) % 1
            
            # test on first few bulk-native atoms of the NC
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
                shifted_f = (f_coords[j] + candidate_shift + 0.5) % 1 - 0.5
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

    # Map each surface atom to its bulk CIF neighbor mapping
    for idx in np.where(surf_mask)[0]:
        sym_i = syms[int(idx)]
        site_q = int(charges.get(sym_i, 0))
        if site_q == 0:
            continue

        # 2. Map to reference bulk site in unit cell (with shift un-biasing)
        f_coord_unshifted = (f_coords[int(idx)] + best_f_shift + 0.5) % 1 - 0.5
        site_idx = int(np.argmin([
            float(np.linalg.norm((site.frac_coords - f_coord_unshifted + 0.5) % 1 - 0.5))
            for site in bulk_struct.sites
        ]))
        bulk_site = bulk_struct.sites[site_idx]

        # 2. Get first-shell opposite-charge bulk neighbors
        all_neigh = bulk_struct.get_neighbors(bulk_site, r=4.5)
        if not all_neigh:
            continue
        min_dist = min(neigh.nn_distance for neigh in all_neigh)
        bulk_neighbors = [
            neigh for neigh in all_neigh 
            if neigh.nn_distance <= 1.15 * min_dist
            and int(charges.get(str(neigh.specie.symbol), 0)) * site_q < 0
        ]
        if not bulk_neighbors:
            continue

        # Build set of ideal bulk neighbor vectors and target coords
        ideal_neigh_list = []
        for neigh in bulk_neighbors:
            vec_ij = neigh.coords - bulk_site.coords
            ideal_neigh_list.append({
                "vec": vec_ij,
                "u_vec": _unit(vec_ij),
                "sym": str(neigh.specie.symbol),
            })

        # 3. Find actual neighbors in the nanocrystal
        max_rcut = max(
            _pair_cut_calibrated(sym_i, s2, cuts)
            for s2 in set(syms)
            if charges.get(s2, 0) * site_q < 0
        ) if any(charges.get(s2, 0) * site_q < 0 for s2 in set(syms)) else 4.0

        neigh_idxs = tree.query_ball_point(pts[idx], r=max_rcut)
        actual_vecs = []
        for j in neigh_idxs:
            if j == idx:
                continue
            if charges.get(syms[j], 0) * site_q >= 0:
                continue
            d = pts[j] - pts[idx]
            dist = float(np.linalg.norm(d))
            if 0.1 < dist <= _pair_cut_calibrated(sym_i, syms[j], cuts):
                actual_vecs.append(_unit(d))

        # 4. Map actual bonds to ideal bulk neighbors to identify missing ones
        assigned_ideal = set()
        for u_a in actual_vecs:
            best_idx = -1
            best_dot = -2.0
            for k, ideal in enumerate(ideal_neigh_list):
                if k in assigned_ideal:
                    continue
                dot = float(np.dot(u_a, ideal["u_vec"]))
                if dot > best_dot:
                    best_idx = k
                    best_dot = dot
            if best_idx >= 0 and best_dot >= 0.70:
                assigned_ideal.add(best_idx)

        # 5. Extract strict missing vacancies pointing outward
        outward = _surface_outward_direction(int(idx), pts, planes, surf_tol)
        for k, ideal in enumerate(ideal_neigh_list):
            if k in assigned_ideal:
                continue
            u_ideal = ideal["u_vec"]
            if float(np.dot(u_ideal, outward)) > 0.05:
                # Perfect Cartesian position where this bulk vacancy sits!
                vacant_pos = pts[idx] + ideal["vec"]
                raw_virtual_sites.append({
                    "host_idx": int(idx),
                    "u_vec": u_ideal,
                    "pos": vacant_pos,
                })

    n_raw = len(raw_virtual_sites)
    if n_raw == 0:
        return []

    # 6. Graph-based proximity clustering (threshold 0.2 Å for exact unit cell alignment)
    adj = {i: [] for i in range(n_raw)}
    for i1 in range(n_raw):
        for i2 in range(i1 + 1, n_raw):
            d = float(np.linalg.norm(raw_virtual_sites[i1]["pos"] - raw_virtual_sites[i2]["pos"]))
            if d < 0.2:
                adj[i1].append(i2)
                adj[i2].append(i1)

    visited = [False] * n_raw
    components = []
    for node in range(n_raw):
        if visited[node]:
            continue
        comp = []
        queue = [node]
        visited[node] = True
        while queue:
            curr = queue.pop(0)
            comp.append(curr)
            for neighbor in adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(comp)

    # 7. Build final merged virtual sites
    merged_sites = []
    for comp in components:
        members = [raw_virtual_sites[k] for k in comp]
        hosts = list(set(m["host_idx"] for m in members))
        merged_pos = np.mean([m["pos"] for m in members], axis=0)
        
        # Build host-specific direction vectors pointing to this intersection position
        u_vecs = {}
        for h in hosts:
            u_vecs[h] = _unit(merged_pos - pts[h])

        merged_sites.append({
            "pos": merged_pos,
            "hosts": hosts,
            "multiplicity": len(hosts),
            "u_vecs": u_vecs,
        })

    # Sort by multiplicity descending
    merged_sites.sort(key=lambda x: x["multiplicity"], reverse=True)
    return merged_sites


