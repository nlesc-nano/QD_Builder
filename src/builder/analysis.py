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
    pts = np.asarray(pts, float)
    tree = cKDTree(pts)
    cn = np.zeros(N, dtype=int)

    uniq = set(symbols)
    max_rcut = 0.0 if not uniq else max(_pair_cut_calibrated(a, b, pair_cuts) for a in uniq for b in uniq)

    for i in range(N):
        si = symbols[i]
        qi = charges.get(si, 0)

        idxs = tree.query_ball_point(pts[i], r=max_rcut)
        if qi == 0:
            # count all neighbors under calibrated cut (rare for cores, but supported)
            count = 0
            for j in idxs:
                if j == i:
                    continue
                if np.linalg.norm(pts[j] - pts[i]) <= _pair_cut_calibrated(si, symbols[j], pair_cuts):
                    count += 1
            cn[i] = count
            continue

        count = 0
        for j in idxs:
            if j == i:
                continue
            if charges.get(symbols[j], 0) * qi >= 0:
                continue
            if np.linalg.norm(pts[j] - pts[i]) <= _pair_cut_calibrated(si, symbols[j], pair_cuts):
                count += 1
        cn[i] = count
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
    pts = np.asarray(pts, float)
    tree = cKDTree(pts)
    uniq = set(symbols)
    max_rcut = 0.0 if not uniq else max(_pair_cut_calibrated(a, b, pair_cuts) for a in uniq for b in uniq)
    cn = np.zeros(len(pts), dtype=int)
    for i, (sym_i, pi) in enumerate(zip(symbols, pts)):
        idxs = tree.query_ball_point(pi, r=max_rcut)
        count = 0
        for j in idxs:
            if j == i:
                continue
            if np.linalg.norm(pts[j] - pi) <= _pair_cut_calibrated(sym_i, symbols[j], pair_cuts):
                count += 1
        cn[i] = count
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
