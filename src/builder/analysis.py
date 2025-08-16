# nanocrystal_builder/analysis.py
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Iterable, Tuple
from scipy.spatial import cKDTree

from .constants import COV_RAD
from .nc_types import Plane, Facet

def _cov_radius(sym: str) -> float:
    try:
        from pymatgen.core.periodic_table import Element
        r = Element(sym).covalent_radius
        if r is not None:
            return float(r)
    except Exception:
        pass
    return COV_RAD.get(sym, 1.20)

def _pair_cut(a: str, b: str) -> float:
    return 1.25 * (_cov_radius(a) + _cov_radius(b))

def coord_numbers(symbols: List[str], pts: NDArray[np.float64]) -> NDArray[np.int_]:
    pts = np.asarray(pts, float)
    tree = cKDTree(pts)
    max_rcut = 1.25 * max(_cov_radius(a) + _cov_radius(b) for a in set(symbols) for b in set(symbols))
    cn = np.zeros(len(pts), dtype=int)
    for i, (sym_i, pi) in enumerate(zip(symbols, pts)):
        idxs = tree.query_ball_point(pi, r=max_rcut)
        count = 0
        for j in idxs:
            if j == i:
                continue
            if np.linalg.norm(pts[j] - pi) <= _pair_cut(sym_i, symbols[j]):
                count += 1
        cn[i] = count
    return cn

def mode(vals: Iterable[int]) -> int:
    arr = np.fromiter(vals, dtype=int, count=-1)
    if arr.size == 0:
        return 0
    return int(np.bincount(arr).argmax())

def _atoms_in_any_shell(pts: NDArray[np.float64], planes: List[Plane], surf_tol: float) -> NDArray[np.bool_]:
    hits = np.zeros(len(pts), dtype=bool)
    for (n, d) in planes:
        hits |= ((d - pts @ n) < surf_tol)
    return hits

def bulk_cn_by_interior(symbols: List[str], pts: NDArray[np.float64], planes: List[Plane], surf_tol: float) -> Dict[str, int]:
    cn = coord_numbers(symbols, pts)
    interior = ~_atoms_in_any_shell(pts, planes, surf_tol)
    bulk: Dict[str, int] = {}
    arr_sym = np.array(symbols, dtype=object)
    for el in set(symbols):
        vals = cn[(arr_sym == el) & interior]
        if vals.size == 0:
            vals = cn[(arr_sym == el)]
        bulk[el] = int(np.bincount(vals).argmax()) if vals.size else 0
    return bulk

def facet_cn_summary(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    facets: List[Facet],
    surf_tol: float,
):
    cn = coord_numbers(symbols, pts)
    bulk = bulk_cn_by_interior(symbols, pts, planes, surf_tol)
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
            hkl = facets[fid]
            print(f"\nFacet ({hkl.h}{hkl.k}{hkl.l})  #atoms={len(shell)}")
            hdr = "  El | " + " ".join(f"CN{c:>3d}" for c in range(max_cn + 1))
            print(hdr)
            print("  ---+" + "-" * (len(hdr) - 5))
            for line in stats:
                print(line)

def surface_report(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    facets: List[Facet],
    surf_tol: float,
):
    cn = coord_numbers(symbols, pts)
    bulk = bulk_cn_by_interior(symbols, pts, planes, surf_tol)
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

# ---------------- New reporting helpers ----------------

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

# --- add this helper anywhere above the reports ---
def _facet_memberships(pts: NDArray[np.float64], planes: List[Plane], surf_tol: float) -> List[List[int]]:
    """Return list of facet-id memberships for each atom (within surf_tol)."""
    mem = [[] for _ in range(len(pts))]
    for fid, (n, d) in enumerate(planes):
        shell = np.where((d - pts @ n) < surf_tol)[0]
        for i in shell:
            mem[i].append(fid)
    return mem

# --- replace the old facet_atom_report with this enhanced version ---
def facet_atom_report(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    facets: List[Facet],
    surf_tol: float,
    charges: Dict[str, int],
):
    """
    Per-facet detailed table with labels:
      role:  unique (1 facet) | edge (2 facets) | vertex (>=3 facets)
      layer: outer | sublayer | (blank if deeper than sublayer threshold)
      deficit: bulk_CN(el) - CN(i)  (>=0)
      target: '*' if (anion & outer & undercoordinated)  <-- NEW rule
    """
    cn = coord_numbers(symbols, pts)
    bulk = bulk_cn_by_interior(symbols, pts, planes, surf_tol)
    memberships = _facet_memberships(pts, planes, surf_tol)

    outer_thr = 0.35 * surf_tol
    subl_thr  = 1.20 * surf_tol

    print("\n=== PER-FACET SURFACE ATOMS (DETAILED) ===")
    print("Legend: role={unique|edge|vertex}  layer={outer|sublayer}  target='*' if (anion & outer & deficit>0)")

    for fid, (n, d) in enumerate(planes):
        shell = np.where((d - pts @ n) < surf_tol)[0]
        if not shell.size:
            continue

        n_unique = n_edge = n_vertex = n_targets = 0

        f = facets[fid]
        print(f"\nFacet ({f.h}{f.k}{f.l})  #atoms={len(shell)}")
        print(" idx  el         x(Å)        y(Å)        z(Å)   CN/bulk  role     layer      deficit  type    tgt")
        for i in sorted(shell.tolist(), key=lambda j: (symbols[j], j)):
            x, y, z = pts[i]
            s = symbols[i]
            depth = d - np.dot(pts[i], n)

            m = len(memberships[i])
            role = "unique" if m == 1 else ("edge" if m == 2 else "vertex")

            layer = "outer" if depth < outer_thr else ("sublayer" if depth < subl_thr else "")

            deficit = max(0, bulk[s] - cn[i])
            q = charges.get(s, 0)
            etype = "anion" if q < 0 else ("cation" if q > 0 else "neutral")

            # NEW: outer-only targeting, no uniqueness requirement
            target = "*" if (layer == "outer" and q < 0 and deficit > 0) else ""

            if role == "unique": n_unique += 1
            elif role == "edge": n_edge += 1
            else: n_vertex += 1
            if target: n_targets += 1

            print(f"{i:4d}  {s:>2s}  {x:10.4f}  {y:10.4f}  {z:10.4f}   {cn[i]}/{bulk[s]}    {role:6s}  {layer:8s}   {deficit:7d}  {etype:7s}  {target:3s}")

        print(f"  -> counts: unique={n_unique}, edge={n_edge}, vertex={n_vertex}, prime_targets={n_targets}")

