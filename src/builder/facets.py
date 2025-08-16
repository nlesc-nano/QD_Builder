# nanocrystal_builder/facets.py
from __future__ import annotations
import math, numpy as np
from typing import Dict, List, Tuple

try:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:
    raise SystemExit("pip install pymatgen[matproj]")

from .nc_types import Facet, Plane
from .geometry import unit_normal
from .constants import EPS

def expand_facets(s: Structure, seeds: list[Facet], proper_only: bool = True) -> list[Facet]:
    ops = SpacegroupAnalyzer(s, symprec=1e-3).get_symmetry_operations(cartesian=True)
    recT = s.lattice.reciprocal_lattice.matrix.T
    out: dict[tuple[int,int,int], Facet] = {}

    for f in seeds:
        n0 = unit_normal(s, (f.h, f.k, f.l))  # use your existing helper
        for op in ops:
            R = op.rotation_matrix
            if proper_only and np.linalg.det(R) < 0.999:   # filter improper (rotoinversions/reflections)
                continue
            n = R @ n0
            # project back to Miller indices
            coeff = np.linalg.solve(recT, n)
            hkl = tuple(int(round(c)) for c in coeff)
            if hkl == (0, 0, 0):
                continue
            # reduce by gcd to primitive integers, preserve SIGN (do not fold to opposite)
            g = math.gcd(math.gcd(abs(hkl[0]), abs(hkl[1])), abs(hkl[2]))
            hkl = (hkl[0] // g, hkl[1] // g, hkl[2] // g)
            # keep first gamma encountered for each signed hkl (or take the max/min—your choice)
            out.setdefault(hkl, Facet(h=hkl[0], k=hkl[1], l=hkl[2], gamma=f.gamma))

    return list(out.values())


def _canon_orientation(n: np.ndarray, tol=1e-6) -> Tuple[int,int,int]:
    """
    Orientation key that treats n and -n as the same direction.
    We use the sign pattern that makes the first non-zero component positive.
    """
    n = n / (np.linalg.norm(n) + 1e-12)
    if abs(n[0]) > tol:
        s = 1.0 if n[0] >= 0 else -1.0
    elif abs(n[1]) > tol:
        s = 1.0 if n[1] >= 0 else -1.0
    else:
        s = 1.0 if n[2] >= 0 else -1.0
    u = tuple(np.round(s * n, 6))
    # map to small integer key to avoid float drift (not strictly necessary)
    return u  # using float tuple is fine for a set key with rounding

def detect_facets_from_nc(
    symbols, pts, lattice, charges, full_facets: List[Facet], surf_tol: float = 2.0
):
    """
    For each unique orientation (ignoring sign), **add both ± planes** by taking the
    outermost intercepts on each side. This guarantees opposite faces are present.
    """
    facets: List[Facet] = []
    planes: List[Plane] = []

    used_orientations = set()

    for f in full_facets:
        n0 = unit_normal(lattice, (f.h, f.k, f.l))
        key = _canon_orientation(n0)
        if key in used_orientations:
            continue
        used_orientations.add(key)

        # + face
        n_plus = n0
        d_plus = float(np.max(pts @ n_plus) + EPS)
        shell_plus = np.where((d_plus - pts @ n_plus) < surf_tol)[0]
        if shell_plus.size:
            facets.append(Facet(f.h, f.k, f.l, f.gamma))
            planes.append((n_plus, d_plus))

        # - face
        n_minus = -n0
        d_minus = float(np.max(pts @ n_minus) + EPS)
        shell_minus = np.where((d_minus - pts @ n_minus) < surf_tol)[0]
        if shell_minus.size:
            facets.append(Facet(-f.h, -f.k, -f.l, f.gamma))
            planes.append((n_minus, d_minus))

    return facets, planes

def halfspaces(s, facets: List[Facet], R: float) -> List[Plane]:
    from .geometry import halfspaces as _hs
    return _hs(s, facets, R)

