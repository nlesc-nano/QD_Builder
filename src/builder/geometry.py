# nanocrystal_builder/geometry.py
from __future__ import annotations
import math
from itertools import product
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

from .constants import EPS
from .nc_types import Facet, Plane

def unit_normal(lattice_or_struct, hkl: Tuple[int,int,int]) -> NDArray[np.float64]:
    """
    Plane normal for Miller indices (hkl) in CARTESIAN coords using the RECIPROCAL lattice:
      n ∝ h b1 + k b2 + l b3
    """
    lattice = lattice_or_struct.lattice if hasattr(lattice_or_struct, "lattice") else lattice_or_struct
    v = lattice.reciprocal_lattice.get_cartesian_coords(hkl)
    n = v / np.linalg.norm(v)
    return n

def halfspaces(s, facets: List[Facet], R: float) -> List[Plane]:
    lam = R / min(f.gamma for f in facets)
    return [(unit_normal(s, (f.h, f.k, f.l)), lam * f.gamma) for f in facets]

def inside(pts: NDArray[np.float64], planes: List[Plane]) -> NDArray[bool]:
    if len(planes) == 0:
        return np.ones(len(pts), dtype=bool)
    N = np.vstack([n for (n, _) in planes])      # F x 3
    d = np.array([d for (_, d) in planes])       # F
    return (pts @ N.T <= (d + EPS)).all(axis=1)

def rep_ranges(lattice, maxd: float):
    a, b, c = lattice.matrix
    n = lambda v: int(math.ceil((maxd + EPS) / np.linalg.norm(v))) + 1
    return (range(-n(a), n(a) + 1), range(-n(b), n(b) + 1), range(-n(c), n(c) + 1))

def build_nanocrystal(struct, facets: List[Facet], R: float):
    planes = halfspaces(struct, facets, R)
    maxd = max(d for _, d in planes)
    rx, ry, rz = rep_ranges(struct.lattice, maxd)

    base = struct.frac_coords @ struct.lattice.matrix
    syms, pts = [], []
    for i, j, k in product(rx, ry, rz):
        shift = (i * struct.lattice.matrix[0]
                 + j * struct.lattice.matrix[1]
                 + k * struct.lattice.matrix[2])
        coords = base + shift
        mask = inside(coords, planes)
        idxs = np.where(mask)[0]
        if idxs.size:
            # >>> changed lines:
            syms.extend([struct.species[int(idx)].symbol for idx in idxs])
            pts.extend(coords[idxs].tolist())
            # <<< 
    return syms, np.asarray(pts, float), planes

def dedupe_points(symbols: List[str], pts: NDArray[np.float64], tol: float = 1e-3):
    """Remove duplicates within 'tol' using grid snap."""
    key = np.round(pts / tol).astype(np.int64)
    _, unique_idx = np.unique(key, axis=0, return_index=True)
    order = np.sort(unique_idx)
    return [symbols[i] for i in order], pts[order]

