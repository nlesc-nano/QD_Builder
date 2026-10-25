# src/builder/geometry.py
from __future__ import annotations
import math
from itertools import product
from typing import List

import numpy as np
from scipy.spatial import cKDTree

from .constants import EPS
from .nc_types import Facet, Plane
from .facets import halfspaces

PLANE_EPS = EPS

# ---------- Common helpers ----------

def rep_ranges(lattice, maxd: float):
    a, b, c = lattice.matrix
    n = lambda v: int(math.ceil((maxd + PLANE_EPS) / np.linalg.norm(v))) + 1
    return (range(-n(a), n(a)+1), range(-n(b), n(b)+1), range(-n(c), n(c)+1))

def inside(pts: np.ndarray, planes: List[Plane]) -> np.ndarray:
    mask = np.ones(len(pts), dtype=bool)
    for n, d in planes:
        mask &= (pts @ n) <= (d + PLANE_EPS)
        if not mask.any():
            break
    return mask

def auto_shift_planes(
    coords: np.ndarray,
    planes: List[Plane],
    threshold: float = 0.05,
    shift_amount: float = 0.25,
) -> List[Plane]:
    shifted_planes = []
    for n, d in planes:
        dists = coords @ n
        diffs = np.abs(dists - d)
        if np.any(diffs < threshold):
            old_d = d
            d = d + shift_amount
            normal_str = f"[{n[0]:.2f}, {n[1]:.2f}, {n[2]:.2f}]"
            print(f"[wulff:shift] Auto-shifted facet (normal: {normal_str}) plane outward by +{shift_amount:.2f} Å (from d={old_d:.4f} to {d:.4f}) to avoid cutting through atomic layer.")
        shifted_planes.append((n, d))
    return shifted_planes

def dedupe_points(symbols: List[str], pts: np.ndarray, tol: float = PLANE_EPS):
    """
    Remove near-duplicates (within tol) while preserving order.
    """
    if len(pts) == 0:
        return symbols, pts
    keep = np.ones(len(pts), bool)
    tree = cKDTree(pts)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        nbrs = tree.query_ball_point(pts[i], r=tol)
        for j in nbrs:
            if j > i:
                keep[j] = False
    return [s for s, k in zip(symbols, keep) if k], pts[keep]


def recut_with_planes(syms: List[str], pts: np.ndarray, planes: List[Plane], tol: float = PLANE_EPS):
    """
    Keep only atoms satisfying all Wulff halfspaces n.x <= d + tol.
    """
    if not planes:
        return syms, pts
    A = np.stack([n for (n, _d) in planes], axis=0)
    b = np.array([d for (_n, d) in planes], float)
    mask = (pts @ A.T <= (b[None, :] + tol)).all(axis=1)
    syms2 = [s for s, keep in zip(syms, mask) if keep]
    pts2 = pts[mask]
    return syms2, pts2

def _dual_termination_families(facets: List[Facet]) -> set:
    """Facet families for which both cation_rich and anion_rich facets are requested."""
    terms: dict = {}
    for f in facets:
        t = getattr(f, "termination", None)
        if getattr(f, "scope", "family") == "family" and t in {"cation_rich", "anion_rich"}:
            terms.setdefault(tuple(abs(int(x)) for x in (f.h, f.k, f.l)), set()).add(t)
    return {fam for fam, ts in terms.items() if len(ts) == 2}


def _bond_vectors(struct) -> List[np.ndarray]:
    """First-shell bond vectors of every unit-cell site (1.15 x the shortest bond)."""
    dmin = min(float(nb.nn_distance) for site in struct.sites for nb in struct.get_neighbors(site, 6.0))
    return [
        np.array([nb.coords - site.coords for nb in struct.get_neighbors(site, 1.15 * dmin)], float).reshape(-1, 3)
        for site in struct.sites
    ]


def place_polar_planes(
    struct,
    facets: List[Facet],
    planes: List[Plane],
    coords: np.ndarray,
    site_index: np.ndarray,
    charges: dict,
    margin: float = 0.25,
    window: float = 6.0,
) -> List[Plane]:
    """
    Move the planes of polar families (both terminations requested) so the
    cut leaves the requested species outermost with the fewest dangling
    bonds, e.g. zinc-blende {111}: In with one bond out on a cation-rich
    face, P with one bond out on an anion-rich one. A Wulff plane that falls
    between the two close layers of an In-P pair leaves the other species,
    with three bonds out, on top; the passivation prepass strips those and
    the face ends up with the opposite termination.

    A plane already above a valid top layer is left untouched; otherwise it
    moves to the nearest position at least `margin` from the layers on both
    sides with a valid top layer (ties outward).
    """
    dual = _dual_termination_families(facets)
    if not dual or len(coords) == 0:
        return planes
    bonds = _bond_vectors(struct)
    sign = [int(np.sign(charges.get(site.specie.symbol, 0))) for site in struct.sites]
    out: List[Plane] = []
    for f, (n, d) in zip(facets, planes):
        term = getattr(f, "termination", None)
        if tuple(abs(int(x)) for x in (f.h, f.k, f.l)) not in dual or term not in {"cation_rich", "anion_rich"}:
            out.append((n, d))
            continue
        want = 1 if term == "cation_rich" else -1
        n_unit = np.asarray(n, float) / np.linalg.norm(n)
        up = [int(np.sum(b @ n_unit > 1e-3 * np.linalg.norm(b, axis=1))) for b in bonds]
        candidates = [s for s in range(len(up)) if sign[s] == want and up[s] > 0]
        if not candidates:
            out.append((n, d))
            continue
        fewest = min(up[s] for s in candidates)
        good = {s for s in candidates if up[s] == fewest}

        z = coords @ n
        near = np.flatnonzero(np.abs(z - d) < window)
        layers: dict = {}
        for zk, s in zip(np.round(z[near], 3).tolist(), site_index[near].tolist()):
            layers.setdefault(zk, set()).add(s)
        zs = sorted(layers)
        valid = [layers[zk] <= good for zk in zs]

        kept = [k for k, zk in enumerate(zs) if zk <= d + PLANE_EPS]
        if kept and valid[kept[-1]]:
            out.append((n, d))
            continue
        best = None
        for k in range(len(zs) - 1):
            if not valid[k]:
                continue
            lo, hi = zs[k], zs[k + 1]
            target = (lo + hi) / 2 if hi - lo <= 2 * margin else min(max(d, lo + margin), hi - margin)
            key = (abs(target - d), -target)
            if best is None or key < best[0]:
                best = (key, target)
        if best is None:
            out.append((n, d))
            continue
        d_new = float(best[1])
        top = "/".join(sorted({struct.sites[s].specie.symbol for s in layers[max(z for z in zs if z <= d_new)]}))
        print(f"[wulff:termination] Moved {term} facet ({f.h} {f.k} {f.l}) plane by {d_new - d:+.2f} Å "
              f"(d={d:.4f} -> {d_new:.4f}) so {top} with {fewest} bond(s) out is the top layer.")
        out.append((n, d_new))
    return out


# ---------- Build a single-material nanocrystal (existing behavior) ----------

def build_nanocrystal(
    struct,
    facets: List[Facet],
    R: float,
    aspect=(1.0, 1.0, 1.0),
    origin_shift: np.ndarray | None = None,
    charges: dict | None = None,
):
    """
    Build a Wulff-cut particle from a bulk Structure with anisotropic aspect.
    With `charges`, planes of polar families requested with both
    terminations are placed between atomic layers (place_polar_planes).
    Returns (symbols, coords[N,3], planes).
    """
    planes = halfspaces(struct, facets, R, aspect=aspect)
    maxd = max(d for _, d in planes)
    rx, ry, rz = rep_ranges(struct.lattice, maxd)

    # Precompute once
    base = struct.frac_coords @ struct.lattice.matrix
    if origin_shift is not None:
        base = base - np.asarray(origin_shift, float)
    site_symbols = [site.specie.symbol for site in struct.sites]

    # Pre-generate all candidate coordinates to run auto-shifting
    all_coords_list = []
    for i, j, k in product(rx, ry, rz):
        shift = (i * struct.lattice.matrix[0]
                 + j * struct.lattice.matrix[1]
                 + k * struct.lattice.matrix[2])
        all_coords_list.append(base + shift)
    all_coords = np.vstack(all_coords_list)

    # Automatically shift planes to avoid cutting through atomic layers
    planes = auto_shift_planes(all_coords, planes, threshold=0.05, shift_amount=0.25)
    if charges is not None:
        site_index = np.tile(np.arange(len(base)), len(all_coords_list))
        planes = place_polar_planes(struct, facets, planes, all_coords, site_index, charges)

    # In case maxd increased, update ranges
    maxd = max(d for _, d in planes)
    rx, ry, rz = rep_ranges(struct.lattice, maxd)

    syms: List[str] = []
    pts: List[np.ndarray] = []

    for i, j, k in product(rx, ry, rz):
        shift = (i * struct.lattice.matrix[0]
                 + j * struct.lattice.matrix[1]
                 + k * struct.lattice.matrix[2])
        coords = base + shift
        mask = inside(coords, planes)
        idxs = np.where(mask)[0]
        if idxs.size:
            syms.extend(site_symbols[idx] for idx in idxs.tolist())
            pts.extend(coords[idxs])

    return syms, np.asarray(pts, float), planes


def fibonacci_sphere_normals(n: int) -> np.ndarray:
    n = max(12, int(n))
    idx = np.arange(n, dtype=float) + 0.5
    z = 1.0 - 2.0 * idx / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    theta = idx * (math.pi * (3.0 - math.sqrt(5.0)))
    normals = np.column_stack((r * np.cos(theta), r * np.sin(theta), z))
    return normals / np.linalg.norm(normals, axis=1)[:, None]


def sphere_halfspaces(R: float, n_planes: int = 192) -> list[Plane]:
    return [(n, float(R)) for n in fibonacci_sphere_normals(n_planes)]


def build_spherical_nanocrystal(
    struct,
    R: float,
    *,
    n_planes: int = 192,
    origin_shift: np.ndarray | None = None,
):
    """
    Build a near-spherical particle from a bulk Structure.
    """
    planes = sphere_halfspaces(R, n_planes=n_planes)
    maxd = float(R)
    rx, ry, rz = rep_ranges(struct.lattice, maxd)

    base = struct.frac_coords @ struct.lattice.matrix
    if origin_shift is not None:
        base = base - np.asarray(origin_shift, float)
    site_symbols = [site.specie.symbol for site in struct.sites]

    # Pre-generate all candidate coordinates to run auto-shifting
    all_coords_list = []
    for i, j, k in product(rx, ry, rz):
        shift = (i * struct.lattice.matrix[0]
                 + j * struct.lattice.matrix[1]
                 + k * struct.lattice.matrix[2])
        all_coords_list.append(base + shift)
    all_coords = np.vstack(all_coords_list)

    # Automatically shift planes to avoid cutting through atomic layers
    planes = auto_shift_planes(all_coords, planes, threshold=0.05, shift_amount=0.25)

    # In case maxd increased, update ranges
    maxd = max(d for _, d in planes)
    rx, ry, rz = rep_ranges(struct.lattice, maxd)

    syms: List[str] = []
    pts: List[np.ndarray] = []

    for i, j, k in product(rx, ry, rz):
        shift = (i * struct.lattice.matrix[0]
                 + j * struct.lattice.matrix[1]
                 + k * struct.lattice.matrix[2])
        coords = base + shift
        mask = inside(coords, planes)
        idxs = np.where(mask)[0]
        if idxs.size:
            syms.extend(site_symbols[idx] for idx in idxs.tolist())
            pts.extend(coords[idxs])

    return syms, np.asarray(pts, float), planes


def apply_core_lattice_fit(
    pts: np.ndarray,
    core_mask: np.ndarray,
    core_planes: List[Plane],
    shell_lattice_matrix: np.ndarray,
    core_lattice_matrix: np.ndarray,
    *,
    strain_width: float = 2.0,
    center_mode: str = "com",
) -> np.ndarray:
    """
    Affinely map core-region atoms from the shell lattice metric toward the core
    lattice metric, with a smooth blend to zero at the core boundary.
    """
    if len(pts) == 0 or not np.any(core_mask):
        return pts
    if not core_planes:
        return pts

    B_shell = np.array(shell_lattice_matrix, float)
    B_core = np.array(core_lattice_matrix, float)
    X = B_core @ np.linalg.inv(B_shell)

    if center_mode == "com":
        center = pts[core_mask].mean(axis=0)
    else:
        center = np.zeros(3, float)

    A = np.stack([n for (n, _d) in core_planes], axis=0)
    b = np.array([d for (_n, d) in core_planes], float)
    depth = (b[None, :] - pts @ A.T).min(axis=1)
    depth = np.clip(depth, 0.0, None)

    width = max(1e-6, float(strain_width))
    t = np.clip(depth / width, 0.0, 1.0)
    w = 0.5 - 0.5 * np.cos(np.pi * t)

    out = pts.copy()
    idx = np.where(core_mask)[0]
    v = out[idx] - center
    mapped = (X @ v.T).T + center
    out[idx] = (1.0 - w[idx, None]) * out[idx] + w[idx, None] * mapped
    return out
