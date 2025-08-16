# src/builder/geometry.py
from __future__ import annotations
import math
from itertools import product
from typing import List, Tuple, Dict

import numpy as np
from scipy.spatial import cKDTree

from .nc_types import Facet, Plane, MaterialSpec
from .facets import expand_facets, halfspaces  # halfspaces in facets.py (no circular import)

EPS = 1e-3

# ---------- Common helpers (existing in your code) ----------

def unit_normal(lattice_or_struct, hkl: Tuple[int,int,int]) -> np.ndarray:
    lattice = lattice_or_struct.lattice if hasattr(lattice_or_struct, "lattice") else lattice_or_struct
    v = lattice.reciprocal_lattice.get_cartesian_coords(hkl)
    return v / np.linalg.norm(v)

def rep_ranges(lattice, maxd: float):
    a, b, c = lattice.matrix
    n = lambda v: int(math.ceil((maxd + EPS) / np.linalg.norm(v))) + 1
    return (range(-n(a), n(a)+1), range(-n(b), n(b)+1), range(-n(c), n(c)+1))

def inside(pts: np.ndarray, planes: List[Plane]) -> np.ndarray:
    mask = np.ones(len(pts), dtype=bool)
    for n, d in planes:
        mask &= (pts @ n) <= (d + EPS)
        if not mask.any():
            break
    return mask

def dedupe_points(symbols: List[str], pts: np.ndarray, tol: float = 1e-3):
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

# ---------- Build a single-material nanocrystal (existing behavior) ----------

def build_nanocrystal(struct, facets: List[Facet], R: float, aspect=(1.0, 1.0, 1.0)):
    """
    Build a Wulff-cut particle from a bulk Structure with anisotropic aspect.
    Returns (symbols, coords[N,3], planes).
    """
    planes = halfspaces(struct, facets, R, aspect=aspect)
    maxd = max(d for _, d in planes)
    rx, ry, rz = rep_ranges(struct.lattice, maxd)

    # Precompute once
    base = struct.frac_coords @ struct.lattice.matrix
    site_symbols = [site.specie.symbol for site in struct.sites]

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
            # Index the Python list with Python ints
            syms.extend(site_symbols[idx] for idx in idxs.tolist())
            pts.extend(coords[idxs])  # coords is NumPy → fancy indexing is fine

    return syms, np.asarray(pts, float), planes


# ---------- Core–shell by labeling (shell-first) ----------

# ---------- Core–shell by labeling (shell-first) ----------

def _inside_halfspaces(planes: List[Plane], pts: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Boolean mask of points inside all planes (n·x ≤ d + tol for all (n,d)).
    """
    if len(pts) == 0:
        return np.zeros((0,), dtype=bool)
    A = np.stack([n for (n, d) in planes], axis=0)   # [P,3]
    b = np.array([d for (n, d) in planes], float)    # [P]
    vals = pts @ A.T                                 # [N,P]
    return (vals <= (b[None, :] + tol)).all(axis=1)

def _region_masks_from_planes(pts: np.ndarray, A: np.ndarray, b: np.ndarray, width: float):
    """
    Given planes defined by A[n]=n_f and b[d]=d_f, compute masks:
      - inner_mask: inside all planes (n·x ≤ d)
      - depth_in:   margin inside (min_f d - n·x), >=0; 0 on boundary
      - depth_out:  margin outside (max_f n·x - d), >=0; 0 on boundary
      - deep_core:  inner_mask & depth_in >= width
      - iface_band: (inner_mask & 0<depth_in<=width)  OR  (~inner_mask & depth_out<=width)
      - shell_mask: ~inner_mask
    """
    import numpy as np
    vals = pts @ A.T
    depth_in  = (b[None, :] - vals).min(axis=1)
    inner_mask = depth_in >= -1e-9
    depth_in = np.clip(depth_in, 0.0, None)

    depth_out = (vals - b[None, :]).max(axis=1)
    depth_out = np.clip(depth_out, 0.0, None)

    shell_mask = ~inner_mask
    deep_core  = inner_mask & (depth_in >= max(1e-6, float(width)))
    iface_band = (inner_mask & (depth_in > 0.0) & (depth_in <= max(1e-6, float(width)))) \
                 | (shell_mask & (depth_out <= max(1e-6, float(width))))
    return inner_mask, shell_mask, deep_core, iface_band, depth_in, depth_out


def _nn_stats_cation_to_anion(syms, pts, charges, cat_sel, an_sel):
    """
    Nearest anion to each selected cation (Euclidean). Returns numpy array of distances (can be empty).
    """
    import numpy as np
    try:
        from scipy.spatial import cKDTree
        use_tree = True
    except Exception:
        use_tree = False

    elems = np.array(syms)
    is_cat = np.array([charges.get(s, 0) > 0 for s in syms], bool)
    is_an  = np.array([charges.get(s, 0) < 0 for s in syms], bool)

    c_idx = np.where(is_cat & cat_sel)[0]
    a_idx = np.where(is_an & an_sel)[0]
    if c_idx.size == 0 or a_idx.size == 0:
        return np.array([], float)

    if use_tree:
        tree = cKDTree(pts[a_idx])
        d, _ = tree.query(pts[c_idx], k=1)
        return d.astype(float, copy=False)
    else:
        # fallback O(N*M) for small selections
        D = np.linalg.norm(pts[c_idx][:, None, :] - pts[a_idx][None, :, :], axis=2)
        return D.min(axis=1)


def build_core_shell_by_labeling(
    core: MaterialSpec,
    shell: MaterialSpec,
    *,
    shell_R: float,
    seeds: List[Facet],
    charges: Dict[str, int],
    surf_tol: float = 1.2,
    verbose: bool = False,
    # existing shrink options
    shrink_to_core_lattice: bool = False,
    strain_width: float = 2.0,
    center_mode: str = "com",
    # NEW:
    print_bond_stats: bool = False,
):
    """
    Build a shell-only NC at radius 'shell_R' (from shell.cif + shell.aspect),
    carve an inner region using core.aspect, then relabel atoms inside
    (shell cation→core cation, shell anion→core anion).

    If 'shrink_to_core_lattice' is True, apply an affine transform to inner atoms
    to map shell lattice → core lattice (B_core @ inv(B_shell)), blended smoothly
    near the core boundary over 'strain_width' Å. The transform is centered at
    the inner-core COM ('com') or the origin ('origin').
    """
    from pymatgen.core import Structure
    import numpy as np

    # 1) Load CIFs
    s_shell = Structure.from_file(shell.cif)
    s_core  = Structure.from_file(core.cif)

    # 2) Build shell NC
    asp_shell = getattr(shell, "aspect", (1.0, 1.0, 1.0)) or (1.0, 1.0, 1.0)
    facets_shell = expand_facets(s_shell, seeds, proper_only=True)
    planes_shell = halfspaces(s_shell, facets_shell, R=shell_R, aspect=asp_shell)
    syms, pts = build_nanocrystal(s_shell, facets_shell, R=shell_R, aspect=asp_shell)[:2]
    if len(pts) == 0:
        return syms, pts

    # 3) Inner region from core aspect (same seeds & R)
    pts_pre = pts.copy()  # keep a copy for "pre" stats
    asp_core   = getattr(core, "aspect", (0.5, 0.5, 0.5)) or (0.5, 0.5, 0.5)
    planes_core = halfspaces(s_shell, facets_shell, R=shell_R, aspect=asp_core)
    # inside test: n·x ≤ d + tol for all planes
    A_planes = np.stack([n for (n, d) in planes_core], axis=0)   # [P,3]
    b_planes = np.array([d for (n, d) in planes_core], float)    # [P]
    inner_mask = (pts @ A_planes.T <= (b_planes[None, :] + 1e-6)).all(axis=1)

    inner_mask_pre, shell_mask_pre, deep_core_pre, iface_band_pre, depth_in_pre, depth_out_pre = \
        _region_masks_from_planes(pts_pre, A_planes, b_planes, width=strain_width)

    def _print_stats(tag, dists):
        if dists.size == 0:
            print(f"[bond-stats] {tag}: no pairs")
            return
        import numpy as np
        mn, mx = float(dists.min()), float(dists.max())
        mean, med = float(dists.mean()), float(np.median(dists))
        p05, p95 = float(np.percentile(dists, 5)), float(np.percentile(dists, 95))
        print(f"[bond-stats] {tag}: N={dists.size}  min={mn:.3f}  p05={p05:.3f}  med={med:.3f}  "
              f"mean={mean:.3f}  p95={p95:.3f}  max={mx:.3f} (Å)")

    # 4) Relabel mapping by charge (1 cation + 1 anion in each material)
    def split_by_charge(elements):
        cats = [e for e in elements if charges.get(e, 0) > 0]
        ans  = [e for e in elements if charges.get(e, 0) < 0]
        return cats, ans

    shell_elems = sorted(set(syms))
    core_elems  = sorted({sp.symbol for sp in s_core.types_of_specie})
    shell_cat, shell_an = split_by_charge(shell_elems)
    core_cat,  core_an  = split_by_charge(core_elems)

    if not (len(shell_cat) == len(shell_an) == len(core_cat) == len(core_an) == 1):
        raise ValueError(
            f"Expected exactly one cation and one anion in shell/core. "
            f"shell: {shell_cat}/{shell_an}, core: {core_cat}/{core_an}"
        )

    cat_from, an_from = shell_cat[0], shell_an[0]
    cat_to,   an_to   = core_cat[0],  core_an[0]

    out_syms = list(syms)
    if inner_mask.any():
        for i, inside in enumerate(inner_mask):
            if not inside:
                continue
            if out_syms[i] == cat_from:
                out_syms[i] = cat_to
            elif out_syms[i] == an_from:
                out_syms[i] = an_to

    # 5) Optional: shrink inner atoms to core lattice + interface strain
    if shrink_to_core_lattice and inner_mask.any():
        B_shell = np.array(s_shell.lattice.matrix, float)  # 3x3
        B_core  = np.array(s_core.lattice.matrix,  float)  # 3x3
        try:
            X = B_core @ np.linalg.inv(B_shell)            # affine map: shell→core in lattice frame
        except np.linalg.LinAlgError:
            if verbose:
                print("WARNING: shell lattice not invertible; skipping core-lattice fit.")
            return out_syms, pts

        # choose center of transform
        if center_mode == "com":
            c = pts[inner_mask].mean(axis=0)
        else:
            c = np.zeros(3, float)

        # softness near boundary (cosine smoothstep)
        # depth = min(d_i - n_i·x) for planes_core; 0 at boundary, larger inside
        depth = (b_planes[None, :] - pts @ A_planes.T).min(axis=1)   # [N]
        depth = np.clip(depth, 0.0, None)

        # Blend only inner atoms; w=1 deep inside, w→0 at boundary over 'strain_width'
        eps = max(1e-6, float(strain_width))
        t = np.clip(depth / eps, 0.0, 1.0)               # 0..1
        w = 0.5 - 0.5 * np.cos(np.pi * t)                # smooth (C1) 0→1→ (cosine ease)

        # Apply map to inner atoms with weight w
        P = pts.copy()
        inner_idx = np.where(inner_mask)[0]
        v = P[inner_idx] - c
        Pv = (X @ v.T).T + c
        # mix
        P[inner_idx] = (1.0 - w[inner_idx, None]) * P[inner_idx] + w[inner_idx, None] * Pv
        pts = P

        if verbose:
            print(f"[shrink] applied core-lattice affine fit with strain width {eps:.3f} Å "
                  f"and center='{center_mode}'")

    if verbose:
        print(f"[label] relabeled {int(inner_mask.sum())} atoms as core "
              f"({cat_from}→{cat_to}, {an_from}→{an_to})")

    if verbose or print_bond_stats:
        import numpy as np

        # masks for POST stats: keep region membership from pre-shrink (apples-to-apples)
        # intra-region (both cat and anion limited to same region)
        core_core_pre  = _nn_stats_cation_to_anion(out_syms, pts_pre, charges, deep_core_pre, deep_core_pre)
        iface_iface_pre= _nn_stats_cation_to_anion(out_syms, pts_pre, charges, iface_band_pre, iface_band_pre)
        shell_shell_pre= _nn_stats_cation_to_anion(out_syms, pts_pre, charges, shell_mask_pre, shell_mask_pre)

        core_core_post   = _nn_stats_cation_to_anion(out_syms, pts, charges, deep_core_pre,  deep_core_pre)
        iface_iface_post = _nn_stats_cation_to_anion(out_syms, pts, charges, iface_band_pre, iface_band_pre)
        shell_shell_post = _nn_stats_cation_to_anion(out_syms, pts, charges, shell_mask_pre, shell_mask_pre)

        print("\n[bond-stats] Nearest cation–anion distances (Å); intra-region only; "
              "regions defined pre-shrink:")
        _print_stats("core(deep)   pre", core_core_pre)
        _print_stats("core(deep)  post", core_core_post)
        _print_stats("interface    pre", iface_iface_pre)
        _print_stats("interface   post", iface_iface_post)
        _print_stats("shell        pre", shell_shell_pre)
        _print_stats("shell       post", shell_shell_post)

        # Also show "to-global" (nearest anion anywhere) for each cation region
        core_any_pre   = _nn_stats_cation_to_anion(out_syms, pts_pre, charges, deep_core_pre,  np.ones(len(out_syms), bool))
        core_any_post  = _nn_stats_cation_to_anion(out_syms, pts,     charges, deep_core_pre,  np.ones(len(out_syms), bool))
        iface_any_pre  = _nn_stats_cation_to_anion(out_syms, pts_pre, charges, iface_band_pre, np.ones(len(out_syms), bool))
        iface_any_post = _nn_stats_cation_to_anion(out_syms, pts,     charges, iface_band_pre, np.ones(len(out_syms), bool))
        shell_any_pre  = _nn_stats_cation_to_anion(out_syms, pts_pre, charges, shell_mask_pre, np.ones(len(out_syms), bool))
        shell_any_post = _nn_stats_cation_to_anion(out_syms, pts,     charges, shell_mask_pre, np.ones(len(out_syms), bool))

        print("\n[bond-stats] Nearest cation–anion distances (Å); cations restricted to region, "
              "nearest anion anywhere:")
        _print_stats("core→any   pre", core_any_pre)
        _print_stats("core→any  post", core_any_post)
        _print_stats("iface→any  pre", iface_any_pre)
        _print_stats("iface→any post", iface_any_post)
        _print_stats("shell→any  pre", shell_any_pre)
        _print_stats("shell→any post", shell_any_post)


    return out_syms, pts



