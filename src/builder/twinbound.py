# twinbound.py
# Generic twin-boundary utilities for QD_Builder
# Dependency: numpy (ASE wrapper is optional)

from __future__ import annotations
import numpy as np
from typing import Iterable, List, Tuple, Dict, Any, Union
from scipy.spatial import KDTree

Array = np.ndarray


# -----------------------------
# ---- Parsing / geometry  ----
# -----------------------------
def parse_hkl(hkl: Union[str, int, Iterable[int]]) -> np.ndarray:
    """
    Parse Miller indices from several common YAML styles:
      - "111", "1-10", "-1-1-1"
      - [1, 1, 1], (1, -1, 0)
      - 111 (int) -> [1,1,1],  100 -> [1,0,0],  110 -> [1,1,0]
    Returns int array [h,k,l].
    """
    if isinstance(hkl, (list, tuple, np.ndarray)):
        arr = np.asarray(hkl, dtype=int).ravel()
        if arr.size != 3:
            raise ValueError(f"hkl iterable must have length 3, got {arr}")
        return arr

    if isinstance(hkl, int):
        s = str(hkl)
        if s[0] == "-":
            raise ValueError(
                "Negative integer HKL like -111 is ambiguous. Use string '-1-1-1' instead."
            )
        if len(s) != 3 or not s.isdigit():
            raise ValueError(f"Integer HKL must have 3 digits (e.g. 111, 110, 100). Got {hkl}")
        return np.array([int(s[0]), int(s[1]), int(s[2])], dtype=int)

    if isinstance(hkl, str):
        s = hkl.replace(",", " ").replace("(", " ").replace(")", " ").strip()
        if " " in s:
            # tokens separated by spaces
            toks = s.split()
            if len(toks) != 3:
                raise ValueError(f"Cannot parse HKL from '{hkl}'")
            return np.array([int(t) for t in toks], dtype=int)
        # compact form like "1-10" or "-1-1-1" or "111"
        vals: List[int] = []
        i = 0
        while i < len(s):
            sign = 1
            if s[i] == "+":
                i += 1
            elif s[i] == "-":
                sign = -1
                i += 1
            if i >= len(s) or not s[i].isdigit():
                raise ValueError(f"Cannot parse HKL from '{hkl}' near position {i}")
            vals.append(sign * int(s[i]))
            i += 1
        if len(vals) != 3:
            raise ValueError(f"HKL must have 3 integers, got '{hkl}' -> {vals}")
        return np.array(vals, dtype=int)

    raise TypeError(f"Unsupported HKL type: {type(hkl)}")

def _unit_normals(planes):
    """Return array of unit normals for planes=[(n,d),...]."""
    N = np.stack([n for (n, d) in planes], axis=0)  # [P,3]
    L = np.linalg.norm(N, axis=1, keepdims=True)
    L[L == 0] = 1.0
    return N / L

def nearest_plane_and_distance(R: np.ndarray, planes):
    """
    For each point x, return (idx, d_perp) of the nearest plane boundary
    of polyhedron { n·x ≤ d }. d_perp ≥ 0 inside.
    """
    A = np.stack([n for (n, d) in planes], axis=0)   # [P,3]
    b = np.array([d for (n, d) in planes], float)    # [P]
    norms = np.linalg.norm(A, axis=1); norms[norms == 0] = 1.0
    slack = b[None, :] - R @ A.T                    # ≥0 inside
    d_perp = slack / norms[None, :]                 # [N,P]
    idx = np.argmin(d_perp, axis=1)
    dmin = d_perp[np.arange(len(R)), idx]
    return idx, dmin


def cell_columns(A: Array) -> Array:
    A = np.asarray(A, float)
    if A.shape != (3, 3):
        raise ValueError(f"Cell must be 3x3, got {A.shape}")
    # Pymatgen: lattice.matrix has vectors as ROWS; convert to columns
    return A.T


def plane_normal_from_hkl(A_cols: Array, hkl: Iterable[int]) -> Array:
    """
    n_hat = normalize( A^{-T} @ [h,k,l] ), with A_cols = [a b c] (columns, Å)
    """
    hkl = np.asarray(hkl, float).ravel()
    n = np.linalg.solve(np.asarray(A_cols).T, hkl)  # A^{-T} * [h,k,l]
    return n / np.linalg.norm(n)

def interplanar_spacing(A_cols: Array, hkl: Iterable[int]) -> float:
    """d_(hkl) = 1 / || A^{-T} [h,k,l] ||"""
    hkl = np.asarray(hkl, float).ravel()
    n_unnorm = np.linalg.solve(np.asarray(A_cols).T, hkl)
    return 1.0 / np.linalg.norm(n_unnorm)

def _mirror_transform(n_hat: Array, p0: Array):
    n_hat = np.asarray(n_hat, float) / np.linalg.norm(n_hat)
    p0    = np.asarray(p0, float)
    def M(X: Array) -> Array:
        X = np.asarray(X, float)
        # vectorized reflection across plane (n_hat, p0)
        return X - 2.0 * np.outer((X - p0) @ n_hat, n_hat)
    return M

def _point_in_halfspaces(pts: Array, H: Array, h: Array, tol: float = 1e-6) -> Array:
    # H x <= h  (row-wise); returns boolean mask for pts inside polyhedron
    return (H @ pts.T <= (h + tol)).all(axis=0)

def _inside_intervals(tvals: Array, intervals: List[Tuple[float, float]]) -> Array:
    mask = np.zeros_like(tvals, dtype=bool)
    for lo, hi in intervals:
        a, b = (lo, hi) if lo <= hi else (hi, lo)
        mask |= (tvals >= a) & (tvals <= b)
    return mask

def _make_twin_transform(
    A_cols: Array,
    n_hat: Array,
    plane_point: Array,
    shift_layers: float,
    d_hkl: float,
    parallel_shift_fractional: Array
):
    """
    T(X) = mirror_{(n_hat, plane_point)}(X) + (shift_layers * d_hkl * n_hat) + A_cols @ parallel_shift_fractional
    """
    A = np.asarray(A_cols, float)
    M = _mirror_transform(n_hat, plane_point)
    s_perp = float(shift_layers) * float(d_hkl) * n_hat
    s_par  = (A @ np.asarray(parallel_shift_fractional, float).reshape(3, 1)).reshape(3,)
    def T(X: Array) -> Array:
        return M(X) + s_par + s_perp
    return T

def refill_by_twin_template(
    *,
    # geometry / lattice
    A_cols: Array,
    hkl: Iterable[int],
    plane_point: Array,                        # point on the mirror plane (e.g., midplane)
    intervals_angstrom: List[Tuple[float,float]],
    parallel_shift_fractional: Array,         # [fu,fv,fw] w.r.t. a,b,c
    shift_layers: float,                      # multiples of d_(hkl)
    # NC data
    parent_pos: Array,                        # current NC positions BEFORE twin ops? (see call below)
    parent_species: List[str],
    # polytope for the *original* Wulff (size/shape)
    halfspaces_H: Array,
    halfspaces_h: Array,
    # controls
    refill_min_separation: float = 1.2,
    refill_dedup_tolerance: float = 3.0,
    # optional species swap within the slab
    swap_sublattice: Dict[str, str] | None = None,
) -> Tuple[Array, List[str]]:
    """
    Generate the ideal *twin-slab* occupancy as a template by transforming the
    parent lattice via the configured twin (mirror + shifts), then add only the
    missing atoms up to the *original* Wulff boundary.

    Returns (new_positions, new_species) to append.
    """
    parent_pos  = np.asarray(parent_pos, float)
    parent_spec = np.asarray(parent_species, object)

    n_hat = plane_normal_from_hkl(A_cols, hkl)
    d_hkl = interplanar_spacing(A_cols, hkl)
    T     = _make_twin_transform(A_cols, n_hat, np.asarray(plane_point, float),
                                 shift_layers, d_hkl, parallel_shift_fractional)

    # 1) Build the twin template by transforming the *parent* lattice
    templ_pos = T(parent_pos)
    templ_spec = parent_spec.copy()

    # 2) Keep only points that lie inside the requested slab intervals (in *twin* coordinates)
    tvals = (templ_pos - plane_point) @ n_hat
    slab_mask = _inside_intervals(tvals, intervals_angstrom)
    templ_pos  = templ_pos[slab_mask]
    templ_spec = templ_spec[slab_mask]

    # 3) Clip to the original Wulff polyhedron (size/shape defined by original lattice)
    if halfspaces_H is not None and halfspaces_h is not None:
        in_poly = _point_in_halfspaces(templ_pos, halfspaces_H, halfspaces_h, tol=1e-6)
        templ_pos  = templ_pos[in_poly]
        templ_spec = templ_spec[in_poly]

    # 4) Species swap rule inside the slab (Cd<->Se etc.), if requested
    if swap_sublattice:
        swap = {str(k): str(v) for k, v in swap_sublattice.items()}
        templ_spec = np.array([swap.get(s, s) for s in templ_spec], dtype=object)

    # 5) Deduplicate vs the *current NC atoms after twin ops so far*.
    #    NOTE: the caller should pass "current_nc_pos" to this filter, not "parent_pos".
    #    If that's not possible in your call site, change 'existing_pos' below.
    existing_pos = parent_pos  # UPDATE to 'current_nc_pos' at call site (see section 2)

    tree_exist = cKDTree(existing_pos)
    dmin, _    = tree_exist.query(templ_pos, k=1, distance_upper_bound=refill_dedup_tolerance)
    add_mask   = np.isinf(dmin)  # only positions not already present (within dedup_tol)

    cand_pos  = templ_pos[add_mask]
    cand_spec = templ_spec[add_mask].tolist()

    if len(cand_pos) == 0:
        return np.zeros((0, 3)), []

    # 6) Enforce min separation vs existing AND among candidates (greedy)
    acc_pos: list[Array] = []
    acc_spec: list[str]  = []
    for p, s in zip(cand_pos, cand_spec):
        # check vs existing
        if tree_exist.query(p, distance_upper_bound=refill_min_separation)[0] != np.inf:
            continue
        # check vs accepted
        if acc_pos:
            if cKDTree(np.array(acc_pos)).query(p, distance_upper_bound=refill_min_separation)[0] != np.inf:
                continue
        acc_pos.append(p)
        acc_spec.append(s)

    if not acc_pos:
        return np.zeros((0, 3)), []

    return np.array(acc_pos), acc_spec

def reflect_about_plane(R: Array, n_hat: Array, c: float) -> Array:
    """
    Reflect coordinates across plane { x | n_hat·x = c }.
    R: (N,3), n_hat: (3,), c scalar. Returns reflected copy (does not modify in-place).
    """
    s = R @ n_hat - c
    return R - 2.0 * s[:, None] * n_hat


def _resolve_origin(R: Array, origin: Union[str, Iterable[float], None]) -> np.ndarray:
    if origin is None or (isinstance(origin, str) and origin.lower() == "center"):
        return R.mean(axis=0)
    arr = np.asarray(origin, float).ravel()
    if arr.size != 3:
        raise ValueError(f"origin must be 'center' or [x,y,z], got {origin}")
    return arr


# -----------------------------
# --------- Merging ----------
# -----------------------------
def merge_close_points(R: Array, tol: float = 0.10) -> Array:
    """
    Deduplicate nearly-overlapping points (Å tolerance).
    Simple grid-hash + unique; preserves approximate geometry, drops exact overlaps at twins.
    """
    if tol <= 0:
        return R

    # Quantize coordinates to a grid of size tol
    Q = np.floor((R - R.min(axis=0)) / tol + 0.5).astype(np.int64)
    # Use structured dtype to unique rows
    dt = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    Qs = Q.view(dt).ravel()
    _, idx = np.unique(Qs, return_index=True)
    return R[np.sort(idx)]

# Add this new function to twinbound.py

def merge_close_points_species_aware(
    syms: List[str], pts: Array, tol: float
) -> Tuple[List[str], Array]:
    """
    Deduplicate nearly-overlapping points with a species-aware tolerance.

    This function only removes points if they are of the SAME species and
    within the specified tolerance (Å). It preserves close contacts between
    different species (e.g., bonds).
    """
    if tol <= 0:
        return syms, pts

    # 1. Create a unique integer ID for each chemical symbol.
    unique_syms = sorted(list(set(syms)))
    sym_to_id = {s: i for i, s in enumerate(unique_syms)}
    s_ids = np.array([sym_to_id[s] for s in syms], dtype=np.int64).reshape(-1, 1)

    # 2. Quantize coordinates to a grid of size `tol`.
    Q = np.floor((pts - pts.min(axis=0)) / tol + 0.5).astype(np.int64)

    # 3. Combine quantized coordinates and species ID into a single array.
    #    Each row is now (qx, qy, qz, species_id).
    QS = np.hstack([Q, s_ids])

    # 4. Find the unique rows. This finds the first occurrence of each
    #    species in each grid cell, effectively merging the duplicates.
    #    We use a structured dtype to unique the rows efficiently.
    dtype = QS.dtype
    structured_qs = np.ascontiguousarray(QS).view(f'V{QS.shape[1] * dtype.itemsize}')
    _, idx = np.unique(structured_qs, return_index=True)

    # 5. Return the filtered symbols and points.
    #    We sort the indices to preserve the original atom ordering as much as possible.
    sorted_idx = np.sort(idx)
    return [syms[i] for i in sorted_idx], pts[sorted_idx]


def _planes_arrays(planes):
    """planes -> (A, b, norms). planes is [(n,d)] with n not necessarily unit length."""
    A = np.stack([n for (n, d) in planes], axis=0)   # [P,3]
    b = np.array([d for (n, d) in planes], float)    # [P]
    norms = np.linalg.norm(A, axis=1)                # [P]
    norms[norms == 0] = 1.0
    return A, b, norms

def min_distance_to_poly_planes(R: np.ndarray, planes, positive_inside: bool = True) -> np.ndarray:
    """
    For NC defined by n·x ≤ d, returns per-point *perpendicular* distance
    to the nearest plane boundary (>=0 inside). If normals aren't unit,
    we normalize by ||n||.
    """
    A, b, norms = _planes_arrays(planes)
    # slack = d - n·x  (>=0 inside), divide by ||n||
    slack = b[None, :] - R @ A.T
    d_perp = slack / norms[None, :]
    dmin = d_perp.min(axis=1)
    return dmin if positive_inside else -dmin

# twinbound.py

# ... (keep all other functions as they are) ...

def refill_from_original_template(
    cur_syms: list,
    cur_pts: np.ndarray,
    tpl_syms: list,
    tpl_pts: np.ndarray,
    planes, # These are the planes of the ORIGINAL Wulff shape
    n_hat: np.ndarray,
    origin: np.ndarray,
    intervals_A: List[Tuple[float, float]],
    min_sep_tol: float = 1.0,
    pad_A: float = 1e-3,
) -> Tuple[list, np.ndarray]:
    """
    Refills voids created by a twin glide using a fully twinned template, but
    constrains the new atoms to lie within the original Wulff shape boundary.
    """
    if tpl_pts.size == 0 or cur_pts.size == 0:
        return cur_syms, cur_pts

    print(f"[refill] Starting boundary-aware refill against twinned template.")

    # 1. Use a KDTree to find which points from the twinned template are missing
    #    from the current (sheared) nanocrystal.
    kdtree = KDTree(cur_pts)
    dist, _ = kdtree.query(tpl_pts, k=1)
    missing_mask = (dist > min_sep_tol)

    if not np.any(missing_mask):
        print("[refill] No missing template sites detected. Nothing to add.")
        return cur_syms, cur_pts

    candidate_pts = tpl_pts[missing_mask]
    candidate_syms = [s for s, m in zip(tpl_syms, missing_mask) if m]
    print(f"[refill] Found {len(candidate_pts)} potential missing sites.")

    # 2. **CRITICAL FILTER**: Keep only candidates that are inside the original
    #    Wulff shape, defined by the 'planes' argument (n·x <= d).
    A = np.stack([n for (n, d) in planes], axis=0)
    b = np.array([d for (n, d) in planes], float)
    
    # A point `p` is inside if (p @ A.T - b) <= 0 for all planes.
    # We add a small tolerance to avoid floating point issues at the surface.
    inside_wulff_mask = (candidate_pts @ A.T <= (b[None, :] + 1e-6)).all(axis=1)
    
    candidate_pts = candidate_pts[inside_wulff_mask]
    candidate_syms = [s for s, m in zip(candidate_syms, inside_wulff_mask) if m]

    if not candidate_syms:
        print("[refill] No missing sites are located inside the Wulff boundary.")
        return cur_syms, cur_pts
    print(f"[refill] {len(candidate_pts)} sites are inside the Wulff boundary.")

    # 3. **SIDE-FILLING FILTER**: Further restrict candidates to the layer-range
    #    of the original twin intervals to fill gaps "on the sides".
    t_cand = (candidate_pts - origin) @ n_hat
    in_interval_mask = np.zeros(len(candidate_pts), dtype=bool)
    for (ta, tb) in intervals_A:
        a, b_upper = min(ta, tb), max(ta, tb)
        in_interval_mask |= (t_cand >= (a - pad_A)) & (t_cand <= (b_upper + pad_A))

    added_pts = candidate_pts[in_interval_mask]
    added_syms = [s for s, m in zip(candidate_syms, in_interval_mask) if m]

    if not added_syms:
        print("[refill] No sites to add after final interval check.")
        return cur_syms, cur_pts

    print(f"[refill] Adding {len(added_syms)} sites to fill voids within the twin slab region.")

    # 4. Combine the original points with the new, filtered, refilled points.
    final_syms = cur_syms + added_syms
    final_pts = np.vstack([cur_pts, added_pts])

    return final_syms, final_pts

# ... (keep all other functions as they are) ...

def refill_against_template(
    cur_syms: list,
    cur_pts: np.ndarray,
    tpl_syms: list,
    tpl_pts: np.ndarray,
    planes,
    n_hat: np.ndarray,
    origin: np.ndarray,
    intervals_A: List[Tuple[float, float]],
    pad_A: float = 1e-3,
    site_match_tol: float = 0.9,
    min_sep_tol: float = 0.8,
    scope: str = "surface",
    shell_thickness: float = 2.0,
    facet_mode: str = "sides",
    top_cos_thresh: float = 0.85,
    refill_region: str = "inside",
    orient_delta: float = 0.20,
    snap_out_eps: float = 0.15,
    snap_offset: float = 0.02,
    layer_gap_tol: float = 0.90,      # consider “same layer” if cross-plane sep < this (Å)
) -> Tuple[list, np.ndarray]:
    """
    Refill missing template sites to repair the Wulff hull cut by a twin glide.

    - Multi-plane facet test: consider all near-min planes (≤ orient_delta).
    - Snap-in: if a candidate lies slightly outside the hull (≤ snap_out_eps),
      project it inside along violated plane(s) by that excess + snap_offset.
    - Layer-aware collision check: reject a candidate only if there exists a current
      atom that is both close in-plane (< min_sep_tol) AND in the same layer
      (< layer_gap_tol along the local facet normal).
    """
    if tpl_pts.size == 0:
        return cur_syms, cur_pts

    # ---- geometry helpers / arrays ----
    U = _unit_normals(planes)
    A = np.stack([n for (n, d) in planes], axis=0)   # [P,3]
    b = np.array([d for (n, d) in planes], float)    # [P]
    norms = np.linalg.norm(A, axis=1); norms[norms == 0] = 1.0

    # ---- region mask (inside or outside the twin intervals) ----
    t_tpl = (tpl_pts - origin) @ n_hat
    inside_any = np.zeros(len(tpl_pts), dtype=bool)
    for (ta, tb) in intervals_A:
        a, b2 = (float(ta), float(tb))
        if a > b2: a, b2 = b2, a
        inside_any |= (t_tpl >= (a - pad_A)) & (t_tpl <= (b2 + pad_A))
    region_mask = inside_any if refill_region.lower() == "inside" else ~inside_any

    # ---- surface shell mask ----
    if scope == "surface":
        slack = b[None, :] - tpl_pts @ A.T          # [N,P]
        d_perp = slack / norms[None, :]
        dmin = d_perp.min(axis=1)
        surf_mask = dmin <= (float(shell_thickness) + 1e-8)
    else:
        surf_mask = np.ones(len(tpl_pts), dtype=bool)

    # ---- facet-orientation mask (near-min planes; exclude top if requested) ----
    slack = b[None, :] - tpl_pts @ A.T             # recompute (N,P)
    d_perp = slack / norms[None, :]                # ≥0 inside
    dmin = d_perp.min(axis=1)
    near = d_perp <= (dmin[:, None] + float(orient_delta))  # planes within Δ of min

    cos_all = np.abs((U @ n_hat).ravel())          # [P], |cosθ| to twin normal
    cosNP = np.broadcast_to(cos_all[None, :], d_perp.shape)

    if facet_mode in ("sides", "exclude_top"):
        facet_mask = (near & (cosNP < float(top_cos_thresh))).any(axis=1)
    elif facet_mode == "top_only":
        facet_mask = (near & (cosNP >= float(top_cos_thresh))).any(axis=1)
    else:
        facet_mask = np.ones(len(tpl_pts), dtype=bool)

    # ---- gather eligible template sites ----
    keep_tpl = region_mask & surf_mask & facet_mask
    n_region = int(region_mask.sum())
    n_surf   = int(surf_mask.sum())
    n_facet  = int(facet_mask.sum())
    n_keep   = int(keep_tpl.sum())
    print(f"[refill:filters] region={n_region} surf={n_surf} facet={n_facet} keep={n_keep}/{len(tpl_pts)}")

    if not keep_tpl.any():
        print("[refill] no eligible template sites (region/scope/facet)")
        return cur_syms, cur_pts

    Pk = tpl_pts[keep_tpl].copy()
    Sk = [s for s, m in zip(tpl_syms, keep_tpl) if m]
    print(f"[refill:candidates] eligible={len(Pk)}  min_sep={min_sep_tol:.2f} Å")

    # ---- precompute near-min plane info for each candidate (for snap & local normal) ----
    slack_k = b[None, :] - Pk @ A.T                # [K,P]
    d_perp_k = slack_k / norms[None, :]
    near_k = d_perp_k <= (d_perp_k.min(axis=1)[:, None] + float(orient_delta))
    rep_plane_idx = np.argmax(near_k, axis=1)      # representative “local facet” per candidate

    # ---- optional de-dup among new additions (grid key) ----
    def qhash(P: np.ndarray, tol: float) -> list[tuple[int,int,int]]:
        Q = np.floor((P - base) / tol + 0.5).astype(np.int64)
        return [(int(q0), int(q1), int(q2)) for q0, q1, q2 in Q]

    chosen_keys: set = set()
    if site_match_tol > 0:
        base = np.minimum(cur_pts.min(axis=0) if len(cur_pts) else Pk.min(axis=0), Pk.min(axis=0))

    # ---- main survivor loop (layer-aware collision + snap-in) ----
    added_syms, added_pts = [], []
    survivors = 0
    for i in range(len(Pk)):
        p = Pk[i]

        # Snap slightly outside (compose from ALL violated planes)
        row = d_perp_k[i]                   # [P]
        viol = row < 0
        if np.any(viol):
            worst = float((-row[viol]).max())
            if worst <= float(snap_out_eps):
                dirs = (A[viol] / norms[viol][:, None])                 # unit plane normals
                mags = (-row[viol]) + float(snap_offset)
                delta = (dirs * mags[:, None]).sum(axis=0)
                n_comb = dirs.sum(axis=0)
                n_norm = np.linalg.norm(n_comb)
                if n_norm > 0:
                    delta += float(snap_offset) * (n_comb / n_norm)     # tiny nudge
                p = p + delta
            else:
                # too far outside → would be cut away anyway
                continue

        # Layer-aware collision check vs current atoms
        if len(cur_pts):
            j = int(rep_plane_idx[i])
            n_loc = (A[j] / norms[j])                                   # unit normal of local facet
            diff = cur_pts - p                                          # [N,3]
            cross = np.abs(diff @ n_loc)                                # cross-plane sep
            inpl = np.linalg.norm(diff - (diff @ n_loc)[:, None] * n_loc[None, :], axis=1)
            if np.any((cross < float(layer_gap_tol)) & (inpl < float(min_sep_tol))):
                continue

        # de-dup among new additions only
        if site_match_tol > 0:
            k = qhash(p[None, :], site_match_tol)[0]
            if k in chosen_keys:
                continue
            chosen_keys.add(k)

        survivors += 1
        added_syms.append(Sk[i])
        added_pts.append(p)

    print(f"[refill:sep] survivors={survivors}")
    print(f"[refill:add] added={len(added_pts)} (snap≤{snap_out_eps:.2f} Å)")

    if added_pts:
        print(f"[refill] +{len(added_pts)} site(s) "
              f"(region={refill_region}, facets={facet_mode}, "
              f"site_tol={site_match_tol:.2f} Å, min_sep={min_sep_tol:.2f} Å, "
              f"Δfacet={orient_delta:.2f} Å, snap≤{snap_out_eps:.2f} Å)")
        cur_syms = list(cur_syms) + added_syms
        cur_pts  = np.vstack([cur_pts, np.asarray(added_pts, float)])
    else:
        print("[refill] nothing to add")

    return cur_syms, cur_pts


# -----------------------------
# ------ Public API -----------
# -----------------------------
# twinbound.py

# ... (other functions like parse_hkl, refill_from_original_template, etc. remain here) ...

def apply_twin_directive(
    R: Array,
    A: Array,
    directive: Dict[str, Any],
    default_origin: Union[str, Iterable[float]] = "center",
    species: Optional[List[str]] = None,
    charges: Optional[Dict[str, int]] = None,
    perform_stitch: bool = True,  # <-- MODIFICATION: Stitching is now optional
) -> Array:
    """
    Twin directive with mirror/glide and optional sublattice swap.
    ... (docstring) ...
    """
    import numpy as _np

    # ---------- Plane / lattice ----------
    if "hkl" not in directive:
        raise ValueError("Twin directive requires 'hkl'.")
    A_cols = cell_columns(A)
    hkl = parse_hkl(directive["hkl"])
    n_hat = plane_normal_from_hkl(A_cols, hkl)
    d_hkl = interplanar_spacing(A_cols, hkl)

    origin = _resolve_origin(R, directive.get("origin", default_origin))

    # ---------- Intervals (Å) ----------
    segments = (
        directive.get("intervals_angstrom")
        or directive.get("segments_angstrom")
        or directive.get("intervals")
        or directive.get("segments")
    )
    segments_layers = directive.get("segments_layers") or directive.get("intervals_layers")
    if segments is None and segments_layers is None:
        raise ValueError("Provide 'intervals_angstrom' (or aliases) or 'intervals_layers' in twins.")

    segs_A: List[Tuple[float, float]] = []
    if segments is not None:
        for (t1, t2) in segments:
            segs_A.append((float(t1), float(t2)))
    if segments_layers is not None:
        for (n1, n2) in segments_layers:
            segs_A.append((float(n1) * d_hkl, float(n2) * d_hkl))

    # ---------- Behavior knobs ----------
    snap = bool(directive.get("snap_to_layers", False))
    mirror_at = str(directive.get("mirror_at", "midplane")).lower()
    tol_merge = float(directive.get("merge_tolerance", 0.0))
    pad = float(directive.get("interval_pad", 1e-3))

    # ---------- Normal (along n̂) shift ----------
    operation = str(directive.get("operation", "mirror")).lower()
    if operation not in ("mirror", "mirror+shift"):
        raise ValueError(f"Unknown twin operation '{operation}'.")
    shift_ang = directive.get("shift_angstrom", None)
    shift_layers = directive.get("shift_layers", None)
    shift_uc = directive.get("shift_unitcell_fraction", None)

    if operation == "mirror+shift":
        if shift_ang is not None:
            s_normal = float(shift_ang)
        elif shift_layers is not None:
            s_normal = float(shift_layers) * d_hkl
        elif shift_uc is not None:
            s_normal = 3.0 * float(shift_uc) * d_hkl
        else:
            s_normal = 0.5 * d_hkl
    else:
        s_normal = 0.0

    # ---------- NEW: Parallel (in-plane) shift ----------
    ref = _np.array([1.0, 0.0, 0.0])
    if abs(_np.dot(ref, n_hat)) > 0.9:
        ref = _np.array([0.0, 1.0, 0.0])
    e1 = _np.cross(n_hat, ref);  e1 /= _np.linalg.norm(e1)
    e2 = _np.cross(n_hat, e1);   e2 /= _np.linalg.norm(e2)

    par = directive.get("parallel_shift_angstrom", None)
    par_cart = directive.get("parallel_shift_cart", None)
    par_frac = directive.get("parallel_shift_fractional", None)

    v_parallel = _np.zeros(3, float)
    if par is not None:
        p1, p2 = float(par[0]), float(par[1])
        v_parallel = p1 * e1 + p2 * e2
    elif par_cart is not None:
        v = _np.asarray(par_cart, float)
        v_parallel = v - _np.dot(v, n_hat) * n_hat
    elif par_frac is not None:
        f = _np.asarray(par_frac, float)
        v = A_cols @ f
        v_parallel = v - _np.dot(v, n_hat) * n_hat

    # ---------- Sublattice swap config ----------
    swap_cfg = directive.get("swap_sublattice", False)
    swap_map: Dict[str, str] = {}
    if swap_cfg:
        if isinstance(swap_cfg, dict):
            swap_map = {str(k): str(v) for k, v in swap_cfg.items()}
        else:
            if charges is None:
                raise ValueError("swap_sublattice requested but 'charges' not provided to apply_twins().")
            pos = [e for e, q in charges.items() if q > 0]
            neg = [e for e, q in charges.items() if q < 0]
            if len(pos) == 1 and len(neg) == 1:
                swap_map = {pos[0]: neg[0], neg[0]: pos[0]}
            else:
                raise ValueError("auto swap needs exactly one cation and one anion in 'charges'.")

    # ---------- Signed coordinate along +n̂ ----------
    t = (R - origin) @ n_hat
    tmin, tmax = float(t.min()), float(t.max())
    print(f"[twin] hkl={tuple(int(x) for x in hkl)}  t-range Å: [{tmin:.3f},{tmax:.3f}]  d_(hkl)={d_hkl:.4f}  "
          f"op={operation}  shift_normal={s_normal:.4f} Å  |shift_parallel|={_np.linalg.norm(v_parallel):.4f} Å")

    R_out = R.copy()
    species_out = list(species) if species is not None else None

    for (t_a, t_b) in segs_A:
        if t_a > t_b:
            t_a, t_b = t_b, t_a
        mask = (t >= (t_a - pad)) & (t <= (t_b + pad))
        n_sel = int(mask.sum())
        if n_sel == 0:
            print(f"[twin]   segment [{t_a:.3f},{t_b:.3f}] Å  -> selected 0 (skip)")
            continue

        t_ref = t_a if mirror_at == "entry" else 0.5 * (t_a + t_b)
        c = t_ref + _np.dot(n_hat, origin)
        if snap:
            c = round((t_ref) / d_hkl) * d_hkl + _np.dot(n_hat, origin)

        R_slab = reflect_about_plane(R_out[mask], n_hat, c)
        delta = s_normal * n_hat + v_parallel
        if _np.any(delta):
            R_slab = R_slab + delta[None, :]
        R_out[mask] = R_slab

        if species_out is not None and swap_map:
            for idx, keep in enumerate(mask):
                if keep:
                    s = species_out[idx]
                    species_out[idx] = swap_map.get(s, s)

        # --------------- NEW: stitch the region beyond the domain ---------------
        # <-- MODIFICATION: This entire block is now conditional
        if perform_stitch:
            stitch_mode = str(directive.get("stitch_beyond", "none")).lower()
            if stitch_mode not in ("none", "auto", "positive", "negative", "true", "false"):
                raise ValueError("stitch_beyond must be one of: none|auto|positive|negative")

            if stitch_mode in ("true", "false"):
                stitch_mode = "auto" if stitch_mode == "true" else "none"

            if stitch_mode != "none":
                if stitch_mode == "auto":
                    side = "positive" if (0.5 * (t_a + t_b)) >= 0.0 else "negative"
                else:
                    side = stitch_mode

                if side == "positive":
                    mask_beyond = (t > (t_b + pad))
                else:
                    mask_beyond = (t < (t_a - pad))
                
                mask_beyond = mask_beyond & (~mask)

                undo_parallel = -v_parallel
                if bool(directive.get("stitch_include_normal", False)):
                    undo_parallel = undo_parallel + (-s_normal) * n_hat

                if np.any(undo_parallel):
                    R_out[mask_beyond] = R_out[mask_beyond] + undo_parallel[None, :]
        # -----------------------------------------------------------------------

        if tol_merge > 0:
            R_out = merge_close_points(R_out, tol_merge)

        print(f"[twin]   segment [{t_a:.3f},{t_b:.3f}]  selected={n_sel}  plane c={c:.4f}")

    if species is not None and species_out is not None:
        species[:] = species_out

    return R_out


def apply_twins(
    R: Array,
    A: Array,
    twins_config: Union[List[Dict[str, Any]], Dict[str, Any]],
    default_origin: Union[str, Iterable[float]] = "center",
    species: Optional[List[str]] = None,
    charges: Optional[Dict[str, int]] = None,
    **kwargs,  # <-- MODIFICATION: Accept keyword arguments
) -> Array:
    """
    Apply multiple twin directives (list or single dict).
    Supports species-aware options and auto sublattice swap via `charges`.
    """
    if isinstance(twins_config, dict):
        directives = [twins_config]
    else:
        directives = list(twins_config)

    R_out = R.copy()
    for d in directives:
        R_out = apply_twin_directive(
            R_out, A, d,
            default_origin=default_origin,
            species=species,
            charges=charges,
            **kwargs,  # <-- MODIFICATION: Pass arguments to the directive handler
        )
    return R_out


def apply_twin_directive_old(
    R: Array,
    A: Array,
    directive: Dict[str, Any],
    default_origin: Union[str, Iterable[float]] = "center",
    species: Optional[List[str]] = None,
    charges: Optional[Dict[str, int]] = None,
) -> Array:
    """
    Twin directive with mirror/glide and optional sublattice swap.

    New keys for *parallel* (in-plane) shift:
      - parallel_shift_cart: [sx, sy, sz] in Å (will be projected into the plane)
      - parallel_shift_fractional: [u, v, w] in lattice fractions (Å = A_cols @ [u,v,w], then projected)
      - parallel_shift_angstrom: [p1, p2] components along an orthonormal in-plane basis {e1,e2}
        (we construct e1,e2 ⟂ n̂ deterministically)

    Normal (along n̂) shift (backward compatible):
      - operation: "mirror" | "mirror+shift"
      - shift_angstrom: s_n (Å along +n̂)
      - shift_layers:   λ_n  (Δ = λ_n * d_(hkl) along +n̂)
      - shift_unitcell_fraction: f (Δ = 3f * d_(111) for cubic (111); general users should prefer the keys above)

    Other existing keys unchanged:
      hkl, origin, (intervals|segments)_(angstrom|layers), snap_to_layers,
      mirror_at, merge_tolerance, interval_pad, swap_sublattice.
    """
    import numpy as _np

    # ---------- Plane / lattice ----------
    if "hkl" not in directive:
        raise ValueError("Twin directive requires 'hkl'.")
    A_cols = cell_columns(A)
    hkl = parse_hkl(directive["hkl"])
    n_hat = plane_normal_from_hkl(A_cols, hkl)
    d_hkl = interplanar_spacing(A_cols, hkl)

    origin = _resolve_origin(R, directive.get("origin", default_origin))

    # ---------- Intervals (Å) ----------
    segments = (
        directive.get("intervals_angstrom")
        or directive.get("segments_angstrom")
        or directive.get("intervals")
        or directive.get("segments")
    )
    segments_layers = directive.get("segments_layers") or directive.get("intervals_layers")
    if segments is None and segments_layers is None:
        raise ValueError("Provide 'intervals_angstrom' (or aliases) or 'intervals_layers' in twins.")

    segs_A: List[Tuple[float, float]] = []
    if segments is not None:
        for (t1, t2) in segments:
            segs_A.append((float(t1), float(t2)))
    if segments_layers is not None:
        for (n1, n2) in segments_layers:
            segs_A.append((float(n1) * d_hkl, float(n2) * d_hkl))

    # ---------- Behavior knobs ----------
    snap = bool(directive.get("snap_to_layers", False))
    mirror_at = str(directive.get("mirror_at", "midplane")).lower()
    tol_merge = float(directive.get("merge_tolerance", 0.0))
    pad = float(directive.get("interval_pad", 1e-3))

    # ---------- Normal (along n̂) shift ----------
    operation = str(directive.get("operation", "mirror")).lower()
    if operation not in ("mirror", "mirror+shift"):
        raise ValueError(f"Unknown twin operation '{operation}'.")
    shift_ang = directive.get("shift_angstrom", None)
    shift_layers = directive.get("shift_layers", None)
    shift_uc = directive.get("shift_unitcell_fraction", None)  # convenience for cubic (111)

    if operation == "mirror+shift":
        if shift_ang is not None:
            s_normal = float(shift_ang)
        elif shift_layers is not None:
            s_normal = float(shift_layers) * d_hkl
        elif shift_uc is not None:
            # For cubic (111): |[111]| = a√3 and d_(111)=a/√3 → Δ = 3f d_(111)
            s_normal = 3.0 * float(shift_uc) * d_hkl
        else:
            s_normal = 0.5 * d_hkl
    else:
        s_normal = 0.0

    # ---------- NEW: Parallel (in-plane) shift ----------
    # Build an orthonormal in-plane basis {e1, e2}
    # Pick a stable reference not parallel to n̂:
    ref = _np.array([1.0, 0.0, 0.0])
    if abs(_np.dot(ref, n_hat)) > 0.9:
        ref = _np.array([0.0, 1.0, 0.0])
    e1 = _np.cross(n_hat, ref);  e1 /= _np.linalg.norm(e1)
    e2 = _np.cross(n_hat, e1);   e2 /= _np.linalg.norm(e2)

    # Accept one of the parallel specs (priority order)
    par = directive.get("parallel_shift_angstrom", None)          # [p1, p2] along e1,e2
    par_cart = directive.get("parallel_shift_cart", None)          # [sx, sy, sz] Å
    par_frac = directive.get("parallel_shift_fractional", None)    # [u, v, w] lattice fractions

    v_parallel = _np.zeros(3, float)
    if par is not None:
        p1, p2 = float(par[0]), float(par[1])
        v_parallel = p1 * e1 + p2 * e2
    elif par_cart is not None:
        v = _np.asarray(par_cart, float)
        v_parallel = v - _np.dot(v, n_hat) * n_hat   # project into plane
    elif par_frac is not None:
        f = _np.asarray(par_frac, float)
        v = A_cols @ f
        v_parallel = v - _np.dot(v, n_hat) * n_hat   # project into plane
    # else: no parallel shift

    # ---------- Sublattice swap config ----------
    swap_cfg = directive.get("swap_sublattice", False)
    swap_map: Dict[str, str] = {}
    if swap_cfg:
        if isinstance(swap_cfg, dict):
            swap_map = {str(k): str(v) for k, v in swap_cfg.items()}
        else:
            if charges is None:
                raise ValueError("swap_sublattice requested but 'charges' not provided to apply_twins().")
            pos = [e for e, q in charges.items() if q > 0]
            neg = [e for e, q in charges.items() if q < 0]
            if len(pos) == 1 and len(neg) == 1:
                swap_map = {pos[0]: neg[0], neg[0]: pos[0]}
            else:
                raise ValueError("auto swap needs exactly one cation and one anion in 'charges'.")

    # ---------- Signed coordinate along +n̂ ----------
    t = (R - origin) @ n_hat
    tmin, tmax = float(t.min()), float(t.max())
    print(f"[twin] hkl={tuple(int(x) for x in hkl)}  t-range Å: [{tmin:.3f},{tmax:.3f}]  d_(hkl)={d_hkl:.4f}  "
          f"op={operation}  shift_normal={s_normal:.4f} Å  |shift_parallel|={_np.linalg.norm(v_parallel):.4f} Å")

    R_out = R.copy()
    species_out = list(species) if species is not None else None

    for (t_a, t_b) in segs_A:
        if t_a > t_b:
            t_a, t_b = t_b, t_a
        mask = (t >= (t_a - pad)) & (t <= (t_b + pad))
        n_sel = int(mask.sum())
        if n_sel == 0:
            print(f"[twin]   segment [{t_a:.3f},{t_b:.3f}] Å  -> selected 0 (skip)")
            continue

        # reflection plane (midplane preserves extent)
        t_ref = t_a if mirror_at == "entry" else 0.5 * (t_a + t_b)
        c = t_ref + _np.dot(n_hat, origin)
        if snap:
            c = round((t_ref) / d_hkl) * d_hkl + _np.dot(n_hat, origin)

        # reflect geometry
        R_slab = reflect_about_plane(R_out[mask], n_hat, c)

        # compose total translation: normal + parallel
        delta = s_normal * n_hat + v_parallel
        if _np.any(delta):
            R_slab = R_slab + delta[None, :]

        # commit geometry
        R_out[mask] = R_slab

        # swap species inside the mirrored slab (labels only)
        if species_out is not None and swap_map:
            for idx, keep in enumerate(mask):
                if keep:
                    s = species_out[idx]
                    species_out[idx] = swap_map.get(s, s)

        # --------------- NEW: stitch the region beyond the domain ---------------
        stitch_mode = str(directive.get("stitch_beyond", "none")).lower()
        if stitch_mode not in ("none", "auto", "positive", "negative", "true", "false"):
            raise ValueError("stitch_beyond must be one of: none|auto|positive|negative")

        if stitch_mode in ("true", "false"):
            stitch_mode = "auto" if stitch_mode == "true" else "none"

        if stitch_mode != "none":
            # decide which side to stitch
            if stitch_mode == "auto":
                side = "positive" if (0.5 * (t_a + t_b)) >= 0.0 else "negative"
            else:
                side = stitch_mode

            if side == "positive":
                mask_beyond = (t > (t_b + pad))    # strictly beyond the exit
            else:
                mask_beyond = (t < (t_a - pad))    # strictly before the entry

            # don't touch the slab we just mirrored
            mask_beyond = mask_beyond & (~mask)

            # undo only the parallel glide by default
            undo_parallel = (-v_parallel)
            # optionally undo the normal component too (usually false)
            if bool(directive.get("stitch_include_normal", False)):
                undo_parallel = undo_parallel + (-s_normal) * n_hat

            if np.any(undo_parallel):
                R_out[mask_beyond] = R_out[mask_beyond] + undo_parallel[None, :]

        # -----------------------------------------------------------------------

        if tol_merge > 0:
            R_out = merge_close_points(R_out, tol_merge)

        print(f"[twin]   segment [{t_a:.3f},{t_b:.3f}]  selected={n_sel}  plane c={c:.4f}")

    # push species back
    if species is not None and species_out is not None:
        species[:] = species_out

    return R_out


def apply_twins_old(
    R: Array,
    A: Array,
    twins_config: Union[List[Dict[str, Any]], Dict[str, Any]],
    default_origin: Union[str, Iterable[float]] = "center",
    species: Optional[List[str]] = None,
    charges: Optional[Dict[str, int]] = None,   # <-- NEW
) -> Array:
    """
    Apply multiple twin directives (list or single dict).
    Supports species-aware options and auto sublattice swap via `charges`.
    """
    if isinstance(twins_config, dict):
        directives = [twins_config]
    else:
        directives = list(twins_config)

    R_out = R.copy()
    for d in directives:
        R_out = apply_twin_directive(
            R_out, A, d,
            default_origin=default_origin,
            species=species,
            charges=charges,          # <-- pass through
        )
    return R_out


# -----------------------------
# ---- Optional ASE shim  -----
# -----------------------------
def apply_twins_to_ase(
    atoms,  # ASE Atoms
    twins_config: Union[List[Dict[str, Any]], Dict[str, Any]],
    default_origin: Union[str, Iterable[float]] = "center",
):
    """
    Convenience wrapper if your pipeline uses ASE.
    Modifies atoms.positions in-place and returns the atoms object.
    """
    A_cols = cell_columns(np.array(atoms.cell))  # ASE stores rows; cell_columns handles it
    R = atoms.get_positions()
    R2 = apply_twins(R, A_cols, twins_config, default_origin=default_origin)
    atoms.set_positions(R2)
    return atoms

