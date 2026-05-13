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

def apply_twin_directive(
    R: Array,
    A: Array,
    directive: Dict[str, Any],
    default_origin: Union[str, Iterable[float]] = "center",
    species: Optional[List[str]] = None,
    charges: Optional[Dict[str, int]] = None,
    perform_stitch: bool = True,
) -> Array:
    """
    Apply one twin directive to positions.

    The selected slab is reflected across an HKL plane and can optionally be
    shifted along the plane normal and/or by an in-plane glide. If species are
    supplied, a sublattice swap can be applied to atoms inside the transformed
    slab.
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

    # ---------- Parallel (in-plane) shift ----------
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
    **kwargs,
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
            **kwargs,
        )
    return R_out
