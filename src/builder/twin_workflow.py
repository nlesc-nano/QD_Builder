from __future__ import annotations

from collections import Counter
from typing import List

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Structure

from .geometry import recut_with_planes
from .nc_types import Config, Plane
from .twinbound import (
    apply_twins,
    cell_columns,
    interplanar_spacing,
    merge_close_points_species_aware,
    parse_hkl,
    plane_normal_from_hkl,
    refill_against_template,
)


def representative_plane_index(pts: NDArray[np.float64], planes: List[Plane]) -> NDArray[np.int_]:
    A = np.stack([n for (n, _d) in planes], axis=0)
    b = np.array([d for (_n, d) in planes], float)
    norms = np.linalg.norm(A, axis=1)
    norms[norms == 0] = 1.0
    slack = b[None, :] - pts @ A.T
    d_perp = slack / norms[None, :]
    near = d_perp <= (d_perp.min(axis=1)[:, None] + 0.20)
    return np.argmax(near, axis=1)


def infer_facet_terminations(
    syms: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    layer_tol: float = 0.60,
) -> dict[int, str]:
    A = np.stack([n for (n, _d) in planes], axis=0)
    b = np.array([d for (_n, d) in planes], float)
    norms = np.linalg.norm(A, axis=1)
    norms[norms == 0] = 1.0
    term = {}
    for j in range(len(planes)):
        dperp = (b[j] - pts @ A[j]) / norms[j]
        m = (dperp >= 0.0) & (dperp <= float(layer_tol))
        if not np.any(m):
            continue
        c = Counter([s for s, keep in zip(syms, m) if keep])
        if c:
            term[j] = max(c, key=c.get)
    return term


def apply_single_material_twins(
    syms: List[str],
    pts: NDArray[np.float64],
    *,
    cfg: Config,
    struct: Structure,
    planes_geo: List[Plane],
) -> tuple[List[str], NDArray[np.float64]]:
    print("\n[4] Building twinned nanocrystal...")
    tw = cfg.twins[0] if isinstance(cfg.twins, list) else cfg.twins

    A_cols = cell_columns(struct.lattice.matrix)
    hkl_t = tuple(int(x) for x in parse_hkl(tw["hkl"]))
    n_hat = plane_normal_from_hkl(A_cols, hkl_t)
    d_hkl = interplanar_spacing(A_cols, hkl_t)

    print("    [4a] Generating twinned template for refilling...")
    syms_tpl_twinned = list(syms)
    pts_tpl_twinned = apply_twins(
        pts.copy(),
        A_cols,
        tw,
        default_origin="center",
        species=syms_tpl_twinned,
        charges=cfg.charges,
        perform_stitch=False,
    )

    print("    [4b] Applying twin glide to working structure...")
    pts = apply_twins(
        pts,
        A_cols,
        tw,
        default_origin="center",
        species=syms,
        charges=cfg.charges,
        perform_stitch=False,
    )
    origin = pts.mean(axis=0)

    if bool(tw.get("refill_missing", True)):
        print("    [4c] Refilling voids using twinned template (boundary-aware)...")
        segsA = [tuple(x) for x in (tw.get("intervals_angstrom") or [])]
        if tw.get("intervals_layers"):
            segsA += [(float(n1) * d_hkl, float(n2) * d_hkl) for (n1, n2) in tw["intervals_layers"]]

        term_map = infer_facet_terminations(syms, pts, planes_geo, layer_tol=0.60)
        facesN = np.stack([n for (n, _d) in planes_geo], axis=0)
        cosang = np.abs(facesN @ n_hat)
        include_top = np.any(cosang > 0.92)
        facet_mode = "all" if include_top else "sides"

        syms_new, pts_new = refill_against_template(
            cur_syms=syms, cur_pts=pts,
            tpl_syms=syms_tpl_twinned, tpl_pts=pts_tpl_twinned,
            planes=planes_geo, n_hat=n_hat, origin=origin,
            intervals_A=segsA,
            pad_A=1e-3,
            site_match_tol=0.90,
            min_sep_tol=float(tw.get("refill_min_separation", 1.2)),
            scope="surface",
            shell_thickness=2.0,
            facet_mode=facet_mode,
            top_cos_thresh=0.92,
            refill_region="inside",
            orient_delta=0.20,
            snap_out_eps=0.00,
            snap_offset=-0.08,
            layer_gap_tol=0.90,
        )

        if len(pts_new) > len(pts):
            added_idx = np.arange(len(pts), len(pts_new))
            which_plane = representative_plane_index(pts_new[added_idx], planes_geo)
            for kk, j in zip(added_idx, which_plane):
                want = term_map.get(int(j), None)
                if want is not None and syms_new[kk] != want:
                    syms_new[kk] = want
            syms, pts = syms_new, pts_new

        syms, pts = recut_with_planes(syms, pts, planes_geo, tol=1e-6)

    stitch_mode = str(tw.get("stitch_beyond", "auto")).lower()
    if stitch_mode not in ("none", "false"):
        print("    [4d] Stitching top layer to align with twinned slab...")
        s_normal = 0.0
        if tw.get("operation") == "mirror+shift" and "shift_layers" in tw:
            s_normal = float(tw["shift_layers"]) * d_hkl

        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, n_hat)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(n_hat, ref)
        e1 /= np.linalg.norm(e1)
        v_parallel = np.zeros(3)
        if "parallel_shift_fractional" in tw:
            f = np.asarray(tw["parallel_shift_fractional"], float)
            v = A_cols @ f
            v_parallel = v - np.dot(v, n_hat) * n_hat

        undo_vec = -v_parallel
        if bool(tw.get("stitch_include_normal", False)):
            undo_vec -= s_normal * n_hat

        t = (pts - origin) @ n_hat
        t_a, t_b = (tw.get("intervals_angstrom") or [[0, 0]])[0]
        if t_a > t_b:
            t_a, t_b = t_b, t_a

        margin = 0.25 * d_hkl
        mask_top = t > (t_b + margin)
        if np.any(undo_vec):
            pts[mask_top] += undo_vec[None, :]

        syms, pts = recut_with_planes(syms, pts, planes_geo, tol=1e-6)

    print("    [4e] Cleaning up interface with species-aware deduplication...")
    dedup_tol = float(tw.get("refill_dedup_tolerance", 3.0))
    syms_deduped, pts_deduped = merge_close_points_species_aware(syms, pts, tol=dedup_tol)
    if len(pts_deduped) < len(pts):
        print(f"         - Merged {len(pts) - len(pts_deduped)} overlapping site(s).")
    return syms_deduped, pts_deduped
