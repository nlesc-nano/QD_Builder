# src/builder/main.py
from __future__ import annotations
import os
import sys
import random
from typing import List
import numpy as np 

try:
    from pymatgen.core import Structure
except ImportError:
    sys.exit("pip install pymatgen[matproj]")

from .config import build_parser, parse_yaml_config
from .nc_types import Config, Facet
from .facets import expand_facets, detect_facets_from_nc, halfspaces, scan_facets_from_cif
from .geometry import build_nanocrystal, dedupe_points, build_core_shell_by_labeling
from .io_utils import write_xyz, write_manifest, center_coords
from .passivation import collect_anion_candidates, charge_balance
from .analysis import facet_families_overview, facet_atom_report
from .cleanup import prune_low_coord_sites
from .twinbound import apply_twins, refill_against_template, merge_close_points_species_aware


def count_near_plane(pts, n_hat, c, w=0.02):
    t = pts @ n_hat
    return int(((t >= c - w) & (t <= c + w)).sum())

def recut_with_planes(syms, pts, planes, tol=1e-6):
    """
    planes: iterable of (n, d) with n·x ≤ d defining the polyhedron.
    Keeps only atoms that still satisfy all halfspaces after a transform.
    """
    if not planes:
        return syms, pts
    A = np.stack([n for (n, d) in planes], axis=0)   # [P,3]
    b = np.array([d for (n, d) in planes], float)    # [P]
    mask = (pts @ A.T <= (b[None, :] + tol)).all(axis=1)
    syms2 = [s for s, keep in zip(syms, mask) if keep]
    pts2  = pts[mask]
    return syms2, pts2

def _swap_all_dicoord_sites(syms, pts, charges, anion_target, cation_target, verbose=False):
    """
    Unconditional pre-pass:
      - CN == 2 anions -> anion_target (e.g., Cl, −1)
      - CN == 2 cations -> cation_target (e.g., Rb, +1)
    CN == 1 are already removed after Wulff construction.
    CN > 2 are left as-is.
    """
    from .analysis import coord_numbers
    cn = coord_numbers(syms, pts)
    out = list(syms)
    n_an = n_cat = 0
    for i, (s, cni) in enumerate(zip(syms, cn)):
        if int(round(cni)) != 2:
            continue
        q = charges.get(s, 0)
        if q < 0:
            out[i] = anion_target
            n_an += 1
        elif q > 0:
            out[i] = cation_target
            n_cat += 1
    if verbose:
        print(f"    [pre] CN=2 swaps → anion:{n_an}, cation:{n_cat}")
    return out

def main(argv: List[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    random.seed(args.seed)

    cfg: Config = parse_yaml_config(args.yaml)

    # ----- Optional facet scan (universal; runs before build) -----
    if args.scan_facets:
        if cfg.mode == "stack":
            for m in cfg.materials:
                rows = scan_facets_from_cif(m.cif if cfg.mode=="stack" else args.cif, cfg.charges, max_index=args.scan_max_index, min_slab_size=args.scan_slab_size, min_vacuum_size=args.scan_vacuum_size, n_shifts=args.scan_shifts)

                print(f"\n[facet-scan] {m.name} ({m.cif}) — |h|,|k|,|l| ≤ {args.scan_max_index}")
                for r in rows:
                    pol = "polar" if r["polar_any"] else "non-polar"
                    pc = r["polar_count"]; nt = r["n_terms_checked"]
                    print(f"  hkl={r['hkl']!s:>10}  fam={r['family']:<6}  {pol:9}  ({pc}/{nt} terminations polar)")
        else:
            rows = scan_facets_from_cif(m.cif if cfg.mode=="stack" else args.cif, cfg.charges, max_index=args.scan_max_index, min_slab_size=args.scan_slab_size, min_vacuum_size=args.scan_vacuum_size, n_shifts=args.scan_shifts)

            print(f"\n[facet-scan] single-mode ({args.cif}) — |h|,|k|,|l| ≤ {args.scan_max_index}")
            for r in rows:
                pol = "polar" if r["polar_any"] else "non-polar"
                pc = r["polar_count"]; nt = r["n_terms_checked"]
                print(f"  hkl={r['hkl']!s:>10}  fam={r['family']:<6}  {pol:9}  ({pc}/{nt} terminations polar)")
        # continue with the normal run afterwards
    
    if cfg.mode == "stack":
        # Multi-material: YAML drives CIFs & radii; CLI --cif/--radius ignored
        if args.verbose:
            print("\n[STACK] Multi-material mode detected from YAML.")
            print(f"  - Regions: {[m.name for m in cfg.materials]}")
            print(f"  - Proper rotations only: {bool(cfg.proper_only)}")
            print(f"  - Pair opposites: {bool(cfg.pair_opposites)}")
        
        # 1) Build geometry using shell-first carve + relabel (default path)
        if args.radius is None:
            raise SystemExit("Please pass -r/--radius to set the full NC size in stack mode.")
        
        if len(cfg.materials) < 2:
            raise SystemExit("Stack mode requires at least two materials (core first, then shell).")
        
        core  = cfg.materials[0]   # core.cif + core.aspect (inner shape)
        shell = cfg.materials[1]   # shell.cif + shell.aspect (outer shape)
        
        syms, pts = build_core_shell_by_labeling(
            core=core,
            shell=shell,
            shell_R=float(args.radius),
            seeds=shell.seeds,
            charges=cfg.charges,
            surf_tol=cfg.passivation.surf_tol,
            verbose=args.verbose,
            shrink_to_core_lattice=args.core_lattice_fit,
            strain_width=args.core_strain_width,
            center_mode=args.core_center,
            print_bond_stats=args.bond_stats,   # <—
        )

        # --- OPTIONAL TWIN BOUNDARIES (stack mode) ---
        if getattr(cfg, "twins", None):
            if args.verbose:
                print("\n[3a] Applying twin boundary transformations (stack mode)...")
        
            shell_struct = Structure.from_file(shell.cif)
        
            # 1) Apply the mirrors on the composite coords
            pts = apply_twins(
                pts,
                shell_struct.lattice.matrix,
                cfg.twins,
                default_origin="center",
                species=syms,
                charges=cfg.charges,
            )
        
            # 2) Recut with OUTER (shell) Wulff planes
            s_shell = shell_struct
            facets_shell = expand_facets(s_shell, shell.seeds, proper_only=cfg.proper_only)
            planes_outer = halfspaces(s_shell, facets_shell, R=float(args.radius), aspect=shell.aspect)
            syms, pts = recut_with_planes(syms, pts, planes_outer, tol=1e-3)
        
            if args.verbose:
                print(f"    - After twins+recut: {len(syms)} atoms")
         
 
        if args.verbose:
            print(f"\n[4] Composite particle atoms: {len(syms)}")
         
        # --- Write core.xyz and shell.xyz (behind --write-all) ---
        if args.write_all:
            # Reconstruct inner mask using core aspect at same R on the shell lattice
            s_shell = Structure.from_file(shell.cif)
            facets_shell = expand_facets(s_shell, shell.seeds, proper_only=cfg.proper_only)
            planes_core = halfspaces(s_shell, facets_shell, R=float(args.radius), aspect=core.aspect)

            # inside test: n·x ≤ d + tol for all planes
            A = np.stack([n for (n, d) in planes_core], axis=0)   # [P,3]
            b = np.array([d for (n, d) in planes_core], float)    # [P]
            inner_mask = (pts @ A.T <= (b[None, :] + 1e-6)).all(axis=1)

            core_syms  = [s for s, keep in zip(syms, inner_mask)  if keep]
            core_pts   =  pts[inner_mask]
            shell_syms = [s for s, keep in zip(syms, ~inner_mask) if keep]
            shell_pts  =  pts[~inner_mask]

            prefix = os.path.splitext(os.path.basename(args.out))[0]
            if args.verbose:
                print(f"    - Writing {prefix}_core.xyz ({len(core_syms)} atoms)")
                print(f"    - Writing {prefix}_shell.xyz ({len(shell_syms)} atoms)")

            write_xyz(f"{prefix}_core.xyz",  core_syms,  center_coords(core_pts)  if args.center else core_pts)
            write_xyz(f"{prefix}_shell.xyz", shell_syms, center_coords(shell_pts) if args.center else shell_pts)

        # 2) Optional prune
        if args.prune_mono:
            if args.verbose:
                print("\n[4b] Pruning low-coordination atoms (pre-facet detection)...")
            syms, pts, n_removed, n_pass = prune_low_coord_sites(
                syms, pts, min_cn=args.prune_min_cn, max_passes=args.prune_passes, verbose=args.verbose
            )
            if args.verbose:
                print(f"    - Pruned {n_removed} atoms in {n_pass} pass(es); remaining {len(syms)} atoms")

        # 3) Detect facets on composite
        if args.verbose:
            print("\n[5] Detecting actual exposed facets (composite)...")
        # Need a lattice for normals; use the core lattice as reference (only for normals in detect function)
        core_cif = cfg.materials[0].cif
        struct = Structure.from_file(core_cif)
        # Expand core seeds once to give detect() a reasonable initial set (not critical)
        seeds0 = expand_facets(struct, cfg.materials[0].seeds, proper_only=cfg.proper_only)
        facets, planes = detect_facets_from_nc(syms, pts, struct.lattice, cfg.charges, seeds0, cfg.passivation.surf_tol)
        if args.verbose:
            print(f"    - Detected {len(facets)} facets")

        # 4) Reports
        if args.verbose:
            print("\n[6] Surface atom and CN reports (composite):")
        facet_families_overview(syms, pts, planes, facets, surf_tol=cfg.passivation.surf_tol, charges=cfg.charges)
        facet_atom_report(syms, pts, planes, facets, surf_tol=cfg.passivation.surf_tol, charges=cfg.charges)

        # --- Unconditional pre-treatment of dicoordinated sites ---
        syms = _swap_all_dicoord_sites(
            syms, pts,
            charges=cfg.charges,
            anion_target=cfg.passivation.ligand,
            cation_target=getattr(cfg.passivation, "cation_ligand", "Rb"),
            verbose=args.verbose,
        )

        # 5) Write snapshot before passivation if requested
        prefix = os.path.splitext(os.path.basename(args.out))[0]
        if args.write_all:
            if args.verbose:
                print(f"\n[7] Writing initial cut XYZ to {prefix}_cut.xyz")
            write_xyz(f"{prefix}_cut.xyz", syms, center_coords(pts) if args.center else pts)

        # 6) Gather outer-layer anion candidates and balance
        if args.verbose:
            print("\n[8] Gathering outer-layer anion candidates (composite)...")
        outer_cands, subl_cands = collect_anion_candidates(
            syms, pts, planes, cfg.charges, cfg.passivation.ligand, cfg.passivation.surf_tol, verbose=args.verbose
        )

        if args.verbose:
            print("\n[10] Balancing charge stepwise (outer anions first; then add/remove ligands if needed)...")
        syms, pts = charge_balance(
            syms, pts,
            outer_cands, subl_cands,
            cfg.charges, cfg.passivation.ligand,
            verbose=args.verbose,
            planes=planes, facets=facets, surf_tol=cfg.passivation.surf_tol,
            rng=random,
            prefer_remove_parity=(args.parity == "remove"),
        )

        # 7) Final write
        if args.verbose:
            print(f"\n[11] Writing final XYZ to {args.out}")
        final_pts = center_coords(pts) if args.center else pts
        write_xyz(args.out, syms, final_pts)

        if args.verbose:
            print(f"[12] Writing JSON manifest to {prefix}.json")
        write_manifest(prefix, syms, cfg.charges)

        if args.verbose:
            from collections import Counter
            cnt = Counter(syms)
            print("\n### ELEMENT COUNTS ###")
            for k in sorted(cnt):
                print(f" {k}: {cnt[k]}")
        return 0

    # ---------------- SINGLE-MATERIAL MODE (legacy) ----------------
    if args.verbose:
        print("\n[1] Reading CIF structure...")
    struct = Structure.from_file(args.cif)
    if args.verbose:
        print(f"    - Loaded {len(struct)} atoms from {args.cif}")

    if args.verbose:
        print("\n[2] Using YAML config (single material)...")
        print(f"    - Facet seeds: {[ (f.h, f.k, f.l) for f in cfg.seeds ]}")
        print(f"    - Ligand: {cfg.passivation.ligand}, surf_tol={cfg.passivation.surf_tol:.3f} Å")
        print(f"    - Charges: {cfg.charges}")
        print(f"    - Pair opposites: {bool(cfg.pair_opposites)}")
        po_cli = getattr(args, "proper_rotations_only", None)
        eff_proper = cfg.proper_only if po_cli is None else bool(po_cli)
        print(f"    - Proper rotations only (effective): {bool(eff_proper)}")

    # Resolve aspect and proper-only (CLI can override)
    aspect = args.aspect if args.aspect is not None else cfg.aspect
    proper_only = cfg.proper_only if getattr(args, "proper_rotations_only", None) is None else bool(args.proper_rotations_only)

    if args.verbose:
        print("\n[3] Expanding symmetry & building Wulff facets...")
    wulff_facets: List[Facet] = expand_facets(struct, cfg.seeds, proper_only=proper_only)
    if args.verbose:
        print(f"    - Expanded to {len(wulff_facets)} oriented facets")

    if args.verbose:
        print("\n[4] Building nanocrystal from Wulff facets...")
    syms, pts, _planes_geo = build_nanocrystal(struct, wulff_facets, args.radius, aspect=aspect)
    syms, pts = dedupe_points(syms, pts, tol=1e-3)
    if args.verbose:
        print(f"    - Cut particle: {len(syms)} atoms")
        ax, ay, az = aspect
        print(f"    - Aspect multipliers (a,b,c): {ax:.3f}, {ay:.3f}, {az:.3f}")

    syms_tpl = list(syms)         # template (pre-twin) symbols
    pts_tpl  = pts.copy()         # template (pre-twin) coords 
    # Prepare a safe handle to (possibly absent) twins config,
    # so downstream code can read optional parameters (e.g., dedup tolerance)
    tw0 = {}
    if getattr(cfg, "twins", None):
        if isinstance(cfg.twins, list) and len(cfg.twins) > 0 and isinstance(cfg.twins[0], dict):
            tw0 = cfg.twins[0]
        elif isinstance(cfg.twins, dict):
            tw0 = cfg.twins
        # else leave as {}

    # --- OPTIONAL TWIN BOUNDARIES (single-material) ---
    if getattr(cfg, "twins", None):
        import numpy as _np
    
        print("\n[4] Building twinned nanocrystal...")
        tw = cfg.twins[0] if isinstance(cfg.twins, list) else cfg.twins
    
        # Lattice in column form
        A_cols = cell_columns(struct.lattice.matrix)
        hkl_t  = tuple(int(x) for x in parse_hkl(tw["hkl"]))
        n_hat  = plane_normal_from_hkl(A_cols, hkl_t)
        d_hkl  = interplanar_spacing(A_cols, hkl_t)
    
        # ---------- helpers ----------
        def _rep_plane_index(pts, planes):
            A = _np.stack([n for (n, d) in planes], axis=0)
            b = _np.array([d for (n, d) in planes], float)
            norms = _np.linalg.norm(A, axis=1); norms[norms == 0] = 1.0
            slack = b[None, :] - pts @ A.T
            d_perp = slack / norms[None, :]
            near = d_perp <= (d_perp.min(axis=1)[:, None] + 0.20)
            return _np.argmax(near, axis=1)  # representative plane per point
    
        def _infer_terminations(syms, pts, planes, layer_tol=0.60):
            """Majority species within 'layer_tol' Å of each plane becomes that plane's termination."""
            A = _np.stack([n for (n, d) in planes], axis=0)
            b = _np.array([d for (n, d) in planes], float)
            norms = _np.linalg.norm(A, axis=1); norms[norms == 0] = 1.0
            term = {}
            for j in range(len(planes)):
                dperp = (b[j] - pts @ A[j]) / norms[j]
                m = (dperp >= 0.0) & (dperp <= float(layer_tol))
                if not _np.any(m):
                    continue
                from collections import Counter
                c = Counter([s for s, keep in zip(syms, m) if keep])
                if c:
                    term[j] = max(c, key=c.get)
            return term  # {plane_index: "Cd" or "Se" ...}
    
        # ---------- Step 1: twinned template (what the slab *should* look like) ----------
        print("    [4a] Generating twinned template for refilling...")
        syms_tpl_twinned = list(syms)       # start from *pre-twin* NC as template base
        pts_tpl_twinned  = pts.copy()
        pts_tpl_twinned = apply_twins(
            pts_tpl_twinned, A_cols, tw,
            default_origin="center", species=syms_tpl_twinned, charges=cfg.charges,
            perform_stitch=False
        )
    
        # ---------- Step 2: apply twin to the working NC ----------
        print("    [4b] Applying twin glide to working structure...")
        pts = apply_twins(
            pts, A_cols, tw,
            default_origin="center", species=syms, charges=cfg.charges,
            perform_stitch=False
        )
        origin = pts.mean(axis=0)
    
        # ---------- Step 3: boundary-aware refill up to original Wulff ----------
        if bool(tw.get("refill_missing", True)):
            print("    [4c] Refilling voids using twinned template (boundary-aware)...")
            # intervals in Å along +n̂ (inside the slab)
            segsA = [tuple(x) for x in (tw.get("intervals_angstrom") or [])]
            if tw.get("intervals_layers"):
                segsA += [(float(n1)*d_hkl, float(n2)*d_hkl) for (n1, n2) in tw["intervals_layers"]]
    
            # infer per-facet termination *before* adding new atoms
            term_map = _infer_terminations(syms, pts, _planes_geo, layer_tol=0.60)

            # --- inside Step 3, before calling refill_against_template ---
            facesN = _np.stack([n for (n, d) in _planes_geo], axis=0)
            cosang  = _np.abs(facesN @ n_hat)
            # if any exposed plane is "top-like" and the slab reaches it, include it in refill
            include_top = _np.any(cosang > 0.92)  # stricter than 0.85
            facet_mode  = "all" if include_top else "sides"
                
            # do NOT snap outside; nudge *inside* and we will recut after, just in case
            syms_new, pts_new = refill_against_template(
                cur_syms=syms, cur_pts=pts,
                tpl_syms=syms_tpl_twinned, tpl_pts=pts_tpl_twinned,
                planes=_planes_geo, n_hat=n_hat, origin=origin,
                intervals_A=segsA,
                pad_A=1e-3,
                site_match_tol=0.90,
                min_sep_tol=float(tw.get("refill_min_separation", 1.2)),
                scope="surface",
                shell_thickness=2.0,
                facet_mode=facet_mode,          # <-- now "all" when needed
                top_cos_thresh=0.92,
                refill_region="inside",
                orient_delta=0.20,
                snap_out_eps=0.00,
                snap_offset=-0.08,              # <-- slightly stronger inward snap
                layer_gap_tol=0.90,
            )
             
            # If we added anything, coerce facet termination species
            if len(pts_new) > len(pts):
                added_idx = _np.arange(len(pts), len(pts_new))
                which_plane = _rep_plane_index(pts_new[added_idx], _planes_geo)
                for kk, j in zip(added_idx, which_plane):
                    want = term_map.get(int(j), None)
                    if want is not None and syms_new[kk] != want:
                        syms_new[kk] = want
                syms, pts = syms_new, pts_new
    
            # hard guard: cut anything that still strays beyond Wulff due to float noise
            syms, pts = recut_with_planes(syms, pts, _planes_geo, tol=1e-6)
    
        # ---------- Step 4: optional stitch (undo in-plane glide beyond the slab) ----------
        stitch_mode = str(tw.get("stitch_beyond", "auto")).lower()
        if stitch_mode not in ("none", "false"):
            print("    [4d] Stitching top layer to align with twinned slab...")
            s_normal = 0.0
            if tw.get("operation") == "mirror+shift" and "shift_layers" in tw:
                s_normal = float(tw["shift_layers"]) * d_hkl
    
            # in-plane component of the user shift (parallel to the slab)
            ref = _np.array([1.0, 0.0, 0.0])
            if abs(_np.dot(ref, n_hat)) > 0.9:
                ref = _np.array([0.0, 1.0, 0.0])
            e1 = _np.cross(n_hat, ref); e1 /= _np.linalg.norm(e1)
            e2 = _np.cross(n_hat, e1);  e2 /= _np.linalg.norm(e2)
            v_parallel = _np.zeros(3)
            if "parallel_shift_fractional" in tw:
                f = _np.asarray(tw["parallel_shift_fractional"], float)
                v = A_cols @ f
                v_parallel = v - _np.dot(v, n_hat) * n_hat
    
            undo_vec = -v_parallel
            if bool(tw.get("stitch_include_normal", False)):
                undo_vec -= s_normal * n_hat
    
            # apply only “above” the slab
            t = (pts - origin) @ n_hat
            t_a, t_b = (tw.get("intervals_angstrom") or [[0, 0]])[0]
            if t_a > t_b: t_a, t_b = t_b, t_a

            margin = 0.25 * d_hkl                # <- do not stitch atoms in the first TB layer
            mask_top = (t > (t_b + margin))
            if _np.any(undo_vec):
                pts[mask_top] += undo_vec[None, :]
    
            # recut again just in case the stitch moved something out
            syms, pts = recut_with_planes(syms, pts, _planes_geo, tol=1e-6)

    # ---------- Step 5: final species-aware dedup ----------
    print("    [4e] Cleaning up interface with species-aware deduplication...")
    dedup_tol = float(tw0.get("refill_dedup_tolerance", 3.0))
    syms_deduped, pts_deduped = merge_close_points_species_aware(syms, pts, tol=dedup_tol)
    if len(pts_deduped) < len(pts):
        print(f"         - Merged {len(pts) - len(pts_deduped)} overlapping site(s).")
    syms, pts = syms_deduped, pts_deduped


     
    if args.prune_mono:
        if args.verbose:
            print("\n[4b] Pruning low-coordination atoms (pre-facet detection)...")
        syms, pts, n_removed, n_pass = prune_low_coord_sites(
            syms, pts, min_cn=args.prune_min_cn, max_passes=args.prune_passes, verbose=args.verbose
        )
        if args.verbose:
            print(f"    - Pruned {n_removed} atoms in {n_pass} pass(es); remaining {len(syms)} atoms")

    if args.verbose:
        print("\n[5] Detecting actual exposed facets...")
    facets, planes = detect_facets_from_nc(syms, pts, struct.lattice, cfg.charges, wulff_facets, cfg.passivation.surf_tol)
    if args.verbose:
        print(f"    - Detected {len(facets)} facets")

    if args.verbose:
        print("\n[6] Surface atom and CN reports:")
    facet_families_overview(syms, pts, planes, facets, surf_tol=cfg.passivation.surf_tol, charges=cfg.charges)
    facet_atom_report(syms, pts, planes, facets, surf_tol=cfg.passivation.surf_tol, charges=cfg.charges)

    # --- Unconditional pre-treatment of dicoordinated sites ---
    syms = _swap_all_dicoord_sites(
        syms, pts,
        charges=cfg.charges,
        anion_target=cfg.passivation.ligand,
        cation_target=getattr(cfg.passivation, "cation_ligand", "Rb"),
        verbose=args.verbose,
    )


    prefix = os.path.splitext(os.path.basename(args.out))[0]
    if args.write_all:
        if args.verbose:
            print(f"\n[7] Writing initial cut XYZ to {prefix}_cut.xyz")
        write_xyz(f"{prefix}_cut.xyz", syms, center_coords(pts) if args.center else pts)

    if args.verbose:
        print("\n[8] Gathering outer-layer anion candidates...")
    outer_cands, subl_cands = collect_anion_candidates(
        syms, pts, planes, cfg.charges, cfg.passivation.ligand, cfg.passivation.surf_tol, verbose=args.verbose
    )

    if args.verbose:
        print("\n[10] Balancing charge stepwise (outer anions first; then add/remove ligands if needed)...")
    syms, pts = charge_balance(
        syms, pts,
        outer_cands, subl_cands,
        cfg.charges, cfg.passivation.ligand,
        verbose=args.verbose,
        planes=planes, facets=facets, surf_tol=cfg.passivation.surf_tol,
        rng=random,
        prefer_remove_parity=(args.parity == "remove"),
    )

    if args.verbose:
        print(f"\n[11] Writing final XYZ to {args.out}")
    final_pts = center_coords(pts) if args.center else pts
    write_xyz(args.out, syms, final_pts)

    if args.verbose:
        print(f"[12] Writing JSON manifest to {prefix}.json")
    write_manifest(prefix, syms, cfg.charges)

    if args.verbose:
        from collections import Counter
        cnt = Counter(syms)
        print("\n### ELEMENT COUNTS ###")
        for k in sorted(cnt):
            print(f" {k}: {cnt[k]}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

