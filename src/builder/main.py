# src/builder/main.py
from __future__ import annotations
import os
import sys
import random
from typing import List

try:
    from pymatgen.core import Structure
except ImportError:
    sys.exit("pip install pymatgen[matproj]")

from .config import build_parser, parse_yaml_config
from .nc_types import Config, Facet
from .facets import expand_facets, detect_facets_from_nc, halfspaces
from .geometry import build_nanocrystal, dedupe_points, build_core_shell_by_labeling
from .io_utils import write_xyz, write_manifest, center_coords
from .passivation import collect_anion_candidates, charge_balance
from .analysis import facet_families_overview, facet_atom_report
from .cleanup import prune_low_coord_sites

def main(argv: List[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    random.seed(args.seed)

    cfg: Config = parse_yaml_config(args.yaml)

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
         
        if args.verbose:
            print(f"\n[4] Composite particle atoms: {len(syms)}")
         
        # --- Write core.xyz and shell.xyz (behind --write-all) ---
        if args.write_all:
            # Reconstruct inner mask using core aspect at same R on the shell lattice
            s_shell = Structure.from_file(shell.cif)
            facets_shell = expand_facets(s_shell, shell.seeds, proper_only=cfg.proper_only)
            planes_core = halfspaces(s_shell, facets_shell, R=float(args.radius), aspect=core.aspect)

            # inside test: n·x ≤ d + tol for all planes
            import numpy as np
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

