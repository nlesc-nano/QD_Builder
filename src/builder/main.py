# nanocrystal_builder/main.py
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
from .nc_types import Facet
from .facets import expand_facets, detect_facets_from_nc
from .geometry import build_nanocrystal, dedupe_points
from .io_utils import write_xyz, write_manifest, center_coords
from .passivation import collect_anion_candidates, charge_balance
from .analysis import facet_families_overview, facet_atom_report
from .cleanup import prune_low_coord_sites


def main(argv: List[str] | None = None) -> int:
    # -----------------------------
    # [0] Parse CLI & seed RNG
    # -----------------------------
    p = build_parser()
    args = p.parse_args(argv)
    random.seed(args.seed)

    # -----------------------------
    # [1] Read CIF
    # -----------------------------
    if args.verbose:
        print("\n[1] Reading CIF structure...")
    struct = Structure.from_file(args.cif)
    if args.verbose:
        print(f"    - Loaded {len(struct)} atoms from {args.cif}")

    # -----------------------------
    # [2] Parse YAML config
    # -----------------------------
    if args.verbose:
        print("\n[2] Parsing YAML config...")
    seeds, ligand, surf_tol, charges, pair_opposites = parse_yaml_config(args.yaml)
    if args.verbose:
        print(f"    - Facet seeds: {[ (f.h, f.k, f.l) for f in seeds ]}")
        print(f"    - Ligand: {ligand}, surf_tol={surf_tol:.3f} Å")
        print(f"    - Charges: {charges}")
        print(f"    - Pair opposites: {bool(pair_opposites)}")
        print(f"    - Proper rotations only: {bool(args.proper_rotations_only)}")

    # -----------------------------
    # [3] Expand symmetry → Wulff facets
    # -----------------------------
    if args.verbose:
        print("\n[3] Expanding symmetry & building Wulff facets...")
    wulff_facets: List[Facet] = expand_facets(struct, seeds, proper_only=args.proper_rotations_only)
    if args.verbose:
        print(f"    - Expanded to {len(wulff_facets)} oriented facets")

    # -----------------------------
    # [4] Build Wulff-cut nanocrystal
    # -----------------------------
    if args.verbose:
        print("\n[4] Building nanocrystal from Wulff facets...")
    syms, pts, _planes_geo = build_nanocrystal(struct, wulff_facets, args.radius)
    syms, pts = dedupe_points(syms, pts, tol=1e-3)
    if args.verbose:
        print(f"    - Cut particle: {len(syms)} atoms")

    # Optional: prune monocoordinated sites
    if args.prune_mono:
        if args.verbose:
            print("\n[4b] Pruning low-coordination atoms (pre-facet detection)...")
        syms, pts, n_removed, n_pass = prune_low_coord_sites(
            syms, pts, min_cn=args.prune_min_cn, max_passes=args.prune_passes, verbose=args.verbose
        )
        if args.verbose:
            print(f"    - Pruned {n_removed} atoms in {n_pass} pass(es); remaining {len(syms)} atoms")

    # -----------------------------
    # [5] Detect actual exposed facets
    # -----------------------------
    if args.verbose:
        print("\n[5] Detecting actual exposed facets...")
    facets, planes = detect_facets_from_nc(syms, pts, struct.lattice, charges, wulff_facets, surf_tol)
    if args.verbose:
        print(f"    - Detected {len(facets)} facets")

    # -----------------------------
    # [6] Reports
    # -----------------------------
    if args.verbose:
        print("\n[6] Surface atom and CN reports:")
    facet_families_overview(syms, pts, planes, facets, surf_tol=surf_tol, charges=charges)
    facet_atom_report(syms, pts, planes, facets, surf_tol=surf_tol, charges=charges)

    # -----------------------------
    # [7] Write initial cut snapshot
    # -----------------------------
    prefix = os.path.splitext(os.path.basename(args.out))[0]
    if args.write_all:
        if args.verbose:
            print(f"\n[7] Writing initial cut XYZ to {prefix}_cut.xyz")
        write_xyz(f"{prefix}_cut.xyz", syms, center_coords(pts) if args.center else pts)

    # -----------------------------
    # [8] Gather outer-layer anion candidates (no mutation)
    # -----------------------------
    if args.verbose:
        print("\n[8] Gathering outer-layer anion candidates...")
    outer_cands, subl_cands = collect_anion_candidates(
        syms, pts, planes, charges, ligand, surf_tol, verbose=args.verbose
    )

    # (Optional) pre-balance snapshot would be identical to _cut at this point
    if args.write_all and args.verbose:
        print("\n[9] Skipping pre-balance snapshot (no swaps applied yet).")

    # -----------------------------
    # [10] Stepwise charge balancing
    # -----------------------------
    if args.verbose:
        print("\n[10] Balancing charge stepwise (outer anions first; then add/remove ligands if needed)...")
    syms, pts = charge_balance(
        syms, pts,
        outer_cands, subl_cands,
        charges, ligand,
        verbose=args.verbose,
        planes=planes, facets=facets, surf_tol=surf_tol,
        rng=random,
        prefer_remove_parity=(args.parity == "remove"),
    )

    # -----------------------------
    # [11] Write final XYZ
    # -----------------------------
    if args.verbose:
        print(f"\n[11] Writing final XYZ to {args.out}")
    final_pts = center_coords(pts) if args.center else pts
    write_xyz(args.out, syms, final_pts)

    # -----------------------------
    # [12] Manifest + counts
    # -----------------------------
    if args.verbose:
        print(f"[12] Writing JSON manifest to {prefix}.json")
    write_manifest(prefix, syms, charges)

    if args.verbose:
        from collections import Counter
        cnt = Counter(syms)
        print("\n### ELEMENT COUNTS ###")
        for k in sorted(cnt):
            print(f" {k}: {cnt[k]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

