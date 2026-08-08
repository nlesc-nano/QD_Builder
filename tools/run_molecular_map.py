#!/usr/bin/env python3
"""Run lattice-free molecular (k,p) map generation.

Example::

    python tools/run_molecular_map.py \\
      --yaml examples/nucleation/cdse_molecular_rules.yaml \\
      --kmax 1 --pmax 3 \\
      --output runs/molecular_cdse_k1

Resume after a killed skeleton dump / decorate only saved cores::

    # finish skeleton dump only for bins not yet in skeletons.csv
    python tools/run_molecular_map.py ... --output runs/X --resume-skeletons \\
        --skeletons-only

    # passivate from saved edge lists (no skeleton re-enumeration)
    python tools/run_molecular_map.py ... --output runs/X --decorate-only \\
        --kmin 5 --kmax 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from builder.nucleation import (  # noqa: E402
    EnumerationLimitError,
    dump_skeletons_upfront,
    generate_molecular_map,
    load_geometry_pack,
    load_nucleation_spec,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml",
        type=Path,
        default=ROOT / "examples/nucleation/cdse_molecular_rules.yaml",
    )
    parser.add_argument(
        "--pack",
        type=Path,
        default=None,
        help="override the geometry_pack referenced by the run YAML",
    )
    parser.add_argument("--kmin", type=int, default=1)
    parser.add_argument("--kmax", type=int, default=1)
    parser.add_argument("--pmin", type=int, default=0)
    parser.add_argument(
        "--pmax",
        type=int,
        default=None,
        help="maximum p (default: derive from Se coordination capacity)",
    )
    parser.add_argument(
        "--max-skeletons",
        type=int,
        default=2000,
        help="safety guard per bin; this is not a chemical p limit",
    )
    parser.add_argument(
        "--max-decoration-assignments",
        type=int,
        default=0,
        help=(
            "optional safety guard per skeleton (0 = unlimited); this counts "
            "Cl assignments and is not the graph-derived chemical p capacity"
        ),
    )
    parser.add_argument(
        "--extra-skeleton-edges",
        type=int,
        default=None,
        help=(
            "cap how many Cd-Se bonds beyond the connected-skeleton minimum "
            "to enumerate (default: no cap, bounded only by coordination "
            "capacity); the run is reported as incomplete when this binds"
        ),
    )
    parser.add_argument(
        "--frame-options",
        type=int,
        default=0,
        help=(
            "how many sound cation-anion frames are kept per coordination vector "
            "before calling a molecule unrealisable (0 = all, the default, "
            "which saturates); lowering it trades structures for time "
            "(k=2,p=3: 20/34/50/55/59 accepted for 1/2/4/8/16)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/molecular_map",
    )
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument(
        "--target-isomers",
        type=int,
        default=0,
        help=(
            "stop each (k,p) once this many isomers are accepted (0 = "
            "exhaustive, the default). Candidates are embedded in a "
            "density-balanced order: bands are the distinct total bond counts "
            "present in that bin, each gets an equal share of the budget, and "
            "they are drawn round-robin -- so a truncated bin spans the bond-"
            "count range instead of piling up at the sparse or dense end"
        ),
    )
    parser.add_argument(
        "--dump-failures",
        action="store_true",
        help=(
            "also write one representative geometry per distinct failure to "
            "k***/p***/failures/*.xyz. Off by default because a run rejects "
            "far more candidates than it accepts; failure_manifest.csv still "
            "records every failure's stage, CN vector, reason and count"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "suppress per-bin/progress messages; files and checkpoints are "
            "still written"
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write bins even when an exact-enumeration safety limit is reached",
    )
    parser.add_argument(
        "--skip-skeleton-dump",
        action="store_true",
        help="do not run the upfront skeleton dump (go straight to decoration)",
    )
    parser.add_argument(
        "--resume-skeletons",
        action="store_true",
        help=(
            "when dumping, skip (k,p) bins already present in output/skeletons.csv "
            "so a killed dump can continue"
        ),
    )
    parser.add_argument(
        "--skeletons-only",
        action="store_true",
        help="only write/resume the skeleton dump; do not decorate",
    )
    parser.add_argument(
        "--decorate-only",
        action="store_true",
        help=(
            "skip skeleton dump and decorate using accepted edge lists from "
            "--skeletons-from (default: --output). Bins without saved edges are "
            "skipped unless you also omit --require-saved-skeletons"
        ),
    )
    parser.add_argument(
        "--skeletons-from",
        type=Path,
        default=None,
        help=(
            "directory with skeletons.csv (edges column) to load for decoration; "
            "defaults to --output when --decorate-only is set"
        ),
    )
    parser.add_argument(
        "--require-saved-skeletons",
        action="store_true",
        help=(
            "with a skeleton catalog: skip (k,p) bins that are not in the dump "
            "instead of re-enumerating them (default when --decorate-only)"
        ),
    )
    args = parser.parse_args(argv)
    if args.kmin < 1 or args.kmax < args.kmin:
        parser.error("require 1 <= --kmin <= --kmax")
    if args.pmin < 0:
        parser.error("--pmin must be >= 0")
    if args.pmax is not None and args.pmax < args.pmin:
        parser.error("require --pmin <= --pmax")
    if args.decorate_only and args.skeletons_only:
        parser.error("use only one of --decorate-only / --skeletons-only")

    decorate_only = bool(args.decorate_only)
    skip_dump = bool(args.skip_skeleton_dump) or decorate_only
    skeletons_from = args.skeletons_from
    if decorate_only and skeletons_from is None:
        skeletons_from = args.output
    require_saved = bool(args.require_saved_skeletons) or decorate_only

    spec = load_nucleation_spec(args.yaml, geometry_pack=args.pack)
    if spec.geometry_pack is None:
        parser.error("molecular runs require nucleation.geometry_pack or --pack")
    pack = load_geometry_pack(spec.geometry_pack)
    print("=" * 88)
    print("MOLECULAR (k,p) STRUCTURE ENUMERATION")
    print(f"  run settings : {args.yaml}")
    print(f"  geometry pack: {pack.name} ({spec.geometry_pack})")
    p_sweep = (
        f"{args.pmin}..{args.pmax}"
        if args.pmax is not None
        else (
            f"{args.pmin}..automatic from accepted p=0 Se slots "
            "(slot-based; global Se bound as fallback)"
        )
    )
    print(f"  sweep        : k={args.kmin}..{args.kmax}, p={p_sweep}")
    embedding_label = (
        "not run (graph-only skeleton catalog)"
        if args.skeletons_only
        else ("off" if args.no_embed else "fixed-rule geometry")
    )
    print(f"  embedding    : {embedding_label}")
    print(f"  output       : {args.output}")
    if skip_dump:
        print("  skeleton dump: skipped")
    elif args.resume_skeletons:
        print("  skeleton dump: resume (skip finished bins)")
    else:
        print("  skeleton dump: full")
    if skeletons_from is not None:
        print(f"  decorate from: {skeletons_from}")
        if require_saved:
            print("  missing bins : skip (require saved skeletons)")
    if args.skeletons_only:
        print("  mode         : skeletons only (no decoration)")
    if decorate_only:
        print("  mode         : decorate only")
    print("=" * 88)

    args.output.mkdir(parents=True, exist_ok=True)
    progress = None if args.quiet else print

    if not skip_dump:
        dump_skeletons_upfront(
            spec,
            args.output,
            pack=pack,
            kmin=args.kmin,
            kmax=args.kmax,
            pmin=args.pmin,
            pmax=args.pmax,
            max_skeletons=args.max_skeletons,
            extra_skeleton_edges=args.extra_skeleton_edges,
            embed=not args.no_embed,
            progress=progress,
            resume=bool(args.resume_skeletons),
        )
        print(
            f"[molecular] graph-only skeleton catalog ready at "
            f"{args.output}/skeletons.csv"
        )
        print("=" * 88)

    if args.skeletons_only:
        print("COMPLETE | skeletons only (decoration skipped)")
        return 0

    try:
        result = generate_molecular_map(
            spec,
            geometry_pack=pack,
            kmin=args.kmin,
            kmax=args.kmax,
            pmin=args.pmin,
            pmax=args.pmax,
            embed=not args.no_embed,
            allow_incomplete=args.allow_incomplete,
            progress=progress,
            max_skeletons=args.max_skeletons,
            max_decoration_assignments=args.max_decoration_assignments,
            extra_skeleton_edges=args.extra_skeleton_edges,
            frame_options=args.frame_options,
            dump_failures=args.dump_failures,
            target_isomers=args.target_isomers,
            incremental_output=args.output,
            skeleton_catalog_dir=skeletons_from,
            require_catalog_skeletons=require_saved and skeletons_from is not None,
        )
    except EnumerationLimitError as exc:
        print("=" * 88)
        print(f"STOPPED BY ENUMERATION SAFETY GUARD | {exc}")
        print(
            f"Skeletons already written under {args.output}/k***/skeletons/"
        )
        return 2
    out = args.output
    print("=" * 88)
    print(f"COMPLETE | wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
