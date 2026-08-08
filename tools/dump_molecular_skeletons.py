#!/usr/bin/env python3
"""Dump Cd–Se skeletons for each (k, p) with CN lists and XYZ.

Does **not** run Cl decoration (fast).  Use this to inspect why high-p bins
have few usable skeletons and what their Cd/Se coordination looks like.

Example
-------

    python tools/dump_molecular_skeletons.py \\
      --yaml examples/nucleation/cdse_molecular_rules.yaml \\
      --kmin 2 --kmax 2 --pmin 3 --pmax 6 \\
      --output runs/skeletons_k2

Writes::

    runs/skeletons_k2/
      skeletons.csv
      k002/skeletons/
        skeletons.md
        skeletons.tsv
        skeleton_k2_p3_001.xyz
        skeleton_k2_p6_001.xyz
        ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from builder.nucleation.geometry_pack import load_geometry_pack  # noqa: E402
from builder.nucleation.molecular import dump_skeletons_upfront  # noqa: E402
from builder.nucleation.spec import load_nucleation_spec  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml",
        type=Path,
        default=ROOT / "examples/nucleation/cdse_molecular_rules.yaml",
    )
    parser.add_argument("--kmin", type=int, default=1)
    parser.add_argument("--kmax", type=int, default=2)
    parser.add_argument("--pmin", type=int, default=0)
    parser.add_argument("--pmax", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output directory for skeleton XYZ + tables",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="skip geometry (no XYZ coords; still write CN tables)",
    )
    args = parser.parse_args(argv)

    spec = load_nucleation_spec(args.yaml)
    pack = None
    if not args.no_embed:
        if spec.geometry_pack:
            pack_path = Path(spec.geometry_pack)
            if not pack_path.is_file():
                pack_path = (args.yaml.parent / spec.geometry_pack).resolve()
            if pack_path.is_file():
                pack = load_geometry_pack(pack_path)
        if pack is None:
            alt = ROOT / "geometry_packs/cdse_cdcl2_molecular.yaml"
            if alt.is_file():
                pack = load_geometry_pack(alt)

    out = dump_skeletons_upfront(
        spec,
        args.output,
        pack=pack,
        kmin=args.kmin,
        kmax=args.kmax,
        pmin=args.pmin,
        pmax=args.pmax,
        embed=not args.no_embed and pack is not None,
        progress=print,
    )
    print(f"Wrote {out}")
    print(f"  global table: {out / 'skeletons.csv'}")
    print("  layout: k###/skeletons/skeleton_k{k}_p{p}_###.xyz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
