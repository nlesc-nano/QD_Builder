#!/usr/bin/env python3
"""Normalize growth manifests and analyze bridge/coordination patterns."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from builder.nucleation.search_analysis import analyze_corpus
from builder.nucleation.spec import load_nucleation_spec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--map",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "geometry_packs/cdse_cdcl2_zb/run_gxtb.yaml",
    )
    ap.add_argument(
        "--graphs-only",
        action="store_true",
        help="Skip geometry consolidation; label minima as unverified",
    )
    args = ap.parse_args()
    if (
        args.output.resolve() == args.corpus.resolve()
        or args.corpus.resolve() in args.output.resolve().parents
    ):
        ap.error("output must be outside the source corpus")
    analyze_corpus(
        args.corpus,
        args.output,
        load_nucleation_spec(args.map),
        geometry=not args.graphs_only,
    )


if __name__ == "__main__":
    main()
