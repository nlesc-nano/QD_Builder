#!/usr/bin/env python3
"""Mine CdSe/CdCl2 molecular geometry rules from DFT trajectories.

Standalone: does **not** use analyze_cp2k_results.py or its analysis trees.

Example::

    python tools/mine_cdse_dft_geometry.py \\
      --root /path/to/cdse_map/dft_all \\
      --root /path/to/cdse_map/dft_k5_partial \\
      --root /path/to/cdse_map/dft_k6_additional \\
      --output runs/cdse_map/geometry_mine \\
      --compare-start
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as script without installing package.
_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from dft_geometry_mine.analyze import analyze_job  # noqa: E402
from dft_geometry_mine.bonds import BondCutoffs  # noqa: E402
from dft_geometry_mine.discover import discover_jobs  # noqa: E402
from dft_geometry_mine.report import write_all_outputs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone geometry miner for CdSe DFT relaxations "
            "(no analyze_cp2k_results dependency)."
        )
    )
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="DFT root directory (repeatable). Walked for CdSe-pos-1.xyz.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for CSV/YAML/Markdown reports.",
    )
    parser.add_argument("--r-cd-se", type=float, default=3.25)
    parser.add_argument("--r-cd-cl", type=float, default=3.10)
    parser.add_argument("--r-cd-cd", type=float, default=3.20)
    parser.add_argument("--r-se-se", type=float, default=3.80)
    parser.add_argument("--r-cl-cl", type=float, default=2.70)
    parser.add_argument(
        "--linear-threshold",
        type=float,
        default=160.0,
        help="Cd CN2 angle (deg) threshold for 'linear' motif flag.",
    )
    parser.add_argument(
        "--compare-start",
        action="store_true",
        help="Also analyse start.xyz / first trajectory frame.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max jobs (0 = all), for smoke tests.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cutoffs = BondCutoffs(
        cd_se=args.r_cd_se,
        cd_cl=args.r_cd_cl,
        cd_cd=args.r_cd_cd,
        se_se=args.r_se_se,
        cl_cl=args.r_cl_cl,
    )
    jobs = discover_jobs(args.root)
    if args.limit and args.limit > 0:
        jobs = jobs[: args.limit]
    if not jobs:
        print("No jobs found under given --root paths.", file=sys.stderr)
        return 1

    print(f"Discovered {len(jobs)} jobs.")
    results = []
    for index, job in enumerate(jobs, start=1):
        if index == 1 or index % 25 == 0 or index == len(jobs):
            print(f"  [{index}/{len(jobs)}] {job.structure_id}")
        results.append(
            analyze_job(
                job,
                cutoffs=cutoffs,
                linear_threshold_deg=args.linear_threshold,
                compare_start=args.compare_start,
            )
        )

    summary = write_all_outputs(
        args.output,
        results,
        cutoffs={
            "cd_se": cutoffs.cd_se,
            "cd_cl": cutoffs.cd_cl,
            "cd_cd": cutoffs.cd_cd,
            "se_se": cutoffs.se_se,
            "cl_cl": cutoffs.cl_cl,
        },
    )
    counts = summary["counts"]
    print(
        f"Done. geometry={counts['with_geometry']} "
        f"clean={counts['clean']} quarantine={counts['quarantine']} "
        f"missing={counts['missing_trajectory']}"
    )
    print(f"Wrote outputs under {args.output}")
    rec = summary["recommendations"]["hard_rules_supported_by_clean_set"]
    print(
        "Clean inorganic connected fraction:",
        rec.get("inorganic_CdSe_connected_fraction"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
