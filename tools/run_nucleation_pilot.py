#!/usr/bin/env python3
"""Run an explicitly budgeted control/experimental nucleation comparison."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from builder.nucleation.pilot import Pilot, PilotConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", type=Path, default=ROOT / "geometry_packs/cdse_cdcl2_zb/pilot.yaml"
    )
    ap.add_argument(
        "--pack-dir", type=Path, default=ROOT / "geometry_packs/cdse_cdcl2_zb"
    )
    ap.add_argument(
        "--growth",
        type=Path,
        default=ROOT / "geometry_packs/cdse_cdcl2_zb/growth_agnostic_k5.yaml",
    )
    ap.add_argument("--seeds", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print budgets; no files or calculations",
    )
    args = ap.parse_args()
    config = PilotConfig.load(args.config)
    if not args.seeds.is_dir():
        ap.error("seed directory does not exist")
    if (
        args.output.resolve() == args.seeds.resolve()
        or args.seeds.resolve() in args.output.resolve().parents
    ):
        ap.error("pilot output must be outside seed data")
    if (args.output / "index.csv").exists() or (
        args.output / "zb_occupations.jsonl"
    ).exists():
        ap.error("refusing to reuse a legacy growth directory")
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    seed_calls=config.seed_calls,
                    per_arm_stage_calls=config.stage_limits(),
                    max_calls=config.max_calls,
                    workers=config.workers,
                    launch_hours=config.launch_hours,
                ),
                indent=2,
            )
        )
        return
    Pilot(
        config, args.pack_dir / "run_gxtb.yaml", args.growth, args.seeds, args.output
    ).run()


if __name__ == "__main__":
    main()
