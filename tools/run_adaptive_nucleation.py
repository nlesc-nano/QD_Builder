#!/usr/bin/env python3
"""Run automated lineage completion (A), k=7 extension (B), and k=8 (C)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from builder.nucleation.adaptive import AdaptiveConfig, AdaptivePilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "geometry_packs/cdse_cdcl2_zb/adaptive_abc.yaml",
    )
    parser.add_argument(
        "--pack-dir", type=Path, default=ROOT / "geometry_packs/cdse_cdcl2_zb"
    )
    parser.add_argument(
        "--growth",
        type=Path,
        default=ROOT / "geometry_packs/cdse_cdcl2_zb/growth_agnostic_k5.yaml",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="completed minima archive, or a chemistry-adapter seed directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = AdaptiveConfig.load(args.config)
    if not (args.source / "minima.json").is_file() and not (
        args.source / "index.csv"
    ).is_file():
        parser.error("source contains neither minima.json nor index.csv")
    if args.source.resolve() == args.output.resolve():
        parser.error("adaptive output must differ from its source archive")
    if args.source.resolve() in args.output.resolve().parents:
        parser.error("adaptive output must not be inside the source archive")
    if (args.output / "index.csv").exists() or (
        args.output / "zb_occupations.jsonl"
    ).exists():
        parser.error("refusing to reuse a legacy growth directory")

    if args.dry_run:
        print(
            json.dumps(
                dict(
                    protocol="adaptive_A_B_C",
                    chemistry_adapter=config.chemistry_adapter,
                    source=str(args.source.resolve()),
                    phases={
                        phase: ks
                        for phase, ks in {
                            "A": config.phase_a_k,
                            "B": [config.phase_b_k],
                            "C": [config.phase_c_k],
                        }.items()
                        if phase in config.enabled_phases
                    },
                    per_k_call_caps=config.stage_limits(),
                    per_operation_call_caps=config.operation_calls,
                    max_calls=config.max_calls,
                    family_slots=config.family_slots,
                    family_policy=dict(
                        coarse_survival=True,
                        subfamilies_per_family=config.subfamilies_per_family,
                        geometries_per_family=config.geometries_per_family,
                        initial_fraction=1.0 - config.admission_fraction,
                        admission_fraction=config.admission_fraction,
                        admission_fraction_by_k=config.admission_fraction_by_k,
                        source_novelty_fraction=config.source_novelty_fraction,
                        minimum_launched_cycles=config.minimum_launched_cycles,
                        import_source_cohorts=config.import_source_cohorts,
                    ),
                    phase_cycle_start=config.phase_cycle_start,
                    plateau=dict(
                        patience=config.convergence_patience,
                        new_families_per_100_calls=config.new_families_per_100_calls,
                        convergence_energy_window_eV=(
                            config.convergence_energy_window_eV
                        ),
                        energy_improvement_eV=config.energy_improvement_eV,
                        minimum_endpoint_fraction=config.minimum_endpoint_fraction,
                    ),
                    workers=config.workers,
                    launch_hours=config.launch_hours,
                ),
                indent=2,
            )
        )
        return

    AdaptivePilot(
        config,
        args.pack_dir / "run_gxtb.yaml",
        args.growth,
        args.source,
        args.output,
    ).run()


if __name__ == "__main__":
    main()
