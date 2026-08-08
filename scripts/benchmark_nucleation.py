#!/usr/bin/env python3
"""Run an opt-in nucleation performance audit without writing a bundle."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from builder.nucleation import generate_nucleation_result, load_nucleation_spec

# Counters worth tracking across an optimisation; keep the names stable so
# recorded ceilings in the test suite stay comparable between runs.
_TRACKED = (
    "theoretical_assignments",
    "orbit_representatives",
    "identical_host_pruned",
    "bridge_search_states",
    "bridge_bases_bound_pruned",
    "bridge_bases_dp_pruned",
    "greedy_incumbent_found",
    "greedy_incumbent_matches_selection",
    "bridge_exactness_certified",
    "bridge_sub_maximum_fallbacks",
    "bridge_sub_maximum_contenders",
    "bridge_sub_maximum_undischarged",
    "bridge_symmetry_pruned",
    "bridge_orbit_representatives",
    "bridge_raw_extensions",
    "dominated_bridge_variants_pruned",
)


def collect_bin_metrics(result) -> list[dict[str, int]]:
    """Extract the per-bin search-effort counters from a nucleation result.

    Shared with the test suite so a performance regression is caught by the
    same numbers the benchmark reports.
    """

    metrics: list[dict[str, int]] = []
    for audit in result.sweep_audit:
        if audit.operation != "dag_bin":
            continue
        row = {"k": int(audit.k), "p": int(audit.p_from)}
        row["theoretical_assignments"] = int(
            audit.stage_counts.get("theoretical_assignments", audit.raw_count)
        )
        for key in _TRACKED[1:]:
            row[key] = int(audit.stage_counts.get(key, 0))
        metrics.append(row)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml", type=Path)
    parser.add_argument("--kmax", type=int)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable metrics instead of the ASCII table.",
    )
    args = parser.parse_args()

    spec = load_nucleation_spec(args.yaml)
    if args.kmax is not None:
        spec = replace(spec, kmax=args.kmax)
    started = time.monotonic()
    result = generate_nucleation_result(
        spec,
        progress=print if args.progress else None,
    )
    elapsed = time.monotonic() - started
    metrics = collect_bin_metrics(result)

    if args.json:
        print(
            json.dumps(
                {
                    "kmax": spec.kmax,
                    "elapsed_seconds": round(elapsed, 3),
                    "totals": {
                        key: sum(row[key] for row in metrics) for key in _TRACKED
                    },
                    "bins": metrics,
                },
                indent=2,
            )
        )
        return

    print("k  p  theoretical  orbit_reps  reduction  bridge_search  dominated")
    for row in metrics:
        orbit_reps = row["orbit_representatives"]
        reduction = (
            float(row["theoretical_assignments"]) / orbit_reps
            if orbit_reps
            else 1.0
        )
        print(
            f"{row['k']:<2d} {row['p']:<2d} "
            f"{row['theoretical_assignments']:<12d} "
            f"{orbit_reps:<11d} {reduction:>8.1f}x "
            f"{row['bridge_search_states']:<13d} "
            f"{row['dominated_bridge_variants_pruned']}"
        )
    print(f"elapsed_seconds {elapsed:.3f}")


if __name__ == "__main__":
    main()
