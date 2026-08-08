#!/usr/bin/env python3
"""Find relaxed-structure descriptors associated with stability within each bin.

This is a post-processing filter for the DFT map.  It compares structures only
within the same ``(k,p)`` bin, so absolute composition-dependent energies do not
drive the result.  The input ``cdcl2_shedding`` directory is produced by
``analyze_cdcl2_shedding.py`` with ``--include-nonready`` when geometry-limit
and technically usable interrupted relaxations should be retained.

Example::

    python scripts/analyze_cdcl2_shedding.py \
        --analysis-root runs/cdse_map/analysis_all_dft \
        --registry-root runs/cdse_map \
        --output runs/cdse_map/analysis_all_dft/cdcl2_shedding \
        --include-nonready

    python scripts/analyze_bin_stability_patterns.py \
        --analysis-root runs/cdse_map/analysis_all_dft \
        --shedding-root runs/cdse_map/analysis_all_dft/cdcl2_shedding \
        --output runs/cdse_map/analysis_all_dft/bin_stability_patterns
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence


HARTREE_TO_KCAL_MOL = 627.5094740631


def as_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_final_descriptors(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """Select final structural descriptors, excluding identifiers and strings."""

    if not rows:
        return []
    fields = [field for field in rows[0] if field.startswith("final_")]
    selected: list[str] = []
    for field in fields:
        if field in {"final_topology_fingerprint", "final_scf_converged"}:
            continue
        if any(as_float(row.get(field)) is not None for row in rows):
            selected.append(field)
    return selected


def add_package_aggregates(
    structures: list[dict[str, Any]],
    package_rows: Sequence[Mapping[str, str]],
) -> None:
    by_structure: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in package_rows:
        by_structure[str(row["structure_id"])].append(row)
    for structure in structures:
        packages = by_structure.get(str(structure["structure_id"]), [])
        for motif in ("no_cl", "terminal_only", "one_4ring", "multi_4ring", "other_cl"):
            structure[f"start_pkg_{motif}_count"] = sum(
                row.get("start_motif") == motif for row in packages
            )
            structure[f"relaxed_pkg_{motif}_count"] = sum(
                row.get("relaxed_motif") == motif for row in packages
            )
        persistence = [
            as_float(row.get("cl_bond_persistence_fraction"))
            for row in packages
            if as_float(row.get("cl_bond_persistence_fraction")) is not None
        ]
        structure["mean_pkg_cl_bond_persistence"] = mean(persistence) if persistence else None
        structure["package_count"] = len(packages)


def prepare_structures(
    original_rows: Sequence[Mapping[str, str]],
    summary_rows: Sequence[Mapping[str, str]],
    package_rows: Sequence[Mapping[str, str]],
    *,
    quality_statuses: set[str] | None,
) -> list[dict[str, Any]]:
    summary_by_id = {str(row["structure_id"]): row for row in summary_rows}
    structures: list[dict[str, Any]] = []
    for original in original_rows:
        structure_id = str(original.get("structure_id", ""))
        summary = summary_by_id.get(structure_id)
        if summary is None:
            continue
        quality = str(summary.get("quality_status", original.get("quality_status", "")))
        if quality_statuses and quality not in quality_statuses:
            continue
        energy = as_float(summary.get("energy_hartree"))
        if energy is None:
            energy = as_float(original.get("energy_hartree"))
        if energy is None:
            continue
        row: dict[str, Any] = dict(original)
        row.update(
            {
                "structure_id": structure_id,
                "k": int(summary["k"]),
                "p": int(summary["p"]),
                "quality_status": quality,
                "energy_hartree": energy,
                "skeleton_family_id": summary.get("skeleton_family_id", ""),
                "bare_skeleton_hash": summary.get("bare_skeleton_hash", ""),
                "core_skeleton_hash": summary.get("core_skeleton_hash", ""),
            }
        )
        structures.append(row)
    add_package_aggregates(structures, package_rows)
    return structures


def rank_bins(
    structures: Sequence[Mapping[str, Any]],
    *,
    winner_window_kcal: float,
    high_gap_kcal: float,
    extreme_gap_kcal: float,
) -> list[dict[str, Any]]:
    bins: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in structures:
        bins[(int(row["k"]), int(row["p"]))].append(row)
    ranked: list[dict[str, Any]] = []
    for (k, p), records in sorted(bins.items()):
        minimum = min(float(row["energy_hartree"]) for row in records)
        for row in records:
            gap = (float(row["energy_hartree"]) - minimum) * HARTREE_TO_KCAL_MOL
            if gap <= winner_window_kcal:
                group = "winner_window"
            elif gap >= extreme_gap_kcal:
                group = "extreme_unstable"
            elif gap >= high_gap_kcal:
                group = "high_gap"
            else:
                group = "intermediate"
            ranked.append(
                {
                    **row,
                    "bin_min_energy_hartree": minimum,
                    "relative_energy_kcal_mol": gap,
                    "stability_group": group,
                    "bin_size": len(records),
                }
            )
    return ranked


def descriptor_comparisons(
    ranked: Sequence[Mapping[str, Any]],
    descriptors: Sequence[str],
    *,
    stable_group: str,
    unstable_group: str,
) -> list[dict[str, Any]]:
    bins: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in ranked:
        bins[(int(row["k"]), int(row["p"]))].append(row)
    output: list[dict[str, Any]] = []
    for descriptor in descriptors:
        deltas: list[float] = []
        stable_values: list[float] = []
        unstable_values: list[float] = []
        matched_bins = 0
        for records in bins.values():
            stable = [
                as_float(row.get(descriptor))
                for row in records
                if row.get("stability_group") == stable_group
            ]
            unstable = [
                as_float(row.get(descriptor))
                for row in records
                if row.get("stability_group") == unstable_group
            ]
            stable = [value for value in stable if value is not None and math.isfinite(value)]
            unstable = [value for value in unstable if value is not None and math.isfinite(value)]
            if not stable or not unstable:
                continue
            matched_bins += 1
            stable_values.extend(stable)
            unstable_values.extend(unstable)
            deltas.append(mean(unstable) - mean(stable))
        if not deltas:
            continue
        scale = max(
            abs(value) for value in stable_values + unstable_values
        ) or 1.0
        output.append(
            {
                "descriptor": descriptor,
                "matched_bin_count": matched_bins,
                "mean_unstable_minus_stable": mean(deltas),
                "median_unstable_minus_stable": median(deltas),
                "pooled_stable_mean": mean(stable_values),
                "pooled_unstable_mean": mean(unstable_values),
                "pooled_difference": mean(unstable_values) - mean(stable_values),
                "normalized_pooled_difference": (
                    (mean(unstable_values) - mean(stable_values)) / scale
                ),
                "direction": "higher_in_unstable"
                if mean(deltas) > 0
                else "lower_in_unstable",
            }
        )
    output.sort(
        key=lambda row: abs(float(row["normalized_pooled_difference"])), reverse=True
    )
    return output


def commonality_rows(
    ranked: Sequence[Mapping[str, Any]],
    descriptors: Sequence[str],
    *,
    group: str,
) -> list[dict[str, Any]]:
    selected = [row for row in ranked if row.get("stability_group") == group]
    output: list[dict[str, Any]] = []
    for descriptor in descriptors:
        values = [
            as_float(row.get(descriptor))
            for row in selected
            if as_float(row.get(descriptor)) is not None
        ]
        if not values:
            continue
        output.append(
            {
                "group": group,
                "descriptor": descriptor,
                "structure_count": len(values),
                "mean": mean(values),
                "median": median(values),
                "min": min(values),
                "max": max(values),
                "nonzero_fraction": mean(float(value > 0.0) for value in values),
            }
        )
    return output


def write_report(
    path: Path,
    ranked: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    extreme_comparisons: Sequence[Mapping[str, Any]],
    commonality: Sequence[Mapping[str, Any]],
    *,
    winner_window_kcal: float,
    high_gap_kcal: float,
    extreme_gap_kcal: float,
) -> None:
    quality = Counter(str(row.get("quality_status", "")) for row in ranked)
    groups = Counter(str(row.get("stability_group", "")) for row in ranked)
    bins = {(int(row["k"]), int(row["p"])) for row in ranked}
    lines = [
        "# Stability patterns within `(k,p)` bins",
        "",
        f"Structures included: **{len(ranked)}**; bins: **{len(bins)}**",
        f"Quality statuses: `{dict(quality)}`",
        f"Stability groups: `{dict(groups)}`",
        "",
        "All energy gaps are recomputed within each `(k,p)` bin from the absolute "
        "DFT energies. The winner window is "
        f"{winner_window_kcal:g} kcal/mol, high-gap is ≥{high_gap_kcal:g}, and "
        f"extreme-gap is ≥{extreme_gap_kcal:g}.",
        "",
        "## Descriptor differences",
        "",
        "The main comparison is high-gap minus winner-window, with both groups "
        "required to occur in the same `(k,p)` bin. Positive values mean the "
        "descriptor is larger in the unstable structures.",
        "",
        "| descriptor | matched bins | mean within-bin difference | pooled stable | pooled unstable | direction |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in comparisons[:30]:
        lines.append(
            f"| {row['descriptor']} | {row['matched_bin_count']} | "
            f"{float(row['mean_unstable_minus_stable']):.3g} | "
            f"{float(row['pooled_stable_mean']):.3g} | "
            f"{float(row['pooled_unstable_mean']):.3g} | {row['direction']} |"
        )
    lines.extend([
        "",
        "## Extreme-gap descriptor differences",
        "",
        "This repeats the comparison for structures at or above the extreme-gap "
        "threshold, which is useful for identifying failure modes rather than "
        "ordinary near-degeneracies.",
        "",
        "| descriptor | matched bins | mean within-bin difference | pooled stable | pooled extreme | direction |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in extreme_comparisons[:30]:
        lines.append(
            f"| {row['descriptor']} | {row['matched_bin_count']} | "
            f"{float(row['mean_unstable_minus_stable']):.3g} | "
            f"{float(row['pooled_stable_mean']):.3g} | "
            f"{float(row['pooled_unstable_mean']):.3g} | {row['direction']} |"
        )
    common_by_descriptor: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in commonality:
        common_by_descriptor[str(row["descriptor"])][str(row["group"])] = row
    commonity_rows: list[tuple[float, str, Mapping[str, Any], Mapping[str, Any]]] = []
    for descriptor, values in common_by_descriptor.items():
        winner = values.get("winner_window")
        extreme = values.get("extreme_unstable")
        if winner is None or extreme is None:
            continue
        difference = float(extreme["nonzero_fraction"]) - float(
            winner["nonzero_fraction"]
        )
        commonity_rows.append((abs(difference), descriptor, winner, extreme))
    lines.extend([
        "",
        "## Motif/descriptor commonality",
        "",
        "The table shows descriptors whose nonzero occurrence changes most between "
        "the winner window and the extreme-gap group.",
        "",
        "| descriptor | winner nonzero fraction | extreme nonzero fraction | difference |",
        "|---|---:|---:|---:|",
    ])
    for _, descriptor, winner, extreme in sorted(commonity_rows, reverse=True)[:20]:
        difference = float(extreme["nonzero_fraction"]) - float(
            winner["nonzero_fraction"]
        )
        lines.append(
            f"| {descriptor} | {float(winner['nonzero_fraction']):.3f} | "
            f"{float(extreme['nonzero_fraction']):.3f} | {difference:+.3f} |"
        )
    lines.extend([
        "",
        "## Bin-level interpretation",
        "",
        "Use the descriptor table as a filter diagnostic, not as a hard chemical "
        "rule. A descriptor is interesting only when it separates structures in "
        "several independent bins and remains meaningful after inspecting the "
        "corresponding XYZ structures.",
        "",
        "Quality-limited structures are retained because their final energies and "
        "geometries are available. They are labelled in the CSV; compare results "
        "with and without `geometry_interrupted` cases before using a rule.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--shedding-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--winner-window-kcal", type=float, default=3.0)
    parser.add_argument("--high-gap-kcal", type=float, default=20.0)
    parser.add_argument("--extreme-gap-kcal", type=float, default=30.0)
    parser.add_argument(
        "--quality-status",
        action="append",
        default=None,
        help="restrict quality statuses; repeat the option. Default: all available statuses",
    )
    args = parser.parse_args()
    analysis_root = args.analysis_root.expanduser().resolve()
    shedding_root = args.shedding_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    original_path = analysis_root / "structures.csv"
    summary_path = shedding_root / "cdcl2_structure_summary.csv"
    package_path = shedding_root / "cdcl2_package_descriptors.csv"
    for path in (original_path, summary_path, package_path):
        if not path.is_file():
            parser.error(f"required input not found: {path}")
    original = read_csv(original_path)
    summary = read_csv(summary_path)
    packages = read_csv(package_path)
    statuses = set(args.quality_status) if args.quality_status else None
    structures = prepare_structures(
        original, summary, packages, quality_statuses=statuses
    )
    ranked = rank_bins(
        structures,
        winner_window_kcal=args.winner_window_kcal,
        high_gap_kcal=args.high_gap_kcal,
        extreme_gap_kcal=args.extreme_gap_kcal,
    )
    descriptors = numeric_final_descriptors(original)
    descriptors.extend(
        sorted(
            field
            for field in ranked[0]
            if field.startswith("start_pkg_")
            or field.startswith("relaxed_pkg_")
            or field == "mean_pkg_cl_bond_persistence"
        )
        if ranked
        else []
    )
    descriptors = list(dict.fromkeys(descriptors))
    comparisons = descriptor_comparisons(
        ranked,
        descriptors,
        stable_group="winner_window",
        unstable_group="high_gap",
    )
    extreme_comparisons = descriptor_comparisons(
        ranked,
        descriptors,
        stable_group="winner_window",
        unstable_group="extreme_unstable",
    )
    commonality = commonality_rows(
        ranked, descriptors, group="winner_window"
    ) + commonality_rows(ranked, descriptors, group="extreme_unstable")
    output.mkdir(parents=True, exist_ok=True)
    ranked_fields = list(ranked[0].keys()) if ranked else []
    write_csv(output / "bin_stability_ranked_structures.csv", ranked, ranked_fields)
    write_csv(
        output / "bin_descriptor_comparisons.csv",
        comparisons,
        list(comparisons[0].keys()) if comparisons else [],
    )
    write_csv(
        output / "bin_descriptor_comparisons_extreme.csv",
        extreme_comparisons,
        list(extreme_comparisons[0].keys()) if extreme_comparisons else [],
    )
    write_csv(
        output / "bin_descriptor_commonality.csv",
        commonality,
        list(commonality[0].keys()) if commonality else [],
    )
    write_report(
        output / "bin_stability_patterns_report.md",
        ranked,
        comparisons,
        extreme_comparisons,
        commonality,
        winner_window_kcal=args.winner_window_kcal,
        high_gap_kcal=args.high_gap_kcal,
        extreme_gap_kcal=args.extreme_gap_kcal,
    )
    print(f"Included structures: {len(ranked)}")
    print(f"Descriptors compared: {len(descriptors)}")
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
