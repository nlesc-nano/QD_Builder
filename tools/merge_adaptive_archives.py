#!/usr/bin/env python3
"""Merge adaptive minima archives without losing lineages or eligibility."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from builder.nucleation.adaptive import coarse_lineage_family, lineage_family
from builder.nucleation.pilot import (
    add_classification_provenance,
    classification_observations,
)
from builder.nucleation.search_analysis import MinimumArchive
from builder.nucleation.spec import load_nucleation_spec


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()


def _unique(values):
    unique = {}
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str, allow_nan=False)
        unique[key] = value
    return [unique[key] for key in sorted(unique)]


def _archive_rows(path: Path, arm: str):
    archive_path = path / "minima.json" if path.is_dir() else path
    data = json.loads(archive_path.read_text())
    rows = data.get(arm)
    if not isinstance(rows, dict) or not rows:
        raise ValueError(f"{archive_path} has no non-empty {arm!r} archive")
    return archive_path, rows


def _prepare(row, minimum_id, label, merge_protocol):
    value = dict(row)
    observations = []
    for observation in classification_observations(value):
        observation = dict(observation)
        observation.setdefault("archive", label)
        observation.setdefault("minimum_id", minimum_id)
        observations.append(observation)
    value["classification_history"] = observations
    value["source_archives"] = sorted(
        set(value.get("source_archives", [])) | {label}
    )
    value["source_minimum_ids"] = sorted(
        set(value.get("source_minimum_ids", [])) | {minimum_id}
    )
    value["source_protocols"] = sorted(
        set(value.get("source_protocols", []))
        | ({str(value.get("protocol"))} if value.get("protocol") else set())
    )
    value["protocol"] = merge_protocol
    value["source"] = f"merge:{label}:{minimum_id}"
    value["minimum_id"] = minimum_id
    add_classification_provenance(value)
    return value


def _merge_rows(old, new, minimum_id, merge_protocol):
    role_rank = {"primary": 0, "audit": 1}
    selected = min(
        (old, new),
        key=lambda row: (
            role_rank.get(row.get("role"), 2),
            float(row["energy_eV"]),
            row.get("structure_id", ""),
        ),
    ).copy()
    selected["minimum_id"] = minimum_id
    selected["protocol"] = merge_protocol
    selected["routes"] = sorted(
        (
            set(old.get("routes", []))
            | set(new.get("routes", []))
            | set(old.get("source_minimum_ids", []))
            | set(new.get("source_minimum_ids", []))
        )
        - {minimum_id}
    )
    selected["occupations"] = _unique(
        list(old.get("occupations", [])) + list(new.get("occupations", []))
    )
    selected["occupation_origins"] = _unique(
        list(old.get("occupation_origins", []))
        + list(new.get("occupation_origins", []))
    )
    for field in ("source_archives", "source_minimum_ids", "source_protocols"):
        selected[field] = sorted(set(old.get(field, [])) | set(new.get(field, [])))
    add_classification_provenance(selected, old, new)
    return selected


def merge_archives(sources, pack, arm="experimental"):
    source_records = []
    fingerprints = []
    for source in sources:
        archive_path, rows = _archive_rows(source, arm)
        label = source.name if source.is_dir() else source.parent.name
        fingerprint = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        source_records.append((label, archive_path, rows, fingerprint))
        fingerprints.append(
            dict(label=label, path=str(archive_path), sha256=fingerprint)
        )
    labels = [record[0] for record in source_records]
    if len(labels) != len(set(labels)):
        raise ValueError(f"source archive labels are not unique: {labels}")
    # Scientific identity is independent of local/HPC mount points.
    merge_protocol = "archive_merge_" + _digest(
        [
            {"label": item["label"], "sha256": item["sha256"]}
            for item in fingerprints
        ]
    )[:20]
    merged = {}
    aliases = {}
    input_counts = Counter()

    occurrences = {}
    for label, _, rows, _ in source_records:
        for minimum_id, row in rows.items():
            occurrences.setdefault(minimum_id, []).append(
                (
                    int(row["k"]),
                    int(row["p"]),
                    0 if row.get("role") == "primary" else 1,
                    float(row["energy_eV"]),
                    label,
                    minimum_id,
                    row,
                )
            )
            input_counts[label] += 1

    # Most continuation archives share a large imported ancestor.  Consolidate
    # those stable minimum IDs directly before doing any costly graph/geometry
    # comparison.  Each branch already compared its new minima with that common
    # ancestor, so only branch-unique IDs need cross-branch geometry matching.
    unique_candidates = []
    for original_id, records in sorted(occurrences.items()):
        if len(records) == 1:
            unique_candidates.append(records[0])
            continue
        prepared = [
            _prepare(row, original_id, label, merge_protocol)
            for _, _, _, _, label, _, row in sorted(records)
        ]
        selected = prepared[0]
        for row in prepared[1:]:
            selected = _merge_rows(selected, row, original_id, merge_protocol)
        merged[original_id] = selected
        for _, _, _, _, label, _, _ in records:
            aliases[f"{label}:{original_id}"] = original_id

    consolidation = MinimumArchive(load_nucleation_spec(pack))
    for _, _, _, _, label, original_id, original in sorted(unique_candidates):
        row = _prepare(original, original_id, label, merge_protocol)
        minimum_id, fresh = consolidation.add(
            row, row["positions"], preferred_id=original_id
        )
        aliases[f"{label}:{original_id}"] = minimum_id
        if fresh:
            # A repeated source ID that names a genuinely different geometry
            # receives the deterministic ID made by MinimumArchive.
            if minimum_id in merged:
                raise ValueError(
                    f"minimum ID collision after consolidation: {minimum_id}"
                )
            row["minimum_id"] = minimum_id
            merged[minimum_id] = row
        else:
            merged[minimum_id] = _merge_rows(
                merged[minimum_id], row, minimum_id, merge_protocol
            )

    primary = [row for row in merged.values() if row.get("role") == "primary"]
    bins = sorted({(row["k"], row["p"]) for row in primary})
    manifest = dict(
        format="adaptive_archive_merge_v1",
        protocol=merge_protocol,
        sources=fingerprints,
        input_minima=dict(sorted(input_counts.items())),
        input_total=sum(input_counts.values()),
        merged_minima=len(merged),
        geometrically_consolidated=sum(input_counts.values()) - len(merged),
        primary_minima=len(primary),
        audit_minima=len(merged) - len(primary),
        primary_coarse_families=len({coarse_lineage_family(row) for row in primary}),
        primary_exact_families=len({lineage_family(row) for row in primary}),
        primary_by_k_p={
            f"k{k}_p{p}": sum(
                row["k"] == k and row["p"] == p for row in primary
            )
            for k, p in bins
        },
        primary_coarse_families_by_k_p={
            f"k{k}_p{p}": len(
                {
                    coarse_lineage_family(row)
                    for row in primary
                    if row["k"] == k and row["p"] == p
                }
            )
            for k, p in bins
        },
        primary_exact_families_by_k_p={
            f"k{k}_p{p}": len(
                {
                    lineage_family(row)
                    for row in primary
                    if row["k"] == k and row["p"] == p
                }
            )
            for k, p in bins
        },
        multi_source_minima=sum(
            len(row.get("source_archives", [])) > 1 for row in merged.values()
        ),
        multi_policy_role_conflicts=sum(
            {"primary", "audit"}
            <= set(row.get("classification_eligibility", {}))
            for row in merged.values()
        ),
        aliases=aliases,
    )
    return {"control": {}, "experimental": merged}, manifest


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--pack",
        type=Path,
        default=ROOT / "geometry_packs/cdse_cdcl2_zb/run_gxtb.yaml",
    )
    parser.add_argument("--arm", default="experimental")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(args.source) < 2:
        parser.error("at least two --source archives are required")
    archive, manifest = merge_archives(args.source, args.pack, args.arm)
    summary = {key: value for key, value in manifest.items() if key != "aliases"}
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    _write_json(args.output / "minima.json", archive)
    _write_json(args.output / "merge_manifest.json", manifest)
    _write_json(
        args.output / "protocol.json",
        {
            "fingerprint": manifest["protocol"],
            "kind": manifest["format"],
            "sources": manifest["sources"],
        },
    )
    _write_json(
        args.output / "status.json",
        {
            "complete": True,
            "merged_minima": manifest["merged_minima"],
            "primary_minima": manifest["primary_minima"],
        },
    )
    with (args.output / "events.jsonl").open("w") as stream:
        stream.write(json.dumps({"event": "archive_merge", **summary}, sort_keys=True))
        stream.write("\n")
    print(f"[merge] wrote canonical archive to {args.output}", flush=True)


if __name__ == "__main__":
    main()
