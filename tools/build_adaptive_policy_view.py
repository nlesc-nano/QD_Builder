#!/usr/bin/env python3
"""Create a non-destructive classification view of an adaptive archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from builder.nucleation.pilot import add_classification_provenance


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def build_policy_view(source, *, k, ps, allowed_violations, mode):
    source_path = source / "minima.json" if source.is_dir() else source
    data = json.loads(source_path.read_text())
    rows = data.get("experimental")
    if not isinstance(rows, dict) or not rows:
        raise ValueError(f"{source_path} has no experimental minima")
    output_rows = {}
    promoted = Counter()
    for minimum_id, original in sorted(rows.items()):
        row = dict(original)
        violations = list(row.get("violations", []))
        categories = {value.split(":", 1)[0] for value in violations}
        eligible = (
            row.get("role") == "audit"
            and int(row.get("k", -1)) == k
            and int(row.get("p", -1)) in ps
            and bool(categories)
            and categories <= allowed_violations
        )
        if eligible:
            strict = dict(row)
            row["role"] = "primary"
            row["violations"] = []
            row["classification_mode"] = mode
            row["policy_view_origin_role"] = strict.get("role")
            row["policy_view_relaxed_violations"] = violations
            row.pop("classification_history", None)
            row.pop("classification_eligibility", None)
            add_classification_provenance(row, strict)
            promoted[int(row["p"])] += 1
        output_rows[minimum_id] = row
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    policy = dict(
        kind="adaptive_archive_policy_view_v1",
        source=str(source_path.resolve()),
        source_sha256=source_hash,
        k=k,
        p=sorted(ps),
        allowed_violations=sorted(allowed_violations),
        classification_mode=mode,
        promoted_by_p={str(p): promoted[p] for p in sorted(promoted)},
        promoted_total=sum(promoted.values()),
    )
    policy["fingerprint"] = "policy_view_" + hashlib.sha256(
        json.dumps(
            {key: value for key, value in policy.items() if key != "source"},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:20]
    for row in output_rows.values():
        row.setdefault("source_protocols", [])
        if row.get("protocol"):
            row["source_protocols"] = sorted(
                set(row["source_protocols"]) | {str(row["protocol"])}
            )
        row["protocol"] = policy["fingerprint"]
    return {"control": {}, "experimental": output_rows}, policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--p", required=True, nargs="+", type=int)
    parser.add_argument(
        "--allow-violation",
        action="append",
        default=[],
        help="violation category to ignore; may be repeated",
    )
    parser.add_argument("--classification-mode", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.allow_violation:
        parser.error("at least one --allow-violation is required")
    archive, policy = build_policy_view(
        args.source,
        k=args.k,
        ps=set(args.p),
        allowed_violations=set(args.allow_violation),
        mode=args.classification_mode,
    )
    print(json.dumps(policy, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    _write_json(args.output / "minima.json", archive)
    _write_json(args.output / "protocol.json", policy)
    _write_json(
        args.output / "status.json",
        {
            "complete": True,
            "promoted_total": policy["promoted_total"],
            "promoted_by_p": policy["promoted_by_p"],
        },
    )
    with (args.output / "events.jsonl").open("w") as stream:
        stream.write(json.dumps({"event": "policy_view", **policy}, sort_keys=True))
        stream.write("\n")
    print(f"[policy] wrote classification view to {args.output}", flush=True)


if __name__ == "__main__":
    main()
