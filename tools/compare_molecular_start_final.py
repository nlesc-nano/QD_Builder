#!/usr/bin/env python3
"""Compare construction (start) molecular graphs to post-DFT (final) graphs.

Use this after a DFT round on lattice-free molecular isomers to measure which
collapse-risk annotations predict graph change.  Results guide enabling:

  reject_closable_terminal_cd2
  require_bridge_maximal
  forbid_cdse_cn_pairs

in ``geometry_packs/cdse_cdcl2_molecular.yaml`` — default remains off.

Example
-------

    python tools/compare_molecular_start_final.py \\
      --annotations path/to/annotations.csv \\
      --start-dir path/to/molecular_map \\
      --final-root path/to/dft_jobs \\
      --output path/to/start_final_report.csv

``annotations.csv`` is written by ``write_molecular_map``.  Each final job
directory is matched by structure_id substring (same convention as
``geometry_mine`` inventory).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Repo layout: tools/ → project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from dft_geometry_mine.bonds import BondCutoffs, analyze_frame  # noqa: E402
from dft_geometry_mine.xyz_io import read_xyz_frames  # noqa: E402

from builder.graph_canon import canonical_form  # noqa: E402


def _load_xyz_frame(path: Path):
    frames = read_xyz_frames(path)
    if not frames:
        raise ValueError(f"no frames in {path}")
    return frames[-1]


def _certificate_from_frame(frame, cutoffs: BondCutoffs) -> Tuple[object, ...]:
    analysis = analyze_frame(frame, cutoffs)
    labels = list(analysis.symbols)
    edges = [
        (i, j, "bond")
        for i, j, _ptype, _length in analysis.edges
    ]
    return canonical_form(labels, edges).certificate


def _find_final_xyz(job_dir: Path) -> Optional[Path]:
    if not job_dir.is_dir():
        return None
    preferred = [
        "cdse_opt-pos-1.xyz",
        "opt-pos-1.xyz",
        "final.xyz",
        "geometry.xyz",
    ]
    for name in preferred:
        candidate = job_dir / name
        if candidate.is_file():
            return candidate
    cands = sorted(job_dir.glob("**/*pos*.xyz")) + sorted(
        job_dir.glob("**/*.xyz")
    )
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_size)


def _index_jobs_by_structure_id(
    final_root: Path,
) -> Dict[str, Path]:
    """Map structure_id → job directory (last match wins)."""

    index: Dict[str, Path] = {}
    for path in final_root.rglob("*"):
        if not path.is_dir():
            continue
        name = path.name
        # Typical: k002_p004_mol0007 or with hash suffix
        if "_mol" not in name and "k0" not in name:
            continue
        # Prefer directory names that contain mol ids
        for part in name.replace("__", "_").split("_"):
            pass
        index[name] = path
        # Also index bare structure id prefix before __hash
        if "__" in name:
            index[name.split("__", 1)[0]] = path
    return index


def _match_job(
    structure_id: str, job_index: Dict[str, Path]
) -> Optional[Path]:
    if structure_id in job_index:
        return job_index[structure_id]
    for key, path in job_index.items():
        if structure_id in key or key in structure_id:
            return path
    return None


def _load_annotations(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_start_xyz(start_dir: Path, k: str, p: str, structure_id: str) -> Optional[Path]:
    # write_molecular_map layout: k00X/p00Y/structure_id.xyz
    try:
        kk, pp = int(k), int(p)
    except ValueError:
        return None
    candidate = (
        start_dir / f"k{kk:03d}" / f"p{pp:03d}" / f"{structure_id}.xyz"
    )
    if candidate.is_file():
        return candidate
    # Fallback search
    hits = list(start_dir.rglob(f"{structure_id}.xyz"))
    return hits[0] if hits else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="annotations.csv from write_molecular_map",
    )
    parser.add_argument(
        "--start-dir",
        type=Path,
        required=True,
        help="molecular map output root (contains k***/p***/.xyz)",
    )
    parser.add_argument(
        "--final-root",
        type=Path,
        required=True,
        help="root of DFT job directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="per-structure comparison CSV",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="optional contingency summary JSON",
    )
    parser.add_argument("--cd-se-cutoff", type=float, default=3.25)
    parser.add_argument("--cd-cl-cutoff", type=float, default=3.10)
    args = parser.parse_args(argv)

    cutoffs = BondCutoffs(
        cd_se=float(args.cd_se_cutoff),
        cd_cl=float(args.cd_cl_cutoff),
    )
    rows = _load_annotations(args.annotations)
    job_index = _index_jobs_by_structure_id(args.final_root)

    out_fields = [
        "k",
        "p",
        "structure_id",
        "status",
        "start_edges",
        "final_edges",
        "n_cd2",
        "n_closable_terminal_cd2",
        "n_cl2se1_near_cd2",
        "n_unsaturated_bridge_candidates",
        "mean_cd_cn",
        "cdse_cn_pairs",
        "final_job",
    ]
    results: List[Dict[str, object]] = []
    contingency: Dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        structure_id = row["structure_id"]
        k, p = row.get("k", ""), row.get("p", "")
        start_xyz = _find_start_xyz(args.start_dir, k, p, structure_id)
        job = _match_job(structure_id, job_index)
        record: Dict[str, object] = {
            "k": k,
            "p": p,
            "structure_id": structure_id,
            "status": "missing",
            "start_edges": "",
            "final_edges": "",
            "n_cd2": row.get("n_cd2", ""),
            "n_closable_terminal_cd2": row.get(
                "n_closable_terminal_cd2", ""
            ),
            "n_cl2se1_near_cd2": row.get("n_cl2se1_near_cd2", ""),
            "n_unsaturated_bridge_candidates": row.get(
                "n_unsaturated_bridge_candidates", ""
            ),
            "mean_cd_cn": row.get("mean_cd_cn", ""),
            "cdse_cn_pairs": row.get("cdse_cn_pairs", ""),
            "final_job": str(job) if job else "",
        }
        if start_xyz is None:
            record["status"] = "missing_start"
            results.append(record)
            continue
        if job is None:
            record["status"] = "missing_final"
            results.append(record)
            continue
        final_xyz = _find_final_xyz(job)
        if final_xyz is None:
            record["status"] = "missing_final_xyz"
            results.append(record)
            continue
        try:
            start_frame = _load_xyz_frame(start_xyz)
            final_frame = _load_xyz_frame(final_xyz)
            start_cert = _certificate_from_frame(start_frame, cutoffs)
            final_cert = _certificate_from_frame(final_frame, cutoffs)
            start_g = analyze_frame(start_frame, cutoffs)
            final_g = analyze_frame(final_frame, cutoffs)
            record["start_edges"] = len(start_g.edges)
            record["final_edges"] = len(final_g.edges)
            if start_cert == final_cert:
                status = "retained"
            elif len(final_g.edges) > len(start_g.edges):
                status = "closed_extra_edges"
            elif len(final_g.edges) < len(start_g.edges):
                status = "opened_lost_edges"
            else:
                status = "rearranged"
            if start_g.has_homonuclear_contact or final_g.has_homonuclear_contact:
                status = "homonuclear_contact"
            record["status"] = status
        except Exception as exc:  # noqa: BLE001 — report per row
            record["status"] = f"error:{exc}"
        results.append(record)

        # Contingency: annotation bins vs retained/closed
        try:
            n_cd2 = int(float(row.get("n_cd2") or 0))
            closable = int(float(row.get("n_closable_terminal_cd2") or 0))
            cl2se1 = int(float(row.get("n_cl2se1_near_cd2") or 0))
        except ValueError:
            n_cd2 = closable = cl2se1 = 0
        outcome = str(record["status"])
        contingency["all"][outcome] += 1
        contingency[f"n_cd2>={2 if n_cd2 >= 2 else n_cd2}"][outcome] += 1
        contingency[f"closable={1 if closable > 0 else 0}"][outcome] += 1
        contingency[f"cl2se1_near={1 if cl2se1 > 0 else 0}"][outcome] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        for record in results:
            writer.writerow(record)

    summary = {
        bin_name: dict(counter)
        for bin_name, counter in sorted(contingency.items())
    }
    summary_path = args.summary_json or args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(f"wrote {summary_path}")
    print("outcome counts:", dict(contingency["all"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
