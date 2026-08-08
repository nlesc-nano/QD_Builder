"""Write CSV / YAML / Markdown mining outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .aggregate import aggregate_groups, rate, summarize_values
from .analyze import StructureResult
from .angles import BondSample, AngleSample


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Stable union of keys
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _yaml_dump(data: Any, indent: int = 0) -> str:
    """Minimal YAML emitter (no PyYAML required)."""

    pad = "  " * indent
    if isinstance(data, dict):
        if not data:
            return pad + "{}\n"
        lines = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(_yaml_dump(value, indent + 1).rstrip("\n"))
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(value)}")
        return "\n".join(lines) + "\n"
    if isinstance(data, list):
        if not data:
            return pad + "[]\n"
        lines = []
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                nested = _yaml_dump(item, indent + 1).rstrip("\n")
                for line in nested.splitlines():
                    lines.append(line)
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    return pad + _yaml_scalar(data) + "\n"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if any(c in text for c in ":#{}[],&*?|>!%@`'\""):
        return json.dumps(text)
    return text


def build_summary(
    results: Sequence[StructureResult],
    *,
    cutoffs: Mapping[str, float],
) -> Dict[str, Any]:
    clean = [r for r in results if r.clean]
    all_with_geo = [r for r in results if r.has_last]
    quarantine = [r for r in results if r.has_last and not r.clean]
    missing = [r for r in results if not r.has_last]

    bond_keys = []
    for r in clean:
        for b in r.bond_samples:
            bond_keys.append(
                (
                    f"{b.pair_type}|cn_cd={b.cn_cd}|cn_other={b.cn_other}|{b.k_window}",
                    b.length,
                )
            )
            bond_keys.append((f"{b.pair_type}|all|{b.k_window}", b.length))
            bond_keys.append((f"{b.pair_type}|all|all_k", b.length))

    angle_keys = []
    for r in clean:
        for a in r.angle_samples:
            angle_keys.append(
                (
                    f"{a.element}|cn={a.cn}|sig={a.neighbor_signature}|pair={a.neighbor_pair}",
                    a.angle_deg,
                )
            )
            angle_keys.append(
                (f"{a.element}|cn={a.cn}|pair={a.neighbor_pair}", a.angle_deg)
            )
            if a.role_signature and a.neighbor_role_pair:
                angle_keys.append(
                    (
                        f"{a.element}|cn={a.cn}|roles={a.role_signature}|"
                        f"pair={a.neighbor_role_pair}",
                        a.angle_deg,
                    )
                )

    improper_keys = []
    proper_keys = []
    for r in clean:
        for sample in r.improper_dihedral_samples:
            improper_keys.append(
                (
                    f"{sample.element}|cn={sample.cn}|"
                    f"sig={sample.neighbor_signature}",
                    sample.improper_deg,
                )
            )
        for sample in r.proper_dihedral_samples:
            proper_keys.append(
                (sample.atom_signature, sample.dihedral_deg)
            )

    # Same-species distance samples (all with geometry, for cutoff tuning)
    same_sp = []
    for r in all_with_geo:
        for el, d in r.same_species_samples:
            if d < 6.0:  # keep chemically relevant range
                same_sp.append((el, d))

    same_agg = aggregate_groups(
        (f"{el}_all_lt6A", d) for el, d in same_sp
    )

    n_clean = len(clean)
    n_inorg_connected = sum(
        1
        for r in clean
        if int(r.structure_row.get("inorganic_connected", 0)) == 1
    )
    n_cd_cn2 = sum(
        1 for r in clean if int(r.structure_row.get("n_cd_cn2", 0) or 0) > 0
    )
    n_cd_cn2_linear = sum(
        1
        for r in clean
        if r.structure_row.get("cd_cn2_all_linear") is True
    )
    n_se_cn5plus = sum(
        1 for r in clean if int(r.structure_row.get("max_cn_se", 0) or 0) >= 5
    )

    # Recommended construction defaults from broad clean aggregates
    bond_agg = aggregate_groups(bond_keys)
    angle_agg = aggregate_groups(angle_keys)
    improper_agg = aggregate_groups(improper_keys)
    proper_agg = aggregate_groups(proper_keys)

    def pick(key: str) -> Optional[Dict[str, object]]:
        return bond_agg.get(key) or angle_agg.get(key)

    recommendations = {
        "bond_lengths_A": {
            "CdSe": pick("CdSe|all|all_k"),
            "CdCl_terminal": pick("CdCl_terminal|all|all_k"),
            "CdCl_bridge": pick("CdCl_bridge|all|all_k"),
        },
        "angles_deg": {
            "Cd_cn2_Se-Se": pick("Cd|cn=2|pair=Se-Se"),
            "Cd_cn2_Cl-Se": pick("Cd|cn=2|pair=Cl-Se"),
            "Cd_cn2_Cl-Cl": pick("Cd|cn=2|pair=Cl-Cl"),
            "Se_cn2_Cd-Cd": pick("Se|cn=2|pair=Cd-Cd"),
            "Cl_cn2_Cd-Cd": pick("Cl|cn=2|pair=Cd-Cd"),
            "Se_cn3_Cd-Cd": pick("Se|cn=3|pair=Cd-Cd"),
            "Cd_cn3_mixed": {
                k: v
                for k, v in angle_agg.items()
                if k.startswith("Cd|cn=3|")
            },
        },
        # Evidence only. The builder promotes only explicitly reviewed
        # improper entries; multimodal proper torsions are never auto-promoted.
        "improper_planarity_deg": {
            key: value
            for key, value in improper_agg.items()
            if key.startswith("Cd|cn=3|")
        },
        "hard_rules_supported_by_clean_set": {
            "inorganic_CdSe_connected_fraction": rate(
                n_inorg_connected, n_clean
            ),
            "cd_cn2_all_linear_fraction_among_structures_with_cd_cn2": rate(
                n_cd_cn2_linear, n_cd_cn2
            ),
            "fraction_with_max_cn_se_ge_5": rate(n_se_cn5plus, n_clean),
            "homonuclear_quarantine_count": len(quarantine),
        },
    }

    return {
        "cutoffs_A": dict(cutoffs),
        "counts": {
            "jobs_total": len(results),
            "with_geometry": len(all_with_geo),
            "clean": n_clean,
            "quarantine": len(quarantine),
            "missing_trajectory": len(missing),
        },
        "k_distribution_clean": dict(
            Counter(str(r.job.k) for r in clean)
        ),
        "k_distribution_all_with_geometry": dict(
            Counter(str(r.job.k) for r in all_with_geo)
        ),
        "recommendations": recommendations,
        "bond_length_tables": bond_agg,
        "angle_tables": angle_agg,
        "improper_dihedral_tables": improper_agg,
        "proper_dihedral_tables": proper_agg,
        "same_species_distance_stats_lt6A": same_agg,
    }


def write_report_md(
    path: Path,
    summary: Mapping[str, Any],
    results: Sequence[StructureResult],
) -> None:
    clean = [r for r in results if r.clean]
    lines: List[str] = []
    lines.append("# CdSe DFT geometry mine (standalone)")
    lines.append("")
    lines.append("Independent of `analyze_cp2k_results.py`. Energies not used.")
    lines.append("")
    counts = summary["counts"]
    lines.append("## Counts")
    lines.append("")
    for key, value in counts.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("## Cutoffs (Å)")
    lines.append("")
    for key, value in summary["cutoffs_A"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Recommended construction defaults (clean set medians)")
    lines.append("")
    rec = summary["recommendations"]
    lines.append("### Bond lengths")
    lines.append("")
    for name, stats in rec["bond_lengths_A"].items():
        if not stats or not stats.get("n"):
            lines.append(f"- **{name}**: insufficient samples")
        else:
            lines.append(
                f"- **{name}**: recommended **{stats['recommended']}** Å "
                f"(n={stats['n']}, mean={stats['mean']}, "
                f"p10–p90={stats['p10']}–{stats['p90']})"
            )
    lines.append("")
    lines.append("### Key angles")
    lines.append("")
    for name, stats in rec["angles_deg"].items():
        if name == "Cd_cn3_mixed":
            continue
        if not stats or not stats.get("n"):
            lines.append(f"- **{name}**: insufficient samples")
        else:
            lines.append(
                f"- **{name}**: recommended **{stats['recommended']}** deg "
                f"(n={stats['n']}, mean={stats['mean']}, "
                f"p10–p90={stats['p10']}–{stats['p90']})"
            )
    lines.append("")
    lines.append("### Hard-rule support rates (clean set)")
    lines.append("")
    for key, value in rec["hard_rules_supported_by_clean_set"].items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("### Cd CN3 improper planarity evidence")
    lines.append("")
    improper = rec.get("improper_planarity_deg", {})
    if not improper:
        lines.append("- (no samples)")
    else:
        for key, stats in sorted(improper.items()):
            lines.append(
                f"- **{key}**: median deviation **{stats['recommended']}** deg "
                f"(n={stats['n']}, p10–p90={stats['p10']}–{stats['p90']})"
            )
    lines.append("")
    lines.append(
        "Ordinary proper torsions are reported as evidence only; their "
        "multimodal distributions are not construction constraints."
    )
    lines.append("")
    lines.append("## Quarantine reasons")
    lines.append("")
    reason_counts: Counter = Counter()
    for r in results:
        if r.clean or not r.has_last:
            continue
        for reason in r.quarantine_reasons:
            head = reason.split(":")[0]
            reason_counts[head] += 1
    if not reason_counts:
        lines.append("- (none)")
    else:
        for reason, count in reason_counts.most_common():
            lines.append(f"- **{reason}**: {count}")
    lines.append("")
    lines.append("## k distribution (clean)")
    lines.append("")
    for key, value in sorted(
        summary["k_distribution_clean"].items(),
        key=lambda kv: (kv[0] == "None", kv[0]),
    ):
        lines.append(f"- k={key}: {value}")
    lines.append("")
    lines.append("## Notes for builder")
    lines.append("")
    lines.append(
        "1. Prefer **Cd–Se inorganic connectivity** as a hard filter "
        f"(clean fraction connected = "
        f"{rec['hard_rules_supported_by_clean_set']['inorganic_CdSe_connected_fraction']})."
    )
    lines.append(
        "2. Reject structures with Cd–Cd / Se–Se / Cl–Cl short contacts "
        "(see quarantine.csv)."
    )
    lines.append(
        "3. Use angle/length tables only where **n** is adequate; "
        "small-n classes are marked insufficient above."
    )
    lines.append(
        "4. Se max CN ≥ 5 appears in a non-trivial clean fraction; "
        "consider allowing Se CN5 at small k after review."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_all_outputs(
    output_dir: Path,
    results: Sequence[StructureResult],
    *,
    cutoffs: Mapping[str, float],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = []
    structures = []
    quarantine = []
    bond_rows = []
    angle_rows = []
    improper_rows = []
    proper_rows = []
    same_rows = []
    start_rows = []

    for r in results:
        inventory.append(
            {
                "root": r.job.root_label,
                "structure_id": r.job.structure_id,
                "job_dir": str(r.job.job_dir),
                "traj_path": str(r.job.traj_path),
                "start_path": str(r.job.start_path) if r.job.start_path else "",
                "k": r.job.k,
                "p": r.job.p,
                "n_frames": r.n_frames,
                "has_last": int(r.has_last),
                "clean": int(r.clean),
            }
        )
        if r.structure_row:
            structures.append(r.structure_row)
        if r.has_last and not r.clean:
            quarantine.append(
                {
                    "root": r.job.root_label,
                    "structure_id": r.job.structure_id,
                    "k": r.job.k,
                    "p": r.job.p,
                    "job_dir": str(r.job.job_dir),
                    "reasons": ";".join(r.quarantine_reasons),
                    **{
                        k: r.structure_row.get(k)
                        for k in (
                            "n_cd_cd",
                            "n_se_se",
                            "n_cl_cl",
                            "formula_msg",
                            "inorganic_connected",
                        )
                    },
                }
            )
        if r.clean:
            for b in r.bond_samples:
                bond_rows.append(
                    {
                        "structure_id": r.job.structure_id,
                        "k": r.job.k,
                        "p": r.job.p,
                        "k_window": b.k_window,
                        "pair_type": b.pair_type,
                        "length": b.length,
                        "cn_cd": b.cn_cd,
                        "cn_other": b.cn_other,
                    }
                )
            for a in r.angle_samples:
                angle_rows.append(
                    {
                        "structure_id": r.job.structure_id,
                        "k": r.job.k,
                        "p": r.job.p,
                        "element": a.element,
                        "cn": a.cn,
                        "neighbor_signature": a.neighbor_signature,
                        "neighbor_pair": a.neighbor_pair,
                        "role_signature": a.role_signature,
                        "neighbor_role_pair": a.neighbor_role_pair,
                        "angle_deg": a.angle_deg,
                    }
                )
            for sample in r.improper_dihedral_samples:
                improper_rows.append(
                    {
                        "structure_id": r.job.structure_id,
                        "k": r.job.k,
                        "p": r.job.p,
                        "element": sample.element,
                        "cn": sample.cn,
                        "neighbor_signature": sample.neighbor_signature,
                        "improper_deg": sample.improper_deg,
                    }
                )
            for sample in r.proper_dihedral_samples:
                proper_rows.append(
                    {
                        "structure_id": r.job.structure_id,
                        "k": r.job.k,
                        "p": r.job.p,
                        "atom_signature": sample.atom_signature,
                        "dihedral_deg": sample.dihedral_deg,
                    }
                )
        for el, d in r.same_species_samples:
            same_rows.append(
                {
                    "structure_id": r.job.structure_id,
                    "k": r.job.k,
                    "p": r.job.p,
                    "element": el,
                    "distance": d,
                    "clean": int(r.clean),
                }
            )
        if r.start_row is not None:
            start_rows.append(r.start_row)

    _write_csv(output_dir / "inventory.csv", inventory)
    _write_csv(output_dir / "structures_geometry.csv", structures)
    _write_csv(output_dir / "quarantine.csv", quarantine)
    _write_csv(output_dir / "samples_bonds.csv", bond_rows)
    _write_csv(output_dir / "samples_angles.csv", angle_rows)
    _write_csv(
        output_dir / "samples_improper_dihedrals.csv", improper_rows
    )
    _write_csv(output_dir / "samples_proper_dihedrals.csv", proper_rows)
    _write_csv(output_dir / "samples_same_species.csv", same_rows)
    if start_rows:
        _write_csv(output_dir / "structures_start_geometry.csv", start_rows)

    summary = build_summary(results, cutoffs=cutoffs)
    (output_dir / "geometry_summary.yaml").write_text(
        _yaml_dump(summary), encoding="utf-8"
    )
    # Also JSON for easy loading
    (output_dir / "geometry_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report_md(output_dir / "geometry_report.md", summary, results)
    return summary
