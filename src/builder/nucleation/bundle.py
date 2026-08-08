from __future__ import annotations

from .graph_ops import *  # private names via __all__

from .surface import *  # private names via __all__

from .scoring import *  # private names via __all__

from .lattice import *  # private names via __all__

from .types import *  # private names via __all__

import json
import shutil
import textwrap
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from ..io_utils import write_xyz
from ..nc_types import NucleationSpec
from .types import (
    AtomRecord,
    ClusterRecord,
    NucleationRegistry,
    NucleationResult,
    SweepAudit,
)

def registry_to_dict(registry: NucleationRegistry) -> Dict[str, object]:
    """Serialize a retained or discarded registry."""

    return {
        str(k): {
            str(p): [_record_to_dict(record) for record in records]
            for p, records in sorted(bins.items())
        }
        for k, bins in sorted(registry.items())
    }


def write_nucleation_json(
    registry: NucleationRegistry,
    path: str | Path,
    *,
    indent: int = 2,
) -> None:
    """Write one registry as deterministic JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(registry_to_dict(registry), indent=indent, sort_keys=True) + "\n"
    )


def nucleation_result_to_dict(result: NucleationResult) -> Dict[str, object]:
    """Serialize the rigid-lattice nucleation bundle schema."""

    return {
        "schema_version": 13,
        "coordinates": {
            "k": "core monomer count",
            "p": "total attached precursor unit count",
        },
        "construction_geometry": {
            "mode": "construction_native_plus_slot_filtered_surface",
            "reference_bond_length": result.reference_bond_length,
            "growth_coordinates": "construction_native",
            "surface_preconditioning": "retained_only_final_cn_geometry_projection",
        },
        "graph_rules": _json_value(result.graph_rules),
        "geometry_rules": _json_value(result.geometry_rules),
        "selection": {
            "primary": "feasible-first minimum coordination",
            "secondary": "maximum bond count",
            "tertiary": "minimum severe maximum-CN deficits",
            "quaternary": "prefer shared vacant CIF-site bridges on ties",
            "surface_gate": (
                "saturated Cd graph neighbors remain its nearest compatible "
                "neighbors after final-CN projection"
            ),
            "symmetry": "element-and-bond graph isomorphism",
            "traversal": (
                "path-independent inorganic-skeleton DAG; stripping is "
                "validation-only"
            ),
            "higher_k_pruning": (
                "surface-valid locally dominated bridge layers are omitted "
                "for k > 2"
            ),
        },
        "completeness": _json_value(result.completeness),
        "registry": registry_to_dict(result.registry),
        "discarded_registry": registry_to_dict(result.discarded_registry),
        "discarded_counts": {
            str(k): {str(p): count for p, count in sorted(bins.items())}
            for k, bins in sorted(result.discarded_counts.items())
        },
        "sweep_audit": [
            {
                "k": audit.k,
                "operation": audit.operation,
                "p_from": audit.p_from,
                "p_to": audit.p_to,
                "source_count": audit.source_count,
                "raw_count": audit.raw_count,
                "valid_count": audit.valid_count,
                "symmetry_duplicate_count": audit.symmetry_duplicate_count,
                "invalid_reasons": dict(sorted(audit.invalid_reasons.items())),
                "stage_counts": dict(sorted(audit.stage_counts.items())),
            }
            for audit in result.sweep_audit
        ],
    }


def write_nucleation_bundle(
    result: NucleationResult,
    output_directory: str | Path,
) -> Path:
    """Write retained/discarded XYZ trees, JSON, and a detailed CN audit log."""

    root = Path(output_directory)
    if root.exists() and not root.is_dir():
        raise ValueError(f"nucleation output must be a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    structures = root / "structures"
    if structures.exists():
        shutil.rmtree(structures)

    # Discarded XYZ trees are only exported for the small-k survey (k<=2).
    # Higher-k discarded counts remain in registry.json only.
    discarded_xyz_through_k = 2
    for status, registry in (
        ("retained", result.registry),
        ("discarded", result.discarded_registry),
    ):
        for k, bins in sorted(registry.items()):
            if status == "discarded" and k > discarded_xyz_through_k:
                continue
            for p, records in sorted(bins.items()):
                directory = structures / f"k{k:03d}" / f"p{p:03d}" / status
                directory.mkdir(parents=True, exist_ok=True)
                for record in records:
                    construction_path = (
                        directory
                        / f"{record.structure_id}_construction_native.xyz"
                    )
                    record.metadata["construction_native_xyz_path"] = str(
                        construction_path.relative_to(root)
                    )
                    write_xyz(
                        str(construction_path),
                        record.symbols,
                        record.coordinates,
                        comment=(
                            f"{_formula(record.symbols)}_construction_native_"
                            f"graph_ranked_bridges_{record.metadata.get('bridge_count', 0)}"
                        ),
                    )
                    if (
                        status == "retained"
                        or record.metadata.get("surface_selection_rejected", False)
                    ):
                        projection_valid = bool(
                            record.metadata.get("surface_geometry", {}).get(
                                "projection_valid", False
                            )
                        )
                        surface_suffix = (
                            "surface"
                            if projection_valid
                            else "surface_rejected"
                        )
                        surface_path = (
                            directory / f"{record.structure_id}_{surface_suffix}.xyz"
                        )
                        record.metadata["surface_xyz_path"] = str(
                            surface_path.relative_to(root)
                        )
                        write_xyz(
                            str(surface_path),
                            record.symbols,
                            record.surface_coordinates,
                            comment=(
                                f"{_formula(record.symbols)}_surface_projected_"
                                f"valid_{str(projection_valid).lower()}"
                            ),
                        )

    (root / "registry.json").write_text(
        json.dumps(nucleation_result_to_dict(result), indent=2, sort_keys=True)
        + "\n"
    )
    (root / "nucleation.log").write_text(_render_log(result))
    return root



def _ascii_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    right_align: Sequence[int] = (),
    max_widths: Optional[Mapping[int, int]] = None,
) -> List[str]:
    """Render a deterministic, wrapped, terminal-safe ASCII table."""

    string_rows = [[str(cell) for cell in row] for row in rows]
    widths: List[int] = []
    for column, header in enumerate(headers):
        content_width = max(
            [len(str(header))]
            + [
                len(row[column]) if column < len(row) else 0
                for row in string_rows
            ]
        )
        if max_widths and column in max_widths:
            content_width = min(content_width, max_widths[column])
        widths.append(max(1, content_width))

    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def wrapped_lines(row: Sequence[str]) -> List[str]:
        cells: List[List[str]] = []
        for column, width in enumerate(widths):
            value = row[column] if column < len(row) else ""
            cells.append(
                textwrap.wrap(
                    value,
                    width=width,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                or [""]
            )
        rendered: List[str] = []
        for line_index in range(max(len(cell) for cell in cells)):
            parts: List[str] = []
            for column, (cell, width) in enumerate(zip(cells, widths)):
                value = cell[line_index] if line_index < len(cell) else ""
                parts.append(
                    value.rjust(width)
                    if column in right_align
                    else value.ljust(width)
                )
            rendered.append("| " + " | ".join(parts) + " |")
        return rendered

    output = [separator]
    output.extend(wrapped_lines([str(header) for header in headers]))
    output.append(separator)
    for row in string_rows:
        output.extend(wrapped_lines(row))
    output.append(separator)
    return output


def _transition_label(audit: SweepAudit) -> Tuple[str, str]:
    if audit.operation in {"core_growth", "core_skeleton_growth"}:
        return (
            f"k{audit.k} p{audit.p_from}",
            f"k{audit.k + 1} p{audit.p_to}",
        )
    return (
        f"k{audit.k} p{audit.p_from}",
        f"k{audit.k} p{audit.p_to}",
    )


def _cn_values_text(record: ClusterRecord) -> str:
    values = record.metadata.get("coordination_by_element", {})
    return " ".join(
        f"{symbol}[{','.join(str(value) for value in cn_values)}]"
        for symbol, cn_values in sorted(values.items())
    )


def _cn_histogram_text(record: ClusterRecord) -> str:
    histograms = record.metadata.get("coordination_histograms", {})
    return " ".join(
        f"{symbol}["
        + ",".join(
            f"CN{value}x{count}"
            for value, count in sorted(
                histogram.items(), key=lambda item: int(item[0])
            )
        )
        + "]"
        for symbol, histogram in sorted(histograms.items())
    )


def _render_log(result: NucleationResult) -> str:
    """Render a complete human-readable nucleation audit as ASCII tables."""

    lines = [
        "QD BUILDER NUCLEATION AUDIT",
        "============================",
        "",
        "RUN CONFIGURATION",
    ]
    min_cn = ", ".join(
        f"{symbol}={minimum}"
        for symbol, minimum in sorted(
            dict(result.graph_rules.get("min_cn", {})).items()
        )
    )
    max_cn = ", ".join(
        f"{symbol}={maximum}"
        for symbol, maximum in sorted(
            dict(result.graph_rules.get("max_cn", {})).items()
        )
    )
    allowed = ", ".join(
        "-".join(pair)
        for pair in result.graph_rules.get("allowed_bonds", [])
    )
    bridge_text = "; ".join(
        f"{ligand}: host={rule.get('host')}, "
        f"shared={rule.get('shared_neighbor')}, "
        f"angle={float(rule.get('surface_angle_deg', 90.0)):.1f} deg"
        for ligand, rule in sorted(
            dict(result.graph_rules.get("bridging", {})).items()
        )
    ) or "disabled"
    geometry_parts: List[str] = []
    for symbol, rules in sorted(
        dict(result.geometry_rules.get("by_cn", {})).items()
    ):
        geometry_parts.append(
            f"{symbol} "
            + ", ".join(
                f"{coordination}={template}"
                for coordination, template in sorted(rules.items())
            )
        )
    for symbol, template in sorted(
        dict(result.geometry_rules.get("all", {})).items()
    ):
        geometry_parts.append(f"{symbol} all={template}")
    geometry_text = "; ".join(geometry_parts)
    lines.extend(
        _ascii_table(
            ("Setting", "Value"),
            (
                (
                    "Growth geometry",
                    "Exact core-CIF virtual sites (construction-native coordinates)",
                ),
                (
                    "Surface geometry",
                    "Retained-only final-CN projection; graph unchanged",
                ),
                ("Reference bond length", f"{result.reference_bond_length:.8f} A"),
                ("Minimum CN", min_cn),
                ("Maximum CN", max_cn),
                ("Allowed bonds", allowed),
                ("Latent bridging", bridge_text),
                (
                    "Traversal",
                    "Merged inorganic-skeleton DAG; stripping validates lineage only",
                ),
                (
                    "Ligand enumeration",
                    "Automorphism-orbit representatives; Raw remains theoretical C(N,L)",
                ),
                ("Geometry templates", geometry_text),
                ("Total CN", "Sum of all graph-node degrees"),
                ("Bond count", "Total CN / 2"),
            ),
            max_widths={1: 80},
        )
    )

    lines.extend(["", "SWEEP SUMMARY"])
    sweep_rows: List[Tuple[object, ...]] = []
    rejection_rows: List[Tuple[object, ...]] = []
    for audit in result.sweep_audit:
        source, target = _transition_label(audit)
        rejected = sum(audit.invalid_reasons.values())
        sweep_rows.append(
            (
                audit.operation,
                source,
                target,
                audit.source_count,
                audit.raw_count,
                audit.stage_counts.get("orbit_representatives", 0),
                audit.stage_counts.get("capacity_pruned", 0),
                audit.stage_counts.get(
                    "symmetry_pruned_before_embedding", 0
                ),
                audit.stage_counts.get("embedded", audit.raw_count),
                audit.stage_counts.get("bridge_variants", 0),
                audit.stage_counts.get("bridge_search_states", 0),
                audit.stage_counts.get(
                    "dominated_bridge_variants_pruned", 0
                ),
                audit.stage_counts.get("parent_routes_merged", 0),
                audit.stage_counts.get("cache_hits", 0),
                audit.valid_count,
                audit.symmetry_duplicate_count,
                rejected,
            )
        )
        for reason, count in sorted(audit.invalid_reasons.items()):
            rejection_rows.append(
                (audit.operation, f"{source} -> {target}", reason, count)
            )
    for k, bins in sorted(result.discarded_registry.items()):
        for p, records in sorted(bins.items()):
            surface_rejected = sum(
                record.selection_reason == "surface_slot_conflict"
                for record in records
            )
            if surface_rejected:
                rejection_rows.append(
                    (
                        "selection",
                        f"k{k} p{p}",
                        "surface_slot_conflict",
                        surface_rejected,
                    )
                )
    lines.extend(
        _ascii_table(
            (
                "Operation",
                "From",
                "To",
                "Sources",
                "Raw",
                "Orbit reps",
                "Cap prune",
                "Pre-sym",
                "Embedded",
                "Bridge variants",
                "Bridge search",
                "Dominated",
                "Routes merged",
                "Cache",
                "Unique",
                "Sym dup",
                "Rejected",
            ),
            sweep_rows,
            right_align=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
        )
    )

    passivation_audits = {
        (audit.k, audit.p_from): audit
        for audit in result.sweep_audit
        if audit.operation == "skeleton_passivation"
    }
    all_k = sorted(
        set(result.registry)
        | set(result.discarded_registry)
        | set(result.discarded_counts)
    )
    lines.extend(["", "BIN SUMMARY"])
    bin_rows: List[Tuple[object, ...]] = []
    for k in all_k:
        p_values = sorted(
            set(result.registry.get(k, {}))
            | set(result.discarded_registry.get(k, {}))
            | set(result.discarded_counts.get(k, {}))
        )
        for p in p_values:
            retained = result.registry.get(k, {}).get(p, [])
            discarded = result.discarded_registry.get(k, {}).get(p, [])
            discarded_count = result.discarded_counts.get(k, {}).get(
                p, len(discarded)
            )
            best = retained[0] if retained else None
            audit = passivation_audits.get((k, p))
            branch = (
                "continues"
                if audit is not None and audit.valid_count > 0
                else "closed"
            )
            compliant = (
                "yes"
                if best and best.metadata.get("min_cn_compliant", False)
                else ("fallback" if best else "-")
            )
            bin_rows.append(
                (
                    k,
                    p,
                    len(retained) + discarded_count,
                    len(retained),
                    discarded_count,
                    best.metadata.get("bond_count", "-") if best else "-",
                    best.metadata.get("total_cn", "-") if best else "-",
                    compliant,
                    branch,
                )
            )
    lines.extend(
        _ascii_table(
            (
                "k",
                "p",
                "Generated",
                "Retained",
                "Discarded",
                "Best bonds",
                "Best total CN",
                "Min CN",
                "p-growth",
            ),
            bin_rows,
            right_align=(0, 1, 2, 3, 4, 5, 6),
        )
    )

    for k in all_k:
        p_values = sorted(
            set(result.registry.get(k, {}))
            | set(result.discarded_registry.get(k, {}))
            | set(result.discarded_counts.get(k, {}))
        )
        for p in p_values:
            retained = result.registry.get(k, {}).get(p, [])
            discarded = result.discarded_registry.get(k, {}).get(p, [])
            lines.extend(["", f"ISOMERS: k={k}, p={p}"])
            isomer_rows: List[Tuple[object, ...]] = []
            for record in [*retained, *discarded]:
                isomer_rows.append(
                    (
                        record.selection_status,
                        record.structure_id,
                        _formula(record.symbols),
                        _cn_values_text(record),
                        _cn_histogram_text(record),
                        record.metadata.get("bond_count", 0),
                        record.metadata.get("total_cn", 0),
                        record.metadata.get("bridge_count", 0),
                        "/".join(
                            str(
                                record.metadata.get("bridge_mode_counts", {}).get(
                                    mode, 0
                                )
                            )
                            for mode in (
                                "shared_vacant_cif_site",
                                "shared_occupied_neighbor",
                            )
                        ),
                        record.metadata.get("min_cn_violation_count", 0),
                        record.metadata.get("min_cn_total_shortfall", 0),
                        record.selection_reason,
                        ",".join(record.source_operations) or "-",
                        record.metadata.get("construction_native_xyz_path", "-"),
                        record.metadata.get("surface_xyz_path", "-"),
                    )
                )
            lines.extend(
                _ascii_table(
                    (
                        "Status",
                        "Structure ID",
                        "Formula",
                        "CN values",
                        "CN histogram",
                        "Bonds",
                        "Total CN",
                        "Bridges",
                        "CIF/Rhombic",
                        "Viol",
                        "Short",
                        "Reason",
                        "Sources",
                        "Construction-native XYZ",
                        "Surface XYZ",
                    ),
                    isomer_rows,
                    right_align=(5, 6, 7, 8, 9, 10),
                    max_widths={
                        1: 38,
                        3: 30,
                        4: 36,
                        11: 26,
                        12: 20,
                        13: 48,
                        14: 48,
                    },
                )
            )

            surface_records = [
                record
                for record in [*retained, *discarded]
                if record.surface_coordinates_data is not None
            ]
            if surface_records:
                geometry_rows: List[Tuple[object, ...]] = []
                for record in surface_records:
                    geometry = record.metadata.get("surface_geometry", {})
                    geometry_rows.append(
                        (
                            record.structure_id,
                            "yes" if geometry.get("projection_valid") else "no",
                            "yes" if geometry.get("coordinates_changed") else "no",
                            len(geometry.get("applied_rules", [])),
                            len(geometry.get("bridge_geometry", [])),
                            ",".join(
                                f"{float(item.get('surface_angle_deg', 0.0)):.2f}"
                                for item in geometry.get("bridge_geometry", [])
                            ) or "-",
                            ",".join(
                                f"{float(item.get('out_of_plane_rotation_deg', 0.0)):.1f}"
                                for item in geometry.get("bridge_geometry", [])
                            ) or "-",
                            len(geometry.get("coordinate_collisions", [])),
                            len(geometry.get("saturated_cd_intrusions", [])),
                            ",".join(
                                f"{float(item.get('angular_rms_deg', 0.0)):.2f}"
                                for item in geometry.get("cn4_tetrahedral_rms", [])
                            ) or "-",
                            f"{float(geometry.get('bond_length_rms_change_angstrom', 0.0)):.6f}",
                            f"{float(geometry.get('max_displacement_angstrom', 0.0)):.6f}",
                            len(geometry.get("unresolved_conflicts", [])),
                            geometry.get("message", "-"),
                        )
                    )
                lines.extend(["", f"SURFACE GEOMETRY: k={k}, p={p}"])
                lines.extend(
                    _ascii_table(
                        (
                            "Structure ID",
                            "Valid",
                            "Changed",
                            "Rules",
                            "Bridges",
                            "Bridge angles",
                            "Bridge rotations",
                            "Exact overlaps",
                            "Slot intrusions",
                            "CN4 RMS deg",
                            "Bond RMS A",
                            "Max shift A",
                            "Conflicts",
                            "Message",
                        ),
                        geometry_rows,
                        right_align=(3, 4, 5, 6, 7, 8, 9, 10, 11),
                        max_widths={0: 38, 13: 52},
                    )
                )
                bridge_host_rows: List[Tuple[object, ...]] = []
                for record in retained:
                    for bridge in record.metadata.get("bridge_edges", []):
                        primary = bridge.get("primary_host_atom_id")
                        secondary = bridge.get("host_atom_id")
                        bridge_host_rows.append(
                            (
                                record.structure_id,
                                bridge.get("ligand_atom_id", "-"),
                                bridge.get("bridge_mode", "-")
                                .replace("shared_vacant_cif_site", "CIF site")
                                .replace("shared_occupied_neighbor", "rhombic"),
                                primary if primary is not None else "-",
                                bridge.get("primary_cn_before_bridge", "-"),
                                record.graph.degree[primary]
                                if isinstance(primary, int)
                                else "-",
                                secondary if secondary is not None else "-",
                                bridge.get("secondary_cn_before_bridge", "-"),
                                record.graph.degree[secondary]
                                if isinstance(secondary, int)
                                else "-",
                            )
                        )
                if bridge_host_rows:
                    lines.extend(["", f"BRIDGE HOST CN: k={k}, p={p}"])
                    lines.extend(
                        _ascii_table(
                            (
                                "Structure ID",
                                "Cl",
                                "Mode",
                                "Primary Cd",
                                "Primary before",
                                "Primary final",
                                "Second Cd",
                                "Second before",
                                "Second final",
                            ),
                            bridge_host_rows,
                            right_align=(1, 3, 4, 5, 6, 7, 8),
                            max_widths={0: 38, 2: 12},
                        )
                    )

                valence_rows = []
                for record in surface_records:
                    block = record.metadata.get("surface_geometry", {}).get(
                        "pauling_valence"
                    )
                    if not block or not block.get("cations"):
                        continue
                    for item in block["cations"]:
                        valence_rows.append(
                            (
                                record.structure_id,
                                f"{item['symbol']}{item['atom_id']}",
                                record.graph.degree[item["atom_id"]],
                                f"{item['pauling_valence']:.2f}",
                                f"{item['deviation']:+.2f}",
                            )
                        )
                if valence_rows:
                    lines.extend(
                        [
                            "",
                            f"PAULING VALENCE: k={k}, p={p}",
                            "  Reporting only -- not used for ranking.  Sum of "
                            "|q(anion)|/CN(anion) over bonded anions; a cation",
                            "  is satisfied at its own formal charge.  DFT on "
                            "k=2 p=3 showed the most oversaturated cation is",
                            "  the one that sheds a ligand on relaxation "
                            "(3/3), while total bond count did not predict it.",
                        ]
                    )
                    lines.extend(
                        _ascii_table(
                            (
                                "Structure ID",
                                "Cation",
                                "CN",
                                "Valence",
                                "Deviation",
                            ),
                            valence_rows,
                            right_align=(2, 3, 4),
                            max_widths={0: 38},
                        )
                    )

    lines.extend(["", "REJECTION SUMMARY"])
    lines.extend(
        _ascii_table(
            ("Operation", "Transition", "Reason", "Count"),
            rejection_rows or [("-", "-", "No rejected candidates", 0)],
            right_align=(3,),
            max_widths={1: 24, 2: 48},
        )
    )

    total_retained = sum(
        len(records)
        for bins in result.registry.values()
        for records in bins.values()
    )
    total_discarded = sum(
        count
        for bins in result.discarded_counts.values()
        for count in bins.values()
    )
    physical_bins = sum(len(bins) for bins in result.registry.values())
    closed_branches = sum(
        audit.valid_count == 0
        for audit in result.sweep_audit
        if audit.operation == "skeleton_passivation"
    )
    lines.extend(["", "FINAL SUMMARY"])
    lines.extend(
        _ascii_table(
            ("Metric", "Count"),
            (
                ("Physical k/p bins", physical_bins),
                ("Retained structures", total_retained),
                ("Discarded structures", total_discarded),
                ("Closed passivation branches", closed_branches),
            ),
            right_align=(1,),
        )
    )
    lines.append("")
    return "\n".join(lines)


def _record_sort_key(record: ClusterRecord) -> Tuple[object, ...]:
    return (
        _graph_hash(record.graph),
        round(float(record.metadata.get("geometry_residual", 0.0)), 12),
        tuple(
            sorted(
                (
                    atom.symbol,
                    round(atom.coordinates[0], 8),
                    round(atom.coordinates[1], 8),
                    round(atom.coordinates[2], 8),
                )
                for atom in record.atoms
            )
        ),
    )


def _record_to_dict(record: ClusterRecord) -> Dict[str, object]:
    counts: Dict[str, int] = {}
    for symbol in record.symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    return {
        "structure_id": record.structure_id,
        "k": record.k,
        "p": record.p,
        "selection": {
            "status": record.selection_status,
            "reason": record.selection_reason,
            "coordination_score": list(record.coordination_score),
        },
        "source_operations": list(record.source_operations),
        "source_structure_ids": list(record.source_structure_ids),
        "skeleton_family_id": record.metadata.get("skeleton_family_id"),
        "ligand_shell_hash": record.metadata.get("ligand_shell_hash"),
        "stoichiometry": counts,
        "symbols": record.symbols,
        "coordinates": record.coordinates.tolist(),
        "surface_coordinates": (
            record.surface_coordinates.tolist()
            if record.surface_coordinates_data is not None
            else None
        ),
        "atoms": [
            {
                "id": atom.atom_id,
                "symbol": atom.symbol,
                "coordinates": list(atom.coordinates),
                "surface_coordinates": (
                    list(record.surface_coordinates[atom.atom_id])
                    if record.surface_coordinates_data is not None
                    else None
                ),
                "role": atom.role,
                "unit_id": atom.unit_id,
            }
            for atom in record.atoms
        ],
        "graph": nx.node_link_data(record.graph, edges="edges"),
        "metadata": _json_value(record.metadata),
    }


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _formula(symbols: Sequence[str]) -> str:
    counts: Dict[str, int] = {}
    order: List[str] = []
    for symbol in symbols:
        if symbol not in counts:
            order.append(symbol)
            counts[symbol] = 0
        counts[symbol] += 1
    return "".join(f"{symbol}{counts[symbol]}" for symbol in order)


def _formal_charge(
    atoms: Sequence[AtomRecord],
    charges: Mapping[str, int],
) -> int:
    return int(sum(charges.get(atom.symbol, 0) for atom in atoms))

__all__ = [
    'registry_to_dict',
    'write_nucleation_json',
    'nucleation_result_to_dict',
    'write_nucleation_bundle',
    '_ascii_table',
    '_transition_label',
    '_cn_values_text',
    '_cn_histogram_text',
    '_render_log',
    '_record_sort_key',
    '_record_to_dict',
    '_json_value',
    '_formula',
    '_formal_charge',
]
