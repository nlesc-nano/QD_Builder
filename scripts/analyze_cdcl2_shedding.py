#!/usr/bin/env python3
"""Analyse local CdCl2 motifs before and after CP2K relaxation.

The script is deliberately a post-processing tool.  It does not modify the
nucleation generator and does not infer a kinetic barrier.  It combines the
merged ``structures.csv`` produced by :mod:`analyze_cp2k_results` with the
registry that generated each starting XYZ file.

Example::

    python scripts/analyze_cdcl2_shedding.py \
        --analysis-root runs/cdse_map/runs/cdse_map/analysis_all_dft \
        --registry-root runs/cdse_map \
        --output runs/cdse_map/runs/cdse_map/analysis_all_dft/cdcl2_shedding

The registry root is the directory containing ``exact_k2/``,
``pathway_k6_redecorated/``, ``broad_k6_equilibrium/``, and similar run
directories.  Absolute paths work as well.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


try:
    from builder.graph_canon import canonical_form
except ModuleNotFoundError:  # Running directly from a source checkout.
    source_root = Path(__file__).resolve().parents[1] / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from builder.graph_canon import canonical_form


HARTREE_TO_KCAL_MOL = 627.5094740631
DEFAULT_CD_SE_CUTOFF = 3.25
DEFAULT_CD_CL_CUTOFF = 3.10
DEFAULT_CDCL2_ENERGY_HARTREE = -76.0795339478
DEFAULT_TEMPERATURE_K = 300.0
HASH_SUFFIX = re.compile(r"__[0-9a-f]{8}$")


Atom = tuple[str, float, float, float]


def read_xyz(path: Path) -> list[Atom]:
    """Read the first XYZ frame from a CP2K trajectory/final XYZ file."""

    with path.open(encoding="utf-8", errors="replace") as handle:
        line = handle.readline()
        if not line:
            raise ValueError(f"empty XYZ file: {path}")
        count = int(line.strip())
        comment = handle.readline()
        if not comment:
            raise ValueError(f"missing XYZ comment: {path}")
        atoms: list[Atom] = []
        for _ in range(count):
            fields = handle.readline().split()
            if len(fields) < 4:
                raise ValueError(f"truncated XYZ frame: {path}")
            atoms.append(
                (
                    fields[0],
                    float(fields[1].replace("D", "E").replace("d", "e")),
                    float(fields[2].replace("D", "E").replace("d", "e")),
                    float(fields[3].replace("D", "E").replace("d", "e")),
                )
            )
        return atoms


def atoms_from_registry(record: Mapping[str, Any]) -> list[Atom]:
    """Reconstruct the starting XYZ when the analyzer stored a registry reference.

    Discarded candidates are sometimes represented in ``structures.csv`` as
    ``registry.json#structure_id`` or point to a construction-native XYZ that
    was not copied into the DFT tree.  The registry still stores the exact
    starting coordinates, including ``surface_coordinates`` when available.
    """

    atoms: list[Atom] = []
    for atom in record.get("atoms", []):
        coordinates = atom.get("surface_coordinates") or atom.get("coordinates")
        if coordinates is None or len(coordinates) != 3:
            raise ValueError("registry atom has no three-dimensional coordinates")
        atoms.append(
            (
                str(atom["symbol"]),
                float(coordinates[0]),
                float(coordinates[1]),
                float(coordinates[2]),
            )
        )
    if not atoms:
        raise ValueError("registry record has no atoms")
    return atoms


def distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def inferred_adjacency(
    atoms: Sequence[Atom], cd_se_cutoff: float, cd_cl_cutoff: float
) -> list[set[int]]:
    """Infer only the Cd-Se and Cd-Cl chemical graph from relaxed XYZ."""

    adjacency = [set() for _ in atoms]
    for left in range(len(atoms)):
        for right in range(left + 1, len(atoms)):
            pair = {atoms[left][0], atoms[right][0]}
            if pair == {"Cd", "Se"}:
                cutoff = cd_se_cutoff
            elif pair == {"Cd", "Cl"}:
                cutoff = cd_cl_cutoff
            else:
                continue
            if distance(atoms[left][1:], atoms[right][1:]) <= cutoff:
                adjacency[left].add(right)
                adjacency[right].add(left)
    return adjacency


def registry_adjacency(record: Mapping[str, Any]) -> list[set[int]]:
    atoms = record.get("atoms", [])
    adjacency = [set() for _ in atoms]
    graph = record.get("graph", {})
    for edge in graph.get("edges", []):
        left = int(edge["source"])
        right = int(edge["target"])
        if 0 <= left < len(adjacency) and 0 <= right < len(adjacency):
            adjacency[left].add(right)
            adjacency[right].add(left)
    return adjacency


def bare_graph_hash(
    record: Mapping[str, Any], *, remove_precursor_center: int | None = None,
    core_only: bool = False,
) -> str:
    """Hash a ligand-free graph, optionally after removing one precursor Cd.

    ``skeleton_family_id`` in the registry retains precursor Cd centers, so it
    cannot match a ``(k,p)`` graph to its ``(k,p-1)`` shed product.  This hash
    removes all Cl atoms and, for a shedding target, one selected precursor Cd.
    The canonical graph representation is exact for the small coloured graphs
    used here.
    """

    atoms = list(record.get("atoms", []))
    keep: list[int] = []
    for index, atom in enumerate(atoms):
        role = str(atom.get("role", ""))
        if role == "precursor_ligand":
            continue
        if remove_precursor_center is not None and index == remove_precursor_center:
            continue
        if core_only and role not in {"core_cation", "core_anion"}:
            continue
        keep.append(index)
    new_index = {old: new for new, old in enumerate(keep)}
    labels = [
        f"{atoms[index].get('symbol', '')}:{atoms[index].get('role', '')}"
        for index in keep
    ]
    edges: list[tuple[int, int, str]] = []
    for edge in (record.get("graph", {}) or {}).get("edges", []):
        left, right = int(edge["source"]), int(edge["target"])
        if left not in new_index or right not in new_index:
            continue
        colour = str(edge.get("kind", edge.get("bond_order", "chemical")))
        edges.append((new_index[left], new_index[right], colour))
    certificate = canonical_form(labels, edges, compress_leaves=False).certificate
    return hashlib.sha1(repr(certificate).encode("utf-8")).hexdigest()[:16]


def simple_cycles(adjacency: Sequence[set[int]], length: int) -> set[tuple[int, ...]]:
    """Return unique node-set representations of simple cycles."""

    cycles: set[tuple[int, ...]] = set()

    def walk(start: int, path: list[int]) -> None:
        if len(path) == length:
            if start in adjacency[path[-1]]:
                cycles.add(tuple(sorted(path)))
            return
        for neighbour in adjacency[path[-1]]:
            if neighbour < start or neighbour in path:
                continue
            walk(start, path + [neighbour])

    for start in range(len(adjacency)):
        walk(start, [start])
    return cycles


def is_cd_cl_se_four_ring(cycle: Iterable[int], atoms: Sequence[Atom]) -> bool:
    symbols = [atoms[index][0] for index in cycle]
    return (
        len(symbols) == 4
        and symbols.count("Cd") == 2
        and symbols.count("Cl") == 1
        and symbols.count("Se") == 1
    )


def is_cd_cl_se_four_ring_symbols(
    cycle: Iterable[int], symbols: Sequence[str]
) -> bool:
    values = [symbols[index] for index in cycle]
    return (
        len(values) == 4
        and values.count("Cd") == 2
        and values.count("Cl") == 1
        and values.count("Se") == 1
    )


def motif_name(four_rings: int, cl_count: int, terminal_cl: int) -> str:
    if four_rings >= 2:
        return "multi_4ring"
    if four_rings == 1:
        return "one_4ring"
    if cl_count and terminal_cl == cl_count:
        return "terminal_only"
    if cl_count:
        return "other_cl"
    return "no_cl"


def resolve_path(path: str, analysis_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate
    for base in (analysis_root, analysis_root.parent, Path.cwd()):
        resolved = base / candidate
        if resolved.is_file():
            return resolved
    return candidate


def load_registry_index(registry_root: Path) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Index retained and discarded records by registry directory and ID."""

    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    for path in sorted(registry_root.rglob("registry.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        index: dict[str, Mapping[str, Any]] = {}
        for section in ("registry", "discarded_registry"):
            for bins in (payload.get(section) or {}).values():
                for records in bins.values():
                    for record in records:
                        if record.get("structure_id"):
                            index[str(record["structure_id"])] = record
        result[path.parent.name] = index
    return result


def source_registry_name(source_xyz: str, indexes: Mapping[str, Any]) -> str | None:
    for name in indexes:
        if f"/{name}/" in source_xyz or source_xyz.startswith(f"{name}/"):
            return name
    return None


def registry_id(run_dir: str) -> str:
    return HASH_SUFFIX.sub("", Path(run_dir).name)


def as_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
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


def package_rows_for_structure(
    record: Mapping[str, Any],
    relaxed: Sequence[Atom],
    start_atoms: Sequence[Atom],
    *,
    cd_se_cutoff: float,
    cd_cl_cutoff: float,
) -> list[dict[str, Any]]:
    """Extract one descriptor row for every precursor Cd center."""

    registry_atoms = record.get("atoms", [])
    if len(registry_atoms) != len(start_atoms) or len(relaxed) != len(start_atoms):
        raise ValueError("atom count differs between registry, start, and relaxed XYZ")
    registry_symbols = [str(atom["symbol"]) for atom in registry_atoms]
    if registry_symbols != [atom[0] for atom in start_atoms]:
        raise ValueError("registry and starting XYZ atom ordering/symbols differ")
    if [atom[0] for atom in relaxed] != [atom[0] for atom in start_atoms]:
        raise ValueError("starting and relaxed XYZ atom ordering/symbols differ")

    start_adj = registry_adjacency(record)
    relaxed_adj = inferred_adjacency(relaxed, cd_se_cutoff, cd_cl_cutoff)
    start_cycles4 = simple_cycles(start_adj, 4)
    relaxed_cycles4 = simple_cycles(relaxed_adj, 4)
    start_cycles6 = simple_cycles(start_adj, 6)
    relaxed_cycles6 = simple_cycles(relaxed_adj, 6)
    start_four = {
        cycle
        for cycle in start_cycles4
        if is_cd_cl_se_four_ring_symbols(cycle, registry_symbols)
    }
    relaxed_four = {
        cycle
        for cycle in relaxed_cycles4
        if is_cd_cl_se_four_ring(cycle, relaxed)
    }
    start_six_with_cl = {
        cycle for cycle in start_cycles6 if any(registry_symbols[i] == "Cl" for i in cycle)
    }
    relaxed_six_with_cl = {
        cycle for cycle in relaxed_cycles6 if any(relaxed[i][0] == "Cl" for i in cycle)
    }
    target_hashes = {
        index: bare_graph_hash(record, remove_precursor_center=index)
        for index, atom in enumerate(registry_atoms)
        if atom.get("role") == "precursor_center"
    }

    rows: list[dict[str, Any]] = []
    for center, atom in enumerate(registry_atoms):
        if atom.get("role") != "precursor_center":
            continue
        start_cl = sorted(
            neighbour for neighbour in start_adj[center] if registry_symbols[neighbour] == "Cl"
        )
        relaxed_cl = sorted(
            neighbour for neighbour in relaxed_adj[center] if relaxed[neighbour][0] == "Cl"
        )
        start_four_here = [cycle for cycle in start_four if center in cycle]
        relaxed_four_here = [cycle for cycle in relaxed_four if center in cycle]
        start_six_here = [cycle for cycle in start_six_with_cl if center in cycle]
        relaxed_six_here = [cycle for cycle in relaxed_six_with_cl if center in cycle]
        start_terminal = sum(len(start_adj[cl]) == 1 for cl in start_cl)
        relaxed_terminal = sum(len(relaxed_adj[cl]) == 1 for cl in relaxed_cl)
        retained = sum(cl in relaxed_cl for cl in start_cl)
        start_coord = start_atoms[center][1:]
        relaxed_coord = relaxed[center][1:]
        rows.append(
            {
                "precursor_center_atom_id": center,
                "precursor_unit_id": atom.get("unit_id"),
                "shed_target_bare_skeleton_hash": target_hashes[center],
                "start_cl_atom_ids": json.dumps(start_cl),
                "relaxed_cl_atom_ids": json.dumps(relaxed_cl),
                "start_cl_count": len(start_cl),
                "start_se_neighbors": sum(
                    registry_symbols[i] == "Se" for i in start_adj[center]
                ),
                "start_cd_neighbors": sum(
                    registry_symbols[i] == "Cd" for i in start_adj[center]
                ),
                "start_bridging_cl_count": sum(len(start_adj[cl]) >= 2 for cl in start_cl),
                "start_terminal_cl_count": start_terminal,
                "start_max_cl_bridge_load": max(
                    (len(start_adj[cl]) for cl in start_cl), default=0
                ),
                "start_4ring_count": len(start_four_here),
                "start_6ring_with_cl_count": len(start_six_here),
                "start_motif": motif_name(len(start_four_here), len(start_cl), start_terminal),
                "relaxed_cl_count": len(relaxed_cl),
                "relaxed_se_neighbors": sum(
                    relaxed[i][0] == "Se" for i in relaxed_adj[center]
                ),
                "relaxed_cd_neighbors": sum(
                    relaxed[i][0] == "Cd" for i in relaxed_adj[center]
                ),
                "relaxed_bridging_cl_count": sum(
                    len(relaxed_adj[cl]) >= 2 for cl in relaxed_cl
                ),
                "relaxed_terminal_cl_count": relaxed_terminal,
                "relaxed_max_cl_bridge_load": max(
                    (len(relaxed_adj[cl]) for cl in relaxed_cl), default=0
                ),
                "relaxed_4ring_count": len(relaxed_four_here),
                "relaxed_6ring_with_cl_count": len(relaxed_six_here),
                "relaxed_motif": motif_name(
                    len(relaxed_four_here), len(relaxed_cl), relaxed_terminal
                ),
                "cl_bond_persistence_count": retained,
                "cl_bond_persistence_fraction": (
                    retained / len(start_cl) if start_cl else None
                ),
                "center_displacement_angstrom": distance(start_coord, relaxed_coord),
            }
        )
    return rows


def summarize_structure(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["start_motif"]) for row in rows)
    persistence = [
        float(row["cl_bond_persistence_fraction"])
        for row in rows
        if row.get("cl_bond_persistence_fraction") not in (None, "")
    ]
    return {
        "package_count": len(rows),
        "start_no_cl_count": counts["no_cl"],
        "start_terminal_only_count": counts["terminal_only"],
        "start_one_4ring_count": counts["one_4ring"],
        "start_multi_4ring_count": counts["multi_4ring"],
        "start_other_cl_count": counts["other_cl"],
        "mean_cl_bond_persistence_fraction": mean(persistence) if persistence else None,
        "mean_center_displacement_angstrom": mean(
            float(row["center_displacement_angstrom"]) for row in rows
        )
        if rows
        else None,
    }


def _matched_energy_rows(
    structures: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
    count_field: str,
) -> list[dict[str, Any]]:
    """Compare motif-containing and motif-free structures within groups."""

    bins: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in structures:
        if row.get("relative_energy_kcal_mol") in (None, ""):
            continue
        key: tuple[Any, ...] = tuple(
            int(row[field]) if field in {"k", "p"} else str(row[field])
            for field in group_fields
        )
        bins[key].append(row)
    output: list[dict[str, Any]] = []
    for motif in ("terminal_only", "one_4ring", "multi_4ring", "other_cl"):
        deltas: list[float] = []
        matched_bins = 0
        motif_wins = 0
        nonmotif_wins = 0
        for _group_key, records in bins.items():
            with_motif = [
                float(record["relative_energy_kcal_mol"])
                for record in records
                if int(record.get(f"start_{motif}_count", 0)) > 0
            ]
            without_motif = [
                float(record["relative_energy_kcal_mol"])
                for record in records
                if int(record.get(f"start_{motif}_count", 0)) == 0
            ]
            if not with_motif or not without_motif:
                continue
            matched_bins += 1
            deltas.append(mean(with_motif) - mean(without_motif))
            motif_wins += sum(
                str(record.get("dft_rank_in_bin", "")) == "1"
                for record in records
                if int(record.get(f"start_{motif}_count", 0)) > 0
            )
            nonmotif_wins += sum(
                str(record.get("dft_rank_in_bin", "")) == "1"
                for record in records
                if int(record.get(f"start_{motif}_count", 0)) == 0
            )
        output.append(
            {
                "motif": motif,
                count_field: matched_bins,
                "mean_within_bin_delta_kcal_mol": mean(deltas) if deltas else None,
                "median_within_bin_delta_kcal_mol": median(deltas) if deltas else None,
                "motif_winner_count": motif_wins,
                "nonmotif_winner_count": nonmotif_wins,
            }
        )
    return output


def matched_energy_rows(structures: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return _matched_energy_rows(
        structures, ("k", "p"), "matched_bin_count"
    )


def matched_family_energy_rows(
    structures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return _matched_energy_rows(
        structures,
        ("k", "p", "skeleton_family_id"),
        "matched_family_count",
    )


def build_shedding_reaction_rows(
    structures: Sequence[Mapping[str, Any]],
    packages: Sequence[Mapping[str, Any]],
    *,
    cdcl2_energy_hartree: float,
    temperature_k: float,
) -> list[dict[str, Any]]:
    """Compute matched neutral-CdCl2 shedding reaction-energy proxies.

    For each package in a parent ``(k,p)`` structure, the target is the
    ligand-free graph obtained by removing that precursor Cd.  Candidate
    ``(k,p-1)`` structures with the same target graph are then compared using

    ``Delta E = E(k,p-1) + E(CdCl2) - E(k,p)``.

    This is an endpoint/reaction-energy proxy, not a transition-state barrier.
    """

    by_target: dict[tuple[int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for structure in structures:
        energy = structure.get("energy_hartree")
        if energy in (None, ""):
            continue
        key = (
            int(structure["k"]),
            int(structure["p"]),
            str(structure.get("bare_skeleton_hash", "")),
        )
        by_target[key].append(structure)
    by_structure_id = {str(row["structure_id"]): row for row in structures}
    output: list[dict[str, Any]] = []
    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    kb_ev_per_k = 8.617333262145e-5
    kt_ev = kb_ev_per_k * temperature_k

    for package in packages:
        parent = by_structure_id.get(str(package["structure_id"]))
        if parent is None or int(parent["p"]) < 1:
            continue
        target_key = (
            int(parent["k"]),
            int(parent["p"]) - 1,
            str(package["shed_target_bare_skeleton_hash"]),
        )
        displaced = by_target.get(target_key, [])
        if not displaced:
            continue
        parent_energy = float(parent["energy_hartree"])
        for product in displaced:
            product_energy = float(product["energy_hartree"])
            delta_hartree = product_energy + cdcl2_energy_hartree - parent_energy
            delta_kcal = delta_hartree * HARTREE_TO_KCAL_MOL
            delta_ev = delta_hartree * 27.211386245988
            signed_exponent = max(-700.0, min(700.0, -delta_ev / kt_ev))
            uphill_exponent = max(
                -700.0, min(700.0, -max(delta_ev, 0.0) / kt_ev)
            )
            output.append(
                {
                    "parent_structure_id": parent["structure_id"],
                    "displaced_structure_id": product["structure_id"],
                    "parent_k": parent["k"],
                    "parent_p": parent["p"],
                    "displaced_p": product["p"],
                    "skeleton_family_parent": parent.get("skeleton_family_id", ""),
                    "skeleton_family_displaced": product.get("skeleton_family_id", ""),
                    "bare_skeleton_hash_after_shedding": package[
                        "shed_target_bare_skeleton_hash"
                    ],
                    "precursor_center_atom_id": package[
                        "precursor_center_atom_id"
                    ],
                    "precursor_unit_id": package.get("precursor_unit_id", ""),
                    "motif": package["start_motif"],
                    "start_4ring_count": package["start_4ring_count"],
                    "start_bridging_cl_count": package[
                        "start_bridging_cl_count"
                    ],
                    "start_terminal_cl_count": package[
                        "start_terminal_cl_count"
                    ],
                    "parent_energy_hartree": parent_energy,
                    "displaced_cluster_energy_hartree": product_energy,
                    "free_cdcl2_energy_hartree": cdcl2_energy_hartree,
                    "delta_e_proxy_hartree": delta_hartree,
                    "delta_e_proxy_kcal_mol": delta_kcal,
                    "delta_e_proxy_ev": delta_ev,
                    "temperature_k": temperature_k,
                    "signed_arrhenius_factor": math.exp(signed_exponent),
                    "uphill_arrhenius_factor": math.exp(uphill_exponent),
                }
            )
    return output


def shedding_reaction_summary(
    reaction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for motif in ("terminal_only", "one_4ring", "multi_4ring", "other_cl", "no_cl"):
        rows = [row for row in reaction_rows if row["motif"] == motif]
        if not rows:
            output.append(
                {
                    "motif": motif,
                    "matched_reaction_count": 0,
                    "mean_delta_e_proxy_kcal_mol": None,
                    "median_delta_e_proxy_kcal_mol": None,
                    "median_signed_arrhenius_factor": None,
                    "median_uphill_arrhenius_factor": None,
                    "positive_delta_fraction": None,
                }
            )
            continue
        output.append(
            {
                "motif": motif,
                "matched_reaction_count": len(rows),
                "mean_delta_e_proxy_kcal_mol": mean(
                    float(row["delta_e_proxy_kcal_mol"]) for row in rows
                ),
                "median_delta_e_proxy_kcal_mol": median(
                    float(row["delta_e_proxy_kcal_mol"]) for row in rows
                ),
                "median_signed_arrhenius_factor": median(
                    float(row["signed_arrhenius_factor"]) for row in rows
                ),
                "median_uphill_arrhenius_factor": median(
                    float(row["uphill_arrhenius_factor"]) for row in rows
                ),
                "positive_delta_fraction": mean(
                    float(float(row["delta_e_proxy_kcal_mol"]) > 0.0)
                    for row in rows
                ),
            }
        )
    return output


def write_report(
    path: Path,
    *,
    structure_rows: Sequence[Mapping[str, Any]],
    package_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    energy_rows: Sequence[Mapping[str, Any]],
    family_energy_rows: Sequence[Mapping[str, Any]],
    reaction_summary_rows: Sequence[Mapping[str, Any]],
    reaction_rows: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
    cd_se_cutoff: float,
    cd_cl_cutoff: float,
) -> None:
    lines = [
        "# CdCl2 local shedding analysis",
        "",
        f"Analysis-ready structures mapped: **{len(structure_rows)}**",
        f"Precursor Cd packages: **{len(package_rows)}**",
        f"Skipped structures: **{len(skipped)}**",
        "",
        "Starting graphs use the generated registry. Relaxed bonds are inferred "
        f"with Cd-Se ≤ {cd_se_cutoff:g} Å and Cd-Cl ≤ {cd_cl_cutoff:g} Å.",
        "",
        "The package rows are local bond-persistence descriptors, not kinetic "
        "barriers or unique per-ligand binding energies. A bridging Cl can be "
        "shared by two Cd centers, so package ownership is intrinsically ambiguous.",
        "",
        "## Starting motif counts and relaxation persistence",
        "",
        "| motif | packages | mean Cd-Cl persistence | mean relaxed bridge count |",
        "|---|---:|---:|---:|",
    ]
    for motif in ("no_cl", "terminal_only", "one_4ring", "multi_4ring", "other_cl"):
        rows = [row for row in package_rows if row["start_motif"] == motif]
        persistence = [
            float(row["cl_bond_persistence_fraction"])
            for row in rows
            if row.get("cl_bond_persistence_fraction") not in (None, "")
        ]
        bridges = [float(row["relaxed_bridging_cl_count"]) for row in rows]
        lines.append(
            f"| {motif} | {len(rows)} | "
            f"{mean(persistence):.3f} | {mean(bridges):.3f} |"
            if rows and persistence
            else (
                f"| {motif} | {len(rows)} | — | "
                f"{mean(bridges):.3f} |"
                if rows
                else f"| {motif} | 0 | — | — |"
            )
        )
    lines.extend([
        "",
        "## Starting → relaxed motif transitions",
        "",
        "| starting motif | relaxed motif | count |",
        "|---|---|---:|",
    ])
    for row in transition_rows:
        lines.append(
            f"| {row['start_motif']} | {row['relaxed_motif']} | {row['count']} |"
        )
    lines.extend([
        "",
        "## Matched-bin energy comparisons",
        "",
        "The energy difference is motif-containing minus motif-free within the "
        "same `(k,p)` bin. Positive values therefore favor the motif-free group.",
        "",
        "| motif | matched bins | mean ΔE (kcal/mol) | median ΔE | motif wins | non-motif wins |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in energy_rows:
        def fmt(value: Any) -> str:
            return "—" if value in (None, "") else f"{float(value):.3f}"

        lines.append(
            f"| {row['motif']} | {row['matched_bin_count']} | "
            f"{fmt(row['mean_within_bin_delta_kcal_mol'])} | "
            f"{fmt(row['median_within_bin_delta_kcal_mol'])} | "
            f"{row['motif_winner_count']} | {row['nonmotif_winner_count']} |"
        )
    lines.extend([
        "",
        "## Matched skeleton-family energy comparisons",
        "",
        "These comparisons require the same `(k,p,skeleton_family_id)`. They are "
        "usually based on fewer structures, but are less confounded by the core "
        "skeleton than the bin-level comparison.",
        "",
        "| motif | matched families | mean ΔE (kcal/mol) | median ΔE | motif wins | non-motif wins |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in family_energy_rows:
        def fmt_family(value: Any) -> str:
            return "—" if value in (None, "") else f"{float(value):.3f}"

        lines.append(
            f"| {row['motif']} | {row['matched_family_count']} | "
            f"{fmt_family(row['mean_within_bin_delta_kcal_mol'])} | "
            f"{fmt_family(row['median_within_bin_delta_kcal_mol'])} | "
            f"{row['motif_winner_count']} | {row['nonmotif_winner_count']} |"
        )
    lines.extend([
        "",
        "## Matched CdCl2 shedding reaction proxies",
        "",
        "For a parent `(k,p)` and a matched `(k,p-1)` product obtained by "
        "removing one precursor Cd from the ligand-free graph, the script uses",
        "",
        "`ΔE_proxy = E(k,p-1) + E(CdCl2) − E(k,p)`.",
        "",
        "The signed Arrhenius factor is `exp(−ΔE_proxy/kBT)`. The uphill factor "
        "uses `max(ΔE_proxy,0)` and is bounded by one. These are endpoint "
        "proxies, not transition-state barriers.",
        "",
        "| motif | matched reactions | mean ΔE (kcal/mol) | median ΔE | median signed factor | median uphill factor | ΔE>0 fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in reaction_summary_rows:
        def fmt_reaction(value: Any) -> str:
            return "—" if value in (None, "") else f"{float(value):.4g}"

        lines.append(
            f"| {row['motif']} | {row['matched_reaction_count']} | "
            f"{fmt_reaction(row['mean_delta_e_proxy_kcal_mol'])} | "
            f"{fmt_reaction(row['median_delta_e_proxy_kcal_mol'])} | "
            f"{fmt_reaction(row['median_signed_arrhenius_factor'])} | "
            f"{fmt_reaction(row['median_uphill_arrhenius_factor'])} | "
            f"{fmt_reaction(row['positive_delta_fraction'])} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Use the package table to calibrate local shedding priorities while keeping "
        "the existing surface-dependent `smax` as a hard upper bound. Persistence "
        "is evidence that a local Cd-Cl environment survives relaxation; it is not "
        "a transition-state barrier. Energy comparisons should be used only within "
        "matched composition bins and, where possible, matched skeleton families.",
        "",
    ])
    if skipped:
        lines.extend(["## Skipped structures", "", "| structure | reason |", "|---|---|"])
        for row in skipped:
            lines.append(f"| {row['structure_id']} | {row['reason']} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument(
        "--registry-root",
        type=Path,
        required=True,
        help="directory containing the nucleation run directories with registry.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cd-se-cutoff", type=float, default=DEFAULT_CD_SE_CUTOFF)
    parser.add_argument("--cd-cl-cutoff", type=float, default=DEFAULT_CD_CL_CUTOFF)
    parser.add_argument(
        "--cdcl2-energy-hartree",
        type=float,
        default=DEFAULT_CDCL2_ENERGY_HARTREE,
        help="free neutral CdCl2 energy used for the shedding reaction proxy",
    )
    parser.add_argument(
        "--temperature-k",
        type=float,
        default=DEFAULT_TEMPERATURE_K,
        help="temperature for Arrhenius-like relative factors",
    )
    parser.add_argument(
        "--include-nonready",
        action="store_true",
        help="include rows that are not marked analysis-ready when a relaxed XYZ exists",
    )
    args = parser.parse_args()
    analysis_root = args.analysis_root.expanduser().resolve()
    registry_root = args.registry_root.expanduser().resolve()
    output = (args.output or analysis_root / "cdcl2_shedding").expanduser().resolve()
    structures_csv = analysis_root / "structures.csv"
    if not structures_csv.is_file():
        parser.error(f"structures.csv not found under {analysis_root}")
    indexes = load_registry_index(registry_root)
    if not indexes:
        parser.error(f"no registry.json files found under {registry_root}")

    package_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    with structures_csv.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    bin_min_hartree: dict[tuple[int, int], float] = {}
    for record in records:
        energy_hartree = as_float(record.get("energy_hartree"))
        if energy_hartree is None:
            continue
        key = (int(record.get("k", 0)), int(record.get("p", 0)))
        bin_min_hartree[key] = min(bin_min_hartree.get(key, energy_hartree), energy_hartree)
    for row in records:
        if row.get("analysis_ready") != "1" and not args.include_nonready:
            continue
        structure_id = row.get("structure_id", "")
        source = row.get("source_xyz", "")
        registry_name = source_registry_name(source, indexes)
        if registry_name is None:
            skipped.append({"structure_id": structure_id, "reason": "registry_root_not_identified"})
            continue
        record = indexes[registry_name].get(registry_id(row.get("run_dir", "")))
        if record is None:
            skipped.append({"structure_id": structure_id, "reason": "registry_record_not_found"})
            continue
        start_path = resolve_path(source, analysis_root)
        relaxed_path = resolve_path(row.get("relaxed_xyz", ""), analysis_root)
        try:
            start_atoms = read_xyz(start_path) if start_path.is_file() else atoms_from_registry(record)
            relaxed_atoms = read_xyz(relaxed_path)
            local_rows = package_rows_for_structure(
                record,
                relaxed_atoms,
                start_atoms,
                cd_se_cutoff=args.cd_se_cutoff,
                cd_cl_cutoff=args.cd_cl_cutoff,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            skipped.append({"structure_id": structure_id, "reason": str(exc)})
            continue
        energy = as_float(row.get("relative_energy_kcal_mol"))
        if energy is None:
            energy_hartree = as_float(row.get("energy_hartree"))
            minimum = bin_min_hartree.get((int(row.get("k", 0)), int(row.get("p", 0))))
            if energy_hartree is not None and minimum is not None:
                energy = (energy_hartree - minimum) * HARTREE_TO_KCAL_MOL
        common = {
            "structure_id": structure_id,
            "registry_structure_id": registry_id(row.get("run_dir", "")),
            "registry_root": registry_name,
            "skeleton_family_id": record.get("skeleton_family_id", ""),
            "bare_skeleton_hash": bare_graph_hash(record),
            "core_skeleton_hash": bare_graph_hash(record, core_only=True),
            "k": int(row.get("k", 0)),
            "p": int(row.get("p", 0)),
            "quality_status": row.get("quality_status", ""),
            "analysis_ready": row.get("analysis_ready", ""),
            "dft_rank_in_bin": row.get("dft_rank_in_bin", ""),
            "energy_hartree": as_float(row.get("energy_hartree")),
            "relative_energy_kcal_mol": energy,
            "source_xyz": str(start_path),
            "relaxed_xyz": str(relaxed_path),
        }
        enriched = [{**common, **local} for local in local_rows]
        package_rows.extend(enriched)
        summary = summarize_structure(enriched)
        structure_rows.append({**common, **summary})

    motif_counter = Counter(
        (row["start_motif"], row["relaxed_motif"]) for row in package_rows
    )
    transition_rows = [
        {"start_motif": start, "relaxed_motif": relaxed, "count": count}
        for (start, relaxed), count in sorted(motif_counter.items())
    ]
    energy_rows = matched_energy_rows(structure_rows)
    family_energy_rows = matched_family_energy_rows(structure_rows)
    reaction_rows = build_shedding_reaction_rows(
        structure_rows,
        package_rows,
        cdcl2_energy_hartree=args.cdcl2_energy_hartree,
        temperature_k=args.temperature_k,
    )
    reaction_summary_rows = shedding_reaction_summary(reaction_rows)
    output.mkdir(parents=True, exist_ok=True)

    package_fields = list(package_rows[0].keys()) if package_rows else []
    structure_fields = list(structure_rows[0].keys()) if structure_rows else []
    write_csv(output / "cdcl2_package_descriptors.csv", package_rows, package_fields)
    write_csv(output / "cdcl2_structure_summary.csv", structure_rows, structure_fields)
    write_csv(
        output / "cdcl2_motif_transitions.csv",
        transition_rows,
        ["start_motif", "relaxed_motif", "count"],
    )
    write_csv(
        output / "cdcl2_matched_bin_energies.csv",
        energy_rows,
        [
            "motif",
            "matched_bin_count",
            "mean_within_bin_delta_kcal_mol",
            "median_within_bin_delta_kcal_mol",
            "motif_winner_count",
            "nonmotif_winner_count",
        ],
    )
    write_csv(
        output / "cdcl2_matched_family_energies.csv",
        family_energy_rows,
        [
            "motif",
            "matched_family_count",
            "mean_within_bin_delta_kcal_mol",
            "median_within_bin_delta_kcal_mol",
            "motif_winner_count",
            "nonmotif_winner_count",
        ],
    )
    write_csv(
        output / "cdcl2_shedding_reactions.csv",
        reaction_rows,
        list(reaction_rows[0].keys()) if reaction_rows else [],
    )
    write_csv(
        output / "cdcl2_shedding_reaction_summary.csv",
        reaction_summary_rows,
        [
            "motif",
            "matched_reaction_count",
            "mean_delta_e_proxy_kcal_mol",
            "median_delta_e_proxy_kcal_mol",
            "median_signed_arrhenius_factor",
            "median_uphill_arrhenius_factor",
            "positive_delta_fraction",
        ],
    )
    write_csv(output / "cdcl2_skipped.csv", skipped, ["structure_id", "reason"])
    write_report(
        output / "cdcl2_shedding_report.md",
        structure_rows=structure_rows,
        package_rows=package_rows,
        transition_rows=transition_rows,
        energy_rows=energy_rows,
        family_energy_rows=family_energy_rows,
        reaction_summary_rows=reaction_summary_rows,
        reaction_rows=reaction_rows,
        skipped=skipped,
        cd_se_cutoff=args.cd_se_cutoff,
        cd_cl_cutoff=args.cd_cl_cutoff,
    )
    print(f"Mapped structures: {len(structure_rows)}")
    print(f"Mapped precursor packages: {len(package_rows)}")
    print(f"Skipped structures: {len(skipped)}")
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
