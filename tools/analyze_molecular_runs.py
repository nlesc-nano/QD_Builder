#!/usr/bin/env python3
"""Analyze saved molecular graph/xTB trials without changing the builder.

The molecular builder intentionally keeps this analysis out of its enumeration
path.  This script consumes the per-bin ``motif_trials.csv`` files and their
initial/relaxed XYZ files, reconstructs graph and geometry descriptors, and
writes energy/filter/lineage summaries.

Example
-------
    python tools/analyze_molecular_runs.py \
        --root runs/cdse_motif_k2p3 \
        --output runs/cdse_motif_k2p3_analysis

The script has no dependency on the builder package; it only needs numpy and
networkx, which are already dependencies of the repository.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


EV_TO_KCAL = 23.060548
ENERGY_RE = re.compile(r"(?:^|\s)energy_eV=([-+0-9.eE]+)")
# One deterministic skeleton mapping is sufficient for the offline overlap
# diagnostic; exact lineage itself is still checked independently.  Exploring
# every automorphism of a star-like skeleton can otherwise dominate the run.
MAX_SYMMETRY_MAPPINGS = 1


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _edges(text: str) -> Tuple[Tuple[int, int], ...]:
    out = []
    for item in (text or "").split("|"):
        if not item:
            continue
        left, right = item.split("-", 1)
        out.append((min(int(left), int(right)), max(int(left), int(right))))
    return tuple(sorted(set(out)))


def _xyz_path(bin_dir: Path, trial_id: str, field_value: str) -> Path:
    """Resolve paths from both old and current checkpoint CSV formats."""

    candidates: List[Path] = []
    if field_value:
        raw = Path(field_value)
        candidates.extend((bin_dir.parent.parent.parent / raw, bin_dir / raw))
    candidates.extend(
        (
            bin_dir / "motif_trials" / f"{trial_id}_xtb.xyz",
            bin_dir / "motif_trials" / f"{trial_id}_initial.xyz",
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _read_xyz(path: Path) -> Tuple[Tuple[str, ...], Optional[np.ndarray], str]:
    if not path.is_file():
        return (), None, ""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return (), None, ""
    try:
        count = int(lines[0].strip())
    except ValueError:
        return (), None, lines[1].strip()
    symbols: List[str] = []
    coords: List[Tuple[float, float, float]] = []
    for line in lines[2 : 2 + count]:
        fields = line.split()
        if len(fields) < 4:
            continue
        symbols.append(fields[0])
        coords.append((float(fields[1]), float(fields[2]), float(fields[3])))
    return tuple(symbols), np.asarray(coords, dtype=float), lines[1].strip()


def _graph(symbols: Sequence[str], edges: Sequence[Tuple[int, int]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(symbols)))
    nx.set_node_attributes(graph, {i: str(s) for i, s in enumerate(symbols)}, "element")
    graph.add_edges_from(edges)
    return graph


def _node_match(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return left.get("element") == right.get("element")


def _lineage_bucket(graph: nx.Graph) -> object:
    """Return a cheap typed bucket key for lineage candidate lookup.

    This is deliberately only a prefilter.  Every bucket candidate is still
    checked with exact NetworkX isomorphism before it is reported as a match.
    The repository's exact canonicalizer is useful for the builder's hot path,
    but running it for every possible Se/Cd deletion is unnecessarily costly in
    an offline diagnostic.
    """

    degree_labels = tuple(
        sorted(
            (
                str(data.get("element", "")),
                int(graph.degree[node]),
            )
            for node, data in graph.nodes(data=True)
        )
    )
    labels: Dict[int, object] = {
        node: (str(data.get("element", "")), int(graph.degree[node]))
        for node, data in graph.nodes(data=True)
    }
    # Two inexpensive refinement rounds separate most non-isomorphic typed
    # skeletons without invoking a general graph hash for every deletion.
    for _ in range(2):
        labels = {
            node: (
                labels[node],
                tuple(sorted(labels[neighbor] for neighbor in graph.neighbors(node))),
            )
            for node in graph
        }
    return (degree_labels, tuple(sorted(labels.values(), key=repr)), graph.number_of_edges())


def _lineage_key(graph: nx.Graph) -> object:
    """Typed prefilter key; exact matching follows for skeleton candidates."""

    return _lineage_bucket(graph)


def _degree_distribution(graph: nx.Graph, nodes: Iterable[int]) -> str:
    counts = Counter(graph.degree[node] for node in nodes)
    return ";".join(f"{degree}:{counts[degree]}" for degree in sorted(counts))


def _simple_cycles_of_length(graph: nx.Graph, length: int) -> List[frozenset]:
    """Enumerate undirected simple cycles as edge sets.

    Molecular skeletons are small, but a bounded DFS avoids the directed-cycle
    multiplicity and ordering ambiguity of ``networkx.simple_cycles``.
    """

    if length < 3:
        return []
    found: set[frozenset] = set()
    for start in sorted(graph):
        stack: List[Tuple[int, Tuple[int, ...]]] = [(start, (start,))]
        while stack:
            current, path = stack.pop()
            if len(path) == length:
                if graph.has_edge(current, start):
                    edges = []
                    for index in range(length):
                        left = path[index]
                        right = path[(index + 1) % length]
                        edges.append((min(left, right), max(left, right)))
                    found.add(frozenset(edges))
                continue
            for neighbor in graph.neighbors(current):
                if neighbor in path:
                    continue
                # The smallest vertex in a cycle is used as the start.  This
                # removes rotational duplicates while retaining both paths
                # until they collapse to the same edge set.
                if neighbor < start:
                    continue
                stack.append((neighbor, path + (neighbor,)))
    return sorted(found, key=lambda item: sorted(item))


def _ring_descriptors(graph: nx.Graph) -> Dict[str, object]:
    inorganic_nodes = [
        node
        for node, data in graph.nodes(data=True)
        if data.get("element") in {"Cd", "Se"}
    ]
    inorganic = graph.subgraph(inorganic_nodes).copy()
    cycles = []
    for cycle in _simple_cycles_of_length(inorganic, 6):
        nodes = set(node for edge in cycle for node in edge)
        if len(nodes) != 6:
            continue
        elements = [inorganic.nodes[node].get("element") for node in nodes]
        if elements.count("Cd") == 3 and elements.count("Se") == 3:
            cycles.append(cycle)
    fused_pairs = sum(
        1
        for left, right in itertools.combinations(cycles, 2)
        if left & right
    )
    ring_se_cd2 = 0
    for cycle in cycles:
        nodes = {node for edge in cycle for node in edge}
        if all(
            inorganic.degree[node] == 2
            for node in nodes
            if inorganic.nodes[node].get("element") == "Se"
        ):
            ring_se_cd2 += 1
    return {
        "six_ring_count": len(cycles),
        "six_ring_fused_pairs": fused_pairs,
        "six_ring_se_cd2_count": ring_se_cd2,
        "inorganic_components": nx.number_connected_components(inorganic)
        if inorganic_nodes
        else 0,
    }


def _bridge_descriptors(graph: nx.Graph) -> Dict[str, object]:
    cl_nodes = [node for node, data in graph.nodes(data=True) if data.get("element") == "Cl"]
    cd_nodes = [node for node, data in graph.nodes(data=True) if data.get("element") == "Cd"]
    degree_counts = Counter()
    pair_counts: Counter[Tuple[int, int]] = Counter()
    cd_bridge_counts: Counter[int] = Counter()
    for cl in cl_nodes:
        hosts = sorted(node for node in graph.neighbors(cl) if node in cd_nodes)
        degree_counts[len(hosts)] += 1
        if len(hosts) >= 2:
            for left, right in itertools.combinations(hosts, 2):
                pair_counts[(left, right)] += 1
            for host in hosts:
                cd_bridge_counts[host] += 1
    overlap = 0
    if pair_counts:
        overlap = sum(max(0, count - 1) for count in pair_counts.values())
    return {
        "cl_terminal_count": degree_counts[1],
        "cl_mu2_count": degree_counts[2],
        "cl_mu3_count": degree_counts[3],
        "cl_other_degree_count": sum(
            count for degree, count in degree_counts.items() if degree not in {1, 2, 3}
        ),
        "max_bridges_per_cd_pair": max(pair_counts.values(), default=0),
        "shared_bridge_overlap_count": overlap,
        "max_bridges_on_one_cd": max(cd_bridge_counts.values(), default=0),
    }


def _kabsch_rmsd(left: np.ndarray, right: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if left is None or right is None or left.shape != right.shape or len(left) == 0:
        return None, None
    a = left - left.mean(axis=0)
    b = right - right.mean(axis=0)
    covariance = a.T @ b
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    aligned = a @ rotation
    displacement = np.linalg.norm(aligned - b, axis=1)
    return float(np.sqrt(np.mean(displacement**2))), float(np.max(displacement))


def _geometry_descriptors(
    symbols: Sequence[str],
    edges: Sequence[Tuple[int, int]],
    coordinates: Optional[np.ndarray],
    initial: Optional[np.ndarray],
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "max_bond_A": None,
        "min_nonbond_A": None,
        "mean_cd_cl_A": None,
        "mean_bridge_angle_deg": None,
        "min_bridge_angle_deg": None,
        "rmsd_initial_relaxed_A": None,
        "max_displacement_A": None,
    }
    if coordinates is None or len(coordinates) != len(symbols):
        return result
    edge_set = {tuple(sorted(edge)) for edge in edges}
    bond_lengths: List[float] = []
    nonbond_lengths: List[float] = []
    for left in range(len(symbols)):
        for right in range(left + 1, len(symbols)):
            distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
            if (left, right) in edge_set:
                bond_lengths.append(distance)
            else:
                nonbond_lengths.append(distance)
    if bond_lengths:
        result["max_bond_A"] = max(bond_lengths)
    if nonbond_lengths:
        result["min_nonbond_A"] = min(nonbond_lengths)
    cd_cl = [
        float(np.linalg.norm(coordinates[left] - coordinates[right]))
        for left, right in edge_set
        if {symbols[left], symbols[right]} == {"Cd", "Cl"}
    ]
    if cd_cl:
        result["mean_cd_cl_A"] = sum(cd_cl) / len(cd_cl)
    bridge_angles: List[float] = []
    for ligand, symbol in enumerate(symbols):
        if symbol != "Cl":
            continue
        hosts = [node for node in range(len(symbols)) if (min(ligand, node), max(ligand, node)) in edge_set and symbols[node] == "Cd"]
        for left, right in itertools.combinations(hosts, 2):
            first = coordinates[left] - coordinates[ligand]
            second = coordinates[right] - coordinates[ligand]
            denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
            if denominator <= 1e-12:
                continue
            cosine = float(np.dot(first, second) / denominator)
            bridge_angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    if bridge_angles:
        result["mean_bridge_angle_deg"] = sum(bridge_angles) / len(bridge_angles)
        result["min_bridge_angle_deg"] = min(bridge_angles)
    rmsd, maximum = _kabsch_rmsd(initial, coordinates)
    result["rmsd_initial_relaxed_A"] = rmsd
    result["max_displacement_A"] = maximum
    return result


def _parse_energy(comment: str) -> Optional[float]:
    match = ENERGY_RE.search(comment)
    return None if match is None else float(match.group(1))


@dataclass
class Trial:
    k: int
    p: int
    trial_id: str
    start: int
    symbols: Tuple[str, ...]
    source_edges: Tuple[Tuple[int, int], ...]
    final_edges: Tuple[Tuple[int, int], ...]
    initial_coordinates: Optional[np.ndarray]
    relaxed_coordinates: Optional[np.ndarray]
    xtb_ok: bool
    xtb_converged: bool
    xtb_error: str
    initial_violations: str
    final_violations: str
    energy_eV: Optional[float]
    relaxed_comment: str
    descriptors: Dict[str, object] = field(default_factory=dict)
    delta_energy_kcal: Optional[float] = None
    source_skeleton_group: str = ""
    skeleton_changed: bool = False

    @property
    def graph_key(self) -> str:
        return f"k{self.k:03d}_p{self.p:03d}_{self.trial_id}"

    @property
    def audit_status(self) -> str:
        if not self.xtb_ok or self.relaxed_coordinates is None:
            return "xtb_failed"
        return "warning" if self.final_violations else "pass"


def _load_trials(root: Path) -> List[Trial]:
    trials: List[Trial] = []
    for csv_path in sorted(root.glob("k*/p*/motif_trials.csv")):
        try:
            k = int(csv_path.parent.parent.name[1:])
            p = int(csv_path.parent.name[1:])
        except (IndexError, ValueError):
            continue
        for row in csv.DictReader(csv_path.open(encoding="utf-8")):
            trial_id = row.get("trial_id", "")
            if not trial_id:
                continue
            bin_dir = csv_path.parent
            initial_path = _xyz_path(bin_dir, trial_id, row.get("initial_xyz", ""))
            relaxed_path = _xyz_path(bin_dir, trial_id, row.get("xtb_xyz", ""))
            initial_symbols, initial_coordinates, _ = _read_xyz(initial_path)
            relaxed_symbols, relaxed_coordinates, relaxed_comment = _read_xyz(relaxed_path)
            symbols = relaxed_symbols or initial_symbols
            energy = _parse_energy(relaxed_comment)
            trials.append(
                Trial(
                    k=k,
                    p=p,
                    trial_id=trial_id,
                    start=int(row.get("start", 0) or 0),
                    symbols=symbols,
                    source_edges=_edges(row.get("source_edges", "")),
                    final_edges=_edges(row.get("final_edges", "")),
                    initial_coordinates=initial_coordinates,
                    relaxed_coordinates=relaxed_coordinates,
                    xtb_ok=_bool(row.get("xtb_ok", "false")),
                    xtb_converged=_bool(row.get("xtb_converged", "false")),
                    xtb_error=row.get("xtb_error", ""),
                    initial_violations=row.get("initial_violations", ""),
                    final_violations=row.get("final_violations", ""),
                    energy_eV=energy,
                    relaxed_comment=relaxed_comment,
                )
            )
    return trials


def _assign_skeleton_groups(trials: Sequence[Trial]) -> None:
    """Group source Cd-Se skeletons within each composition bin."""

    by_bin: Dict[Tuple[int, int], List[Trial]] = defaultdict(list)
    for trial in trials:
        by_bin[(trial.k, trial.p)].append(trial)
    for (k, p), group in sorted(by_bin.items()):
        representatives: List[nx.Graph] = []
        for trial in group:
            source = _skeleton(_graph(trial.symbols, trial.source_edges))
            assigned = ""
            for index, representative in enumerate(representatives, start=1):
                if _lineage_bucket(source) != _lineage_bucket(representative):
                    continue
                if nx.is_isomorphic(source, representative, node_match=_node_match):
                    assigned = f"k{k:03d}_p{p:03d}_skel{index:03d}"
                    break
            if not assigned:
                representatives.append(source)
                assigned = f"k{k:03d}_p{p:03d}_skel{len(representatives):03d}"
            trial.source_skeleton_group = assigned
            final_graph = _stage_graph(trial, "final")
            trial.skeleton_changed = bool(
                final_graph is not None
                and not nx.is_isomorphic(
                    source,
                    _skeleton(final_graph),
                    node_match=_node_match,
                )
            )


def _inventory(row: Mapping[str, object], prefix: str) -> str:
    return (
        f"Cl_t={int(row.get(prefix + 'cl_terminal_count') or 0)};"
        f"Cl_mu2={int(row.get(prefix + 'cl_mu2_count') or 0)};"
        f"Cl_mu3={int(row.get(prefix + 'cl_mu3_count') or 0)}"
    )


def _descriptor_row(trial: Trial) -> Dict[str, object]:
    source_graph = _graph(trial.symbols, trial.source_edges)
    final_graph = _graph(trial.symbols, trial.final_edges or trial.source_edges)
    edges = trial.final_edges or trial.source_edges
    graph = final_graph
    cd_nodes = [node for node, data in graph.nodes(data=True) if data.get("element") == "Cd"]
    se_nodes = [node for node, data in graph.nodes(data=True) if data.get("element") == "Se"]
    source_cd_nodes = [node for node, data in source_graph.nodes(data=True) if data.get("element") == "Cd"]
    source_se_nodes = [node for node, data in source_graph.nodes(data=True) if data.get("element") == "Se"]
    source_skeleton = source_graph.subgraph(source_cd_nodes + source_se_nodes).copy()
    final_skeleton = final_graph.subgraph(
        [node for node, data in final_graph.nodes(data=True) if data.get("element") in {"Cd", "Se"}]
    ).copy()
    source_bridge_ligands = int(
        _bridge_descriptors(source_graph)["cl_mu2_count"]
        + _bridge_descriptors(source_graph)["cl_mu3_count"]
    )
    final_bridge_ligands = int(
        _bridge_descriptors(final_graph)["cl_mu2_count"]
        + _bridge_descriptors(final_graph)["cl_mu3_count"]
    )
    source_bridge_incidence = int(
        2 * _bridge_descriptors(source_graph)["cl_mu2_count"]
        + 3 * _bridge_descriptors(source_graph)["cl_mu3_count"]
    )
    final_bridge_incidence = int(
        2 * _bridge_descriptors(final_graph)["cl_mu2_count"]
        + 3 * _bridge_descriptors(final_graph)["cl_mu3_count"]
    )
    row: Dict[str, object] = {
        "k": trial.k,
        "p": trial.p,
        "trial_id": trial.trial_id,
        "graph_key": trial.graph_key,
        "source_skeleton_group": trial.source_skeleton_group,
        "skeleton_changed_after_xtb": str(trial.skeleton_changed).lower(),
        "xtb_ok": str(trial.xtb_ok).lower(),
        "xtb_converged": str(trial.xtb_converged).lower(),
        "audit_status": trial.audit_status,
        "final_violations": trial.final_violations,
        "energy_eV": trial.energy_eV,
        "delta_energy_kcal": trial.delta_energy_kcal,
        "source_edge_count": len(trial.source_edges),
        "final_edge_count": len(trial.final_edges),
        "source_skeleton_total_cn": sum(dict(source_skeleton.degree()).values()),
        "final_skeleton_total_cn": sum(dict(final_skeleton.degree()).values()),
        "source_total_cn": sum(dict(source_graph.degree()).values()),
        "final_total_cn": sum(dict(final_graph.degree()).values()),
        "source_skeleton_max_cn": max((degree for _, degree in source_skeleton.degree()), default=0),
        "final_skeleton_max_cn": max((degree for _, degree in final_skeleton.degree()), default=0),
        "source_total_bonds": len(trial.source_edges),
        "final_total_bonds": len(trial.final_edges),
        "source_max_atom_cn": max((degree for _, degree in source_graph.degree()), default=0),
        "final_max_atom_cn": max((degree for _, degree in final_graph.degree()), default=0),
        "source_bridge_ligand_count": source_bridge_ligands,
        "final_bridge_ligand_count": final_bridge_ligands,
        "source_bridge_host_incidence": source_bridge_incidence,
        "final_bridge_host_incidence": final_bridge_incidence,
        "connectivity_gained": len(set(trial.final_edges) - set(trial.source_edges)),
        "connectivity_lost": len(set(trial.source_edges) - set(trial.final_edges)),
        "cd_cn_skeleton": _degree_distribution(source_skeleton, source_cd_nodes),
        "se_cn_skeleton": _degree_distribution(source_skeleton, source_se_nodes),
        "cd_cn_final": _degree_distribution(graph, cd_nodes),
        "se_cn_final": _degree_distribution(graph, se_nodes),
    }
    row.update({f"source_{key}": value for key, value in _ring_descriptors(source_graph).items()})
    row.update({f"final_{key}": value for key, value in _ring_descriptors(final_graph).items()})
    row.update({f"source_{key}": value for key, value in _bridge_descriptors(source_graph).items()})
    row.update({f"final_{key}": value for key, value in _bridge_descriptors(final_graph).items()})
    row["source_bridge_inventory"] = _inventory(row, "source_")
    row["final_bridge_inventory"] = _inventory(row, "final_")
    row.update(
        _geometry_descriptors(
            trial.symbols,
            edges,
            trial.relaxed_coordinates,
            trial.initial_coordinates,
        )
    )
    trial.descriptors = row
    return row


def _skeleton_group_rows(trials: Sequence[Trial]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Trial]] = defaultdict(list)
    bin_totals: Counter[Tuple[int, int]] = Counter()
    for trial in trials:
        groups[trial.source_skeleton_group].append(trial)
        bin_totals[(trial.k, trial.p)] += 1
    rows: List[Dict[str, object]] = []
    for group_id, members in sorted(groups.items()):
        energies = [member.delta_energy_kcal for member in members if member.delta_energy_kcal is not None]
        first = members[0].descriptors
        rows.append(
            {
                "k": members[0].k,
                "p": members[0].p,
                "skeleton_group": group_id,
                "trial_count": len(members),
                "shared_trial_fraction_in_bin": len(members) / bin_totals[(members[0].k, members[0].p)],
                "xTB_converged": sum(member.xtb_converged for member in members),
                "audit_warnings": sum(member.audit_status == "warning" for member in members),
                "skeleton_changed_after_xtb": sum(member.skeleton_changed for member in members),
                "source_cd_cn": first.get("cd_cn_skeleton", ""),
                "source_se_cn": first.get("se_cn_skeleton", ""),
                "source_skeleton_total_cn": first.get("source_skeleton_total_cn", 0),
                "source_skeleton_max_cn": first.get("source_skeleton_max_cn", 0),
                "source_total_bonds": first.get("source_total_bonds", 0),
                "source_six_ring_count": first.get("source_six_ring_count", 0),
                "source_six_ring_se_cd2_count": first.get("source_six_ring_se_cd2_count", 0),
                "energy_min_delta_kcal": min(energies) if energies else "",
                "energy_max_delta_kcal": max(energies) if energies else "",
                "energy_mean_delta_kcal": (sum(energies) / len(energies)) if energies else "",
            }
        )
    return rows


def _decoration_transition_rows(trials: Sequence[Trial]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[int, int, str, str, str, bool], List[Trial]] = defaultdict(list)
    for trial in trials:
        row = trial.descriptors
        key = (
            trial.k,
            trial.p,
            trial.source_skeleton_group,
            str(row.get("source_bridge_inventory", "")),
            str(row.get("final_bridge_inventory", "")),
            trial.skeleton_changed,
        )
        groups[key].append(trial)
    rows: List[Dict[str, object]] = []
    for (k, p, skeleton_group, source_inventory, final_inventory, changed), members in sorted(groups.items()):
        energies = [member.delta_energy_kcal for member in members if member.delta_energy_kcal is not None]
        rows.append(
            {
                "k": k,
                "p": p,
                "skeleton_group": skeleton_group,
                "source_bridge_inventory": source_inventory,
                "final_bridge_inventory": final_inventory,
                "skeleton_changed_after_xtb": str(changed).lower(),
                "trial_count": len(members),
                "xTB_converged": sum(member.xtb_converged for member in members),
                "audit_warnings": sum(member.audit_status == "warning" for member in members),
                "energy_min_delta_kcal": min(energies) if energies else "",
                "energy_mean_delta_kcal": (sum(energies) / len(energies)) if energies else "",
                "energy_max_delta_kcal": max(energies) if energies else "",
                "representative_trials": "|".join(member.trial_id for member in members[:8]),
            }
        )
    return rows


def _count_transition_rows(
    trials: Sequence[Trial],
    source_field: str,
    final_field: str,
    name: str,
) -> List[Dict[str, object]]:
    groups: Dict[Tuple[int, int, object, object], List[Trial]] = defaultdict(list)
    for trial in trials:
        groups[
            (
                trial.k,
                trial.p,
                trial.descriptors.get(source_field, ""),
                trial.descriptors.get(final_field, ""),
            )
        ].append(trial)
    rows: List[Dict[str, object]] = []
    for (k, p, source_value, final_value), members in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1], str(item[0][2]), str(item[0][3]))
    ):
        energies = [member.delta_energy_kcal for member in members if member.delta_energy_kcal is not None]
        rows.append(
            {
                "k": k,
                "p": p,
                "transition": name,
                "source_value": source_value,
                "final_value": final_value,
                "count": len(members),
                "low_energy_count_delta_le_3_kcal": sum(
                    member.delta_energy_kcal is not None and member.delta_energy_kcal <= 3.0
                    for member in members
                ),
                "xTB_converged": sum(member.xtb_converged for member in members),
                "audit_warnings": sum(member.audit_status == "warning" for member in members),
                "energy_min_delta_kcal": min(energies) if energies else "",
                "energy_mean_delta_kcal": (sum(energies) / len(energies)) if energies else "",
                "energy_max_delta_kcal": max(energies) if energies else "",
                "representative_trials": "|".join(member.trial_id for member in members[:8]),
            }
        )
    return rows


def _skeleton(graph: nx.Graph) -> nx.Graph:
    nodes = [node for node, data in graph.nodes(data=True) if data.get("element") in {"Cd", "Se"}]
    return graph.subgraph(nodes).copy()


def _stage_graph(trial: Trial, stage: str) -> Optional[nx.Graph]:
    edges = trial.final_edges if stage == "final" else trial.source_edges
    if not edges or not trial.symbols:
        return None
    return _graph(trial.symbols, edges)


def _best_cd_cl_overlap(
    parent_graph: nx.Graph,
    child_graph: nx.Graph,
    parent_skeleton: nx.Graph,
    child_skeleton: nx.Graph,
) -> Tuple[int, int, int, float]:
    """Compare ligand incidence after every valid skeleton symmetry mapping."""

    parent_cl = [
        node for node, data in parent_graph.nodes(data=True)
        if data.get("element") == "Cl"
    ]
    child_cl = [
        node for node, data in child_graph.nodes(data=True)
        if data.get("element") == "Cl"
    ]
    parent_cd = {
        node for node, data in parent_graph.nodes(data=True)
        if data.get("element") == "Cd"
    }
    child_cd = {
        node for node, data in child_graph.nodes(data=True)
        if data.get("element") == "Cd"
    }
    parent_edge_count = sum(
        1 for cl in parent_cl for cd in parent_cd if parent_graph.has_edge(cl, cd)
    )
    child_edge_count = sum(
        1 for cl in child_cl for cd in child_cd if child_graph.has_edge(cl, cd)
    )
    if not parent_cl or not child_cl:
        overlap = min(parent_edge_count, child_edge_count)
        changed = parent_edge_count + child_edge_count - 2 * overlap
        fraction = overlap / max(parent_edge_count, child_edge_count, 1)
        return overlap, changed, parent_edge_count + child_edge_count, fraction

    best_overlap = 0
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        parent_skeleton,
        child_skeleton,
        node_match=_node_match,
    )
    for mapping_index, mapping in enumerate(matcher.isomorphisms_iter()):
        if mapping_index >= MAX_SYMMETRY_MAPPINGS:
            break
        weights: Dict[Tuple[Tuple[str, int], Tuple[str, int]], int] = {}
        for parent_index, parent_ligand in enumerate(parent_cl):
            mapped_hosts = {
                mapping[host]
                for host in parent_graph.neighbors(parent_ligand)
                if host in parent_cd and host in mapping
            }
            for child_index, child_ligand in enumerate(child_cl):
                child_hosts = {
                    host
                    for host in child_graph.neighbors(child_ligand)
                    if host in child_cd
                }
                weights[(('p', parent_index), ('c', child_index))] = len(
                    mapped_hosts & child_hosts
                )
        # The Cl counts are small (at most 2p in the intended runs).  A
        # bitmask assignment is both deterministic and much cheaper here than
        # constructing a general blossom matching graph for every symmetry
        # mapping.
        scores = [
            [weights[(('p', i), ('c', j))] for j in range(len(child_cl))]
            for i in range(len(parent_cl))
        ]
        assignments = {0: 0}
        for row in scores:
            updated = dict(assignments)
            for mask, score in assignments.items():
                for column, value in enumerate(row):
                    if mask & (1 << column):
                        continue
                    next_mask = mask | (1 << column)
                    updated[next_mask] = max(
                        updated.get(next_mask, -1), score + value
                    )
            assignments = updated
        overlap = max(assignments.values(), default=0)
        best_overlap = max(best_overlap, int(overlap))
    changed = parent_edge_count + child_edge_count - 2 * best_overlap
    fraction = best_overlap / max(parent_edge_count, child_edge_count, 1)
    return best_overlap, changed, parent_edge_count + child_edge_count, fraction


def _lineage_matches(
    child: Trial,
    stage: str,
    parents_by_skeleton: Mapping[object, Sequence[Tuple[Trial, nx.Graph]]],
) -> Iterator[Dict[str, object]]:
    child_graph = _stage_graph(child, stage)
    if child_graph is None:
        return
    child_skeleton = _skeleton(child_graph)
    child_se = [node for node, data in child_skeleton.nodes(data=True) if data.get("element") == "Se"]
    child_cd = [node for node, data in child_skeleton.nodes(data=True) if data.get("element") == "Cd"]
    if len(child_se) < 2 or len(child_cd) < 2:
        return
    for removed_se, removed_cd in itertools.product(child_se, child_cd):
        retained = [
            node for node in child_skeleton if node not in {removed_se, removed_cd}
        ]
        reduced = child_skeleton.subgraph(retained).copy()
        candidates = parents_by_skeleton.get(_lineage_key(reduced), ())
        if not candidates:
            continue
        reduced_full_nodes = [
            node for node in child_graph if node not in {removed_se, removed_cd}
        ]
        reduced_full = child_graph.subgraph(reduced_full_nodes)
        for parent, parent_graph in candidates:
            parent_skeleton = _skeleton(parent_graph)
            if (
                len(child_se)
                != sum(
                    data.get("element") == "Se"
                    for _, data in parent_skeleton.nodes(data=True)
                )
                + 1
                or len(child_cd)
                != sum(
                    data.get("element") == "Cd"
                    for _, data in parent_skeleton.nodes(data=True)
                )
                + 1
            ):
                continue
            # The bucket is a fast prefilter; this exact check prevents bucket
            # collisions from being reported as lineage.
            if not nx.is_isomorphic(
                parent_skeleton, reduced, node_match=_node_match
            ):
                continue
            # Full decorated graphs contain many interchangeable terminal Cl
            # leaves.  A canonical proof for those graphs is disproportionately
            # expensive, so retain a typed incidence bucket as a diagnostic
            # candidate and reserve exact lineage for the inorganic skeleton.
            decorated_bucket_match = (
                _lineage_bucket(parent_graph) == _lineage_bucket(reduced_full)
            )
            overlap, changed, total_cd_cl_edges, overlap_fraction = _best_cd_cl_overlap(
                parent_graph,
                reduced_full,
                parent_skeleton,
                reduced,
            )
            yield {
                "stage": stage,
                "parent_graph": parent.graph_key,
                "child_graph": child.graph_key,
                "removed_child_se": removed_se,
                "removed_child_cd": removed_cd,
                "exact_inorganic_lineage": True,
                "decorated_bucket_match": decorated_bucket_match,
                "cd_cl_overlap_edges": overlap,
                "cd_cl_changed_edges": changed,
                "cd_cl_total_edges": total_cd_cl_edges,
                "cd_cl_overlap_fraction": overlap_fraction,
                "parent_energy_eV": parent.energy_eV,
                "child_energy_eV": child.energy_eV,
                "parent_audit_status": parent.audit_status,
                "child_audit_status": child.audit_status,
            }


def _lineage_rows(trials: Sequence[Trial]) -> List[Dict[str, object]]:
    by_bin: Dict[Tuple[int, int], List[Trial]] = defaultdict(list)
    for trial in trials:
        by_bin[(trial.k, trial.p)].append(trial)
    rows: List[Dict[str, object]] = []
    for (k, p), children in sorted(by_bin.items()):
        parents = by_bin.get((k - 1, p), [])
        if not parents:
            continue
        for stage in ("source", "final"):
            parent_catalog: Dict[object, List[Tuple[Trial, nx.Graph]]] = defaultdict(list)
            for parent in parents:
                parent_graph = _stage_graph(parent, stage)
                if parent_graph is None:
                    continue
                parent_catalog[_lineage_key(_skeleton(parent_graph))].append(
                    (parent, parent_graph)
                )
            for child in children:
                matches = list(_lineage_matches(child, stage, parent_catalog) or [])
                if matches:
                    rows.extend(matches)
                else:
                    rows.append(
                        {
                            "stage": stage,
                            "parent_graph": "",
                            "child_graph": child.graph_key,
                            "removed_child_se": "",
                            "removed_child_cd": "",
                            "exact_inorganic_lineage": False,
                            "decorated_bucket_match": False,
                            "cd_cl_overlap_edges": "",
                            "cd_cl_changed_edges": "",
                            "cd_cl_total_edges": "",
                            "cd_cl_overlap_fraction": "",
                            "parent_energy_eV": "",
                            "child_energy_eV": child.energy_eV,
                            "parent_audit_status": "",
                            "child_audit_status": child.audit_status,
                        }
                    )
    return rows


def _energy_normalize(trials: Sequence[Trial]) -> None:
    by_bin: Dict[Tuple[int, int], List[Trial]] = defaultdict(list)
    for trial in trials:
        if trial.energy_eV is not None:
            by_bin[(trial.k, trial.p)].append(trial)
    for group in by_bin.values():
        reference = min(float(trial.energy_eV) for trial in group if trial.energy_eV is not None)
        for trial in group:
            trial.delta_energy_kcal = (float(trial.energy_eV) - reference) * EV_TO_KCAL


def _filter_predicates(row: Mapping[str, object]) -> Dict[str, bool]:
    return {
        "source_has_six_ring": int(row.get("source_six_ring_count") or 0) >= 1,
        "source_has_se_cd2_six_ring": int(row.get("source_six_ring_se_cd2_count") or 0) >= 1,
        "final_has_six_ring": int(row.get("final_six_ring_count") or 0) >= 1,
        "final_has_se_cd2_six_ring": int(row.get("final_six_ring_se_cd2_count") or 0) >= 1,
        "source_has_mu2_bridge": int(row.get("source_cl_mu2_count") or 0) >= 1,
        "final_has_mu2_bridge": int(row.get("final_cl_mu2_count") or 0) >= 1,
        "source_no_shared_bridge_pair": int(row.get("source_max_bridges_per_cd_pair") or 0) <= 1,
        "final_no_shared_bridge_pair": int(row.get("final_max_bridges_per_cd_pair") or 0) <= 1,
        "source_no_mu3_host_overlap": int(row.get("source_shared_bridge_overlap_count") or 0) == 0,
        "final_no_mu3_host_overlap": int(row.get("final_shared_bridge_overlap_count") or 0) == 0,
        "cd_cn_at_most_4": max(
            [int(value.split(":", 1)[0]) for value in str(row.get("cd_cn_final") or "").split(";") if value]
            or [0]
        )
        <= 4,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(trials: Sequence[Trial]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[int, int], List[Trial]] = defaultdict(list)
    for trial in trials:
        groups[(trial.k, trial.p)].append(trial)
    rows: List[Dict[str, object]] = []
    for (k, p), group in sorted(groups.items()):
        energies = [trial.delta_energy_kcal for trial in group if trial.delta_energy_kcal is not None]
        for predicate in sorted(_filter_predicates(group[0].descriptors)) if group else []:
            values = [_filter_predicates(trial.descriptors)[predicate] for trial in group]
            selected = [trial for trial, value in zip(group, values) if value]
            low = [trial for trial in group if trial.delta_energy_kcal is not None and trial.delta_energy_kcal <= 3.0]
            selected_low = [trial for trial in selected if trial.delta_energy_kcal is not None and trial.delta_energy_kcal <= 3.0]
            rows.append(
                {
                    "k": k,
                    "p": p,
                    "predicate": predicate,
                    "trials": len(group),
                    "selected": len(selected),
                    "selected_fraction": len(selected) / len(group) if group else 0.0,
                    "energy_trials": len(energies),
                    "low_energy_trials_delta_le_3_kcal": len(low),
                    "low_energy_selected": len(selected_low),
                    "low_energy_retention": len(selected_low) / len(low) if low else "",
                    "selected_warnings": sum(trial.audit_status == "warning" for trial in selected),
                    "selected_converged": sum(trial.xtb_converged for trial in selected),
                }
            )
    return rows


def _report(
    root: Path,
    output: Path,
    trials: Sequence[Trial],
    skeleton_groups: Sequence[Mapping[str, object]],
    lineage: Sequence[Mapping[str, object]],
    transitions: Sequence[Mapping[str, object]],
    bond_transitions: Sequence[Mapping[str, object]],
    bridge_transitions: Sequence[Mapping[str, object]],
) -> None:
    groups: Dict[Tuple[int, int], List[Trial]] = defaultdict(list)
    for trial in trials:
        groups[(trial.k, trial.p)].append(trial)
    lines = [
        "# Molecular run analysis",
        "",
        f"Input: `{root}`",
        f"Trials with xTB coordinates: **{sum(trial.relaxed_coordinates is not None for trial in trials)}**",
        f"Total trial records: **{len(trials)}**",
        "",
        "## Bins",
        "",
        "| k | p | trials | xTB converged | audit warnings | source six-rings | final six-rings | energy range (kcal/mol) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, group in sorted(groups.items()):
        energies = [trial.delta_energy_kcal for trial in group if trial.delta_energy_kcal is not None]
        lines.append(
            f"| {key[0]} | {key[1]} | {len(group)} | "
            f"{sum(trial.xtb_converged for trial in group)} | "
            f"{sum(trial.audit_status == 'warning' for trial in group)} | "
            f"{sum(int(trial.descriptors.get('source_six_ring_count') or 0) > 0 for trial in group)} | "
            f"{sum(int(trial.descriptors.get('final_six_ring_count') or 0) > 0 for trial in group)} | "
            f"{('%.2f' % (max(energies) - min(energies))) if energies else '-'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Candidate filters are descriptive only; they were not applied to graph construction.",
            "Source six-ring predicates describe the constructed graph; final six-ring predicates describe the distance-inferred xTB graph. They are not interchangeable.",
            "Raw xTB energies are compared only within one `(k,p)` bin; the reported energy is relative to that bin's minimum.",
            "",
            "## Shared inorganic skeletons",
            "",
            "A skeleton group is an exact isomorphism class of the source Cd–Se graph within one `(k,p)` bin. The same group can therefore carry several chloride decorations.",
            "`source_skeleton_total_cn` is the sum of Cd/Se degrees in the Cd–Se subgraph; `source_total_cn` includes Cl decoration; `source_total_bonds` is the full graph edge count at construction. The corresponding final columns use distance-inferred xTB connectivity.",
            "",
            "| k | p | skeleton group | trials sharing skeleton | fraction of bin | source Cd CN | source Se CN | skeleton total CN | skeleton max CN | total bonds | changed after xTB |",
            "|---:|---:|---|---:|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        skeleton_groups,
        key=lambda item: (
            -int(item.get("trial_count", 0)),
            int(item.get("k", 0)),
            int(item.get("p", 0)),
            str(item.get("skeleton_group", "")),
        ),
    )[:40]:
        lines.append(
            f"| {row['k']} | {row['p']} | {row['skeleton_group']} | {row['trial_count']} | "
            f"{float(row['shared_trial_fraction_in_bin']):.3f} | {row['source_cd_cn']} | {row['source_se_cn']} | "
            f"{row['source_skeleton_total_cn']} | {row['source_skeleton_max_cn']} | {row['source_total_bonds']} | "
            f"{row['skeleton_changed_after_xtb']} |"
        )
    lines.extend(
        [
            "",
            "## Most common decoration transitions",
            "",
            "| k | p | source inventory | final inventory | count | mean ΔE (kcal/mol) | warnings |",
            "|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(
        transitions,
        key=lambda item: (
            -int(item.get("trial_count", 0)),
            float(item.get("energy_mean_delta_kcal") or 1e9),
        ),
    )[:20]:
        lines.append(
            f"| {row['k']} | {row['p']} | {row['source_bridge_inventory']} | "
            f"{row['final_bridge_inventory']} | {row['trial_count']} | "
            f"{row['energy_mean_delta_kcal'] if row['energy_mean_delta_kcal'] != '' else '-'} | "
            f"{row['audit_warnings']} |"
        )
    lines.extend(
        [
            "",
            "## Bond-count and bridge-count transitions",
            "",
            "| type | k | p | source | final | count | low-energy | mean ΔE (kcal/mol) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        list(bond_transitions) + list(bridge_transitions),
        key=lambda item: (-int(item.get("count", 0)), str(item.get("transition", ""))),
    )[:30]:
        lines.append(
            f"| {row['transition']} | {row['k']} | {row['p']} | "
            f"{row['source_value']} | {row['final_value']} | {row['count']} | "
            f"{row['low_energy_count_delta_le_3_kcal']} | "
            f"{row['energy_mean_delta_kcal'] if row['energy_mean_delta_kcal'] != '' else '-'} |"
        )
    lines.extend(
        [
            "",
            "## Lineage",
            "",
        ]
    )
    if not lineage:
        lines.append("- Cross-bin parent/child lineage matching was not run; use `--lineage` when that expensive comparison is needed.")
    for stage in ("source", "final"):
        stage_rows = [row for row in lineage if row.get("stage") == stage]
        children = {row.get("child_graph") for row in stage_rows}
        matched = {
            row.get("child_graph")
            for row in stage_rows
            if row.get("exact_inorganic_lineage") is True or str(row.get("exact_inorganic_lineage")).lower() == "true"
        }
        decorated_candidates = {
            row.get("child_graph")
            for row in stage_rows
            if row.get("decorated_bucket_match") is True or str(row.get("decorated_bucket_match")).lower() == "true"
        }
        lines.append(
            f"- `{stage}` graphs: {len(matched)}/{len(children)} have an exact inorganic parent; "
            f"{len(decorated_candidates)}/{len(children)} have a decorated-incidence bucket match."
        )
    lines.append("")
    (output / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(root: Path, output: Path, *, include_lineage: bool = False) -> int:
    trials = _load_trials(root)
    if not trials:
        raise SystemExit(f"no k*/p*/motif_trials.csv files found below {root}")
    _assign_skeleton_groups(trials)
    _energy_normalize(trials)
    descriptor_rows = [_descriptor_row(trial) for trial in trials]
    transition_rows = _decoration_transition_rows(trials)
    bond_transition_rows = _count_transition_rows(
        trials,
        "source_total_bonds",
        "final_total_bonds",
        "total_bonds",
    )
    cn_transition_rows = _count_transition_rows(
        trials,
        "source_max_atom_cn",
        "final_max_atom_cn",
        "max_atom_cn",
    )
    skeleton_total_cn_transition_rows = _count_transition_rows(
        trials,
        "source_skeleton_total_cn",
        "final_skeleton_total_cn",
        "skeleton_total_cn",
    )
    skeleton_max_cn_transition_rows = _count_transition_rows(
        trials,
        "source_skeleton_max_cn",
        "final_skeleton_max_cn",
        "skeleton_max_cn",
    )
    total_cn_transition_rows = _count_transition_rows(
        trials,
        "source_total_cn",
        "final_total_cn",
        "total_cn",
    )
    bridge_transition_rows = _count_transition_rows(
        trials,
        "source_bridge_ligand_count",
        "final_bridge_ligand_count",
        "bridge_ligands",
    )
    bridge_incidence_transition_rows = _count_transition_rows(
        trials,
        "source_bridge_host_incidence",
        "final_bridge_host_incidence",
        "bridge_host_incidence",
    )
    lineage_rows = _lineage_rows(trials) if include_lineage else []
    output.mkdir(parents=True, exist_ok=True)
    descriptor_fields = sorted({key for row in descriptor_rows for key in row})
    _write_csv(output / "trial_descriptors.csv", descriptor_rows, descriptor_fields)
    lineage_fields = sorted({key for row in lineage_rows for key in row}) or [
        "stage", "parent_graph", "child_graph"
    ]
    _write_csv(output / "lineage.csv", lineage_rows, lineage_fields)
    group_rows = _skeleton_group_rows(trials)
    _write_csv(
        output / "skeleton_groups.csv",
        group_rows,
        sorted({key for row in group_rows for key in row}) or ["skeleton_group"],
    )
    _write_csv(
        output / "decoration_transitions.csv",
        transition_rows,
        sorted({key for row in transition_rows for key in row}) or ["skeleton_group"],
    )
    _write_csv(
        output / "bond_count_transitions.csv",
        bond_transition_rows,
        sorted({key for row in bond_transition_rows for key in row}) or ["transition"],
    )
    _write_csv(
        output / "coordination_transitions.csv",
        cn_transition_rows
        + skeleton_total_cn_transition_rows
        + skeleton_max_cn_transition_rows
        + total_cn_transition_rows,
        sorted(
            {
                key
                for row in cn_transition_rows + skeleton_total_cn_transition_rows + skeleton_max_cn_transition_rows
                for key in row
            }
        ) or ["transition"],
    )
    _write_csv(
        output / "bridge_count_transitions.csv",
        bridge_transition_rows + bridge_incidence_transition_rows,
        sorted({key for row in bridge_transition_rows + bridge_incidence_transition_rows for key in row}) or ["transition"],
    )
    summary_rows = _summary_rows(trials)
    _write_csv(output / "filter_candidates.csv", summary_rows, sorted({key for row in summary_rows for key in row}) or ["k", "p"])
    _report(
        root,
        output,
        trials,
        group_rows,
        lineage_rows,
        transition_rows,
        bond_transition_rows
        + cn_transition_rows
        + skeleton_total_cn_transition_rows
        + skeleton_max_cn_transition_rows
        + total_cn_transition_rows,
        bridge_transition_rows + bridge_incidence_transition_rows,
    )
    print(f"analyzed {len(trials)} trial records")
    print(f"wrote {output / 'trial_descriptors.csv'}")
    print(f"wrote {output / 'lineage.csv'}")
    print(f"wrote {output / 'skeleton_groups.csv'}")
    print(f"wrote {output / 'decoration_transitions.csv'}")
    print(f"wrote {output / 'bond_count_transitions.csv'}")
    print(f"wrote {output / 'coordination_transitions.csv'}")
    print(f"wrote {output / 'bridge_count_transitions.csv'}")
    print(f"wrote {output / 'filter_candidates.csv'}")
    print(f"wrote {output / 'analysis_report.md'}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="saved molecular run directory")
    parser.add_argument("--output", required=True, type=Path, help="analysis output directory")
    parser.add_argument(
        "--lineage",
        action="store_true",
        help="also run the slower parent-child lineage comparison",
    )
    args = parser.parse_args(argv)
    return analyze(
        args.root.expanduser().resolve(),
        args.output.expanduser().resolve(),
        include_lineage=args.lineage,
    )


if __name__ == "__main__":
    raise SystemExit(main())
