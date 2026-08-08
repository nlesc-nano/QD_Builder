#!/usr/bin/env python3
"""Extract and analyse relaxed CP2K nucleation structures.

Run from the directory containing ``runs/``::

    python analyze_cp2k_results.py --root runs/cdse_map/dft_all

Merge several calculation trees into one analysis::

    python analyze_cp2k_results.py \
        --root runs/cdse_map/dft_all \
        --root runs/cdse_map/dft_k5_partial \
        --root runs/cdse_map/dft_k6_additional \
        --output runs/cdse_map/analysis_all_dft

Scan several experimental chemical-potential shifts (values are eV per
formula unit)::

    python analyze_cp2k_results.py --root runs/cdse_map/dft_all \
        --delta-mu-cdse-ev -0.4 -0.2 0.0 0.2 \
        --delta-mu-cdcl2-ev -0.6 -0.3 0.0 0.3

The script reads the last complete frame of every ``CdSe-pos-1.xyz``, takes its
energy from the XYZ comment (falling back to ``cp2k_job.out``), writes clean
relaxed XYZ files, copies the complete geometry trajectories, and compares
isomers only within the same ``(k, p)`` formula.  It also constructs the
chemical-potential-dependent grand-potential surface and its p-minimized
nucleation path.  It is dependency-free; plots are added when matplotlib is
available.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import math
from pathlib import Path
import re
import shutil
from statistics import mean
from typing import Iterable, Mapping, Optional, Sequence


HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL_MOL = 627.5094740631
DEFAULT_MU_CDSE0_HARTREE = -55.539224
DEFAULT_CDSE_ENERGY_HARTREE = -55.4099660628
DEFAULT_CDCL2_ENERGY_HARTREE = -76.0795339478
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
ENERGY_COMMENT_RE = re.compile(rf"(?:^|[,\s])E\s*=\s*({FLOAT_PATTERN})")
STEP_COMMENT_RE = re.compile(r"(?:^|[,\s])i\s*=\s*(-?\d+)")
BIN_RE = re.compile(r"^(?P<axis>[kp])(?P<value>\d+)$")

Atom = tuple[str, float, float, float]
Edge = tuple[int, int]


def as_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def read_last_xyz_frame(path: Path) -> dict[str, object]:
    """Return the last complete XYZ frame, retaining a prior frame if truncated."""

    last: Optional[dict[str, object]] = None
    frames = 0
    truncated = False
    with path.open(encoding="utf-8", errors="replace") as handle:
        while True:
            line = handle.readline()
            while line and not line.strip():
                line = handle.readline()
            if not line:
                break
            try:
                count = int(line.strip())
            except ValueError:
                truncated = True
                break
            comment = handle.readline()
            if not comment:
                truncated = True
                break
            atoms: list[Atom] = []
            complete = True
            for _ in range(count):
                atom_line = handle.readline()
                fields = atom_line.split()
                if len(fields) < 4:
                    complete = False
                    break
                try:
                    atoms.append(
                        (
                            fields[0],
                            as_float(fields[1]),
                            as_float(fields[2]),
                            as_float(fields[3]),
                        )
                    )
                except ValueError:
                    complete = False
                    break
            if not complete:
                truncated = True
                break
            energy_match = ENERGY_COMMENT_RE.search(comment)
            step_match = STEP_COMMENT_RE.search(comment)
            last = {
                "atoms": atoms,
                "comment": comment.strip(),
                "energy_hartree": (
                    as_float(energy_match.group(1)) if energy_match else None
                ),
                "step": int(step_match.group(1)) if step_match else frames,
            }
            frames += 1
    if last is None:
        raise ValueError(f"no complete XYZ frame in {path}")
    last["frame_count"] = frames
    last["trajectory_truncated"] = truncated
    return last


def read_single_xyz(path: Path) -> list[Atom]:
    return list(read_last_xyz_frame(path)["atoms"])  # type: ignore[arg-type]


def parse_cp2k_output(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "program_ended": False,
        "geometry_converged": False,
        "max_geo_steps_reached": False,
        "scf_converged_count": 0,
        "scf_nonconverged_count": 0,
        "final_scf_converged": None,
        "output_energy_hartree": None,
    }
    if not path.is_file():
        return result
    last_energy: Optional[float] = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            upper = line.upper()
            if "PROGRAM ENDED AT" in upper:
                result["program_ended"] = True
            if "GEOMETRY OPTIMIZATION COMPLETED" in upper:
                result["geometry_converged"] = True
            if (
                "MAXIMUM NUMBER OF OPTIMIZATION STEPS" in upper
                or "MAXIMUM NUMBER OF GEO_OPT" in upper
            ):
                result["max_geo_steps_reached"] = True
            if "SCF RUN NOT CONVERGED" in upper:
                result["scf_nonconverged_count"] = (
                    int(result["scf_nonconverged_count"]) + 1
                )
                result["final_scf_converged"] = False
            elif "SCF RUN CONVERGED IN" in upper:
                result["scf_converged_count"] = (
                    int(result["scf_converged_count"]) + 1
                )
                result["final_scf_converged"] = True
            if "ENERGY|" in upper and "TOTAL FORCE_EVAL" in upper:
                numbers = re.findall(FLOAT_PATTERN, line)
                if numbers:
                    last_energy = as_float(numbers[-1])
    result["output_energy_hartree"] = last_energy
    return result


def kp_from_path(path: Path) -> tuple[int, int]:
    values: dict[str, int] = {}
    for part in path.parts:
        match = BIN_RE.match(part)
        if match:
            values[match.group("axis")] = int(match.group("value"))
    if set(values) != {"k", "p"}:
        raise ValueError(f"cannot infer k and p from {path}")
    return values["k"], values["p"]


def discover_runs(root: Path) -> list[dict[str, object]]:
    manifest = root / "manifest.tsv"
    entries: list[dict[str, object]] = []
    if manifest.is_file():
        with manifest.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                run_dir = root / row["run_dir"]
                entries.append(
                    {
                        "index": int(row["index"]),
                        "k": int(row["k"]),
                        "p": int(row["p"]),
                        "structure_id": row["structure_id"],
                        "box_angstrom": row.get("box_angstrom", ""),
                        "run_dir": run_dir,
                        "source_xyz": row.get("source_xyz", ""),
                    }
                )
        return entries

    for index, run_dir in enumerate(sorted(root.glob("k*/p*/*"))):
        if not run_dir.is_dir():
            continue
        k, p = kp_from_path(run_dir)
        entries.append(
            {
                "index": index,
                "k": k,
                "p": p,
                "structure_id": run_dir.name,
                "box_angstrom": "",
                "run_dir": run_dir,
                "source_xyz": "",
            }
        )
    return entries


def distance(left: Atom, right: Atom) -> float:
    return math.sqrt(sum((left[axis] - right[axis]) ** 2 for axis in (1, 2, 3)))


def vector_from(center: Atom, neighbor: Atom) -> tuple[float, float, float]:
    return tuple(neighbor[axis] - center[axis] for axis in (1, 2, 3))  # type: ignore[return-value]


def vector_dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(vector_dot(vector, vector))


def angle_degrees(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    denominator = vector_norm(left) * vector_norm(right)
    if denominator <= 1.0e-15:
        return None
    cosine = max(-1.0, min(1.0, vector_dot(left, right) / denominator))
    return math.degrees(math.acos(cosine))


def tetrahedral_order(vectors: Sequence[Sequence[float]]) -> Optional[float]:
    """Return the conventional q tetrahedral order parameter (ideal q=1)."""

    if len(vectors) != 4:
        return None
    penalty = 0.0
    for left in range(4):
        for right in range(left + 1, 4):
            angle = angle_degrees(vectors[left], vectors[right])
            if angle is None:
                return None
            penalty += (math.cos(math.radians(angle)) + 1.0 / 3.0) ** 2
    return 1.0 - 3.0 * penalty / 8.0


def point_plane_distance(
    point: Sequence[float],
    first: Sequence[float],
    second: Sequence[float],
    third: Sequence[float],
) -> Optional[float]:
    left = tuple(second[i] - first[i] for i in range(3))
    right = tuple(third[i] - first[i] for i in range(3))
    normal = (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )
    norm = vector_norm(normal)
    if norm <= 1.0e-15:
        return None
    displacement = tuple(point[i] - first[i] for i in range(3))
    return abs(vector_dot(displacement, normal)) / norm


def infer_edges(
    atoms: Sequence[Atom], cd_se_cutoff: float, cd_cl_cutoff: float
) -> set[Edge]:
    edges: set[Edge] = set()
    for left in range(len(atoms)):
        for right in range(left + 1, len(atoms)):
            pair = frozenset((atoms[left][0], atoms[right][0]))
            cutoff: Optional[float] = None
            if pair == frozenset(("Cd", "Se")):
                cutoff = cd_se_cutoff
            elif pair == frozenset(("Cd", "Cl")):
                cutoff = cd_cl_cutoff
            if cutoff is not None and distance(atoms[left], atoms[right]) <= cutoff:
                edges.add((left, right))
    return edges


def adjacency(node_count: int, edges: Iterable[Edge]) -> list[set[int]]:
    graph = [set() for _ in range(node_count)]
    for left, right in edges:
        graph[left].add(right)
        graph[right].add(left)
    return graph


def cycle_node_sets(graph: Sequence[set[int]], length: int) -> set[frozenset[Edge]]:
    """Enumerate undirected simple cycles as unique edge sets."""

    cycles: set[frozenset[Edge]] = set()
    for start in range(len(graph)):
        stack: list[tuple[int, tuple[int, ...], frozenset[int]]] = [
            (start, (start,), frozenset((start,)))
        ]
        while stack:
            current, path, used = stack.pop()
            if len(path) == length:
                if start in graph[current]:
                    cycle_edges: list[Edge] = []
                    for index in range(length - 1):
                        a, b = path[index], path[index + 1]
                        cycle_edges.append((min(a, b), max(a, b)))
                    cycle_edges.append((min(current, start), max(current, start)))
                    cycles.add(frozenset(cycle_edges))
                continue
            for neighbor in graph[current]:
                if neighbor == start or neighbor in used:
                    continue
                stack.append((neighbor, path + (neighbor,), used | {neighbor}))
    return cycles


def cycle_nodes(cycle: frozenset[Edge]) -> set[int]:
    return {node for edge in cycle for node in edge}


def topology_fingerprint(atoms: Sequence[Atom], edges: set[Edge]) -> str:
    """Stable WL-style labelled-graph fingerprint for relaxed topology families."""

    graph = adjacency(len(atoms), edges)
    labels = [atom[0] for atom in atoms]
    for _ in range(max(4, len(atoms))):
        refined = []
        for node, label in enumerate(labels):
            payload = label + "|" + "|".join(sorted(labels[n] for n in graph[node]))
            refined.append(hashlib.sha256(payload.encode()).hexdigest()[:20])
        if refined == labels:
            break
        labels = refined
    edge_labels = sorted(
        (min(labels[left], labels[right]), max(labels[left], labels[right]))
        for left, right in edges
    )
    payload = repr((sorted(labels), edge_labels)).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def structure_descriptors(
    atoms: Sequence[Atom],
    *,
    cd_se_cutoff: float,
    cd_cl_cutoff: float,
    unexpected_cutoff: float,
) -> tuple[dict[str, object], set[Edge]]:
    edges = infer_edges(atoms, cd_se_cutoff, cd_cl_cutoff)
    graph = adjacency(len(atoms), edges)
    symbols = [atom[0] for atom in atoms]
    counts = {symbol: symbols.count(symbol) for symbol in ("Cd", "Se", "Cl")}
    pair_counts = {"Cd_Se_bonds": 0, "Cd_Cl_bonds": 0}
    for left, right in edges:
        pair = frozenset((symbols[left], symbols[right]))
        if pair == frozenset(("Cd", "Se")):
            pair_counts["Cd_Se_bonds"] += 1
        elif pair == frozenset(("Cd", "Cl")):
            pair_counts["Cd_Cl_bonds"] += 1

    descriptor: dict[str, object] = {
        "atom_count": len(atoms),
        "Cd_count": counts["Cd"],
        "Se_count": counts["Se"],
        "Cl_count": counts["Cl"],
        "total_bonds": len(edges),
        **pair_counts,
    }
    targets = {"Cd": 4, "Se": 4, "Cl": 2}
    for symbol in ("Cd", "Se", "Cl"):
        degrees = [len(graph[index]) for index, value in enumerate(symbols) if value == symbol]
        descriptor[f"mean_CN_{symbol}"] = mean(degrees) if degrees else 0.0
        descriptor[f"min_CN_{symbol}"] = min(degrees) if degrees else 0
        descriptor[f"max_CN_{symbol}"] = max(degrees) if degrees else 0
        descriptor[f"CN_deficit_{symbol}"] = sum(
            max(0, targets[symbol] - degree) for degree in degrees
        )

    cl_degrees = [len(graph[index]) for index, symbol in enumerate(symbols) if symbol == "Cl"]
    descriptor["bridging_Cl"] = sum(degree >= 2 for degree in cl_degrees)
    descriptor["terminal_Cl"] = sum(degree == 1 for degree in cl_degrees)
    descriptor["isolated_Cl"] = sum(degree == 0 for degree in cl_degrees)

    cd_nodes = [index for index, symbol in enumerate(symbols) if symbol == "Cd"]
    se_nodes = [index for index, symbol in enumerate(symbols) if symbol == "Se"]
    bridge_cl_nodes = {
        index for index, symbol in enumerate(symbols)
        if symbol == "Cl"
        and sum(symbols[neighbor] == "Cd" for neighbor in graph[index]) >= 2
    }
    terminal_cl_nodes = {
        index for index, symbol in enumerate(symbols)
        if symbol == "Cl"
        and sum(symbols[neighbor] == "Cd" for neighbor in graph[index]) == 1
    }
    bridge_load = {
        node: sum(neighbor in bridge_cl_nodes for neighbor in graph[node])
        for node in cd_nodes
    }
    terminal_load = {
        node: sum(neighbor in terminal_cl_nodes for neighbor in graph[node])
        for node in cd_nodes
    }
    bridge_load_values = list(bridge_load.values())
    bridge_load_mean = mean(bridge_load_values) if bridge_load_values else 0.0
    descriptor["mean_bridge_load_Cd"] = bridge_load_mean
    descriptor["max_bridge_load_Cd"] = max(bridge_load_values, default=0)
    descriptor["std_bridge_load_Cd"] = (
        math.sqrt(mean((value - bridge_load_mean) ** 2 for value in bridge_load_values))
        if bridge_load_values else 0.0
    )
    descriptor["Cd_with_one_bridge"] = sum(value == 1 for value in bridge_load_values)
    descriptor["Cd_with_multiple_bridges"] = sum(value >= 2 for value in bridge_load_values)
    descriptor["Cd_with_mixed_terminal_bridge"] = sum(
        bridge_load[node] >= 1 and terminal_load[node] >= 1 for node in cd_nodes
    )

    bridge_pair_counts: dict[Edge, int] = defaultdict(int)
    bridge_angles: list[float] = []
    for ligand in bridge_cl_nodes:
        hosts = sorted(
            neighbor for neighbor in graph[ligand] if symbols[neighbor] == "Cd"
        )
        for left in range(len(hosts)):
            for right in range(left + 1, len(hosts)):
                pair = (hosts[left], hosts[right])
                bridge_pair_counts[pair] += 1
                angle = angle_degrees(
                    vector_from(atoms[ligand], atoms[hosts[left]]),
                    vector_from(atoms[ligand], atoms[hosts[right]]),
                )
                if angle is not None:
                    bridge_angles.append(angle)
    descriptor["bridged_Cd_pairs"] = len(bridge_pair_counts)
    descriptor["max_shared_bridges_per_Cd_pair"] = max(
        bridge_pair_counts.values(), default=0
    )
    descriptor["Cd_pairs_with_multiple_bridges"] = sum(
        value >= 2 for value in bridge_pair_counts.values()
    )
    descriptor["mean_Cd_Cl_Cd_angle_deg"] = (
        mean(bridge_angles) if bridge_angles else None
    )
    descriptor["std_Cd_Cl_Cd_angle_deg"] = (
        math.sqrt(mean((value - mean(bridge_angles)) ** 2 for value in bridge_angles))
        if bridge_angles else None
    )

    se_tetrahedrality: list[float] = []
    cd_tetrahedrality: list[float] = []
    cd_cn3_angle_rms: list[float] = []
    cd_cn3_plane_distance: list[float] = []
    for node in [*se_nodes, *cd_nodes]:
        neighbors = sorted(graph[node])
        vectors = [vector_from(atoms[node], atoms[neighbor]) for neighbor in neighbors]
        order = tetrahedral_order(vectors)
        if order is not None:
            if symbols[node] == "Se":
                se_tetrahedrality.append(order)
            else:
                cd_tetrahedrality.append(order)
        if symbols[node] == "Cd" and len(neighbors) == 3:
            angles = [
                angle_degrees(vectors[left], vectors[right])
                for left in range(3) for right in range(left + 1, 3)
            ]
            if all(value is not None for value in angles):
                cd_cn3_angle_rms.append(
                    math.sqrt(mean((float(value) - 120.0) ** 2 for value in angles))
                )
            points = [atoms[neighbor][1:] for neighbor in neighbors]
            plane_distance = point_plane_distance(
                atoms[node][1:], points[0], points[1], points[2]
            )
            if plane_distance is not None:
                cd_cn3_plane_distance.append(plane_distance)
    descriptor["Se_CN4_count"] = len(se_tetrahedrality)
    descriptor["mean_Se_tetrahedrality"] = (
        mean(se_tetrahedrality) if se_tetrahedrality else None
    )
    descriptor["min_Se_tetrahedrality"] = (
        min(se_tetrahedrality) if se_tetrahedrality else None
    )
    descriptor["Cd_CN4_count"] = len(cd_tetrahedrality)
    descriptor["mean_Cd_tetrahedrality"] = (
        mean(cd_tetrahedrality) if cd_tetrahedrality else None
    )
    descriptor["min_Cd_tetrahedrality"] = (
        min(cd_tetrahedrality) if cd_tetrahedrality else None
    )
    descriptor["Cd_CN3_count"] = len(cd_cn3_angle_rms)
    descriptor["mean_Cd_CN3_angle_rms_deg"] = (
        mean(cd_cn3_angle_rms) if cd_cn3_angle_rms else None
    )
    descriptor["max_Cd_CN3_angle_rms_deg"] = (
        max(cd_cn3_angle_rms) if cd_cn3_angle_rms else None
    )
    descriptor["mean_Cd_CN3_plane_distance"] = (
        mean(cd_cn3_plane_distance) if cd_cn3_plane_distance else None
    )
    descriptor["max_Cd_CN3_plane_distance"] = (
        max(cd_cn3_plane_distance) if cd_cn3_plane_distance else None
    )

    bond_lengths: dict[str, list[float]] = {"Cd_Se": [], "Cd_Cl": []}
    for left, right in edges:
        pair = frozenset((symbols[left], symbols[right]))
        if pair == frozenset(("Cd", "Se")):
            bond_lengths["Cd_Se"].append(distance(atoms[left], atoms[right]))
        elif pair == frozenset(("Cd", "Cl")):
            bond_lengths["Cd_Cl"].append(distance(atoms[left], atoms[right]))
    for pair_name, values in bond_lengths.items():
        pair_mean = mean(values) if values else None
        descriptor[f"mean_{pair_name}_distance"] = pair_mean
        descriptor[f"std_{pair_name}_distance"] = (
            math.sqrt(mean((value - float(pair_mean)) ** 2 for value in values))
            if values else None
        )

    cycles4 = cycle_node_sets(graph, 4)
    cycles6 = cycle_node_sets(graph, 6)
    descriptor["rings_4"] = len(cycles4)
    descriptor["rings_6"] = len(cycles6)
    descriptor["Cl_rings_4"] = sum(
        any(symbols[node] == "Cl" for node in cycle_nodes(cycle)) for cycle in cycles4
    )
    descriptor["Cl_rings_6"] = sum(
        any(symbols[node] == "Cl" for node in cycle_nodes(cycle)) for cycle in cycles6
    )

    inorganic_nodes = [index for index, symbol in enumerate(symbols) if symbol != "Cl"]
    remap = {node: index for index, node in enumerate(inorganic_nodes)}
    inorganic_edges = {
        (min(remap[left], remap[right]), max(remap[left], remap[right]))
        for left, right in edges
        if left in remap and right in remap
    }
    inorganic_graph = adjacency(len(inorganic_nodes), inorganic_edges)
    descriptor["inorganic_rings_6"] = len(cycle_node_sets(inorganic_graph, 6))

    coordinates = [(atom[1], atom[2], atom[3]) for atom in atoms]
    centre = tuple(mean(point[axis] for point in coordinates) for axis in range(3))
    descriptor["radius_of_gyration"] = math.sqrt(
        mean(
            sum((point[axis] - centre[axis]) ** 2 for axis in range(3))
            for point in coordinates
        )
    )
    spans = [
        max(point[axis] for point in coordinates) - min(point[axis] for point in coordinates)
        for axis in range(3)
    ]
    descriptor["max_span"] = max(spans)
    descriptor["max_pair_distance"] = max(
        (distance(atoms[left], atoms[right]) for left in range(len(atoms)) for right in range(left + 1, len(atoms))),
        default=0.0,
    )
    unexpected = 0
    for left in range(len(atoms)):
        for right in range(left + 1, len(atoms)):
            pair = frozenset((symbols[left], symbols[right]))
            if pair in {frozenset(("Cd", "Se")), frozenset(("Cd", "Cl"))}:
                continue
            if distance(atoms[left], atoms[right]) <= unexpected_cutoff:
                unexpected += 1
    descriptor["unexpected_close_contacts"] = unexpected
    descriptor["topology_fingerprint"] = topology_fingerprint(atoms, edges)
    return descriptor, edges


def edge_pair_name(edge: Edge, atoms: Sequence[Atom]) -> str:
    return "_".join(sorted((atoms[edge[0]][0], atoms[edge[1]][0])))


def write_xyz(path: Path, atoms: Sequence[Atom], comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(atoms)}\n{comment}\n")
        for symbol, x, y, z in atoms:
            handle.write(f"{symbol:<3s} {x: .10f} {y: .10f} {z: .10f}\n")


def csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def format_optional(value: object, digits: int = 3) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x_mean, y_mean = mean(xs), mean(ys)
    dx = [value - x_mean for value in xs]
    dy = [value - y_mean for value in ys]
    denominator = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denominator <= 1.0e-15:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / denominator


def analyse_correlations(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    descriptors = [
        "final_total_bonds", "final_Cd_Se_bonds", "final_Cd_Cl_bonds",
        "final_bridging_Cl", "final_terminal_Cl", "final_CN_deficit_Cd",
        "final_CN_deficit_Se", "final_inorganic_rings_6", "final_Cl_rings_4",
        "final_Cl_rings_6", "final_radius_of_gyration", "final_max_span",
        "final_mean_bridge_load_Cd", "final_max_bridge_load_Cd",
        "final_std_bridge_load_Cd", "final_Cd_with_one_bridge",
        "final_Cd_with_multiple_bridges", "final_Cd_with_mixed_terminal_bridge",
        "final_max_shared_bridges_per_Cd_pair",
        "final_Cd_pairs_with_multiple_bridges",
        "final_mean_Se_tetrahedrality", "final_mean_Cd_tetrahedrality",
        "final_mean_Cd_CN3_angle_rms_deg",
        "final_mean_Cd_CN3_plane_distance",
        "final_std_Cd_Se_distance", "final_std_Cd_Cl_distance",
        "formed_bonds", "broken_bonds",
    ]
    groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if record.get("energy_hartree") is not None:
            groups[(int(record["k"]), int(record["p"]))].append(record)

    results: list[dict[str, object]] = []
    for descriptor in descriptors:
        centred_x: list[float] = []
        centred_y: list[float] = []
        winner_values: list[float] = []
        other_values: list[float] = []
        bins_used = 0
        for group in groups.values():
            usable = [row for row in group if row.get(descriptor) not in (None, "")]
            if len(usable) < 2:
                continue
            xs = [float(row[descriptor]) for row in usable]
            ys = [float(row["energy_hartree"]) * HARTREE_TO_KCAL_MOL for row in usable]
            x_mean, y_mean = mean(xs), mean(ys)
            centred_x.extend(value - x_mean for value in xs)
            centred_y.extend(value - y_mean for value in ys)
            bins_used += 1
            for row, value in zip(usable, xs):
                if int(row.get("dft_rank_in_bin", 0)) == 1:
                    winner_values.append(value)
                else:
                    other_values.append(value)
        correlation = pearson(centred_x, centred_y)
        variance = sum(value * value for value in centred_x)
        slope = (
            sum(x * y for x, y in zip(centred_x, centred_y)) / variance
            if variance > 1.0e-15 else None
        )
        results.append(
            {
                "descriptor": descriptor,
                "within_bin_pearson_r": correlation,
                "slope_kcal_mol_per_unit": slope,
                "bins_used": bins_used,
                "structures_used": len(centred_x),
                "winner_mean": mean(winner_values) if winner_values else None,
                "nonwinner_mean": mean(other_values) if other_values else None,
            }
        )
    results.sort(
        key=lambda row: abs(float(row["within_bin_pearson_r"]))
        if row["within_bin_pearson_r"] is not None else -1.0,
        reverse=True,
    )
    return results


def analyse_grand_potential(
    records: Sequence[dict[str, object]],
    *,
    mu_cdse0_hartree: float,
    cdcl2_energy_hartree: float,
    delta_mu_cdse_ev: Sequence[float],
    delta_mu_cdcl2_ev: Sequence[float],
    quality_mode: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build the two-dimensional grand-potential grid and its p-minimized path.

    Raw CP2K energies are used, so the free CdCl2 reference energy is included
    explicitly.  This is equivalent to first subtracting ``p * E(CdCl2)`` and
    setting the standard ligand chemical potential to zero.
    """

    by_bin: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if record.get("energy_hartree") is None:
            continue
        if quality_mode == "ready" and not record.get("analysis_ready"):
            continue
        by_bin[(int(record["k"]), int(record["p"]))].append(record)

    bin_minima: dict[tuple[int, int], dict[str, object]] = {
        key: min(group, key=lambda row: float(row["energy_hartree"]))
        for key, group in by_bin.items()
    }
    grand_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    condition = 0
    for delta_cdse_ev in delta_mu_cdse_ev:
        for delta_cdcl2_ev in delta_mu_cdcl2_ev:
            delta_cdse_h = float(delta_cdse_ev) / HARTREE_TO_EV
            delta_cdcl2_h = float(delta_cdcl2_ev) / HARTREE_TO_EV
            mu_cdse = mu_cdse0_hartree + delta_cdse_h
            mu_cdcl2 = cdcl2_energy_hartree + delta_cdcl2_h
            condition_rows: list[dict[str, object]] = []
            for (k, p), winner in sorted(bin_minima.items()):
                energy = float(winner["energy_hartree"])
                ligand_referenced_energy = energy - p * cdcl2_energy_hartree
                omega = energy - k * mu_cdse - p * mu_cdcl2
                row = {
                    "condition_id": condition,
                    "delta_mu_CdSe_eV": float(delta_cdse_ev),
                    "delta_mu_CdCl2_eV": float(delta_cdcl2_ev),
                    "mu_CdSe_hartree": mu_cdse,
                    "mu_CdCl2_hartree": mu_cdcl2,
                    "k": k,
                    "p": p,
                    "M_CdCl2": p,
                    "structure_id": winner["structure_id"],
                    "quality_status": winner.get("quality_status"),
                    "provisional": not bool(winner.get("analysis_ready")),
                    "cluster_energy_hartree": energy,
                    "ligand_referenced_energy_hartree": ligand_referenced_energy,
                    "grand_potential_hartree": omega,
                    "grand_potential_eV": omega * HARTREE_TO_EV,
                    "grand_potential_kcal_mol": omega * HARTREE_TO_KCAL_MOL,
                    "optimal_p_for_k": False,
                }
                grand_rows.append(row)
                condition_rows.append(row)

            by_k: dict[int, list[dict[str, object]]] = defaultdict(list)
            for row in condition_rows:
                by_k[int(row["k"])].append(row)
            for k, candidates in sorted(by_k.items()):
                optimum = min(
                    candidates, key=lambda row: float(row["grand_potential_hartree"])
                )
                optimum["optimal_p_for_k"] = True
                profile_rows.append(
                    {
                        "condition_id": condition,
                        "delta_mu_CdSe_eV": float(delta_cdse_ev),
                        "delta_mu_CdCl2_eV": float(delta_cdcl2_ev),
                        "k": k,
                        "p_star": optimum["p"],
                        "M_CdCl2_star": optimum["p"],
                        "structure_id": optimum["structure_id"],
                        "quality_status": optimum["quality_status"],
                        "provisional": optimum["provisional"],
                        "grand_potential_star_hartree": optimum["grand_potential_hartree"],
                        "grand_potential_star_eV": optimum["grand_potential_eV"],
                        "grand_potential_star_kcal_mol": optimum["grand_potential_kcal_mol"],
                    }
                )
            condition += 1
    return grand_rows, profile_rows


def make_grand_potential_plots(
    output: Path,
    grand_rows: Sequence[dict[str, object]],
    profile_rows: Sequence[dict[str, object]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return []

    plots: list[str] = []
    by_condition: dict[int, list[dict[str, object]]] = defaultdict(list)
    profiles: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in grand_rows:
        by_condition[int(row["condition_id"])].append(row)
    for row in profile_rows:
        profiles[int(row["condition_id"])].append(row)

    for condition, rows in sorted(by_condition.items()):
        by_p: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_p[int(row["p"])].append(row)
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        for p, values in sorted(by_p.items()):
            values = sorted(values, key=lambda row: int(row["k"]))
            ax.plot(
                [int(row["k"]) for row in values],
                [float(row["grand_potential_eV"]) for row in values],
                marker="o",
                linewidth=1.0,
                alpha=0.65,
                label=f"p={p}",
            )
        envelope = sorted(profiles.get(condition, ()), key=lambda row: int(row["k"]))
        if envelope:
            ax.plot(
                [int(row["k"]) for row in envelope],
                [float(row["grand_potential_star_eV"]) for row in envelope],
                color="black",
                marker="o",
                linewidth=2.6,
                label=r"$\Delta\Omega^*(k)$",
            )
            for row in envelope:
                ax.annotate(
                    f"p={row['p_star']}",
                    (int(row["k"]), float(row["grand_potential_star_eV"])),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        reference = rows[0]
        ax.axhline(0.0, color="0.45", linestyle="--", linewidth=0.8)
        ax.set_xlabel("CdSe units, k")
        ax.set_ylabel(r"$\Delta\Omega$ (eV)")
        ax.set_title(
            rf"$\Delta\mu_{{CdSe}}$={float(reference['delta_mu_CdSe_eV']):g} eV, "
            rf"$\Delta\mu_{{CdCl_2}}$={float(reference['delta_mu_CdCl2_eV']):g} eV"
        )
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        path = output / f"grand_potential_condition_{condition:03d}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plots.append(path.name)
    return plots


def make_plots(
    output: Path,
    records: Sequence[dict[str, object]],
    correlations: Sequence[dict[str, object]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return []
    plots: list[str] = []
    usable = [row for row in records if row.get("relative_energy_kcal_mol") is not None]
    if usable:
        labels = sorted({(int(row["k"]), int(row["p"])) for row in usable})
        position = {key: index for index, key in enumerate(labels)}
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.45), 4.8))
        for row in usable:
            key = (int(row["k"]), int(row["p"]))
            ax.scatter(position[key], float(row["relative_energy_kcal_mol"]), s=24, alpha=0.8)
        ax.axhline(3.0, color="tab:red", linestyle="--", linewidth=1, label="3 kcal/mol")
        ax.set_xticks(range(len(labels)), [f"{k},{p}" for k, p in labels], rotation=60)
        ax.set_xlabel("(k,p) bin")
        ax.set_ylabel("Relative energy within bin (kcal/mol)")
        ax.legend()
        fig.tight_layout()
        path = output / "relative_energies.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plots.append(path.name)
    ranked = [row for row in correlations if row["within_bin_pearson_r"] is not None][:10]
    if ranked:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = [str(row["descriptor"]).removeprefix("final_") for row in reversed(ranked)]
        values = [float(row["within_bin_pearson_r"]) for row in reversed(ranked)]
        ax.barh(names, values)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Within-bin Pearson r with DFT energy")
        fig.tight_layout()
        path = output / "descriptor_correlations.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plots.append(path.name)
    return plots


def write_report(
    output: Path,
    records: Sequence[dict[str, object]],
    bins: Sequence[dict[str, object]],
    correlations: Sequence[dict[str, object]],
    near_threshold: float,
    plots: Sequence[str],
    cd_se_cutoff: float,
    cd_cl_cutoff: float,
) -> None:
    extracted = [row for row in records if row.get("energy_hartree") is not None]
    converged = [row for row in extracted if row.get("geometry_converged")]
    earlier_scf_bad = [
        row for row in records if int(row.get("scf_nonconverged_count", 0)) > 0
    ]
    final_scf_bad = [row for row in extracted if row.get("final_scf_converged") is not True]
    ready = [row for row in extracted if row.get("quality_status") == "ready"]
    formula_bad = [row for row in extracted if not row.get("formula_ok")]
    changed = [row for row in extracted if int(row.get("formed_bonds", 0)) + int(row.get("broken_bonds", 0)) > 0]
    lines = [
        "# CP2K nucleation-label analysis",
        "",
        "Absolute total energies are compared only between isomers with the same `(k,p)` composition. "
        "Cross-composition stability requires chemical-potential terms.",
        "",
        f"Relaxed bonds are inferred with Cd-Se <= {cd_se_cutoff:g} Å and "
        f"Cd-Cl <= {cd_cl_cutoff:g} Å. Re-run with adjusted cutoffs if the "
        "relaxed distance distributions show a different first-shell minimum.",
        "",
        "## Data quality",
        "",
        f"- Manifest/run directories: {len(records)}",
        f"- Final structures with energies: {len(extracted)}",
        f"- CP2K geometry-converged: {len(converged)}",
        f"- Analysis-ready calculations: {len(ready)}",
        f"- Final SCF not confirmed converged: {len(final_scf_bad)}",
        f"- Runs with one or more earlier unconverged SCF cycles: {len(earlier_scf_bad)}",
        f"- Formula mismatches: {len(formula_bad)}",
        f"- Relaxations changing inferred bonds: {len(changed)}",
        "",
        "## Lowest-energy structure per bin",
        "",
        "| k | p | winner | quality | E (Ha) | second gap (kcal/mol) | near-degenerate | bridge Cl | inorganic 6-rings |",
        "|---:|---:|:---|:---|---:|---:|---:|---:|---:|",
    ]
    for row in bins:
        lines.append(
            f"| {row['k']} | {row['p']} | {row['winner_structure_id']} | "
            f"{row['winner_quality_status']} | "
            f"{float(row['winner_energy_hartree']):.10f} | "
            f"{format_optional(row['second_gap_kcal_mol'])} | "
            f"{row['near_degenerate_count']} | {row['winner_bridging_Cl']} | "
            f"{row['winner_inorganic_rings_6']} |"
        )
    lines.extend(
        [
            "",
            f"Near-degenerate means within {near_threshold:g} kcal/mol of the bin minimum.",
            "",
            "## Descriptor patterns",
            "",
            "Correlations are calculated after subtracting each `(k,p)` bin mean, so atom-count/composition "
            "differences do not dominate. Negative `r` means larger descriptor values are associated with lower energy.",
            "",
            "| descriptor | within-bin r | slope (kcal/mol per unit) | bins | structures | winner mean | nonwinner mean |",
            "|:---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in correlations[:12]:
        correlation = row["within_bin_pearson_r"]
        slope = row["slope_kcal_mol_per_unit"]
        lines.append(
            f"| {row['descriptor']} | "
            f"{format_optional(correlation)} | "
            f"{format_optional(slope)} | "
            f"{row['bins_used']} | {row['structures_used']} | "
            f"{format_optional(row['winner_mean'])} | "
            f"{format_optional(row['nonwinner_mean'])} |"
        )
    if correlations and correlations[0]["within_bin_pearson_r"] is not None:
        strongest = correlations[0]
        direction = "lower" if float(strongest["within_bin_pearson_r"]) < 0 else "higher"
        lines.extend(
            [
                "",
                f"The strongest pooled association is `{strongest['descriptor']}` "
                f"(`r={float(strongest['within_bin_pearson_r']):.3f}`): larger values tend to have {direction} DFT energy. "
                "Treat this as a screening signal, not a causal result, especially when few bins contain multiple isomers.",
            ]
        )
    if plots:
        lines.extend(["", "## Plots", ""] + [f"- `{name}`" for name in plots])
    lines.extend(
        [
            "",
            "## Output tables",
            "",
            "- `structures.csv`: one row per attempted calculation.",
            "- `bin_minima.csv`: DFT winner and energy gap per `(k,p)`.",
            "- `near_degenerate.csv`: structures inside the selected energy window.",
            "- `descriptor_correlations.csv`: within-bin pattern analysis.",
            "- `topology_families.csv`: relaxed structures sharing an inferred topology.",
            "- `relaxed/`: standardized final XYZ structures.",
            "- `relaxations/`: original multiframe CP2K geometry trajectories.",
            "- `grand_potential.csv`: grand potentials for every requested condition.",
            "- `nucleation_profiles.csv`: ligand-minimized nucleation paths.",
            "- `grand_potential_report.md`: chemical-potential references and profiles.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_grand_potential_report(
    output: Path,
    profile_rows: Sequence[dict[str, object]],
    *,
    mu_cdse0_hartree: float,
    cdse_energy_hartree: float,
    cdcl2_energy_hartree: float,
    mu_cdse_source: str,
    quality_mode: str,
    plots: Sequence[str],
) -> None:
    binding = mu_cdse0_hartree - cdse_energy_hartree
    conditions: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in profile_rows:
        conditions[int(row["condition_id"])].append(row)
    lines = [
        "# Grand-potential nucleation profiles",
        "",
        "The raw CP2K cluster energies are transformed using",
        "",
        "`DeltaOmega(k,p) = E_cluster(k,p) - k*(mu_CdSe^0 + DeltaMu_CdSe) "
        "- p*(E_CdCl2 + DeltaMu_CdCl2)`.",
        "",
        "This is equivalent to subtracting `p*E_CdCl2` first and taking the "
        "standard CdCl2 chemical potential as zero.",
        "",
        f"- `mu_CdSe^0`: {mu_cdse0_hartree:.12f} Ha",
        f"- CdSe chemical-potential source: {mu_cdse_source}",
        f"- isolated `E_CdSe`: {cdse_energy_hartree:.12f} Ha",
        f"- free `E_CdCl2`: {cdcl2_energy_hartree:.12f} Ha",
        f"- ligated-reference shift relative to isolated CdSe: {binding:.12f} Ha "
        f"({binding * HARTREE_TO_KCAL_MOL:.3f} kcal/mol)",
        f"- calculation-quality selection: `{quality_mode}`",
        "",
        "At every k, the reported nucleation path is "
        "`DeltaOmega*(k) = min_p DeltaOmega(k,p)`.",
    ]
    for condition, rows in sorted(conditions.items()):
        rows = sorted(rows, key=lambda row: int(row["k"]))
        first = rows[0]
        lines.extend(
            [
                "",
                f"## Condition {condition}",
                "",
                f"`DeltaMu_CdSe = {float(first['delta_mu_CdSe_eV']):g} eV`; "
                f"`DeltaMu_CdCl2 = {float(first['delta_mu_CdCl2_eV']):g} eV`.",
                "",
                "| k | p* | structure | DeltaOmega* (eV) | quality |",
                "|---:|---:|:---|---:|:---|",
            ]
        )
        for row in rows:
            quality = str(row["quality_status"])
            if row.get("provisional"):
                quality += " (provisional)"
            lines.append(
                f"| {row['k']} | {row['p_star']} | {row['structure_id']} | "
                f"{float(row['grand_potential_star_eV']):.6f} | {quality} |"
            )
    if plots:
        lines.extend(["", "## Plots", ""] + [f"- `{name}`" for name in plots])
    lines.extend(
        [
            "",
            "## Tables",
            "",
            "- `grand_potential.csv`: every available `(k,p)` bin under every condition.",
            "- `nucleation_profiles.csv`: the p-minimized one-dimensional path.",
        ]
    )
    (output / "grand_potential_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help="calculation tree; repeat to merge several DFT batches",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--trajectory-name", default="CdSe-pos-1.xyz")
    parser.add_argument("--cp2k-output-name", default="cp2k_job.out")
    parser.add_argument("--near-threshold-kcal", type=float, default=3.0)
    parser.add_argument("--cd-se-cutoff", type=float, default=3.25)
    parser.add_argument("--cd-cl-cutoff", type=float, default=3.10)
    parser.add_argument("--unexpected-cutoff", type=float, default=2.60)
    parser.add_argument(
        "--mu-cdse0-hartree", type=float, default=DEFAULT_MU_CDSE0_HARTREE,
        help="standard CdSe chemical potential; default is the CdSe.2CdCl2 reference",
    )
    parser.add_argument(
        "--mu-cdse-baseline-ligands", type=int, choices=(0, 1, 2, 3), default=None,
        help=(
            "derive mu_CdSe^0 from the DFT minimum of k=1,p=M as "
            "E(1,M)-M*E(CdCl2), for M=0,1,2,3; this overrides "
            "--mu-cdse0-hartree"
        ),
    )
    parser.add_argument(
        "--cdse-energy-hartree", type=float, default=DEFAULT_CDSE_ENERGY_HARTREE,
        help="isolated CdSe reference energy, recorded for consistency checks",
    )
    parser.add_argument(
        "--cdcl2-energy-hartree", type=float, default=DEFAULT_CDCL2_ENERGY_HARTREE,
        help="free CdCl2 DFT energy used to put raw cluster energies on the ligand-zero scale",
    )
    parser.add_argument(
        "--delta-mu-cdse-ev", type=float, nargs="+", default=[0.0],
        help="one or more CdSe chemical-potential shifts in eV per CdSe unit",
    )
    parser.add_argument(
        "--delta-mu-cdcl2-ev", type=float, nargs="+", default=[0.0],
        help="one or more CdCl2 chemical-potential shifts in eV per CdCl2 unit",
    )
    parser.add_argument(
        "--grand-quality", choices=("all", "ready"), default="all",
        help="use all extracted bin minima or only fully converged analysis-ready structures",
    )
    parser.add_argument(
        "--no-copy-relaxations", action="store_true",
        help="do not copy the original multiframe CP2K geometry trajectories",
    )
    args = parser.parse_args()

    roots = [path.resolve() for path in (args.root or [Path("runs/cdse_map/dft_all")])]
    if args.output is not None:
        output = args.output.resolve()
    elif len(roots) == 1:
        output = (roots[0] / "analysis").resolve()
    else:
        output = (roots[0].parent / "analysis_all_dft").resolve()
    output.mkdir(parents=True, exist_ok=True)
    root_names = [root.name or f"root{index + 1}" for index, root in enumerate(roots)]
    name_counts: dict[str, int] = defaultdict(int)
    source_labels: list[str] = []
    for name in root_names:
        name_counts[name] += 1
        source_labels.append(
            name if name_counts[name] == 1 else f"{name}_{name_counts[name]}"
        )
    entries: list[dict[str, object]] = []
    for root, source_label in zip(roots, source_labels):
        discovered = discover_runs(root)
        for entry in discovered:
            annotated = dict(entry)
            original_id = str(entry["structure_id"])
            annotated["source_root"] = str(root)
            annotated["source_label"] = source_label
            annotated["source_structure_id"] = original_id
            # Distinguish identical structure IDs from different manifests;
            # the original ID remains available in source_structure_id.
            if len(roots) > 1:
                annotated["structure_id"] = f"{source_label}__{original_id}"
            annotated["index"] = len(entries)
            entries.append(annotated)
    if not entries:
        raise RuntimeError(
            "no calculation directories found under "
            + ", ".join(str(root) for root in roots)
        )

    records: list[dict[str, object]] = []
    for entry in entries:
        run_dir = Path(entry["run_dir"])
        trajectory = run_dir / args.trajectory_name
        if not trajectory.is_file():
            alternatives = sorted(run_dir.glob("*-pos-1.xyz"))
            trajectory = alternatives[0] if len(alternatives) == 1 else trajectory
        output_info = parse_cp2k_output(run_dir / args.cp2k_output_name)
        record = dict(entry)
        record["run_dir"] = str(run_dir)
        record["trajectory_path"] = str(trajectory) if trajectory.is_file() else ""
        record.update(output_info)
        record.update(
            {
                "energy_hartree": None,
                "energy_eV": None,
                "relative_energy_eV": None,
                "relative_energy_kcal_mol": None,
                "dft_rank_in_bin": None,
                "near_degenerate": False,
                "formula_ok": False,
                "quality_status": "not_evaluated",
                "analysis_ready": False,
                "formed_bonds": 0,
                "broken_bonds": 0,
                "energy_xyz_output_difference_hartree": None,
            }
        )
        if not trajectory.is_file():
            record["extraction_status"] = "missing_trajectory"
            record["quality_status"] = "missing_trajectory"
            records.append(record)
            continue
        try:
            frame = read_last_xyz_frame(trajectory)
        except (OSError, ValueError) as exc:
            record["extraction_status"] = f"trajectory_error: {exc}"
            record["quality_status"] = "trajectory_error"
            records.append(record)
            continue

        final_atoms: list[Atom] = list(frame["atoms"])  # type: ignore[arg-type]
        energy = frame["energy_hartree"]
        if energy is None:
            energy = output_info["output_energy_hartree"]
            energy_source = "cp2k_job.out"
        else:
            energy_source = args.trajectory_name
        record.update(
            {
                "extraction_status": "ok" if energy is not None else "missing_energy",
                "energy_source": energy_source if energy is not None else "",
                "energy_hartree": energy,
                "energy_eV": float(energy) * HARTREE_TO_EV if energy is not None else None,
                "optimizer_step": frame["step"],
                "trajectory_frames": frame["frame_count"],
                "trajectory_truncated": frame["trajectory_truncated"],
            }
        )
        if energy is not None and output_info["output_energy_hartree"] is not None:
            record["energy_xyz_output_difference_hartree"] = (
                float(energy) - float(output_info["output_energy_hartree"])
            )

        final_desc, final_edges = structure_descriptors(
            final_atoms,
            cd_se_cutoff=args.cd_se_cutoff,
            cd_cl_cutoff=args.cd_cl_cutoff,
            unexpected_cutoff=args.unexpected_cutoff,
        )
        record.update({f"final_{key}": value for key, value in final_desc.items()})
        k, p = int(record["k"]), int(record["p"])
        record["formula_ok"] = (
            int(final_desc["Cd_count"]) == k + p
            and int(final_desc["Se_count"]) == k
            and int(final_desc["Cl_count"]) == 2 * p
        )

        energy_difference = record.get("energy_xyz_output_difference_hartree")
        if energy is None:
            quality_status = "missing_energy"
        elif not record["formula_ok"]:
            quality_status = "formula_mismatch"
        elif output_info["final_scf_converged"] is not True:
            quality_status = "final_scf_not_converged"
        elif (
            energy_difference is not None
            and abs(float(energy_difference)) > 1.0e-6
        ):
            quality_status = "trajectory_output_energy_mismatch"
        elif not output_info["geometry_converged"]:
            quality_status = (
                "geometry_step_limit"
                if output_info["max_geo_steps_reached"]
                else "geometry_interrupted"
            )
        elif not output_info["program_ended"]:
            quality_status = "program_interrupted"
        else:
            quality_status = "ready"
        record["quality_status"] = quality_status
        record["analysis_ready"] = quality_status == "ready"

        start_path = run_dir / "start.xyz"
        if start_path.is_file():
            try:
                start_atoms = read_single_xyz(start_path)
                start_desc, start_edges = structure_descriptors(
                    start_atoms,
                    cd_se_cutoff=args.cd_se_cutoff,
                    cd_cl_cutoff=args.cd_cl_cutoff,
                    unexpected_cutoff=args.unexpected_cutoff,
                )
                record.update({f"start_{key}": value for key, value in start_desc.items()})
                if len(start_atoms) == len(final_atoms) and [a[0] for a in start_atoms] == [a[0] for a in final_atoms]:
                    formed = final_edges - start_edges
                    broken = start_edges - final_edges
                    record["formed_bonds"] = len(formed)
                    record["broken_bonds"] = len(broken)
                    formed_types: dict[str, int] = defaultdict(int)
                    broken_types: dict[str, int] = defaultdict(int)
                    for edge in formed:
                        formed_types[edge_pair_name(edge, final_atoms)] += 1
                    for edge in broken:
                        broken_types[edge_pair_name(edge, start_atoms)] += 1
                    record["formed_bond_types"] = ";".join(
                        f"{key}:{value}" for key, value in sorted(formed_types.items())
                    )
                    record["broken_bond_types"] = ";".join(
                        f"{key}:{value}" for key, value in sorted(broken_types.items())
                    )
            except (OSError, ValueError):
                record["start_parse_failed"] = True

        relaxed_path = (
            output / "relaxed" / f"k{k:03d}" / f"p{p:03d}"
            / f"{record['structure_id']}_relaxed.xyz"
        )
        comment = f"{record['structure_id']} k={k} p={p}"
        if energy is not None:
            comment += f" E_hartree={float(energy):.12f}"
        write_xyz(relaxed_path, final_atoms, comment)
        record["relaxed_xyz"] = str(relaxed_path.relative_to(output))
        if not args.no_copy_relaxations:
            relaxation_path = (
                output / "relaxations" / f"k{k:03d}" / f"p{p:03d}"
                / f"{record['structure_id']}_relaxation.xyz"
            )
            relaxation_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trajectory, relaxation_path)
            record["relaxation_trajectory"] = str(
                relaxation_path.relative_to(output)
            )
        records.append(record)

    by_bin: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if record.get("energy_hartree") is not None:
            by_bin[(int(record["k"]), int(record["p"]))].append(record)

    bin_rows: list[dict[str, object]] = []
    near_rows: list[dict[str, object]] = []
    for (k, p), group in sorted(by_bin.items()):
        ranked = sorted(group, key=lambda row: float(row["energy_hartree"]))
        minimum = float(ranked[0]["energy_hartree"])
        for rank, record in enumerate(ranked, start=1):
            delta_h = float(record["energy_hartree"]) - minimum
            record["relative_energy_eV"] = delta_h * HARTREE_TO_EV
            record["relative_energy_kcal_mol"] = delta_h * HARTREE_TO_KCAL_MOL
            record["dft_rank_in_bin"] = rank
            record["near_degenerate"] = (
                float(record["relative_energy_kcal_mol"]) <= args.near_threshold_kcal
            )
            if record["near_degenerate"]:
                near_rows.append(record)
        winner = ranked[0]
        second_gap = (
            (float(ranked[1]["energy_hartree"]) - minimum) * HARTREE_TO_KCAL_MOL
            if len(ranked) > 1 else None
        )
        bin_rows.append(
            {
                "k": k,
                "p": p,
                "isomer_count": len(ranked),
                "winner_structure_id": winner["structure_id"],
                "winner_quality_status": winner.get("quality_status"),
                "analysis_ready_isomer_count": sum(
                    bool(row.get("analysis_ready")) for row in ranked
                ),
                "winner_energy_hartree": minimum,
                "winner_energy_eV": minimum * HARTREE_TO_EV,
                "second_gap_kcal_mol": second_gap,
                "near_degenerate_count": sum(bool(row["near_degenerate"]) for row in ranked),
                "winner_bridging_Cl": winner.get("final_bridging_Cl"),
                "winner_inorganic_rings_6": winner.get("final_inorganic_rings_6"),
                "winner_topology_fingerprint": winner.get("final_topology_fingerprint"),
            }
        )

    topology_groups: dict[tuple[int, int, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        fingerprint = record.get("final_topology_fingerprint")
        if fingerprint:
            topology_groups[(int(record["k"]), int(record["p"]), str(fingerprint))].append(record)
    topology_rows: list[dict[str, object]] = []
    for (k, p, fingerprint), group in sorted(topology_groups.items()):
        energies = [float(row["energy_hartree"]) for row in group if row.get("energy_hartree") is not None]
        topology_rows.append(
            {
                "k": k,
                "p": p,
                "topology_fingerprint": fingerprint,
                "structure_count": len(group),
                "structure_ids": ";".join(str(row["structure_id"]) for row in group),
                "minimum_energy_hartree": min(energies) if energies else None,
                "energy_spread_kcal_mol": (
                    (max(energies) - min(energies)) * HARTREE_TO_KCAL_MOL
                    if len(energies) >= 2 else 0.0
                ),
            }
        )

    correlations = analyse_correlations(records)
    mu_cdse0_hartree = float(args.mu_cdse0_hartree)
    mu_cdse_source = "configured CdSe.2CdCl2 reference"
    if args.mu_cdse_baseline_ligands is not None:
        baseline_p = int(args.mu_cdse_baseline_ligands)
        baseline_candidates = [
            row for row in records
            if int(row["k"]) == 1
            and int(row["p"]) == baseline_p
            and row.get("energy_hartree") is not None
        ]
        if not baseline_candidates:
            raise RuntimeError(
                f"cannot derive CdSe baseline: no energy for k=1,p={baseline_p}"
            )
        baseline_winner = min(
            baseline_candidates, key=lambda row: float(row["energy_hartree"])
        )
        mu_cdse0_hartree = (
            float(baseline_winner["energy_hartree"])
            - baseline_p * float(args.cdcl2_energy_hartree)
        )
        mu_cdse_source = (
            f"derived from {baseline_winner['structure_id']} "
            f"(k=1,p={baseline_p})"
        )
    grand_rows, profile_rows = analyse_grand_potential(
        records,
        mu_cdse0_hartree=mu_cdse0_hartree,
        cdcl2_energy_hartree=args.cdcl2_energy_hartree,
        delta_mu_cdse_ev=args.delta_mu_cdse_ev,
        delta_mu_cdcl2_ev=args.delta_mu_cdcl2_ev,
        quality_mode=args.grand_quality,
    )
    preferred = [
        "index", "k", "p", "structure_id", "box_angstrom", "extraction_status",
        "quality_status", "analysis_ready",
        "program_ended", "geometry_converged", "max_geo_steps_reached",
        "scf_converged_count", "scf_nonconverged_count", "final_scf_converged",
        "formula_ok", "energy_source", "energy_hartree",
        "energy_eV", "relative_energy_eV", "relative_energy_kcal_mol",
        "dft_rank_in_bin", "near_degenerate", "optimizer_step", "trajectory_frames",
        "trajectory_truncated", "energy_xyz_output_difference_hartree",
        "formed_bonds", "broken_bonds", "formed_bond_types", "broken_bond_types",
        "final_total_bonds", "final_Cd_Se_bonds", "final_Cd_Cl_bonds",
        "final_bridging_Cl", "final_terminal_Cl", "final_isolated_Cl",
        "final_mean_bridge_load_Cd", "final_max_bridge_load_Cd",
        "final_std_bridge_load_Cd", "final_Cd_with_one_bridge",
        "final_Cd_with_multiple_bridges", "final_Cd_with_mixed_terminal_bridge",
        "final_bridged_Cd_pairs", "final_max_shared_bridges_per_Cd_pair",
        "final_Cd_pairs_with_multiple_bridges", "final_mean_Cd_Cl_Cd_angle_deg",
        "final_std_Cd_Cl_Cd_angle_deg",
        "final_mean_CN_Cd", "final_mean_CN_Se", "final_mean_CN_Cl",
        "final_CN_deficit_Cd", "final_CN_deficit_Se", "final_CN_deficit_Cl",
        "final_Se_CN4_count", "final_mean_Se_tetrahedrality",
        "final_min_Se_tetrahedrality", "final_Cd_CN4_count",
        "final_mean_Cd_tetrahedrality", "final_min_Cd_tetrahedrality",
        "final_Cd_CN3_count", "final_mean_Cd_CN3_angle_rms_deg",
        "final_max_Cd_CN3_angle_rms_deg", "final_mean_Cd_CN3_plane_distance",
        "final_max_Cd_CN3_plane_distance", "final_mean_Cd_Se_distance",
        "final_std_Cd_Se_distance", "final_mean_Cd_Cl_distance",
        "final_std_Cd_Cl_distance",
        "final_rings_4", "final_rings_6", "final_Cl_rings_4", "final_Cl_rings_6",
        "final_inorganic_rings_6", "final_radius_of_gyration", "final_max_span",
        "final_max_pair_distance", "final_unexpected_close_contacts",
        "final_topology_fingerprint", "relaxed_xyz", "relaxation_trajectory",
        "run_dir", "trajectory_path",
    ]
    extra_fields = sorted({key for row in records for key in row}.difference(preferred))
    write_csv(output / "structures.csv", records, [*preferred, *extra_fields])
    write_csv(
        output / "bin_minima.csv",
        bin_rows,
        [
            "k", "p", "isomer_count", "winner_structure_id", "winner_energy_hartree",
            "winner_quality_status", "analysis_ready_isomer_count",
            "winner_energy_eV", "second_gap_kcal_mol", "near_degenerate_count",
            "winner_bridging_Cl", "winner_inorganic_rings_6", "winner_topology_fingerprint",
        ],
    )
    write_csv(output / "near_degenerate.csv", near_rows, preferred)
    write_csv(
        output / "descriptor_correlations.csv",
        correlations,
        [
            "descriptor", "within_bin_pearson_r", "slope_kcal_mol_per_unit",
            "bins_used", "structures_used", "winner_mean", "nonwinner_mean",
        ],
    )
    write_csv(
        output / "topology_families.csv",
        topology_rows,
        [
            "k", "p", "topology_fingerprint", "structure_count", "structure_ids",
            "minimum_energy_hartree", "energy_spread_kcal_mol",
        ],
    )
    write_csv(
        output / "grand_potential.csv",
        grand_rows,
        [
            "condition_id", "delta_mu_CdSe_eV", "delta_mu_CdCl2_eV",
            "mu_CdSe_hartree", "mu_CdCl2_hartree", "k", "p", "M_CdCl2",
            "structure_id", "quality_status", "provisional",
            "cluster_energy_hartree", "ligand_referenced_energy_hartree",
            "grand_potential_hartree", "grand_potential_eV",
            "grand_potential_kcal_mol", "optimal_p_for_k",
        ],
    )
    write_csv(
        output / "nucleation_profiles.csv",
        profile_rows,
        [
            "condition_id", "delta_mu_CdSe_eV", "delta_mu_CdCl2_eV", "k",
            "p_star", "M_CdCl2_star", "structure_id", "quality_status",
            "provisional", "grand_potential_star_hartree",
            "grand_potential_star_eV", "grand_potential_star_kcal_mol",
        ],
    )
    plots = make_plots(output, records, correlations)
    grand_plots = make_grand_potential_plots(output, grand_rows, profile_rows)
    write_report(
        output,
        records,
        bin_rows,
        correlations,
        args.near_threshold_kcal,
        plots,
        args.cd_se_cutoff,
        args.cd_cl_cutoff,
    )
    write_grand_potential_report(
        output,
        profile_rows,
        mu_cdse0_hartree=mu_cdse0_hartree,
        cdse_energy_hartree=args.cdse_energy_hartree,
        cdcl2_energy_hartree=args.cdcl2_energy_hartree,
        mu_cdse_source=mu_cdse_source,
        quality_mode=args.grand_quality,
        plots=grand_plots,
    )

    print(f"Calculation directories: {len(records)}")
    print(f"Final energies extracted: {sum(row.get('energy_hartree') is not None for row in records)}")
    print(f"Analysed (k,p) bins: {len(bin_rows)}")
    print(f"Near-degenerate structures: {len(near_rows)}")
    print(f"Grand-potential conditions: {len(args.delta_mu_cdse_ev) * len(args.delta_mu_cdcl2_ev)}")
    print(f"CdSe chemical-potential reference: {mu_cdse0_hartree:.12f} Ha ({mu_cdse_source})")
    print(f"Analysis directory: {output}")
    print(f"Report: {output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
