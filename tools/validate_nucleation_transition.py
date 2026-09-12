#!/usr/bin/env python3
"""Validate the high-chloride transition in an adaptive minima archive.

The script is deliberately a post-processing diagnostic.  It does not change
the archive and it does not decide whether a graph-rule violation is
chemically acceptable.  It selects one representative per relaxed graph,
recomputes Cd--Cl coordination from the archived graph, and reports the
distance distribution around mu3/bridge-overlap contacts.

Typical use::

    python tools/validate_nucleation_transition.py RUN \
        --output RUN/transition_validation --p 9 10 11 --top 12

The generated ``validation.xyz`` is a multi-frame XYZ file.  Its comment line
starts with the energy in Hartree, so it can be opened directly by Molden.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HARTREE_TO_EV = 27.211386245988


def _input_path(path: Path) -> Path:
    return path / "minima.json" if path.is_dir() else path


def _edge_set(edges: Iterable[Iterable[int]]) -> set[tuple[int, int]]:
    return {tuple(sorted((int(a), int(b)))) for a, b in edges}


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _graph_key(row: dict[str, Any]) -> str:
    return str(row.get("final", {}).get("graph_hash") or row.get("minimum_id"))


def _cd_cl_contacts(row: dict[str, Any]) -> dict[str, Any]:
    symbols = row["symbols"]
    positions = row["positions"]
    edges = _edge_set(row.get("edges", []))
    cd = [i for i, symbol in enumerate(symbols) if symbol == "Cd"]
    cl = [i for i, symbol in enumerate(symbols) if symbol == "Cl"]
    bonded: list[dict[str, Any]] = []
    unbonded: list[dict[str, Any]] = []
    cl_hosts: dict[int, list[int]] = {}
    for ligand in cl:
        hosts = [
            host for host in cd if tuple(sorted((host, ligand))) in edges
        ]
        cl_hosts[ligand] = hosts
        for host in cd:
            item = {
                "cd": host,
                "cl": ligand,
                "distance_A": _distance(positions[host], positions[ligand]),
                "bonded": host in hosts,
            }
            (bonded if host in hosts else unbonded).append(item)

    overlaps: list[dict[str, Any]] = []
    for cap, hosts in cl_hosts.items():
        if len(hosts) < 3:
            continue
        for host in hosts:
            for bridge, bridge_hosts in cl_hosts.items():
                if bridge == cap or host not in bridge_hosts or len(bridge_hosts) < 2:
                    continue
                overlaps.append(
                    {
                        "cap_cl": cap,
                        "shared_cd": host,
                        "bridge_cl": bridge,
                        "cap_cd_distance_A": _distance(
                            positions[host], positions[cap]
                        ),
                        "bridge_cd_distance_A": _distance(
                            positions[host], positions[bridge]
                        ),
                        "cap_cn": len(hosts),
                        "bridge_cn": len(bridge_hosts),
                    }
                )

    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"min": None, "median": None, "max": None}
        values = sorted(values)
        return {
            "min": values[0],
            "median": values[(len(values) - 1) // 2],
            "max": values[-1],
        }

    return {
        "n_cd": len(cd),
        "n_cl": len(cl),
        "bonded_cd_cl": len(bonded),
        "bonded_distance_A": _stats([x["distance_A"] for x in bonded]),
        "unbonded_distance_A": _stats([x["distance_A"] for x in unbonded]),
        "mu2_from_graph": sum(len(hosts) == 2 for hosts in cl_hosts.values()),
        "mu3_or_higher_from_graph": sum(len(hosts) >= 3 for hosts in cl_hosts.values()),
        "terminal_from_graph": sum(len(hosts) == 1 for hosts in cl_hosts.values()),
        "overlap_contacts": overlaps,
        "overlap_count": len(overlaps),
    }


def _select(rows: list[dict[str, Any]], ps: set[int], top: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        k, p = int(row.get("k", -1)), int(row.get("p", -1))
        if k != 8 or p not in ps or not row.get("converged", False):
            continue
        role = str(row.get("role", "unknown"))
        grouped[(k, p, role, _graph_key(row))].append(row)
    for (k, p, role, _), members in sorted(grouped.items()):
        selected.append(min(members, key=lambda row: float(row["energy_eV"])))
    # Keep the lowest-energy unique graphs per composition.  Audit structures
    # are retained because the transition question is precisely whether their
    # violation is robust, not because they are accepted minima.
    result: list[dict[str, Any]] = []
    for p in sorted(ps):
        for role in ("primary", "audit", "unknown"):
            bin_rows = [
                row for row in selected
                if int(row["p"]) == p and row.get("role", "") == role
            ]
            bin_rows.sort(key=lambda row: float(row["energy_eV"]))
            result.extend(bin_rows[:top])
    return result


def _write_xyz(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as handle:
        for rank, row in enumerate(rows, start=1):
            energy = float(row["energy_eV"])
            final = row.get("final", {})
            handle.write(f"{len(row['symbols'])}\n")
            handle.write(
                f"{energy / HARTREE_TO_EV:.10f} {row.get('minimum_id', '')} "
                f"E_eV={energy:.8f} rank={rank} k={row.get('k')} p={row.get('p')} "
                f"role={row.get('role', '')} graph={_graph_key(row)} "
                f"mu2={final.get('mu2', 'NA')} mu3={final.get('mu3', 'NA')}\n"
            )
            for symbol, position in zip(row["symbols"], row["positions"]):
                handle.write(
                    f"{symbol:<2s} {float(position[0]): .10f} "
                    f"{float(position[1]): .10f} {float(position[2]): .10f}\n"
                )


def validate(source: Path, output: Path, *, ps: set[int], top: int) -> dict[str, Any]:
    source = _input_path(source)
    archive = json.loads(source.read_text())
    rows = list(archive.get("experimental", {}).values())
    selected = _select(rows, ps, top)
    records: list[dict[str, Any]] = []
    for row in selected:
        contacts = _cd_cl_contacts(row)
        final = row.get("final", {})
        records.append(
            {
                "minimum_id": row.get("minimum_id"),
                "structure_id": row.get("structure_id", ""),
                "k": int(row["k"]),
                "p": int(row["p"]),
                "role": row.get("role", ""),
                "energy_eV": float(row["energy_eV"]),
                "graph_hash": final.get("graph_hash", ""),
                "core_hash": final.get("core_hash", ""),
                "stored_violations": list(row.get("violations", [])),
                "stored_mu2": final.get("mu2"),
                "stored_mu3": final.get("mu3"),
                "stored_terminal": final.get("terminal"),
                "contacts": contacts,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    _write_xyz(selected, output / "validation.xyz")
    fields = [
        "minimum_id", "k", "p", "role", "energy_eV", "graph_hash",
        "stored_mu2", "stored_mu3", "stored_terminal", "stored_violations",
        "mu2_from_graph", "mu3_or_higher_from_graph", "terminal_from_graph",
        "overlap_count", "bonded_min_A", "bonded_median_A", "bonded_max_A",
        "unbonded_min_A", "unbonded_median_A", "unbonded_max_A",
    ]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            contacts = record["contacts"]
            row = {key: record.get(key, "") for key in fields}
            row["stored_violations"] = ";".join(record["stored_violations"])
            for prefix in ("bonded", "unbonded"):
                stats = contacts[f"{prefix}_distance_A"]
                for stat in ("min", "median", "max"):
                    row[f"{prefix}_{stat}_A"] = stats[stat]
            row.update(
                mu2_from_graph=contacts["mu2_from_graph"],
                mu3_or_higher_from_graph=contacts["mu3_or_higher_from_graph"],
                terminal_from_graph=contacts["terminal_from_graph"],
                overlap_count=contacts["overlap_count"],
            )
            writer.writerow(row)
    report = {
        "source": str(source.resolve()),
        "selection": {"k": 8, "p": sorted(ps), "top_unique_graphs": top},
        "structures": len(records),
        "records": records,
        "interpretation": {
            "overlap_count_is_graph_contact_pairs": True,
            "energies_are_only_comparable_within_identical_k_p": True,
            "this_is_a_diagnostic_not_an_acceptance_rule": True,
        },
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="run directory or minima.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p", type=int, nargs="+", default=[9, 10, 11])
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    report = validate(args.source, args.output, ps=set(args.p), top=args.top)
    print(
        f"[transition-validation] selected {report['structures']} structures; "
        f"wrote {args.output / 'validation.json'} and {args.output / 'validation.xyz'}"
    )


if __name__ == "__main__":
    main()
