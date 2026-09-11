#!/usr/bin/env python3
"""Export adaptive ``minima.json`` archives as ranked multi-frame XYZ stacks."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


EV_TO_KCAL_MOL = 23.060548
HARTREE_TO_EV = 27.211386245988


def _json_cell(value) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"))


def _input_path(path: Path) -> Path:
    return path / "minima.json" if path.is_dir() else path


def _comment(row, minimum_id, rank, delta_eV) -> str:
    final = row.get("final", {})
    energy = float(row["energy_eV"])
    graph_members = int(row.get("_export_graph_members", 1))
    # Molden's concatenated XYZ reader parses the second line with atof(), so
    # the energy must be the first token.  Metadata may safely follow it.
    return (
        f"{energy / HARTREE_TO_EV:.10f} {minimum_id} "
        f"E_Ha={energy / HARTREE_TO_EV:.10f} E_eV={energy:.10f} "
        f"dE_eV={delta_eV:.6f} "
        f"dE_kcal_mol={delta_eV * EV_TO_KCAL_MOL:.3f} rank={rank} "
        f"k={row['k']} p={row['p']} role={row.get('role', 'unknown')} "
        f"graph_members={graph_members} "
        f"mu2={final.get('mu2', 'NA')} n4={final.get('n4', 'NA')} "
        f"n6={final.get('n6', 'NA')}"
    )


def export_archive(
    source: Path,
    output: Path,
    *,
    arm: str = "experimental",
    roles=("primary",),
    selected_k=None,
    selected_p=None,
    max_per_bin=None,
    one_per_graph=False,
):
    """Write one energy-ranked XYZ trajectory per selected ``(k, p)`` bin."""

    source = _input_path(Path(source))
    data = json.loads(source.read_text())
    if arm not in data or not isinstance(data[arm], dict):
        raise ValueError(f"archive has no {arm!r} minimum map")
    role_set = set(roles)
    k_set = None if selected_k is None else set(selected_k)
    p_set = None if selected_p is None else set(selected_p)
    grouped = defaultdict(list)
    skipped = 0
    for archive_id, original in data[arm].items():
        row = dict(original)
        minimum_id = row.get("minimum_id", archive_id)
        try:
            k = int(row["k"])
            p = int(row["p"])
            energy = float(row["energy_eV"])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        if (
            row.get("role") not in role_set
            or (k_set is not None and k not in k_set)
            or (p_set is not None and p not in p_set)
        ):
            continue
        symbols = row.get("symbols")
        positions = row.get("positions")
        if (
            not math.isfinite(energy)
            or not isinstance(symbols, list)
            or not isinstance(positions, list)
            or len(symbols) != len(positions)
            or not symbols
        ):
            skipped += 1
            continue
        grouped[k, p].append((energy, str(minimum_id), row))

    output.mkdir(parents=True, exist_ok=True)
    index_fields = [
        "stack",
        "frame",
        "rank",
        "k",
        "p",
        "minimum_id",
        "structure_id",
        "role",
        "energy_eV",
        "relative_energy_eV",
        "relative_energy_kcal_mol",
        "graph_members",
        "n_atoms",
        "mu2",
        "mu3",
        "terminal",
        "n4",
        "n6",
        "se_cn",
        "cd_environments",
        "cl_cn",
        "core_hash",
        "graph_hash",
        "source",
    ]
    index_rows = []
    bin_summary = []
    for (k, p), records in sorted(grouped.items()):
        records.sort(key=lambda item: (item[0], item[1]))
        available = len(records)
        if one_per_graph:
            graph_groups = defaultdict(list)
            for record in records:
                graph_hash = record[2].get("final", {}).get("graph_hash")
                graph_groups[graph_hash or f"missing:{record[1]}"].append(record)
            representatives = []
            for members in graph_groups.values():
                representative = members[0]
                representative[2]["_export_graph_members"] = len(members)
                representatives.append(representative)
            records = sorted(representatives, key=lambda item: (item[0], item[1]))
        if max_per_bin is not None:
            records = records[:max_per_bin]
        reference = records[0][0]
        stack_name = f"k{k}_p{p}.xyz"
        with (output / stack_name).open("w") as xyz:
            for rank, (energy, minimum_id, row) in enumerate(records, start=1):
                delta_eV = energy - reference
                symbols = row["symbols"]
                positions = row["positions"]
                xyz.write(f"{len(symbols)}\n")
                xyz.write(_comment(row, minimum_id, rank, delta_eV) + "\n")
                for symbol, coordinates in zip(symbols, positions):
                    if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 3:
                        raise ValueError(f"{minimum_id} has invalid Cartesian coordinates")
                    x, y, z = (float(value) for value in coordinates)
                    xyz.write(f"{symbol:<2s} {x: .10f} {y: .10f} {z: .10f}\n")
                final = row.get("final", {})
                index_rows.append(
                    dict(
                        stack=stack_name,
                        frame=rank,
                        rank=rank,
                        k=k,
                        p=p,
                        minimum_id=minimum_id,
                        structure_id=row.get("structure_id", ""),
                        role=row.get("role", ""),
                        energy_eV=f"{energy:.10f}",
                        relative_energy_eV=f"{delta_eV:.10f}",
                        relative_energy_kcal_mol=f"{delta_eV * EV_TO_KCAL_MOL:.6f}",
                        graph_members=int(row.get("_export_graph_members", 1)),
                        n_atoms=len(symbols),
                        mu2=final.get("mu2", ""),
                        mu3=final.get("mu3", ""),
                        terminal=final.get("terminal", ""),
                        n4=final.get("n4", ""),
                        n6=final.get("n6", ""),
                        se_cn=_json_cell(final.get("se_cn")),
                        cd_environments=_json_cell(final.get("cd_environments")),
                        cl_cn=_json_cell(final.get("cl_cn")),
                        core_hash=final.get("core_hash", ""),
                        graph_hash=final.get("graph_hash", ""),
                        source=row.get("source", ""),
                    )
                )
        bin_summary.append(
            dict(k=k, p=p, written=len(records), available=available, file=stack_name)
        )

    with (output / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=index_fields)
        writer.writeheader()
        writer.writerows(index_rows)
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                source=str(source.resolve()),
                arm=arm,
                roles=sorted(role_set),
                sorting="ascending raw g-xTB energy within each fixed (k,p) composition",
                molden_energy="first XYZ comment token, converted from eV to Hartree",
                relative_energy_reference="lowest exported energy in the same (k,p) stack",
                max_per_bin=max_per_bin,
                one_per_graph=one_per_graph,
                structures=len(index_rows),
                skipped_invalid=skipped,
                bins=bin_summary,
            ),
            indent=2,
        )
        + "\n"
    )
    return dict(structures=len(index_rows), bins=len(bin_summary), skipped=skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="run directory or minima.json")
    parser.add_argument(
        "--output",
        type=Path,
        help="destination (default: <run>/xyz_stacks)",
    )
    parser.add_argument("--arm", choices=("experimental", "control"), default="experimental")
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=("primary", "audit"),
        default=("primary",),
    )
    parser.add_argument("--k", type=int, action="append", help="select k; repeat as needed")
    parser.add_argument("--p", type=int, action="append", help="select p; repeat as needed")
    parser.add_argument(
        "--max-per-bin",
        type=int,
        help="write only the N lowest-energy structures in each (k,p) stack",
    )
    parser.add_argument(
        "--one-per-graph",
        action="store_true",
        help=(
            "write only the lowest-energy representative of each relaxed graph "
            "topology; the complete minima.json is never modified"
        ),
    )
    args = parser.parse_args()
    if args.max_per_bin is not None and args.max_per_bin < 1:
        parser.error("--max-per-bin must be positive")
    source = _input_path(args.source)
    output = args.output or source.parent / "xyz_stacks"
    summary = export_archive(
        source,
        output,
        arm=args.arm,
        roles=args.roles,
        selected_k=args.k,
        selected_p=args.p,
        max_per_bin=args.max_per_bin,
        one_per_graph=args.one_per_graph,
    )
    print(
        f"[export] wrote {summary['structures']} structures in "
        f"{summary['bins']} (k,p) stacks to {output}"
    )
    if summary["skipped"]:
        print(f"[export] skipped {summary['skipped']} invalid archive rows")


if __name__ == "__main__":
    main()
