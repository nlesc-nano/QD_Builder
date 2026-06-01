#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def _format_rep(value: float) -> str:
    return f"{float(value):g}"


def _parse_sizes(args) -> list[float]:
    if args.sizes:
        return [float(v) for v in args.sizes]
    if args.start is None or args.stop is None:
        return [1.0, 1.5, 2.0]

    step = float(args.step)
    if step <= 0:
        raise SystemExit("--step must be > 0")

    sizes = []
    current = float(args.start)
    stop = float(args.stop)
    while current <= stop + 1e-9:
        sizes.append(round(current, 10))
        current += step
    return sizes


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def _read_yaml_metadata(path: Path) -> tuple[dict[str, int], list[str]]:
    if yaml is None:
        return {}, []
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}

    charges = {str(el): int(q) for el, q in (data.get("charges") or {}).items()}
    passivation = data.get("passivation") or {}
    ligands = []
    for key in ("ligand", "anion_ligand", "cation_ligand"):
        if passivation.get(key):
            ligands.append(str(passivation[key]))
    return charges, ligands


def _count_columns(
    rows: list[dict[str, Any]],
    *,
    charges: dict[str, int],
    ligands: list[str],
) -> list[str]:
    elems = set()
    for row in rows:
        elems.update(row.get("counts", {}).keys())

    ligand_set = set(ligands)
    inorganic = [el for el in charges if el in elems and el not in ligand_set]
    remaining_inorganic = sorted(el for el in elems if el not in ligand_set and el not in inorganic)
    ligand_cols = [el for el in ligands if el in elems]
    remaining_ligands = sorted(el for el in elems if el in ligand_set and el not in ligand_cols)
    return inorganic + remaining_inorganic + ligand_cols + remaining_ligands


def _angstrom_to_nm(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) / 10.0


def _center_from_name(stem: str) -> str:
    for part in stem.split("_"):
        if part.startswith("c") and len(part) > 1:
            return part[1:]
    return ""


def _row_from_manifest(rep: float, json_path: Path, data: dict[str, Any]) -> dict[str, Any]:
    actual_rep = data.get("actual_size_unit_cells")
    if actual_rep is None:
        # Fall back to construction_radius_ang divided by min lattice if present
        actual_radius = data.get("actual_radius_ang", data.get("construction_radius_ang"))
        if actual_radius is not None:
            actual_rep = float(actual_radius) / 5.0  # approximate min lattice
        else:
            actual_rep = rep
    return {
        "rep": _format_rep(rep),
        "actual_rep": _format_rep(actual_rep),
        "center": _center_from_name(json_path.stem),
        "file": json_path.with_suffix(".xyz").name,
        "json": json_path.name,
        "total_charge": data.get("total_charge"),
        "input_D_nm": _angstrom_to_nm(data.get("construction_diameter_ang")),
        "final_D_nm": _angstrom_to_nm(data.get("size_metrics", {}).get("diameter_hull")),
        "counts": data.get("counts", {}),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _table_data(
    rows: list[dict[str, Any]],
    *,
    charges: dict[str, int],
    ligands: list[str],
) -> tuple[list[str], list[list[Any]], list[str]]:
    count_cols = _count_columns(rows, charges=charges, ligands=ligands)
    headers = [
        "rep",
        "actual_rep",
        "center",
        *count_cols,
        "Q",
        "input D (nm)",
        "final D (nm)",
        "file",
    ]

    table_rows = []
    for row in rows:
        counts = row.get("counts", {})
        table_rows.append([
            row["rep"],
            row["actual_rep"],
            row["center"],
            *[counts.get(el, 0) for el in count_cols],
            row["total_charge"],
            row["input_D_nm"],
            row["final_D_nm"],
            row["file"],
        ])
    return headers, table_rows, count_cols


def _format_aligned_table(headers: list[str], table_rows: list[list[Any]]) -> str:
    rows_s = [[_fmt(cell) for cell in row] for row in table_rows]
    widths = [
        max(len(str(headers[i])), *(len(row[i]) for row in rows_s))
        for i in range(len(headers))
    ]

    def fmt_row(row):
        cells = []
        for i, cell in enumerate(row):
            text = str(cell)
            if i >= len(row) - 1:
                cells.append(text.ljust(widths[i]))
            else:
                cells.append(text.rjust(widths[i]))
        return "  ".join(cells)

    lines = [fmt_row(headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt_row(row) for row in rows_s)
    return "\n".join(lines)


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    charges: dict[str, int],
    ligands: list[str],
) -> None:
    headers, table_rows, _count_cols = _table_data(rows, charges=charges, ligands=ligands)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("# Size-Cell Scan Summary\n\n")
        fh.write("Diameters are reported in nm. Element columns list inorganic atoms first, then ligand atoms.\n\n")
        fh.write("```text\n")
        fh.write(_format_aligned_table(headers, table_rows))
        fh.write("\n```\n")


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    charges: dict[str, int],
    ligands: list[str],
) -> None:
    count_cols = _count_columns(rows, charges=charges, ligands=ligands)
    fields = [
        "rep",
        "actual_rep",
        "center",
        *count_cols,
        "total_charge",
        "input_D_nm",
        "final_D_nm",
        "file",
        "json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fields}
            for el in count_cols:
                flat[el] = row.get("counts", {}).get(el, 0)
            writer.writerow(flat)


def _print_collected_row(row: dict[str, Any], count_cols: list[str]) -> None:
    counts = row.get("counts", {})
    counts_text = " ".join(f"{el}={counts.get(el, 0)}" for el in count_cols)
    print(
        f"[row] rep={row['rep']} actual_rep={row['actual_rep']} center={row['center']} "
        f"{counts_text} Q={row['total_charge']} "
        f"D_in={_fmt(row['input_D_nm'])} nm D_final={_fmt(row['final_D_nm'])} nm "
        f"file={row['file']}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run builder over several --size-unit-cells values and summarize outputs."
    )
    parser.add_argument("cif", help="Input CIF")
    parser.add_argument("yaml", help="Builder YAML input")
    parser.add_argument("--sizes", nargs="+", type=float, help="Explicit size-cell values, e.g. 1 1.5 2 2.5")
    parser.add_argument("--start", type=float, help="First size-cell value if --sizes is omitted")
    parser.add_argument("--stop", type=float, help="Last size-cell value if --sizes is omitted")
    parser.add_argument("--step", type=float, default=0.5, help="Size-cell increment for --start/--stop")
    parser.add_argument("--out-dir", default="size_cell_scan", help="Directory for generated structures/logs")
    parser.add_argument("--summary", default="size_cell_scan_summary.md", help="Markdown summary path")
    parser.add_argument("--csv", default=None, help="Optional CSV summary path")
    parser.add_argument("--positive-q-mode", choices=["add", "remove"], default="add")
    parser.add_argument("--center", action="store_true", help="Pass --center to builder")
    parser.add_argument("--verbose-builder", action="store_true", help="Pass --verbose to builder")
    args, builder_args = parser.parse_known_args(argv)

    charges, ligands = _read_yaml_metadata(Path(args.yaml))
    sizes = _parse_sizes(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for rep in sizes:
        rep_label = _format_rep(rep)
        before = time.time()
        cmd = [
            sys.executable,
            "-m",
            "builder",
            args.cif,
            args.yaml,
            "--size-unit-cells",
            rep_label,
            "-o",
            str(out_dir / "scan.xyz"),
            "--positive-q-mode",
            args.positive_q_mode,
        ]
        if args.center:
            cmd.append("--center")
        if args.verbose_builder:
            cmd.append("--verbose")
        extra = builder_args
        if extra and extra[0] == "--":
            extra = extra[1:]
        cmd.extend(extra)

        print(f"[run] rep={rep_label} -> builder", flush=True)
        proc = subprocess.run(cmd, text=True, capture_output=True)
        log_path = out_dir / f"rep{rep_label}.log"
        log_path.write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            raise SystemExit(f"builder failed for rep={rep_label}; see {log_path}")

        manifests = sorted(
            path for path in out_dir.glob(f"*_rep{rep_label}.json")
            if path.stat().st_mtime >= before - 1.0
        )
        if not manifests:
            raise SystemExit(f"no manifests found for rep={rep_label} in {out_dir}")

        for manifest in manifests:
            row = _row_from_manifest(rep, manifest, _read_json(manifest))
            rows.append(row)
            count_cols = _count_columns(rows, charges=charges, ligands=ligands)
            _print_collected_row(row, count_cols)

    rows.sort(key=lambda r: (float(r["rep"]), r["center"], r["file"]))

    summary_path = Path(args.summary)
    _write_markdown(summary_path, rows, charges=charges, ligands=ligands)
    if args.csv:
        _write_csv(Path(args.csv), rows, charges=charges, ligands=ligands)

    headers, table_rows, _count_cols = _table_data(rows, charges=charges, ligands=ligands)
    print("\n" + _format_aligned_table(headers, table_rows))
    print(f"\nWrote {summary_path}")
    if args.csv:
        print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
