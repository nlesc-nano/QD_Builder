#!/usr/bin/env python3
"""Prepare a manifest-driven InAs/InCl3 CP2K tree.

Each retained surface XYZ becomes:

    output/k###/p###/structure_id/start.xyz
    output/k###/p###/structure_id/cp2k_job.in

All isomers in the same (k,p) bin receive one common cubic box.  The manifest
is consumed by the companion ``cp2k.slurm`` and ``submit_jobs.sh`` scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import re
import shutil
from typing import Sequence


BIN_PART = re.compile(r"^(?P<axis>[kp])(?P<value>\d+)$")


def read_xyz(path: Path) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"invalid XYZ: {path}")
    count = int(lines[0].strip())
    atom_lines = lines[2 : 2 + count]
    if len(atom_lines) != count:
        raise ValueError(f"XYZ atom count mismatch: {path}")
    symbols: list[str] = []
    coords: list[tuple[float, ...]] = []
    for line in atom_lines:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"malformed XYZ line in {path}: {line!r}")
        symbols.append(fields[0])
        coords.append(tuple(float(value) for value in fields[1:4]))
    return tuple(symbols), tuple(coords)


def kp_from_path(path: Path) -> tuple[int, int]:
    values: dict[str, int] = {}
    for part in path.parts:
        match = BIN_PART.match(part)
        if match:
            values[match.group("axis")] = int(match.group("value"))
    if set(values) != {"k", "p"}:
        raise ValueError(f"cannot determine k/p from {path}")
    return values["k"], values["p"]


def structure_id(path: Path) -> str:
    suffix = "_surface"
    return path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem


def geometry_key(
    symbols: Sequence[str], coords: Sequence[Sequence[float]]
) -> tuple[object, ...]:
    return (
        tuple(symbols),
        tuple(tuple(round(float(value), 8) for value in xyz) for xyz in coords),
    )


def required_side(coords: Sequence[Sequence[float]], padding: float) -> float:
    spans = [
        max(point[axis] for point in coords)
        - min(point[axis] for point in coords)
        for axis in range(3)
    ]
    return max(spans) + float(padding)


def rounded_side(value: float, minimum: float, quantum: float) -> float:
    value = max(float(value), float(minimum))
    return math.ceil(value / quantum - 1.0e-12) * quantum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--max-k", type=int, default=3)
    parser.add_argument("--padding", type=float, default=12.0)
    parser.add_argument("--min-box", type=float, default=20.0)
    parser.add_argument("--round-box", type=float, default=1.0)
    args = parser.parse_args()

    if args.max_k < 1:
        parser.error("--max-k must be positive")
    if args.padding <= 0 or args.min_box <= 0 or args.round_box <= 0:
        parser.error("padding, min-box, and round-box must be positive")

    source = args.source.resolve()
    template = args.template.resolve().read_text()
    required = ("__XYZ_FILE__", "__PREFIX__", "__ABC_SIZE__")
    for placeholder in required:
        if placeholder not in template:
            raise ValueError(f"template is missing {placeholder}")
    if not source.is_dir():
        raise FileNotFoundError(source)

    candidates: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for xyz in sorted(source.glob("structures/k*/p*/retained/*_surface.xyz")):
        k, p = kp_from_path(xyz)
        if k > args.max_k:
            continue
        symbols, coords = read_xyz(xyz)
        key = (k, p, geometry_key(symbols, coords))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "k": k,
                "p": p,
                "id": structure_id(xyz),
                "source": xyz.resolve(),
                "symbols": symbols,
                "coords": coords,
                "required_side": required_side(coords, args.padding),
                "geometry_key": key,
            }
        )
    if not candidates:
        raise RuntimeError("no retained *_surface.xyz files found")

    bin_boxes: dict[tuple[int, int], float] = {}
    for item in candidates:
        key = (int(item["k"]), int(item["p"]))
        bin_boxes[key] = max(
            bin_boxes.get(key, 0.0), float(item["required_side"])
        )
    bin_boxes = {
        key: rounded_side(value, args.min_box, args.round_box)
        for key, value in bin_boxes.items()
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates.sort(
        key=lambda item: (
            int(item["k"]), int(item["p"]), str(item["id"]), str(item["source"])
        )
    )
    rows: list[tuple[object, ...]] = []
    used_dirs: set[tuple[int, int, str]] = set()
    for index, item in enumerate(candidates, start=1):
        k, p = int(item["k"]), int(item["p"])
        sid = str(item["id"])
        key = (k, p, sid)
        if key in used_dirs:
            digest = hashlib.sha256(repr(item["geometry_key"]).encode()).hexdigest()[:8]
            sid = f"{sid}__{digest}"
            key = (k, p, sid)
        used_dirs.add(key)
        run_dir = output / f"k{k:03d}" / f"p{p:03d}" / sid
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["source"], run_dir / "start.xyz")
        box = bin_boxes[(k, p)]
        box_text = f"{box:.6f}".rstrip("0").rstrip(".")
        rendered = (
            template.replace("__XYZ_FILE__", "start.xyz")
            .replace("__PREFIX__", sid)
            .replace("__ABC_SIZE__", f"{box_text} {box_text} {box_text}")
        )
        (run_dir / "cp2k_job.in").write_text(rendered)
        rows.append(
            (
                index,
                k,
                p,
                sid,
                box_text,
                run_dir.relative_to(output).as_posix(),
                str(item["source"]),
            )
        )

    with (output / "manifest.tsv").open("w", encoding="utf-8") as handle:
        handle.write(
            "index\tk\tp\tstructure_id\tbox_angstrom\trun_dir\tsource_xyz\n"
        )
        for row in rows:
            handle.write("\t".join(str(value) for value in row) + "\n")

    with (output / "box_sizes.tsv").open("w", encoding="utf-8") as handle:
        handle.write("k\tp\tbox_angstrom\n")
        for (k, p), box in sorted(bin_boxes.items()):
            handle.write(f"{k}\t{p}\t{box:g}\n")

    print(f"Prepared {len(rows)} unique retained InAs structures")
    print(f"Output tree: {output}")
    print(f"Manifest: {output / 'manifest.tsv'}")
    print(f"Bins: {len(bin_boxes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
