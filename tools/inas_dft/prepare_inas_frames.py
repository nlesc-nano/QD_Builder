#!/usr/bin/env python3
"""Prepare flat InAs/InCl3 CP2K frames from a nucleation bundle.

The output is designed for the range-based ``cp2k.slurm`` workflow:

    dft_inas_k3/
      frame_1.xyz ... frame_N.xyz
      frames.tsv
      template_inas_opt.in

Only retained surface structures are selected by default.  All isomers in a
given (k,p) bin receive the same cubic box, computed from the largest Cartesian
span in that bin plus the requested padding.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import re
import shutil
from typing import Iterable, Sequence


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
    coordinates: list[tuple[float, ...]] = []
    for line in atom_lines:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"malformed XYZ line in {path}: {line!r}")
        symbols.append(fields[0])
        coordinates.append(tuple(float(value) for value in fields[1:4]))
    return tuple(symbols), tuple(coordinates)


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
    symbols: Sequence[str], coordinates: Sequence[Sequence[float]]
) -> tuple[object, ...]:
    return (
        tuple(symbols),
        tuple(tuple(round(float(value), 8) for value in xyz) for xyz in coordinates),
    )


def required_side(coordinates: Sequence[Sequence[float]], padding: float) -> float:
    spans = [
        max(point[axis] for point in coordinates)
        - min(point[axis] for point in coordinates)
        for axis in range(3)
    ]
    return max(spans) + float(padding)


def rounded_side(value: float, minimum: float, quantum: float) -> float:
    value = max(float(value), float(minimum))
    return math.ceil(value / quantum - 1.0e-12) * quantum


def source_xyzs(source: Path, max_k: int) -> list[Path]:
    if not source.is_dir():
        raise FileNotFoundError(f"nucleation bundle not found: {source}")
    paths = sorted(source.glob("structures/k*/p*/retained/*_surface.xyz"))
    selected: list[Path] = []
    for path in paths:
        k, _p = kp_from_path(path)
        if k <= max_k:
            selected.append(path.resolve())
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="nucleation bundle containing structures/k*/p*/retained/*_surface.xyz",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).with_name("template_inas_opt.in"),
    )
    parser.add_argument("--max-k", type=int, default=3)
    parser.add_argument("--padding", type=float, default=12.0)
    parser.add_argument("--min-box", type=float, default=20.0)
    parser.add_argument("--round-box", type=float, default=1.0)
    args = parser.parse_args()

    if args.max_k < 1:
        parser.error("--max-k must be positive")
    if args.padding <= 0 or args.min_box <= 0 or args.round_box <= 0:
        parser.error("padding, min-box, and round-box must be positive")

    template = args.template.resolve().read_text()
    required_placeholders = (
        "__XYZ_FILE__",
        "__PREFIX__",
        "__ABC_SIZE__",
        "__WFN_FILE__",
        "__SCF_GUESS__",
    )
    for placeholder in required_placeholders:
        if placeholder not in template:
            raise ValueError(f"template is missing {placeholder}")

    candidates: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for xyz in source_xyzs(args.source.resolve(), args.max_k):
        symbols, coordinates = read_xyz(xyz)
        k, p = kp_from_path(xyz)
        key = (k, p, geometry_key(symbols, coordinates))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "k": k,
                "p": p,
                "structure_id": structure_id(xyz),
                "source": xyz,
                "symbols": symbols,
                "coordinates": coordinates,
                "required_side": required_side(coordinates, args.padding),
                "geometry_key": key,
            }
        )
    if not candidates:
        raise RuntimeError("no retained surface XYZ files found")

    bin_sides: dict[tuple[int, int], float] = {}
    for item in candidates:
        key = (int(item["k"]), int(item["p"]))
        bin_sides[key] = max(
            bin_sides.get(key, 0.0), float(item["required_side"])
        )
    bin_sides = {
        key: rounded_side(value, args.min_box, args.round_box)
        for key, value in bin_sides.items()
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.template.resolve(), output / "template_inas_opt.in")

    candidates.sort(
        key=lambda item: (
            int(item["k"]),
            int(item["p"]),
            str(item["structure_id"]),
            str(item["source"]),
        )
    )
    rows: list[tuple[object, ...]] = []
    used_names: set[str] = set()
    for frame, item in enumerate(candidates, start=1):
        k, p = int(item["k"]), int(item["p"])
        sid = str(item["structure_id"])
        frame_name = f"frame_{frame}.xyz"
        if frame_name in used_names:
            digest = hashlib.sha256(
                repr(item["geometry_key"]).encode()
            ).hexdigest()[:8]
            frame_name = f"frame_{frame}_{digest}.xyz"
        used_names.add(frame_name)
        shutil.copy2(item["source"], output / frame_name)
        box = bin_sides[(k, p)]
        rows.append(
            (
                frame,
                k,
                p,
                sid,
                f"{box:.6f}".rstrip("0").rstrip("."),
                frame_name,
                str(item["source"]),
            )
        )

    with (output / "frames.tsv").open("w", encoding="utf-8") as handle:
        handle.write("frame\tk\tp\tstructure_id\tbox_angstrom\txyz_file\tsource_xyz\n")
        for row in rows:
            handle.write("\t".join(str(value) for value in row) + "\n")

    with (output / "box_sizes.tsv").open("w", encoding="utf-8") as handle:
        handle.write("k\tp\tbox_angstrom\n")
        for (k, p), box in sorted(bin_sides.items()):
            handle.write(f"{k}\t{p}\t{box:g}\n")

    print(f"Prepared {len(rows)} unique retained InAs surface structures")
    print(f"Output: {output}")
    print(f"Frames: {output / 'frames.tsv'}")
    print(
        f"Bins: {len(bin_sides)}; box range: "
        f"{min(bin_sides.values()):g}-{max(bin_sides.values()):g} A"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
