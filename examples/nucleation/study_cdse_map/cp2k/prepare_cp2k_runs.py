#!/usr/bin/env python3
"""Prepare isolated CP2K run folders from retained nucleation surface XYZs.

All isomers in one ``(k, p)`` bin receive the same cubic box.  Its side is the
largest Cartesian span of any selected isomer in that bin plus ``--padding``,
rounded upward and bounded below by ``--min-box``.
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
    if not lines:
        raise ValueError(f"empty XYZ: {path}")
    count = int(lines[0].strip())
    atom_lines = lines[2 : 2 + count]
    if len(atom_lines) != count:
        raise ValueError(f"XYZ atom count mismatch: {path}")
    symbols: list[str] = []
    coordinates: list[tuple[float, ...]] = []
    for line in atom_lines:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"malformed XYZ atom line in {path}: {line!r}")
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
        raise ValueError(f"cannot determine k/p bin from {path}")
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


def surface_xyzs(sources: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for source in sources:
        if not source.is_dir():
            raise FileNotFoundError(f"nucleation bundle not found: {source}")
        paths.extend(source.glob("structures/k*/p*/retained/*_surface.xyz"))
    return sorted(path.resolve() for path in paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="Nucleation bundle; repeat to merge exact/guided bundles",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).with_name("template.in"),
    )
    parser.add_argument("--padding", type=float, default=12.0)
    parser.add_argument("--min-box", type=float, default=20.0)
    parser.add_argument("--round-box", type=float, default=1.0)
    args = parser.parse_args()

    if args.padding <= 0 or args.min_box <= 0 or args.round_box <= 0:
        parser.error("padding, min-box, and round-box must be positive")
    template = args.template.resolve().read_text()
    for placeholder in ("@XYZ_FILE@", "@BOX_SIZE@"):
        if placeholder not in template:
            raise ValueError(f"template is missing {placeholder}")

    candidates: list[dict[str, object]] = []
    seen_geometries: set[tuple[object, ...]] = set()
    for xyz in surface_xyzs(args.source):
        symbols, coordinates = read_xyz(xyz)
        k, p = kp_from_path(xyz)
        key = (k, p, geometry_key(symbols, coordinates))
        if key in seen_geometries:
            continue
        seen_geometries.add(key)
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
        raise RuntimeError("no retained *_surface.xyz files found")

    bin_side: dict[tuple[int, int], float] = {}
    for item in candidates:
        key = (int(item["k"]), int(item["p"]))
        bin_side[key] = max(bin_side.get(key, 0.0), float(item["required_side"]))
    bin_side = {
        key: rounded_side(value, args.min_box, args.round_box)
        for key, value in bin_side.items()
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    used_folders: set[tuple[int, int, str]] = set()
    rows: list[tuple[object, ...]] = []
    candidates.sort(
        key=lambda item: (item["k"], item["p"], item["structure_id"], str(item["source"]))
    )
    for index, item in enumerate(candidates):
        k, p = int(item["k"]), int(item["p"])
        folder_id = str(item["structure_id"])
        folder_key = (k, p, folder_id)
        if folder_key in used_folders:
            digest = hashlib.sha256(repr(item["geometry_key"]).encode()).hexdigest()[:8]
            folder_id = f"{folder_id}__{digest}"
            folder_key = (k, p, folder_id)
        used_folders.add(folder_key)

        relative_dir = Path(f"k{k:03d}") / f"p{p:03d}" / folder_id
        run_dir = output / relative_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["source"], run_dir / "start.xyz")
        box = bin_side[(k, p)]
        box_text = f"{box:.6f}".rstrip("0").rstrip(".")
        rendered = template.replace("@XYZ_FILE@", "start.xyz").replace(
            "@BOX_SIZE@", box_text
        )
        (run_dir / "cp2k_job.in").write_text(rendered)
        rows.append(
            (
                index,
                k,
                p,
                folder_id,
                box_text,
                relative_dir.as_posix(),
                str(item["source"]),
            )
        )

    manifest = output / "manifest.tsv"
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("index\tk\tp\tstructure_id\tbox_angstrom\trun_dir\tsource_xyz\n")
        for row in rows:
            handle.write("\t".join(str(value) for value in row) + "\n")
    with (output / "box_sizes.tsv").open("w", encoding="utf-8") as handle:
        handle.write("k\tp\tbox_angstrom\n")
        for (k, p), box in sorted(bin_side.items()):
            handle.write(f"{k}\t{p}\t{box:g}\n")

    print(f"Prepared {len(rows)} unique retained surface structures")
    print(f"Run root: {output}")
    print(f"Manifest: {manifest}")
    print(f"Bins: {len(bin_side)}; box range: {min(bin_side.values()):g}-{max(bin_side.values()):g} A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
