#!/usr/bin/env python3
"""Prepare a CP2K DFT tree from a lattice-free molecular map.

Reads isomer XYZ files written by ``write_molecular_map``::

    molecular_map/k###/p###/<structure_id>.xyz

and produces a parallel tree ready for HPC::

    dft_out/k###/p###/<structure_id>/start.xyz
    dft_out/k###/p###/<structure_id>/cp2k_job.in
    dft_out/manifest.tsv
    dft_out/box_sizes.tsv
    dft_out/cp2k.slurm          (copied)
    dft_out/submit_jobs.sh      (copied)

All isomers in the same (k, p) bin share one cubic vacuum box (max extent in
the bin + padding, rounded).  The manifest is consumed by ``cp2k.slurm`` /
``submit_jobs.sh`` the same way as the InAs preparer.

Example
-------

    python tools/molecular_dft/prepare_molecular_dft.py \\
      --source runs/molecular_cdse_k2 \\
      --output runs/molecular_cdse_k2_dft \\
      --max-k 2

Then rsync ``runs/molecular_cdse_k2_dft`` to the HPC and::

    cd molecular_cdse_k2_dft
    ./submit_jobs.sh manifest.tsv 1 8
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = HERE / "template_cdse_opt.in"
DEFAULT_SLURM = HERE / "cp2k_one.slurm"
DEFAULT_SLURM_ARRAY = HERE / "cp2k.slurm"  # optional multi-task array (legacy)
DEFAULT_SUBMIT = HERE / "submit_jobs.sh"

BIN_PART = re.compile(r"^(?P<axis>[kp])(?P<value>\d+)$")


def read_xyz(path: Path) -> Tuple[Tuple[str, ...], Tuple[Tuple[float, ...], ...]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"invalid XYZ: {path}")
    count = int(lines[0].strip().split()[0])
    atom_lines = lines[2 : 2 + count]
    if len(atom_lines) != count:
        raise ValueError(f"XYZ atom count mismatch: {path}")
    symbols: List[str] = []
    coords: List[Tuple[float, ...]] = []
    for line in atom_lines:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"malformed XYZ line in {path}: {line!r}")
        symbols.append(fields[0])
        coords.append(
            (float(fields[1]), float(fields[2]), float(fields[3]))
        )
    return tuple(symbols), tuple(coords)


def kp_from_path(path: Path) -> Tuple[int, int]:
    values: Dict[str, int] = {}
    for part in path.parts:
        match = BIN_PART.match(part)
        if match:
            values[match.group("axis")] = int(match.group("value"))
    if set(values) != {"k", "p"}:
        raise ValueError(f"cannot determine k/p from {path}")
    return values["k"], values["p"]


def geometry_key(
    symbols: Sequence[str], coords: Sequence[Sequence[float]]
) -> Tuple[object, ...]:
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


def discover_isomers(
    source: Path,
    *,
    max_k: int,
    min_k: int,
    max_p: Optional[int],
    min_p: int,
) -> List[Dict[str, object]]:
    """Collect unique molecular-map XYZ isomers under source."""

    # write_molecular_map layout: k###/p###/<structure_id>.xyz
    # Also accept nested maps or index-relative trees.
    patterns = (
        "k*/p*/*.xyz",
        "**/k*/p*/*.xyz",
    )
    found: List[Path] = []
    for pattern in patterns:
        found.extend(source.glob(pattern))
    # De-duplicate paths
    unique_paths = sorted({path.resolve() for path in found if path.is_file()})

    candidates: List[Dict[str, object]] = []
    seen: set = set()
    for xyz in unique_paths:
        # Skip non-isomer junk if any
        if xyz.name in {"start.xyz"}:
            continue
        try:
            k, p = kp_from_path(xyz)
        except ValueError:
            continue
        if k < min_k or k > max_k:
            continue
        if p < min_p or (max_p is not None and p > max_p):
            continue
        symbols, coords = read_xyz(xyz)
        if not symbols:
            continue
        key = (k, p, geometry_key(symbols, coords))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "k": k,
                "p": p,
                "id": xyz.stem,
                "source": xyz,
                "symbols": symbols,
                "coords": coords,
                "required_side": required_side(coords, 0.0),  # padding later
                "geometry_key": key,
            }
        )
    return candidates


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="molecular map root (contains k###/p###/*.xyz)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="DFT tree root to create",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help=f"CP2K input template (default: {DEFAULT_TEMPLATE.name})",
    )
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=2)
    parser.add_argument("--min-p", type=int, default=0)
    parser.add_argument(
        "--max-p",
        type=int,
        default=None,
        help="maximum p inclusive (default: no cap)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=12.0,
        help="vacuum padding (Å) added to max span in each (k,p) bin",
    )
    parser.add_argument("--min-box", type=float, default=20.0)
    parser.add_argument("--round-box", type=float, default=1.0)
    parser.add_argument(
        "--no-slurm-copy",
        action="store_true",
        help="do not copy cp2k.slurm / submit_jobs.sh into the output root",
    )
    args = parser.parse_args(argv)

    if args.min_k < 1 or args.max_k < args.min_k:
        parser.error("require 1 <= --min-k <= --max-k")
    if args.min_p < 0:
        parser.error("--min-p must be >= 0")
    if args.max_p is not None and args.max_p < args.min_p:
        parser.error("require --min-p <= --max-p")
    if args.padding <= 0 or args.min_box <= 0 or args.round_box <= 0:
        parser.error("padding, min-box, and round-box must be positive")

    source = args.source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)

    template_path = args.template.resolve()
    if not template_path.is_file():
        raise FileNotFoundError(template_path)
    template = template_path.read_text(encoding="utf-8")
    for placeholder in ("__XYZ_FILE__", "__PREFIX__", "__ABC_SIZE__"):
        if placeholder not in template:
            raise ValueError(f"template is missing {placeholder}")

    raw = discover_isomers(
        source,
        max_k=args.max_k,
        min_k=args.min_k,
        max_p=args.max_p,
        min_p=args.min_p,
    )
    if not raw:
        raise RuntimeError(
            f"no molecular XYZ isomers found under {source} "
            f"(expected k*/p*/*.xyz for k={args.min_k}..{args.max_k})"
        )

    # Apply padding to per-structure required sides, then take max per bin.
    for item in raw:
        item["required_side"] = required_side(
            item["coords"], args.padding  # type: ignore[arg-type]
        )

    bin_boxes: Dict[Tuple[int, int], float] = {}
    for item in raw:
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
    raw.sort(
        key=lambda item: (
            int(item["k"]),
            int(item["p"]),
            str(item["id"]),
            str(item["source"]),
        )
    )

    rows: List[Tuple[object, ...]] = []
    used_dirs: set = set()
    for index, item in enumerate(raw, start=1):
        k, p = int(item["k"]), int(item["p"])
        sid = str(item["id"])
        key = (k, p, sid)
        if key in used_dirs:
            digest = hashlib.sha256(
                repr(item["geometry_key"]).encode()
            ).hexdigest()[:8]
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
        (run_dir / "cp2k_job.in").write_text(rendered, encoding="utf-8")

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
        handle.write("k\tp\tbox_angstrom\tn_structures\n")
        counts: Dict[Tuple[int, int], int] = {}
        for item in raw:
            key = (int(item["k"]), int(item["p"]))
            counts[key] = counts.get(key, 0) + 1
        for (k, p), box in sorted(bin_boxes.items()):
            handle.write(f"{k}\t{p}\t{box:g}\t{counts[(k, p)]}\n")

    # Carry annotations if present next to the molecular map.
    ann_src = source / "annotations.csv"
    if ann_src.is_file():
        shutil.copy2(ann_src, output / "annotations.csv")

    if not args.no_slurm_copy:
        for helper in (DEFAULT_SLURM, DEFAULT_SUBMIT, DEFAULT_SLURM_ARRAY):
            if helper.is_file():
                dest = output / helper.name
                shutil.copy2(helper, dest)
                dest.chmod(dest.stat().st_mode | 0o111)

    readme = output / "README.txt"
    readme.write_text(
        "CdSe/CdCl2 molecular DFT tree (CP2K GEO_OPT).\n"
        f"source molecular map: {source}\n"
        f"template: {template_path}\n"
        f"structures: {len(rows)}\n"
        f"bins: {len(bin_boxes)}\n"
        "\n"
        "Per isomer:\n"
        "  start.xyz     — construction geometry (input coordinates)\n"
        "  cp2k_job.in   — rendered from template (PROJECT = structure_id)\n"
        "\n"
        "On the HPC, from this directory:\n"
        "  ./submit_jobs.sh manifest.tsv\n  # → one independent sbatch per structure (fills the cluster)\n"
        "\n"
        "After jobs finish, compare graphs with:\n"
        "  python tools/compare_molecular_start_final.py \\\n"
        "    --annotations annotations.csv \\\n"
        "    --start-dir <molecular_map> \\\n"
        "    --final-root <this_dft_tree> \\\n"
        "    --output start_final_report.csv\n",
        encoding="utf-8",
    )

    print(f"Prepared {len(rows)} molecular isomers for CP2K")
    print(f"Output tree: {output}")
    print(f"Manifest:    {output / 'manifest.tsv'}")
    print(f"Bins:        {len(bin_boxes)}")
    for (k, p), box in sorted(bin_boxes.items()):
        n = sum(1 for item in raw if int(item["k"]) == k and int(item["p"]) == p)
        print(f"  k={k} p={p}: {n} structures, box={box:g} Å")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
