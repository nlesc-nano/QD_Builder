#!/usr/bin/env python3
"""Build a light transfer tree: start.xyz + final.xyz only.

Reads a submitted molecular CP2K DFT tree::

    dft_root/
      k001/p001/<structure_id>/
        start.xyz
        cp2k_job.in
        <PROJECT>-pos-1.xyz   # multi-frame CP2K trajectory
        cp2k_job.out
        ...

and writes a **new** folder with the same ``k###/p###/id`` layout but
**only** the two geometries per isomer (easy to ``scp`` / ``rsync``)::

    dft_light/
      inventory.csv
      k001/p001/<structure_id>/
        start.xyz
        final.xyz
      k002/p003/<structure_id>/
        start.xyz
        final.xyz

``final.xyz`` = last complete frame of the CP2K ``*pos*.xyz`` trajectory.

Does **not** copy inputs, restarts, trajectories, or logs into the light tree.
Safe to re-run; existing light files are overwritten.

Examples
--------

On the HPC (recommended — then transfer only the light folder)::

    python extract_start_final.py \\
        --root /path/to/heavy_dft \\
        --out  /path/to/dft_light

Only finished jobs (``PROGRAM ENDED`` in ``cp2k_job.out``)::

    python extract_start_final.py --root . --out ../dft_light --only-done

Also write ``final.xyz`` inside the heavy tree (optional)::

    python extract_start_final.py --root . --out ../dft_light --write-in-place
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


BIN_PART = re.compile(r"^(?P<axis>[kp])(?P<value>\d+)$")


def _read_last_xyz_frame_text(path: Path) -> Tuple[str, int]:
    """Return (xyz_text_of_last_complete_frame, n_complete_frames)."""

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    i = 0
    n_frames = 0
    last_block: Optional[List[str]] = None
    n_lines = len(lines)
    while i < n_lines:
        while i < n_lines and not lines[i].strip():
            i += 1
        if i >= n_lines:
            break
        try:
            count = int(lines[i].strip().split()[0])
        except (ValueError, IndexError):
            i += 1
            continue
        end = i + 2 + count
        if end > n_lines:
            break
        block = lines[i:end]
        ok = True
        for row in block[2:]:
            parts = row.split()
            if len(parts) < 4:
                ok = False
                break
            try:
                float(parts[1])
                float(parts[2])
                float(parts[3])
            except ValueError:
                ok = False
                break
        if ok and len(block) == count + 2:
            last_block = block
            n_frames += 1
        i = end
    if last_block is None:
        raise ValueError(f"no complete XYZ frames in {path}")
    return "\n".join(last_block) + "\n", n_frames


def _find_pos_trajectory(calc_dir: Path) -> Optional[Path]:
    """Prefer CP2K ``*-pos-*.xyz``; fall back to any ``*pos*.xyz``."""

    candidates = list(calc_dir.glob("*-pos-*.xyz")) + list(
        calc_dir.glob("*pos*.xyz")
    )
    candidates = [
        p
        for p in candidates
        if p.name not in {"start.xyz", "final.xyz"} and p.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def _kp_from_path(path: Path) -> Tuple[Optional[int], Optional[int]]:
    values: dict[str, int] = {}
    for part in path.parts:
        match = BIN_PART.match(part)
        if match:
            values[match.group("axis")] = int(match.group("value"))
    return values.get("k"), values.get("p")


def _iter_calc_dirs(root: Path) -> List[Path]:
    found: List[Path] = []
    for start in root.glob("k*/p*/*/start.xyz"):
        found.append(start.parent)
    for job in root.glob("k*/p*/*/cp2k_job.in"):
        if job.parent not in found:
            found.append(job.parent)
    return sorted(set(found), key=lambda p: str(p))


def _job_done(calc_dir: Path) -> bool:
    out = calc_dir / "cp2k_job.out"
    if not out.is_file():
        return False
    try:
        text = out.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "PROGRAM ENDED" in text


def _light_dest(
    out_dir: Path,
    k: Optional[int],
    p: Optional[int],
    structure_id: str,
) -> Path:
    if k is None or p is None:
        return out_dir / structure_id
    return out_dir / f"k{k:03d}" / f"p{p:03d}" / structure_id


def _write_light_pair(
    dest: Path,
    start_path: Path,
    final_xyz_text: str,
) -> None:
    """Write only start.xyz + final.xyz into dest (create parents)."""

    dest.mkdir(parents=True, exist_ok=True)
    # Ensure the light folder contains nothing but the two geometries.
    for extra in dest.iterdir():
        if extra.name not in {"start.xyz", "final.xyz"}:
            if extra.is_file():
                extra.unlink()
            elif extra.is_dir():
                shutil.rmtree(extra)
    shutil.copy2(start_path, dest / "start.xyz")
    (dest / "final.xyz").write_text(final_xyz_text, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="heavy DFT tree root (default: current directory)",
    )
    parser.add_argument(
        "--out",
        "--export-dir",
        dest="out",
        type=Path,
        default=None,
        help=(
            "light transfer tree (required for transfer use). "
            "Same k###/p###/id layout; only start.xyz + final.xyz per isomer."
        ),
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help="CSV inventory path (default: <out>/inventory.csv or <root>/start_final_inventory.csv)",
    )
    parser.add_argument(
        "--only-done",
        action="store_true",
        help="only process dirs whose cp2k_job.out contains PROGRAM ENDED",
    )
    parser.add_argument(
        "--write-in-place",
        action="store_true",
        help="also write final.xyz inside the heavy DFT tree",
    )
    parser.add_argument(
        "--include-pending",
        action="store_true",
        help="also copy start.xyz for isomers without a trajectory (no final.xyz)",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    out_dir = args.out.resolve() if args.out is not None else None
    if out_dir is None and not args.write_in_place:
        print(
            "Specify --out <light_dir> for a transfer tree, "
            "and/or --write-in-place to write final.xyz in the heavy tree.",
            file=sys.stderr,
        )
        return 2

    if out_dir is not None:
        if out_dir.resolve() == root.resolve():
            print(
                "--out must differ from --root (light tree vs heavy DFT tree)",
                file=sys.stderr,
            )
            return 2
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.inventory is not None:
        inventory_path = args.inventory.resolve()
    elif out_dir is not None:
        inventory_path = out_dir / "inventory.csv"
    else:
        inventory_path = root / "start_final_inventory.csv"

    calc_dirs = _iter_calc_dirs(root)
    if not calc_dirs:
        print(f"No isomer folders under {root} (expected k*/p/*/start.xyz)")
        return 1

    rows: List[dict] = []
    n_ok = n_pending = n_fail = n_start_only = 0

    for calc_dir in calc_dirs:
        rel = calc_dir.relative_to(root).as_posix()
        k, p = _kp_from_path(calc_dir)
        structure_id = calc_dir.name
        start_path = calc_dir / "start.xyz"
        row = {
            "k": k if k is not None else "",
            "p": p if p is not None else "",
            "structure_id": structure_id,
            "run_dir": rel,
            "status": "",
            "has_start": int(start_path.is_file()),
            "pos_file": "",
            "n_frames": "",
            "light_dir": "",
            "job_done": int(_job_done(calc_dir)),
        }

        if args.only_done and not row["job_done"]:
            row["status"] = "pending"
            n_pending += 1
            rows.append(row)
            continue

        if not start_path.is_file():
            row["status"] = "missing_start"
            n_fail += 1
            rows.append(row)
            continue

        pos = _find_pos_trajectory(calc_dir)
        if pos is None:
            row["status"] = "missing_pos"
            n_pending += 1
            if args.include_pending and out_dir is not None:
                dest = _light_dest(out_dir, k, p, structure_id)
                dest.mkdir(parents=True, exist_ok=True)
                for extra in list(dest.iterdir()):
                    if extra.name != "start.xyz":
                        if extra.is_file():
                            extra.unlink()
                        elif extra.is_dir():
                            shutil.rmtree(extra)
                shutil.copy2(start_path, dest / "start.xyz")
                row["light_dir"] = dest.relative_to(out_dir).as_posix()
                row["status"] = "start_only"
                n_start_only += 1
            rows.append(row)
            continue

        row["pos_file"] = pos.name
        try:
            last_xyz, n_frames = _read_last_xyz_frame_text(pos)
        except ValueError as exc:
            row["status"] = f"bad_pos:{exc}"
            n_fail += 1
            rows.append(row)
            continue

        row["n_frames"] = n_frames

        if args.write_in_place:
            (calc_dir / "final.xyz").write_text(last_xyz, encoding="utf-8")

        if out_dir is not None:
            dest = _light_dest(out_dir, k, p, structure_id)
            _write_light_pair(dest, start_path, last_xyz)
            row["light_dir"] = dest.relative_to(out_dir).as_posix()

        row["status"] = "ok"
        n_ok += 1
        rows.append(row)
        print(
            f"[ok] {rel}  frames={n_frames}  "
            f"start.xyz + final.xyz  (from {pos.name})"
        )

    fields = [
        "k",
        "p",
        "structure_id",
        "run_dir",
        "status",
        "has_start",
        "pos_file",
        "n_frames",
        "light_dir",
        "job_done",
    ]
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print()
    print(f"Heavy root:  {root}")
    if out_dir is not None:
        print(f"Light out:   {out_dir}")
        print(f"             (only start.xyz + final.xyz per isomer)")
    print(f"Isomers:     {len(calc_dirs)}")
    print(f"Extracted:   {n_ok}")
    print(f"Pending:     {n_pending} (no pos / not done)")
    if n_start_only:
        print(f"Start-only:  {n_start_only}")
    print(f"Failed:      {n_fail}")
    print(f"Inventory:   {inventory_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
