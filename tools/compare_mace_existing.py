#!/usr/bin/env python3
"""Compare MACE relaxations with the existing CP2K CdSe calibration archive.

This deliberately samples already analyzed structures; it does not generate a
new nucleation search. It is intended as a first backend sanity check before
using MACE in growth. The MACE worker is run once and kept resident.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path


def read_xyz(path: Path, *, last: bool = False):
    lines = path.read_text().splitlines()
    offset = 0
    if last:
        while offset < len(lines):
            n = int(lines[offset].split()[0])
            next_offset = offset + n + 2
            if next_offset >= len(lines):
                break
            offset = next_offset
    n = int(lines[offset].split()[0])
    symbols, positions = [], []
    for line in lines[offset + 2 : offset + 2 + n]:
        fields = line.split()
        symbols.append(fields[0])
        positions.append([float(value) for value in fields[1:4]])
    return symbols, positions


def resolve_relocated(path: Path) -> Path:
    """The archived manifests predate the map→backup/map relocation."""
    if path.is_file():
        return path
    text = str(path)
    marker = "/Programs/Nucleation/map/"
    if marker in text:
        candidate = Path(text.replace(marker, "/Programs/Nucleation/backup/map/", 1))
        if candidate.is_file():
            return candidate
    return path


def rmsd_aligned(a, b):
    """Kabsch RMSD, preserving atom order from the CP2K trajectories."""
    import numpy as np

    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        return None
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    covariance = x.T @ y
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = u @ correction @ vt
    return float(np.sqrt(np.mean(np.sum((x @ rotation - y) ** 2, axis=1))))


def run_gxtb(entry, binary: Path):
    """Relax one of the same starts with the local g-xTB executable."""
    with tempfile.TemporaryDirectory(prefix="compare_gxtb_") as directory:
        work = Path(directory)
        xyz = work / "in.xyz"
        lines = [str(len(entry["symbols"])), entry["id"]]
        lines.extend(
            f"{symbol} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}"
            for symbol, pos in zip(entry["symbols"], entry["positions"])
        )
        xyz.write_text("\n".join(lines) + "\n")
        (work / "xcontrol").write_text("$opt\n  maxcycle=150\n$end\n")
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "1")
        try:
            proc = subprocess.run(
                [str(binary), "in.xyz", "--gxtb", "--opt", "--input", "xcontrol"],
                cwd=work, capture_output=True, text=True, env=env,
                timeout=600, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        output = work / "xtbopt.xyz"
        if not output.is_file():
            detail = (proc.stdout + proc.stderr).strip().splitlines()
            return {"ok": False, "error": detail[-1] if detail else "no xtbopt.xyz"}
        symbols, positions = read_xyz(output)
        comment = output.read_text().splitlines()[1]
        match = re.search(r"energy:\s*(-?\d+\.\d+)", comment)
        energy = float(match.group(1)) * 27.211386245988 if match else None
        return {"ok": energy is not None, "energy_eV": energy, "positions": positions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            "/Users/ivaninfante/Documents/University/Programs/Nucleation/"
            "backup/map/runs/cdse_map/analysis_all_dft/structures.csv"
        ),
    )
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--head", default="omat_pbe")
    parser.add_argument("--gxtb", type=Path, default=None, help="also run local g-xTB on the same starts")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--max-structures", type=int, default=8)
    args = parser.parse_args()

    analysis_dir = args.analysis_dir or args.archive.parent
    rows = [row for row in csv.DictReader(args.archive.open()) if row.get("analysis_ready") == "1"]
    # One DFT bin winner per k, with a few p values spanning the existing map.
    selected = []
    seen_bins = set()
    for row in rows:
        if row.get("dft_rank_in_bin") != "1":
            continue
        k = int(row["k"])
        bin_key = (k, int(row["p"]))
        if bin_key in seen_bins:
            continue
        final = resolve_relocated(analysis_dir / row["relaxed_xyz"])
        trajectory = resolve_relocated(Path(row.get("trajectory_path") or ""))
        if not final.is_file() and trajectory.is_file():
            final = trajectory
        source = resolve_relocated(Path(row["source_xyz"]))
        if source.is_file() and final.is_file():
            selected.append((row, source, final))
            seen_bins.add(bin_key)
        if len(selected) >= args.max_structures:
            break
    if not selected:
        raise SystemExit("no analysis-ready DFT structures could be resolved")

    structures = []
    metadata = []
    for row, source, final in selected:
        symbols, positions = read_xyz(source)
        final_symbols, final_positions = read_xyz(final, last=final.name == "CdSe-pos-1.xyz")
        if symbols != final_symbols:
            raise SystemExit(f"atom ordering differs for {row['structure_id']}")
        structures.append({"id": row["structure_id"], "symbols": symbols, "positions": positions})
        metadata.append(
            {
                "id": row["structure_id"],
                "k": int(row["k"]),
                "p": int(row["p"]),
                "dft_energy_eV": float(row["energy_eV"]),
                "source_xyz": str(source),
                "dft_xyz": str(final),
                "rmsd_start_to_dft_A": rmsd_aligned(positions, final_positions),
                "dft_positions": final_positions,
            }
        )

    request = {
        "command": "relax",
        "settings": {
            "device": args.device,
            "dtype": args.dtype,
            "optimizer": "BFGS",
            "fmax": 0.02,
            "max_steps": args.max_steps,
            "maxstep": 0.20,
            "accept_maxcycle": True,
        },
        "structures": structures,
    }
    worker = Path(__file__).resolve().with_name("mace_worker.py")
    proc = subprocess.Popen(
        [str(args.python), str(worker), "--model", str(args.model), "--head", args.head,
         "--device", args.device, "--dtype", args.dtype],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    stdout, _ = proc.communicate(json.dumps(request) + "\n", timeout=3600)
    reply = json.loads(stdout.strip().splitlines()[-1])
    if not reply.get("ok"):
        raise SystemExit(reply.get("error", "MACE worker failed"))
    by_id = {result["id"]: result for result in reply["results"]}
    for item in metadata:
        result = by_id[item["id"]]
        item.update(
            {
                key: result.get(key)
                for key in ("ok", "energy_eV", "steps", "converged", "max_force", "status", "error", "wall_seconds")
            }
        )
        item["rmsd_mace_to_dft_A"] = rmsd_aligned(result.get("positions"), item.pop("dft_positions"))
        if args.gxtb:
            gx = run_gxtb(
                next(entry for entry in structures if entry["id"] == item["id"]),
                args.gxtb,
            )
            item.update({f"gxtb_{key}": value for key, value in gx.items() if key != "positions"})
            item["rmsd_gxtb_to_dft_A"] = rmsd_aligned(gx.get("positions"), read_xyz(Path(item["dft_xyz"]), last=Path(item["dft_xyz"]).name == "CdSe-pos-1.xyz")[1]) if gx.get("ok") else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"settings": vars(args), "results": metadata}, indent=2, default=str) + "\n")
    print(json.dumps({"output": str(args.output), "n": len(metadata), "results": metadata}, indent=2))


if __name__ == "__main__":
    main()
