#!/usr/bin/env python3
"""MACE-MH-1 vs CP2K/PBE on the full CdSe/CdCl2 DFT calibration archive.

Handoff step 3, with coordination descriptors rather than only RMSD.  For
every analysis-ready DFT structure whose relaxed endpoint is on disk:

A. ``from_dft``   -- relax the DFT endpoint with MACE.  If PBE's minimum is
                    also a MACE minimum the structure barely moves; a shift
                    to higher Se coordination / longer Cd–Se contacts means
                    the two surfaces disagree.
B. ``from_start`` -- when the pre-DFT starting geometry is also on disk,
                    relax it with MACE and compare the MACE endpoint to the
                    DFT endpoint (same-basin test from the same start).

Descriptors (``describe`` from ``mace_reopt_subset``) are computed for the
DFT endpoint and for each MACE endpoint: Cd–Se graph, Se/Cd/Cl CN
histograms, mu2/mu3, bridge load, min pair distances, plus Kabsch RMSD.

Energies stay in their own namespaces (``dft_energy_eV`` from the CSV,
``mace_energy_eV`` from the worker) and are never compared directly.

Output: ``<out>/results.jsonl`` (resumable, one line per DFT structure),
``<out>/manifest.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_mace_existing import read_xyz  # noqa: E402
from mace_reopt_subset import (  # noqa: E402
    MACE_PYTHON,
    MODEL,
    Worker,
    append_jsonl,
    artifact_flags,
    describe,
    kabsch_rmsd,
)

ARCHIVE = Path(
    "/Users/ivaninfante/Documents/University/Programs/Nucleation/backup/map/runs/cdse_map/analysis_all_dft/structures.csv"
)


def reloc(path: str) -> Optional[Path]:
    for cand in (path, path.replace("/Programs/Nucleation/map/", "/Programs/Nucleation/backup/map/", 1)):
        if cand and os.path.isfile(cand):
            return Path(cand)
    return None


def dft_endpoint(analysis_dir: Path, row: Dict[str, str]) -> Optional[Path]:
    rel = row.get("relaxed_xyz") or ""
    for cand in (analysis_dir / rel, analysis_dir / rel.replace("relaxed/", "relaxed 2/", 1)):
        if rel and cand.is_file():
            return cand
    return reloc(row.get("trajectory_path") or "")


def public(desc: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in desc.items() if not k.startswith("_") and k != "edges"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--preflight", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--python", default=MACE_PYTHON)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--head", default="omat_pbe")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    pre = json.loads(args.preflight.read_text())
    if not pre.get("ok") or pre.get("fingerprint", {}).get("device") != args.device:
        raise SystemExit("preflight did not pass for the requested device; refusing to run")
    fingerprint = pre["fingerprint"]

    analysis_dir = args.archive.parent
    rows = [r for r in csv.DictReader(args.archive.open()) if r.get("analysis_ready") == "1"]
    jobs = []
    for r in rows:
        end = dft_endpoint(analysis_dir, r)
        if not end:
            continue
        jobs.append((r, end, reloc(r.get("source_xyz") or "")))
    if args.limit:
        jobs = jobs[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.jsonl"
    done = set()
    if results_path.exists():
        with results_path.open() as fh:
            done = {json.loads(line)["structure_id"] for line in fh}
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "archive": str(args.archive),
                "dft_method": "CP2K PBE / DZVP-MOLOPT-GGA-GTH / no dispersion / non-periodic",
                "fingerprint": fingerprint,
                "settings": {"fmax": args.fmax, "max_steps": args.max_steps, "optimizer": "BFGS", "maxstep": 0.20},
                "n_rows": len(rows),
                "n_with_endpoint": len(jobs),
                "n_with_start": sum(1 for _, _, s in jobs if s),
            },
            indent=2,
        )
    )
    print(f"[dft] {len(jobs)} DFT endpoints ({sum(1 for _,_,s in jobs if s)} with start), {len(done)} done", file=sys.stderr)

    settings = {
        "device": args.device, "dtype": "float32", "optimizer": "BFGS",
        "fmax": args.fmax, "max_steps": args.max_steps, "maxstep": 0.20, "accept_maxcycle": True,
    }
    worker = Worker(args.python, args.model, args.head, args.device, args.out / "worker.log")
    try:
        for n, (r, end_path, start_path) in enumerate(jobs, 1):
            sid = r["structure_id"]
            if sid in done:
                continue
            symbols, dft_pos = read_xyz(end_path, last=end_path.name == "CdSe-pos-1.xyz")
            dft_pos = np.asarray(dft_pos, float)
            dft_desc = describe(symbols, dft_pos)
            row: Dict[str, Any] = {
                "structure_id": sid,
                "k": int(r["k"]),
                "p": int(r["p"]),
                "dft_energy_eV": float(r["energy_eV"]) if r.get("energy_eV") else None,
                "dft_rank_in_bin": r.get("dft_rank_in_bin"),
                "dft_geometry_converged": r.get("geometry_converged"),
                "dft_xyz": str(end_path),
                "start_xyz": str(start_path) if start_path else None,
                "mace_protocol": fingerprint,
                "symbols": symbols,
                "dft": public(dft_desc),
                "dft_positions": dft_pos.tolist(),
            }

            def run(tag: str, pos: np.ndarray) -> None:
                res = worker.relax({"id": f"{sid}:{tag}", "symbols": symbols, "positions": pos.tolist()}, settings)
                block: Dict[str, Any] = {
                    "ok": bool(res.get("ok")), "status": res.get("status"), "error": res.get("error") or "",
                    "steps": res.get("steps"), "wall_s": round(res.get("wall_seconds", 0.0), 2),
                }
                if res.get("ok") and res.get("positions"):
                    endp = np.asarray(res["positions"], float)
                    d = describe(symbols, endp)
                    block.update(
                        {
                            "mace_energy_eV": res["energy_eV"],
                            "rmsd_to_dft_A": round(kabsch_rmsd(endp, dft_pos), 4),
                            "rmsd_from_input_A": round(kabsch_rmsd(endp, pos), 4),
                            "core_same_as_dft": d["core_hash"] == dft_desc["core_hash"],
                            "core_edges_shared_with_dft": len(d["_core_edges"] & dft_desc["_core_edges"]),
                            "artifact_flags": artifact_flags(d["min_distances"]),
                            "desc": public(d),
                            "positions": endp.tolist(),
                        }
                    )
                row[tag] = block

            run("from_dft", dft_pos)
            if start_path:
                s_sym, s_pos = read_xyz(start_path)
                if s_sym == symbols:
                    s_pos = np.asarray(s_pos, float)
                    row["rmsd_start_to_dft_A"] = round(kabsch_rmsd(s_pos, dft_pos), 4)
                    run("from_start", s_pos)
                else:
                    row["from_start"] = {"ok": False, "error": "atom ordering differs from DFT endpoint"}
            append_jsonl(results_path, row)
            fd = row["from_dft"]
            msg = f"[dft] {n}/{len(jobs)} {sid} k={row['k']} p={row['p']} from_dft: {fd['status']} steps={fd['steps']}"
            if fd.get("ok"):
                msg += (
                    f" rmsd={fd['rmsd_to_dft_A']} core_same={fd['core_same_as_dft']}"
                    f" SeCN {row['dft']['se_cn']}->{fd['desc']['se_cn']} mu2 {row['dft']['mu2']}->{fd['desc']['mu2']}"
                    f" art={fd['artifact_flags'] or '-'}"
                )
            else:
                msg += f" ERROR {fd['error'][:100]}"
            if "from_start" in row and row["from_start"].get("ok"):
                fs = row["from_start"]
                msg += f" | from_start: rmsd_to_dft={fs['rmsd_to_dft_A']} (start was {row['rmsd_start_to_dft_A']}) core_same={fs['core_same_as_dft']}"
            print(msg, file=sys.stderr, flush=True)
    finally:
        worker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
