#!/usr/bin/env python3
"""
Generate a size series of builder structures for the QDSpace library.

For every centre species and size in the config, build the preset recipe
(clean) and, when the {111} reconstruction applies, a reconstructed variant.
Structures are deduplicated by fingerprint (the smallest size that produces a
structure keeps it; larger sizes are recorded in `size.unit_cells_all`),
checked (neutral, no atom clashes, minimum core size) and written as

    <out>/<material>/<id>/start.xyz
    <out>/<material>/<id>/recipe.yaml
    <out>/<material>/<id>/record.json
    <out>/<material>/review.csv  and  review.md   (every build, incl. rejects)

Usage:
    python -m builder.scripts.generate_library examples/library/cdse_zb.yaml [--out DIR]
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from ..library_record import describe_structure, make_record, min_distance, read_xyz_first_frame
from ..main import main as builder_main


def _qd_builder_revision() -> Dict[str, object]:
    here = Path(__file__).resolve().parent
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=here, capture_output=True, text=True, check=True
        ).stdout.strip()
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--", "src"], cwd=root, capture_output=True, text=True
        ).stdout.strip())
        return {"qd_builder_commit": rev, "qd_builder_dirty": dirty}
    except Exception:
        return {"qd_builder_commit": None, "qd_builder_dirty": None}


def _sizes(spec) -> List[float]:
    if isinstance(spec, dict):
        start, stop, step = float(spec["start"]), float(spec["stop"]), float(spec["step"])
        n = int(round((stop - start) / step)) + 1
        return [round(start + i * step, 4) for i in range(n)]
    return [float(x) for x in spec]


def _build(cif: str, recipe: dict, positive_q_mode: str, workdir: Path) -> Optional[dict]:
    """Run the builder in-process; return symbols, coordinates, manifest and log."""
    workdir.mkdir(parents=True, exist_ok=True)
    yml = workdir / "recipe.yaml"
    yml.write_text(yaml.safe_dump(recipe, sort_keys=False))
    out = workdir / "out.xyz"
    log = io.StringIO()
    argv = [cif, str(yml), "-o", str(out), "--center", "--positive-q-mode", positive_q_mode]
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            rc = builder_main(argv)
    except SystemExit as exc:
        rc = exc.code
    except Exception as exc:  # keep the scan going; the review table records it
        return {"error": f"{type(exc).__name__}: {exc}", "log": log.getvalue()}
    if rc not in (0, None) or not out.exists():
        return {"error": f"builder exit code {rc}", "log": log.getvalue()}
    symbols, pts = read_xyz_first_frame(str(out))
    manifest_path = workdir / "out.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return {"symbols": symbols, "pts": pts, "manifest": manifest, "log": log.getvalue()}


def generate(config_path: str, out_dir: Optional[str] = None) -> Path:
    cfg_path = Path(config_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    material = cfg["material"]
    family = cfg["family"]
    phase = cfg["phase"]
    cif = str((cfg_path.parent / cfg["cif"]).resolve())
    native_order = list(cfg["native_elements"])
    preset = cfg["preset"]
    charges = dict(preset["charges"])
    posq = cfg.get("positive_q_mode", "add")
    min_core = int(cfg.get("min_core_atoms", 20))
    min_dist = float(cfg.get("min_interatomic_distance", 1.8))
    reconstruction = cfg.get("reconstruction", "auto")
    out_root = Path(out_dir or (cfg_path.parent / cfg.get("out_dir", "out"))).resolve() / material
    out_root.mkdir(parents=True, exist_ok=True)
    revision = _qd_builder_revision()

    kept: Dict[str, dict] = {}   # fingerprint -> record
    review: List[dict] = []

    def consider(centre: str, size: float, surface: str, recipe: dict, res: Optional[dict], parent: Optional[str]):
        row = {"centre": centre, "unit_cells": size, "surface": surface, "id": "", "formula": "",
               "d_nm": "", "n_atoms": "", "Q": "", "status": ""}
        if res is None or "skip" in res:
            reason = (res or {}).get("skip", "not applicable")
            row["status"] = f"skipped: {reason}"
            review.append(row)
            return None
        if "error" in res:
            row["status"] = f"rejected: {res['error']}"
            review.append(row)
            return None
        desc = describe_structure(res["symbols"], res["pts"], native_order=native_order, charges=charges)
        row.update(formula=desc["formula"], d_nm=desc["size"]["d_nm"], n_atoms=desc["n_atoms"],
                   Q=desc["total_charge"])
        reasons = []
        if desc["total_charge"] != 0:
            reasons.append(f"charge {desc['total_charge']:+d}")
        if desc["size"]["n_core"] < min_core:
            reasons.append(f"core {desc['size']['n_core']} < {min_core} atoms")
        dmin = min_distance(res["pts"])
        if dmin < min_dist:
            reasons.append(f"min distance {dmin:.2f} Å")
        if desc["centre"] != centre:
            reasons.append(f"centre detected as {desc['centre']}")
        if reasons:
            row["status"] = "rejected: " + "; ".join(reasons)
            review.append(row)
            return None
        fp = desc["fingerprint"]
        if fp in kept:
            rec = kept[fp]
            rec["size"]["unit_cells_all"].append(size)
            row.update(id=rec["id"], status=f"duplicate of {rec['id']}")
            review.append(row)
            return rec
        desc["size"].update(unit_cells=size, unit_cells_all=[size])
        origin = {
            "generator": "builder.scripts.generate_library",
            "config": cfg_path.name,
            **revision,
            "cif": Path(cif).name,
            "positive_q_mode": posq,
            "recipe": recipe,
        }
        recon_ledger = res["manifest"].get("surface_reconstruction_ledger")
        if recon_ledger:
            origin["reconstruction_ledger"] = recon_ledger
        rec = make_record(
            material=material, family=family, phase=phase, surface=surface, source="builder",
            description=desc,
            stages=[{"stage": "start", "file": "start.xyz"}],
            origin=origin,
            extra={"parent": parent} if parent else None,
        )
        kept[fp] = rec
        target = out_root / rec["id"]
        target.mkdir(parents=True, exist_ok=True)
        _write_xyz(target / "start.xyz", res["symbols"], res["pts"], rec["id"])
        (target / "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
        row.update(id=rec["id"], status="kept")
        review.append(row)
        return rec

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for centre in cfg["centres"]:
            for size in _sizes(cfg["sizes"]):
                recipe = copy.deepcopy(preset)
                recipe["size_unit_cells"] = [size, size, size]
                recipe["construction_origin"] = {"center_on_species": centre}
                tag = f"{centre}_{size}"
                clean = consider(centre, size, "clean", recipe,
                                 _build(cif, recipe, posq, tmp / f"{tag}_clean"), None)
                if reconstruction == "never":
                    continue
                r_recipe = copy.deepcopy(recipe)
                r_recipe.setdefault("post_treatment", {})["surface_reconstruction"] = {
                    "enabled": True, "ligand": preset["passivation"].get("ligand", "Cl"),
                }
                res = _build(cif, r_recipe, posq, tmp / f"{tag}_recon")
                ledger = (res or {}).get("manifest", {}).get("surface_reconstruction_ledger", {}) if res else {}
                if res and "error" not in res and ledger.get("status") != "applied":
                    res = {"skip": ledger.get("reason", "reconstruction not applied")}
                consider(centre, size, "reconstructed", r_recipe, res, clean["id"] if clean else None)

    for rec in kept.values():
        (out_root / rec["id"] / "record.json").write_text(json.dumps(rec, indent=2))
    _write_review(out_root, review, material)
    return out_root


def _write_xyz(path: Path, symbols, pts, comment: str) -> None:
    lines = [str(len(symbols)), comment]
    lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, np.asarray(pts))]
    path.write_text("\n".join(lines) + "\n")


def _write_review(out_root: Path, review: List[dict], material: str) -> None:
    cols = ["centre", "unit_cells", "surface", "status", "id", "formula", "d_nm", "n_atoms", "Q"]
    with open(out_root / "review.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(review)
    kept = [r for r in review if r["status"] == "kept"]
    md = [f"# {material} library series", "",
          f"{len(kept)} unique structures kept out of {len(review)} builds.", "",
          "| centre | cells | surface | status | formula | d (nm) | atoms |",
          "|---|---|---|---|---|---|---|"]
    for r in review:
        md.append(f"| {r['centre']} | {r['unit_cells']} | {r['surface']} | {r['status']} | "
                  f"{r['formula']} | {r['d_nm']} | {r['n_atoms']} |")
    (out_root / "review.md").write_text("\n".join(md) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="library series config (YAML)")
    ap.add_argument("--out", help="output directory (default: out_dir in the config)")
    args = ap.parse_args(argv)
    out = generate(args.config, args.out)
    print(f"Library series written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
