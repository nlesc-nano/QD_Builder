#!/usr/bin/env python3
"""Export viewable multi-frame XYZ files from a lattice growth run.

For every (k,p) bin of a run directory (``zb_occupations.jsonl`` +
``k###/p###/*.xyz``) writes ``<out>/k###_p###.xyz`` with the
propagation-eligible endpoints sorted by MACE energy (lowest first), and
optionally ``<out>/k###_p###_offpath.xyz`` with the rejected/off-path ones.
Each frame's comment line carries: rank, structure id, E (eV), dE to the bin
minimum (eV), status, Se-skeleton family (short), parent, mu2/mu3 and
zb_score from the two-tier descriptors.  Open them in any viewer that reads
multi-frame XYZ (VESTA, Avogadro, ASE gui, OVITO, Jmol).

Also writes ``<out>/summary.csv`` with one row per exported frame.

Example::

    python tools/export_growth_view.py \\
        /path/growth_zb/mace/growth --parents /path/growth_zb/mace/k1_seed \\
        --out /path/growth_zb/mace/view --top 20 --offpath
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from builder.nucleation.mace_descriptors import describe  # noqa: E402
from builder.nucleation.molecular_zb_growth import (  # noqa: E402
    occupation_anion_skeleton_key,
    occupation_from_record,
)


def basin_map(run_dirs, pack_yaml, growth_yaml):
    """structure_id -> (basin_id, multiplicity, representative_id) using the driver's consolidation.

    Runs ``consolidate_relaxed_minima`` (graph+energy merge included) on the
    propagation-eligible endpoints of every (k,p) bin found in ``run_dirs``.
    """
    from builder.nucleation.geometry_pack import load_geometry_pack
    from builder.nucleation.molecular_growth import (
        GrowthConfig,
        consolidate_relaxed_minima,
        load_parents_from_run,
    )
    from builder.nucleation.spec import load_nucleation_spec

    spec = load_nucleation_spec(str(pack_yaml))
    pack = load_geometry_pack(pack_yaml)
    cfg = GrowthConfig.from_yaml(growth_yaml).minimum_consolidation
    out = {}
    for run_dir in run_dirs:
        index = Path(run_dir) / "index.csv"
        if not index.is_file():
            continue
        ks = sorted({int(line.split(",")[0]) for line in index.read_text().splitlines()[1:] if line.strip()})
        for k in ks:
            parents = load_parents_from_run(Path(run_dir), k=k, spec=spec, pack=pack)
            for p in sorted({x.p for x in parents}):
                members = [x for x in parents if x.p == p and x.propagation_eligible]
                for c in consolidate_relaxed_minima(members, cfg, spec):
                    ids = [m.structure_id for m in c.members]
                    for sid in ids:
                        out[sid] = (c.representative.structure_id, len(ids))
    return out


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    sym, pos = [], []
    for line in lines[2 : 2 + n]:
        f = line.split()
        sym.append(f[0])
        pos.append([float(x) for x in f[1:4]])
    return sym, np.asarray(pos, float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--parents", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top", type=int, default=0, help="frames per bin (0 = all)")
    ap.add_argument("--offpath", action="store_true", help="also export rejected / off-path endpoints")
    ap.add_argument("--pack", type=Path, help="run_mace_mh1.yaml (with --growth: export one frame per relaxed basin instead of per endpoint)")
    ap.add_argument("--growth", type=Path, help="growth_mace.yaml (consolidation thresholds)")
    args = ap.parse_args(argv)
    basins = basin_map([args.run_dir] + list(args.parents), args.pack, args.growth) if args.pack and args.growth else None

    records = []
    for base in [args.run_dir] + list(args.parents):
        base = base.expanduser().resolve()
        for line in (base / "zb_occupations.jsonl").read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                rec["_base"] = base
                records.append(rec)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    bins = {}
    for rec in records:
        occ = occupation_from_record(rec["occupation"])
        rec["_k"], rec["_p"] = int(occ.k), int(occ.p)
        rec["_family"] = occupation_anion_skeleton_key(occ, "Se", 0.2)[:8]
        bins.setdefault((rec["_k"], rec["_p"]), []).append(rec)

    for (k, p), recs in sorted(bins.items()):
        for eligible, suffix in ((True, ""), (False, "_offpath")):
            if not eligible and not args.offpath:
                continue
            sel = [r for r in recs if bool(r.get("propagation_eligible")) == eligible and r.get("energy_eV") is not None and r.get("xyz")]
            if basins is not None and eligible:
                # one frame per basin: the consolidation representative
                sel = [r for r in sel if basins.get(r["structure_id"], (r["structure_id"], 1))[0] == r["structure_id"]]
            sel.sort(key=lambda r: float(r["energy_eV"]))
            if args.top:
                sel = sel[: args.top]
            if not sel:
                continue
            emin = min(float(r["energy_eV"]) for r in recs if r.get("propagation_eligible") and r.get("energy_eV") is not None) if any(r.get("propagation_eligible") for r in recs) else float(sel[0]["energy_eV"])
            frames = []
            for rank, r in enumerate(sel, 1):
                xyz = r["_base"] / r["xyz"]
                if not xyz.is_file():
                    continue
                sym, pos = read_xyz(xyz)
                d = describe(sym, pos)
                e = float(r["energy_eV"])
                status = "in-path" if eligible else ("off-path:" + r.get("topology_status", "") if r.get("chemically_ok") else "failed:" + ",".join(sorted({v.split(":")[0] for v in r.get("violations", [])})))
                mult = basins.get(r["structure_id"], (None, 1))[1] if (basins is not None and eligible) else 1
                comment = (f"rank={rank} id={r['structure_id']} E_eV={e:.6f} dE_eV={e - emin:.4f} status={status} basin_multiplicity={mult} "
                           f"family={r['_family']} parent={r.get('parent_structure_id', '')} mu2={d['mu2']} mu3={d['mu3']} "
                           f"zb_score={d['zb_score']:.3f} se_cn={d['se_cn_strong']}")
                frames.append(f"{len(sym)}\n{comment}\n" + "\n".join(f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(sym, pos)))
                rows.append({"k": k, "p": p, "rank": rank, "structure_id": r["structure_id"], "E_eV": e, "dE_eV": round(e - emin, 4), "status": status, "basin_multiplicity": mult,
                             "family": r["_family"], "parent": r.get("parent_structure_id", ""), "mu2": d["mu2"], "mu3": d["mu3"], "zb_score": round(d["zb_score"] or 0, 3), "file": f"k{k:03d}_p{p:03d}{suffix}.xyz"})
            (args.out / f"k{k:03d}_p{p:03d}{suffix}.xyz").write_text("\n".join(frames) + "\n")
    with (args.out / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} frames in {len(bins)} bins to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
