#!/usr/bin/env python3
"""
Recompute the size descriptors of existing library records.

Rewrites `size` in every <series>/<id>/record.json from its start.xyz with
the current `library_record.size_descriptors`, without rebuilding the
structures.  The bulk volume per atom comes from the CIF of the record's
series config (origin.config, looked up in --configs); the generator's
`unit_cells` bookkeeping is kept.

Usage:
    python -m builder.scripts.refresh_library_sizes <series dir> [<series dir> ...] [--configs DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

from ..library_record import bulk_volume_per_atom, read_xyz_first_frame, size_descriptors

DEFAULT_CONFIGS = Path(__file__).resolve().parents[3] / "examples" / "library"


def material_bulk_volumes(configs: Path = DEFAULT_CONFIGS) -> Dict[Tuple[str, str], dict]:
    """Bulk volume per atom by element for every (material, phase) of the series configs."""
    out: Dict[Tuple[str, str], dict] = {}
    for cfg_path in sorted(Path(configs).glob("*.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        key = (cfg.get("material"), cfg.get("phase"))
        cif = cfg_path.parent / str(cfg.get("cif", ""))
        if None not in key and key not in out and cif.is_file():
            out[key] = bulk_volume_per_atom(str(cif))
    return out


def _bulk_volume(rec: dict, configs: Path, cache: Dict[str, Optional[dict]]) -> Optional[dict]:
    origin = rec.get("origin", {})
    key = f"{origin.get('config')}|{origin.get('cif')}"
    if key not in cache:
        cif = None
        cfg_path = configs / str(origin.get("config", ""))
        if cfg_path.is_file():
            cif = (cfg_path.parent / yaml.safe_load(cfg_path.read_text())["cif"]).resolve()
        elif origin.get("cif"):
            cif = next(configs.rglob(str(origin["cif"])), None)
        cache[key] = bulk_volume_per_atom(str(cif)) if cif and Path(cif).is_file() else None
    return cache[key]


def refresh(series_dirs, configs: Path = DEFAULT_CONFIGS) -> int:
    cache: Dict[str, Optional[dict]] = {}
    n = 0
    for d in series_dirs:
        for rec_path in sorted(Path(d).glob("*/record.json")):
            rec = json.loads(rec_path.read_text())
            symbols, pts = read_xyz_first_frame(str(rec_path.parent / "start.xyz"))
            bulk = _bulk_volume(rec, configs, cache)
            if bulk is None:
                print(f"[warn] {rec['id']}: no CIF for {rec.get('origin', {}).get('config')}; no volume diameter")
            kept = {k: v for k, v in rec.get("size", {}).items() if k.startswith("unit_cells")}
            rec["size"] = {**size_descriptors(symbols, pts, rec["core"], bulk), **kept}
            rec_path.write_text(json.dumps(rec, indent=2))
            n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("series", nargs="+", help="series directories holding <id>/record.json")
    ap.add_argument("--configs", default=str(DEFAULT_CONFIGS),
                    help="directory of the library series configs (default: examples/library)")
    args = ap.parse_args(argv)
    n = refresh(args.series, Path(args.configs))
    print(f"{n} records refreshed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
