#!/usr/bin/env python3
"""Grow molecular (k+1) cores from a finished map run (package growth).

Example::

    python tools/run_molecular_growth.py \\
      --map-yaml geometry_packs/cdse_cdcl2/run_gxtb.yaml \\
      --growth geometry_packs/cdse_cdcl2/growth.yaml \\
      --parents /path/to/gxtb_cdse_target_k3_p1p7 \\
      --k-from 3 \\
      --output runs/growth_k3_to_k4 \\
      --cores-only

Without ``--cores-only``, each child bin is decorated and (if the pack has
relaxation enabled) fully optimised via the existing molecular pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from builder.nucleation.geometry_pack import load_geometry_pack  # noqa: E402
from builder.nucleation.molecular_growth import (  # noqa: E402
    GrowthConfig,
    run_growth_step,
    write_growth_summary,
)
from builder.nucleation.spec import load_nucleation_spec  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-yaml",
        type=Path,
        default=ROOT / "geometry_packs/cdse_cdcl2/run_gxtb.yaml",
        help="map/driver YAML (graph rules + relaxation)",
    )
    parser.add_argument(
        "--growth",
        type=Path,
        default=ROOT / "geometry_packs/cdse_cdcl2/growth.yaml",
        help="growth.yaml (packages, parents, shed, thermo refs)",
    )
    parser.add_argument(
        "--parents",
        type=Path,
        required=True,
        help="finished map run directory (index.csv + k###/p###/*_xtb.xyz)",
    )
    parser.add_argument("--k-from", type=int, required=True, help="parent k")
    parser.add_argument(
        "--p-parents",
        type=str,
        default="",
        help="optional comma list of parent p bins (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/molecular_growth",
    )
    parser.add_argument(
        "--cores-only",
        action="store_true",
        help="only build child core catalogs (no decorate/opt)",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="decorate graphs but skip 3D/opt",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    growth = GrowthConfig.from_yaml(args.growth)
    spec = load_nucleation_spec(str(args.map_yaml))
    pack = None
    if spec.geometry_pack:
        pack = load_geometry_pack(spec.geometry_pack)
    else:
        # driver with include: load pack from map yaml dir
        pack = load_geometry_pack(args.map_yaml)

    p_parents = None
    if args.p_parents.strip():
        p_parents = [int(x) for x in args.p_parents.split(",") if x.strip()]

    def progress(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    # copy growth config into output for provenance
    (args.output / "growth_used.yaml").write_text(args.growth.read_text())

    result = run_growth_step(
        run_dir=args.parents,
        k_from=args.k_from,
        growth=growth,
        map_spec=spec,
        pack=pack,
        p_parents=p_parents,
        decorate=not args.cores_only,
        embed=not args.no_embed and not args.cores_only,
        output_dir=None if args.cores_only else args.output,
        progress=progress,
    )
    write_growth_summary(result, args.output / "growth_channels.csv")
    with (args.output / "growth_parents.json").open("w") as handle:
        json.dump(result.parent_records, handle, indent=2)
    catalog_meta = {
        f"k{k}_p{p}": len(cores)
        for (k, p), cores in sorted(result.skeleton_catalog.items())
    }
    with (args.output / "growth_catalog.json").open("w") as handle:
        json.dump(
            {
                "k_from": result.k_from,
                "k_to": result.k_to,
                "parents_selected": result.parents_selected,
                "bins": catalog_meta,
                "n_channels": len(result.channels),
            },
            handle,
            indent=2,
        )
    # dump core edge lists for decorate-only resume
    cat_dir = args.output / "child_cores"
    cat_dir.mkdir(exist_ok=True)
    for (k, p), cores in result.skeleton_catalog.items():
        lines = [f"# k={k} p={p} n={len(cores)}"]
        for i, edges in enumerate(cores):
            edge_s = "|".join(f"{a}-{b}" for a, b in edges)
            lines.append(f"{i}\t{edge_s}")
        (cat_dir / f"k{k:03d}_p{p:03d}.tsv").write_text("\n".join(lines) + "\n")

    progress(
        f"[growth] done: parents={result.parents_selected} "
        f"channels={len(result.channels)} bins={catalog_meta}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
