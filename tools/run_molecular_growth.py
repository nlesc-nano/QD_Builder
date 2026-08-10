#!/usr/bin/env python3
"""Grow molecular (k+1) from a finished map run (package growth).

Simple usage (recommended) — one pack folder that contains
``run_gxtb.yaml`` + ``growth.yaml`` (+ includes)::

    python tools/run_molecular_growth.py \\
      --pack-dir /path/to/graphs/growth \\
      --parents  /path/to/runs/gxtb_cdse_target_k1k2_p1p5 \\
      --k-from 2 \\
      --output  /path/to/runs/growth_k2_to_k3

Optional: ``--p-parents 2,3`` to limit parent p-bins.
Optional: ``--cores-only`` to skip decorate/opt (catalog only).
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


def _resolve_pack_files(
    pack_dir: Path | None,
    map_yaml: Path | None,
    growth_yaml: Path | None,
) -> tuple[Path, Path]:
    """Return (map_yaml, growth_yaml) absolute paths."""

    if pack_dir is not None:
        pack_dir = pack_dir.expanduser().resolve()
        if not pack_dir.is_dir():
            raise SystemExit(f"--pack-dir is not a directory: {pack_dir}")
        map_p = pack_dir / "run_gxtb.yaml"
        grow_p = pack_dir / "growth.yaml"
        if not map_p.is_file():
            raise SystemExit(f"missing {map_p}")
        if not grow_p.is_file():
            raise SystemExit(f"missing {grow_p}")
        return map_p, grow_p

    if map_yaml is None or growth_yaml is None:
        raise SystemExit(
            "Provide --pack-dir DIR  (preferred), or both --map-yaml and --growth"
        )
    return map_yaml.expanduser().resolve(), growth_yaml.expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=None,
        help=(
            "folder with run_gxtb.yaml + growth.yaml (+ graph_rules/motifs/embed). "
            "Preferred single switch instead of --map-yaml/--growth."
        ),
    )
    parser.add_argument(
        "--map-yaml",
        type=Path,
        default=None,
        help="(advanced) map/driver YAML; use --pack-dir instead",
    )
    parser.add_argument(
        "--growth",
        type=Path,
        default=None,
        help="(advanced) growth.yaml; use --pack-dir instead",
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
        required=True,
        help="output directory for child map / catalogs",
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

    map_yaml, growth_yaml = _resolve_pack_files(
        args.pack_dir, args.map_yaml, args.growth
    )
    growth = GrowthConfig.from_yaml(growth_yaml)
    spec = load_nucleation_spec(str(map_yaml))
    pack = load_geometry_pack(map_yaml)

    p_parents = None
    if args.p_parents.strip():
        p_parents = [int(x) for x in args.p_parents.split(",") if x.strip()]

    def progress(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "growth_used.yaml").write_text(growth_yaml.read_text())
    (out / "map_used.yaml").write_text(
        f"# loaded from\n# {map_yaml}\n"
    )

    progress(f"[growth] pack map   : {map_yaml}")
    progress(f"[growth] pack growth: {growth_yaml}")
    progress(f"[growth] parents    : {args.parents}")
    progress(f"[growth] k-from     : {args.k_from}")

    result = run_growth_step(
        run_dir=args.parents.expanduser().resolve(),
        k_from=args.k_from,
        growth=growth,
        map_spec=spec,
        pack=pack,
        p_parents=p_parents,
        decorate=not args.cores_only,
        embed=not args.no_embed and not args.cores_only,
        output_dir=None if args.cores_only else out,
        progress=progress,
    )
    write_growth_summary(result, out / "growth_channels.csv")
    with (out / "growth_parents.json").open("w") as handle:
        json.dump(result.parent_records, handle, indent=2)
    catalog_meta = {
        f"k{k}_p{p}": len(cores)
        for (k, p), cores in sorted(result.skeleton_catalog.items())
    }
    with (out / "growth_catalog.json").open("w") as handle:
        json.dump(
            {
                "k_from": result.k_from,
                "k_to": result.k_to,
                "parents_selected": result.parents_selected,
                "bins": catalog_meta,
                "n_channels": len(result.channels),
                "pack_dir": str(map_yaml.parent),
            },
            handle,
            indent=2,
        )
    cat_dir = out / "child_cores"
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
