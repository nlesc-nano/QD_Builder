#!/usr/bin/env python3
"""Grow molecular cores from a finished map / growth run (package growth).

Simple usage (recommended) — one pack folder that contains
``run_gxtb.yaml`` + ``growth.yaml`` (+ includes)::

    python tools/run_molecular_growth.py \\
      --pack-dir /path/to/graphs/growth \\
      --growth  growth_k2k3.yaml \\
      --parents  /path/to/runs/gxtb_cdse_target_k1k2_p1p5 \\
      --k-from 2 \\
      --p-parents all \\
      --output  /path/to/runs/growth_k2_to_k3

Named envelopes in the pack: growth_survey.yaml, growth_k2k3.yaml,
growth_k4k8.yaml, growth_k9k13.yaml, growth_k3k13.yaml (wide k=3→13),
growth_k1k13.yaml (older mixed envelope).
For a clean zb-only k=1→4 test use geometry_packs/cdse_cdcl2_zb
(--pack-dir .../cdse_cdcl2_zb --growth growth.yaml --k-from 1 --k-to 4).
Default is growth.yaml.

Parent p bins::

    --p-parents all          # default: every p present for k-from
    --p-parents 2,3,4        # only these p

Multi-step growth (chain k-from … k-to)::

    --k-from 2 --k-to 4      # runs k=2→3 then k=3→4
                             # parents of step 2 = --output of step 1

Restart (default on)::

    Resubmit the *same* command and *same* --output. Finished move-B opts,
    finished move-A bins, and finished k-steps are skipped. Logs append to
    ``<output>/growth_run.log`` (and to your shell redirect if you use ``>>``).

Optional: ``--cores-only`` to skip decorate/opt (catalog only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from builder.nucleation.geometry_pack import load_geometry_pack  # noqa: E402
from builder.nucleation.molecular_growth import (  # noqa: E402
    GrowthConfig,
    GrowthLog,
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
        if not map_p.is_file():
            raise SystemExit(f"missing {map_p}")
        if growth_yaml is not None:
            grow_p = Path(growth_yaml).expanduser()
            if not grow_p.is_file():
                grow_p = pack_dir / growth_yaml
            grow_p = grow_p.resolve()
        else:
            grow_p = pack_dir / "growth.yaml"
        if not grow_p.is_file():
            raise SystemExit(f"missing {grow_p}")
        return map_p, grow_p

    if map_yaml is None or growth_yaml is None:
        raise SystemExit(
            "Provide --pack-dir DIR  (preferred), or both --map-yaml and --growth"
        )
    return map_yaml.expanduser().resolve(), growth_yaml.expanduser().resolve()


def _parse_p_parents(raw: str) -> Optional[List[int]]:
    """Return None = all p; else list of p values.

    Accepts ``\"\"``, ``all``, ``*``, or ``1,2,3``.
    """

    text = (raw or "").strip()
    if not text or text.lower() in {"all", "*", "any"}:
        return None
    vals: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(int(part))
        except ValueError as exc:
            raise SystemExit(
                f"invalid --p-parents entry {part!r} "
                f"(use 'all' or comma list of integers)"
            ) from exc
    if not vals:
        return None
    return vals


def _run_one_step(
    *,
    parents_dir: Path,
    k_from: int,
    growth: GrowthConfig,
    spec,
    pack,
    p_parents: Optional[Sequence[int]],
    decorate: bool,
    embed: bool,
    output_dir: Path,
    log: GrowthLog,
    growth_yaml: Path,
    map_yaml: Path,
    resume: bool,
) -> object:
    """One k → k+1 growth; write channels / catalog under output_dir."""

    log.line(f"---- growth step k={k_from} -> k={k_from + 1} ----")
    log.line(f"pack     : {map_yaml.parent}")
    log.line(f"parents  : {parents_dir}")
    log.line(f"k-from   : {k_from}")
    log.line(f"k-to     : {k_from + 1}  (this step)")
    log.line(f"output   : {output_dir}")
    log.line(f"p-parents: {'all' if p_parents is None else list(p_parents)}")
    log.line(f"resume   : {resume}")

    result = run_growth_step(
        run_dir=parents_dir,
        k_from=k_from,
        growth=growth,
        map_spec=spec,
        pack=pack,
        p_parents=p_parents,
        decorate=decorate,
        embed=embed,
        output_dir=None if not decorate else output_dir,
        progress=log,
        resume=resume,
    )
    # per-step side files (suffix with k so multi-step does not clobber)
    tag = f"k{k_from:03d}_to_k{k_from + 1:03d}"
    write_growth_summary(result, output_dir / f"growth_channels_{tag}.csv")
    with (output_dir / f"growth_parents_{tag}.json").open("w") as handle:
        json.dump(result.parent_records, handle, indent=2)
    catalog_meta = {
        f"k{k}_p{p}": len(cores)
        for (k, p), cores in sorted(result.skeleton_catalog.items())
    }
    with (output_dir / f"growth_catalog_{tag}.json").open("w") as handle:
        json.dump(
            {
                "k_from": result.k_from,
                "k_to": result.k_to,
                "parents_selected": result.parents_selected,
                "bins": catalog_meta,
                "n_channels": len(result.channels),
                "pack_dir": str(map_yaml.parent),
                "p_parents": "all" if p_parents is None else list(p_parents),
                "resume": resume,
            },
            handle,
            indent=2,
        )
    cat_dir = output_dir / "child_cores"
    cat_dir.mkdir(exist_ok=True)
    for (k, p), cores in result.skeleton_catalog.items():
        lines = [f"# k={k} p={p} n={len(cores)}"]
        for i, edges in enumerate(cores):
            edge_s = "|".join(f"{a}-{b}" for a, b in edges)
            lines.append(f"{i}\t{edge_s}")
        (cat_dir / f"k{k:03d}_p{p:03d}.tsv").write_text("\n".join(lines) + "\n")

    log.line(
        f"DONE step k={k_from}->{k_from + 1}  parents={result.parents_selected}  "
        f"channels={len(result.channels)}  bins={catalog_meta}"
    )
    return result


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
        help=(
            "growth YAML.  With --pack-dir, a bare name such as "
            "growth_k2k3.yaml is resolved inside the pack folder.  "
            "Default: <pack-dir>/growth.yaml"
        ),
    )
    parser.add_argument(
        "--parents",
        type=Path,
        required=True,
        help="finished map/growth run (index.csv + k###/p###/*_xtb.xyz)",
    )
    parser.add_argument(
        "--k-from",
        type=int,
        required=True,
        help="starting parent k (first growth step)",
    )
    parser.add_argument(
        "--k-to",
        type=int,
        default=None,
        help=(
            "final child k to reach (default: k-from+1). "
            "Example: --k-from 2 --k-to 4 runs k=2→3 then k=3→4, "
            "using --output as the parent tree for later steps."
        ),
    )
    parser.add_argument(
        "--p-parents",
        type=str,
        default="all",
        help=(
            "parent p bins: 'all' (default) or comma list e.g. 2,3,4. "
            "Empty string also means all."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output directory for child map / catalogs (shared across --k-to steps)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "skip finished work under --output (default: on). "
            "Use --no-resume to force a full redo."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "append-only log path (default: <output>/growth_run.log). "
            "Also use shell '>>' for Slurm redirects so stdout continues."
        ),
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress almost all progress",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also print detailed [molecular] motif lines",
    )
    args = parser.parse_args(argv)

    map_yaml, growth_yaml = _resolve_pack_files(
        args.pack_dir, args.map_yaml, args.growth
    )
    growth = GrowthConfig.from_yaml(growth_yaml)
    spec = load_nucleation_spec(str(map_yaml))
    pack = load_geometry_pack(map_yaml)

    p_parents = _parse_p_parents(args.p_parents)
    k_from = int(args.k_from)
    k_to = int(args.k_to) if args.k_to is not None else k_from + 1
    if k_to <= k_from:
        raise SystemExit(f"--k-to ({k_to}) must be > --k-from ({k_from})")

    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log_path = (
        args.log_file.expanduser().resolve()
        if args.log_file is not None
        else out / "growth_run.log"
    )
    log = GrowthLog(
        verbose=args.verbose,
        quiet=args.quiet,
        log_path=log_path,
    )

    (out / "growth_used.yaml").write_text(growth_yaml.read_text())
    (out / "map_used.yaml").write_text(f"# loaded from\n# {map_yaml}\n")

    log.line(f"pack     : {map_yaml.parent}")
    log.line(f"parents0 : {args.parents}")
    log.line(f"k-from   : {k_from}")
    log.line(f"k-to     : {k_to}  (steps: {list(range(k_from, k_to))})")
    log.line(f"p-parents: {'all' if p_parents is None else p_parents}")
    log.line(f"output   : {out}")
    log.line(f"resume   : {args.resume}")
    log.line(f"log-file : {log_path}  (append-only)")

    decorate = not args.cores_only
    embed = not args.no_embed and not args.cores_only
    parents_dir = args.parents.expanduser().resolve()

    try:
        for k in range(k_from, k_to):
            step_parents = parents_dir if k == k_from else out
            if k > k_from and not decorate:
                log.line(
                    f"warning: --cores-only with --k-to>k-from+1: "
                    f"step k={k} has no energies in {step_parents}; "
                    f"parent load may be empty"
                )
            _run_one_step(
                parents_dir=step_parents,
                k_from=k,
                growth=growth,
                spec=spec,
                pack=pack,
                p_parents=p_parents,
                decorate=decorate,
                embed=embed,
                output_dir=out,
                log=log,
                growth_yaml=growth_yaml,
                map_yaml=map_yaml,
                resume=bool(args.resume),
            )

        log.line(f"ALL DONE  k={k_from} -> k={k_to}  output={out}")
    finally:
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
