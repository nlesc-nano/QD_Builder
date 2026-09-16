#!/usr/bin/env python3
"""Build the k=1 parent tree for a lattice growth run directly from the ZB lattice.

The g-xTB archive started from a lattice-free combinatorial map
(``gxtb_cdse_target_k1k2_p1p5``) whose k=1 rows were snapped onto lattice
occupations at growth time.  For a restart "built on the underlying lattice
from k=1" the seeds are generated here from the CIF lattice model itself:
one Cd-Se pair plus p precursor Cd on vacant tetrahedral sites of the Se
(``_add_precursor_cd``, every distinct cubic-invariant occupation, most
compact first), for p = 1 .. p_max(1) of the growth p-envelope.  Each
occupation is then decorated, embedded and relaxed by the same Move-Z bin
optimizer the growth steps use (``_opt_zb_occupations``), so the output is
a complete parents tree: ``index.csv``, ``k001/p00N/*_xtb.xyz`` and
``zb_occupations.jsonl`` with the stored occupations.

Example::

    /Users/ivaninfante/miniforge3/envs/orchestr_ai-mace/bin/python \\
        tools/bootstrap_zb_k1.py \\
        --pack-dir geometry_packs/cdse_cdcl2_zb \\
        --map-yaml run_mace_mh1.yaml --growth growth_mace.yaml \\
        --output runs/mace_growth/k1_seed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from builder.nucleation.geometry_pack import load_geometry_pack  # noqa: E402
from builder.nucleation.molecular_growth import (  # noqa: E402
    GrowthConfig,
    GrowthLog,
    _opt_zb_occupations,
)
from builder.nucleation.molecular_zb_growth import (  # noqa: E402
    ZbGrowStats,
    _add_precursor_cd,
    lattice_model,
    seed_occupation,
)
from builder.nucleation.spec import load_nucleation_spec  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack-dir", type=Path, default=ROOT / "geometry_packs/cdse_cdcl2_zb")
    ap.add_argument("--map-yaml", default="run_mace_mh1.yaml", help="driver YAML inside --pack-dir")
    ap.add_argument("--growth", default="growth_mace.yaml", help="growth YAML inside --pack-dir")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--p-max", type=int, default=0, help="override p_max(1) from the growth envelope")
    ap.add_argument("--cap", type=int, default=8, help="distinct Cd realisations kept per p (most compact first)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    map_yaml = args.pack_dir / args.map_yaml
    growth = GrowthConfig.from_yaml(args.pack_dir / args.growth)
    spec = load_nucleation_spec(str(map_yaml))
    pack = load_geometry_pack(map_yaml)
    model = lattice_model(spec)
    window = growth.window_for(1)
    p_max = int(args.p_max) if args.p_max > 0 else window.p_surf(1) + int(window.p_slack)

    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = GrowthLog(verbose=args.verbose, log_path=out / "bootstrap_k1.log")
    log.line(f"k=1 lattice bootstrap: pack={map_yaml} growth={args.growth} p=1..{p_max} cap={args.cap}")

    seed = seed_occupation(spec, model)
    zb_seeds = {}
    for p in range(1, p_max + 1):
        occupations = _add_precursor_cd(seed, p, spec, model, cap=int(args.cap))
        if not occupations:
            log.line(f"  p={p}: no lattice occupation")
            continue
        zb_seeds[(1, p)] = list(occupations)
        log.line(f"  p={p}: {len(occupations)} occupation(s): " + ", ".join(o.occupation_id[-8:] for o in occupations))

    child_minima = {}
    bin_ranks = {}
    stats = ZbGrowStats()
    try:
        _opt_zb_occupations(
            zb_seeds,
            growth=growth,
            map_spec=spec,
            pack=pack,
            output_dir=out,
            progress=log,
            child_minima=child_minima,
            bin_ranks=bin_ranks,
            zb_stats=stats,
            resume=not args.no_resume,
        )
        log.line(stats.as_log())
        for (k, p), best in sorted(child_minima.items()):
            log.line(f"  bin k={k} p={p}: best {best['structure_id']} E={best['energy_eV']:.6f} eV, {len(bin_ranks.get((k, p), []))} accepted")
        log.line(f"DONE k=1 bootstrap -> {out}")
    finally:
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
