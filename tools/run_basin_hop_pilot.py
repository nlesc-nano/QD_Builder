#!/usr/bin/env python3
"""Basin-hopping pilot on one finished ``(k, p)`` bin of a growth run.

Answers, for a composition where the enumerator's answer is already known:

1. does a global search find anything below the bin minimum, and by how much?
2. do the new minima keep the seed's Cd-Se core, i.e. could they feed the
   existing lineage, or would they fork it?
3. how many of the enumerator's distinct minima does the search rediscover?
4. what does a new distinct minimum cost, against the enumerator's own rate?

Nothing in the growth pipeline is touched -- seeds are read from structures a
finished run already wrote.

    python tools/run_basin_hop_pilot.py \\
      --pack-dir geometry_packs/cdse_cdcl2_zb \\
      --run      /path/to/growth_k1_to_k4_zb2 \\
      --k 4 --p 3 \\
      --n-eligible 6 --n-offpath 3 \\
      --steps 200 --workers 8 \\
      --output   /path/to/bh_k4p3

Walkers are independent processes.  Every g-xTB call already runs in its own
``tempfile.mkdtemp`` scratch directory with ``OMP_NUM_THREADS=1``, so this is
also the smallest safe test of parallelising the backend before the growth loop
itself is touched.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from builder.nucleation.basin_hop import MOVES, basin_hop  # noqa: E402
from builder.nucleation.geometry_pack import load_geometry_pack  # noqa: E402
from builder.nucleation.spec import load_nucleation_spec  # noqa: E402
from builder.nucleation.xtb_relax import XtbSettings  # noqa: E402


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------


def _read_xyz(path: Path) -> Tuple[List[str], np.ndarray, Optional[float]]:
    """Symbols, coordinates and the ``energy_eV=`` field of a run's XYZ."""

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    count = int(lines[0].split()[0])
    energy: Optional[float] = None
    for token in lines[1].split():
        if token.startswith("energy_eV="):
            try:
                energy = float(token.split("=", 1)[1])
            except ValueError:
                energy = None
    symbols: List[str] = []
    points: List[List[float]] = []
    for line in lines[2 : 2 + count]:
        parts = line.split()
        symbols.append(parts[0])
        points.append([float(x) for x in parts[1:4]])
    return symbols, np.asarray(points, dtype=float), energy


def collect_seeds(
    run_dir: Path,
    *,
    k: int,
    p: int,
    n_eligible: int,
    n_offpath: int,
    spec: Any,
) -> List[Dict[str, Any]]:
    """Lowest-energy propagation-eligible and off-path endpoints of one bin."""

    from builder.nucleation.molecular_growth import load_parents_from_run

    seeds: List[Dict[str, Any]] = []

    parents = [
        parent
        for parent in load_parents_from_run(run_dir, k=k, spec=spec, p_values=[p])
        if parent.p == p
    ]
    parents.sort(key=lambda item: float(item.energy_eV))
    for parent in parents[: max(0, n_eligible)]:
        seeds.append(
            {
                "id": parent.structure_id,
                "symbols": list(parent.symbols),
                "positions": np.asarray(parent.coordinates, dtype=float),
                "energy_eV": float(parent.energy_eV),
                "origin": "eligible",
            }
        )

    if n_offpath > 0:
        bin_dir = run_dir / f"k{k:03d}" / f"p{p:03d}"
        offpath: List[Tuple[float, Path]] = []
        for path in sorted(bin_dir.glob("*_offpath.xyz")):
            if " 2." in path.name:  # rsync duplicate, same bytes as the original
                continue
            try:
                _sym, _pts, energy = _read_xyz(path)
            except (OSError, ValueError, IndexError):
                continue
            if energy is not None:
                offpath.append((energy, path))
        offpath.sort(key=lambda item: item[0])
        for energy, path in offpath[:n_offpath]:
            symbols, points, _energy = _read_xyz(path)
            seeds.append(
                {
                    "id": path.stem,
                    "symbols": symbols,
                    "positions": points,
                    "energy_eV": float(energy),
                    "origin": "offpath",
                }
            )

    return seeds


# ---------------------------------------------------------------------------
# walker process
# ---------------------------------------------------------------------------


def _serialise(minimum: Any) -> Dict[str, Any]:
    return {
        "structure_id": minimum.structure_id,
        "seed_id": minimum.seed_id,
        "step": minimum.step,
        "energy_eV": minimum.energy_eV,
        "symbols": list(minimum.symbols),
        "coordinates": [list(map(float, point)) for point in minimum.coordinates],
        "edges": [list(edge) for edge in minimum.edges],
        "core_edges": [list(edge) for edge in minimum.core_edges],
        "core_preserved": minimum.core_preserved,
        "zb_embeddable": minimum.zb_embeddable,
        "zb_reason": minimum.zb_reason,
        "violations": list(minimum.violations),
        "n_components": minimum.n_components,
        "converged": minimum.converged,
        "clean": minimum.clean,
    }


def _walk(job: Dict[str, Any]) -> Dict[str, Any]:
    """One walker, in its own process; re-loads the pack so args stay picklable.

    Streams every step to stdout and to ``walkers/<seed>.log``, and appends each
    new basin to ``walkers/<seed>.jsonl`` as it is found, so a long run is
    followable with ``tail -f`` instead of going silent for half an hour.
    """

    from builder.nucleation.molecular_zb_growth import lattice_model

    pack_yaml = Path(job["pack_yaml"])
    spec = load_nucleation_spec(str(pack_yaml))
    pack = load_geometry_pack(str(pack_yaml))
    settings = XtbSettings.from_pack((pack.raw or {}).get("relaxation"))
    # Pre-QC overlap wall, the same constant motif reconstruction uses before
    # handing a geometry to g-xTB.  Not the post-relaxation artifact floors.
    overlap_min_A = float(
        ((pack.raw or {}).get("reconstruction") or {}).get("overlap_min_A", 0.75)
    )
    try:
        zb_model = lattice_model(spec)
    except Exception:  # noqa: BLE001 — the zb label is a diagnostic
        zb_model = None

    seed = dict(job["seed"])
    seed["positions"] = np.asarray(seed["positions"], dtype=float)

    tag = str(seed["id"])[-14:]
    label = f"k{job['k']}p{job['p']} {tag}"
    walk_dir = Path(job["out_dir"]) / "walkers"
    walk_dir.mkdir(parents=True, exist_ok=True)
    log_path = walk_dir / f"{seed['id']}.log"
    min_path = walk_dir / f"{seed['id']}.jsonl"
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)
    min_handle = min_path.open("w", encoding="utf-8", buffering=1)

    def progress(line: str) -> None:
        print(f"[{label}] {line}", flush=True)
        log_handle.write(line + "\n")

    def on_minimum(minimum: Any) -> None:
        min_handle.write(json.dumps(_serialise(minimum)) + "\n")

    started = time.perf_counter()
    result = basin_hop(
        seed,
        settings,
        spec,
        k=int(job["k"]),
        p=int(job["p"]),
        steps=int(job["steps"]),
        temperature_eV=float(job["temperature_eV"]),
        moves=tuple(job["moves"]),
        amplitude_A=float(job["amplitude_A"]),
        rng_seed=int(job["rng_seed"]),
        zb_model=zb_model,
        overlap_min_A=overlap_min_A,
        motif_definitions=(pack.raw or {}).get("motifs"),
        progress=progress,
        on_minimum=on_minimum,
    )
    log_handle.close()
    min_handle.close()
    return {
        "seed_id": result.seed_id,
        "origin": seed.get("origin", "?"),
        "seed_energy_eV": result.seed_energy_eV,
        "steps_run": result.steps_run,
        "n_relaxations": result.n_relaxations,
        "n_accepted": result.n_accepted,
        "n_rejected_pre_qc": result.n_rejected_pre_qc,
        "n_failed": result.n_failed,
        "gain_eV": result.gain_eV,
        "wall_s": time.perf_counter() - started,
        "minima": [_serialise(m) for m in result.minima],
        "trajectory": [[s, e, bool(a)] for s, e, a in result.trajectory],
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _known_bin_minima(run_dir: Path, k: int, p: int, spec: Any) -> List[Any]:
    """The enumerator's own distinct basins for this bin."""

    from builder.nucleation.molecular_growth import (
        MinimumConsolidation,
        consolidate_relaxed_minima,
        load_parents_from_run,
    )

    parents = [
        parent
        for parent in load_parents_from_run(run_dir, k=k, spec=spec, p_values=[p])
        if parent.p == p
    ]
    if not parents:
        return []
    return consolidate_relaxed_minima(parents, MinimumConsolidation(enabled=True), spec)


def write_summary(
    out_dir: Path,
    *,
    k: int,
    p: int,
    run_dir: Path,
    walkers: Sequence[Dict[str, Any]],
    spec: Any,
    steps: int,
) -> Path:
    from builder.nucleation.basin_hop import BasinHopMinimum, _as_parent
    from builder.nucleation.molecular_growth import (
        MinimumConsolidation,
        relaxed_minimum_similarity,
    )

    # allow_reflection: see basin_hop.basin_hop -- an enantiomorph is the same
    # isomer to a search, even though it is a distinct route to the lineage.
    config = MinimumConsolidation(enabled=True, allow_reflection=True)
    clusters = _known_bin_minima(run_dir, k, p, spec)
    known_best = (
        min(float(c.representative.energy_eV) for c in clusters) if clusters else None
    )

    # Deduplicate across walkers so "distinct minima found" is a global count.
    found: List[Tuple[Dict[str, Any], Any]] = []
    for walker in walkers:
        for record in walker["minima"]:
            minimum = BasinHopMinimum(
                structure_id=record["structure_id"],
                seed_id=record["seed_id"],
                step=record["step"],
                energy_eV=record["energy_eV"],
                symbols=tuple(record["symbols"]),
                coordinates=np.asarray(record["coordinates"], dtype=float),
                edges=tuple(tuple(e) for e in record["edges"]),
                core_edges=tuple(tuple(e) for e in record["core_edges"]),
                core_preserved=record["core_preserved"],
                converged=record["converged"],
            )
            parent = _as_parent(minimum, k, p)
            if all(
                relaxed_minimum_similarity(other, parent, config, spec) is None
                for _rec, other in found
            ):
                found.append((record, parent))

    n_relax = sum(int(w["n_relaxations"]) for w in walkers)
    wall = sum(float(w["wall_s"]) for w in walkers)

    # A walker that wanders back into its own seed basin must not be reported as
    # a discovery, and neither must the seed record itself (step 0).  Without
    # this the headline reads as a search win whenever an off-path seed happens
    # to sit below the best eligible one -- which is exactly the case at k4p3.
    seed_parents = [parent for rec, parent in found if int(rec["step"]) == 0]
    discovered = [
        (rec, parent)
        for rec, parent in found
        if int(rec["step"]) > 0
        and all(
            relaxed_minimum_similarity(sp, parent, config, spec) is None
            for sp in seed_parents
        )
    ]
    seed_best = min(
        (
            float(w["seed_energy_eV"])
            for w in walkers
            if str(w.get("origin", "")) == "eligible"
        ),
        default=min((float(w["seed_energy_eV"]) for w in walkers), default=None),
    )
    bh_best = min((float(r["energy_eV"]) for r, _ in discovered), default=None)

    rediscovered = 0
    for cluster in clusters:
        if any(
            relaxed_minimum_similarity(cluster.representative, parent, config, spec)
            is not None
            for _rec, parent in discovered
        ):
            rediscovered += 1

    novel = [
        rec
        for rec, parent in discovered
        if all(
            relaxed_minimum_similarity(c.representative, parent, config, spec) is None
            for c in clusters
        )
    ]
    novel_clean = [r for r in novel if r["clean"]]
    novel_core = [r for r in novel if r["core_preserved"]]

    lines: List[str] = []
    lines.append(f"# Basin-hopping pilot — k={k} p={p}\n")
    lines.append(f"Run: `{run_dir}`\n")
    lines.append(
        f"{len(walkers)} walkers x {steps} steps, "
        f"{n_relax} g-xTB optimisations, {wall/3600:.2f} h of walker time.\n"
    )

    lines.append("\n## 1. Does the search beat the enumerator's bin minimum?\n\n")
    if known_best is None:
        lines.append("No eligible endpoints in this bin; nothing to compare.\n")
    elif bh_best is None:
        lines.append(
            "The search left every seed basin without finding a new minimum.\n"
        )
    else:
        gap = bh_best - known_best
        lines.append(
            f"- enumerator best: **{known_best:.6f} eV**\n"
            f"- best *discovered* minimum: **{bh_best:.6f} eV** "
            "(seed basins excluded)\n"
            f"- **gap: {gap:+.3f} eV** "
            f"({'search wins' if gap < -0.005 else 'enumerator validated'})\n"
        )
        if seed_best is not None:
            lines.append(
                f"\nSeeded from {len(walkers)} structures, best eligible seed "
                f"{seed_best:.6f} eV.  Discoveries are counted only when they "
                "fall outside every seed's basin, so a walker returning to where "
                "it started never scores.\n"
            )

    lines.append("\n## 2. Do the new minima keep the Cd-Se core?\n\n")
    lines.append(
        f"- distinct minima discovered: **{len(discovered)}** "
        f"(plus {len(seed_parents)} seed basins)\n"
        f"- of which not in the enumerator's set: **{len(novel)}**\n"
        f"- novel with the seed's Cd-Se core intact: **{len(novel_core)}**\n"
        f"- novel and audit-clean (converged, connected, no violations): "
        f"**{len(novel_clean)}**\n"
        "\nOnly core-preserving minima could feed the existing lineage; the rest\n"
        "would fork it.  `zb` below is the stricter lattice-snap test, and it is\n"
        "False even for endpoints the growth pipeline accepted, because a relaxed\n"
        "core drifts about 1 A off ideal sites -- reported, never used to filter.\n"
    )
    if novel:
        lines.append(
            "\n| structure | seed | step | E (eV) | core | zb | clean | violations |\n"
        )
        lines.append("|---|---|---:|---:|:-:|:-:|:-:|---|\n")
        for rec in sorted(novel, key=lambda r: r["energy_eV"])[:20]:
            viol = ", ".join(rec["violations"][:3]) or "—"
            lines.append(
                f"| `{rec['structure_id']}` | `{rec['seed_id']}` | {rec['step']} | "
                f"{rec['energy_eV']:.4f} | {'y' if rec['core_preserved'] else 'n'} | "
                f"{'y' if rec['zb_embeddable'] else 'n'} | "
                f"{'y' if rec['clean'] else 'n'} | {viol} |\n"
            )

    lines.append("\n## 3. Rediscovery of the enumerator's minima\n\n")
    lines.append(
        f"- enumerator distinct minima in this bin: **{len(clusters)}**\n"
        f"- rediscovered by the search: **{rediscovered}**"
        f" ({100*rediscovered/len(clusters):.0f}%)\n"
        if clusters
        else "- enumerator produced no minima here\n"
    )

    lines.append("\n## 4. Cost per distinct minimum\n\n")
    per = n_relax / max(1, len(discovered))
    clean_per = n_relax / max(1, len(novel_clean))
    lines.append(
        f"- basin hopping: {n_relax} optimisations / {len(discovered)} discovered "
        f"minima = **{per:.1f} per minimum**\n"
        f"- counting only novel *and* audit-clean minima: "
        f"**{clean_per:.1f} per minimum**\n"
    )
    if clusters:
        lines.append(
            f"- enumerator, same bin: see the run's own accounting "
            f"({len(clusters)} minima)\n"
        )

    lines.append("\n## Walkers\n\n")
    lines.append(
        "| seed | origin | E_seed | relax | escaped | accepted | failed "
        "| gain (eV) | wall |\n"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for w in sorted(walkers, key=lambda x: float(x["seed_energy_eV"])):
        escaped = max(0, len(w["minima"]) - 1)
        lines.append(
            f"| `{w['seed_id']}` | {w['origin']} | {w['seed_energy_eV']:.4f} | "
            f"{w['n_relaxations']} | {escaped} | {w['n_accepted']} | {w['n_failed']} | "
            f"{w['gain_eV']:+.3f} | {w['wall_s']/60:.1f} m |\n"
        )

    accepts = [w["n_accepted"] / max(1, w["n_relaxations"]) for w in walkers]
    escapes = [
        max(0, len(w["minima"]) - 1) / max(1, w["n_relaxations"]) for w in walkers
    ]
    if accepts:
        lines.append(
            f"\nMean acceptance {100*statistics.mean(accepts):.0f}%, "
            f"mean escape rate {100*statistics.mean(escapes):.0f}%.\n\n"
            "Acceptance counts a proposal that relaxed straight back into the "
            "same basin, so it runs high whenever the move is too small to "
            "leave; **escape rate** is the number that matters. If it is under "
            "~20%, raise `--amplitude` (0.35 A is only 0.14x the Cd-Se bond) or "
            "weight `surface_swap` more heavily.\n"
        )

    path = out_dir / "summary.md"
    path.write_text("".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack-dir", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="finished growth run directory")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--n-eligible", type=int, default=6, help="lowest-energy eligible seeds")
    ap.add_argument("--n-offpath", type=int, default=3, help="lowest-energy off-path seeds")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.15, help="Metropolis kT in eV")
    ap.add_argument("--amplitude", type=float, default=0.35, help="shake sigma in A")
    ap.add_argument("--moves", default=",".join(MOVES))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rng-seed", type=int, default=1729)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    pack_yaml = args.pack_dir / "run_gxtb.yaml"
    if not pack_yaml.is_file():
        raise SystemExit(f"no run_gxtb.yaml in {args.pack_dir}")
    spec = load_nucleation_spec(str(pack_yaml))

    moves = tuple(m.strip() for m in str(args.moves).split(",") if m.strip())
    for move in moves:
        if move not in MOVES:
            raise SystemExit(f"unknown move {move!r}; expected from {MOVES}")

    seeds = collect_seeds(
        args.run,
        k=args.k,
        p=args.p,
        n_eligible=args.n_eligible,
        n_offpath=args.n_offpath,
        spec=spec,
    )
    if not seeds:
        raise SystemExit(f"no seeds found for k={args.k} p={args.p} in {args.run}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "seeds.json").write_text(
        json.dumps(
            [
                {
                    "id": s["id"],
                    "origin": s["origin"],
                    "energy_eV": s["energy_eV"],
                    "n_atoms": len(s["symbols"]),
                }
                for s in seeds
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[bh] {len(seeds)} seeds, {args.steps} steps each, {args.workers} worker(s)")
    print(f"[bh] live: tail -f {args.output}/walkers/*.log")
    for seed in seeds:
        print(f"[bh]   {seed['origin']:8s} {seed['id']:34s} E={seed['energy_eV']:.6f}")

    jobs = [
        {
            "pack_yaml": str(pack_yaml),
            "seed": {**seed, "positions": [list(map(float, x)) for x in seed["positions"]]},
            "k": args.k,
            "p": args.p,
            "steps": args.steps,
            "temperature_eV": args.temperature,
            "moves": list(moves),
            "amplitude_A": args.amplitude,
            "rng_seed": args.rng_seed + index,
            "out_dir": str(args.output),
        }
        for index, seed in enumerate(seeds)
    ]

    started = time.perf_counter()
    walkers: List[Dict[str, Any]] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_walk, job): job["seed"]["id"] for job in jobs}
            for future in as_completed(futures):
                walker = future.result()
                walkers.append(walker)
                print(
                    f"[bh] done {walker['seed_id']}: {walker['n_relaxations']} opts, "
                    f"{len(walker['minima'])} minima, gain {walker['gain_eV']:+.3f} eV"
                )
    else:
        for job in jobs:
            walker = _walk(job)
            walkers.append(walker)
            print(
                f"[bh] done {walker['seed_id']}: {walker['n_relaxations']} opts, "
                f"{len(walker['minima'])} minima, gain {walker['gain_eV']:+.3f} eV"
            )

    with (args.output / "bh_minima.jsonl").open("w", encoding="utf-8") as handle:
        for walker in walkers:
            for record in walker["minima"]:
                handle.write(json.dumps({**record, "origin": walker["origin"]}) + "\n")
    (args.output / "bh_walkers.json").write_text(
        json.dumps(
            [{key: value for key, value in w.items() if key != "minima"} for w in walkers],
            indent=2,
        ),
        encoding="utf-8",
    )

    path = write_summary(
        args.output,
        k=args.k,
        p=args.p,
        run_dir=args.run,
        walkers=walkers,
        spec=spec,
        steps=args.steps,
    )
    print(f"[bh] wall {(time.perf_counter()-started)/60:.1f} min -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
