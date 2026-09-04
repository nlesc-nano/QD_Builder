#!/usr/bin/env python3
"""Report whether a growth run is still on the road to a target core.

Growth adds one anion per step and never removes one -- ``shed`` is Z-type
(CdCl2) only, so the anion sublattice is strictly monotone.  A run can
therefore reach a target core at size K only if, at every k < K, it built an
anion backbone that is a connected sub-skeleton of that target.  Once the
lineage is lost it cannot be recovered at any p, because p decorates a core
and does not move anion sites.

This reads the run's ``zb_occupations.jsonl`` and reports, per k, how many of
the target's connected sub-skeletons the run actually built.  The first k with
zero is where the road was lost.

Usage::

    PYTHONPATH=src python scripts/check_wulff_ancestry.py RUN_DIR
    PYTHONPATH=src python scripts/check_wulff_ancestry.py RUN_DIR \\
        --reference geometry_packs/cdse_cdcl2_zb/k13_wulff_core.yaml \\
        --anion Se --tolerance 0.2
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from builder.nucleation.molecular_zb_growth import _occupation_shape_certificate

#: fcc second-neighbour separation, a/sqrt(2); two anions sharing a cation.
ANION_ADJACENCY_A = 4.5


def target_sub_skeletons(
    points: np.ndarray, anion: str, tolerance: float, k_max: int
) -> dict[int, set[str]]:
    """Every connected sub-skeleton of the target, by size, up to k_max."""

    n = len(points)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    adj = (d > 0.1) & (d <= ANION_ADJACENCY_A)
    out: dict[int, set[str]] = collections.defaultdict(set)
    for size in range(1, min(n, k_max) + 1):
        for sub in itertools.combinations(range(n), size):
            if size > 1:
                graph = {i: set() for i in sub}
                for a, b in itertools.combinations(sub, 2):
                    if adj[a, b]:
                        graph[a].add(b)
                        graph[b].add(a)
                seen = {sub[0]}
                stack = [sub[0]]
                while stack:
                    for nxt in graph[stack.pop()]:
                        if nxt not in seen:
                            seen.add(nxt)
                            stack.append(nxt)
                if len(seen) != size:
                    continue
            out[size].add(
                _occupation_shape_certificate(
                    [anion] * size, points[list(sub)], tolerance
                )
            )
    return out


def built_skeletons(
    run_dir: Path, anion: str, tolerance: float
) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
    """Anion backbones the run actually produced: all built vs preserved."""

    built: dict[int, set[str]] = collections.defaultdict(set)
    preserved: dict[int, set[str]] = collections.defaultdict(set)
    path = run_dir / "zb_occupations.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            occupation = record.get("occupation") or {}
        except json.JSONDecodeError:
            continue
        k = occupation.get("k")
        if not k:
            continue
        symbols = np.asarray(occupation["symbols"])
        coords = np.asarray(occupation["lattice_coordinates"], dtype=float)
        pts = coords[symbols == anion]
        if not len(pts):
            continue
        cert = _occupation_shape_certificate([anion] * len(pts), pts, tolerance)
        k_int = int(k)
        built[k_int].add(cert)
        is_preserved = bool(
            record.get("propagation_eligible", False)
            or str(record.get("topology_status", "")).lower() == "preserved"
        )
        if is_preserved:
            preserved[k_int].add(cert)
    return built, preserved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument(
        "--reference",
        type=Path,
        default=Path("geometry_packs/cdse_cdcl2_zb/k13_wulff_core.yaml"),
    )
    ap.add_argument("--anion", default="Se")
    ap.add_argument("--tolerance", type=float, default=0.2)
    args = ap.parse_args()

    raw = yaml.safe_load(args.reference.read_text(encoding="utf-8")) or {}
    symbols = np.asarray(raw["symbols"])
    coords = np.asarray(raw["coordinates"], dtype=float)
    target = coords[symbols == args.anion]
    print(
        f"target {args.reference.name}: {len(target)} {args.anion} "
        f"({(symbols != args.anion).sum()} cations)"
    )

    built, preserved = built_skeletons(args.run_dir, args.anion, args.tolerance)
    k_max = max(built) if built else len(target)
    anc = target_sub_skeletons(target, args.anion, args.tolerance, min(k_max, len(target)))

    print(f"\nrun {args.run_dir.name}")
    print("   k | target sub-skeletons | built by run | on-road (built) | on-road (preserved)")
    lost_at = None
    for k in sorted(anc):
        hit_built = anc[k] & built.get(k, set())
        hit_pres = anc[k] & preserved.get(k, set())
        flag = ""
        if not hit_pres and lost_at is None and k > 1 and (anc[k - 1] & preserved.get(k - 1, set())):
            lost_at = k
            flag = "  <-- preserved lineage lost here"
        print(
            f"  {k:2} | {len(anc[k]):20} | {len(built.get(k, set())):12} |"
            f" {len(hit_built):15} | {len(hit_pres):19}{flag}"
        )
    if lost_at is None and anc:
        alive = max(k for k in sorted(anc) if anc[k] & preserved.get(k, set()))
        print(f"\npreserved lineage still alive at k={alive}")
    else:
        print(f"\npreserved lineage lost at k={lost_at}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
