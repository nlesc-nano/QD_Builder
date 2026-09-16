#!/usr/bin/env python3
"""Mine an ``embed_*.yaml`` metric table from MACE / PBE relaxed endpoints.

The embedding tables (``geometry_packs/cdse_cdcl2_zb/embed.yaml``) hold the
bond lengths and angles a constructed graph is morphed onto before the
first relaxation.  The shipped table is a g-xTB median; a MACE-MH-1 /
PBE optimizer wants its own (Cd-Se bonded median 2.655 vs 2.578 Å, Cd-Cl
2.543 vs 2.495 Å, and a 2.9-3.3 Å contact tail g-xTB lacks).  This tool
rebuilds the table in the same schema from relaxed endpoints:

* ``--reopt results.jsonl``  -- ``tools/mace_reopt_subset.py`` journals
  (MACE endpoints of the g-xTB archive; ``symbols`` / ``positions``);
* ``--dft results.jsonl``    -- ``tools/compare_mace_dft_full.py`` journals
  (``from_dft.positions`` = MACE-relaxed PBE endpoints, and
  ``dft_positions`` = the PBE endpoints themselves with ``--dft-endpoints``).

Bond graph and coordination numbers use the two-tier *strong* cutoffs
(``mace_descriptors.MACE_CUTOFFS["strong"]``), which is what constructed
graphs correspond to.  Cl roles follow ``molecular._neighbor_geometry_role``
(``Cl_t`` terminal, ``Cl_b2s`` μ2 whose hosts share an anion, ``Cl_b2n``
μ2 otherwise, ``Cl_b3`` μ3).

Bonds, angles, ``angle_sum_cn3``, ``nonbonded`` and ``nonbonded_1_4`` are
mined; ``dihedrals`` and ``rings`` (geometric ZB-chair relations) and the
``reconstruction`` solver knobs are carried over from the reference table
and marked as such.  Structures with any artifact contact (Se-Se, Cd-Cd,
Cl-Se, Cl-Cl < 2.80 Å) or p = 0 are excluded.

Example::

    /Users/ivaninfante/miniforge3/envs/orchestr_ai-mace/bin/python \\
        tools/mine_mace_embed_table.py \\
        --reopt runs/mace_import/reopt_full_k10/results.jsonl \\
        --dft runs/mace_import/dft_compare/results.jsonl \\
        --reference geometry_packs/cdse_cdcl2_zb/embed.yaml \\
        --out geometry_packs/cdse_cdcl2_zb/embed_mace.yaml
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_FLOOR = 2.80
MIN_COUNT = 20  # minimum observations for a table entry


def _load_descriptors():
    path = ROOT / "src" / "builder" / "nucleation" / "mace_descriptors.py"
    spec = importlib.util.spec_from_file_location("mace_descriptors", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


MD = _load_descriptors()
STRONG = MD.MACE_CUTOFFS["strong"]


# --------------------------------------------------------------------------
# structures
# --------------------------------------------------------------------------
def iter_structures(args) -> Iterable[Tuple[str, List[str], np.ndarray, int, int]]:
    for path in args.reopt:
        with Path(path).open() as fh:
            for line in fh:
                r = json.loads(line)
                if not r.get("ok") or r.get("status") != "converged" or r.get("artifact_flags"):
                    continue
                if int(r["p"]) == 0:
                    continue
                yield f"reopt:{r['minimum_id']}", list(r["symbols"]), np.asarray(r["positions"], float), int(r["k"]), int(r["p"])
    for path in args.dft:
        with Path(path).open() as fh:
            for line in fh:
                r = json.loads(line)
                if int(r["p"]) == 0:
                    continue
                fd = r.get("from_dft") or {}
                if fd.get("ok") and fd.get("status") == "converged" and not fd.get("artifact_flags"):
                    yield f"mace_from_dft:{r['structure_id']}", list(r["symbols"]), np.asarray(fd["positions"], float), int(r["k"]), int(r["p"])
                if args.dft_endpoints and r.get("dft_positions"):
                    yield f"dft:{r['structure_id']}", list(r["symbols"]), np.asarray(r["dft_positions"], float), int(r["k"]), int(r["p"])


def has_artifact(symbols: Sequence[str], dist: np.ndarray) -> bool:
    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):
            pair = tuple(sorted((symbols[i], symbols[j])))
            if pair in {("Cd", "Cd"), ("Se", "Se"), ("Cl", "Se"), ("Cl", "Cl")} and dist[i, j] < ARTIFACT_FLOOR:
                return True
    return False


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def angle_deg(a: np.ndarray, c: np.ndarray, b: np.ndarray) -> float:
    v1, v2 = a - c, b - c
    cosang = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosang))))


def cl_role(symbols: Sequence[str], nbrs: Mapping[int, List[int]], center: int, neighbor: int) -> str:
    sym = symbols[neighbor]
    if sym != "Cl":
        return sym
    hosts = [h for h in nbrs[neighbor] if symbols[h] == "Cd"]
    degree = len(hosts)
    if degree <= 1:
        return "Cl_t"
    if degree == 2:
        others = [h for h in hosts if h != center]
        shared = any(
            any(symbols[c] != "Cl" and c != neighbor for c in set(nbrs[center]) & set(nbrs[o]))
            for o in others
        )
        return "Cl_b2s" if shared else "Cl_b2n"
    return f"Cl_b{degree}"


def robust(values: Sequence[float], lo: float, hi: float, nd: int = 2) -> Tuple[float, float]:
    arr = np.asarray(values, float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826
    return round(med, nd), round(min(hi, max(lo, mad)), nd)


def two_modes(values: Sequence[float]) -> Optional[List[Dict[str, float]]]:
    """Simple 2-means split; returns two modes when the spread is genuinely bimodal."""
    arr = np.sort(np.asarray(values, float))
    if len(arr) < 2 * MIN_COUNT or float(np.std(arr)) < 18.0:
        return None
    c1, c2 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    for _ in range(50):
        left = arr[np.abs(arr - c1) <= np.abs(arr - c2)]
        right = arr[np.abs(arr - c1) > np.abs(arr - c2)]
        if len(left) < MIN_COUNT or len(right) < MIN_COUNT:
            return None
        n1, n2 = float(np.mean(left)), float(np.mean(right))
        if abs(n1 - c1) < 1e-6 and abs(n2 - c2) < 1e-6:
            break
        c1, c2 = n1, n2
    if abs(c2 - c1) < 25.0:
        return None
    return [
        {"deg": round(float(np.median(left)), 0), "tol_deg": round(max(8.0, min(20.0, float(np.std(left)) * 1.5)), 0)},
        {"deg": round(float(np.median(right)), 0), "tol_deg": round(max(8.0, min(20.0, float(np.std(right)) * 1.5)), 0)},
    ]


# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reopt", nargs="*", default=[], help="mace_reopt_subset results.jsonl")
    ap.add_argument("--dft", nargs="*", default=[], help="compare_mace_dft_full results.jsonl")
    ap.add_argument("--dft-endpoints", action="store_true", help="also include the PBE endpoints themselves")
    ap.add_argument("--reference", type=Path, default=ROOT / "geometry_packs/cdse_cdcl2_zb/embed.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k-max", type=int, default=0, help="ignore structures above this k (0 = all)")
    args = ap.parse_args(argv)

    ref = yaml.safe_load(args.reference.read_text())

    bonds: Dict[Tuple[str, int, int], List[float]] = collections.defaultdict(list)
    ang_se: Dict[int, List[float]] = collections.defaultdict(list)
    ang_cl: Dict[int, List[float]] = collections.defaultdict(list)
    ang_cd_pair: Dict[Tuple[int, str], List[float]] = collections.defaultdict(list)  # (cn, "A-B")
    ang_cd_all: Dict[int, List[float]] = collections.defaultdict(list)
    ang_cd3_sig: Dict[Tuple[str, str], List[float]] = collections.defaultdict(list)  # (signature, pair)
    ang_sum3: Dict[str, List[float]] = collections.defaultdict(list)
    nonbonded: Dict[str, List[float]] = collections.defaultdict(list)
    geminal: Dict[str, List[float]] = collections.defaultdict(list)
    n_struct = 0
    n_skip = 0
    sources = collections.Counter()
    kp = collections.Counter()

    for sid, symbols, pos, k, p in iter_structures(args):
        if args.k_max and k > args.k_max:
            continue
        dist = MD.distance_matrix(pos)
        if has_artifact(symbols, dist):
            n_skip += 1
            continue
        edges = MD.bonded_edges(symbols, dist, STRONG)
        nbrs: Dict[int, List[int]] = collections.defaultdict(list)
        for i, j in edges:
            nbrs[i].append(j)
            nbrs[j].append(i)
        bonded = {(min(i, j), max(i, j)) for i, j in edges}
        n_struct += 1
        sources[sid.split(":")[0]] += 1
        kp[(k, p)] += 1
        n = len(symbols)
        deg = {i: len(nbrs[i]) for i in range(n)}
        # bonds
        for i, j in edges:
            cd, other = (i, j) if symbols[i] == "Cd" else (j, i)
            if symbols[other] == "Se":
                bonds[("CdSe", deg[cd], deg[other])].append(dist[cd, other])
            else:
                pair = "CdCl_terminal" if deg[other] <= 1 else "CdCl_bridge"
                bonds[(pair, deg[cd], deg[other])].append(dist[cd, other])
        # angles
        for c in range(n):
            ns = nbrs[c]
            if len(ns) < 2:
                continue
            angs = [angle_deg(pos[a], pos[c], pos[b]) for a, b in itertools.combinations(ns, 2)]
            if symbols[c] == "Se":
                ang_se[len(ns)].extend(angs)
            elif symbols[c] == "Cl":
                ang_cl[len(ns)].extend(angs)
            else:
                roles = {a: cl_role(symbols, nbrs, c, a) for a in ns}
                cn = len(ns)
                ang_cd_all[cn].extend(angs)
                sig = "+".join(sorted(roles.values()))
                for (a, b), ang in zip(itertools.combinations(ns, 2), angs):
                    pair = "-".join(sorted((roles[a], roles[b])))
                    ang_cd_pair[(cn, pair)].append(ang)
                    if cn == 3:
                        ang_cd3_sig[(sig, pair)].append(ang)
            if len(ns) == 3:
                ang_sum3[symbols[c]].append(sum(angs))
        # non-bonded floors and geminal (1-3) distances
        for i in range(n):
            for j in range(i + 1, n):
                key = "-".join(sorted((symbols[i], symbols[j])))
                if (i, j) in bonded:
                    continue
                if set(nbrs[i]) & set(nbrs[j]):
                    geminal[key].append(dist[i, j])
                elif dist[i, j] < 6.0:
                    nonbonded[key].append(dist[i, j])

    if n_struct == 0:
        raise SystemExit("no usable structures")

    # ---- assemble ----
    out: Dict[str, Any] = {}
    bond_rows = []
    for (pair, cn_cd, cn_other), vals in sorted(bonds.items()):
        if len(vals) < MIN_COUNT:
            continue
        r, tol = robust(vals, 0.04, 0.14)
        bond_rows.append({"pair": pair, "cn_cd": cn_cd, "cn_other": cn_other, "r_A": r, "tol_A": tol, "n": len(vals)})
    out["bonds"] = bond_rows

    def angle_entry(vals, *, hard=False):
        d, tol = robust(vals, 8.0, 16.0, 0)
        entry = {"deg": d, "tol_deg": tol}
        if hard:
            entry["hard_min_deg"] = round(float(np.percentile(vals, 2)), 0)
            entry["hard_max_deg"] = round(float(np.percentile(vals, 98)), 0)
        entry["n"] = len(vals)
        return entry

    angles: Dict[str, Any] = {"Se": {}, "Cl": {}, "Cd": {}}
    for cn, vals in sorted(ang_se.items()):
        if len(vals) >= MIN_COUNT:
            block = {"default": angle_entry(vals, hard=cn >= 4)}
            modes = two_modes(vals)
            if modes:
                block["modes"] = modes
            angles["Se"][f"cn{cn}"] = block
    for cn, vals in sorted(ang_cl.items()):
        if len(vals) >= MIN_COUNT:
            block = {"default": angle_entry(vals)}
            modes = two_modes(vals)
            if modes:
                block["modes"] = modes
            angles["Cl"][f"cn{cn}"] = block
    for cn, vals in sorted(ang_cd_all.items()):
        if len(vals) < MIN_COUNT:
            continue
        block: Dict[str, Any] = {"default": angle_entry(vals, hard=cn == 4)}
        by_pair = {
            pair: angle_entry(pv)
            for (c2, pair), pv in sorted(ang_cd_pair.items())
            if c2 == cn and len(pv) >= MIN_COUNT
        }
        if cn == 3:
            sigs: Dict[str, Dict[str, Any]] = {}
            for (sig, pair), pv in sorted(ang_cd3_sig.items()):
                if len(pv) >= MIN_COUNT:
                    sigs.setdefault(sig, {})[pair] = angle_entry(pv)
            if sigs:
                block["by_role_signature"] = sigs
        elif by_pair:
            block["by_pair"] = by_pair
        angles["Cd"][f"cn{cn}"] = block
    out["angles"] = angles
    out["angle_sum_cn3"] = {
        sym: {"target_deg": round(float(np.median(v)), 0), "tol_deg": round(max(8.0, min(20.0, float(np.std(v)))), 0), "n": len(v)}
        for sym, v in sorted(ang_sum3.items())
        if len(v) >= MIN_COUNT
    }
    out["dihedrals"] = ref.get("dihedrals")
    out["nonbonded_1_4"] = {
        key: {"min_A": round(float(np.percentile(v, 1)), 2), "tol_A": 0.35, "n": len(v)}
        for key, v in sorted(geminal.items())
        if key in {"Cd-Cl", "Cd-Se"} and len(v) >= MIN_COUNT
    }
    nb = {}
    for key, v in sorted(nonbonded.items()):
        if len(v) >= MIN_COUNT:
            floor = float(np.percentile(v, 1))
            pair = tuple(sorted(key.split("-")))
            if pair in {("Cd", "Cd"), ("Se", "Se"), ("Cl", "Se"), ("Cl", "Cl")}:
                floor = max(floor, ARTIFACT_FLOOR)
            nb[key] = {"min_A": round(floor, 2), "tol_A": 0.1, "n": len(v)}
    out["nonbonded"] = nb
    out["rings"] = ref.get("rings")
    out["reconstruction"] = ref.get("reconstruction")
    out["cl_seed"] = {"from_table": True}

    header = (
        "# Metric tables for zb occupation growth -- MACE-MH-1 (omat_pbe) / PBE\n"
        "# endpoints.  Same schema and role as embed.yaml (g-xTB medians):\n"
        "# after Cl is seeded on the CIF core, a short least_squares morphs\n"
        "# core+Cl onto these lengths/angles; nothing here decides which graphs\n"
        "# are built.\n#\n"
        f"# Mined {time.strftime('%Y-%m-%d')} by tools/mine_mace_embed_table.py from\n"
        f"# {n_struct} relaxed structures ({dict(sources)}), {n_skip} excluded for\n"
        f"# artifact contacts; {sum(len(v) for v in bonds.values())} bonds, "
        f"{sum(len(v) for v in ang_se.values()) + sum(len(v) for v in ang_cl.values()) + sum(len(v) for v in ang_cd_all.values())} angles.\n"
        f"# k range {min(k for k, _ in kp)}-{max(k for k, _ in kp)}; bond graph = two-tier strong cutoffs "
        f"(Cd-Se {STRONG[('Cd','Se')]}, Cd-Cl {STRONG[('Cd','Cl')]}).\n"
        "# Observation counts per entry are in the .provenance.json sidecar.  dihedrals / rings /\n"
        "# reconstruction are carried over from embed.yaml (geometric relations,\n"
        "# not re-mined).  cl_seed.from_table makes the geometric Cl seed use the\n"
        "# CdCl_terminal / CdCl_bridge radii below.\n"
        "# Do not hand-edit numbers; re-run the miner.\n\n"
    )
    # The pack loader rejects non-executable keys, so observation counts go
    # to a sidecar instead of the table itself.
    counts: Dict[str, int] = {}

    def strip_counts(node: Any, path: str = "") -> Any:
        if isinstance(node, dict):
            if "n" in node and isinstance(node.get("n"), int):
                counts[path or "root"] = int(node.pop("n"))
            return {k: strip_counts(v, f"{path}.{k}" if path else str(k)) for k, v in node.items()}
        if isinstance(node, list):
            return [strip_counts(v, f"{path}[{i}]") for i, v in enumerate(node)]
        return node

    out = strip_counts(out)
    args.out.write_text(header + yaml.safe_dump(out, sort_keys=False, default_flow_style=None, width=100))
    sidecar = args.out.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps({
        "mined": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inputs": {"reopt": [str(x) for x in args.reopt], "dft": [str(x) for x in args.dft], "dft_endpoints": bool(args.dft_endpoints)},
        "n_structures": n_struct, "n_excluded_artifact": n_skip, "sources": dict(sources),
        "kp_bins": {f"{k},{p}": v for (k, p), v in sorted(kp.items())},
        "cutoffs_strong": {"-".join(k): v for k, v in STRONG.items()},
        "observation_counts": counts,
    }, indent=1))
    print(f"wrote {args.out}: {n_struct} structures, {len(bond_rows)} bond rows, "
          f"Se cn {sorted(angles['Se'])}, Cd cn {sorted(angles['Cd'])}, Cl cn {sorted(angles['Cl'])}")
    print(f"k,p coverage: {len(kp)} bins, k={min(k for k, _ in kp)}..{max(k for k, _ in kp)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
