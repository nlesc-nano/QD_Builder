#!/usr/bin/env python3
"""Re-optimize a tier-stratified subset of g-xTB minima with MACE-MH-1.

Handoff step 2: take existing g-xTB endpoints, relax them with MACE through
the persistent MPS worker, and record -- per structure -- what MACE did to
the scaffold, the chloride bridging, and the coordination environment.  No
cross-protocol energy comparison is made; ``gxtb_energy_eV`` and
``mace_energy_eV`` live in separate namespaces in the output.

Selection per ``(k,p)`` bin (see docs/nucleation/gxtb_trust_tiers.md):

* ``--n-b`` Tier AB minima: lowest g-xTB energy, at most one per
  construction core hash;
* ``--n-c`` Tier AC minima with distinct *final* core hashes (these test
  whether MACE agrees with g-xTB's scaffold rearrangements);
* ``--n-d`` Tier AD ``scaffold_lost`` minima as diagnostics (flagged
  ``namespace: diagnostic``).

Results are appended to ``<out>/results.jsonl`` one line per structure as
soon as the worker returns, so a killed run resumes by skipping ids already
present.  ``<out>/manifest.json`` records the selection and the preflight
fingerprint.

Example::

    /Users/ivaninfante/miniforge3/envs/orchestr_ai-mace/bin/python \\
        tools/mace_reopt_subset.py \\
        --minima .../nucleation_phase_c_k10_canonical_v1/minima.json \\
        --tiers runs/mace_import/k10_tiers.jsonl \\
        --preflight runs/mace_import/preflight.json \\
        --k 8 9 --p 2 4 6 8 --n-b 2 --n-c 2 --n-d 1 \\
        --out runs/mace_import/reopt_pilot
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

MACE_PYTHON = "/Users/ivaninfante/miniforge3/envs/orchestr_ai-mace/bin/python"
MODEL = "/Users/ivaninfante/.cache/mace/macemh1model"
WORKER = Path(__file__).resolve().with_name("mace_worker.py")

# Bond cutoffs from geometry_packs/cdse_cdcl2_zb/graph_rules.yaml
BOND_MAX = {("Cd", "Se"): 3.25, ("Cd", "Cl"): 2.90}
# Artifact floors from run_mace_mh1.yaml
ARTIFACT_FLOORS = {("Cd", "Cd"): 2.80, ("Se", "Se"): 2.80, ("Cl", "Se"): 2.80, ("Cl", "Cl"): 2.80}


# --------------------------------------------------------------------------
# Geometry descriptors
# --------------------------------------------------------------------------
def _pair(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def bonded_edges(symbols: Sequence[str], pos: np.ndarray) -> List[Tuple[int, int]]:
    edges = []
    for i, j in itertools.combinations(range(len(symbols)), 2):
        cut = BOND_MAX.get(_pair(symbols[i], symbols[j]))
        if cut is not None and float(np.linalg.norm(pos[i] - pos[j])) <= cut:
            edges.append((i, j))
    return edges


def min_distances(symbols: Sequence[str], pos: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, j in itertools.combinations(range(len(symbols)), 2):
        key = "-".join(_pair(symbols[i], symbols[j]))
        r = float(np.linalg.norm(pos[i] - pos[j]))
        out[key] = min(out.get(key, math.inf), r)
    return out


def n_components(n: int, edges: Iterable[Tuple[int, int]], keep: Optional[set] = None) -> int:
    nodes = set(range(n)) if keep is None else set(keep)
    adj = collections.defaultdict(set)
    for i, j in edges:
        if i in nodes and j in nodes:
            adj[i].add(j)
            adj[j].add(i)
    seen, comps = set(), 0
    for s in nodes:
        if s in seen:
            continue
        comps += 1
        stack = [s]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(adj[u] - seen)
    return comps


def core_hash(symbols: Sequence[str], edges: Iterable[Tuple[int, int]]) -> str:
    """Order-independent hash of the Cd–Se bonding graph (labelled by index).

    Not the archive's canonical ``core_hash`` (that relabels by graph
    canonicalization), but adequate for *before/after* comparison of the
    same atom ordering, which is all this script needs.
    """
    core = sorted((i, j) for i, j in edges if {symbols[i], symbols[j]} == {"Cd", "Se"})
    return hashlib.md5(json.dumps(core).encode()).hexdigest()


def describe(symbols: Sequence[str], pos: np.ndarray) -> Dict[str, Any]:
    edges = bonded_edges(symbols, pos)
    n = len(symbols)
    cn = collections.Counter()
    nbrs: Dict[int, List[int]] = collections.defaultdict(list)
    for i, j in edges:
        nbrs[i].append(j)
        nbrs[j].append(i)
    se_cn = collections.Counter(len(nbrs[i]) for i in range(n) if symbols[i] == "Se")
    cd_cn = collections.Counter(len(nbrs[i]) for i in range(n) if symbols[i] == "Cd")
    cl_cn = collections.Counter(len(nbrs[i]) for i in range(n) if symbols[i] == "Cl")
    cd_idx = {i for i in range(n) if symbols[i] == "Cd"}
    cd_se_edges = [(i, j) for i, j in edges if {symbols[i], symbols[j]} == {"Cd", "Se"}]
    core_nodes = {i for i in range(n) if symbols[i] in ("Cd", "Se")}
    # bridge load per Cd: number of mu2/mu3 Cl bonded to it
    bridge_load = []
    for i in sorted(cd_idx):
        bridge_load.append(sum(1 for j in nbrs[i] if symbols[j] == "Cl" and len(nbrs[j]) >= 2))
    core_edges_set = {tuple(sorted(e)) for e in cd_se_edges}
    return {
        "edges": edges,
        "core_hash": core_hash(symbols, edges),
        "n_components": n_components(n, edges),
        "core_components": n_components(n, cd_se_edges, keep=core_nodes),
        "mu1": cl_cn.get(1, 0),
        "mu2": cl_cn.get(2, 0),
        "mu3": sum(v for k, v in cl_cn.items() if k >= 3),
        "isolated_cl": cl_cn.get(0, 0),
        "se_cn": {str(k): v for k, v in sorted(se_cn.items())},
        "cd_cn": {str(k): v for k, v in sorted(cd_cn.items())},
        "cl_cn": {str(k): v for k, v in sorted(cl_cn.items())},
        "se_cn4_fraction": se_cn.get(4, 0) / max(1, sum(se_cn.values())),
        "cd_cn4_fraction": cd_cn.get(4, 0) / max(1, sum(cd_cn.values())),
        "bridge_load_mean": float(np.mean(bridge_load)) if bridge_load else 0.0,
        "bridge_load_max": int(max(bridge_load)) if bridge_load else 0,
        "bridge_load_var": float(np.var(bridge_load)) if bridge_load else 0.0,
        "min_distances": {k: round(v, 3) for k, v in min_distances(symbols, pos).items()},
        "_core_edges": core_edges_set,
    }


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    dmat = np.diag([1, 1, d])
    r = vt.T @ dmat @ u.T
    return float(np.sqrt(((a @ r.T - b) ** 2).sum(axis=1).mean()))


def artifact_flags(mind: Mapping[str, float]) -> List[str]:
    flags = []
    for (x, y), floor in ARTIFACT_FLOORS.items():
        key = "-".join(_pair(x, y))
        if mind.get(key, math.inf) < floor:
            flags.append(f"{key}<{floor}")
    return flags


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def load_tiers(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            out[row["minimum_id"]] = row
    return out


def select(tiers: Mapping[str, Dict[str, Any]], ks: Sequence[int], ps: Sequence[int], nb: int, nc: int, nd: int):
    by_bin: Dict[Tuple[int, int], List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in tiers.values():
        if row["k"] in ks and (not ps or row["p"] in ps):
            by_bin[(row["k"], row["p"])].append(row)
    chosen: List[Dict[str, Any]] = []
    for key in sorted(by_bin):
        rows = by_bin[key]
        b = sorted((r for r in rows if r["tier"] == "AB"), key=lambda r: r["gxtb_energy_eV"])
        seen = set()
        for r in b:
            if len([c for c in chosen if c["_bin"] == key and c["_stratum"] == "B"]) >= nb:
                break
            if nb < 10**9 and r["construction_core_hash"] in seen:
                continue
            seen.add(r["construction_core_hash"])
            chosen.append({**r, "_bin": key, "_stratum": "B", "namespace": "population"})
        c = sorted((r for r in rows if r["tier"] == "AC"), key=lambda r: r["gxtb_energy_eV"])
        seen = set()
        for r in c:
            if len([x for x in chosen if x["_bin"] == key and x["_stratum"] == "C"]) >= nc:
                break
            if nc < 10**9 and r["final_core_hash"] in seen:
                continue
            seen.add(r["final_core_hash"])
            chosen.append({**r, "_bin": key, "_stratum": "C", "namespace": "population"})
        d = sorted(
            (r for r in rows if r["tier"] == "AD" and any("scaffold_lost" in x for x in r["reasons"])),
            key=lambda r: r["gxtb_energy_eV"],
        )
        for r in d[:nd]:
            chosen.append({**r, "_bin": key, "_stratum": "D", "namespace": "diagnostic"})
    return chosen


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
class Worker:
    def __init__(self, python: str, model: str, head: str, device: str, log: Path) -> None:
        self.log = log.open("a")
        self.proc = subprocess.Popen(
            [python, str(WORKER), "--model", model, "--head", head, "--device", device, "--dtype", "float32"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )

    def relax(self, structure: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps({"command": "relax", "settings": settings, "structures": [structure]}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"MACE worker exited (rc={self.proc.poll()}); see {self.log.name}")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "worker error"))
        return reply["results"][0]

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.write(json.dumps({"command": "close"}) + "\n")
                self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            self.proc.kill()
        self.log.close()


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as fh:
        fh.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    with path.open("a") as out, tmp.open() as src:
        out.write(src.read())
        out.flush()
        os.fsync(out.fileno())
    tmp.unlink()


# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minima", type=Path, required=True)
    ap.add_argument("--tiers", type=Path, required=True)
    ap.add_argument("--preflight", type=Path, required=True, help="passing preflight JSON (fingerprint is embedded)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, nargs="+", required=True)
    ap.add_argument("--p", type=int, nargs="*", default=[])
    ap.add_argument("--n-b", type=int, default=2)
    ap.add_argument("--n-c", type=int, default=2)
    ap.add_argument("--n-d", type=int, default=1)
    ap.add_argument(
        "--all",
        action="store_true",
        help="ignore the per-bin quotas and take every AB/AC minimum plus every AD scaffold_lost diagnostic",
    )
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--python", default=MACE_PYTHON)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--head", default="omat_pbe")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dry-run", action="store_true", help="select and write manifest only")
    args = ap.parse_args(argv)

    pre = json.loads(args.preflight.read_text())
    if not pre.get("ok") or pre.get("fingerprint", {}).get("device") != args.device:
        raise SystemExit("preflight did not pass for the requested device; refusing to run")
    fingerprint = pre["fingerprint"]

    tiers = load_tiers(args.tiers)
    big = 10**9
    chosen = (
        select(tiers, args.k, args.p, big, big, big)
        if args.all
        else select(tiers, args.k, args.p, args.n_b, args.n_c, args.n_d)
    )
    if not chosen:
        raise SystemExit("selection is empty")
    minima = json.loads(args.minima.read_text())
    minima = minima.get("experimental", minima)

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.jsonl"
    done = set()
    if results_path.exists():
        with results_path.open() as fh:
            for line in fh:
                done.add(json.loads(line)["minimum_id"])

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "minima": str(args.minima),
        "tiers": str(args.tiers),
        "fingerprint": fingerprint,
        "settings": {"fmax": args.fmax, "max_steps": args.max_steps, "optimizer": "BFGS", "maxstep": 0.20},
        "selection": {"k": args.k, "p": args.p, "n_b": args.n_b, "n_c": args.n_c, "n_d": args.n_d},
        "n_selected": len(chosen),
        "strata": dict(collections.Counter(c["_stratum"] for c in chosen)),
        "ids": [c["minimum_id"] for c in chosen],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[reopt] selected {len(chosen)} ({manifest['strata']}), {len(done)} already done", file=sys.stderr)
    if args.dry_run:
        return 0

    settings = {
        "device": args.device,
        "dtype": "float32",
        "optimizer": "BFGS",
        "fmax": args.fmax,
        "max_steps": args.max_steps,
        "maxstep": 0.20,
        "accept_maxcycle": True,
    }
    worker = Worker(args.python, args.model, args.head, args.device, args.out / "worker.log")
    try:
        for n, c in enumerate(chosen, 1):
            mid = c["minimum_id"]
            if mid in done:
                continue
            m = minima[mid]
            symbols = list(m["symbols"])
            start = np.asarray(m["positions"], dtype=float)
            before = describe(symbols, start)
            t0 = time.time()
            res = worker.relax({"id": mid, "symbols": symbols, "positions": start.tolist()}, settings)
            row: Dict[str, Any] = {
                "minimum_id": mid,
                "structure_id": c.get("structure_id"),
                "k": c["k"],
                "p": c["p"],
                "stratum": c["_stratum"],
                "tier": c["tier"],
                "namespace": c["namespace"],
                "gxtb_energy_eV": c["gxtb_energy_eV"],
                "gxtb_reasons": c["reasons"],
                "mace_protocol": fingerprint,
                "ok": bool(res.get("ok")),
                "status": res.get("status"),
                "error": res.get("error") or "",
                "steps": res.get("steps"),
                "wall_s": round(res.get("wall_seconds", time.time() - t0), 2),
            }
            if res.get("ok") and res.get("positions"):
                end = np.asarray(res["positions"], dtype=float)
                after = describe(symbols, end)
                retained = len(before["_core_edges"] & after["_core_edges"])
                row.update(
                    {
                        "mace_energy_eV": res["energy_eV"],
                        "max_force": res.get("max_force"),
                        "rmsd_gxtb_to_mace_A": round(kabsch_rmsd(start, end), 4),
                        "core_retained": before["core_hash"] == after["core_hash"],
                        "core_edges_before": len(before["_core_edges"]),
                        "core_edges_after": len(after["_core_edges"]),
                        "core_edges_kept": retained,
                        "artifact_flags": artifact_flags(after["min_distances"]),
                        "before": {k: v for k, v in before.items() if not k.startswith("_") and k != "edges"},
                        "after": {k: v for k, v in after.items() if not k.startswith("_") and k != "edges"},
                        "positions": end.tolist(),
                        "symbols": symbols,
                    }
                )
            append_jsonl(results_path, row)
            msg = (
                f"[reopt] {n}/{len(chosen)} {mid} k={row['k']} p={row['p']} {row['stratum']} "
                f"status={row['status']} steps={row['steps']} wall={row['wall_s']}s"
            )
            if row.get("ok"):
                msg += (
                    f" core_retained={row['core_retained']} kept={row['core_edges_kept']}/{row['core_edges_before']}"
                    f" mu2 {row['before']['mu2']}->{row['after']['mu2']} rmsd={row['rmsd_gxtb_to_mace_A']}"
                    f" artifacts={row['artifact_flags'] or '-'}"
                )
            else:
                msg += f" ERROR {row['error'][:120]}"
            print(msg, file=sys.stderr, flush=True)
    finally:
        worker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
