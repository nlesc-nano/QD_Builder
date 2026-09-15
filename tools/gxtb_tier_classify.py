#!/usr/bin/env python3
"""Assign g-xTB archive trust tiers before import into a MACE archive.

Implements the criteria in ``docs/nucleation/gxtb_trust_tiers.md``.  Reads one
or more adaptive ``minima.json`` archives (``{"control": {...},
"experimental": {...}}``), assigns every minimum a *record tier* (A or not-A,
"is the construction/lineage record internally valid?") and an *endpoint tier*
(B, C or D, "how far can the relaxed g-xTB geometry be trusted?"), and writes
a JSONL manifest plus a per-``(k,p)`` summary.

Tiers are assigned from the archived descriptors and coordinates only; no
energies are compared across protocols and no re-relaxation is performed.

Usage::

    python tools/gxtb_tier_classify.py \
        /path/to/growth_zb/nucleation_phase_c_k9_canonical_v1/minima.json \
        --out runs/mace_import/k9_tiers.jsonl --summary runs/mace_import/k9_tiers.md
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

# --------------------------------------------------------------------------
# Thresholds (Å).  Sources:
#   * artifact floors: geometry_packs/cdse_cdcl2_zb/run_mace_mh1.yaml
#     ``relaxation.artifact_min_distance``
#   * bond floors: 1st-percentile of relaxed g-xTB distances in the k9
#     canonical archive is Cd–Se 2.37 / Cd–Cl 2.29; floors are set ~0.1 Å
#     below that so only genuinely collapsed contacts trip them.
#   * bond ceilings: geometry_packs/cdse_cdcl2_zb/graph_rules.yaml
#     ``bond_max_distance`` (Cd–Se 3.25, Cd–Cl 2.90).
# --------------------------------------------------------------------------
ARTIFACT_FLOORS: Dict[str, float] = {
    "Cd-Cd": 2.80,
    "Se-Se": 2.80,
    "Cl-Se": 2.80,
    "Cl-Cl": 2.80,
}
BOND_FLOORS: Dict[str, float] = {"Cd-Se": 2.25, "Cd-Cl": 2.15}
BOND_MAX: Dict[str, float] = {"Cd-Se": 3.25, "Cd-Cl": 2.90}

#: Violations that mean the endpoint is a reconstruction / unusable parent.
HARD_VIOLATIONS = {"inorganic_disconnected", "min_cn"}
#: Violations that keep the geometry usable as a *diverse* seed but exclude
#: it from Tier B (chemically clean).  Everything not listed here or above is
#: treated as soft as well; unknown tags are reported so the list can grow.
SOFT_VIOLATIONS = {
    "mu3_host_bridge_overlap",
    "max_cn",
    "unsupported_motif",
    "bridges_per_cd_pair",
    "min_bridged_host_cn",
    "cl_on_cn4_cd",
    "mono_se_dual_terminal",
}

#: Se coordination-number loss that counts as "scaffold lost" even when the
#: core hash changed for a benign reason (e.g. one borrowed edge).
SE_CN_LOSS_FRACTION = 0.25


def _pair_key(a: str, b: str) -> str:
    return "-".join(sorted((a, b)))


def min_pair_distances(symbols: List[str], positions: List[List[float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, j in itertools.combinations(range(len(symbols)), 2):
        r = math.dist(positions[i], positions[j])
        key = _pair_key(symbols[i], symbols[j])
        if r < out.get(key, math.inf):
            out[key] = r
    return out


def composition_ok(symbols: List[str], k: int, p: int) -> bool:
    c = collections.Counter(symbols)
    return c.get("Cd", 0) == k + p and c.get("Se", 0) == k and c.get("Cl", 0) == 2 * p


def se_cn_mass(hist: Mapping[str, int], min_cn: int) -> int:
    return sum(int(n) for cn, n in hist.items() if int(cn) >= min_cn)


def classify_record(m: Mapping[str, Any]) -> Dict[str, Any]:
    """Return tier assignment and the reasons for one archived minimum."""
    reasons: List[str] = []
    k, p = int(m.get("k", -1)), int(m.get("p", -1))
    symbols = list(m.get("symbols") or [])
    positions = list(m.get("positions") or [])
    cons = dict(m.get("construction") or {})
    fin = dict(m.get("final") or {})
    violations = [str(v) for v in m.get("violations") or []]
    vkinds = {v.split(":", 1)[0] for v in violations}

    # ---------------- record tier (A / not-A) ----------------
    record_ok = True
    if k < 1 or p < 0:
        record_ok = False
        reasons.append("record:bad_kp")
    if not symbols or not composition_ok(symbols, k, p):
        record_ok = False
        reasons.append("record:composition_mismatch")
    if not m.get("edges"):
        record_ok = False
        reasons.append("record:no_edges")
    for key in ("core_hash", "graph_hash"):
        if not cons.get(key):
            record_ok = False
            reasons.append(f"record:missing_construction_{key}")
    if int(cons.get("n_components", 1)) != 1 or int(cons.get("core_components", 1)) != 1:
        record_ok = False
        reasons.append("record:construction_disconnected")
    if not (m.get("routes") or m.get("source_minimum_ids") or m.get("occupations")):
        # No ancestry at all: still a valid *structure* record but not a
        # lineage record, so it cannot seed family tracking.
        record_ok = False
        reasons.append("record:no_ancestry")
    record_tier = "A" if record_ok else "-"

    # ---------------- endpoint tier (B / C / D) ----------------
    endpoint = "B"

    def demote(to: str, why: str) -> None:
        nonlocal endpoint
        order = {"B": 0, "C": 1, "D": 2}
        if order[to] > order[endpoint]:
            endpoint = to
        reasons.append(why)

    if not positions or m.get("energy_eV") is None:
        demote("D", "endpoint:missing_geometry_or_energy")
        return _result(m, record_tier, "D", reasons, {})

    status = str(m.get("status") or ("converged" if m.get("converged") else ""))
    if not m.get("converged") and status not in {"maxcycle"}:
        demote("D", f"endpoint:not_converged:{status or 'unknown'}")
    elif status == "maxcycle":
        demote("C", "endpoint:maxcycle")

    if int(fin.get("n_components", 1)) != 1:
        demote("D", "endpoint:disconnected")
    if vkinds & HARD_VIOLATIONS:
        demote("D", "endpoint:hard_violation:" + ",".join(sorted(vkinds & HARD_VIOLATIONS)))

    dmin = min_pair_distances(symbols, positions)
    for key, floor in ARTIFACT_FLOORS.items():
        if dmin.get(key, math.inf) < floor:
            demote("D", f"endpoint:artifact_contact:{key}={dmin[key]:.2f}")
    for key, floor in BOND_FLOORS.items():
        if dmin.get(key, math.inf) < floor:
            demote("D", f"endpoint:collapsed_bond:{key}={dmin[key]:.2f}")

    # Scaffold retention: exact certificate first, Se-CN mass as backstop.
    core_retained = bool(cons.get("core_hash")) and cons.get("core_hash") == fin.get("core_hash")
    se_before = se_cn_mass(cons.get("se_cn") or {}, 3)
    se_after = se_cn_mass(fin.get("se_cn") or {}, 3)
    se_loss = (se_before - se_after) / se_before if se_before else 0.0
    if not core_retained:
        if int(fin.get("core_components", 1)) != 1:
            demote("D", "endpoint:scaffold_lost:core_split")
        elif se_loss > SE_CN_LOSS_FRACTION:
            demote("D", f"endpoint:scaffold_lost:se_cn3plus_loss={se_loss:.2f}")
        else:
            demote("C", "endpoint:scaffold_changed")

    soft = vkinds - HARD_VIOLATIONS
    if soft:
        demote("C", "endpoint:soft_violation:" + ",".join(sorted(soft)))
    unknown = soft - SOFT_VIOLATIONS
    if unknown:
        reasons.append("endpoint:unknown_violation:" + ",".join(sorted(unknown)))

    metrics = {
        "core_retained": core_retained,
        "graph_retained": cons.get("graph_hash") == fin.get("graph_hash"),
        "connectivity_preserved": bool(m.get("connectivity_preserved", False)),
        "se_cn3plus_loss": round(se_loss, 3),
        "mu2_over_p": (fin.get("mu2") / p) if p else None,
        "min_distances": {k_: round(v, 3) for k_, v in dmin.items()},
    }
    return _result(m, record_tier, endpoint, reasons, metrics)


def _result(m, record_tier, endpoint_tier, reasons, metrics) -> Dict[str, Any]:
    return {
        "minimum_id": m.get("minimum_id"),
        "structure_id": m.get("structure_id"),
        "k": m.get("k"),
        "p": m.get("p"),
        "role": m.get("role"),
        "source_protocol": m.get("source_protocol"),
        "source_archives": m.get("source_archives"),
        "gxtb_energy_eV": m.get("energy_eV"),
        "record_tier": record_tier,
        "endpoint_tier": endpoint_tier,
        "tier": f"{record_tier}{endpoint_tier}",
        "reasons": reasons,
        "metrics": metrics,
        "construction_core_hash": (m.get("construction") or {}).get("core_hash"),
        "final_core_hash": (m.get("final") or {}).get("core_hash"),
        "mace_status": "pending",
    }


def iter_minima(path: Path) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "experimental" in data:
        for arm in ("control", "experimental"):
            for mid, m in (data.get(arm) or {}).items():
                yield arm, m
    elif isinstance(data, dict):
        for mid, m in data.items():
            yield "experimental", m
    else:
        for m in data:
            yield "experimental", m


def summarize(rows: List[Dict[str, Any]]) -> str:
    by_bin: Dict[Tuple[int, int], collections.Counter] = collections.defaultdict(collections.Counter)
    total = collections.Counter()
    reason_counts = collections.Counter()
    for r in rows:
        by_bin[(r["k"], r["p"])][r["tier"]] += 1
        total[r["tier"]] += 1
        for why in r["reasons"]:
            reason_counts[why.split("=")[0]] += 1
    tiers = sorted(total)
    lines = ["| k | p | n | " + " | ".join(tiers) + " |", "|---|---|---:|" + "---:|" * len(tiers)]
    for (k, p) in sorted(by_bin):
        c = by_bin[(k, p)]
        lines.append(f"| {k} | {p} | {sum(c.values())} | " + " | ".join(str(c.get(t, 0)) for t in tiers) + " |")
    lines.append(f"| **all** | | {sum(total.values())} | " + " | ".join(str(total[t]) for t in tiers) + " |")
    lines.append("")
    lines.append("Reason counts:")
    for why, n in reason_counts.most_common():
        lines.append(f"- `{why}`: {n}")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("minima", nargs="+", type=Path, help="minima.json archive(s)")
    ap.add_argument("--out", type=Path, required=True, help="JSONL manifest")
    ap.add_argument("--summary", type=Path, help="Markdown summary")
    ap.add_argument("--arm", default="experimental", choices=["experimental", "control", "all"])
    args = ap.parse_args(argv)

    rows: List[Dict[str, Any]] = []
    for path in args.minima:
        for arm, m in iter_minima(path):
            if args.arm != "all" and arm != args.arm:
                continue
            row = classify_record(m)
            row["archive"] = path.parent.name
            row["arm"] = arm
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    text = summarize(rows)
    if args.summary:
        args.summary.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
