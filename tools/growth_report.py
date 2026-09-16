#!/usr/bin/env python3
"""Per-transition growth report for a lattice (Move Z) growth run.

Answers the two questions that decide whether to extend k: *is the ZB
growth pattern surviving* (per Se-skeleton family) and *where does the
proposal generator fail* (per (k,p) bin).  Reads ``zb_occupations.jsonl``
and the relaxed XYZ files of a run directory written by
``tools/run_molecular_growth.py`` / ``tools/bootstrap_zb_k1.py``.

For every transition k -> k+1 present in the run:

* families at k = distinct ``occupation_anion_skeleton_key`` (cubic-invariant
  Se lattice skeleton, p-agnostic) among propagation-eligible endpoints;
* a family *survives* when at least one propagation-eligible child at k+1
  lists one of its members in ``parent_structure_ids``;
* per surviving family: number of distinct child Se skeletons, the best
  child relative energy (dE to the lowest eligible endpoint of the same
  (k+1,p) bin -- energies are never compared across p), mu2/p, zb_score
  (two-tier descriptors of the relaxed endpoint) and the parent->child
  zb_score trend;
* per (k+1,p) bin: counts of propagate / lattice-route / failed endpoints
  and the failure causes (artifact by pair, fragmentation, unconverged,
  topology changed / split, chemistry rule).

Prints a Markdown report, writes ``growth_report.md`` and
``growth_report.json`` into the run directory, and with ``--gate FRACTION``
exits 2 when the family survival of the last transition is below FRACTION
(so a driver loop can stop extending k).
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from builder.nucleation.mace_descriptors import describe as two_tier_describe  # noqa: E402
from builder.nucleation.molecular_zb_growth import (  # noqa: E402
    occupation_anion_skeleton_key,
    occupation_from_record,
)


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    symbols, pos = [], []
    for line in lines[2 : 2 + n]:
        f = line.split()
        symbols.append(f[0])
        pos.append([float(x) for x in f[1:4]])
    return symbols, np.asarray(pos, float)


def failure_cause(rec: Dict[str, Any]) -> str:
    if rec.get("propagation_eligible"):
        return "propagate"
    viol = [str(v) for v in rec.get("violations") or []]
    kinds = {v.split(":")[0] for v in viol}
    if not rec.get("xtb_converged", True):
        return "unconverged"
    if "artifact" in kinds:
        pairs = sorted({v.split(":")[1] for v in viol if v.startswith("artifact:") and v.count(":") >= 2})
        return "artifact:" + ",".join(pairs)
    if "inorganic_disconnected" in kinds or "cn0" in kinds:
        return "fragmented"
    if rec.get("chemically_ok") and rec.get("topology_status") in {"changed", "split"}:
        return f"topology_{rec['topology_status']}"
    if kinds:
        return "rule:" + ",".join(sorted(kinds))
    return "other"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--anion", default="Se")
    ap.add_argument("--site-tolerance", type=float, default=0.2)
    ap.add_argument("--gate", type=float, default=None, help="minimum family survival fraction for the last transition")
    ap.add_argument("--parents", type=Path, nargs="*", default=[], help="additional run dirs whose manifests hold the first parents (e.g. the k=1 seed run)")
    args = ap.parse_args(argv)

    run = args.run_dir.expanduser().resolve()
    records: List[Dict[str, Any]] = []
    for base in [run] + [Path(x).expanduser().resolve() for x in args.parents]:
        manifest = base / "zb_occupations.jsonl"
        if not manifest.is_file():
            raise SystemExit(f"no zb_occupations.jsonl in {base}")
        with manifest.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    rec["_base"] = base
                    records.append(rec)

    # families and descriptors
    by_id: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        occ = occupation_from_record(rec["occupation"])
        rec["_k"], rec["_p"] = int(occ.k), int(occ.p)
        rec["_family"] = occupation_anion_skeleton_key(occ, args.anion, args.site_tolerance)
        rec["_desc"] = None
        if rec.get("propagation_eligible") and rec.get("xyz"):
            xyz = rec["_base"] / rec["xyz"]
            if xyz.is_file():
                sym, pos = read_xyz(xyz)
                d = two_tier_describe(sym, pos)
                rec["_desc"] = {"mu2": d["mu2"], "mu3": d["mu3"], "zb_score": d["zb_score"], "n_cdse_long": d["n_cdse_long"], "se_cn_strong": d["se_cn_strong"]}
        by_id[rec["structure_id"]] = rec

    ks = sorted({r["_k"] for r in records})
    report: Dict[str, Any] = {"run": str(run), "transitions": []}
    lines = [f"# Growth report: `{run.name}`", ""]
    last_survival = None
    for k in ks:
        children = [r for r in records if r["_k"] == k + 1]
        if not children:
            continue
        parents = [r for r in records if r["_k"] == k and r.get("propagation_eligible")]
        fam_members: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in parents:
            fam_members[r["_family"]].append(r)
        eligible_children = [c for c in children if c.get("propagation_eligible")]
        # bin minima at k+1 (eligible only)
        emin: Dict[int, float] = {}
        for c in eligible_children:
            e = float(c["energy_eV"])
            emin[c["_p"]] = min(emin.get(c["_p"], e), e)
        fam_rows = []
        survived = 0
        for fam, members in sorted(fam_members.items(), key=lambda kv: min(float(m["energy_eV"]) for m in kv[1])):
            member_ids = {m["structure_id"] for m in members}
            kids = [c for c in eligible_children if member_ids & set(c.get("parent_structure_ids") or [c.get("parent_structure_id")])]
            attempts = [c for c in children if member_ids & set(c.get("parent_structure_ids") or [c.get("parent_structure_id")])]
            ok = bool(kids)
            survived += ok
            child_fams = {c["_family"] for c in kids}
            best_de = min((float(c["energy_eV"]) - emin[c["_p"]] for c in kids), default=None)
            zb_parent = [m["_desc"]["zb_score"] for m in members if m.get("_desc") and m["_desc"]["zb_score"] is not None]
            zb_child = [c["_desc"]["zb_score"] for c in kids if c.get("_desc") and c["_desc"]["zb_score"] is not None]
            mu2p = [c["_desc"]["mu2"] / c["_p"] for c in kids if c.get("_desc") and c["_p"]]
            row = {
                "family": fam[:12], "n_members": len(members), "p_values": sorted({m["_p"] for m in members}),
                "attempted_children": len(attempts), "valid_children": len(kids), "survived": ok,
                "child_skeletons": len(child_fams), "best_child_dE_eV": best_de,
                "child_p": sorted({c["_p"] for c in kids}),
                "zb_parent": round(st.mean(zb_parent), 3) if zb_parent else None,
                "zb_child": round(st.mean(zb_child), 3) if zb_child else None,
                "mu2_over_p_child": round(st.mean(mu2p), 2) if mu2p else None,
            }
            fam_rows.append(row)
        survival = survived / len(fam_members) if fam_members else 0.0
        last_survival = survival
        # bin outcomes
        bins = {}
        for p in sorted({c["_p"] for c in children}):
            cs = [c for c in children if c["_p"] == p]
            causes = collections.Counter(failure_cause(c) for c in cs)
            bins[p] = {"n": len(cs), "causes": dict(causes.most_common())}
        report["transitions"].append({"k": k, "k_child": k + 1, "n_families": len(fam_members), "survived": survived, "survival": survival, "families": fam_rows, "bins": bins})

        lines += [f"## k={k} -> {k+1}", "", f"Se-skeleton families at k={k}: **{len(fam_members)}**, with a valid child: **{survived}** (survival {survival:.2f})", ""]
        lines += ["| family | members (p) | attempted | valid | child skeletons | best child dE (eV) | zb parent -> child | mu2/p child |", "|---|---|---:|---:|---:|---:|---|---:|"]
        for r in fam_rows:
            lines.append(f"| {r['family']} | {r['n_members']} ({','.join(map(str, r['p_values']))}) | {r['attempted_children']} | {r['valid_children']} | {r['child_skeletons']} | "
                         f"{'-' if r['best_child_dE_eV'] is None else f'{r['best_child_dE_eV']:.3f}'} | {r['zb_parent']} -> {r['zb_child']} | {r['mu2_over_p_child'] if r['mu2_over_p_child'] is not None else '-'} |")
        lines += ["", f"Outcomes at k={k+1} by p:", "", "| p | n | causes |", "|---|---:|---|"]
        for p, b in bins.items():
            lines.append(f"| {p} | {b['n']} | " + ", ".join(f"{c}: {n}" for c, n in b["causes"].items()) + " |")
        lines.append("")

    text = "\n".join(lines)
    (run / "growth_report.md").write_text(text + "\n")
    (run / "growth_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(text)
    if args.gate is not None and last_survival is not None and last_survival < args.gate:
        print(f"\nGATE: family survival {last_survival:.2f} < {args.gate}; do not extend k", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
