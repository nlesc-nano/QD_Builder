#!/usr/bin/env python3
"""Re-score MACE re-optimization / DFT-comparison journals with the two-tier
descriptors from ``builder.nucleation.mace_descriptors``.

Reads a ``results.jsonl`` written by ``mace_reopt_subset.py`` (g-xTB start ->
MACE endpoint) or ``compare_mace_dft_full.py`` (DFT endpoint -> MACE
endpoint), computes ``describe()`` for the before and after geometries, and
writes ``<results>.rescored.jsonl`` plus a Markdown summary comparing the
legacy single-cutoff verdicts with the two-tier ones.

The descriptor module is loaded by file path so this runs in the MACE
environment without importing the ``builder`` package (which needs pymatgen).
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_descriptors():
    path = ROOT / "src" / "builder" / "nucleation" / "mace_descriptors.py"
    spec = importlib.util.spec_from_file_location("mace_descriptors", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--minima", type=Path, help="minima.json for g-xTB start geometries (reopt journals)")
    ap.add_argument("--summary", type=Path)
    args = ap.parse_args(argv)
    md = load_descriptors()

    minima = None
    if args.minima:
        minima = json.loads(args.minima.read_text())
        minima = minima.get("experimental", minima)

    rows = [json.loads(l) for l in args.results.open()]
    out_path = args.results.with_suffix(".rescored.jsonl")
    scored: List[Dict[str, Any]] = []
    with out_path.open("w") as fh:
        for r in rows:
            if "from_dft" in r:  # DFT comparison journal
                if not r["from_dft"].get("ok"):
                    continue
                sym = r["symbols"]
                before = md.describe(sym, r["dft_positions"])
                after = md.describe(sym, r["from_dft"]["positions"])
                legacy_same = bool(r["from_dft"]["core_same_as_dft"])
                sid, k, p, strat = r["structure_id"], r["k"], r["p"], "dft"
            else:  # reopt journal
                if not r.get("ok") or minima is None:
                    continue
                m = minima[r["minimum_id"]]
                sym = m["symbols"]
                before = md.describe(sym, m["positions"])
                after = md.describe(sym, r["positions"])
                legacy_same = bool(r["core_retained"])
                sid, k, p, strat = r["minimum_id"], r["k"], r["p"], r["stratum"]
            change = md.scaffold_change(before, after)
            row = {
                "id": sid, "k": k, "p": p, "stratum": strat,
                "legacy_core_same": legacy_same,
                "change": change,
                "before": before, "after": after,
            }
            scored.append(row)
            fh.write(json.dumps(row, separators=(",", ":"), default=float) + "\n")

    # ---- summary ----
    lines = [f"# Two-tier descriptor rescoring of `{args.results}`", "", f"n = {len(scored)}", ""]
    lab = collections.Counter(s["change"]["label"] for s in scored)
    lines += ["| scaffold change (strong graph) | n |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in lab.most_common()]
    lines += ["", f"legacy single-cutoff `core_retained`: {sum(s['legacy_core_same'] for s in scored)} / {len(scored)}",
              f"two-tier strong-hash retained: {sum(s['change']['strong_hash_same'] for s in scored)} / {len(scored)}", ""]
    def pooled(key, side):
        c = collections.Counter()
        for s in scored:
            for cn, n in s[side][key].items():
                c[int(cn)] += n
        return dict(sorted(c.items()))
    for key in ("se_cn_strong", "se_cn_all", "cd_cn_se_strong"):
        lines.append(f"- `{key}` before {pooled(key,'before')} -> after {pooled(key,'after')}")
    lines.append(f"- long Cd–Se contacts: before {sum(s['before']['n_cdse_long'] for s in scored)} -> after {sum(s['after']['n_cdse_long'] for s in scored)}")
    lines.append(f"- Se with a long contact: before {sum(s['before']['se_with_long_contact'] for s in scored)} -> after {sum(s['after']['se_with_long_contact'] for s in scored)}")
    zb_b = [s["before"]["zb_score"] for s in scored if s["before"]["zb_score"] is not None]
    zb_a = [s["after"]["zb_score"] for s in scored if s["after"]["zb_score"] is not None]
    lines.append(f"- zb_score median before {st.median(zb_b):.3f} -> after {st.median(zb_a):.3f}")
    for key in ("q4_se_strong", "pyr_se_strong", "q4_nn4_se"):
        b = [s["before"][key]["mean"] for s in scored if s["before"][key]["mean"] is not None]
        a = [s["after"][key]["mean"] for s in scored if s["after"][key]["mean"] is not None]
        if b and a:
            lines.append(f"- `{key}` mean-of-means before {st.mean(b):.3f} -> after {st.mean(a):.3f}")
    lines.append(f"- 4-ring fraction (strong core) mean before {np.nanmean([s['before']['ring4_fraction'] if s['before']['ring4_fraction'] is not None else np.nan for s in scored]):.3f} -> after {np.nanmean([s['after']['ring4_fraction'] if s['after']['ring4_fraction'] is not None else np.nan for s in scored]):.3f}")
    lines.append("")
    lines += ["| stratum | n | retained | retained_long_changed | minor | changed | split |", "|---|---:|---:|---:|---:|---:|---:|"]
    for strat in sorted({s["stratum"] for s in scored}):
        ss = [s for s in scored if s["stratum"] == strat]
        c = collections.Counter(s["change"]["label"] for s in ss)
        lines.append(f"| {strat} | {len(ss)} | {c['retained']} | {c['retained_long_contacts_changed']} | {c['minor']} | {c['changed']} | {c['split']} |")
    text = "\n".join(lines)
    print(text)
    if args.summary:
        args.summary.write_text(text + "\n")
    print(f"\nwrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
