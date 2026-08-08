#!/usr/bin/env python3
"""Start → relaxed graph analysis on a light DFT tree.

Reads ``k###/p###/<id>/{start.xyz,final.xyz}`` (as produced by
``extract_start_final.py``), rebuilds chemical graphs with pack cutoffs,
compares exact + approximate topology, records edge events and Kabsch RMSD,
and correlates breakdown with start-only predictors for candidate construction
rules.

Example
-------

    python tools/analyze_dft_start_final.py \\
      --root /path/to/dft_partial \\
      --output /path/to/dft_partial/start_final_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from dft_geometry_mine.bonds import (  # noqa: E402
    BondCutoffs,
    GraphAnalysis,
    analyze_frame,
)
from dft_geometry_mine.xyz_io import Frame, read_xyz_frames  # noqa: E402

# NOTE: start/final share fixed atom indices (CP2K GEO_OPT). Compare edge
# *sets on those indices* — do NOT use isomorphism certificates here.
# Cd–Se-only canonical_form on k=2 frames was ~5 s/isomer: many identical
# isolated Cd vertices explode individualisation. Index equality is O(E).

CLOSABLE_A = 3.50
NEAR_JACCARD = 0.85


def _load_xyz(path: Path) -> Frame:
    frames = read_xyz_frames(path)
    if not frames:
        raise ValueError(f"no frames in {path}")
    return frames[-1]


def _edge_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _edge_set(analysis: GraphAnalysis) -> Set[Tuple[int, int]]:
    return {_edge_key(i, j) for i, j, _t, _d in analysis.edges}


def _edge_set_typed(
    analysis: GraphAnalysis, pair_type: Optional[str] = None
) -> Set[Tuple[int, int]]:
    if pair_type is None:
        return _edge_set(analysis)
    return {
        _edge_key(i, j)
        for i, j, ptype, _d in analysis.edges
        if ptype == pair_type
    }


def _edge_map(analysis: GraphAnalysis) -> Dict[Tuple[int, int], Tuple[str, float]]:
    out: Dict[Tuple[int, int], Tuple[str, float]] = {}
    for i, j, ptype, length in analysis.edges:
        out[_edge_key(i, j)] = (ptype, float(length))
    return out

def _jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _cl_roles(analysis: GraphAnalysis) -> Dict[str, int]:
    symbols = analysis.symbols
    n_term = n_mu2 = n_mu3 = 0
    for idx, sym in enumerate(symbols):
        if sym != "Cl":
            continue
        deg = analysis.degrees[idx]
        if deg <= 1:
            n_term += 1
        elif deg == 2:
            n_mu2 += 1
        else:
            n_mu3 += 1
    return {"n_terminal_cl": n_term, "n_mu2_cl": n_mu2, "n_mu3_cl": n_mu3}


def _cd_cn_hist(analysis: GraphAnalysis) -> str:
    counts = Counter(
        analysis.degrees[i]
        for i, s in enumerate(analysis.symbols)
        if s == "Cd"
    )
    return ",".join(f"cn{cn}:{counts[cn]}" for cn in sorted(counts))


def _cd_cn_bins(analysis: GraphAnalysis) -> Tuple[int, int, int, float]:
    cds = [
        analysis.degrees[i]
        for i, s in enumerate(analysis.symbols)
        if s == "Cd"
    ]
    if not cds:
        return 0, 0, 0, 0.0
    n2 = sum(1 for d in cds if d == 2)
    n3 = sum(1 for d in cds if d == 3)
    n4 = sum(1 for d in cds if d >= 4)
    return n2, n3, n4, float(sum(cds) / len(cds))


def _cdse_adjacency(analysis: GraphAnalysis) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = defaultdict(list)
    for i, j, ptype, _d in analysis.edges:
        if ptype != "CdSe":
            continue
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _shortest_path(
    adj: Dict[int, List[int]], source: int, target: int
) -> Optional[int]:
    if source == target:
        return 0
    seen = {source}
    q = deque([(source, 0)])
    while q:
        node, dist = q.popleft()
        for neigh in adj.get(node, []):
            if neigh in seen:
                continue
            if neigh == target:
                return dist + 1
            seen.add(neigh)
            q.append((neigh, dist + 1))
    return None


def _mu2_host_pairs(
    analysis: GraphAnalysis,
) -> List[Tuple[int, Tuple[int, int]]]:
    """Return (cl_index, (host_a, host_b)) for each μ₂ Cl."""

    symbols = analysis.symbols
    pairs: List[Tuple[int, Tuple[int, int]]] = []
    for cl, sym in enumerate(symbols):
        if sym != "Cl" or analysis.degrees[cl] != 2:
            continue
        hosts = sorted(
            n for n in analysis.neighbors[cl] if symbols[n] == "Cd"
        )
        if len(hosts) == 2:
            pairs.append((cl, (hosts[0], hosts[1])))
    return pairs


def _mu2_path_stats(analysis: GraphAnalysis) -> Tuple[int, int, int, int]:
    """Counts of μ₂ with Cd–Se path length 2, 4, other, disconnected."""

    adj = _cdse_adjacency(analysis)
    n2 = n4 = n_other = n_disc = 0
    for _cl, (a, b) in _mu2_host_pairs(analysis):
        d = _shortest_path(adj, a, b)
        if d is None:
            n_disc += 1
        elif d == 2:
            n2 += 1
        elif d == 4:
            n4 += 1
        else:
            n_other += 1
    return n2, n4, n_other, n_disc


def _max_shared_pair(analysis: GraphAnalysis) -> int:
    pair_count: Counter = Counter()
    for _cl, pair in _mu2_host_pairs(analysis):
        pair_count[pair] += 1
    # also μ3 contributes pairs but max_shared usually about μ2
    return max(pair_count.values()) if pair_count else 0


def _start_predictors(analysis: GraphAnalysis) -> Dict[str, object]:
    symbols = analysis.symbols
    n_cd2, n_cd3, n_cd4, mean_cn = _cd_cn_bins(analysis)
    roles = _cl_roles(analysis)
    n_closable = 0
    n_unsat = 0
    n_cl2se1_dual = 0
    n_term_on_cd2 = 0
    dist = analysis.distances

    cd_ids = [i for i, s in enumerate(symbols) if s == "Cd"]
    se_ids = [i for i, s in enumerate(symbols) if s == "Se"]

    # free valence assuming max CN 4
    free = {
        c: max(0, 4 - analysis.degrees[c]) for c in cd_ids
    }

    for cl, sym in enumerate(symbols):
        if sym != "Cl":
            continue
        hosts = [n for n in analysis.neighbors[cl] if symbols[n] == "Cd"]
        if len(hosts) == 1:
            h = hosts[0]
            if analysis.degrees[h] == 2:
                n_term_on_cd2 += 1
            # closable: another Cd CN2 within CLOSABLE_A, not already bonded
            for u in cd_ids:
                if u == h:
                    continue
                if analysis.degrees[u] != 2:
                    continue
                if u in analysis.neighbors[cl]:
                    continue
                if float(dist[cl, u]) <= CLOSABLE_A:
                    n_closable += 1
                    break
            # unsaturated bridge candidate: some other Cd with free valence
            for u in cd_ids:
                if u == h or free[u] <= 0:
                    continue
                if u in analysis.neighbors[h]:
                    # already bonded via? check common - any path ok
                    pass
                if free[h] >= 0 and free[u] > 0:
                    # terminal on h, u has room for a new Cl bond
                    n_unsat += 1
                    break

    # mono-Se dual terminal
    for c in cd_ids:
        neigh = analysis.neighbors[c]
        n_se = sum(1 for n in neigh if symbols[n] == "Se")
        term_cl = [
            n
            for n in neigh
            if symbols[n] == "Cl" and analysis.degrees[n] == 1
        ]
        if n_se == 1 and len(term_cl) >= 2:
            n_cl2se1_dual += 1

    mu2_d2, mu2_d4, mu2_other, mu2_disc = _mu2_path_stats(analysis)

    # cdse cn pairs
    pair_counts: Counter = Counter()
    for i, j, ptype, _d in analysis.edges:
        if ptype != "CdSe":
            continue
        cd = i if symbols[i] == "Cd" else j
        se = j if symbols[i] == "Cd" else i
        pair_counts[f"{analysis.degrees[cd]}-{analysis.degrees[se]}"] += 1
    cdse_pairs = ",".join(
        f"{k}:{pair_counts[k]}" for k in sorted(pair_counts)
    )

    return {
        "n_cd2": n_cd2,
        "n_cd3": n_cd3,
        "n_cd4": n_cd4,
        "mean_cd_cn": round(mean_cn, 4),
        "n_terminal_cl": roles["n_terminal_cl"],
        "n_mu2_cl": roles["n_mu2_cl"],
        "n_mu3_cl": roles["n_mu3_cl"],
        "n_closable_terminal_cd2": n_closable,
        "n_unsaturated_bridge_candidates": n_unsat,
        "n_cl2se1_dual_terminal": n_cl2se1_dual,
        "n_terminal_on_cd2": n_term_on_cd2,
        "n_mu2_dist2": mu2_d2,
        "n_mu2_dist4": mu2_d4,
        "n_mu2_path_other": mu2_other,
        "n_mu2_disconnected": mu2_disc,
        "max_shared_pair": _max_shared_pair(analysis),
        "cdse_cn_pairs": cdse_pairs,
        "cd_cn_hist": _cd_cn_hist(analysis),
    }


def _kabsch_rmsd(
    ref: np.ndarray, mob: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Return (rmsd, mobile aligned to ref)."""

    assert ref.shape == mob.shape
    n = ref.shape[0]
    if n == 0:
        return 0.0, mob.copy()
    ref_c = ref.mean(axis=0)
    mob_c = mob.mean(axis=0)
    r = ref - ref_c
    m = mob - mob_c
    h = m.T @ r
    u, _s, vt = np.linalg.svd(h)
    d = np.linalg.det(vt.T @ u.T)
    sign = np.diag([1.0, 1.0, 1.0 if d >= 0 else -1.0])
    rot = u @ sign @ vt
    aligned = (m @ rot) + ref_c
    diff = aligned - ref
    rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    return rmsd, aligned


def _element_rmsd(
    symbols: Sequence[str], ref: np.ndarray, aligned: np.ndarray, el: str
) -> float:
    idx = [i for i, s in enumerate(symbols) if s == el]
    if not idx:
        return float("nan")
    diff = aligned[idx] - ref[idx]
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _classify_outcome(
    *,
    cert_match: bool,
    cdse_cert_match: bool,
    jaccard: float,
    n_start: int,
    n_final: int,
    final_contact: bool,
    min_jaccard_near: float,
) -> str:
    if final_contact:
        return "contact_violation"
    if cert_match:
        return "retained"
    delta = n_final - n_start
    if jaccard >= min_jaccard_near and abs(delta) <= 1:
        return "near_retained"
    # Cd–Se skeleton changed (and not a tiny near-retain)
    if not cdse_cert_match:
        return "skeleton_break"
    if delta > 0:
        return "closed_extra"
    if delta < 0:
        return "opened_lost"
    return "rearranged"


def _discover_pairs(root: Path) -> List[Tuple[Path, Path, int, int, str]]:
    """Return (start, final, k, p, structure_id)."""

    found: List[Tuple[Path, Path, int, int, str]] = []
    for start in sorted(root.glob("k*/p*/*/start.xyz")):
        final = start.parent / "final.xyz"
        if not final.is_file():
            continue
        structure_id = start.parent.name
        k = p = -1
        for part in start.parts:
            if part.startswith("k") and part[1:].isdigit():
                k = int(part[1:])
            if part.startswith("p") and part[1:].isdigit():
                p = int(part[1:])
        found.append((start, final, k, p, structure_id))
    return found


def _lift_table(
    rows: List[Dict[str, object]], feature: str
) -> Dict[str, object]:
    """Contingency of binary feature vs breakdown."""

    def is_break(outcome: str) -> bool:
        return outcome not in {"retained", "near_retained"}

    def feat_on(row: Dict[str, object]) -> bool:
        val = row.get(feature, 0)
        try:
            return float(val) > 0
        except (TypeError, ValueError):
            return bool(val)

    n = len(rows)
    if n == 0:
        return {}
    n_break = sum(1 for r in rows if is_break(str(r["outcome"])))
    p_break = n_break / n
    on = [r for r in rows if feat_on(r)]
    off = [r for r in rows if not feat_on(r)]
    p_break_on = (
        sum(1 for r in on if is_break(str(r["outcome"]))) / len(on)
        if on
        else float("nan")
    )
    p_break_off = (
        sum(1 for r in off if is_break(str(r["outcome"]))) / len(off)
        if off
        else float("nan")
    )
    lift = (
        p_break_on / p_break
        if p_break > 0 and not math.isnan(p_break_on)
        else float("nan")
    )
    return {
        "feature": feature,
        "n_on": len(on),
        "n_off": len(off),
        "p_break_global": round(p_break, 4),
        "p_break_on": None if math.isnan(p_break_on) else round(p_break_on, 4),
        "p_break_off": None
        if math.isnan(p_break_off)
        else round(p_break_off, 4),
        "lift": None if math.isnan(lift) else round(lift, 3),
        "outcome_on": dict(Counter(str(r["outcome"]) for r in on)),
        "outcome_off": dict(Counter(str(r["outcome"]) for r in off)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="light DFT tree root (k*/p*/*/start.xyz + final.xyz)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output directory for CSVs and summaries",
    )
    parser.add_argument("--cd-se-cutoff", type=float, default=3.25)
    parser.add_argument("--cd-cl-cutoff", type=float, default=3.10)
    parser.add_argument(
        "--min-jaccard-near",
        type=float,
        default=NEAR_JACCARD,
        help="jaccard threshold for near_retained (default 0.85)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=8,
        help="breakdown vignettes in summary.md",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cutoffs = BondCutoffs(
        cd_se=float(args.cd_se_cutoff),
        cd_cl=float(args.cd_cl_cutoff),
    )
    pairs = _discover_pairs(root)
    if not pairs:
        print(f"No start+final pairs under {root}", file=sys.stderr)
        return 1

    per_rows: List[Dict[str, object]] = []
    edge_rows: List[Dict[str, object]] = []
    examples: List[Dict[str, object]] = []

    for pair_index, (start_path, final_path, k, p, structure_id) in enumerate(
        pairs, start=1
    ):
        if pair_index == 1 or pair_index % 25 == 0 or pair_index == len(pairs):
            print(
                f"  … {pair_index}/{len(pairs)} {structure_id}",
                flush=True,
            )
        row: Dict[str, object] = {
            "k": k,
            "p": p,
            "structure_id": structure_id,
            "run_dir": start_path.parent.relative_to(root).as_posix(),
        }
        try:
            start_fr = _load_xyz(start_path)
            final_fr = _load_xyz(final_path)
            if start_fr.n_atoms != final_fr.n_atoms:
                row["outcome"] = "error:atom_count_mismatch"
                per_rows.append(row)
                continue
            if start_fr.symbols != final_fr.symbols:
                # still analyze but flag
                row["symbol_order_match"] = 0
            else:
                row["symbol_order_match"] = 1

            start_g = analyze_frame(start_fr, cutoffs)
            final_g = analyze_frame(final_fr, cutoffs)
            e_s = _edge_set(start_g)
            e_f = _edge_set(final_g)
            map_s = _edge_map(start_g)
            map_f = _edge_map(final_g)
            gained = e_f - e_s
            lost = e_s - e_f
            jacc = _jaccard(e_s, e_f)
            # Fixed atom indices: exact topology match ≡ identical edge sets.
            cert_match = e_s == e_f
            cdse_s = _edge_set_typed(start_g, "CdSe")
            cdse_f = _edge_set_typed(final_g, "CdSe")
            cdse_match = cdse_s == cdse_f
            roles_s = _cl_roles(start_g)
            roles_f = _cl_roles(final_g)
            pred = _start_predictors(start_g)

            # RMSD
            ref = np.asarray(start_fr.coordinates, dtype=float)
            mob = np.asarray(final_fr.coordinates, dtype=float)
            rmsd_all, aligned = _kabsch_rmsd(ref, mob)
            disp = np.linalg.norm(aligned - ref, axis=1)
            max_i = int(np.argmax(disp))
            rmsd_cd = _element_rmsd(start_fr.symbols, ref, aligned, "Cd")
            rmsd_se = _element_rmsd(start_fr.symbols, ref, aligned, "Se")
            rmsd_cl = _element_rmsd(start_fr.symbols, ref, aligned, "Cl")

            # μ2 path change summary
            mu2_s = {
                pair: _shortest_path(_cdse_adjacency(start_g), pair[0], pair[1])
                for _cl, pair in _mu2_host_pairs(start_g)
            }
            mu2_f_pairs = {
                pair for _cl, pair in _mu2_host_pairs(final_g)
            }
            n_mu2_path_changed = 0
            for pair, d0 in mu2_s.items():
                if pair not in mu2_f_pairs:
                    n_mu2_path_changed += 1
                    continue
                d1 = _shortest_path(_cdse_adjacency(final_g), pair[0], pair[1])
                if d0 != d1:
                    n_mu2_path_changed += 1

            outcome = _classify_outcome(
                cert_match=cert_match,
                cdse_cert_match=cdse_match,
                jaccard=jacc,
                n_start=len(e_s),
                n_final=len(e_f),
                final_contact=final_g.has_homonuclear_contact,
                min_jaccard_near=float(args.min_jaccard_near),
            )

            row.update(pred)
            row.update(
                {
                    "outcome": outcome,
                    "cert_match": int(cert_match),
                    "cdse_cert_match": int(cdse_match),
                    "edge_jaccard": round(jacc, 4),
                    "n_edges_start": len(e_s),
                    "n_edges_final": len(e_f),
                    "n_gained": len(gained),
                    "n_lost": len(lost),
                    "delta_edges": len(e_f) - len(e_s),
                    "n_terminal_cl_final": roles_f["n_terminal_cl"],
                    "n_mu2_cl_final": roles_f["n_mu2_cl"],
                    "n_mu3_cl_final": roles_f["n_mu3_cl"],
                    "delta_terminal_cl": roles_f["n_terminal_cl"]
                    - roles_s["n_terminal_cl"],
                    "delta_mu2_cl": roles_f["n_mu2_cl"] - roles_s["n_mu2_cl"],
                    "delta_mu3_cl": roles_f["n_mu3_cl"] - roles_s["n_mu3_cl"],
                    "cd_cn_hist_final": _cd_cn_hist(final_g),
                    "n_mu2_path_changed": n_mu2_path_changed,
                    "rmsd_all": round(rmsd_all, 4),
                    "rmsd_cd": round(rmsd_cd, 4)
                    if not math.isnan(rmsd_cd)
                    else "",
                    "rmsd_se": round(rmsd_se, 4)
                    if not math.isnan(rmsd_se)
                    else "",
                    "rmsd_cl": round(rmsd_cl, 4)
                    if not math.isnan(rmsd_cl)
                    else "",
                    "max_disp": round(float(disp[max_i]), 4),
                    "max_disp_atom": max_i,
                    "max_disp_symbol": start_fr.symbols[max_i],
                    "final_homonuclear": int(final_g.has_homonuclear_contact),
                    "start_homonuclear": int(start_g.has_homonuclear_contact),
                }
            )
            per_rows.append(row)

            for a, b in sorted(gained):
                ptype, d_f = map_f[(a, b)]
                d_s = float(start_g.distances[a, b])
                edge_rows.append(
                    {
                        "k": k,
                        "p": p,
                        "structure_id": structure_id,
                        "event": "gained",
                        "i": a,
                        "j": b,
                        "symbol_i": start_fr.symbols[a],
                        "symbol_j": start_fr.symbols[b],
                        "pair_type": ptype,
                        "d_start": round(d_s, 4),
                        "d_final": round(d_f, 4),
                        "outcome": outcome,
                    }
                )
            for a, b in sorted(lost):
                ptype, d_s = map_s[(a, b)]
                d_f = float(final_g.distances[a, b])
                edge_rows.append(
                    {
                        "k": k,
                        "p": p,
                        "structure_id": structure_id,
                        "event": "lost",
                        "i": a,
                        "j": b,
                        "symbol_i": start_fr.symbols[a],
                        "symbol_j": start_fr.symbols[b],
                        "pair_type": ptype,
                        "d_start": round(d_s, 4),
                        "d_final": round(d_f, 4),
                        "outcome": outcome,
                    }
                )

            if outcome not in {"retained", "near_retained"}:
                top = np.argsort(-disp)[:3]
                examples.append(
                    {
                        "structure_id": structure_id,
                        "k": k,
                        "p": p,
                        "outcome": outcome,
                        "jaccard": round(jacc, 3),
                        "rmsd": round(rmsd_all, 3),
                        "gained": len(gained),
                        "lost": len(lost),
                        "delta_mu2": roles_f["n_mu2_cl"] - roles_s["n_mu2_cl"],
                        "top_disp": [
                            f"{start_fr.symbols[i]}{i}:{disp[i]:.2f}Å"
                            for i in top
                        ],
                        "predictors": {
                            "closable": pred["n_closable_terminal_cd2"],
                            "mu2_dist2": pred["n_mu2_dist2"],
                            "unsat": pred["n_unsaturated_bridge_candidates"],
                            "n_cd2": pred["n_cd2"],
                        },
                    }
                )
        except Exception as exc:  # noqa: BLE001
            row["outcome"] = f"error:{exc}"
            per_rows.append(row)

    # --- write CSVs ---
    per_path = out_dir / "per_isomer.csv"
    if per_rows:
        fields = list(per_rows[0].keys())
        # stable union of keys
        for r in per_rows[1:]:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with per_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(per_rows)

    edge_path = out_dir / "edge_events.csv"
    edge_fields = [
        "k",
        "p",
        "structure_id",
        "event",
        "i",
        "j",
        "symbol_i",
        "symbol_j",
        "pair_type",
        "d_start",
        "d_final",
        "outcome",
    ]
    with edge_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=edge_fields)
        writer.writeheader()
        writer.writerows(edge_rows)

    # --- summary stats ---
    ok_rows = [
        r
        for r in per_rows
        if not str(r.get("outcome", "")).startswith("error")
    ]
    outcomes = Counter(str(r["outcome"]) for r in ok_rows)
    n = len(ok_rows)
    n_hold = sum(
        1
        for r in ok_rows
        if r["outcome"] in {"retained", "near_retained"}
    )
    by_kp: Dict[str, Dict[str, int]] = defaultdict(Counter)
    for r in ok_rows:
        key = f"k{int(r['k']):03d}_p{int(r['p']):03d}"
        by_kp[key][str(r["outcome"])] += 1

    features = [
        "n_closable_terminal_cd2",
        "n_unsaturated_bridge_candidates",
        "n_cl2se1_dual_terminal",
        "n_terminal_on_cd2",
        "n_mu2_dist2",
        "n_mu2_dist4",
        "n_cd2",
        "n_mu3_cl",
        "max_shared_pair",
    ]
    lifts = [_lift_table(ok_rows, f) for f in features]
    lifts_sorted = sorted(
        [x for x in lifts if x.get("n_on", 0) >= 3],
        key=lambda x: (x.get("lift") is None, -(x.get("lift") or 0)),
    )

    # edge event aggregates
    gained_types = Counter(
        e["pair_type"] for e in edge_rows if e["event"] == "gained"
    )
    lost_types = Counter(
        e["pair_type"] for e in edge_rows if e["event"] == "lost"
    )
    # terminal→bridge signature: gained CdCl + lost CdCl with role roles
    n_gained_cdcl = gained_types.get("CdCl", 0)
    n_lost_cdcl = lost_types.get("CdCl", 0)
    n_gained_cdse = gained_types.get("CdSe", 0)
    n_lost_cdse = lost_types.get("CdSe", 0)

    mean_jacc = (
        float(np.mean([float(r["edge_jaccard"]) for r in ok_rows]))
        if ok_rows
        else 0.0
    )
    mean_rmsd = (
        float(np.mean([float(r["rmsd_all"]) for r in ok_rows]))
        if ok_rows
        else 0.0
    )

    summary = {
        "n_pairs_found": len(pairs),
        "n_analyzed": n,
        "n_errors": len(per_rows) - n,
        "outcomes": dict(outcomes),
        "fraction_hold_exact_or_near": round(n_hold / n, 4) if n else 0.0,
        "fraction_retained_exact": round(
            outcomes.get("retained", 0) / n, 4
        )
        if n
        else 0.0,
        "mean_edge_jaccard": round(mean_jacc, 4),
        "mean_rmsd_all": round(mean_rmsd, 4),
        "by_kp": {k: dict(v) for k, v in sorted(by_kp.items())},
        "edge_events": {
            "n_gained": sum(1 for e in edge_rows if e["event"] == "gained"),
            "n_lost": sum(1 for e in edge_rows if e["event"] == "lost"),
            "gained_by_type": dict(gained_types),
            "lost_by_type": dict(lost_types),
            "n_gained_cdcl": n_gained_cdcl,
            "n_lost_cdcl": n_lost_cdcl,
            "n_gained_cdse": n_gained_cdse,
            "n_lost_cdse": n_lost_cdse,
        },
        "predictor_lifts": lifts_sorted,
        "cutoffs": {
            "cd_se": cutoffs.cd_se,
            "cd_cl": cutoffs.cd_cl,
            "closable_a": CLOSABLE_A,
            "min_jaccard_near": float(args.min_jaccard_near),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # --- rule candidates ---
    rules: List[str] = []
    rules.append("# Rule candidates from start→final DFT (partial set)\n")
    rules.append(
        f"Analyzed **{n}** isomers under `{root}`.\n"
        f"Hold (exact+near): **{n_hold}/{n}** "
        f"({100 * n_hold / n:.1f}%). "
        f"Exact retained: **{outcomes.get('retained', 0)}**.\n"
    )
    rules.append("## Do not enable yet without review\n")
    rules.append(
        "These are **evidence notes**, not automatic pack changes.\n"
    )
    for item in lifts_sorted:
        lift = item.get("lift")
        n_on = item["n_on"]
        if lift is None or n_on < 5:
            continue
        name = item["feature"]
        p_on = item["p_break_on"]
        p_off = item["p_break_off"]
        if lift >= 1.5:
            rules.append(
                f"- **`{name}`** lift={lift:.2f} "
                f"(break rate on={p_on}, off={p_off}, n_on={n_on}) → "
                f"consider filter correlated with this start feature.\n"
            )
        elif lift <= 0.7:
            rules.append(
                f"- `{name}` lift={lift:.2f} (protective or rare); "
                f"n_on={n_on}.\n"
            )

    # Map features to pack flags
    rules.append("\n## Suggested pack knobs (if lifts hold on full DFT)\n")
    feat_to_flag = {
        "n_closable_terminal_cd2": "`reject_closable_terminal_cd2: true`",
        "n_unsaturated_bridge_candidates": "`require_bridge_maximal: true`",
        "n_mu2_dist2": "optional `mu2_min_cdse_path_length: 4` (forbid 4-ring Cl)",
        "n_cl2se1_dual_terminal": "already have `forbid_mono_se_dual_terminal`",
    }
    for item in lifts_sorted:
        feat = item["feature"]
        if feat not in feat_to_flag:
            continue
        lift = item.get("lift")
        if lift is None or item["n_on"] < 5:
            rules.append(
                f"- {feat_to_flag[feat]} — insufficient or weak "
                f"(lift={lift}, n_on={item['n_on']}).\n"
            )
        elif lift >= 1.5:
            rules.append(
                f"- **Promote review:** {feat_to_flag[feat]} "
                f"(lift={lift:.2f}, n_on={item['n_on']}).\n"
            )
        else:
            rules.append(
                f"- {feat_to_flag[feat]} — lift only {lift:.2f}; "
                f"keep OFF for now.\n"
            )

    if n_gained_cdcl > n_gained_cdse * 2 and n_gained_cdcl > 0:
        rules.append(
            f"\nEdge events: gained Cd–Cl (**{n_gained_cdcl}**) ≫ gained "
            f"Cd–Se (**{n_gained_cdse}**) → breaks are mostly **Cl rebonding** "
            f"(terminal↔bridge), not inorganic skeleton rewrite.\n"
        )
    if n_lost_cdse + n_gained_cdse > 0:
        rules.append(
            f"Cd–Se edge churn: gained={n_gained_cdse}, lost={n_lost_cdse}.\n"
        )

    rules_path = out_dir / "rule_candidates.md"
    rules_path.write_text("".join(rules), encoding="utf-8")

    # --- summary.md ---
    md: List[str] = []
    md.append("# Start → final DFT analysis\n\n")
    md.append(f"Root: `{root}`\n\n")
    md.append(f"- Pairs analyzed: **{n}**\n")
    md.append(
        f"- Hold (retained + near_retained): **{n_hold}** "
        f"({100 * n_hold / n:.1f}%)\n"
    )
    md.append(f"- Mean edge Jaccard: **{mean_jacc:.3f}**\n")
    md.append(f"- Mean Kabsch RMSD: **{mean_rmsd:.3f} Å**\n\n")
    md.append("## Outcomes\n\n")
    md.append("| outcome | count |\n|---|---:|\n")
    for name, cnt in outcomes.most_common():
        md.append(f"| {name} | {cnt} |\n")
    md.append("\n## By (k, p)\n\n")
    for key in sorted(by_kp):
        parts = ", ".join(f"{o}:{c}" for o, c in sorted(by_kp[key].items()))
        md.append(f"- `{key}`: {parts}\n")
    md.append("\n## Edge events\n\n")
    md.append(
        f"- Gained: {summary['edge_events']['n_gained']} "
        f"{dict(gained_types)}\n"
    )
    md.append(
        f"- Lost: {summary['edge_events']['n_lost']} {dict(lost_types)}\n"
    )
    md.append("\n## Predictor lifts (breakdown)\n\n")
    md.append(
        "| feature | n_on | P(break\\|on) | P(break\\|off) | lift |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    for item in lifts_sorted:
        md.append(
            f"| {item['feature']} | {item['n_on']} | "
            f"{item.get('p_break_on')} | {item.get('p_break_off')} | "
            f"{item.get('lift')} |\n"
        )
    md.append("\n## Breakdown examples\n\n")
    for ex in examples[: int(args.max_examples)]:
        md.append(
            f"### {ex['structure_id']} (k={ex['k']}, p={ex['p']}) — "
            f"**{ex['outcome']}**\n\n"
        )
        md.append(
            f"- jaccard={ex['jaccard']}, rmsd={ex['rmsd']} Å, "
            f"gained={ex['gained']}, lost={ex['lost']}, "
            f"Δμ₂={ex['delta_mu2']}\n"
        )
        md.append(f"- top displacements: {', '.join(ex['top_disp'])}\n")
        md.append(f"- start predictors: {ex['predictors']}\n\n")
    md.append("\nSee `rule_candidates.md` for filter proposals.\n")
    md_path = out_dir / "summary.md"
    md_path.write_text("".join(md), encoding="utf-8")

    # stdout
    print(f"Analyzed {n} isomers from {root}")
    print(f"Outcomes: {dict(outcomes)}")
    print(
        f"Hold (exact+near): {n_hold}/{n} "
        f"({100 * n_hold / n:.1f}%)"
        if n
        else "Hold: n/a"
    )
    print(f"Mean Jaccard={mean_jacc:.3f}  mean RMSD={mean_rmsd:.3f} Å")
    print("Top predictor lifts:")
    for item in lifts_sorted[:5]:
        print(
            f"  {item['feature']}: lift={item.get('lift')} "
            f"n_on={item['n_on']} P(break|on)={item.get('p_break_on')}"
        )
    print(f"Wrote {per_path}")
    print(f"Wrote {edge_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {rules_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
