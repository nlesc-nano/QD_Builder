#!/usr/bin/env python3
"""Export open vs closed CdSe-core isomers for DFT comparison.

Definitions
-----------
* **Open**  — fewest inorganic Cd–Se 6-rings (prefer 0).
* **Closed** — most inorganic Cd–Se 6-rings (at least 1 when available).

Coordination ranking usually *retains* compact closed cores, so open candidates
often live in ``discarded_registry``. Generate open maps with::

    discarded_through_k: 3

Pair mode (default)
-------------------
Match open↔closed structures that are **structurally similar** except for
ring closure.  Distance is computed in two stages (fast by default):

1. **cheap** (all open×closed): CN-histogram L1, |Δedges−1|, sorted-id Kabsch RMSD
2. **optional refine** (only top candidates): exact graph-edit distance

Greedy bipartite matching then writes the best pairs::

  ./k003_p002_pairs/
    pair001/
      open_...xyz
      closed_...xyz
      pair.json
    ...
  ./ring_comparison_manifest.json

Usage::

  python scripts/export_ring_comparison.py \\
    --bundle-open  out_open \\
    --bundle-closed out_closed \\
    --k 3 --p 2,3 \\
    --top-pairs 8
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover
    raise SystemExit("networkx is required for export_ring_comparison.py") from exc


# ---------------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------------


def _load_registry(bundle: Path) -> Mapping[str, Any]:
    path = bundle / "registry.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing registry.json under {bundle}")
    return json.loads(path.read_text())


def _bin_records(
    registry: Mapping[str, Any],
    k: int,
    p: int,
    *,
    include_discarded: bool,
) -> List[dict]:
    rows: List[dict] = []
    retained = (
        registry.get("registry", {}).get(str(k), {}).get(str(p), []) or []
    )
    for record in retained:
        item = dict(record)
        item["_pool"] = "retained"
        rows.append(item)
    if include_discarded:
        discarded = (
            registry.get("discarded_registry", {})
            .get(str(k), {})
            .get(str(p), [])
            or []
        )
        for record in discarded:
            item = dict(record)
            item["_pool"] = "discarded"
            rows.append(item)
    return rows


# ---------------------------------------------------------------------------
# Core graph / ring helpers
# ---------------------------------------------------------------------------


def _ligand_symbol(record: Mapping[str, Any]) -> str:
    for atom in record.get("atoms") or []:
        if atom.get("role") == "precursor_ligand":
            return str(atom.get("symbol", "Cl"))
    return "Cl"


def _ligand_node_ids(record: Mapping[str, Any], graph: nx.Graph) -> set[int]:
    ligand = _ligand_symbol(record)
    nodes: set[int] = set()
    for atom in record.get("atoms") or []:
        if atom.get("role") == "precursor_ligand" or atom.get("symbol") == ligand:
            nodes.add(int(atom["id"]))
    for node, data in graph.nodes(data=True):
        if data.get("element") == ligand or data.get("role") == "precursor_ligand":
            nodes.add(int(node))
    return nodes


def _full_graph(record: Mapping[str, Any]) -> nx.Graph:
    payload = record.get("graph")
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"record {record.get('structure_id')} has no graph payload"
        )
    return nx.node_link_graph(payload, edges="edges")


def _inorganic_graph(record: Mapping[str, Any]) -> nx.Graph:
    graph = _full_graph(record)
    graph = graph.copy()
    graph.remove_nodes_from(_ligand_node_ids(record, graph))
    return graph


def _count_six_cycles(graph: nx.Graph) -> int:
    if graph.number_of_nodes() < 6:
        return 0
    raw = 0
    for start in graph.nodes:
        stack = [(start, (start,), frozenset({start}))]
        while stack:
            current, path, used = stack.pop()
            if len(path) == 6:
                if start in graph[current] and path[1] < path[-1]:
                    raw += 1
                continue
            for neighbor in graph[current]:
                if neighbor in used or neighbor == start:
                    continue
                stack.append((neighbor, path + (neighbor,), used | {neighbor}))
    return raw // 6


def _inorganic_six_rings(record: Mapping[str, Any]) -> int:
    meta = record.get("metadata") or {}
    rings = meta.get("rings") or {}
    if "inorganic_six_rings" in rings:
        return int(rings["inorganic_six_rings"])
    inorganic = rings.get("inorganic_rings_by_length") or {}
    if "6" in inorganic:
        return int(inorganic["6"])
    return _count_six_cycles(_inorganic_graph(record))


def _atom_coords(record: Mapping[str, Any]) -> Dict[int, np.ndarray]:
    coords: Dict[int, np.ndarray] = {}
    for atom in record.get("atoms") or []:
        coords[int(atom["id"])] = np.asarray(atom["coordinates"], dtype=float)
    return coords


def _element_of(graph: nx.Graph, node: int, record: Mapping[str, Any]) -> str:
    data = graph.nodes[node]
    if "element" in data:
        return str(data["element"])
    for atom in record.get("atoms") or []:
        if int(atom["id"]) == int(node):
            return str(atom["symbol"])
    return "?"


def _cn_histogram(graph: nx.Graph, record: Mapping[str, Any]) -> Dict[str, Tuple[int, ...]]:
    """Sorted degree multiset per element on the inorganic graph."""

    buckets: Dict[str, List[int]] = {}
    for node in graph.nodes:
        symbol = _element_of(graph, node, record)
        buckets.setdefault(symbol, []).append(int(graph.degree[node]))
    return {
        symbol: tuple(sorted(values, reverse=True))
        for symbol, values in sorted(buckets.items())
    }


def _cn_l1(left: Mapping[str, Tuple[int, ...]], right: Mapping[str, Tuple[int, ...]]) -> int:
    symbols = set(left) | set(right)
    total = 0
    for symbol in symbols:
        a = list(left.get(symbol, ()))
        b = list(right.get(symbol, ()))
        n = max(len(a), len(b))
        a.extend([0] * (n - len(a)))
        b.extend([0] * (n - len(b)))
        total += sum(abs(x - y) for x, y in zip(sorted(a), sorted(b)))
    return total


# ---------------------------------------------------------------------------
# Similarity / pairing (cheap by default)
# ---------------------------------------------------------------------------


def _kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """RMSD after optimal rotation (centroids already removed)."""

    if len(p) == 0:
        return 0.0
    h = p.T @ q
    u, _s, vt = np.linalg.svd(h)
    d = np.linalg.det(vt.T @ u.T)
    sign = np.array([1.0, 1.0, 1.0 if d >= 0 else -1.0])
    r = vt.T @ np.diag(sign) @ u.T
    p_rot = p @ r.T
    return float(np.sqrt(np.mean(np.sum((p_rot - q) ** 2, axis=1))))


def _fast_core_rmsd(
    left: nx.Graph,
    right: nx.Graph,
    record_l: Mapping[str, Any],
    record_r: Mapping[str, Any],
) -> float:
    """O(n) RMSD: pair same-element atoms by sorted atom id (no permutations)."""

    coords_l = _atom_coords(record_l)
    coords_r = _atom_coords(record_r)
    by_el_l: Dict[str, List[int]] = {}
    by_el_r: Dict[str, List[int]] = {}
    for node in left.nodes:
        by_el_l.setdefault(_element_of(left, node, record_l), []).append(int(node))
    for node in right.nodes:
        by_el_r.setdefault(_element_of(right, node, record_r), []).append(int(node))
    if set(by_el_l) != set(by_el_r):
        return float("inf")
    p_ids: List[int] = []
    q_ids: List[int] = []
    for symbol in sorted(by_el_l):
        left_ids = sorted(by_el_l[symbol])
        right_ids = sorted(by_el_r[symbol])
        if len(left_ids) != len(right_ids):
            return float("inf")
        p_ids.extend(left_ids)
        q_ids.extend(right_ids)
    p = np.vstack([coords_l[i] for i in p_ids])
    q = np.vstack([coords_r[i] for i in q_ids])
    p = p - p.mean(axis=0)
    q = q - q.mean(axis=0)
    return _kabsch_rmsd(p, q)


def _graph_edit_distance(
    left: nx.Graph,
    right: nx.Graph,
    record_l: Mapping[str, Any],
    record_r: Mapping[str, Any],
    *,
    timeout: float,
) -> float:
    """Element-respecting GED — only for optional refine of shortlisted pairs."""

    def node_match(n1, n2) -> bool:
        return _element_of(left, n1, record_l) == _element_of(right, n2, record_r)

    try:
        distance = nx.graph_edit_distance(
            left,
            right,
            node_match=node_match,
            timeout=timeout,
        )
    except Exception:
        return float("inf")
    if distance is None:
        return float("inf")
    return float(distance)


def _wl_fingerprint(graph: nx.Graph, record: Mapping[str, Any]) -> str:
    """Fast structural sketch for pre-filtering."""

    labelled = nx.Graph()
    for node in graph.nodes:
        labelled.add_node(node, _el=_element_of(graph, node, record))
    for u, v in graph.edges:
        labelled.add_edge(u, v)
    return nx.weisfeiler_lehman_graph_hash(
        labelled, node_attr="_el", iterations=3
    )


@dataclass
class _CoreFeatures:
    graph: nx.Graph
    cn: Dict[str, Tuple[int, ...]]
    edges: int
    six: int
    wl: str
    n_nodes: int


def _core_features(record: Mapping[str, Any]) -> _CoreFeatures:
    graph = _inorganic_graph(record)
    return _CoreFeatures(
        graph=graph,
        cn=_cn_histogram(graph, record),
        edges=graph.number_of_edges(),
        six=_inorganic_six_rings(record),
        wl=_wl_fingerprint(graph, record),
        n_nodes=graph.number_of_nodes(),
    )


def _cheap_pair_distance(
    open_rec: Mapping[str, Any],
    closed_rec: Mapping[str, Any],
    feat_o: _CoreFeatures,
    feat_c: _CoreFeatures,
    *,
    w_cn: float,
    w_edges: float,
    w_rmsd: float,
    w_wl: float,
) -> Tuple[float, Dict[str, float]]:
    """Fast distance — no GED, no factorial RMSD."""

    if feat_o.n_nodes != feat_c.n_nodes:
        # Different core atom counts cannot be near-isomers at fixed (k,p)
        # unless ligand stripping differed; treat as far.
        return 1.0e6, {"distance": 1.0e6, "reject": 1.0}

    cn = float(_cn_l1(feat_o.cn, feat_c.cn))
    edge_term = float(abs((feat_c.edges - feat_o.edges) - 1))
    wl_term = 0.0 if feat_o.wl == feat_c.wl else 1.0
    rmsd = _fast_core_rmsd(
        feat_o.graph, feat_c.graph, open_rec, closed_rec
    )
    if not math.isfinite(rmsd):
        rmsd = 50.0
    total = w_cn * cn + w_edges * edge_term + w_rmsd * rmsd + w_wl * wl_term
    detail = {
        "distance": total,
        "cn_histogram_l1": cn,
        "abs_edge_diff_minus_1": edge_term,
        "inorganic_edges_open": float(feat_o.edges),
        "inorganic_edges_closed": float(feat_c.edges),
        "core_rmsd_angstrom": rmsd,
        "wl_mismatch": wl_term,
        "inorganic_six_rings_open": float(feat_o.six),
        "inorganic_six_rings_closed": float(feat_c.six),
        "graph_edit_distance": -1.0,  # not computed in cheap pass
    }
    return total, detail


def _select_open_pool(
    records: Sequence[dict],
    *,
    prefer_zero: bool,
    max_tier: int,
) -> Tuple[List[dict], Dict[str, Any]]:
    if not records:
        return [], {"warning": "empty open pool"}
    scored = [(_inorganic_six_rings(r), r) for r in records]
    min_six = min(six for six, _ in scored)
    chosen = [r for six, r in scored if six == min_six]
    # Stable order then cap for pairing speed (still representative).
    chosen.sort(key=lambda r: str(r.get("structure_id", "")))
    capped = False
    if max_tier > 0 and len(chosen) > max_tier:
        chosen = chosen[:max_tier]
        capped = True
    warning = None
    if prefer_zero and min_six > 0:
        warning = (
            f"no zero-CdSe-6-ring structures; using minimum "
            f"inorganic_six_rings={min_six}. Raise discarded_through_k."
        )
    return chosen, {
        "inorganic_six_rings": min_six,
        "pool_size": len(records),
        "tier_size": len(chosen),
        "tier_capped": capped,
        "warning": warning,
    }


def _select_closed_pool(
    records: Sequence[dict],
    *,
    min_rings: int = 1,
    max_tier: int = 0,
) -> Tuple[List[dict], Dict[str, Any]]:
    if not records:
        return [], {"warning": "empty closed pool"}
    scored = [(_inorganic_six_rings(r), r) for r in records]
    max_six = max(six for six, _ in scored)
    chosen = [r for six, r in scored if six == max_six and six >= min_rings]
    if not chosen and max_six >= min_rings:
        chosen = [r for six, r in scored if six >= min_rings]
    warning = None
    if max_six < min_rings:
        warning = (
            f"closed pool max inorganic_six_rings={max_six} "
            f"< required {min_rings}"
        )
        chosen = [r for six, r in scored if six == max_six]
    chosen.sort(key=lambda r: str(r.get("structure_id", "")))
    capped = False
    if max_tier > 0 and len(chosen) > max_tier:
        chosen = chosen[:max_tier]
        capped = True
    return chosen, {
        "inorganic_six_rings": max_six,
        "pool_size": len(records),
        "tier_size": len(chosen),
        "tier_capped": capped,
        "warning": warning,
    }


def _greedy_pairs(
    open_recs: Sequence[dict],
    closed_recs: Sequence[dict],
    *,
    top_pairs: int,
    refine_ged: bool,
    ged_timeout: float,
    refine_multiplier: int,
    weights: Mapping[str, float],
) -> List[Dict[str, Any]]:
    """Two-stage greedy matching: cheap all-pairs, optional GED refine."""

    if not open_recs or not closed_recs:
        return []

    t0 = time.perf_counter()
    feat_o = [_core_features(r) for r in open_recs]
    feat_c = [_core_features(r) for r in closed_recs]
    cheap: List[Tuple[float, int, int, Dict[str, float]]] = []
    for i, o in enumerate(open_recs):
        for j, c in enumerate(closed_recs):
            dist, detail = _cheap_pair_distance(
                o,
                c,
                feat_o[i],
                feat_c[j],
                w_cn=weights["cn"],
                w_edges=weights["edges"],
                w_rmsd=weights["rmsd"],
                w_wl=weights.get("wl", 0.5),
            )
            cheap.append((dist, i, j, detail))
    cheap.sort(key=lambda item: (item[0], item[1], item[2]))
    t_cheap = time.perf_counter() - t0
    print(
        f"  cheap scored {len(open_recs)}×{len(closed_recs)}="
        f"{len(cheap)} pairs in {t_cheap:.2f}s"
    )

    limit = top_pairs if top_pairs > 0 else min(len(open_recs), len(closed_recs))
    # Keep a shortlist larger than final pairs for optional GED re-ranking.
    shortlist_n = (
        min(len(cheap), max(limit * refine_multiplier, limit))
        if refine_ged
        else min(len(cheap), max(limit * 20, limit))  # enough for greedy fill
    )
    shortlist = cheap[:shortlist_n]

    if refine_ged:
        t1 = time.perf_counter()
        refined: List[Tuple[float, int, int, Dict[str, float]]] = []
        for _dist, i, j, detail in shortlist:
            ged = _graph_edit_distance(
                feat_o[i].graph,
                feat_c[j].graph,
                open_recs[i],
                closed_recs[j],
                timeout=ged_timeout,
            )
            if not math.isfinite(ged):
                ged = 10.0 + detail["cn_histogram_l1"] + abs(
                    feat_c[j].edges - feat_o[i].edges
                )
            detail = dict(detail)
            detail["graph_edit_distance"] = ged
            detail["distance_cheap"] = detail["distance"]
            total = (
                weights.get("ged", 1.0) * ged
                + weights["cn"] * detail["cn_histogram_l1"]
                + weights["edges"] * detail["abs_edge_diff_minus_1"]
                + weights["rmsd"] * detail["core_rmsd_angstrom"]
                + weights.get("wl", 0.5) * detail.get("wl_mismatch", 0.0)
            )
            detail["distance"] = total
            refined.append((total, i, j, detail))
        refined.sort(key=lambda item: (item[0], item[1], item[2]))
        shortlist = refined
        print(
            f"  GED refined {len(refined)} candidates in "
            f"{time.perf_counter() - t1:.2f}s"
        )

    used_o: set[int] = set()
    used_c: set[int] = set()
    pairs: List[Dict[str, Any]] = []
    for dist, i, j, detail in shortlist:
        if i in used_o or j in used_c:
            continue
        used_o.add(i)
        used_c.add(j)
        pairs.append(
            {
                "distance": dist,
                "detail": detail,
                "open": open_recs[i],
                "closed": closed_recs[j],
            }
        )
        if len(pairs) >= limit:
            break
    return pairs


# ---------------------------------------------------------------------------
# XYZ export
# ---------------------------------------------------------------------------


def _resolve_xyz(bundle: Path, record: Mapping[str, Any]) -> Path:
    sid = str(record.get("structure_id", "unknown"))
    meta = record.get("metadata") or {}
    rel = meta.get("construction_native_xyz_path")
    pool = str(record.get("_pool", "retained"))
    if rel:
        src = bundle / str(rel)
    else:
        k = int(record["k"])
        p = int(record["p"])
        src = (
            bundle
            / "structures"
            / f"k{k:03d}"
            / f"p{p:03d}"
            / pool
            / f"{sid}_construction_native.xyz"
        )
    if not src.is_file():
        raise FileNotFoundError(f"missing XYZ for {sid}: {src}")
    return src


def _copy_record_xyz(
    bundle: Path,
    record: Mapping[str, Any],
    dest_dir: Path,
    *,
    prefix: str,
) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = _resolve_xyz(bundle, record)
    dest_name = f"{prefix}__{src.name}"
    shutil.copy2(src, dest_dir / dest_name)
    meta = record.get("metadata") or {}
    surface_rel = meta.get("surface_xyz_path")
    if surface_rel:
        surface_src = bundle / str(surface_rel)
        if surface_src.is_file():
            shutil.copy2(
                surface_src,
                dest_dir / f"{prefix}__{surface_src.name}",
            )
    return dest_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bundle-open", type=Path, required=True)
    parser.add_argument("--bundle-closed", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--p", type=str, default="2,3")
    parser.add_argument(
        "--mode",
        choices=("pairs", "pools", "both"),
        default="pairs",
        help="pairs: matched open/closed; pools: full open/closed tiers; both",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=8,
        help="Max open–closed pairs per (k,p) in pair mode (default 8).",
    )
    parser.add_argument(
        "--max-tier",
        type=int,
        default=40,
        help=(
            "Cap open/closed tier size used for pairing (default 40). "
            "0 = use full tier (can be slow for large discarded sets)."
        ),
    )
    parser.add_argument(
        "--max-per-arm",
        type=int,
        default=0,
        help="In pool mode, cap structures per arm (0 = full tier).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Parent directory (default: cwd).",
    )
    parser.add_argument(
        "--refine-ged",
        action="store_true",
        help=(
            "Optional: re-rank a shortlist with graph-edit distance "
            "(much slower; off by default)."
        ),
    )
    parser.add_argument(
        "--ged-timeout",
        type=float,
        default=1.0,
        help="Seconds per GED call when --refine-ged is set (default 1).",
    )
    parser.add_argument(
        "--refine-multiplier",
        type=int,
        default=5,
        help="With --refine-ged, score this many × top-pairs candidates.",
    )
    parser.add_argument(
        "--closed-min-rings",
        type=int,
        default=1,
        help="Minimum inorganic 6-rings required for the closed tier.",
    )
    parser.add_argument(
        "--open-from-retained-only",
        action="store_true",
        help="Do not use discarded_registry for the open pool.",
    )
    args = parser.parse_args(argv)

    open_bundle = args.bundle_open.resolve()
    closed_bundle = args.bundle_closed.resolve()
    open_reg = _load_registry(open_bundle)
    closed_reg = _load_registry(closed_bundle)
    p_values = [int(x.strip()) for x in args.p.split(",") if x.strip()]
    parent = (args.out or Path.cwd()).resolve()
    parent.mkdir(parents=True, exist_ok=True)

    # Cheap-pass weights (GED weight only used with --refine-ged).
    weights = {"ged": 1.0, "cn": 0.5, "edges": 1.0, "rmsd": 0.25, "wl": 0.5}

    manifest: Dict[str, Any] = {
        "k": args.k,
        "p": p_values,
        "mode": args.mode,
        "top_pairs": args.top_pairs,
        "parent": str(parent),
        "pair_distance": {
            "weights": weights,
            "refine_ged": bool(args.refine_ged),
            "ged_timeout_s": args.ged_timeout,
            "description": (
                "default: distance = w_cn*CN_L1 + w_edges*|ΔE-1| + w_rmsd*RMSD "
                "+ w_wl*WL_mismatch (fast). Optional --refine-ged adds GED on a "
                "shortlist only."
            ),
        },
        "bins": {},
        "folders": {},
    }

    for p in p_values:
        open_pool = _bin_records(
            open_reg,
            args.k,
            p,
            include_discarded=not args.open_from_retained_only,
        )
        closed_pool = _bin_records(
            closed_reg, args.k, p, include_discarded=True
        )
        open_tier, open_info = _select_open_pool(
            open_pool, prefer_zero=True, max_tier=args.max_tier
        )
        closed_tier, closed_info = _select_closed_pool(
            closed_pool,
            min_rings=args.closed_min_rings,
            max_tier=args.max_tier,
        )
        bin_key = f"k{args.k:03d}_p{p:03d}"
        if open_info.get("warning"):
            print(f"WARNING {bin_key} open: {open_info['warning']}")
        if closed_info.get("warning"):
            print(f"WARNING {bin_key} closed: {closed_info['warning']}")
        print(
            f"{bin_key}: open_tier={open_info.get('tier_size')} "
            f"(pool={open_info.get('pool_size')}), "
            f"closed_tier={closed_info.get('tier_size')} "
            f"(pool={closed_info.get('pool_size')})"
        )

        bin_entry: Dict[str, Any] = {
            "open_selection": open_info,
            "closed_selection": closed_info,
        }

        # ---- pair mode ----
        if args.mode in {"pairs", "both"}:
            pairs = _greedy_pairs(
                open_tier,
                closed_tier,
                top_pairs=args.top_pairs,
                refine_ged=bool(args.refine_ged),
                ged_timeout=args.ged_timeout,
                refine_multiplier=max(1, int(args.refine_multiplier)),
                weights=weights,
            )
            pairs_root = parent / f"{bin_key}_pairs"
            if pairs_root.exists():
                shutil.rmtree(pairs_root)
            pairs_root.mkdir(parents=True, exist_ok=True)
            pair_rows: List[Dict[str, Any]] = []
            for index, pair in enumerate(pairs, start=1):
                pair_dir = pairs_root / f"pair{index:03d}"
                pair_dir.mkdir(parents=True, exist_ok=True)
                open_name = _copy_record_xyz(
                    open_bundle, pair["open"], pair_dir, prefix="open"
                )
                closed_name = _copy_record_xyz(
                    closed_bundle, pair["closed"], pair_dir, prefix="closed"
                )
                pair_meta = {
                    "pair_index": index,
                    "distance": pair["distance"],
                    "detail": pair["detail"],
                    "open": {
                        "structure_id": pair["open"].get("structure_id"),
                        "pool": pair["open"].get("_pool"),
                        "xyz": open_name,
                        "inorganic_six_rings": _inorganic_six_rings(pair["open"]),
                    },
                    "closed": {
                        "structure_id": pair["closed"].get("structure_id"),
                        "pool": pair["closed"].get("_pool"),
                        "xyz": closed_name,
                        "inorganic_six_rings": _inorganic_six_rings(
                            pair["closed"]
                        ),
                    },
                }
                (pair_dir / "pair.json").write_text(
                    json.dumps(pair_meta, indent=2, sort_keys=True) + "\n"
                )
                pair_rows.append(pair_meta)
                ged = pair["detail"].get("graph_edit_distance", -1.0)
                ged_txt = (
                    f"GED={ged:.1f} "
                    if isinstance(ged, (int, float)) and ged >= 0
                    else ""
                )
                print(
                    f"{bin_key} pair{index:03d}: dist={pair['distance']:.3f} "
                    f"{ged_txt}"
                    f"CN_L1={pair['detail']['cn_histogram_l1']:.0f} "
                    f"RMSD={pair['detail']['core_rmsd_angstrom']:.3f} Å "
                    f"6rings {pair_meta['open']['inorganic_six_rings']}→"
                    f"{pair_meta['closed']['inorganic_six_rings']} "
                    f"-> {pair_dir}"
                )
            bin_entry["pairs"] = pair_rows
            manifest["folders"][pairs_root.name] = {
                "kind": "pairs",
                "k": args.k,
                "p": p,
                "count": len(pair_rows),
                "path": str(pairs_root),
            }
            if not pair_rows:
                print(f"WARNING {bin_key}: no pairs formed")

        # ---- pool dump ----
        if args.mode in {"pools", "both"}:
            for arm, tier, bundle, info in (
                ("open", open_tier, open_bundle, open_info),
                ("closed", closed_tier, closed_bundle, closed_info),
            ):
                recs = list(tier)
                if args.max_per_arm > 0:
                    recs = recs[: args.max_per_arm]
                folder = parent / f"{bin_key}_{arm}"
                if folder.exists():
                    shutil.rmtree(folder)
                folder.mkdir(parents=True, exist_ok=True)
                exported = []
                for record in recs:
                    name = _copy_record_xyz(
                        bundle, record, folder, prefix=str(record.get("_pool"))
                    )
                    exported.append(
                        {
                            "structure_id": record.get("structure_id"),
                            "xyz": name,
                            "inorganic_six_rings": _inorganic_six_rings(record),
                        }
                    )
                bin_entry.setdefault("pools", {})[arm] = exported
                manifest["folders"][folder.name] = {
                    "kind": "pool",
                    "arm": arm,
                    "k": args.k,
                    "p": p,
                    "count": len(exported),
                    "inorganic_six_rings": info.get("inorganic_six_rings"),
                    "path": str(folder),
                }
                print(
                    f"{folder.name}: {len(exported)} structures, "
                    f"inorganic_CdSe_6rings={info.get('inorganic_six_rings')} "
                    f"-> {folder}"
                )

        manifest["bins"][bin_key] = bin_entry

    manifest_path = parent / "ring_comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
