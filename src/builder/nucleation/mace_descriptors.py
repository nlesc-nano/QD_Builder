"""Geometry descriptors calibrated on MACE-MH-1 / PBE relaxed CdSe–CdCl2 clusters.

Why a second descriptor set
---------------------------
``search_analysis.descriptors`` classifies bonds with a single Cd–Se cutoff
(3.25 Å) that was tuned on g-xTB endpoints, whose Cd–Se pair distribution has
a clean gap between 2.9 and 3.2 Å.  MACE-MH-1 and CP2K/PBE endpoints do not:
~10 % of bonded Cd–Se contacts sit in a 2.95–3.35 Å tail and the first RDF
minimum is at 3.32 Å on both (``runs/mace_import/*/REPORT.md``).  A single
cutoff inside that tail makes coordination numbers, tetrahedrality and the
Weisfeiler–Lehman core hash flicker with sub-0.1 Å motions and reports Se
as 5/6-coordinate when it is really 4 strong bonds plus long contacts.

This module therefore uses a **two-tier bond model**:

* *strong* bonds: Cd–Se ≤ 2.95 Å, Cd–Cl ≤ 2.80 Å (covers the main peak,
  p95 of the bonded population on MACE/PBE);
* *all* bonds: Cd–Se ≤ 3.35 Å, Cd–Cl ≤ 3.15 Å (first RDF minimum);
* *long* contacts: the difference.

Scaffold identity, coordination and tetrahedrality are reported on the
strong graph (stable), long contacts are reported separately (they carry
chemistry: they are how PBE packs Se beyond CN4), and the all-bond graph is
kept for bridging (``mu2``/``mu3``) where a 3.0 Å Cd–Cl contact still
bridges.

Cutoff-free order parameters (``q4_nn4`` from the four nearest Cd of every
Se, angle-based ZB scores) are included so that scaffold classification does
not depend on any threshold at all.

Only numpy and networkx are required, so ``tools/`` scripts in the MACE
environment can load this module by path without importing the package.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

# --------------------------------------------------------------------------
# Calibrated cutoffs (Å).  See module docstring and
# docs/nucleation/mace_descriptors.md for the pair-distance
# profiles they were read from.
# --------------------------------------------------------------------------
MACE_CUTOFFS: Dict[str, Dict[Tuple[str, str], float]] = {
    "strong": {("Cd", "Se"): 2.95, ("Cd", "Cl"): 2.80},
    "all": {("Cd", "Se"): 3.35, ("Cd", "Cl"): 3.15},
}
#: Legacy single-tier cutoffs from geometry_packs/cdse_cdcl2_zb/graph_rules.yaml
LEGACY_CUTOFFS: Dict[Tuple[str, str], float] = {("Cd", "Se"): 3.25, ("Cd", "Cl"): 2.90}

TETRAHEDRAL_DEG = math.degrees(math.acos(-1.0 / 3.0))  # 109.47
ANGLE_TOL_DEG = 15.0


def _pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def distance_matrix(positions: np.ndarray) -> np.ndarray:
    pos = np.asarray(positions, float)
    return np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)


def bonded_edges(
    symbols: Sequence[str], dist: np.ndarray, cutoffs: Mapping[Tuple[str, str], float]
) -> List[Tuple[int, int]]:
    edges = []
    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):
            cut = cutoffs.get(_pair(symbols[i], symbols[j]))
            if cut is not None and dist[i, j] <= cut:
                edges.append((i, j))
    return edges


def _graph(symbols: Sequence[str], edges: Iterable[Tuple[int, int]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from((i, {"element": s}) for i, s in enumerate(symbols))
    g.add_edges_from(edges)
    return g


def _core(graph: nx.Graph, symbols: Sequence[str]) -> nx.Graph:
    core = graph.copy()
    core.remove_nodes_from([i for i, s in enumerate(symbols) if s == "Cl"])
    core.remove_edges_from([(a, b) for a, b in list(core.edges) if symbols[a] == symbols[b]])
    return core


def _count_cycles(graph: nx.Graph, length: int, limit: int = 256) -> int:
    """Number of simple cycles of exactly ``length`` (bounded DFS)."""
    if graph.number_of_nodes() < length or graph.number_of_edges() < length:
        return 0
    found = set()
    nbrs = {n: sorted(graph.neighbors(n)) for n in graph.nodes()}

    def dfs(start: int, current: int, path: List[int]) -> None:
        if len(found) >= limit:
            return
        if len(path) == length:
            if start in nbrs[current]:
                found.add(frozenset(path))
            return
        for nxt in nbrs[current]:
            if nxt > start and nxt not in path:
                dfs(start, nxt, path + [nxt])

    for s in sorted(graph.nodes()):
        dfs(s, s, [s])
    return len(found)


def q4(vectors: np.ndarray) -> float:
    """Standard tetrahedral order parameter for exactly four neighbour vectors.

    ``1`` for an ideal tetrahedron, ``0`` for a random arrangement.
    """
    v = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    return 1.0 - 3.0 / 8.0 * sum((float(v[a] @ v[b]) + 1.0 / 3.0) ** 2 for a in range(4) for b in range(a))


def pyramidality(vectors: np.ndarray) -> float:
    """Tetrahedral pyramidality for three neighbours, in [0, 1].

    Uses ``s = |sum of unit neighbour vectors|``: ``s = 1`` when the three
    bonds point along three tetrahedral directions (the missing fourth
    direction is well defined, as for a ZB surface Se or Cd), ``s = 0`` for
    trigonal planar, ``s -> 3`` for an over-acute pyramid.  Returned as
    ``1 - |1 - s|`` clipped to [0, 1] so ideal tetrahedral = 1.
    """
    v = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    s = float(np.linalg.norm(v.sum(axis=0)))
    return max(0.0, min(1.0, 1.0 - abs(1.0 - s)))


def core_hash(graph: nx.Graph, symbols: Sequence[str]) -> str:
    core = _core(graph, symbols)
    return nx.weisfeiler_lehman_graph_hash(core, node_attr="element")


def describe(
    symbols: Sequence[str],
    positions: Sequence[Sequence[float]],
    cutoffs: Mapping[str, Mapping[Tuple[str, str], float]] = MACE_CUTOFFS,
) -> Dict[str, Any]:
    """Two-tier coordination / scaffold descriptors for one cluster."""
    symbols = list(symbols)
    pos = np.asarray(positions, float)
    n = len(symbols)
    dist = distance_matrix(pos)

    strong_edges = bonded_edges(symbols, dist, cutoffs["strong"])
    all_edges = bonded_edges(symbols, dist, cutoffs["all"])
    g_strong = _graph(symbols, strong_edges)
    g_all = _graph(symbols, all_edges)
    core_strong = _core(g_strong, symbols)
    core_all = _core(g_all, symbols)

    cd = [i for i, s in enumerate(symbols) if s == "Cd"]
    se = [i for i, s in enumerate(symbols) if s == "Se"]
    cl = [i for i, s in enumerate(symbols) if s == "Cl"]

    def nb(g: nx.Graph, i: int, elem: str) -> List[int]:
        return [j for j in g[i] if symbols[j] == elem]

    # ---- coordination, strong and all ----
    se_cn_strong = Counter(len(nb(g_strong, i, "Cd")) for i in se)
    se_cn_all = Counter(len(nb(g_all, i, "Cd")) for i in se)
    cd_cn_se_strong = Counter(len(nb(g_strong, i, "Se")) for i in cd)
    cd_cn_se_all = Counter(len(nb(g_all, i, "Se")) for i in cd)
    cd_env_strong = Counter(f"{len(nb(g_strong, i, 'Se'))},{len(nb(g_strong, i, 'Cl'))}" for i in cd)
    cl_cn_all = Counter(len(nb(g_all, i, "Cd")) for i in cl)
    cl_cn_strong = Counter(len(nb(g_strong, i, "Cd")) for i in cl)

    n_long_cdse = len(all_edges) - len(strong_edges) - (
        sum(1 for a, b in all_edges if {symbols[a], symbols[b]} == {"Cd", "Cl"})
        - sum(1 for a, b in strong_edges if {symbols[a], symbols[b]} == {"Cd", "Cl"})
    )
    n_long_cdcl = (len(all_edges) - len(strong_edges)) - n_long_cdse
    se_with_long = sum(1 for i in se if len(nb(g_all, i, "Cd")) > len(nb(g_strong, i, "Cd")))

    # ---- tetrahedrality: q4 for strong-CN4, pyramidality for strong-CN3,
    #      plus cutoff-free q4 from the four nearest counter-ions ----
    q4_se, q4_cd, pyr_se, pyr_cd, q4_nn4_se, q4_nn4_cd = [], [], [], [], [], []
    for i in se:
        ns = nb(g_strong, i, "Cd")
        if len(ns) == 4:
            q4_se.append(q4(pos[ns] - pos[i]))
        elif len(ns) == 3:
            pyr_se.append(pyramidality(pos[ns] - pos[i]))
        near = sorted(cd, key=lambda j: dist[i, j])[:4]
        if len(near) == 4:
            q4_nn4_se.append(q4(pos[near] - pos[i]))
    for i in cd:
        ns = nb(g_strong, i, "Se") + nb(g_strong, i, "Cl")
        if len(ns) == 4:
            q4_cd.append(q4(pos[ns] - pos[i]))
        elif len(ns) == 3:
            pyr_cd.append(pyramidality(pos[ns] - pos[i]))
        near = sorted(se + cl, key=lambda j: dist[i, j])[:4]
        if len(near) == 4:
            q4_nn4_cd.append(q4(pos[near] - pos[i]))

    # ---- angle-based ZB score on the strong core graph ----
    def angles(center_list: List[int], elem: str) -> List[float]:
        out = []
        for i in center_list:
            ns = nb(g_strong, i, elem)
            for a, b in itertools.combinations(ns, 2):
                v1, v2 = pos[a] - pos[i], pos[b] - pos[i]
                c = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                out.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
        return out

    cdsecd = angles(se, "Cd")
    secdse = angles(cd, "Se")

    def frac_tet(a: List[float]) -> Optional[float]:
        if not a:
            return None
        return float(np.mean([abs(x - TETRAHEDRAL_DEG) <= ANGLE_TOL_DEG for x in a]))

    # ---- rings on strong core (ZB: only 6-rings; 4-rings mark non-ZB packing)
    n4 = _count_cycles(core_strong, 4)
    n6 = _count_cycles(core_strong, 6)

    # ---- bridge load per Cd on the all-bond graph ----
    bridge_load = []
    for i in cd:
        bridge_load.append(sum(1 for j in nb(g_all, i, "Cl") if len(nb(g_all, j, "Cd")) >= 2))

    # ---- min pair distances ----
    mind: Dict[str, float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = "-".join(_pair(symbols[i], symbols[j]))
            if dist[i, j] < mind.get(key, math.inf):
                mind[key] = float(dist[i, j])

    core_pts = pos[[i for i in range(n) if symbols[i] != "Cl"]]
    radius = float(np.sqrt(np.mean(np.sum((core_pts - core_pts.mean(0)) ** 2, axis=1)))) if len(core_pts) else 0.0

    def hist(c: Counter) -> Dict[str, int]:
        return {str(k): int(v) for k, v in sorted(c.items())}

    def stat(v: List[float]) -> Dict[str, Optional[float]]:
        v = [x for x in v if not math.isnan(x)]
        if not v:
            return {"mean": None, "min": None, "frac_ge_0.7": None, "n": 0}
        return {
            "mean": float(np.mean(v)),
            "min": float(min(v)),
            "frac_ge_0.7": float(np.mean([x >= 0.7 for x in v])),
            "n": len(v),
        }

    out: Dict[str, Any] = {
        "cutoffs": {tier: {"-".join(k): v for k, v in c.items()} for tier, c in cutoffs.items()},
        # graph certificates
        "core_hash_strong": nx.weisfeiler_lehman_graph_hash(core_strong, node_attr="element"),
        "core_hash_all": nx.weisfeiler_lehman_graph_hash(core_all, node_attr="element"),
        "graph_hash_strong": nx.weisfeiler_lehman_graph_hash(g_strong, node_attr="element"),
        "n_components_all": nx.number_connected_components(g_all) if n else 0,
        "core_components_strong": nx.number_connected_components(core_strong) if len(core_strong) else 0,
        "core_components_all": nx.number_connected_components(core_all) if len(core_all) else 0,
        # components of the all-bond core that contain at least one Se: the
        # number of separate Se scaffolds.  A Cd with no Se neighbour (a shed
        # CdCl2 unit held only by Cl bridges) is counted separately below and
        # does not split the scaffold.
        "se_scaffold_components_all": sum(
            1 for comp in nx.connected_components(core_all) if any(symbols[i] == "Se" for i in comp)
        ) if len(core_all) else 0,
        "cd_detached_all": int(cd_cn_se_all.get(0, 0)),
        "cd_detached_strong": int(cd_cn_se_strong.get(0, 0)),
        "n_cdse_strong": sum(1 for a, b in strong_edges if {symbols[a], symbols[b]} == {"Cd", "Se"}),
        "n_cdse_long": int(n_long_cdse),
        "n_cdcl_long": int(n_long_cdcl),
        "se_with_long_contact": int(se_with_long),
        # coordination
        "se_cn_strong": hist(se_cn_strong),
        "se_cn_all": hist(se_cn_all),
        "cd_cn_se_strong": hist(cd_cn_se_strong),
        "cd_cn_se_all": hist(cd_cn_se_all),
        "cd_environments_strong": {k: int(v) for k, v in sorted(cd_env_strong.items())},
        "cl_cn_all": hist(cl_cn_all),
        "cl_cn_strong": hist(cl_cn_strong),
        "se_cn4_fraction_strong": (se_cn_strong.get(4, 0) / len(se)) if se else None,
        "se_cn34_fraction_strong": ((se_cn_strong.get(3, 0) + se_cn_strong.get(4, 0)) / len(se)) if se else None,
        # bridging (all-bond graph)
        "terminal": int(cl_cn_all.get(1, 0)),
        "mu2": int(cl_cn_all.get(2, 0)),
        "mu3": int(sum(v for k, v in cl_cn_all.items() if k >= 3)),
        "isolated_cl": int(cl_cn_all.get(0, 0)),
        "bridge_load_mean": float(np.mean(bridge_load)) if bridge_load else 0.0,
        "bridge_load_max": int(max(bridge_load)) if bridge_load else 0,
        "bridge_load_var": float(np.var(bridge_load)) if bridge_load else 0.0,
        # tetrahedrality / ZB
        "q4_se_strong": stat(q4_se),
        "q4_cd_strong": stat(q4_cd),
        "pyr_se_strong": stat(pyr_se),
        "pyr_cd_strong": stat(pyr_cd),
        "q4_nn4_se": stat(q4_nn4_se),
        "q4_nn4_cd": stat(q4_nn4_cd),
        "cdsecd_tet_fraction": frac_tet(cdsecd),
        "secdse_tet_fraction": frac_tet(secdse),
        "cdsecd_angle_median": float(np.median(cdsecd)) if cdsecd else None,
        "rings4_strong": int(n4),
        "rings6_strong": int(n6),
        "ring4_fraction": (n4 / (n4 + n6)) if (n4 + n6) else None,
        # geometry
        "radius_A": radius,
        "min_distances": {k: round(v, 3) for k, v in sorted(mind.items())},
    }
    # Composite ZB score in [0,1]: mean of the available angle/order measures.
    parts = [
        out["q4_se_strong"]["mean"],
        out["q4_cd_strong"]["mean"],
        out["pyr_se_strong"]["mean"],
        out["pyr_cd_strong"]["mean"],
        out["cdsecd_tet_fraction"],
        out["secdse_tet_fraction"],
        (1.0 - out["ring4_fraction"]) if out["ring4_fraction"] is not None else None,
    ]
    parts = [max(0.0, min(1.0, float(x))) for x in parts if x is not None]
    out["zb_score"] = float(np.mean(parts)) if parts else None
    return out


def scaffold_change(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    """Classify what a relaxation did to the scaffold, using strong-graph certificates."""
    strong_same = before["core_hash_strong"] == after["core_hash_strong"]
    all_same = before["core_hash_all"] == after["core_hash_all"]
    se_cn_b = Counter({int(k): v for k, v in before["se_cn_strong"].items()})
    se_cn_a = Counter({int(k): v for k, v in after["se_cn_strong"].items()})
    lost_34 = (se_cn_b[3] + se_cn_b[4]) - (se_cn_a[3] + se_cn_a[4])
    if after["se_scaffold_components_all"] > max(1, before["se_scaffold_components_all"]):
        label = "split"
    elif strong_same:
        label = "retained" if all_same else "retained_long_contacts_changed"
    elif lost_34 <= 0 and abs(after["n_cdse_strong"] - before["n_cdse_strong"]) <= 1:
        label = "minor"
    else:
        label = "changed"
    return {
        "label": label,
        "strong_hash_same": strong_same,
        "all_hash_same": all_same,
        "d_n_cdse_strong": after["n_cdse_strong"] - before["n_cdse_strong"],
        "d_n_cdse_long": after["n_cdse_long"] - before["n_cdse_long"],
        "d_se_cn34": -lost_34,
        "d_cd_detached": after["cd_detached_all"] - before["cd_detached_all"],
        "d_zb_score": (after["zb_score"] - before["zb_score"]) if (after["zb_score"] is not None and before["zb_score"] is not None) else None,
    }


def descriptor_digest(desc: Mapping[str, Any]) -> str:
    """Stable digest of the descriptor block for journaling."""
    return hashlib.sha256(json.dumps(desc, sort_keys=True, default=float).encode()).hexdigest()[:16]
