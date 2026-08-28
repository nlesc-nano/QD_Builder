"""Zinc-blende motif census and Channel B occupation seeds.

Occupations are CIF Cd–Se cuts.  Motifs (chair 6-rings, adamantane cages,
T1 / T3 supertetrahedra) are scored on that graph, not on the relaxed XYZ.
Channel B injects a named cut when a bin's generated occupations do not
contain it, then the usual decorate / embed / chem path runs.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..nc_types import NucleationSpec
from .types import AtomRecord, _LatticeModel, _State

Edge = Tuple[int, int]


def expected_motifs(k: int, p: int) -> Tuple[str, ...]:
    """Named ZB units that can appear as a subgraph at this (k, p)."""

    x = int(k) + int(p)
    y = int(k)
    out: List[str] = []
    if y >= 1 and x >= 4:
        out.append("T1")
    if y >= 3:
        out.append("chair")
    if y >= 4 and x >= 6:
        out.append("adamantane")
    if y >= 4 and x >= 10:
        out.append("T3")
    if y >= 5 and x >= 9:
        out.append("two_cage")
    if y >= 6 and x >= 4:
        out.append("adamantane_Cd4Se6")
    return tuple(out)


def job_status_label(
    *,
    chemically_ok: bool,
    propagation_eligible: bool,
    topology_status: str,
    violations: Sequence[str] = (),
    error: str = "",
) -> Tuple[str, str]:
    """Return ``(kind, cause)`` for the per-job growth log column.

    kind is ``in-path``, ``off-path``, or ``failed``.
    """

    if propagation_eligible:
        return "in-path", ""
    if chemically_ok:
        topo = str(topology_status or "").lower()
        if topo == "changed":
            return "off-path", "topology_changed"
        if topo == "preserved":
            return "off-path", "unconverged"
        if topo == "unavailable":
            return "off-path", "unavailable"
        return "off-path", topo or "changed"
    return "failed", short_fail_cause(violations, error)


def short_fail_cause(
    violations: Sequence[str] = (), error: str = ""
) -> str:
    """Compress a violation list / g-xTB error to one log token."""

    raw = str(error or "").strip()
    if raw:
        lower = raw.lower()
        if "abnormal" in lower:
            return "gxtb_abort"
        if "gxtb" in lower or "xtb" in lower:
            return "gxtb_failed"
        token = raw.split(":")[0].strip().replace(" ", "_")
        if token:
            return token[:40]
    for value in violations:
        text = str(value)
        if text.startswith("artifact:"):
            parts = text.split(":")
            return "artifact:" + (parts[1] if len(parts) > 1 else "pair")
        if "abnormal" in text.lower():
            return "gxtb_abort"
        return text.split(":")[0][:40]
    return "unknown"


def classify_occupation_motifs(occupation: Any) -> Dict[str, int]:
    """Count ZB units on one occupation's Cd–Se graph."""

    symbols = tuple(occupation.symbols)
    edges = tuple(occupation.core_edges)
    n = len(symbols)
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(
        (min(int(a), int(b)), max(int(a), int(b))) for a, b in edges
    )
    cycles6 = _all_n_cycles(graph, 6)
    pair_c = Counter(
        _classify_pair(a, b) for a, b in combinations(cycles6, 2)
    )
    adams = _find_adamantanes(cycles6, symbols)
    n_cd6se4 = sum(1 for item in adams if item == "Cd6Se4")
    n_cd4se6 = sum(1 for item in adams if item == "Cd4Se6")
    t1 = _count_t1(graph, symbols)
    t3 = _count_t3(graph, symbols, n_cd6se4)
    two = _count_two_cage(adams, cycles6, symbols)
    return {
        "n6": len(cycles6),
        "chair": int(len(cycles6) > 0),
        "cage_3path": int(pair_c.get("cage_3path", 0)),
        "ribbon": int(pair_c.get("single_edge", 0)),
        "adamantane": n_cd6se4,
        "adamantane_Cd4Se6": n_cd4se6,
        "T1": t1,
        "T3": t3,
        "two_cage": two,
    }


def occupation_has_motif(occupation: Any, name: str) -> bool:
    counts = classify_occupation_motifs(occupation)
    if name == "chair":
        return counts["n6"] > 0
    return int(counts.get(name, 0) or 0) > 0


def census_occupations(occupations: Sequence[Any]) -> Dict[str, int]:
    """Unique occupations that carry each motif."""

    totals = Counter()
    n = 0
    seen = set()
    for occ in occupations:
        oid = str(getattr(occ, "occupation_id", "") or id(occ))
        if oid in seen:
            continue
        seen.add(oid)
        n += 1
        counts = classify_occupation_motifs(occ)
        for key, value in counts.items():
            if key == "n6":
                if value:
                    totals["chair"] += 1
                continue
            if value:
                totals[key] += 1
    totals["n_unique"] = n
    return dict(totals)


def motif_status(k: int, p: int, census: Mapping[str, int]) -> Dict[str, str]:
    """``present`` / ``missing`` / ``n/a`` per expected motif."""

    out: Dict[str, str] = {}
    for name in expected_motifs(k, p):
        out[name] = "present" if int(census.get(name, 0) or 0) > 0 else "missing"
    return out


def format_motif_log_line(
    k: int,
    p: int,
    *,
    n_occ: int,
    census: Mapping[str, int],
    inpath: Mapping[str, int],
    injected: Sequence[str] = (),
    n_inpath: int = 0,
    n_offpath: int = 0,
) -> str:
    expect = expected_motifs(k, p)
    status = motif_status(k, p, census)
    missing = [name for name in expect if status.get(name) == "missing"]
    present = [name for name in expect if status.get(name) == "present"]
    flag = "ok" if not missing else "missing:" + ",".join(missing)
    if injected:
        flag = flag + " injected:" + ",".join(injected)
    bits = [
        f"[zb-motifs] k={int(k)} p={int(p)} Cd{int(k)+int(p)}Se{int(k)}",
        f"occ={int(n_occ)}",
        f"inpath={int(n_inpath)} offpath={int(n_offpath)}",
        f"chair={int(census.get('chair', 0))}",
        f"cage={int(census.get('cage_3path', 0))}",
        f"ribbon={int(census.get('ribbon', 0))}",
        f"adam={int(census.get('adamantane', 0))}",
        f"T1={int(census.get('T1', 0))}",
        f"T3={int(census.get('T3', 0))}",
        f"two_cage={int(census.get('two_cage', 0))}",
        f"inpath_adam={int(inpath.get('adamantane', 0))}",
        f"expect={','.join(expect) or '-'}",
        f"present={','.join(present) or '-'}",
        f"STATUS {flag}",
    ]
    return "  ".join(bits)


def inject_missing_motifs(
    occupations: Sequence[Any],
    k: int,
    p: int,
    spec: NucleationSpec,
    model: _LatticeModel,
) -> Tuple[List[Any], List[str]]:
    """Append Channel B cuts for expected motifs absent from ``occupations``."""

    from .molecular_zb_growth import occupation_compactness_key

    existing = list(occupations)
    seen = {
        str(getattr(item, "occupation_id", "") or "")
        for item in existing
        if getattr(item, "occupation_id", "")
    }
    census = census_occupations(existing)
    injected: List[str] = []
    for name in expected_motifs(k, p):
        if int(census.get(name, 0) or 0) > 0:
            continue
        built = build_motif_occupation(name, spec, model, k=k, p=p)
        if built is None:
            continue
        oid = str(built.occupation_id or "")
        if oid and oid in seen:
            continue
        existing.append(built)
        if oid:
            seen.add(oid)
        injected.append(name)
        census = census_occupations(existing)
    existing.sort(
        key=lambda item: (
            0 if str(getattr(item, "parent_id", "")) == "channel_b" else 1,
            occupation_compactness_key(item)
            if getattr(item, "core_edges", None) is not None
            else (0, 0, 0, 0, str(getattr(item, "occupation_id", ""))),
        )
    )
    return existing, injected


def build_motif_occupation(
    name: str,
    spec: NucleationSpec,
    model: _LatticeModel,
    *,
    k: int,
    p: int,
) -> Optional[Any]:
    """Build a named ZB cut and raise/pad it onto ``(k, p)``."""

    native = _native_motif_occupation(name, spec, model)
    if native is None:
        return None
    fitted = _fit_occupation_to_bin(native, spec, model, k=k, p=p)
    if fitted is None:
        return None
    fitted.parent_id = "channel_b"
    fitted.notes = f"channel_b:{name}"
    fitted.shed = 0
    fitted.p_m = max(0, int(p) - int(native.p))
    return fitted


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------


def _canon_cycle(cyc: Sequence[int]) -> Tuple[int, ...]:
    n = len(cyc)
    seq = list(cyc)
    rots = [tuple(seq[i:] + seq[:i]) for i in range(n)]
    rev = list(reversed(seq))
    rots += [tuple(rev[i:] + rev[:i]) for i in range(n)]
    return min(rots)


def _cycle_edges(cyc: Sequence[int]) -> frozenset:
    n = len(cyc)
    return frozenset(
        (min(cyc[i], cyc[(i + 1) % n]), max(cyc[i], cyc[(i + 1) % n]))
        for i in range(n)
    )


def _all_n_cycles(graph: Any, length: int) -> List[Tuple[int, ...]]:
    """Simple cycles of exact ``length``.  Bounded DFS; not nx.simple_cycles."""

    if graph.number_of_nodes() < length or graph.number_of_edges() < length:
        return []
    found = set()
    nodes = sorted(graph.nodes())
    nbrs = {n: sorted(graph.neighbors(n)) for n in nodes}

    def dfs(start: int, current: int, path: List[int], blocked: set) -> None:
        if len(found) >= 64:
            return
        if len(path) == length:
            if start in nbrs[current] and start == min(path):
                found.add(_canon_cycle(path))
            return
        for nxt in nbrs[current]:
            if nxt in blocked:
                continue
            if nxt < start:
                continue
            blocked.add(nxt)
            path.append(nxt)
            dfs(start, nxt, path, blocked)
            path.pop()
            blocked.remove(nxt)

    for start in nodes:
        dfs(start, start, [start], {start})
        if len(found) >= 64:
            break
    return list(found)


def _shared_is_path(shared_edges: Iterable[Edge], shared_verts: Iterable[int]) -> bool:
    import networkx as nx

    edges = list(shared_edges)
    if not edges:
        return False
    graph = nx.Graph()
    graph.add_edges_from(edges)
    graph.add_nodes_from(shared_verts)
    if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
        return False
    deg = [d for _, d in graph.degree()]
    if len(deg) == 2 and sorted(deg) == [1, 1]:
        return True
    return sorted(deg) == [1, 1] + [2] * (len(deg) - 2)


def _classify_pair(c1: Sequence[int], c2: Sequence[int]) -> str:
    v1, v2 = set(c1), set(c2)
    e1, e2 = _cycle_edges(c1), _cycle_edges(c2)
    shared_v, shared_e = v1 & v2, e1 & e2
    if len(shared_e) == 1 and len(shared_v) == 2:
        return "single_edge"
    if (
        len(shared_e) == 2
        and len(shared_v) == 3
        and _shared_is_path(shared_e, shared_v)
    ):
        return "cage_3path"
    return "other"


def _find_adamantanes(
    cycles6: Sequence[Sequence[int]], symbols: Sequence[str]
) -> List[str]:
    """Return ``Cd6Se4`` / ``Cd4Se6`` labels for each unique 10-vertex cage."""

    if len(cycles6) < 4:
        return []
    found: Dict[frozenset, str] = {}
    for comb in combinations(cycles6, 4):
        verts = set().union(*[set(c) for c in comb])
        if len(verts) != 10:
            continue
        edges: set = set()
        for cyc in comb:
            edges |= set(_cycle_edges(cyc))
        if len(edges) != 12:
            continue
        if any(_classify_pair(a, b) != "cage_3path" for a, b in combinations(comb, 2)):
            continue
        import networkx as nx

        graph = nx.Graph()
        graph.add_edges_from(edges)
        deg = {i: graph.degree(i) for i in verts}
        if sum(d == 3 for d in deg.values()) != 4:
            continue
        if sum(d == 2 for d in deg.values()) != 6:
            continue
        bh = [i for i, d in deg.items() if d == 3]
        br = [i for i, d in deg.items() if d == 2]
        if len({symbols[i] for i in bh}) != 1 or len({symbols[i] for i in br}) != 1:
            continue
        if symbols[bh[0]] == symbols[br[0]]:
            continue
        n_cd = sum(1 for i in verts if symbols[i] == "Cd")
        kind = "Cd6Se4" if n_cd == 6 else "Cd4Se6" if n_cd == 4 else ""
        if kind:
            found[frozenset(verts)] = kind
    return list(found.values())


def _count_t1(graph: Any, symbols: Sequence[str]) -> int:
    n = 0
    for i, sym in enumerate(symbols):
        if sym != "Se":
            continue
        cd = sum(1 for nbr in graph.neighbors(i) if symbols[nbr] == "Cd")
        if cd >= 4:
            n += 1
    return n


def _count_t3(graph: Any, symbols: Sequence[str], n_adam: int) -> int:
    if n_adam <= 0:
        return 0
    n_se = sum(1 for s in symbols if s == "Se")
    n_cd = sum(1 for s in symbols if s == "Cd")
    if n_se < 4 or n_cd < 10:
        return 0
    se_cn4 = 0
    for i, sym in enumerate(symbols):
        if sym != "Se":
            continue
        if sum(1 for nbr in graph.neighbors(i) if symbols[nbr] == "Cd") >= 4:
            se_cn4 += 1
    return 1 if se_cn4 >= 4 else 0


def _count_two_cage(
    adams: Sequence[str],
    cycles6: Sequence[Sequence[int]],
    symbols: Sequence[str],
) -> int:
    if sum(1 for item in adams if item == "Cd6Se4") < 1:
        return 0
    n_se = sum(1 for s in symbols if s == "Se")
    n_cd = sum(1 for s in symbols if s == "Cd")
    if n_se < 5 or n_cd < 9:
        return 0
    n_cage_pairs = 0
    for a, b in combinations(cycles6, 2):
        if _classify_pair(a, b) == "cage_3path":
            n_cage_pairs += 1
    return 1 if n_cage_pairs >= 2 and n_se >= 5 else 0


# ---------------------------------------------------------------------------
# Channel B builders
# ---------------------------------------------------------------------------


def _native_motif_occupation(
    name: str, spec: NucleationSpec, model: _LatticeModel
) -> Optional[Any]:
    if name == "T1":
        return _build_t1(spec, model)
    if name == "chair":
        return _build_chair(spec, model)
    if name == "adamantane":
        return _build_adamantane(spec, model, se_bridgeheads=True)
    if name == "adamantane_Cd4Se6":
        return _build_adamantane(spec, model, se_bridgeheads=False)
    if name == "T3":
        return _build_t3(spec, model)
    if name == "two_cage":
        return _build_two_cage(spec, model)
    return None


def _fit_occupation_to_bin(
    occ: Any,
    spec: NucleationSpec,
    model: _LatticeModel,
    *,
    k: int,
    p: int,
) -> Optional[Any]:
    from .molecular_zb_growth import (
        _add_precursor_cd,
        attach_cdse,
        occupation_compactness_key,
    )

    cur = occ
    if int(cur.k) > int(k):
        return None
    while int(cur.k) < int(k):
        kids = attach_cdse(cur, spec, model, cap=16)
        if not kids:
            return None
        kids.sort(key=occupation_compactness_key)
        nxt = kids[0]
        nxt.parent_id = occ.parent_id
        nxt.notes = occ.notes
        cur = nxt
    if int(cur.p) > int(p):
        return None
    if int(cur.p) == int(p):
        cur.parent_id = occ.parent_id
        cur.notes = occ.notes
        return cur
    added = _add_precursor_cd(cur, int(p) - int(cur.p), spec, model, cap=8)
    if not added:
        return None
    added.sort(key=occupation_compactness_key)
    best = added[0]
    best.parent_id = occ.parent_id
    best.notes = occ.notes
    return best


_CLOUD: Dict[int, _State] = {}


def _cloud(spec: NucleationSpec, model: _LatticeModel) -> _State:
    from .molecular_zb_growth import _zb_site_cloud, seed_occupation

    cached = _CLOUD.get(id(model))
    if cached is not None:
        return cached
    seed = seed_occupation(spec, model)
    center = np.mean(np.asarray(seed.coordinates, dtype=float), axis=0)
    radius = float(model.bond_length) * 10.0
    state = _zb_site_cloud(model, spec, center, radius, max_sites=120)
    _CLOUD[id(model)] = state
    return state


def _occupation_from_ids(
    cloud: _State,
    atom_ids: Sequence[int],
    spec: NucleationSpec,
    model: _LatticeModel,
    notes: str,
) -> Optional[Any]:
    from .lattice import _make_core_graph
    from .molecular_zb_growth import occupation_from_state

    want = {int(i) for i in atom_ids}
    picked = [atom for atom in cloud.atoms if int(atom.atom_id) in want]
    if len(picked) != len(want):
        return None
    cation, anion = spec.core.cation, spec.core.anion
    n_se = sum(1 for atom in picked if atom.symbol == anion)
    n_cd = sum(1 for atom in picked if atom.symbol == cation)
    p = n_cd - n_se
    if n_se < 1 or p < 0:
        return None
    atoms = [
        AtomRecord(
            index,
            atom.symbol,
            tuple(float(x) for x in atom.coordinates),
            "core_anion" if atom.symbol == anion else "core_cation",
        )
        for index, atom in enumerate(picked)
    ]
    state = _make_core_graph(atoms, model, spec)
    occ = occupation_from_state(
        state,
        spec,
        model=model,
        k=n_se,
        p=p,
        parent_id="channel_b",
        notes=notes,
    )
    return occ


def _bond_tol(model: _LatticeModel) -> Tuple[float, float, float]:
    bond = float(model.bond_length)
    tol = max(float(model.site_tolerance) * 2.0, 0.25)
    tet = bond * (8.0 / 3.0) ** 0.5
    return bond, tet, tol


def _neighbors_by_distance(
    points: np.ndarray, target: float, tol: float
) -> List[List[int]]:
    n = len(points)
    nbrs: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(float(np.linalg.norm(points[i] - points[j])) - target) <= tol:
                nbrs[i].append(j)
                nbrs[j].append(i)
    return nbrs


def _regular_tetrahedra(nbrs: Sequence[Sequence[int]]) -> List[Tuple[int, ...]]:
    tets = []
    seen = set()
    for i, nbs in enumerate(nbrs):
        if len(nbs) < 3:
            continue
        nset = set(nbs)
        for a, b, c in combinations(nbs, 3):
            if b not in nset or c not in nset:
                continue
            if a not in set(nbrs[b]) or a not in set(nbrs[c]) or b not in set(nbrs[c]):
                continue
            key = tuple(sorted((i, a, b, c)))
            if key not in seen:
                seen.add(key)
                tets.append(key)
    return tets


def _cd_bridging(
    cloud: _State,
    se_atoms: Sequence[AtomRecord],
    spec: NucleationSpec,
    model: _LatticeModel,
) -> Dict[Tuple[int, int], int]:
    """Map unordered Se-atom_id pair -> Cd atom_id sitting on that edge."""

    cation = spec.core.cation
    bond, _tet, tol = _bond_tol(model)
    se_by_id = {int(atom.atom_id): atom for atom in se_atoms}
    cd_atoms = [atom for atom in cloud.atoms if atom.symbol == cation]
    bridges: Dict[Tuple[int, int], int] = {}
    se_ids = [int(atom.atom_id) for atom in se_atoms]
    for cd in cd_atoms:
        cd_xyz = np.asarray(cd.coordinates, dtype=float)
        hit = []
        for sid in se_ids:
            se_xyz = np.asarray(se_by_id[sid].coordinates, dtype=float)
            if abs(float(np.linalg.norm(cd_xyz - se_xyz)) - bond) <= tol:
                hit.append(sid)
        if len(hit) == 2:
            key = (min(hit), max(hit))
            bridges.setdefault(key, int(cd.atom_id))
    return bridges


def _build_t1(spec: NucleationSpec, model: _LatticeModel) -> Optional[Any]:
    from .molecular_zb_growth import lattice_k1_occupation

    occ = lattice_k1_occupation(spec, model, p=3)
    if occ is None:
        return None
    occ.parent_id = "channel_b"
    occ.notes = "channel_b:T1"
    return occ


def _build_chair(spec: NucleationSpec, model: _LatticeModel) -> Optional[Any]:
    cloud = _cloud(spec, model)
    anion, cation = spec.core.anion, spec.core.cation
    seed = next((atom for atom in cloud.atoms if atom.symbol == anion), None)
    if seed is None:
        return None
    origin = np.asarray(seed.coordinates, dtype=float)
    radius = float(model.bond_length) * 3.2
    inorg = [
        atom.atom_id
        for atom in cloud.atoms
        if atom.symbol in {cation, anion}
        and float(np.linalg.norm(np.asarray(atom.coordinates) - origin)) <= radius
    ]
    sub = cloud.graph.subgraph(inorg)
    cycles = _all_n_cycles(sub, 6)
    by_id = {int(atom.atom_id): atom for atom in cloud.atoms}
    for cyc in cycles:
        els = [by_id[i].symbol for i in cyc]
        if els.count(anion) == 3 and els.count(cation) == 3:
            occ = _occupation_from_ids(cloud, cyc, spec, model, "channel_b:chair")
            if occ is not None and occ.k == 3 and occ.p == 0:
                return occ
    return None


def _build_adamantane(
    spec: NucleationSpec,
    model: _LatticeModel,
    *,
    se_bridgeheads: bool,
) -> Optional[Any]:
    cloud = _cloud(spec, model)
    anion = spec.core.anion if se_bridgeheads else spec.core.cation
    cation = spec.core.cation if se_bridgeheads else spec.core.anion
    origin = np.asarray(cloud.atoms[0].coordinates, dtype=float)
    heads = [atom for atom in cloud.atoms if atom.symbol == anion]
    heads.sort(
        key=lambda atom: float(
            np.linalg.norm(np.asarray(atom.coordinates) - origin)
        )
    )
    heads = heads[:40]
    if len(heads) < 4:
        return None
    pts = np.asarray([atom.coordinates for atom in heads], dtype=float)
    _bond, tet, tol = _bond_tol(model)
    nbrs = _neighbors_by_distance(pts, tet, tol)
    tets = _regular_tetrahedra(nbrs)[:24]
    for tet_idx in tets:
        se_atoms = [heads[i] for i in tet_idx]
        centroid = np.mean(
            np.asarray([atom.coordinates for atom in se_atoms], dtype=float),
            axis=0,
        )
        # Skip T1-style tetrahedra that enclose a cation at the centre.
        occupied = False
        bond, _, btol = _bond_tol(model)
        for atom in cloud.atoms:
            if atom.symbol != cation:
                continue
            if float(np.linalg.norm(np.asarray(atom.coordinates) - centroid)) <= btol:
                occupied = True
                break
        if occupied:
            continue
        bridges = _cd_bridging(cloud, se_atoms, spec, model)
        se_ids = [int(atom.atom_id) for atom in se_atoms]
        needed = [
            (min(se_ids[i], se_ids[j]), max(se_ids[i], se_ids[j]))
            for i, j in combinations(range(4), 2)
        ]
        if not all(edge in bridges for edge in needed):
            continue
        cd_ids = [bridges[edge] for edge in needed]
        occ = _occupation_from_ids(
            cloud,
            se_ids + cd_ids,
            spec,
            model,
            "channel_b:adamantane" if se_bridgeheads else "channel_b:adamantane_Cd4Se6",
        )
        if occ is None:
            continue
        if se_bridgeheads and occ.k == 4 and occ.p == 2:
            return occ
        if (not se_bridgeheads) and occ.k == 6 and occ.p == 0:
            return occ
    return None


def _build_t3(spec: NucleationSpec, model: _LatticeModel) -> Optional[Any]:
    adam = _build_adamantane(spec, model, se_bridgeheads=True)
    if adam is None:
        return None
    cloud = _cloud(spec, model)
    bond, _tet, tol = _bond_tol(model)
    anion, cation = spec.core.anion, spec.core.cation
    se_pts = [
        (i, np.asarray(pt, dtype=float))
        for i, (sym, pt) in enumerate(zip(adam.symbols, adam.coordinates))
        if sym == anion
    ]
    used = {
        tuple(np.round(np.asarray(pt, dtype=float) / max(tol, 1e-6)))
        for pt in adam.coordinates
    }
    extra: List[int] = []
    cloud_cd = [atom for atom in cloud.atoms if atom.symbol == cation]
    for _idx, se_xyz in se_pts:
        candidates = []
        for atom in cloud_cd:
            key = tuple(np.round(np.asarray(atom.coordinates) / max(tol, 1e-6)))
            if key in used:
                continue
            dist = float(np.linalg.norm(np.asarray(atom.coordinates) - se_xyz))
            if abs(dist - bond) <= tol:
                candidates.append(atom)
        if not candidates:
            return None
        centroid = np.mean(np.asarray(adam.coordinates, dtype=float), axis=0)
        pick = max(
            candidates,
            key=lambda atom: float(
                np.linalg.norm(np.asarray(atom.coordinates) - centroid)
            ),
        )
        extra.append(int(pick.atom_id))
        used.add(tuple(np.round(np.asarray(pick.coordinates) / max(tol, 1e-6))))
    # Map adam sites back onto the cloud, then add vertex Cd.
    from .molecular_zb_growth import _position_key

    cloud_key = {
        _position_key(np.asarray(atom.coordinates), model.site_tolerance): int(
            atom.atom_id
        )
        for atom in cloud.atoms
    }
    adam_ids = []
    for pt in adam.coordinates:
        key = _position_key(np.asarray(pt), model.site_tolerance)
        if key not in cloud_key:
            return None
        adam_ids.append(cloud_key[key])
    occ = _occupation_from_ids(
        cloud, adam_ids + extra, spec, model, "channel_b:T3"
    )
    if occ is not None and occ.k == 4 and occ.p == 6:
        return occ
    return None


def _build_two_cage(spec: NucleationSpec, model: _LatticeModel) -> Optional[Any]:
    cloud = _cloud(spec, model)
    origin = np.asarray(cloud.atoms[0].coordinates, dtype=float)
    heads = [atom for atom in cloud.atoms if atom.symbol == spec.core.anion]
    heads.sort(
        key=lambda atom: float(
            np.linalg.norm(np.asarray(atom.coordinates) - origin)
        )
    )
    heads = heads[:40]
    if len(heads) < 5:
        return None
    pts = np.asarray([atom.coordinates for atom in heads], dtype=float)
    _bond, tet, tol = _bond_tol(model)
    nbrs = _neighbors_by_distance(pts, tet, tol)
    tets = _regular_tetrahedra(nbrs)[:16]
    for a, b in combinations(tets, 2):
        sa, sb = set(a), set(b)
        shared = sa & sb
        if len(shared) != 3:
            continue
        union = sa | sb
        if len(union) != 5:
            continue
        se_atoms = [heads[i] for i in union]
        bridges = _cd_bridging(cloud, se_atoms, spec, model)
        se_ids = [int(atom.atom_id) for atom in se_atoms]
        needed = []
        for i, j in combinations(range(5), 2):
            d = float(
                np.linalg.norm(
                    np.asarray(se_atoms[i].coordinates)
                    - np.asarray(se_atoms[j].coordinates)
                )
            )
            if abs(d - tet) <= tol:
                needed.append((min(se_ids[i], se_ids[j]), max(se_ids[i], se_ids[j])))
        if len(needed) < 9:
            continue
        if not all(edge in bridges for edge in needed):
            continue
        cd_ids = [bridges[edge] for edge in needed]
        occ = _occupation_from_ids(
            cloud, se_ids + cd_ids, spec, model, "channel_b:two_cage"
        )
        if occ is not None and occ.k == 5 and occ.p == 4:
            return occ
    # Fallback: adamantane + one CdSe (k=5).  Padding to bin p happens in
    # ``_fit_occupation_to_bin``.
    adam = _build_adamantane(spec, model, se_bridgeheads=True)
    if adam is None:
        return None
    from .molecular_zb_growth import attach_cdse, occupation_compactness_key

    kids = attach_cdse(adam, spec, model, cap=12)
    if not kids:
        return None
    kids.sort(key=occupation_compactness_key)
    kid = kids[0]
    kid.parent_id = "channel_b"
    kid.notes = "channel_b:two_cage"
    return kid
