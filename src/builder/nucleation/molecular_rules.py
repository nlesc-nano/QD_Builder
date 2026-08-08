"""Hard graph filters for molecular (and optional lattice) nucleation.

Implements agreed rules H1, H4–H7 as pure checks on a chemical graph, plus
post-embed contact floors for forbidden / non-bonded pairs (Cd–Cd, Se–Se,
Cl–Cl, Se–Cl, …).
Does not depend on DFT analysis scripts.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from ..nc_types import NucleationSpec
from .types import AtomRecord, _State

# Default floors (Å) for pairs that must not approach like a bond.
# Used when NucleationSpec.contact_min_distance is empty.
DEFAULT_CONTACT_MIN_DISTANCE: Dict[str, float] = {
    "Cd-Cd": 3.20,
    "Se-Se": 3.80,
    "Cl-Cl": 2.70,
    "Cl-Se": 2.80,  # Se–Cl is never an allowed edge
}


def pair_key(sym_a: str, sym_b: str) -> str:
    a, b = sorted((str(sym_a), str(sym_b)))
    return f"{a}-{b}"


def allowed_bond_pairs(spec: NucleationSpec) -> set[Tuple[str, str]]:
    """Return canonical allowed pairs from new or legacy graph rules."""

    if spec.graph_rules.pair_rules:
        return {
            tuple(rule.elements)
            for rule in spec.graph_rules.pair_rules.values()
            if rule.bond_allowed
        }
    return {tuple(sorted(pair)) for pair in spec.graph_rules.allowed_bonds}


def resolved_contact_cutoffs(
    spec: Optional[NucleationSpec] = None,
) -> Dict[str, float]:
    """Merge defaults with optional YAML overrides."""

    cutoffs = dict(DEFAULT_CONTACT_MIN_DISTANCE)
    if spec is not None and spec.contact_min_distance:
        for key, value in spec.contact_min_distance.items():
            parts = str(key).replace("_", "-").split("-")
            if len(parts) != 2:
                continue
            cutoffs[pair_key(parts[0], parts[1])] = float(value)
    return cutoffs


def coordinate_contact_violations(
    symbols: Sequence[str],
    coordinates: Sequence[Sequence[float]],
    *,
    bonded_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    cutoffs: Optional[Mapping[str, float]] = None,
    check_bonded: bool = False,
) -> List[str]:
    """Flag pairs closer than a contact floor.

    By default only **non-bonded** pairs are checked (bonded pairs use
    geometry-pack bond lengths and may be shorter).  Forbidden pairs such as
    Se–Cl are never bonded, so they are always checked when a cutoff exists.
    """

    if cutoffs is None:
        cutoffs = DEFAULT_CONTACT_MIN_DISTANCE
    bonded = set()
    if bonded_pairs is not None:
        for a, b in bonded_pairs:
            bonded.add((min(int(a), int(b)), max(int(a), int(b))))
    coords = np.asarray(coordinates, dtype=float)
    n = len(symbols)
    if coords.shape[0] != n:
        return ["contact:coordinate_length_mismatch"]
    violations: List[str] = []

    for i in range(n):
        for j in range(i + 1, n):
            key = pair_key(symbols[i], symbols[j])
            limit = cutoffs.get(key)
            if limit is None:
                continue
            is_bonded = (i, j) in bonded
            if is_bonded and not check_bonded:
                # Allowed chemical bonds may be shorter than contact floors
                # for same-species (should not be bonded) or hetero pairs.
                # Only skip if this pair type is an allowed bond species pair
                # — same-species never skip.
                if symbols[i] != symbols[j]:
                    # Cd–Se, Cd–Cl: skip bonded
                    continue
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            if dist < float(limit):
                violations.append(
                    f"contact:{key}:{i}-{j}:{dist:.3f}<{float(limit):.3f}"
                )
    return violations


def inorganic_component_count(
    state: _State,
    spec: NucleationSpec,
) -> int:
    """Number of connected components in the cation–anion subgraph."""

    cation = {spec.core.cation, spec.precursor.center}
    anion = {spec.core.anion}
    inorganic = [
        atom.atom_id
        for atom in state.atoms
        if atom.symbol in cation or atom.symbol in anion
    ]
    if not inorganic:
        return 0
    sub = state.graph.subgraph(inorganic).copy()
    # Keep only cation–anion edges.
    drop = [
        (u, v)
        for u, v in sub.edges
        if not (
            (
                state.atoms[u].symbol in cation
                and state.atoms[v].symbol in anion
            )
            or (
                state.atoms[u].symbol in anion
                and state.atoms[v].symbol in cation
            )
        )
    ]
    sub.remove_edges_from(drop)
    if sub.number_of_nodes() == 0:
        return 0
    return nx.number_connected_components(sub)


def shortest_cycle_length(graph: nx.Graph) -> Optional[int]:
    """Girth of ``graph``, or ``None`` when it is acyclic.

    A breadth-first sweep from every vertex: the first time a level meets an
    already-seen vertex it closes a cycle, and the shortest such closure over
    all roots is the girth.  Exact, and cheap enough for graphs of this size
    that pulling in a cycle-basis routine would be doing far more work than the
    question needs.
    """

    best: Optional[int] = None
    for root in graph:
        distance = {root: 0}
        parent = {root: None}
        queue = [root]
        while queue:
            node = queue.pop(0)
            for neighbor in graph[node]:
                if neighbor not in distance:
                    distance[neighbor] = distance[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
                elif neighbor != parent[node]:
                    length = distance[node] + distance[neighbor] + 1
                    if best is None or length < best:
                        best = length
        if best is not None and best <= 4:
            break
    return best


def ring_size_violations(
    state: _State,
    spec: NucleationSpec,
) -> List[str]:
    """Flag cycles shorter than ``graph_rules.min_ring_size`` allows.

    A ring that is simply not part of the reference chemistry -- Cd2Se2 is the
    motivating case -- is invisible to every other rule here: its coordination
    numbers, bond pairs and distances are all perfectly legal.  It has to be
    excluded topologically or not at all.

    Each rule names the element pair whose alternating subgraph it constrains,
    so ligands bridging two hosts that share an anion are untouched; those
    Cd-Se-Cd-Cl rings are a real motif and stay governed by
    ``max_shared_ligands_per_host_pair``.
    """

    violations: List[str] = []
    for key, minimum in spec.graph_rules.min_ring_size.items():
        elements = set(key.split("-"))
        nodes = [
            atom.atom_id for atom in state.atoms if atom.symbol in elements
        ]
        subgraph = nx.Graph()
        subgraph.add_nodes_from(nodes)
        subgraph.add_edges_from(
            (left, right)
            for left, right in state.graph.subgraph(nodes).edges
            if state.atoms[left].symbol != state.atoms[right].symbol
        )
        girth = shortest_cycle_length(subgraph)
        if girth is not None and girth < minimum:
            violations.append(f"ring_too_small:{key}:{girth}<{minimum}")
    return violations


def bridge_count_per_host_pair(
    state: _State,
    spec: NucleationSpec,
) -> Dict[Tuple[int, int], int]:
    """Count ligand atoms that bridge each unordered host–host pair.

    Hosts are atoms with symbol ``bridge_rule.host`` (typically Cd). A ligand
    with degree >= 2 whose neighbors include two hosts contributes one to
    that pair (and to every host pair among its host neighbors if μ₃).
    """

    ligand = spec.precursor.ligand
    host_symbols = {rule.host for rule in spec.graph_rules.bridge_rules}
    if not host_symbols:
        # Fallback: precursor center / core cation.
        host_symbols = {spec.core.cation, spec.precursor.center}
    counts: Dict[Tuple[int, int], int] = {}
    for atom in state.atoms:
        if atom.symbol != ligand:
            continue
        hosts = sorted(
            neighbor
            for neighbor in state.graph.neighbors(atom.atom_id)
            if state.atoms[neighbor].symbol in host_symbols
        )
        if len(hosts) < 2:
            continue
        for i, left in enumerate(hosts):
            for right in hosts[i + 1 :]:
                key = (left, right)
                counts[key] = counts.get(key, 0) + 1
    return counts


def max_bridges_per_host_pair(state: _State, spec: NucleationSpec) -> int:
    counts = bridge_count_per_host_pair(state, spec)
    return max(counts.values()) if counts else 0


def _has_cycle_of_length(graph: "nx.Graph", length: int, want: int) -> bool:
    """Whether ``graph`` holds at least ``want`` distinct cycles of ``length``.

    Stops as soon as the quota is met, so a gate costs far less than a full
    census; only a rejected graph pays for the whole search.
    """

    if want <= 0:
        return True
    seen: set = set()
    for start in graph.nodes():
        stack = [(start, [start])]
        while stack:
            current, path = stack.pop()
            if len(path) == length:
                if graph.has_edge(current, start):
                    seen.add(
                        frozenset(
                            frozenset((path[i], path[(i + 1) % length]))
                            for i in range(length)
                        )
                    )
                    if len(seen) >= want:
                        return True
                continue
            for neighbor in graph.neighbors(current):
                # ``start`` is the smallest node of any cycle it reports, so
                # each cycle is discovered from exactly one root.
                if neighbor in path or neighbor < start:
                    continue
                stack.append((neighbor, path + [neighbor]))
    return False


def required_ring_violations(
    state: _State, spec: NucleationSpec
) -> List[str]:
    """Enforce ``graph_rules.required_rings`` -- rings a graph *must* contain.

    ``min_ring_size`` forbids rings that are too small; this is the opposite
    gate, and it is what bounds the combinatorics at large k.  Eight-rings
    (Cd4X4 macrocycles) were the most consistent stability correlate measured
    across k=2 bins, so demanding one from k>=4 removes the open, floppy
    frameworks before they are ever decorated::

        graph_rules:
          required_rings:
            - {size: 8, min_count: 1, from_k: 4}

    ``from_k`` makes the gate apply only at or above that core size, leaving
    the small bins exhaustive.  Rings are counted on the *whole* graph, so
    ligand-containing macrocycles count -- that is what was measured.
    """

    rules = getattr(spec.graph_rules, "required_rings", None) or ()
    if not rules:
        return []
    anion = spec.core.anion
    k = sum(1 for atom in state.atoms if atom.symbol == anion)
    violations: List[str] = []
    for rule in rules:
        size = int(rule.get("size", 0))
        if size < 3:
            continue
        from_k = int(rule.get("from_k", 0))
        if k < from_k:
            continue
        want = int(rule.get("min_count", 1))
        if not _has_cycle_of_length(state.graph, size, want):
            violations.append(f"missing_required_ring:{size}x{want}:k{k}")
    return violations


def molecular_graph_violations(
    state: _State,
    spec: NucleationSpec,
    *,
    require_inorganic_connected: Optional[bool] = None,
    bridges_per_cd_pair: Optional[int] = None,
    enforce_min_cn: Optional[bool] = None,
) -> List[str]:
    """Return human-readable violation codes; empty means legal.

    Optional overrides default to the corresponding ``NucleationSpec`` fields.
    """

    req_conn = (
        spec.require_inorganic_connected
        if require_inorganic_connected is None
        else require_inorganic_connected
    )
    bridge_cap = (
        spec.bridges_per_cd_pair
        if bridges_per_cd_pair is None
        else bridges_per_cd_pair
    )
    do_min = (
        spec.enforce_min_cn if enforce_min_cn is None else enforce_min_cn
    )

    violations: List[str] = []

    allowed = allowed_bond_pairs(spec)
    for left, right in state.graph.edges:
        edge_symbols = tuple(
            sorted((state.atoms[left].symbol, state.atoms[right].symbol))
        )
        if edge_symbols not in allowed:
            violations.append(
                f"forbidden_edge:{edge_symbols[0]}-{edge_symbols[1]}:"
                f"{left}-{right}"
            )

    # H7 + degree vs max_cn always checked when called as full molecular validate.
    for atom in state.atoms:
        degree = state.graph.degree[atom.atom_id]
        max_cn = spec.graph_rules.max_cn.get(atom.symbol)
        min_cn = spec.graph_rules.min_cn.get(atom.symbol, 0)
        if degree < 1:
            violations.append(f"cn0:{atom.symbol}:{atom.atom_id}")
        if max_cn is not None and degree > max_cn:
            violations.append(
                f"max_cn:{atom.symbol}:{atom.atom_id}:{degree}>{max_cn}"
            )
        if do_min and degree < min_cn:
            violations.append(
                f"min_cn:{atom.symbol}:{atom.atom_id}:{degree}<{min_cn}"
            )

    # H1 inorganic connectivity
    if req_conn:
        n_comp = inorganic_component_count(state, spec)
        if n_comp > 1:
            violations.append(f"inorganic_disconnected:{n_comp}")
        elif n_comp == 0 and any(
            atom.symbol in {spec.core.cation, spec.core.anion, spec.precursor.center}
            for atom in state.atoms
        ):
            violations.append("inorganic_empty")

    violations.extend(ring_size_violations(state, spec))
    violations.extend(required_ring_violations(state, spec))

    # H6 bridges per Cd–Cd pair
    if bridge_cap > 0:
        peak = max_bridges_per_host_pair(state, spec)
        if peak > bridge_cap:
            violations.append(
                f"bridges_per_cd_pair:{peak}>{bridge_cap}"
            )

    return violations


def molecular_graph_ok(
    state: _State,
    spec: NucleationSpec,
    **kwargs: object,
) -> bool:
    return not molecular_graph_violations(state, spec, **kwargs)  # type: ignore[arg-type]


def molecular_geometry_ok(
    state: _State,
    coordinates: Sequence[Sequence[float]],
    spec: NucleationSpec,
) -> Tuple[bool, List[str]]:
    """Post-embed contact filter: reject too-close Cd–Cd, Se–Se, Cl–Cl, Se–Cl."""

    symbols = [atom.symbol for atom in state.atoms]
    coords = np.asarray(coordinates, dtype=float)
    count = len(symbols)
    if coords.shape != (count, 3):
        return False, ["geometry:coordinate_shape"]
    if not np.all(np.isfinite(coords)):
        return False, ["geometry:non_finite"]

    bonded = {
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in state.graph.edges
    }
    viol: List[str] = []
    for left in range(count):
        for right in range(left + 1, count):
            distance = float(np.linalg.norm(coords[left] - coords[right]))
            key = pair_key(symbols[left], symbols[right])
            if distance < 0.75:
                viol.append(
                    f"overlap:{key}:{left}-{right}:{distance:.3f}<0.750"
                )
            pair_rule = spec.graph_rules.pair_rules.get(key)
            is_bonded = (left, right) in bonded
            if pair_rule is not None:
                if not pair_rule.bond_allowed:
                    limit = float(pair_rule.min_distance or 0.0)
                    if distance < limit:
                        viol.append(
                            f"contact:{key}:{left}-{right}:"
                            f"{distance:.3f}<{limit:.3f}"
                        )
                else:
                    limit = float(pair_rule.bond_max_distance or 0.0)
                    if is_bonded and distance > limit:
                        viol.append(
                            f"bond_too_long:{key}:{left}-{right}:"
                            f"{distance:.3f}>{limit:.3f}"
                        )
                    elif not is_bonded and distance <= limit:
                        viol.append(
                            f"missing_edge:{key}:{left}-{right}:"
                            f"{distance:.3f}<={limit:.3f}"
                        )

    if not spec.graph_rules.pair_rules:
        viol.extend(
            coordinate_contact_violations(
                symbols,
                coords,
                bonded_pairs=list(bonded),
                cutoffs=resolved_contact_cutoffs(spec),
            )
        )
    return (not viol, viol)
