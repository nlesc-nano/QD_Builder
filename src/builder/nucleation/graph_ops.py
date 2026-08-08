from __future__ import annotations

from ..graph_canon import canonical_form, compress_leaves

from .types import *  # private names via __all__

from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .types import AtomRecord, FloatArray, _State, _EnumerationCache

def _count_simple_cycles_of_length(graph: nx.Graph, length: int) -> int:
    """Count undirected simple cycles of exact ``length`` (each cycle once).

    Used only on small nucleation skeletons (tens of vertices).  Each undirected
    cycle is discovered ``2 * length`` times in the directed DFS (``length``
    starts and two directions), so the raw count is divided by that factor.
    """

    if length < 3 or graph.number_of_nodes() < length:
        return 0
    raw = 0
    for start in graph.nodes:
        # path holds the walk exclusive of the return edge to start
        stack: List[Tuple[int, Tuple[int, ...], frozenset]] = [
            (start, (start,), frozenset({start}))
        ]
        while stack:
            current, path, used = stack.pop()
            depth = len(path)
            if depth == length:
                if start in graph[current] and path[1] < path[-1]:
                    # Orientation guard path[1] < path[-1] removes one direction;
                    # remaining multiplicity is ``length`` (one per start).
                    raw += 1
                continue
            for neighbor in graph[current]:
                if neighbor in used:
                    continue
                # Keep start free for the closing edge only at full length.
                if neighbor == start:
                    continue
                stack.append(
                    (neighbor, path + (neighbor,), used | {neighbor})
                )
    # With the orientation guard, each cycle is counted once per start vertex.
    return raw // length


def _new_six_ring_count(
    parent: _State, child: _State, length: int = 6
) -> int:
    """How many new undirected cycles of ``length`` appear after core add.

    Default length 6 is the zincblende chair; pass
    ``spec.inorganic_ring_length`` for other lattices.
    """

    parent_cycles = _count_simple_cycles_of_length(parent.graph, length)
    child_cycles = _count_simple_cycles_of_length(child.graph, length)
    return max(0, child_cycles - parent_cycles)


def _enumerate_simple_cycle_edge_sets(
    graph: nx.Graph, length: int
) -> List[frozenset]:
    """Return each undirected simple cycle of ``length`` as its edge frozenset."""

    if length < 3 or graph.number_of_nodes() < length:
        return []
    edge_sets: List[frozenset] = []
    seen: set[frozenset] = set()
    for start in graph.nodes:
        stack: List[Tuple[int, Tuple[int, ...], frozenset]] = [
            (start, (start,), frozenset({start}))
        ]
        while stack:
            current, path, used = stack.pop()
            if len(path) == length:
                if start in graph[current] and path[1] < path[-1]:
                    edges = []
                    for idx in range(length - 1):
                        a, b = path[idx], path[idx + 1]
                        edges.append((a, b) if a < b else (b, a))
                    a, b = path[-1], start
                    edges.append((a, b) if a < b else (b, a))
                    key = frozenset(edges)
                    if key not in seen:
                        seen.add(key)
                        edge_sets.append(key)
                continue
            for neighbor in graph[current]:
                if neighbor in used or neighbor == start:
                    continue
                stack.append(
                    (neighbor, path + (neighbor,), used | {neighbor})
                )
    return edge_sets


def _fused_chair_metrics(
    graph: nx.Graph, length: int = 6
) -> Tuple[int, int]:
    """Return ``(ring_count, fused_pair_count)`` on an inorganic graph.

    ``fused_pair_count`` is the number of unordered pairs of distinct cycles
    of ``length`` that share at least one edge.  Default length 6 is the
    zincblende chair; pass ``spec.inorganic_ring_length`` otherwise.
    """

    edge_sets = _enumerate_simple_cycle_edge_sets(graph, length)
    n = len(edge_sets)
    if n <= 1:
        return n, 0
    fused = 0
    for i in range(n):
        for j in range(i + 1, n):
            if edge_sets[i] & edge_sets[j]:
                fused += 1
    return n, fused


def _skeleton_ring_metrics(
    state: _State, length: int = 6
) -> Tuple[int, int, int]:
    """``(bonds, preferred_rings, fused_pairs)`` on a ligand-free skeleton."""

    bonds = state.graph.number_of_edges()
    rings, fused = _fused_chair_metrics(state.graph, length)
    return bonds, rings, fused


def _count_cycles_on_graph(
    graph: nx.Graph,
    length: int,
    *,
    ligand_nodes: Optional[set[int]] = None,
) -> Tuple[int, int]:
    """Return ``(total_cycles, cycles_with_any_ligand_node)`` of exact length."""

    if length < 3 or graph.number_of_nodes() < length:
        return 0, 0
    ligand_nodes = ligand_nodes or set()
    raw_total = 0
    raw_cl = 0
    for start in graph.nodes:
        stack: List[Tuple[int, Tuple[int, ...], frozenset, bool]] = [
            (
                start,
                (start,),
                frozenset({start}),
                start in ligand_nodes,
            )
        ]
        while stack:
            current, path, used, has_cl = stack.pop()
            if len(path) == length:
                if start in graph[current] and path[1] < path[-1]:
                    raw_total += 1
                    if has_cl:
                        raw_cl += 1
                continue
            for neighbor in graph[current]:
                if neighbor in used or neighbor == start:
                    continue
                stack.append(
                    (
                        neighbor,
                        path + (neighbor,),
                        used | {neighbor},
                        has_cl or neighbor in ligand_nodes,
                    )
                )
    return raw_total // length, raw_cl // length


def _normalize_atoms(atoms: Sequence[AtomRecord]) -> Tuple[AtomRecord, ...]:
    return tuple(
        AtomRecord(
            atom_id=index,
            symbol=atom.symbol,
            coordinates=tuple(float(x) for x in atom.coordinates),
            role=atom.role,
            unit_id=atom.unit_id,
        )
        for index, atom in enumerate(atoms)
    )


def _without_ligands(
    atoms: Sequence[AtomRecord],
    spec: NucleationSpec,
) -> Tuple[AtomRecord, ...]:
    return _normalize_atoms(
        [atom for atom in atoms if atom.symbol != spec.precursor.ligand]
    )


@lru_cache(maxsize=65536)
def _cached_certificate(
    elements: Tuple[str, ...],
    edges: Tuple[Tuple[int, int, str], ...],
) -> Tuple[object, ...]:
    return canonical_form(list(elements), list(edges)).certificate


def _graph_automorphisms(
    graph: nx.Graph,
    cache: Optional[_EnumerationCache] = None,
) -> Tuple[Tuple[Tuple[int, ...], ...], int]:
    """Return automorphisms as index permutations over ``sorted(graph.nodes)``.

    ``permutation[i]`` is the index of the image of the ``i``-th node.  This is
    the shape the orbit reductions consume, and matches what the previous
    previous ``networkx`` automorphism loop produced.

    Leaves are deliberately *not* compressed here: callers permute concrete atom
    ids, so every original vertex must survive.  Both call sites already avoid
    the factorial blowup by other means -- one works on the ligand-free
    skeleton, the other folds terminal-ligand counts into node labels.
    """

    key = _graph_fingerprint(graph)
    cached = cache.automorphisms.get(key) if cache is not None else None
    if cached is not None:
        return cached, 1
    elements, edges = key
    # Automorphism consumers need concrete permutations of every atom.  VF2 is
    # exact and, on these leaf-free skeleton/environment graphs, avoids the
    # large individualisation tree that a full canonical form can create.
    # (Certificate calls still use the canonical labeller with leaf
    # compression.)
    node_order = sorted(graph.nodes)
    node_index = {node: index for index, node in enumerate(node_order)}
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph,
        graph,
        node_match=nx.algorithms.isomorphism.categorical_node_match(
            "element", ""
        ),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match(
            "bond_order", 1
        ),
    )
    permutations_found = tuple(
        sorted(
            tuple(node_index[mapping[node]] for node in node_order)
            for mapping in matcher.isomorphisms_iter()
        )
    )
    if cache is not None:
        cache.automorphisms[key] = permutations_found
    return permutations_found, 0


def _graph_certificate(graph: nx.Graph) -> Tuple[object, ...]:
    """Return a value equal for two graphs exactly when they are isomorphic.

    Same equivalence relation as the ``networkx`` matcher this replaces -- node
    ``element`` and edge ``bond_order`` -- so no dedup decision changes.  The
    difference is that equivalence classes can now be looked up in a dict
    instead of discovered by a matching search against every member of a hash
    bucket, and that interchangeable terminal ligands are folded away rather
    than permuted.
    """

    return _cached_certificate(*_graph_fingerprint(graph))


def _graphs_isomorphic(left: nx.Graph, right: nx.Graph) -> bool:
    return _graph_certificate(left) == _graph_certificate(right)


def _bridge_mode_by_ligand(graph: nx.Graph) -> Dict[int, str]:
    """Map each bridging ligand to the motif it sits in.

    Bridge character belongs to the *ligand*, not to one of its two bonds.  Once
    a bridge forms, the ligand is bonded symmetrically to both cations -- the
    surface projection places it equidistant between them -- so which cation
    happened to own it during construction is history, not chemistry, and must
    not distinguish two structures.  What does distinguish them is where the
    ligand sits: a rhombic bridge holds it in the cation--anion--cation plane at
    the rule's angle (90 degrees for CdSe/CdCl2), an exact-CIF-site bridge puts
    it on the shared vacant anion site at the tetrahedral angle.
    """

    modes: Dict[int, str] = {}
    for left, right, data in graph.edges(data=True):
        if data.get("kind") != "surface_bridge":
            continue
        mode = str(data.get("bridge_mode", "shared_occupied_neighbor"))
        for node in (left, right):
            if graph.nodes[node].get("role") == "precursor_ligand":
                modes[node] = mode
    return modes


def _graph_fingerprint(
    graph: nx.Graph,
) -> Tuple[Tuple[str, ...], Tuple[Tuple[int, int, str], ...]]:
    """Return the exact labelled adjacency used by graph matching and WL."""

    nodes = sorted(graph.nodes)
    remap = {node: index for index, node in enumerate(nodes)}
    bridge_modes = _bridge_mode_by_ligand(graph)
    elements = tuple(
        (
            f"{graph.nodes[node].get('element', '')}|bridge={bridge_modes[node]}"
            if node in bridge_modes
            else str(graph.nodes[node].get("element", ""))
        )
        for node in nodes
    )
    edges = tuple(
        sorted(
            (
                min(remap[left], remap[right]),
                max(remap[left], remap[right]),
                str(data.get("bond_order", 1)),
            )
            for left, right, data in graph.edges(data=True)
        )
    )
    return elements, edges


@lru_cache(maxsize=65536)
def _cached_graph_hash(
    elements: Tuple[str, ...],
    edges: Tuple[Tuple[int, int, str], ...],
) -> str:
    labelled = nx.Graph()
    for node, element in enumerate(elements):
        labelled.add_node(node, _element=element)
    for left, right, bond_order in edges:
        labelled.add_edge(left, right, _bond_label=bond_order)
    return nx.weisfeiler_lehman_graph_hash(
        labelled,
        node_attr="_element",
        edge_attr="_bond_label",
        iterations=4,
    )


def _graph_hash(graph: nx.Graph) -> str:
    return _cached_graph_hash(*_graph_fingerprint(graph))

__all__ = [
    '_count_simple_cycles_of_length',
    '_new_six_ring_count',
    '_enumerate_simple_cycle_edge_sets',
    '_fused_chair_metrics',
    '_skeleton_ring_metrics',
    '_count_cycles_on_graph',
    '_normalize_atoms',
    '_without_ligands',
    '_cached_certificate',
    '_graph_automorphisms',
    '_graph_certificate',
    '_graphs_isomorphic',
    '_bridge_mode_by_ligand',
    '_graph_fingerprint',
    '_cached_graph_hash',
    '_graph_hash',
]
