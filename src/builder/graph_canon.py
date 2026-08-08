"""Exact canonical labelling for small vertex- and edge-coloured graphs.

The nucleation search compares graphs constantly: deduplicating ligand
embeddings, merging skeletons reached by different growth routes, collapsing
isomorphic bridge variants.  Doing that with pairwise VF2 isomorphism costs a
subgraph-matching search per comparison, and it is at its worst exactly where
these graphs live -- a cation carrying several interchangeable terminal ligands
gives VF2 a factorial number of equivalent leaf matchings to walk.

This module computes a *certificate* instead: a hashable value that is equal for
two graphs if and only if they are isomorphic.  Comparison then costs a dict
lookup, and the pairwise search disappears.  It also returns the automorphism
group, which the orbit reductions need.

Three stages, each exact:

1. **Leaf compression.**  Degree-1 vertices hanging off a higher-degree vertex
   are folded into that vertex's colour as counts.  This is what removes the
   factorial: three chemically identical terminal Cl on one Cd become the label
   ``Cd|leaves=Cl:1x3`` rather than 3! interchangeable vertices.
2. **Colour refinement.**  The 1-dimensional Weisfeiler-Leman fixpoint, which
   separates most vertices for free.
3. **Individualise and refine.**  Where refinement stalls, branch over the first
   non-singleton cell and keep the lexicographically smallest encoding over all
   branches.  The set of branches is isomorphism-invariant, so the minimum is a
   true canonical form, and the branches attaining it give the automorphisms.

Labels are reduced to strings on entry, and the certificate carries those
strings rather than per-graph colour indices -- indices would be assigned
relative to whichever labels a particular graph happens to contain, so two
graphs over different elements could otherwise collide.

Pure Python and dependency-free by design: the graphs here are tens of vertices,
where the constant factor of a C extension does not pay for the packaging cost.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Sequence, Tuple

__all__ = [
    "CanonicalForm",
    "canonical_form",
    "compress_leaves",
    "label_key",
]

Edge = Tuple[int, int, Hashable]
Permutation = Tuple[int, ...]


def label_key(label: Hashable) -> str:
    """Reduce a vertex or edge colour to a canonical string.

    Callers must ensure this is injective over the labels they use, which holds
    for the element symbols and bond orders in this project.  Working in one
    string domain keeps colours mutually comparable for sorting without
    scattering conversions through the algorithm.
    """

    return label if isinstance(label, str) else repr(label)


class CanonicalForm:
    """The canonical encoding of one graph, plus its automorphism group."""

    __slots__ = ("certificate", "automorphisms", "positions", "_orbits")

    def __init__(
        self,
        certificate: Tuple[object, ...],
        automorphisms: Tuple[Permutation, ...],
        positions: Permutation = (),
    ) -> None:
        self.certificate = certificate
        self.automorphisms = automorphisms
        #: ``positions[v]`` is the slot vertex ``v`` takes in the canonical
        #: ordering.  Relabelling the input permutes this map but leaves the
        #: *ordered* graph identical, which is what lets a caller do work in an
        #: order that does not depend on how the input happened to be numbered.
        self.positions = positions
        self._orbits: Optional[Tuple[Tuple[int, ...], ...]] = None

    @property
    def orbits(self) -> Tuple[Tuple[int, ...], ...]:
        """Vertex orbits under the automorphism group, each sorted."""

        if self._orbits is None:
            self._orbits = _orbits_of(self.automorphisms)
        return self._orbits

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CanonicalForm):
            return NotImplemented
        return self.certificate == other.certificate

    def __hash__(self) -> int:
        return hash(self.certificate)

    def __repr__(self) -> str:
        return (
            f"CanonicalForm(vertices={len(self.certificate[0])}, "
            f"automorphisms={len(self.automorphisms)})"
        )


def _orbits_of(
    automorphisms: Sequence[Permutation],
) -> Tuple[Tuple[int, ...], ...]:
    if not automorphisms:
        return ()
    count = len(automorphisms[0])
    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for permutation in automorphisms:
        for vertex, image in enumerate(permutation):
            left, right = find(vertex), find(image)
            if left != right:
                parent[left] = right
    groups: Dict[int, List[int]] = {}
    for vertex in range(count):
        groups.setdefault(find(vertex), []).append(vertex)
    return tuple(sorted(tuple(sorted(group)) for group in groups.values()))


def _adjacency(
    count: int,
    edges: Sequence[Tuple[int, int, str]],
) -> List[List[Tuple[int, str]]]:
    adjacency: List[List[Tuple[int, str]]] = [[] for _ in range(count)]
    for left, right, colour in edges:
        adjacency[left].append((right, colour))
        adjacency[right].append((left, colour))
    return adjacency


def _compress_leaves(
    labels: Sequence[str],
    edges: Sequence[Tuple[int, int, str]],
) -> Tuple[List[str], List[Tuple[int, int, str]]]:
    """Fold interchangeable pendant vertices into their neighbour's label.

    Exact because an isomorphism must map degree-1 vertices to degree-1 vertices
    and preserve colours, so it induces an isomorphism of the compressed graph;
    conversely any isomorphism of the compressed graphs extends by matching
    pendants of equal label and edge colour, of which there are equally many at
    corresponding vertices.

    A vertex is only folded away when its single neighbour has degree above one,
    which keeps an isolated edge -- both endpoints degree 1 -- intact.
    """

    count = len(labels)
    adjacency = _adjacency(count, edges)
    degree = [len(adjacency[vertex]) for vertex in range(count)]
    tally: Dict[int, Dict[Tuple[str, str], int]] = {}
    removed = [False] * count
    for vertex in range(count):
        if degree[vertex] != 1:
            continue
        neighbour, edge_colour = adjacency[vertex][0]
        if degree[neighbour] <= 1:
            continue
        removed[vertex] = True
        counts = tally.setdefault(neighbour, {})
        key = (labels[vertex], edge_colour)
        counts[key] = counts.get(key, 0) + 1

    kept = [vertex for vertex in range(count) if not removed[vertex]]
    index_of = {vertex: position for position, vertex in enumerate(kept)}
    compressed: List[str] = []
    for vertex in kept:
        counts = tally.get(vertex)
        if not counts:
            compressed.append(labels[vertex])
            continue
        folded = ",".join(
            f"{leaf}:{colour}x{number}"
            for (leaf, colour), number in sorted(counts.items())
        )
        compressed.append(f"{labels[vertex]}|leaves={folded}")
    remaining = [
        (index_of[left], index_of[right], colour)
        for left, right, colour in edges
        if left in index_of and right in index_of
    ]
    return compressed, remaining


def compress_leaves(
    node_labels: Sequence[Hashable],
    edges: Sequence[Edge],
) -> Tuple[List[str], List[Tuple[int, int, str]]]:
    """Public wrapper: reduce a graph by folding away interchangeable pendants.

    Isomorphism of the compressed graphs is equivalent to isomorphism of the
    originals (see :func:`_compress_leaves`), so this is a sound preprocessing
    step for *any* exact isomorphism test -- including a VF2 call, which is what
    the nucleation search uses it for.  Matching on the reduced graph is what
    removes the factorial cost of permuting identical terminal ligands.
    """

    labels = [label_key(label) for label in node_labels]
    coloured = [
        (int(left), int(right), label_key(colour)) for left, right, colour in edges
    ]
    return _compress_leaves(labels, coloured)


def _refine(
    colours: List[int],
    adjacency: Sequence[List[Tuple[int, str]]],
) -> List[int]:
    """Run 1-dimensional colour refinement to its fixpoint."""

    count = len(colours)
    while True:
        signatures = [
            (
                colours[vertex],
                tuple(
                    sorted(
                        (edge_colour, colours[neighbour])
                        for neighbour, edge_colour in adjacency[vertex]
                    )
                ),
            )
            for vertex in range(count)
        ]
        ranks = {
            signature: rank
            for rank, signature in enumerate(sorted(set(signatures)))
        }
        refined = [ranks[signature] for signature in signatures]
        if refined == colours:
            return colours
        colours = refined


def canonical_form(
    node_labels: Sequence[Hashable],
    edges: Sequence[Edge],
    *,
    compress_leaves: bool = True,
) -> CanonicalForm:
    """Return the canonical form of a vertex- and edge-coloured graph.

    ``node_labels[v]`` colours vertex ``v``; ``edges`` holds ``(u, v, colour)``
    triples for an undirected graph without self-loops or parallel edges.

    With ``compress_leaves`` the certificate is cheaper, but the automorphisms
    then range over only the surviving vertices.  Callers that must permute
    *every* original vertex -- the orbit reductions do -- pass ``False``.
    """

    labels = [label_key(label) for label in node_labels]
    coloured = [
        (int(left), int(right), label_key(colour)) for left, right, colour in edges
    ]
    if compress_leaves:
        labels, coloured = _compress_leaves(labels, coloured)
    return _canonicalise(labels, coloured)


def _canonicalise(
    labels: Sequence[str],
    edges: Sequence[Tuple[int, int, str]],
) -> CanonicalForm:
    count = len(labels)
    if count == 0:
        return CanonicalForm(((), ()), ((),), ())

    adjacency = _adjacency(count, edges)
    ranks = {label: rank for rank, label in enumerate(sorted(set(labels)))}
    initial = [ranks[label] for label in labels]

    best_code: Optional[Tuple[object, ...]] = None
    best_positions: List[Permutation] = []

    def discrete_code(colours: List[int]) -> Tuple[Tuple[object, ...], List[int]]:
        order = sorted(range(count), key=lambda vertex: colours[vertex])
        position = [0] * count
        for slot, vertex in enumerate(order):
            position[vertex] = slot
        code = (
            # Real label strings, not per-graph colour indices: indices are
            # relative to the labels this graph happens to contain, so they
            # would let graphs over different elements collide.
            tuple(labels[vertex] for vertex in order),
            tuple(
                sorted(
                    (
                        min(position[left], position[right]),
                        max(position[left], position[right]),
                        colour,
                    )
                    for left, right, colour in edges
                )
            ),
        )
        return code, position

    def descend(colours: List[int]) -> None:
        nonlocal best_code
        colours = _refine(colours, adjacency)
        cells: Dict[int, List[int]] = {}
        for vertex, colour in enumerate(colours):
            cells.setdefault(colour, []).append(vertex)
        target: Optional[int] = None
        for colour in sorted(cells):
            if len(cells[colour]) > 1:
                target = colour
                break

        if target is None:
            code, position = discrete_code(colours)
            if best_code is None or code < best_code:
                best_code = code
                best_positions.clear()
                best_positions.append(tuple(position))
            elif code == best_code:
                best_positions.append(tuple(position))
            return

        for vertex in cells[target]:
            child = list(colours)
            # Individualise by pushing this vertex ahead of every other cell.
            # Refinement re-ranks colours, so the sentinel only has to be
            # smaller than every current colour.
            child[vertex] = -1
            descend(child)

    descend(list(initial))
    assert best_code is not None

    # Two orderings that produce the same encoding differ by an automorphism.
    # Composing the first such ordering's inverse with each of the others walks
    # the whole group exactly once.
    reference = best_positions[0]
    inverse_reference = [0] * count
    for vertex, slot in enumerate(reference):
        inverse_reference[slot] = vertex
    automorphisms = tuple(
        sorted(
            tuple(inverse_reference[slot] for slot in positions)
            for positions in best_positions
        )
    )
    return CanonicalForm(best_code, automorphisms, reference)
