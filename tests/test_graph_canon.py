"""Tests for the exact canonical-labelling module.

The certificate replaces pairwise VF2 isomorphism throughout the nucleation
search, so it has to be exactly as discriminating as VF2 was -- no more, no
less.  These tests check that against NetworkX on random relabellings and on the
symmetric motifs the chemistry actually produces.
"""

from __future__ import annotations

import itertools
import random

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
import pytest

from builder.graph_canon import canonical_form

NODE_MATCH = nx.algorithms.isomorphism.categorical_node_match("element", "")
EDGE_MATCH = nx.algorithms.isomorphism.categorical_edge_match("bond", "")


def _to_inputs(graph):
    order = sorted(graph.nodes)
    index = {node: position for position, node in enumerate(order)}
    labels = [graph.nodes[node]["element"] for node in order]
    edges = [
        (index[left], index[right], data.get("bond", "1"))
        for left, right, data in graph.edges(data=True)
    ]
    return labels, edges


def _certificate(graph, *, compress_leaves=True):
    labels, edges = _to_inputs(graph)
    return canonical_form(
        labels, edges, compress_leaves=compress_leaves
    ).certificate


def _relabelled(graph, rng):
    order = list(graph.nodes)
    shuffled = order[:]
    rng.shuffle(shuffled)
    return nx.relabel_nodes(graph, dict(zip(order, shuffled)), copy=True)


def _star(centre_element, leaf_element, leaf_count, bond="1"):
    graph = nx.Graph()
    graph.add_node(0, element=centre_element)
    for leaf in range(1, leaf_count + 1):
        graph.add_node(leaf, element=leaf_element)
        graph.add_edge(0, leaf, bond=bond)
    return graph


def _ring(elements, bond="1"):
    graph = nx.Graph()
    for index, element in enumerate(elements):
        graph.add_node(index, element=element)
    for index in range(len(elements)):
        graph.add_edge(index, (index + 1) % len(elements), bond=bond)
    return graph


def _cdse_like():
    """A branched Cd/Se skeleton with terminal Cl, as the search produces."""

    graph = nx.Graph()
    for node, element in enumerate(["Cd", "Se", "Cd", "Se", "Cd"]):
        graph.add_node(node, element=element)
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)], bond="1")
    ligand = 5
    for host, count in ((0, 3), (2, 1), (4, 3)):
        for _ in range(count):
            graph.add_node(ligand, element="Cl")
            graph.add_edge(host, ligand, bond="1")
            ligand += 1
    return graph


GRAPHS = {
    "empty": nx.Graph(),
    "single": _star("Cd", "Cl", 0),
    "isolated_edge": _star("Cd", "Cl", 1),
    "cd_cl3": _star("Cd", "Cl", 3),
    "cd_cl4": _star("Cd", "Cl", 4),
    "ring4_alternating": _ring(["Cd", "Se", "Cd", "Se"]),
    "ring4_uniform": _ring(["Cd", "Cd", "Cd", "Cd"]),
    "ring6_alternating": _ring(["Cd", "Se", "Cd", "Se", "Cd", "Se"]),
    "ring3": _ring(["Cd", "Se", "Cd"]),
    "cdse_like": _cdse_like(),
}


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_certificate_is_invariant_under_relabelling(name) -> None:
    graph = GRAPHS[name]
    if graph.number_of_nodes() == 0:
        pytest.skip("relabelling an empty graph is vacuous")
    rng = random.Random(20260726)
    reference = _certificate(graph)
    for _ in range(60):
        assert _certificate(_relabelled(graph, rng)) == reference


def test_large_refinement_cell_is_invariant_under_relabelling() -> None:
    """Regression: selecting one branch in a cell larger than eight was unsafe."""

    graph = nx.Graph()
    graph.add_nodes_from((node, {"element": "X"}) for node in range(10))
    graph.add_edges_from(
        [
            (0, 1), (0, 4), (0, 7), (1, 2), (1, 5),
            (2, 3), (2, 8), (3, 4), (3, 9), (4, 7),
            (5, 6), (5, 9), (6, 8), (6, 9), (7, 8),
        ],
        bond="1",
    )
    mapping = {0: 3, 1: 4, 2: 0, 3: 8, 4: 2, 5: 1, 6: 5, 7: 9, 8: 7, 9: 6}
    relabelled = nx.relabel_nodes(graph, mapping, copy=True)
    assert _certificate(relabelled) == _certificate(graph)


@pytest.mark.parametrize("compress", [True, False])
def test_certificate_agrees_with_vf2_on_every_pair(compress) -> None:
    """Certificate equality must mean exactly what VF2 isomorphism meant."""

    names = sorted(GRAPHS)
    for left_name, right_name in itertools.combinations_with_replacement(names, 2):
        left, right = GRAPHS[left_name], GRAPHS[right_name]
        by_certificate = _certificate(left, compress_leaves=compress) == _certificate(
            right, compress_leaves=compress
        )
        by_vf2 = GraphMatcher(
            left, right, node_match=NODE_MATCH, edge_match=EDGE_MATCH
        ).is_isomorphic()
        assert by_certificate == by_vf2, (
            f"{left_name} vs {right_name}: certificate={by_certificate} vf2={by_vf2}"
        )


def test_edge_colour_is_discriminating() -> None:
    """Two graphs differing only in a bond label are not isomorphic.

    This is the property nucleation needs in order to stop merging a rhombic
    bridge with an exact-CIF-site bridge.
    """

    left = _ring(["Cd", "Se", "Cd", "Se"])
    right = _ring(["Cd", "Se", "Cd", "Se"])
    right[0][1]["bond"] = "bridge"
    assert _certificate(left) != _certificate(right)


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_automorphisms_are_genuine_and_complete(name) -> None:
    """Every returned permutation is an automorphism, and none are missing.

    Without leaf compression the group must match VF2's enumeration exactly,
    which also pins down the composition direction used to build it.
    """

    graph = GRAPHS[name]
    if graph.number_of_nodes() == 0:
        pytest.skip("no automorphisms of an empty graph to compare")
    labels, edges = _to_inputs(graph)
    form = canonical_form(labels, edges, compress_leaves=False)
    edge_set = {
        (min(left, right), max(left, right)): colour
        for left, right, colour in edges
    }

    for permutation in form.automorphisms:
        assert sorted(permutation) == list(range(len(labels))), "not a bijection"
        for vertex, image in enumerate(permutation):
            assert labels[vertex] == labels[image], "colour not preserved"
        mapped = {
            (
                min(permutation[left], permutation[right]),
                max(permutation[left], permutation[right]),
            ): colour
            for (left, right), colour in edge_set.items()
        }
        assert mapped == edge_set, "adjacency or bond colour not preserved"

    expected = sum(
        1
        for _ in GraphMatcher(
            graph, graph, node_match=NODE_MATCH, edge_match=EDGE_MATCH
        ).isomorphisms_iter()
    )
    assert len(form.automorphisms) == expected


def test_leaf_compression_removes_the_factorial_blowup() -> None:
    """A hub with many identical pendants must not cost factorial time.

    ``Cd`` with 8 interchangeable ``Cl`` has 8! = 40320 automorphisms.  The
    compressed certificate collapses them, so the search tree stays trivial
    while the certificate still separates 8 pendants from 7.
    """

    eight = _star("Cd", "Cl", 8)
    seven = _star("Cd", "Cl", 7)
    assert _certificate(eight) != _certificate(seven)
    compressed = canonical_form(*_to_inputs(eight), compress_leaves=True)
    assert len(compressed.certificate[0]) == 1
    assert len(compressed.automorphisms) == 1
    uncompressed = canonical_form(*_to_inputs(eight), compress_leaves=False)
    assert len(uncompressed.automorphisms) == 40320


def test_pendants_on_different_hosts_are_distinguished() -> None:
    """Compression must not lose *where* the pendants are attached."""

    left = nx.Graph()
    left.add_nodes_from([(0, {"element": "Cd"}), (1, {"element": "Cd"})])
    left.add_edge(0, 1, bond="1")
    right = left.copy()
    node = 2
    for host, count in ((0, 3), (1, 1)):
        for _ in range(count):
            left.add_node(node, element="Cl")
            left.add_edge(host, node, bond="1")
            node += 1
    node = 2
    for host, count in ((0, 2), (1, 2)):
        for _ in range(count):
            right.add_node(node, element="Cl")
            right.add_edge(host, node, bond="1")
            node += 1
    assert _certificate(left) != _certificate(right)


def test_orbits_group_equivalent_vertices() -> None:
    ring = _ring(["Cd", "Cd", "Cd", "Cd"])
    form = canonical_form(*_to_inputs(ring), compress_leaves=False)
    assert form.orbits == ((0, 1, 2, 3),)

    chain = nx.Graph()
    for node, element in enumerate(["Cl", "Cd", "Se", "Cd", "Cl"]):
        chain.add_node(node, element=element)
    chain.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)], bond="1")
    form = canonical_form(*_to_inputs(chain), compress_leaves=False)
    assert form.orbits == ((0, 4), (1, 3), (2,))
