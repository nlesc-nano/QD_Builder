"""Implementation-independent references for nucleation regression guards.

Nothing here may import a private canonicalisation, scoring, or enumeration
helper from :mod:`builder.nucleation`.  The whole point of these references is
to stay valid while those internals are replaced, so they recompute what they
need from ``atoms`` and ``graph`` alone.

Two references live here:

``registry_digest``
    A topological fingerprint of a whole :class:`NucleationResult`.  It captures
    composition, connectivity, coordination, bridge character and the selection
    verdict, and deliberately ignores coordinates, search counters and structure
    ids.  Refactors that only make the search faster must not change it.

``exhaustive_bridge_sets``
    Every feasible terminal-ligand bridge set for one base, of every
    cardinality, with its selection score.  This is the brute-force oracle the
    bridge enumerator is checked against on small systems.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Mapping, Sequence, Tuple

import networkx as nx


# --------------------------------------------------------------------------
# Topological digest
# --------------------------------------------------------------------------


def graph_certificate(graph: nx.Graph, atoms: Sequence[object]) -> str:
    """Return an isomorphism-invariant certificate keyed on element labels.

    Computed here rather than through ``nucleation._graph_hash`` so that
    replacing the module's hashing strategy cannot silently invalidate the
    guard.  Bridge edges are labelled by their ``bridge_mode`` because a
    rhombic bridge and an exact-CIF-site bridge are chemically distinct even
    though both are a single Cd--Cl bond.
    """

    labelled = nx.Graph()
    for atom in atoms:
        labelled.add_node(atom.atom_id, _element=atom.symbol)
    for left, right, data in graph.edges(data=True):
        if data.get("kind") == "surface_bridge":
            label = f"bridge:{data.get('bridge_mode', 'shared_occupied_neighbor')}"
        else:
            label = "chemical"
        labelled.add_edge(left, right, _bond=label)
    return nx.weisfeiler_lehman_graph_hash(
        labelled, node_attr="_element", edge_attr="_bond", iterations=5
    )


def _formula(atoms: Sequence[object]) -> str:
    counts: Dict[str, int] = {}
    for atom in atoms:
        counts[atom.symbol] = counts.get(atom.symbol, 0) + 1
    return "".join(
        f"{symbol}{counts[symbol]}" for symbol in sorted(counts)
    )


def _degree_histogram(graph: nx.Graph, atoms: Sequence[object]) -> str:
    by_symbol: Dict[str, List[int]] = {}
    for atom in atoms:
        by_symbol.setdefault(atom.symbol, []).append(
            int(graph.degree[atom.atom_id])
        )
    return " ".join(
        f"{symbol}{sorted(values, reverse=True)}"
        for symbol, values in sorted(by_symbol.items())
    )


def _bridge_modes(graph: nx.Graph) -> str:
    counts: Dict[str, int] = {}
    for _left, _right, data in graph.edges(data=True):
        if data.get("kind") != "surface_bridge":
            continue
        mode = str(data.get("bridge_mode", "shared_occupied_neighbor"))
        counts[mode] = counts.get(mode, 0) + 1
    return ",".join(f"{mode}={counts[mode]}" for mode in sorted(counts)) or "none"


def record_signature(record) -> Tuple[str, ...]:
    """Topological + chemical signature of one retained/discarded record."""

    return (
        _formula(record.atoms),
        graph_certificate(record.graph, record.atoms),
        _degree_histogram(record.graph, record.atoms),
        f"edges={record.graph.number_of_edges()}",
        _bridge_modes(record.graph),
        f"status={record.selection_status}",
        f"reason={record.selection_reason}",
    )


def registry_digest(
    result, *, retained_only: bool = False
) -> Dict[str, List[Tuple[str, ...]]]:
    """Reduce a NucleationResult to a coordinate-free, counter-free digest.

    Keys are ``"k{k}p{p}:retained"`` / ``":discarded"``.  Values are sorted
    lists of :func:`record_signature`, so the digest is stable under any
    reordering of equivalent structures.

    ``retained_only`` skips the discarded registry.  Use it above k=2, where the
    discarded count is documented to be a *lower bound* -- bases that provably
    cannot win their bin are pruned before they ever become records, so that
    number legitimately shrinks whenever pruning improves and pinning it would
    fight the optimisation rather than guard the science.
    """

    registries = [("retained", result.registry)]
    if not retained_only:
        registries.append(("discarded", result.discarded_registry))

    digest: Dict[str, List[Tuple[str, ...]]] = {}
    for label, registry in registries:
        for k, bins in sorted(registry.items()):
            for p, records in sorted(bins.items()):
                digest[f"k{k}p{p}:{label}"] = sorted(
                    record_signature(record) for record in records
                )
    return digest


def digest_diff(
    expected: Mapping[str, List[Tuple[str, ...]]],
    actual: Mapping[str, List[Tuple[str, ...]]],
) -> List[str]:
    """Return human-readable differences between two digests."""

    lines: List[str] = []
    for key in sorted(set(expected) | set(actual)):
        before = expected.get(key, [])
        after = actual.get(key, [])
        if before == after:
            continue
        lines.append(f"{key}: {len(before)} -> {len(after)}")
        for item in sorted(set(map(tuple, before)) - set(map(tuple, after))):
            lines.append(f"  - {item}")
        for item in sorted(set(map(tuple, after)) - set(map(tuple, before))):
            lines.append(f"  + {item}")
    return lines


# --------------------------------------------------------------------------
# Exhaustive bridge reference
# --------------------------------------------------------------------------


def bridge_candidates(state, model, spec, nucleation_module):
    """Re-derive the bridge opportunities of one base.

    Mirrors ``_latent_bridge_variants`` lines 2483-2559.  ``nucleation_module``
    is passed in rather than imported so the caller controls which build is
    under test; only ``_BridgeCandidate`` and the public vacancy helper are
    used, never the enumerator itself.

    Returns ``(terminal_by_primary, candidates)``.
    """

    rules = {rule.ligand: rule for rule in spec.graph_rules.bridge_rules}
    terminal_by_primary: Dict[int, List[int]] = {}
    candidates = []
    if not rules:
        return terminal_by_primary, candidates

    for primary_atom in state.atoms:
        matching = [r for r in rules.values() if primary_atom.symbol == r.host]
        if not matching:
            continue
        rule = matching[0]
        primary = primary_atom.atom_id
        terminal = sorted(
            neighbor
            for neighbor in state.graph.neighbors(primary)
            if state.atoms[neighbor].symbol == rule.ligand
            and state.graph.degree[neighbor] < spec.graph_rules.max_cn[rule.ligand]
        )
        if not terminal:
            continue
        terminal_by_primary[primary] = terminal
        pair_candidates: Dict[int, int] = {}
        for shared in state.graph.neighbors(primary):
            if state.atoms[shared].symbol != rule.shared_neighbor:
                continue
            for second in state.graph.neighbors(shared):
                if second == primary:
                    continue
                if state.atoms[second].symbol != rule.host:
                    continue
                if state.graph.degree[second] >= spec.graph_rules.max_cn[rule.host]:
                    continue
                pair_candidates.setdefault(second, shared)
        for second, shared in sorted(pair_candidates.items()):
            candidates.append(
                nucleation_module._BridgeCandidate(
                    primary=primary,
                    host=second,
                    rule=rule,
                    mode="shared_occupied_neighbor",
                    shared_neighbor=shared,
                )
            )

    rule_by_host = {rule.host: rule for rule in rules.values()}
    for vacancy in nucleation_module._anion_vacancies_on_cations(state, model, spec):
        if len(vacancy.hosts) < 2:
            continue
        site = tuple(float(value) for value in vacancy.position)
        for first, second in combinations(sorted(vacancy.hosts), 2):
            for primary, host in ((first, second), (second, first)):
                rule = rule_by_host.get(state.atoms[primary].symbol)
                if rule is None or primary not in terminal_by_primary:
                    continue
                if state.atoms[host].symbol != rule.host:
                    continue
                if state.graph.degree[host] >= spec.graph_rules.max_cn[rule.host]:
                    continue
                candidates.append(
                    nucleation_module._BridgeCandidate(
                        primary=primary,
                        host=host,
                        rule=rule,
                        mode="shared_vacant_cif_site",
                        virtual_site=site,
                        virtual_hosts=tuple(sorted(vacancy.hosts)),
                    )
                )
    return terminal_by_primary, candidates


def _preoccupied_host_pairs(state, spec) -> set:
    """Cd pairs already stitched by an existing two-host ligand (line 2576)."""

    ligands = {rule.ligand for rule in spec.graph_rules.bridge_rules}
    hosts_of = {rule.ligand: rule.host for rule in spec.graph_rules.bridge_rules}
    pairs = set()
    for atom in state.atoms:
        if atom.symbol not in ligands:
            continue
        neighbors = [
            neighbor
            for neighbor in state.graph.neighbors(atom.atom_id)
            if state.atoms[neighbor].symbol == hosts_of[atom.symbol]
        ]
        if len(neighbors) == 2:
            pairs.add(tuple(sorted(neighbors)))
    return pairs


def arc_set_is_feasible(state, spec, terminal_by_primary, subset) -> bool:
    """Donor supply, acceptor capacity, one bridge per Cd pair (line 2590)."""

    used_pairs = _preoccupied_host_pairs(state, spec)
    supply: Dict[int, int] = {}
    added: Dict[int, int] = {}
    for candidate in subset:
        primary, host = candidate.primary, candidate.host
        pair = tuple(sorted((primary, host)))
        if pair in used_pairs:
            return False
        if supply.get(primary, 0) >= len(terminal_by_primary[primary]):
            return False
        capacity = spec.graph_rules.max_cn[state.atoms[host].symbol]
        if state.graph.degree[host] + added.get(host, 0) >= capacity:
            return False
        used_pairs.add(pair)
        supply[primary] = supply.get(primary, 0) + 1
        added[host] = added.get(host, 0) + 1
    return True


def build_bridged_graph(state, terminal_by_primary, subset) -> nx.Graph:
    """Materialise the bridged graph exactly as ``build_graph`` does (line 2692)."""

    graph = state.graph.copy()
    by_primary: Dict[int, List] = {}
    for candidate in subset:
        by_primary.setdefault(candidate.primary, []).append(candidate)
    for primary, choices in sorted(by_primary.items()):
        choices.sort(
            key=lambda item: (
                0 if item.mode == "shared_vacant_cif_site" else 1,
                item.host,
                item.shared_neighbor if item.shared_neighbor is not None else -1,
                item.virtual_site or (),
            )
        )
        for ligand, candidate in zip(terminal_by_primary[primary], choices):
            graph.add_edge(
                ligand,
                candidate.host,
                kind="surface_bridge",
                bond_order=1,
                bridge_mode=candidate.mode,
                shared_neighbor=candidate.shared_neighbor,
                virtual_site=candidate.virtual_site,
            )
    return graph


def exhaustive_bridge_sets(state, model, spec, nucleation_module, *, max_candidates=13):
    """Every feasible bridge set of every cardinality, with its score.

    Returns ``None`` when the base has more than ``max_candidates``
    opportunities, since the reference is deliberately brute force.  Otherwise
    returns ``(candidates, [(cardinality, score, subset), ...])``.
    """

    terminal_by_primary, candidates = bridge_candidates(
        state, model, spec, nucleation_module
    )
    if not candidates or len(candidates) > max_candidates:
        return None

    scored = []
    for size in range(len(candidates) + 1):
        for subset in combinations(candidates, size):
            if not arc_set_is_feasible(state, spec, terminal_by_primary, subset):
                continue
            graph = build_bridged_graph(state, terminal_by_primary, subset)
            score = nucleation_module._graph_coordination_score(
                state.atoms, graph, spec
            )
            scored.append((size, score, subset))
    return candidates, scored
