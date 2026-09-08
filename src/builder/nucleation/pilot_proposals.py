"""Bounded, charge-conserving proposals. No electronic structure calls here."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from itertools import combinations
from collections import Counter
import networkx as nx
import numpy as np

from .search_analysis import digest, descriptors
from .types import AtomRecord, _State


@dataclass
class Proposal:
    k: int
    p: int
    symbols: list
    positions: list
    edges: list
    parents: list
    channel: str
    occupation: dict | None = None
    audit_derived: bool = False
    attempt: int = 0
    origin_id: str = ""

    @property
    def id(self):
        return (
            self.origin_id
            or "proposal_"
            + digest(
                [
                    self.k,
                    self.p,
                    self.symbols,
                    np.round(self.positions, 7).tolist(),
                    sorted(self.edges),
                    self.channel,
                ]
            )[:24]
        )

    def payload(self):
        return dict(
            id=self.id, symbols=self.symbols, positions=self.positions, edges=self.edges
        )

    def record(self):
        return asdict(self)


def validate_composition(proposal):
    return Counter(proposal.symbols) == Counter(
        Cd=proposal.k + proposal.p, Se=proposal.k, Cl=2 * proposal.p
    )


def graph_state(symbols, edges, xyz=None):
    if xyz is None:
        xyz = np.zeros((len(symbols), 3))
    graph = nx.Graph()
    graph.add_nodes_from(range(len(symbols)))
    graph.add_edges_from(edges)
    return _State(
        atoms=[
            AtomRecord(i, s, tuple(xyz[i]), "search") for i, s in enumerate(symbols)
        ],
        graph=graph,
    )


def bounded_shells(
    symbols,
    core_edges,
    p,
    rng,
    coordinates=None,
    *,
    total_nodes=10000,
    tier_nodes=2000,
    tier_emissions=40,
):
    """Sample bridge b-matchings, then bounded terminal allocations.

    The largest *discovered feasible* tier is distinguished from proof of
    optimality. Limits count failed nodes too, including unproductive tiers.
    """
    cd = [i for i, s in enumerate(symbols) if s == "Cd"]
    degree = Counter(i for edge in core_edges for i in edge)
    room = {i: max(0, 4 - degree[i]) for i in cd}
    pairs = [
        (a, b)
        for a, b in combinations(cd, 2)
        if room[a]
        and room[b]
        and (
            coordinates is None
            or np.linalg.norm(np.asarray(coordinates[a]) - coordinates[b]) <= 4.75
        )
    ]
    rng.shuffle(pairs)
    maximum = min(2 * p, sum(room.values()) // 2, len(pairs))
    shells = []
    used = 0
    productive = 0
    known = set()
    symmetry_buckets = {}
    all_symbols = list(symbols) + ["Cl"] * (2 * p)
    for target in range(maximum, -1, -1):
        nodes = 0
        emitted = 0

        def walk(start, selected):
            nonlocal nodes, used, emitted
            if used >= total_nodes or nodes >= tier_nodes or emitted >= tier_emissions:
                return
            nodes += 1
            used += 1
            if len(selected) == target:
                for _ in range(4):
                    slots = dict(room)
                    hosts = []
                    for _ in range(2 * p - target):
                        possible = [i for i in cd if slots[i] > 0]
                        if not possible:
                            break
                        # Prefer to repair CN1 hosts, but do not enumerate every allocation.
                        deficit = [i for i in possible if 4 - slots[i] < 2]
                        host = int(rng.choice(deficit or possible))
                        hosts.append(host)
                        slots[host] -= 1
                    if len(hosts) != 2 * p - target:
                        continue
                    edges = list(core_edges)
                    for n, (a, b) in enumerate(selected):
                        edges.extend([(a, len(symbols) + n), (b, len(symbols) + n)])
                    for n, a in enumerate(hosts, target):
                        edges.append((a, len(symbols) + n))
                    state = graph_state(all_symbols, edges)
                    if any(state.graph.degree(i) < 2 for i in range(len(symbols))):
                        continue
                    key = tuple(sorted((min(a, b), max(a, b)) for a, b in edges))
                    if key in known:
                        continue
                    symmetry = state.graph.copy()
                    nx.set_node_attributes(
                        symmetry, dict(enumerate(all_symbols)), "element"
                    )
                    nx.set_edge_attributes(symmetry, "bond", "kind")
                    if coordinates is not None:
                        for a, b in combinations(range(len(symbols)), 2):
                            distance = round(
                                float(
                                    np.linalg.norm(
                                        np.asarray(coordinates[a]) - coordinates[b]
                                    )
                                ),
                                5,
                            )
                            symmetry.add_edge(
                                a,
                                b,
                                kind=f"core:{distance}:{state.graph.has_edge(a,b)}",
                            )
                    signature = nx.weisfeiler_lehman_graph_hash(
                        symmetry, node_attr="element", edge_attr="kind"
                    )
                    if any(
                        nx.is_isomorphic(
                            symmetry,
                            other,
                            node_match=lambda a, b: a["element"] == b["element"],
                            edge_match=lambda a, b: a["kind"] == b["kind"],
                        )
                        for other in symmetry_buckets.get(signature, [])
                    ):
                        continue
                    symmetry_buckets.setdefault(signature, []).append(symmetry)
                    known.add(key)
                    shells.append((target, state))
                    emitted += 1
                    if emitted >= tier_emissions:
                        break
                return
            need = target - len(selected)
            for index in range(start, len(pairs) - need + 1):
                if (
                    used >= total_nodes
                    or nodes >= tier_nodes
                    or emitted >= tier_emissions
                ):
                    break
                a, b = pairs[index]
                if room[a] and room[b]:
                    room[a] -= 1
                    room[b] -= 1
                    walk(index + 1, selected + [(a, b)])
                    room[a] += 1
                    room[b] += 1

        before = len(shells)
        walk(0, [])
        if len(shells) > before:
            productive += 1
        if productive >= 3 or used >= total_nodes:
            break
    # Exploratory shells: drop a bridge arm, or promote a bridge to mu3.
    exploratory = []
    for _, state in shells[:40]:
        graph = state.graph.copy()
        bridges = [
            i for i, s in enumerate(all_symbols) if s == "Cl" and graph.degree(i) == 2
        ]
        if not bridges:
            continue
        cl = int(rng.choice(bridges))
        host = int(rng.choice(list(graph[cl])))
        if rng.random() < 0.5 and graph.degree(host) > 2:
            graph.remove_edge(host, cl)
        else:
            available = [i for i in cd if i not in graph[cl] and graph.degree(i) < 4]
            if not available:
                continue
            graph.add_edge(int(rng.choice(available)), cl)
        exploratory.append(_State(atoms=state.atoms, graph=graph))
    return (
        [s for _, s in shells],
        exploratory,
        dict(
            nodes=used,
            exhausted=used >= total_nodes,
            feasible_tiers=productive,
            upper_bound=maximum,
        ),
    )


def choose_shells(primary, exploratory, rng, limit=3, prior=0.8):
    chosen = []
    seen = set()
    # The draw occurs for each slot, avoiding rounding 20% away on small lists.
    for _ in range(limit):
        pool = primary if rng.random() < prior or not exploratory else exploratory
        candidates = []
        for state in pool:
            key = tuple(sorted(tuple(sorted(e)) for e in state.graph.edges))
            if key not in seen:
                candidates.append((key, state))
        if not candidates:
            for state in primary + exploratory:
                key = tuple(sorted(tuple(sorted(e)) for e in state.graph.edges))
                if key not in seen:
                    candidates.append((key, state))
        if not candidates:
            break
        key, state = candidates[int(rng.integers(len(candidates)))]
        seen.add(key)
        chosen.append(state)
    return chosen


def embed_core(
    k,
    p,
    symbols,
    edges,
    coordinates,
    rng,
    spec,
    pack,
    parents,
    channel,
    occupation=None,
    prior=0.8,
    limit=3,
):
    from .molecular_motif_reconstruct import reconstruct_motif_state
    from .molecular_zb_growth import place_cl_on_zb_core

    primary, exploratory, stats = bounded_shells(symbols, edges, p, rng, coordinates)
    proposals = []
    for state in choose_shells(primary, exploratory, rng, limit, prior):
        sy = [a.symbol for a in state.atoms]
        anchored = (
            {i: point for i, point in enumerate(coordinates)}
            if coordinates is not None
            else {}
        )
        xyz = place_cl_on_zb_core(state, anchored, spec, pack) if anchored else None
        if xyz is None:
            rebuilt = reconstruct_motif_state(
                state,
                pack,
                spec,
                starts=1,
                keep=1,
                max_nfev=40,
                overlap_min_A=0.75,
                start_max_bond_error_A=1.0,
                core_coordinates=anchored or None,
            )
            if not rebuilt.candidates:
                continue
            xyz = np.asarray(rebuilt.candidates[0].coordinates)
        proposals.append(
            Proposal(
                k,
                p,
                sy,
                np.asarray(xyz).tolist(),
                sorted([list(sorted(e)) for e in state.graph.edges]),
                parents,
                channel,
                occupation,
            )
        )
    return proposals, stats


def fresh_cores(k, p, rng, limit=16):
    """Bounded small-k graph sampling; includes rings without requiring them."""
    symbols = ["Se"] * k + ["Cd"] * (k + p)
    out = []
    known = {}
    for _ in range(2000):
        graph = nx.Graph()
        graph.add_nodes_from(range(len(symbols)))
        for se in range(k):
            for cd in rng.choice(
                range(k, len(symbols)), size=min(2, k + p), replace=False
            ):
                graph.add_edge(se, int(cd))
        for cd in range(k, len(symbols)):
            if not graph.degree(cd):
                graph.add_edge(int(rng.integers(k)), cd)
        if rng.random() < 0.5:
            a = int(rng.integers(k))
            b = int(rng.integers(k, len(symbols)))
            graph.add_edge(a, b)
        if max(dict(graph.degree).values()) > 4 or not nx.is_connected(graph):
            continue
        nx.set_node_attributes(graph, dict(enumerate(symbols)), "element")
        key = nx.weisfeiler_lehman_graph_hash(graph, node_attr="element")
        if any(
            nx.is_isomorphic(
                graph, g, node_match=lambda a, b: a["element"] == b["element"]
            )
            for g in known.get(key, [])
        ):
            continue
        known.setdefault(key, []).append(graph)
        out.append((symbols, sorted(graph.edges)))
        if len(out) >= limit:
            break
    return out


def local_proposals(parent, rng, spec, pack, *, channel, limit=8):
    from .molecular_growth import (
        _outward_direction,
        place_monomer_and_packages,
        ParentStructure,
        _cdcl_bond_A,
    )

    sy = list(parent["symbols"])
    xyz = np.asarray(parent["positions"])
    edges = [tuple(e) for e in parent["edges"]]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(sy)))
    graph.add_edges_from(edges)
    k, p = parent["k"], parent["p"]
    out = []

    def emit(symbols, coords, es, kk, pp):
        proposal = Proposal(
            kk,
            pp,
            list(symbols),
            np.asarray(coords).tolist(),
            sorted([list(sorted(e)) for e in es]),
            [parent["minimum_id"]],
            channel,
            audit_derived=parent.get("role") == "audit"
            or parent.get("audit_derived", False),
        )
        if validate_composition(proposal):
            out.append(proposal)

    if channel == "topology":
        # Deliberately change the Cd--Se graph before relaxation.  A coordinate
        # perturbation alone almost always returns to the same local basin and
        # therefore cannot repair a lineage search that has lost a core family.
        # These are search moves, not claims about elementary reaction steps.
        core = graph.subgraph(i for i, s in enumerate(sy) if s != "Cl").copy()
        cd = [i for i, s in enumerate(sy) if s == "Cd"]
        se = [i for i, s in enumerate(sy) if s == "Se"]
        candidates = [(a, b) for a in cd for b in se if not core.has_edge(a, b)]
        rng.shuffle(candidates)
        core_degree = dict(core.degree)

        def moved_coordinates(new_edges, removed=()):
            trial = xyz.copy()
            for a, b in new_edges:
                # Move the cation part-way toward its new anion neighbour.  A
                # full ideal-bond projection is too violent when ligands remain
                # attached; 35% is enough to put the new basin in reach.
                cation, anion = (a, b) if sy[a] == "Cd" else (b, a)
                vector = trial[anion] - trial[cation]
                distance = float(np.linalg.norm(vector))
                if distance > 1e-8:
                    trial[cation] += max(0.0, distance - 2.66) * 0.35 * vector / distance
            touched = sorted({i for edge in list(new_edges) + list(removed) for i in edge})
            if touched:
                trial[touched] += np.clip(
                    rng.normal(0.0, 0.12, (len(touched), 3)), -0.3, 0.3
                )
            return trial

        # Ring closure / coordination increase.
        for a, b in candidates:
            if core_degree.get(a, 0) >= 4 or core_degree.get(b, 0) >= 4:
                continue
            changed = sorted(set(edges + [(min(a, b), max(a, b))]))
            emit(sy, moved_coordinates([(a, b)]), changed, k, p)
            if len(out) >= max(1, limit // 2):
                break

        # Bond migration/swap.  Keep the inorganic graph connected and both
        # atom types coordinated; this supplies open-chain and ring-changing
        # moves without enumerating all labelled graphs.
        core_edges = list(core.edges)
        rng.shuffle(core_edges)
        for old_a, old_b in core_edges:
            for new_a, new_b in candidates:
                if {old_a, old_b} == {new_a, new_b}:
                    continue
                trial_core = core.copy()
                trial_core.remove_edge(old_a, old_b)
                trial_core.add_edge(new_a, new_b)
                if not nx.is_connected(trial_core):
                    continue
                if any(trial_core.degree(i) == 0 or trial_core.degree(i) > 4 for i in trial_core):
                    continue
                changed = [e for e in edges if set(e) != {old_a, old_b}]
                changed.append((min(new_a, new_b), max(new_a, new_b)))
                emit(
                    sy,
                    moved_coordinates([(new_a, new_b)], [(old_a, old_b)]),
                    changed,
                    k,
                    p,
                )
                break
            if len(out) >= limit:
                break
    elif channel == "reconstruction":
        surface = [
            i for i, s in enumerate(sy) if s == "Cl" or graph.degree(i) < 4
        ] or list(graph)
        for j in range(limit):
            trial = xyz.copy()
            ids = [int(rng.choice(surface))] if j % 2 else surface
            trial[ids] += np.clip(rng.normal(0, 0.25, (len(ids), 3)), -0.6, 0.6)
            emit(sy, trial, edges, k, p)
    elif channel == "exchange":
        # Geminal departure, including bridging Cl. Never borrow remote Cl.
        for cd, s in enumerate(sy):
            if s != "Cd" or p <= 1:
                continue
            lig = [j for j in graph[cd] if sy[j] == "Cl"]
            for a, b in combinations(lig, 2):
                keep = [i for i in graph if i not in {cd, a, b}]
                g = graph.subgraph(keep)
                if not nx.is_connected(g):
                    continue
                ids = {old: new for new, old in enumerate(keep)}
                emit(
                    [sy[i] for i in keep],
                    xyz[keep],
                    [(ids[a], ids[b]) for a, b in g.edges],
                    k,
                    p - 1,
                )
                if len(out) >= limit // 2:
                    break
            if len(out) >= limit // 2:
                break
        candidates = [i for i, s in enumerate(sy) if s == "Se" and graph.degree(i) < 4]
        rng.shuffle(candidates)
        for host in candidates[: limit - len(out)]:
            d = _outward_direction(xyz, host, list(graph[host]))
            cd = len(sy)
            point = xyz[host] + 2.66 * d
            axis = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
            axis /= np.linalg.norm(axis)
            r = _cdcl_bond_A(pack)
            pts = np.vstack(
                [
                    xyz,
                    point,
                    point + r * (0.3 * d + 0.953939 * axis),
                    point + r * (0.3 * d - 0.953939 * axis),
                ]
            )
            emit(
                sy + ["Cd", "Cl", "Cl"],
                pts,
                edges + [(host, cd), (cd, cd + 1), (cd, cd + 2)],
                k,
                p + 1,
            )
    elif channel == "growth":
        hosts = [i for i, s in enumerate(sy) if s == "Cd"]
        hosts.sort(key=lambda i: (graph.degree(i), i))
        if len(hosts) > 2:
            tail = hosts[2:]
            rng.shuffle(tail)
            hosts = hosts[:2] + tail
        for host in hosts[: limit // 2]:
            # Explicit host choice avoids silently reverting to one lowest-CN atom.
            for pm in [0, 1]:
                symbols, coords, es = place_monomer_and_packages(
                    sy,
                    xyz,
                    edges,
                    k_parent=k,
                    p_after_shed=p,
                    p_m=pm,
                    spec=spec,
                    pack=pack,
                    attach_host=host,
                )
                emit(symbols, coords, es, k + 1, p + pm)
    return out[:limit]
