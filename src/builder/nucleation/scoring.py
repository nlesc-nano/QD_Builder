from __future__ import annotations

from .graph_ops import *  # private names via __all__

from .lattice import *  # private names via __all__

from .types import *  # private names via __all__

from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from ..nc_types import NucleationSpec
from .types import AtomRecord, ClusterRecord, FloatArray, _LatticeModel, _State, _EnumerationCache

def _state_valid(
    state: _State,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> bool:
    if not nx.is_connected(state.graph):
        return False
    positions = _atom_positions(state.atoms)
    count = len(state.atoms)
    distances, allowed, bonded = _bonded_pair_matrix(
        positions, [atom.symbol for atom in state.atoms], model, spec
    )
    if not _same_species_contacts_valid(
        distances, [atom.symbol for atom in state.atoms], model
    ):
        return False
    upper = np.triu(np.ones((count, count), dtype=bool), 1)

    # No two sites may coincide, nothing may sit closer than a bond, and a
    # forbidden pair may not come within bonding range at all.
    if np.any(upper & (distances < model.site_tolerance)):
        return False
    if np.any(upper & (distances < model.bond_length - model.site_tolerance)):
        return False
    if np.any(
        upper
        & ~allowed
        & (distances <= model.bond_length + model.site_tolerance)
    ):
        return False

    # The graph must record exactly the geometrically present bonds.
    present = np.zeros((count, count), dtype=bool)
    if state.graph.number_of_edges():
        edges = np.fromiter(
            (node for edge in state.graph.edges for node in edge),
            dtype=int,
            count=2 * state.graph.number_of_edges(),
        ).reshape(-1, 2)
        present[edges[:, 0], edges[:, 1]] = True
        present[edges[:, 1], edges[:, 0]] = True
    if np.any(upper & (present != bonded)):
        return False

    for atom in state.atoms:
        degree = state.graph.degree[atom.atom_id]
        if degree > spec.graph_rules.max_cn[atom.symbol]:
            return False
        if atom.role == "precursor_ligand":
            if degree < 1:
                return False
        if atom.role in {"core_cation", "precursor_center"} and not any(
            state.atoms[neighbor].symbol == spec.core.anion
            for neighbor in state.graph.neighbors(atom.atom_id)
        ):
            return False
        if not _atom_geometry_is_rigid(atom, state, model, spec, positions):
            return False

    # Optional molecular hard filters (off by default for lattice maps).
    if (
        spec.require_inorganic_connected
        or spec.bridges_per_cd_pair > 0
        or spec.enforce_min_cn
    ):
        from .molecular_rules import molecular_graph_violations

        if molecular_graph_violations(state, spec):
            return False
    return True


@lru_cache(maxsize=256)
def _saturates_all_rows(adjacency: Sequence[int], column_count: int) -> bool:
    """Whether every row can be matched to a distinct compatible column.

    Kuhn's augmenting-path algorithm over bitmask adjacency.  The lists here are
    at most 4x4 (a tetrahedral CIF environment), so this replaces a NetworkX
    bipartite graph construction that dominated ``_state_valid``.
    """

    if len(adjacency) > column_count:
        return False
    matched_row_of_column = [-1] * column_count

    def augment(row: int, visited: List[bool]) -> bool:
        mask = adjacency[row]
        for column in range(column_count):
            if not (mask >> column) & 1 or visited[column]:
                continue
            visited[column] = True
            occupant = matched_row_of_column[column]
            if occupant == -1 or augment(occupant, visited):
                matched_row_of_column[column] = row
                return True
        return False

    for row in range(len(adjacency)):
        if not augment(row, [False] * column_count):
            return False
    return True


def _atom_geometry_is_rigid(
    atom: AtomRecord,
    state: _State,
    model: _LatticeModel,
    spec: NucleationSpec,
    positions: Optional[FloatArray] = None,
) -> bool:
    """Check that incident edges occupy distinct sites of one CIF environment."""

    neighbors = list(state.graph.neighbors(atom.atom_id))
    if not neighbors:
        return True
    template_symbol = (
        spec.core.cation
        if atom.role in {"core_cation", "precursor_center"}
        else spec.core.anion
    )
    if positions is None:
        positions = _atom_positions(state.atoms)
    actual = positions[neighbors] - positions[atom.atom_id]
    for environment in model.environments[template_symbol]:
        ideal = _environment_array(environment)
        if len(actual) > len(ideal):
            continue
        deltas = actual[:, None, :] - ideal[None, :, :]
        compatible = (
            np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))
            <= model.site_tolerance
        )
        adjacency = [
            sum(1 << column for column in range(len(ideal)) if row[column])
            for row in compatible
        ]
        if _saturates_all_rows(tuple(adjacency), len(ideal)):
            return True
    return False


def _base_coordination_valid(
    state: _State,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> bool:
    positions = _atom_positions(state.atoms)
    distances = _pair_distances(positions)
    if not _same_species_contacts_valid(
        distances, [atom.symbol for atom in state.atoms], model
    ):
        return False
    for atom in state.atoms:
        if state.graph.degree[atom.atom_id] > spec.graph_rules.max_cn[atom.symbol]:
            return False
        if atom.role in {"core_cation", "precursor_center"} and not any(
            state.atoms[neighbor].symbol == spec.core.anion
            for neighbor in state.graph.neighbors(atom.atom_id)
        ):
            return False
    return True


def _coordination_score(
    record: ClusterRecord,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> Tuple[int, ...]:
    return _graph_coordination_score(record.atoms, record.graph, spec)


class _BridgeScoreContext:
    """Score a bridge assignment from degree deltas, without building a graph.

    ``_graph_coordination_score`` reads only three things off a graph: every
    atom's degree, the edge count, and how many bridges sit on exact CIF sites.
    All three follow from the chosen arcs, because a bridge

    * lifts its acceptor Cd by one coordination,
    * turns one terminal Cl of its donor into a two-coordinate bridging Cl,
    * and leaves the donor Cd's own coordination untouched,

    and because terminal Cl on one host are interchangeable degree-1 leaves, so
    only *how many* of them bridge matters, never which.  That removes a full
    graph copy per candidate assignment from the search.

    ``test_bridge_score_context_matches_graph_scoring`` pins this to
    ``_graph_coordination_score`` over real bases; the two must never diverge.
    """

    __slots__ = (
        "_base_degrees",
        "_targets",
        "_min_cn",
        "_max_target",
        "_species",
        "_base_edges",
        "_skeleton_scope",
        "_cand_primary",
        "_cand_host",
        "_cand_exact",
        "_terminal_by_primary",
    )

    def __init__(
        self,
        state: "_State",
        spec: NucleationSpec,
        cand_primary: Sequence[int],
        cand_host: Sequence[int],
        cand_exact: Sequence[bool],
        terminal_by_primary: Mapping[int, Sequence[int]],
    ) -> None:
        self._base_degrees = [
            state.graph.degree[atom.atom_id] for atom in state.atoms
        ]
        self._targets = [
            spec.graph_rules.max_cn[atom.symbol] for atom in state.atoms
        ]
        self._min_cn = [
            spec.graph_rules.min_cn[atom.symbol] for atom in state.atoms
        ]
        self._max_target = max(self._targets)
        self._species = [
            (
                target,
                [
                    atom.atom_id
                    for atom in state.atoms
                    if atom.symbol == symbol
                ],
            )
            for symbol, target in sorted(spec.graph_rules.max_cn.items())
        ]
        # Under "skeleton" scope only cation-anion bonds count, and a bridge adds
        # a cation-ligand bond -- so the base count excludes ligand bonds and the
        # chosen arcs contribute nothing.  ``_skeleton_scope`` carries that to
        # ``score`` so the graph-free path and ``_graph_coordination_score`` stay
        # in lockstep; ``test_bridge_score_context_matches_graph_scoring`` pins it.
        self._skeleton_scope = spec.bond_count_scope == "skeleton"
        if self._skeleton_scope:
            self._base_edges = sum(
                1
                for left, right in state.graph.edges
                if state.atoms[left].symbol != spec.precursor.ligand
                and state.atoms[right].symbol != spec.precursor.ligand
            )
        else:
            self._base_edges = state.graph.number_of_edges()
        self._cand_primary = cand_primary
        self._cand_host = cand_host
        self._cand_exact = cand_exact
        self._terminal_by_primary = terminal_by_primary

    def degrees(self, subset: Sequence[int]) -> List[int]:
        """Final degree of every atom after applying ``subset``."""

        degrees = list(self._base_degrees)
        per_primary: Dict[int, int] = {}
        for index in subset:
            degrees[self._cand_host[index]] += 1
            primary = self._cand_primary[index]
            per_primary[primary] = per_primary.get(primary, 0) + 1
        for primary, count in per_primary.items():
            for ligand in self._terminal_by_primary[primary][:count]:
                degrees[ligand] += 1
        return degrees

    def score(self, subset: Sequence[int]) -> Tuple[int, ...]:
        degrees = self.degrees(subset)
        violation_count = 0
        total_shortfall = 0
        deficits: List[int] = []
        for target, minimum, degree in zip(self._targets, self._min_cn, degrees):
            deficits.append(target - degree if target > degree else 0)
            shortfall = minimum - degree
            if shortfall > 0:
                violation_count += 1
                total_shortfall += shortfall
        severe = tuple(
            -sum(deficit >= threshold for deficit in deficits)
            for threshold in range(self._max_target, 1, -1)
        )
        species_hist: List[int] = []
        for target, members in self._species:
            values = [degrees[member] for member in members]
            species_hist.extend(
                sum(value == cn for value in values)
                for cn in range(target, -1, -1)
            )
        return (
            int(violation_count == 0),
            -violation_count,
            -total_shortfall,
            self._base_edges
            + (0 if self._skeleton_scope else len(subset)),
            *severe,
            *species_hist,
            sum(1 for index in subset if self._cand_exact[index]),
        )


def _graph_coordination_score(
    atoms: Sequence[AtomRecord],
    graph: nx.Graph,
    spec: NucleationSpec,
) -> Tuple[int, ...]:
    """Return the selection score without constructing a registry record."""

    targets = [spec.graph_rules.max_cn[atom.symbol] for atom in atoms]
    degrees = [graph.degree[atom.atom_id] for atom in atoms]
    if spec.bond_count_scope == "skeleton":
        # Rank on the inorganic framework alone: ligand bonds, and therefore
        # bridges, stop buying rank.
        bond_count = sum(
            1
            for left, right in graph.edges
            if atoms[left].symbol != spec.precursor.ligand
            and atoms[right].symbol != spec.precursor.ligand
        )
    else:
        bond_count = graph.number_of_edges()
    deficits = [max(0, target - degree) for target, degree in zip(targets, degrees)]
    minimum_shortfalls = [
        max(0, spec.graph_rules.min_cn[atom.symbol] - degree)
        for atom, degree in zip(atoms, degrees)
    ]
    violation_count = sum(shortfall > 0 for shortfall in minimum_shortfalls)
    total_shortfall = sum(minimum_shortfalls)
    severe = tuple(
        -sum(deficit >= threshold for deficit in deficits)
        for threshold in range(max(targets), 1, -1)
    )
    species_hist: List[int] = []
    for symbol, target in sorted(spec.graph_rules.max_cn.items()):
        cn_values = [
            graph.degree[atom.atom_id]
            for atom in atoms if atom.symbol == symbol
        ]
        species_hist.extend(
            sum(value == cn for value in cn_values)
            for cn in range(target, -1, -1)
        )
    return (
        int(violation_count == 0),
        -violation_count,
        -total_shortfall,
        bond_count,
        *severe,
        *species_hist,
        sum(
            data.get("bridge_mode") == "shared_vacant_cif_site"
            for _left, _right, data in graph.edges(data=True)
        ),
    )


def _greedy_resolved(costs: Sequence[int], budget: int) -> int:
    """How many items can be paid for cheapest-first within ``budget``."""

    resolved = 0
    for cost in sorted(costs):
        if cost > budget:
            break
        budget -= cost
        resolved += 1
    return resolved


def _optimistic_bridge_score(
    state: _State,
    spec: NucleationSpec,
) -> Tuple[int, ...]:
    """Bound every score reachable by adding allowed terminal-ligand bridges.

    Geometry and host incidence are still relaxed -- any host with a free slot is
    treated as reachable -- but the bound now respects *who can receive a degree
    increment at all*, which the previous version did not:

    * A bridge raises exactly one host cation by one, and turns exactly one
      terminal ligand from one bond into two.  Nothing else moves.  In
      particular the **anion coordination numbers cannot change at all**, so
      their deficits and histogram entries are exact rather than assumed
      maximal.
    * Because each bridge contributes one increment to the cation pool *and* one
      to the ligand pool, the two pools have independent budgets of
      ``bridge_bound`` each.  The old ``degree_budget = 2 * bridge_bound``
      pooled them, which let a single bridge repair two cation shortfalls -- an
      over-estimate of a factor of two on the leading score components.
    * A terminal ligand can gain at most one bond, so it can only ever move from
      its current coordination to one above it.

    Every component is still computed as the most favourable value reachable in
    isolation, so the tuple dominates any achievable score componentwise, and
    componentwise domination implies lexicographic domination.  The bound is
    therefore safe for pruning: if it is strictly below an already surface-valid
    score, no bridge arrangement on this base can win the bin.
    """

    bridge_rules = tuple(spec.graph_rules.bridge_rules)
    if not bridge_rules:
        return _graph_coordination_score(state.atoms, state.graph, spec)
    ligand_symbols = {rule.ligand for rule in bridge_rules}
    host_symbols = {rule.host for rule in bridge_rules}

    targets = [spec.graph_rules.max_cn[atom.symbol] for atom in state.atoms]
    degrees = [state.graph.degree[atom.atom_id] for atom in state.atoms]

    # A host may absorb up to its free capacity; a terminal ligand exactly one
    # further bond.  Anything else has no reachable increment.
    max_gain: List[int] = []
    for atom, target, degree in zip(state.atoms, targets, degrees):
        if atom.symbol in host_symbols:
            max_gain.append(max(0, target - degree))
        elif atom.symbol in ligand_symbols and degree < target:
            max_gain.append(1)
        else:
            max_gain.append(0)

    donor_supply = sum(
        1
        for atom, gain in zip(state.atoms, max_gain)
        if atom.symbol in ligand_symbols and gain > 0
    )
    host_slots = sum(
        gain
        for atom, gain in zip(state.atoms, max_gain)
        if atom.symbol in host_symbols
    )
    host_count = sum(1 for atom in state.atoms if atom.symbol in host_symbols)
    # At most one bridge per unordered host pair.
    pair_cap = host_count * (host_count - 1) // 2
    bridge_bound = min(donor_supply, host_slots, pair_cap)

    def budget_for(symbol: str) -> int:
        if symbol in host_symbols or symbol in ligand_symbols:
            return bridge_bound
        return 0

    deficits = [
        target - degree if target > degree else 0
        for target, degree in zip(targets, degrees)
    ]
    shortfalls = [
        max(0, spec.graph_rules.min_cn[atom.symbol] - degree)
        for atom, degree in zip(state.atoms, degrees)
    ]

    # Group the repair budgets by pool so a cation increment cannot be spent
    # twice, once as itself and once as its partner ligand increment.
    violation_count = sum(value > 0 for value in shortfalls)
    resolved_violations = 0
    repairable_shortfall = 0
    for pool in (host_symbols, ligand_symbols):
        costs = [
            min(shortfall, gain)
            for atom, shortfall, gain in zip(state.atoms, shortfalls, max_gain)
            if atom.symbol in pool and shortfall > 0 and gain > 0
        ]
        fully = [
            shortfall
            for atom, shortfall, gain in zip(state.atoms, shortfalls, max_gain)
            if atom.symbol in pool and 0 < shortfall <= gain
        ]
        resolved_violations += _greedy_resolved(fully, bridge_bound)
        repairable_shortfall += min(sum(costs), bridge_bound)
    optimistic_violations = max(0, violation_count - resolved_violations)
    optimistic_shortfall = max(0, sum(shortfalls) - repairable_shortfall)

    severe: List[int] = []
    for threshold in range(max(targets), 1, -1):
        unfixable = 0
        for pool in (host_symbols, ligand_symbols, None):
            costs = []
            for atom, deficit, gain in zip(state.atoms, deficits, max_gain):
                if deficit < threshold:
                    continue
                in_pool = (
                    atom.symbol not in host_symbols
                    and atom.symbol not in ligand_symbols
                    if pool is None
                    else atom.symbol in pool
                )
                if not in_pool:
                    continue
                needed = deficit - threshold + 1
                # Unreachable within this atom's own ceiling: never fixable.
                costs.append(needed if needed <= gain else None)
            payable = [cost for cost in costs if cost is not None]
            unfixable += len(costs) - _greedy_resolved(
                payable, bridge_bound if pool is not None else 0
            )
        severe.append(-unfixable)

    species_hist: List[int] = []
    for symbol, target in sorted(spec.graph_rules.max_cn.items()):
        members = [
            (degree, gain)
            for atom, degree, gain in zip(state.atoms, degrees, max_gain)
            if atom.symbol == symbol
        ]
        budget = budget_for(symbol)
        for coordination in range(target, -1, -1):
            already = sum(1 for degree, _gain in members if degree == coordination)
            arrivals = [
                coordination - degree
                for degree, gain in members
                if degree < coordination and coordination - degree <= gain
            ]
            # Upper bound: keep everyone already at this coordination and move as
            # many others onto it as the pool budget allows.
            species_hist.append(
                already + _greedy_resolved(arrivals, budget)
            )

    return (
        int(optimistic_violations == 0),
        -optimistic_violations,
        -optimistic_shortfall,
        state.graph.number_of_edges() + bridge_bound,
        *severe,
        *species_hist,
        bridge_bound,
    )


def _reachable_bridge_score_max(
    state: _State,
    model: _LatticeModel,
    spec: NucleationSpec,
    cache: Optional[_EnumerationCache] = None,
) -> Tuple[int, ...]:
    """Lexicographic maximum of the score over the relaxed bridge space.

    The componentwise bound (`_optimistic_bridge_score`) evaluates each score
    component at its own best increment distribution; those distributions are
    usually mutually exclusive, which is exactly the measured slack.  This
    routine instead enumerates *joint* distributions and takes the true
    lexicographic maximum over them, so a base is pruned as soon as no single
    consistent way of spending the bridges can beat the incumbent.

    The distributions honour the base's real bridge opportunities per atom:

    * an acceptor cation can rise by at most ``min(free slots, distinct donors
      with an arc to it)``;
    * a donor cation can lend at most ``min(its terminal ligands, distinct
      acceptors it reaches)`` -- and only its own terminals can become bridging;
    * the total is capped by donor supply, acceptor intake and the number of
      distinct usable host pairs.

    What is still relaxed -- which combinations of arcs can be selected
    *simultaneously* -- only enlarges the space, so every achievable score is
    lexicographically at or below the returned maximum, and pruning on
    ``max < incumbent`` is safe.

    Atoms with identical (symbol, degree, cap) are interchangeable for scoring,
    so the enumeration runs over classes, not atoms, and the result is memoised
    on the class signature -- which repeats across the bases of one bin.
    """

    bridge_rules = tuple(spec.graph_rules.bridge_rules)
    if not bridge_rules:
        return _graph_coordination_score(state.atoms, state.graph, spec)
    max_cn = spec.graph_rules.max_cn
    min_cn = spec.graph_rules.min_cn
    host_symbols = {rule.host for rule in bridge_rules}
    ligand_symbols = {rule.ligand for rule in bridge_rules}
    host_symbol = next(iter(host_symbols))
    ligand_symbol = next(iter(ligand_symbols))

    # Late import: arcs live in engine; engine already depends on scoring.
    from .engine import _bridge_candidate_arcs

    terminal_by_primary, arcs = _bridge_candidate_arcs(state, model, spec)
    # Arcs on a pair already stitched by an existing two-host ligand can never
    # be selected; drop them before deriving any capacity.
    rules_by_ligand = {rule.ligand: rule for rule in bridge_rules}
    stitched: set[Tuple[int, int]] = set()
    for atom in state.atoms:
        rule = rules_by_ligand.get(atom.symbol)
        if rule is None:
            continue
        hosts = [
            neighbor
            for neighbor in state.graph.neighbors(atom.atom_id)
            if state.atoms[neighbor].symbol == rule.host
        ]
        if len(hosts) == 2:
            stitched.add((min(hosts), max(hosts)))
    usable = [
        arc
        for arc in arcs
        if (min(arc.primary, arc.host), max(arc.primary, arc.host))
        not in stitched
    ]

    donors_of_host: Dict[int, set[int]] = {}
    hosts_of_donor: Dict[int, set[int]] = {}
    usable_pairs: set[Tuple[int, int]] = set()
    for arc in usable:
        donors_of_host.setdefault(arc.host, set()).add(arc.primary)
        hosts_of_donor.setdefault(arc.primary, set()).add(arc.host)
        usable_pairs.add(
            (min(arc.primary, arc.host), max(arc.primary, arc.host))
        )

    degrees = {
        atom.atom_id: state.graph.degree[atom.atom_id] for atom in state.atoms
    }
    in_cap = {
        atom.atom_id: min(
            max_cn[atom.symbol] - degrees[atom.atom_id],
            len(donors_of_host.get(atom.atom_id, ())),
        )
        for atom in state.atoms
        if atom.symbol in host_symbols
    }
    out_cap = {
        primary: min(
            len(terminal_by_primary.get(primary, ())),
            len(hosts_of_donor.get(primary, ())),
        )
        for primary in hosts_of_donor
    }
    bumpable_ligands: set[int] = set()
    for primary, cap in out_cap.items():
        if cap > 0:
            bumpable_ligands.update(terminal_by_primary.get(primary, ()))

    # Partition atoms into host classes (may rise, keyed by degree and cap),
    # bumpable ligand classes (may take exactly one bond) and fixed atoms.
    host_classes: Dict[Tuple[int, int], int] = {}
    donor_classes: Dict[int, int] = {}
    fixed: List[Tuple[str, int]] = []
    for atom in state.atoms:
        degree = degrees[atom.atom_id]
        if atom.symbol in host_symbols and in_cap.get(atom.atom_id, 0) > 0:
            key = (degree, in_cap[atom.atom_id])
            host_classes[key] = host_classes.get(key, 0) + 1
        elif atom.atom_id in bumpable_ligands:
            donor_classes[degree] = donor_classes.get(degree, 0) + 1
        else:
            fixed.append((atom.symbol, degree))

    bridge_bound = min(
        sum(out_cap.values()),
        sum(
            cap * count for (_degree, cap), count in host_classes.items()
        ),
        len(usable_pairs),
    )

    signature = (
        tuple(sorted(fixed)),
        tuple(sorted(host_classes.items())),
        tuple(sorted(donor_classes.items())),
        bridge_bound,
    )
    if cache is not None:
        memoised = cache.reachable_scores.get(signature)
        if memoised is not None:
            return memoised

    base_edges = state.graph.number_of_edges()
    targets_max = max(max_cn[atom.symbol] for atom in state.atoms)
    species_order = sorted(max_cn.items())

    def score_of(
        host_final: Mapping[int, int],
        donor_bumped: Mapping[int, int],
        bridges: int,
    ) -> Tuple[int, ...]:
        # Assemble the final (symbol, degree) counts for the whole cluster.
        counts: Dict[Tuple[str, int], int] = {}
        for symbol, degree in fixed:
            counts[(symbol, degree)] = counts.get((symbol, degree), 0) + 1
        for degree, number in host_final.items():
            if number:
                key = (host_symbol, degree)
                counts[key] = counts.get(key, 0) + number
        for degree, count in donor_classes.items():
            bumped = donor_bumped.get(degree, 0)
            if bumped:
                key = (ligand_symbol, degree + 1)
                counts[key] = counts.get(key, 0) + bumped
            if count - bumped:
                key = (ligand_symbol, degree)
                counts[key] = counts.get(key, 0) + (count - bumped)

        violation_count = 0
        total_shortfall = 0
        deficit_counts: Dict[int, int] = {}
        for (symbol, degree), number in counts.items():
            shortfall = min_cn[symbol] - degree
            if shortfall > 0:
                violation_count += number
                total_shortfall += shortfall * number
            deficit = max_cn[symbol] - degree
            if deficit > 0:
                deficit_counts[deficit] = (
                    deficit_counts.get(deficit, 0) + number
                )
        severe = tuple(
            -sum(
                number
                for deficit, number in deficit_counts.items()
                if deficit >= threshold
            )
            for threshold in range(targets_max, 1, -1)
        )
        species_hist: List[int] = []
        for symbol, target in species_order:
            for coordination in range(target, -1, -1):
                species_hist.append(counts.get((symbol, coordination), 0))
        return (
            int(violation_count == 0),
            -violation_count,
            -total_shortfall,
            base_edges + bridges,
            *severe,
            *species_hist,
            bridges,
        )

    best: Optional[Tuple[int, ...]] = None

    donor_degrees = sorted(donor_classes)
    host_degrees = sorted(host_classes)

    def donor_distributions(
        index: int, remaining: int, chosen: Dict[int, int]
    ) -> List[Dict[int, int]]:
        if index == len(donor_degrees):
            return [dict(chosen)] if remaining == 0 else []
        degree = donor_degrees[index]
        results: List[Dict[int, int]] = []
        for bumped in range(min(remaining, donor_classes[degree]) + 1):
            chosen[degree] = bumped
            results.extend(
                donor_distributions(index + 1, remaining - bumped, chosen)
            )
        chosen.pop(degree, None)
        return results

    donor_options_by_b: Dict[int, List[Dict[int, int]]] = {}

    def class_options(degree: int, cap: int, count: int, budget: int):
        """All ways to lift `count` atoms from `degree` by <= `cap` each."""

        ceiling = degree + cap
        levels = list(range(degree, ceiling + 1))
        option: List[Tuple[Dict[int, int], int]] = []

        def rec(level_index: int, left: int, cost: int, acc: Dict[int, int]):
            if cost > budget:
                return
            if level_index == len(levels) - 1:
                final = dict(acc)
                final[levels[-1]] = final.get(levels[-1], 0) + left
                total = cost + left * (levels[-1] - degree)
                if total <= budget:
                    option.append((final, total))
                return
            level = levels[level_index]
            for here in range(left + 1):
                acc[level] = acc.get(level, 0) + here
                rec(
                    level_index + 1,
                    left - here,
                    cost + here * (level - degree),
                    acc,
                )
                acc[level] -= here

        rec(0, count, 0, {})
        return option

    def enumerate_tables(index: int, spent: int, final_counts: Dict[int, int]):
        nonlocal best
        if index == len(host_degrees):
            options = donor_options_by_b.get(spent)
            if options is None:
                options = donor_distributions(0, spent, {})
                donor_options_by_b[spent] = options
            for bumped in options:
                candidate = score_of(final_counts, bumped, spent)
                if best is None or candidate > best:
                    best = candidate
            return
        degree, cap = host_degrees[index]
        for class_final, cost in class_options(
            degree, cap, host_classes[(degree, cap)], bridge_bound - spent
        ):
            for level, number in class_final.items():
                final_counts[level] = final_counts.get(level, 0) + number
            enumerate_tables(index + 1, spent + cost, final_counts)
            for level, number in class_final.items():
                final_counts[level] -= number

    enumerate_tables(0, 0, {})
    assert best is not None  # b=0 with no donors bumped is always enumerated
    if cache is not None:
        cache.reachable_scores[signature] = best
    return best


def _coordination_metadata(
    record: ClusterRecord,
    model: _LatticeModel,
    spec: NucleationSpec,
    *,
    include_rings: bool = True,
) -> Dict[str, object]:
    by_symbol: Dict[str, List[int]] = {}
    deficits: Dict[str, List[int]] = {}
    minimum_shortfalls: Dict[str, List[int]] = {}
    for atom in record.atoms:
        cn = int(record.graph.degree[atom.atom_id])
        by_symbol.setdefault(atom.symbol, []).append(cn)
        deficits.setdefault(atom.symbol, []).append(
            max(0, _target_cn(atom, model, spec) - cn)
        )
        minimum_shortfalls.setdefault(atom.symbol, []).append(
            max(0, spec.graph_rules.min_cn[atom.symbol] - cn)
        )
    histograms = {
        symbol: {
            str(cn): values.count(cn)
            for cn in sorted(set(values))
        }
        for symbol, values in by_symbol.items()
    }
    total_cn = sum(sum(values) for values in by_symbol.values())
    violation_count = sum(
        shortfall > 0
        for values in minimum_shortfalls.values()
        for shortfall in values
    )
    total_shortfall = sum(
        sum(values) for values in minimum_shortfalls.values()
    )
    # Ring DFS is fine on small graphs but gets costly once many ligands are
    # present; callers that only need CN scores skip it until retention.
    rings = (
        _ring_counts_for_record(record, spec) if include_rings else {}
    )
    return {
        "rings": rings,
        "coordination_by_element": {
            symbol: sorted(values, reverse=True)
            for symbol, values in sorted(by_symbol.items())
        },
        "coordination_histograms": histograms,
        "coordination_deficits": {
            symbol: sorted(values, reverse=True)
            for symbol, values in sorted(deficits.items())
        },
        "minimum_cn_shortfalls": {
            symbol: sorted(values, reverse=True)
            for symbol, values in sorted(minimum_shortfalls.items())
        },
        "minimum_cn_violations_by_element": {
            symbol: sum(shortfall > 0 for shortfall in values)
            for symbol, values in sorted(minimum_shortfalls.items())
        },
        "min_cn_violation_count": violation_count,
        "min_cn_total_shortfall": total_shortfall,
        "min_cn_compliant": violation_count == 0,
        "total_cn": total_cn,
        "bond_count": record.graph.number_of_edges(),
        "bridge_count": sum(
            data.get("kind") == "surface_bridge"
            for _left, _right, data in record.graph.edges(data=True)
        ),
        "bridge_mode_counts": {
            mode: sum(
                data.get("kind") == "surface_bridge"
                and data.get("bridge_mode", "shared_occupied_neighbor") == mode
                for _left, _right, data in record.graph.edges(data=True)
            )
            for mode in (
                "shared_vacant_cif_site",
                "shared_occupied_neighbor",
            )
        },
        "geometry_residual": float(record.metadata.get("geometry_residual", 0.0)),
    }


def _target_cn(
    atom: AtomRecord,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> int:
    return spec.graph_rules.max_cn[atom.symbol]

__all__ = [
    '_state_valid',
    '_saturates_all_rows',
    '_atom_geometry_is_rigid',
    '_base_coordination_valid',
    '_coordination_score',
    '_BridgeScoreContext',
    '_graph_coordination_score',
    '_greedy_resolved',
    '_optimistic_bridge_score',
    '_reachable_bridge_score_max',
    '_coordination_metadata',
    '_target_cn',
]
