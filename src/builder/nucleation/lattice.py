from __future__ import annotations

from .graph_ops import *  # private names via __all__

from .types import *  # private names via __all__

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from pymatgen.core import Structure

from ..analysis import cif_first_shell_vector_sets, derive_pair_cuts_from_cif
from ..nc_types import NucleationSpec
from .types import AtomRecord, FloatArray, _LatticeModel, _State, _Vacancy

def _soft_clash_radius(model: _LatticeModel) -> float:
    """Minimum allowed separation for non-coinciding atoms (matches ``_state_valid``).

    Using only ``site_tolerance`` (~0.2 Å) for free-site placement let two Cl sit
    ~0.3–0.5 Å apart on "different" virtual sites.  Continuous decoration must
    refuse any placement closer than a shortened bulk bond.
    """

    return float(max(model.site_tolerance, model.bond_length - model.site_tolerance))


def _position_clashes(
    position: FloatArray,
    occupied: FloatArray,
    radius: float,
) -> bool:
    if occupied is None or len(occupied) == 0:
        return False
    return bool(
        np.any(np.linalg.norm(occupied - position, axis=1) < radius)
    )


def _state_has_soft_clashes(
    state: _State,
    model: _LatticeModel,
) -> bool:
    """True if any atom pair is unphysically close (same rule as ``_state_valid``)."""

    positions = _atom_positions(state.atoms)
    n = len(positions)
    if n < 2:
        return False
    radius = _soft_clash_radius(model)
    for i in range(n):
        if np.any(
            np.linalg.norm(positions[i + 1 :] - positions[i], axis=1) < radius
        ):
            return True
    return False


def _pair_distances(positions: FloatArray) -> FloatArray:
    """Return the full symmetric pairwise distance matrix in one operation."""

    deltas = positions[:, None, :] - positions[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))


def _allowed_pair_matrix(
    symbols: Sequence[str],
    spec: NucleationSpec,
) -> NDArray[np.bool_]:
    """Return an ``(n,n)`` mask of which atom pairs may form a bond at all."""

    order = sorted({*symbols})
    index = {symbol: position for position, symbol in enumerate(order)}
    table = np.zeros((len(order), len(order)), dtype=bool)
    for left, right in spec.graph_rules.allowed_bonds:
        if left in index and right in index:
            table[index[left], index[right]] = True
            table[index[right], index[left]] = True
    codes = np.fromiter(
        (index[symbol] for symbol in symbols), dtype=int, count=len(symbols)
    )
    return table[np.ix_(codes, codes)]


def _bonded_pair_matrix(
    positions: FloatArray,
    symbols: Sequence[str],
    model: _LatticeModel,
    spec: NucleationSpec,
) -> Tuple[FloatArray, NDArray[np.bool_], NDArray[np.bool_]]:
    """Return ``(distances, allowed_mask, bonded_mask)`` for every atom pair.

    ``bonded_mask`` marks the pairs that are both chemically allowed and sitting
    at the reference bond length within ``site_tolerance`` -- the single rule
    that decides connectivity on the rigid lattice.  The diagonal is cleared so
    callers may use the mask directly.
    """

    distances = _pair_distances(positions)
    allowed = _allowed_pair_matrix(symbols, spec)
    at_bond_length = (
        np.abs(distances - model.bond_length) <= model.site_tolerance
    )
    bonded = allowed & at_bond_length
    np.fill_diagonal(bonded, False)
    return distances, allowed, bonded


def _same_species_contacts_valid(
    distances: FloatArray,
    symbols: Sequence[str],
    model: _LatticeModel,
) -> bool:
    """Return whether no same-species pair violates a hard contact floor."""

    if not model.same_species_min_distance:
        return True
    for symbol, cutoff in model.same_species_min_distance.items():
        indices = [
            index for index, candidate in enumerate(symbols)
            if candidate == symbol
        ]
        if len(indices) < 2:
            continue
        sub = distances[np.ix_(indices, indices)]
        sub_upper = np.triu(np.ones(sub.shape, dtype=bool), 1)
        if np.any(sub_upper & (sub < float(cutoff))):
            return False
    return True


def _make_core_graph(
    atoms: Sequence[AtomRecord],
    model: _LatticeModel,
    spec: NucleationSpec,
) -> _State:
    """Build all and only geometrically present bonds allowed by graph rules."""

    normalized = _normalize_atoms(atoms)
    graph = nx.Graph()
    for atom in normalized:
        graph.add_node(
            atom.atom_id,
            element=atom.symbol,
            role=atom.role,
            unit_id=atom.unit_id,
        )
    positions = _atom_positions(normalized)
    _distances, _allowed, bonded = _bonded_pair_matrix(
        positions, [atom.symbol for atom in normalized], model, spec
    )
    graph.add_edges_from(
        (int(left), int(right), {"kind": "chemical", "bond_order": 1})
        for left, right in zip(*np.nonzero(np.triu(bonded, 1)))
    )
    return _State(normalized, graph)


def _extend_core_graph(
    base: _State,
    additions: Sequence[AtomRecord],
    model: _LatticeModel,
    spec: NucleationSpec,
) -> _State:
    """Append atoms to a base state, deriving only the genuinely new bonds.

    Produces exactly what ``_make_core_graph`` would over the concatenated atom
    list: the existing atoms have not moved, so their mutual bonds are already
    settled in ``base.graph``, and only the ``len(additions) x n`` new pairs need
    a distance test.  That turns the per-candidate cost from ``O(n^2)`` into
    ``O(len(additions) * n)`` while leaving the result bit-identical.
    """

    if not additions:
        return base
    normalized = _normalize_atoms([*base.atoms, *additions])
    graph = base.graph.copy()
    offset = len(base.atoms)
    for index, atom in enumerate(normalized[offset:], start=offset):
        graph.add_node(
            index,
            element=atom.symbol,
            role=atom.role,
            unit_id=atom.unit_id,
        )
    positions = _atom_positions(normalized)
    symbols = [atom.symbol for atom in normalized]
    allowed = _allowed_pair_matrix(symbols, spec)[offset:]
    deltas = positions[offset:, None, :] - positions[None, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))
    bonded = allowed & (
        np.abs(distances - model.bond_length) <= model.site_tolerance
    )
    for row, column in zip(*np.nonzero(bonded)):
        left, right = int(row) + offset, int(column)
        if left == right:
            continue
        graph.add_edge(left, right, kind="chemical", bond_order=1)
    return _State(normalized, graph)


def _seed_state(model: _LatticeModel) -> _State:
    vector = np.asarray(model.environments[model.core.cation][0][0], dtype=float)
    positions = np.vstack([np.zeros(3), vector])
    positions -= positions.mean(axis=0)
    atoms = (
        AtomRecord(0, model.core.cation, tuple(positions[0]), "core_cation"),
        AtomRecord(1, model.core.anion, tuple(positions[1]), "core_anion"),
    )
    graph = nx.Graph()
    for atom in atoms:
        graph.add_node(atom.atom_id, element=atom.symbol, role=atom.role)
    graph.add_edge(0, 1, kind="chemical", bond_order=1)
    return _State(atoms, graph)


def _cation_vacancies_on_anions(
    state: _State,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> List[_Vacancy]:
    return _vacancies(
        state,
        host_symbol=spec.core.anion,
        target_symbol=spec.core.cation,
        model=model,
        spec=spec,
    )


def _anion_vacancies_on_cations(
    state: _State,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> List[_Vacancy]:
    return _vacancies(
        state,
        host_symbol=spec.core.cation,
        target_symbol=spec.core.anion,
        model=model,
        spec=spec,
    )


def _vacancies(
    state: _State,
    *,
    host_symbol: str,
    target_symbol: str,
    model: _LatticeModel,
    spec: NucleationSpec,
    count_ligands_as_neighbors: bool = False,
) -> List[_Vacancy]:
    """Find vacant CIF directions on hosts.

    When ``count_ligands_as_neighbors`` is True (decorated k-growth), ligand
    atoms count as occupied tetrahedral directions: they enter the local
    environment match and block sites via occupation.  Historical bare growth
    ignores ligands in the environment vectors (they are already stripped).
    """

    positions = _atom_positions(state.atoms)
    vacancies: List[_Vacancy] = []
    ligand = spec.precursor.ligand
    for atom in state.atoms:
        if atom.symbol != host_symbol:
            continue
        actual = []
        for neighbor in state.graph.neighbors(atom.atom_id):
            if (
                not count_ligands_as_neighbors
                and state.atoms[neighbor].symbol == ligand
            ):
                continue
            actual.append(positions[neighbor] - positions[atom.atom_id])
        for environment in _best_environments(
            model.environments[host_symbol], actual
        ):
            for vector in environment:
                position = positions[atom.atom_id] + np.asarray(vector, dtype=float)
                if _position_occupied(position, positions, model.site_tolerance):
                    continue
                _merge_vacancy(
                    vacancies,
                    target_symbol,
                    position,
                    atom.atom_id,
                    model.site_tolerance,
                )
    vacancies.sort(key=lambda site: _position_key(site.position, model.site_tolerance))
    return vacancies


def _partner_slots(
    anchor_position: FloatArray,
    anchor_species: str,
    occupied: FloatArray,
    model: _LatticeModel,
) -> List[FloatArray]:
    actual = [
        point - anchor_position
        for point in occupied
        if abs(
            float(np.linalg.norm(point - anchor_position)) - model.bond_length
        ) <= model.site_tolerance
    ]
    slots: List[FloatArray] = []
    for environment in _best_environments(
        model.environments[anchor_species], actual
    ):
        for vector in environment:
            position = anchor_position + np.asarray(vector, dtype=float)
            if _position_occupied(position, occupied, model.site_tolerance):
                continue
            if not any(
                np.linalg.norm(position - old) < model.site_tolerance
                for old in slots
            ):
                slots.append(position)
    return slots


def _environment_array(
    environment: Tuple[Tuple[float, float, float], ...],
) -> FloatArray:
    """Cache the array form of one frozen CIF environment."""

    return np.asarray(environment, dtype=float)


def _same_species_contact_cutoffs(
    structure: Structure,
    spec: NucleationSpec,
    bond_length: float,
) -> Dict[str, float]:
    """Derive hard minimum distances for same-species construction contacts.

    Cd--Cd and Se--Se are not chemical bonds in the bipartite CdSe graph, but
    the lattice still has a well-defined nearest homonuclear shell.  A
    candidate closer than that shell (minus the site tolerance) is a collapsed
    construction, not a valid surface contact.  Cl has no site in the CdSe CIF,
    so use 90% of its van-der-Waals diameter as a conservative compressed
    ligand-contact floor; this rejects the bond-like 3.1 Å Cl--Cl contacts seen
    in the earlier k=3 DFT inputs.
    """

    cutoffs: Dict[str, float] = {}
    core_symbols = {spec.core.cation, spec.core.anion}
    for symbol in core_symbols:
        first_shell: List[float] = []
        for site in structure:
            if str(site.specie.symbol) != symbol:
                continue
            neighbors = structure.get_neighbors(site, 10.0)
            distances = [
                float(neighbor.nn_distance)
                for neighbor in neighbors
                if str(neighbor.specie.symbol) == symbol
                and float(neighbor.nn_distance) > 1.0e-8
            ]
            if distances:
                first_shell.append(min(distances))
        if first_shell:
            cutoffs[symbol] = max(
                0.0,
                min(first_shell) - float(spec.site_tolerance),
            )

    ligand = spec.precursor.ligand
    try:
        from pymatgen.core.periodic_table import Element

        vdw = Element(ligand).van_der_waals_radius
    except Exception:
        vdw = None
    if vdw is not None:
        cutoffs[ligand] = max(
            cutoffs.get(ligand, 0.0),
            0.90 * 2.0 * float(vdw),
        )
    else:
        cutoffs.setdefault(
            ligand,
            float(bond_length) + float(spec.site_tolerance),
        )
    return cutoffs


def _build_lattice_model(spec: NucleationSpec) -> _LatticeModel:
    structure = Structure.from_file(spec.cif)
    environments = {
        spec.core.cation: _freeze_environments(
            cif_first_shell_vector_sets(
                structure, spec.core.cation, [spec.core.anion]
            )
        ),
        spec.core.anion: _freeze_environments(
            cif_first_shell_vector_sets(
                structure, spec.core.anion, [spec.core.cation]
            )
        ),
    }
    if not environments[spec.core.cation] or not environments[spec.core.anion]:
        raise ValueError("CIF does not define core first-shell environments")
    # Keep calibration as a validation side effect used by the parent package.
    derive_pair_cuts_from_cif(spec.cif, spec.charges, safety=1.00)
    lengths = [
        np.linalg.norm(np.asarray(vector, dtype=float))
        for environment in environments[spec.core.cation]
        for vector in environment
    ]
    bond_length = float(np.median(lengths))
    return _LatticeModel(
        structure=structure,
        core=spec.core,
        environments=environments,
        bond_length=bond_length,
        site_tolerance=spec.site_tolerance,
        same_species_min_distance=_same_species_contact_cutoffs(
            structure, spec, bond_length
        ),
    )


def _freeze_environments(
    environments: Sequence[Sequence[FloatArray]],
) -> Tuple[Tuple[Tuple[float, float, float], ...], ...]:
    return tuple(
        tuple(tuple(float(x) for x in vector) for vector in environment)
        for environment in environments
    )


def _best_environments(
    environments: Sequence[Sequence[Tuple[float, float, float]]],
    actual_vectors: Sequence[FloatArray],
) -> List[Sequence[Tuple[float, float, float]]]:
    if not actual_vectors:
        return list(environments)
    scores = [_environment_score(env, actual_vectors) for env in environments]
    best = max(scores)
    return [
        environment
        for environment, score in zip(environments, scores)
        if abs(score - best) < 1e-8
    ]


def _environment_score(
    environment: Sequence[Tuple[float, float, float]],
    actual_vectors: Sequence[FloatArray],
) -> float:
    ideal = [
        np.asarray(vector, dtype=float) / np.linalg.norm(vector)
        for vector in environment
    ]
    actual = [
        vector / np.linalg.norm(vector)
        for vector in actual_vectors
        if np.linalg.norm(vector) > 1e-12
    ]
    assigned: set[int] = set()
    score = 0.0
    for vector in actual:
        choices = [
            (float(np.dot(vector, candidate)), index)
            for index, candidate in enumerate(ideal)
            if index not in assigned
        ]
        if not choices:
            score -= 1.0
            continue
        dot, index = max(choices)
        assigned.add(index)
        score += dot
    return score


def _atom_positions(atoms: Sequence[AtomRecord]) -> FloatArray:
    return np.asarray([atom.coordinates for atom in atoms], dtype=float)


def _position_occupied(
    position: FloatArray,
    occupied: FloatArray,
    tolerance: float,
) -> bool:
    return bool(
        len(occupied)
        and np.any(np.linalg.norm(occupied - position, axis=1) < tolerance)
    )


def _merge_vacancy(
    vacancies: List[_Vacancy],
    species: str,
    position: FloatArray,
    host: int,
    tolerance: float,
) -> None:
    for vacancy in vacancies:
        if (
            vacancy.species == species
            and np.linalg.norm(vacancy.position - position) < tolerance
        ):
            vacancy.hosts.add(host)
            return
    vacancies.append(
        _Vacancy(species, np.asarray(position, dtype=float), {host})
    )


def _position_key(position: FloatArray, tolerance: float) -> Tuple[int, int, int]:
    return tuple(int(round(float(value) / tolerance)) for value in position)


def _count_symbol(atoms: Sequence[AtomRecord], symbol: str) -> int:
    return sum(atom.symbol == symbol for atom in atoms)

__all__ = [
    '_soft_clash_radius',
    '_position_clashes',
    '_state_has_soft_clashes',
    '_pair_distances',
    '_allowed_pair_matrix',
    '_bonded_pair_matrix',
    '_same_species_contacts_valid',
    '_make_core_graph',
    '_extend_core_graph',
    '_seed_state',
    '_cation_vacancies_on_anions',
    '_anion_vacancies_on_cations',
    '_vacancies',
    '_partner_slots',
    '_environment_array',
    '_same_species_contact_cutoffs',
    '_build_lattice_model',
    '_freeze_environments',
    '_best_environments',
    '_environment_score',
    '_atom_positions',
    '_position_occupied',
    '_merge_vacancy',
    '_position_key',
    '_count_symbol',
]
