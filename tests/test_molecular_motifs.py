from __future__ import annotations

import networkx as nx

from builder.nucleation.molecular_motifs import (
    construction_atom_type,
    coordination_motif_inventory,
    local_coordination_signature,
)
from builder.nucleation.types import AtomRecord, _State


def _state() -> _State:
    # Cd0-Se1-Cd2 with one terminal Cl and one mu2 Cl.
    symbols = ["Cd", "Se", "Cd", "Cl", "Cl"]
    roles = [
        "core_cation",
        "core_anion",
        "precursor_center",
        "precursor_ligand",
        "precursor_ligand",
    ]
    atoms = tuple(
        AtomRecord(
            atom_id=i,
            symbol=symbol,
            coordinates=(0.0, 0.0, 0.0),
            role=roles[i],
        )
        for i, symbol in enumerate(symbols)
    )
    graph = nx.Graph()
    graph.add_nodes_from(range(len(atoms)))
    graph.add_edges_from([(0, 1), (1, 2), (0, 3), (0, 4), (2, 4)])
    return _State(atoms=atoms, graph=graph)


def test_construction_types_use_final_graph_degree() -> None:
    state = _state()
    assert construction_atom_type(state, 0) == "Cd3"
    assert construction_atom_type(state, 1) == "Se2"
    assert construction_atom_type(state, 3) == "Cl_t"
    assert construction_atom_type(state, 4) == "Cl_b2"


def test_local_signature_and_inventory() -> None:
    state = _state()
    assert local_coordination_signature(state, 0) == "Se1Cl2"
    inventory = coordination_motif_inventory(state)
    assert inventory["Cl_t"] == 1
    assert inventory["Cl_b2"] == 1
    assert inventory["Cd3:Se1Cl2"] == 1
    assert inventory["Cd-Se"] == 2
    assert inventory["Cd-Cl"] == 3

