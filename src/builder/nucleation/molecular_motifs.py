"""Graph-level coordination motifs for molecular CdSe/CdCl2 builds.

Motifs in this module are descriptive graph labels, not 3D templates.  The
graph rules remain the source of chemical legality; these helpers make the
local coordination state explicit before an embedding is attempted.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Mapping, Sequence

from .types import _State


def construction_atom_type(
    state: _State,
    index: int,
    *,
    cation_symbols: Iterable[str] = ("Cd",),
    anion_symbols: Iterable[str] = ("Se",),
    ligand_symbols: Iterable[str] = ("Cl",),
) -> str:
    """Return the local construction type used by motif/geometry tables.

    CN values are deliberately derived from the *complete graph*.  Cd/Se
    values above the table range collapse to the highest supported bucket,
    while chloride degree identifies terminal/μ2/μ3 roles.
    """

    atom = state.atoms[int(index)]
    degree = int(state.graph.degree[int(index)])
    symbol = str(atom.symbol)
    if symbol in set(cation_symbols):
        return f"Cd{min(max(degree, 2), 4)}"
    if symbol in set(anion_symbols):
        return f"Se{min(max(degree, 2), 4)}"
    if symbol in set(ligand_symbols):
        return {1: "Cl_t", 2: "Cl_b2", 3: "Cl_b3"}.get(
            degree, f"Cl_deg{degree}"
        )
    return symbol


def local_coordination_signature(
    state: _State,
    center: int,
    *,
    cation_symbols: Iterable[str] = ("Cd",),
    anion_symbols: Iterable[str] = ("Se",),
    ligand_symbols: Iterable[str] = ("Cl",),
) -> str:
    """Return a compact local signature such as ``Cl2Se1`` or ``Se2``."""

    cations = set(cation_symbols)
    anions = set(anion_symbols)
    ligands = set(ligand_symbols)
    counts: Dict[str, int] = {"Cd": 0, "Se": 0, "Cl": 0, "other": 0}
    for neighbor in state.graph.neighbors(int(center)):
        symbol = str(state.atoms[int(neighbor)].symbol)
        key = (
            "Cd" if symbol in cations else
            "Se" if symbol in anions else
            "Cl" if symbol in ligands else
            "other"
        )
        counts[key] += 1
    return "".join(
        f"{key}{counts[key]}"
        for key in ("Cd", "Se", "Cl", "other")
        if counts[key]
    ) or "empty"


def coordination_motif_inventory(
    state: _State,
    *,
    cation_symbols: Iterable[str] = ("Cd",),
    anion_symbols: Iterable[str] = ("Se",),
    ligand_symbols: Iterable[str] = ("Cl",),
) -> Mapping[str, int]:
    """Summarise local coordination motifs in a completed graph.

    The inventory is intentionally side-effect free and does not reject a
    graph.  Existing graph rules decide legality; this inventory is used for
    diagnostics, stratified counts, and later motif-specific constraints.
    """

    cations = set(cation_symbols)
    anions = set(anion_symbols)
    ligands = set(ligand_symbols)
    out: Counter[str] = Counter()
    for left, right in state.graph.edges:
        symbols = {str(state.atoms[left].symbol), str(state.atoms[right].symbol)}
        if symbols == cations | anions and len(symbols) == 2:
            out["Cd-Se"] += 1
        elif symbols == cations | ligands and len(symbols) == 2:
            out["Cd-Cl"] += 1
    for atom in state.atoms:
        symbol = str(atom.symbol)
        degree = int(state.graph.degree[atom.atom_id])
        if symbol in ligands:
            role = {1: "Cl_t", 2: "Cl_b2", 3: "Cl_b3"}.get(
                degree, f"Cl_deg{degree}"
            )
            out[role] += 1
        elif symbol in cations or symbol in anions:
            typ = construction_atom_type(
                state,
                atom.atom_id,
                cation_symbols=cations,
                anion_symbols=anions,
                ligand_symbols=ligands,
            )
            out[f"{typ}:{local_coordination_signature(state, atom.atom_id, cation_symbols=cations, anion_symbols=anions, ligand_symbols=ligands)}"] += 1
    return dict(sorted(out.items()))

