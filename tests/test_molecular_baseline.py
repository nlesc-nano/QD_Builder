"""Golden-file regression for lattice-free molecular (k, p) enumeration.

The enumeration is exact, so any change to the generator, the filters, or the
embedder must leave the accepted isomer set *and* its coordinates untouched
unless that change is intentional.  Optimisation work in particular has no
licence to move a single atom, and the fast paths it introduces are only sound
because of invariants that are easy to break silently -- hence a golden file
rather than a smoke test.

Regenerate deliberately, and review the diff, with::

    python tests/test_molecular_baseline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from builder.nucleation import load_nucleation_spec
from builder.nucleation.geometry_pack import load_geometry_pack
from builder.nucleation.molecular import enumerate_molecular_bin

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(__file__).with_name("molecular_map_baseline.json")
RUN_YAML = ROOT / "examples/nucleation/cdse_molecular_rules.yaml"

# Kept small enough to stay a test rather than a batch job; (2, 3) and beyond
# are covered by the invariant test below instead.
BINS = [(1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2)]


def _capture() -> dict:
    spec = load_nucleation_spec(RUN_YAML)
    pack = load_geometry_pack(spec.geometry_pack)
    captured = {}
    for k, p in BINS:
        result = enumerate_molecular_bin(k, p, spec, pack=pack, embed=True)
        captured[f"k{k}p{p}"] = {
            json.dumps(isomer.certificate, sort_keys=True, default=str): [
                [round(float(value), 9) for value in point]
                for point in (
                    () if isomer.coordinates is None else isomer.coordinates
                )
            ]
            for isomer in result.isomers
        }
    return captured


@pytest.fixture(scope="module")
def captured() -> dict:
    return _capture()


def test_accepted_isomers_match_baseline(captured) -> None:
    expected = json.loads(BASELINE.read_text())
    assert sorted(captured) == sorted(expected)
    for name in sorted(expected):
        lost = set(expected[name]) - set(captured[name])
        gained = set(captured[name]) - set(expected[name])
        assert not lost, f"{name}: {len(lost)} accepted isomer(s) disappeared"
        assert not gained, f"{name}: {len(gained)} unexpected new isomer(s)"


def test_embedded_coordinates_match_baseline(captured) -> None:
    expected = json.loads(BASELINE.read_text())
    for name in sorted(expected):
        for certificate, coordinates in expected[name].items():
            assert captured[name][certificate] == coordinates, (
                f"{name}: constructed coordinates changed for {certificate[:80]}"
            )


@pytest.mark.parametrize("k, p", [(1, 2), (1, 3)])
def test_acceptance_does_not_depend_on_atom_numbering(k: int, p: int) -> None:
    """Renumbering chemically identical atoms must not change the verdict.

    Construction order decides the geometry -- which anion seeds the pass, which
    template slot each neighbour takes -- and driving that by atom id made
    acceptance an artifact of numbering: 22 of 32 accepted structures at
    k=2, p=3 flipped to rejected under a permutation of like atoms.  The ranks
    in ``_canonical_ranks`` exist to stop that, and this is what pins it.
    """

    import random

    import networkx as nx

    from builder.nucleation.molecular import (
        bridge_feasibility_violations,
        embed_molecular_state,
        frame_violations,
        inorganic_coordinates,
        ExactEmbeddingError,
        _exact_bond_violations,
        _exact_local_geometry_violations,
    )
    from builder.nucleation.molecular_rules import molecular_geometry_ok
    from builder.nucleation.types import _State

    spec = load_nucleation_spec(RUN_YAML)
    pack = load_geometry_pack(spec.geometry_pack)
    result = enumerate_molecular_bin(k, p, spec, pack=pack, embed=True)
    assert result.isomers
    atoms = result.isomers[0].atoms
    by_element: dict = {}
    for atom in atoms:
        by_element.setdefault(atom.symbol, []).append(atom.atom_id)

    def accepted(edges) -> bool:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(atoms)))
        graph.add_edges_from(edges)
        state = _State(atoms=atoms, graph=graph)
        try:
            frame = inorganic_coordinates(state, pack, spec)
            if frame_violations(state, frame[0], pack, spec):
                return False
            if bridge_feasibility_violations(state, frame[0], pack, spec):
                return False
            coordinates = embed_molecular_state(
                state, pack, spec, inorganic=frame
            )
        except ExactEmbeddingError:
            return False
        if _exact_bond_violations(
            state, coordinates, pack, spec
        ) or _exact_local_geometry_violations(state, coordinates, pack, spec):
            return False
        return molecular_geometry_ok(state, coordinates, spec)[0]

    rng = random.Random(20240803)
    for isomer in result.isomers:
        for _ in range(5):
            permutation = {}
            for ids in by_element.values():
                shuffled = ids[:]
                rng.shuffle(shuffled)
                permutation.update(dict(zip(ids, shuffled)))
            relabelled = [
                (permutation[u], permutation[v]) for u, v in isomer.graph.edges
            ]
            assert accepted(relabelled), (
                f"{isomer.structure_id} is accepted as numbered but rejected "
                "after permuting chemically identical atoms"
            )


@pytest.mark.parametrize("k, p", [(1, 3), (2, 2)])
def test_reduced_graph_check_agrees_with_full_check(k: int, p: int) -> None:
    """The per-decoration fast path must equal the full rule set.

    ``enumerate_molecular_bin`` only re-checks the cation coordination floor per
    decoration; everything else in ``molecular_graph_violations`` is either a
    skeleton invariant or already guaranteed by the generator.  If that ever
    stops holding, the fast path would silently accept illegal graphs.
    """

    spec = load_nucleation_spec(RUN_YAML)
    pack = load_geometry_pack(spec.geometry_pack)
    fast = enumerate_molecular_bin(k, p, spec, pack=pack, embed=True)
    full = enumerate_molecular_bin(
        k, p, spec, pack=pack, embed=True, validate_every_graph=True
    )
    assert [isomer.certificate for isomer in fast.isomers] == [
        isomer.certificate for isomer in full.isomers
    ]
    assert fast.rejected == full.rejected
    assert fast.rejection_reasons == full.rejection_reasons


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    BASELINE.write_text(json.dumps(_capture(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {BASELINE}")
