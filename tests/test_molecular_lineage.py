"""Lineage growth: correctness against exhaustive enumeration."""
from __future__ import annotations

from pathlib import Path

import pytest

from builder.nucleation.molecular import (
    _enumerate_inorganic_edge_sets,
    _molecular_check_spec,
)
from builder.nucleation.molecular_lineage import (
    core_certificate,
    grow_generation,
    shed_and_grow,
)
from builder.nucleation.spec import load_nucleation_spec

PACK = Path(__file__).resolve().parents[1] / "geometry_packs" / "cdse_cdcl2_motif_v1.yaml"


@pytest.fixture(scope="module")
def spec():
    return _molecular_check_spec(load_nucleation_spec(PACK))


def _exhaustive_certs(k, p, spec):
    sets, _ = _enumerate_inorganic_edge_sets(
        k, p, spec, max_skeletons=100000, extra_skeleton_edges=None
    )
    return {
        core_certificate(
            tuple(sorted((min(a, b), max(a, b)) for a, b in s)), k, p, spec
        )
        for s in sets
    }


def test_children_are_all_legal_cores(spec):
    """Every child must be a core the exhaustive enumerator would also accept."""
    parents = _exhaustive_certs(1, 2, spec)
    assert parents, "fixture precondition: k=1 p=2 has cores"
    sets, _ = _enumerate_inorganic_edge_sets(
        1, 2, spec, max_skeletons=1000, extra_skeleton_edges=None
    )
    edges = [tuple(sorted((min(a, b), max(a, b)) for a, b in s)) for s in sets]
    children = grow_generation(edges, k=1, p=2, p_out=2, spec=spec)
    exhaustive = _exhaustive_certs(2, 2, spec)
    produced = {core_certificate(c, 2, 2, spec) for c in children}
    # Soundness: no child outside the exhaustive set.  A spurious core would
    # mean the growth step invents an illegal skeleton.
    assert produced <= exhaustive


def test_shedding_reduces_precursor_count(spec):
    sets, _ = _enumerate_inorganic_edge_sets(
        2, 3, spec, max_skeletons=1000, extra_skeleton_edges=None
    )
    parent = tuple(sorted((min(a, b), max(a, b)) for a, b in sets[0]))
    children = shed_and_grow(parent, k=2, p=3, p_out=1, spec=spec)
    # (2,3) -> (3,1): 3 Se and 4 Cd expected in every child.
    for child in children:
        nodes = {n for edge in child for n in edge}
        assert max(nodes) < 3 + 3 + 1
    assert children


def test_dedup_is_by_isomorphism(spec):
    """Two runs over the same parents must not report isomorphic duplicates."""
    sets, _ = _enumerate_inorganic_edge_sets(
        1, 3, spec, max_skeletons=1000, extra_skeleton_edges=None
    )
    edges = [tuple(sorted((min(a, b), max(a, b)) for a, b in s)) for s in sets]
    children = grow_generation(edges, k=1, p=3, p_out=3, spec=spec)
    certs = [core_certificate(c, 2, 3, spec) for c in children]
    assert len(certs) == len(set(certs))
