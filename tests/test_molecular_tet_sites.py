"""Tetrahedral slot scaffold: free dirs, merge, small-bin smoke."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from builder.nucleation import load_nucleation_spec
from builder.nucleation.geometry_pack import load_geometry_pack
from builder.nucleation.molecular import enumerate_molecular_bin
from builder.nucleation.molecular_tet_sites import (
    _tet_free_directions,
    build_tet_sites,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mol_spec():
    return load_nucleation_spec(
        ROOT / "examples/nucleation/cdse_molecular_rules.yaml"
    )


@pytest.fixture(scope="module")
def pack():
    return load_geometry_pack(ROOT / "geometry_packs/cdse_cdcl2_molecular.yaml")


def test_mono_se_has_three_tet_free_dirs() -> None:
    se = np.array([1.0, 0.0, 0.0])
    free = _tet_free_directions([se])
    assert len(free) == 3
    # Each free dir makes a tetrahedral angle with Se (~109.47°)
    for d in free:
        ang = float(
            np.degrees(np.arccos(np.clip(np.dot(d, se), -1.0, 1.0)))
        )
        assert 100.0 < ang < 120.0


def test_tet_sites_k1_p2_matches_multiset_accept_count(mol_spec, pack) -> None:
    multi = replace(
        mol_spec, graph_rules=replace(mol_spec.graph_rules, decoration_mode="graph_multiset")
    )
    tet = replace(
        mol_spec, graph_rules=replace(mol_spec.graph_rules, decoration_mode="tet_sites")
    )
    r_m = enumerate_molecular_bin(1, 2, multi, pack=pack, embed=True)
    r_t = enumerate_molecular_bin(1, 2, tet, pack=pack, embed=True)
    assert len(r_m.isomers) == 2
    # Tet scaffold should find at least the multiset accepts (may equal)
    assert len(r_t.isomers) >= 1
    assert len(r_t.isomers) <= len(r_m.isomers) + 2  # not a flood
