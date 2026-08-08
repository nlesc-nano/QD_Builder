"""Pack virtual-site decorator: CN refresh after bridges, small-bin smoke."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from builder.nucleation import load_nucleation_spec
from builder.nucleation.geometry_pack import load_geometry_pack
from builder.nucleation.molecular import enumerate_molecular_bin
from builder.nucleation.molecular_sites import terminal_directions_for_new_cn

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "geometry_packs/cdse_cdcl2_molecular.yaml"


@pytest.fixture(scope="module")
def mol_spec():
    return load_nucleation_spec(
        ROOT / "examples/nucleation/cdse_molecular_rules.yaml"
    )


@pytest.fixture(scope="module")
def pack():
    return load_geometry_pack(PACK)


def test_decoration_mode_from_pack(mol_spec) -> None:
    # Production default is graph_multiset (fastest at low–mid p).
    assert mol_spec.graph_rules.decoration_mode == "graph_multiset"


def test_terminal_dirs_after_bridge_use_cn3_not_linear(pack) -> None:
    """User rule: after a bridge, acceptor CN→3 terminal uses CN3 geometry.

    Pre-bridge mono-Se is nearly linear (CN2 completion opposite Se).  After one
    bridge the next terminal must *not* be collinear with Se; CN3 defaults give
    bent placements.
    """

    se_dir = np.array([1.0, 0.0, 0.0])
    # CN2 slot: only opposite Se
    cn2 = terminal_directions_for_new_cn([se_dir], new_cn=2, pack=pack)
    assert len(cn2) >= 1
    assert float(np.dot(cn2[0], se_dir)) < -0.9  # nearly opposite

    # After bridge: fixed = Se + bridge direction (e.g. roughly perpendicular)
    bridge_dir = np.array([0.0, 1.0, 0.0])
    cn3 = terminal_directions_for_new_cn(
        [se_dir, bridge_dir], new_cn=3, pack=pack
    )
    assert len(cn3) >= 1
    # None of the CN3 candidates should be the pure -Se linear hole alone as
    # the only option; at least one should have a significant z or mixed component
    # relative to the Se–bridge plane... stronger check: angle to Se not ~180
    # for the primary (first) candidate opposite the wedge.
    primary = cn3[0]
    angle_to_se = float(
        np.degrees(np.arccos(np.clip(np.dot(primary, se_dir), -1.0, 1.0)))
    )
    # CN3 completion is not collinear with Se (would be ~180°)
    assert angle_to_se < 170.0


def test_pack_sites_k1_p1_finds_isomers(mol_spec, pack) -> None:
    rules = replace(mol_spec.graph_rules, decoration_mode="pack_sites")
    spec = replace(mol_spec, graph_rules=rules)
    bin_res = enumerate_molecular_bin(1, 1, spec, pack=pack, embed=True)
    assert bin_res.raw_graphs > 0
    assert len(bin_res.isomers) >= 1
    for iso in bin_res.isomers:
        assert iso.coordinates is not None


def test_pack_sites_k1_p2_nonzero(mol_spec, pack) -> None:
    rules = replace(mol_spec.graph_rules, decoration_mode="pack_sites")
    spec = replace(mol_spec, graph_rules=rules)
    bin_res = enumerate_molecular_bin(1, 2, spec, pack=pack, embed=True)
    assert bin_res.raw_graphs > 0
    assert len(bin_res.isomers) >= 1


def test_tet_sites_k1_p1_and_p2(mol_spec, pack) -> None:
    rules = replace(mol_spec.graph_rules, decoration_mode="tet_sites")
    spec = replace(mol_spec, graph_rules=rules)
    r1 = enumerate_molecular_bin(1, 1, spec, pack=pack, embed=True)
    assert len(r1.isomers) >= 1
    r2 = enumerate_molecular_bin(1, 2, spec, pack=pack, embed=True)
    assert len(r2.isomers) >= 1
    for iso in r2.isomers:
        assert iso.coordinates is not None


def test_graph_multiset_default_k1_p1(mol_spec, pack) -> None:
    bin_res = enumerate_molecular_bin(1, 1, mol_spec, pack=pack, embed=True)
    assert len(bin_res.isomers) >= 1
