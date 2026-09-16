"""Growth-driver changes for the MACE restart: two-tier cutoffs, topology gate, pre-relax screen."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from builder.nucleation.geometry_pack import load_geometry_pack
from builder.nucleation.mace_descriptors import MACE_CUTOFFS
from builder.nucleation.molecular_growth import (
    bond_cutoffs_from_spec,
    classify_core_topology,
    prerelax_wall_violations,
)
from builder.nucleation.spec import load_nucleation_spec

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "geometry_packs" / "cdse_cdcl2_zb"


@pytest.fixture(scope="module")
def spec():
    return load_nucleation_spec(PACK / "run_gxtb.yaml")


def test_bond_cutoffs_follow_pack_descriptor_model(spec):
    legacy = bond_cutoffs_from_spec(spec)
    assert legacy[("Cd", "Se")] == 3.25 and legacy[("Cd", "Cl")] == 2.90
    # g-xTB pack: unchanged
    gxtb = bond_cutoffs_from_spec(spec, load_geometry_pack(PACK / "run_gxtb.yaml"))
    assert gxtb == legacy
    # MACE pack (descriptor_cutoffs: two_tier): strong tier
    mace = bond_cutoffs_from_spec(spec, load_geometry_pack(PACK / "run_mace_mh1.yaml"))
    assert mace[("Cd", "Se")] == MACE_CUTOFFS["strong"][("Cd", "Se")]
    assert mace[("Cd", "Cl")] == MACE_CUTOFFS["strong"][("Cd", "Cl")]


def test_classify_core_topology():
    symbols = ["Se", "Se", "Cd", "Cd", "Cd"]
    expected = [(0, 2), (0, 3), (1, 3), (1, 4)]
    same = classify_core_topology(expected, expected, symbols, anion="Se", subgraph_gate=True)
    assert same == ("preserved", 0, 0)
    added = expected + [(0, 4)]
    assert classify_core_topology(expected, added, symbols, anion="Se", subgraph_gate=True) == ("extended", 0, 1)
    # the same added contact is a 'changed' topology under the legacy exact gate
    assert classify_core_topology(expected, added, symbols, anion="Se", subgraph_gate=False)[0] == "changed"
    lost = [(0, 2), (0, 3), (1, 3)]
    assert classify_core_topology(expected, lost, symbols, anion="Se", subgraph_gate=True) == ("changed", 1, 0)
    # losing the only bridge between the two Se splits the scaffold
    split = [(0, 2), (1, 4)]
    assert classify_core_topology(expected, split, symbols, anion="Se", subgraph_gate=True)[0] == "split"


def test_prerelax_wall_screen(spec):
    symbols = ["Se", "Se", "Cd", "Cl"]
    ok = np.array([[0, 0, 0], [4.3, 0, 0], [2.15, 1.5, 0], [2.15, 4.0, 0]])
    assert prerelax_wall_violations(symbols, ok, spec) == []
    # Se-Se dimer in the start: hits the pack wall (3.80) and would collapse
    dimer = ok.copy()
    dimer[1] = [2.4, 0, 0]
    tags = prerelax_wall_violations(symbols, dimer, spec)
    assert tags and tags[0].startswith("wall:Se-Se:0-1:2.40<3.80")
    # artifact floors raise a wall when they are stricter
    close_cl = ok.copy()
    close_cl[3] = [0.0, 2.75, 0.0]  # Cl-Se 2.75: above the 2.70 pair wall, below a 2.80 floor
    assert prerelax_wall_violations(symbols, close_cl, spec) == []
    assert prerelax_wall_violations(symbols, close_cl, spec, floors={"Cl-Se": 2.80})
