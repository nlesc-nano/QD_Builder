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


def _occ(k, p, se_pts, cd_pts, oid):
    from builder.nucleation.molecular_zb_growth import ZbOccupation

    pts = np.vstack([se_pts, cd_pts])
    symbols = tuple(["Se"] * len(se_pts) + ["Cd"] * len(cd_pts))
    return ZbOccupation(
        k=k, p=p, symbols=symbols, coordinates=pts, core_edges=(), occupation_id=oid
    )


def _parent(k, p, sid, energy, occ, topology="preserved"):
    from builder.nucleation.molecular_growth import ParentStructure

    return ParentStructure(
        k=k, p=p, structure_id=sid, symbols=occ.symbols, coordinates=occ.coordinates,
        energy_eV=energy, edges=(), core_edges=(), zb_occupation=occ,
        topology_status=topology, propagation_eligible=topology in {"preserved", "extended"},
    )


def test_se_skeleton_family_selection_spans_p(spec):
    from dataclasses import replace

    from builder.nucleation.molecular_growth import GrowthConfig, select_parents

    growth = GrowthConfig.from_yaml(PACK / "growth_agnostic_k5.yaml")
    growth = replace(
        growth, family_mode="se_skeleton", families_cap=2, p_states_per_family=2,
        realizations_per_family=1, compact_rank_below_k=0, reserved_compact_seats=0,
    )
    a = 6.1405 / 4
    se_A = np.array([[0, 0, 0], [2 * a, 2 * a, 0], [2 * a, 0, 2 * a]])       # skeleton A
    se_B = np.array([[0, 0, 0], [2 * a, 2 * a, 0], [0, 2 * a, 2 * a]])       # skeleton A rotated (same family)
    se_C = np.array([[0, 0, 0], [2 * a, 2 * a, 0], [4 * a, 4 * a, 0]])       # skeleton C (different)
    cd1 = np.array([[a, a, a]])
    cd2 = np.array([[a, a, a], [3 * a, a, a]])
    # the k=3 overlay in growth_agnostic_k5.yaml sets min_p: 2, so use p=2,3
    parents = [
        # family A at p=2 (best in bin) and p=3 (rotated copy, best in bin)
        _parent(3, 2, "A_p2", -100.0, _occ(3, 2, se_A, cd1, "occA1")),
        _parent(3, 3, "A_p3", -130.0, _occ(3, 3, se_B, cd2, "occA2")),
        # family C at p=2 only, 0.3 eV above A, and an off-path member at p=3
        _parent(3, 2, "C_p2", -99.7, _occ(3, 2, se_C, cd1, "occC1")),
        _parent(3, 3, "C_p3_off", -129.0, _occ(3, 3, se_C, cd2, "occC2"), topology="changed"),
        # a third skeleton far above the window: dropped by families_cap=2
        _parent(3, 2, "D_p2", -95.0, _occ(3, 2, se_C * 1.5, cd1, "occD1")),
    ]
    selected = select_parents(parents, growth, spec)
    ids = {p.structure_id for p in selected}
    assert {"A_p2", "A_p3"} <= ids            # same Se skeleton at two p -> one family, both p carried
    assert "C_p2" in ids                      # second family kept
    assert "D_p2" not in ids                  # families_cap honoured
    assert "C_p3_off" not in ids              # off-path member not chosen while an in-path p exists


def test_cl_seed_radii_follow_table_only_when_opted_in():
    from builder.nucleation.molecular_zb_growth import _cdcl_seed_radii

    pack = load_geometry_pack(PACK / "run_gxtb.yaml")
    assert _cdcl_seed_radii(pack) == (2.45, 2.45)  # frozen g-xTB behaviour
    pack.raw["cl_seed"] = {"from_table": True}
    terminal, bridge = _cdcl_seed_radii(pack)
    assert terminal == 2.31 and bridge == 2.47  # CdCl_terminal cn3 / CdCl_bridge cn3,cn2


def test_growth_mace_yaml_loads():
    from builder.nucleation.molecular_growth import GrowthConfig

    growth = GrowthConfig.from_yaml(PACK / "growth_mace.yaml")
    assert growth.child_full_opt == "MACE-MH-1" and growth.child_full_opt_cycles == 500
    assert growth.family_mode == "se_skeleton"
    assert growth.references is not None and growth.references.method == "MACE-MH-1"
    win = growth.window_for(3)
    assert win.family_mode == "se_skeleton" and win.energy_window_eV == 0.80


def test_decoration_regimes_resolve_per_bin():
    from builder.nucleation.molecular_growth import decoration_regime

    gxtb = load_nucleation_spec(PACK / "run_gxtb.yaml")
    assert decoration_regime(5, 3, gxtb.graph_rules) is None  # no policy: historical behaviour
    mace = load_nucleation_spec(PACK / "run_mace_mh1.yaml")
    # p_max(5) = floor(2.7*5^(2/3)) + 1 = 8
    low = decoration_regime(5, 2, mace.graph_rules)
    mid = decoration_regime(5, 5, mace.graph_rules)
    high = decoration_regime(5, 8, mace.graph_rules)
    assert (low["name"], mid["name"], high["name"]) == ("low", "mid", "high")
    assert low["p_max"] == 8 and abs(high["p_fraction"] - 1.0) < 1e-9
    assert low["overrides"]["bridge_target_count_window"] == 1 and low["max_bridge_load_var"] is None
    assert high["overrides"]["selection_max_per_skeleton"] == 4 and high["max_bridge_load_var"] == 0.60
    # regimes never change the decoration mode (mu3 stays impossible at k>=3)
    assert "decoration_mode" not in high["overrides"]


def test_decoration_policy_rejects_unknown_regime_keys(tmp_path):
    import yaml

    from builder.nucleation.spec import _parse_decoration_policy

    ok = {"p_max": {"beta": 2.7, "slack": 1}, "regimes": [{"name": "a", "max_p_fraction": 0.5}, {"name": "b"}]}
    assert _parse_decoration_policy(ok)
    with pytest.raises(ValueError, match="unsupported keys"):
        _parse_decoration_policy({"regimes": [{"name": "a", "max_cn": {"Cd": 9}}]})
    with pytest.raises(ValueError, match="increasing"):
        _parse_decoration_policy({"regimes": [{"max_p_fraction": 0.7}, {"max_p_fraction": 0.4}]})


def test_bridge_load_variance():
    from types import SimpleNamespace

    from builder.nucleation.molecular_growth import bridge_load_variance

    # two Cd sharing one mu2 Cl and each carrying one terminal: loads (1,1) -> var 0
    iso = SimpleNamespace(symbols=["Se", "Cd", "Cd", "Cl", "Cl", "Cl"], edges=[(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (2, 5)], graph=None)
    assert bridge_load_variance(iso) == 0.0
    # both bridges on one Cd pair vs. none elsewhere: loads (2,2,0) -> var > 0
    iso = SimpleNamespace(symbols=["Se", "Cd", "Cd", "Cd", "Cl", "Cl"], edges=[(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (1, 5), (2, 5)], graph=None)
    assert bridge_load_variance(iso) > 0.8
