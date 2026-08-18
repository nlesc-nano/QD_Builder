"""Soft ranking terms: diamonds, F6 quality, terminal Se3Cl."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from builder.nucleation.molecular_growth import GrowthConfig
from builder.nucleation.soft_rules import (
    SoftDescriptors,
    SoftRulesConfig,
    _share_hist,
    describe_graph,
    describe_structure,
)

PACK = Path(__file__).resolve().parents[1] / "geometry_packs" / "cdse_cdcl2"


def _diamond_xyz():
    # Planar Cd2Se2: Cd–Se ~ 2.64 Å
    symbols = ["Se", "Se", "Cd", "Cd"]
    coords = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.00, 3.80, 0.00],
            [1.85, 1.90, 0.00],
            [-1.85, 1.90, 0.00],
        ],
        dtype=float,
    )
    return symbols, coords


def test_describe_counts_cdse_diamond_not_cl_rhombus() -> None:
    symbols, coords = _diamond_xyz()
    desc = describe_structure(symbols, coords)
    assert desc.n4 == 1
    assert desc.n4_fused == 0
    # add a μ2 Cl on the two Cd — that is a Cd–Se–Cd–Cl rhombus, not n4
    symbols2 = symbols + ["Cl"]
    coords2 = np.vstack([coords, [[0.0, 1.90, -2.4]]])
    desc2 = describe_structure(symbols2, coords2)
    assert desc2.n4 == 1


def test_f6_share_two_is_clean_share_four_is_dirty() -> None:
    # two 6-rings that share exactly two nodes
    a = frozenset({0, 1, 2, 3, 4, 5})
    b = frozenset({0, 1, 6, 7, 8, 9})
    c = frozenset({0, 1, 2, 3, 10, 11})  # share 4 with a
    h_clean = _share_hist([a, b])
    assert h_clean.get(2, 0) == 1
    assert sum(v for s, v in h_clean.items() if s >= 4) == 0
    h_dirty = _share_hist([a, c])
    assert h_dirty.get(4, 0) == 1


def test_terminal_se3cl_not_bridging() -> None:
    # Cd at origin, 3 Se in plane, one terminal Cl on +z
    symbols = ["Cd", "Se", "Se", "Se", "Cl"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.6, 0.0, 0.0],
            [-1.3, 2.25, 0.0],
            [-1.3, -2.25, 0.0],
            [0.0, 0.0, 2.5],
        ],
        dtype=float,
    )
    term = describe_structure(symbols, coords)
    assert term.n_term_se3cl == 1
    # second Cd bound to the same Cl → μ2, not terminal
    symbols_b = symbols + ["Cd"]
    coords_b = np.vstack([coords, [[0.0, 0.0, 5.0]]])
    brid = describe_structure(symbols_b, coords_b)
    assert brid.n_term_se3cl == 0


def test_penalty_signs_and_from_k() -> None:
    rules = SoftRulesConfig.from_raw(
        {
            "enabled": True,
            "diamond": {"enabled": True, "weight_eV": 0.15, "fused_weight_eV": 0.30, "from_k": 2},
            "f6": {"enabled": True, "weight_eV": -0.04, "dirty_weight_eV": 0.20, "from_k": 3},
            "terminal_se3cl": {"enabled": True, "weight_eV": 0.20, "from_k": 4},
            "se1cl3": {"enabled": False},
            "asphericity": {"enabled": False},
        }
    )
    dirty = SoftDescriptors(n4=2, n4_fused=2, f6_clean=5, f6_dirty=3, n_term_se3cl=1)
    # k=2: only diamond
    assert rules.penalty_eV(dirty, 2) == pytest.approx(0.15 * 2 + 0.30 * 2)
    # k=3: diamond + f6
    p3 = rules.penalty_eV(dirty, 3)
    assert p3 == pytest.approx(0.15 * 2 + 0.30 * 2 - 0.04 * 5 + 0.20 * 3)
    # k=4: plus terminal
    assert rules.penalty_eV(dirty, 4) == pytest.approx(p3 + 0.20)


def test_asphericity_only_if_dirty() -> None:
    rules = SoftRulesConfig.from_raw(
        {
            "enabled": True,
            "diamond": {"enabled": False},
            "f6": {"enabled": False},
            "terminal_se3cl": {"enabled": False},
            "se1cl3": {"enabled": False},
            "asphericity": {
                "enabled": True,
                "weight_eV": 2.0,
                "from_k": 1,
                "only_if_dirty_f6": True,
            },
        }
    )
    clean = SoftDescriptors(f6_dirty=0, asphericity=0.3)
    dirty = SoftDescriptors(f6_dirty=2, asphericity=0.3)
    assert rules.penalty_eV(clean, 8) == 0.0
    assert rules.penalty_eV(dirty, 8) == pytest.approx(0.6)


def test_growth_yaml_exposes_soft_rules() -> None:
    cfg = GrowthConfig.from_yaml(PACK / "growth_k9k13.yaml")
    assert cfg.soft_rules.enabled is True
    assert cfg.soft_rules.diamond.weight_eV == 0.15
    assert cfg.soft_rules.f6.extra["dirty_weight_eV"] == 0.20
    assert cfg.soft_rules.asphericity.enabled is False
    w8 = cfg.window_for(8)
    assert w8.soft_rules.enabled is True
    w13 = cfg.window_for(13)
    assert w13.soft_rules.diamond.from_k == 2


def test_graph_rules_expose_construction_bias() -> None:
    from builder.nucleation.spec import load_nucleation_spec
    import yaml

    driver = yaml.safe_load((PACK / "run_gxtb.yaml").read_text())
    rules = yaml.safe_load((PACK / "graph_rules.yaml").read_text())
    merged = {k: v for k, v in driver.items() if k != "include"}
    merged.update(rules)
    merged["cif"] = str(PACK.parents[1] / "examples" / "cifs" / "CdSe_zb.cif")
    path = PACK.parents[1] / "geometry_packs" / "cdse_cdcl2"  # unused
    tmp = Path(__file__).resolve().parents[1] / "examples" / "cifs" / "CdSe_zb.cif"
    import tempfile

    out = Path(tempfile.mkdtemp()) / "map.yaml"
    merged["cif"] = str(tmp)
    out.write_text(yaml.safe_dump(merged, sort_keys=False))
    spec = load_nucleation_spec(str(out))
    assert spec.graph_rules.reject_new_cdse_4rings is True
    assert spec.graph_rules.rank_cores_by_fusion is True
    assert spec.graph_rules.rank_decorations_by_motifs is True
    assert spec.graph_rules.construction_score["n4"] == 15
    assert spec.graph_rules.construction_score["f6_clean"] == -4


def test_describe_graph_matches_coord_counts() -> None:
    symbols, coords = _diamond_xyz()
    from_xyz = describe_structure(symbols, coords)
    edges = [(0, 2), (0, 3), (1, 2), (1, 3)]
    from_graph = describe_graph(symbols, edges)
    assert from_graph.n4 == from_xyz.n4 == 1
    assert from_graph.n4_fused == 0
