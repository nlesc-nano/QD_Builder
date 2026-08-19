"""Zinc-blende occupation growth: snap, attach, reject diamonds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from builder.nucleation.molecular_growth import (
    GrowthConfig,
    ParentStructure,
    grow_cores_from_parents,
)
from builder.nucleation.molecular_zb_growth import (
    attach_cdse,
    grow_zb_children,
    lattice_k1_occupation,
    lattice_model,
    place_cl_2p,
    seed_occupation,
    zb_embeddable,
)
from builder.nucleation.soft_rules import describe_structure
from builder.nucleation.spec import load_nucleation_spec

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "geometry_packs" / "cdse_cdcl2"
PACK_ZB = ROOT / "geometry_packs" / "cdse_cdcl2_zb"


@pytest.fixture(scope="module")
def map_spec(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("zb_spec")
    driver = yaml.safe_load((PACK / "run_gxtb.yaml").read_text())
    rules = yaml.safe_load((PACK / "graph_rules.yaml").read_text())
    merged = {k: v for k, v in driver.items() if k != "include"}
    merged.update(yaml.safe_load((PACK / "motifs.yaml").read_text()))
    merged.update(yaml.safe_load((PACK / "embed.yaml").read_text()))
    merged.update(rules)
    merged["cif"] = str(ROOT / "examples/cifs/CdSe_zb.cif")
    merged.setdefault("relaxation", {})["enabled"] = False
    path = tmp / "map.yaml"
    path.write_text(yaml.safe_dump(merged, sort_keys=False))
    return load_nucleation_spec(str(path))


def test_seed_and_k1_occupation(map_spec) -> None:
    model = lattice_model(map_spec)
    seed = seed_occupation(map_spec, model)
    assert seed.k == 1 and seed.p == 0
    assert len(seed.core_edges) == 1
    occ = lattice_k1_occupation(map_spec, model, p=2)
    assert occ is not None
    assert occ.k == 1 and occ.p == 2
    assert occ.symbols.count("Se") == 1
    assert occ.symbols.count("Cd") == 3


def test_attach_cdse_has_no_diamond(map_spec) -> None:
    model = lattice_model(map_spec)
    parent = lattice_k1_occupation(map_spec, model, p=1)
    assert parent is not None
    kids = attach_cdse(parent, map_spec, model, cap=20)
    assert kids
    for kid in kids:
        assert kid.k == 2
        assert kid.p == 1
        desc = describe_structure(kid.symbols, kid.coordinates, map_spec)
        assert desc.n4 == 0
        ok, emb, why = zb_embeddable(
            kid.symbols, kid.coordinates, map_spec, model
        )
        assert ok, why
        assert emb is not None


def test_grow_zb_children_stoich(map_spec) -> None:
    model = lattice_model(map_spec)
    parent = lattice_k1_occupation(map_spec, model, p=2)
    assert parent is not None
    kids = grow_zb_children(
        parent, s=1, p_m=1, spec=map_spec, model=model, cap=12
    )
    assert kids
    for kid in kids:
        assert kid.k == 2
        assert kid.p == 2  # 2 - 1 + 1
        assert kid.symbols.count("Se") == 2
        assert kid.symbols.count("Cd") == 4


def test_diamond_is_not_zb_embeddable(map_spec) -> None:
    model = lattice_model(map_spec)
    symbols = ["Se", "Se", "Cd", "Cd"]
    coords = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.00, 3.80, 0.00],
            [1.85, 1.90, 0.00],
            [-1.85, 1.90, 0.00],
        ]
    )
    assert describe_structure(symbols, coords, map_spec).n4 == 1
    ok, occ, why = zb_embeddable(symbols, coords, map_spec, model)
    assert not ok
    assert occ is None
    assert why.startswith("n4")


def test_place_cl_2p_count(map_spec) -> None:
    model = lattice_model(map_spec)
    occ = lattice_k1_occupation(map_spec, model, p=2)
    assert occ is not None
    symbols, coords, edges = place_cl_2p(occ, map_spec)
    assert symbols.count("Cl") == 4
    assert coords.shape[0] == len(symbols)


def test_zb_pack_is_clean() -> None:
    spec = load_nucleation_spec(str(PACK_ZB / "run_gxtb.yaml"))
    rules = spec.graph_rules
    assert rules.reject_new_cdse_4rings is False
    assert rules.rank_cores_by_fusion is False
    assert rules.rank_decorations_by_motifs is False
    assert not getattr(rules, "min_ring_size", None) or not rules.min_ring_size.get("Cd-Se")
    cfg = GrowthConfig.from_yaml(PACK_ZB / "growth.yaml")
    assert cfg.soft_rules.enabled is False
    assert cfg.move_zb_sites is True
    assert cfg.move_graph is False
    assert cfg.move_coord is False
    w1 = cfg.window_for(1)
    w3 = cfg.window_for(3)
    assert w1.move_zb_sites is True
    assert w1.soft_rules.enabled is False
    assert w3.min_p_parent == 2
    assert w3.energy_window_eV == 0.60
    # this pack only grows through parent k=3 (child k=4)
    w4 = cfg.window_for(4)
    assert w4.move_zb_sites is True


def test_grow_cores_zb_move(map_spec) -> None:
    model = lattice_model(map_spec)
    occ = lattice_k1_occupation(map_spec, model, p=1)
    assert occ is not None
    parent = ParentStructure(
        k=1,
        p=1,
        structure_id="k001_p001_zb",
        symbols=occ.symbols,
        coordinates=occ.coordinates,
        energy_eV=-1.0,
        edges=occ.core_edges,
        core_edges=occ.core_edges,
    )
    cfg = GrowthConfig.from_yaml(PACK / "growth_k1k13.yaml")
    result = grow_cores_from_parents([parent], growth=cfg, spec=map_spec)
    assert result.zb_seeds
    assert result.zb_stats is not None
    assert result.zb_stats.snapped >= 1
    moves = {ch.move for ch in result.channels}
    assert "zb_sites" in moves
    assert "coord" not in moves
    kids = [c for occs in result.zb_seeds.values() for c in occs]
    assert kids
    assert all(c.k == 2 for c in kids)


def test_grow_cores_uses_zb_pack_growth(map_spec) -> None:
    """The clean k1k4 pack grows Z children and does not enable A/B."""

    model = lattice_model(map_spec)
    occ = lattice_k1_occupation(map_spec, model, p=1)
    assert occ is not None
    parent = ParentStructure(
        k=1,
        p=1,
        structure_id="k001_p001_zb",
        symbols=occ.symbols,
        coordinates=occ.coordinates,
        energy_eV=-1.0,
        edges=occ.core_edges,
        core_edges=occ.core_edges,
    )
    cfg = GrowthConfig.from_yaml(PACK_ZB / "growth.yaml")
    result = grow_cores_from_parents([parent], growth=cfg, spec=map_spec)
    assert {ch.move for ch in result.channels} == {"zb_sites"}
    assert result.zb_seeds
    assert not result.coord_seeds or not any(result.coord_seeds.values())
