"""Molecular package growth: parents, packages, core inflation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from builder.nucleation.molecular_growth import (
    GrowthConfig,
    GrowthLog,
    ParentStructure,
    full_opt_relaxation_raw,
    grow_cores_from_parents,
    identify_packages,
    parent_k_inventory,
    select_parents,
)
from builder.nucleation.molecular_lineage import shed_and_grow
from builder.nucleation.spec import load_nucleation_spec

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "geometry_packs" / "cdse_cdcl2"
GROWTH_YAML = PACK / "growth.yaml"


@pytest.fixture(scope="module")
def map_spec(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("growth_spec")
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


def test_parent_k_inventory_from_index(tmp_path) -> None:
    (tmp_path / "index.csv").write_text(
        "k,p,structure_id,xtb_energy_eV,xtb_converged\n"
        "3,2,a,-1.0,True\n"
        "3,3,b,-1.0,False\n"
        "3,4,c,-1.0,True\n"
        "5,1,d,-1.0,True\n",
        encoding="utf-8",
    )
    assert parent_k_inventory(tmp_path) == {3: 2, 5: 1}


def test_growth_config_loads() -> None:
    cfg = GrowthConfig.from_yaml(GROWTH_YAML)
    assert cfg.monomer_p_values == (1, 2, 3)
    assert cfg.max_shed >= 1
    assert cfg.references is not None
    assert cfg.references.energy_cdse_eV < 0
    assert cfg.use_coord_carry
    assert cfg.local_cleanup_cycles == 20
    assert cfg.child_full_opt_cycles == 150
    assert cfg.shed_mode == "wbo"
    assert cfg.soft_rules.enabled is True
    assert cfg.soft_rules.asphericity.enabled is False


def test_full_opt_cycles_cap_not_cleanup() -> None:
    cfg = GrowthConfig.from_yaml(GROWTH_YAML)
    raw = full_opt_relaxation_raw(None, cfg)
    assert raw["max_steps"] == 150
    assert raw["method"] == "g-xTB"
    # cleanup still uses its own 20-cycle cap, even if the pack said 500
    assert cfg.local_cleanup_cycles == 20
    cfg.child_full_opt_cycles = 80
    assert full_opt_relaxation_raw(None, cfg)["max_steps"] == 80


def test_shed_packages_coords_and_place(map_spec) -> None:
    from builder.nucleation.molecular_growth import (
        place_monomer_and_packages,
        shed_packages_coords,
    )

    symbols = ("Se", "Cd", "Cd", "Cl", "Cl")
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 2.3, 0.0],
            [5.0, -2.3, 0.0],
        ],
        dtype=float,
    )
    edges = ((0, 1), (1, 2), (2, 3), (2, 4))
    parent = ParentStructure(
        k=1,
        p=1,
        structure_id="fake",
        symbols=symbols,
        coordinates=coords,
        energy_eV=-1.0,
        edges=edges,
        core_edges=((0, 1), (1, 2)),
        wbo={(2, 3): 0.5, (2, 4): 0.4},
    )
    pkgs = identify_packages(parent, map_spec)
    # s=0 keep all
    s0, c0, e0, sc0 = shed_packages_coords(parent, s=0, packages=pkgs)
    assert len(s0) == 5 and sc0 == ()
    # s=1 removes the package Cd+2Cl → Se + core Cd remain
    s1, c1, e1, sc1 = shed_packages_coords(parent, s=1, packages=pkgs)
    assert len(s1) == 2
    assert set(s1) == {"Se", "Cd"}
    assert len(sc1) == 1
    # place monomer + p_m=1 package
    s2, c2, e2 = place_monomer_and_packages(
        s1,
        c1,
        e1,
        k_parent=1,
        p_after_shed=0,
        p_m=1,
        spec=map_spec,
        pack=None,
    )
    # Se, Cd (core host), + Se, Cd (monomer), + Cd, Cl, Cl (package)
    assert s2.count("Se") == 2
    assert s2.count("Cd") == 3
    assert s2.count("Cl") == 2
    assert c2.shape[0] == len(s2)


def test_identify_packages_two_cl(map_spec) -> None:
    # Se0, Cd1 (core), Cd2 (precursor), Cl3, Cl4  — minimal fake at k=1 p=1
    # Standard layout k=1 p=1: Se, Cd_core, Cd_pre, Cl, Cl
    symbols = ("Se", "Cd", "Cd", "Cl", "Cl")
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 2.3, 0.0],
            [5.0, -2.3, 0.0],
        ],
        dtype=float,
    )
    edges = ((0, 1), (1, 2), (2, 3), (2, 4))  # Cd2 has two Cl
    parent = ParentStructure(
        k=1,
        p=1,
        structure_id="fake",
        symbols=symbols,
        coordinates=coords,
        energy_eV=-1.0,
        edges=edges,
        core_edges=((0, 1), (1, 2)),
        wbo={(2, 3): 0.5, (2, 4): 0.4},
    )
    pkgs = identify_packages(parent, map_spec)
    assert len(pkgs) >= 1
    assert pkgs[0].cd == 2
    # weaker sum WBO sheds first when sorted ascending (plus tiny Se bias)
    assert pkgs[0].score == pytest.approx(0.9, abs=0.05)


def test_identify_packages_nonoverlapping() -> None:
    """Shared Cl must not make two packages claim the same ligand."""
    from builder.nucleation.molecular_growth import identify_packages
    from builder.nucleation.spec import load_nucleation_spec
    import yaml
    from pathlib import Path

    # Two Cd share one Cl → only one package can be kept
    # Se0, Cd1 (core), Cd2, Cd3, Cl4, Cl5, Cl6  (Cl5 shared if both want it)
    symbols = ("Se", "Cd", "Cd", "Cd", "Cl", "Cl", "Cl")
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 3.0, 0.0],
            [5.0, -2.3, 0.0],
            [5.0, 1.5, 0.0],  # shared bridge-like Cl
            [5.0, 5.0, 0.0],
        ],
        dtype=float,
    )
    edges = (
        (0, 1),
        (1, 2),
        (2, 4),
        (2, 5),  # package A: Cd2-Cl4,Cl5
        (3, 5),
        (3, 6),  # package B: Cd3-Cl5,Cl6  shares Cl5
    )
    parent = ParentStructure(
        k=1,
        p=2,
        structure_id="overlap",
        symbols=symbols,
        coordinates=coords,
        energy_eV=-1.0,
        edges=edges,
        core_edges=((0, 1), (1, 2)),
        wbo=None,
    )
    # minimal spec-like object via real pack
    ROOT = Path(__file__).resolve().parents[1]
    PACK = ROOT / "geometry_packs" / "cdse_cdcl2"
    driver = yaml.safe_load((PACK / "run_gxtb.yaml").read_text())
    rules = yaml.safe_load((PACK / "graph_rules.yaml").read_text())
    merged = {k: v for k, v in driver.items() if k != "include"}
    merged.update(yaml.safe_load((PACK / "motifs.yaml").read_text()))
    merged.update(yaml.safe_load((PACK / "embed.yaml").read_text()))
    merged.update(rules)
    merged["cif"] = str(ROOT / "examples/cifs/CdSe_zb.cif")
    merged.setdefault("relaxation", {})["enabled"] = False
    import tempfile
    from pathlib import Path as P

    tmp = P(tempfile.mkdtemp()) / "s.yaml"
    tmp.write_text(yaml.safe_dump(merged, sort_keys=False))
    map_spec = load_nucleation_spec(str(tmp))
    pkgs = identify_packages(parent, map_spec)
    atoms = []
    for pk in pkgs:
        atoms.extend([pk.cd, pk.cl[0], pk.cl[1]])
    assert len(atoms) == len(set(atoms)), "packages must be atom-disjoint"
    assert len(pkgs) == 1  # only one non-overlapping package fits


def test_select_parents_respects_window_and_dec(map_spec) -> None:
    def make(i, e, p=2):
        # k=2 p=2: 2 Se + 4 Cd + 4 Cl = 10 atoms — stub coords
        n = 2 + 4 + 4
        return ParentStructure(
            k=2,
            p=p,
            structure_id=f"m{i}",
            symbols=tuple(["Se"] * 2 + ["Cd"] * 4 + ["Cl"] * 4),
            coordinates=np.zeros((n, 3)),
            energy_eV=e,
            edges=(),
            core_edges=((0, 2), (0, 3), (1, 3), (1, 4)),
        )

    parents = [
        make(0, 0.0),
        make(1, 0.2),
        make(2, 0.5),
        make(3, 2.0),  # outside 1 eV window
    ]
    cfg = GrowthConfig.from_yaml(GROWTH_YAML)
    # force small diversity
    cfg = GrowthConfig(
        raw=cfg.raw,
        monomer_p_values=cfg.monomer_p_values,
        references=cfg.references,
        energy_window_eV=1.0,
        decorations_per_skeleton=2,
        max_skeletons_frac=1.0,
        max_skeletons_cap=10,
        max_shed=cfg.max_shed,
    )
    sel = select_parents(parents, cfg, map_spec)
    ids = {p.structure_id for p in sel}
    assert "m3" not in ids
    assert "m0" in ids


def test_select_parents_prefers_fewer_diamonds(map_spec) -> None:
    from builder.nucleation.soft_rules import describe_structure

    def parent(sid, energy, coords, symbols):
        return ParentStructure(
            k=3,
            p=2,
            structure_id=sid,
            symbols=tuple(symbols),
            coordinates=np.asarray(coords, dtype=float),
            energy_eV=energy,
            edges=(),
            core_edges=(),
        )

    free_sym = ["Se", "Cd", "Cd"]
    free_xyz = np.array([[0.0, 0.0, 0.0], [2.6, 0.0, 0.0], [0.0, 2.6, 0.0]])
    d_sym = ["Se", "Se", "Cd", "Cd"]
    d_xyz = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.00, 3.80, 0.00],
            [1.85, 1.90, 0.00],
            [-1.85, 1.90, 0.00],
        ],
        dtype=float,
    )
    assert describe_structure(d_sym, d_xyz).n4 == 1
    assert describe_structure(free_sym, free_xyz).n4 == 0
    cfg = GrowthConfig.from_yaml(PACK / "growth_k2k3.yaml")
    cfg = GrowthConfig(
        raw=cfg.raw,
        energy_window_eV=1.0,
        max_skeletons_frac=1.0,
        max_skeletons_cap=10,
        decorations_per_skeleton=1,
        soft_rules=cfg.soft_rules,
    )
    sel = select_parents(
        [
            parent("diamond", 0.0, d_xyz, d_sym),
            parent("free", 0.05, free_xyz, free_sym),
        ],
        cfg,
        map_spec,
    )
    assert sel[0].structure_id == "free"


def test_surface_law_matches_lattice_engine() -> None:
    from builder.nucleation.engine import _p_surf
    from builder.nucleation.molecular_growth import (
        effective_s_max,
        p_surf_capacity,
    )

    for beta in (1.5, 2.0, 2.5):
        for k in range(1, 13):
            assert p_surf_capacity(k, beta) == _p_surf(k, beta)
    # β=2: p_surf(2)=3, (3)=4, (7)=7.
    # 8^{2/3} is 4 in exact arithmetic but 3.999… in float, so floor(2·) = 7.
    assert p_surf_capacity(2, 2.0) == 3
    assert p_surf_capacity(3, 2.0) == 4
    assert p_surf_capacity(7, 2.0) == 7
    assert p_surf_capacity(8, 2.0) == 7
    # s_max = min(p, floor(α p_surf), hard).  At k=7, p_surf=7 so the
    # surface term is not the tight bound — YAML max_shed (hard) is.
    assert effective_s_max(7, 7, beta=2.0, alpha=1.0, hard=2) == 2
    assert effective_s_max(7, 7, beta=2.0, alpha=1.0, hard=1) == 1
    assert effective_s_max(2, 5, beta=2.0, alpha=1.0, hard=2) == 2
    assert effective_s_max(4, 4, beta=0.0, alpha=1.0, hard=2) == 2


def test_k2k3_wide_parent_window() -> None:
    from builder.nucleation.molecular_growth import GrowthConfig

    cfg = GrowthConfig.from_yaml(PACK / "growth_k2k3.yaml")
    w2 = cfg.window_for(2)
    w3 = cfg.window_for(3)
    assert w2.monomer_p_values == (1, 2, 3)
    assert w2.max_shed == 2
    assert w2.energy_window_eV == 2.0
    assert w2.max_skeletons_cap == 50
    assert w2.max_skeletons_frac == 1.0
    assert w2.child_redecorate is True
    assert not w2.child_redecorate_slack
    assert w2.allow_redecorate(3, w2.p_surf(3))
    assert not w2.allow_redecorate(3, w2.p_surf(3) + 1)
    assert w2.selection_max_per_skeleton == 6
    assert w3.energy_window_eV == 2.0
    assert w3.max_shed == 2
    assert w3.child_redecorate is True


def test_k3k13_one_shot_windows() -> None:
    from builder.nucleation.molecular_growth import GrowthConfig

    cfg = GrowthConfig.from_yaml(PACK / "growth_k3k13.yaml")
    w3 = cfg.window_for(3)
    w5 = cfg.window_for(5)
    w6 = cfg.window_for(6)
    w8 = cfg.window_for(8)
    w11 = cfg.window_for(11)
    w12 = cfg.window_for(12)
    assert w3.child_redecorate is True
    assert w3.move_graph is True
    assert w3.max_shed == 2
    assert w3.monomer_p_values == (1, 2)
    assert w3.energy_window_eV == 2.0
    assert w5.child_redecorate is False
    assert w5.move_graph is False
    assert w5.move_coord is True
    assert w5.max_shed == 2
    assert w6.max_shed == 1
    assert w6.energy_window_eV == 1.0
    assert w8.monomer_p_values == (1,)
    assert w8.max_skeletons_cap == 15
    assert w11.energy_window_eV == 0.75
    assert w12.max_shed == 0
    assert w12.p_slack == 0
    assert w12.energy_window_eV == 0.50
    assert cfg.soft_rules.enabled is True
    assert cfg.soft_rules.asphericity.enabled is False
    assert w8.soft_rules.diamond.weight_eV == 0.15


def test_by_k_window_picks_k7() -> None:
    from builder.nucleation.molecular_growth import GrowthConfig

    cfg = GrowthConfig.from_yaml(PACK / "growth_k4k8.yaml")
    w4 = cfg.window_for(4)
    w7 = cfg.window_for(7)
    assert w4.monomer_p_values == (1, 2)
    assert w4.max_shed == 2
    assert w4.energy_window_eV == 2.0
    assert w4.max_skeletons_cap == 50
    assert w4.max_skeletons_frac == 1.0
    assert w7.monomer_p_values == (1,)
    assert w7.max_shed == 1
    assert w7.energy_window_eV == 1.0
    assert w7.attach == "local"
    assert not w7.allow_p_child(8, w7.p_surf(8) + w7.p_slack + 1)
    assert w7.allow_p_child(8, w7.p_surf(8))
    assert w4.child_redecorate is False
    assert not w4.allow_redecorate(4, w4.p_surf(4))
    assert w4.allow_p_child(4, w4.p_surf(4) + w4.p_slack)
    w5 = cfg.window_for(5)
    assert w5.child_redecorate is False
    assert w5.max_shed == 2
    assert not w5.allow_redecorate(6, w5.p_surf(6))


def test_by_k_window_k9k13() -> None:
    from builder.nucleation.molecular_growth import GrowthConfig

    cfg = GrowthConfig.from_yaml(PACK / "growth_k9k13.yaml")
    w9 = cfg.window_for(9)
    w12 = cfg.window_for(12)
    w13 = cfg.window_for(13)
    assert w9.monomer_p_values == (1,)
    assert w9.max_shed == 1
    assert w9.child_redecorate is True
    assert w9.attach == "local"
    assert w9.selection_max_per_skeleton == 3
    assert w9.max_children_per_channel == 20
    assert w9.max_opts_per_k == 80
    assert w12.max_shed == 0
    assert w12.child_redecorate is False
    assert w13.max_shed == 0
    assert w13.child_redecorate is False
    assert w13.energy_window_eV == 0.25
    # p_slack=0: child p may not exceed p_surf(k+1)
    assert w9.allow_p_child(10, w9.p_surf(10))
    assert not w9.allow_p_child(10, w9.p_surf(10) + 1)


def test_survey_yaml_budget_and_attach() -> None:
    from builder.nucleation.molecular_growth import GrowthConfig

    cfg = GrowthConfig.from_yaml(PACK / "growth_survey.yaml")
    w = cfg.window_for(3)
    assert w.attach == "enumerate"
    assert w.max_opts_per_k == 120
    assert w.surface_beta == 2.5


def test_opt_budget_caps_seeds_then_cores() -> None:
    from builder.nucleation.molecular_growth import (
        CoordSeed,
        GrowthStepResult,
        _apply_opt_budget,
    )

    seeds = {
        (3, 2): [
            CoordSeed(
                k=3,
                p=2,
                structure_id=f"s{i}",
                parent_id="p",
                shed=0,
                p_m=1,
                symbols=("Se",),
                coordinates=np.zeros((1, 3)),
                core_edges=(),
            )
            for i in range(5)
        ]
    }
    catalog = {(3, 2): [((0, 1),), ((0, 2),), ((0, 3),)]}
    result = GrowthStepResult(
        k_from=2,
        k_to=3,
        parents_selected=1,
        channels=[],
        skeleton_catalog=catalog,
        coord_seeds=seeds,
    )
    capped = _apply_opt_budget(result, 4)
    assert sum(len(v) for v in capped.coord_seeds.values()) == 4
    assert sum(len(v) for v in capped.skeleton_catalog.values()) == 0
    result2 = GrowthStepResult(
        k_from=2,
        k_to=3,
        parents_selected=1,
        channels=[],
        skeleton_catalog=catalog,
        coord_seeds=seeds,
    )
    capped2 = _apply_opt_budget(result2, 7)
    assert sum(len(v) for v in capped2.coord_seeds.values()) == 5
    assert sum(len(v) for v in capped2.skeleton_catalog.values()) == 2


def test_wbo_persist_and_reload(tmp_path) -> None:
    from builder.nucleation.molecular_growth import (
        load_parent_wbo,
        write_wbo_file,
    )

    orders = [
        [0.0, 0.8, 0.1],
        [0.8, 0.0, 0.0],
        [0.1, 0.0, 0.0],
    ]
    xyz = tmp_path / "k002_p001_mol0001_xtb.xyz"
    xyz.write_text("1\ntest\nSe 0 0 0\n")
    write_wbo_file(xyz.with_suffix(".wbo"), orders)
    parsed, src = load_parent_wbo(
        xyz, structure_id="k002_p001_mol0001", run_dir=tmp_path
    )
    assert src == "wbo_file"
    assert parsed[(0, 1)] == pytest.approx(0.8)
    # missing wbo → none (distance fallback is the caller)
    empty = tmp_path / "other.xyz"
    empty.write_text("1\nx\nSe 0 0 0\n")
    parsed2, src2 = load_parent_wbo(
        empty, structure_id="missing", run_dir=tmp_path
    )
    assert parsed2 is None
    assert src2 == "none"


def test_grow_cores_respects_p_surf_cap(map_spec) -> None:
    """High p_m channels above p_surf(k+1)+slack are dropped."""
    from builder.nucleation.molecular import _enumerate_inorganic_edge_sets
    from builder.nucleation.molecular_growth import GrowthConfig

    sets, _ = _enumerate_inorganic_edge_sets(
        2, 2, map_spec, max_skeletons=20, mode="free"
    )
    if not sets:
        pytest.skip("no free skeletons at k2p2")
    core = tuple(sorted((min(a, b), max(a, b)) for a, b in sets[0]))
    symbols = tuple(["Se"] * 2 + ["Cd"] * 4 + ["Cl"] * 4)
    parent = ParentStructure(
        k=2,
        p=2,
        structure_id="core0",
        symbols=symbols,
        coordinates=np.random.RandomState(0).randn(len(symbols), 3) * 2.0,
        energy_eV=-100.0,
        edges=core,
        core_edges=core,
    )
    cfg = GrowthConfig.from_yaml(GROWTH_YAML)
    cfg = GrowthConfig(
        raw=cfg.raw,
        monomer_p_values=(1, 2, 3),
        references=cfg.references,
        energy_window_eV=5.0,
        max_shed=1,
        prefer_low_shed=True,
        max_children_per_channel=20,
        decorations_per_skeleton=1,
        max_skeletons_frac=1.0,
        max_skeletons_cap=5,
        start_from="graph_only",
        local_cleanup_enabled=False,
        surface_beta=2.0,
        surface_alpha=1.0,
        p_slack=0,
        move_coord=False,
        attach="local",
    )
    # p_surf(3)=4, slack=0 → p_child ≤ 4.  parent p=2, s=0, p_m=3 → 5 dropped.
    result = grow_cores_from_parents([parent], growth=cfg, spec=map_spec)
    child_ps = {ch.p_child for ch in result.channels}
    assert 5 not in child_ps
    assert all(p <= 4 for p in child_ps)


def test_grow_cores_produces_child_bins(map_spec) -> None:
    """Synthetic connected parent core grows to k+1 cores for packages."""
    # Use real shed_and_grow path: need legal parent core at k=2 p=2
    from builder.nucleation.molecular import _enumerate_inorganic_edge_sets

    sets, _ = _enumerate_inorganic_edge_sets(
        2, 2, map_spec, max_skeletons=20, mode="free"
    )
    if not sets:
        pytest.skip("no free skeletons at k2p2")
    core = tuple(
        sorted((min(a, b), max(a, b)) for a, b in sets[0])
    )
    # Build a ParentStructure with standard symbols and core edges
    symbols = tuple(["Se"] * 2 + ["Cd"] * 4 + ["Cl"] * 4)
    n = len(symbols)
    parent = ParentStructure(
        k=2,
        p=2,
        structure_id="core0",
        symbols=symbols,
        coordinates=np.random.RandomState(0).randn(n, 3) * 2.0,
        energy_eV=-100.0,
        edges=core,
        core_edges=core,
    )
    cfg = GrowthConfig.from_yaml(GROWTH_YAML)
    cfg = GrowthConfig(
        raw=cfg.raw,
        monomer_p_values=(1, 2),
        references=cfg.references,
        energy_window_eV=5.0,
        max_shed=1,
        prefer_low_shed=True,
        max_children_per_channel=50,
        decorations_per_skeleton=1,
        max_skeletons_frac=1.0,
        max_skeletons_cap=5,
        start_from="relaxed_coords",
        local_cleanup_enabled=False,  # no gxtb in unit test
    )
    # parent_core_in_blocks needs matching layout — force core as block edges
    result = grow_cores_from_parents([parent], growth=cfg, spec=map_spec)
    assert result.k_to == 3
    # expect some child bins
    assert result.skeleton_catalog or result.channels
    # composition check: p_child = 2 - s + p_m
    for ch in result.channels:
        assert ch.p_child == ch.p_parent - ch.shed + ch.p_m
    # move B seeds present when start_from=relaxed_coords
    assert result.coord_seeds
    moves = {ch.move for ch in result.channels}
    assert "graph" in moves
    assert "coord" in moves
    from builder.nucleation.molecular_growth import parse_compact_serial

    for seeds in result.coord_seeds.values():
        for seed in seeds:
            assert parse_compact_serial(seed.structure_id) is not None
            assert seed.structure_id.startswith(f"k{seed.k:03d}_p{seed.p:03d}_B")
            assert "from_" not in seed.structure_id
            assert len(seed.structure_id) <= 16


def test_compact_growth_id_does_not_nest_parent() -> None:
    from builder.nucleation.molecular_growth import (
        CoordSeed,
        assign_compact_b_ids,
        compact_growth_id,
        parse_compact_serial,
    )

    long_parent = (
        "coord_k005_p004_from_coord_k004_p003_from_"
        "k003_p002_mol0001_s1_pm1_s0_pm2"
    )
    assert compact_growth_id(6, 5, "B", 7) == "k006_p005_B0007"
    assert parse_compact_serial("k006_p005_B0007") == 7
    assert parse_compact_serial(long_parent) is None
    seeds = [
        CoordSeed(
            k=6,
            p=5,
            structure_id="tmp",
            parent_id=long_parent,
            shed=1,
            p_m=1,
            symbols=("Se",),
            coordinates=np.zeros((1, 3)),
            core_edges=(),
        ),
        CoordSeed(
            k=6,
            p=5,
            structure_id="tmp",
            parent_id="k005_p004_B0003",
            shed=0,
            p_m=1,
            symbols=("Se",),
            coordinates=np.zeros((1, 3)),
            core_edges=(),
        ),
    ]
    assign_compact_b_ids(seeds, output_dir=None)
    assert seeds[0].structure_id == "k006_p005_B0001"
    assert seeds[1].structure_id == "k006_p005_B0002"
    assert all(len(s.structure_id) == 15 for s in seeds)


def test_assign_compact_b_ids_reuses_finished_lineage(tmp_path) -> None:
    from builder.nucleation.molecular_growth import (
        CoordSeed,
        assign_compact_b_ids,
    )

    (tmp_path / "index.csv").write_text(
        "k,p,structure_id,xtb_energy_eV,xtb_converged,move,shed,p_m,parent_id\n"
        "4,3,k004_p003_B0004,-10.0,True,coord,1,1,k003_p002_mol0001\n",
        encoding="utf-8",
    )
    seed = CoordSeed(
        k=4,
        p=3,
        structure_id="tmp",
        parent_id="k003_p002_mol0001",
        shed=1,
        p_m=1,
        symbols=("Se",),
        coordinates=np.zeros((1, 3)),
        core_edges=(),
    )
    other = CoordSeed(
        k=4,
        p=3,
        structure_id="tmp",
        parent_id="k003_p002_mol0002",
        shed=0,
        p_m=1,
        symbols=("Se",),
        coordinates=np.zeros((1, 3)),
        core_edges=(),
    )
    assign_compact_b_ids([seed, other], output_dir=tmp_path)
    assert seed.structure_id == "k004_p003_B0004"
    assert other.structure_id == "k004_p003_B0005"


def test_growth_log_global_and_block_index(tmp_path) -> None:
    path = tmp_path / "growth.log"
    log = GrowthLog(quiet=True, log_path=path)
    log.set_block_plan(2)
    log.begin_block(2, label="B k=4 p=3")
    log(
        "[growth-job] k=4 p=3 move=B s=1 p_m=2 "
        "k_parent=3 p_parent=2 parent=k003_p002_mol0001 "
        "id=c1 t_s=1.2 recon_s=0.3 steps=87 max_steps=150 "
        "E_eV=-10.0 relax=ok"
    )
    log(
        "[growth-job] k=4 p=3 move=B s=1 p_m=2 "
        "k_parent=3 p_parent=2 parent=k003_p002_mol0001 "
        "id=c2 t_s=1.1 recon_s=0.2 steps=150 max_steps=150 "
        "E_eV=-10.1 relax=ok"
    )
    log.begin_bin(k=4, p=6, cores=2, cores_done=0, cores_total=2, jobs=2)
    log(
        "[growth-job] k=4 p=6 move=A id=a1 "
        "E_eV=-11.0 t_s=2.0 recon_s=0.4 steps=40 max_steps=150 relax=ok"
    )
    log(
        "[growth-job] k=4 p=6 move=A id=a2 "
        "E_eV=-11.1 t_s=2.1 recon_s=0.5 steps=12 max_steps=150 relax=ok"
    )
    log.close()
    text = path.read_text(encoding="utf-8")
    assert "block 2/2" in text
    # global index keeps climbing; block index resets per block
    assert "     1   1/2  [CdSe]_3(CdCl2)_2" in text
    assert "cyc=87/150" in text
    assert "     2   2/2  [CdSe]_3(CdCl2)_2" in text
    assert "cyc=150/150" in text
    assert "     3   1/2  k=4 p=6" in text
    assert "     4   2/2  k=4 p=6" in text
    # extra jobs in a block grow the denominator instead of showing i > N
    log2 = GrowthLog(quiet=True, log_path=path)
    log2.begin_block(1, label="overflow")
    log2(
        "[growth-job] k=5 p=1 move=A id=x1 "
        "E_eV=-1.0 t_s=0.1 recon_s=0.0 relax=ok"
    )
    log2(
        "[growth-job] k=5 p=1 move=A id=x2 "
        "E_eV=-1.1 t_s=0.1 recon_s=0.0 relax=ok"
    )
    log2.close()
    extra = path.read_text(encoding="utf-8")
    assert "     1   1/1  k=5 p=1" in extra
    assert "     2   2/2  k=5 p=1" in extra
