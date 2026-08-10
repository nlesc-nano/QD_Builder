"""Molecular package growth: parents, packages, core inflation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from builder.nucleation.molecular_growth import (
    GrowthConfig,
    ParentStructure,
    grow_cores_from_parents,
    identify_packages,
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


def test_growth_config_loads() -> None:
    cfg = GrowthConfig.from_yaml(GROWTH_YAML)
    assert cfg.monomer_p_values == (1, 2, 3)
    assert cfg.max_shed >= 1
    assert cfg.references is not None
    assert cfg.references.energy_cdse_eV < 0
    assert cfg.use_coord_carry
    assert cfg.local_cleanup_cycles > 0
    assert cfg.shed_mode == "wbo"


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
