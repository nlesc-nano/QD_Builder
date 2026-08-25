"""Zinc-blende occupation growth: snap, attach, reject diamonds."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from builder.nucleation.molecular_growth import (
    GrowthConfig,
    MinimumConsolidation,
    ParentStructure,
    _opt_zb_occupations,
    consolidate_relaxed_minima,
    grow_cores_from_parents,
    relaxed_minimum_similarity,
    select_parents,
    write_minimum_clusters,
)
from builder.nucleation.geometry_pack import load_geometry_pack
from builder.nucleation.molecular_zb_growth import (
    attach_cdse,
    endpoint_similarity_diagnostic,
    ensure_occupation_identity,
    grow_zb_children,
    lattice_k1_occupation,
    lattice_model,
    load_reference_occupation,
    occupation_from_record,
    occupation_to_record,
    place_cl_2p,
    seed_occupation,
    shed_occupations,
    zb_embeddable,
)
from builder.nucleation.soft_rules import describe_structure
from builder.nucleation.spec import load_nucleation_spec
from builder.nucleation.xtb_relax import XtbResult

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


def test_shedding_is_exhaustive_before_soft_priority(map_spec) -> None:
    model = lattice_model(map_spec)
    parent = lattice_k1_occupation(map_spec, model, p=1)
    assert parent is not None
    k2 = grow_zb_children(
        parent, s=0, p_m=1, spec=map_spec, model=model, cap=0
    )[0]
    removals = shed_occupations(k2, 1, map_spec, model)
    assert len(removals) >= 2

    # WBO is a post-enumeration preference only: it must not change the set of
    # legal children when no cap is requested.
    baseline = grow_zb_children(
        k2, s=1, p_m=1, spec=map_spec, model=model, cap=0
    )
    biased = grow_zb_children(
        k2,
        s=1,
        p_m=1,
        spec=map_spec,
        model=model,
        cap=0,
        parent_wbo={(0, 2): 0.1, (1, 3): 1.5},
    )
    assert {item.occupation_id for item in baseline} == {
        item.occupation_id for item in biased
    }


def test_occupation_manifest_round_trip(map_spec) -> None:
    model = lattice_model(map_spec)
    occupation = lattice_k1_occupation(map_spec, model, p=2)
    assert occupation is not None
    occupation.parent_occupation_ids = ("parent-occ-a", "parent-occ-b")
    occupation.parent_structure_ids = ("parent-a", "parent-b")
    restored = occupation_from_record(occupation_to_record(occupation))
    ensure_occupation_identity(restored, model)
    assert restored.occupation_id == occupation.occupation_id
    assert restored.site_ids == occupation.site_ids
    assert np.allclose(restored.coordinates, occupation.coordinates)
    assert restored.parent_occupation_ids == occupation.parent_occupation_ids
    assert restored.parent_structure_ids == occupation.parent_structure_ids


def test_occupation_identity_is_spatial_and_cubic_invariant(map_spec) -> None:
    model = lattice_model(map_spec)
    parent = lattice_k1_occupation(map_spec, model, p=1)
    assert parent is not None
    children = attach_cdse(parent, map_spec, model, cap=0)
    same_graph = [
        child for child in children if child.core_edges == children[0].core_edges
    ]
    assert len(same_graph) >= 2
    assert same_graph[0].occupation_id != same_graph[1].occupation_id

    transformed = replace(
        same_graph[0],
        coordinates=same_graph[0].coordinates[:, [2, 0, 1]]
        * np.array([-1.0, 1.0, -1.0])
        + np.array([12.0, -7.0, 3.0]),
        site_ids=(),
        occupation_id="",
    )
    ensure_occupation_identity(transformed, model)
    assert transformed.occupation_id == same_graph[0].occupation_id


def test_k13_wulff_endpoint_is_diagnostic_only(map_spec) -> None:
    model = lattice_model(map_spec)
    reference = load_reference_occupation(
        PACK_ZB / "k13_wulff_core.yaml", map_spec, model
    )
    diagnostic = endpoint_similarity_diagnostic(reference, reference)
    assert reference.k == 13 and reference.p == 3
    assert diagnostic["site_overlap_fraction"] == pytest.approx(1.0)
    assert diagnostic["assignment_rmsd_A"] == pytest.approx(0.0)
    assert diagnostic["ranking_or_filtering_effect"] == "none"


def _similarity_parent(
    structure_id: str,
    *,
    coordinates: np.ndarray,
    symbols=("Se", "Cd", "Cd", "Cl", "Cl"),
    edges=((0, 1), (0, 2), (1, 3), (2, 4)),
    occupation_id: str = "occ-a",
) -> ParentStructure:
    return ParentStructure(
        k=1,
        p=1,
        structure_id=structure_id,
        symbols=tuple(symbols),
        coordinates=np.asarray(coordinates, dtype=float),
        energy_eV=-10.0,
        edges=tuple(edges),
        core_edges=((0, 1), (0, 2)),
        zb_occupation=SimpleNamespace(occupation_id=occupation_id),
    )


def test_minimum_similarity_is_rigid_motion_and_permutation_invariant(
    map_spec,
) -> None:
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.8, 1.2, 1.0],
            [-1.7, -1.3, 1.1],
            [3.1, 1.0, 0.2],
            [-3.0, -1.1, 0.1],
        ]
    )
    left = _similarity_parent("left", coordinates=coordinates)
    # new index -> old index; simultaneously exchange both equivalent arms
    permutation = [0, 2, 1, 4, 3]
    old_to_new = {old: new for new, old in enumerate(permutation)}
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transformed = coordinates[permutation] @ rotation + np.array([8.0, -3.0, 5.0])
    right = _similarity_parent(
        "right",
        coordinates=transformed,
        symbols=tuple(left.symbols[index] for index in permutation),
        edges=tuple(
            sorted(
                tuple(sorted((old_to_new[a], old_to_new[b])))
                for a, b in left.edges
            )
        ),
    )
    config = MinimumConsolidation(enabled=True)
    metrics = relaxed_minimum_similarity(left, right, config, map_spec)
    assert metrics is not None
    assert metrics.pair_distance_rms_A == pytest.approx(0.0, abs=1.0e-10)
    assert metrics.core_rmsd_A == pytest.approx(0.0, abs=1.0e-10)
    assert metrics.full_rmsd_A == pytest.approx(0.0, abs=1.0e-10)


def test_internal_geometry_prevents_equal_energy_false_merge(map_spec) -> None:
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.8, 1.2, 1.0],
            [-1.7, -1.3, 1.1],
            [3.1, 1.0, 0.2],
            [-3.0, -1.1, 0.1],
        ]
    )
    left = _similarity_parent("left", coordinates=coordinates)
    deformed = coordinates.copy()
    deformed[4] += np.array([0.8, -0.5, 0.4])
    right = _similarity_parent("right", coordinates=deformed)
    assert (
        relaxed_minimum_similarity(
            left, right, MinimumConsolidation(enabled=True), map_spec
        )
        is None
    )


def test_minimum_similarity_does_not_treat_reflection_as_rotation(map_spec) -> None:
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    symbols = ("Cd", "Se", "Cl", "X", "Y")
    edges = ((0, 1), (0, 2), (0, 3), (0, 4))
    left = _similarity_parent(
        "left", coordinates=coordinates, symbols=symbols, edges=edges
    )
    mirrored = coordinates.copy()
    mirrored[:, 0] *= -1.0
    right = _similarity_parent(
        "right", coordinates=mirrored, symbols=symbols, edges=edges
    )
    proper_only = MinimumConsolidation(enabled=True)
    assert relaxed_minimum_similarity(left, right, proper_only, map_spec) is None
    reflection_allowed = replace(proper_only, allow_reflection=True)
    metrics = relaxed_minimum_similarity(
        left, right, reflection_allowed, map_spec
    )
    assert metrics is not None
    assert metrics.full_rmsd_A == pytest.approx(0.0, abs=1.0e-10)


def test_minimum_cluster_merges_endpoints_but_preserves_zb_routes(
    map_spec, tmp_path: Path
) -> None:
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.8, 1.2, 1.0],
            [-1.7, -1.3, 1.1],
            [3.1, 1.0, 0.2],
            [-3.0, -1.1, 0.1],
        ]
    )
    endpoints = [
        _similarity_parent("a-start0", coordinates=coordinates, occupation_id="occ-a"),
        _similarity_parent("a-start1", coordinates=coordinates, occupation_id="occ-a"),
        _similarity_parent("b-start0", coordinates=coordinates, occupation_id="occ-b"),
    ]
    endpoints[1].energy_eV = -9.9999
    endpoints[2].energy_eV = -9.9998
    clusters = consolidate_relaxed_minima(
        endpoints, MinimumConsolidation(enabled=True), map_spec
    )
    assert len(clusters) == 1
    assert len(clusters[0].members) == 3
    routes = clusters[0].route_representatives()
    assert [route.structure_id for route in routes] == ["a-start0", "b-start0"]
    assert all(route.minimum_multiplicity == 3 for route in routes)
    assert all(route.minimum_occupation_ids == ("occ-a", "occ-b") for route in routes)

    inventory = write_minimum_clusters(
        tmp_path,
        1,
        1,
        clusters,
        config=MinimumConsolidation(enabled=True),
    )
    persisted = json.loads(inventory.read_text())
    assert persisted["raw_endpoint_count"] == 3
    assert persisted["minimum_count"] == 1
    assert persisted["clusters"][0]["occupation_ids"] == ["occ-a", "occ-b"]
    assert persisted["criteria"]["allow_reflection"] is False

    growth = GrowthConfig.from_yaml(PACK_ZB / "growth.yaml")
    selected = select_parents(endpoints, growth, map_spec)
    assert [route.structure_id for route in selected] == ["a-start0", "b-start0"]
    assert len({route.minimum_id for route in selected}) == 1


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
    from builder.nucleation.molecular_zb_growth import construction_clash

    model = lattice_model(map_spec)
    occ = lattice_k1_occupation(map_spec, model, p=2)
    assert occ is not None
    placed = place_cl_2p(occ, map_spec, model=model)
    assert placed is not None
    symbols, coords, edges = placed
    assert symbols.count("Cl") == 4
    assert coords.shape[0] == len(symbols)
    assert not construction_clash(symbols, coords, map_spec, bonded=edges)
    se_idx = [i for i, s in enumerate(symbols) if s == "Se"]
    cl_idx = [i for i, s in enumerate(symbols) if s == "Cl"]
    for i in se_idx:
        for j in cl_idx:
            assert float(np.linalg.norm(coords[i] - coords[j])) > 2.20


def test_zb_metric_bridge_pairs_drops_long_and_occupied_segment(map_spec) -> None:
    from dataclasses import replace

    from builder.nucleation.molecular_zb_growth import zb_metric_bridge_pairs
    from builder.nc_types import NucleationGraphRules

    spec = replace(
        map_spec,
        graph_rules=replace(map_spec.graph_rules, bridge_cd_cd_max_distance=4.75),
    )
    # Three collinear Cd: 0 --4.34-- 1 --4.34-- 2 (midpoint of 0-2 is 1).
    coords = np.array(
        [
            [0.00, 0.00, 0.00],
            [4.342, 0.00, 0.00],
            [8.684, 0.00, 0.00],
            [2.171, 3.760, 0.00],
        ]
    )
    pairs = [(0, 1), (1, 2), (0, 2), (0, 3)]
    kept = set(zb_metric_bridge_pairs(pairs, coords, [0, 1, 2, 3], spec))
    assert (0, 1) in kept
    assert (1, 2) in kept
    assert (0, 3) in kept
    assert (0, 2) not in kept


def test_place_cl_mu2_prefers_outward_site(map_spec) -> None:
    import networkx as nx

    from builder.nucleation.molecular_zb_growth import place_cl_on_zb_core
    from builder.nucleation.types import AtomRecord, _State

    # Interior Cd 0 at origin; surface Cd 1 and 2 form an equilateral 4.342 Å
    # triangle with it.  A μ2 on 1-2 must not land on Cd 0.
    a = 4.342
    c1 = np.array([a, 0.0, 0.0])
    c2 = np.array([0.5 * a, a * np.sqrt(3) / 2.0, 0.0])
    atoms = (
        AtomRecord(0, "Cd", (0.0, 0.0, 0.0), "core_cation"),
        AtomRecord(1, "Cd", tuple(c1), "core_cation"),
        AtomRecord(2, "Cd", tuple(c2), "core_cation"),
        AtomRecord(3, "Cl", (0.0, 0.0, 0.0), "precursor_ligand"),
    )
    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(1, 3), (2, 3)])
    state = _State(atoms=atoms, graph=graph)
    anchored = {0: (0.0, 0.0, 0.0), 1: tuple(c1), 2: tuple(c2)}
    xyz = place_cl_on_zb_core(state, anchored, map_spec)
    assert xyz is not None
    assert float(np.linalg.norm(xyz[3] - xyz[0])) > 2.90
    assert float(np.linalg.norm(xyz[3] - xyz[1])) < 3.10
    assert float(np.linalg.norm(xyz[3] - xyz[2])) < 3.10


def test_adapt_to_embed_table_is_a_start_not_a_relax() -> None:
    import networkx as nx

    from builder.nucleation.geometry_pack import load_geometry_pack
    from builder.nucleation.molecular_motif_reconstruct import adapt_to_embed_table
    from builder.nucleation.types import AtomRecord, _State

    spec = load_nucleation_spec(str(PACK_ZB / "run_gxtb.yaml"))
    pack = load_geometry_pack(PACK_ZB / "run_gxtb.yaml")
    atoms = (
        AtomRecord(0, "Cd", (0.0, 0.0, 0.0), "core_cation"),
        AtomRecord(1, "Se", (3.40, 0.0, 0.0), "core_anion"),
    )
    graph = nx.Graph()
    graph.add_nodes_from((0, 1))
    graph.add_edges_from([(0, 1)])
    state = _State(atoms=atoms, graph=graph)
    start = np.array([[0.0, 0.0, 0.0], [3.40, 0.0, 0.0]])
    out = adapt_to_embed_table(state, start, pack, spec, max_nfev=16)
    assert out is not None
    stretched = float(np.linalg.norm(out[1] - out[0]))
    # Table CdSe is ~2.4–2.6 Å; a 16-step start should move toward it, not
    # stay at the 3.40 Å lattice-like guess, and must not collapse.
    assert 2.20 < stretched < 3.20


def test_adapt_to_embed_table_keeps_clash_free_start_if_fit_clashes() -> None:
    import networkx as nx

    from builder.nucleation.geometry_pack import load_geometry_pack
    from builder.nucleation.molecular_motif_reconstruct import adapt_to_embed_table
    from builder.nucleation.types import AtomRecord, _State

    spec = load_nucleation_spec(str(PACK_ZB / "run_gxtb.yaml"))
    pack = load_geometry_pack(PACK_ZB / "run_gxtb.yaml")
    # Two non-bonded Cd already legal; a Cl far from both.
    atoms = (
        AtomRecord(0, "Cd", (0.0, 0.0, 0.0), "core_cation"),
        AtomRecord(1, "Cd", (4.34, 0.0, 0.0), "core_cation"),
        AtomRecord(2, "Cl", (4.34, 2.45, 0.0), "precursor_ligand"),
    )
    graph = nx.Graph()
    graph.add_nodes_from(range(3))
    graph.add_edges_from([(1, 2)])
    state = _State(atoms=atoms, graph=graph)
    start = np.array([a.coordinates for a in atoms], dtype=float)
    out = adapt_to_embed_table(state, start, pack, spec, max_nfev=16)
    assert out is not None
    assert float(np.linalg.norm(out[0] - out[2])) >= 2.90


def test_construction_clash_rejects_cl_on_cn4_cd(map_spec) -> None:
    from builder.nucleation.molecular_zb_growth import construction_clash

    # Interior Cd at origin with four Se. Cl is outside the generic 2.20 Å
    # floor but inside the Cd-Cl bond of the 4-Se cation.
    symbols = ["Cd", "Se", "Se", "Se", "Se", "Cd", "Cl"]
    coords = np.array(
        [
            [0.00, 0.00, 0.00],
            [1.50, 1.50, 1.50],
            [1.50, -1.50, -1.50],
            [-1.50, 1.50, -1.50],
            [-1.50, -1.50, 1.50],
            [4.00, 0.00, 0.00],
            [2.40, 0.00, 0.00],
        ]
    )
    bonded = [(5, 6)]
    assert construction_clash(symbols, coords, map_spec, bonded=bonded)


def test_place_cl_on_k2_attach_does_not_clash(map_spec) -> None:
    from builder.nucleation.molecular_zb_growth import construction_clash

    model = lattice_model(map_spec)
    parent = lattice_k1_occupation(map_spec, model, p=1)
    assert parent is not None
    kids = grow_zb_children(
        parent, s=0, p_m=1, spec=map_spec, model=model, cap=8
    )
    assert kids
    ok = 0
    for kid in kids:
        placed = place_cl_2p(kid, map_spec, model=model)
        if placed is None:
            continue
        symbols, coords, edges = placed
        if not construction_clash(symbols, coords, map_spec, bonded=edges):
            ok += 1
    assert ok >= 1


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
    assert w1.child_redecorate is True
    assert w1.soft_rules.enabled is False
    assert w3.min_p_parent == 2
    assert w3.energy_window_eV == 0.60
    assert spec.graph_rules.decoration_mode == "motif_bridge_first"
    assert spec.graph_rules.bridge_cd_cd_max_distance == pytest.approx(4.75)
    assert cfg.local_cleanup_enabled is True
    assert cfg.local_cleanup_freeze_core is True
    w12 = cfg.window_for(12)
    assert w12.move_zb_sites is True
    assert cfg.endpoint_diagnostic_k == 13
    assert cfg.endpoint_reference == (PACK_ZB / "k13_wulff_core.yaml").resolve()
    assert cfg.minimum_consolidation.enabled is True
    assert cfg.minimum_consolidation.allow_reflection is False
    assert cfg.minimum_consolidation.pair_distance_rms_A == pytest.approx(0.05)
    assert cfg.minimum_consolidation.core_rmsd_A == pytest.approx(0.10)
    assert cfg.minimum_consolidation.full_rmsd_A == pytest.approx(0.15)
    assert cfg.minimum_consolidation.max_occupations_per_minimum == 0


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
    """The clean k1k13 pack grows Z children and does not enable A/B."""

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


def test_later_growth_uses_stored_lattice_not_relaxed_coordinates(map_spec) -> None:
    model = lattice_model(map_spec)
    k1 = lattice_k1_occupation(map_spec, model, p=1)
    assert k1 is not None
    k2 = grow_zb_children(
        k1, s=0, p_m=1, spec=map_spec, model=model, cap=1
    )[0]
    distorted = np.asarray(k2.coordinates, dtype=float).copy()
    distorted[0] += np.array([7.0, -4.0, 3.0])
    parent = ParentStructure(
        k=k2.k,
        p=k2.p,
        structure_id="relaxed_but_distorted",
        symbols=k2.symbols,
        coordinates=distorted,
        energy_eV=-2.0,
        edges=k2.core_edges,
        core_edges=k2.core_edges,
        zb_occupation=k2,
    )
    cfg = GrowthConfig.from_yaml(PACK_ZB / "growth.yaml")
    result = grow_cores_from_parents([parent], growth=cfg, spec=map_spec)
    assert result.zb_stats is not None
    assert result.zb_stats.snapped == 1
    assert result.zb_stats.snap_fail == 0
    assert result.zb_seeds
    assert all(
        child.parent_occupation_ids == (k2.occupation_id,)
        for children in result.zb_seeds.values()
        for child in children
    )


def test_zb_opt_indexes_only_topology_preserving_endpoints(
    map_spec, monkeypatch, tmp_path: Path
) -> None:
    import builder.nucleation.xtb_relax as xtb_relax

    model = lattice_model(map_spec)
    occupation = lattice_k1_occupation(map_spec, model, p=1)
    assert occupation is not None
    occupation.parent_id = "parent-structure"
    occupation.parent_structure_ids = ("parent-structure",)
    growth = replace(
        GrowthConfig.from_yaml(PACK_ZB / "growth.yaml"),
        local_cleanup_enabled=False,
        local_cleanup_freeze_core=False,
    )
    pack = load_geometry_pack(PACK_ZB / "run_gxtb.yaml")

    def run_fake(output: Path, *, preserve: bool):
        calls = []

        def fake_relax(entries, _settings, _cutoffs=None, **_kwargs):
            calls.extend(entries)
            results = []
            for entry in entries:
                edges = list(entry["edges"])
                if not preserve:
                    edges.remove(tuple(occupation.core_edges[0]))
                results.append(
                    XtbResult(
                        ok=True,
                        energy_eV=-12.5,
                        converged=True,
                        coordinates=tuple(
                            tuple(float(value) for value in point)
                            for point in entry["positions"]
                        ),
                        relaxed_edges=tuple(edges),
                    )
                )
            return results

        monkeypatch.setattr(xtb_relax, "relax_structures", fake_relax)
        minima = {}
        ranks = {}
        _opt_zb_occupations(
            {(1, 1): [occupation]},
            growth=growth,
            map_spec=map_spec,
            pack=pack,
            output_dir=output,
            progress=None,
            child_minima=minima,
            bin_ranks=ranks,
        )
        records = [
            json.loads(line)
            for line in (output / "zb_occupations.jsonl").read_text().splitlines()
        ]
        return calls, minima, ranks, records

    calls, minima, ranks, records = run_fake(tmp_path / "preserved", preserve=True)
    assert 1 <= len(calls) <= 2
    assert minima and ranks
    assert records[0]["propagation_eligible"] is True
    assert records[0]["topology_status"] == "preserved"
    # Core may leave CIF in the embed.yaml morph; g-xTB sees that start.

    _calls, minima, ranks, records = run_fake(tmp_path / "changed", preserve=False)
    assert not minima and not ranks
    assert records[0]["propagation_eligible"] is False
    assert records[0]["topology_status"] == "changed"
    assert list((tmp_path / "changed").rglob("*_offpath.xyz"))
    assert not (tmp_path / "changed" / "index.csv").exists()
