"""Regression tests for rigid-lattice, graph-configurable nucleation maps."""

from __future__ import annotations

from collections import Counter
import hashlib
import copy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from networkx.algorithms.isomorphism import GraphMatcher
import networkx as nx
import numpy as np
import pytest

import builder.nucleation as nucleation_module
from _nucleation_reference import (
    digest_diff,
    exhaustive_bridge_sets,
    graph_certificate,
    registry_digest,
)
from builder.nucleation import (
    AtomRecord,
    _CandidateAccumulator,
    _State,
    _build_lattice_model,
    _graph_coordination_score,
    _latent_bridge_variants,
    _optimistic_bridge_score,
    generate_nucleation_map,
    generate_nucleation_result,
    load_nucleation_spec,
    nucleation_result_to_dict,
    registry_to_dict,
    write_nucleation_bundle,
    write_nucleation_json,
)

ROOT = Path(__file__).resolve().parents[1]


def _node_match(left, right):
    return left.get("element") == right.get("element")


def _edge_match(left, right):
    return left.get("bond_order", 1) == right.get("bond_order", 1)


@pytest.fixture(scope="module")
def cdse_k1_result():
    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    assert spec.kmax == 1
    return generate_nucleation_result(spec)


@pytest.fixture(scope="module")
def cdse_k2_result():
    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    return generate_nucleation_result(replace(spec, kmax=2))


@pytest.fixture(scope="module")
def cdse_k1_unrestricted_result():
    """k=1 with ``min_bridged_host_cn`` off -- the pre-DFT default.

    The shipped default now forbids a bridge that would leave a cation at CN 2,
    which is right (see ``test_dft_k1p1_...``) but removes the only bridge at
    k=1 p=1.  The construction machinery for rhombic bridges still needs
    exercising at that size, so the tests that document *how a bridge is built*
    -- as opposed to *whether one should be* -- run against this counterfactual.
    It is the configuration shipped as ``cdse_cdcl2_dft_rules.yaml``.
    """

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    return generate_nucleation_result(_with_min_bridged_host_cn(spec, 1, 1))


@pytest.fixture(scope="module")
def cdse_k2_unrestricted_result():
    """k=2 counterpart of :func:`cdse_k1_unrestricted_result`."""

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    return generate_nucleation_result(_with_min_bridged_host_cn(spec, 1, 2))


def test_guided_yaml_exposes_multi_shell_knobs() -> None:
    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2_guided.yaml")
    assert spec.mode == "guided"
    assert spec.shells_per_skeleton >= 1
    assert spec.shell_score_layers >= 1
    assert spec.shell_enum_max_assignments >= 0


def test_surface_scenario_a_shed_and_p_cap_helpers() -> None:
    """Scenario A: p_surf=β k^{2/3}, s_max, inventory p_child max."""

    import math
    from builder.nucleation import (
        _channel_p_child_max,
        _effective_max_shed,
        _effective_p_cap,
        _p_surf,
    )

    beta = 3.0
    assert _p_surf(13, beta) == int(math.floor(3.0 * 13 ** (2.0 / 3.0)))
    assert _p_surf(13, beta) == 16  # matches user's A table ballpark

    guided = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2_guided.yaml")
    assert guided.p_surf_beta == pytest.approx(3.0)
    # s_max = min(p, p_surf(k)); parent p=16 at k=13 → s_max=16
    assert _effective_max_shed(13, 16, guided) == 16
    assert _effective_max_shed(13, 5, guided) == 5
    # Inventory: residual (16-3)=13 + pool (3+2)=5 → 18, capped by p_surf(14)
    p_ch = _channel_p_child_max(16, 3, 2, 14, guided)
    assert p_ch == min(16 + 2, _p_surf(14, beta))
    # s=0 must NOT collapse to p_m only: residual p kept → min(p+p_m, p_surf)
    p_noshed = _channel_p_child_max(16, 0, 2, 14, guided)
    assert p_noshed == min(16 + 2, _p_surf(14, beta))
    assert p_noshed >= 16
    # Ladder cap at destination k is p_surf, not 3k
    cap = _effective_p_cap(
        13, capacity_cap=3 * 13, max_inherited=10, spec=guided
    )
    assert cap == _p_surf(13, beta)
    assert cap < 3 * 13

    # Legacy fixed shed when beta=0
    legacy = replace(guided, p_surf_beta=0.0, k_growth_max_shed=2)
    assert _effective_max_shed(13, 16, legacy) == 2


def test_default_shedding_is_limited_only_by_parent_inventory() -> None:
    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    from builder.nucleation import _effective_max_shed

    assert spec.p_surf_beta == pytest.approx(0.0)
    assert spec.k_growth_max_shed == 0
    assert _effective_max_shed(3, 0, spec) == 0
    assert _effective_max_shed(3, 4, spec) == 4


def test_default_shedding_enumerates_all_counts(cdse_k1_result) -> None:
    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    model = _build_lattice_model(spec)
    record = cdse_k1_result.registry[1][2][0]
    state = nucleation_module._make_core_graph(record.atoms, model, spec)
    variants = nucleation_module._shed_parent_variants(
        state, 2, model, spec, k=1
    )
    assert {shed for _state, _p_out, shed in variants} == {0, 1, 2}


def test_se_capacity_counts_remaining_anion_slots() -> None:
    from builder.nucleation import _build_lattice_model, _se_coordination_capacity

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    model = _build_lattice_model(spec)
    state = nucleation_module._seed_state(model)
    # The CdSe monomer has one Se--Cd bond and therefore three remaining Se
    # slots for precursor Cd centers.
    assert _se_coordination_capacity(state, spec) == 3


def test_same_species_bond_like_contacts_are_hard_rejected() -> None:
    from builder.nucleation import (
        _build_lattice_model,
        _pair_distances,
        _same_species_contacts_valid,
    )

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    model = _build_lattice_model(spec)
    assert model.same_species_min_distance["Cd"] == pytest.approx(
        4.3420191712 - spec.site_tolerance
    )
    assert not _same_species_contacts_valid(
        _pair_distances(np.asarray([[0.0, 0.0, 0.0], [3.43, 0.0, 0.0]])),
        ["Cd", "Cd"],
        model,
    )
    assert not _same_species_contacts_valid(
        _pair_distances(np.asarray([[0.0, 0.0, 0.0], [3.14, 0.0, 0.0]])),
        ["Cl", "Cl"],
        model,
    )
    assert _same_species_contacts_valid(
        _pair_distances(np.asarray([[0.0, 0.0, 0.0], [4.34, 0.0, 0.0]])),
        ["Cd", "Cd"],
        model,
    )


def test_guided_multi_shell_keeps_more_isomers_per_skeleton() -> None:
    """Same core, multiple passivations when shells_per_skeleton > 1."""

    guided = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2_guided.yaml")
    single = generate_nucleation_result(
        replace(
            guided,
            kmax=1,
            shells_per_skeleton=1,
            shell_score_layers=1,
            retain_max_per_bin=0,
            retain_score_layers=3,
        )
    )
    multi = generate_nucleation_result(
        replace(
            guided,
            kmax=1,
            shells_per_skeleton=6,
            shell_score_layers=3,
            shells_per_score_layer=1,
            retain_max_per_bin=0,
            retain_score_layers=3,
        )
    )
    n_single = sum(len(recs) for recs in single.registry[1].values())
    n_multi = sum(len(recs) for recs in multi.registry[1].values())
    assert n_multi >= n_single
    # At least one (k,p) bin should show >1 retained isomer under multi-shell.
    assert any(
        len(recs) > 1 for recs in multi.registry[1].values()
    ), "expected multi-shell guided to retain >1 isomer in some k=1 bin"

    # Isomers in a multi-shell bin share the same skeleton family when they
    # are passivation variants of one core.
    for p, recs in multi.registry[1].items():
        if len(recs) < 2:
            continue
        families = {r.metadata["skeleton_family_id"] for r in recs}
        assert len(families) == 1, (
            f"p={p}: multi-shell bin should be one skeleton family, got {families}"
        )
        break


def test_select_shells_by_score_band_respects_layers_and_cap() -> None:
    """Unit: score-layer band + hard shell cap."""

    from builder.nucleation import _select_shells_by_score_band, _State
    import networkx as nx
    from builder.nucleation import AtomRecord

    def _tiny(tag: int) -> _State:
        g = nx.Graph()
        g.add_node(0)
        atom = AtomRecord(
            atom_id=0,
            symbol="Cd",
            coordinates=(float(tag), 0.0, 0.0),
            role="core_cation",
        )
        return _State(atoms=(atom,), graph=g)

    s_hi_a = _tiny(1)
    s_hi_b = _tiny(2)
    s_mid = _tiny(3)
    s_lo = _tiny(4)
    score_hi = (10, 0, 0)
    score_mid = (9, 0, 0)
    score_lo = (8, 0, 0)
    scored = [
        (score_hi, s_hi_a),
        (score_hi, s_hi_b),
        (score_mid, s_mid),
        (score_lo, s_lo),
    ]
    # Two layers, whole layer (per_score_layer=0): high+mid, drop low.
    selected = _select_shells_by_score_band(
        scored, score_layers=2, max_shells=3, per_score_layer=0
    )
    assert len(selected) == 3
    assert s_lo not in selected
    # Cap 2 with whole layers: both highs first, mid cut by cap.
    selected_cap = _select_shells_by_score_band(
        scored, score_layers=2, max_shells=2, per_score_layer=0
    )
    assert len(selected_cap) == 2
    assert set(selected_cap) == {s_hi_a, s_hi_b}

    # One top per score layer: best@10, best@9, best@8 → 3 shells.
    one_each = _select_shells_by_score_band(
        scored, score_layers=3, max_shells=6, per_score_layer=1
    )
    assert len(one_each) == 3
    assert s_mid in one_each
    assert s_lo in one_each
    # Exactly one of the two high-score ties.
    assert sum(1 for s in one_each if s in (s_hi_a, s_hi_b)) == 1


def test_graph_rules_are_strict_unordered_pairs() -> None:
    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    assert spec.graph_rules.min_cn == {"Cd": 2, "Se": 2, "Cl": 1}
    assert spec.graph_rules.max_cn == {"Cd": 4, "Se": 4, "Cl": 2}
    assert spec.graph_rules.allowed_bonds == (("Cd", "Cl"), ("Cd", "Se"))
    assert spec.ligand_max_cn == 2
    assert len(spec.graph_rules.bridge_rules) == 1
    bridge = spec.graph_rules.bridge_rules[0]
    assert (bridge.ligand, bridge.host, bridge.shared_neighbor) == (
        "Cl",
        "Cd",
        "Se",
    )
    assert bridge.surface_angle_deg == 90.0
    assert spec.geometry_rules.by_cn["Cd"] == {
        2: "linear",
        3: "trigonal_planar",
        4: "tetrahedral",
    }
    assert spec.geometry_rules.all_cn == {
        "Cl": "tetrahedral",
        "Se": "tetrahedral",
    }


def test_k1_p1_prefers_one_bridge_plus_terminal_cl(
    cdse_k1_unrestricted_result,
) -> None:
    """Ranking behaviour when bridging is unrestricted.

    This is the pre-DFT answer, kept as a guard on the scoring machinery: given
    that a bridge is *allowed*, the bridged Cd[2,3] outranks the unbridged
    Cd[2,2] on bond count.  Whether it *should* be allowed is settled the other
    way by ``test_dft_k1p1_min_bridged_host_cn_reproduces_the_relaxed_structure``,
    and the shipped default follows DFT.
    """

    cdse_k1_result = cdse_k1_unrestricted_result
    record = cdse_k1_result.registry[1][1][0]
    assert Counter(record.symbols) == Counter({"Cd": 2, "Se": 1, "Cl": 2})
    assert record.graph.number_of_edges() == 5

    by_symbol = {
        symbol: sorted(
            record.graph.degree[atom.atom_id]
            for atom in record.atoms
            if atom.symbol == symbol
        )
        for symbol in {"Cd", "Se", "Cl"}
    }
    assert by_symbol == {"Cd": [2, 3], "Se": [2], "Cl": [1, 2]}
    assert any(len(cycle) == 4 for cycle in nx.cycle_basis(record.graph))
    assert record.metadata["total_cn"] == 10
    assert record.metadata["bridge_count"] == 1
    assert record.metadata["min_cn_compliant"] is True
    assert record.metadata["min_cn_violation_count"] == 0
    assert record.selection_reason == "min_cn_compliant"

    discarded = cdse_k1_result.discarded_registry[1][1]
    assert len(discarded) == 2
    violating = next(
        item for item in discarded
        if item.metadata["min_cn_violation_count"] == 1
    )
    assert violating.metadata["coordination_by_element"] == {
        "Cd": [3, 1],
        "Cl": [1, 1],
        "Se": [2],
    }
    assert violating.metadata["min_cn_compliant"] is False
    assert violating.metadata["min_cn_total_shortfall"] == 1
    assert violating.selection_reason == "min_cn_violation"
    baseline = next(
        item for item in discarded if item.metadata["bridge_count"] == 0
        and item.metadata["min_cn_compliant"]
    )
    assert baseline.metadata["coordination_by_element"] == {
        "Cd": [2, 2], "Cl": [1, 1], "Se": [2]
    }

    reference = cdse_k1_result.reference_bond_length
    allowed = {frozenset(("Cd", "Se")), frozenset(("Cd", "Cl"))}
    for left, right, data in record.graph.edges(data=True):
        assert frozenset((record.symbols[left], record.symbols[right])) in allowed
        distance = np.linalg.norm(
            record.coordinates[left] - record.coordinates[right]
        )
        if data.get("kind") == "surface_bridge":
            assert distance > reference + 0.20
        else:
            assert distance == pytest.approx(reference, abs=1e-7)


def test_k1_p1_surface_geometry_builds_90_degree_coplanar_bridge(
    cdse_k1_unrestricted_result,
) -> None:
    """How a rhombic bridge is constructed, at the smallest size that has one.

    Runs unrestricted because the shipped default (correctly) leaves k=1 p=1
    unbridged; this test is about bridge geometry, not bridge policy.
    """

    cdse_k1_result = cdse_k1_unrestricted_result
    record = cdse_k1_result.registry[1][1][0]
    assert record.surface_coordinates_data is not None
    assert not np.allclose(record.surface_coordinates, record.coordinates)
    assert record.metadata["surface_geometry"]["graph_preserved"] is True
    geometry = record.metadata["surface_geometry"]
    assert geometry["converged"] is True
    assert geometry["coordinates_changed"] is True
    assert geometry["mode"] == "retained_only_final_cn_geometry_projection"
    assert geometry["projection_valid"] is True
    assert len(geometry["bridge_geometry"]) == 1
    bridge = geometry["bridge_geometry"][0]
    assert bridge["surface_angle_deg"] == pytest.approx(90.0, abs=1e-8)
    assert bridge["plane_distance_angstrom"] == pytest.approx(0.0, abs=1e-8)
    assert bridge["surface_cd_ligand_distances_angstrom"][0] == pytest.approx(
        bridge["surface_cd_ligand_distances_angstrom"][1], abs=1e-8
    )

    ligand = bridge["ligand_atom_id"]
    first, second = bridge["host_atom_ids"]
    shared = bridge["shared_neighbor_atom_id"]
    midpoint = 0.5 * (
        record.surface_coordinates[first] + record.surface_coordinates[second]
    )
    assert np.dot(
        record.surface_coordinates[ligand] - midpoint,
        record.surface_coordinates[shared] - midpoint,
    ) < 0.0
    for atom in record.atoms:
        if atom.symbol == "Se":
            assert np.allclose(
                record.surface_coordinates[atom.atom_id],
                record.coordinates[atom.atom_id],
            )

    # Hosts that carry the rhombus bridge place their remaining terminal on the
    # Se–Cd–Cl_bridge bisector (not a CIF residual).
    bridge_slot_rule = next(
        rule for rule in geometry["applied_rules"]
        if rule["rule"] == "cd_cn3_se_bridge_terminal_bisector"
    )
    assert bridge_slot_rule["center_atom_id"] in bridge["host_atom_ids"]
    assert int(bridge_slot_rule["bridge_ligand_id"]) == ligand

    for atom in record.atoms:
        if atom.symbol == "Cd":
            assert any(
                record.atoms[neighbor].symbol == "Se"
                for neighbor in record.graph.neighbors(atom.atom_id)
            )


def test_unbridged_rule_still_constructs_linear_and_planar_cd() -> None:
    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    spec = replace(
        spec,
        graph_rules=replace(spec.graph_rules, bridge_rules=()),
    )
    result = generate_nucleation_result(spec)
    p1 = result.registry[1][1][0]
    for cd in (atom for atom in p1.atoms if atom.symbol == "Cd"):
        neighbors = list(p1.graph.neighbors(cd.atom_id))
        assert _angle_at(
            p1.surface_coordinates, cd.atom_id, neighbors[0], neighbors[1]
        ) == pytest.approx(180.0, abs=1e-5)

    p2 = result.registry[1][2][0]
    cd = next(
        atom for atom in p2.atoms
        if atom.symbol == "Cd" and p2.graph.degree[atom.atom_id] == 3
    )
    angles = _neighbor_angles_for_test(
        p2.surface_coordinates,
        cd.atom_id,
        list(p2.graph.neighbors(cd.atom_id)),
    )
    assert angles == pytest.approx([120.0, 120.0, 120.0], abs=1e-5)


def test_p2_uses_at_most_one_cl_bridge_per_cd_pair(cdse_k1_result) -> None:
    record = cdse_k1_result.registry[1][2][0]
    assert record.metadata["bridge_count"] == 3
    host_pairs = {
        tuple(sorted(item["host_atom_ids"]))
        for item in record.metadata["surface_geometry"]["bridge_geometry"]
    }
    assert len(host_pairs) == record.metadata["bridge_count"]
    assert all(
        record.graph.degree[atom.atom_id] <= 2
        for atom in record.atoms if atom.symbol == "Cl"
    )
    assert any(
        rule["rule"] == "cd_two_bridge_terminal_bisector"
        for rule in record.metadata["surface_geometry"]["applied_rules"]
    )
    model = _build_lattice_model(
        load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    )
    _assert_cif_terminal_directions(record, model)


def test_bridge_sets_enforce_final_cn_and_cn4_uses_cif_sites(
    cdse_k2_result,
) -> None:
    max_cd = cdse_k2_result.graph_rules["max_cn"]["Cd"]
    model = _build_lattice_model(
        load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    )
    saturated_donor_found = False
    for bins in cdse_k2_result.registry.values():
        for records in bins.values():
            for record in records:
                for atom in record.atoms:
                    maximum = cdse_k2_result.graph_rules["max_cn"][atom.symbol]
                    assert record.graph.degree[atom.atom_id] <= maximum
                for bridge in record.metadata["bridge_edges"]:
                    assert bridge["secondary_cn_before_bridge"] < max_cd
                    saturated_donor_found |= (
                        bridge["primary_cn_before_bridge"] == max_cd
                    )
                geometry = record.metadata["surface_geometry"]
                assert geometry["projection_valid"] is True
                assert geometry["coordinate_collisions"] == []
                for bridge_geometry in geometry["bridge_geometry"]:
                    if (
                        bridge_geometry["bridge_mode"]
                        == "shared_vacant_cif_site"
                    ):
                        assert bridge_geometry[
                            "shared_neighbor_atom_id"
                        ] is None
                        assert bridge_geometry[
                            "plane_distance_angstrom"
                        ] is None
                        assert bridge_geometry[
                            "surface_angle_deg"
                        ] == pytest.approx(109.47122063, abs=1.0e-8)
                        ligand = bridge_geometry["ligand_atom_id"]
                        assert np.allclose(
                            record.surface_coordinates[ligand],
                            bridge_geometry["virtual_site_position"],
                        )
                    else:
                        assert bridge_geometry[
                            "surface_angle_deg"
                        ] == pytest.approx(90.0, abs=1.0e-8)
                        assert bridge_geometry[
                            "plane_distance_angstrom"
                        ] == pytest.approx(0.0, abs=1.0e-8)
                    assert bridge_geometry[
                        "out_of_plane_rotation_deg"
                    ] == pytest.approx(0.0, abs=1.0e-8)
                    distances = bridge_geometry[
                        "surface_cd_ligand_distances_angstrom"
                    ]
                    assert distances[0] == pytest.approx(
                        distances[1], abs=1.0e-8
                    )
                _assert_cif_terminal_directions(record, model)
    assert saturated_donor_found
    # Bisector terminal placement (vs pure CIF residual) can surface-reject one
    # extra k=2 p=4 shell where the ideal ∠Cl–Cd–Cl completion collides.
    assert len(cdse_k2_result.registry[2][4]) == 8
    assert len(cdse_k2_result.registry[2][5]) == 1
    surface_rejected = [
        record
        for record in cdse_k2_result.discarded_registry[2][4]
        if record.selection_reason == "surface_slot_conflict"
    ]
    assert len(surface_rejected) == 11
    assert all(
        record.metadata["surface_geometry"]["saturated_cd_intrusions"]
        for record in surface_rejected
    )
    p4_to_p5 = next(
        audit
        for audit in cdse_k2_result.sweep_audit
        if audit.operation == "skeleton_passivation"
        and audit.k == 2
        and audit.p_from == 4
    )
    assert p4_to_p5.source_count == 2
    assert p4_to_p5.stage_counts["ligand_enumerations"] == 0


def test_k2_p4_closes_shared_vacant_cif_sites(cdse_k2_result) -> None:
    expected_sites = {
        (-2.3027034, 0.7675678, -3.837839),
        (0.7675678, -2.3027034, -3.837839),
    }
    for record in cdse_k2_result.registry[2][4]:
        exact = [
            bridge for bridge in record.metadata["bridge_edges"]
            if bridge["bridge_mode"] == "shared_vacant_cif_site"
        ]
        assert len(exact) == 2
        observed = {
            tuple(round(float(value), 7) for value in bridge["virtual_site_position"])
            for bridge in exact
        }
        assert observed == expected_sites
        for bridge in exact:
            ligand = bridge["ligand_atom_id"]
            primary = bridge["primary_host_atom_id"]
            secondary = bridge["host_atom_id"]
            assert record.graph.degree[ligand] == 2
            assert record.graph.degree[primary] <= 4
            assert record.graph.degree[secondary] <= 4
            assert np.allclose(
                record.surface_coordinates[ligand],
                bridge["virtual_site_position"],
            )


def _assert_cif_terminal_directions(record, model) -> None:
    ideal_directions = [
        np.asarray(vector) / np.linalg.norm(vector)
        for environment in model.environments["Cd"]
        for vector in environment
    ]
    for rule in record.metadata["surface_geometry"]["applied_rules"]:
        if rule["rule"] not in {
            "cd_cn4_terminal_cl_cif_tetrahedral",
            "cd_cn3_bridge_terminal_cl_cif_tetrahedral",
        }:
            continue
        center = rule["center_atom_id"]
        ligand_ids = rule.get("atom_ids", [rule.get("atom_id")])
        for ligand in ligand_ids:
            vector = (
                record.surface_coordinates[ligand]
                - record.surface_coordinates[center]
            )
            assert np.linalg.norm(vector) == pytest.approx(
                model.bond_length, abs=1.0e-8
            )
            direction = vector / np.linalg.norm(vector)
            assert max(
                np.dot(direction, ideal) for ideal in ideal_directions
            ) == pytest.approx(1.0, abs=1.0e-8)


def test_terminal_two_bridge_bisector_is_planar_and_equal_angles() -> None:
    """CN3 + two bridges: terminal on H₂O-like ∠Cl–Cd–Cl bisector completion."""

    center = np.asarray([0.0, 0.0, 0.0])
    bridge_a = np.asarray([1.0, 0.0, 0.0])
    bridge_b = np.asarray([0.0, 1.0, 0.0])
    # Native is out of plane; the preferred site must still be coplanar with
    # the two bridges (σ_d plane of the Cl–Cd–Cl angle).
    native = np.asarray([0.2, 0.1, 1.0])
    bond_length = 2.5
    position = nucleation_module._terminal_position_two_bridge_bisector(
        center,
        bridge_a,
        bridge_b,
        bond_length=bond_length,
        native_ligand=native,
        cluster_com=np.asarray([0.0, 0.0, 0.0]),
    )
    assert position is not None
    vector = position - center
    assert np.linalg.norm(vector) == pytest.approx(bond_length, abs=1.0e-10)
    # Coplanar with Cd and the two bridges (z ≈ 0 for this setup).
    assert abs(float(position[2])) < 1.0e-10
    # Equal angles to both bridges.
    def _angle(left, right):
        return float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(left, right)
                        / (np.linalg.norm(left) * np.linalg.norm(right)),
                        -1.0,
                        1.0,
                    )
                )
            )
        )

    angle_a = _angle(vector, bridge_a - center)
    angle_b = _angle(vector, bridge_b - center)
    assert angle_a == pytest.approx(angle_b, abs=1.0e-8)
    # Completes opposite the bridge wedge, not between the bridges.
    assert angle_a > 90.0


def test_terminal_pair_c2v_two_fixed_equal_angles_and_tt() -> None:
    """P2: two terminals C2v about two fixed ligands (Se + bridge class)."""

    center = np.asarray([0.0, 0.0, 0.0])
    se = np.asarray([1.0, 0.0, 0.0])
    bridge = np.asarray([0.0, 1.0, 0.0])  # 90° fixed angle (off tet is OK)
    bond_length = 2.5
    native = (
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([0.0, 0.0, -1.0]),
    )
    pair = nucleation_module._terminal_pair_c2v_two_fixed(
        center,
        se,
        bridge,
        bond_length=bond_length,
        native_ligands=native,
    )
    assert pair is not None
    p1, p2 = pair

    def _angle(left, right):
        return float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(left, right)
                        / (np.linalg.norm(left) * np.linalg.norm(right)),
                        -1.0,
                        1.0,
                    )
                )
            )
        )

    v1, v2 = p1 - center, p2 - center
    assert np.linalg.norm(v1) == pytest.approx(bond_length, abs=1.0e-10)
    assert np.linalg.norm(v2) == pytest.approx(bond_length, abs=1.0e-10)
    # Equal angles to both fixed ligands (C2v).
    assert _angle(v1, se - center) == pytest.approx(
        _angle(v1, bridge - center), abs=1.0e-6
    )
    assert _angle(v2, se - center) == pytest.approx(
        _angle(v2, bridge - center), abs=1.0e-6
    )
    assert _angle(v1, se - center) == pytest.approx(
        _angle(v2, se - center), abs=1.0e-6
    )
    # Terminal–terminal ≈ tetrahedral.
    assert _angle(v1, v2) == pytest.approx(
        math.degrees(math.acos(-1.0 / 3.0)), abs=1.0e-4
    )
    # Both open vs the fixed wedge (not the acute CIF residual).
    assert _angle(v1, se - center) > 90.0


def test_terminal_se_bridge_bisector_is_planar_and_equal_angles() -> None:
    """CN3 Se + bridge + terminal: terminal on ∠Se–Cd–Cl bisector completion."""

    center = np.asarray([0.0, 0.0, 0.0])
    se = np.asarray([1.0, 0.0, 0.0])
    bridge = np.asarray([0.0, 1.0, 0.0])
    native = np.asarray([0.2, 0.1, 1.0])
    bond_length = 2.5
    position = nucleation_module._terminal_position_two_fixed_bisector(
        center,
        se,
        bridge,
        bond_length=bond_length,
        native_ligand=native,
        cluster_com=np.asarray([0.0, 0.0, 0.0]),
    )
    assert position is not None
    vector = position - center
    assert np.linalg.norm(vector) == pytest.approx(bond_length, abs=1.0e-10)
    assert abs(float(position[2])) < 1.0e-10

    def _angle(left, right):
        return float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(left, right)
                        / (np.linalg.norm(left) * np.linalg.norm(right)),
                        -1.0,
                        1.0,
                    )
                )
            )
        )

    angle_se = _angle(vector, se - center)
    angle_br = _angle(vector, bridge - center)
    assert angle_se == pytest.approx(angle_br, abs=1.0e-8)
    assert angle_se > 90.0


def test_k2_cn3_se_bridge_terminal_uses_bisector(cdse_k2_result) -> None:
    """Retained k=2 shells: CN3 Se+bridge+terminal on ∠Se–Cd–Cl bisector."""

    found = False
    for p, records in sorted(cdse_k2_result.registry[2].items()):
        for record in records:
            final_rules = [
                rule
                for rule in record.metadata["surface_geometry"]["applied_rules"]
                if rule["rule"] == "cd_cn3_se_bridge_terminal_bisector"
                and rule.get("pass") == "final"
            ]
            if not final_rules:
                continue
            found = True
            coords = record.surface_coordinates
            for rule in final_rules:
                center = int(rule["center_atom_id"])
                terminal = int(rule["atom_id"])
                se = int(rule["se_atom_id"])
                bridge = int(rule["bridge_ligand_id"])
                angle_se = _angle_at(coords, center, terminal, se)
                angle_br = _angle_at(coords, center, terminal, bridge)
                assert angle_se == pytest.approx(angle_br, abs=1.0e-5)
                assert angle_se > 90.0
                c = coords[center]
                va = coords[se] - c
                vb = coords[bridge] - c
                vt = coords[terminal] - c
                volume = abs(float(np.dot(vt, np.cross(va, vb))))
                scale = float(
                    np.linalg.norm(va) * np.linalg.norm(vb) * np.linalg.norm(vt)
                )
                assert volume / max(scale, 1.0e-12) < 1.0e-6
    assert found, (
        "expected at least one CN3 Se+bridge+terminal bisector on k=2 retained"
    )


def test_c2v_pair_opens_angles_vs_off_lattice_bridge() -> None:
    """Cd9-class numbers: Se + bridge at ~80°, two terminals must not go 66°/170°."""

    center = np.asarray([2.302703, -0.767568, -2.302703])
    se = np.asarray([0.767568, 0.767568, -0.767568])
    bridge = np.asarray([0.767568, -2.938577, -0.767568])
    bond = 2.658932855707733
    pair = nucleation_module._terminal_pair_c2v_two_fixed(
        center,
        se,
        bridge,
        bond_length=bond,
        native_ligands=(
            np.asarray([0.767568, -2.302703, -3.837839]),
            np.asarray([3.837839, 0.767568, -3.837839]),
        ),
    )
    assert pair is not None

    def _ang(i, j, c=center):
        va, vb = i - c, j - c
        return float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(va, vb)
                        / (np.linalg.norm(va) * np.linalg.norm(vb)),
                        -1.0,
                        1.0,
                    )
                )
            )
        )

    for pos in pair:
        assert _ang(pos, bridge) > 95.0
        assert _ang(pos, se) > 95.0
    assert abs(_ang(pair[0], bridge) - _ang(pair[1], bridge)) < 1.0e-4
    assert _ang(pair[0], pair[1]) == pytest.approx(
        math.degrees(math.acos(-1.0 / 3.0)), abs=1.0e-3
    )


def test_k1_p2_two_bridge_terminal_uses_bisector(cdse_k1_result) -> None:
    """Seed (1,2): host with two bridge Cl + one terminal uses ∠Cl–Cd–Cl bisector.

    On the shipped (1,2) graph the terminal sits on a CN4 Cd (Se + two bridges
    + terminal).  CIF residual alone can leave the Cl near both bridges (~66°);
    the bisector completion must equalize bridge angles and leave the Cl–Cd–Cl
    plane (coplanar with the two bridges).
    """

    record = cdse_k1_result.registry[1][2][0]
    assert record.surface_coordinates_data is not None
    geometry = record.metadata["surface_geometry"]
    assert geometry["projection_valid"] is True

    final_bisectors = [
        rule
        for rule in geometry["applied_rules"]
        if rule["rule"] == "cd_two_bridge_terminal_bisector"
        and rule.get("pass") == "final"
    ]
    assert final_bisectors, (
        "expected at least one two-bridge terminal bisector on (1,2); "
        f"rules={[r['rule'] for r in geometry['applied_rules']]}"
    )

    coords = record.surface_coordinates
    for rule in final_bisectors:
        center = int(rule["center_atom_id"])
        terminal = int(rule["atom_id"])
        bridges = [int(index) for index in rule["bridge_ligand_ids"]]
        assert len(bridges) == 2
        angle_a = _angle_at(coords, center, terminal, bridges[0])
        angle_b = _angle_at(coords, center, terminal, bridges[1])
        assert angle_a == pytest.approx(angle_b, abs=1.0e-5)
        # Must open the terminal away from the bridge wedge (not the ~66° CIF
        # residual that sits between the two bridges).
        assert angle_a > 90.0

        # Terminal coplanar with Cd and the two bridges.
        c = coords[center]
        va = coords[bridges[0]] - c
        vb = coords[bridges[1]] - c
        vt = coords[terminal] - c
        volume = abs(float(np.dot(vt, np.cross(va, vb))))
        scale = float(np.linalg.norm(va) * np.linalg.norm(vb) * np.linalg.norm(vt))
        assert volume / max(scale, 1.0e-12) < 1.0e-6


def test_surface_slot_rejection_writes_inspectable_xyz(
    tmp_path: Path,
    cdse_k2_result,
) -> None:
    rejected = next(
        record
        for record in cdse_k2_result.discarded_registry[2][4]
        if record.selection_reason == "surface_slot_conflict"
    )
    reduced = copy.deepcopy(cdse_k2_result)
    reduced.registry = {2: {4: []}}
    reduced.discarded_registry = {2: {4: [copy.deepcopy(rejected)]}}
    reduced.discarded_counts = {2: {4: 1}}
    bundle = write_nucleation_bundle(reduced, tmp_path / "surface_rejected")
    rejected_directory = bundle / "structures/k002/p004/discarded"
    assert list(rejected_directory.glob("*_construction_native.xyz"))
    surface_files = list(rejected_directory.glob("*_surface_rejected.xyz"))
    assert len(surface_files) == 1
    assert surface_files[0].read_text().splitlines()[1].endswith(
        "surface_projected_valid_false"
    )


def _neighbor_angles_for_test(coordinates, center, neighbors):
    return sorted(
        _angle_at(coordinates, center, left, right)
        for index, left in enumerate(neighbors)
        for right in neighbors[index + 1 :]
    )


def _angle_at(
    coordinates: np.ndarray,
    center: int,
    left: int,
    right: int,
) -> float:
    first = coordinates[left] - coordinates[center]
    second = coordinates[right] - coordinates[center]
    return float(
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(first, second)
                    / (np.linalg.norm(first) * np.linalg.norm(second)),
                    -1.0,
                    1.0,
                )
            )
        )
    )


def test_bare_seed_uses_minimum_cn_fallback(cdse_k1_result) -> None:
    seed = cdse_k1_result.registry[1][0][0]
    assert seed.metadata["coordination_by_element"] == {"Cd": [1], "Se": [1]}
    assert seed.metadata["min_cn_compliant"] is False
    assert seed.metadata["min_cn_violation_count"] == 2
    assert seed.metadata["min_cn_total_shortfall"] == 2
    assert seed.selection_reason == "minimum_cn_shortfall"


def test_k1_bins_formula_cn_and_forbidden_contacts(cdse_k1_result) -> None:
    assert sorted(cdse_k1_result.registry) == [1]
    assert sorted(cdse_k1_result.registry[1]) == [0, 1, 2, 3]
    max_cn = {"Cd": 4, "Se": 4, "Cl": 2}
    allowed = {frozenset(("Cd", "Se")), frozenset(("Cd", "Cl"))}
    reference = cdse_k1_result.reference_bond_length

    for registry in (
        cdse_k1_result.registry,
        cdse_k1_result.discarded_registry,
    ):
        for p, records in registry[1].items():
            for record in records:
                counts = Counter(record.symbols)
                assert counts["Cd"] == 1 + p
                assert counts["Se"] == 1
                assert counts["Cl"] == 2 * p
                assert record.metadata["formal_charge"] == 0
                assert record.metadata["total_cn"] == 2 * record.metadata[
                    "bond_count"
                ]
                for atom in record.atoms:
                    assert record.graph.degree[atom.atom_id] <= max_cn[atom.symbol]
                for left, right in record.graph.edges:
                    assert (
                        frozenset((record.symbols[left], record.symbols[right]))
                        in allowed
                    )
                for left in range(len(record.atoms)):
                    for right in range(left + 1, len(record.atoms)):
                        pair = frozenset(
                            (record.symbols[left], record.symbols[right])
                        )
                        distance = np.linalg.norm(
                            record.coordinates[left] - record.coordinates[right]
                        )
                        if pair not in allowed:
                            assert distance > reference + 0.20


def test_declared_building_blocks_must_each_be_charge_neutral() -> None:
    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    bad_core = replace(spec, charges={"Cd": 1, "Se": -2, "Cl": -1})
    with pytest.raises(ValueError, match="core monomer must be charge neutral"):
        nucleation_module._validate_spec(bad_core)

    bad_precursor = replace(spec, charges={"Cd": 2, "Se": -2, "Cl": -2})
    with pytest.raises(ValueError, match="precursor package must be charge neutral"):
        nucleation_module._validate_spec(bad_precursor)


def test_no_isomorphic_pair_survives_in_one_bin(cdse_k1_result) -> None:
    for p, retained in cdse_k1_result.registry[1].items():
        records = [
            *retained,
            *cdse_k1_result.discarded_registry[1].get(p, []),
        ]
        for index, left in enumerate(records):
            for right in records[index + 1 :]:
                assert not GraphMatcher(
                    left.graph,
                    right.graph,
                    node_match=_node_match,
                    edge_match=_edge_match,
                ).is_isomorphic()


def test_sweeps_stop_when_all_se_sites_are_occupied(cdse_k1_result) -> None:
    operations = Counter(audit.operation for audit in cdse_k1_result.sweep_audit)
    assert operations == {
        "dag_bin": 4,
        "skeleton_passivation": 4,
        "strip_validation": 3,
    }
    terminal = next(
        audit
        for audit in cdse_k1_result.sweep_audit
        if audit.operation == "skeleton_passivation" and audit.p_from == 3
    )
    assert terminal.valid_count == 0
    assert all(
        audit.stage_counts["ligand_enumerations"] == 0
        for audit in cdse_k1_result.sweep_audit
        if audit.operation == "strip_validation"
    )


def test_ligand_subsets_are_generated_as_symmetry_orbits(
    cdse_k1_result,
) -> None:
    p3 = next(
        audit for audit in cdse_k1_result.sweep_audit
        if audit.operation == "dag_bin" and audit.p_from == 3
    )
    assert p3.raw_count == 924
    assert p3.stage_counts["theoretical_assignments"] == 924
    assert p3.stage_counts["orbit_representatives"] == 5
    assert p3.stage_counts["identical_host_pruned"] == 919
    assert p3.stage_counts["duplicate_orbit_extensions_pruned"] > 0
    assert p3.stage_counts["bridge_symmetry_bases"] > 0
    assert p3.stage_counts["bridge_symmetry_pruned"] > 0
    assert (
        p3.stage_counts["bridge_raw_extensions"]
        > p3.stage_counts["bridge_orbit_representatives"]
    )
    p1 = next(
        audit for audit in cdse_k1_result.sweep_audit
        if audit.operation == "dag_bin" and audit.p_from == 1
    )
    assert p1.stage_counts["bridge_identity_fallback_bases"] > 0


def test_bridge_score_bound_dominates_small_exhaustive_variants(
    cdse_k1_unrestricted_result,
) -> None:
    # Unrestricted, so that k=1 p=1 still offers more than one arc set to bound.
    cdse_k1_result = cdse_k1_unrestricted_result
    spec = _with_min_bridged_host_cn(
        load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml"), 1, 1
    )
    model = _build_lattice_model(spec)
    records = [
        *cdse_k1_result.registry[1][1],
        *cdse_k1_result.discarded_registry[1][1],
    ]
    base_record = next(
        record
        for record in records
        if record.metadata.get("bridge_count", 0) == 0
    )
    base = _State(tuple(base_record.atoms), base_record.graph.copy())
    variants = _latent_bridge_variants(
        base, model, spec, prune_dominated=False
    )
    assert len(variants) > 1
    bound = _optimistic_bridge_score(base, spec)
    assert all(
        _graph_coordination_score(variant.atoms, variant.graph, spec) <= bound
        for variant in variants
    )


TOPOLOGY_BASELINE = ROOT / "tests/nucleation_topology_baseline.json"


def _load_topology_baseline(key: str) -> dict[str, list[tuple[str, ...]]]:
    raw = json.loads(TOPOLOGY_BASELINE.read_text(encoding="utf-8"))[key]
    return {
        bin_key: [tuple(signature) for signature in signatures]
        for bin_key, signatures in raw.items()
    }


@pytest.mark.parametrize("kmax", [1, 2])
def test_topology_digest_matches_recorded_baseline(
    kmax,
    cdse_k1_result,
    cdse_k2_result,
) -> None:
    """Guard the enumerated topologies against unintended change.

    The digest carries composition, an element-labelled isomorphism
    certificate, the coordination histogram, bridge character and the selection
    verdict -- but no coordinates, structure ids or search counters.  Any
    optimisation that only makes the search faster must leave it untouched; a
    change here is either a bug or a deliberate, reviewable chemistry change.
    """

    result = cdse_k1_result if kmax == 1 else cdse_k2_result
    expected = _load_topology_baseline(f"kmax{kmax}")
    actual = registry_digest(result)
    differences = digest_diff(expected, actual)
    assert not differences, "topology changed:\n" + "\n".join(differences)


def _with_min_bridged_host_cn(spec, value, kmax):
    rules = tuple(
        replace(rule, min_bridged_host_cn=value)
        for rule in spec.graph_rules.bridge_rules
    )
    return replace(
        spec, kmax=kmax,
        graph_rules=replace(spec.graph_rules, bridge_rules=rules),
    )


def test_dft_k1p2_three_bridges_hold_without_planar_distortion(
    cdse_k1_result,
) -> None:
    """DFT anchor: k=1 p=2 relaxes to three bridges, Cd at CN 3,3,4.

    The relaxation stays close to the construction geometry and the CN-3 Cd,
    each carrying two bridges, do **not** flatten to trigonal planar.  That is
    why the cation shift excludes bridge hosts: an earlier attempt to project
    them onto their fixed-neighbour plane moved a Cd far enough to compress
    Cd-Se from 2.659 to 1.300 A and collapse a Cd-Cl-Cd angle to 60 degrees.
    """

    record = cdse_k1_result.registry[1][2][0]
    assert record.metadata["bridge_count"] == 3
    assert record.metadata["coordination_by_element"]["Cd"] == [4, 3, 3]
    assert record.graph.number_of_edges() == 10

    rules = {
        str(rule["rule"])
        for rule in record.metadata["surface_geometry"]["applied_rules"]
    }
    assert "cd_cn3_three_se_plane" not in rules, (
        "a bridge-hosting CN-3 Cd was flattened to planar; DFT says it keeps "
        "its near-tetrahedral geometry"
    )


def test_dft_k2p0_cn2_cation_is_linear(cdse_k2_result) -> None:
    """DFT anchor: k=2 p=0 relaxes with Se-Cd-Se pseudo-linear.

    The projection places that Cd at the midpoint of its two Se, which is
    exactly 180 degrees but shortens Cd-Se from 2.659 to 2.171 A.  The
    compression looked wrong until the relaxation showed it converges to a
    pseudo-linear geometry, so it is a sound pre-relaxation guess rather than an
    error -- the surface view is explicitly a starting structure.
    """

    record = cdse_k2_result.registry[2][0][0]
    surface = record.surface_coordinates
    linear = []
    for atom in record.atoms:
        neighbors = sorted(record.graph.neighbors(atom.atom_id))
        if atom.symbol != "Cd" or len(neighbors) != 2:
            continue
        first = surface[neighbors[0]] - surface[atom.atom_id]
        second = surface[neighbors[1]] - surface[atom.atom_id]
        angle = np.degrees(
            np.arccos(
                np.clip(
                    first @ second
                    / (np.linalg.norm(first) * np.linalg.norm(second)),
                    -1.0,
                    1.0,
                )
            )
        )
        linear.append(angle)
    assert linear, "expected a CN-2 Cd in k=2 p=0"
    assert all(angle == pytest.approx(180.0, abs=1e-4) for angle in linear), linear


def test_dft_k1p1_min_bridged_host_cn_reproduces_the_relaxed_structure() -> None:
    """DFT anchor: k=1 p=1 relaxes to Cd[2,2] with terminal Cl and no bridge.

    By default the code retains the 5-bond bridged Cd[3,2], which scores higher
    on bond count.  With ``min_bridged_host_cn: 3`` -- no bridge may leave a
    cation at CN 2 -- the retained structure becomes the relaxed one, while
    k=1 p=2's three bridges are untouched because their cations are at CN 3 and
    4.  One rule, one negative and one positive observation.

    The rule is stated on the finished structure on purpose.  Phrasing it as a
    minimum coordination for the bridge *donor* does nothing at all: the same
    structure is reachable from a ligand arrangement whose donor is
    three-coordinate, and the route-merging DAG rebuilds it there.  This test
    pins the working formulation against that regression.
    """

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    # The rule now ships on by default; the pre-DFT behaviour is the
    # counterfactual, kept runnable as cdse_cdcl2_dft_rules.yaml.
    assert spec.graph_rules.bridge_rules[0].min_bridged_host_cn == 3

    default = generate_nucleation_result(_with_min_bridged_host_cn(spec, 1, 1))
    baseline = default.registry[1][1][0]
    assert baseline.metadata["bridge_count"] == 1
    assert baseline.metadata["coordination_by_element"]["Cd"] == [3, 2]

    ruled = generate_nucleation_result(_with_min_bridged_host_cn(spec, 3, 1))
    relaxed = ruled.registry[1][1][0]
    assert relaxed.metadata["bridge_count"] == 0
    assert relaxed.graph.number_of_edges() == 4
    assert relaxed.metadata["coordination_by_element"] == {
        "Cd": [2, 2],
        "Cl": [1, 1],
        "Se": [2],
    }
    assert relaxed.metadata["min_cn_compliant"] is True

    # k=1 p=2 is unaffected: its bridges sit between CN 3 and CN 4 cations.
    held = ruled.registry[1][2][0]
    assert held.metadata["bridge_count"] == 3
    assert held.metadata["coordination_by_element"]["Cd"] == [4, 3, 3]


def test_dft_k2p3_retains_the_relaxed_family() -> None:
    """DFT anchor: k=2 p=3 relaxations pick the Se[3,3] family, not the default.

    Three isomers of Cd5Se2Cl6 were relaxed and land within 0.87 kcal/mol of each
    other.  Matched by graph certificate they are the three structures this bin
    retains under ``min_bridged_host_cn: 3`` (a fourth graph-isomorph used to
    pass surface gate with CIF residual terminals but is rejected once the
    two-bridge terminal sits on the ∠Cl–Cd–Cl bisector).  Without the host-CN
    rule the bin retains a single 18-edge ``Cd[2,4,4,4,4] Se[2,4]`` structure
    that is not any of them -- one extra bond bought by stranding a Cd and an
    Se at CN 2, because ``edges`` sits ahead of the evenness term in the score
    tuple.

    Bond count is deliberately *not* asserted as a stability proxy here: across
    the three relaxations it went 17->16, 17->16 and 17->18 while the energies
    stayed within 1 kcal/mol, so it neither orders them nor even moves in a
    consistent direction.
    """

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")

    ruled = generate_nucleation_result(_with_min_bridged_host_cn(spec, 3, 2))
    family = ruled.registry[2][3]
    assert len(family) == 3

    for record in family:
        assert record.graph.number_of_edges() == 17
        coordination = record.metadata["coordination_by_element"]
        assert coordination["Se"] == [3, 3]
        environments = sorted(
            (
                sum(
                    1
                    for neighbor in record.graph.neighbors(atom.atom_id)
                    if record.atoms[neighbor].symbol == "Se"
                ),
                sum(
                    1
                    for neighbor in record.graph.neighbors(atom.atom_id)
                    if record.atoms[neighbor].symbol == "Cl"
                ),
            )
            for atom in record.atoms
            if atom.symbol == "Cd"
        )
        assert environments == [(1, 2), (1, 2), (1, 2), (1, 3), (2, 2)]

    certificates = {
        graph_certificate(record.graph, record.atoms) for record in family
    }
    assert len(certificates) == 3, "the three retained structures must be distinct"

    # Negative control: without the rule the bin collapses to a structure that
    # is not in the relaxed family at all.
    unruled = generate_nucleation_result(_with_min_bridged_host_cn(spec, 1, 2))
    (other,) = unruled.registry[2][3]
    assert other.graph.number_of_edges() == 18
    assert sorted(other.metadata["coordination_by_element"]["Se"]) == [2, 4]
    assert sorted(other.metadata["coordination_by_element"]["Cd"]) == [2, 4, 4, 4, 4]
    assert graph_certificate(other.graph, other.atoms) not in certificates


def test_dft_k2p1_prefers_the_unbridged_structure(
    cdse_k2_result, cdse_k2_unrestricted_result,
) -> None:
    """DFT anchor: the strongest single datum for ``min_bridged_host_cn: 3``.

    k=2 p=1 (Cd3Se2Cl2) is where the rule is most aggressive -- it drops the bin
    from 8 bonds to 6.  Relaxed, both structures keep their connectivity exactly,
    so both are genuine minima, and the rule's 6-bond ``Cd[2,2,2]`` comes out
    **5.4 kcal/mol below** the 8-bond bridged ``Cd[2,3,3]`` (-187.101563 vs
    -187.092894 Ha).  Two fewer bonds, decisively lower, well outside the
    ~2 kcal/mol DFT error -- which is why the rule ships on by default.
    """

    (ruled,) = cdse_k2_result.registry[2][1]
    assert ruled.graph.number_of_edges() == 6
    assert ruled.metadata["bridge_count"] == 0
    assert sorted(ruled.metadata["coordination_by_element"]["Cd"]) == [2, 2, 2]
    assert sorted(ruled.metadata["coordination_by_element"]["Cl"]) == [1, 1]

    (unruled,) = cdse_k2_unrestricted_result.registry[2][1]
    assert unruled.graph.number_of_edges() == 8
    assert unruled.metadata["bridge_count"] == 2
    assert sorted(unruled.metadata["coordination_by_element"]["Cd"]) == [2, 3, 3]


def test_pauling_valence_is_reported_and_changes_nothing(cdse_k2_result) -> None:
    """The valence diagnostic must be observable and completely inert.

    It exists because three k=2 p=3 relaxations agreed 3/3 on which cation
    sheds a ligand, and bond count did not.  Three points is too thin to rank
    on, so this asserts the number is emitted *and* that emitting it leaves the
    enumerated topology untouched -- the moment it starts steering selection,
    the recorded baselines would move and this guard would fire.
    """

    record = cdse_k2_result.registry[1][2][0]
    block = record.metadata["surface_geometry"]["pauling_valence"]

    # k=1 p=2 is Cd3SeCl4 with Cd[4,3,3]: Se is CN 3, and of the four Cl three
    # bridge (CN 2, worth 1/2 each) and one is terminal (CN 1, worth 1).  The
    # CN-4 Cd carries two bridges plus the terminal Cl, so it sits at
    # 2/3 + 2*(1/2) + 1 = 2.667; the two CN-3 Cd carry two bridges each, at
    # 2/3 + 2*(1/2) = 1.667.  Only the first is oversaturated.
    assert block["max_pauling_valence"] == pytest.approx(2.666667, abs=1e-6)
    assert sorted(
        item["pauling_valence"] for item in block["cations"]
    ) == pytest.approx([1.666667, 1.666667, 2.666667], abs=1e-6)
    assert {item["atom_id"] for item in block["cations"]} == {
        atom.atom_id for atom in record.atoms if atom.symbol == "Cd"
    }
    for item in block["cations"]:
        assert item["target"] == 2.0
        assert item["deviation"] == pytest.approx(
            item["pauling_valence"] - 2.0, abs=1e-9
        )

    # Every cation of every retained structure carries the field.
    for bins in cdse_k2_result.registry.values():
        for records in bins.values():
            for item in records:
                reported = item.metadata["surface_geometry"]["pauling_valence"]
                assert reported["max_pauling_valence"] is not None

    # Inertness: the digest is pinned by the recorded baseline, so agreement
    # with it is the proof that adding the diagnostic re-ranked nothing.
    assert registry_digest(cdse_k2_result) == _load_topology_baseline("kmax2")


def test_min_bridged_host_cn_holds_on_every_retained_structure() -> None:
    """No retained structure may violate the rule that produced it."""

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    result = generate_nucleation_result(_with_min_bridged_host_cn(spec, 3, 2))
    checked = 0
    for k, bins in result.registry.items():
        for p, records in bins.items():
            for record in records:
                for bridge in record.metadata["bridge_edges"]:
                    hosts = (
                        bridge["primary_host_atom_id"],
                        bridge["host_atom_id"],
                    )
                    for host in hosts:
                        assert record.graph.degree[host] >= 3, (
                            f"k={k} p={p} {record.structure_id}: bridge host "
                            f"{host} is at CN {record.graph.degree[host]}"
                        )
                        checked += 1
    assert checked > 0, "no bridges survived, so the rule was not exercised"


def test_exact_mode_declares_completeness_and_guided_declares_loss() -> None:
    """A consumer must never have to infer completeness from a missing warning.

    Two independent approximations exist -- guided ligand placement, and
    narrowing skeleton growth to retained cores -- and each must name itself in
    ``registry.json``.  Equally, an exact run must positively assert that it was
    exact rather than merely stay quiet.
    """

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")

    exact_messages: list[str] = []
    exact = generate_nucleation_result(
        replace(spec, kmax=2), progress=exact_messages.append
    )
    report = exact.completeness
    assert report["mode"] == "exact"
    assert report["approximations"] == []
    assert report["enumeration_complete_through_k"] == 2
    assert report["guarantees"], "an exact run must state its guarantees"
    assert not any("WARNING" in message for message in exact_messages)

    guided_messages: list[str] = []
    guided = generate_nucleation_result(
        replace(spec, kmax=2, mode="guided"), progress=guided_messages.append
    )
    report = guided.completeness
    assert report["mode"] == "guided"
    stages = {item["stage"] for item in report["approximations"]}
    assert "ligand_placement" in stages
    assert report["enumeration_complete_through_k"] == 0
    assert any("WARNING" in message for message in guided_messages), (
        "an approximate run must warn on the progress stream, not only in JSON"
    )

def test_guided_never_outscores_exact() -> None:
    """Guided must never beat exact -- if it does, exact missed a structure.

    Guided places one ligand shell per skeleton, and that shell is one of the
    arrangements the exact run enumerates, so the exact optimum can only be
    better or equal.  This is a statement about *scores*, not counts: a
    lower-scoring layer can be attained by more tied structures, so guided
    legitimately retains more records than exact in some bins.

    This is the property that caught a real completeness bug.  Bases used to be
    merged on graph isomorphism alone, but a ``shared_vacant_cif_site`` bridge
    needs two cations to share a vacant anion site -- a property of the
    coordinates.  Graph-isomorphic bases could therefore offer different
    bridges, and keeping one representative discarded the other's options: at
    k=2 p=1 guided found a min-CN-compliant Cd=[3,3,2] at (1,0,0,8,0,0,-3) while
    exact retained Cd=[4,2,2] at (1,0,0,8,0,0,-4), with no CIF-site bridge
    anywhere in that bin.  Base equivalence now includes the bridge options
    (``_bridge_opportunity_graph``), and this asserts it stays that way.
    """

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    exact = generate_nucleation_result(replace(spec, kmax=2))
    guided = generate_nucleation_result(replace(spec, kmax=2, mode="guided"))

    offenders = []
    for k, bins in guided.registry.items():
        for p, records in bins.items():
            reference = exact.registry.get(k, {}).get(p, [])
            if not records or not reference:
                continue
            if records[0].coordination_score > reference[0].coordination_score:
                offenders.append(
                    f"k={k} p={p}: guided={records[0].coordination_score[:7]} "
                    f"Cd={records[0].metadata['coordination_by_element']['Cd']} "
                    f"beat exact={reference[0].coordination_score[:7]} "
                    f"Cd={reference[0].metadata['coordination_by_element']['Cd']}"
                )
    assert not offenders, (
        "exact enumeration missed structures guided found:\n  "
        + "\n  ".join(offenders)
    )


def test_guided_mode_still_obeys_the_hard_coordination_rules() -> None:
    """Approximate placement may sample, but must not break the chemistry.

    Max-CN caps, allowed bonds and the surface gate are hard constraints; the
    guided path routes through the same ``_state_valid`` and surface projection,
    and this pins that it really does.
    """

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    result = generate_nucleation_result(replace(spec, kmax=2, mode="guided"))
    allowed = {frozenset(pair) for pair in spec.graph_rules.allowed_bonds}
    seen = 0
    for k, bins in result.registry.items():
        for p, records in bins.items():
            for record in records:
                seen += 1
                assert Counter(record.symbols)["Cl"] == 2 * p
                for atom in record.atoms:
                    degree = record.graph.degree[atom.atom_id]
                    assert degree <= spec.graph_rules.max_cn[atom.symbol]
                for left, right in record.graph.edges:
                    pair = frozenset(
                        (record.symbols[left], record.symbols[right])
                    )
                    assert pair in allowed
                assert record.metadata["surface_geometry"]["projection_valid"]
    assert seen > 0


def test_narrowed_growth_reports_only_a_binding_cut_as_loss() -> None:
    """Switching the narrowing rule on is not itself a loss of completeness.

    If every skeleton in the row happened to be a retained core, nothing was
    dropped and the run is still exact.  Claiming otherwise would train the
    reader to ignore the warning.
    """

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    # k=1 has one skeleton per bin and each is retained, so narrowing from k=1
    # is active but cuts nothing.
    result = generate_nucleation_result(
        replace(spec, kmax=2, exact_through_k=1)
    )
    stages = {
        item["stage"] for item in result.completeness["approximations"]
    }
    assert "skeleton_growth" not in stages, (
        "narrowing that dropped no skeleton was reported as a loss"
    )
    assert any(
        "dropped no skeleton" in guarantee
        for guarantee in result.completeness["guarantees"]
    )


def test_soft_retain_band_keeps_more_than_top_score_layer() -> None:
    """retain_score_layers>1 widens the retained set without breaking chemistry."""

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    default = generate_nucleation_result(replace(spec, kmax=1))
    banded = generate_nucleation_result(
        replace(spec, kmax=1, retain_score_layers=3, retain_max_per_bin=50)
    )
    default_n = sum(len(v) for v in default.registry[1].values())
    banded_n = sum(len(v) for v in banded.registry[1].values())
    assert banded_n >= default_n
    # Top-layer winners remain retained under a wider band.
    for p, records in default.registry[1].items():
        default_scores = {r.coordination_score for r in records}
        banded_scores = {
            r.coordination_score for r in banded.registry[1].get(p, ())
        }
        assert default_scores <= banded_scores
    assert banded.completeness["retain_score_layers"] == 3
    stages = {a["stage"] for a in banded.completeness["approximations"]}
    assert "soft_retain_band" in stages


def test_seed_band_parent_mode_limits_growth_sources() -> None:
    """Legacy parent_p_mode=seed_band filters which p bins feed k→k+1."""

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    result = generate_nucleation_result(
        replace(
            spec,
            kmax=2,
            exact_through_k=1,
            seed_p=2,
            seed_p_window=0,
            parent_p_mode="seed_band",
            mode="guided",
        )
    )
    assert 2 in result.registry
    stages = {a["stage"] for a in result.completeness["approximations"]}
    assert "parent_p_filter" in stages
    assert sum(len(v) for v in result.registry[2].values()) >= 1


def test_building_block_p0_accounting() -> None:
    """Product p0 = p_parent - shed + p_m."""

    from builder.nucleation import _product_p0, _monomer_packages
    from builder.nucleation import load_nucleation_spec
    from dataclasses import replace

    assert _product_p0(4, 0, 2) == 6
    assert _product_p0(4, 1, 2) == 5
    assert _product_p0(2, 2, 0) == 0
    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    bb = replace(
        spec,
        monomer_p_values=(0, 1, 2),
        parent_p_mode="all_retained",
        p_ladder_mode="product_window",
    )
    assert _monomer_packages(bb) == (0, 1, 2)
    # Empty packages + seed_p derives a band.
    derived = replace(spec, seed_p=2, seed_p_window=1, monomer_p_values=())
    assert _monomer_packages(derived) == (1, 2, 3)


def test_empty_retained_bins_pruned_from_registry() -> None:
    """Registry must not keep (k,p) keys with zero retained structures."""

    from builder.nucleation import _prune_empty_retained_bins

    registry = {
        1: {0: ["a"], 1: []},  # type: ignore[list-item]
        2: {3: []},
        3: {1: ["b"], 2: ["c"]},  # type: ignore[list-item]
    }
    # Use empty lists of real type
    registry = {
        1: {0: [], 1: []},
        2: {0: []},
    }
    # properly typed with ClusterRecord-free stub: function only checks truthiness
    _prune_empty_retained_bins(registry)  # type: ignore[arg-type]
    assert registry == {}


def test_bridge_terminal_prefers_slot_away_from_bridge_cl() -> None:
    """On a bridged Cd, terminal Cl should not pick the near-bridge CIF slot."""

    import numpy as np
    from builder.nucleation import (
        _build_lattice_model,
        _cif_tetrahedral_terminal_positions,
        load_nucleation_spec,
    )

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    model = _build_lattice_model(spec)
    # Synthetic tetrahedron around origin; bond length scaled.
    bl = model.bond_length
    # Ideal zb-like directions (normalized then scaled).
    dirs = [
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, -1.0, -1.0]),
        np.array([-1.0, 1.0, -1.0]),
        np.array([-1.0, -1.0, 1.0]),
    ]
    dirs = [d / np.linalg.norm(d) * bl for d in dirs]
    # atoms: 0=Cd center, 1=Se, 2=bridge Cl (fixed), 3=terminal Cl
    # Put native terminal near slot 1 (adjacent to bridge on slot 0).
    surface = np.zeros((4, 3))
    surface[0] = 0.0
    surface[1] = dirs[2]  # Se
    surface[2] = dirs[0]  # bridge Cl direction
    surface[3] = dirs[1]  # native-ish near adjacent slot
    native = surface.copy()
    # Without repulsion, native bias keeps slot near bridge; with repulsion,
    # slot opposite bridge (dirs[3] side) should win.
    pos_native_bias, _ = _cif_tetrahedral_terminal_positions(
        center=0,
        terminal_ligands=[3],
        fixed_neighbors=[1, 2],
        surface=surface,
        native=native,
        model=model,
        spec=spec,
        repulsive_neighbors=(),
    )
    pos_repel, _ = _cif_tetrahedral_terminal_positions(
        center=0,
        terminal_ligands=[3],
        fixed_neighbors=[1, 2],
        surface=surface,
        native=native,
        model=model,
        spec=spec,
        repulsive_neighbors=[2],
    )
    assert pos_repel is not None and 3 in pos_repel
    bridge_pos = surface[2]
    d_repel = float(np.linalg.norm(pos_repel[3] - bridge_pos))
    if pos_native_bias is not None:
        d_native = float(np.linalg.norm(pos_native_bias[3] - bridge_pos))
        assert d_repel + 1.0e-6 >= d_native
    # Absolute: terminal should not sit in the near-bridge hemisphere only;
    # angle to bridge direction should be the larger available slot.
    center = surface[0]
    u_bridge = (bridge_pos - center) / np.linalg.norm(bridge_pos - center)
    u_term = (pos_repel[3] - center) / np.linalg.norm(pos_repel[3] - center)
    # Opposite-ish: cos should be negative for tetrahedral opposite pair.
    assert float(np.dot(u_bridge, u_term)) < 0.0


def test_soft_clash_rejects_near_coincident_atoms() -> None:
    """Continuous decoration must not keep Cl ~0.3 Å apart."""

    import numpy as np
    from builder.nucleation import (
        _State,
        _build_lattice_model,
        _make_core_graph,
        _place_n_ligands_free_sites,
        _soft_clash_radius,
        _state_has_soft_clashes,
        AtomRecord,
        load_nucleation_spec,
    )

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    model = _build_lattice_model(spec)
    radius = _soft_clash_radius(model)
    assert radius > 1.0  # much stricter than site_tolerance alone
    # Two Cl almost on top of each other on a tiny fragment.
    atoms = (
        AtomRecord(0, "Cd", (0.0, 0.0, 0.0), "core_cation"),
        AtomRecord(1, "Se", (2.6, 0.0, 0.0), "core_anion"),
        AtomRecord(2, "Cl", (0.0, 0.0, 2.6), "precursor_ligand"),
        AtomRecord(3, "Cl", (0.0, 0.1, 2.7), "precursor_ligand"),
    )
    state = _make_core_graph(atoms, model, spec)
    assert _state_has_soft_clashes(state, model)
    # Free-site placer must refuse to add a ligand on top of an existing one.
    # Start with only one Cl; try to place another on an occupied direction.
    clean = _make_core_graph(atoms[:3], model, spec)
    # Over-request ligands so the placer would need free sites; if it returns
    # a state, it must be clash-free.
    placed = _place_n_ligands_free_sites(clean, 4, model, spec)
    if placed is not None:
        assert not _state_has_soft_clashes(placed, model)


def test_building_block_all_retained_parents_reach_high_p() -> None:
    """With packages and product_window, k=2 is not stuck near p<=3 only."""

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    result = generate_nucleation_result(
        replace(
            spec,
            kmax=2,
            exact_through_k=1,
            mode="guided",
            monomer_p_values=(0, 1, 2),
            parent_p_mode="all_retained",
            p_ladder_mode="product_window",
            k_growth_max_shed=2,
            k_growth_max_add=2,
            core_growth_occupation="decorated",
            continuous_decoration=True,
        )
    )
    assert 2 in result.registry
    max_p = max(result.registry[2])
    # (1,2)+(1,2) → p0=4, + max_add 2 → at least p=4 available if chemistry allows.
    assert max_p >= 4, f"expected ligand-rich bins at k=2, got max_p={max_p}"
    assert result.completeness["parent_p_mode"] == "all_retained"
    assert result.completeness["p_ladder_mode"] == "product_window"
    assert any(
        a.operation == "continuous_passivation" for a in result.sweep_audit
    )
    # Lineage tags for ligand-diffusion sorting on the same skeleton.
    for bins in result.registry.values():
        for records in bins.values():
            for rec in records:
                assert rec.metadata.get("skeleton_family_id", "").startswith(
                    "fam_"
                )
                assert rec.metadata.get("ligand_shell_hash")


def test_redecorated_growth_materializes_injected_precursor_centers(
    cdse_k1_result,
) -> None:
    """A non-continuous p_m channel must add its Cd before Cl rebuilding.

    This is the pathway used by ``05_pathway_k6_redecorated_calibration.yaml``:
    the decorated parent controls accessible attachment sites, then Cl is
    stripped and rebuilt in the destination bin.  Before the fix, p=0 + p_m=1
    was labeled as destination p=1 while still containing zero precursor Cd,
    eventually producing a charged k=2/p=1 record.
    """

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    spec = replace(
        spec,
        core_growth_occupation="decorated",
        continuous_decoration=False,
        monomer_p_values=(0, 1),
        k_growth_max_shed=1,
        k_growth_max_add=1,
        p_ladder_mode="product_window",
    )
    model = _build_lattice_model(spec)
    records = cdse_k1_result.registry[1][0]
    by_p, attempted, stats = nucleation_module._decorated_core_children_by_p(
        records,
        k_from=1,
        p=0,
        p_m=1,
        model=model,
        spec=spec,
    )
    assert attempted > 0
    assert by_p.get(1)
    assert stats["package_center_placements_attempted"] > 0
    for child, _routes in by_p[1]:
        roles = Counter(atom.role for atom in child.atoms)
        assert roles == {
            "core_cation": 2,
            "core_anion": 2,
            "precursor_center": 1,
        }
        assert nucleation_module._formal_charge(child.atoms, spec.charges) == 2


def test_decorated_growth_blocks_ligand_occupied_sites() -> None:
    """Decorated occupation must not invent free sites under Cl.

    On a fully passivated k=1 parent, bare strip-growth can place the next
    monomer on directions Cl already fills; decorated growth must use the
    passivated graph so occupation checks see those atoms.
    """

    from builder.nucleation import (
        _build_lattice_model,
        _decorated_core_children_by_p,
        _monomer_pair_placements,
        _make_core_graph,
        _retained_core_sources,
        _core_skeleton_children,
        _bare_package_core_children,
    )

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    result = generate_nucleation_result(replace(spec, kmax=1, mode="guided"))
    # Prefer a high-p bin where ligands pack the surface.
    p_candidates = sorted(
        (p for p, recs in result.registry[1].items() if recs), reverse=True
    )
    assert p_candidates
    p = p_candidates[0]
    records = result.registry[1][p]
    model = _build_lattice_model(spec)

    bare_sources = _retained_core_sources(records, model, spec)
    bare_by_p, bare_attempted, _ = _bare_package_core_children(
        bare_sources, k_from=1, p=p, model=model, spec=spec, p_m=0
    )
    bare_children = [child for items in bare_by_p.values() for child, _ in items]
    dec_by_p, dec_attempted, dec_stats = _decorated_core_children_by_p(
        records, k_from=1, p=p, model=model, spec=spec
    )
    dec_children = sum(len(v) for v in dec_by_p.values())
    # With capacity-only shedding, a decorated parent may first shed packages
    # and then expose additional sites.  The per-parent vacancy comparison
    # below still checks that Cl occupation itself never invents a site.
    # Vacancy helper: counting ligands as neighbors yields fewer-or-equal free sites.
    parent = _make_core_graph(records[0].atoms, model, spec)
    bare_pairs = _monomer_pair_placements(
        parent, model, spec, count_ligands_as_neighbors=False
    )
    dec_pairs = _monomer_pair_placements(
        parent, model, spec, count_ligands_as_neighbors=True
    )
    assert len(dec_pairs) <= len(bare_pairs)


def test_shedding_removes_a_complete_neutral_precursor_package(
    cdse_k1_result,
) -> None:
    """Bridging Cl must not be left behind when its CdCl2 unit is shed."""

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    spec = replace(spec, k_growth_max_shed=1, p_surf_beta=0.0)
    model = _build_lattice_model(spec)
    record = cdse_k1_result.registry[1][2][0]
    state = nucleation_module._make_core_graph(record.atoms, model, spec)

    variants = nucleation_module._shed_parent_variants(
        state, 2, model, spec, k=1
    )
    shed = [item for item in variants if item[2] == 1]
    assert shed
    for stripped, p_out, shed_count in shed:
        roles = Counter(atom.role for atom in stripped.atoms)
        assert shed_count == 1
        assert p_out == 1
        assert roles == {
            "core_cation": 1,
            "core_anion": 1,
            "precursor_center": 1,
            "precursor_ligand": 2,
        }
        assert nucleation_module._formal_charge(stripped.atoms, spec.charges) == 0


def test_surface_cap_prunes_instead_of_relabelling_composition(
    cdse_k1_result,
) -> None:
    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    spec = replace(
        spec,
        p_surf_beta=0.1,
        k_growth_max_shed=0,
        continuous_decoration=True,
    )
    model = _build_lattice_model(spec)
    records = cdse_k1_result.registry[1][2][:1]
    by_p, attempted, stats = nucleation_module._decorated_core_children_by_p(
        records, k_from=1, p=2, model=model, spec=spec
    )
    assert by_p == {}
    assert attempted == 0
    assert stats["surface_channel_cap_pruned"] == 1


def test_guided_yaml_loads_new_growth_controls() -> None:
    spec = load_nucleation_spec(
        ROOT / "examples/nucleation/cdse_cdcl2_guided.yaml"
    )
    assert spec.retain_score_layers == 2
    assert spec.retain_max_per_bin == 8
    assert spec.growth_max_parents_per_bin == 4
    assert spec.core_growth_occupation == "decorated"
    assert spec.continuous_decoration is True
    assert spec.monomer_p_values == (1, 2)
    assert spec.parent_p_mode == "all_retained"
    assert spec.p_ladder_mode == "product_window"
    assert spec.p_surf_beta == pytest.approx(3.0)
    assert spec.shed_alpha == pytest.approx(1.0)
    # Hard shed/add caps off when surface law is active (0 / -1).
    assert spec.k_growth_max_shed == 0
    assert spec.k_growth_max_add == -1


def test_retain_max_per_bin_is_per_kp_not_global() -> None:
    """Cap applies independently to each (k, p) bin."""

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    result = generate_nucleation_result(
        replace(spec, kmax=1, retain_score_layers=5, retain_max_per_bin=1)
    )
    for p, records in result.registry[1].items():
        assert len(records) <= 1, f"bin p={p} exceeded per-bin cap"


def test_checkpoint_flushes_each_retained_bin(tmp_path: Path) -> None:
    """Structures and partial JSON appear as each (k,p) is retained."""

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    out = tmp_path / "nuc_out"
    result = generate_nucleation_result(
        replace(spec, kmax=1, mode="guided"),
        checkpoint_dir=out,
        verbose=False,
    )
    assert (out / "checkpoint" / "k001" / "DONE").is_file()
    assert (out / "checkpoint" / "k001" / "progress.json").is_file()
    progress = json.loads(
        (out / "checkpoint" / "k001" / "progress.json").read_text()
    )
    assert progress["status"] == "done"
    # Per-bin folders under structures/k001/pXXX/retained/
    for p in result.registry[1]:
        retained_dir = out / "structures" / "k001" / f"p{p:03d}" / "retained"
        assert retained_dir.is_dir(), f"missing {retained_dir}"
        assert list(retained_dir.glob("*.xyz")), f"empty {retained_dir}"


K3_TOPOLOGY_BASELINE = ROOT / "tests/nucleation_topology_baseline_k3.json"


@pytest.mark.skipif(
    not os.environ.get("QD_NUCLEATION_K3_GUARD"),
    reason="k=3 costs minutes; set QD_NUCLEATION_K3_GUARD=1 to run",
)
def test_k3_retained_topology_matches_recorded_baseline() -> None:
    """Guard the k=3 retained set, which no other test observes.

    Everything else here stops at k=2, so a change above that passes silently --
    and it has: adding bridge mode to the certificate left k<=2 untouched while
    un-merging a genuine pair at k=3 p=7.  Any change that narrows the DAG above
    k=2 must be measured against this, not against the k<=2 baselines.

    Retained only, deliberately: the k>2 discarded count is a documented lower
    bound that shrinks as pruning improves (see registry.json ``completeness``),
    so pinning it would fight the optimisation instead of guarding the chemistry.

    Opt-in because one run is minutes, not seconds.  Re-record only with a
    reviewed diff::

        QD_NUCLEATION_K3_GUARD=1 PYTHONPATH=src:tests python -c "
        import json, pathlib
        from dataclasses import replace
        from builder.nucleation import load_nucleation_spec, generate_nucleation_result
        from _nucleation_reference import registry_digest
        spec = load_nucleation_spec('examples/nucleation/cdse_cdcl2.yaml')
        res = generate_nucleation_result(replace(spec, kmax=3))
        d = registry_digest(res, retained_only=True)
        pathlib.Path('tests/nucleation_topology_baseline_k3.json').write_text(
            json.dumps({k: [list(s) for s in v] for k, v in d.items()},
                       indent=2, sort_keys=True) + chr(10))
        "
    """

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    result = generate_nucleation_result(replace(spec, kmax=3))
    raw = json.loads(K3_TOPOLOGY_BASELINE.read_text(encoding="utf-8"))
    expected = {
        key: [tuple(signature) for signature in signatures]
        for key, signatures in raw.items()
    }
    actual = registry_digest(result, retained_only=True)
    differences = digest_diff(expected, actual)
    assert not differences, "k=3 retained topology changed:\n" + "\n".join(
        differences
    )


BUNDLE_BASELINE = ROOT / "tests/nucleation_bundle_baseline.json"


def test_bundle_bytes_match_recorded_baseline(tmp_path: Path) -> None:
    """Catch churn in user-visible output that the topology digest cannot see.

    The digest sorts its signatures, so it is deliberately blind to *ordering*.
    But record order decides which isomer is labelled ``iso0001``, and therefore
    the XYZ filenames, the ``registry.json`` ordering and the log tables.  A
    refactor that reorders equivalent structures changes every one of those for
    no scientific gain, so pin the bytes.

    Regenerate deliberately with::

        PYTHONPATH=src python -c "import json,pathlib,hashlib,tempfile; ..."

    and review the diff -- never to make this test pass.
    """

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    result = generate_nucleation_result(spec)
    out = tmp_path / "bundle"
    write_nucleation_bundle(result, out)

    digests = {}
    for path in sorted(out.rglob("*")):
        if path.is_file():
            digests[str(path.relative_to(out))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

    expected = json.loads(BUNDLE_BASELINE.read_text(encoding="utf-8"))
    assert sorted(digests) == sorted(expected), (
        "bundle file set changed:\n"
        f"  added:   {sorted(set(digests) - set(expected))}\n"
        f"  removed: {sorted(set(expected) - set(digests))}"
    )
    changed = [name for name in sorted(digests) if digests[name] != expected[name]]
    assert not changed, "bundle contents changed for:\n  " + "\n  ".join(changed)


def test_topology_digest_is_stable_across_regeneration(cdse_k1_result) -> None:
    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    regenerated = generate_nucleation_result(spec)
    assert registry_digest(regenerated) == registry_digest(cdse_k1_result)


def test_every_bin_respects_the_derived_p_at_most_three_k_bound(
    cdse_k2_result,
) -> None:
    """``p <= 3k`` follows from the hard rules, so no bin may exceed it.

    Every cation needs at least one anion neighbour and the only anions are the
    ``k`` core Se, each capped at CN 4.  So cation-anion bonds number at most
    ``4k`` while there are ``k + p`` cations, giving ``k + p <= 4k``.
    """

    for k, bins in cdse_k2_result.registry.items():
        for p in bins:
            assert p <= 3 * k, f"bin k={k} p={p} exceeds the p <= 3k capacity bound"


# Recorded search effort for kmax=2 on examples/nucleation/cdse_cdcl2.yaml.
# Guards are one-sided: an optimisation that lowers a number passes, one that
# silently loses pruning fails.  Re-record deliberately, never to make a test
# pass.
K2_SEARCH_EFFORT_CEILING = {
    "theoretical_assignments": 16783,
    "orbit_representatives": 297,
    # Rose from 815 / 291 when base equivalence started accounting for bridge
    # geometry: graph-isomorphic bases with different CIF-site bridge options no
    # longer collapse, so more of them reach the bridge search.  That is the
    # price of not losing structures, and it is deliberately recorded here
    # rather than treated as a regression.
    #
    # Rose again 1447 -> 2390 when min_bridged_host_cn: 3 became the default.
    # The rule is non-monotone in the arc set, so maximum-cardinality
    # enumeration alone no longer suffices and the sub-maximum fallback runs on
    # far more bases.  Also a correctness cost, not a pruning regression.
    "bridge_search_states": 2390,
    "bridge_raw_extensions": 666,
}


def _search_effort_totals(result) -> dict[str, int]:
    totals: dict[str, int] = {}
    for audit in result.sweep_audit:
        if audit.operation != "dag_bin":
            continue
        for key in (*K2_SEARCH_EFFORT_CEILING, "bridge_symmetry_pruned"):
            default = audit.raw_count if key == "theoretical_assignments" else 0
            totals[key] = totals.get(key, 0) + int(
                audit.stage_counts.get(key, default)
            )
    return totals


def test_k2_search_effort_does_not_regress(cdse_k2_result) -> None:
    """Catch a silent loss of symmetry or capacity pruning.

    ``bridge_search_states`` counts nodes actually visited by the bridge
    enumerator, so it rises the moment a pruning rule stops firing even though
    every structural assertion still passes.
    """

    totals = _search_effort_totals(cdse_k2_result)
    for key, ceiling in K2_SEARCH_EFFORT_CEILING.items():
        assert totals[key] <= ceiling, (
            f"{key} rose to {totals[key]} (ceiling {ceiling}); "
            "pruning regressed or the reduction was disabled"
        )
    # The reductions must still be doing real work, not trivially passing.
    assert totals["orbit_representatives"] < totals["theoretical_assignments"]
    assert totals["bridge_symmetry_pruned"] > 0


def _unbridged_bases(result, spec):
    """Rebuild every base (a record carrying no bridge) as a ``_State``."""

    for registry in (result.registry, result.discarded_registry):
        for bins in registry.values():
            for records in bins.values():
                for record in records:
                    if record.metadata.get("bridge_count", 0):
                        continue
                    yield _State(tuple(record.atoms), record.graph.copy())


def test_optimistic_bound_dominates_every_reachable_bridge_score(
    cdse_k2_result,
) -> None:
    """The pruning bound must dominate every achievable score, not just the
    ones the enumerator happens to visit.

    ``_optimistic_bridge_score`` gates whole bases out of the bridge search at
    k>2, so a bound that is too low silently deletes structures.  This checks it
    against the brute-force score of every feasible arc set of every cardinality
    on real bases -- componentwise, which is the property the proof relies on,
    and lexicographically, which is how the pruning test actually compares.
    """

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    model = _build_lattice_model(spec)
    checked = 0
    for base in _unbridged_bases(cdse_k2_result, spec):
        reference = exhaustive_bridge_sets(
            base, model, spec, nucleation_module, max_candidates=11
        )
        if reference is None:
            continue
        _candidates, scored = reference
        bound = _optimistic_bridge_score(base, spec)
        for _size, score, _subset in scored:
            assert len(score) == len(bound)
            assert all(
                actual <= limit for actual, limit in zip(score, bound)
            ), f"bound violated componentwise: {score} vs {bound}"
            assert score <= bound
            checked += 1
    assert checked > 200, f"only {checked} scores compared against the bound"


def test_reachable_score_max_dominates_every_achievable_score(
    cdse_k2_result,
) -> None:
    """The joint decision bound must never claim a score is unreachable when a
    real bridge arrangement achieves it.

    ``_reachable_bridge_score_max`` gates whole bases out of the k>2 bridge
    search; if it under-estimates, structures are silently deleted.  Check it
    lexicographically against every feasible arc set of every cardinality, and
    require it to be at least as sharp as the componentwise bound on real bases
    (that is what justifies running it at all).
    """

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    model = _build_lattice_model(spec)
    cache = nucleation_module._EnumerationCache()
    compared = 0
    sharper = 0
    for base in _unbridged_bases(cdse_k2_result, spec):
        reference = exhaustive_bridge_sets(
            base, model, spec, nucleation_module, max_candidates=11
        )
        if reference is None:
            continue
        _candidates, scored = reference
        reachable = nucleation_module._reachable_bridge_score_max(
            base, model, spec, cache
        )
        componentwise = _optimistic_bridge_score(base, spec)
        if reachable < componentwise:
            sharper += 1
        for _size, score, _subset in scored:
            assert score <= reachable, (
                "decision bound claims unreachable a score a real arc set "
                f"achieves: {score} > {reachable}"
            )
            compared += 1
    assert compared > 200, f"only {compared} scores compared"
    assert sharper > 0, "decision bound is never sharper than the tuple bound"


def test_bridge_score_context_matches_graph_scoring(cdse_k2_result) -> None:
    """Pin the graph-free bridge score to the graph-based reference.

    ``_BridgeScoreContext`` derives the selection score from degree deltas so
    the search never has to copy a graph.  It is only exercised on the k>2
    prune-dominated path, which no k<=2 fixture reaches, so without this test it
    would ship unverified.  Every feasible arc set of every cardinality is
    checked, not just the maximum ones the search visits.
    """

    from _nucleation_reference import bridge_candidates, build_bridged_graph

    base_spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    model = _build_lattice_model(base_spec)
    compared = 0
    # Both scopes: under "skeleton" the graph path drops ligand bonds from its
    # count while the graph-free path drops the subset contribution, and those
    # two must land on the same number.
    for scope in ("all", "skeleton"):
        spec = replace(base_spec, bond_count_scope=scope)
        for base in _unbridged_bases(cdse_k2_result, spec):
            reference = exhaustive_bridge_sets(
                base, model, spec, nucleation_module, max_candidates=10
            )
            if reference is None:
                continue
            candidates, scored = reference
            terminal_by_primary, _ = bridge_candidates(
                base, model, spec, nucleation_module
            )
            context = nucleation_module._BridgeScoreContext(
                base,
                spec,
                [candidate.primary for candidate in candidates],
                [candidate.host for candidate in candidates],
                [
                    candidate.mode == "shared_vacant_cif_site"
                    for candidate in candidates
                ],
                terminal_by_primary,
            )
            positions = {
                id(candidate): index for index, candidate in enumerate(candidates)
            }
            for _size, expected, subset in scored:
                indices = [positions[id(candidate)] for candidate in subset]
                assert context.score(indices) == expected, (
                    "graph-free bridge score diverged from "
                    f"_graph_coordination_score at bond_count_scope={scope!r}"
                )
                graph = build_bridged_graph(base, terminal_by_primary, subset)
                assert context.degrees(indices) == [
                    graph.degree[atom.atom_id] for atom in base.atoms
                ]
                compared += 1
    assert compared > 400, f"only {compared} arc sets compared"


def test_maximum_cardinality_restriction_is_discharged_not_assumed(
    cdse_k2_result, cdse_k2_unrestricted_result,
) -> None:
    """Every bin must either prove maximum-cardinality optimal or check it.

    The bridge search enumerates maximum-cardinality arc sets.  That is sound
    only because for feasible ``S`` subset ``S'`` the score strictly increases,
    so the optimum sits at an inclusion-maximal set, and because a maximum set
    leaving no minimum-CN violation is optimal outright (components 1-3 are then
    at their absolute best and it wins the bond count).  Where that certificate
    fails, a fallback enumerates the smaller sets and keeps any that tie or win.

    ``bridge_sub_maximum_undischarged`` counts bases where neither route
    applied; it must be zero, otherwise the bin is carrying an unproven
    restriction and the run should say so rather than imply completeness.
    """

    def tally(result):
        totals = dict.fromkeys(
            (
                "bridge_exactness_certified",
                "bridge_sub_maximum_fallbacks",
                "bridge_sub_maximum_contenders",
                "bridge_sub_maximum_undischarged",
            ),
            0,
        )
        for audit in result.sweep_audit:
            if audit.operation != "dag_bin":
                continue
            for key in totals:
                totals[key] += audit.stage_counts.get(key, 0)
        return totals

    for label, result in (
        ("default", cdse_k2_result),
        ("unrestricted", cdse_k2_unrestricted_result),
    ):
        totals = tally(result)
        assert totals["bridge_sub_maximum_undischarged"] == 0, (
            f"{label}: {totals['bridge_sub_maximum_undischarged']} bases left "
            "the maximum-cardinality restriction unproven; completeness would "
            "be overstated"
        )
        assert (
            totals["bridge_exactness_certified"] > 0
            and totals["bridge_sub_maximum_fallbacks"] > 0
        ), f"{label}: expected both discharge routes to be exercised ({totals})"

    # With bridging unrestricted the score is monotone in the arc set, so no
    # smaller set can ever tie or beat a maximum one.  If this fires, the
    # restriction was genuinely lossy and the extra structures are the finding
    # -- update the baselines deliberately, do not relax this.
    assert tally(cdse_k2_unrestricted_result)["bridge_sub_maximum_contenders"] == 0

    # ``min_bridged_host_cn`` breaks that monotonicity on purpose: adding a
    # bridge raises its acceptor's CN, so a larger arc set can satisfy the rule
    # where a smaller one cannot and vice versa.  Sub-maximum contenders are
    # therefore expected under the shipped default, and finding them is the
    # fallback working rather than a completeness hole -- which is exactly why
    # ``undischarged`` above must stay zero in both regimes.
    assert tally(cdse_k2_result)["bridge_sub_maximum_contenders"] > 0


def test_maximum_cardinality_bridge_sets_attain_the_global_optimum(
    cdse_k2_result,
) -> None:
    """Pin down the one assumption in the present bridge search.

    ``enumerate_maximum`` only yields bridge sets of maximum cardinality, on the
    reasoning that forming more bonds always scores better.  That is not
    provable in general: a single well-placed bridge can lift a CN-1 Cd to CN 2
    and clear a minimum-CN violation (score component 1) that no larger set
    reaches, and component 1 outranks the bond count at component 4.

    This test brute-forces every feasible bridge set of every cardinality on
    real bases and asserts the restriction is nonetheless lossless here, so the
    day it stops being true we find out from a failing test rather than from a
    missing isomer.

    Bases with more opportunities than ``max_candidates`` are skipped because
    the reference is deliberately brute force -- the k=2 bases run up to 28
    candidates.  Set ``QD_NUCLEATION_DEEP_REFERENCE`` to widen the sweep.
    """

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    model = _build_lattice_model(spec)
    max_candidates = int(os.environ.get("QD_NUCLEATION_DEEP_REFERENCE", "13"))
    checked = 0
    for base in _unbridged_bases(cdse_k2_result, spec):
        reference = exhaustive_bridge_sets(
            base, model, spec, nucleation_module, max_candidates=max_candidates
        )
        if reference is None:
            continue
        _candidates, scored = reference
        if not scored:
            continue
        checked += 1
        largest = max(size for size, _score, _subset in scored)
        best_overall = max(score for _size, score, _subset in scored)
        best_maximum = max(
            score for size, score, _subset in scored if size == largest
        )
        assert best_maximum == best_overall, (
            "a sub-maximum bridge set outscores every maximum one; the "
            "cardinality restriction in enumerate_maximum is now lossy"
        )
    assert checked > 20, f"reference covered only {checked} bases"


def test_bridge_symmetry_matches_exhaustive_graph_classes(
    cdse_k1_result,
    monkeypatch,
) -> None:
    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    model = _build_lattice_model(spec)
    records = [
        *cdse_k1_result.registry[1][3],
        *cdse_k1_result.discarded_registry[1][3],
    ]
    symmetric_variants = None
    symmetric_base = None
    symmetric_stats = None
    for record in records:
        if record.metadata.get("bridge_count", 0) != 0:
            continue
        base = _State(tuple(record.atoms), record.graph.copy())
        stats: dict[str, int] = {}
        variants = _latent_bridge_variants(
            base,
            model,
            spec,
            prune_dominated=False,
            stats_out=stats,
        )
        if stats.get("bridge_symmetry_used", 0):
            symmetric_base = base
            symmetric_variants = variants
            symmetric_stats = stats
            break
    assert symmetric_base is not None
    assert symmetric_variants is not None
    assert symmetric_stats is not None
    assert symmetric_stats["bridge_symmetry_pruned"] > 0

    def identity_only(_state, candidates, _terminal, _cache):
        return (tuple(range(len(candidates))),), 0

    monkeypatch.setattr(
        nucleation_module,
        "_bridge_candidate_permutations",
        identity_only,
    )
    exhaustive = _latent_bridge_variants(
        symmetric_base, model, spec, prune_dominated=False
    )
    assert len(symmetric_variants) == len(exhaustive)
    assert {
        _graph_coordination_score(variant.atoms, variant.graph, spec)
        for variant in symmetric_variants
    } == {
        _graph_coordination_score(variant.atoms, variant.graph, spec)
        for variant in exhaustive
    }
    assert all(
        any(
            GraphMatcher(
                candidate.graph,
                reference.graph,
                node_match=_node_match,
                edge_match=_edge_match,
            ).is_isomorphic()
            for reference in exhaustive
        )
        for candidate in symmetric_variants
    )


def test_skeleton_dag_merges_routes_before_ligand_enumeration(
    cdse_k2_result,
) -> None:
    p2 = next(
        audit for audit in cdse_k2_result.sweep_audit
        if audit.operation == "dag_bin"
        and audit.k == 2
        and audit.p_from == 2
    )
    assert p2.stage_counts["parent_routes_merged"] > 0
    assert (
        p2.stage_counts["base_embeddings_after_cross_skeleton_symmetry"]
        <= p2.stage_counts["base_embeddings_before_cross_skeleton_symmetry"]
    )
    validations = [
        audit for audit in cdse_k2_result.sweep_audit
        if audit.k == 2 and audit.operation == "strip_validation"
    ]
    assert validations
    assert all(audit.invalid_reasons == {} for audit in validations)
    assert all(
        audit.stage_counts["ligand_enumerations"] == 0
        for audit in validations
    )


def test_streaming_candidate_dedup_merges_routes_deterministically() -> None:
    graph = nx.Graph()
    graph.add_node(0, element="Cd", role="core_cation")
    graph.add_node(1, element="Se", role="core_anion")
    graph.add_edge(0, 1, kind="chemical", bond_order=1)
    later = _State(
        (
            AtomRecord(0, "Cd", (1.0, 0.0, 0.0), "core_cation"),
            AtomRecord(1, "Se", (2.0, 0.0, 0.0), "core_anion"),
        ),
        graph.copy(),
    )
    earlier = _State(
        (
            AtomRecord(0, "Cd", (0.0, 0.0, 0.0), "core_cation"),
            AtomRecord(1, "Se", (1.0, 0.0, 0.0), "core_anion"),
        ),
        graph.copy(),
    )
    accumulator = _CandidateAccumulator()
    assert accumulator.add(later, ("route_b",))
    assert not accumulator.add(earlier, ("route_a",))
    assert accumulator.isomorphism_checks == 1
    assert accumulator.result() == [
        (earlier, ("route_a", "route_b"))
    ]


def test_schema_records_native_and_surface_geometry(cdse_k1_result) -> None:
    data = nucleation_result_to_dict(cdse_k1_result)
    assert data["schema_version"] == 13
    assert (
        data["construction_geometry"]["mode"]
        == "construction_native_plus_slot_filtered_surface"
    )
    assert (
        data["construction_geometry"]["growth_coordinates"]
        == "construction_native"
    )
    assert data["graph_rules"] == {
        "min_cn": {"Cd": 2, "Cl": 1, "Se": 2},
        "max_cn": {"Cd": 4, "Cl": 2, "Se": 4},
        "allowed_bonds": [["Cd", "Cl"], ["Cd", "Se"]],
        "bridging": {
            "Cl": {
                "host": "Cd",
                "shared_neighbor": "Se",
                "surface_angle_deg": 90.0,
                "min_bridged_host_cn": 3,
            }
        },
    }
    assert data["geometry_rules"]["by_cn"]["Cd"]["cn2"] == "linear"
    assert data["registry"]["1"]["1"][0]["selection"]["status"] == "retained"
    assert data["registry"]["1"]["1"][0]["surface_coordinates"] is not None
    assert (
        data["discarded_registry"]["1"]["1"][0]["surface_coordinates"]
        is None
    )


def test_bundle_writes_shell_safe_xyz_and_detailed_log(
    tmp_path: Path,
    cdse_k1_result,
) -> None:
    bundle = write_nucleation_bundle(cdse_k1_result, tmp_path / "map")
    retained_count = sum(
        len(records)
        for bins in cdse_k1_result.registry.values()
        for records in bins.values()
    )
    discarded_count = sum(
        len(records)
        for bins in cdse_k1_result.discarded_registry.values()
        for records in bins.values()
    )
    expected = 2 * retained_count + discarded_count
    xyz_files = sorted((bundle / "structures").rglob("*.xyz"))
    assert len(xyz_files) == expected
    assert (
        bundle
        / "structures/k001/p001/retained/"
        "k001_p001_retained_iso0001_construction_native.xyz"
    ).is_file()
    assert (
        bundle
        / "structures/k001/p001/retained/"
        "k001_p001_retained_iso0001_surface.xyz"
    ).is_file()
    assert (
        bundle
        / "structures/k001/p001/discarded/"
        "k001_p001_discarded_iso0001_construction_native.xyz"
    ).is_file()
    assert not list(
        (bundle / "structures/k001/p001/discarded").glob("*_surface.xyz")
    )

    safe = re.compile(r"^[A-Za-z0-9_]+$")
    for path in xyz_files:
        relative = path.relative_to(bundle / "structures")
        assert all(safe.fullmatch(part) for part in relative.parts[:-1])
        assert safe.fullmatch(path.stem)
        lines = path.read_text().splitlines()
        assert re.fullmatch(
            r"(?:[A-Z][a-z]?\d+)+_(?:construction_native_graph_ranked_"
            r"bridges_\d+|surface_projected_valid_(?:true|false))",
            lines[1],
        )
        assert len(lines) == int(lines[0]) + 2

    log = (bundle / "nucleation.log").read_text()
    for heading in (
        "RUN CONFIGURATION",
        "SWEEP SUMMARY",
        "BIN SUMMARY",
        "ISOMERS: k=1, p=1",
        "REJECTION SUMMARY",
        "FINAL SUMMARY",
    ):
        assert heading in log
    assert "Exact core-CIF virtual sites (construction-native coordinates)" in log
    assert "Retained-only final-CN projection" in log
    assert "Cl: host=Cd, shared=Se, angle=90.0 deg" in log
    assert "SURFACE GEOMETRY: k=1, p=1" in log
    # The bridge section renders only for bins that have a bridge.  Under the
    # shipped ``min_bridged_host_cn: 3`` that is no longer k=1 p=1 (its only
    # candidate bridge would leave a Cd at CN 2), but k=1 p=2 keeps all three.
    assert "BRIDGE HOST CN: k=1, p=2" in log
    assert "Cd=2, Cl=1, Se=2" in log
    assert "Cd=4, Cl=2, Se=4" in log
    assert "Cd-Cl, Cd-Se" in log
    assert "Cd[2,2] Cl[1,1] Se[2]" in log
    assert "Cd[3,1] Cl[1,1] Se[2]" in log
    assert "min_cn_compliant" in log
    assert "min_cn_violation" in log
    assert "| 1 | 3 |         7 |        1 |         6 |" in log
    assert "| yes      | closed" in log
    assert "score=" not in log
    assert all(
        character == "\n" or 32 <= ord(character) <= 126
        for character in log
    )
    all_records = [
        record
        for registry in (
            cdse_k1_result.registry,
            cdse_k1_result.discarded_registry,
        )
        for bins in registry.values()
        for records in bins.values()
        for record in records
    ]
    for record in all_records:
        id_rows = [
            line for line in log.splitlines()
            if record.structure_id in line
            and line.startswith("|")
        ]
        expected_rows = 1
        if record.selection_status == "retained":
            expected_rows += 1  # SURFACE GEOMETRY
            expected_rows += int(record.metadata.get("bridge_count", 0))
            # PAULING VALENCE lists one row per cation.
            expected_rows += len(
                record.metadata["surface_geometry"]["pauling_valence"]["cations"]
            )
        assert len(id_rows) == expected_rows

    json_data = json.loads((bundle / "registry.json").read_text())
    assert json_data["schema_version"] == 13


def test_registry_writer_and_generation_are_deterministic(tmp_path: Path) -> None:
    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    first = generate_nucleation_map(spec)
    second = generate_nucleation_map(spec)
    assert registry_to_dict(first) == registry_to_dict(second)
    output = tmp_path / "registry.json"
    write_nucleation_json(first, output)
    assert json.loads(output.read_text()) == registry_to_dict(first)


def test_progress_callback_reports_long_running_phases() -> None:
    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    messages: list[str] = []
    result = generate_nucleation_result(
        spec,
        progress=messages.append,
        progress_interval=0.0,
    )
    text = "\n".join(messages)
    assert "[nucleation] starting map: kmax=1" in text
    assert "k=1: starting merged skeleton DAG" in text
    assert "sites=6, ligands=2, assignments=15" in text
    assert "orbit processed=1/2" in text
    assert "theoretical=15" in text
    assert "cross-skeleton filtering complete" in text
    assert "bridge search complete" in text
    assert "bin symmetry filtering complete" in text
    assert "ranked surface screening complete" in text
    assert "symmetry_duplicates=" in text
    assert "DAG complete" in text
    # 11 before min_bridged_host_cn: 3 became the default: the k=1 p=1 bridged
    # variant is no longer generated, so it is not discarded either.
    assert "map complete: retained=4, discarded=10" in text
    p1_audit = next(
        audit
        for audit in result.sweep_audit
        if audit.operation == "dag_bin" and audit.p_from == 1
    )
    assert p1_audit.raw_count == 15
    assert p1_audit.stage_counts["orbit_representatives"] == 2
    assert p1_audit.stage_counts["embedded"] == 2
    # 2 before min_bridged_host_cn: 3 became the default.  The only bridge
    # available at k=1 p=1 would leave a Cd at CN 2, so no bridged variant is
    # generated at all and the bin resolves from the bare bases.
    assert p1_audit.stage_counts["bridge_variants"] == 0
    assert (
        p1_audit.stage_counts["symmetry_pruned_before_embedding"] == 13
    )


def test_bundle_never_writes_discarded_structures_above_k2(
    tmp_path: Path,
    cdse_k1_result,
) -> None:
    result = copy.deepcopy(cdse_k1_result)
    record = copy.deepcopy(result.discarded_registry[1][1][0])
    record.k = 3
    record.p = 1
    record.structure_id = "k003_p001_discarded_iso0001"
    result.discarded_registry[3] = {1: [record]}
    result.discarded_counts[3] = {1: 1}
    bundle = write_nucleation_bundle(result, tmp_path / "discard_limit")
    assert not (bundle / "structures/k003").exists()


def _write_recipe(path: Path, graph_rules: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "cif: CdSe_zb.cif",
                "charges:",
                "  Cd: +2",
                "  Se: -2",
                "  Cl: -1",
                "nucleation:",
                "  kmax: 1",
                "  core_monomer: {cation: Cd, anion: Se}",
                "  precursor: {center: Cd, ligand: Cl, ligand_count: 2}",
                "  site_tolerance: 0.20",
                *graph_rules,
                "",
            ]
        )
    )


_VALID_GRAPH_RULES = [
    "  graph_rules:",
    "    min_cn: {Cd: 2, Se: 2, Cl: 1}",
    "    max_cn: {Cd: 4, Se: 4, Cl: 2}",
    "    allowed_bonds: [[Cd, Se], [Cd, Cl]]",
]


def test_mode_and_exact_threshold_are_validated(tmp_path: Path) -> None:
    """A misspelt mode must fail loudly, not silently fall back to exact.

    Silently treating ``mode: greedy`` as exact would hand back a complete map
    to someone who asked for an approximate one, or vice versa -- the one class
    of error the completeness block exists to prevent.
    """

    shutil.copy2(ROOT / "examples/cifs/CdSe_zb.cif", tmp_path / "CdSe_zb.cif")

    bad_mode = tmp_path / "bad_mode.yaml"
    _write_recipe(bad_mode, [*_VALID_GRAPH_RULES, "  mode: greedy"])
    with pytest.raises(ValueError, match="nucleation.mode must be one of"):
        load_nucleation_spec(bad_mode)

    bad_threshold = tmp_path / "bad_threshold.yaml"
    _write_recipe(
        bad_threshold, [*_VALID_GRAPH_RULES, "  exact_through_k: 0"]
    )
    with pytest.raises(ValueError, match="exact_through_k must be at least 1"):
        load_nucleation_spec(bad_threshold)

    for mode in ("exact", "guided"):
        good = tmp_path / f"good_{mode}.yaml"
        _write_recipe(good, [*_VALID_GRAPH_RULES, f"  mode: {mode}"])
        assert load_nucleation_spec(good).mode == mode

    bad_scope = tmp_path / "bad_scope.yaml"
    _write_recipe(bad_scope, [*_VALID_GRAPH_RULES, "  bond_count_scope: core"])
    with pytest.raises(ValueError, match="bond_count_scope must be one of"):
        load_nucleation_spec(bad_scope)
    for scope in ("all", "skeleton"):
        good = tmp_path / f"scope_{scope}.yaml"
        _write_recipe(good, [*_VALID_GRAPH_RULES, f"  bond_count_scope: {scope}"])
        assert load_nucleation_spec(good).bond_count_scope == scope

    bad_growth = tmp_path / "bad_growth.yaml"
    _write_recipe(
        bad_growth, [*_VALID_GRAPH_RULES, "  core_growth_policy: greedy"]
    )
    with pytest.raises(
        ValueError, match="core_growth_policy must be one of"
    ):
        load_nucleation_spec(bad_growth)
    for policy in ("all", "max_bonds", "compact_ring"):
        good = tmp_path / f"growth_{policy}.yaml"
        _write_recipe(
            good,
            [
                *_VALID_GRAPH_RULES,
                f"  core_growth_policy: {policy}",
                "  compact_from_k: 3",
            ],
        )
        loaded = load_nucleation_spec(good)
        assert loaded.core_growth_policy == policy
        assert loaded.compact_from_k == 3


def test_core_growth_filter_keeps_only_max_bond_children() -> None:
    """max_bonds must drop a single-bond attachment when a multi-bond one exists."""

    parent_atoms = (
        AtomRecord(0, "Cd", (0.0, 0.0, 0.0), "core_cation"),
        AtomRecord(1, "Se", (1.0, 0.0, 0.0), "core_anion"),
    )
    parent = _State(
        parent_atoms,
        nx.Graph([(0, 1)]),
    )
    # Child A: +1 edge relative to parent (dangling).
    dangling = _State(
        (
            *parent_atoms,
            AtomRecord(2, "Cd", (2.0, 0.0, 0.0), "core_cation"),
            AtomRecord(3, "Se", (3.0, 0.0, 0.0), "core_anion"),
        ),
        nx.Graph([(0, 1), (1, 2), (2, 3)]),
    )
    # Child B: +3 edges (more compact).
    compact = _State(
        (
            *parent_atoms,
            AtomRecord(2, "Cd", (0.0, 1.0, 0.0), "core_cation"),
            AtomRecord(3, "Se", (1.0, 1.0, 0.0), "core_anion"),
        ),
        nx.Graph([(0, 1), (0, 3), (1, 2), (2, 3)]),
    )
    kept = nucleation_module._filter_core_children_by_policy(
        parent,
        [(dangling, ("a",)), (compact, ("b",))],
        "max_bonds",
    )
    assert len(kept) == 1
    assert kept[0][1] == ("b",)


def test_six_cycle_counter_counts_c6_once() -> None:
    graph = nx.cycle_graph(6)
    assert nucleation_module._count_simple_cycles_of_length(graph, 6) == 1


def test_fused_chair_metrics_two_edge_sharing_hexagons() -> None:
    """Two 6-cycles sharing an edge → fused_pair_count == 1."""

    # Build two hexagons sharing edge (0,1):
    # cycle A: 0-1-2-3-4-5-0
    # cycle B: 0-1-6-7-8-9-0
    g = nx.Graph()
    g.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (1, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 0),
        ]
    )
    six, fused = nucleation_module._fused_chair_metrics(g)
    assert six == 2
    assert fused == 1


def test_p_skeleton_beam_keeps_top_b() -> None:
    """Beam must keep highest-ranked skeletons only."""

    from builder.nucleation import (
        _State,
        _apply_p_skeleton_beam,
        _ProgressReporter,
    )
    from builder.nc_types import (
        CoreMonomerSpec,
        NucleationGeometryRules,
        NucleationGraphRules,
        NucleationSpec,
        PrecursorUnitSpec,
    )

    def _state_with_edges(n_edges: int) -> _State:
        g = nx.path_graph(n_edges + 1)
        atoms = tuple(
            AtomRecord(i, "Cd" if i % 2 == 0 else "Se", (float(i), 0.0, 0.0), "core")
            for i in range(n_edges + 1)
        )
        return _State(atoms, g)

    skeletons = [
        (_state_with_edges(1), ("a",)),
        (_state_with_edges(5), ("b",)),
        (_state_with_edges(3), ("c",)),
    ]
    spec = NucleationSpec(
        cif="examples/cifs/CdSe_zb.cif",
        charges={"Cd": 2, "Se": -2, "Cl": -1},
        core=CoreMonomerSpec(cation="Cd", anion="Se"),
        precursor=PrecursorUnitSpec(center="Cd", ligand="Cl", ligand_count=2),
        kmax=5,
        graph_rules=NucleationGraphRules(
            min_cn={"Cd": 2, "Se": 2, "Cl": 1},
            max_cn={"Cd": 4, "Se": 4, "Cl": 2},
            allowed_bonds=(("Cd", "Cl"), ("Cd", "Se")),
        ),
        geometry_rules=NucleationGeometryRules(by_cn={}, all_cn={}),
        p_skeleton_beam=2,
        p_beam_from_k=1,
        p_beam_rank="bonds",
    )
    messages: list[str] = []
    kept, dropped = _apply_p_skeleton_beam(
        skeletons,
        k=4,
        p=2,
        spec=spec,
        progress=_ProgressReporter(callback=messages.append, interval_seconds=0),
    )
    assert dropped == 1
    assert len(kept) == 2
    assert kept[0][0].graph.number_of_edges() == 5
    assert kept[1][0].graph.number_of_edges() == 3


def test_compact_growth_discloses_when_it_prunes() -> None:
    """Path-dependent core growth must appear in completeness when it binds."""

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    # Force compact growth already on the k=1 -> k=2 step so a short run
    # exercises the completeness path without a k=3 exact map.
    result = generate_nucleation_result(
        replace(
            spec,
            kmax=2,
            mode="guided",
            core_growth_policy="max_bonds",
            compact_from_k=2,
        )
    )
    report = result.completeness
    assert report["core_growth_policy"] == "max_bonds"
    assert report["compact_from_k"] == 2
    stages = {item["stage"] for item in report["approximations"]}
    # Either the policy pruned (approximation) or it was configured but idle
    # (guarantee).  Both are honest; the silent case is the bug.
    if "core_monomer_growth" not in stages:
        assert any(
            "core_growth_policy=max_bonds" in text
            for text in report["guarantees"]
        )
    growth_audits = [
        audit
        for audit in result.sweep_audit
        if audit.operation == "core_skeleton_growth"
    ]
    assert growth_audits
    assert any(
        "core_growth_after_policy" in audit.stage_counts
        for audit in growth_audits
    )


def test_ring_metadata_and_prefer_policy_keep_min_bridged_host_cn() -> None:
    """Ring ranking is post-placement; bridge host floor stays in the rules."""

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    assert spec.graph_rules.bridge_rules[0].min_bridged_host_cn == 3
    result = generate_nucleation_result(
        replace(
            spec,
            kmax=1,
            mode="guided",
            passivation_ring_policy="prefer_cl_rings",
            ring_lengths=(4, 6),
        )
    )
    assert result.completeness["passivation_ring_policy"] == "prefer_cl_rings"
    stages = {item["stage"] for item in result.completeness["approximations"]}
    assert "passivation_ring_selection" in stages
    sample = next(
        record
        for bins in result.registry.values()
        for records in bins.values()
        for record in records
    )
    rings = sample.metadata["rings"]
    assert "cl_rings_by_length" in rings
    assert "4" in rings["cl_rings_by_length"]
    # Bridging rule is unchanged in the serialized graph_rules block.
    assert result.graph_rules["bridging"]["Cl"]["min_bridged_host_cn"] == 3


def test_checkpoint_restart_extends_kmax(tmp_path: Path) -> None:
    """A finished k=1 checkpoint can be resumed to k=2 without redoing k=1."""

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    out = tmp_path / "bundle"
    first = generate_nucleation_result(
        replace(spec, kmax=1, mode="guided"),
        checkpoint_dir=out,
        progress_interval=0,
    )
    write_nucleation_bundle(first, out)
    assert (out / "checkpoint" / "k001" / "DONE").is_file()
    k1_ids = {
        record.structure_id
        for records in first.registry[1].values()
        for record in records
    }

    second = generate_nucleation_result(
        replace(spec, kmax=2, mode="guided"),
        checkpoint_dir=out,
        restart=True,
        progress_interval=0,
    )
    assert set(second.registry) >= {1, 2}
    resumed_k1 = {
        record.structure_id
        for records in second.registry[1].values()
        for record in records
    }
    assert resumed_k1 == k1_ids
    assert second.registry[2], "k=2 should be newly generated on resume"


def test_restart_recomputes_incompatible_partial_inherited_row(
    tmp_path: Path,
) -> None:
    """Keep DONE rows but reject a partial row with composition-label drift."""

    from builder.nucleation import _seed_state, write_nucleation_checkpoint

    base = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    spec = replace(
        base,
        kmax=2,
        mode="guided",
        exact_through_k=1,
        core_growth_occupation="decorated",
        continuous_decoration=False,
        monomer_p_values=(0, 1),
        p_ladder_mode="product_window",
        k_growth_max_shed=1,
        k_growth_max_add=1,
    )
    out = tmp_path / "bundle"
    generate_nucleation_result(
        replace(spec, kmax=1),
        checkpoint_dir=out,
        progress_interval=0,
    )
    model = _build_lattice_model(spec)
    # Simulate the historical bug: an inherited destination p=1 entry that is
    # actually only the k=1/p=0 seed (no added core pair or precursor center).
    write_nucleation_checkpoint(
        root=out,
        spec=spec,
        k=2,
        retained={},
        discarded={},
        skeletons={},
        discarded_counts={},
        mark_done=False,
        last_completed_p=-1,
        p_cap=6,
        max_inherited=1,
        inherited={1: [(_seed_state(model), ("stale_bad_channel",))]},
    )
    messages: list[str] = []
    result = generate_nucleation_result(
        spec,
        checkpoint_dir=out,
        restart=True,
        progress=messages.append,
        progress_interval=0,
    )
    assert result.registry[2]
    assert any(
        "ignoring incompatible partial k=2 checkpoint" in message
        for message in messages
    )


def test_checkpoint_fingerprint_rejects_policy_change(tmp_path: Path) -> None:
    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    out = tmp_path / "bundle"
    generate_nucleation_result(
        replace(spec, kmax=1, mode="guided"),
        checkpoint_dir=out,
        progress_interval=0,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        generate_nucleation_result(
            replace(spec, kmax=2, mode="guided", core_growth_policy="max_bonds"),
            checkpoint_dir=out,
            restart=True,
            progress_interval=0,
        )


def test_min_bridged_host_cn_is_parsed_and_validated(tmp_path: Path) -> None:
    """The bridging rule must round-trip and reject impossible thresholds.

    A threshold above the host's maximum CN can never be satisfied, so every
    bridge would be silently forbidden -- which would look like "this chemistry
    has no bridges" rather than like a misconfiguration.
    """

    shutil.copy2(ROOT / "examples/cifs/CdSe_zb.cif", tmp_path / "CdSe_zb.cif")
    bridging = [
        "    bridging:",
        "      Cl:",
        "        host: Cd",
        "        shared_neighbor: Se",
    ]

    good = tmp_path / "good_rule.yaml"
    _write_recipe(good, [*_VALID_GRAPH_RULES, *bridging,
                         "        min_bridged_host_cn: 3"])
    rule = load_nucleation_spec(good).graph_rules.bridge_rules[0]
    assert rule.min_bridged_host_cn == 3

    default = tmp_path / "default_rule.yaml"
    _write_recipe(default, [*_VALID_GRAPH_RULES, *bridging])
    assert (
        load_nucleation_spec(default).graph_rules.bridge_rules[0]
        .min_bridged_host_cn
        == 1
    )

    too_low = tmp_path / "too_low.yaml"
    _write_recipe(too_low, [*_VALID_GRAPH_RULES, *bridging,
                            "        min_bridged_host_cn: 0"])
    with pytest.raises(ValueError, match="min_bridged_host_cn.*at least 1"):
        load_nucleation_spec(too_low)

    unsatisfiable = tmp_path / "unsatisfiable.yaml"
    _write_recipe(unsatisfiable, [*_VALID_GRAPH_RULES, *bridging,
                                  "        min_bridged_host_cn: 5"])
    with pytest.raises(ValueError, match="exceeds.*maximum CN"):
        load_nucleation_spec(unsatisfiable)


def test_strict_graph_rule_validation(tmp_path: Path) -> None:
    shutil.copy2(ROOT / "examples/cifs/CdSe_zb.cif", tmp_path / "CdSe_zb.cif")
    missing = tmp_path / "missing.yaml"
    _write_recipe(missing, [])
    with pytest.raises(KeyError, match="requires graph_rules"):
        load_nucleation_spec(missing)

    duplicate = tmp_path / "duplicate.yaml"
    _write_recipe(
        duplicate,
        [
            "  graph_rules:",
            "    min_cn: {Cd: 2, Se: 2, Cl: 1}",
            "    max_cn: {Cd: 4, Se: 4, Cl: 2}",
            "    allowed_bonds:",
            "      - [Cd, Se]",
            "      - [Se, Cd]",
            "      - [Cd, Cl]",
        ],
    )
    with pytest.raises(ValueError, match="duplicate unordered allowed bond"):
        load_nucleation_spec(duplicate)

    incomplete = tmp_path / "incomplete.yaml"
    _write_recipe(
        incomplete,
        [
            "  graph_rules:",
            "    min_cn: {Cd: 2, Se: 2, Cl: 1}",
            "    max_cn: {Cd: 4, Se: 4}",
            "    allowed_bonds: [[Cd, Se], [Cd, Cl]]",
        ],
    )
    with pytest.raises(ValueError, match="missing.*max_cn.*Cl"):
        load_nucleation_spec(incomplete)

    missing_minimum = tmp_path / "missing_minimum.yaml"
    _write_recipe(
        missing_minimum,
        [
            "  graph_rules:",
            "    max_cn: {Cd: 4, Se: 4, Cl: 2}",
            "    allowed_bonds: [[Cd, Se], [Cd, Cl]]",
        ],
    )
    with pytest.raises(KeyError, match="requires min_cn"):
        load_nucleation_spec(missing_minimum)

    inverted = tmp_path / "inverted.yaml"
    _write_recipe(
        inverted,
        [
            "  graph_rules:",
            "    min_cn: {Cd: 5, Se: 2, Cl: 1}",
            "    max_cn: {Cd: 4, Se: 4, Cl: 2}",
            "    allowed_bonds: [[Cd, Se], [Cd, Cl]]",
        ],
    )
    with pytest.raises(ValueError, match="minimum CN exceeds maximum CN.*Cd"):
        load_nucleation_spec(inverted)

    invalid_geometry = tmp_path / "invalid_geometry.yaml"
    _write_recipe(
        invalid_geometry,
        [
            "  geometry_rules:",
            "    Cd: {cn2: bent}",
            "  graph_rules:",
            "    min_cn: {Cd: 2, Se: 2, Cl: 1}",
            "    max_cn: {Cd: 4, Se: 4, Cl: 2}",
            "    allowed_bonds: [[Cd, Se], [Cd, Cl]]",
        ],
    )
    with pytest.raises(ValueError, match="unsupported geometry template"):
        load_nucleation_spec(invalid_geometry)

    invalid_bridge = tmp_path / "invalid_bridge.yaml"
    _write_recipe(
        invalid_bridge,
        [
            "  graph_rules:",
            "    min_cn: {Cd: 2, Se: 2, Cl: 1}",
            "    max_cn: {Cd: 4, Se: 4, Cl: 2}",
            "    allowed_bonds: [[Cd, Se], [Cd, Cl]]",
            "    bridging:",
            "      Cl: {host: Cd, shared_neighbor: Se, surface_angle_deg: 180}",
        ],
    )
    with pytest.raises(ValueError, match="surface_angle_deg must be between"):
        load_nucleation_spec(invalid_bridge)


def test_nucleation_cli_resolves_external_cif_and_yaml(tmp_path: Path) -> None:
    shutil.copy2(ROOT / "examples/cifs/CdSe_zb.cif", tmp_path / "CdSe_zb.cif")
    recipe = tmp_path / "cdse_cdcl2.yaml"
    _write_recipe(
        recipe,
        [
            "  graph_rules:",
            "    min_cn: {Cd: 2, Se: 2, Cl: 1}",
            "    max_cn: {Cd: 4, Se: 4, Cl: 2}",
            "    allowed_bonds: [[Cd, Se], [Cd, Cl]]",
        ],
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "builder", recipe.name],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    output = tmp_path / "cdse_cdcl2_nucleation"
    assert (output / "registry.json").is_file()
    assert (output / "nucleation.log").is_file()
    assert list((output / "structures/k001/p001/retained").glob("*.xyz"))
    assert "[nucleation] starting map: kmax=1" in completed.stdout
    assert "sites=6, ligands=2, assignments=15" in completed.stdout
    assert "DAG complete" in completed.stdout
    assert "map complete: retained=4, discarded=7" in completed.stdout
    assert "physical k/p bins" in completed.stdout


def test_minimum_cdse_wulff_reference_is_unchanged(tmp_path: Path) -> None:
    requested = tmp_path / "minimum.xyz"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "builder",
            str(ROOT / "examples/cifs/CdSe_zb.cif"),
            str(ROOT / "examples/core-only/cdse_minimum_wulff.yaml"),
            "-o",
            str(requested),
            "--positive-q-mode",
            "add",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(requested.with_suffix(".json").read_text())
    assert manifest["counts"] == {"Cd": 16, "Se": 13, "Cl": 6}
    assert manifest["total_charge"] == 0
