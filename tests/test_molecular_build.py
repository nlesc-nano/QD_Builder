"""Smoke tests for lattice-free molecular enumeration + embedding."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import math

import numpy as np
import pytest

from builder.nucleation import load_nucleation_spec
from builder.nucleation.geometry_pack import load_geometry_pack
from builder.nucleation.molecular import (
    EnumerationLimitError,
    embed_molecular_state,
    enumerate_molecular_bin,
    generate_molecular_map,
    molecular_isomer_log_line,
    molecular_max_p_from_accepted_p0_slots,
    molecular_max_p_from_se_capacity,
    molecular_stoichiometry_label,
    resolve_molecular_max_p,
    write_molecular_map,
    _cl_attachments,
    _dihedral_deg_points,
    _enumerate_inorganic_edge_sets,
)
from builder.nucleation.molecular_rules import molecular_geometry_ok
from builder.nucleation.types import _State

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


def test_geometry_pack_bond_lookup(pack) -> None:
    r = pack.bond_length("CdCl_bridge", 3, 2)
    assert 2.4 < r < 2.7
    r2 = pack.bond_length("CdSe", 3, 3)
    assert 2.5 < r2 < 2.7
    assert pack.center_angle_deg("Cd", 2) == pytest.approx(175.603)
    assert pack.center_angle_deg(
        "Cd",
        3,
        neighbor_pair="Cl-Cl",
        signature="Cl2Se1",
        role_pair="Cl_b2s-Cl_t",
        role_signature="Cl_b2s+Cl_t+Se",
    ) == pytest.approx(122.366)
    assert pack.improper_angle_deg("Cd", 3, "Cl1Se2") == 0.0
    assert pack.improper_angle_deg("Se", 3, "Cd3") is None


def test_geometry_pack_proper_dihedral_and_soft_clearance(pack) -> None:
    assert pack.preferred_dihedral(("Se", "Cd", "Cl", "Cd")) == pytest.approx(
        (0.0, 30.0)
    )
    assert pack.dihedral_weight(("Cd", "Se", "Cd", "Cl")) == "clearance"
    assert pack.dihedral_excluded(
        ("Cd", "Se", "Cd", "Se"), endocyclic=True
    )
    assert not pack.dihedral_excluded(
        ("Cd", "Se", "Cd", "Se"), endocyclic=False
    )
    assert pack.soft_contact_penalty("Se-Cl", 3.20) == pytest.approx(0.30)
    assert pack.one_four_hard_min("Cl-Se") == pytest.approx(3.30)
    bridge_pack = load_geometry_pack(
        ROOT / "geometry_packs/cdse_cdcl2_bridge_first.yaml"
    )
    assert bridge_pack.preferred_dihedral(("Cd", "Cl", "Cd", "Se")) == pytest.approx(
        (0.0, 30.0)
    )


def test_enumerate_k1_p1_finds_isomers(mol_spec, pack) -> None:
    """Cd2SeCl2: connected Cd–Se, min CN 2, unique graphs."""

    bin_res = enumerate_molecular_bin(1, 1, mol_spec, pack=pack, embed=True)
    assert bin_res.raw_graphs > 0
    assert len(bin_res.isomers) >= 1
    for iso in bin_res.isomers:
        assert iso.coordinates is not None
        assert len(iso.coordinates) == len(iso.atoms)
        # composition
        from collections import Counter

        c = Counter(a.symbol for a in iso.atoms)
        assert c["Cd"] == 2 and c["Se"] == 1 and c["Cl"] == 2
        # coordinates finite and not all zero
        xyz = np.asarray(iso.coordinates)
        assert np.all(np.isfinite(xyz))
        assert float(np.max(np.linalg.norm(xyz, axis=1))) > 0.5


def test_bridge_candidates_use_preferred_cd_cl_cd_se_torsion(mol_spec, pack) -> None:
    result = enumerate_molecular_bin(1, 2, mol_spec, pack=pack, embed=True)
    measured = []
    for iso in result.isomers:
        assert iso.coordinates is not None
        xyz = np.asarray(iso.coordinates)
        for cl_atom in iso.atoms:
            cl = cl_atom.atom_id
            if cl_atom.symbol != "Cl" or iso.graph.degree[cl] != 2:
                continue
            hosts = list(iso.graph.neighbors(cl))
            for host, other in ((hosts[0], hosts[1]), (hosts[1], hosts[0])):
                for se in iso.graph.neighbors(host):
                    if iso.atoms[se].symbol == "Se":
                        measured.append(
                            abs(
                                _dihedral_deg_points(
                                    xyz[other], xyz[cl], xyz[host], xyz[se]
                                )
                            )
                        )
    assert measured
    assert min(measured) < 1.0


def test_molecular_progress_labels_stoichiometry_and_bond_roles(
    mol_spec, pack
) -> None:
    blocks, total = molecular_stoichiometry_label(mol_spec, 1, 2)
    assert blocks == "[CdSe]1(CdCl2)2"
    assert total == "Cd3Se1Cl4"
    assert molecular_max_p_from_se_capacity(mol_spec, 1) == 4
    assert molecular_max_p_from_se_capacity(mol_spec, 2) == 7
    assert molecular_max_p_from_se_capacity(mol_spec, 3) == 10
    result = enumerate_molecular_bin(1, 2, mol_spec, pack=pack, embed=True)
    mu3 = next(
        isomer
        for isomer in result.isomers
        if any(
            atom.symbol == "Cl" and isomer.graph.degree[atom.atom_id] >= 3
            for atom in isomer.atoms
        )
    )
    line = molecular_isomer_log_line(mu3, mol_spec)
    assert "bonds: skeleton=" in line
    assert "Cd-Cl terminal=" in line
    assert "multiple_mu3+=3" in line
    assert "Cl motifs:" in line


def test_k1_p0_empty_under_min_cn(mol_spec, pack) -> None:
    """Bare CdSe cannot satisfy min_cn 2 for both atoms."""

    bin_res = enumerate_molecular_bin(1, 0, mol_spec, pack=pack, embed=False)
    assert len(bin_res.isomers) == 0


def test_slot_based_pmax_from_accepted_p0(mol_spec) -> None:
    """Automatic pmax uses free Se on accepted p=0 cores (no k-growth)."""

    # k=1: no bare CdSe → global Se bound
    info1 = molecular_max_p_from_accepted_p0_slots(mol_spec, 1)
    assert info1.source == "global_fallback"
    assert info1.n_p0_accepted == 0
    assert info1.pmax == molecular_max_p_from_se_capacity(mol_spec, 1) == 4

    # k=2: only p=0 graph is Cd2Se2 four-ring (C4-banned) → global fallback
    info2 = molecular_max_p_from_accepted_p0_slots(mol_spec, 2)
    assert info2.n_p0_accepted == 0
    assert info2.source == "global_fallback"
    assert info2.pmax == molecular_max_p_from_se_capacity(mol_spec, 2) == 7

    # k=3: accepted Se[2,2,2] free=9 with max Se=5 → pmax=9 (global 10)
    info3 = molecular_max_p_from_accepted_p0_slots(mol_spec, 3)
    assert info3.source == "slots"
    assert info3.n_p0_accepted >= 1
    assert info3.max_free_slots == 9
    assert info3.global_bound == 10
    assert info3.pmax == 9

    # Explicit pmax overrides slot logic
    p_user, diag = resolve_molecular_max_p(mol_spec, 3, pmax=2)
    assert p_user == 2 and diag is None
    p_auto, diag2 = resolve_molecular_max_p(mol_spec, 3, pmax=None)
    assert p_auto == 9 and diag2 is not None and diag2.source == "slots"


def test_ring_first_required_composition() -> None:
    """Min pattern Cd[3,3,4] Se[3,3,3] possible iff k≥3 and (p≥1 or k≥4)."""

    from builder.nucleation.molecular import (
        ring_first_required,
        two_ring_possible,
        max_structure_level_possible,
        _enumerate_inorganic_edge_sets,
        count_cdse_six_rings,
    )
    from builder.nucleation import load_nucleation_spec
    from pathlib import Path
    from dataclasses import replace

    assert not ring_first_required(1, 5)
    assert not ring_first_required(2, 5)
    assert not ring_first_required(3, 0)
    assert ring_first_required(3, 1)
    assert ring_first_required(4, 0)
    assert ring_first_required(5, 1)
    assert ring_first_required(6, 0)

    assert not two_ring_possible(3, 5)
    assert two_ring_possible(4, 0)
    assert two_ring_possible(5, 1)

    spec = load_nucleation_spec(
        Path(__file__).resolve().parents[1]
        / "examples/nucleation/cdse_molecular_rules.yaml"
    )
    assert max_structure_level_possible(3, 1, spec) == 1
    assert max_structure_level_possible(4, 1, spec) == 2
    assert max_structure_level_possible(2, 2, spec) == 0

    # k=3 p=0: free path (pattern not possible). min_cn≥2 may force a
    # 6-cycle graph, but it is NOT ring_first / min-pattern chemistry.
    rules = replace(
        spec.graph_rules,
        ring_first_when_pattern_possible=True,
        ring_min_pattern_cd=(3, 3, 4),
        ring_min_pattern_se=(3, 3, 3),
    )
    spec0 = replace(spec, graph_rules=rules)
    assert max_structure_level_possible(3, 0, spec0) == 0
    skels0, _ = _enumerate_inorganic_edge_sets(3, 0, spec0, mode="free")
    assert skels0, "expected free skeletons at (3,0)"
    # free_fallback at (3,1) prefers acyclic when any exist
    skels_fb, _ = _enumerate_inorganic_edge_sets(3, 1, spec0, mode="free")
    open_fb = [e for e in skels_fb if count_cdse_six_rings(e, 3, 1) == 0]
    if open_fb:
        assert all(count_cdse_six_rings(e, 3, 1) == 0 for e in skels_fb)


def test_fused2_all_modes_have_two_six_rings(mol_spec) -> None:
    """Fused-2 seeds (all modes) produce skeletons with ≥2 Cd–Se 6-rings."""

    from dataclasses import replace
    import networkx as nx
    from builder.nucleation.molecular import _enumerate_inorganic_edge_sets

    rules = replace(
        mol_spec.graph_rules,
        ring_first_when_pattern_possible=True,
        multi_ring_ladder=True,
        max_cn={**mol_spec.graph_rules.max_cn, "Se": 5},
    )
    spec = replace(mol_spec, graph_rules=rules)

    def n_six_rings(edges, k, p):
        se_ids = set(range(0, k))
        cd_ids = list(range(k, k + k + p))
        g = nx.Graph()
        g.add_nodes_from(list(se_ids) + cd_ids)
        g.add_edges_from(edges)
        rings = set()
        for i, c1 in enumerate(cd_ids):
            for c2 in cd_ids[i + 1 :]:
                for c3 in cd_ids[cd_ids.index(c2) + 1 :]:
                    n1 = set(g.neighbors(c1)) & se_ids
                    n2 = set(g.neighbors(c2)) & se_ids
                    n3 = set(g.neighbors(c3)) & se_ids
                    for a in n1 & n2:
                        for b in n2 & n3:
                            for c in n3 & n1:
                                if len({a, b, c}) == 3:
                                    rings.add(frozenset([c1, c2, c3, a, b, c]))
        return len(rings)

    skels, _ = _enumerate_inorganic_edge_sets(5, 1, spec, mode="fused2")
    assert skels, "expected fused2 skeletons at (5,1)"
    for skel in skels[:20]:
        assert n_six_rings(skel, 5, 1) >= 2, skel


def test_ring_first_fallback_when_zero_accepts(mol_spec, pack) -> None:
    """Ring-first with 0 passivated isomers falls back to free skeletons."""

    from dataclasses import replace
    from builder.nucleation.molecular import enumerate_molecular_bin

    rules = replace(
        mol_spec.graph_rules,
        ring_first_when_pattern_possible=True,
        ring_first_fallback_to_open=True,
        decoration_mode="skeleton_bridge_first",
    )
    spec = replace(mol_spec, graph_rules=rules)
    # Small bin that still can run both paths
    res = enumerate_molecular_bin(
        1, 1, spec, pack=pack, embed=True, skeleton_mode="auto"
    )
    # k=1 never ring-first; free path
    assert res.skeleton_mode_used in {"free", "precomputed"}
    assert len(res.isomers) >= 1

    # Auto ladder on k=3 p=1 (1-ring possible, fused2 not) with fallback
    res2 = enumerate_molecular_bin(
        3,
        1,
        spec,
        pack=pack,
        embed=True,
        skeleton_mode="auto",
        allow_ring_fallback=True,
    )
    assert res2.skeleton_mode_used in {
        "ring_first",
        "free",
        "free_fallback",
    }
    if res2.ring_first_proved:
        assert res2.proved_level >= 1
        assert len(res2.isomers) >= 1
    # else: fell back toward free after structured seeds failed


def test_ring_first_builds_closed_six_ring(mol_spec) -> None:
    """When ring-first is ON, every skeleton contains a Cd–Se 6-cycle."""

    from dataclasses import replace
    import networkx as nx
    from builder.nucleation.molecular import (
        _enumerate_inorganic_edge_sets,
        ring_first_required_for_spec,
    )

    rules = replace(
        mol_spec.graph_rules,
        ring_first_when_pattern_possible=True,
        ring_min_pattern_cd=(3, 3, 4),
        ring_min_pattern_se=(3, 3, 3),
        max_cn={**mol_spec.graph_rules.max_cn, "Se": 4},
    )
    spec = replace(mol_spec, graph_rules=rules)
    assert ring_first_required_for_spec(3, 1, spec)
    assert not ring_first_required_for_spec(3, 0, spec)

    def has_6ring(edges, k, p):
        se_ids = set(range(0, k))
        cd_ids = set(range(k, k + k + p))
        g = nx.Graph()
        g.add_nodes_from(se_ids | cd_ids)
        g.add_edges_from(edges)
        cds = list(cd_ids)
        for i, c1 in enumerate(cds):
            for c2 in cds[i + 1 :]:
                for c3 in cds[cds.index(c2) + 1 :]:
                    n1, n2, n3 = (
                        set(g.neighbors(c1)) & se_ids,
                        set(g.neighbors(c2)) & se_ids,
                        set(g.neighbors(c3)) & se_ids,
                    )
                    for a in n1 & n2:
                        for b in n2 & n3:
                            for c in n3 & n1:
                                if len({a, b, c}) == 3:
                                    return True
        return False

    for k, p in ((3, 1), (4, 0)):
        skels, _ = _enumerate_inorganic_edge_sets(k, p, spec)
        assert not skels, "hard Se333 plus no-C4 makes this ring attempt infeasible"
    for k, p in ((4, 2), (5, 1)):
        skels, _ = _enumerate_inorganic_edge_sets(k, p, spec)
        assert skels, f"expected skeletons at k={k} p={p}"
        for skel in skels:
            assert has_6ring(skel, k, p), (k, p, skel)

    # Below onset: free enum still works (may or may not have rings)
    skels0, _ = _enumerate_inorganic_edge_sets(3, 0, spec)
    # k=3 p=0: ring-first OFF; accepted skeletons can exist without full pattern
    assert isinstance(skels0, list)


def test_skeleton_catalog_roundtrip_and_decorate(mol_spec, pack, tmp_path) -> None:
    """Dump edges → load catalog → decorate without re-enumeration."""

    from builder.nucleation.molecular import (
        dump_skeletons_upfront,
        enumerate_molecular_bin,
        format_skeleton_edges,
        load_skeleton_catalog,
        parse_skeleton_edges,
    )

    edges = ((0, 2), (1, 2), (1, 3))
    assert parse_skeleton_edges(format_skeleton_edges(edges)) == edges

    dump_skeletons_upfront(
        mol_spec,
        tmp_path,
        pack=pack,
        kmin=1,
        kmax=1,
        pmin=1,
        pmax=1,
        embed=True,
    )
    csv_path = tmp_path / "skeletons.csv"
    assert csv_path.is_file()
    text = csv_path.read_text(encoding="utf-8")
    assert "edges" in text.splitlines()[0]
    catalog = load_skeleton_catalog(tmp_path, accepted_only=True, require_edges=True)
    assert (1, 1) in catalog
    assert catalog[(1, 1)]

    res = enumerate_molecular_bin(
        1,
        1,
        mol_spec,
        pack=pack,
        embed=True,
        precomputed_skeletons=catalog[(1, 1)],
    )
    assert res.raw_graphs >= 1
    assert len(res.isomers) >= 1


def test_orderly_c4free_skeleton_generation(mol_spec) -> None:
    """Orderly bipartite enum: unique, C4-free when min_ring=6, fast at k=4."""

    from dataclasses import replace
    import networkx as nx
    from builder.nucleation.molecular import (
        _atoms_for_composition,
        _roles_for_composition,
        _skeleton_graph_violations,
        _State,
        _symbols_for_composition,
    )

    rules = replace(
        mol_spec.graph_rules,
        max_cn={**mol_spec.graph_rules.max_cn, "Se": 4},
    )
    spec4 = replace(mol_spec, graph_rules=rules)

    # k=3 p=0: single girth-6 core (no isomorphic flood)
    skels, trunc = _enumerate_inorganic_edge_sets(3, 0, spec4)
    assert not trunc
    assert len(skels) == 1
    assert len(set(skels)) == 1

    # k=4 p=1 and p=2: finishes quickly with only unique C4-free graphs
    for p in (1, 2):
        skels, trunc = _enumerate_inorganic_edge_sets(4, p, spec4, mode="free")
        assert not trunc
        assert len(skels) == len(set(skels)) >= 1
        symbols = _symbols_for_composition(spec4, 4, p)
        roles = _roles_for_composition(spec4, 4, p)
        atoms = _atoms_for_composition(symbols, roles)
        for skel in skels:
            g = nx.Graph()
            g.add_nodes_from(range(len(atoms)))
            g.add_edges_from(skel)
            # Incremental C4 ban ⇒ no ring_too_small left for post-filter
            viol = _skeleton_graph_violations(
                _State(atoms=atoms, graph=g), spec4
            )
            assert not any(v.startswith("ring_too_small") for v in viol), viol
            # Explicit: any two Cd share at most one Se
            se = [a.atom_id for a in atoms if a.symbol == "Se"]
            cd = [a.atom_id for a in atoms if a.symbol == "Cd"]
            for i, c1 in enumerate(cd):
                n1 = {x for x in g.neighbors(c1) if x in se}
                for c2 in cd[i + 1 :]:
                    n2 = {x for x in g.neighbors(c2) if x in se}
                    assert len(n1 & n2) <= 1


def test_isomers_have_unique_certificates(mol_spec, pack) -> None:
    bin_res = enumerate_molecular_bin(1, 2, mol_spec, pack=pack, embed=False)
    certs = [iso.certificate for iso in bin_res.isomers]
    assert len(certs) == len(set(certs))


def test_exact_k1_p2_retains_feasible_geometry(mol_spec, pack) -> None:
    result = enumerate_molecular_bin(1, 2, mol_spec, pack=pack, embed=True)
    assert result.isomers
    found_bridged_cd_cn3 = False
    for isomer in result.isomers:
        state = _State(atoms=isomer.atoms, graph=isomer.graph)
        ok, reasons = molecular_geometry_ok(
            state,
            () if isomer.coordinates is None else isomer.coordinates,
            mol_spec,
        )
        assert ok, reasons
        coordinates = np.asarray(isomer.coordinates)
        for atom in isomer.atoms:
            neighbors = sorted(isomer.graph.neighbors(atom.atom_id))
            if atom.symbol != "Cd" or len(neighbors) != 3:
                continue
            vectors = [
                (coordinates[neighbor] - coordinates[atom.atom_id])
                / np.linalg.norm(
                    coordinates[neighbor] - coordinates[atom.atom_id]
                )
                for neighbor in neighbors
            ]
            improper_deg = abs(
                math.degrees(
                    math.asin(
                        max(
                            -1.0,
                            min(
                                1.0,
                                float(
                                    np.dot(
                                        vectors[0],
                                        np.cross(vectors[1], vectors[2]),
                                    )
                                ),
                            ),
                        )
                    )
                )
            )
            # A CN3 Cd is planar to within the pack's band, not to machine
            # precision.  The old ``< 1e-6`` encoded an exact-planarity
            # contract that the pack's own angle medians contradict: three
            # angles sum to 360 only for a planar centre, and the DFT medians
            # sum to less than that.  The band is what the audit enforces.
            band = pack.audit_improper_tolerance_deg
            if band is not None:
                assert improper_deg <= band, improper_deg
            cl_neighbors = [
                neighbor
                for neighbor in neighbors
                if isomer.atoms[neighbor].symbol == "Cl"
            ]
            se_neighbors = [
                neighbor
                for neighbor in neighbors
                if isomer.atoms[neighbor].symbol == "Se"
            ]
            if len(cl_neighbors) == 2 and len(se_neighbors) == 1:
                cl_vectors = [
                    (coordinates[neighbor] - coordinates[atom.atom_id])
                    / np.linalg.norm(
                        coordinates[neighbor] - coordinates[atom.atom_id]
                    )
                    for neighbor in cl_neighbors
                ]
                cl_cd_cl = np.degrees(
                    np.arccos(
                        np.clip(
                            float(np.dot(cl_vectors[0], cl_vectors[1])),
                            -1.0,
                            1.0,
                        )
                    )
                )
                has_bridge = any(
                    isomer.graph.degree[neighbor] > 1
                    for neighbor in cl_neighbors
                )
                cl_degrees = sorted(
                    isomer.graph.degree[neighbor]
                    for neighbor in cl_neighbors
                )
                expected_by_roles = {
                    (1, 1): 121.857,
                    (1, 2): 122.366,
                    (1, 3): 111.797,
                    (2, 2): 138.068,
                }
                expected = expected_by_roles[tuple(cl_degrees)]
                assert cl_cd_cl == pytest.approx(expected, abs=1.0e-6)
                found_bridged_cd_cn3 |= has_bridge
    assert found_bridged_cd_cn3


def test_contact_threshold_never_changes_constructed_coordinates(
    mol_spec, pack
) -> None:
    result = enumerate_molecular_bin(1, 1, mol_spec, pack=pack, embed=True)
    isomer = result.isomers[0]
    state = _State(atoms=isomer.atoms, graph=isomer.graph)
    baseline = np.asarray(embed_molecular_state(state, pack, mol_spec))
    rules = dict(mol_spec.graph_rules.pair_rules)
    rules["Cl-Se"] = replace(rules["Cl-Se"], min_distance=10.0)
    strict_spec = replace(
        mol_spec,
        graph_rules=replace(mol_spec.graph_rules, pair_rules=rules),
    )
    strict = np.asarray(embed_molecular_state(state, pack, strict_spec))
    assert np.array_equal(baseline, strict)
    ok, reasons = molecular_geometry_ok(state, strict, strict_spec)
    assert not ok
    assert any(reason.startswith("contact:Cl-Se") for reason in reasons)


def test_exact_enumeration_limit_is_not_silent(mol_spec) -> None:
    with pytest.raises(EnumerationLimitError):
        enumerate_molecular_bin(
            1,
            3,
            mol_spec,
            embed=False,
            max_decoration_assignments=1,
        )


def test_cl_mu3_is_an_explicit_graph_mode(mol_spec) -> None:
    skeleton = _enumerate_inorganic_edge_sets(1, 3, mol_spec)[0][0]
    decorations, truncated = _cl_attachments(
        1, 3, skeleton, mol_spec, max_assignments=5000
    )
    assert not truncated
    cl_ids = range(1 + 1 + 3, 1 + 1 + 3 + 6)
    assert any(
        any(
            sum(1 for edge in decoration if cl in edge) == 3
            for cl in cl_ids
        )
        for decoration in decorations
    )


def test_degree_first_pruning_emits_fewer_graphs_than_unconditioned(
    mol_spec, pack
) -> None:
    """Geometry-conditioned generation must not enlarge the candidate set.

    Degree-first exact surplus cover + per-frame mode support is a necessary
    filter on the unconditioned multiset enumerator.  On a modest bin it must
    therefore emit strictly fewer raw graphs while keeping every accepted
    isomer the unconditioned path would keep under the same screen.
    """

    from builder.nucleation.molecular import (
        iter_cl_attachments,
        survey_skeleton_frames,
        _DecorationStatus,
        _atoms_for_composition,
        _roles_for_composition,
        _symbols_for_composition,
    )
    import networkx as nx

    k, p = 1, 3
    symbols = _symbols_for_composition(mol_spec, k, p)
    atoms = _atoms_for_composition(
        symbols, _roles_for_composition(mol_spec, k, p)
    )
    skeleton = _enumerate_inorganic_edge_sets(k, p, mol_spec)[0][0]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(atoms)))
    graph.add_edges_from(skeleton)
    cation_ids = [
        atom.atom_id
        for atom in atoms
        if atom.symbol in {mol_spec.core.cation, mol_spec.precursor.center}
    ]
    survey = survey_skeleton_frames(
        _State(atoms=atoms, graph=graph),
        cation_ids,
        pack,
        mol_spec,
        2 * p,
        eager=True,
    )
    assert survey.degree_slices is not None
    assert survey.degree_slices

    status_fast = _DecorationStatus()
    fast = list(
        iter_cl_attachments(
            k,
            p,
            skeleton,
            mol_spec,
            status=status_fast,
            degree_slices=survey.degree_slices,
        )
    )
    status_full = _DecorationStatus()
    full = list(
        iter_cl_attachments(
            k,
            p,
            skeleton,
            mol_spec,
            status=status_full,
        )
    )
    assert len(fast) <= len(full)
    assert len(fast) < len(full)
    # Every degree-first decoration must realise some surveyed CN vector.
    cd_ids = list(cation_ids)
    allowed_degrees = {slice_.degree for slice_ in survey.degree_slices}
    base = [graph.degree[c] for c in cd_ids]
    for decoration in fast:
        added = {c: 0 for c in cd_ids}
        for _cl, host in decoration:
            if host in added:
                added[host] += 1
        final = tuple(base[i] + added[cd_ids[i]] for i in range(len(cd_ids)))
        assert final in allowed_degrees


def test_construction_defaults_from_pack(mol_spec, pack) -> None:
    """Molecular pack enables bridge-host CN floor and mono-Se dual-terminal ban."""

    assert mol_spec.graph_rules.min_bridged_host_cn == 3
    assert mol_spec.graph_rules.forbid_mono_se_dual_terminal is True
    assert mol_spec.graph_rules.reject_closable_terminal_cd2 is False
    assert mol_spec.graph_rules.require_bridge_maximal is False
    result = enumerate_molecular_bin(1, 2, mol_spec, pack=pack, embed=True)
    assert len(result.isomers) >= 1
    # No accepted isomer may host a bridge on final CN2 or dual-terminal mono-Se.
    from builder.nucleation.molecular import (
        min_bridged_host_cn_violations,
        mono_se_dual_terminal_violations,
    )

    for isomer in result.isomers:
        state = _State(atoms=isomer.atoms, graph=isomer.graph)
        assert not min_bridged_host_cn_violations(state, mol_spec)
        assert not mono_se_dual_terminal_violations(state, mol_spec)


def test_min_bridged_host_cn_removes_cn2_bridge_hosts(mol_spec, pack) -> None:
    from dataclasses import replace

    loose = replace(
        mol_spec,
        graph_rules=replace(
            mol_spec.graph_rules,
            min_bridged_host_cn=1,
            forbid_mono_se_dual_terminal=False,
        ),
    )
    strict = mol_spec
    base = enumerate_molecular_bin(2, 2, loose, pack=pack, embed=True)
    filt = enumerate_molecular_bin(2, 2, strict, pack=pack, embed=True)
    assert len(filt.isomers) <= len(base.isomers)


def test_reject_closable_terminal_cd2_flag_reduces_accepts(
    mol_spec, pack
) -> None:
    from dataclasses import replace

    strict = replace(
        mol_spec,
        graph_rules=replace(
            mol_spec.graph_rules,
            reject_closable_terminal_cd2=True,
        ),
    )
    baseline = enumerate_molecular_bin(2, 2, mol_spec, pack=pack, embed=True)
    filtered = enumerate_molecular_bin(2, 2, strict, pack=pack, embed=True)
    assert len(filtered.isomers) <= len(baseline.isomers)
    # Every survivor must have zero closable contacts under the pack distance.
    for isomer in filtered.isomers:
        assert isomer.annotations is not None
        assert isomer.annotations.n_closable_terminal_cd2 == 0


def test_forbid_cdse_cn_pairs_rejects_matching_graphs(
    mol_spec, pack
) -> None:
    from dataclasses import replace

    # Forbid a pair that appears in small-k accepts when present.
    baseline = enumerate_molecular_bin(1, 2, mol_spec, pack=pack, embed=True)
    assert baseline.isomers
    # Collect observed pairs from one isomer and forbid one of them.
    from builder.nucleation.molecular import annotate_molecular_state
    from builder.nucleation.types import _State

    iso = baseline.isomers[0]
    ann = annotate_molecular_state(
        _State(atoms=iso.atoms, graph=iso.graph), mol_spec, iso.coordinates
    )
    if not ann.cdse_cn_pairs:
        return
    first = ann.cdse_cn_pairs.split(",")[0]
    cd_cn, se_cn = first.split(":")[0].split("-")
    strict = replace(
        mol_spec,
        graph_rules=replace(
            mol_spec.graph_rules,
            forbid_cdse_cn_pairs=((int(cd_cn), int(se_cn)),),
        ),
    )
    filtered = enumerate_molecular_bin(1, 2, strict, pack=pack, embed=True)
    assert len(filtered.isomers) < len(baseline.isomers)
    assert filtered.rejection_reasons.get("forbidden_cdse_cn_pair", 0) > 0


def test_collapse_annotations_flag_unsaturated_and_closable(
    mol_spec, pack
) -> None:
    """Accepted isomers carry DFT-collapse risk labels without changing the set."""

    from builder.nucleation.molecular import annotate_molecular_state

    result = enumerate_molecular_bin(1, 2, mol_spec, pack=pack, embed=True)
    assert result.isomers
    for isomer in result.isomers:
        assert isomer.annotations is not None
        assert isomer.annotations.n_cd2 + isomer.annotations.n_cd3 + (
            isomer.annotations.n_cd4
        ) <= sum(
            1
            for atom in isomer.atoms
            if atom.symbol in {mol_spec.core.cation, mol_spec.precursor.center}
        )
        # Recompute matches stored annotation.
        state = _State(atoms=isomer.atoms, graph=isomer.graph)
        again = annotate_molecular_state(
            state, mol_spec, isomer.coordinates
        )
        assert again.n_cd2 == isomer.annotations.n_cd2
        assert again.n_terminal_cl == isomer.annotations.n_terminal_cl
        assert (
            again.n_unsaturated_bridge_candidates
            == isomer.annotations.n_unsaturated_bridge_candidates
        )


def test_write_molecular_map(tmp_path, mol_spec, pack) -> None:
    progress = []
    output = tmp_path / "molmap"
    result = generate_molecular_map(
        mol_spec,
        geometry_pack=pack,
        kmax=1,
        pmax=2,
        embed=True,
        incremental_output=output,
        progress=progress.append,
    )
    assert (1, 1) in result.bins
    out = write_molecular_map(result, output)
    assert (out / "index.csv").is_file()
    assert (out / "annotations.csv").is_file()
    assert (out / "rejections.csv").is_file()
    assert (out / "k001" / "p001" / "bin_meta.txt").is_file()
    ann_header = (out / "annotations.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "n_cd2" in ann_header
    assert "n_closable_terminal_cd2" in ann_header
    assert any(
        line.startswith("[molecular] CHECKPOINT k=1 p=1")
        for line in progress
    )
    assert any(line.startswith("[molecular] SAVED k=1 p=1") for line in progress)
    # at least one xyz for p>=1
    xyzs = list(out.rglob("*.xyz"))
    assert len(xyzs) >= 1
