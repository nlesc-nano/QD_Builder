"""Scientific invariants and budget/resume behaviour; no quantum backend required."""
import csv
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest

from builder.nucleation.molecular_zb_growth import (
    ZbOccupation,
    _occupation_shape_certificate,
    _compactness_from_core,
    _growth_site_priority,
)
from builder.nucleation.pilot import Pilot, PilotConfig, round_robin, select_population
from builder.nucleation.adaptive import (
    AdaptiveConfig,
    AdaptivePilot,
    coarse_lineage_family,
    lineage_family,
)
from builder.nucleation.geometry_pack import load_geometry_pack
from builder.nucleation.molecular import mu3_host_bridge_overlap_violations
from builder.nucleation.pilot_proposals import (
    Proposal,
    bounded_shells,
    graph_state,
    local_proposals,
    validate_composition,
)
from builder.nucleation.search_analysis import (
    descriptors,
    iter_corpus,
    corrected_occupation,
    analyze_corpus,
)
from builder.nucleation.xtb_relax import XtbResult
from builder.nucleation.spec import load_nucleation_spec
from tools.build_adaptive_policy_view import build_policy_view
from tools.export_adaptive_minima_xyz import export_archive

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "geometry_packs/cdse_cdcl2_zb"


def test_identity_translation_cubic_and_permutation():
    symbols = ["Se", "Se", "Cd"]
    xyz = np.array(
        [[0.0, 0, 0], [3.0702712, 3.0702712, 0], [1.5351356, 1.5351356, 1.5351356]]
    )
    key = _occupation_shape_certificate(symbols, xyz, 0.2)
    for shift in [0.05, 0.1, 3.0702712, 123.45]:
        for axes in [[0, 1, 2], [1, 2, 0]]:
            for signs in [[1, 1, 1], [-1, 1, -1]]:
                coords = (xyz[:, axes] * signs) + shift
                assert (
                    _occupation_shape_certificate(symbols[::-1], coords[::-1], 0.2)
                    == key
                )
    assert _occupation_shape_certificate(symbols, xyz * 2, 0.2) != key


def test_shell_counts_and_bounded_exhaustion():
    rng = np.random.default_rng(5)
    sy = ["Se", "Se", "Cd", "Cd", "Cd", "Cd"]
    edges = [(0, 2), (0, 3), (1, 3), (1, 4), (1, 5)]
    primary, explore, stats = bounded_shells(
        sy, edges, 2, rng, total_nodes=25, tier_nodes=12
    )
    assert stats["nodes"] <= 25
    assert stats["feasible_tiers"] <= 3
    for state in primary + explore:
        symbols = [a.symbol for a in state.atoms]
        f = descriptors(symbols, state.graph.edges)
        assert sum(f["cl_cn"].values()) == 4
        assert f["mu2"] == sum(
            state.graph.degree(i) == 2 for i, s in enumerate(symbols) if s == "Cl"
        )


def test_six_ring_census_is_not_cycle_basis():
    # K3,3 has six distinct six-cycles, but its cycle basis has only four cycles.
    edges = [(a, b) for a in range(3) for b in range(3, 6)]
    symbols = ["Se"] * 3 + ["Cd"] * 3
    assert _compactness_from_core(symbols, np.zeros((6, 3)), edges)[0] == -6
    assert _compactness_from_core(symbols, np.zeros((6, 3)), edges[::-1])[0] == -6


def test_ligands_follow_core_frame_and_shedding():
    xyz = np.array([[0.0, 0, 0], [2.6, 0, 0], [0, 2.6, 0]])
    parent = ZbOccupation(
        1, 1, ("Se", "Cd", "Cd"), xyz, ((0, 1), (0, 2)), ("a", "b", "c")
    )
    child = ZbOccupation(
        2,
        0,
        ("Se", "Cd", "Cd", "Se"),
        np.vstack([xyz, [5.2, 0, 0]]),
        ((0, 1), (0, 2), (1, 3)),
        ("a", "b", "c", "d"),
    )
    ligands = np.array([[2.6, 0, 2.3], [2.6, 0, -2.3]])
    kwargs = dict(parent_wbo=None, ligand_bond_length=2.5)
    key = _growth_site_priority(
        child,
        parent,
        relaxed_parent_coordinates=xyz,
        parent_ligand_coordinates=ligands,
        **kwargs,
    )
    rotation = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
    moved = _growth_site_priority(
        child,
        parent,
        relaxed_parent_coordinates=xyz @ rotation + 10,
        parent_ligand_coordinates=ligands @ rotation + 10,
        **kwargs,
    )
    assert key == moved
    shed = _growth_site_priority(
        replace(child, shed_ligand_indices=(0, 1)),
        parent,
        relaxed_parent_coordinates=xyz,
        parent_ligand_coordinates=ligands,
        **kwargs,
    )
    assert shed[3] == key[3] - 2  # two newly free ligand coordination slots


def test_legacy_metadata_and_aliases(tmp_path):
    row = dict(
        structure_id="legacy",
        occupation=dict(
            k=1,
            p=1,
            symbols=["Se", "Cd", "Cd"],
            lattice_coordinates=[[0, 0, 0], [1.5, 1.5, 1.5], [-1.5, -1.5, 1.5]],
            core_edges=[[0, 1], [0, 2]],
            occupation_id="old",
        ),
        energy_eV=-10,
        xtb_converged=True,
        violations=[],
        final_edges=[[0, 1], [0, 2], [1, 3], [2, 4]],
        topology_status="changed",
    )
    (tmp_path / "zb_occupations.jsonl").write_text(json.dumps(row) + "\n")
    got = list(iter_corpus(tmp_path))[0]
    assert got["accepted"]
    assert got["protocol_verified"] is False
    assert got["occupation"]["identity_aliases"] == ["old"]
    from builder.nucleation.spec import load_nucleation_spec

    result = analyze_corpus(
        tmp_path,
        tmp_path / "analysis",
        load_nucleation_spec(PACK / "run_gxtb.yaml"),
        geometry=False,
    )
    assert result["bins"][0]["mu2"] == 0
    assert (tmp_path / "analysis" / "assessment.md").is_file()
    row["xtb_converged"] = False
    (tmp_path / "zb_occupations.jsonl").write_text(json.dumps(row) + "\n")
    assert not list(iter_corpus(tmp_path))[0]["accepted"]


def make_pilot(tmp_path, backend, **kwargs):
    config = PilotConfig(k_max=3, workers=2, max_calls=12, seed_calls=4, **kwargs)
    seeds = tmp_path / "seeds"
    seeds.mkdir(exist_ok=True)
    (seeds / "index.csv").write_text("k,p,structure_id\n")
    return Pilot(
        config,
        PACK / "run_gxtb.yaml",
        PACK / "growth_agnostic_k5.yaml",
        seeds,
        tmp_path / "out",
        backend=backend,
    )


def example(index=0):
    return Proposal(
        1,
        1,
        ["Se", "Cd", "Cd", "Cl", "Cl"],
        [[0.0, 0, 0], [2.6, 0, 0], [-2.6, 0, 0], [4.9, 0, 0], [-4.9, 0, 0]],
        [[0, 1], [0, 2], [1, 3], [2, 4]],
        ["root"],
        f"seed{index}",
    )


def test_reservation_retry_resume_and_protocol(tmp_path):
    calls = []

    def backend(payload, settings, cutoffs):
        calls.extend(payload)
        return [
            XtbResult(
                ok=True,
                energy_eV=-10,
                coordinates=tuple(map(tuple, payload[0]["positions"])),
                converged=False,
            )
        ]

    pilot = make_pilot(tmp_path, backend)
    pilot.setup()
    pilot.evaluate([("shared", example())], 1)
    assert len(calls) == 2  # initial plus one charged retry
    assert pilot.calls["shared", 1] == 2
    assert not pilot.rows["experimental"]  # unconverged energies cannot enter minima
    resumed = make_pilot(tmp_path, backend)
    resumed.setup()
    resumed.evaluate([("shared", example())], 1)
    assert len(calls) == 2
    resumed.config.seed = 42
    with pytest.raises(ValueError, match="fingerprint"):
        resumed.setup()


def test_failed_calls_are_charged_and_global_cap(tmp_path):
    def backend(*args):
        raise RuntimeError("simulated solver failure")

    pilot = make_pilot(tmp_path, backend)
    pilot.setup()
    pilot.evaluate([("shared", example(i)) for i in range(20)], 1)
    assert pilot.calls["shared", 1] == 4
    assert len(pilot.completed) == 4


def test_batch_progress_is_compact_and_reports_outcomes(tmp_path, capsys):
    def backend(*args):
        return [XtbResult(ok=False, error="test failure")]

    pilot = make_pilot(tmp_path, backend)
    pilot.setup()
    pilot.evaluate([("shared", example(0)), ("shared", example(1))], 1)
    output = capsys.readouterr().out
    assert "[pilot] launch k=1 batch=1 size=2" in output
    assert "channels=seed0:1,seed1:1 compositions=p1:2" in output
    assert "calls_total=2/12" in output
    assert "[pilot] finish k=1 batch=1 outcomes=failed:2" in output
    assert "queue_left=0" in output


def test_export_adaptive_archive_writes_energy_ranked_xyz_stacks(tmp_path):
    first = dict(
        example(0).record(),
        minimum_id="higher",
        structure_id="higher-route",
        energy_eV=-9.0,
        role="primary",
        final={
            "graph_hash": "same-relaxed-graph",
            "mu2": 2,
            "n4": 1,
            "n6": 0,
            "se_cn": {"2": 1},
        },
        source="test",
    )
    second = dict(first, minimum_id="lower", energy_eV=-10.0)
    audit = dict(first, minimum_id="audit", energy_eV=-11.0, role="audit")
    archive = tmp_path / "run"
    archive.mkdir()
    (archive / "minima.json").write_text(
        json.dumps(
            {
                "experimental": {row["minimum_id"]: row for row in (first, second, audit)},
                "control": {},
            }
        )
    )
    summary = export_archive(archive, tmp_path / "xyz")
    assert summary == {"structures": 2, "bins": 1, "skipped": 0}
    lines = (tmp_path / "xyz/k1_p1.xyz").read_text().splitlines()
    frame_lines = len(first["symbols"]) + 2
    assert lines[1].startswith(
        "-0.3674932218 lower E_Ha=-0.3674932218 E_eV=-10.0000000000"
    )
    assert lines[frame_lines + 1].startswith(
        "-0.3307438996 higher E_Ha=-0.3307438996 E_eV=-9.0000000000"
    )
    with (tmp_path / "xyz/index.csv").open() as handle:
        exported = list(csv.DictReader(handle))
    assert [row["minimum_id"] for row in exported] == ["lower", "higher"]
    assert exported[1]["relative_energy_kcal_mol"] == "23.060548"

    collapsed = export_archive(
        archive, tmp_path / "collapsed", one_per_graph=True
    )
    assert collapsed == {"structures": 1, "bins": 1, "skipped": 0}
    with (tmp_path / "collapsed/index.csv").open() as handle:
        representative = next(csv.DictReader(handle))
    assert representative["minimum_id"] == "lower"
    assert representative["graph_members"] == "2"


def test_policy_view_promotes_only_selected_overlap_only_audits(tmp_path):
    base = dict(
        example().record(),
        k=10,
        p=13,
        role="audit",
        energy_eV=-10.0,
        protocol="strict",
        classification_mode="strict",
    )
    overlap = dict(
        base,
        minimum_id="overlap",
        violations=["mu3_host_bridge_overlap:1-2-3"],
    )
    multiple = dict(
        base,
        minimum_id="multiple",
        violations=["mu3_host_bridge_overlap:1-2-3", "max_cn:Cd:1:5>4"],
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "minima.json").write_text(
        json.dumps(
            {
                "control": {},
                "experimental": {"overlap": overlap, "multiple": multiple},
            }
        )
    )
    archive, policy = build_policy_view(
        source,
        k=10,
        ps={11, 12, 13},
        allowed_violations={"mu3_host_bridge_overlap"},
        mode="relaxed-overlap",
    )
    assert policy["promoted_by_p"] == {"13": 1}
    promoted = archive["experimental"]["overlap"]
    assert promoted["role"] == "primary"
    assert promoted["violations"] == []
    assert promoted["classification_eligibility"] == {
        "audit": ["strict"],
        "primary": ["relaxed-overlap"],
    }
    assert archive["experimental"]["multiple"]["role"] == "audit"


def test_frozen_queue_and_changed_seed_reject_resume(tmp_path):
    pilot = make_pilot(tmp_path, lambda *args: [])
    seed_xyz = pilot.seed_dir / "seed.xyz"
    seed_xyz.write_text("1\nseed\nCd 0 0 0\n")
    pilot.setup()
    original = pilot.prepare_queue("experimental", 2, 0, lambda: [example()])
    resumed = make_pilot(tmp_path, lambda *args: [])
    resumed.setup()
    restored = resumed.prepare_queue(
        "experimental", 2, 0, lambda: pytest.fail("rebuilt frozen queue")
    )
    assert [p.id for p in restored] == [p.id for p in original]
    seed_xyz.write_text("1\nseed\nCd 1 0 0\n")
    with pytest.raises(ValueError, match="fingerprint"):
        resumed.setup()


def test_audit_quota_uses_actual_calls(tmp_path):
    def backend(*args):
        return [XtbResult(ok=False, error="test")]

    pilot = make_pilot(tmp_path, backend)
    pilot.config.max_calls = 100
    pilot.setup()
    pilot.evaluate([("experimental", replace(example(0), audit_derived=True))], 2)
    assert pilot.calls["experimental", 2] == 0
    pilot.evaluate([("experimental", example(i)) for i in range(1, 10)], 2)
    pilot.evaluate([("experimental", replace(example(10), audit_derived=True))], 2)
    assert pilot.calls["experimental", 2] == 10
    assert pilot.audit_calls["experimental", 2] == 1
    pilot.evaluate([("experimental", replace(example(11), audit_derived=True))], 2)
    assert pilot.calls["experimental", 2] == 10


def test_shared_audit_seed_does_not_enter_control(tmp_path):
    pilot = make_pilot(tmp_path, lambda *args: [])
    p = example()
    row = dict(
        p.record(),
        structure_id=p.id,
        energy_eV=-10,
        role="audit",
        final=descriptors(p.symbols, p.edges, p.positions),
        protocol="test",
        source="seed:audit",
    )
    pilot.store("shared", row)
    assert len(pilot.rows["experimental"]) == 1
    assert not pilot.rows["control"]


def test_deadline_and_reaction_composition(tmp_path):
    pilot = make_pilot(tmp_path, lambda *args: pytest.fail("deadline ignored"))
    pilot.setup()
    pilot.started -= 47 * 3600
    pilot.evaluate([("shared", example())], 1)
    assert not pilot.calls
    parent = dict(example().record(), minimum_id="seed", role="primary")
    for channel in ["growth", "exchange", "reconstruction", "topology"]:
        jobs = local_proposals(
            parent,
            np.random.default_rng(5),
            pilot.spec,
            pilot.pack,
            channel=channel,
            limit=8,
        )
        assert len(jobs) <= 8
        for job in jobs:
            assert validate_composition(job)
            assert job.k == parent["k"] + (channel == "growth")


def test_topology_moves_change_core_graph_and_preserve_formula(tmp_path):
    pilot = make_pilot(tmp_path, lambda *args: [])
    parent = dict(
        k=2,
        p=1,
        symbols=["Se", "Se", "Cd", "Cd", "Cd", "Cl", "Cl"],
        positions=[
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 2.6, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 2.6, 0.0],
            [0.0, 4.9, 0.0],
            [3.0, 4.9, 0.0],
        ],
        edges=[[0, 2], [0, 3], [1, 3], [1, 4], [2, 5], [4, 6]],
        minimum_id="parent",
        role="primary",
    )
    jobs = local_proposals(
        parent,
        np.random.default_rng(7),
        pilot.spec,
        pilot.pack,
        channel="topology",
        limit=6,
    )
    assert jobs
    old_core = {
        tuple(edge)
        for edge in parent["edges"]
        if {parent["symbols"][edge[0]], parent["symbols"][edge[1]]} == {"Cd", "Se"}
    }
    assert all(validate_composition(job) for job in jobs)
    assert any(
        {
            tuple(edge)
            for edge in job.edges
            if {job.symbols[edge[0]], job.symbols[edge[1]]} == {"Cd", "Se"}
        }
        != old_core
        for job in jobs
    )


def test_adaptive_archive_import_preserves_ids_without_self_routes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    proposal = example()
    row = dict(
        proposal.record(),
        structure_id=proposal.id,
        minimum_id="minimum_source",
        energy_eV=-10.0,
        role="primary",
        final=descriptors(proposal.symbols, proposal.edges, proposal.positions),
        protocol="old-protocol",
        source="old",
        routes=["minimum_source", "seed:root"],
        occupations=[],
        occupation_origins=[],
    )
    (source / "minima.json").write_text(
        json.dumps({"control": {}, "experimental": {"minimum_source": row}})
    )
    config = AdaptiveConfig(
        workers=1,
        max_calls=5,
        stage_calls={1: 1, 2: 1, 3: 1},
        family_slots={1: 1, 2: 1, 3: 1},
        p_max={1: 3, 2: 5, 3: 6},
        phase_a_k=[1],
        phase_b_k=2,
        phase_c_k=3,
    )
    pilot = AdaptivePilot(
        config,
        PACK / "run_gxtb.yaml",
        PACK / "growth_agnostic_k5.yaml",
        source,
        tmp_path / "adaptive",
        backend=lambda *args: [],
    )
    pilot.setup()
    pilot.initialize_source()
    imported = pilot.rows["experimental"]["minimum_source"]
    assert imported["routes"] == ["seed:root"]
    assert imported["source_protocol"] == "old-protocol"
    assert lineage_family(imported).startswith("family_")
    resumed = AdaptivePilot(
        config,
        PACK / "run_gxtb.yaml",
        PACK / "growth_agnostic_k5.yaml",
        source,
        tmp_path / "adaptive",
        backend=lambda *args: [],
    )
    resumed.setup()
    assert list(resumed.rows["experimental"]) == ["minimum_source"]


def test_adaptive_import_uses_preconsolidated_merge_archive(tmp_path):
    source = tmp_path / "merged"
    source.mkdir()
    proposal = example()
    row = dict(
        proposal.record(),
        structure_id=proposal.id,
        minimum_id="minimum_merged",
        energy_eV=-10.0,
        role="primary",
        final=descriptors(proposal.symbols, proposal.edges, proposal.positions),
        protocol="merged-protocol",
        source="merged",
        routes=[],
        occupations=[],
        occupation_origins=[],
    )
    (source / "minima.json").write_text(
        json.dumps({"control": {}, "experimental": {"minimum_merged": row}})
    )
    (source / "protocol.json").write_text(
        json.dumps({"kind": "adaptive_archive_merge_v1"})
    )
    config = AdaptiveConfig(
        workers=1,
        max_calls=5,
        stage_calls={1: 1, 2: 1, 3: 1},
        family_slots={1: 1, 2: 1, 3: 1},
        p_max={1: 3, 2: 5, 3: 6},
        phase_a_k=[1],
        phase_b_k=2,
        phase_c_k=3,
    )
    output = tmp_path / "adaptive"
    pilot = AdaptivePilot(
        config,
        PACK / "run_gxtb.yaml",
        PACK / "growth_agnostic_k5.yaml",
        source,
        output,
        backend=lambda *args: [],
    )
    pilot.setup()
    pilot.initialize_source()
    assert pilot.rows["experimental"]["minimum_merged"]["minimum_id"] == (
        "minimum_merged"
    )
    event = next(e for e in pilot.events if e["event"] == "archive_import")
    assert event["preconsolidated"] is True

    resumed = AdaptivePilot(
        config,
        PACK / "run_gxtb.yaml",
        PACK / "growth_agnostic_k5.yaml",
        source,
        output,
        backend=lambda *args: [],
    )
    resumed.setup()
    assert list(resumed.rows["experimental"]) == ["minimum_merged"]


def test_basin_consolidation_does_not_downgrade_primary_role(tmp_path):
    proposal = example()
    final = descriptors(proposal.symbols, proposal.edges, proposal.positions)
    config = AdaptiveConfig(
        workers=1,
        max_calls=5,
        stage_calls={1: 1, 2: 1, 3: 1},
        family_slots={1: 1, 2: 1, 3: 1},
        p_max={1: 3, 2: 5, 3: 6},
        phase_a_k=[1],
        phase_b_k=2,
        phase_c_k=3,
    )
    pilot = AdaptivePilot(
        config,
        PACK / "run_gxtb.yaml",
        PACK / "growth_agnostic_k5.yaml",
        tmp_path,
        tmp_path / "adaptive",
        backend=lambda *args: [],
    )
    pilot.protocol = {"fingerprint": "test"}
    base = dict(
        proposal.record(),
        structure_id="strict-primary",
        energy_eV=-10.0,
        role="primary",
        violations=[],
        final=final,
        protocol="test",
        source="strict-primary",
        routes=[],
        occupations=[],
        occupation_origins=[],
        classification_mode="relaxed_high_p",
    )
    minimum_id, fresh = pilot.store("experimental", base)
    assert fresh
    audit = dict(
        base,
        structure_id="strict-audit",
        source="strict-audit",
        energy_eV=-10.001,
        role="audit",
        violations=["mu3_host_bridge_overlap:1-2-3"],
        classification_mode="strict",
    )
    duplicate_id, fresh = pilot.store("experimental", audit)
    assert not fresh
    assert duplicate_id == minimum_id
    retained = pilot.rows["experimental"][minimum_id]
    assert retained["role"] == "primary"
    assert retained["energy_eV"] == -10.0
    assert retained["classification_eligibility"] == {
        "audit": ["strict"],
        "primary": ["relaxed_high_p"],
    }
    assert len(retained["classification_history"]) == 2


def test_adaptive_continuation_imports_and_freezes_source_cohort(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    proposal = example()
    row = dict(
        proposal.record(),
        structure_id=proposal.id,
        minimum_id="minimum_source",
        energy_eV=-10.0,
        role="primary",
        final=descriptors(proposal.symbols, proposal.edges, proposal.positions),
        protocol="old-protocol",
        source="old",
        routes=[],
        occupations=[],
        occupation_origins=[],
    )
    family = coarse_lineage_family(row)
    (source / "minima.json").write_text(
        json.dumps({"control": {}, "experimental": {"minimum_source": row}})
    )
    (source / "events.jsonl").write_text(
        json.dumps(
            dict(
                event="cohort_initialized",
                phase="A",
                k=1,
                cycle=0,
                capacity=1,
                families=[family],
                known_families=[family],
            )
        )
        + "\n"
    )
    config = AdaptiveConfig(
        workers=1,
        max_calls=5,
        stage_calls={1: 1, 2: 1, 3: 1},
        family_slots={1: 1, 2: 1, 3: 1},
        p_max={1: 3, 2: 5, 3: 6},
        phase_a_k=[1],
        phase_b_k=2,
        phase_c_k=3,
        enabled_phases=["A"],
        import_source_cohorts=True,
        phase_cycle_start={"A": 5},
        max_cycles=8,
    )
    pilot = AdaptivePilot(
        config,
        PACK / "run_gxtb.yaml",
        PACK / "growth_agnostic_k5.yaml",
        source,
        tmp_path / "adaptive",
        backend=lambda *args: [],
    )
    pilot.setup()
    pilot.initialize_source()
    initialized, cohort = pilot._cohort_state("A", 1)
    assert cohort == [family]
    assert initialized["cycle"] == 5
    assert initialized["imported_from"] == str(source)
    assert pilot._update_cohort("A", 1, 6) == [family]
    assert any(e["event"] == "source_cohorts_imported" for e in pilot.events)


def test_coarse_family_keeps_exact_graphs_as_subfamilies():
    base = dict(
        k=5,
        final=dict(
            core_hash="exact-a",
            se_cn={"2": 1, "3": 4},
            cd_environments={"1,2": 2, "2,1": 6},
            n4=1,
            n6=3,
            radius_A=3.4,
            tetrahedral_q4=[0.7, 0.8],
        ),
    )
    changed = json.loads(json.dumps(base))
    changed["final"]["core_hash"] = "exact-b"
    assert lineage_family(base) != lineage_family(changed)
    assert coarse_lineage_family(base) == coarse_lineage_family(changed)

    # Continuous geometry descriptors can cross arbitrary reporting bins after
    # a numerically negligible re-relaxation.  They must not rename a lineage.
    changed["final"]["radius_A"] = 20.0
    changed["final"]["tetrahedral_q4"] = [-0.8, 0.99]
    assert coarse_lineage_family(base) == coarse_lineage_family(changed)


def test_energy_window_limits_convergence_family_novelty(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "minima.json").write_text('{"experimental": {}}')
    config = AdaptiveConfig(
        workers=1,
        max_calls=10,
        stage_calls={1: 2, 2: 2, 3: 2},
        family_slots={1: 4, 2: 1, 3: 1},
        p_max={1: 3, 2: 5, 3: 6},
        phase_a_k=[1],
        phase_b_k=2,
        phase_c_k=3,
        convergence_energy_window_eV=1.0,
    )
    pilot = AdaptivePilot(
        config,
        PACK / "run_gxtb.yaml",
        PACK / "growth_agnostic_k5.yaml",
        source,
        tmp_path / "out",
        backend=lambda *args: [],
    )
    low = dict(
        example().record(),
        minimum_id="low",
        energy_eV=-10.0,
        role="primary",
        final=descriptors(example().symbols, example().edges, example().positions),
    )
    high = json.loads(json.dumps(low))
    high["minimum_id"] = "high"
    high["energy_eV"] = -8.0
    high["final"]["n4"] += 1
    pilot.rows["experimental"] = {"low": low, "high": high}
    snapshot = pilot._snapshot([1])[1]
    assert len(snapshot["families"]) == 2
    assert len(snapshot["relevant_families"]) == 1


def test_composition_coverage_debt_counts_independent_primary_families(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "minima.json").write_text('{"experimental": {}}')
    config = AdaptiveConfig(
        workers=1,
        max_calls=10,
        stage_calls={1: 2, 2: 2, 3: 2},
        family_slots={1: 4, 2: 1, 3: 1},
        p_max={1: 3, 2: 5, 3: 6},
        phase_a_k=[1],
        phase_b_k=2,
        phase_c_k=3,
        required_primary_families_by_p={1: {1: 2, 2: 1}},
    )
    pilot = AdaptivePilot(
        config,
        PACK / "run_gxtb.yaml",
        PACK / "growth_agnostic_k5.yaml",
        source,
        tmp_path / "out",
        backend=lambda *args: [],
    )
    proposal = example()
    row = dict(
        proposal.record(),
        minimum_id="only_p1",
        energy_eV=-10.0,
        role="primary",
        final=descriptors(proposal.symbols, proposal.edges, proposal.positions),
    )
    pilot.rows["experimental"] = {"only_p1": row}
    assert pilot._composition_coverage_debt([1]) == {
        "k1:p1": 1,
        "k1:p2": 1,
    }


def test_source_novelty_reserve_is_in_initial_cohort(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "minima.json").write_text('{"experimental": {}}')
    config = AdaptiveConfig(
        workers=1,
        max_calls=10,
        stage_calls={1: 2, 2: 2, 3: 2},
        family_slots={1: 4, 2: 1, 3: 1},
        p_max={1: 3, 2: 5, 3: 6},
        phase_a_k=[1],
        phase_b_k=2,
        phase_c_k=3,
        admission_fraction=0.0,
        source_novelty_fraction=0.25,
    )
    pilot = AdaptivePilot(
        config,
        PACK / "run_gxtb.yaml",
        PACK / "growth_agnostic_k5.yaml",
        source,
        tmp_path / "out",
        backend=lambda *args: [],
    )
    pilot.setup()

    rows = {}
    for index in range(5):
        proposal = example(index)
        row = dict(
            proposal.record(),
            minimum_id=f"minimum_{index}",
            energy_eV=-20.0 + index,
            role="primary",
            final=dict(
                descriptors(proposal.symbols, proposal.edges, proposal.positions),
                se_cn={str(index + 1): 1},
            ),
        )
        rows[row["minimum_id"]] = row
    pilot.rows["experimental"] = rows
    ranked, _ = pilot._ranked_families(1)
    reserved = ranked[-1]
    pilot._source_novel_families_cache = {1: [reserved]}
    cohort = pilot._update_cohort("A", 1, 0)
    assert len(cohort) == 4
    assert reserved in cohort
    initialized, _ = pilot._cohort_state("A", 1)
    assert initialized["source_novelty_families"] == [reserved]


def test_composition_stratified_cohort_reserves_sparse_p_family(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "minima.json").write_text('{"experimental": {}}')
    config = AdaptiveConfig(
        workers=1,
        max_calls=10,
        stage_calls={1: 2, 2: 2, 3: 2},
        family_slots={1: 3, 2: 1, 3: 1},
        p_max={1: 3, 2: 5, 3: 6},
        phase_a_k=[1],
        phase_b_k=2,
        phase_c_k=3,
        admission_fraction=0.0,
        cohort_min_primary_families_by_p={1: {2: 1}},
    )
    pilot = AdaptivePilot(
        config,
        PACK / "run_gxtb.yaml",
        PACK / "growth_agnostic_k5.yaml",
        source,
        tmp_path / "out",
        backend=lambda *args: [],
    )
    pilot.setup()
    rows = {}
    for index in range(4):
        proposal = example(index)
        row = dict(
            proposal.record(),
            p=2 if index == 3 else 1,
            minimum_id=f"minimum_{index}",
            energy_eV=-20.0 + index,
            role="primary",
            final=dict(
                descriptors(proposal.symbols, proposal.edges, proposal.positions),
                se_cn={str(index + 1): 1},
            ),
        )
        rows[row["minimum_id"]] = row
    pilot.rows["experimental"] = rows
    sparse_family = pilot.adapter.family(rows["minimum_3"])
    cohort = pilot._update_cohort("A", 1, 0)
    assert len(cohort) == 3
    assert sparse_family in cohort
    initialized, _ = pilot._cohort_state("A", 1)
    assert initialized["composition_reserved_families"] == [sparse_family]


def test_adaptive_cohort_is_stable_and_only_admits_new_families(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "minima.json").write_text('{"experimental": {}}')
    config = AdaptiveConfig(
        workers=1,
        max_calls=10,
        stage_calls={1: 2, 2: 2, 3: 2},
        family_slots={1: 4, 2: 1, 3: 1},
        p_max={1: 3, 2: 5, 3: 6},
        phase_a_k=[1],
        phase_b_k=2,
        phase_c_k=3,
        admission_fraction=0.25,
    )
    pilot = AdaptivePilot(
        config,
        PACK / "run_gxtb.yaml",
        PACK / "growth_agnostic_k5.yaml",
        source,
        tmp_path / "out",
        backend=lambda *args: [],
    )
    pilot.setup()

    def row(index, radius, energy):
        proposal = example(index)
        return dict(
            proposal.record(),
            structure_id=proposal.id,
            minimum_id=f"minimum_{index}",
            energy_eV=energy,
            role="primary",
            final=dict(
                descriptors(proposal.symbols, proposal.edges, proposal.positions),
                radius_A=radius,
                se_cn={str(index + 1): 1},
                core_hash=f"exact-{index}",
            ),
            protocol="test",
            source=f"test:{index}",
            routes=[],
            occupations=[],
            occupation_origins=[],
        )

    pilot.rows["experimental"] = {
        f"minimum_{i}": row(i, 2.0 + i, -20.0 + i) for i in range(5)
    }
    initial = pilot._update_cohort("A", 1, 0)
    assert len(initial) == 3
    excluded_existing = {
        pilot.adapter.family(r) for r in pilot.rows["experimental"].values()
    } - set(initial)
    new = row(9, 20.0, -100.0)
    pilot.rows["experimental"]["minimum_9"] = new
    updated = pilot._update_cohort("A", 1, 1)
    assert updated[:3] == initial
    assert pilot.adapter.family(new) in updated
    assert excluded_existing.isdisjoint(updated)


def test_operation_budgets_are_independent(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "minima.json").write_text('{"experimental": {}}')
    config = AdaptiveConfig(
        workers=1,
        max_calls=4,
        operation_calls={1: {"fixed": 1, "growth": 2}},
        operation_calls_by_p={
            1: {"fixed": {1: 1}, "growth": {1: 1, 2: 1}}
        },
        family_slots={1: 1, 2: 1, 3: 1},
        p_max={1: 3, 2: 5, 3: 6},
        phase_a_k=[1],
        phase_b_k=2,
        phase_c_k=3,
    )
    pilot = AdaptivePilot(
        config,
        PACK / "run_gxtb.yaml",
        PACK / "growth_agnostic_k5.yaml",
        source,
        tmp_path / "out",
        backend=lambda *args: [],
    )
    fixed = replace(example(), search_operation="fixed")
    growth = replace(example(), search_operation="growth")
    assert pilot.allowed("experimental", 1, fixed)
    assert pilot.allowed("experimental", 1, growth)
    pilot.operation_p_calls_used[1, "growth", 1] = 1
    assert not pilot.allowed("experimental", 1, growth)
    assert pilot.allowed("experimental", 1, replace(growth, p=2))
    pilot.operation_p_calls_used.clear()
    pilot.operation_calls_used[1, "fixed"] = 1
    assert not pilot.allowed("experimental", 1, fixed)
    assert pilot.allowed("experimental", 1, growth)


def test_round_robin_compositions():
    proposals = [replace(example(i), p=p) for i, p in enumerate([1, 1, 1, 2, 2, 3])]
    assert [p.p for p in round_robin(proposals)] == [1, 2, 3, 1, 2, 1]


def test_default_budget_and_invalid_config(tmp_path):
    config = PilotConfig.load(PACK / "pilot.yaml")
    assert sum(config.stage_limits().values()) * 2 + config.seed_calls == 4000
    path = tmp_path / "bad.yaml"
    path.write_text("max_calls: 10000\n")
    with pytest.raises(ValueError):
        PilotConfig.load(path)


def test_integration_through_k3_without_solver(tmp_path, monkeypatch):
    def backend(payload, *args):
        return [
            XtbResult(
                ok=True,
                energy_eV=-10,
                coordinates=tuple(map(tuple, payload[0]["positions"])),
                converged=True,
            )
        ]

    pilot = make_pilot(tmp_path, backend)
    monkeypatch.setattr(pilot, "seeds", lambda: [example()])
    monkeypatch.setattr(
        pilot, "proposals", lambda arm, k, r: [example(k)] if r == 0 else []
    )

    # Geometry-independent mock auditing, while exercising real archive/budget/journal paths.
    def classify(p, xr, arm):
        return dict(
            structure_id=p.id,
            k=p.k,
            p=p.p,
            symbols=p.symbols,
            positions=p.positions,
            edges=p.edges,
            energy_eV=xr.energy_eV,
            role="primary",
            final=descriptors(p.symbols, p.edges, p.positions),
            protocol=pilot.protocol["fingerprint"],
            source=f"{arm}:{p.id}",
            routes=p.parents,
            occupations=[],
        )

    monkeypatch.setattr(pilot, "classify", classify)
    pilot.run()
    assert ("experimental", 3, 2) in pilot.stage_done
    assert ("control", 3, 0) in pilot.stage_done
    assert (pilot.output / "assessment.md").exists()
    curves = json.loads((pilot.output / "discovery_curves.json").read_text())
    assert curves and any(point["primary_minima"] for point in curves)
    assert (pilot.output / "composition_comparison.json").exists()
    assert sum(pilot.calls.values()) <= pilot.config.max_calls


def test_ligand_repair_moves_target_mu3_overlap_and_preserve_formula():
    spec_path = PACK / "run_gxtb.yaml"
    spec = load_nucleation_spec(spec_path)
    pack = load_geometry_pack(spec_path)
    parent = dict(
        k=1,
        p=2,
        minimum_id="audit_parent",
        role="audit",
        audit_derived=False,
        symbols=["Se", "Cd", "Cd", "Cd", "Cl", "Cl", "Cl", "Cl"],
        positions=[
            [0, 0, 0],
            [2.6, 0, 0],
            [-1.3, 2.25, 0],
            [-1.3, -2.25, 0],
            [0, 0, 2],
            [0.65, 1.125, 2],
            [4.9, 0, 0],
            [-3.5, -2.25, 0],
        ],
        edges=[
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 4],
            [2, 4],
            [3, 4],
            [1, 5],
            [2, 5],
            [1, 6],
            [3, 7],
        ],
    )
    before = mu3_host_bridge_overlap_violations(
        graph_state(parent["symbols"], parent["edges"], parent["positions"]), spec
    )
    repairs = local_proposals(
        parent,
        np.random.default_rng(4),
        spec,
        pack,
        channel="repair",
        limit=8,
    )
    assert before and repairs
    assert all(validate_composition(proposal) for proposal in repairs)
    assert all(proposal.audit_derived for proposal in repairs)
    assert any(
        not mu3_host_bridge_overlap_violations(
            graph_state(proposal.symbols, proposal.edges, proposal.positions), spec
        )
        for proposal in repairs
    )


def test_phase_c_completion_configuration_has_composition_call_caps():
    config = AdaptiveConfig.load(PACK / "adaptive_phase_c_completion.yaml")
    assert config.phase_cycle_start == {"C": 8}
    assert config.parent_roles_by_k[8] == ["primary", "audit"]
    assert config.operation_p_limit(8, "growth", 9) == 400
    assert config.operation_p_limit(8, "fixed", 11) == 900
    assert config.operation_p_limit(8, "fixed", 8) == 0
