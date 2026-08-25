"""Tests for molecular hard graph filters (H1, H4–H7)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import networkx as nx
import pytest

from builder.nc_types import (
    CoreMonomerSpec,
    NucleationGeometryRules,
    NucleationGraphRules,
    NucleationBridgeRule,
    NucleationSpec,
    PrecursorUnitSpec,
)
from builder.nucleation.molecular_rules import (
    bridge_count_per_host_pair,
    cl_on_cn4_cd_violations,
    inorganic_component_count,
    molecular_graph_ok,
    molecular_graph_violations,
    molecular_geometry_ok,
)
from builder.nucleation.types import AtomRecord, _State
from builder.nucleation import load_nucleation_spec

ROOT = Path(__file__).resolve().parents[1]


def _spec(**kwargs) -> NucleationSpec:
    base = NucleationSpec(
        cif=str(ROOT / "examples/cifs/CdSe_zb.cif"),
        charges={"Cd": 2, "Se": -2, "Cl": -1},
        core=CoreMonomerSpec(cation="Cd", anion="Se"),
        precursor=PrecursorUnitSpec(center="Cd", ligand="Cl", ligand_count=2),
        kmax=1,
        graph_rules=NucleationGraphRules(
            min_cn={"Cd": 2, "Se": 2, "Cl": 1},
            max_cn={"Cd": 4, "Se": 5, "Cl": 3},
            allowed_bonds=(("Cd", "Cl"), ("Cd", "Se")),
            bridge_rules=(
                NucleationBridgeRule(
                    ligand="Cl",
                    host="Cd",
                    shared_neighbor="Se",
                ),
            ),
        ),
        geometry_rules=NucleationGeometryRules(by_cn={}, all_cn={}),
        require_inorganic_connected=True,
        bridges_per_cd_pair=2,
        enforce_min_cn=True,
    )
    return replace(base, **kwargs) if kwargs else base


def _state(symbols_roles_edges):
    """symbols_roles_edges: (symbols, roles, edges)."""
    symbols, roles, edges = symbols_roles_edges
    atoms = tuple(
        AtomRecord(
            atom_id=i,
            symbol=sym,
            coordinates=(float(i), 0.0, 0.0),
            role=roles[i],
        )
        for i, sym in enumerate(symbols)
    )
    g = nx.Graph()
    g.add_nodes_from(range(len(atoms)))
    g.add_edges_from(edges)
    return _State(atoms=atoms, graph=g)


def test_inorganic_connected_accepts_single_cdse_cl() -> None:
    # Cd0-Se1-Cd2 with Cl3 on Cd0 and Cl4 bridging Cd0-Cd2
    st = _state(
        (
            ["Cd", "Se", "Cd", "Cl", "Cl"],
            ["core_cation", "core_anion", "precursor_center", "precursor_ligand", "precursor_ligand"],
            [(0, 1), (1, 2), (0, 3), (0, 4), (2, 4)],
        )
    )
    spec = _spec()
    assert inorganic_component_count(st, spec) == 1
    assert molecular_graph_ok(st, spec)


def test_inorganic_disconnected_via_cl_only_rejected() -> None:
    # (Cd0-Se1) and (Cd2-Se3) linked only by Cl4 between Cd0-Cd2
    st = _state(
        (
            ["Cd", "Se", "Cd", "Se", "Cl"],
            ["core_cation", "core_anion", "core_cation", "core_anion", "precursor_ligand"],
            [(0, 1), (2, 3), (0, 4), (2, 4)],
        )
    )
    spec = _spec()
    assert inorganic_component_count(st, spec) == 2
    viol = molecular_graph_violations(st, spec)
    assert any(v.startswith("inorganic_disconnected") for v in viol)


def test_cd_cn1_rejected_when_enforce_min_cn() -> None:
    # Cd0-Se1 only: both CN1
    st = _state(
        (
            ["Cd", "Se"],
            ["core_cation", "core_anion"],
            [(0, 1)],
        )
    )
    viol = molecular_graph_violations(st, _spec(enforce_min_cn=True))
    assert any(v.startswith("min_cn:Cd") for v in viol)
    assert any(v.startswith("min_cn:Se") for v in viol)


def test_bridges_per_cd_pair_cap() -> None:
    # Two Cd linked by Se and three bridging Cl → max bridges on pair = 3
    st = _state(
        (
            ["Cd", "Se", "Cd", "Cl", "Cl", "Cl"],
            [
                "core_cation",
                "core_anion",
                "core_cation",
                "precursor_ligand",
                "precursor_ligand",
                "precursor_ligand",
            ],
            [
                (0, 1),
                (1, 2),
                (0, 3),
                (2, 3),
                (0, 4),
                (2, 4),
                (0, 5),
                (2, 5),
            ],
        )
    )
    counts = bridge_count_per_host_pair(st, _spec())
    assert counts[(0, 2)] == 3
    viol = molecular_graph_violations(st, _spec(bridges_per_cd_pair=2))
    assert any(v.startswith("bridges_per_cd_pair") for v in viol)
    assert molecular_graph_ok(st, _spec(bridges_per_cd_pair=3))


def test_yaml_loads_molecular_keywords() -> None:
    # Write temp yaml snippet via replace on existing example
    from dataclasses import replace

    spec = load_nucleation_spec(ROOT / "examples/nucleation/cdse_cdcl2.yaml")
    assert spec.require_inorganic_connected is False
    assert spec.bridges_per_cd_pair == 0
    assert spec.enforce_min_cn is False
    # defaults on NucleationSpec
    mol = replace(
        spec,
        require_inorganic_connected=True,
        bridges_per_cd_pair=2,
        enforce_min_cn=True,
    )
    assert mol.bridges_per_cd_pair == 2


def test_yaml_loads_complete_pair_rules() -> None:
    spec = load_nucleation_spec(
        ROOT / "examples/nucleation/cdse_molecular_rules.yaml"
    )
    rules = spec.graph_rules.pair_rules
    assert set(rules) == {
        "Cd-Cd", "Cd-Cl", "Cd-Se", "Cl-Cl", "Cl-Se", "Se-Se"
    }
    assert rules["Cd-Se"].bond_allowed
    assert rules["Cd-Se"].bond_max_distance == pytest.approx(3.25)
    assert not rules["Cl-Se"].bond_allowed
    assert rules["Cl-Se"].min_distance == pytest.approx(2.70)
    assert spec.graph_rules.allowed_neighbor_signatures["Cl"] == (
        "Cd1", "Cd2", "Cd3"
    )


def test_legacy_contact_min_distance_is_parsed(tmp_path) -> None:
    config = tmp_path / "legacy_contacts.yaml"
    config.write_text(
        f"""cif: {ROOT / 'examples/cifs/CdSe_zb.cif'}
charges: {{Cd: 2, Se: -2, Cl: -1}}
nucleation:
  kmax: 1
  contact_min_distance: {{Se_Cl: 3.05}}
  core_monomer: {{cation: Cd, anion: Se}}
  precursor: {{center: Cd, ligand: Cl, ligand_count: 2}}
  graph_rules:
    min_cn: {{Cd: 2, Se: 2, Cl: 1}}
    max_cn: {{Cd: 4, Se: 5, Cl: 3}}
    allowed_bonds: [[Cd, Se], [Cd, Cl]]
""",
        encoding="utf-8",
    )
    spec = load_nucleation_spec(config)
    assert spec.contact_min_distance == {"Cl-Se": pytest.approx(3.05)}


def test_pair_rule_rejects_forbidden_edge() -> None:
    spec = load_nucleation_spec(
        ROOT / "examples/nucleation/cdse_molecular_rules.yaml"
    )
    state = _state(
        (
            ["Cd", "Cd"],
            ["core_cation", "precursor_center"],
            [(0, 1)],
        )
    )
    assert any(
        reason.startswith("forbidden_edge:Cd-Cd")
        for reason in molecular_graph_violations(state, spec)
    )


def test_pair_geometry_detects_missing_edge_and_forbidden_contact() -> None:
    spec = load_nucleation_spec(
        ROOT / "examples/nucleation/cdse_molecular_rules.yaml"
    )
    state = _state(
        (
            ["Cd", "Se", "Cl"],
            ["core_cation", "core_anion", "precursor_ligand"],
            [(0, 1)],
        )
    )
    ok, reasons = molecular_geometry_ok(
        state,
        [(0.0, 0.0, 0.0), (2.60, 0.0, 0.0), (2.00, 0.0, 1.00)],
        spec,
    )
    assert not ok
    assert any(reason.startswith("contact:Cl-Se") for reason in reasons)
    assert any(reason.startswith("missing_edge:Cd-Cl") for reason in reasons)


def _cn4_core_state():
    """Interior Cd with four Se; a second Cd holds the Cl in the graph."""

    # 0 Cd (CN_Se=4), 1-4 Se, 5 Cd, 6 Cl bonded only to Cd 5
    return _state(
        (
            ["Cd", "Se", "Se", "Se", "Se", "Cd", "Cl"],
            [
                "core_cation",
                "core_anion",
                "core_anion",
                "core_anion",
                "core_anion",
                "precursor_center",
                "precursor_ligand",
            ],
            [(0, 1), (0, 2), (0, 3), (0, 4), (5, 6)],
        )
    )


def test_graph_cl_edge_on_cn4_cd_is_illegal() -> None:
    spec = _spec(enforce_min_cn=False)
    state = _state(
        (
            ["Cd", "Se", "Se", "Se", "Se", "Cl"],
            ["core_cation"] + ["core_anion"] * 4 + ["precursor_ligand"],
            [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
        )
    )
    codes = molecular_graph_violations(state, spec)
    assert any(code.startswith("cl_on_cn4_cd:0:5") for code in codes)


def test_spatial_cl_on_cn4_cd_is_illegal_without_graph_edge() -> None:
    spec = _spec(enforce_min_cn=False)
    state = _cn4_core_state()
    assert cl_on_cn4_cd_violations(state, spec) == []
    # Cl is graph-bonded to Cd 5 but sits on the interior Cd (0.9 Å).
    coords = [
        (0.0, 0.0, 0.0),
        (1.5, 1.5, 1.5),
        (1.5, -1.5, -1.5),
        (-1.5, 1.5, -1.5),
        (-1.5, -1.5, 1.5),
        (4.0, 0.0, 0.0),
        (0.9, 0.0, 0.0),
    ]
    codes = cl_on_cn4_cd_violations(state, spec, coords)
    assert any(code.startswith("cl_on_cn4_cd:0:6:") for code in codes)


def test_distant_cl_on_cn4_core_is_allowed() -> None:
    spec = _spec(enforce_min_cn=False)
    state = _cn4_core_state()
    coords = [
        (0.0, 0.0, 0.0),
        (1.5, 1.5, 1.5),
        (1.5, -1.5, -1.5),
        (-1.5, 1.5, -1.5),
        (-1.5, -1.5, 1.5),
        (4.0, 0.0, 0.0),
        (6.4, 0.0, 0.0),
    ]
    assert cl_on_cn4_cd_violations(state, spec, coords) == []


def test_geometry_pack_file_exists() -> None:
    path = ROOT / "geometry_packs/cdse_cdcl2_molecular.yaml"
    assert path.is_file()
    text = path.read_text()
    assert "schema_version: 2" in text
    assert "require_inorganic_connected" in text
    assert "max_shared_ligands_per_host_pair" in text
    assert "CdCl_bridge" in text
    assert "dihedrals:" in text
    assert "shape:" not in text
    assert "local_geometry:" not in text
