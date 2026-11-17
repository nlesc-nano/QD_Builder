import numpy as np
import pytest

from builder.analysis import (
    _facet_family_label,
    bulk_cn_opposite_by_interior,
    coord_numbers_bipartite,
    surface_atom_records,
)
from builder.nc_types import Facet
from builder.surface_metadata import (
    build_surface_metadata_document,
    resolve_prior_ligand_provenance,
    validate_metadata_matches_xyz,
    write_surface_metadata_json,
)

_PLANES = [
    (np.array([1.0, 0.0, 0.0]), 0.1),
    (np.array([0.0, 1.0, 0.0]), 1.1),
    (np.array([0.0, 0.0, 1.0]), 1.1),
]
_FACETS = [Facet(1, 0, 0, 1.0), Facet(1, -1, 1, 1.0), Facet(1, 1, 1, 1.0)]


def _inp_surface_fixture():
    # In0 facet(plane0), In1 edge(planes 0,1), In2 vertex(planes 0,1,2),
    # In3 deep bulk (no facet membership -> classification "internal"). P0 also
    # facet(plane0) -- an anion surface atom, proving qd_atoms is not
    # cation-only. Cl sits 2.0 A from In0, *outward* along plane 0's own
    # normal (i.e. past the surface, into the vacuum growth direction) --
    # well within its generic covalent bond cutoff of In0 and clearly closer
    # to In0 than to any other surface atom, but also genuinely within
    # surf_tol of plane 0 itself: the ligand's own position is real surface
    # geometry (facet_ids=[0]), not just a bond-distance fixture artifact,
    # since facet labels are now recomputed from the ligand's own position
    # regardless of which host a given test binds it to.
    symbols = ["In", "In", "In", "In", "P", "Cl"]
    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [-5.0, -5.0, -5.0],
        [0.05, -3.0, -3.0],
        [2.0, 0.0, 0.0],
    ])
    charges = {"In": 3, "P": -3, "Cl": -1}
    native_species = {"In", "P"}
    return symbols, pts, _PLANES, _FACETS, charges, native_species


def _cl_provenance(anchor_idx=5, host_idx=0):
    return [{
        "type": "Cl",
        "anchor_atom_index_0based": anchor_idx,
        "atom_indices_0based": [anchor_idx],
        "bound_host_atom_index_0based": host_idx,
        "provenance": "x_type",
    }]


def _synthetic_material_fixture():
    # Deliberately non-real element symbols to prove no element name (In, Cd,
    # Zn, ...) or hardcoded charge-sign assumption is baked into which atoms
    # become qd_atoms. Xx=cation, Yy=anion, Ww=present in symbols but
    # absent from `charges` entirely (-> sublattice must be None for it).
    # Same facet/edge/vertex geometry pattern as the InP fixture.
    symbols = ["Xx", "Xx", "Xx", "Xx", "Yy", "Ww"]
    pts = np.array([
        [0.0, 0.0, 0.0],       # Xx0 -> facet
        [0.0, 1.0, 0.0],       # Xx1 -> edge
        [0.0, 1.0, 1.0],       # Xx2 -> vertex
        [-5.0, -5.0, -5.0],    # Xx3 -> core
        [0.05, -3.0, -3.0],    # Yy0 -> facet (anion)
        [0.05, -6.0, -6.0],    # Ww0 -> facet (no charges entry)
    ])
    charges = {"Xx": 2, "Yy": -2}  # "Ww" intentionally absent
    native_species = {"Xx", "Yy", "Ww"}
    return symbols, pts, _PLANES, _FACETS, charges, native_species


def _write_xyz(path, symbols, pts):
    with open(path, "w") as fh:
        fh.write(f"{len(symbols)}\ntest\n")
        for s, (x, y, z) in zip(symbols, pts):
            fh.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")


def _read_xyz(path):
    with open(path) as fh:
        n = int(fh.readline())
        fh.readline()
        symbols, pts = [], []
        for _ in range(n):
            parts = fh.readline().split()
            symbols.append(parts[0])
            pts.append([float(v) for v in parts[1:4]])
    return symbols, np.array(pts)


# ---------------------------------------------------------------------
# surface_atom_records: shape, roles, HKLs, families, coordination,
# sublattice, surface/core classification
# ---------------------------------------------------------------------

def test_surface_atom_records_shape_and_xyz_order():
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    native_idx = [i for i, s in enumerate(symbols) if s in native]
    records = surface_atom_records(
        symbols, pts, planes, facets, 0.2, charges,
        true_bulk_cn={"In": 4, "P": 4, "Cl": 1},
        include_indices=native_idx,
    )

    core_keys = {
        "qd_atom_id", "atom_index_0based", "element", "position",
        "classification", "coordination_number", "bulk_coordination_number",
        "coordination_deficit", "sublattice", "provenance",
    }
    surface_keys = core_keys | {
        "role", "layer", "facet_ids", "facet_hkls", "facet_families", "surface_normal",
    }
    for row in records:
        assert set(row.keys()) == (surface_keys if row["classification"] == "surface" else core_keys)
        assert "eligible_for_ligand_exchange" not in row
        assert "current_ligand" not in row

    # Every native atom is emitted now (surface AND internal) -- In0/In1/In2
    # (facet/edge/vertex) and P0 (facet) are "surface"; In3 (deep bulk) is
    # "internal"; Cl (not native) is excluded.
    assert [row["atom_index_0based"] for row in records] == [0, 1, 2, 3, 4]
    assert {row["element"] for row in records} == {"In", "P"}
    by_index = {row["atom_index_0based"]: row for row in records}
    assert by_index[3]["classification"] == "internal"
    assert "role" not in by_index[3]
    assert "facet_ids" not in by_index[3]
    # In3 sits isolated in this fixture (no real neighbors) -- deficit is a
    # real, computed value (not null) even for a "core" atom: 4 (pinned
    # true_bulk_cn) - 0 (its actual coordination_number) = 4.
    assert by_index[3]["coordination_deficit"] == 4
    for i in (0, 1, 2, 4):
        assert by_index[i]["classification"] == "surface"
    assert all(row["provenance"] == "native" for row in records)


def test_role_classification_facet_edge_vertex():
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    native_idx = [i for i, s in enumerate(symbols) if s in native]
    records = surface_atom_records(
        symbols, pts, planes, facets, 0.2, charges,
        true_bulk_cn={"In": 4, "P": 4, "Cl": 1}, include_indices=native_idx,
    )
    by_index = {row["atom_index_0based"]: row for row in records}
    assert by_index[0]["role"] == "facet"
    assert by_index[1]["role"] == "edge"
    assert by_index[2]["role"] == "vertex"
    assert by_index[4]["role"] == "facet"  # the anion


def test_signed_hkls_preserved_not_reduced_to_family():
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    native_idx = [i for i, s in enumerate(symbols) if s in native]
    records = surface_atom_records(
        symbols, pts, planes, facets, 0.2, charges,
        true_bulk_cn={"In": 4, "P": 4, "Cl": 1}, include_indices=native_idx,
    )
    by_index = {row["atom_index_0based"]: row for row in records}
    assert by_index[1]["facet_hkls"] == [[1, 0, 0], [1, -1, 1]]
    assert by_index[2]["facet_hkls"] == [[1, 0, 0], [1, -1, 1], [1, 1, 1]]
    assert [1, -1, 1] in by_index[2]["facet_hkls"]


def test_facet_family_labels():
    assert _facet_family_label(1, 1, 1) == "{111}"
    assert _facet_family_label(1, -1, 1) == "{111}"
    assert _facet_family_label(-1, -1, -1) == "{111}"
    assert _facet_family_label(2, -1, 0) == "{210}"

    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    native_idx = [i for i, s in enumerate(symbols) if s in native]
    records = surface_atom_records(
        symbols, pts, planes, facets, 0.2, charges,
        true_bulk_cn={"In": 4, "P": 4, "Cl": 1}, include_indices=native_idx,
    )
    by_index = {row["atom_index_0based"]: row for row in records}
    assert by_index[1]["facet_families"] == ["{100}", "{111}"]
    assert by_index[2]["facet_families"] == ["{100}", "{111}", "{111}"]


def test_coordination_matches_existing_qd_builder_analysis_and_includes_ligand_bonds():
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    native_idx = [i for i, s in enumerate(symbols) if s in native]
    true_bulk_cn = {"In": 4, "P": 4, "Cl": 1}
    records = surface_atom_records(
        symbols, pts, planes, facets, 0.2, charges,
        true_bulk_cn=true_bulk_cn, include_indices=native_idx,
    )

    # Coordination is computed against the FULL array (native + ligand), so a
    # passivated atom's true (ligand-inclusive) bonding is reflected, not just
    # its lattice-only neighbor count.
    existing_cn = coord_numbers_bipartite(symbols, pts, charges)
    existing_bulk = bulk_cn_opposite_by_interior(
        symbols, pts, planes, 0.2, charges, true_bulk_cn=true_bulk_cn,
    )
    for row in records:
        i = row["atom_index_0based"]
        assert row["coordination_number"] == int(existing_cn[i])
        assert row["bulk_coordination_number"] == int(existing_bulk[symbols[i]])
        # Computed the same way regardless of classification -- a core
        # atom's deficit is a real value (e.g. 0 for a fully-coordinated
        # one), never null.
        assert row["coordination_deficit"] == max(
            0, row["bulk_coordination_number"] - row["coordination_number"]
        )


def test_sublattice_from_charges_not_periodic_table():
    symbols, pts, planes, facets, charges, native = _synthetic_material_fixture()
    native_idx = [i for i, s in enumerate(symbols) if s in native]
    records = surface_atom_records(
        symbols, pts, planes, facets, 0.2, charges, include_indices=native_idx,
    )
    by_element = {row["element"]: row["sublattice"] for row in records}
    assert by_element["Xx"] == "cation"
    assert by_element["Yy"] == "anion"
    assert by_element["Ww"] is None  # absent from `charges` -> unknown, not guessed


# ---------------------------------------------------------------------
# qd_atom_id determinism / neighbor lists / normals
# ---------------------------------------------------------------------

def test_qd_atom_ids_unique_deterministic_and_index_sorted():
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    native_idx = [i for i, s in enumerate(symbols) if s in native]
    kwargs = dict(true_bulk_cn={"In": 4, "P": 4, "Cl": 1}, include_indices=native_idx)
    first = surface_atom_records(symbols, pts, planes, facets, 0.2, charges, **kwargs)
    second = surface_atom_records(symbols, pts, planes, facets, 0.2, charges, **kwargs)

    assert first == second
    ids = [row["qd_atom_id"] for row in first]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    assert ids == list(range(len(first)))
    assert [row["atom_index_0based"] for row in first] == sorted(
        row["atom_index_0based"] for row in first
    )


def test_surface_normals_are_unit_vectors_for_surface_atoms():
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    native_idx = [i for i, s in enumerate(symbols) if s in native]
    records = surface_atom_records(
        symbols, pts, planes, facets, 0.2, charges,
        true_bulk_cn={"In": 4, "P": 4, "Cl": 1}, include_indices=native_idx,
    )
    for row in records:
        if row["classification"] == "internal":
            assert "surface_normal" not in row
            continue
        norm = float(np.linalg.norm(row["surface_normal"]))
        assert norm == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------
# build_surface_metadata_document: schema, XYZ-order consistency,
# determinism, validation, ligand integration. There is no geometric
# fallback any more -- a ligand atom with no matching `ligand_provenance`
# entry is reported with `provenance: None` and listed in `warnings`.
# ---------------------------------------------------------------------

def test_document_schema_shape_and_two_elements_present():
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    doc = build_surface_metadata_document(
        symbols, pts, planes, facets, charges, 0.2,
        native_species=native,
        material="InP",
        xyz_filename="InP_test.xyz",
        true_bulk_cn={"In": 4, "P": 4, "Cl": 1},
        ligand_provenance=_cl_provenance(),
    )
    assert doc["structure"]["atom_count"] == len(symbols)
    assert doc["structure"]["coordinate_units"] == "angstrom"
    assert doc["structure"]["material"] == "InP"
    assert set(doc.keys()) >= {"structure", "builder", "facets", "qd_atoms", "ligands", "warnings"}

    elements = {row["element"] for row in doc["qd_atoms"]}
    assert elements == {"In", "P"}  # both a cation and an anion surface element

    for row in doc["qd_atoms"]:
        assert "eligible_for_ligand_exchange" not in row
        assert "current_ligand" not in row

    assert len(doc["ligands"]) == 1
    assert doc["warnings"] == []
    lig = doc["ligands"][0]
    assert lig["type"] == "Cl"
    assert lig["provenance"] == "x_type"
    assert lig["binding_site"] == "μ1-terminal"
    assert lig["host_elements"] == ["In"]
    assert lig["host_atom_indices_0based"] == [0]  # In0 is atom_index_0based 0 in this fixture
    # facet label is unioned from the real host (In0, facet-only per fixture),
    # never computed independently for the ligand's own position.
    assert lig["facet_ids"] == [0]
    assert lig["facet_hkls"] == [[1, 0, 0]]
    assert "swap" not in lig


def test_ligand_with_no_provenance_is_reported_and_warned():
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    doc = build_surface_metadata_document(
        symbols, pts, planes, facets, charges, 0.2,
        native_species=native,
        true_bulk_cn={"In": 4, "P": 4, "Cl": 1},
    )
    assert len(doc["ligands"]) == 1
    lig = doc["ligands"][0]
    assert lig["type"] == "Cl"
    assert lig["provenance"] is None
    assert lig["binding_site"] is None
    assert lig["anchor_atom_index_0based"] is None
    assert lig["host_atom_indices_0based"] == []
    assert doc["warnings"] == [{
        "atom_index_0based": 5,
        "reason": "no_prior_metadata_supplied",
    }]


def test_synthetic_material_is_handled_generically():
    # No In/Cd/Zn/etc anywhere -- proves inclusion in qd_atoms has no
    # hardcoded element name or charge-sign filter.
    symbols, pts, planes, facets, charges, native = _synthetic_material_fixture()
    doc = build_surface_metadata_document(
        symbols, pts, planes, facets, charges, 0.2, native_species=native,
    )
    elements = {row["element"] for row in doc["qd_atoms"]}
    assert elements == {"Xx", "Yy", "Ww"}


def test_document_atom_indices_match_written_xyz_order(tmp_path):
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    doc = build_surface_metadata_document(
        symbols, pts, planes, facets, charges, 0.2,
        native_species=native,
        xyz_filename="fixture.xyz",
        true_bulk_cn={"In": 4, "P": 4, "Cl": 1},
        ligand_provenance=_cl_provenance(),
    )

    xyz_path = tmp_path / "fixture.xyz"
    _write_xyz(xyz_path, symbols, pts)
    xyz_symbols, xyz_pts = _read_xyz(xyz_path)

    assert xyz_symbols == symbols
    validate_metadata_matches_xyz(doc, xyz_symbols, xyz_pts)
    for row in doc["qd_atoms"]:
        idx = row["atom_index_0based"]
        assert xyz_symbols[idx] == row["element"]
        assert np.allclose(xyz_pts[idx], row["position"])
    for lig in doc["ligands"]:
        assert xyz_symbols[lig["anchor_atom_index_0based"]] == lig["anchor_element"]
        for i in lig["atom_indices_0based"]:
            assert 0 <= i < len(xyz_symbols)
        for host_idx, host_el in zip(lig["host_atom_indices_0based"], lig["host_elements"]):
            assert xyz_symbols[host_idx] == host_el


def test_document_is_deterministic_across_calls():
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    kwargs = dict(
        native_species=native, material="InP", xyz_filename="fixture.xyz",
        true_bulk_cn={"In": 4, "P": 4, "Cl": 1},
        ligand_provenance=_cl_provenance(),
    )
    first = build_surface_metadata_document(symbols, pts, planes, facets, charges, 0.2, **kwargs)
    second = build_surface_metadata_document(symbols, pts, planes, facets, charges, 0.2, **kwargs)
    assert first == second


def test_validate_metadata_matches_xyz_raises_on_corrupted_document():
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    doc = build_surface_metadata_document(
        symbols, pts, planes, facets, charges, 0.2,
        native_species=native, true_bulk_cn={"In": 4, "P": 4, "Cl": 1},
        ligand_provenance=_cl_provenance(),
    )

    with pytest.raises(ValueError):
        validate_metadata_matches_xyz(doc, symbols[:-1])  # wrong atom count

    corrupted = dict(doc)
    corrupted["qd_atoms"] = [dict(doc["qd_atoms"][0])]
    corrupted["qd_atoms"][0]["element"] = "Zz"  # doesn't match XYZ symbol at that index
    with pytest.raises(ValueError):
        validate_metadata_matches_xyz(corrupted, symbols)

    corrupted2 = dict(doc)
    corrupted2["ligands"] = [dict(doc["ligands"][0])]
    assert corrupted2["ligands"][0]["host_atom_indices_0based"], "fixture ligand must have a resolved host to corrupt"
    corrupted2["ligands"][0]["host_atom_indices_0based"] = [9999]
    with pytest.raises(ValueError):
        validate_metadata_matches_xyz(corrupted2, symbols)

    # Position checks (only run when xyz_pts is given): a document whose
    # positions have gone stale relative to the XYZ it's meant to describe
    # (the write-ordering class of bug -- a later step rewrote the XYZ
    # without regenerating the metadata) isn't caught by index/element
    # checks alone.
    validate_metadata_matches_xyz(doc, symbols, pts)  # real data: must pass
    bad_pts = np.asarray(pts, float).copy()
    bad_pts[doc["qd_atoms"][0]["atom_index_0based"]] += 1.0
    with pytest.raises(ValueError):
        validate_metadata_matches_xyz(doc, symbols, bad_pts)

    bad_pts2 = np.asarray(pts, float).copy()
    some_ligand = next(l for l in doc["ligands"] if l["host_atom_indices_0based"])
    bad_pts2[some_ligand["host_atom_indices_0based"][0]] += 1.0
    with pytest.raises(ValueError):
        validate_metadata_matches_xyz(doc, symbols, bad_pts2)


def test_write_surface_metadata_json_roundtrip(tmp_path):
    import json

    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    doc = build_surface_metadata_document(
        symbols, pts, planes, facets, charges, 0.2,
        native_species=native,
        xyz_filename="fixture.xyz",
        true_bulk_cn={"In": 4, "P": 4, "Cl": 1},
        ligand_provenance=_cl_provenance(),
    )
    out_path = tmp_path / "fixture_metadata.json"
    write_surface_metadata_json(str(out_path), doc)

    with open(out_path) as fh:
        reloaded = json.load(fh)
    assert reloaded == doc


# ---------------------------------------------------------------------
# Real build-time ligand provenance -- every mechanism threads its own
# `provenance` mechanism name straight through (see
# builder.main._run_passivation_and_write_outputs /
# backend.api_builder._run_repassivation_posttreatment).
# ---------------------------------------------------------------------

def test_ligand_provenance_is_used_directly():
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    provenance = [{
        "type": "Cl",
        "anchor_atom_index_0based": 5,
        "atom_indices_0based": [5],
        "bound_host_atom_index_0based": 2,
        "provenance": "x_type",
    }]
    doc = build_surface_metadata_document(
        symbols, pts, planes, facets, charges, 0.2,
        native_species=native,
        true_bulk_cn={"In": 4, "P": 4, "Cl": 1},
        ligand_provenance=provenance,
    )
    assert len(doc["ligands"]) == 1
    lig = doc["ligands"][0]
    assert lig["provenance"] == "x_type"
    assert lig["anchor_element"] == "Cl"
    assert lig["anchor_atom_index_0based"] == 5
    assert lig["host_atom_indices_0based"] == [2]  # In2 is atom_index_0based 2 in this fixture
    assert lig["host_elements"] == ["In"]
    assert lig["binding_site"] == "μ1-terminal"
    # Recomputed from the ligand's own position (facet 0 only), independent
    # of which host (In2, the vertex) is actually claimed.
    assert lig["facet_ids"] == [0]
    assert "swap" not in lig


def test_ligand_provenance_reports_bridging_to_multiple_real_hosts():
    # Real provenance can name more than one host (bridging/multidentate
    # binding) -- both must come through, closest-to-anchor first, not
    # collapsed to a single primary host.
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    provenance = [{
        "type": "Cl",
        "anchor_atom_index_0based": 5,
        "atom_indices_0based": [5],
        "bound_host_atom_indices_0based": [2, 0],  # In2 first (deliberately out of distance order)
        "provenance": "l_type",
    }]
    doc = build_surface_metadata_document(
        symbols, pts, planes, facets, charges, 0.2,
        native_species=native,
        true_bulk_cn={"In": 4, "P": 4, "Cl": 1},
        ligand_provenance=provenance,
    )
    lig = doc["ligands"][0]
    assert lig["provenance"] == "l_type"
    assert lig["binding_site"] == "μ2-bridging"
    assert set(lig["host_atom_indices_0based"]) == {0, 2}  # In0/In2 are atom_index_0based 0/2 in this fixture
    assert lig["host_elements"] == ["In", "In"]
    # Recomputed from the ligand's own position (facet 0 only), independent
    # of which two hosts are actually claimed.
    assert lig["facet_ids"] == [0]
    # Sorted closest-to-anchor first: Cl (index 5, at [2,0,0]) is nearer to
    # In0 ([0,0,0], dist 2.0) than to In2 ([0,1,1], dist ~2.45).
    assert lig["host_atom_indices_0based"][0] == 0
    assert lig["anchor_host_distances"] == sorted(lig["anchor_host_distances"])


def test_ligand_provenance_partial_coverage_warns_for_the_rest():
    # Two independent ligand fragments: only the first (monatomic) has real
    # provenance; the second (multi-atom) has none at all -- it must still
    # be reported (symbols/positions intact), with `provenance: None` and a
    # `warnings` entry per atom, not a geometric guess.
    symbols = ["Mm", "Mm", "Zz", "Nn", "Hh"]
    pts = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [2.2, 0.0, 0.0],   # Zz: provenance-covered, bound to host 0
        [10.0, 2.0, 0.0],  # Nn: not covered -> no provenance, warned
        [10.0, 2.0, 1.0],
    ])
    planes = [(np.array([1.0, 0.0, 0.0]), 0.1), (np.array([-1.0, 0.0, 0.0]), 10.1)]
    facets = [Facet(1, 0, 0, 1.0), Facet(-1, 0, 0, 1.0)]
    charges = {"Mm": 2, "Zz": -1, "Nn": -3, "Hh": 1}
    native = {"Mm"}

    provenance = [{
        "type": "Zz",
        "anchor_atom_index_0based": 2,
        "atom_indices_0based": [2],
        "bound_host_atom_index_0based": 0,
        "provenance": "x_type",
    }]
    doc = build_surface_metadata_document(
        symbols, pts, planes, facets, charges, 0.2,
        native_species=native, ligand_provenance=provenance,
    )
    # The uncovered fragment (Nn, Hh) has no provenance captured for either
    # atom, so each is reported as its own unresolved ligand record.
    assert len(doc["ligands"]) == 3
    by_provenance = {}
    for lig in doc["ligands"]:
        by_provenance.setdefault(lig["provenance"], []).append(lig)
    assert by_provenance["x_type"][0]["anchor_element"] == "Zz"
    unresolved = by_provenance[None]
    assert len(unresolved) == 2
    for rec in unresolved:
        assert rec["anchor_element"] is None
        assert rec["anchor_atom_index_0based"] is None
        assert rec["host_elements"] == []
        assert rec["binding_site"] is None
    assert {i for rec in unresolved for i in rec["atom_indices_0based"]} == {3, 4}
    assert {w["atom_index_0based"] for w in doc["warnings"]} == {3, 4}
    assert {w["reason"] for w in doc["warnings"]} == {"no_prior_metadata_supplied"}


def test_ligand_provenance_with_unresolvable_host_degrades_gracefully():
    # Provenance references a real anchor/atom_indices but a host index that
    # doesn't resolve to any known qd_atom -- anchor/atom indices must still
    # be reported correctly; only the host cross-reference goes empty.
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    provenance = [{
        "type": "Cl",
        "anchor_atom_index_0based": 5,
        "atom_indices_0based": [5],
        "bound_host_atom_index_0based": 9999,  # no such atom
        "provenance": "x_type",
    }]
    doc = build_surface_metadata_document(
        symbols, pts, planes, facets, charges, 0.2,
        native_species=native,
        true_bulk_cn={"In": 4, "P": 4, "Cl": 1},
        ligand_provenance=provenance,
    )
    lig = doc["ligands"][0]
    assert lig["provenance"] == "x_type"
    assert lig["anchor_atom_index_0based"] == 5
    assert lig["host_atom_indices_0based"] == []
    assert lig["host_elements"] == []
    assert lig["binding_site"] is None
    validate_metadata_matches_xyz(doc, symbols)


def test_malformed_ledger_entry_gets_its_own_warning_reason():
    # A real-mechanism ledger entry that references this atom but fails
    # internal validation (missing anchor) is a genuine bug signal --
    # distinct from "no provenance was ever supplied for this atom at all".
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    provenance = [{
        "type": "Cl",
        "anchor_atom_index_0based": None,  # malformed: no anchor claimed
        "atom_indices_0based": [5],
        "bound_host_atom_index_0based": 0,
        "provenance": "x_type",
    }]
    doc = build_surface_metadata_document(
        symbols, pts, planes, facets, charges, 0.2,
        native_species=native,
        true_bulk_cn={"In": 4, "P": 4, "Cl": 1},
        ligand_provenance=provenance,
    )
    assert doc["warnings"] == [{"atom_index_0based": 5, "reason": "malformed_ledger_entry"}]


def test_unresolved_ligand_reason_reflects_prior_metadata_supplied():
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    doc = build_surface_metadata_document(
        symbols, pts, planes, facets, charges, 0.2,
        native_species=native,
        true_bulk_cn={"In": 4, "P": 4, "Cl": 1},
        prior_metadata_supplied=True,
    )
    assert doc["warnings"] == [{"atom_index_0based": 5, "reason": "prior_position_match_failed"}]


def test_chained_lattice_replacement_gets_uniform_fields_and_swap_tag():
    # A lattice-replacement ligand chained forward from a prior document
    # (see resolve_prior_ligand_provenance, which carries `swap` through
    # directly from the prior document's own record -- the provenance
    # string alone no longer implies it, since several real mechanisms now
    # share "charge_balance") arrives through the generic ligand_provenance
    # path, not _build_lattice_substitution_records -- it must still get
    # the exact same uniform field set as a freshly-resolved one:
    # binding_site from real host count, facet label recomputed from the
    # ligand's own position, and the `swap` tag.
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    provenance = [{
        "type": "Cl",
        "anchor_atom_index_0based": 5,
        "atom_indices_0based": [5],
        "bound_host_atom_indices_0based": [0, 2],  # In0 (facet 0 only) + In2 (vertex, all three)
        "provenance": "polar_surface_reconstruction",
        "swap": True,
    }]
    doc = build_surface_metadata_document(
        symbols, pts, planes, facets, charges, 0.2,
        native_species=native,
        true_bulk_cn={"In": 4, "P": 4, "Cl": 1},
        ligand_provenance=provenance,
    )
    lig = doc["ligands"][0]
    assert lig["provenance"] == "polar_surface_reconstruction"
    assert lig["binding_site"] == "μ2-bridging"
    # Recomputed from the ligand's own position (facet 0 only), independent
    # of which two hosts are actually claimed.
    assert lig["facet_ids"] == [0]
    assert lig["swap"] is True


# ---------------------------------------------------------------------
# _ordered_xyz_view(..., return_order=True) (main.py)
# ---------------------------------------------------------------------

def test_ordered_xyz_view_return_order_is_a_valid_permutation():
    from builder.main import _ordered_xyz_view

    symbols = ["In", "Cl", "P", "In", "Cl"]
    pts = np.arange(15, dtype=float).reshape(5, 3)
    charges = {"In": 3, "P": -3, "Cl": -1}

    out_syms_2tuple, out_pts_2tuple = _ordered_xyz_view(symbols, pts, charges, "Cl")
    out_syms, out_pts, order = _ordered_xyz_view(symbols, pts, charges, "Cl", return_order=True)

    assert out_syms == out_syms_2tuple
    assert np.array_equal(out_pts, out_pts_2tuple)
    assert sorted(order) == list(range(len(symbols)))
    assert out_syms == [symbols[i] for i in order]
    assert np.array_equal(out_pts, pts[order])


# ---------------------------------------------------------------------
# lattice_substitution_provenance (a native anion converted in place into
# the ligand species -- by polar facet reconstruction, see
# builder.facet_reconstruction.reconstruct_polar_facets, or by the baseline
# prepass, see builder.passivation.prepass_surface_cleanup).
# ---------------------------------------------------------------------

def _lattice_substitution_fixture():
    # In0 -> facet (plane0). In1 -> edge (planes 0,1). P0 -> facet (plane0),
    # native anion, uninvolved. Cl0 sits at the midpoint of In0/In1, within
    # generic covalent cutoff of both -- a lattice-substitution anchor
    # bridging two real hosts. Cl1/In2 are a second, independent
    # lattice-substitution pair -- having >=2 real pairs lets
    # _detect_global_offset's majority vote correctly identify the true
    # offset (zero or otherwise) rather than trivially matching a single
    # entry to whatever lone candidate happens to exist.
    symbols = ["In", "In", "P", "Cl", "In", "Cl"]
    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.05, -3.0, -3.0],
        [0.0, 0.5, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -0.5, 0.0],
    ])
    charges = {"In": 3, "P": -3, "Cl": -1}
    native_species = {"In", "P"}
    return symbols, pts, charges, native_species


def _entry(pts, anchor_i, host_is, offset=np.zeros(3), replaced_element=None,
           mechanism="polar_surface_reconstruction", host_symbols=None,
           z_type_relocated=None, neutrality_redistributed=None):
    entry = {
        "position": (pts[anchor_i] - offset).tolist(),
        "symbol": "Cl",
        "host_positions": [(pts[i] - offset).tolist() for i in host_is],
        "host_symbols": host_symbols if host_symbols is not None else ["In"] * len(host_is),
        "mechanism": mechanism,
    }
    if replaced_element is not None:
        entry["replaced_element"] = replaced_element
    if z_type_relocated is not None:
        entry["z_type_relocated"] = z_type_relocated
    if neutrality_redistributed is not None:
        entry["neutrality_redistributed"] = neutrality_redistributed
    return entry


def test_lattice_substitution_resolves_by_position_with_global_offset():
    symbols, pts, charges, native = _lattice_substitution_fixture()
    # Simulate a uniform recentering translation (e.g. center_coords) applied
    # between when the ledger entries were captured and this call's `pts`.
    # Two real pairs (not one) so _detect_global_offset's majority vote has
    # enough data to identify the true offset rather than trivially matching
    # a lone entry to whatever lone same-symbol candidate happens to exist.
    offset = np.array([10.0, -5.0, 3.0])
    entries = [
        # Cl0 bridges In0 (facet 0) + In1 (edge, facets 0,1)
        _entry(pts, 3, [0, 1], offset, replaced_element="Se"),
        _entry(pts, 5, [4], offset, replaced_element="Se"),  # Cl1 bonded to In4 (facet 0 only)
    ]
    doc = build_surface_metadata_document(
        symbols, pts, _PLANES, _FACETS, charges, 0.2,
        native_species=native, lattice_substitution_provenance=entries,
    )
    lattice_ligs = {l["anchor_atom_index_0based"]: l for l in doc["ligands"] if l.get("provenance") == "polar_surface_reconstruction"}
    assert set(lattice_ligs) == {3, 5}
    cl0 = lattice_ligs[3]
    assert cl0["host_elements"] == ["In", "In"]
    assert set(cl0["host_atom_indices_0based"]) == {0, 1}  # In0/In1 are atom_index_0based 0/1 in this fixture
    # Same uniform field set as any other ligand record now -- binding_site
    # from real host count, facet label recomputed from the ligand's own
    # position (facet 0 only -- Cl0 at [0, 0.5, 0] doesn't itself satisfy
    # plane 1 or 2, regardless of host In1 being on an edge), no
    # role/coordination context merged in.
    assert set(cl0.keys()) == set(doc["ligands"][1].keys())
    assert cl0["binding_site"] == "μ2-bridging"
    assert cl0["facet_ids"] == [0]
    assert cl0["facet_hkls"] == [[1, 0, 0]]
    assert cl0["swap"] is True
    assert cl0["replaced_element"] == "Se"
    assert "role" not in cl0
    assert "coordination_number" not in cl0
    cl1 = lattice_ligs[5]
    assert cl1["binding_site"] == "μ1-terminal"
    assert cl1["facet_ids"] == [0]
    # Never emitted as a parallel qd_atoms entry.
    assert {3, 5}.isdisjoint({r["atom_index_0based"] for r in doc["qd_atoms"]})
    assert {r["element"] for r in doc["qd_atoms"]} == {"In", "P"}
    assert doc["warnings"] == []
    validate_metadata_matches_xyz(doc, symbols)


def test_ligand_records_never_carry_facet_terminations():
    # Termination (cation_rich/anion_rich/stoichiometric) is a property of
    # which native species occupies a lattice site under a given Wulff seed
    # configuration -- it doesn't apply to a ligand sitting in open space
    # beyond the lattice, so ligand records never carry a
    # facet_terminations key at all, even when facet_seed_terminations is
    # supplied (it still governs the top-level facets[] list's own
    # termination field, unaffected by this).
    symbols, pts, charges, native = _lattice_substitution_fixture()
    entries = [_entry(pts, 3, [0, 1])]
    doc = build_surface_metadata_document(
        symbols, pts, _PLANES, _FACETS, charges, 0.2,
        native_species=native, lattice_substitution_provenance=entries,
        facet_seed_terminations={(1, 0, 0): "cation_rich", (1, -1, 1): "stoichiometric"},
    )
    lig = next(l for l in doc["ligands"] if l["anchor_atom_index_0based"] == 3)
    assert "facet_terminations" not in lig
    assert doc["facets"][0]["termination"] == "cation_rich"


def test_lattice_substitution_both_mechanisms_get_uniform_fields():
    # The baseline charge-balancing machinery's always-on CN==2 anion-swap
    # step (prepass_surface_cleanup, tagged "charge_balance") feeds the same
    # lattice_substitution_provenance list as reconstruct_polar_facets's
    # optional reconstruction -- both are real in-place native-anion-to-
    # ligand conversions, so both get the exact same field set (binding_site
    # from host count, facet label recomputed from the ligand's own
    # position, swap=True); only `provenance` tells them apart.
    symbols, pts, charges, native = _lattice_substitution_fixture()
    entries = [
        _entry(pts, 3, [0, 1], mechanism="polar_surface_reconstruction", replaced_element="Se"),
        _entry(pts, 5, [4], mechanism="charge_balance", replaced_element="Se"),
    ]
    doc = build_surface_metadata_document(
        symbols, pts, _PLANES, _FACETS, charges, 0.2,
        native_species=native, lattice_substitution_provenance=entries,
    )
    by_anchor = {l["anchor_atom_index_0based"]: l for l in doc["ligands"]}
    assert by_anchor[3]["provenance"] == "polar_surface_reconstruction"
    assert by_anchor[3]["binding_site"] == "μ2-bridging"
    assert by_anchor[3]["facet_ids"] == [0]
    assert by_anchor[3]["swap"] is True
    assert by_anchor[5]["provenance"] == "charge_balance"
    assert by_anchor[5]["binding_site"] == "μ1-terminal"
    assert by_anchor[5]["facet_ids"] == [0]
    assert by_anchor[5]["swap"] is True
    assert set(by_anchor[3].keys()) == set(by_anchor[5].keys())


def test_charge_balance_addition_entry_gets_no_swap():
    # reconstruct_polar_facets's step 4 (fresh compensating ligands on
    # cation-facet sites) uses the same select_ligand_sites placement as the
    # baseline charge-balance add-path, so it's tagged "charge_balance" too
    # (never "polar_surface_reconstruction", which is reserved for the real
    # in-place anion->ligand swaps, steps 1-3) -- no replaced_element, no swap.
    symbols, pts, charges, native = _lattice_substitution_fixture()
    entries = [_entry(pts, 3, [0, 1], mechanism="charge_balance")]
    doc = build_surface_metadata_document(
        symbols, pts, _PLANES, _FACETS, charges, 0.2,
        native_species=native, lattice_substitution_provenance=entries,
    )
    lig = next(l for l in doc["ligands"] if l["anchor_atom_index_0based"] == 3)
    assert lig["provenance"] == "charge_balance"
    assert "swap" not in lig
    assert "replaced_element" not in lig


def test_ligand_facet_label_recomputed_from_own_position_not_host():
    # The ligand's facet label is recomputed geometrically from its own
    # position by the same standard procedure as qd_atoms, never inherited
    # from its host(s). Here the anchor's own position geometrically
    # satisfies all three of this fixture's planes at once, even though its
    # only real host (In0) is facet-only (one facet) -- the record must
    # report the anchor's own, broader real membership, not a narrower
    # label copied from the host.
    symbols = ["In", "P", "Cl"]
    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.05, -3.0, -3.0],
        [0.0, 0.95, 0.95],  # satisfies planes 0, 1, and 2 within surf_tol=0.2
    ])
    charges = {"In": 3, "P": -3, "Cl": -1}
    native = {"In", "P"}
    entries = [_entry(pts, 2, [0])]
    doc = build_surface_metadata_document(
        symbols, pts, _PLANES, _FACETS, charges, 0.2,
        native_species=native, lattice_substitution_provenance=entries,
    )
    lig = next(l for l in doc["ligands"] if l["anchor_atom_index_0based"] == 2)
    assert lig["provenance"] == "polar_surface_reconstruction"
    assert lig["facet_ids"] == [0, 1, 2]
    assert lig["facet_hkls"] == [[1, 0, 0], [1, -1, 1], [1, 1, 1]]
    assert lig["facet_families"] == ["{100}", "{111}", "{111}"]
    assert "facet_terminations" not in lig


def test_lattice_substitution_unresolved_host_degrades_gracefully():
    symbols, pts, charges, native = _lattice_substitution_fixture()
    entries = [
        _entry(pts, 3, [0, 1]),  # fully valid -- anchors the offset vote at zero
        dict(_entry(pts, 5, [4]), host_positions=[[500.0, 500.0, 500.0]]),
    ]
    doc = build_surface_metadata_document(
        symbols, pts, _PLANES, _FACETS, charges, 0.2,
        native_species=native, lattice_substitution_provenance=entries,
    )
    lattice_ligs = {l["anchor_atom_index_0based"]: l for l in doc["ligands"] if l.get("provenance") == "polar_surface_reconstruction"}
    assert set(lattice_ligs) == {3, 5}
    # Anchor still resolves; the one bad host position is just dropped.
    assert lattice_ligs[5]["host_elements"] == []
    assert lattice_ligs[5]["host_atom_indices_0based"] == []


def test_lattice_substitution_unresolved_anchor_is_warned_not_guessed():
    symbols, pts, charges, native = _lattice_substitution_fixture()
    entries = [
        _entry(pts, 3, [0, 1]),  # fully valid -- anchors the offset vote at zero
        dict(_entry(pts, 5, [4]), position=[500.0, 500.0, 500.0]),  # no such atom
    ]
    doc = build_surface_metadata_document(
        symbols, pts, _PLANES, _FACETS, charges, 0.2,
        native_species=native, lattice_substitution_provenance=entries,
    )
    # The corrupted entry is dropped entirely -- Cl1 (atom 5) still gets a
    # ligand record (type/position intact), but with no provenance at all --
    # there is no geometric fallback to guess it from.
    by_anchor = {l["anchor_atom_index_0based"]: l for l in doc["ligands"] if l["anchor_atom_index_0based"] is not None}
    assert by_anchor[3]["provenance"] == "polar_surface_reconstruction"
    unresolved = [l for l in doc["ligands"] if l["anchor_atom_index_0based"] is None]
    assert len(unresolved) == 1
    assert unresolved[0]["atom_indices_0based"] == [5]
    assert unresolved[0]["provenance"] is None
    assert {w["atom_index_0based"] for w in doc["warnings"]} == {5}
    assert {w["reason"] for w in doc["warnings"]} == {"no_prior_metadata_supplied"}


def test_lattice_substitution_mixed_host_symbols_resolve_independently():
    # Regression for the real-world bug: an anion bonded to two DIFFERENT
    # cation species (e.g. one native, one alloy-replaced) must resolve
    # both hosts independently -- collapsing to one shared `host_symbol`
    # (Counter.most_common) used to silently drop whichever species didn't
    # "win" the count, even though the atom is still there.
    symbols = ["In", "Ga", "P", "Cl"]
    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.05, -3.0, -3.0],
        [0.0, 0.5, 0.0],
    ])
    charges = {"In": 3, "Ga": 3, "P": -3, "Cl": -1}
    native = {"In", "Ga", "P"}
    entries = [_entry(pts, 3, [0, 1], host_symbols=["In", "Ga"])]
    doc = build_surface_metadata_document(
        symbols, pts, _PLANES, _FACETS, charges, 0.2,
        native_species=native, lattice_substitution_provenance=entries,
    )
    lig = next(l for l in doc["ligands"] if l["anchor_atom_index_0based"] == 3)
    assert lig["host_atom_indices_0based"] == [0, 1]
    assert lig["host_elements"] == ["In", "Ga"]


def test_lattice_substitution_mixed_host_symbols_surviving_host_not_dropped():
    # The Bug A+B cascade: one of the two hosts captured at ledger-write
    # time is genuinely gone by the time this call runs (removed by a
    # later charge-balancing step, e.g.) -- the surviving host, of a
    # DIFFERENT species than the removed one, must still resolve. A single
    # shared `host_symbol` would have failed this (checked the surviving
    # host against the removed host's species) and reported zero hosts.
    symbols = ["In", "P", "Cl"]  # Ga1 from the original capture is gone
    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.05, -3.0, -3.0],
        [0.0, 0.5, 0.0],
    ])
    charges = {"In": 3, "Ga": 3, "P": -3, "Cl": -1}
    native = {"In", "Ga", "P"}
    removed_ga_pos = [0.0, 1.0, 0.0]
    entry = {
        "position": pts[2].tolist(),
        "symbol": "Cl",
        "host_positions": [pts[0].tolist(), removed_ga_pos],
        "host_symbols": ["In", "Ga"],
        "mechanism": "charge_balance",
    }
    doc = build_surface_metadata_document(
        symbols, pts, _PLANES, _FACETS, charges, 0.2,
        native_species=native, lattice_substitution_provenance=[entry],
    )
    lig = next(l for l in doc["ligands"] if l["anchor_atom_index_0based"] == 2)
    assert lig["host_atom_indices_0based"] == [0]
    assert lig["host_elements"] == ["In"]


def test_z_type_relocated_keeps_real_provenance_and_drops_swap():
    # z-type's orphan-ligand relocation feeds lattice_substitution_provenance
    # too (see z_type_displacement_posttreat._relocate_orphan_ligands) -- it
    # no longer uses its own "z_type_relocated" mechanism string; instead it
    # keeps the ligand's real original `mechanism` (here "charge_balance",
    # a swap, so `replaced_element` carries through and `swap` still shows)
    # and layers `z_type_relocated: True` on top, so the full history (what
    # it really was, plus the fact that it moved) survives. A ligand moved
    # by a *fresh addition* (no `replaced_element`) must not get `swap: True`
    # even though it's still a real, known-origin relocation.
    symbols, pts, charges, native = _lattice_substitution_fixture()
    entries = [
        _entry(pts, 3, [0, 1], mechanism="charge_balance", replaced_element="Se", z_type_relocated=True),
        _entry(pts, 5, [4], mechanism="polar_surface_reconstruction", z_type_relocated=True),
    ]
    doc = build_surface_metadata_document(
        symbols, pts, _PLANES, _FACETS, charges, 0.2,
        native_species=native, lattice_substitution_provenance=entries,
    )
    by_anchor = {l["anchor_atom_index_0based"]: l for l in doc["ligands"]}
    assert by_anchor[3]["provenance"] == "charge_balance"
    assert by_anchor[3]["swap"] is True
    assert by_anchor[3]["z_type_relocated"] is True
    assert by_anchor[5]["provenance"] == "polar_surface_reconstruction"
    assert "swap" not in by_anchor[5]
    assert by_anchor[5]["z_type_relocated"] is True


def test_z_type_relocated_unrecoverable_origin_stays_null_not_guessed():
    # When z-type's orphan relocation can't recover the ligand's real prior
    # mechanism at all (see _lookup_original_provenance returning None),
    # `provenance` must stay null rather than defaulting to a guess --
    # `z_type_relocated: True` alone still flags it as "moved, origin
    # unknown" instead of being indistinguishable from any other
    # unexplained null.
    symbols, pts, charges, native = _lattice_substitution_fixture()
    entries = [_entry(pts, 3, [0, 1], mechanism=None, z_type_relocated=True)]
    doc = build_surface_metadata_document(
        symbols, pts, _PLANES, _FACETS, charges, 0.2,
        native_species=native, lattice_substitution_provenance=entries,
    )
    lig = next(l for l in doc["ligands"] if l["anchor_atom_index_0based"] == 3)
    assert lig["provenance"] is None
    assert lig["z_type_relocated"] is True
    assert "swap" not in lig


def test_neutrality_redistributed_flag_propagates():
    # A ligand moved by post-passivation redistribution (Cause B) is not a
    # swap -- it already was the ligand, it just relocated to resolve
    # undercoordination -- so it gets `neutrality_redistributed: True`
    # instead of `swap`.
    symbols, pts, charges, native = _lattice_substitution_fixture()
    entries = [_entry(pts, 3, [0, 1], mechanism="charge_balance", neutrality_redistributed=True)]
    doc = build_surface_metadata_document(
        symbols, pts, _PLANES, _FACETS, charges, 0.2,
        native_species=native, lattice_substitution_provenance=entries,
    )
    lig = next(l for l in doc["ligands"] if l["anchor_atom_index_0based"] == 3)
    assert lig["provenance"] == "charge_balance"
    assert lig["neutrality_redistributed"] is True
    assert "swap" not in lig


def test_charge_balance_ligand_resolves_by_position_surviving_index_shift():
    # Regression: charge_balance-placed ligands (baseline halide
    # placeholder / post-alloying rebalance) can run *before* z-type in the
    # real pipeline, unlike x_type/l_type/z_type_neutral_exchange which only
    # ever run after it. z-type's own atom removal shifts/invalidates any
    # raw index captured earlier with no remap of its own -- so
    # charge_balance_ledger entries are resolved by position
    # (lattice_substitution_provenance), same as prepass_cleanup, not
    # through the raw atom_indices_0based/anchor_atom_index_0based path.
    # Simulate exactly that: the entry's raw (now-stale) atom_indices point
    # nowhere near reality, but its real captured position still matches,
    # and the ligand must not get `swap: True` (it's a freely-placed
    # exterior ligand, not a converted native lattice site).
    symbols, pts, charges, native = _lattice_substitution_fixture()
    entries = [
        {
            "atom_indices": [9999],  # stale, shifted-away-by-z-type raw index
            "anchor_local_idx": 0,
            "host_env_indices": [9999],
            "position": pts[3].tolist(),
            "symbol": "Cl",
            "host_positions": [pts[0].tolist(), pts[1].tolist()],
            "host_symbols": ["In", "In"],
            "mechanism": "charge_balance",
        },
    ]
    doc = build_surface_metadata_document(
        symbols, pts, _PLANES, _FACETS, charges, 0.2,
        native_species=native, lattice_substitution_provenance=entries,
    )
    lig = next(l for l in doc["ligands"] if l["anchor_atom_index_0based"] == 3)
    assert lig["provenance"] == "charge_balance"
    assert lig["host_atom_indices_0based"] == [0, 1]
    assert "swap" not in lig


def test_binding_site_mu_notation_thresholds():
    # mu-notation binding mode is a pure function of real host count, the
    # same for every ligand regardless of which mechanism placed it.
    symbols = ["In", "In", "In", "In", "Cl", "Cl", "Cl", "Cl"]
    pts = np.array([
        [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [15.0, 0.0, 0.0],
        [0.0, 5.0, 0.0], [0.0, 10.0, 0.0], [0.0, 15.0, 0.0], [0.0, 20.0, 0.0],
    ])
    charges = {"In": 3, "Cl": -1}
    native = {"In"}
    provenance = [
        {"type": "Cl", "anchor_atom_index_0based": 4, "atom_indices_0based": [4],
         "bound_host_atom_indices_0based": [0], "provenance": "x_type"},
        {"type": "Cl", "anchor_atom_index_0based": 5, "atom_indices_0based": [5],
         "bound_host_atom_indices_0based": [0, 1], "provenance": "x_type"},
        {"type": "Cl", "anchor_atom_index_0based": 6, "atom_indices_0based": [6],
         "bound_host_atom_indices_0based": [0, 1, 2], "provenance": "x_type"},
        {"type": "Cl", "anchor_atom_index_0based": 7, "atom_indices_0based": [7],
         "bound_host_atom_indices_0based": [0, 1, 2, 3], "provenance": "x_type"},
    ]
    doc = build_surface_metadata_document(
        symbols, pts, [], [], charges, 0.2,
        native_species=native, ligand_provenance=provenance,
    )
    by_anchor = {l["anchor_atom_index_0based"]: l for l in doc["ligands"]}
    assert by_anchor[4]["binding_site"] == "μ1-terminal"
    assert by_anchor[5]["binding_site"] == "μ2-bridging"
    assert by_anchor[6]["binding_site"] == "μ3-face-capping"
    assert by_anchor[7]["binding_site"] == "μ4-tetrahedro-capping"


# ---------------------------------------------------------------------
# resolve_prior_ligand_provenance -- chaining a prior surface-metadata
# document's `ligands` forward by position (QDSpaceWebApp repassivation:
# an atom already present on upload, with no ledger of its own for it).
# ---------------------------------------------------------------------

def test_resolve_prior_ligand_provenance_survives_reorder_and_translation():
    prior_symbols = ["In", "In", "Cl"]
    prior_pts = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    prior_ligands = [{
        "type": "Cl", "anchor_atom_index_0based": 2, "atom_indices_0based": [2],
        "host_atom_indices_0based": [0], "host_elements": ["In"],
        "provenance": "x_type",
    }]

    # Current structure: same atoms, globally translated, reordered, plus a
    # brand-new atom this repassivation call itself placed.
    offset = np.array([10.0, 10.0, 10.0])
    symbols = ["In", "Cl", "In", "N"]
    pts = np.array([
        [5.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [50.0, 50.0, 50.0],
    ]) + offset

    out = resolve_prior_ligand_provenance(
        symbols, pts, prior_symbols, prior_pts, prior_ligands,
    )
    # Output is in the ligand_provenance *input* shape build_surface_metadata_document
    # expects (bound_host_atom_indices_0based, real-XYZ-index space) -- not
    # the final-JSON host_atom_indices_0based/host_elements shape it was read from.
    assert out == [{
        "type": "Cl",
        "anchor_atom_index_0based": 1,
        "atom_indices_0based": [1],
        "bound_host_atom_indices_0based": [2],
        "provenance": "x_type",
    }]


def test_resolve_prior_ligand_provenance_drops_removed_atoms():
    prior_symbols = ["In", "In", "Cl"]
    prior_pts = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    prior_ligands = [{
        "type": "Cl", "anchor_atom_index_0based": 2, "atom_indices_0based": [2],
        "host_atom_indices_0based": [0], "host_elements": ["In"],
        "provenance": "x_type",
    }]
    offset = np.array([10.0, 10.0, 10.0])
    # The Cl ligand itself was removed by this repassivation call (e.g.
    # exchanged away) -- it must not be reported at all, not with a guessed
    # position.
    symbols = ["In", "In"]
    pts = np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) + offset
    assert resolve_prior_ligand_provenance(
        symbols, pts, prior_symbols, prior_pts, prior_ligands,
    ) == []


def test_resolve_prior_ligand_provenance_keeps_host_alloying_replaced_in_place():
    # Regression for the real-world bug: a chained-forward ligand's host was
    # recorded as "In" in the prior document (built before alloying was ever
    # enabled on this structure); THIS call's own alloying pass then
    # replaces that exact atom with "Ga" in place. The bond is still real --
    # without `alloying_provenance`, an exact-symbol check would silently
    # drop it (atom "moved" from the chaining logic's point of view); with
    # it, the host resolves to its new real species instead of vanishing.
    prior_symbols = ["In", "In", "Cl"]
    prior_pts = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    prior_ligands = [{
        "type": "Cl", "anchor_atom_index_0based": 2, "atom_indices_0based": [2],
        "host_atom_indices_0based": [0, 1], "host_elements": ["In", "In"],
        "provenance": "prepass_cleanup",
    }]
    offset = np.array([10.0, 10.0, 10.0])
    # host 0 (prior "In" at [0,0,0]) has been alloy-replaced to "Ga" in place;
    # host 1 (prior "In" at [5,0,0]) is untouched.
    symbols = ["Ga", "In", "Cl"]
    pts = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.0, 0.0, 0.0]]) + offset
    alloying_provenance = [
        {"position": (pts[0]).tolist(), "symbol": "Ga", "provenance": "alloying"},
    ]

    without_fix = resolve_prior_ligand_provenance(
        symbols, pts, prior_symbols, prior_pts, prior_ligands,
    )
    assert without_fix[0]["bound_host_atom_indices_0based"] == [1]

    with_fix = resolve_prior_ligand_provenance(
        symbols, pts, prior_symbols, prior_pts, prior_ligands,
        alloying_provenance=alloying_provenance,
    )
    assert with_fix[0]["bound_host_atom_indices_0based"] == [0, 1]


def test_resolve_prior_ligand_provenance_does_not_rechain_an_already_null_anchor():
    # Regression for a real-world bug: a prior document can itself already
    # carry an unresolved ligand placeholder (anchor_atom_index_0based:
    # None, e.g. from an even earlier chaining gap) alongside its real
    # atom_indices_0based. Blindly re-chaining that forward as a "resolved"
    # entry with anchor_atom_index_0based: None makes
    # _assign_ligands_from_provenance flag it as *malformed* (a genuine-bug
    # signal) on every subsequent repassivation, forever -- it must instead
    # be dropped here, same as if no prior entry existed for it at all.
    prior_symbols = ["In", "Cl"]
    prior_pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    prior_ligands = [{
        "type": "Cl", "anchor_atom_index_0based": None, "atom_indices_0based": [1],
        "host_atom_indices_0based": [], "host_elements": [],
        "provenance": None,
    }]
    offset = np.array([10.0, 10.0, 10.0])
    symbols = ["In", "Cl"]
    pts = prior_pts + offset
    assert resolve_prior_ligand_provenance(
        symbols, pts, prior_symbols, prior_pts, prior_ligands,
    ) == []


def test_resolve_prior_ligand_provenance_feeds_build_surface_metadata_document():
    # End-to-end: a prior ligand chained forward resolves exactly like a
    # freshly-captured one once handed to build_surface_metadata_document.
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    prior_symbols = ["In", "Cl"]
    prior_pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    prior_ligands = [{
        "type": "Cl", "anchor_atom_index_0based": 1, "atom_indices_0based": [1],
        "host_atom_indices_0based": [0], "host_elements": ["In"],
        "provenance": "x_type",
    }]
    chained = resolve_prior_ligand_provenance(
        symbols, pts, prior_symbols, prior_pts, prior_ligands,
    )
    doc = build_surface_metadata_document(
        symbols, pts, planes, facets, charges, 0.2,
        native_species=native,
        true_bulk_cn={"In": 4, "P": 4, "Cl": 1},
        ligand_provenance=chained,
        prior_metadata_supplied=True,
    )
    assert doc["warnings"] == []
    lig = doc["ligands"][0]
    assert lig["provenance"] == "x_type"
    assert lig["host_elements"] == ["In"]


# ---------------------------------------------------------------------
# prepass_mode="role-aware": role computed with the real build's own
# geometric classifier (passivation._role_by_geometry), not the plain
# facet-membership-count heuristic.
# ---------------------------------------------------------------------

def test_role_aware_mode_uses_real_geometric_classifier():
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    native_idx = [i for i, s in enumerate(symbols) if s in native]
    kwargs = dict(true_bulk_cn={"In": 4, "P": 4, "Cl": 1}, include_indices=native_idx)

    standard = surface_atom_records(symbols, pts, planes, facets, 0.2, charges, prepass_mode="standard", **kwargs)
    role_aware = surface_atom_records(symbols, pts, planes, facets, 0.2, charges, prepass_mode="role-aware", **kwargs)

    std_by_idx = {r["atom_index_0based"]: r.get("role") for r in standard}
    geo_by_idx = {r["atom_index_0based"]: r.get("role") for r in role_aware}
    # Same atoms classified either way, valid labels, core atoms unaffected.
    assert set(std_by_idx) == set(geo_by_idx)
    assert all(v in (None, "facet", "edge", "vertex") for v in geo_by_idx.values())
    assert std_by_idx[3] is None and geo_by_idx[3] is None  # In3: core either way (no "role" key)


def test_role_aware_mode_is_opt_in_default_unaffected():
    symbols, pts, planes, facets, charges, native = _inp_surface_fixture()
    native_idx = [i for i, s in enumerate(symbols) if s in native]
    kwargs = dict(true_bulk_cn={"In": 4, "P": 4, "Cl": 1}, include_indices=native_idx)
    default = surface_atom_records(symbols, pts, planes, facets, 0.2, charges, **kwargs)
    standard = surface_atom_records(symbols, pts, planes, facets, 0.2, charges, prepass_mode="standard", **kwargs)
    assert default == standard
