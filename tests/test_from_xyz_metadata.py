"""--from-xyz must produce a real surface-metadata JSON, same as a fresh build."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CIF = ROOT / "examples/cifs/InP_zb.cif"


def _run(args: list[str]) -> None:
    subprocess.run(
        [sys.executable, "-m", "builder", *args],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )


def test_from_xyz_writes_metadata_with_ligand_provenance(tmp_path):
    base_out = tmp_path / "base.xyz"
    _run([str(CIF), "tests/test_from_xyz_baseline.yaml", "-o", str(base_out)])
    assert (tmp_path / "base_metadata.json").exists()

    final_out = tmp_path / "final.xyz"
    _run([
        str(CIF), "tests/test_from_xyz_posttreat.yaml",
        "--from-xyz", str(base_out), "-o", str(final_out),
    ])

    metadata_path = tmp_path / "final_metadata.json"
    assert metadata_path.exists(), "--from-xyz must write a metadata JSON alongside the output xyz"
    doc = json.loads(metadata_path.read_text())

    assert doc["qd_atoms"]
    provenances = {l.get("provenance") for l in doc["ligands"]}
    assert "x_type" in provenances
    assert "l_type" in provenances

    from ase.io import read
    from builder.surface_metadata import validate_metadata_matches_xyz

    atoms = read(str(final_out), format="xyz")
    validate_metadata_matches_xyz(doc, atoms.get_chemical_symbols(), atoms.get_positions())


def test_from_xyz_prior_metadata_chains_ligand_provenance(tmp_path):
    base_out = tmp_path / "base.xyz"
    _run([str(CIF), "tests/test_from_xyz_baseline.yaml", "-o", str(base_out)])
    base_metadata = tmp_path / "base_metadata.json"
    assert base_metadata.exists()

    unchained_out = tmp_path / "unchained.xyz"
    _run([
        str(CIF), "tests/test_from_xyz_posttreat.yaml",
        "--from-xyz", str(base_out), "-o", str(unchained_out),
    ])
    unchained_doc = json.loads((tmp_path / "unchained_metadata.json").read_text())
    null_before = [l for l in unchained_doc["ligands"] if l.get("provenance") is None]
    assert null_before, "the baseline Cl ligands should be unresolved without --prior-metadata"

    chained_out = tmp_path / "chained.xyz"
    _run([
        str(CIF), "tests/test_from_xyz_posttreat.yaml",
        "--from-xyz", str(base_out),
        "--prior-xyz", str(base_out),
        "--prior-metadata", str(base_metadata),
        "-o", str(chained_out),
    ])
    chained_doc = json.loads((tmp_path / "chained_metadata.json").read_text())
    null_after = [l for l in chained_doc["ligands"] if l.get("provenance") is None]
    assert not null_after, f"expected all ligands to resolve via prior-metadata chaining, got {len(null_after)} null"
    assert not chained_doc.get("warnings")


def test_from_xyz_builder_params_merge_with_post_treatment_recipe(tmp_path):
    base_out = tmp_path / "base.xyz"
    _run([str(CIF), "tests/test_from_xyz_baseline.yaml", "-o", str(base_out)])

    base_doc = json.loads((tmp_path / "base_metadata.json").read_text())
    assert base_doc["builder"]["shape"] == "wulff"
    assert base_doc["builder"]["size_unit_cells"] == [2.0, 2.0, 2.0]
    assert base_doc["builder"]["center_ion"] == "In"
    assert "post_treatment" not in base_doc["builder"], (
        "the baseline recipe enables no post_treatment mechanisms"
    )

    original_params = tmp_path / "original_params.json"
    original_params.write_text(json.dumps({
        "radius_A": 20.0, "size_unit_cells": [2, 2, 2], "aspect": [1.0, 1.0, 1.0],
        "shape": "wulff", "center_ion": "In", "shells": [],
    }))

    final_out = tmp_path / "final.xyz"
    _run([
        str(CIF), "tests/test_from_xyz_posttreat.yaml",
        "--from-xyz", str(base_out),
        "--builder-params-json", str(original_params),
        "-o", str(final_out),
    ])
    doc = json.loads((tmp_path / "final_metadata.json").read_text())
    assert doc["builder"]["radius_A"] == 20.0
    assert doc["builder"]["center_ion"] == "In"
    assert doc["builder"]["post_treatment"]["ligand_exchange"]["enabled"] is True
    assert doc["builder"]["post_treatment"]["neutral_ligands"]["enabled"] is True
