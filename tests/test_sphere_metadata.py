"""Sphere-mode builds must get a real surface-metadata JSON, same as faceted builds."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_builder(cif: Path, yaml_path: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "builder", str(cif), str(yaml_path), "-o", str(out)]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)


def test_sphere_build_writes_metadata_with_placeholder_facets(tmp_path):
    cif = ROOT / "examples/cifs/InP_zb.cif"
    yaml_path = ROOT / "tests/test_sphere_metadata.yaml"
    out = tmp_path / "sphere.xyz"
    _run_builder(cif, yaml_path, out)

    metadata_path = tmp_path / "sphere_metadata.json"
    assert metadata_path.exists(), "sphere-mode build must still write a metadata JSON"
    doc = json.loads(metadata_path.read_text())

    assert len(doc["facets"]) == 24
    assert all(f["hkl"] == [0, 0, 0] for f in doc["facets"])
    assert all(f["family_label"] == "{000}" for f in doc["facets"])

    assert doc["qd_atoms"], "qd_atoms must be populated for a sphere build"
    surface_atoms = [a for a in doc["qd_atoms"] if a["classification"] == "surface"]
    assert surface_atoms, "at least some atoms must classify as surface"
    for atom in surface_atoms:
        assert atom["facet_ids"]
        assert all(hkl == [0, 0, 0] for hkl in atom["facet_hkls"])
        assert atom["coordination_number"] is not None

    null_ligands = [l for l in doc["ligands"] if l.get("provenance") is None]
    assert not null_ligands, f"expected no null-provenance ligands, got {len(null_ligands)}"

    assert doc["builder"]["shape"] == "sphere"
    assert doc["builder"]["size_unit_cells"] == [2.0, 2.0, 2.0]
    assert doc["builder"]["center_ion"] == "all"


def test_sphere_build_role_aware_prepass_does_not_fail(tmp_path):
    """Metadata assembly's own facet-frame (edge/vertex) geometry must handle the
    many-plane sphere set, not just passivation's own role-aware prepass."""
    cif = ROOT / "examples/cifs/InP_zb.cif"
    yaml_path = ROOT / "tests/test_sphere_metadata_role_aware.yaml"
    out = tmp_path / "sphere_ra.xyz"
    _run_builder(cif, yaml_path, out)

    metadata_path = tmp_path / "sphere_ra_metadata.json"
    assert metadata_path.exists()
    doc = json.loads(metadata_path.read_text())
    assert doc["qd_atoms"]
