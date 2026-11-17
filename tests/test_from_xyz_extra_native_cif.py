"""--from-xyz's --extra-native-cif must keep a multi-material structure's other
materials classified as qd_atoms, not misattributed as ligands."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "builder", *args],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )


def test_extra_native_cif_keeps_core_atoms_out_of_ligands(tmp_path):
    stack_out = tmp_path / "stack.xyz"
    _run([str(ROOT / "tests/stack_znse_cdse.yaml"), "-o", str(stack_out), "--no-core-lattice-fit"])
    assert stack_out.exists()

    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "size_unit_cells: [1, 1, 1]\n"
        "facets:\n"
        "  - hkl: \"100\"\n"
        "    gamma: 1.0\n"
        "passivation:\n"
        "  ligand: Cl\n"
        "  surf_tol: 2.0\n"
        "charges:\n"
        "  Zn: +2\n"
        "  Cd: +2\n"
        "  Se: -2\n"
        "  Cl: -1\n"
        "symmetry:\n"
        "  proper_rotations_only: true\n"
        "construction_origin:\n"
        "  center_on_species: all\n"
    )
    shell_cif = ROOT / "examples/cifs/CdSe_zb.cif"
    core_cif = ROOT / "examples/cifs/ZnSe_zb.cif"

    without_out = tmp_path / "without.xyz"
    _run([str(shell_cif), str(recipe), "--from-xyz", str(stack_out), "-o", str(without_out)])
    without_doc = json.loads((tmp_path / "without_metadata.json").read_text())
    assert "Zn" not in {a["element"] for a in without_doc["qd_atoms"]}, (
        "sanity check: without --extra-native-cif, Zn core atoms should misclassify as ligands"
    )

    with_out = tmp_path / "with.xyz"
    _run([
        str(shell_cif), str(recipe), "--from-xyz", str(stack_out),
        "--extra-native-cif", str(core_cif), "-o", str(with_out),
    ])
    with_doc = json.loads((tmp_path / "with_metadata.json").read_text())
    qd_elements = {a["element"] for a in with_doc["qd_atoms"]}
    assert {"Zn", "Cd", "Se"} <= qd_elements
