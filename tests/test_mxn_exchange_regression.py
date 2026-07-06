import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


BASE_YAML = """\
cif: examples/cifs/CdSe_zb.cif
size_unit_cells: [1.25, 1.25, 1.25]
facets:
  - hkl: "100"
    scope: family
    termination: cation_rich
    gamma: 1.0
  - hkl: "111"
    scope: family
    termination: cation_rich
    gamma: 1.0
  - hkl: "-1-1-1"
    scope: family
    termination: anion_rich
    gamma: 1.0

passivation:
  ligand: Cl
  surf_tol: 2.0
  prepass_mode: role-aware
  prepass_min_cn_terrace: 2
  prepass_min_cn_edge: 2
  prepass_min_cn_vertex: 1

charges:
  Cd: +2
  Se: -2
  Cl: -1
  Br: -1
  In: +3
  Zn: +2
  Cs: +1
  C: 0
  N: 0
  O: -1
  S: -1
  H: 0

symmetry:
  proper_rotations_only: true

construction_origin:
  center_on_species: Se

experimental:
  exhausted_positive_q_fallback: true
"""


CHARGES = {
    "Cd": 2,
    "Se": -2,
    "Cl": -1,
    "Br": -1,
    "In": 3,
    "Zn": 2,
    "Cs": 1,
    "C": 0,
    "N": 0,
    "O": -1,
    "S": -1,
    "H": 0,
}


def _write_yaml(path: Path, smiles: str | None = None, exchange_type: str = "mxn"):
    text = BASE_YAML
    if smiles is not None:
        text += f"""
post_treatment:
  neutral_exchange:
    enabled: true
    seed: 1337
    passes:
      - cation: Cd
        anion: Cl
        anion_count: 2
        target_count: 1
        exchange_type: {exchange_type}
        smiles: "{smiles}"
"""
    path.write_text(text)


def _read_xyz_counts(path: Path) -> Counter:
    lines = path.read_text().splitlines()
    n = int(lines[0])
    return Counter(line.split()[0] for line in lines[2:2 + n])


def _formal_charge(counts: Counter) -> int:
    return int(sum(CHARGES.get(sym, 0) * count for sym, count in counts.items()))


def _run_builder(yaml_path: Path, out_path: Path):
    cmd = [
        sys.executable,
        "-m",
        "builder",
        str(ROOT / "examples/cifs/CdSe_zb.cif"),
        str(yaml_path),
        "-o",
        str(out_path),
    ]
    result = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout


def test_cdcl2_mxn_formula_exchange_counts(tmp_path):
    base_yaml = tmp_path / "cdse_base.yaml"
    base_out = tmp_path / "cdse_base.xyz"
    _write_yaml(base_yaml)
    _run_builder(base_yaml, base_out)
    base = _read_xyz_counts(base_out)

    cases = [
        ("InBr3", "In", 3),
        ("ZnBr2", "Zn", 2),
        ("CsBr", "Cs", 1),
    ]
    for smiles, cation, expected_br in cases:
        yaml_path = tmp_path / f"cdse_{cation}.yaml"
        out_path = tmp_path / f"cdse_{cation}.xyz"
        _write_yaml(yaml_path, smiles)
        _run_builder(yaml_path, out_path)
        counts = _read_xyz_counts(out_path)

        assert counts["Cd"] == base["Cd"] - 1
        assert counts[cation] == 1
        assert counts["Cl"] == base["Cl"] - 2
        assert counts["Br"] == expected_br
        assert _formal_charge(counts) == _formal_charge(base)


def test_cdcl2_mxn_ionic_smiles_acetate_exchange(tmp_path):
    try:
        import rdkit  # noqa: F401
    except Exception:
        return

    base_yaml = tmp_path / "cdse_base.yaml"
    base_out = tmp_path / "cdse_base.xyz"
    _write_yaml(base_yaml)
    _run_builder(base_yaml, base_out)
    base = _read_xyz_counts(base_out)

    for idx, smiles in enumerate([
        "[Zn+2].CC(=O)[O-].CC(=O)[O-]",
        "[Zn2+].[CC(=O)O-].[CC(=O)O-]",
    ]):
        yaml_path = tmp_path / f"cdse_zn_acetate_{idx}.yaml"
        out_path = tmp_path / f"cdse_zn_acetate_{idx}.xyz"
        _write_yaml(yaml_path, smiles)
        _run_builder(yaml_path, out_path)
        counts = _read_xyz_counts(out_path)

        assert counts["Cd"] == base["Cd"] - 1
        assert counts["Zn"] == 1
        assert counts["Cl"] == base["Cl"] - 2
        assert counts["C"] >= 4
        assert counts["O"] >= 4


def test_cdcl2_zwitterion_and_l_type_exchange_have_bound_groups(tmp_path):
    try:
        import rdkit  # noqa: F401
    except Exception:
        return

    base_yaml = tmp_path / "cdse_base.yaml"
    base_out = tmp_path / "cdse_base.xyz"
    _write_yaml(base_yaml)
    _run_builder(base_yaml, base_out)
    base = _read_xyz_counts(base_out)

    cases = [
        ("zwitterion", "[NH3+]CC[S-]", "N"),
        ("l_type", "CCCCC(=O)O", "C"),
    ]
    for exchange_type, smiles, marker in cases:
        yaml_path = tmp_path / f"cdse_{exchange_type}.yaml"
        out_path = tmp_path / f"cdse_{exchange_type}.xyz"
        _write_yaml(yaml_path, smiles, exchange_type=exchange_type)
        stdout = _run_builder(yaml_path, out_path)
        counts = _read_xyz_counts(out_path)

        assert "Candidates: 0 groups" not in stdout
        assert "Charge compensation:" not in stdout
        assert counts["Cd"] == base["Cd"] - 1
        assert counts["Cl"] == base["Cl"] - 2
        assert counts[marker] > base[marker]


def test_cdcl2_availability_completes_groups_from_removable_cl_pool():
    from builder.neutral_exchange_posttreat import _select_groups
    from builder.z_type_displacement_posttreat import _count_bound_groups

    syms = ["Cd", "Cd", "Cd", "Cl", "Cl", "Cl", "Cl", "Cl", "Cl"]
    pts = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [14.0, 0.0, 0.0],
            [21.0, 0.0, 0.0],
            [24.0, 0.0, 0.0],
        ],
        float,
    )
    cation_indices = [0, 1, 2]
    anion_indices = [3, 4, 5, 6, 7, 8]

    assert _count_bound_groups(
        syms,
        pts,
        cation_indices,
        anion_indices,
        2,
        None,
        allow_unbound_completion=False,
    ) == 0
    assert _count_bound_groups(
        syms,
        pts,
        cation_indices,
        anion_indices,
        2,
        None,
        allow_unbound_completion=True,
    ) == 3

    groups = _select_groups(
        syms,
        pts,
        cation_indices,
        anion_indices,
        2,
        1.0,
        0,
        "uniform",
        11,
        6.0,
    )
    assert len(groups) == 3
    assert all(len(ais) == 2 for _ci, ais in groups)


def test_neutral_ligand_pass_accepts_target_symbol(tmp_path):
    from builder.config import parse_yaml_config

    yaml_path = tmp_path / "target_symbol.yaml"
    yaml_path.write_text(BASE_YAML + """
post_treatment:
  neutral_ligands:
    enabled: true
    passes:
      - target: cation
        target_symbol: Cd
        smiles: "CN"
        target_count: 1
        distribution: uniform
""")
    cfg = parse_yaml_config(str(yaml_path))
    spec = cfg.post_treatment.neutral_ligands

    assert spec.enabled
    assert spec.passes[0].target == "cation"
    assert spec.passes[0].target_symbol == "Cd"
