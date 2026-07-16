import numpy as np
import pytest


rdkit = pytest.importorskip("rdkit")


def test_non_neutral_phosphonate_zwitterion_suggests_explicit_anion():
    from builder.neutral_exchange_posttreat import _validate_zwitterion_smiles

    with pytest.raises(ValueError) as exc_info:
        _validate_zwitterion_smiles("COP(=O)OCC[NH3+]")

    message = str(exc_info.value)
    assert "formal charge is +1" in message
    assert "Did you mean 'COP(=O)([O-])OCC[NH3+]'?" in message


def test_explicit_phosphonate_uses_charged_anchors():
    from rdkit import Chem

    from builder.neutral_exchange_posttreat import (
        _detect_zwitterion_anchors,
        _validate_zwitterion_smiles,
    )

    mol = Chem.AddHs(_validate_zwitterion_smiles("COP(=O)([O-])OCC[NH3+]"))
    cat_idx, an_idx = _detect_zwitterion_anchors(mol)

    assert mol.GetAtomWithIdx(cat_idx).GetSymbol() == "N"
    assert mol.GetAtomWithIdx(cat_idx).GetFormalCharge() == 1
    assert mol.GetAtomWithIdx(an_idx).GetSymbol() == "O"
    assert mol.GetAtomWithIdx(an_idx).GetFormalCharge() == -1


@pytest.mark.parametrize("seed", [1, 2, 3, 1337, 2026])
def test_phosphonate_methyl_branch_faces_outward(seed):
    from rdkit import Chem

    from builder.neutral_exchange_posttreat import (
        _detect_zwitterion_anchors,
        _place_zwitterion,
        _validate_zwitterion_smiles,
        _zwitterion_side_branch,
    )

    smiles = "COP(=O)([O-])OCC[NH3+]"
    mol = Chem.AddHs(_validate_zwitterion_smiles(smiles))
    cat_idx, an_idx = _detect_zwitterion_anchors(mol)
    center_idx, branch_atoms = _zwitterion_side_branch(mol, cat_idx, an_idx)

    assert center_idx is not None
    assert branch_atoms
    assert all(mol.GetAtomWithIdx(idx).GetSymbol() == "C" for idx in branch_atoms)

    _symbols, coords = _place_zwitterion(
        smiles,
        ci_pos=np.array([0.0, 0.0, 0.0]),
        ai_pos=np.array([3.0, 0.0, 0.0]),
        n_surf=np.array([1.0, 0.0, 0.0]),
        seed=seed,
    )
    branch_vec = coords[branch_atoms].mean(axis=0) - coords[int(center_idx)]
    outward_projection = float(
        np.dot(branch_vec / np.linalg.norm(branch_vec), np.array([1.0, 0.0, 0.0]))
    )

    assert outward_projection > 0.5


@pytest.mark.parametrize("seed", [1, 1337, 2026])
def test_phosphonate_methyl_branch_faces_away_from_core_at_edges(seed):
    from rdkit import Chem

    from builder.neutral_exchange_posttreat import (
        _detect_zwitterion_anchors,
        _place_zwitterion,
        _validate_zwitterion_smiles,
        _zwitterion_side_branch,
    )

    smiles = "COP(=O)([O-])OCC[NH3+]"
    mol = Chem.AddHs(_validate_zwitterion_smiles(smiles))
    cat_idx, an_idx = _detect_zwitterion_anchors(mol)
    center_idx, branch_atoms = _zwitterion_side_branch(mol, cat_idx, an_idx)
    outward = np.array([1.0, 1.0, 1.0])
    outward /= np.linalg.norm(outward)

    _symbols, coords = _place_zwitterion(
        smiles,
        ci_pos=np.array([3.0, 3.0, 3.0]),
        ai_pos=np.array([3.0, 3.0, 6.0]),
        n_surf=outward,
        seed=seed,
    )
    branch_vec = coords[branch_atoms].mean(axis=0) - coords[int(center_idx)]
    outward_projection = float(np.dot(branch_vec / np.linalg.norm(branch_vec), outward))

    assert outward_projection > 0.05
