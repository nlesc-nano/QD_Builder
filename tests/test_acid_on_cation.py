"""A carboxylic acid as an L-type ligand on cations binds through its carbonyl O (the C=O away from the cation, the
O-H turned toward a nearby O when one is in reach); on anions it keeps the acidic-H anchor."""
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECIPE = """size_unit_cells: [1.5, 1.5, 1.5]
facets:
  - {{hkl: "100", scope: family, termination: cation_rich, gamma: 1.0}}
  - {{hkl: "111", scope: family, termination: cation_rich, gamma: 1.0}}
  - {{hkl: "-1-1-1", scope: family, termination: anion_rich, gamma: 1.0}}
charges: {{Cd: 2, Se: -2, Cl: -1}}
passivation: {{ligand: Cl, surf_tol: 2.0}}
post_treatment:
  neutral_ligands:
    enabled: true
    seed: 1337
    passes:
      - {{target: {target}, smiles: "CC(=O)O", distribution: uniform, ratio: 1.0}}
"""


def _read_xyz(path):
    lines = Path(path).read_text().strip().splitlines()
    n = int(lines[0])
    rows = [l.split() for l in lines[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])


def _build(tmp_path, target):
    (tmp_path / "r.yaml").write_text(RECIPE.format(target=target))
    out = tmp_path / f"{target}.xyz"
    subprocess.run([sys.executable, "-m", "builder", str(ROOT / "examples/cifs/CdSe_zb.cif"), str(tmp_path / "r.yaml"),
                    "-o", str(out)], cwd=ROOT, check=True, capture_output=True, text=True)
    return _read_xyz(out)


def _acids(syms, pts):
    """(carbonyl O, hydroxyl O, hydroxyl H) coordinates of every acid, from the bonds (a full build reorders atoms)."""
    syms = np.asarray(syms)
    o_idx, h_idx = np.where(syms == "O")[0], np.where(syms == "H")[0]
    acids = []
    for c in np.where(syms == "C")[0]:
        oo = [o for o in o_idx if np.linalg.norm(pts[o] - pts[c]) < 1.45]
        if len(oo) != 2:
            continue
        hb = {o: [h for h in h_idx if np.linalg.norm(pts[h] - pts[o]) < 1.15] for o in oo}
        oh = [o for o in oo if hb[o]]
        if len(oh) != 1:
            continue
        oc = [o for o in oo if o != oh[0]][0]
        acids.append((pts[oc], pts[oh[0]], pts[hb[oh[0]][0]]))
    return acids


def test_acid_binds_cations_by_its_carbonyl(tmp_path):
    syms, pts = _build(tmp_path, "cation")
    acids = _acids(syms, pts)
    assert acids
    cd = pts[[i for i, s in enumerate(syms) if s == "Cd"]]
    for oc, oh, hh in acids:
        d = lambda p: np.linalg.norm(cd - p, axis=1).min()          # noqa: E731
        assert d(oc) < d(oh) and d(oc) < d(hh)                      # the carbonyl O is the one on the Cd
        assert d(oc) < 3.4


def test_acid_on_anions_keeps_the_proton_anchor(tmp_path):
    syms, pts = _build(tmp_path, "anion")
    acids = _acids(syms, pts)
    assert acids
    se = pts[[i for i, s in enumerate(syms) if s == "Se"]]
    for oc, oh, hh in acids:
        assert np.linalg.norm(se - hh, axis=1).min() < np.linalg.norm(se - oc, axis=1).min()
