"""Two-tier MACE/PBE descriptors: cutoff tiers, order parameters, scaffold change."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

# Load by path so the test also runs in environments without the full
# ``builder`` package dependencies (the module itself needs only numpy/networkx).
_PATH = Path(__file__).resolve().parents[1] / "src" / "builder" / "nucleation" / "mace_descriptors.py"
_spec = importlib.util.spec_from_file_location("mace_descriptors", _PATH)
md = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(md)

A = 6.08  # CdSe zinc-blende lattice constant, Å
TET = [np.array(v) for v in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]


def zb_adamantane():
    """Se-centred Cd4 tetrahedron with one extra Se on each Cd (Cd4Se5 core)."""
    bond = A * math.sqrt(3) / 4
    symbols = ["Se"]
    pos = [np.zeros(3)]
    for t in TET:
        symbols.append("Cd")
        pos.append(t / math.sqrt(3) * bond)
    # each Cd gets a second Se along another tetrahedral direction
    for i, t in enumerate(TET):
        other = TET[(i + 1) % 4]
        symbols.append("Se")
        pos.append(pos[1 + i] - other / math.sqrt(3) * bond)
    return symbols, np.array(pos)


def test_q4_ideal_and_pyramidality_limits():
    v = np.array(TET, float)
    assert md.q4(v) == pytest.approx(1.0)
    assert md.pyramidality(v[:3]) == pytest.approx(1.0)
    planar = np.array([[1, 0, 0], [-0.5, math.sqrt(3) / 2, 0], [-0.5, -math.sqrt(3) / 2, 0]])
    assert md.pyramidality(planar) == pytest.approx(0.0)


def test_two_tier_bond_classification():
    # Cd–Se at 2.60 (strong), 3.10 (long), 3.50 (none)
    symbols = ["Cd", "Se", "Se", "Se"]
    pos = np.array([[0, 0, 0], [2.60, 0, 0], [0, 3.10, 0], [0, 0, 3.50]])
    d = md.describe(symbols, pos)
    assert d["n_cdse_strong"] == 1
    assert d["n_cdse_long"] == 1
    assert d["cd_cn_se_strong"] == {"1": 1}
    assert d["cd_cn_se_all"] == {"2": 1}
    assert d["core_hash_strong"] != d["core_hash_all"]


def test_zb_fragment_scores_tetrahedral():
    symbols, pos = zb_adamantane()
    d = md.describe(symbols, pos)
    assert d["se_cn_strong"] == {"1": 4, "4": 1}
    assert d["q4_se_strong"]["mean"] == pytest.approx(1.0)
    assert d["cdsecd_tet_fraction"] == pytest.approx(1.0)
    assert d["rings4_strong"] == 0
    assert d["n_cdse_long"] == 0
    assert d["se_scaffold_components_all"] == 1
    assert d["zb_score"] > 0.9


def test_scaffold_change_labels():
    symbols, pos = zb_adamantane()
    before = md.describe(symbols, pos)
    # stretch one Cd–Se bond into the long-contact window: strong hash changes
    # by one edge, no CN3/4 lost on the central Se -> "minor"
    moved = pos.copy()
    moved[5] = pos[1] + (pos[5] - pos[1]) * (3.10 / np.linalg.norm(pos[5] - pos[1]))
    after = md.describe(symbols, moved)
    ch = md.scaffold_change(before, after)
    assert ch["label"] == "minor"
    assert ch["d_n_cdse_strong"] == -1
    assert ch["d_n_cdse_long"] == 1
    # identical geometry -> retained
    assert md.scaffold_change(before, before)["label"] == "retained"
    # remove the central Se's bonds entirely -> scaffold splits into 4 pieces
    far = pos.copy()
    far[0] = [0, 0, 20.0]
    split = md.scaffold_change(before, md.describe(symbols, far))
    assert split["label"] == "split"


def test_shed_cd_is_not_a_split():
    symbols, pos = zb_adamantane()
    symbols += ["Cd", "Cl", "Cl"]
    # a CdCl2 unit 6 Å away, bridged to nothing
    pos = np.vstack([pos, [[9, 0, 0], [9, 2.4, 0], [9, -2.4, 0]]])
    d = md.describe(symbols, pos)
    assert d["cd_detached_all"] == 1
    assert d["se_scaffold_components_all"] == 1
