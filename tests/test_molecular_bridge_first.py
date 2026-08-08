"""Skeleton bridge-first decoration smoke tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from builder.nucleation import load_nucleation_spec
from builder.nucleation.geometry_pack import load_geometry_pack
from builder.nucleation.molecular import enumerate_molecular_bin

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mol_spec():
    return load_nucleation_spec(
        ROOT / "examples/nucleation/cdse_molecular_rules.yaml"
    )


@pytest.fixture(scope="module")
def pack():
    return load_geometry_pack(
        ROOT / "geometry_packs/cdse_cdcl2_molecular.yaml"
    )


def _bridge_spec(mol_spec):
    return replace(
        mol_spec,
        graph_rules=replace(
            mol_spec.graph_rules, decoration_mode="skeleton_bridge_first"
        ),
    )


def test_bridge_first_k1_p1(mol_spec, pack) -> None:
    res = enumerate_molecular_bin(
        1, 1, _bridge_spec(mol_spec), pack=pack, embed=True
    )
    assert res.raw_graphs >= 1
    assert len(res.isomers) >= 1


def test_bridge_first_k1_p2(mol_spec, pack) -> None:
    res = enumerate_molecular_bin(
        1, 2, _bridge_spec(mol_spec), pack=pack, embed=True
    )
    assert res.raw_graphs >= 1
    assert len(res.isomers) >= 1


def test_bridge_first_graphs_do_not_depend_on_embed_k2_p2(mol_spec, pack) -> None:
    with_embed = enumerate_molecular_bin(
        2, 2, _bridge_spec(mol_spec), pack=pack, embed=True
    )
    graph_only = enumerate_molecular_bin(
        2, 2, _bridge_spec(mol_spec), pack=pack, embed=False
    )
    assert with_embed.raw_graphs == graph_only.raw_graphs
    assert with_embed.raw_graphs >= 1


def test_ring_skeleton_dump_is_graph_only_and_falls_back(mol_spec, pack) -> None:
    from builder.nucleation.molecular import (
        dump_skeletons_upfront,
        count_cdse_six_rings,
    )
    import tempfile
    from pathlib import Path
    from dataclasses import replace

    rules = replace(
        mol_spec.graph_rules,
        ring_first_when_pattern_possible=True,
        ring_min_pattern_cd=(3, 3, 4),
        ring_min_pattern_se=(3, 3, 3),
    )
    spec = replace(mol_spec, graph_rules=rules)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "rings"
        dump_skeletons_upfront(
            spec,
            out,
            pack=pack,
            kmin=3,
            kmax=3,
            pmin=0,
            pmax=1,
            max_skeletons=20,
            embed=True,
        )
        xyz = list(out.rglob("skeleton_*.xyz"))
        assert not xyz
        text = (out / "skeletons.csv").read_text()
        assert "skeleton_mode,forced_rings" in text
        # k3p1 cannot give every forced-ring Se CN3 without a Cd2Se2 C4.
        assert ",3,1," not in text or ",free," in text

    res = enumerate_molecular_bin(
        3, 3, _bridge_spec(spec), pack=pack, embed=True, max_skeletons=20
    )
    assert res.raw_graphs >= 1
    assert all(r.coordinates is None for r in res.skeleton_records)


def test_pack_chair_boat_conformations(pack) -> None:
    import numpy as np
    from builder.nucleation.molecular import _six_ring_template_positions

    assert pack.cdse6_conformations() == ("chair", "boat")
    pat = pack.cdse6_ring_pattern()
    assert pat.cd_cn == (3, 3, 4)
    assert pat.se_cn == (3, 3, 3)
    assert pat.bond_cdse_A == pytest.approx(2.69)
    assert pat.angle_at_cd_deg == pytest.approx(116.5)
    assert pat.angle_at_se_deg == pytest.approx(89.5)
    d_ch = pack.cdse6_dihedrals("chair")
    d_bo = pack.cdse6_dihedrals("boat")
    assert d_ch != d_bo
    pc = np.vstack(
        _six_ring_template_positions(
            bond_length=pat.bond_cdse_A,
            angle_at_cd_deg=pat.angle_at_cd_deg,
            angle_at_se_deg=pat.angle_at_se_deg,
            conformation="chair",
            dihedrals_deg=d_ch,
        )
    )
    pb = np.vstack(
        _six_ring_template_positions(
            bond_length=pat.bond_cdse_A,
            angle_at_cd_deg=pat.angle_at_cd_deg,
            angle_at_se_deg=pat.angle_at_se_deg,
            conformation="boat",
            dihedrals_deg=d_bo,
        )
    )
    # Chair: alternating height signs; boat: different pattern
    def heights(arr):
        c = arr.mean(0)
        _, _, vt = np.linalg.svd(arr - c)
        return (arr - c) @ vt[-1]

    hc, hb = heights(pc), heights(pb)
    chair_alt = sum(
        1 for i in range(6) if hc[i] * hc[(i + 1) % 6] < 0
    )
    assert chair_alt >= 4, ("chair should alternate", hc)
    pc0, pb0 = pc - pc.mean(0), pb - pb.mean(0)
    H = pc0.T @ pb0
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    rmsd = float(np.sqrt(((pc0 @ R.T - pb0) ** 2).sum() / 6.0))
    assert rmsd > 0.3, "chair and boat must be geometrically distinct"
    for conf, arr in (("chair", pc), ("boat", pb)):
        bonds = [
            float(np.linalg.norm(arr[(i + 1) % 6] - arr[i])) for i in range(6)
        ]
        assert min(bonds) > 2.4, (conf, bonds)
        assert all(abs(b - pat.bond_cdse_A) < 0.08 for b in bonds), (conf, bonds)


def test_dump_never_writes_early_ring_xyz(mol_spec, pack) -> None:
    from dataclasses import replace
    from builder.nucleation.molecular import dump_skeletons_upfront
    import tempfile
    from pathlib import Path

    rules = replace(
        mol_spec.graph_rules,
        ring_first_when_pattern_possible=True,
        ring_min_pattern_cd=(3, 3, 4),
        ring_min_pattern_se=(3, 3, 3),
    )
    spec = replace(mol_spec, graph_rules=rules)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "d"
        dump_skeletons_upfront(
            spec,
            out,
            pack=pack,
            kmin=3,
            kmax=3,
            pmin=1,
            pmax=1,
            max_skeletons=10,
            embed=True,
        )
        names = {p.name for p in out.rglob("skeleton_*.xyz")}
        assert not names


def test_completed_fused_ring_graph_does_not_require_early_3d(mol_spec, pack) -> None:
    rules = replace(
        mol_spec.graph_rules,
        decoration_mode="skeleton_bridge_first",
        ring_first_when_pattern_possible=True,
        multi_ring_ladder=True,
        ring_min_pattern_cd=(3, 3, 4),
        ring_min_pattern_se=(3, 3, 3),
        max_cn={**mol_spec.graph_rules.max_cn, "Se": 4},
    )
    spec = replace(mol_spec, graph_rules=rules)
    result = enumerate_molecular_bin(
        4,
        2,
        spec,
        pack=pack,
        embed=False,
        skeleton_mode="fused2",
        allow_ring_fallback=False,
        max_skeletons=20,
        max_decoration_assignments=1,
        allow_incomplete=True,
    )
    assert result.raw_graphs == 1
    assert result.skeleton_records
    assert len(result.skeleton_records[0].forced_rings) == 2
    assert result.skeleton_records[0].coordinates is None
