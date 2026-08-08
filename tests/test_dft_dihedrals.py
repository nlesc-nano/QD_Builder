from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from dft_geometry_mine.bonds import BondCutoffs, analyze_frame  # noqa: E402
from dft_geometry_mine.angles import collect_angles  # noqa: E402
from dft_geometry_mine.dihedrals import (  # noqa: E402
    collect_improper_dihedrals,
)
from dft_geometry_mine.xyz_io import Frame  # noqa: E402


def test_cd_cn3_improper_reports_planarity_deviation() -> None:
    planar = Frame(
        symbols=("Cd", "Se", "Se", "Cl"),
        coordinates=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-0.5, 0.8660254, 0.0),
            (-0.5, -0.8660254, 0.0),
        ),
    )
    graph = analyze_frame(planar, BondCutoffs())
    samples = [
        sample
        for sample in collect_improper_dihedrals(planar, graph)
        if sample.element == "Cd"
    ]
    assert len(samples) == 1
    assert samples[0].neighbor_signature == "Cl1Se2"
    assert samples[0].improper_deg == pytest.approx(0.0, abs=1.0e-10)


def test_angle_miner_distinguishes_terminal_and_shared_bridge_cl() -> None:
    frame = Frame(
        symbols=("Cd", "Se", "Cl", "Cl", "Cd"),
        coordinates=(
            (0.0, 0.0, 0.0),
            (2.5, 0.0, 0.0),
            (-1.2, -2.0, 0.0),
            (0.0, 2.5, 0.0),
            (2.5, 2.5, 0.0),
        ),
    )
    graph = analyze_frame(frame, BondCutoffs())
    samples = [
        sample
        for sample in collect_angles(frame, graph)
        if sample.element == "Cd"
        and sample.cn == 3
        and sample.neighbor_pair == "Cl-Cl"
    ]
    assert len(samples) == 1
    assert samples[0].role_signature == "Cl_b2s+Cl_t+Se"
    assert samples[0].neighbor_role_pair == "Cl_b2s-Cl_t"
