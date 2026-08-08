"""Smoke tests for multi-lattice nucleation packs (rock-salt, wurtzite)."""

from __future__ import annotations

from pathlib import Path

import pytest

from builder.nucleation import (
    _build_lattice_model,
    _seed_state,
    generate_nucleation_result,
    load_nucleation_spec,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "yaml_name,cation,anion,ring_len,env_cn,terminal",
    [
        ("pbs_pbcl2_k1.yaml", "Pb", "S", 4, 6, "none"),
        ("pbse_pbcl2_k1.yaml", "Pb", "Se", 4, 6, "none"),
        ("cdse_wz_cdcl2_k1.yaml", "Cd", "Se", 6, 4, "none"),
        ("inp_incl3_k1.yaml", "In", "P", 6, 4, "none"),
    ],
)
def test_multilattice_packs_load_and_build_lattice(
    yaml_name: str,
    cation: str,
    anion: str,
    ring_len: int,
    env_cn: int,
    terminal: str,
) -> None:
    spec = load_nucleation_spec(ROOT / "examples/nucleation" / yaml_name)
    assert spec.core.cation == cation
    assert spec.core.anion == anion
    assert spec.inorganic_ring_length == ring_len
    assert spec.terminal_motifs == terminal
    model = _build_lattice_model(spec)
    envs = model.environments[cation]
    assert envs
    assert len(envs[0]) == env_cn
    seed = _seed_state(model)
    assert sum(1 for a in seed.atoms if a.symbol == anion) == 1
    assert sum(1 for a in seed.atoms if a.symbol == cation) == 1


def test_wurtzite_cdse_k1_map_runs() -> None:
    """Full k=1 WZ map stays cheap and retains the bare seed."""

    spec = load_nucleation_spec(
        ROOT / "examples/nucleation/cdse_wz_cdcl2_k1.yaml"
    )
    result = generate_nucleation_result(spec)
    assert 1 in result.registry
    assert 0 in result.registry[1]
    assert len(result.registry[1][0]) >= 1


def test_pbs_k1_map_runs_and_respects_cn6_capacity() -> None:
    """Rock-salt k=1 reaches higher p than ZB (anion max_cn 6)."""

    spec = load_nucleation_spec(ROOT / "examples/nucleation/pbs_pbcl2_k1.yaml")
    result = generate_nucleation_result(spec)
    assert 1 in result.registry
    p_bins = sorted(result.registry[1])
    assert 0 in p_bins
    # With max_cn(S)=6 and one Pb–S bond on the seed, free slots allow p>3.
    assert max(p_bins) >= 4
