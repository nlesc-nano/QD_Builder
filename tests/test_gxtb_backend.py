"""g-xTB CLI backend: dispatch, parsing, and pack wiring."""
from __future__ import annotations

from pathlib import Path

import pytest

from builder.nucleation.xtb_relax import (
    XtbSettings,
    _is_gxtb,
    _is_mace,
    relax_structures,
)

PACK = Path(__file__).resolve().parents[1] / "geometry_packs" / "cdse_cdcl2_motif_GXTB.yaml"


def test_method_dispatch():
    assert _is_gxtb("g-xTB") and _is_gxtb("gxtb") and _is_gxtb("G-XTB")
    assert not _is_gxtb("GFN1-xTB")
    assert not _is_gxtb("GFN2-xTB")
    assert _is_mace("MACE-MH-1") and _is_mace("mace")
    assert not _is_mace("g-xTB")


def test_mace_settings_and_dispatch(monkeypatch):
    st = XtbSettings.from_pack(
        {
            "enabled": True,
            "backend": "mace",
            "method": "MACE-MH-1",
            "python": "/opt/mace/bin/python",
            "model": "/models/macemh1model",
            "head": "omat_pbe",
            "device": "mps",
            "dtype": "float32",
            "optimizer": "BFGS",
            "maxstep": 0.15,
        }
    )
    assert st.backend == "mace"
    assert st.model == "/models/macemh1model"
    assert st.head == "omat_pbe" and st.maxstep == 0.15

    called = {}

    def fake_mace(payload, settings, on_result=None):
        called["ids"] = [row["id"] for row in payload]
        return [
            {
                "id": "m1",
                "ok": True,
                "energy_eV": -12.5,
                "steps": 7,
                "converged": True,
                "max_force": 0.01,
                "positions": [[0, 0, 0], [0, 0, 2.6]],
                "status": "converged",
            }
        ]

    monkeypatch.setattr("builder.nucleation.xtb_relax._run_mace", fake_mace)
    result = relax_structures(
        [
            {
                "id": "m1",
                "symbols": ["Cd", "Se"],
                "positions": [[0, 0, 0], [0, 0, 2.5]],
                "edges": [[0, 1]],
            }
        ],
        st,
        {("Cd", "Se"): 3.25},
    )[0]
    assert called["ids"] == ["m1"]
    assert result.ok and result.converged and result.energy_eV == -12.5
    assert result.max_force == 0.01 and result.relaxed_edges == ((0, 1),)


def test_settings_round_trip():
    st = XtbSettings.from_pack(
        {"enabled": True, "method": "g-xTB", "binary": "gxtb",
         "xtb_path": "/opt/share/xtb", "charge": -1, "timeout_s": 60}
    )
    assert st.method == "g-xTB"
    assert st.binary.endswith("gxtb")
    assert st.xtb_path == "/opt/share/xtb"
    assert st.charge == -1
    # g-xTB default maxcycle is 100 when the pack omits max_steps
    assert st.max_steps == 100
    assert st.accept_maxcycle is True


def test_gxtb_maxcycle_default_and_override():
    st = XtbSettings.from_pack({"enabled": True, "method": "g-xTB"})
    assert st.max_steps == 100
    st2 = XtbSettings.from_pack(
        {"enabled": True, "method": "g-xTB", "max_steps": 50}
    )
    assert st2.max_steps == 50
    gfn = XtbSettings.from_pack({"enabled": True, "method": "GFN1-xTB"})
    assert gfn.max_steps == 500


def test_artifact_min_distance_from_yaml():
    st = XtbSettings.from_pack(
        {
            "enabled": True,
            "method": "g-xTB",
            "artifact_min_distance": {
                "Cd-Cd": 2.80,
                "Se-Se": 2.75,
                "Cl-Se": 2.80,
            },
        }
    )
    assert st.artifact_min_distance["Cd-Cd"] == 2.80
    assert st.artifact_min_distance["Se-Se"] == 2.75
    assert st.artifact_min_distance["Cl-Se"] == 2.80
    # missing pair is simply absent; code defaults fill at prune time
    assert "Cl-Cl" not in st.artifact_min_distance


def test_xcontrol_always_written_for_positive_max_steps(tmp_path):
    from builder.nucleation.xtb_relax import _cli_command, _write_cli_xcontrol

    st = XtbSettings.from_pack(
        {"enabled": True, "method": "g-xTB", "max_steps": 100}
    )
    _write_cli_xcontrol(tmp_path, st)
    text = (tmp_path / "xcontrol").read_text()
    assert "maxcycle=100" in text
    cmd = _cli_command("gxtb", st)
    assert "--input" in cmd and "xcontrol" in cmd


def test_xcontrol_writes_fix_block(tmp_path):
    from builder.nucleation.xtb_relax import _cli_command, _write_cli_xcontrol

    st = XtbSettings.from_pack(
        {"enabled": True, "method": "g-xTB", "max_steps": 25, "opt_level": "crude"}
    )
    _write_cli_xcontrol(tmp_path, st, freeze_atoms=range(5))
    text = (tmp_path / "xcontrol").read_text()
    assert "maxcycle=25" in text
    assert "$fix" in text
    assert "atoms: 1-5" in text
    assert "engine=rf" in text
    cmd = _cli_command("gxtb", st, opt_level="crude", need_input=True)
    assert "--opt" in cmd and "crude" in cmd


def test_parse_cli_opt_status_maxcycle():
    from builder.nucleation.xtb_relax import _parse_cli_opt_status

    conv, steps, note = _parse_cli_opt_status(
        "GEOMETRY OPTIMIZATION CONVERGED\n*** CONVERGED AFTER 42 ITERATIONS",
        100,
    )
    assert conv and steps == 42 and note == ""
    conv, steps, note = _parse_cli_opt_status(
        "FAILED TO CONVERGE IN 100 CYCLES\ncycle 100",
        100,
    )
    assert not conv and note == "maxcycle"
    assert steps >= 100


def test_pack_selects_gxtb():
    import yaml
    raw = yaml.safe_load(PACK.read_text())
    st = XtbSettings.from_pack(raw["relaxation"])
    assert _is_gxtb(st.method)
    assert st.enabled
    # GFN1's 90 s guard is far too short: 29 atoms already takes ~47 s.
    assert st.timeout_s >= 600
    assert st.max_steps == 100
    assert st.accept_maxcycle is True


def test_missing_binary_reports_error_not_crash():
    """A bad binary must degrade to an error result, not raise."""
    st = XtbSettings.from_pack(
        {"enabled": True, "method": "g-xTB",
         "binary": "/nonexistent/gxtb", "timeout_s": 10}
    )
    out = relax_structures(
        [{"id": "x", "symbols": ["Cd", "Se"],
          "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]], "edges": []}],
        st,
    )
    assert len(out) == 1 and not out[0].ok and out[0].error


def test_descriptor_model_resolution():
    from builder.nucleation.mace_descriptors import MACE_CUTOFFS
    from builder.nucleation.xtb_relax import (
        backend_provenance,
        descriptor_bond_cutoffs,
        descriptor_model,
    )

    legacy = {("Cd", "Se"): 3.25, ("Cd", "Cl"): 2.90}
    mace = XtbSettings.from_pack(
        {"backend": "mace", "method": "MACE-MH-1", "model": "/models/m", "python": "/opt/p"}
    )
    gxtb = XtbSettings.from_pack({"method": "g-xTB", "binary": "gxtb"})
    # auto: two_tier only for MACE
    assert descriptor_model(mace) == "two_tier"
    assert descriptor_model(gxtb) == "legacy"
    assert descriptor_bond_cutoffs(mace, legacy) == {**legacy, **MACE_CUTOFFS["strong"]}
    assert descriptor_bond_cutoffs(gxtb, legacy) == legacy
    # explicit override in either direction
    assert descriptor_model(XtbSettings.from_pack({"method": "g-xTB", "descriptor_cutoffs": "two_tier"})) == "two_tier"
    forced = XtbSettings.from_pack(
        {"backend": "mace", "method": "MACE-MH-1", "model": "/models/m", "descriptor_cutoffs": "legacy"}
    )
    assert descriptor_model(forced) == "legacy"
    with pytest.raises(ValueError):
        descriptor_model(XtbSettings.from_pack({"method": "g-xTB", "descriptor_cutoffs": "nope"}))
    # the choice is part of the MACE protocol identity and of the payload
    assert backend_provenance(mace)["descriptor_model"] == "two_tier"
    assert mace.payload()["descriptor_cutoffs"] == "auto"


def test_energy_plateau_settings_round_trip():
    st = XtbSettings.from_pack(
        {"backend": "mace", "method": "MACE-MH-1", "model": "/m", "python": "/p",
         "energy_tol_eV": 0.002, "energy_window": 5, "energy_fmax_gate": 0.05}
    )
    assert (st.energy_tol_eV, st.energy_window, st.energy_fmax_gate) == (0.002, 5, 0.05)
    payload = st.payload()
    assert payload["energy_tol_eV"] == 0.002 and payload["energy_fmax_gate"] == 0.05
    from builder.nucleation.xtb_relax import backend_provenance

    assert backend_provenance(st)["energy_tol_eV"] == 0.002
    assert XtbSettings.from_pack({"method": "g-xTB"}).energy_tol_eV == 0.0
