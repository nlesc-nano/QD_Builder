"""g-xTB CLI backend: dispatch, parsing, and pack wiring."""
from __future__ import annotations

from pathlib import Path

import pytest

from builder.nucleation.xtb_relax import XtbSettings, _is_gxtb, relax_structures

PACK = Path(__file__).resolve().parents[1] / "geometry_packs" / "cdse_cdcl2_motif_GXTB.yaml"


def test_method_dispatch():
    assert _is_gxtb("g-xTB") and _is_gxtb("gxtb") and _is_gxtb("G-XTB")
    assert not _is_gxtb("GFN1-xTB")
    assert not _is_gxtb("GFN2-xTB")


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
