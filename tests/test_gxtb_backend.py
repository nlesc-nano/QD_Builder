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


def test_pack_selects_gxtb():
    import yaml
    raw = yaml.safe_load(PACK.read_text())
    st = XtbSettings.from_pack(raw["relaxation"])
    assert _is_gxtb(st.method)
    assert st.enabled
    # GFN1's 90 s guard is far too short: 29 atoms already takes ~47 s.
    assert st.timeout_s >= 600


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
