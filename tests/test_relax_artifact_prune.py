"""Post-relax artifact prune: mid-gap floors (not construction pair_rules).

Bond-like g-xTB collapses (Se–Cl ~2.2, Se–Se ~2.3, bare Cd–Cd ~2.65) must be
pruned; ordinary non-bonded contacts and μ-bridged Cd…Cd (~2.9+) must pass.
"""

from __future__ import annotations

from builder.nucleation.molecular_rules import (
    DEFAULT_RELAX_ARTIFACT_MIN_DISTANCE,
    forbidden_pair_contact_violations,
    is_hard_relax_artifact,
    resolved_relax_artifact_floors,
)


def test_default_floors_are_mid_gap_not_construction():
    floors = DEFAULT_RELAX_ARTIFACT_MIN_DISTANCE
    # Construction pair_rules are longer (Cd–Cd 3.0, Se–Se 3.8) — must not reuse.
    assert floors["Cd-Cd"] == 2.80
    assert floors["Se-Se"] == 2.80
    assert floors["Cl-Se"] == 2.80
    assert floors["Cl-Cl"] == 2.80
    assert floors["Cd-Cd"] < 3.00
    assert floors["Se-Se"] < 3.80


def test_se_cl_bond_like_pruned_nonbond_kept():
    # Terminal Se–Cl at 2.15 (Cl not on two Cd) — still an artifact
    symbols = ["Cd", "Se", "Cl"]
    bonded = [
        (0.0, 0.0, 0.0),
        (2.5, 0.0, 0.0),
        (2.5, 2.15, 0.0),
    ]
    viol = forbidden_pair_contact_violations(symbols, bonded)
    assert any("Cl-Se" in v for v in viol)

    # Normal non-bonded Cl…Se at 3.6 (bulk starts ~3.5)
    nonbond = [
        (0.0, 0.0, 0.0),
        (2.5, 0.0, 0.0),
        (2.5, 3.6, 0.0),
    ]
    assert forbidden_pair_contact_violations(symbols, nonbond) == []


def test_rhombic_cl_se_kept_even_if_short():
    """μ2 Cl on a Cd–Se–Cd face: short Cl…Se is allowed (g-xTB overbinds)."""

    # Cd 0, Cd 1, Se 2, Cl 3 — diamond like mol0042
    symbols = ["Cd", "Cd", "Se", "Cl"]
    coords = [
        (0.0, 0.0, 0.0),
        (3.60, 0.0, 0.0),
        (1.80, 1.85, 0.0),
        (1.80, -1.70, 0.0),
    ]
    # Cl–Se ~ 3.55 in this layout — squeeze Cl toward Se to ~2.40
    coords[3] = (1.80, 1.85 - 2.40, 0.0)
    import numpy as np

    d_clse = float(np.linalg.norm(np.array(coords[3]) - np.array(coords[2])))
    d_cdcl = [
        float(np.linalg.norm(np.array(coords[3]) - np.array(coords[i])))
        for i in (0, 1)
    ]
    assert d_clse < 2.80
    assert all(x < 2.90 for x in d_cdcl)
    assert forbidden_pair_contact_violations(symbols, coords) == []


def test_cd_cd_bare_pruned_bridged_kept():
    # Bare metal-like Cd–Cd at 2.65 (dataset peak 2.64–2.78)
    symbols = ["Cd", "Cd", "Se", "Cl"]
    bare = [
        (0.0, 0.0, 0.0),
        (2.65, 0.0, 0.0),
        (0.0, 2.5, 0.0),
        (2.65, 2.5, 0.0),
    ]
    viol = forbidden_pair_contact_violations(symbols, bare)
    assert any("Cd-Cd" in v for v in viol)

    # Doubly-bridged-like Cd…Cd at 2.95 (dataset floor ~2.90 for μ-X₂)
    bridged = [
        (0.0, 0.0, 0.0),
        (2.95, 0.0, 0.0),
        (1.48, 1.8, 0.0),
        (1.48, -1.8, 0.0),
    ]
    assert forbidden_pair_contact_violations(symbols, bridged) == []


def test_se_se_bond_like_pruned():
    symbols = ["Cd", "Se", "Se"]
    # Covalent Se–Se ~2.34
    coords = [
        (0.0, 0.0, 0.0),
        (2.5, 0.0, 0.0),
        (2.5, 2.34, 0.0),
    ]
    viol = forbidden_pair_contact_violations(symbols, coords)
    assert any("Se-Se" in v for v in viol)

    # Ordinary Se…Se ~4.0
    far = [
        (0.0, 0.0, 0.0),
        (2.5, 0.0, 0.0),
        (2.5, 4.0, 0.0),
    ]
    assert forbidden_pair_contact_violations(symbols, far) == []


def test_construction_floor_would_be_wrong_but_is_not_used():
    # Cd–Cd at 3.05 is a normal non-bonded contact; construction min_distance
    # is 3.00 and would falsely prune.  Mid-gap 2.80 must keep it.
    symbols = ["Cd", "Cd"]
    coords = [(0.0, 0.0, 0.0), (3.05, 0.0, 0.0)]
    assert forbidden_pair_contact_violations(symbols, coords) == []


def test_floor_override():
    floors = resolved_relax_artifact_floors({"Cd-Cd": 2.70})
    assert floors["Cd-Cd"] == 2.70
    assert floors["Se-Se"] == 2.80
    symbols = ["Cd", "Cd"]
    # 2.75: default 2.80 would prune; override 2.70 keeps
    coords = [(0.0, 0.0, 0.0), (2.75, 0.0, 0.0)]
    assert forbidden_pair_contact_violations(
        symbols, coords, floors={"Cd-Cd": 2.70}
    ) == []
    assert forbidden_pair_contact_violations(symbols, coords)  # default prunes


def test_yaml_floors_via_xtb_settings():
    from builder.nucleation.xtb_relax import XtbSettings

    st = XtbSettings.from_pack(
        {
            "enabled": True,
            "method": "g-xTB",
            "artifact_min_distance": {"Cd-Cd": 2.70, "Se-Se": 2.80},
        }
    )
    symbols = ["Cd", "Cd"]
    coords = [(0.0, 0.0, 0.0), (2.75, 0.0, 0.0)]
    assert (
        forbidden_pair_contact_violations(
            symbols, coords, floors=st.artifact_min_distance
        )
        == []
    )


def test_hard_artifact_codes_only_artifact_prefix():
    assert is_hard_relax_artifact("artifact:Cl-Se:0-1:2.150<2.800")
    assert is_hard_relax_artifact("artifact:overlap:Cd-Se:0-1:0.400<0.750")
    # construction-style contact: must NOT rank-disqualify by itself
    assert not is_hard_relax_artifact("contact:Cd-Cd:1-2:2.950<3.000")
    assert not is_hard_relax_artifact("bond_too_long:Cd-Se:0-1:3.5>3.25")
