"""Formation / grand-potential reporting (report-only thermo)."""

from __future__ import annotations

from pathlib import Path

import pytest

from builder.nucleation.formation import (
    HARTREE_EV,
    MonomerReferences,
    load_monomer_references,
)

ROOT = Path(__file__).resolve().parents[1]
GROWTH = ROOT / "geometry_packs" / "cdse_cdcl2" / "growth.yaml"


def test_load_growth_yaml_references() -> None:
    refs = load_monomer_references(GROWTH)
    assert refs.energy_cdse_eV < 0
    assert refs.energy_cdcl2_eV < 0
    assert refs.energy_cdse_hartree is not None
    # Ha→eV consistency
    assert refs.energy_cdse_eV == pytest.approx(
        refs.energy_cdse_hartree * HARTREE_EV, rel=1e-9
    )


def test_formation_and_omega_algebra() -> None:
    refs = MonomerReferences(
        energy_cdse_eV=-100.0,
        energy_cdcl2_eV=-10.0,
    )
    # E = k*(-100) + p*(-10) + 3  →  ΔE_f = 3
    e = -100.0 * 2 + -10.0 * 3 + 3.0
    assert refs.formation_eV(e, 2, 3) == pytest.approx(3.0)
    # Ω = 3 - 3*Δμ
    assert refs.grand_potential_eV(e, 2, 3, delta_mu_cdcl2_eV=1.0) == pytest.approx(
        0.0
    )
    assert refs.package_energy_eV(-130.0, 3) == pytest.approx(-100.0)
