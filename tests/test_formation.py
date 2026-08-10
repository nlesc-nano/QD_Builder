"""Formation / grand-potential reporting (report-only thermo)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from builder.nucleation.formation import (
    HARTREE_EV,
    MonomerReferences,
    format_bin_ranking,
    load_monomer_references,
    select_display_delta_mu,
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
        package_cluster_eV={1: -115.0},  # E(1,1)
    )
    # E = k*(-100) + p*(-10) + 3  →  ΔE_f = 3
    e = -100.0 * 2 + -10.0 * 3 + 3.0
    assert refs.formation_eV(e, 2, 3) == pytest.approx(3.0)
    # Ω_free = 3 - 3*Δμ
    assert refs.grand_potential_eV(e, 2, 3, delta_mu_cdcl2_eV=1.0) == pytest.approx(
        0.0
    )
    # explicit free μ_CdSe⁰
    assert refs.grand_potential_eV(
        e, 2, 3, 1.0, mu_cdse0_eV=refs.mu_cdse0_free_eV()
    ) == pytest.approx(0.0)
    assert refs.package_energy_eV(-130.0, 3) == pytest.approx(-100.0)
    # (3,4) vs package (1,1): E - 3*E(1,1) - 1*E(CdCl2)
    e34 = -500.0
    de_pkg = refs.formation_from_package_eV(e34, 3, 4, 1)
    assert de_pkg == pytest.approx(-500.0 - 3 * (-115.0) - 1 * (-10.0))


def test_ligated_mu_and_omega() -> None:
    """μ_CdSe⁰(p_m)=E(1,p_m)−p_m E(CdCl2); Ω_lig = dE_pkg − p Δμ."""

    refs = MonomerReferences(
        energy_cdse_eV=-100.0,
        energy_cdcl2_eV=-10.0,
        package_cluster_eV={1: -125.0, 2: -140.0},
    )
    # μ_CdSe⁰(1) = -125 - 1*(-10) = -115
    assert refs.mu_cdse0_ligated_eV(1) == pytest.approx(-115.0)
    assert refs.mu_cdse0_ligated_eV(2) == pytest.approx(-120.0)
    assert refs.mu_cdse0_ligated_eV(9) is None

    # (k,p)=(2,3), E chosen so free dE_f = 1
    e = 2 * (-100.0) + 3 * (-10.0) + 1.0  # -229
    de_pkg1 = refs.formation_from_package_eV(e, 2, 3, 1)
    # excess = 3 - 2*1 = 1 → E - 2*(-125) - 1*(-10) = -229 + 250 + 10 = 31
    assert de_pkg1 == pytest.approx(31.0)

    # Ω_lig = E - k μ0 - p (E_Cl + Δμ) = dE_pkg - p Δμ
    dmu = 0.5
    om = refs.grand_potential_ligated_eV(e, 2, 3, 1, dmu)
    assert om == pytest.approx(de_pkg1 - 3 * dmu)
    assert refs.grand_potential_package_eV(e, 2, 3, 1, dmu) == pytest.approx(om)

    # same via explicit μ0
    mu0 = refs.mu_cdse0_ligated_eV(1)
    assert refs.grand_potential_eV(e, 2, 3, dmu, mu_cdse0_eV=mu0) == pytest.approx(
        om
    )


def test_select_display_delta_mu() -> None:
    assert select_display_delta_mu([-1.0, -0.5, 0.0, 0.5, 1.0]) == (
        -1.0,
        0.0,
        1.0,
    )
    assert select_display_delta_mu([0.0]) == (0.0,)
    assert select_display_delta_mu([-0.3, 0.2, 0.8]) == (-0.3, 0.2, 0.8)


def test_growth_yaml_loads_packages() -> None:
    refs = load_monomer_references(GROWTH)
    assert 1 in refs.package_cluster_eV
    de = refs.formation_from_package_eV(
        refs.package_cluster_eV[1] * 2 + refs.energy_cdcl2_eV,
        2,
        3,
        1,
    )
    # 2 * E(1,1) + 1 CdCl2 → (2,3); ΔE_pkg = 0
    assert de == pytest.approx(0.0, abs=1e-6)


def test_package_growth_profile_matrix() -> None:
    refs = MonomerReferences(
        energy_cdse_eV=-100.0,
        energy_cdcl2_eV=-10.0,
        package_cluster_eV={1: -115.0, 2: -130.0},
        method="g-xTB",
    )
    # Stoichiometric path p_m=1: (1,1), (2,2), (3,3)
    # E(1,1)=-115 → dE_f*=0
    # E(2,2)=2*(-115)+1 = -229 → dE_f*=1
    # E(3,3)=3*(-115)+3 = -342 → dE_f*=3
    minima = {
        (1, 1): {"energy_eV": -115.0, "structure_id": "k1p1"},
        (2, 2): {"energy_eV": -229.0, "structure_id": "k2p2"},
        (3, 3): {"energy_eV": -342.0, "structure_id": "k3p3"},
        (1, 2): {"energy_eV": -130.0, "structure_id": "k1p2"},
        # (2,4) and (3,6) missing for p_m=2
    }
    from builder.nucleation.formation import format_package_growth_profile

    text = format_package_growth_profile(
        minima,
        refs=refs,
        package_p_m=(1, 2),
        k_values=(1, 2, 3),
        delta_mu=(-1.0, 0.0, 1.0),
    )
    assert "package growth profile" in text
    assert "relative formation" in text
    assert "absolute formation" in text
    assert "p_m=1" in text
    assert "p_m=2" in text
    assert "k = 1" in text
    assert "k = 2" in text
    assert "0.000" in text
    assert "1.000" in text  # dE_f* at k=2, p_m=1
    assert "3.000" in text  # dE_f* at k=3, p_m=1
    assert "—" in text  # missing p_m=2 path
    assert "(1,1)" in text
    assert "(2,2)" in text
    # no structure-id clutter in the matrix view
    assert "k1p1" not in text


def test_format_bin_ranking_readable() -> None:
    refs = MonomerReferences(
        energy_cdse_eV=-100.0,
        energy_cdcl2_eV=-10.0,
        package_cluster_eV={1: -115.0, 2: -130.0},
        method="g-xTB",
    )
    isos = [
        SimpleNamespace(
            structure_id="k002_p003_mol0002",
            xtb_energy_eV=-232.0,
            xtb_converged=True,
        ),
        SimpleNamespace(
            structure_id="k002_p003_mol0001",
            xtb_energy_eV=-231.0,
            xtb_converged=True,
        ),
    ]
    text = format_bin_ranking(
        isos,
        k=2,
        p=3,
        refs=refs,
        package_p_m=(1, 2),
        delta_mu=(-1.0, -0.5, 0.0, 0.5, 1.0),
    )
    assert "ranking k=2 p=3" in text
    assert "k002_p003_mol0002" in text
    assert "grand potential" in text
    assert "free CdSe baseline" in text
    assert "ligated baselines" in text
    assert "Ω_free" in text
    assert "p_m" in text
    # lean / 0 / rich only — not the intermediate ±0.5 grid points
    assert "Ω@-0.5" not in text
    assert "Ω@-1.0" in text
    assert "Ω@+0.0" in text
    assert "Ω@+1.0" in text
    # winner first
    pos2 = text.index("k002_p003_mol0002")
    pos1 = text.index("k002_p003_mol0001")
    assert pos2 < pos1
