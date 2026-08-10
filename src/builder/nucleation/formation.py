"""Formation and grand-potential reporting for molecular (k, p) maps.

These quantities are **report-only**.  They must not filter graph generation,
parent selection, or growth channels.  Growth and enumeration use total
energies ``E`` only; chemical-potential tables re-rank stored results for
lean vs rich precursor interpretation.

Definitions (neutral composition ``k CdSe + p CdCl2``)::

    # Free-molecule reservoir (standard formation)
    ΔE_f(k, p) = E(k, p) - k E(CdSe) - p E(CdCl2)

    # Building-block packages (1, p_m): cluster energy E(1, p_m) of the k=1 min
    # Assemble (k, p) as  k packages + free excess CdCl2:
    #   k · [CdSe](CdCl2)_{p_m}  +  (p - k p_m) · CdCl2
    ΔE_pkg(k, p; p_m) = E(k, p) - k E(1, p_m) - (p - k·p_m) E(CdCl2)
    # e.g. (3,4) vs package (1,1):
    #   E(3,4) - 3 E(1,1) - 1 E(CdCl2)

    # Grand potential (CdCl2 chemical potential offset Δμ from free molecule)
    Ω(k, p; Δμ) = ΔE_f(k, p) - p Δμ
                = E - k E(CdSe) - p (E(CdCl2) + Δμ)

    # Package-oriented grand potential (packages + free CdCl2 reservoir):
    Ω_pkg(k, p; p_m, Δμ) = ΔE_pkg(k, p; p_m) - (p - k·p_m) Δμ
    # Low Δμ (lean precursor) → fewer free CdCl2 preferred;
    # high Δμ (rich) → high p preferred.  Report-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import yaml

# Hartree → eV (CODATA-consistent with xtb_relax)
HARTREE_EV = 27.211386245988


@dataclass(frozen=True)
class MonomerReferences:
    """Free CdSe/CdCl2 and optional k=1 package cluster energies (eV)."""

    energy_cdse_eV: float
    energy_cdcl2_eV: float
    method: str = "g-xTB"
    energy_cdse_hartree: Optional[float] = None
    energy_cdcl2_hartree: Optional[float] = None
    cdse_dir: Optional[str] = None
    cdcl2_dir: Optional[str] = None
    #: p_m → min total energy E(1, p_m) of the k=1 bin (eV)
    package_cluster_eV: Mapping[int, float] = field(default_factory=dict)

    def formation_eV(self, energy_eV: float, k: int, p: int) -> float:
        """ΔE_f = E - k E(CdSe) - p E(CdCl2)."""

        return (
            float(energy_eV)
            - int(k) * self.energy_cdse_eV
            - int(p) * self.energy_cdcl2_eV
        )

    def grand_potential_eV(
        self,
        energy_eV: float,
        k: int,
        p: int,
        delta_mu_cdcl2_eV: float,
    ) -> float:
        """Ω = ΔE_f - p Δμ (informative; does not steer growth)."""

        return self.formation_eV(energy_eV, k, p) - int(p) * float(
            delta_mu_cdcl2_eV
        )

    def package_energy_eV(self, energy_1_pm_eV: float, p_m: int) -> float:
        """E_pkg binding = E(1, p_m) - p_m E(CdCl2) (relative to free CdCl2)."""

        return float(energy_1_pm_eV) - int(p_m) * self.energy_cdcl2_eV

    def formation_from_package_eV(
        self,
        energy_eV: float,
        k: int,
        p: int,
        p_m: int,
        *,
        energy_1_pm_eV: Optional[float] = None,
    ) -> Optional[float]:
        """Formation vs k packages (1, p_m) + free excess CdCl2.

        ΔE = E(k,p) - k E(1,p_m) - (p - k p_m) E(CdCl2)

        Example: (k,p)=(3,4), p_m=1 → E - 3 E(1,1) - 1 E(CdCl2).
        """

        e1 = energy_1_pm_eV
        if e1 is None:
            e1 = self.package_cluster_eV.get(int(p_m))
        if e1 is None:
            return None
        excess = int(p) - int(k) * int(p_m)
        return (
            float(energy_eV)
            - int(k) * float(e1)
            - excess * self.energy_cdcl2_eV
        )

    def grand_potential_package_eV(
        self,
        energy_eV: float,
        k: int,
        p: int,
        p_m: int,
        delta_mu_cdcl2_eV: float,
        *,
        energy_1_pm_eV: Optional[float] = None,
    ) -> Optional[float]:
        """Ω_pkg = ΔE_pkg - (p - k p_m) Δμ  (excess CdCl2 only)."""

        de = self.formation_from_package_eV(
            energy_eV, k, p, p_m, energy_1_pm_eV=energy_1_pm_eV
        )
        if de is None:
            return None
        excess = int(p) - int(k) * int(p_m)
        return de - excess * float(delta_mu_cdcl2_eV)


def _hartree_to_eV(eh: float) -> float:
    return float(eh) * HARTREE_EV


def load_monomer_references(
    source: Union[str, Path, Mapping[str, Any]],
) -> MonomerReferences:
    """Load references from a growth.yaml path or a ``references:`` mapping."""

    if isinstance(source, Mapping):
        raw = dict(source)
        full = None
    else:
        path = Path(source)
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, Mapping):
            raise ValueError(f"growth file is not a mapping: {path}")
        full = data
        raw = data.get("references") or data
        if not isinstance(raw, Mapping):
            raise ValueError(f"no references block in {path}")

    method = str(raw.get("method", "g-xTB"))
    eh_cdse = raw.get("energy_cdse_hartree")
    eh_cdcl2 = raw.get("energy_cdcl2_hartree")
    ev_cdse = raw.get("energy_cdse_eV")
    ev_cdcl2 = raw.get("energy_cdcl2_eV")

    if ev_cdse is None and eh_cdse is not None:
        ev_cdse = _hartree_to_eV(float(eh_cdse))
    if ev_cdcl2 is None and eh_cdcl2 is not None:
        ev_cdcl2 = _hartree_to_eV(float(eh_cdcl2))

    if ev_cdse is None or ev_cdcl2 is None:
        raise ValueError(
            "references need energy_cdse_eV/energy_cdcl2_eV "
            "(or the corresponding *_hartree fields)"
        )

    # Optional: energy_package_1_1_eV or package_cluster_eV: {1: ..., 2: ...}
    packages: Dict[int, float] = {}
    pkg_map = raw.get("package_cluster_eV") or {}
    if isinstance(pkg_map, Mapping):
        for key, val in pkg_map.items():
            packages[int(key)] = float(val)
    for key, val in raw.items():
        # energy_package_1_2_eV → p_m=2
        if str(key).startswith("energy_package_1_") and str(key).endswith(
            "_eV"
        ):
            mid = str(key)[len("energy_package_1_") : -len("_eV")]
            try:
                packages[int(mid)] = float(val)
            except ValueError:
                continue

    return MonomerReferences(
        energy_cdse_eV=float(ev_cdse),
        energy_cdcl2_eV=float(ev_cdcl2),
        method=method,
        energy_cdse_hartree=(
            None if eh_cdse is None else float(eh_cdse)
        ),
        energy_cdcl2_hartree=(
            None if eh_cdcl2 is None else float(eh_cdcl2)
        ),
        cdse_dir=None if raw.get("cdse_dir") is None else str(raw["cdse_dir"]),
        cdcl2_dir=(
            None if raw.get("cdcl2_dir") is None else str(raw["cdcl2_dir"])
        ),
        package_cluster_eV=packages,
    )


def load_delta_mu_grid(
    growth_yaml: Union[str, Path],
) -> Sequence[float]:
    """Δμ_CdCl2 grid for report-only grand-potential tables."""

    data = yaml.safe_load(Path(growth_yaml).read_text())
    block = (data or {}).get("chemical_potential") or {}
    if not block.get("enabled", True):
        return ()
    grid = block.get("delta_mu_cdcl2_eV") or ()
    return tuple(float(x) for x in grid)


def format_bin_ranking(
    isomers: Sequence[Any],
    *,
    k: int,
    p: int,
    refs: Optional[MonomerReferences],
    package_p_m: Sequence[int] = (1, 2, 3),
    delta_mu: Sequence[float] = (-0.5, 0.0, 0.5),
) -> str:
    """Text table: most stable first; package ΔE and Ω columns (report-only)."""

    rows = []
    for iso in isomers:
        e = getattr(iso, "xtb_energy_eV", None)
        if e is None:
            continue
        if not getattr(iso, "xtb_converged", False):
            # still rank if energy present
            pass
        rows.append(iso)
    if not rows:
        return "  (no converged energies to rank)"

    rows.sort(key=lambda iso: float(iso.xtb_energy_eV))
    emin = float(rows[0].xtb_energy_eV)
    lines = [
        f"  ranking k={k} p={p}  (most stable → least; energies in eV)",
        "  "
        + f"{'rk':>3}  {'id':28s}  {'E':>14}  {'dE_bin':>8}  "
        + f"{'dE_f':>8}  "
        + "  ".join(f"{'dE_pkg'+str(pm):>10}" for pm in package_p_m)
        + "  "
        + "  ".join(f"{'Ω(Δμ='+f'{dm:+.1f}'+')':>12}" for dm in delta_mu),
    ]
    for rank, iso in enumerate(rows, start=1):
        e = float(iso.xtb_energy_eV)
        de_bin = e - emin
        de_f = refs.formation_eV(e, k, p) if refs else float("nan")
        pkg_cols = []
        for pm in package_p_m:
            if refs is None:
                pkg_cols.append(f"{'n/a':>10}")
                continue
            de_p = refs.formation_from_package_eV(e, k, p, pm)
            pkg_cols.append(
                f"{de_p:10.3f}" if de_p is not None else f"{'n/a':>10}"
            )
        omega_cols = []
        for dm in delta_mu:
            if refs is None:
                omega_cols.append(f"{'n/a':>12}")
            else:
                om = refs.grand_potential_eV(e, k, p, dm)
                omega_cols.append(f"{om:12.3f}")
        lines.append(
            "  "
            + f"{rank:3d}  {iso.structure_id:28s}  {e:14.4f}  {de_bin:8.3f}  "
            + f"{de_f:8.3f}  "
            + "  ".join(pkg_cols)
            + "  "
            + "  ".join(omega_cols)
        )
    lines.append(
        "  notes: dE_f = E − k E(CdSe) − p E(CdCl2);  "
        "dE_pkg(p_m) = E − k E(1,p_m) − (p−k p_m) E(CdCl2);  "
        "Ω = dE_f − p Δμ  (Δμ = μ_CdCl2 − E(CdCl2); report-only)"
    )
    return "\n".join(lines)


__all__ = [
    "HARTREE_EV",
    "MonomerReferences",
    "load_monomer_references",
    "load_delta_mu_grid",
    "format_bin_ranking",
]
