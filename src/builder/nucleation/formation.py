"""Formation and grand-potential reporting for molecular (k, p) maps.

These quantities are **report-only**.  They must not filter graph generation,
parent selection, or growth channels.  Growth and enumeration use total
energies ``E`` only; chemical-potential tables re-rank stored results for
lean vs rich precursor interpretation.

Definitions (neutral composition ``k CdSe + p CdCl2``)::

    ΔE_f(k, p) = E(k, p) - k E(CdSe) - p E(CdCl2)

    Ω(k, p; Δμ) = ΔE_f(k, p) - p Δμ

with reservoir origin ``μ_CdSe = E(CdSe)``, ``μ_CdCl2 = E(CdCl2) + Δμ``.

Package building-block energy at k=1::

    E_pkg(p_m) = E(1, p_m) - p_m E(CdCl2)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import yaml

# Hartree → eV (CODATA-consistent with xtb_relax)
HARTREE_EV = 27.211386245988


@dataclass(frozen=True)
class MonomerReferences:
    """Free CdSe and CdCl2 energies for formation reporting."""

    energy_cdse_eV: float
    energy_cdcl2_eV: float
    method: str = "g-xTB"
    energy_cdse_hartree: Optional[float] = None
    energy_cdcl2_hartree: Optional[float] = None
    cdse_dir: Optional[str] = None
    cdcl2_dir: Optional[str] = None

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
        """E_pkg(p_m) = E(1, p_m) - p_m E(CdCl2)."""

        return float(energy_1_pm_eV) - int(p_m) * self.energy_cdcl2_eV


def _hartree_to_eV(eh: float) -> float:
    return float(eh) * HARTREE_EV


def load_monomer_references(
    source: Union[str, Path, Mapping[str, Any]],
) -> MonomerReferences:
    """Load references from a growth.yaml path or a ``references:`` mapping."""

    if isinstance(source, Mapping):
        raw = dict(source)
    else:
        path = Path(source)
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, Mapping):
            raise ValueError(f"growth file is not a mapping: {path}")
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


__all__ = [
    "HARTREE_EV",
    "MonomerReferences",
    "load_monomer_references",
    "load_delta_mu_grid",
]
