"""Formation and grand-potential reporting for molecular (k, p) maps.

These quantities are **report-only**.  They must not filter graph generation,
parent selection, or growth channels.  Growth and enumeration use total
energies ``E`` only; chemical-potential tables re-rank stored results for
lean vs rich precursor interpretation.

Definitions (neutral composition ``k CdSe + p CdCl2``), same algebra as the
nucleation grand-potential reconstruction but on **g-xTB** energies::

    # Free-molecule formation
    ΔE_f(k, p) = E(k, p) − k E(CdSe) − p E(CdCl2)

    # Building-block packages (1, p_m): min total energy E(1, p_m) of the k=1 bin
    # Assemble (k, p) as  k packages + free excess CdCl2:
    #   k · [CdSe](CdCl2)_{p_m}  +  (p − k p_m) · CdCl2
    ΔE_pkg(k, p; p_m) = E(k, p) − k E(1, p_m) − (p − k·p_m) E(CdCl2)

    # Grand potential (CdCl2 supersaturation Δμ only; Δμ_CdSe ≡ 0)
    #   μ_CdCl2 = E(CdCl2) + Δμ
    #   Ω = E − k μ_CdSe⁰ − p μ_CdCl2
    #
    # Free CdSe baseline:
    #   μ_CdSe⁰ = E(CdSe)
    #   Ω_free = ΔE_f − p Δμ
    #
    # Ligated package baseline (same idea as CP2K --mu-cdse-baseline-ligands):
    #   μ_CdSe⁰(p_m) = E(1, p_m) − p_m E(CdCl2)
    #   Ω_lig(p_m) = E − k μ_CdSe⁰(p_m) − p (E(CdCl2) + Δμ)
    #              = ΔE_pkg − p Δμ
    #
    # Low Δμ (lean precursor) → fewer free CdCl2 preferred;
    # high Δμ (rich) → high p preferred.  Report-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

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
        """ΔE_f = E − k E(CdSe) − p E(CdCl2)."""

        return (
            float(energy_eV)
            - int(k) * self.energy_cdse_eV
            - int(p) * self.energy_cdcl2_eV
        )

    def mu_cdse0_free_eV(self) -> float:
        """Free-molecule CdSe chemical-potential zero (eV)."""

        return float(self.energy_cdse_eV)

    def mu_cdse0_ligated_eV(self, p_m: int) -> Optional[float]:
        """Ligated CdSe zero from k=1 package: E(1,p_m) − p_m E(CdCl2)."""

        e1 = self.package_cluster_eV.get(int(p_m))
        if e1 is None:
            return None
        return float(e1) - int(p_m) * self.energy_cdcl2_eV

    def grand_potential_eV(
        self,
        energy_eV: float,
        k: int,
        p: int,
        delta_mu_cdcl2_eV: float,
        *,
        mu_cdse0_eV: Optional[float] = None,
    ) -> float:
        """Ω = E − k μ_CdSe⁰ − p (E(CdCl2) + Δμ).

        Default μ_CdSe⁰ is free E(CdSe).  Pass a ligated zero for package
        baselines.  Equivalent to ΔE_f − p Δμ when μ_CdSe⁰ = E(CdSe).
        """

        mu_se = (
            self.energy_cdse_eV
            if mu_cdse0_eV is None
            else float(mu_cdse0_eV)
        )
        return (
            float(energy_eV)
            - int(k) * mu_se
            - int(p) * (self.energy_cdcl2_eV + float(delta_mu_cdcl2_eV))
        )

    def grand_potential_ligated_eV(
        self,
        energy_eV: float,
        k: int,
        p: int,
        p_m: int,
        delta_mu_cdcl2_eV: float,
        *,
        energy_1_pm_eV: Optional[float] = None,
    ) -> Optional[float]:
        """Ω with μ_CdSe⁰ = E(1,p_m) − p_m E(CdCl2)  (= ΔE_pkg − p Δμ)."""

        if energy_1_pm_eV is not None:
            mu0 = float(energy_1_pm_eV) - int(p_m) * self.energy_cdcl2_eV
        else:
            mu0 = self.mu_cdse0_ligated_eV(p_m)
        if mu0 is None:
            return None
        return self.grand_potential_eV(
            energy_eV, k, p, delta_mu_cdcl2_eV, mu_cdse0_eV=mu0
        )

    def package_energy_eV(self, energy_1_pm_eV: float, p_m: int) -> float:
        """E_pkg binding = E(1, p_m) − p_m E(CdCl2) (relative to free CdCl2)."""

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

        ΔE = E(k,p) − k E(1,p_m) − (p − k p_m) E(CdCl2)

        Example: (k,p)=(3,4), p_m=1 → E − 3 E(1,1) − 1 E(CdCl2).
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
        """Alias: nucleation-style Ω with ligated μ_CdSe⁰(p_m).

        Ω = ΔE_pkg − p Δμ  (= E − k μ_CdSe⁰(p_m) − p (E(CdCl2)+Δμ)).
        """

        return self.grand_potential_ligated_eV(
            energy_eV,
            k,
            p,
            p_m,
            delta_mu_cdcl2_eV,
            energy_1_pm_eV=energy_1_pm_eV,
        )


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


def select_display_delta_mu(
    delta_mu: Sequence[float],
    *,
    max_points: int = 3,
) -> Tuple[float, ...]:
    """Pick a small lean / neutral / rich subset for log tables.

    Prefer min, 0 (if present or nearest), max — at most ``max_points``
    unique values.  Full grids stay in CSV / config; logs stay readable.
    """

    if not delta_mu:
        return (-1.0, 0.0, 1.0)[:max_points]
    vals = sorted({float(x) for x in delta_mu})
    if len(vals) <= max_points:
        return tuple(vals)
    # lean, near-zero, rich
    lean = vals[0]
    rich = vals[-1]
    zeroish = min(vals, key=lambda x: abs(x))
    picked = sorted({lean, zeroish, rich})
    if len(picked) >= max_points:
        return tuple(picked[:max_points])
    # fill from remaining closest to zero then extremes already used
    for v in vals:
        if v not in picked:
            picked.append(v)
        if len(picked) >= max_points:
            break
    return tuple(sorted(picked)[:max_points])


def format_bin_ranking(
    isomers: Sequence[Any],
    *,
    k: int,
    p: int,
    refs: Optional[MonomerReferences],
    package_p_m: Sequence[int] = (1, 2, 3),
    delta_mu: Sequence[float] = (-1.0, 0.0, 1.0),
    max_isomers: int = 20,
) -> str:
    """Human-readable bin ranking + compact grand-potential snapshot.

    Isomer table: E, dE_bin, dE_f, dE_pkg(p_m).  Grand-potential block uses
    g-xTB free and ligated μ_CdSe⁰ baselines with a *few* Δμ_CdCl₂ points
    (not a full 2D surface).  Report-only.
    """

    rows = []
    for iso in isomers:
        e = getattr(iso, "xtb_energy_eV", None)
        if e is None:
            continue
        rows.append(iso)
    if not rows:
        return "  (no converged energies to rank)"

    rows.sort(key=lambda iso: float(iso.xtb_energy_eV))
    emin = float(rows[0].xtb_energy_eV)
    winner = rows[0]
    e_win = emin
    n_show = min(len(rows), int(max_isomers))
    pms = tuple(int(x) for x in package_p_m)
    dmu_show = select_display_delta_mu(delta_mu, max_points=3)

    # --- isomer ranking (ASCII only: safe for HPC log locales) --------
    pkg_hdr = "  ".join(f"{'dE_pkg'+str(pm):>8}" for pm in pms)
    lines: list[str] = [
        f"  -- ranking k={k} p={p}  "
        f"({len(rows)} isomers with E; most stable -> least; eV) --",
        "  "
        + f"{'rk':>3}  {'id':28s}  {'E':>14}  {'dE_bin':>7}  {'dE_f':>8}"
        + (f"  {pkg_hdr}" if pms else ""),
    ]
    for rank, iso in enumerate(rows[:n_show], start=1):
        e = float(iso.xtb_energy_eV)
        de_bin = e - emin
        de_f = refs.formation_eV(e, k, p) if refs else float("nan")
        pkg_cols = []
        for pm in pms:
            if refs is None:
                pkg_cols.append(f"{'n/a':>8}")
                continue
            de_p = refs.formation_from_package_eV(e, k, p, pm)
            pkg_cols.append(
                f"{de_p:8.3f}" if de_p is not None else f"{'n/a':>8}"
            )
        line = (
            "  "
            + f"{rank:3d}  {iso.structure_id:28s}  {e:14.4f}  "
            + f"{de_bin:7.3f}  {de_f:8.3f}"
        )
        if pkg_cols:
            line += "  " + "  ".join(pkg_cols)
        lines.append(line)
    if len(rows) > n_show:
        lines.append(f"  ... ({len(rows) - n_show} more isomers not shown)")

    lines.append(
        "  dE_f = E - k E(CdSe) - p E(CdCl2);  "
        "dE_pkg(p_m) = E - k E(1,p_m) - (p-k p_m) E(CdCl2)"
    )

    if refs is None:
        lines.append("  (no monomer references - grand potential skipped)")
        return "\n".join(lines)

    # --- grand potential (compact, ASCII) -----------------------------
    method = refs.method or "g-xTB"
    dmu_labels = "  ".join(f"{dm:+.1f}".rjust(10) for dm in dmu_show)
    free_vals = "  ".join(
        f"{refs.grand_potential_eV(e_win, k, p, dm):10.3f}" for dm in dmu_show
    )
    lines.extend(
        [
            "",
            f"  -- grand potential ({method}, report-only) --",
            "  Omega = E - k*mu_CdSe0 - p*(E(CdCl2) + dmu)",
            "      dmu = mu_CdCl2 - E(CdCl2);  dmu_CdSe = 0 (not scanned)",
            f"  bin winner: {winner.structure_id}   E = {e_win:.4f} eV",
            "",
            f"  free CdSe baseline   mu_CdSe0 = E(CdSe) = "
            f"{refs.mu_cdse0_free_eV():.4f} eV",
            f"    {'dmu_CdCl2':>10}  {dmu_labels}",
            f"    {'Omega_free':>10}  {free_vals}",
            "",
            "  ligated baselines   mu_CdSe0(p_m) = E(1,p_m) - p_m E(CdCl2)",
            "  "
            + f"{'p_m':>4}  {'mu_CdSe0':>12}  {'dE_pkg':>8}  "
            + "  ".join(f"{('O@'+f'{dm:+.1f}'):>10}" for dm in dmu_show),
        ]
    )
    any_pkg = False
    for pm in pms:
        mu0 = refs.mu_cdse0_ligated_eV(pm)
        de_pkg = refs.formation_from_package_eV(e_win, k, p, pm)
        if mu0 is None or de_pkg is None:
            lines.append(
                "  "
                + f"{pm:4d}  {'(no E(1,p_m))':>12}  {'n/a':>8}  "
                + "  ".join(f"{'n/a':>10}" for _ in dmu_show)
            )
            continue
        any_pkg = True
        om_cols = [
            f"{refs.grand_potential_ligated_eV(e_win, k, p, pm, dm):10.3f}"
            for dm in dmu_show
        ]
        lines.append(
            "  "
            + f"{pm:4d}  {mu0:12.4f}  {de_pkg:8.3f}  "
            + "  ".join(om_cols)
        )
    if not any_pkg and pms:
        lines.append(
            "  (package_cluster_eV missing in growth.yaml references)"
        )

    lines.append(
        "  note: within one (k,p) bin Omega only shifts by -p dmu - "
        "isomer order = total energy; compare Omega across p (or k) for lean/rich"
    )
    return "\n".join(lines)


def _stoichiometric_path_energy(
    bin_minima: Mapping[Tuple[int, int], Mapping[str, Any]],
    refs: MonomerReferences,
    *,
    k: int,
    p_m: int,
) -> Optional[float]:
    """Bin-minimum E(k, k·p_m) if available; k=1 may use package_cluster_eV."""

    p = int(k) * int(p_m)
    row = bin_minima.get((int(k), p))
    if row is not None and row.get("energy_eV") is not None:
        return float(row["energy_eV"])
    if int(k) == 1:
        e1 = refs.package_cluster_eV.get(int(p_m))
        if e1 is not None:
            return float(e1)
    return None


def format_package_growth_profile(
    bin_minima: Mapping[Tuple[int, int], Mapping[str, Any]],
    *,
    refs: Optional[MonomerReferences],
    package_p_m: Sequence[int] = (1, 2, 3),
    k_values: Sequence[int] = (1, 2, 3),
    delta_mu: Sequence[float] = (0.0,),
) -> str:
    """Number-only package profile matrices (k × p_m), baseline 0 at k=1.

    Path for column ``p_m`` is stoichiometric ``(k, p=k·p_m)``.  On that path::

        dE_f*(k) = dE_f(k) − k·dE_f(1) = E(k, k p_m) − k E(1, p_m) = dE_pkg

    so the relative matrix is unique.  Also prints absolute ``dE_f`` (eV).
    Missing bins as ``—``.  Report-only.
    """

    if refs is None:
        return "  (no monomer references - package profile skipped)"

    pms = tuple(int(x) for x in package_p_m)
    ks = tuple(int(x) for x in k_values)
    method = refs.method or "g-xTB"
    _ = delta_mu

    col_w = 10
    # header label width for row stubs like "k = 3"
    stub_w = 14
    # ASCII placeholder for missing bins (em dash breaks non-UTF8 HPC logs)
    miss = "n/a"

    def _col_headers() -> str:
        return "".join(f"{('p_m=' + str(pm)):>{col_w}s}" for pm in pms)

    def _fmt_cell(val: Optional[float]) -> str:
        if val is None:
            return f"{miss:>{col_w}s}"
        return f"{val:{col_w}.3f}"

    # Build matrices: dE_f* (relative) and dE_f (absolute)
    de_f_star: Dict[Tuple[int, int], Optional[float]] = {}
    de_f_abs: Dict[Tuple[int, int], Optional[float]] = {}
    for pm in pms:
        e1 = _stoichiometric_path_energy(bin_minima, refs, k=1, p_m=pm)
        if e1 is None:
            for k in ks:
                de_f_star[(k, pm)] = None
                de_f_abs[(k, pm)] = None
            continue
        de_f_1 = refs.formation_eV(e1, 1, pm)
        for k in ks:
            e = _stoichiometric_path_energy(bin_minima, refs, k=k, p_m=pm)
            if e is None:
                de_f_star[(k, pm)] = None
                de_f_abs[(k, pm)] = None
                continue
            p = k * pm
            de_f = refs.formation_eV(e, k, p)
            de_f_abs[(k, pm)] = de_f
            de_f_star[(k, pm)] = de_f - k * de_f_1

    lines: list[str] = [
        f"  -- package growth profile ({method}, report-only) --",
        "  path: column p_m uses composition (k, p = k*p_m)",
        "  dE_f* = dE_f(k) - k*dE_f(1) = dE_pkg   "
        "(0 at k=1; more negative -> more stable)",
        "",
        "  relative formation  dE_f*  (eV)",
        "  " + f"{'':>{stub_w}s}" + _col_headers(),
    ]
    for k in ks:
        # show composition pattern once in the stub for clarity
        stub = f"k = {k}"
        row = "".join(_fmt_cell(de_f_star.get((k, pm))) for pm in pms)
        lines.append("  " + f"{stub:<{stub_w}s}" + row)

    lines.extend(
        [
            "",
            "  absolute formation  dE_f  (eV)   "
            "[Omega_free at dmu=0; not zeroed at k=1]",
            "  " + f"{'':>{stub_w}s}" + _col_headers(),
        ]
    )
    for k in ks:
        stub = f"k = {k}"
        row = "".join(_fmt_cell(de_f_abs.get((k, pm))) for pm in pms)
        lines.append("  " + f"{stub:<{stub_w}s}" + row)

    # composition key (numbers only, no structure ids)
    lines.extend(
        [
            "",
            "  compositions (k, p)",
            "  " + f"{'':>{stub_w}s}" + _col_headers(),
        ]
    )
    for k in ks:
        stub = f"k = {k}"
        cells = []
        for pm in pms:
            p = k * pm
            label = f"({k},{p})"
            cells.append(f"{label:>{col_w}s}")
        lines.append("  " + f"{stub:<{stub_w}s}" + "".join(cells))

    n_miss = sum(1 for v in de_f_star.values() if v is None)
    if n_miss:
        lines.append(
            "  n/a = bin not available yet (map/growth has no energy for that "
            "(k, k*p_m))"
        )
    return "\n".join(lines)


__all__ = [
    "HARTREE_EV",
    "MonomerReferences",
    "load_monomer_references",
    "load_delta_mu_grid",
    "select_display_delta_mu",
    "format_bin_ranking",
    "format_package_growth_profile",
]
