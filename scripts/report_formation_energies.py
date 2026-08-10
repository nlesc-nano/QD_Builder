#!/usr/bin/env python3
"""Report formation energies and grand potentials from finished map index.csv.

Does **not** change which structures exist.  Pure post-process:

    ΔE_f = E - k E(CdSe) - p E(CdCl2)
    Ω(Δμ) = ΔE_f - p Δμ

Example::

    python scripts/report_formation_energies.py \\
      --growth geometry_packs/cdse_cdcl2/growth.yaml \\
      --index /path/to/gxtb_cdse_target_k3_p1p7/index.csv \\
      --out analysis/formation_k3.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from builder.nucleation.formation import (  # noqa: E402
    load_delta_mu_grid,
    load_monomer_references,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--growth",
        type=Path,
        default=ROOT / "geometry_packs/cdse_cdcl2/growth.yaml",
        help="growth.yaml with references + optional Δμ grid",
    )
    ap.add_argument(
        "--index",
        type=Path,
        action="append",
        required=True,
        help="map index.csv (repeatable)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output CSV path",
    )
    args = ap.parse_args()

    refs = load_monomer_references(args.growth)
    dmu_grid = load_delta_mu_grid(args.growth)

    rows_out = []
    by_bin_min = defaultdict(lambda: float("inf"))
    raw = []
    for idx in args.index:
        with idx.open() as f:
            for r in csv.DictReader(f):
                if r.get("xtb_converged", "").lower() != "true":
                    continue
                try:
                    k, p = int(r["k"]), int(r["p"])
                    e = float(r["xtb_energy_eV"])
                except (KeyError, ValueError):
                    continue
                raw.append((k, p, e, r.get("structure_id", ""), idx.parent.name))
                by_bin_min[(k, p)] = min(by_bin_min[(k, p)], e)

    for k, p, e, sid, src in raw:
        de_f = refs.formation_eV(e, k, p)
        de_bin = e - by_bin_min[(k, p)]
        row = {
            "source": src,
            "structure_id": sid,
            "k": k,
            "p": p,
            "E_eV": f"{e:.8f}",
            "dE_bin_eV": f"{de_bin:.8f}",
            "dE_f_eV": f"{de_f:.8f}",
            "E_CdSe_eV": f"{refs.energy_cdse_eV:.8f}",
            "E_CdCl2_eV": f"{refs.energy_cdcl2_eV:.8f}",
            "method_ref": refs.method,
        }
        for dmu in dmu_grid:
            omega = refs.grand_potential_eV(e, k, p, dmu)
            key = f"Omega_dmu_{dmu:+.2f}".replace(".", "p").replace("+", "p").replace("-", "m")
            row[key] = f"{omega:.8f}"
        rows_out.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not rows_out:
        print("no converged rows", file=sys.stderr)
        return 1
    fields = list(rows_out[0].keys())
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    # Package summary at k=1
    print(f"wrote {len(rows_out)} rows → {args.out}")
    print(f"refs method={refs.method}")
    print(f"  E(CdSe)  = {refs.energy_cdse_eV:.6f} eV")
    print(f"  E(CdCl2) = {refs.energy_cdcl2_eV:.6f} eV")
    k1 = [(p, e) for k, p, e, sid, src in raw if k == 1]
    if k1:
        by_p = defaultdict(list)
        for p, e in k1:
            by_p[p].append(e)
        print("package energies E_pkg(p_m)=E(1,p_m)-p_m E(CdCl2) [min per p]:")
        for p in sorted(by_p):
            emin = min(by_p[p])
            pkg = refs.package_energy_eV(emin, p)
            print(f"  p_m={p}: E_min={emin:.6f}  E_pkg={pkg:.6f} eV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
