#!/usr/bin/env python3
"""GFN-xTB relaxation worker.

Reads one JSON batch on stdin and writes one JSON batch on stdout, so the
builder can use xTB without sharing an environment with it: xtb-python and ASE
live in their own conda env, while the builder needs networkx/pymatgen which
that env does not have.  One subprocess per *batch* keeps interpreter start-up
off the per-structure cost.

Run directly to self-test::

    ~/miniconda3/envs/xtb/bin/python tools/xtb_worker.py --selftest
"""

from __future__ import annotations

import json
import sys

HARTREE_EV = 27.211386245988
BOHR_PER_ANGSTROM = 1.8897261254535


def _electronic(numbers, positions_A, method: str, accuracy: float):
    """HOMO/LUMO and Wiberg bond orders from the low-level xTB API.

    The ASE calculator exposes only energy/forces/charges/dipole, so the gap
    and the bond orders have to come from ``xtb.interface`` directly.
    """

    import numpy as np
    from xtb.interface import Calculator, Param
    from xtb.libxtb import VERBOSITY_MUTED

    params = {"GFN1-XTB": Param.GFN1xTB, "GFN2-XTB": Param.GFN2xTB}
    param = params.get(method.upper().replace("_", "-"), Param.GFN1xTB)
    calc = Calculator(
        param,
        np.asarray(numbers),
        np.asarray(positions_A, dtype=float) * BOHR_PER_ANGSTROM,
    )
    calc.set_verbosity(VERBOSITY_MUTED)
    calc.set_accuracy(float(accuracy))
    res = calc.singlepoint()
    energies = res.get_orbital_eigenvalues()
    occupations = res.get_orbital_occupations()
    occupied = [e for e, o in zip(energies, occupations) if o > 0.5]
    virtual = [e for e, o in zip(energies, occupations) if o <= 0.5]
    homo = max(occupied) * HARTREE_EV if occupied else None
    lumo = min(virtual) * HARTREE_EV if virtual else None
    return {
        "homo_eV": homo,
        "lumo_eV": lumo,
        "gap_eV": (None if homo is None or lumo is None else lumo - homo),
        "bond_orders": np.asarray(res.get_bond_orders()).tolist(),
    }


def relax_one(entry, settings):
    from ase import Atoms
    from ase.optimize import BFGS
    from xtb.ase.calculator import XTB
    import numpy as np

    symbols = list(entry["symbols"])
    atoms = Atoms(symbols=symbols, positions=np.asarray(entry["positions"], float))
    atoms.calc = XTB(
        method=settings.get("method", "GFN1-xTB"),
        accuracy=float(settings.get("accuracy", 1.0)),
        electronic_temperature=float(settings.get("electronic_temperature", 300.0)),
        max_iterations=int(settings.get("max_iterations", 500)),
    )
    out = {"id": entry.get("id"), "ok": False}
    try:
        opt = BFGS(atoms, logfile=None)
        converged = opt.run(
            fmax=float(settings.get("fmax", 0.02)),
            steps=int(settings.get("max_steps", 500)),
        )
        forces = atoms.get_forces()
        out.update(
            ok=True,
            converged=bool(converged),
            steps=int(opt.get_number_of_steps()),
            energy_eV=float(atoms.get_potential_energy()),
            max_force=float(np.abs(forces).max()),
            positions=np.asarray(atoms.get_positions(), float).tolist(),
        )
        out.update(
            _electronic(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                settings.get("method", "GFN1-xTB"),
                settings.get("accuracy", 1.0),
            )
        )
    except Exception as exc:  # noqa: BLE001  - a failed structure must not
        out["error"] = f"{type(exc).__name__}: {exc}"  # kill the whole batch
    return out


def main(argv):
    if "--selftest" in argv:
        payload = {
            "settings": {"method": "GFN1-xTB", "fmax": 0.05, "max_steps": 50},
            "structures": [
                {
                    "id": "h2o",
                    "symbols": ["O", "H", "H"],
                    "positions": [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
                }
            ],
        }
    else:
        payload = json.load(sys.stdin)
    settings = payload.get("settings") or {}
    results = [relax_one(e, settings) for e in payload.get("structures") or []]
    json.dump({"results": results}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
