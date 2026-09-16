#!/usr/bin/env python3
"""Persistent ASE/MACE geometry-optimization worker.

Standard output is reserved for one JSON response per JSON request. Human
progress goes to stderr so the parent process can safely keep the model loaded
while submitting many structures.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
import traceback


def _load_calculator(args):
    import torch

    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("torch MPS is not available to the MACE worker")
        # mace-torch 0.3.16 calls Tensor.double() inside the forward path, while
        # Apple MPS has no float64 kernels. Keep the compatibility workaround
        # isolated inside this subprocess.
        original_double = torch.Tensor.double

        def mps_safe_double(self, *call_args, **call_kwargs):
            if self.device.type == "mps":
                return self
            return original_double(self, *call_args, **call_kwargs)

        torch.Tensor.double = mps_safe_double

    from mace.calculators import MACECalculator

    print(f"[mace] loading {args.model} on CPU", file=sys.stderr, flush=True)
    model = torch.load(args.model, map_location="cpu", weights_only=False)
    if args.dtype == "float32":
        model = model.float()
    elif args.dtype == "float64":
        model = model.double()
    else:
        raise ValueError(f"unsupported dtype: {args.dtype}")
    calc = MACECalculator(
        models=[model],
        device=args.device,
        default_dtype=args.dtype,
        head=args.head,
    )
    print(
        f"[mace] ready head={calc.head} device={args.device} dtype={args.dtype}",
        file=sys.stderr,
        flush=True,
    )
    return calc


def _optimizer(name):
    from ase.optimize import BFGS, FIRE, LBFGS

    choices = {"bfgs": BFGS, "lbfgs": LBFGS, "fire": FIRE}
    try:
        return choices[str(name).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported ASE optimizer: {name}") from exc


def _finite_matrix(values):
    return all(math.isfinite(float(value)) for row in values for value in row)


def relax_one(entry, settings, calculator):
    import numpy as np
    import torch
    from ase import Atoms
    from ase.constraints import FixAtoms

    structure_id = str(entry.get("id", ""))
    started = time.perf_counter()
    try:
        symbols = list(entry["symbols"])
        positions = np.asarray(entry["positions"], dtype=float)
        if positions.shape != (len(symbols), 3) or not _finite_matrix(positions):
            raise ValueError("invalid or non-finite input coordinates")
        atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
        frozen = [int(i) for i in (entry.get("freeze_atoms") or ())]
        if frozen:
            if min(frozen) < 0 or max(frozen) >= len(atoms):
                raise ValueError("freeze_atoms contains an out-of-range index")
            atoms.set_constraint(FixAtoms(indices=frozen))
        atoms.calc = calculator
        optimizer_class = _optimizer(settings.get("optimizer", "BFGS"))
        optimizer = optimizer_class(
            atoms,
            logfile=None,
            trajectory=None,
            maxstep=float(settings.get("maxstep", 0.2)),
        )
        fmax = float(settings.get("fmax", 0.02))
        max_steps = int(settings.get("max_steps", 500))
        # Energy-plateau stop.  For a ranking search the geometry is
        # structurally settled long before every force is below fmax; once the
        # total energy has moved by less than ``energy_tol_eV`` over the last
        # ``energy_window`` steps the optimisation is stopped and reported as
        # converged (status ``energy_converged``).  0 disables it.
        energy_tol = float(settings.get("energy_tol_eV", 0.0) or 0.0)
        energy_window = max(1, int(settings.get("energy_window", 5) or 5))
        # the plateau stop only fires once every force is already below this
        # (a BFGS creep across a shoulder shows a flat energy with forces
        # still ~0.05 eV/A and can drop hundreds of meV later)
        energy_fmax_gate = float(settings.get("energy_fmax_gate", 0.05) or 0.05)
        history = []
        stop_reason = ""
        converged = False
        if energy_tol > 0.0:
            for force_converged in optimizer.irun(fmax=fmax, steps=max_steps):
                history.append(float(atoms.get_potential_energy()))
                if force_converged:
                    converged = True
                    stop_reason = "fmax"
                    break
                if len(history) > energy_window:
                    span = max(history[-energy_window - 1:]) - min(history[-energy_window - 1:])
                    if span < energy_tol:
                        fnow = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
                        if fnow <= energy_fmax_gate:
                            converged = True
                            stop_reason = "energy_plateau"
                            break
            else:
                converged = False
        else:
            converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
            stop_reason = "fmax" if converged else ""
        if str(settings.get("device", "")).lower() == "mps":
            torch.mps.synchronize()
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        final_positions = np.asarray(atoms.get_positions(), dtype=float)
        if not math.isfinite(energy) or not _finite_matrix(forces):
            raise FloatingPointError("MACE returned a non-finite energy or force")
        if not _finite_matrix(final_positions):
            raise FloatingPointError("optimizer returned non-finite coordinates")
        max_force = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0
        if converged:
            status = "energy_converged" if stop_reason == "energy_plateau" else "converged"
        else:
            status = "maxcycle"
        accept = converged or bool(settings.get("accept_maxcycle", True))
        return {
            "id": structure_id,
            "ok": bool(accept),
            "energy_eV": energy,
            "steps": int(optimizer.nsteps),
            "converged": converged,
            "stop_reason": stop_reason,
            "max_force": max_force,
            "positions": final_positions.tolist(),
            "bond_orders": None,
            "status": status,
            "error": "" if accept else "not_converged:maxcycle",
            "wall_seconds": time.perf_counter() - started,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "id": structure_id,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
            "wall_seconds": time.perf_counter() - started,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--head", default="omat_pbe")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    args = parser.parse_args()

    # Third-party initialization can print banners. Redirect those to stderr;
    # stdout is the machine-readable transport.
    with contextlib.redirect_stdout(sys.stderr):
        calculator = _load_calculator(args)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request.get("command", "relax")
            if command == "close":
                return
            if command != "relax":
                raise ValueError(f"unknown command: {command}")
            structures = list(request.get("structures") or [])
            settings = dict(request.get("settings") or {})
            settings.setdefault("device", args.device)
            results = []
            for index, entry in enumerate(structures, start=1):
                print(
                    f"[mace] relax {index}/{len(structures)} id={entry.get('id', '')}",
                    file=sys.stderr,
                    flush=True,
                )
                results.append(relax_one(entry, settings, calculator))
            response = {"ok": True, "results": results}
        except Exception as exc:  # noqa: BLE001
            response = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
        print(json.dumps(response, allow_nan=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
