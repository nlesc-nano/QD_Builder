#!/usr/bin/env python3
"""MACE-MH-1 / MPS production preflight.

Runs the exact worker recipe from ``tools/mace_worker.py`` end to end in the
*current* process environment and refuses (exit 1) unless every stage passes
on MPS.  There is deliberately no CPU fallback: the production run must not
silently degrade, and a CPU-only run needs a different protocol fingerprint.

Stages:

  1. platform  -- arm64, macOS version, Python/Torch/MACE/ASE/e3nn versions
  2. mps       -- ``is_built`` and ``is_available``, a real tensor allocation,
                  a matmul checked against CPU, and a float64 request (which
                  MPS lacks -- this is why the worker patches ``Tensor.double``)
  3. model     -- checkpoint exists, sha256, loads on CPU, casts to float32,
                  requested head is present
  4. calculator-- ``MACECalculator(device="mps", default_dtype="float32",
                  head=...)`` builds with the ``Tensor.double`` workaround
  5. numerics  -- energy/forces of a small Cd/Se/Cl cluster on MPS agree with
                  the same float32 model on CPU (tolerance below), and are
                  finite and translation-invariant
  6. relax     -- a short BFGS relaxation runs to fmax on MPS and the energy
                  decreases monotonically to within tolerance
  7. timing    -- warm per-call cost, so epoch budgets can be sized

Writes a JSON fingerprint (``--json``) that the controller should embed in
every result it journals; ``--require-fingerprint`` fails if the live
environment differs from a previously recorded one (guards against a torch
upgrade half-way through a production run).

Run with the same interpreter the run YAML names under ``relaxation.python``::

    /Users/ivaninfante/miniforge3/envs/orchestr_ai-mace/bin/python \
        tools/mace_preflight.py --json runs/mace_import/preflight.json

Notes on "MPS unavailable": ``torch.backends.mps.is_available()`` returns
False -- with ``is_built() == True`` -- when the process cannot reach the
Metal device.  On this machine that happens inside sandboxed agent shells,
not because of a Torch/macOS incompatibility.  If stage 2 fails, rerun the
preflight from a plain login shell / launchd job before assuming the stack is
broken; the fingerprint records the sandbox indicators it could see.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List

DEFAULT_MODEL = "/Users/ivaninfante/.cache/mace/macemh1model"
DEFAULT_HEAD = "omat_pbe"

# Tolerances for MPS-vs-CPU float32 agreement.  Float32 MACE on two backends
# differs at the 1e-4 eV level for ~10 atoms; anything above the values below
# indicates a broken kernel or a dtype leak, not round-off.
ENERGY_TOL_EV = 2.0e-3
FORCE_TOL_EV_A = 2.0e-2
RELAX_ENERGY_RISE_TOL_EV = 1.0e-3

# Small, asymmetric, charge-balanced Cd3Se2Cl2 test cluster (k=2, p=1),
# deliberately not at a minimum so forces are non-trivial.
TEST_SYMBOLS = ["Cd", "Cd", "Cd", "Se", "Se", "Cl", "Cl"]
TEST_POSITIONS = [
    [0.00, 0.00, 0.00],
    [2.60, 0.00, 0.00],
    [0.00, 2.60, 0.00],
    [1.30, 1.30, 2.10],
    [1.30, 1.30, -2.10],
    [-2.30, 0.00, 0.00],
    [0.00, -2.30, 0.00],
]


class Preflight:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.report: Dict[str, Any] = {
            "ok": False,
            "stages": {},
            "fingerprint": {},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self._failed = False

    # -- helpers -----------------------------------------------------------
    def stage(self, name: str, ok: bool, **info: Any) -> bool:
        self.report["stages"][name] = {"ok": bool(ok), **info}
        flag = "PASS" if ok else "FAIL"
        detail = ", ".join(f"{k}={v}" for k, v in info.items() if k != "traceback")
        print(f"[preflight] {flag} {name}: {detail}", file=sys.stderr, flush=True)
        if not ok:
            self._failed = True
            if "traceback" in info:
                print(info["traceback"], file=sys.stderr)
        return bool(ok)

    @staticmethod
    def _exc() -> Dict[str, Any]:
        import traceback

        return {"error": traceback.format_exc().strip().splitlines()[-1], "traceback": traceback.format_exc()}

    # -- stages --------------------------------------------------------------
    def check_platform(self) -> bool:
        info: Dict[str, Any] = {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "sandbox_hints": {
                k: os.environ[k]
                for k in ("CLAUDECODE", "SANDBOX", "SANDBOX_RUNTIME", "TERM_PROGRAM")
                if k in os.environ
            },
        }
        try:
            import torch
            import mace
            import ase
            import e3nn

            info.update(
                torch=torch.__version__,
                mace=mace.__version__,
                ase=ase.__version__,
                e3nn=e3nn.__version__,
                torch_build_sha=getattr(torch.version, "git_version", None),
            )
        except Exception:
            return self.stage("platform", False, **info, **self._exc())
        self.report["fingerprint"].update(
            {k: info[k] for k in ("python", "torch", "mace", "ase", "e3nn", "macos", "machine")}
        )
        return self.stage("platform", info["machine"] == "arm64", **info)

    def check_mps(self) -> bool:
        import torch

        info: Dict[str, Any] = {
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        }
        if not (info["mps_built"] and info["mps_available"]):
            info["hint"] = (
                "MPS built but not available -> process cannot reach Metal. "
                "Rerun from a non-sandboxed shell before assuming a torch/macOS problem."
                if info["mps_built"]
                else "torch wheel has no MPS support; reinstall an arm64 macOS wheel."
            )
            return self.stage("mps", False, **info)
        try:
            a = torch.randn(256, 256)
            b = torch.randn(256, 256)
            ref = a @ b
            got = (a.to("mps") @ b.to("mps")).cpu()
            torch.mps.synchronize()
            info["matmul_max_abs_err"] = float((ref - got).abs().max())
            # MPS has no float64 kernels; record what a .double() request does
            # so the worker's Tensor.double patch is known to be still needed.
            try:
                torch.ones(2, device="mps").double()
                info["mps_float64"] = "supported"
            except Exception as exc:  # noqa: BLE001
                info["mps_float64"] = f"unsupported ({type(exc).__name__})"
        except Exception:
            return self.stage("mps", False, **info, **self._exc())
        self.report["fingerprint"]["device"] = "mps"
        self.report["fingerprint"]["mps_float64"] = info["mps_float64"]
        return self.stage("mps", info["matmul_max_abs_err"] < 1e-2, **info)

    def _patch_double(self) -> None:
        import torch

        original = torch.Tensor.double

        def mps_safe_double(self, *a, **k):  # noqa: ANN001
            if self.device.type == "mps":
                return self
            return original(self, *a, **k)

        torch.Tensor.double = mps_safe_double

    def check_model(self) -> bool:
        import torch

        path = self.args.model
        info: Dict[str, Any] = {"model": path, "head": self.args.head}
        if not os.path.isfile(path):
            return self.stage("model", False, **info, error="checkpoint not found")
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        info["sha256"] = h.hexdigest()
        info["size_MB"] = round(os.path.getsize(path) / 1e6, 1)
        try:
            t0 = time.time()
            model = torch.load(path, map_location="cpu", weights_only=False)
            model = model.float()
            info["load_s"] = round(time.time() - t0, 2)
            heads = list(getattr(model, "heads", []) or [])
            info["heads"] = heads
            info["r_max"] = float(getattr(model, "r_max", float("nan")))
            info["n_params"] = sum(p.numel() for p in model.parameters())
            info["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
        except Exception:
            return self.stage("model", False, **info, **self._exc())
        self._model = model
        ok = (self.args.head in heads if heads else True) and info["param_dtypes"] == ["torch.float32"]
        self.report["fingerprint"].update(
            model_sha256=info["sha256"], head=self.args.head, dtype="float32", r_max=info["r_max"]
        )
        return self.stage("model", ok, **info)

    def _calculator(self, device: str):
        from mace.calculators import MACECalculator

        # MACECalculator calls ``model.to(device)`` in place.  Each calculator
        # must therefore own its own copy: sharing one model object between an
        # MPS and a CPU calculator moves the weights back and forth and the
        # MPS calculator then fails with "mat2 is on CPU".  The worker keeps a
        # single calculator, so this only matters here -- but it is also why
        # the worker must never build a second calculator from its model.
        return MACECalculator(
            models=[copy.deepcopy(self._model)],
            device=device,
            default_dtype="float32",
            head=self.args.head,
        )

    def _atoms(self):
        from ase import Atoms

        return Atoms(symbols=TEST_SYMBOLS, positions=TEST_POSITIONS)

    def check_calculator(self) -> bool:
        try:
            self._patch_double()
            t0 = time.time()
            self._calc_mps = self._calculator("mps")
            info = {"build_s": round(time.time() - t0, 2)}
        except Exception:
            return self.stage("calculator", False, **self._exc())
        return self.stage("calculator", True, **info)

    def check_numerics(self) -> bool:
        import numpy as np
        import torch

        try:
            a_mps = self._atoms()
            a_mps.calc = self._calc_mps
            t0 = time.time()
            e_mps = float(a_mps.get_potential_energy())
            f_mps = np.asarray(a_mps.get_forces())
            torch.mps.synchronize()
            first_call_s = time.time() - t0

            # Translation invariance on MPS
            a_shift = self._atoms()
            a_shift.positions += np.array([3.1, -1.7, 0.4])
            a_shift.calc = self._calc_mps
            e_shift = float(a_shift.get_potential_energy())

            # Same float32 weights on CPU as the reference
            a_cpu = self._atoms()
            a_cpu.calc = self._calculator("cpu")
            e_cpu = float(a_cpu.get_potential_energy())
            f_cpu = np.asarray(a_cpu.get_forces())

            info = {
                "E_mps_eV": round(e_mps, 6),
                "E_cpu_eV": round(e_cpu, 6),
                "dE_abs_eV": round(abs(e_mps - e_cpu), 6),
                "dF_max_eV_A": round(float(np.abs(f_mps - f_cpu).max()), 6),
                "translation_dE_eV": round(abs(e_mps - e_shift), 6),
                "net_force_eV_A": round(float(np.abs(f_mps.sum(axis=0)).max()), 6),
                "Fmax_eV_A": round(float(np.abs(f_mps).max()), 4),
                "first_call_s": round(first_call_s, 3),
            }
        except Exception:
            return self.stage("numerics", False, **self._exc())
        ok = (
            np.isfinite(e_mps)
            and np.isfinite(f_mps).all()
            and info["dE_abs_eV"] < ENERGY_TOL_EV
            and info["dF_max_eV_A"] < FORCE_TOL_EV_A
            and info["translation_dE_eV"] < ENERGY_TOL_EV
            and info["net_force_eV_A"] < FORCE_TOL_EV_A
        )
        return self.stage("numerics", ok, **info)

    def check_relax(self) -> bool:
        import numpy as np
        from ase.optimize import BFGS

        try:
            atoms = self._atoms()
            atoms.calc = self._calc_mps
            energies: List[float] = []

            opt = BFGS(atoms, maxstep=0.20, logfile=None)
            opt.attach(lambda: energies.append(float(atoms.get_potential_energy())), interval=1)
            t0 = time.time()
            converged = bool(opt.run(fmax=0.05, steps=60))
            dt = time.time() - t0
            rises = [b - a for a, b in zip(energies, energies[1:]) if b - a > RELAX_ENERGY_RISE_TOL_EV]
            dmin = self._min_distance(atoms)
            info = {
                "converged": converged,
                "steps": opt.get_number_of_steps(),
                "E0_eV": round(energies[0], 4),
                "E_final_eV": round(energies[-1], 4),
                "n_energy_rises": len(rises),
                "max_rise_eV": round(max(rises), 4) if rises else 0.0,
                "min_pair_distance_A": round(dmin, 3),
                "wall_s": round(dt, 2),
                "s_per_step": round(dt / max(1, opt.get_number_of_steps()), 3),
            }
        except Exception:
            return self.stage("relax", False, **self._exc())
        ok = converged and len(rises) <= 2 and dmin > 2.0 and np.isfinite(energies[-1])
        self.report["fingerprint"]["test_relax_E_final_eV"] = info["E_final_eV"]
        return self.stage("relax", ok, **info)

    @staticmethod
    def _min_distance(atoms) -> float:
        import numpy as np

        d = atoms.get_all_distances()
        return float(d[np.triu_indices(len(atoms), 1)].min())

    def check_timing(self) -> bool:
        import torch

        try:
            atoms = self._atoms()
            atoms.calc = self._calc_mps
            atoms.get_potential_energy()
            times = []
            for i in range(5):
                atoms.positions[0, 0] += 0.001
                t0 = time.time()
                atoms.get_potential_energy()
                atoms.get_forces()
                torch.mps.synchronize()
                times.append(time.time() - t0)
            info = {"warm_call_s_median": round(sorted(times)[2], 4), "n_atoms": len(atoms)}
        except Exception:
            return self.stage("timing", False, **self._exc())
        return self.stage("timing", True, **info)

    # -- driver --------------------------------------------------------------
    def run(self) -> int:
        steps = [
            self.check_platform,
            self.check_mps,
            self.check_model,
            self.check_calculator,
            self.check_numerics,
            self.check_relax,
            self.check_timing,
        ]
        for step in steps:
            if not step():
                break
        self.report["ok"] = not self._failed

        if self.report["ok"] and self.args.require_fingerprint:
            try:
                with open(self.args.require_fingerprint) as fh:
                    prev = json.load(fh).get("fingerprint", {})
            except Exception:
                prev = {}
            diff = {
                key: (prev.get(key), val)
                for key, val in self.report["fingerprint"].items()
                if key != "test_relax_E_final_eV" and prev.get(key) != val
            }
            self.stage("fingerprint_match", not diff, diff=diff or None)
            self.report["ok"] = not self._failed

        if self.args.json:
            os.makedirs(os.path.dirname(os.path.abspath(self.args.json)), exist_ok=True)
            with open(self.args.json, "w") as fh:
                json.dump(self.report, fh, indent=2, sort_keys=True)
        print(json.dumps({"ok": self.report["ok"], "fingerprint": self.report["fingerprint"]}))
        return 0 if self.report["ok"] else 1


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--head", default=DEFAULT_HEAD)
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument(
        "--require-fingerprint",
        metavar="PREV_JSON",
        help="fail if versions/model/device differ from a previous preflight report",
    )
    ap.add_argument(
        "--python",
        help="re-exec the preflight under this interpreter (e.g. the run YAML's relaxation.python)",
    )
    args = ap.parse_args(argv)

    if args.python and os.path.abspath(args.python) != os.path.abspath(sys.executable):
        cmd = [args.python, os.path.abspath(__file__)] + [
            a for a in (argv if argv is not None else sys.argv[1:]) if not a.startswith("--python")
        ]
        cmd = [c for c in cmd if c != args.python]
        return subprocess.call(cmd)

    return Preflight(args).run()


if __name__ == "__main__":
    sys.exit(main())
