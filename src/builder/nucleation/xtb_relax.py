"""GFN-xTB relaxation of accepted molecular isomers.

The constructed geometry satisfies a table of restraints; xTB replaces that
with a real semi-empirical surface, so the final coordinates are close to DFT
quality and every isomer comes with a total energy that can rank it against its
siblings -- which no bond-count score can do.

xtb-python and ASE usually live in their own conda environment (the builder
needs networkx/pymatgen, which that environment does not carry).  When xTB is
importable here it is used directly; otherwise the work is handed to
``tools/xtb_worker.py`` under the interpreter named in the pack, one subprocess
per batch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = ["XtbSettings", "XtbResult", "relax_structures"]

_WORKER = Path(__file__).resolve().parents[3] / "tools" / "xtb_worker.py"


@dataclass(frozen=True)
class XtbSettings:
    """``relaxation:`` block of a geometry pack."""

    enabled: bool = False
    method: str = "GFN1-xTB"
    accuracy: float = 1.0
    electronic_temperature: float = 300.0
    max_iterations: int = 500
    fmax: float = 0.02
    max_steps: int = 500
    #: interpreter that has xtb-python; ``None`` means "this one"
    python: Optional[str] = None
    #: recompute the graph after relaxation and report how it differs
    check_connectivity: bool = True
    #: wall-clock guard for one worker batch; 0 disables the guard
    timeout_s: float = 0.0

    @classmethod
    def from_pack(cls, raw: Optional[Mapping[str, Any]]) -> "XtbSettings":
        if not isinstance(raw, Mapping):
            return cls()
        python = raw.get("python")
        return cls(
            enabled=bool(raw.get("enabled", False)),
            method=str(raw.get("method", "GFN1-xTB")),
            accuracy=float(raw.get("accuracy", 1.0)),
            electronic_temperature=float(raw.get("electronic_temperature", 300.0)),
            max_iterations=int(raw.get("max_iterations", 500)),
            fmax=float(raw.get("fmax", 0.02)),
            max_steps=int(raw.get("max_steps", 500)),
            python=None if python is None else str(Path(python).expanduser()),
            check_connectivity=bool(raw.get("check_connectivity", True)),
            timeout_s=float(raw.get("timeout_s", 0.0)),
        )

    def payload(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "accuracy": self.accuracy,
            "electronic_temperature": self.electronic_temperature,
            "max_iterations": self.max_iterations,
            "fmax": self.fmax,
            "max_steps": self.max_steps,
        }


@dataclass
class XtbResult:
    ok: bool
    energy_eV: Optional[float] = None
    gap_eV: Optional[float] = None
    homo_eV: Optional[float] = None
    lumo_eV: Optional[float] = None
    steps: int = 0
    converged: bool = False
    max_force: Optional[float] = None
    coordinates: Optional[Tuple[Tuple[float, float, float], ...]] = None
    bond_orders: Optional[List[List[float]]] = None
    error: str = ""
    #: pairs that gained or lost a bond relative to the input graph
    connectivity_changed: Tuple[Tuple[int, int], ...] = ()
    #: the graph implied by the relaxed coordinates
    relaxed_edges: Tuple[Tuple[int, int], ...] = ()


def _run_in_process(structures, settings) -> Optional[List[Dict[str, Any]]]:
    try:
        import xtb  # noqa: F401
        import ase  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    sys.path.insert(0, str(_WORKER.parent))
    try:
        import xtb_worker  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    return [xtb_worker.relax_one(e, settings.payload()) for e in structures]


def _run_in_subprocess(structures, settings) -> List[Dict[str, Any]]:
    interpreter = settings.python or sys.executable
    payload = {"settings": settings.payload(), "structures": structures}
    try:
        proc = subprocess.run(
            [interpreter, str(_WORKER)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
            timeout=(None if settings.timeout_s <= 0.0 else settings.timeout_s),
        )
    except subprocess.TimeoutExpired:
        return [
            {
                "id": e["id"],
                "ok": False,
                "error": f"timeout after {settings.timeout_s:g} s",
            }
            for e in structures
        ]
    except OSError as exc:
        return [{"id": e["id"], "ok": False, "error": str(exc)} for e in structures]
    if proc.returncode != 0 or not proc.stdout.strip():
        detail = (proc.stderr or "").strip().splitlines()
        message = detail[-1] if detail else f"exit {proc.returncode}"
        return [{"id": e["id"], "ok": False, "error": message} for e in structures]
    try:
        return json.loads(proc.stdout)["results"]
    except Exception as exc:  # noqa: BLE001
        return [
            {"id": e["id"], "ok": False, "error": f"bad worker output: {exc}"}
            for e in structures
        ]


def relaxed_edges(
    symbols: Sequence[str],
    positions: Sequence[Sequence[float]],
    cutoffs: Mapping[Tuple[str, str], float],
) -> List[Tuple[int, int]]:
    """Bonds present after relaxation, by the pack's own distance criterion.

    Distance from the bond table rather than a Wiberg order: the table is what
    defines a bond everywhere else in the pipeline, and for ionic Cd-Se/Cd-Cl a
    bond-order threshold is a second, differently-calibrated opinion.
    """

    import math

    out: List[Tuple[int, int]] = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            limit = cutoffs.get(tuple(sorted((symbols[i], symbols[j]))))
            if limit is None:
                continue
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dz = positions[i][2] - positions[j][2]
            if math.sqrt(dx * dx + dy * dy + dz * dz) <= limit:
                out.append((i, j))
    return out


def _connectivity_drift(
    relaxed: Sequence[Tuple[int, int]],
    edges: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[int, int], ...]:
    """Pairs that gained or lost a bond relative to the enumerated graph."""

    before = {(min(a, b), max(a, b)) for a, b in edges}
    after = {(min(a, b), max(a, b)) for a, b in relaxed}
    return tuple(sorted(before ^ after))


def relax_structures(
    structures: Sequence[Mapping[str, Any]],
    settings: XtbSettings,
    bond_cutoffs: Optional[Mapping[Tuple[str, str], float]] = None,
) -> List[XtbResult]:
    """Relax a batch.  ``structures`` need ``id``, ``symbols``, ``positions``
    and -- when connectivity is checked -- ``edges``."""

    if not settings.enabled or not structures:
        return [XtbResult(ok=False, error="disabled") for _ in structures]
    payload = [
        {
            "id": str(entry["id"]),
            "symbols": list(entry["symbols"]),
            "positions": [list(map(float, p)) for p in entry["positions"]],
        }
        for entry in structures
    ]
    raw = _run_in_process(payload, settings)
    if raw is None:
        raw = _run_in_subprocess(payload, settings)
    by_id = {str(r.get("id")): r for r in raw}

    results: List[XtbResult] = []
    for entry in structures:
        r = by_id.get(str(entry["id"]), {"ok": False, "error": "missing result"})
        if not r.get("ok"):
            results.append(XtbResult(ok=False, error=str(r.get("error", "failed"))))
            continue
        coords = tuple(
            (float(x), float(y), float(z)) for x, y, z in r["positions"]
        )
        drift: Tuple[Tuple[int, int], ...] = ()
        after: Tuple[Tuple[int, int], ...] = ()
        if settings.check_connectivity and bond_cutoffs and entry.get("edges"):
            after = tuple(
                relaxed_edges(entry["symbols"], coords, bond_cutoffs)
            )
            drift = _connectivity_drift(after, entry["edges"])
        results.append(
            XtbResult(
                ok=True,
                energy_eV=r.get("energy_eV"),
                gap_eV=r.get("gap_eV"),
                homo_eV=r.get("homo_eV"),
                lumo_eV=r.get("lumo_eV"),
                steps=int(r.get("steps", 0)),
                converged=bool(r.get("converged", False)),
                max_force=r.get("max_force"),
                coordinates=coords,
                bond_orders=r.get("bond_orders"),
                connectivity_changed=drift,
                relaxed_edges=after,
            )
        )
    return results
