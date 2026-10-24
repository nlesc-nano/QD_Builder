# src/builder/composition.py
"""
Composition of a passivated dot as [MA]_m (MX_q)_k.

m counts formula units of the bulk compound in the core (the limiting
species). The excess core cations, which the X-type ligand balances, give
k_X in units of MX_q (CdCl2 for CdSe/Cl: k_X = N_Cd - N_Se). Z-type MX_q
complexes placed on top add k_Z, and k = k_X + k_Z.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np


def _formula(counts: Dict[str, int]) -> str:
    return "".join(f"{el}{n if n != 1 else ''}" for el, n in counts.items())


def core_composition(
    struct,
    syms: Sequence[str],
    core: np.ndarray,
    charges: Dict[str, float],
    ligand: Optional[str],
    z_type_ledger: Optional[List[dict]] = None,
) -> dict:
    """The `composition` manifest entry."""
    comp = struct.composition.element_composition.reduced_composition
    unit = {str(el.symbol): int(round(n)) for el, n in comp.items()}
    core = np.asarray(core, bool)
    n_core = Counter(s for s, c in zip(syms, core) if c)
    m = min(n_core.get(el, 0) // n for el, n in unit.items()) if unit else 0
    excess = {el: int(n_core.get(el, 0) - m * n) for el, n in unit.items()}
    n_lig = sum(1 for s, c in zip(syms, core) if not c and s == ligand)

    z_units: Dict[str, int] = {}
    z_ligand_atoms = 0
    for entry in z_type_ledger or []:
        z_units[entry["formula"]] = z_units.get(entry["formula"], 0) + int(entry.get("placed", 0))
        anions = entry.get("anion_symbols") or []
        z_ligand_atoms += int(entry.get("placed", 0)) * sum(1 for a in anions if a == ligand)

    out = {
        "formula_unit": _formula(unit),
        "m": int(m),
        "core_excess": {el: n for el, n in excess.items() if n},
        "x_type": {"ligand": ligand, "n": int(n_lig - z_ligand_atoms)},
        "z_type": z_units,
    }
    # One excess cation balanced by a monovalent-or-higher ligand: write it as MX_q.
    cations = [el for el, n in excess.items() if n > 0]
    q_lig = abs(float(charges.get(ligand, 0) or 0)) if ligand else 0.0
    if len(cations) == 1 and all(n == 0 for el, n in excess.items() if el != cations[0]) and q_lig > 0:
        metal = cations[0]
        q = float(charges.get(metal, 0)) / q_lig
        if q > 0 and abs(q - round(q)) < 1e-9:
            mx = f"{metal}{ligand}{int(round(q)) if round(q) != 1 else ''}"
            k_x = excess[metal]
            k_z = z_units.get(mx, 0)
            out.update({
                "mx_unit": mx,
                "k_X": int(k_x),
                "k_Z": int(k_z),
                "k": int(k_x + k_z),
                "formula": f"[{_formula(unit)}]_{m}({mx})_{k_x + k_z}",
                "x_type_balanced": out["x_type"]["n"] == k_x * int(round(q)),
            })
    elif all(n == 0 for n in excess.values()):
        out.update({"k_X": 0, "k_Z": 0, "k": 0, "formula": f"[{_formula(unit)}]_{m}"})
    return out
