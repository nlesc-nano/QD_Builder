"""Post-treatments on an existing nanocrystal (``nc-builder CIF RECIPE --from-xyz DOT.xyz``).

No Wulff cut and no first charge balance: the input is a dot that a build already
passivated (ligand placeholders such as Cl on the surface).  The recipe's
``post_treatment`` block runs on it in the build's order:

  1. surface reconstruction (polar facets)
  2. alloying, then charge rebalance ('add', falling back to 'remove' if Q > 0 is left)
  3. Z-type displacement
  4. neutral exchange
  5. charged (X-type) ligand exchange, then ligand-only charge compensation
  6. Z-type ligands (MXn complexes on free surface anions)
  7. neutral (L-type) ligands

The Wulff planes used for site discovery are detected on the dot itself.  The XYZ is
written by ASE in the input atom order, so the run is reproducible from its own output.
The manifest carries the build's ledger keys plus ``post_treatment_ledger``, every
step's entries in the order they ran.
"""
from __future__ import annotations

import io
import os
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Structure

from .config import Config
from .facet_reconstruction import _native_facets_and_planes, reconstruct_polar_facets
from .facets import detect_facets_from_nc, expand_facets
from .io_utils import center_coords, write_manifest
from .passivation_iterative import charge_balance_iterative


def post_treat(
    syms: List[str],
    pts: NDArray[np.float64],
    cfg: Config,
    struct: Structure,
    cif_path: str,
    *,
    prefix: str,
    verbose: bool = False,
    write_all: bool = False,
) -> Tuple[List[str], NDArray[np.float64], Dict[str, object]]:
    """Run ``cfg.post_treatment`` on a passivated dot; return the dot and the step ledgers."""
    syms = list(syms)
    pts = np.asarray(pts, float)
    anion_lig = cfg.passivation.ligand or "Cl"
    facet_seeds = expand_facets(struct, cfg.seeds, proper_only=cfg.proper_only)
    pt = getattr(cfg, "post_treatment", None)
    flat: List[dict] = []
    ledgers: Dict[str, object] = {"post_treatment": flat}

    def balance(syms, pts, planes, strategy):
        return charge_balance_iterative(
            syms, pts,
            cfg.charges, anion_lig,
            verbose=verbose,
            planes=planes,
            surf_tol=cfg.passivation.surf_tol,
            cif_path=cif_path,
            positive_q_strategy=strategy,
            write_all=write_all,
            prefix=prefix,
            experimental_exhausted_positive_q_fallback=bool(
                getattr(cfg, "experimental", {}).get("exhausted_positive_q_fallback", False)
            ),
            prepass_mode=cfg.passivation.prepass_mode,
            prepass_min_cn_terrace=cfg.passivation.prepass_min_cn_terrace,
            prepass_min_cn_edge=cfg.passivation.prepass_min_cn_edge,
            prepass_min_cn_vertex=cfg.passivation.prepass_min_cn_vertex,
            include_sublayer=cfg.passivation.include_sublayer,
        )

    z_target = getattr(pt, "z_type_target", None)
    if z_target is not None and z_target.enabled:
        raise SystemExit("post_treatment.z_type_target is not supported with --from-xyz")

    # 1. Surface reconstruction (polar facets)
    recon_spec = getattr(pt, "surface_reconstruction", cfg.facet_reconstruction)
    if recon_spec is not None and recon_spec.enabled:
        recon_ledger: dict = {}
        syms, pts = reconstruct_polar_facets(
            syms,
            pts,
            struct=struct,
            facet_seeds=facet_seeds,
            charges=cfg.charges,
            ligand=anion_lig,
            surf_tol=cfg.passivation.surf_tol,
            cif_path=cif_path,
            spec=recon_spec,
            verbose=verbose,
            write_all=write_all,
            prefix=prefix,
            ledger=recon_ledger,
        )
        if recon_ledger:
            ledgers["surface_reconstruction"] = recon_ledger
            flat.append({"surface_reconstruction": True, **recon_ledger})

    _facets, planes = _native_facets_and_planes(
        syms, pts, struct, cfg.charges, facet_seeds, anion_lig, cfg.passivation.surf_tol
    )
    if not planes:
        _facets, planes = detect_facets_from_nc(
            syms, pts, struct.lattice, cfg.charges, facet_seeds, cfg.passivation.surf_tol
        )
    print(f"[from-xyz] Detected {len(planes)} Wulff plane(s) for post-treatment site discovery")

    # 2. Inorganic alloying, then charge rebalance.  A higher-valent dopant (In3+ on a
    # Cd2+ site) is compensated by extra X-type ligand; if the free ligand sites cannot
    # absorb it all, restart from the alloyed dot and remove surface atoms.
    alloying_spec = getattr(pt, "alloying", None)
    if alloying_spec is not None and alloying_spec.enabled:
        from .alloying_posttreat import run_alloying_posttreatment
        syms, pts, alloy_ledger = run_alloying_posttreatment(syms, pts, cfg, struct, planes)
        if alloy_ledger:
            for entry in alloy_ledger:
                cfg.charges.setdefault(entry["replacement"], int(entry["replacement_charge"]))
            ledgers["alloying"] = alloy_ledger
            flat.extend({"alloying": True, **entry} for entry in alloy_ledger)
            alloyed_syms, alloyed_pts = list(syms), np.asarray(pts, float).copy()
            for strategy in ("add", "remove"):
                if strategy == "remove":
                    if sum(int(cfg.charges.get(s, 0)) for s in syms) <= 0:
                        break
                    syms, pts = list(alloyed_syms), alloyed_pts.copy()
                syms, pts = balance(syms, pts, planes, strategy)

    # 3. Z-type displacement (neutral inorganic groups; before X-type exchange)
    z_spec = getattr(pt, "z_type_displacement", None)
    if z_spec is not None and z_spec.enabled:
        from .z_type_displacement_posttreat import run_z_type_displacement_posttreatment
        syms, pts, z_ledger = run_z_type_displacement_posttreatment(syms, pts, cfg, struct, planes, cif_path)
        ledgers["z_type_displacement"] = z_ledger
        flat.extend({"z_type": True, **entry} for entry in z_ledger)

    # 4. Neutral exchange (after displacement, before X-type exchange)
    ne_spec = getattr(pt, "neutral_exchange", None)
    if ne_spec is not None and ne_spec.enabled:
        from .neutral_exchange_posttreat import run_neutral_exchange_posttreatment
        syms, pts, ne_ledger = run_neutral_exchange_posttreatment(syms, pts, cfg, struct, planes, cif_path)
        ledgers["neutral_exchange"] = ne_ledger
        flat.extend({"neutral_exchange": True, **entry} for entry in ne_ledger)

    # 5. Charged ligand exchange (X-type), then ligand-only compensation of any residual
    x_ledger: List[dict] = []
    lig_spec = getattr(pt, "ligand_exchange", None)
    if lig_spec is not None and lig_spec.enabled:
        from .ligand_exchange_posttreat import (
            rebalance_ligand_exchange_charge,
            run_ligand_exchange_posttreatment,
        )
        syms, pts, x_ledger = run_ligand_exchange_posttreatment(syms, pts, cfg, struct, planes, cif_path)
        flat.extend(x_ledger)
        if x_ledger:
            from .main import _charge_with_ligand_exchange
            q_total = _charge_with_ligand_exchange(syms, cfg.charges, x_ledger)[3]
            if q_total != 0:
                print(
                    f"[ligand-exchange:charge-balance] residual Q={q_total:+d}; "
                    "running ligand-only exchange compensation"
                )
                syms, pts = rebalance_ligand_exchange_charge(
                    syms, pts, cfg, struct, planes, cif_path, x_ledger, verbose=verbose
                )
    ledgers["ligand_exchange"] = x_ledger

    # 6. Z-type ligands (MXn complexes on free surface anions; before L-type)
    zl_spec = getattr(pt, "z_type_ligands", None)
    if zl_spec is not None and zl_spec.enabled:
        from .z_type_ligand_posttreat import run_z_type_ligand_posttreatment
        syms, pts, zl_ledger = run_z_type_ligand_posttreatment(syms, pts, cfg, struct, planes)
        ledgers["z_type_ligand"] = zl_ledger
        flat.extend({"z_type_ligand": True, **entry} for entry in zl_ledger)

    # 7. Neutral (L-type) ligands, the last step
    neutral_spec = getattr(pt, "neutral_ligands", cfg.passivation.neutral_ligands)
    if neutral_spec is not None and neutral_spec.enabled:
        from .neutral_ligand_posttreat import run_neutral_ligand_posttreatment
        syms, pts = run_neutral_ligand_posttreatment(syms, pts, cfg, struct, planes)

    return syms, pts, ledgers


def xyz_text(syms: List[str], pts: NDArray[np.float64]) -> str:
    """The dot as ASE writes it (plain XYZ, full precision)."""
    from ase import Atoms
    from ase.io import write as ase_write

    out = io.StringIO()
    ase_write(out, Atoms(symbols=syms, positions=pts), format="xyz")
    return out.getvalue()


def run_from_xyz(args, cfg: Config) -> int:
    """``--from-xyz``: post-treat ``args.from_xyz``, write ``args.out`` and its manifest."""
    from ase.io import read as ase_read

    from .main import (
        _charge_with_ligand_exchange,
        _material_label_from_cif_or_structure,
        get_cluster_size_metrics,
    )

    struct = Structure.from_file(args.cif)
    cif_path = os.path.abspath(args.cif)
    atoms = ase_read(args.from_xyz, format="xyz")
    prefix = os.path.splitext(args.out)[0]
    if args.verbose:
        print(f"[from-xyz] {len(atoms)} atoms from {args.from_xyz}; post-treatment only (no Wulff cut)")

    syms, pts, ledgers = post_treat(
        atoms.get_chemical_symbols(), atoms.get_positions(), cfg, struct, cif_path,
        prefix=prefix, verbose=args.verbose, write_all=args.write_all,
    )

    final_pts = center_coords(pts) if args.center else pts
    with open(args.out, "w") as fh:
        fh.write(xyz_text(syms, final_pts))

    anion_lig = cfg.passivation.ligand or "Cl"
    material_label = _material_label_from_cif_or_structure(args.cif, struct, cfg.charges, anion_lig)
    extra = {
        "material": material_label,
        "from_xyz": os.path.abspath(args.from_xyz),
        "size_metrics": get_cluster_size_metrics(final_pts, atom_symbols=syms, material_name=material_label),
    }
    # Charge accounting as a build does: molecular anions of Z-type complexes count by
    # their molecular charge, like exchanged X-type ligands.
    charge_ledger = list(ledgers["ligand_exchange"]) + [
        {"z_type_ligand": entry["formula"], "charge": entry["charge"],
         "ignored_element_charge": entry["ignored_element_charge"]}
        for entry in ledgers.get("z_type_ligand", []) if "charge" in entry
    ]
    if charge_ledger:
        q_element, q_ignored, q_exchange, q_total = _charge_with_ligand_exchange(syms, cfg.charges, charge_ledger)
        extra.update({
            "element_symbol_charge": q_element,
            "ignored_exchanged_ligand_element_charge": q_ignored,
            "yaml_exchanged_ligand_charge": q_exchange,
            "total_charge": q_total,
            "ligand_exchange_charge_ledger": charge_ledger,
        })
    for key, name in (("z_type_displacement", "z_type_displacement_ledger"),
                      ("z_type_ligand", "z_type_ligand_ledger"),
                      ("alloying", "alloying_ledger"),
                      ("surface_reconstruction", "surface_reconstruction_ledger"),
                      ("neutral_exchange", "neutral_exchange_ledger")):
        if ledgers.get(key):
            extra[name] = ledgers[key]
    extra["post_treatment_ledger"] = ledgers["post_treatment"]
    if args.verbose:
        print(f"[from-xyz] Writing {args.out} ({len(syms)} atoms) and {prefix}.json")
    write_manifest(prefix, syms, cfg.charges, extra=extra)
    return 0
