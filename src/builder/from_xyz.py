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
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Structure

from .config import Config
from .facet_reconstruction import _native_facets_and_planes, reconstruct_polar_facets
from .facets import detect_facets_from_nc, expand_facets
from .io_utils import center_coords, write_manifest
from .passivation_iterative import charge_balance_iterative, derive_true_bulk_cn_from_cif
from .surface_metadata import (
    build_surface_metadata_document,
    resolve_prior_ligand_provenance,
    write_surface_metadata_json,
)


def _finalize_placement_ledger(entries: List[dict]) -> None:
    """Normalize raw atom_indices/host_env_indices/anchor_local_idx into the
    _0based field names build_surface_metadata_document expects. No index
    remap needed here (unlike main.py's _remap_placement_ledger), since
    run_from_xyz never reorders atoms before the final XYZ write."""
    for entry in entries:
        atoms = entry.get("atom_indices")
        if not atoms:
            entry["atom_indices_0based"] = None
            entry["bound_host_atom_indices_0based"] = []
            entry["anchor_atom_index_0based"] = None
            continue
        entry["atom_indices_0based"] = sorted(atoms)
        entry["anchor_atom_index_0based"] = min(atoms) + int(entry.get("anchor_local_idx", 0))
        entry["bound_host_atom_indices_0based"] = list(entry.get("host_env_indices") or [])


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
    extra_native_species: Iterable[str] | None = None,
) -> Tuple[List[str], NDArray[np.float64], Dict[str, object]]:
    """Run ``cfg.post_treatment`` on a passivated dot; return the dot and the step ledgers."""
    syms = list(syms)
    pts = np.asarray(pts, float)
    anion_lig = cfg.passivation.ligand or "Cl"
    facet_seeds = expand_facets(struct, cfg.seeds, proper_only=cfg.proper_only)
    pt = getattr(cfg, "post_treatment", None)
    flat: List[dict] = []
    ledgers: Dict[str, object] = {"post_treatment": flat}

    # For build_surface_metadata_document, same shape as main.py's fresh-build path.
    native_species = {str(s.specie.symbol) for s in struct.sites}
    if extra_native_species:
        native_species.update(extra_native_species)
    charge_balance_ledger: List[dict] = []
    prepass_lattice_substitution_ledger: List[dict] = []
    neutral_exchange_provenance_ledger: List[dict] = []
    neutral_ligand_ledger: List[dict] = []
    resolved_charge_ledger: List[dict] = []
    atom_provenance: List[dict] = []
    lattice_substitution_records: List[dict] = []

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
            ledger=charge_balance_ledger,
            lattice_substitution_ledger=prepass_lattice_substitution_ledger,
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
            lattice_substitution_records = recon_ledger.get("lattice_substitutions", [])

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
            native_species.update(entry["replacement"] for entry in alloy_ledger)
            atom_provenance.extend(
                {"position": pos, "symbol": entry["replacement"], "provenance": "alloying"}
                for entry in alloy_ledger
                for pos in entry.get("atom_positions", [])
            )
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
        syms, pts, z_ledger = run_z_type_displacement_posttreatment(
            syms, pts, cfg, struct, planes, cif_path,
            lattice_substitution_entries=lattice_substitution_records + prepass_lattice_substitution_ledger,
        )
        ledgers["z_type_displacement"] = z_ledger
        flat.extend({"z_type": True, **entry} for entry in z_ledger)

    # 4. Neutral exchange (after displacement, before X-type exchange)
    ne_spec = getattr(pt, "neutral_exchange", None)
    if ne_spec is not None and ne_spec.enabled:
        from .neutral_exchange_posttreat import run_neutral_exchange_posttreatment
        syms, pts, ne_ledger = run_neutral_exchange_posttreatment(
            syms, pts, cfg, struct, planes, cif_path,
            provenance_ledger=neutral_exchange_provenance_ledger,
        )
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
        syms, pts, x_ledger = run_ligand_exchange_posttreatment(
            syms, pts, cfg, struct, planes, cif_path,
            resolved_charge_ledger=resolved_charge_ledger,
        )
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
        _finalize_placement_ledger(x_ledger)
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
        syms, pts = run_neutral_ligand_posttreatment(
            syms, pts, cfg, struct, planes, ledger=neutral_ligand_ledger,
        )
        _finalize_placement_ledger(neutral_ligand_ledger)

    ledgers["facets"] = _facets
    ledgers["planes"] = planes
    ledgers["facet_seeds"] = facet_seeds
    ledgers["native_species"] = native_species
    ledgers["atom_provenance"] = atom_provenance
    ledgers["charge_balance_ledger"] = charge_balance_ledger
    ledgers["prepass_lattice_substitution_ledger"] = prepass_lattice_substitution_ledger
    ledgers["neutral_exchange_provenance"] = neutral_exchange_provenance_ledger
    ledgers["neutral_ligand"] = neutral_ligand_ledger
    ledgers["resolved_charge_ledger"] = resolved_charge_ledger
    return syms, pts, ledgers


def xyz_text(syms: List[str], pts: NDArray[np.float64]) -> str:
    """The dot as ASE writes it (plain XYZ, full precision)."""
    from ase import Atoms
    from ase.io import write as ase_write

    out = io.StringIO()
    ase_write(out, Atoms(symbols=syms, positions=pts), format="xyz")
    return out.getvalue()


def run_from_xyz(args, cfg: Config) -> int:
    """``--from-xyz``: post-treat ``args.from_xyz``, write ``args.out``, its manifest,
    and its surface-metadata JSON."""
    from ase.io import read as ase_read

    from .main import (
        _charge_with_ligand_exchange,
        _material_label_from_cif_or_structure,
        _post_treatment_summary,
        get_cluster_size_metrics,
    )

    struct = Structure.from_file(args.cif)
    cif_path = os.path.abspath(args.cif)
    atoms = ase_read(args.from_xyz, format="xyz")
    prefix = os.path.splitext(args.out)[0]
    if args.verbose:
        print(f"[from-xyz] {len(atoms)} atoms from {args.from_xyz}; post-treatment only (no Wulff cut)")

    extra_native_species: set = set()
    for extra_cif in (getattr(args, "extra_native_cif", None) or []):
        extra_struct = Structure.from_file(extra_cif)
        extra_native_species.update(str(s.specie.symbol) for s in extra_struct.sites)

    syms, pts, ledgers = post_treat(
        atoms.get_chemical_symbols(), atoms.get_positions(), cfg, struct, cif_path,
        prefix=prefix, verbose=args.verbose, write_all=args.write_all,
        extra_native_species=extra_native_species,
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

    _facets, planes = ledgers.get("facets"), ledgers.get("planes")
    if _facets and planes:
        ligand_provenance = [
            {**e, "type": e.get("type") or e.get("kind") or e.get("smiles"), "provenance": "x_type"}
            for e in ledgers.get("ligand_exchange", [])
        ] + [
            {**e, "type": e.get("type") or e.get("kind") or e.get("smiles"), "provenance": "l_type"}
            for e in ledgers.get("neutral_ligand", [])
        ] + [
            {**e, "provenance": "z_type_neutral_exchange"}
            for e in ledgers.get("neutral_exchange_provenance", [])
        ]
        z_type_relocated_records = [
            rec
            for entry in ledgers.get("z_type_displacement", [])
            for rec in entry.get("relocated_ligand_records", [])
        ]
        lattice_substitution_provenance = (
            ledgers.get("surface_reconstruction", {}).get("lattice_substitutions", [])
            + ledgers.get("prepass_lattice_substitution_ledger", [])
            + z_type_relocated_records
            + ledgers.get("charge_balance_ledger", [])
        )
        facet_seed_terminations = {
            (f.h, f.k, f.l): f.termination
            for f in ledgers.get("facet_seeds", [])
            if getattr(f, "termination", None)
        }
        material_provenance = None
        prior_metadata_supplied = False
        prior_xyz_path = getattr(args, "prior_xyz", None)
        prior_metadata_path = getattr(args, "prior_metadata", None)
        if prior_xyz_path and prior_metadata_path:
            with open(prior_metadata_path) as fh:
                prior_doc = json.load(fh)
            prior_atoms = ase_read(prior_xyz_path, format="xyz")
            prior_syms = prior_atoms.get_chemical_symbols()
            prior_pts = prior_atoms.get_positions()
            chained = resolve_prior_ligand_provenance(
                syms, final_pts, prior_syms, prior_pts,
                prior_doc.get("ligands"),
                alloying_provenance=ledgers.get("atom_provenance"),
            )
            already_covered = {
                i for entry in ligand_provenance for i in (entry.get("atom_indices_0based") or [])
            }
            ligand_provenance = ligand_provenance + [
                e for e in chained
                if not already_covered.intersection(e.get("atom_indices_0based") or [])
            ]
            material_provenance = [
                {"position": r["position"], "symbol": r["element"], "material": r["material"]}
                for r in prior_doc.get("qd_atoms", [])
                if r.get("material") is not None
            ] or None
            prior_metadata_supplied = True
        if args.verbose:
            print(f"[from-xyz] Writing surface metadata JSON to {prefix}_metadata.json")
        true_bulk_cn = derive_true_bulk_cn_from_cif(cif_path, cfg.charges)
        # --from-xyz doesn't know the original build's own params; a caller
        # that does can pass them through via --builder-params-json.
        builder_params: dict = {}
        builder_params_json_path = getattr(args, "builder_params_json", None)
        if builder_params_json_path:
            with open(builder_params_json_path) as fh:
                builder_params.update(json.load(fh))
        post_treatment_summary = _post_treatment_summary(cfg.post_treatment)
        if post_treatment_summary:
            builder_params["post_treatment"] = post_treatment_summary
        metadata_doc = build_surface_metadata_document(
            syms, final_pts, planes, _facets, cfg.charges, cfg.passivation.surf_tol,
            native_species=ledgers.get("native_species"),
            material=material_label,
            xyz_filename=os.path.basename(args.out),
            builder_params=builder_params or None,
            facet_seed_terminations=facet_seed_terminations,
            ligand_provenance=ligand_provenance,
            lattice_substitution_provenance=lattice_substitution_provenance,
            atom_provenance=ledgers.get("atom_provenance"),
            material_provenance=material_provenance,
            prepass_mode=cfg.passivation.prepass_mode,
            true_bulk_cn=true_bulk_cn,
            prior_metadata_supplied=prior_metadata_supplied,
        )
        write_surface_metadata_json(f"{prefix}_metadata.json", metadata_doc)
    return 0
