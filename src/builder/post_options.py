"""Post-treatment options and size descriptors of a dot (``nc-builder CIF RECIPE --options DOT.xyz``).

What a front end needs to offer post-treatments on a built dot, as one JSON file:

  z_type_options          Z-type displacement candidates (MXn units that can leave)
  alloying_options        inorganic substitutions
  x_type_options          charged-ligand exchange sites, per element
  l_type_options          neutral-ligand host sites, per native element
  z_type_ligand_options   free core anions that can take an MXn complex
  size_metrics            SAXS and volume diameters and aspect ratio

The Wulff planes are detected on the dot itself.  Each detector runs on a fresh copy
of the recipe, the structure and the dot; one that fails leaves its lists empty (the
surface detectors keep what they found before the failure) and reports to stderr.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from pymatgen.core import Structure

from .config import Config


def _read_xyz(path: str):
    from ase.io import read as ase_read

    atoms = ase_read(path, format="xyz")
    return list(atoms.get_chemical_symbols()), atoms.get_positions()


def _planes(syms, pts, cfg: Config, struct: Structure):
    from .facet_reconstruction import _native_facets_and_planes
    from .facets import detect_facets_from_nc, expand_facets

    facet_seeds = expand_facets(struct, cfg.seeds, proper_only=cfg.proper_only)
    anion_lig = cfg.passivation.ligand or "Cl"
    _facets, planes = _native_facets_and_planes(
        syms, pts, struct, cfg.charges, facet_seeds, anion_lig, cfg.passivation.surf_tol
    )
    if not planes:
        _facets, planes = detect_facets_from_nc(
            syms, pts, struct.lattice, cfg.charges, facet_seeds, cfg.passivation.surf_tol
        )
    return planes


def z_type_options(xyz_path: str, cfg: Config, cif_path: str) -> List[dict]:
    """Z-type displacement candidates on the dot; [] if detection fails."""
    try:
        from .z_type_displacement_posttreat import detect_z_type_displacement_options

        cfg = copy.deepcopy(cfg)
        struct = Structure.from_file(cif_path)
        syms, pts = _read_xyz(xyz_path)
        planes = _planes(syms, pts, cfg, struct)
        return detect_z_type_displacement_options(syms, pts, cfg, struct, planes, cif_path)
    except Exception as exc:
        print(f"[options] Z-type option detection failed: {exc!r}", file=sys.stderr)
        return []


def surface_options(xyz_path: str, cfg: Config, cif_path: str) -> Dict[str, list]:
    """Alloying, X-type, L-type and Z-type-ligand options on the dot."""
    out = {"alloying_options": [], "x_type_options": [], "l_type_options": [], "z_type_ligand_options": []}
    try:
        from .alloying_posttreat import detect_alloying_options
        from .analysis import bulk_cn_opposite_by_interior, coord_numbers_bipartite, derive_pair_cuts_from_cif
        from .ligand_exchange_posttreat import _bound_hosts
        from .neutral_ligand_posttreat import neutral_ligand_host_sites

        cfg = copy.deepcopy(cfg)
        struct = Structure.from_file(cif_path)
        syms, pts = _read_xyz(xyz_path)
        planes = _planes(syms, pts, cfg, struct)

        surface = np.zeros(len(pts), bool)
        for normal, d in planes:
            normal = np.asarray(normal, float)
            surface |= ((float(d) - pts @ normal) < float(cfg.passivation.surf_tol))
        pair_cuts = derive_pair_cuts_from_cif(cif_path, cfg.charges, safety=1.00)
        cn = coord_numbers_bipartite(syms, pts, cfg.charges, pair_cuts=pair_cuts)
        bulk_cn = bulk_cn_opposite_by_interior(
            syms, pts, planes, cfg.passivation.surf_tol, cfg.charges, pair_cuts=pair_cuts
        )
        native = {s.specie.symbol for s in struct.sites}
        non_native_coords = [pts[j] for j, s in enumerate(syms) if s not in native]

        def is_passivated(idx: int, cutoff: float = 3.5) -> bool:
            if not non_native_coords:
                return False
            dists = np.linalg.norm(np.asarray(non_native_coords) - pts[idx], axis=1)
            return bool(np.min(dists) < cutoff)

        out["alloying_options"] = detect_alloying_options(syms, pts, cfg, struct, planes)

        # X-type: free surface sites of native species (an undercoordinated surface atom
        # with no non-native neighbour within 3.5 Å and a bonded host), or every atom of
        # a placeholder species (Cl from the passivation).
        for sym in sorted(set(syms)):
            q = int(cfg.charges.get(sym, 0))
            if q == 0:
                continue
            if sym in native:
                available = 0
                for i, s in enumerate(syms):
                    if s != sym or not bool(surface[i]) or is_passivated(i):
                        continue
                    target_cn = int(bulk_cn.get(sym, 0))
                    if target_cn > 0 and int(cn[i]) >= target_cn:
                        continue
                    if not _bound_hosts(syms, pts, i, cfg.charges, pair_cuts):
                        continue
                    available += 1
                source = "native_surface"
            else:
                available = int(sum(1 for s in syms if s == sym))
                source = "placeholder"
            if available <= 0:
                continue
            out["x_type_options"].append({
                "replace": sym,
                "dummy": sym,
                "charge": q,
                "replace_charge": q,
                "ligand_charge": -1 if q < 0 else 1,
                "site_type": "anion" if q < 0 else "cation",
                "available_count": available,
                "source": source,
            })

        # L-type: the host sites the neutral-ligand pass itself would use.  A surface
        # atom that already carries X-type Cl but still has a free bond counts.
        l_type = []
        for sym in sorted(native):
            q = int(cfg.charges.get(sym, 0))
            if q == 0:
                continue
            site_type = "cation" if q > 0 else "anion"
            with contextlib.redirect_stdout(io.StringIO()):
                sites = neutral_ligand_host_sites(
                    syms, pts, cfg, struct, planes, native, target=site_type, target_symbol=sym,
                )
            if sites:
                l_type.append({
                    "element": sym,
                    "target_symbol": sym,
                    "target": site_type,
                    "site_type": site_type,
                    "available_count": len(sites),
                })
        out["l_type_options"] = sorted(l_type, key=lambda item: (item["site_type"], item["element"]))
        out["z_type_ligand_options"] = _z_type_ligand_options(syms, pts, cfg, struct, planes, native)
    except Exception as exc:
        print(f"[options] post-treatment option detection failed: {exc!r}", file=sys.stderr)
    return out


def _z_type_ligand_options(syms, pts, cfg: Config, struct, planes, native) -> List[dict]:
    """Free core anions that can take a Z-type complex, per anion species."""
    from .z_type_ligand_posttreat import z_type_ligand_host_sites

    cations = sorted(s for s in native if int(cfg.charges.get(s, 0)) > 0)
    ligand = cfg.passivation.ligand or "Cl"
    q_lig = abs(int(cfg.charges.get(ligand, -1))) or 1
    default_complex = ""
    if cations:
        q_m = int(cfg.charges[cations[0]])
        n = q_m // q_lig if q_m % q_lig == 0 else 0
        if n:
            default_complex = f"{cations[0]}{ligand}{n if n > 1 else ''}"
    options = []
    for sym in sorted(native):
        if int(cfg.charges.get(sym, 0)) >= 0:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            sites = z_type_ligand_host_sites(syms, pts, cfg, struct, planes, native, target_symbol=sym)
        if sites:
            options.append({
                "element": sym,
                "target_symbol": sym,
                "site_type": "anion",
                "available_count": len(sites),
                "default_complex": default_complex,
            })
    return options


def size_metrics(xyz_path: str, cif_paths: List[str]) -> Optional[dict]:
    """SAXS/volume size descriptors; bulk volumes from the CIFs, the first one winning
    an element they share (core first).  None if they cannot be computed."""
    try:
        from .library_record import bulk_volume_per_atom, size_descriptors

        bulk: Dict[str, float] = {}
        for cif in cif_paths:
            for el, v in bulk_volume_per_atom(str(cif)).items():
                bulk.setdefault(el, v)
        syms, pts = _read_xyz(xyz_path)
        return size_descriptors(syms, pts, set(bulk), bulk)
    except Exception as exc:
        print(f"[options] size metrics failed: {exc!r}", file=sys.stderr)
        return None


def run_options(args, cfg: Config) -> int:
    """``--options``: write the options and size metrics of ``args.options`` to ``<out stem>.json``."""
    out_path = os.path.splitext(args.out)[0] + ".json"
    result = {"z_type_options": z_type_options(args.options, cfg, args.cif)}
    result.update(surface_options(args.options, cfg, args.cif))
    result["size_metrics"] = size_metrics(args.size_xyz or args.options, args.size_cif or [args.cif])
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=1)
    if args.verbose:
        counts = ", ".join(f"{k}={len(v)}" for k, v in result.items() if isinstance(v, list))
        print(f"[options] {args.options}: {counts}; wrote {out_path}")
    return 0
