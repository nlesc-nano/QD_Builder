# src/builder/main.py
from __future__ import annotations
import os
import sys
import logging
import copy
import re
import dataclasses
from typing import List
import numpy as np 

try:
    from pymatgen.core import Structure
except ImportError:
    sys.exit("pip install pymatgen[matproj]")

from .config import build_parser, parse_yaml_config
from .nc_types import Config, Facet
from .facets import expand_facets, detect_facets_from_nc, halfspaces, scan_facets_from_cif
from .facets import unit_normal
from .geometry import apply_core_lattice_fit, build_nanocrystal, build_spherical_nanocrystal, dedupe_points, recut_with_planes, sphere_halfspaces
from .io_utils import write_xyz, write_manifest, center_coords
from .passivation_iterative import charge_balance_iterative
from .analysis import (
    bulk_cn_opposite_by_interior,
    coord_numbers_bipartite,
    derive_pair_cuts_from_cif,
    facet_atom_report,
    facet_families_overview,
    merge_pair_cuts_from_cifs,
    PairCuts,
)
from .cleanup import prune_low_coord_sites
from .stack import (
    build_layer_planes,
    cumulative_size_unit_cells,
    region_masks_from_layer_planes,
    relabel_regions_by_material,
    size_unit_cells_to_radius_aspect,
)
from .twin_workflow import apply_single_material_twins
from .facet_reconstruction import reconstruct_polar_facets

# A custom logging handler that forces a flush after every message.
class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

from .twinbound import apply_twins

MATERIAL_ELEMENTS = {
    "CSPBCL3": ["Cs", "Pb", "Cl"], "CSPBBR3": ["Cs", "Pb", "Br"], "CSPBI3": ["Cs", "Pb", "I"],
    "MAPBI3": ["Pb", "I"], "FAPBI3": ["Pb", "I"],
    "ZNS": ["Zn", "S"], "ZNSE": ["Zn", "Se"], "ZNTE": ["Zn", "Te"],
    "CDS": ["Cd", "S"], "CDSE": ["Cd", "Se"], "CDTE": ["Cd", "Te"],
    "HGS": ["Hg", "S"], "HGSE": ["Hg", "Se"], "HGTE": ["Hg", "Te"],
    "ALP": ["Al", "P"], "ALAS": ["Al", "As"], "ALSB": ["Al", "Sb"],
    "GAP": ["Ga", "P"], "GAAS": ["Ga", "As"], "GASB": ["Ga", "Sb"],
    "INP": ["In", "P"], "INAS": ["In", "As"], "INSB": ["In", "Sb"],
    "PBS": ["Pb", "S"], "PBSE": ["Pb", "Se"],
}


def get_cluster_size_metrics(coords_ang, atom_symbols=None, material_name=None):
    """Calculate an AABB-based radius/diameter for lattice-cut QDs."""
    coords = np.asarray(coords_ang, dtype=float)

    if atom_symbols is not None and material_name is not None:
        m_name = str(material_name).upper()
        if m_name in MATERIAL_ELEMENTS:
            core_elements = {el.lower() for el in MATERIAL_ELEMENTS[m_name]}
            core_coords = [
                coords[i]
                for i, sym in enumerate(atom_symbols)
                if str(sym).lower() in core_elements
            ]
            if core_coords:
                coords = np.asarray(core_coords, dtype=float)

    if len(coords) < 2:
        return {"R_eff_hull": 1.0, "diameter_hull": 2.0}

    spans = np.ptp(coords, axis=0)
    spans += 2.5
    avg_diameter = float(np.mean(spans))
    return {
        "R_eff_hull": avg_diameter / 2.0,
        "diameter_hull": avg_diameter,
    }


def _metrics_for_indices(pts, indices):
    if not indices:
        return None
    return get_cluster_size_metrics(np.asarray(pts, float)[indices])


def _ordered_xyz_view(symbols, pts, charges, ligand_symbol: str, *, layer_planes=None, pts_for_layers=None):
    pts = np.asarray(pts, float)
    pts_ref = np.asarray(pts_for_layers, float) if pts_for_layers is not None else pts
    if layer_planes is None:
        order = sorted(
            range(len(symbols)),
            key=lambda i: (
                1 if symbols[i] == ligand_symbol else 0,
                0 if charges.get(symbols[i], 0) > 0 else 1,
                str(symbols[i]),
                i,
            ),
        )
        return [symbols[i] for i in order], pts[order]

    region_masks = region_masks_from_layer_planes(pts_ref, layer_planes)

    def layer_of(i: int) -> int:
        for k, mask in enumerate(region_masks):
            if bool(mask[i]):
                return k
        return len(region_masks)

    def role_rank(i: int) -> int:
        if symbols[i] == ligand_symbol:
            return 2
        return 0 if charges.get(symbols[i], 0) > 0 else 1

    order = sorted(
        range(len(symbols)),
        key=lambda i: (
            len(region_masks) + 1 if symbols[i] == ligand_symbol else layer_of(i),
            role_rank(i),
            str(symbols[i]),
            i,
        ),
    )
    return [symbols[i] for i in order], pts[order]


def _stack_size_metrics(symbols, pts, charges, ligand_symbol: str, materials_cfg, layer_planes):
    pts = np.asarray(pts, float)
    region_masks = region_masks_from_layer_planes(pts, layer_planes)
    out = {}
    for k, (m, mask) in enumerate(zip(materials_cfg, region_masks)):
        indices = [
            i for i, keep in enumerate(mask)
            if keep and symbols[i] != ligand_symbol and charges.get(symbols[i], 0) != 0
        ]
        label = "core" if k == 0 else f"shell{k}"
        out[label] = {
            "name": m.name,
            "size_metrics": _metrics_for_indices(pts, indices),
        }
    native_indices = [
        i for i, s in enumerate(symbols)
        if s != ligand_symbol and charges.get(s, 0) != 0
    ]
    out["overall"] = {
        "name": "overall",
        "size_metrics": _metrics_for_indices(pts, native_indices),
    }
    return out


def _print_stack_summary(symbols, charges, materials_cfg, ligand_symbol: str, *, pts=None, layer_planes=None):
    """
    Print per-layer counts for core/shell systems.
    Uses the element sets derived from each material's CIF structure.
    """
    from collections import Counter

    cnt = Counter(symbols)
    Q_total = sum(charges.get(el, 0) * v for el, v in cnt.items())
    region_masks = None
    if pts is not None and layer_planes is not None:
        region_masks = region_masks_from_layer_planes(np.asarray(pts, float), layer_planes)

    print("\n### CORE–SHELL SUMMARY ###")

    for layer_idx, m in enumerate(materials_cfg):
        label = "CORE" if layer_idx == 0 else f"SHELL {layer_idx}"
        # parse element set directly from the CIF
        try:
            struct = Structure.from_file(m.cif)
            elems = []
            for site in struct.sites:
                sym = str(site.specie.symbol)
                if sym not in elems:
                    elems.append(sym)
        except Exception:
            elems = []

        layer_symbols = symbols
        if region_masks is not None and layer_idx < len(region_masks):
            layer_symbols = [s for s, keep in zip(symbols, region_masks[layer_idx]) if keep]
        layer_cnt = Counter(layer_symbols)

        print(f"\n{label} ({m.name}):")
        for el in elems:
            if el == ligand_symbol:
                continue
            n = layer_cnt.get(el, 0)
            print(f"  number of {el}: {n}")

    # ligand placeholders (global)
    n_ligand = cnt.get(ligand_symbol, 0)
    if n_ligand:
        print(f"\nLigand placeholders ({ligand_symbol}): {n_ligand}")

    print(f"\nTotal Charge = {Q_total:+d}")


def _print_single_material_summary(symbols, charges, ligand_symbol: str, title: str = None):
    from collections import Counter
    cnt = Counter(symbols)

    n_ligand  = cnt.get(ligand_symbol, 0)

    # total Q
    Q = sum(charges.get(el, 0) * v for el, v in cnt.items())

    if title:
        print(f"\n### {title} ###")
    # print cations first, then anions, then ligands
    for el, q in charges.items():
        if el == ligand_symbol:
            continue
        if cnt.get(el, 0) > 0 and q > 0:
            print(f"number of {el}: {cnt[el]}")
    for el, q in charges.items():
        if el == ligand_symbol:
            continue
        if cnt.get(el, 0) > 0 and q < 0:
            print(f"number of {el}: {cnt[el]}")
    if n_ligand:
        print(f"number of ligand placeholder {ligand_symbol}: {n_ligand}")
    # also print per-element lines in a stable order
    print(f"\nTotal Charge = {Q:+d}")


def construction_origin_shift(struct: Structure, spec: dict | None) -> np.ndarray | None:
    """
    Return a Cartesian shift r0 for construction-time cuts n.(r-r0) <= d.
    """
    if not spec:
        return None

    if "cartesian_shift" in spec:
        shift = np.asarray(spec["cartesian_shift"], float)
        if shift.shape != (3,):
            raise ValueError("construction_origin.cartesian_shift must have three values")
        return shift

    if "fractional_shift" in spec:
        frac = np.asarray(spec["fractional_shift"], float)
        if frac.shape != (3,):
            raise ValueError("construction_origin.fractional_shift must have three values")
        return frac @ struct.lattice.matrix

    species = spec.get("center_on_species", spec.get("center_on"))
    if species:
        if isinstance(species, (list, tuple)):
            raise ValueError(
                "construction_origin.center_on_species lists are handled as multiple "
                "variants; pass a single species to construction_origin_shift."
            )
        species = str(species)
        sites = [site for site in struct.sites if site.specie.symbol == species]
        if not sites:
            raise ValueError(f"construction_origin species '{species}' not found in CIF")
        coords = np.asarray([site.coords for site in sites], float)
        return coords[np.argmin(np.linalg.norm(coords, axis=1))]

    return None


def _is_all_centers_spec(spec: dict | None) -> bool:
    if not spec:
        return False
    species = spec.get("center_on_species", spec.get("center_on"))
    return isinstance(species, str) and species.strip().lower() == "all"


def _center_species_list(spec: dict | None) -> List[str] | None:
    if not spec:
        return None
    species = spec.get("center_on_species", spec.get("center_on"))
    if species is None:
        return None
    if isinstance(species, str):
        raw = species.strip()
        if raw.lower() == "all":
            return None
        # Be tolerant of user-written "(Cs, Pb)" or "Cs, Pb" in addition to
        # proper YAML lists like [Cs, Pb].
        compact = raw.strip("()[]")
        if "," in compact:
            out = [part.strip().strip("'\"") for part in compact.split(",") if part.strip()]
            if not out:
                raise ValueError("construction_origin.center_on_species list cannot be empty")
            if any(item.lower() == "all" for item in out):
                raise ValueError("construction_origin.center_on_species cannot mix 'all' with a list")
            return out
        return [raw]
    if isinstance(species, (list, tuple)):
        out = [str(item) for item in species]
        if not out:
            raise ValueError("construction_origin.center_on_species list cannot be empty")
        if any(item.strip().lower() == "all" for item in out):
            raise ValueError("construction_origin.center_on_species cannot mix 'all' with a list")
        return out
    raise TypeError("construction_origin.center_on_species must be a string or list of strings")


def _safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "center"


def _format_rep_value(value) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = [float(v) for v in value]
        if vals and all(abs(v - vals[0]) < 1e-12 for v in vals):
            return f"{vals[0]:g}"
        return "x".join(f"{v:g}" for v in vals)
    return f"{float(value):g}"


def _material_label_from_cif_or_structure(
    cif_path: str,
    struct: Structure,
    charges,
    ligand_symbol: str | None,
) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "", os.path.splitext(os.path.basename(cif_path))[0]).upper()
    for key in sorted(MATERIAL_ELEMENTS, key=len, reverse=True):
        if key in stem:
            return "".join(MATERIAL_ELEMENTS[key])

    elems = _native_species_in_cif_order(struct, charges, ligand_symbol)
    if elems:
        return "".join(elems)

    seen = []
    for site in struct.sites:
        sym = str(site.specie.symbol)
        if sym not in seen:
            seen.append(sym)
    return "".join(seen) or "Material"


def _native_species_in_cif_order(struct: Structure, charges, ligand_symbol: str | None) -> List[str]:
    out: List[str] = []
    for site in struct.sites:
        sym = str(site.specie.symbol)
        if sym == ligand_symbol:
            continue
        if charges.get(sym, 0) == 0:
            continue
        if sym not in out:
            out.append(sym)
    return out


def _center_variants(struct: Structure, cfg: Config, ligand_symbol: str | None):
    if cfg.construction_origin and not _is_all_centers_spec(cfg.construction_origin):
        species_list = _center_species_list(cfg.construction_origin)
        if species_list:
            variants = []
            for species in species_list:
                variants.append((_safe_label(species), construction_origin_shift(
                    struct, {"center_on_species": species}
                )))
            return variants
        return [("custom", construction_origin_shift(struct, cfg.construction_origin))]

    variants = []
    for sym in _native_species_in_cif_order(struct, cfg.charges, ligand_symbol):
        variants.append((_safe_label(sym), construction_origin_shift(
            struct, {"center_on_species": sym}
        )))
    return variants or [("origin", None)]


def _variant_out_path(
    out_path: str,
    label: str,
    multiple: bool,
    *,
    args=None,
    material_label: str | None = None,
) -> str:
    root, ext = os.path.splitext(out_path)
    if not ext:
        ext = ".xyz"

    if args is not None and args.size_unit_cells is not None:
        out_dir = os.path.dirname(out_path)
        stem = (
            f"{_safe_label(material_label or 'Material')}"
            f"_c{_safe_label(label)}"
            f"_rep{_format_rep_value(args.size_unit_cells)}"
        )
        return os.path.join(out_dir, f"{stem}{ext}")

    if not multiple:
        return out_path
    return f"{root}_{_safe_label(label)}{ext}"


def _radius_from_size_args(struct: Structure, args) -> float:
    if args.size_unit_cells is not None:
        if isinstance(args.size_unit_cells, (list, tuple, np.ndarray)):
            radius, _aspect = size_unit_cells_to_radius_aspect(struct, tuple(float(x) for x in args.size_unit_cells))
            return radius
        a, b, c = struct.lattice.matrix
        return float(args.size_unit_cells) * min(
            float(np.linalg.norm(a)),
            float(np.linalg.norm(b)),
            float(np.linalg.norm(c)),
        )
    if args.radius is None:
        raise SystemExit("Please pass either -r/--radius or --size-unit-cells.")
    return float(args.radius)


def _effective_single_size_unit_cells(cfg: Config, args):
    if args.size_unit_cells is not None:
        x = float(args.size_unit_cells)
        return (x, x, x)
    return cfg.size_unit_cells


def _validate_ligand_not_native(cif_paths: List[str], ligand: str | None) -> None:
    if not ligand:
        return
    for cif_path in cif_paths:
        struct = Structure.from_file(cif_path)
        native = {str(site.specie.symbol) for site in struct.sites}
        if ligand in native:
            raise SystemExit(
                f"passivation.ligand='{ligand}' is present as a native species in {cif_path}. "
                "Use a distinct placeholder ligand symbol and assign its charge in YAML."
            )


def _surface_charge_for_signed_hkl(struct: Structure, hkl, charges) -> int:
    n = unit_normal(struct, hkl)
    coords = np.asarray([site.coords for site in struct.sites], float)
    proj = coords @ n
    top = np.max(proj)
    tol = max(1e-4, 1e-3 * max(1.0, abs(top)))
    return int(sum(
        int(charges.get(str(site.specie.symbol), 0))
        for site, p in zip(struct.sites, proj)
        if top - p <= tol
    ))


def _resolve_facet_terminations(struct: Structure, seeds: List[Facet], charges) -> List[Facet]:
    resolved: List[Facet] = []
    for f in seeds:
        term = getattr(f, "termination", None)
        if not term:
            resolved.append(f)
            continue

        hkl = (abs(int(f.h)), abs(int(f.k)), abs(int(f.l)))
        if hkl == (0, 0, 0):
            resolved.append(f)
            continue
        candidates = [hkl, (-hkl[0], -hkl[1], -hkl[2])]
        scored = [(cand, _surface_charge_for_signed_hkl(struct, cand, charges)) for cand in candidates]
        if term == "cation_rich":
            chosen, _q = max(scored, key=lambda rec: rec[1])
        elif term == "anion_rich":
            chosen, _q = min(scored, key=lambda rec: rec[1])
        else:
            chosen = (f.h, f.k, f.l)
        resolved.append(Facet(chosen[0], chosen[1], chosen[2], f.gamma, termination=term, scope=getattr(f, "scope", "family")))
    return resolved


def _run_passivation_and_write_outputs(
    syms,
    pts,
    *,
    args,
    cfg: Config,
    anion_lig: str,
    planes,
    cif_path: str,
    struct: Structure,
    facet_seeds,
    material_label: str | None = None,
    construction_radius_override: float | None = None,
    pair_cuts_override: PairCuts | None = None,
    output_layer_planes=None,
    output_materials=None,
):
    prefix = os.path.splitext(args.out)[0]

    if args.write_all:
        if args.verbose:
            print(f"\n[7] Writing initial cut XYZ to {prefix}_cut.xyz")
        write_xyz(f"{prefix}_cut.xyz", syms, center_coords(pts) if args.center else pts)

    if args.verbose:
        print("\n[8] Balancing charge stepwise (outer anions first; then add/remove ligands if needed)...")
    syms, pts = charge_balance_iterative(
        syms, pts,
        cfg.charges, anion_lig,
        verbose=args.verbose,
        planes=planes,
        surf_tol=cfg.passivation.surf_tol,
        cif_path=cif_path,
        positive_q_strategy=args.positive_q_mode,
        write_all=args.write_all,
        prefix=prefix,
        experimental_exhausted_positive_q_fallback=bool(
            getattr(cfg, "experimental", {}).get("exhausted_positive_q_fallback", False)
        ),
        pair_cuts_override=pair_cuts_override,
    )

    if cfg.facet_reconstruction.enabled:
        syms, pts = reconstruct_polar_facets(
            syms,
            pts,
            struct=struct,
            facet_seeds=facet_seeds,
            charges=cfg.charges,
            ligand=anion_lig,
            surf_tol=cfg.passivation.surf_tol,
            cif_path=cif_path,
            spec=cfg.facet_reconstruction,
            charge_balance_fn=charge_balance_iterative,
            verbose=args.verbose,
            write_all=args.write_all,
            prefix=prefix,
        )

    if args.verbose:
        print(f"\n[11] Writing final XYZ to {args.out}")
    final_pts = center_coords(pts) if args.center else pts
    out_syms, out_pts = _ordered_xyz_view(
        syms,
        final_pts,
        cfg.charges,
        anion_lig,
        layer_planes=output_layer_planes,
        pts_for_layers=pts,
    )
    write_xyz(args.out, out_syms, out_pts)
    construction_radius = (
        float(construction_radius_override)
        if construction_radius_override is not None
        else _radius_from_size_args(struct, args)
    )
    construction_diameter = 2.0 * construction_radius
    size_metrics = get_cluster_size_metrics(
        final_pts,
        atom_symbols=syms,
        material_name=material_label,
    )
    stack_sizes = None
    if output_layer_planes is not None and output_materials is not None:
        stack_sizes = _stack_size_metrics(
            syms,
            pts,
            cfg.charges,
            anion_lig,
            output_materials,
            output_layer_planes,
        )
    if args.verbose:
        print(
            "[size] "
            f"R_eff_hull={size_metrics['R_eff_hull']:.3f} Å, "
            f"diameter_hull={size_metrics['diameter_hull']:.3f} Å"
        )
        print(
            "[size-summary] "
            f"input_estimate: R={construction_radius:.3f} Å, D={construction_diameter:.3f} Å | "
            f"final_obtained: R={size_metrics['R_eff_hull']:.3f} Å, "
            f"D={size_metrics['diameter_hull']:.3f} Å"
        )
        if stack_sizes:
            print("[stack-size-summary] native-only final sizes:")
            for key, rec in stack_sizes.items():
                metrics = rec.get("size_metrics")
                if metrics is None:
                    print(f"  {key} ({rec.get('name')}): empty")
                    continue
                print(
                    f"  {key} ({rec.get('name')}): "
                    f"R={metrics['R_eff_hull']:.3f} Å, "
                    f"D={metrics['diameter_hull']:.3f} Å"
                )

    if args.verbose:
        print(f"[12] Writing JSON manifest to {prefix}.json")
    extra = {
        "material": material_label,
        "size_unit_cells": args.size_unit_cells,
        "construction_radius_ang": construction_radius,
        "construction_diameter_ang": construction_diameter,
        "size_metrics": size_metrics,
    }
    if stack_sizes:
        extra["stack_size_metrics"] = stack_sizes
    write_manifest(prefix, syms, cfg.charges, extra=extra)

    return syms, pts


def _print_surface_reports(syms, pts, planes, facets, charges, surf_tol: float, *, label: str, verbose: bool):
    if verbose:
        print(f"\n[6] Surface atom and CN reports{label}:")
    facet_families_overview(syms, pts, planes, facets, surf_tol=surf_tol, charges=charges)
    facet_atom_report(syms, pts, planes, facets, surf_tol=surf_tol, charges=charges)


def _native_surface_view(syms, pts, ligand_symbol: str):
    indices = [i for i, s in enumerate(syms) if s != ligand_symbol]
    return [syms[i] for i in indices], pts[indices], indices


def _native_facet_memberships(native_pts, planes, surf_tol: float):
    memberships = [[] for _ in range(len(native_pts))]
    for fid, (n, d) in enumerate(planes):
        shell = np.where((d - native_pts @ n) < surf_tol)[0]
        for i in shell:
            memberships[int(i)].append(fid)
    return memberships


def _print_native_facet_families_overview(native_syms, native_pts, planes, facets, charges, surf_tol: float):
    def fam_key(h, k, l):
        return tuple(sorted((abs(h), abs(k), abs(l))))

    families = {}
    for fid, f in enumerate(facets):
        families.setdefault(fam_key(f.h, f.k, f.l), []).append(fid)

    print("\n=== FACET FAMILIES OVERVIEW ===")
    for fam, ids in sorted(families.items()):
        label = "".join(str(x) for x in fam)
        print(f"Family {label}: {len(ids)} faces")

    print("\nFacet charges (native scaffold shell only):")
    for fid, (n, d) in enumerate(planes):
        shell = np.where((d - native_pts @ n) < surf_tol)[0]
        q_shell = int(sum(charges.get(native_syms[int(i)], 0) for i in shell))
        f = facets[fid]
        label = f"({f.h}{f.k}{f.l})"
        richness = "cation-rich" if q_shell > 0 else ("anion-rich" if q_shell < 0 else "neutral")
        print(f"  {label:>8s}  #atoms={len(shell):3d}  Q={q_shell:+d}  {richness}")


def _print_native_facet_atom_report_with_full_cn(
    syms,
    pts,
    native_syms,
    native_pts,
    native_indices,
    planes,
    facets,
    charges,
    surf_tol: float,
    pair_cuts,
):
    cn_full = coord_numbers_bipartite(syms, pts, charges, pair_cuts=pair_cuts)
    bulk = bulk_cn_opposite_by_interior(
        syms,
        pts,
        planes,
        surf_tol,
        charges,
        pair_cuts=pair_cuts,
    )
    memberships = _native_facet_memberships(native_pts, planes, surf_tol)

    outer_thr = 0.35 * surf_tol
    subl_thr = 1.20 * surf_tol

    print("\n=== PER-FACET SURFACE ATOMS (DETAILED; native facets, full CN incl. ligands) ===")
    print("Legend: CN counts Cd/Se plus attached ligands; target='*' if native anion is unsaturated.")
    for fid, (n, d) in enumerate(planes):
        shell = np.where((d - native_pts @ n) < surf_tol)[0]
        if not shell.size:
            continue

        f = facets[fid]
        print(f"\nFacet ({f.h}{f.k}{f.l})  #atoms={len(shell)}")
        print(" idx  el         x(Å)        y(Å)        z(Å)   CN/bulk  role     layer      deficit  type    tgt")
        for local_i in sorted(shell.tolist(), key=lambda j: (native_syms[j], native_indices[j])):
            global_i = native_indices[local_i]
            x, y, z = pts[global_i]
            s = syms[global_i]
            depth = d - float(np.dot(native_pts[local_i], n))

            m = len(memberships[local_i])
            role = "unique" if m == 1 else ("edge" if m == 2 else "vertex")
            layer = "outer" if depth < outer_thr else ("sublayer" if depth < subl_thr else "")
            deficit = max(0, int(bulk.get(s, 0)) - int(cn_full[global_i]))
            q = int(charges.get(s, 0))
            etype = "anion" if q < 0 else ("cation" if q > 0 else "neutral")
            target = "*" if (q < 0 and deficit > 0) else ""
            print(
                f"{global_i:4d}  {s:>2s}  {x:10.4f}  {y:10.4f}  {z:10.4f}"
                f"   {int(cn_full[global_i])}/{int(bulk.get(s, 0))}    {role:6s}"
                f"  {layer:8s}   {deficit:7d}  {etype:7s}  {target:3s}"
            )


def _print_native_surface_reports_after_charge_balance(
    syms,
    pts,
    *,
    struct: Structure,
    charges,
    facet_seeds,
    surf_tol: float,
    ligand_symbol: str,
    cif_path: str,
):
    native_syms, native_pts, native_indices = _native_surface_view(syms, pts, ligand_symbol)
    facets, planes = detect_facets_from_nc(
        native_syms,
        native_pts,
        struct.lattice,
        charges,
        facet_seeds,
        surf_tol,
    )
    print("\n[8b] Native surface reports after charge balance, before Lannoo reconstruction:")
    print("     Facets are defined by Cd/Se only; CN is computed on the full structure including Cl.")
    pair_cuts = derive_pair_cuts_from_cif(cif_path, charges, safety=1.00)
    _print_native_facet_families_overview(native_syms, native_pts, planes, facets, charges, surf_tol)
    _print_native_facet_atom_report_with_full_cn(
        syms,
        pts,
        native_syms,
        native_pts,
        native_indices,
        planes,
        facets,
        charges,
        surf_tol,
        pair_cuts,
    )


def _prune_before_facet_detection(syms, pts, *, args):
    if not args.prune_mono:
        return syms, pts
    if args.verbose:
        print("\n[4b] Pruning low-coordination atoms (pre-facet detection)...")
    syms, pts, n_removed, n_pass = prune_low_coord_sites(
        syms, pts, min_cn=args.prune_min_cn, max_passes=args.prune_passes, verbose=args.verbose
    )
    if args.verbose:
        print(f"    - Pruned {n_removed} atoms in {n_pass} pass(es); remaining {len(syms)} atoms")
    return syms, pts


def _detect_facets_and_report(
    syms,
    pts,
    struct: Structure,
    charges,
    facet_seeds,
    surf_tol: float,
    *,
    verbose: bool,
    label: str,
):
    if verbose:
        print(f"\n[5] Detecting actual exposed facets{label}...")
    facets, planes = detect_facets_from_nc(syms, pts, struct.lattice, charges, facet_seeds, surf_tol)
    if verbose:
        print(f"    - Detected {len(facets)} facets")
    _print_surface_reports(
        syms,
        pts,
        planes,
        facets,
        charges,
        surf_tol,
        label=label,
        verbose=verbose,
    )
    return facets, planes


def _setup_unbuffered_logging_from_env() -> None:
    if not os.environ.get("QD_BUILDER_UNBUFFERED"):
        return

    handler = FlushingStreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(handler)

    import io

    class _LoggerWriter(io.TextIOBase):
        def __init__(self, log_fn):
            self._log_fn = log_fn

        def write(self, s):
            for part in s.splitlines():
                part = part.rstrip("\n")
                if part:
                    self._log_fn(part)
            return len(s)

        def flush(self):
            pass

    sys.stdout = _LoggerWriter(root_logger.info)
    sys.stderr = _LoggerWriter(root_logger.warning)


def main(argv: List[str] | None = None) -> int:
    _setup_unbuffered_logging_from_env()
    
    # ------------------------------------
    p = build_parser()
    args = p.parse_args(argv)
    cfg: Config = parse_yaml_config(args.yaml)
    # --- Passivation ligand selection (backward compatible) ---
    pass_cfg = getattr(cfg, "passivation", None)
    if pass_cfg:
        # legacy: 'ligand' means anion ligand
        anion_lig = getattr(pass_cfg, "anion_ligand", getattr(pass_cfg, "ligand", None))
        cation_lig = getattr(pass_cfg, "cation_ligand", None) or "Rb"
    else:
        anion_lig, cation_lig = None, None

    # Ensure ligand charges are present (do it in-place so downstream uses cfg.charges)
    if anion_lig and (anion_lig not in cfg.charges):
        cfg.charges[anion_lig] = -1.0
    if cation_lig and (cation_lig not in cfg.charges):
        cfg.charges[cation_lig] = +1.0


    # ----- Optional facet scan (universal; runs before build) -----
    if args.scan_facets:
        if cfg.mode == "stack":
            for m in cfg.materials:
                rows = scan_facets_from_cif(m.cif if cfg.mode=="stack" else args.cif, cfg.charges, max_index=args.scan_max_index, min_slab_size=args.scan_slab_size, min_vacuum_size=args.scan_vacuum_size, n_shifts=args.scan_shifts)

                print(f"\n[facet-scan] {m.name} ({m.cif}) — |h|,|k|,|l| ≤ {args.scan_max_index}")
                for r in rows:
                    pol = "polar" if r["polar_any"] else "non-polar"
                    pc = r["polar_count"]; nt = r["n_terms_checked"]
                    print(f"  hkl={r['hkl']!s:>10}  fam={r['family']:<6}  {pol:9}  ({pc}/{nt} terminations polar)")
        else:
            rows = scan_facets_from_cif(m.cif if cfg.mode=="stack" else args.cif, cfg.charges, max_index=args.scan_max_index, min_slab_size=args.scan_slab_size, min_vacuum_size=args.scan_vacuum_size, n_shifts=args.scan_shifts)

            print(f"\n[facet-scan] single-mode ({args.cif}) — |h|,|k|,|l| ≤ {args.scan_max_index}")
            for r in rows:
                pol = "polar" if r["polar_any"] else "non-polar"
                pc = r["polar_count"]; nt = r["n_terms_checked"]
                print(f"  hkl={r['hkl']!s:>10}  fam={r['family']:<6}  {pol:9}  ({pc}/{nt} terminations polar)")
        # continue with the normal run afterwards
    
    if cfg.mode == "stack":
        # Multi-material: YAML drives CIFs; CLI radius sets the outer cut.
        _validate_ligand_not_native([m.cif for m in cfg.materials], anion_lig)
        if args.verbose:
            print("\n[STACK] Multi-material mode detected from YAML.")
            print(f"  - Regions: {[m.name for m in cfg.materials]}")
            print(f"  - Proper rotations only: {bool(cfg.proper_only)}")
            print(f"  - Pair opposites: {bool(cfg.pair_opposites)}")
    
        if len(cfg.materials) < 2:
            raise SystemExit("Stack mode requires at least two materials (core first, then shell).")
    
        cumulative_sizes = cumulative_size_unit_cells(cfg.materials)
        resolved_materials = []
        for material in cfg.materials:
            sm = Structure.from_file(material.cif)
            resolved_materials.append(dataclasses.replace(
                material,
                seeds=_resolve_facet_terminations(sm, material.seeds, cfg.charges),
            ))

        # === MINIMAL CHANGE: cut once on OUTERMOST shell, then relabel regions ===
        # Build OUTERMOST cut
        outer_idx = len(resolved_materials) - 1
        active_layer_indices = list(range(len(resolved_materials)))
        if cumulative_sizes is not None:
            outer_idx = 0
            active_layer_indices = []
            prev = np.zeros(3, dtype=float)
            for idx, size in enumerate(cumulative_sizes):
                arr = np.asarray(size, dtype=float)
                if not np.allclose(arr, prev, atol=1e-12):
                    outer_idx = idx
                    active_layer_indices.append(idx)
                prev = arr
            if not active_layer_indices:
                raise SystemExit("At least one material must have nonzero size_unit_cells in stack mode.")
        outer_cfg = resolved_materials[outer_idx]
        struct_outer = Structure.from_file(outer_cfg.cif)
        if cumulative_sizes is None:
            radius_eff = _radius_from_size_args(struct_outer, args)
            aspect_outer = outer_cfg.aspect
        else:
            radius_eff, aspect_outer = size_unit_cells_to_radius_aspect(struct_outer, cumulative_sizes[outer_idx])
        if outer_cfg.shape_mode == "sphere":
            facets_outer = []
            syms, pts, _ = build_spherical_nanocrystal(
                struct_outer,
                radius_eff,
                n_planes=outer_cfg.sphere_planes,
            )
        else:
            facets_outer = expand_facets(struct_outer, outer_cfg.seeds, proper_only=cfg.proper_only)
            syms, pts, _ = build_nanocrystal(struct_outer, facets_outer, radius_eff, aspect=aspect_outer)
        syms, pts = dedupe_points(syms, pts, tol=1e-3)
        if args.verbose:
            if cumulative_sizes is not None:
                print("    - Cumulative size_unit_cells:")
                prev_size = np.zeros(3, dtype=float)
                for idx, (m, size) in enumerate(zip(resolved_materials, cumulative_sizes)):
                    if idx > 0 and np.allclose(size, prev_size, atol=1e-12):
                        print(
                            f"      {m.name}: size={tuple(f'{x:g}' for x in size)} "
                            "-> zero-thickness layer; reuses previous boundary"
                        )
                        prev_size = np.asarray(size, dtype=float)
                        continue
                    r_i, aspect_i = size_unit_cells_to_radius_aspect(Structure.from_file(m.cif), size)
                    print(
                        f"      {m.name}: size={tuple(f'{x:g}' for x in size)} "
                        f"-> radius={r_i:.6f} Å, aspect={tuple(round(x, 4) for x in aspect_i)}"
                    )
                    prev_size = np.asarray(size, dtype=float)
            if outer_cfg.shape_mode == "sphere":
                print(
                    f"    - Outermost spherical cut atoms: {len(syms)} "
                    f"(from {outer_cfg.name}, planes={outer_cfg.sphere_planes})"
                )
            else:
                print(f"    - Outermost cut atoms: {len(syms)} (from {outer_cfg.name})")
    
        layer_planes = build_layer_planes(
            resolved_materials,
            radius_eff,
            cfg.proper_only,
            cumulative_sizes=cumulative_sizes,
        )
        region_masks = region_masks_from_layer_planes(pts, layer_planes)
        syms = relabel_regions_by_material(
            syms,
            region_masks,
            resolved_materials,
            cfg.charges,
            getattr(cfg.passivation, "ligand", None),
            verbose=args.verbose,
        )

        if args.core_lattice_fit:
            if args.verbose:
                print("\n[3b] Applying core lattice fit with smooth interface blend...")
            try:
                core_struct = Structure.from_file(resolved_materials[0].cif)
                shell_struct = Structure.from_file(resolved_materials[-1].cif)
                pts = apply_core_lattice_fit(
                    pts,
                    region_masks[0],
                    layer_planes[0],
                    shell_struct.lattice.matrix,
                    core_struct.lattice.matrix,
                    strain_width=args.core_strain_width,
                    center_mode=args.core_center,
                )
                if args.verbose:
                    print(
                        f"    - Core lattice fit applied "
                        f"(width={args.core_strain_width:.3f} Å, center={args.core_center})"
                    )
            except np.linalg.LinAlgError:
                print("WARNING: shell lattice not invertible; skipping core lattice fit.")
    
        if args.verbose:
            print(f"\n[4] Composite particle atoms: {len(syms)}")
    
        # --- OPTIONAL TWIN BOUNDARIES (stack mode) ---
        if getattr(cfg, "twins", None):
            if args.verbose:
                print("\n[3a] Applying twin boundary transformations (stack mode)...")
            # Use OUTERMOST shell lattice for twins and recut
            outer_shell = resolved_materials[outer_idx]
            shell_struct = Structure.from_file(outer_shell.cif)
    
            # (1) Apply mirrors
            pts = apply_twins(
                pts,
                shell_struct.lattice.matrix,
                cfg.twins,
                default_origin="center",
                species=syms,
                charges=cfg.charges,
            )
    
            # (2) Recut with OUTER (outermost shell) Wulff planes
            if outer_shell.shape_mode == "sphere":
                planes_outer = sphere_halfspaces(radius_eff, n_planes=outer_shell.sphere_planes)
            else:
                facets_shell = expand_facets(shell_struct, outer_shell.seeds, proper_only=cfg.proper_only)
                planes_outer = halfspaces(shell_struct, facets_shell, R=radius_eff, aspect=aspect_outer)
            syms, pts = recut_with_planes(syms, pts, planes_outer, tol=1e-3)
    
            if args.verbose:
                print(f"    - After twins+recut: {len(syms)} atoms")
    
        # --- Write core.xyz and shell.xyz (behind --write-all) ---
        if args.write_all:
            region_masks = region_masks_from_layer_planes(pts, layer_planes)
            prefix = os.path.splitext(os.path.basename(args.out))[0]
            for k, mask in enumerate(region_masks):
                tag = "core" if k == 0 else f"shell{k}"
                part_syms = [s for s, keep in zip(syms, mask) if keep]
                part_pts  =  pts[mask]
                if args.verbose:
                    print(f"    - Writing {prefix}_{tag}.xyz ({len(part_syms)} atoms)")
                part_pts_out = center_coords(part_pts) if args.center else part_pts
                part_syms_out, part_pts_out = _ordered_xyz_view(
                    part_syms,
                    part_pts_out,
                    cfg.charges,
                    anion_lig,
                )
                write_xyz(f"{prefix}_{tag}.xyz", part_syms_out, part_pts_out)
    
        syms, pts = _prune_before_facet_detection(syms, pts, args=args)

        stack_pair_cuts = merge_pair_cuts_from_cifs(
            [resolved_materials[idx].cif for idx in active_layer_indices],
            cfg.charges,
            safety=1.00,
        )
    
        # --- Detect facets on composite ---
        # Use outermost shell lattice only for normal directions in detect()
        core_cif = resolved_materials[outer_idx].cif
        struct = Structure.from_file(core_cif)
        if resolved_materials[outer_idx].shape_mode == "sphere":
            seeds0 = []
            planes = sphere_halfspaces(radius_eff, n_planes=resolved_materials[outer_idx].sphere_planes)
            if args.verbose:
                print(
                    "\n[5] Using synthetic spherical surface planes "
                    f"(composite, {len(planes)} planes); Miller facet report skipped."
                )
        else:
            seeds0 = expand_facets(struct, resolved_materials[outer_idx].seeds, proper_only=cfg.proper_only)
            _facets, planes = _detect_facets_and_report(
                syms,
                pts,
                struct,
                cfg.charges,
                seeds0,
                cfg.passivation.surf_tol,
                label=" (composite)",
                verbose=args.verbose,
            )
    
        syms, pts = _run_passivation_and_write_outputs(
            syms,
            pts,
            args=args,
            cfg=cfg,
            anion_lig=anion_lig,
            planes=planes,
            cif_path=resolved_materials[outer_idx].cif,
            struct=struct,
            facet_seeds=seeds0,
            material_label=resolved_materials[outer_idx].name,
            construction_radius_override=radius_eff,
            pair_cuts_override=stack_pair_cuts,
            output_layer_planes=layer_planes,
            output_materials=resolved_materials,
        )
    
        if args.verbose:
            print("\n### ELEMENT COUNTS ###")
            _print_stack_summary(
                syms,
                cfg.charges,
                resolved_materials,
                anion_lig,
                pts=pts,
                layer_planes=layer_planes,
            )
    
        return 0
    

    # ---------------- SINGLE-MATERIAL MODE (legacy) ----------------
    if args.verbose:
        print("\n[1] Reading CIF structure...")
    struct = Structure.from_file(args.cif)
    _validate_ligand_not_native([args.cif], anion_lig)
    material_label = _material_label_from_cif_or_structure(
        args.cif,
        struct,
        cfg.charges,
        anion_lig,
    )
    if args.verbose:
        print(f"    - Loaded {len(struct)} atoms from {args.cif}")

    if args.verbose:
        print("\n[2] Using YAML config (single material)...")
        print(f"    - Shape mode: {cfg.shape_mode}")
        if cfg.shape_mode == "sphere":
            print(f"    - Sphere planes: {cfg.sphere_planes}")
        else:
            print(f"    - Facet seeds: {[ (f.h, f.k, f.l) for f in cfg.seeds ]}")
        print(f"    - Ligands: anion={anion_lig}, cation={cation_lig}, surf_tol={cfg.passivation.surf_tol:.3f} Å")
        print(f"    - Charges: {cfg.charges}")
        print(f"    - Pair opposites: {bool(cfg.pair_opposites)}")
        po_cli = getattr(args, "proper_rotations_only", None)
        eff_proper = cfg.proper_only if po_cli is None else bool(po_cli)
        print(f"    - Proper rotations only (effective): {bool(eff_proper)}")

    # Resolve size/aspect and proper-only (CLI can override YAML size).
    size_unit_cells_eff = _effective_single_size_unit_cells(cfg, args)
    aspect = args.aspect if args.aspect is not None else cfg.aspect
    proper_only = cfg.proper_only if getattr(args, "proper_rotations_only", None) is None else bool(args.proper_rotations_only)
    if size_unit_cells_eff is not None:
        radius_eff, aspect = size_unit_cells_to_radius_aspect(struct, size_unit_cells_eff)
    else:
        radius_eff = _radius_from_size_args(struct, args)

    if cfg.shape_mode == "sphere":
        if args.verbose:
            print("\n[3] Preparing spherical cut...")
            if size_unit_cells_eff is not None:
                print(
                    f"    - Size unit cells: {tuple(f'{x:g}' for x in size_unit_cells_eff)} "
                    f"-> radius {radius_eff:.6f} Å, aspect={tuple(round(x, 4) for x in aspect)}"
                )
            print(f"    - Synthetic sphere planes: {cfg.sphere_planes}")
        seeds = []
        wulff_facets: List[Facet] = []
    else:
        if args.verbose:
            print("\n[3] Expanding symmetry & building Wulff facets...")
        seeds = _resolve_facet_terminations(struct, cfg.seeds, cfg.charges)
        if args.verbose:
            resolved = [
                ((f.h, f.k, f.l), f.termination)
                for f in seeds
                if getattr(f, "termination", None)
            ]
            if resolved:
                print(f"    - Resolved terminated facet seeds: {resolved}")
            if size_unit_cells_eff is not None:
                print(
                    f"    - Size unit cells: {tuple(f'{x:g}' for x in size_unit_cells_eff)} "
                    f"-> radius {radius_eff:.6f} Å, aspect={tuple(round(x, 4) for x in aspect)}"
                )
        wulff_facets = expand_facets(struct, seeds, proper_only=proper_only)
        if args.verbose:
            print(f"    - Expanded to {len(wulff_facets)} oriented facets")

    variants = _center_variants(struct, cfg, anion_lig)
    multiple_variants = len(variants) > 1
    if args.verbose and multiple_variants:
        print(f"    - Construction center variants: {[label for label, _ in variants]}")

    for variant_label, origin_shift in variants:
        run_args = copy.copy(args)
        run_args.size_unit_cells = size_unit_cells_eff
        run_args.out = _variant_out_path(
            args.out,
            variant_label,
            multiple_variants,
            args=args,
            material_label=material_label,
        )

        if args.verbose:
            heading = f" [{variant_label}]" if multiple_variants else ""
            if cfg.shape_mode == "sphere":
                print(f"\n[4] Building nanocrystal from spherical cut{heading}...")
            else:
                print(f"\n[4] Building nanocrystal from Wulff facets{heading}...")
            if origin_shift is not None:
                print(
                    "    - Construction origin shift "
                    f"r0 = [{origin_shift[0]:.6f}, {origin_shift[1]:.6f}, {origin_shift[2]:.6f}] Å"
                )
            print(f"    - Output: {run_args.out}")

        if cfg.shape_mode == "sphere":
            syms, pts, _planes_geo = build_spherical_nanocrystal(
                struct,
                radius_eff,
                n_planes=cfg.sphere_planes,
                origin_shift=origin_shift,
            )
        else:
            syms, pts, _planes_geo = build_nanocrystal(
                struct,
                wulff_facets,
                radius_eff,
                aspect=aspect,
                origin_shift=origin_shift,
            )
        syms, pts = dedupe_points(syms, pts, tol=1e-3)
        if args.verbose:
            print(f"    - Cut particle: {len(syms)} atoms")
            if cfg.shape_mode == "sphere":
                print(f"    - Spherical cut planes: {cfg.sphere_planes}")
            else:
                ax, ay, az = aspect
                print(f"    - Aspect multipliers (a,b,c): {ax:.3f}, {ay:.3f}, {az:.3f}")

        # --- OPTIONAL TWIN BOUNDARIES (single-material) ---
        if getattr(cfg, "twins", None):
            syms, pts = apply_single_material_twins(
                syms,
                pts,
                cfg=cfg,
                struct=struct,
                planes_geo=_planes_geo,
            )

        syms, pts = _prune_before_facet_detection(syms, pts, args=args)

        report_label = f" ({variant_label})" if multiple_variants else ""
        if cfg.shape_mode == "sphere":
            planes = _planes_geo
            if args.verbose:
                print(
                    f"\n[5] Using synthetic spherical surface planes{report_label} "
                    f"({len(planes)} planes); Miller facet report skipped."
                )
        else:
            _facets, planes = _detect_facets_and_report(
                syms,
                pts,
                struct,
                cfg.charges,
                wulff_facets,
                cfg.passivation.surf_tol,
                label=report_label,
                verbose=args.verbose,
            )

        syms, pts = _run_passivation_and_write_outputs(
            syms,
            pts,
            args=run_args,
            cfg=cfg,
            anion_lig=anion_lig,
            planes=planes,
            cif_path=args.cif,
            struct=struct,
            facet_seeds=wulff_facets,
            material_label=material_label,
        )

        if args.verbose:
            print("\n### ELEMENT COUNTS ###")
            title = f"ROLE COUNTS (single material{report_label})"
            _print_single_material_summary(syms, cfg.charges, anion_lig, title=title)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
