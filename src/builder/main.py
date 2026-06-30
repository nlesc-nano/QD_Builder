# src/builder/main.py
from __future__ import annotations
import os
import sys
import logging
import copy
import re
import dataclasses
from collections import Counter
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
    reference_radius_from_size,
    region_masks_from_layer_planes,
    relabel_regions_by_material,
    select_geometry_reference,
    size_unit_cells_to_aspect,
    size_unit_cells_to_radius_aspect,
    validate_stack_symmetry,
    material_cation_anion,
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


def _charge_with_ligand_exchange(symbols, charges, ligand_exchange_charge_ledger=None):
    """Total charge with exchanged ligands counted by their YAML molecular charge."""
    ledger = ligand_exchange_charge_ledger or []
    q_element = int(sum(int(charges.get(sym, 0)) for sym in symbols))
    q_ignored = int(sum(int(entry.get("ignored_element_charge", 0)) for entry in ledger))
    q_exchange = int(sum(int(entry.get("charge", 0)) for entry in ledger))
    return q_element, q_ignored, q_exchange, q_element - q_ignored + q_exchange


def _print_stack_summary(
    symbols,
    charges,
    materials_cfg,
    ligand_symbol: str,
    *,
    pts=None,
    layer_planes=None,
    ligand_exchange_charge_ledger=None,
):
    """
    Print per-layer counts for core/shell systems.
    Uses the element sets derived from each material's CIF structure.
    """
    from collections import Counter

    cnt = Counter(symbols)
    Q_element, Q_ignored, Q_exchange, Q_total = _charge_with_ligand_exchange(
        symbols, charges, ligand_exchange_charge_ledger
    )
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

    if ligand_exchange_charge_ledger:
        print(f"\nElement-symbol Charge = {Q_element:+d}")
        print(f"Ignored exchanged-ligand element charge = {-Q_ignored:+d}")
        print(f"YAML exchanged-ligand charge = {Q_exchange:+d}")
    print(f"\nTotal Charge = {Q_total:+d}")


def _print_single_material_summary(
    symbols,
    charges,
    ligand_symbol: str,
    title: str = None,
    *,
    ligand_exchange_charge_ledger=None,
):
    from collections import Counter
    cnt = Counter(symbols)

    n_ligand  = cnt.get(ligand_symbol, 0)

    # total Q
    Q_element, Q_ignored, Q_exchange, Q = _charge_with_ligand_exchange(
        symbols, charges, ligand_exchange_charge_ledger
    )

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
    if ligand_exchange_charge_ledger:
        print(f"\nElement-symbol Charge = {Q_element:+d}")
        print(f"Ignored exchanged-ligand element charge = {-Q_ignored:+d}")
        print(f"YAML exchanged-ligand charge = {Q_exchange:+d}")
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
    # Determine the materials to search in
    if getattr(cfg, "mode", "single") == "stack" and getattr(cfg, "materials", None):
        materials = cfg.materials
    else:
        materials = None

    if cfg.construction_origin and not _is_all_centers_spec(cfg.construction_origin):
        species_list = _center_species_list(cfg.construction_origin)
        if species_list:
            variants = []
            for species in species_list:
                # Find which material structure has this species
                target_struct = struct
                if materials:
                    for m in materials:
                        try:
                            m_struct = Structure.from_file(m.cif)
                            if any(site.specie.symbol == species for site in m_struct.sites):
                                target_struct = m_struct
                                break
                        except Exception:
                            pass
                variants.append((_safe_label(species), construction_origin_shift(
                    target_struct, {"center_on_species": species}
                )))
            
            # Deduplicate variants by origin_shift
            unique_variants = []
            for name, shift in variants:
                if not any(
                    (shift is None and u_shift is None) or 
                    (shift is not None and u_shift is not None and np.allclose(shift, u_shift, atol=1e-5))
                    for _, u_shift in unique_variants
                ):
                    unique_variants.append((name, shift))
            return unique_variants

        return [("custom", construction_origin_shift(struct, cfg.construction_origin))]

    # Fallback / 'all' species
    # Collect all unique native species across materials
    native_species = []
    if materials:
        for m in materials:
            try:
                m_struct = Structure.from_file(m.cif)
                for sym in _native_species_in_cif_order(m_struct, cfg.charges, ligand_symbol):
                    if sym not in native_species:
                        native_species.append(sym)
            except Exception:
                pass
    else:
        native_species = _native_species_in_cif_order(struct, cfg.charges, ligand_symbol)

    variants = []
    for sym in native_species:
        target_struct = struct
        if materials:
            for m in materials:
                try:
                    m_struct = Structure.from_file(m.cif)
                    if any(site.specie.symbol == sym for site in m_struct.sites):
                        target_struct = m_struct
                        break
                except Exception:
                    pass
        variants.append((_safe_label(sym), construction_origin_shift(
            target_struct, {"center_on_species": sym}
        )))

    # Deduplicate variants by origin_shift
    unique_variants = []
    for name, shift in variants:
        if not any(
            (shift is None and u_shift is None) or 
            (shift is not None and u_shift is not None and np.allclose(shift, u_shift, atol=1e-5))
            for _, u_shift in unique_variants
        ):
            unique_variants.append((name, shift))
    return unique_variants or [("origin", None)]



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

        scope = getattr(f, "scope", "family")
        hkl_in = (int(f.h), int(f.k), int(f.l))
        hkl = hkl_in if scope == "facet" else tuple(abs(x) for x in hkl_in)
        if hkl == (0, 0, 0):
            resolved.append(f)
            continue

        if scope == "facet":
            # Facet-scoped hkl is an exact exposed orientation supplied by the
            # recipe. Wulff halfspaces expose the layer opposite the plane
            # normal, so only flip the construction normal; do not choose
            # between the two signed terminations.
            chosen = hkl
        else:
            candidates = [hkl, (-hkl[0], -hkl[1], -hkl[2])]
            scored = [(cand, _surface_charge_for_signed_hkl(struct, cand, charges)) for cand in candidates]
            if term == "cation_rich":
                chosen, _q = max(scored, key=lambda rec: rec[1])
            elif term == "anion_rich":
                chosen, _q = min(scored, key=lambda rec: rec[1])
            else:
                chosen = hkl_in
        # Bulk slab scoring uses max projection; Wulff halfspaces expose the opposite polar layer.
        chosen = (-chosen[0], -chosen[1], -chosen[2])
        resolved.append(Facet(chosen[0], chosen[1], chosen[2], f.gamma, termination=term, scope=scope))
    return _swap_dual_family_effective_terminations(resolved)


def _swap_dual_family_effective_terminations(facets: List[Facet]) -> List[Facet]:
    groups: dict[tuple[int, int, int], dict[str, float]] = {}
    for f in facets:
        term = getattr(f, "termination", None)
        if getattr(f, "scope", "family") != "family" or term not in {"cation_rich", "anion_rich"}:
            continue
        fam = tuple(abs(int(x)) for x in (f.h, f.k, f.l))
        groups.setdefault(fam, {})[term] = f.gamma

    dual = {fam: terms for fam, terms in groups.items() if "cation_rich" in terms and "anion_rich" in terms}
    if not dual:
        return facets

    opposite = {"cation_rich": "anion_rich", "anion_rich": "cation_rich"}
    out: List[Facet] = []
    for f in facets:
        term = getattr(f, "termination", None)
        fam = tuple(abs(int(x)) for x in (f.h, f.k, f.l))
        if getattr(f, "scope", "family") == "family" and fam in dual and term in opposite:
            new_term = opposite[term]
            out.append(Facet(f.h, f.k, f.l, dual[fam][new_term], termination=new_term, scope=f.scope))
        else:
            out.append(f)
    return out


def _effective_termination_from_charge(q: float) -> str | None:
    if q > 0:
        return "cation_rich"
    if q < 0:
        return "anion_rich"
    return None


def _termination_mismatches_for_detected_facets(
    syms: List[str],
    pts: np.ndarray,
    requested_facets: List[Facet],
    detected_facets: List[Facet],
    detected_planes,
    charges,
    surf_tol: float,
) -> List[dict]:
    requested_by_hkl: dict[tuple[int, int, int], str] = {}
    for f in requested_facets:
        term = getattr(f, "termination", None)
        if term not in {"cation_rich", "anion_rich"}:
            continue
        requested_by_hkl[(int(f.h), int(f.k), int(f.l))] = term
    if not requested_by_hkl:
        return []

    mismatches: List[dict] = []
    for f, (n, d) in zip(detected_facets, detected_planes):
        hkl = (int(f.h), int(f.k), int(f.l))
        requested = requested_by_hkl.get(hkl)
        if requested is None:
            continue
        # Requested cation/anion-rich terminations refer to the exposed outer
        # atomic layer, not the charge of a multi-layer surface shell.
        shell = np.where((float(d) - pts @ n) < min(0.25, max(1e-3, surf_tol * 0.125)))[0]
        if shell.size == 0:
            continue
        q = float(sum(float(charges.get(syms[i], 0.0)) for i in shell.tolist()))
        effective = _effective_termination_from_charge(q)
        if effective != requested:
            mismatches.append({
                "hkl": hkl,
                "requested": requested,
                "effective": effective or "balanced",
                "charge": q,
                "atoms": int(shell.size),
            })
    return mismatches


def _flip_terminated_facets(facets: List[Facet]) -> List[Facet]:
    return [
        Facet(-f.h, -f.k, -f.l, f.gamma, termination=f.termination, scope=f.scope)
        if getattr(f, "termination", None) in {"cation_rich", "anion_rich"}
        else f
        for f in facets
    ]


def _format_termination_mismatches(mismatches: List[dict]) -> str:
    parts = []
    for m in mismatches[:8]:
        h, k, l = m["hkl"]
        parts.append(
            f"({h},{k},{l}) requested={m['requested']} effective={m['effective']} "
            f"Q={m['charge']:+.0f} atoms={m['atoms']}"
        )
    extra = len(mismatches) - len(parts)
    if extra > 0:
        parts.append(f"... +{extra} more")
    return "; ".join(parts)


def _build_wulff_cut(
    struct: Structure,
    wulff_facets: List[Facet],
    radius_eff: float,
    *,
    aspect,
    origin_shift,
):
    syms, pts, planes_geo = build_nanocrystal(
        struct,
        wulff_facets,
        radius_eff,
        aspect=aspect,
        origin_shift=origin_shift,
    )
    syms, pts = dedupe_points(syms, pts, tol=1e-3)
    return syms, pts, planes_geo


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
    region_masks=None,
    stack_passivation: bool = False,
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
        region_masks=region_masks,
        stack_passivation=stack_passivation,
        prepass_mode=cfg.passivation.prepass_mode,
        prepass_min_cn_terrace=cfg.passivation.prepass_min_cn_terrace,
        prepass_min_cn_edge=cfg.passivation.prepass_min_cn_edge,
        prepass_min_cn_vertex=cfg.passivation.prepass_min_cn_vertex,
        include_sublayer=cfg.passivation.include_sublayer,
    )

    surface_reconstruction_spec = getattr(
        getattr(cfg, "post_treatment", None),
        "surface_reconstruction",
        cfg.facet_reconstruction,
    )
    if surface_reconstruction_spec.enabled:
        from functools import partial
        balance_fn = partial(
            charge_balance_iterative,
            prepass_mode=cfg.passivation.prepass_mode,
            prepass_min_cn_terrace=cfg.passivation.prepass_min_cn_terrace,
            prepass_min_cn_edge=cfg.passivation.prepass_min_cn_edge,
            prepass_min_cn_vertex=cfg.passivation.prepass_min_cn_vertex,
        )
        syms, pts = reconstruct_polar_facets(
            syms,
            pts,
            struct=struct,
            facet_seeds=facet_seeds,
            charges=cfg.charges,
            ligand=anion_lig,
            surf_tol=cfg.passivation.surf_tol,
            cif_path=cif_path,
            spec=surface_reconstruction_spec,
            charge_balance_fn=balance_fn,
            verbose=args.verbose,
            write_all=args.write_all,
            prefix=prefix,
        )

    if output_layer_planes is not None:
        syms, pts = _perform_core_shell_swapping_and_rebalance(
            syms,
            pts,
            args=args,
            cfg=cfg,
            anion_lig=anion_lig,
            layer_planes=output_layer_planes,
            resolved_materials=output_materials,
            charges=cfg.charges,
            planes=planes,
            cif_path=cif_path,
            pair_cuts_override=pair_cuts_override,
            verbose=args.verbose,
        )

    ligand_exchange_charge_ledger = []

    # ── Charged ligand exchange (optional; after reconstruction, before neutral ligands)
    ligand_exchange_spec = getattr(
        getattr(cfg, "post_treatment", None),
        "ligand_exchange",
        None,
    )
    if ligand_exchange_spec is not None and ligand_exchange_spec.enabled:
        from .ligand_exchange_posttreat import run_ligand_exchange_posttreatment
        syms, pts, ligand_exchange_charge_ledger = run_ligand_exchange_posttreatment(
            syms, pts, cfg, struct, planes, cif_path
        )
        if ligand_exchange_charge_ledger:
            q_element, q_ignored, q_exchange, q_total = _charge_with_ligand_exchange(
                syms, cfg.charges, ligand_exchange_charge_ledger
            )
            print(
                "[ligand-exchange:charge] "
                f"element-symbol Q={q_element:+d}, "
                f"ignored exchanged-ligand element Q={-q_ignored:+d}, "
                f"YAML exchanged-ligand Q={q_exchange:+d}, "
                f"total Q={q_total:+d}"
            )

    # ── Neutral-ligand post-treatment (optional; final post-treatment step) ───
    neutral_ligand_spec = getattr(
        getattr(cfg, "post_treatment", None),
        "neutral_ligands",
        cfg.passivation.neutral_ligands,
    )
    if neutral_ligand_spec.enabled:
        from .neutral_ligand_posttreat import run_neutral_ligand_posttreatment
        syms, pts = run_neutral_ligand_posttreatment(
            syms, pts, cfg, struct, planes
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
    actual_radius = max(d for _, d in planes) if planes else construction_radius
    a, b, c = struct.lattice.matrix
    min_lat = min(float(np.linalg.norm(a)), float(np.linalg.norm(b)), float(np.linalg.norm(c)))
    actual_size_cells = actual_radius / min_lat

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
            f"actual_used: R={actual_radius:.3f} Å (cell_size={actual_size_cells:.2f}) | "
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
        "actual_size_unit_cells": actual_size_cells,
        "construction_radius_ang": construction_radius,
        "actual_radius_ang": actual_radius,
        "construction_diameter_ang": construction_diameter,
        "size_metrics": size_metrics,
    }
    if stack_sizes:
        extra["stack_size_metrics"] = stack_sizes
    if ligand_exchange_charge_ledger:
        q_element, q_ignored, q_exchange, q_total = _charge_with_ligand_exchange(
            syms, cfg.charges, ligand_exchange_charge_ledger
        )
        extra.update({
            "element_symbol_charge": q_element,
            "ignored_exchanged_ligand_element_charge": q_ignored,
            "yaml_exchanged_ligand_charge": q_exchange,
            "total_charge": q_total,
            "ligand_exchange_charge_ledger": ligand_exchange_charge_ledger,
        })
    write_manifest(prefix, syms, cfg.charges, extra=extra)

    return syms, pts, ligand_exchange_charge_ledger


def _fps_subsample(coords: np.ndarray, num_to_pick: int) -> np.ndarray:
    """
    Selects num_to_pick indices from coords using Farthest Point Sampling.
    To ensure reproducibility (tie-breaking), we use a deterministic seed
    or fixed initial pick (index with max norm).
    """
    n = len(coords)
    if n == 0 or num_to_pick <= 0:
        return np.array([], dtype=int)
    if num_to_pick >= n:
        return np.arange(n)

    picks = []
    # Deterministic start: pick the point furthest from the origin (max norm)
    norms = np.linalg.norm(coords, axis=1)
    first_pick = int(np.argmax(norms))
    picks.append(first_pick)

    # Track minimum distance from each point to the selected set
    min_dists = np.linalg.norm(coords - coords[first_pick], axis=1)

    for _ in range(1, num_to_pick):
        next_pick = int(np.argmax(min_dists))
        picks.append(next_pick)
        dists_to_next = np.linalg.norm(coords - coords[next_pick], axis=1)
        min_dists = np.minimum(min_dists, dists_to_next)

    return np.array(picks, dtype=int)


def _perform_core_shell_swapping_and_rebalance(
    syms,
    pts,
    *,
    args,
    cfg,
    anion_lig,
    layer_planes,
    resolved_materials,
    charges,
    planes,
    cif_path,
    pair_cuts_override,
    verbose,
):
    if not layer_planes or not resolved_materials:
        return syms, pts

    # Dynamically compute region masks using current post-passivation coordinates
    region_masks = region_masks_from_layer_planes(pts, layer_planes)

    # 1. Relabel all regions based on the abrupt materials definition
    # This correctly changes core species and shell species.
    # Passivating ligands are kept as anion_lig.
    syms = relabel_regions_by_material(
        syms,
        region_masks,
        resolved_materials,
        charges,
        anion_lig,
        verbose=verbose,
    )

    # 2. Check if a mixed interface is requested.
    interface_type = "abrupt"
    mixing_width = 3.0
    mixing_ratio = 0.5

    # Check top-level stack config first
    stack_cfg = getattr(cfg, "stack", None)
    if stack_cfg is not None:
        interface_type = getattr(stack_cfg, "interface", "abrupt")
        mixing_width = getattr(stack_cfg, "mixing_width", 3.0)

    # Check shell material config (resolved_materials[1] interface)
    if len(resolved_materials) > 1 and resolved_materials[1].interface is not None:
        shell_int = resolved_materials[1].interface
        interface_type = str(shell_int.get("type", interface_type)).strip().lower()
        mixing_width = float(shell_int.get("mixing_width", mixing_width))
        mixing_ratio = float(shell_int.get("mixing_ratio", mixing_ratio))

    if interface_type == "mixed":
        if verbose:
            print(f"\n[Mixed Interface] Blending interface (width: {mixing_width} Å, ratio: {mixing_ratio}) via Farthest Point Sampling...")

        # We assume 2 materials: core (0) and shell (1)
        core_cat, core_an = material_cation_anion(resolved_materials[0], charges)
        shell_cat, shell_an = material_cation_anion(resolved_materials[1], charges)

        # Core non-ligand indices
        core_idx_mask = region_masks[0] & (np.array(syms) != anion_lig)
        # Shell non-ligand indices
        shell_idx_mask = region_masks[1] & (np.array(syms) != anion_lig)

        core_indices = np.where(core_idx_mask)[0]
        shell_indices = np.where(shell_idx_mask)[0]

        if len(core_indices) > 0 and len(shell_indices) > 0:
            from scipy.spatial import cKDTree
            shell_tree = cKDTree(pts[shell_indices])
            dists, _ = shell_tree.query(pts[core_indices], distance_upper_bound=mixing_width)
            valid_query = dists <= mixing_width
            mixing_core_global_indices = core_indices[valid_query]

            # Separate mixing core atoms into cations and anions
            mixing_cations = [idx for idx in mixing_core_global_indices if syms[idx] == core_cat]
            mixing_anions = [idx for idx in mixing_core_global_indices if syms[idx] == core_an]

            if verbose:
                print(f"    - Found {len(mixing_core_global_indices)} core atoms in mixing zone (cations: {len(mixing_cations)}, anions: {len(mixing_anions)})")

            num_cat_swap = int(round(mixing_ratio * len(mixing_cations)))
            if len(mixing_cations) > 0 and num_cat_swap > 0:
                cat_coords = pts[mixing_cations]
                fps_cat_local_indices = _fps_subsample(cat_coords, num_cat_swap)
                for local_idx in fps_cat_local_indices:
                    global_idx = mixing_cations[local_idx]
                    syms[global_idx] = shell_cat

            num_an_swap = int(round(mixing_ratio * len(mixing_anions)))
            if len(mixing_anions) > 0 and num_an_swap > 0:
                an_coords = pts[mixing_anions]
                fps_an_local_indices = _fps_subsample(an_coords, num_an_swap)
                for local_idx in fps_an_local_indices:
                    global_idx = mixing_anions[local_idx]
                    syms[global_idx] = shell_an

            if verbose:
                print(f"    - Swapped {num_cat_swap} cations to {shell_cat} and {num_an_swap} anions to {shell_an}")

    # Output the core_mixed.xyz containing only the core region (region_masks[0])
    prefix = os.path.splitext(args.out)[0]
    out_dir = os.path.dirname(args.out)
    core_mixed_path = os.path.join(out_dir, "core_mixed.xyz") if out_dir else "core_mixed.xyz"
    core_mixed_prefix_path = f"{prefix}_core_mixed.xyz"

    core_mask = region_masks[0]
    core_syms = [s for s, keep in zip(syms, core_mask) if keep]
    core_pts = pts[core_mask]

    if len(core_syms) > 0:
        if verbose:
            print(f"    - Writing core-only mixed atomic output to {core_mixed_path} and {core_mixed_prefix_path} ({len(core_syms)} atoms)")
        core_pts_out = center_coords(core_pts) if args.center else core_pts
        write_xyz(core_mixed_path, core_syms, core_pts_out)
        write_xyz(core_mixed_prefix_path, core_syms, core_pts_out)

    # 3. Post-swap charge balancing
    if verbose:
        print("\n[Post-Swap Rebalancing] Checking charge and running passivation rebalance if needed...")

    syms, pts = charge_balance_iterative(
        syms, pts,
        charges, anion_lig,
        verbose=verbose,
        planes=planes,
        surf_tol=cfg.passivation.surf_tol,
        cif_path=resolved_materials[-1].cif,
        positive_q_strategy=args.positive_q_mode,
        write_all=args.write_all,
        prefix=prefix,
        experimental_exhausted_positive_q_fallback=bool(
            getattr(cfg, "experimental", {}).get("exhausted_positive_q_fallback", False)
        ),
        pair_cuts_override=pair_cuts_override,
        region_masks=region_masks,
        stack_passivation=False,
        prepass_mode="none",
        include_sublayer=cfg.passivation.include_sublayer,
    )

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


def _prune_before_facet_detection(syms, pts, *, args, cfg=None):
    if not args.prune_mono:
        return syms, pts
    
    # Determine the prune threshold dynamically
    min_cn = args.prune_min_cn
    if cfg is not None and getattr(cfg, "passivation", None) is not None:
        pass_spec = cfg.passivation
        if pass_spec.prepass_mode == "role-aware":
            min_cn = min(min_cn, pass_spec.prepass_min_cn_vertex)

    if args.verbose:
        print(f"\n[4b] Pruning low-coordination atoms (pre-facet detection, min_cn={min_cn})...")
    syms, pts, n_removed, n_pass = prune_low_coord_sites(
        syms, pts, min_cn=min_cn, max_passes=args.prune_passes, verbose=args.verbose
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
    if len(args.inputs) == 1:
        args.yaml = args.inputs[0]
        args.cif = None
    elif len(args.inputs) == 2:
        args.cif, args.yaml = args.inputs[0], args.inputs[1]
    else:
        p.error("expected one argument (YAML for stack mode) or two arguments (CIF YAML for single-material mode)")

    cfg: Config = parse_yaml_config(args.yaml)
    # Merge CLI options into PassivationSpec
    pass_spec = cfg.passivation
    updated_pass_fields = {}
    if getattr(args, "prepass_mode", None) is not None:
        updated_pass_fields["prepass_mode"] = args.prepass_mode
    if getattr(args, "prepass_min_cn_terrace", None) is not None:
        updated_pass_fields["prepass_min_cn_terrace"] = args.prepass_min_cn_terrace
    if getattr(args, "prepass_min_cn_edge", None) is not None:
        updated_pass_fields["prepass_min_cn_edge"] = args.prepass_min_cn_edge
    if getattr(args, "prepass_min_cn_vertex", None) is not None:
        updated_pass_fields["prepass_min_cn_vertex"] = args.prepass_min_cn_vertex
    
    # If prepass_mode was overridden to role-aware and vertex was not specified, default to 1
    if updated_pass_fields.get("prepass_mode") == "role-aware" and "prepass_min_cn_vertex" not in updated_pass_fields and pass_spec.prepass_min_cn_vertex == 3:
        updated_pass_fields["prepass_min_cn_vertex"] = 1

    if updated_pass_fields:
        pass_spec = dataclasses.replace(pass_spec, **updated_pass_fields)
        cfg = dataclasses.replace(cfg, passivation=pass_spec)

    if cfg.mode != "stack" and args.cif is None:
        p.error("single-material mode requires: nc-builder STRUCT.cif RECIPE.yaml")
    if cfg.mode == "stack" and args.cif is None:
        args.cif = cfg.materials[0].cif
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
        try:
            validate_stack_symmetry(cfg.materials)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        reference_cfg = select_geometry_reference(
            cfg.materials,
            mode=cfg.stack.geometry_reference,
        )
        resolved_materials = []
        for material in cfg.materials:
            mat_struct = Structure.from_file(material.cif)
            resolved_materials.append(dataclasses.replace(
                material,
                seeds=_resolve_facet_terminations(mat_struct, material.seeds, cfg.charges),
            ))
        core_cfg = resolved_materials[0]

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

        # Force unified crystal to be cut as a single Wulff polyhedron using shell lattice
        struct_ref = Structure.from_file(outer_cfg.cif)
        if cumulative_sizes is None:
            radius_eff = _radius_from_size_args(struct_ref, args)
            aspect_outer = outer_cfg.aspect
        else:
            radius_eff, aspect_outer = reference_radius_from_size(struct_ref, cumulative_sizes[outer_idx])

        if args.verbose:
            print(
                f"    - Geometry reference lattice: {reference_cfg.name} ({reference_cfg.cif}); "
                f"policy={cfg.stack.geometry_reference}; core material: {core_cfg.name}"
            )

        # Get construction center variants
        variants = _center_variants(struct_ref, cfg, anion_lig)
        multiple_variants = len(variants) > 1
        if args.verbose and multiple_variants:
            print(f"    - Construction center variants: {[label for label, _ in variants]}")

        for variant_label, origin_shift in variants:
            run_args = copy.copy(args)
            run_args.out = _variant_out_path(
                args.out,
                variant_label,
                multiple_variants,
                args=None, # pass args=None so it doesn't add _rep3 to the variant filename
                material_label=outer_cfg.name,
            )

            if args.verbose:
                heading = f" [{variant_label}]" if multiple_variants else ""
                if outer_cfg.shape_mode == "sphere":
                    print(f"\n[4] Building nanocrystal from spherical cut{heading}...")
                else:
                    print(f"\n[4] Building nanocrystal from Wulff facets{heading}...")
                if origin_shift is not None:
                    print(
                        "    - Construction origin shift "
                        f"r0 = [{origin_shift[0]:.6f}, {origin_shift[1]:.6f}, {origin_shift[2]:.6f}] Å"
                    )
                print(f"    - Output: {run_args.out}")

            if outer_cfg.shape_mode == "sphere":
                facets_outer = []
                syms, pts, _ = build_spherical_nanocrystal(
                    struct_ref,
                    radius_eff,
                    n_planes=outer_cfg.sphere_planes,
                    origin_shift=origin_shift,
                )
            else:
                facets_outer = expand_facets(struct_ref, outer_cfg.seeds, proper_only=cfg.proper_only)
                syms, pts, _ = build_nanocrystal(
                    struct_ref,
                    facets_outer,
                    radius_eff,
                    aspect=aspect_outer,
                    origin_shift=origin_shift,
                )
            syms, pts = dedupe_points(syms, pts)
            if args.verbose:
                if cumulative_sizes is not None:
                    print("    - Cumulative size_unit_cells (replica topology on reference lattice):")
                    prev_size = np.zeros(3, dtype=float)
                    for idx, (m, size) in enumerate(zip(resolved_materials, cumulative_sizes)):
                        if idx > 0 and np.allclose(size, prev_size, atol=1e-12):
                            print(
                                f"      {m.name}: size={tuple(f'{x:g}' for x in size)} "
                                "-> zero-thickness layer; reuses previous boundary"
                            )
                            prev_size = np.asarray(size, dtype=float)
                            continue
                        r_i, aspect_i = reference_radius_from_size(struct_ref, size)
                        r_phys, _ = size_unit_cells_to_radius_aspect(Structure.from_file(m.cif), size)
                        print(
                            f"      {m.name}: size={tuple(f'{x:g}' for x in size)} "
                            f"-> radius={r_i:.6f} Å (ref), aspect={tuple(round(x, 4) for x in aspect_i)}; "
                            f"physical_if_native={r_phys:.6f} Å"
                        )
                        prev_size = np.asarray(size, dtype=float)
                if outer_cfg.shape_mode == "sphere":
                    print(
                        f"    - Outermost spherical cut atoms: {len(syms)} "
                        f"(reference={reference_cfg.name}, planes={outer_cfg.sphere_planes})"
                    )
                else:
                    print(
                        f"    - Outermost cut atoms: {len(syms)} "
                        f"(reference={reference_cfg.name}, outer layer={outer_cfg.name})"
                    )

            layer_planes = build_layer_planes(
                resolved_materials,
                struct_ref,
                cfg.proper_only,
                cumulative_sizes=cumulative_sizes,
                radius=radius_eff if cumulative_sizes is None else None,
            )
            region_masks = region_masks_from_layer_planes(pts, layer_planes)

            syms, pts = _prune_before_facet_detection(syms, pts, args=run_args, cfg=cfg)
            region_masks = region_masks_from_layer_planes(pts, layer_planes)

            stack_pair_cuts = merge_pair_cuts_from_cifs(
                [resolved_materials[idx].cif for idx in active_layer_indices],
                cfg.charges,
                safety=1.00,
            )

            surface_cfg = resolved_materials[outer_idx]
            struct = struct_ref
            if surface_cfg.shape_mode == "sphere":
                seeds0 = []
                passivation_planes = sphere_halfspaces(radius_eff, n_planes=surface_cfg.sphere_planes)
                if args.verbose:
                    print(
                        "\n[5] Using synthetic spherical surface planes "
                        f"(composite, {len(passivation_planes)} planes); Miller facet report skipped."
                    )
            else:
                seeds0 = expand_facets(struct_ref, surface_cfg.seeds, proper_only=cfg.proper_only)
                _facets, passivation_planes = _detect_facets_and_report(
                    syms,
                    pts,
                    struct,
                    cfg.charges,
                    seeds0,
                    cfg.passivation.surf_tol,
                    label=" (composite)",
                    verbose=args.verbose,
                )

            if args.verbose:
                print(f"\n[4] Composite particle atoms (pre-passivation): {len(syms)}")

            # --- OPTIONAL TWIN BOUNDARIES (stack mode) ---
            if getattr(cfg, "twins", None):
                if args.verbose:
                    print("\n[3a] Applying twin boundary transformations (stack mode)...")
                # Use reference lattice for twins and recut
                outer_shell = resolved_materials[outer_idx]

                # (1) Apply mirrors
                pts = apply_twins(
                    pts,
                    struct_ref.lattice.matrix,
                    cfg.twins,
                    default_origin="center",
                    species=syms,
                    charges=cfg.charges,
                )

                # (2) Recut with outer Wulff planes on reference lattice
                if outer_shell.shape_mode == "sphere":
                    planes_outer = sphere_halfspaces(radius_eff, n_planes=outer_shell.sphere_planes)
                else:
                    facets_shell = expand_facets(struct_ref, outer_shell.seeds, proper_only=cfg.proper_only)
                    planes_outer = halfspaces(struct_ref, facets_shell, R=radius_eff, aspect=aspect_outer)
                syms, pts = recut_with_planes(syms, pts, planes_outer)
        
                if args.verbose:
                    print(f"    - After twins+recut: {len(syms)} atoms")
                if surface_cfg.shape_mode == "sphere":
                    passivation_planes = sphere_halfspaces(radius_eff, n_planes=surface_cfg.sphere_planes)
                else:
                    _facets, passivation_planes = _detect_facets_and_report(
                        syms,
                        pts,
                        struct,
                        cfg.charges,
                        seeds0,
                        cfg.passivation.surf_tol,
                        label=" (composite, post-twin)",
                        verbose=args.verbose,
                    )

            # --- Write core.xyz and shell.xyz (behind --write-all) ---
            if args.write_all:
                region_masks = region_masks_from_layer_planes(pts, layer_planes)
                layer_prefix = os.path.splitext(run_args.out)[0]
                for k, mask in enumerate(region_masks):
                    tag = "core" if k == 0 else f"shell{k}"
                    part_syms = [s for s, keep in zip(syms, mask) if keep]
                    part_pts = pts[mask]
                    layer_path = f"{layer_prefix}_{tag}.xyz"
                    if args.verbose:
                        print(f"    - Writing {layer_path} ({len(part_syms)} atoms)")
                    part_pts_out = center_coords(part_pts) if args.center else part_pts
                    part_syms_out, part_pts_out = _ordered_xyz_view(
                        part_syms,
                        part_pts_out,
                        cfg.charges,
                        anion_lig,
                    )
                    write_xyz(layer_path, part_syms_out, part_pts_out)

            syms, pts, ligand_exchange_charge_ledger = _run_passivation_and_write_outputs(
                syms,
                pts,
                args=run_args,
                cfg=cfg,
                anion_lig=anion_lig,
                planes=passivation_planes,
                cif_path=surface_cfg.cif,
                struct=struct,
                facet_seeds=seeds0,
                material_label=surface_cfg.name,
                construction_radius_override=radius_eff,
                pair_cuts_override=stack_pair_cuts,
                output_layer_planes=layer_planes,
                output_materials=resolved_materials,
                region_masks=region_masks,
                stack_passivation=False,
            )

            use_core_lattice_fit = not args.no_core_lattice_fit
            if use_core_lattice_fit:
                region_masks = region_masks_from_layer_planes(pts, layer_planes)
                if args.verbose:
                    print("\n[3b] Applying core lattice fit with smooth interface blend...")
                try:
                    core_struct = Structure.from_file(core_cfg.cif)
                    pts = apply_core_lattice_fit(
                        pts,
                        region_masks[0],
                        layer_planes[0],
                        struct_ref.lattice.matrix,
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
                    print("WARNING: reference lattice not invertible; skipping core lattice fit.")
                if args.center:
                    pts = center_coords(pts)
                write_xyz(run_args.out, syms, pts)
                if args.verbose:
                    print(f"[12] Rewrote passivated structure after core lattice fit → {run_args.out}")

            # --- ALWAYS write core.xyz by default using finalized coordinates ---
            final_masks = region_masks_from_layer_planes(pts, layer_planes)
            core_mask = final_masks[0]
            core_syms = [s for s, keep in zip(syms, core_mask) if keep]
            core_pts = pts[core_mask]
            
            layer_prefix = os.path.splitext(run_args.out)[0]
            core_out_path = f"{layer_prefix}_core.xyz"
            
            out_dir = os.path.dirname(args.out)
            default_core_path = os.path.join(out_dir, "core.xyz") if out_dir else "core.xyz"
            
            if args.verbose:
                print(f"    - [Default Output] Writing core region to {core_out_path} and {default_core_path} ({len(core_syms)} atoms)")
            
            write_xyz(default_core_path, core_syms, core_pts)
            write_xyz(core_out_path, core_syms, core_pts)

            if multiple_variants and variant_label == variants[0][0]:
                import shutil
                if args.verbose:
                    print(f"    - [Default Output] Copying first variant '{variant_label}' to default output: {args.out}")
                shutil.copyfile(run_args.out, args.out)
                try:
                    shutil.copyfile(
                        f"{layer_prefix}.json",
                        f"{os.path.splitext(args.out)[0]}.json"
                    )
                except Exception:
                    pass
        
            if args.verbose:
                print("\n### ELEMENT COUNTS ###")
                _print_stack_summary(
                    syms,
                    cfg.charges,
                    resolved_materials,
                    anion_lig,
                    pts=pts,
                    layer_planes=layer_planes,
                    ligand_exchange_charge_ledger=ligand_exchange_charge_ledger,
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
                ((f.h, f.k, f.l), f.gamma, f.termination)
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
            if any(getattr(f, "termination", None) for f in wulff_facets) and len(wulff_facets) <= 16:
                expanded = [
                    ((f.h, f.k, f.l), f.gamma, getattr(f, "termination", None))
                    for f in wulff_facets
                ]
                print(f"    - Expanded terminated facets: {expanded}")

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

        facets_for_variant = wulff_facets
        last_mismatches: List[dict] = []
        variant_failed = False
        for termination_attempt in range(2):
            if cfg.shape_mode == "sphere":
                syms, pts, _planes_geo = build_spherical_nanocrystal(
                    struct,
                    radius_eff,
                    n_planes=cfg.sphere_planes,
                    origin_shift=origin_shift,
                )
                syms, pts = dedupe_points(syms, pts, tol=1e-3)
            else:
                syms, pts, _planes_geo = _build_wulff_cut(
                    struct,
                    facets_for_variant,
                    radius_eff,
                    aspect=aspect,
                    origin_shift=origin_shift,
                )
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

            syms, pts = _prune_before_facet_detection(syms, pts, args=args, cfg=cfg)
            if cfg.shape_mode == "sphere":
                break

            detected_facets, detected_planes = detect_facets_from_nc(
                syms,
                pts,
                struct.lattice,
                cfg.charges,
                facets_for_variant,
                cfg.passivation.surf_tol,
            )
            mismatches = _termination_mismatches_for_detected_facets(
                syms,
                pts,
                facets_for_variant,
                detected_facets,
                detected_planes,
                cfg.charges,
                cfg.passivation.surf_tol,
            )
            if not mismatches:
                if termination_attempt and args.verbose:
                    print("    - Effective facet termination check passed after flipping terminated Wulff seeds")
                wulff_facets = facets_for_variant
                break
            last_mismatches = mismatches
            if termination_attempt == 0:
                if args.verbose:
                    print(
                        "    - Effective facet termination mismatch after pruning; "
                        "retrying with opposite terminated Wulff seeds: "
                        f"{_format_termination_mismatches(mismatches)}"
                    )
                facets_for_variant = _flip_terminated_facets(facets_for_variant)
                continue
            msg = (
                "Requested facet termination could not be realized by the finite Wulff cut after pruning. "
                f"{_format_termination_mismatches(last_mismatches)}"
            )
            if multiple_variants:
                if args.verbose:
                    print(f"    - Skipping center variant '{variant_label}': {msg}")
                variant_failed = True
                break
            raise SystemExit(msg)

        if variant_failed:
            continue

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

        syms, pts, ligand_exchange_charge_ledger = _run_passivation_and_write_outputs(
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

        if multiple_variants and variant_label == variants[0][0]:
            import shutil
            if args.verbose:
                print(f"    - [Default Output] Copying first variant '{variant_label}' to default output: {args.out}")
            shutil.copyfile(run_args.out, args.out)
            try:
                shutil.copyfile(
                    f"{os.path.splitext(run_args.out)[0]}.json",
                    f"{os.path.splitext(args.out)[0]}.json"
                )
            except Exception:
                pass

        if args.verbose:
            print("\n### ELEMENT COUNTS ###")
            title = f"ROLE COUNTS (single material{report_label})"
            _print_single_material_summary(
                syms,
                cfg.charges,
                anion_lig,
                title=title,
                ligand_exchange_charge_ledger=ligand_exchange_charge_ledger,
            )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
