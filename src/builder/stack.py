from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Structure

from .facets import expand_facets, halfspaces
from .geometry import sphere_halfspaces
from .nc_types import MaterialSpec, Plane


def size_unit_cells_to_radius_aspect(struct: Structure, size: tuple[float, float, float]) -> tuple[float, tuple[float, float, float]]:
    lengths = np.array([np.linalg.norm(v) for v in struct.lattice.matrix], dtype=float)
    physical = np.array(size, dtype=float) * lengths
    radius = float(np.min(physical))
    if radius <= 0:
        raise ValueError(f"size_unit_cells must produce positive dimensions, got {size}")
    aspect = tuple(float(x / radius) for x in physical)
    return radius, aspect


def cumulative_size_unit_cells(materials: List[MaterialSpec]) -> List[tuple[float, float, float]] | None:
    sizes = [m.build.size_unit_cells for m in materials]
    if all(size is None for size in sizes):
        return None
    if any(size is None for size in sizes):
        raise ValueError("In stack mode, either every material or no material must define size_unit_cells")
    cumulative = np.zeros(3, dtype=float)
    out: list[tuple[float, float, float]] = []
    for size in sizes:
        cumulative += np.array(size, dtype=float)
        out.append(tuple(float(x) for x in cumulative))
    return out


def inside_planes(points: NDArray[np.float64], planes: List[Plane], tol: float = 1e-6) -> NDArray[np.bool_]:
    if not planes:
        return np.zeros(len(points), dtype=bool)
    A = np.stack([n for (n, _d) in planes], axis=0)
    b = np.array([d for (_n, d) in planes], float)
    return (points @ A.T <= (b[None, :] + tol)).all(axis=1)


def build_layer_planes(
    materials: List[MaterialSpec],
    radius: float,
    proper_only: bool,
    *,
    cumulative_sizes: List[tuple[float, float, float]] | None = None,
) -> List[List[Plane]]:
    layer_planes = []
    for idx, m in enumerate(materials):
        if (
            cumulative_sizes is not None
            and idx > 0
            and np.allclose(cumulative_sizes[idx], cumulative_sizes[idx - 1], atol=1e-12)
        ):
            layer_planes.append(layer_planes[-1])
            continue
        sm = Structure.from_file(m.cif)
        if cumulative_sizes is None:
            r_eff, aspect = radius, m.aspect
        else:
            r_eff, aspect = size_unit_cells_to_radius_aspect(sm, cumulative_sizes[idx])
        if m.shape_mode == "sphere":
            layer_planes.append(sphere_halfspaces(r_eff, n_planes=m.sphere_planes))
        else:
            fm = expand_facets(sm, m.seeds, proper_only=proper_only)
            layer_planes.append(halfspaces(sm, fm, R=r_eff, aspect=aspect))
    return layer_planes


def region_masks_from_layer_planes(
    points: NDArray[np.float64],
    layer_planes: List[List[Plane]],
) -> List[NDArray[np.bool_]]:
    inside_layers = [inside_planes(points, pl) for pl in layer_planes]
    region_masks = []
    for k in range(len(layer_planes)):
        if k == 0:
            region_masks.append(inside_layers[0])
        else:
            region_masks.append(inside_layers[k] & (~inside_layers[k-1]))
    return region_masks


def material_cation_anion(material_cfg: MaterialSpec, charges: dict) -> tuple[str, str]:
    struct = Structure.from_file(material_cfg.cif)
    elems = sorted(set(str(site.specie.symbol) for site in struct.sites))
    cat = next((e for e in elems if charges.get(e, 0) > 0), None)
    an = next((e for e in elems if charges.get(e, 0) < 0), None)
    if cat is None or an is None:
        raise SystemExit(f"Cannot infer cation/anion for {material_cfg.name}. Check CIF and charges.")
    return cat, an


def relabel_regions_by_material(
    syms: List[str],
    region_masks: List[NDArray[np.bool_]],
    materials: List[MaterialSpec],
    charges: dict,
    ligand: str | None,
    *,
    verbose: bool,
) -> List[str]:
    for k, m in enumerate(materials):
        mask = region_masks[k]
        if verbose:
            lab = "CORE" if k == 0 else f"SHELL {k}"
            print(f"    - Region {lab}: {int(mask.sum())} atoms (aspect={m.aspect})")
        if not mask.any():
            continue

        cat_el, an_el = material_cation_anion(m, charges)
        idxs = np.where(mask)[0]
        for i in idxs:
            el = syms[i]
            q = charges.get(el, 0)
            if q > 0:
                syms[i] = cat_el
            elif q < 0 and el != ligand:
                syms[i] = an_el
    return syms
