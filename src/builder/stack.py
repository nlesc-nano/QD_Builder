from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Structure

try:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:
    raise SystemExit("pip install pymatgen[matproj]")

from .constants import EPS as PLANE_EPS
from .facets import expand_facets, halfspaces
from .geometry import sphere_halfspaces
from .nc_types import MaterialSpec, Plane


def size_unit_cells_to_aspect(size: tuple[float, float, float]) -> tuple[float, tuple[float, float, float]]:
    """
    Convert cumulative unit-cell counts to Wulff aspect in replica space.

    The minimum count defines R=1 in unit-cell space; lattice constants scale
    this to physical Å separately via the reference structure.
    """
    arr = np.array(size, dtype=float)
    r_cells = float(np.min(arr))
    if r_cells <= 0:
        raise ValueError(f"size_unit_cells must produce positive dimensions, got {size}")
    aspect = tuple(float(x / r_cells) for x in arr)
    return r_cells, aspect


def size_unit_cells_to_radius_aspect(struct: Structure, size: tuple[float, float, float]) -> tuple[float, tuple[float, float, float]]:
    """Physical Å radius and aspect for a given structure (single-material / reporting)."""
    lengths = np.array([np.linalg.norm(v) for v in struct.lattice.matrix], dtype=float)
    physical = np.array(size, dtype=float) * lengths
    radius = float(np.min(physical))
    if radius <= 0:
        raise ValueError(f"size_unit_cells must produce positive dimensions, got {size}")
    aspect = tuple(float(x / radius) for x in physical)
    return radius, aspect


def reference_radius_from_size(struct: Structure, size: tuple[float, float, float]) -> tuple[float, tuple[float, float, float]]:
    """Physical Å Wulff radius and aspect from replica counts on a reference lattice."""
    r_cells, aspect = size_unit_cells_to_aspect(size)
    ref_min = float(min(np.linalg.norm(v) for v in struct.lattice.matrix))
    return r_cells * ref_min, aspect


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


def validate_stack_symmetry(materials: List[MaterialSpec], *, symprec: float = 1e-3) -> None:
    """Raise if stack materials do not share the same space group."""
    groups: list[tuple[str, str, int]] = []
    for m in materials:
        s = Structure.from_file(m.cif)
        analyzer = SpacegroupAnalyzer(s, symprec=symprec)
        groups.append((m.name, analyzer.get_space_group_symbol(), analyzer.get_space_group_number()))
    unique = {(sg_num, sg_sym) for _name, sg_sym, sg_num in groups}
    if len(unique) > 1:
        detail = ", ".join(f"{name}={sym} (#{num})" for name, sym, num in groups)
        raise ValueError(
            "Stack materials must share the same space group. Found: "
            f"{detail}. Use CIFs with matching symmetry (e.g. F-43m for zinc-blende)."
        )


def select_geometry_reference(
    materials: List[MaterialSpec],
    *,
    mode: str = "core",
) -> MaterialSpec:
    """
    Pick one material to define the shared replica lattice for all stack layers.

    Default ``core`` pins topology to the first YAML material so catA-anA @
    catB-anB stacks with identical size_unit_cells share the same discrete Wulff
    grid regardless of shared cation/anion chemistry.  Use ``shortest`` for
    legacy shortest-lattice selection.
    """
    if not materials:
        raise ValueError("select_geometry_reference requires at least one material")
    ref_mode = str(mode).strip().lower()
    if ref_mode == "core":
        return materials[0]
    if ref_mode == "shell":
        return materials[-1]
    if ref_mode == "shortest":
        best = materials[0]
        best_len = _min_lattice_length(best.cif)
        for m in materials[1:]:
            length = _min_lattice_length(m.cif)
            if length < best_len - 1e-9:
                best = m
                best_len = length
        return best
    raise ValueError(
        "stack.geometry_reference must be 'core', 'shell', or 'shortest'; "
        f"got {mode!r}"
    )


def _min_lattice_length(cif_path: str) -> float:
    struct = Structure.from_file(cif_path)
    return float(min(np.linalg.norm(v) for v in struct.lattice.matrix))


def inside_planes(points: NDArray[np.float64], planes: List[Plane], tol: float = PLANE_EPS) -> NDArray[np.bool_]:
    if not planes:
        return np.zeros(len(points), dtype=bool)
    A = np.stack([n for (n, _d) in planes], axis=0)
    b = np.array([d for (_n, d) in planes], float)
    return (points @ A.T <= (b[None, :] + tol)).all(axis=1)


def build_layer_planes(
    materials: List[MaterialSpec],
    reference_struct: Structure,
    proper_only: bool,
    *,
    cumulative_sizes: List[tuple[float, float, float]] | None = None,
    radius: float | None = None,
) -> List[List[Plane]]:
    """
    Build concentric Wulff boundary planes for each stack layer.

    All planes share the reference (core) lattice geometry; replica counts set
    the Wulff aspect.  Per-material facet energies/terminations still apply.
    """
    layer_planes = []
    for idx, m in enumerate(materials):
        if (
            cumulative_sizes is not None
            and idx > 0
            and np.allclose(cumulative_sizes[idx], cumulative_sizes[idx - 1], atol=1e-12)
        ):
            layer_planes.append(layer_planes[-1])
            continue
        if cumulative_sizes is None:
            if radius is None:
                raise ValueError("build_layer_planes requires radius when cumulative_sizes is None")
            r_eff, aspect = radius, m.aspect
        else:
            r_eff, aspect = reference_radius_from_size(reference_struct, cumulative_sizes[idx])
        if m.shape_mode == "sphere":
            layer_planes.append(sphere_halfspaces(r_eff, n_planes=m.sphere_planes))
        else:
            fm = expand_facets(reference_struct, m.seeds, proper_only=proper_only)
            layer_planes.append(halfspaces(reference_struct, fm, R=r_eff, aspect=aspect))
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
