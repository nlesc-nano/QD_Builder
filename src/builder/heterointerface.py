from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator, get_symmetrically_distinct_miller_indices
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.interfaces.zsl import ZSLGenerator

from .facets import expand_facets, halfspaces, unit_normal
from .geometry import build_nanocrystal, build_spherical_nanocrystal, dedupe_points
from .nc_types import Facet


@dataclass(frozen=True)
class FacetTermination:
    material: str
    cif: str
    family: str
    hkl: tuple[int, int, int]
    charge: int
    richness: str
    counts: dict[str, int]
    stoich_multiple: bool
    family_status: str


@dataclass(frozen=True)
class LatticeMatch:
    area: float
    area_mismatch: float
    max_length_mismatch: float
    angle_mismatch_deg: float
    film_transformation: tuple[tuple[float, float], tuple[float, float]]
    substrate_transformation: tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class InterfaceCandidate:
    core: FacetTermination
    shell: FacetTermination
    compatibility: str
    score: tuple
    lattice_match: LatticeMatch | None = None


def hkl_label(hkl: tuple[int, int, int]) -> str:
    return "(" + "".join(str(x) for x in hkl) + ")"


def counts_label(counts: dict[str, int]) -> str:
    return " ".join(f"{el}:{n}" for el, n in counts.items())


def charge_class(q: int) -> str:
    if q > 0:
        return "+"
    if q < 0:
        return "-"
    return "0"


def _gcd3(hkl: tuple[int, int, int]) -> int:
    return math.gcd(math.gcd(abs(hkl[0]), abs(hkl[1])), abs(hkl[2])) or 1


def _primitive_hkl(hkl: Iterable[float | int]) -> tuple[int, int, int]:
    raw = tuple(int(round(x)) for x in hkl)
    g = _gcd3(raw)
    return (raw[0] // g, raw[1] // g, raw[2] // g)


def _family_label(hkl: tuple[int, int, int]) -> str:
    vals = sorted((abs(hkl[0]), abs(hkl[1]), abs(hkl[2])), reverse=True)
    return "{" + "".join(str(v) for v in vals) + "}"


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return 0.0
    cosang = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _area_2d(vectors: np.ndarray) -> float:
    return float(np.linalg.norm(np.cross(vectors[0], vectors[1])))


def _as_tuple2(matrix: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    arr = np.asarray(matrix, float)
    return (
        (float(arr[0, 0]), float(arr[0, 1])),
        (float(arr[1, 0]), float(arr[1, 1])),
    )


def _signed_equivalents(
    struct: Structure,
    hkl: tuple[int, int, int],
    *,
    proper_only: bool,
    include_opposites: bool = True,
) -> list[tuple[int, int, int]]:
    ops = SpacegroupAnalyzer(struct, symprec=1e-3).get_symmetry_operations(cartesian=True)
    rec_t = struct.lattice.reciprocal_lattice.matrix.T
    g0 = struct.lattice.reciprocal_lattice.get_cartesian_coords(hkl)
    out: set[tuple[int, int, int]] = set()
    for op in ops:
        if proper_only and np.linalg.det(op.rotation_matrix) < 0.999:
            continue
        g_rot = op.rotation_matrix @ g0
        coeff = np.linalg.solve(rec_t, g_rot)
        hkl_i = _primitive_hkl(coeff)
        if hkl_i != (0, 0, 0):
            out.add(hkl_i)
    out.add(_primitive_hkl(hkl))
    if include_opposites:
        out.update(tuple(-x for x in item) for item in list(out))
    return sorted(out, key=lambda t: (abs(t[0]) + abs(t[1]) + abs(t[2]), t))


def _bulk_formula_counts(struct: Structure, charges: dict[str, int]) -> Counter:
    counts = Counter(str(site.specie.symbol) for site in struct.sites if str(site.specie.symbol) in charges)
    if not counts:
        counts = Counter(str(site.specie.symbol) for site in struct.sites)
    g = 0
    for value in counts.values():
        g = math.gcd(g, int(value))
    if g > 1:
        return Counter({key: value // g for key, value in counts.items()})
    return counts


def _is_stoich_multiple(counts: Counter, formula: Counter) -> bool:
    if not counts:
        return False
    elems = set(formula)
    if set(counts) != elems:
        return False
    ratios = []
    for el in sorted(elems):
        if formula[el] == 0:
            return False
        ratios.append(counts[el] / formula[el])
    return max(ratios) - min(ratios) < 1e-8


def _layer_groups(
    struct: Structure,
    hkl: tuple[int, int, int],
    charges: dict[str, int],
    *,
    layer_tol: float,
) -> list[dict]:
    gvec = struct.lattice.reciprocal_lattice.get_cartesian_coords(hkl)
    period = 1.0 / np.linalg.norm(gvec)
    phase_tol = layer_tol / period

    frac = np.asarray([site.frac_coords for site in struct.sites], dtype=float)
    phases = (frac @ np.asarray(hkl, dtype=float)) % 1.0
    order = np.argsort(phases)

    raw_groups: list[list[int]] = []
    current: list[int] = []
    last = None
    for idx in order:
        phase = float(phases[idx])
        if last is None or abs(phase - last) <= phase_tol:
            current.append(int(idx))
        else:
            raw_groups.append(current)
            current = [int(idx)]
        last = phase
    if current:
        raw_groups.append(current)

    if len(raw_groups) > 1:
        first = raw_groups[0]
        last_group = raw_groups[-1]
        gap_wrap = (float(phases[first[0]]) + 1.0) - float(phases[last_group[-1]])
        if gap_wrap <= phase_tol:
            raw_groups = [last_group + first] + raw_groups[1:-1]

    formula = _bulk_formula_counts(struct, charges)
    layers = []
    for group in raw_groups:
        counts = Counter(str(struct.sites[i].specie.symbol) for i in group)
        q = int(sum(charges.get(el, 0) * n_el for el, n_el in counts.items()))
        layers.append({
            "counts": dict(sorted(counts.items())),
            "charge": q,
            "stoich_multiple": _is_stoich_multiple(counts, formula),
        })
    return layers


def _dedupe_patterns(layers: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for layer in layers:
        key = (tuple(sorted(layer["counts"].items())), int(layer["charge"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(layer)
    return out


def _has_opposite_charged_patterns(patterns: list[dict]) -> bool:
    charges = [int(row["charge"]) for row in patterns]
    return any(q > 0 for q in charges) and any(q < 0 for q in charges)


def _classify_patterns(patterns: list[dict]) -> str:
    if not patterns:
        return "unknown"
    charges = [int(row["charge"]) for row in patterns]
    if all(q == 0 for q in charges):
        if all(bool(row["stoich_multiple"]) for row in patterns):
            return "non-polar"
        return "neutral-mixed"
    if any(q > 0 for q in charges) and any(q < 0 for q in charges):
        return "polar"
    return "charged-mixed"


def _richness(pattern: dict, charges: dict[str, int]) -> str:
    q = int(pattern["charge"])
    if q > 0:
        positives = [el for el in pattern["counts"] if charges.get(el, 0) > 0]
        return "cation-rich" if positives else "positive"
    if q < 0:
        negatives = [el for el in pattern["counts"] if charges.get(el, 0) < 0]
        return "anion-rich" if negatives else "negative"
    if pattern["stoich_multiple"]:
        return "stoichiometric"
    return "neutral-mixed"


def analyze_terminations(
    cif_path: str,
    charges: dict[str, int],
    *,
    material_name: str = "material",
    max_index: int = 1,
    proper_only: bool = True,
    layer_tol: float = 0.08,
) -> list[FacetTermination]:
    """
    Analyze low-index facet layer terminations directly from a CIF.

    This intentionally uses layer chemistry and formal charge rather than
    material-specific cation/anion labels, so it also works for ternaries.
    """
    struct = Structure.from_file(cif_path)
    missing = sorted({str(site.specie.symbol) for site in struct.sites} - set(charges))
    if missing:
        raise ValueError(f"charges missing for CIF species {missing}")

    reps = get_symmetrically_distinct_miller_indices(struct, max_index=max_index)
    terms: list[FacetTermination] = []

    for rep in sorted(reps, key=lambda t: (abs(t[0]) + abs(t[1]) + abs(t[2]), t)):
        symmetry_signed = set(_signed_equivalents(
            struct,
            tuple(rep),
            proper_only=proper_only,
            include_opposites=False,
        ))
        signed = _signed_equivalents(
            struct,
            tuple(rep),
            proper_only=proper_only,
            include_opposites=True,
        )

        by_hkl: dict[tuple[int, int, int], list[dict]] = {}
        has_charged = False
        has_polar_split = False
        classifications = []
        for hkl in signed:
            patterns = _dedupe_patterns(_layer_groups(
                struct,
                hkl,
                charges,
                layer_tol=layer_tol,
            ))
            by_hkl[hkl] = patterns
            classifications.append(_classify_patterns(patterns))
            if _has_opposite_charged_patterns(patterns):
                has_charged = True
            opposite = tuple(-x for x in hkl)
            if (
                _has_opposite_charged_patterns(patterns)
                and hkl in symmetry_signed
                and opposite not in symmetry_signed
            ):
                has_polar_split = True

        if has_polar_split:
            family_status = "polar"
        elif has_charged:
            family_status = "termination-sensitive"
        elif all(cls in {"non-polar", "neutral-mixed"} for cls in classifications):
            family_status = "non-polar"
        else:
            family_status = "mixed"

        family = _family_label(tuple(rep))
        seen: set[tuple] = set()
        for hkl in signed:
            for pattern in by_hkl[hkl]:
                key = (hkl, tuple(sorted(pattern["counts"].items())), int(pattern["charge"]))
                if key in seen:
                    continue
                seen.add(key)
                terms.append(FacetTermination(
                    material=material_name,
                    cif=cif_path,
                    family=family,
                    hkl=hkl,
                    charge=int(pattern["charge"]),
                    richness=_richness(pattern, charges),
                    counts={str(k): int(v) for k, v in pattern["counts"].items()},
                    stoich_multiple=bool(pattern["stoich_multiple"]),
                    family_status=family_status,
                ))
    return terms


def classify_pair(
    core: FacetTermination,
    shell: FacetTermination,
    *,
    allow_charged_neutral: bool = False,
) -> str | None:
    qc = int(core.charge)
    qs = int(shell.charge)
    if qc == 0 and qs == 0:
        return "neutral-neutral"
    if qc * qs < 0:
        return "opposite-charge"
    if allow_charged_neutral and (qc == 0 or qs == 0):
        return "charged-neutral"
    return None


def enumerate_interface_candidates(
    core_terms: list[FacetTermination],
    shell_terms: list[FacetTermination],
    *,
    allow_charged_neutral: bool = False,
) -> list[InterfaceCandidate]:
    candidates: list[InterfaceCandidate] = []
    for core in core_terms:
        for shell in shell_terms:
            compatibility = classify_pair(
                core,
                shell,
                allow_charged_neutral=allow_charged_neutral,
            )
            if compatibility is None:
                continue
            abs_residual = abs(int(core.charge) + int(shell.charge))
            class_rank = {
                "opposite-charge": 0,
                "neutral-neutral": 1,
                "charged-neutral": 2,
            }.get(compatibility, 9)
            area_proxy = sum(abs(x) for x in core.hkl) + sum(abs(x) for x in shell.hkl)
            candidates.append(InterfaceCandidate(
                core=core,
                shell=shell,
                compatibility=compatibility,
                score=(class_rank, abs_residual, area_proxy),
            ))
    return sorted(candidates, key=lambda cand: cand.score)


def surface_lattice_vectors(cif_path: str, hkl: tuple[int, int, int]) -> np.ndarray:
    """
    Return two in-plane vectors for a Miller surface.

    Pymatgen's SlabGenerator orients the unit cell with the first two lattice
    vectors in the surface plane and the third along the slab normal.
    """
    struct = Structure.from_file(cif_path)
    gen = SlabGenerator(
        struct,
        hkl,
        min_slab_size=8.0,
        min_vacuum_size=8.0,
        center_slab=True,
        in_unit_planes=True,
        reorient_lattice=True,
    )
    return np.asarray(gen.oriented_unit_cell.lattice.matrix[:2], float)


def best_zsl_match(
    core: FacetTermination,
    shell: FacetTermination,
    *,
    max_area_ratio_tol: float = 0.09,
    max_area: float = 400.0,
    max_length_tol: float = 0.03,
    max_angle_tol: float = 0.01,
) -> LatticeMatch | None:
    """
    Find the lowest-area Zur-McGill/ZSL 2D lattice match for a candidate pair.

    The shell is treated as the film and the core as the substrate. This only
    screens epitaxial commensurability; it does not build slabs or rank energies.
    """
    core_vecs = surface_lattice_vectors(core.cif, core.hkl)
    shell_vecs = surface_lattice_vectors(shell.cif, shell.hkl)
    zsl = ZSLGenerator(
        max_area_ratio_tol=max_area_ratio_tol,
        max_area=max_area,
        max_length_tol=max_length_tol,
        max_angle_tol=max_angle_tol,
        bidirectional=False,
    )
    match = next(zsl(shell_vecs, core_vecs, lowest=True), None)
    if match is None:
        return None

    film = np.asarray(match.film_sl_vectors, float)
    substrate = np.asarray(match.substrate_sl_vectors, float)
    film_lengths = np.linalg.norm(film, axis=1)
    substrate_lengths = np.linalg.norm(substrate, axis=1)
    length_mismatch = np.abs(film_lengths - substrate_lengths) / np.maximum(substrate_lengths, 1e-12)
    area_film = _area_2d(film)
    area_sub = _area_2d(substrate)
    area_mismatch = abs(area_film - area_sub) / max(area_sub, 1e-12)
    angle_mismatch = abs(_angle_deg(film[0], film[1]) - _angle_deg(substrate[0], substrate[1]))

    return LatticeMatch(
        area=0.5 * (area_film + area_sub),
        area_mismatch=float(area_mismatch),
        max_length_mismatch=float(np.max(length_mismatch)),
        angle_mismatch_deg=float(angle_mismatch),
        film_transformation=_as_tuple2(match.film_transformation),
        substrate_transformation=_as_tuple2(match.substrate_transformation),
    )


def filter_lattice_matched_candidates(
    candidates: list[InterfaceCandidate],
    *,
    max_area_ratio_tol: float = 0.09,
    max_area: float = 400.0,
    max_length_tol: float = 0.03,
    max_angle_tol: float = 0.01,
) -> list[InterfaceCandidate]:
    matched: list[InterfaceCandidate] = []
    for cand in candidates:
        lm = best_zsl_match(
            cand.core,
            cand.shell,
            max_area_ratio_tol=max_area_ratio_tol,
            max_area=max_area,
            max_length_tol=max_length_tol,
            max_angle_tol=max_angle_tol,
        )
        if lm is None:
            continue
        score = (
            cand.score[0],
            cand.score[1],
            lm.max_length_mismatch,
            lm.angle_mismatch_deg,
            lm.area,
            *cand.score[2:],
        )
        matched.append(InterfaceCandidate(
            core=cand.core,
            shell=cand.shell,
            compatibility=cand.compatibility,
            score=score,
            lattice_match=lm,
        ))
    return sorted(matched, key=lambda cand: cand.score)


def unique_family_terminations(terms: list[FacetTermination]) -> list[FacetTermination]:
    """
    Collapse symmetry-related signed facets into one family/termination pattern.

    The full signed list is useful when a builder needs explicit orientation.
    For candidate screening, this compact view keeps the table readable.
    """
    seen: set[tuple] = set()
    out: list[FacetTermination] = []
    for term in terms:
        key = (
            term.material,
            term.family,
            int(term.charge),
            term.richness,
            tuple(sorted(term.counts.items())),
            term.family_status,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
    return out


def _local_frame_from_lattice(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(matrix[0], float)
    b = np.asarray(matrix[1], float)
    c = np.asarray(matrix[2], float)
    u = a / np.linalg.norm(a)
    n = np.cross(a, b)
    if np.dot(n, c) < 0:
        n *= -1.0
    n = n / np.linalg.norm(n)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    return u, v, n


def _to_local_matrix(matrix: np.ndarray) -> np.ndarray:
    u, v, n = _local_frame_from_lattice(matrix)
    frame = np.stack([u, v, n], axis=1)
    return np.asarray(matrix, float) @ frame


def _coords_in_local_frame(coords: np.ndarray, lattice_matrix: np.ndarray) -> np.ndarray:
    u, v, n = _local_frame_from_lattice(lattice_matrix)
    frame = np.stack([u, v, n], axis=1)
    return np.asarray(coords, float) @ frame


def _oriented_supercell_local(
    cif_path: str,
    hkl: tuple[int, int, int],
    *,
    radius: float,
    pad: float,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    struct = Structure.from_file(cif_path)
    slab_gen = SlabGenerator(
        struct,
        hkl,
        min_slab_size=8.0,
        min_vacuum_size=8.0,
        center_slab=True,
        in_unit_planes=True,
        reorient_lattice=True,
    )
    oriented = slab_gen.oriented_unit_cell
    matrix = np.asarray(oriented.lattice.matrix, float)
    lengths = np.linalg.norm(matrix, axis=1)
    reps = [int(math.ceil((radius + pad) / max(length, 1e-12))) + 2 for length in lengths]
    syms: list[str] = []
    pts: list[np.ndarray] = []
    base = np.asarray(oriented.cart_coords, float)
    site_symbols = [str(site.specie.symbol) for site in oriented.sites]
    for i in range(-reps[0], reps[0] + 1):
        for j in range(-reps[1], reps[1] + 1):
            for k in range(-reps[2], reps[2] + 1):
                shift = i * matrix[0] + j * matrix[1] + k * matrix[2]
                coords = base + shift
                local = _coords_in_local_frame(coords, matrix)
                syms.extend(site_symbols)
                pts.extend(local)
    return syms, np.asarray(pts, float), _to_local_matrix(matrix)


def _layer_keys_for_points(
    syms: list[str],
    pts: np.ndarray,
    charges: dict[str, int],
    *,
    layer_tol: float,
) -> list[tuple[float, dict[str, int], int]]:
    if len(pts) == 0:
        return []
    z = np.asarray(pts, float)[:, 2]
    order = np.argsort(z)
    groups: list[list[int]] = []
    current: list[int] = []
    last = None
    for idx in order:
        val = float(z[idx])
        if last is None or abs(val - last) <= layer_tol:
            current.append(int(idx))
        else:
            groups.append(current)
            current = [int(idx)]
        last = val
    if current:
        groups.append(current)

    out = []
    for group in groups:
        counts = Counter(syms[i] for i in group)
        q = int(sum(charges.get(el, 0) * n_el for el, n_el in counts.items()))
        out.append((float(np.mean(z[group])), dict(sorted(counts.items())), q))
    return out


def _counts_proportional(a: dict[str, int], b: dict[str, int]) -> bool:
    if set(a) != set(b) or not a:
        return False
    ratios = [a[key] / b[key] for key in sorted(a)]
    return max(ratios) - min(ratios) < 1e-8


def _pick_interface_layer_z(
    syms: list[str],
    pts: np.ndarray,
    term: FacetTermination,
    charges: dict[str, int],
    *,
    layer_tol: float,
    side: str,
) -> float:
    layers = _layer_keys_for_points(syms, pts, charges, layer_tol=layer_tol)
    z_mid = float(np.median(pts[:, 2])) if len(pts) else 0.0
    matches = [
        z for z, counts, q in layers
        if int(q) == int(term.charge) and _counts_proportional(counts, term.counts)
    ]
    if matches:
        return min(matches, key=lambda z: abs(z - z_mid))
    # Fallback: choose the charged layer with the correct sign, then the sidemost layer.
    sign = 1 if term.charge > 0 else (-1 if term.charge < 0 else 0)
    if sign:
        signed = [z for z, _counts, q in layers if q * sign > 0]
        if signed:
            return min(signed, key=lambda z: abs(z - z_mid))
    return z_mid


def _dedupe_by_distance(
    syms: list[str],
    pts: np.ndarray,
    *,
    min_dist: float,
) -> tuple[list[str], np.ndarray, int]:
    if len(pts) == 0:
        return syms, pts, 0
    keep = np.ones(len(pts), dtype=bool)
    tree = cKDTree(pts)
    removed = 0
    for i in range(len(pts)):
        if not keep[i]:
            continue
        nbrs = tree.query_ball_point(pts[i], r=min_dist)
        for j in nbrs:
            if j <= i or not keep[j]:
                continue
            keep[j] = False
            removed += 1
    return [s for s, ok in zip(syms, keep) if ok], pts[keep], removed


def _interface_layer_summary(
    syms: list[str],
    pts: np.ndarray,
    charges: dict[str, int],
    *,
    side: str,
    layer_tol: float,
) -> dict:
    if len(pts) == 0:
        return {"counts": {}, "charge": 0, "atoms": 0, "z": None}
    z = np.asarray(pts, float)[:, 2]
    z0 = float(np.max(z) if side == "core" else np.min(z))
    if side == "core":
        mask = z >= z0 - layer_tol
    else:
        mask = z <= z0 + layer_tol
    counts = Counter(s for s, keep in zip(syms, mask) if keep)
    q = int(sum(charges.get(el, 0) * n for el, n in counts.items()))
    return {
        "counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "charge": q,
        "atoms": int(sum(counts.values())),
        "z": z0,
    }


def _unique_layer_positions(z_values: np.ndarray, *, layer_tol: float) -> list[float]:
    if len(z_values) == 0:
        return []
    vals = np.sort(np.asarray(z_values, float))
    groups: list[list[float]] = []
    cur: list[float] = []
    last = None
    for val in vals:
        if last is None or abs(float(val) - last) <= layer_tol:
            cur.append(float(val))
        else:
            groups.append(cur)
            cur = [float(val)]
        last = float(val)
    if cur:
        groups.append(cur)
    return [float(np.mean(group)) for group in groups]


def _layer_cut_bounds(
    core_pts: np.ndarray,
    shell_pts: np.ndarray,
    *,
    core_layers: int,
    shell_layers: int,
    interface_distance: float,
    layer_tol: float,
) -> tuple[float, float]:
    core_layers_z = [z for z in _unique_layer_positions(core_pts[:, 2], layer_tol=layer_tol) if z <= layer_tol]
    shell_layers_z = [
        z for z in _unique_layer_positions(shell_pts[:, 2], layer_tol=layer_tol)
        if z >= interface_distance - layer_tol
    ]
    core_layers_z.sort(reverse=True)
    shell_layers_z.sort()
    if not core_layers_z:
        core_min = -float("inf")
    else:
        idx = min(max(int(core_layers), 1), len(core_layers_z)) - 1
        core_min = core_layers_z[idx] - layer_tol
    if not shell_layers_z:
        shell_max = float("inf")
    else:
        idx = min(max(int(shell_layers), 1), len(shell_layers_z)) - 1
        shell_max = shell_layers_z[idx] + layer_tol
    return core_min, shell_max


def _apply_shell_zsl_match(
    shell_pts: np.ndarray,
    shell_local_matrix: np.ndarray,
    core_local_matrix: np.ndarray,
    match: LatticeMatch | None,
) -> tuple[np.ndarray, dict | None]:
    if match is None:
        return shell_pts, None
    film_prim = np.asarray(shell_local_matrix[:2, :2], float)
    substrate_prim = np.asarray(core_local_matrix[:2, :2], float)
    film_t = np.asarray(match.film_transformation, float)
    substrate_t = np.asarray(match.substrate_transformation, float)
    film_super = film_t @ film_prim
    substrate_super = substrate_t @ substrate_prim
    try:
        transform = np.linalg.solve(film_super, substrate_super)
    except np.linalg.LinAlgError:
        return shell_pts, None
    out = shell_pts.copy()
    out[:, :2] = out[:, :2] @ transform
    return out, {
        "in_plane_transform": [
            [float(transform[0, 0]), float(transform[0, 1])],
            [float(transform[1, 0]), float(transform[1, 1])],
        ],
    }


def _convex_footprint_mask(
    points_xy: np.ndarray,
    footprint_xy: np.ndarray,
    *,
    margin: float,
) -> tuple[np.ndarray, dict]:
    points_xy = np.asarray(points_xy, float)
    footprint_xy = np.asarray(footprint_xy, float)
    if len(points_xy) == 0:
        return np.zeros(0, dtype=bool), {"method": "empty", "margin": float(margin)}
    if len(footprint_xy) == 0:
        return np.zeros(len(points_xy), dtype=bool), {"method": "empty", "margin": float(margin)}

    xmin, ymin = np.min(footprint_xy, axis=0) - float(margin)
    xmax, ymax = np.max(footprint_xy, axis=0) + float(margin)
    bbox_mask = (
        (points_xy[:, 0] >= xmin)
        & (points_xy[:, 0] <= xmax)
        & (points_xy[:, 1] >= ymin)
        & (points_xy[:, 1] <= ymax)
    )
    meta = {
        "method": "bbox",
        "margin": float(margin),
        "bounds": [float(xmin), float(xmax), float(ymin), float(ymax)],
    }

    centered = footprint_xy - np.mean(footprint_xy, axis=0)
    if len(footprint_xy) < 3 or np.linalg.matrix_rank(centered, tol=1e-8) < 2:
        return bbox_mask, meta

    try:
        hull = ConvexHull(footprint_xy)
    except QhullError:
        return bbox_mask, meta

    equations = np.asarray(hull.equations, float)
    values = points_xy @ equations[:, :2].T + equations[:, 2]
    hull_mask = np.all(values <= float(margin), axis=1)
    meta.update({
        "method": "convex_hull",
        "vertices": int(len(hull.vertices)),
    })
    return hull_mask, meta


def _mushroom_footprint_mask(
    shell_pts: np.ndarray,
    core_footprint_xy: np.ndarray,
    shell_z_mask: np.ndarray,
    *,
    interface_z: float,
    margin: float,
    overhang: float | None = None,
) -> tuple[np.ndarray, dict]:
    shell_pts = np.asarray(shell_pts, float)
    core_footprint_xy = np.asarray(core_footprint_xy, float)
    if len(shell_pts) == 0 or len(core_footprint_xy) == 0:
        return np.zeros(len(shell_pts), dtype=bool), {
            "method": "mushroom",
            "margin": float(margin),
            "overhang": 0.0,
        }

    xmin0, ymin0 = np.min(core_footprint_xy, axis=0) - float(margin)
    xmax0, ymax0 = np.max(core_footprint_xy, axis=0) + float(margin)
    core_half_x = 0.5 * (xmax0 - xmin0)
    core_half_y = 0.5 * (ymax0 - ymin0)
    cx = 0.5 * (xmin0 + xmax0)
    cy = 0.5 * (ymin0 + ymax0)

    shell_half = shell_pts[shell_z_mask] if np.any(shell_z_mask) else shell_pts
    z = shell_pts[:, 2]
    z_shell = shell_half[:, 2]
    z_min = float(max(interface_z, np.min(z_shell)))
    z_max = float(np.max(z_shell))
    z_span = max(1e-8, z_max - z_min)

    if overhang is None:
        max_shell_x = float(np.max(np.abs(shell_half[:, 0] - cx)))
        max_shell_y = float(np.max(np.abs(shell_half[:, 1] - cy)))
        overhang_x = max(0.0, max_shell_x - core_half_x)
        overhang_y = max(0.0, max_shell_y - core_half_y)
    else:
        overhang_x = max(0.0, float(overhang))
        overhang_y = max(0.0, float(overhang))

    t = np.clip((z - z_min) / z_span, 0.0, 1.0)
    # Smoothly open the neck into the spherical cap without an abrupt ledge.
    f = np.sin(0.5 * np.pi * t)
    hx = core_half_x + overhang_x * f
    hy = core_half_y + overhang_y * f
    mask = (np.abs(shell_pts[:, 0] - cx) <= hx) & (np.abs(shell_pts[:, 1] - cy) <= hy)
    return mask, {
        "method": "mushroom",
        "margin": float(margin),
        "overhang_x": float(overhang_x),
        "overhang_y": float(overhang_y),
        "interface_z": float(interface_z),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "bounds_neck": [float(xmin0), float(xmax0), float(ymin0), float(ymax0)],
    }


def _outer_box_planes(
    *,
    hx: float,
    hy: float,
    core_z_min: float,
    shell_z_max: float,
) -> list[tuple[np.ndarray, float]]:
    return [
        (np.array([1.0, 0.0, 0.0]), float(hx)),
        (np.array([-1.0, 0.0, 0.0]), float(hx)),
        (np.array([0.0, 1.0, 0.0]), float(hy)),
        (np.array([0.0, -1.0, 0.0]), float(hy)),
        (np.array([0.0, 0.0, -1.0]), float(-core_z_min)),
        (np.array([0.0, 0.0, 1.0]), float(shell_z_max)),
    ]


def _resolve_facet_terminations_for_structure(
    struct: Structure,
    seeds: list[Facet],
    charges: dict[str, int],
) -> list[Facet]:
    resolved: list[Facet] = []
    for f in seeds:
        term = getattr(f, "termination", None)
        if not term:
            resolved.append(f)
            continue
        hkl = (abs(int(f.h)), abs(int(f.k)), abs(int(f.l)))
        if hkl == (0, 0, 0):
            resolved.append(f)
            continue
        scored = []
        for cand in (hkl, (-hkl[0], -hkl[1], -hkl[2])):
            n = unit_normal(struct, cand)
            coords = np.asarray([site.coords for site in struct.sites], float)
            proj = coords @ n
            top = float(np.max(proj))
            tol = max(1e-4, 1e-3 * max(1.0, abs(top)))
            q = int(sum(
                charges.get(str(site.specie.symbol), 0)
                for site, p in zip(struct.sites, proj)
                if top - float(p) <= tol
            ))
            scored.append((cand, q))
        chosen, _q = max(scored, key=lambda rec: rec[1]) if term == "cation_rich" else min(scored, key=lambda rec: rec[1])
        resolved.append(Facet(chosen[0], chosen[1], chosen[2], f.gamma, termination=term))
    return resolved


def _size_to_radius_aspect(struct: Structure, size_unit_cells) -> tuple[float, tuple[float, float, float]]:
    if isinstance(size_unit_cells, (int, float)):
        size = np.array([float(size_unit_cells)] * 3, dtype=float)
    else:
        size = np.asarray(size_unit_cells, dtype=float)
    lengths = np.array([np.linalg.norm(v) for v in struct.lattice.matrix], dtype=float)
    physical = size * lengths
    radius = float(np.min(physical))
    if radius <= 0:
        raise ValueError(f"size_unit_cells must be positive, got {size_unit_cells!r}")
    return radius, tuple(float(v / radius) for v in physical)


def _hkl_to_local_z(points: np.ndarray, struct: Structure, hkl: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    n = unit_normal(struct, hkl)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(ref, n))) > 0.85:
        ref = np.array([0.0, 1.0, 0.0])
    u = ref - float(np.dot(ref, n)) * n
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    frame = np.stack([u, v, n], axis=1)
    return np.asarray(points, float) @ frame, frame


def _wulff_outer_planes_local(
    planes: list[tuple[np.ndarray, float]],
    frame: np.ndarray,
    exclude_normal: np.ndarray,
    *,
    exclude_cos: float = 0.95,
) -> list[tuple[np.ndarray, float]]:
    out = []
    for n, d in planes:
        n_local = frame.T @ np.asarray(n, float)
        n_norm = np.linalg.norm(n_local)
        if n_norm == 0:
            continue
        n_local = n_local / n_norm
        if abs(float(np.dot(n_local, exclude_normal))) >= exclude_cos:
            continue
        out.append((n_local, float(d)))
    return out


def _planes_from_actual_hull(
    pts: np.ndarray,
    directions: list[np.ndarray],
    *,
    exclude_interface: bool = True,
    interface_cos: float = 0.95,
) -> list[tuple[np.ndarray, float]]:
    if len(pts) == 0:
        return []
    out: list[tuple[np.ndarray, float]] = []
    seen: set[tuple[float, float, float]] = set()
    z_axis = np.array([0.0, 0.0, 1.0])
    for raw in directions:
        n = np.asarray(raw, float)
        norm = float(np.linalg.norm(n))
        if norm <= 0:
            continue
        n = n / norm
        if exclude_interface and abs(float(np.dot(n, z_axis))) >= interface_cos:
            continue
        key = tuple(np.round(n, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((n, float(np.max(pts @ n) + 1e-3)))
    return out


def build_janus_candidate(
    candidate: InterfaceCandidate,
    charges: dict[str, int],
    *,
    radius: float,
    interface_distance: float = 2.8,
    layer_tol: float = 0.08,
    min_separation: float = 1.2,
) -> tuple[list[str], np.ndarray, dict]:
    """
    Build an experimental bare Janus particle for one interface candidate.

    The interface is placed at z=0 in a common local frame. Core atoms occupy
    z <= 0 and shell atoms occupy z >= interface_distance. No ligand passivation
    is applied here; this keeps the buried interface untouched.
    """
    pad = max(8.0, interface_distance + 4.0)
    core_syms, core_pts, core_local_matrix = _oriented_supercell_local(
        candidate.core.cif,
        candidate.core.hkl,
        radius=radius,
        pad=pad,
    )
    shell_syms, shell_pts, shell_local_matrix = _oriented_supercell_local(
        candidate.shell.cif,
        candidate.shell.hkl,
        radius=radius,
        pad=pad,
    )
    shell_pts, zsl_transform_meta = _apply_shell_zsl_match(
        shell_pts,
        shell_local_matrix,
        core_local_matrix,
        candidate.lattice_match,
    )
    for arr in (core_pts, shell_pts):
        arr[:, 0] -= float(np.mean(arr[:, 0]))
        arr[:, 1] -= float(np.mean(arr[:, 1]))
    core_z = _pick_interface_layer_z(
        core_syms,
        core_pts,
        candidate.core,
        charges,
        layer_tol=layer_tol,
        side="core",
    )
    shell_z = _pick_interface_layer_z(
        shell_syms,
        shell_pts,
        candidate.shell,
        charges,
        layer_tol=layer_tol,
        side="shell",
    )
    core_pts = core_pts.copy()
    shell_pts = shell_pts.copy()
    core_pts[:, 2] -= core_z
    shell_pts[:, 2] += interface_distance - shell_z

    core_r = np.linalg.norm(core_pts, axis=1)
    shell_r = np.linalg.norm(shell_pts, axis=1)
    core_mask = (core_r <= radius) & (core_pts[:, 2] <= layer_tol)
    shell_mask = (shell_r <= radius) & (shell_pts[:, 2] >= interface_distance - layer_tol)

    core_kept_syms = [s for s, keep in zip(core_syms, core_mask) if keep]
    core_kept_pts = core_pts[core_mask]
    shell_kept_pts = shell_pts[shell_mask]
    syms = list(core_kept_syms)
    pts_parts = [core_kept_pts]
    shell_kept_syms = [s for s, keep in zip(shell_syms, shell_mask) if keep]
    syms.extend(shell_kept_syms)
    pts_parts.append(shell_kept_pts)
    pts = np.vstack(pts_parts) if pts_parts else np.zeros((0, 3), float)
    syms, pts, removed = _dedupe_by_distance(syms, pts, min_dist=min_separation)
    pts = pts - pts.mean(axis=0) if len(pts) else pts

    actual_core = _interface_layer_summary(
        core_kept_syms,
        core_kept_pts,
        charges,
        side="core",
        layer_tol=layer_tol,
    )
    actual_shell = _interface_layer_summary(
        shell_kept_syms,
        shell_kept_pts,
        charges,
        side="shell",
        layer_tol=layer_tol,
    )

    meta = {
        "core_atoms": int(np.count_nonzero(core_mask)),
        "shell_atoms": int(np.count_nonzero(shell_mask)),
        "overlap_removed": int(removed),
        "interface_distance": float(interface_distance),
        "radius": float(radius),
        "actual_interface": {
            "core": actual_core,
            "shell": actual_shell,
            "charge_sum": int(actual_core["charge"] + actual_shell["charge"]),
        },
        "zsl_transform_applied": zsl_transform_meta,
    }
    return syms, pts, meta


def build_janus_candidate_cells(
    candidate: InterfaceCandidate,
    charges: dict[str, int],
    *,
    lateral_cells: tuple[float, float],
    core_layers: int,
    shell_layers: int,
    interface_distance: float = 2.8,
    layer_tol: float = 0.08,
    min_separation: float = 1.2,
) -> tuple[list[str], np.ndarray, dict, list[tuple[np.ndarray, float]]]:
    """
    Build an experimental Janus particle from a common interface-cell footprint.

    x/y are cut as a rectangular patch in repetitions of the matched surface
    cell; z is cut by explicit core/shell layer counts. The returned planes are
    only the external box faces, never the buried interface faces.
    """
    nx, ny = float(lateral_cells[0]), float(lateral_cells[1])
    pad = max(12.0, interface_distance + 6.0)
    nominal_radius = max(nx, ny) * 12.0 + pad
    core_syms, core_pts, core_local_matrix = _oriented_supercell_local(
        candidate.core.cif,
        candidate.core.hkl,
        radius=nominal_radius,
        pad=pad,
    )
    shell_syms, shell_pts, shell_local_matrix = _oriented_supercell_local(
        candidate.shell.cif,
        candidate.shell.hkl,
        radius=nominal_radius,
        pad=pad,
    )
    shell_pts, zsl_transform_meta = _apply_shell_zsl_match(
        shell_pts,
        shell_local_matrix,
        core_local_matrix,
        candidate.lattice_match,
    )
    for arr in (core_pts, shell_pts):
        arr[:, 0] -= float(np.mean(arr[:, 0]))
        arr[:, 1] -= float(np.mean(arr[:, 1]))
    core_z = _pick_interface_layer_z(
        core_syms,
        core_pts,
        candidate.core,
        charges,
        layer_tol=layer_tol,
        side="core",
    )
    shell_z = _pick_interface_layer_z(
        shell_syms,
        shell_pts,
        candidate.shell,
        charges,
        layer_tol=layer_tol,
        side="shell",
    )
    core_pts = core_pts.copy()
    shell_pts = shell_pts.copy()
    core_pts[:, 2] -= core_z
    shell_pts[:, 2] += interface_distance - shell_z

    core_surface_lengths = np.linalg.norm(np.asarray(core_local_matrix[:2, :2], float), axis=1)
    hx = 0.5 * nx * float(core_surface_lengths[0])
    hy = 0.5 * ny * float(core_surface_lengths[1])
    core_z_min, shell_z_max = _layer_cut_bounds(
        core_pts,
        shell_pts,
        core_layers=core_layers,
        shell_layers=shell_layers,
        interface_distance=interface_distance,
        layer_tol=layer_tol,
    )
    core_mask = (
        (np.abs(core_pts[:, 0]) <= hx)
        & (np.abs(core_pts[:, 1]) <= hy)
        & (core_pts[:, 2] <= layer_tol)
        & (core_pts[:, 2] >= core_z_min)
    )
    shell_mask = (
        (np.abs(shell_pts[:, 0]) <= hx)
        & (np.abs(shell_pts[:, 1]) <= hy)
        & (shell_pts[:, 2] >= interface_distance - layer_tol)
        & (shell_pts[:, 2] <= shell_z_max)
    )

    core_kept_syms = [s for s, keep in zip(core_syms, core_mask) if keep]
    core_kept_pts = core_pts[core_mask]
    shell_kept_syms = [s for s, keep in zip(shell_syms, shell_mask) if keep]
    shell_kept_pts = shell_pts[shell_mask]
    syms = list(core_kept_syms) + list(shell_kept_syms)
    pts = np.vstack([core_kept_pts, shell_kept_pts]) if syms else np.zeros((0, 3), float)
    syms, pts, removed = _dedupe_by_distance(syms, pts, min_dist=min_separation)

    planes = _outer_box_planes(hx=hx, hy=hy, core_z_min=core_z_min, shell_z_max=shell_z_max)
    actual_core = _interface_layer_summary(
        core_kept_syms,
        core_kept_pts,
        charges,
        side="core",
        layer_tol=layer_tol,
    )
    actual_shell = _interface_layer_summary(
        shell_kept_syms,
        shell_kept_pts,
        charges,
        side="shell",
        layer_tol=layer_tol,
    )
    shift = pts.mean(axis=0) if len(pts) else np.zeros(3)
    pts = pts - shift if len(pts) else pts
    centered_planes = []
    for n, d in planes:
        centered_planes.append((n, d - float(np.dot(n, shift))))

    meta = {
        "core_atoms": int(np.count_nonzero(core_mask)),
        "shell_atoms": int(np.count_nonzero(shell_mask)),
        "overlap_removed": int(removed),
        "interface_distance": float(interface_distance),
        "interface_mid_z": float(0.5 * interface_distance - shift[2]),
        "lateral_cells": [nx, ny],
        "core_layers": int(core_layers),
        "shell_layers": int(shell_layers),
        "box": {
            "hx": float(hx),
            "hy": float(hy),
            "core_z_min": float(core_z_min),
            "shell_z_max": float(shell_z_max),
        },
        "actual_interface": {
            "core": actual_core,
            "shell": actual_shell,
            "charge_sum": int(actual_core["charge"] + actual_shell["charge"]),
        },
        "zsl_transform_applied": zsl_transform_meta,
    }
    return syms, pts, meta, centered_planes


def build_janus_candidate_wulff(
    candidate: InterfaceCandidate,
    charges: dict[str, int],
    *,
    core_facets: list[Facet],
    shell_facets: list[Facet],
    core_size_unit_cells,
    shell_size_unit_cells,
    proper_only: bool = True,
    interface_distance: float = 2.8,
    layer_tol: float = 0.08,
    min_separation: float = 1.2,
    match_core_footprint: bool = True,
    footprint_margin: float = 1.0,
    footprint_shape: str = "bbox",
    mushroom_overhang: float | None = None,
    core_shape_mode: str = "wulff",
    shell_shape_mode: str = "wulff",
    core_sphere_planes: int = 192,
    shell_sphere_planes: int = 192,
) -> tuple[list[str], np.ndarray, dict, list[tuple[np.ndarray, float]]]:
    core_struct = Structure.from_file(candidate.core.cif)
    shell_struct = Structure.from_file(candidate.shell.cif)
    core_r, core_aspect = _size_to_radius_aspect(core_struct, core_size_unit_cells)
    shell_r, shell_aspect = _size_to_radius_aspect(shell_struct, shell_size_unit_cells)

    if str(core_shape_mode).lower() == "sphere":
        core_full = []
        core_syms, core_pts_cart, core_planes_cart = build_spherical_nanocrystal(
            core_struct,
            core_r,
            n_planes=core_sphere_planes,
        )
    else:
        core_seed = _resolve_facet_terminations_for_structure(core_struct, core_facets, charges)
        core_full = expand_facets(core_struct, core_seed, proper_only=proper_only)
        core_syms, core_pts_cart, core_planes_cart = build_nanocrystal(
            core_struct,
            core_full,
            core_r,
            aspect=core_aspect,
        )
    if str(shell_shape_mode).lower() == "sphere":
        shell_full = []
        shell_syms, shell_pts_cart, shell_planes_cart = build_spherical_nanocrystal(
            shell_struct,
            shell_r,
            n_planes=shell_sphere_planes,
        )
    else:
        shell_seed = _resolve_facet_terminations_for_structure(shell_struct, shell_facets, charges)
        shell_full = expand_facets(shell_struct, shell_seed, proper_only=proper_only)
        shell_syms, shell_pts_cart, shell_planes_cart = build_nanocrystal(
            shell_struct,
            shell_full,
            shell_r,
            aspect=shell_aspect,
        )
    core_syms, core_pts_cart = dedupe_points(core_syms, core_pts_cart, tol=1e-3)
    shell_syms, shell_pts_cart = dedupe_points(shell_syms, shell_pts_cart, tol=1e-3)

    core_pts, core_frame = _hkl_to_local_z(core_pts_cart, core_struct, candidate.core.hkl)
    shell_pts, shell_frame = _hkl_to_local_z(shell_pts_cart, shell_struct, candidate.shell.hkl)

    core_surface_xy = surface_lattice_vectors(candidate.core.cif, candidate.core.hkl) @ core_frame
    shell_surface_xy = surface_lattice_vectors(candidate.shell.cif, candidate.shell.hkl) @ shell_frame
    shell_pts, zsl_transform_meta = _apply_shell_zsl_match(
        shell_pts,
        shell_surface_xy[:, :2],
        core_surface_xy[:, :2],
        candidate.lattice_match,
    )
    for arr in (core_pts, shell_pts):
        arr[:, 0] -= float(np.mean(arr[:, 0]))
        arr[:, 1] -= float(np.mean(arr[:, 1]))
    core_z = _pick_interface_layer_z(
        core_syms, core_pts, candidate.core, charges, layer_tol=layer_tol, side="core"
    )
    shell_z = _pick_interface_layer_z(
        shell_syms, shell_pts, candidate.shell, charges, layer_tol=layer_tol, side="shell"
    )
    core_pts = core_pts.copy()
    shell_pts = shell_pts.copy()
    core_pts[:, 2] -= core_z
    shell_pts[:, 2] += interface_distance - shell_z

    core_mask = core_pts[:, 2] <= layer_tol
    shell_z_mask = shell_pts[:, 2] >= interface_distance - layer_tol
    shell_mask = shell_z_mask.copy()
    footprint_meta = {
        "enabled": bool(match_core_footprint),
        "method": "none",
        "margin": float(footprint_margin),
        "shell_atoms_before": int(np.count_nonzero(shell_z_mask)),
        "shell_atoms_after": int(np.count_nonzero(shell_mask)),
        "shell_atoms_removed": 0,
    }
    if match_core_footprint and np.any(core_mask):
        raw_footprint_meta = {}
        shape = str(footprint_shape).lower()
        if shape in {"mushroom", "cap", "hemisphere", "hemispherical"}:
            footprint_mask, raw_footprint_meta = _mushroom_footprint_mask(
                shell_pts,
                core_pts[core_mask][:, :2],
                shell_z_mask,
                interface_z=interface_distance,
                margin=footprint_margin,
                overhang=mushroom_overhang,
            )
        elif shape in {"convex", "convex_hull", "hull"}:
            footprint_mask, raw_footprint_meta = _convex_footprint_mask(
                shell_pts[:, :2],
                core_pts[core_mask][:, :2],
                margin=footprint_margin,
            )
        else:
            footprint_mask, raw_footprint_meta = _convex_footprint_mask(
                shell_pts[:, :2],
                core_pts[core_mask][:, :2],
                margin=footprint_margin,
            )
            xmin, xmax, ymin, ymax = raw_footprint_meta["bounds"]
            footprint_mask = (
                (shell_pts[:, 0] >= xmin)
                & (shell_pts[:, 0] <= xmax)
                & (shell_pts[:, 1] >= ymin)
                & (shell_pts[:, 1] <= ymax)
            )
            raw_footprint_meta["method"] = "bbox"
        shell_mask = shell_z_mask & footprint_mask
        footprint_meta.update(raw_footprint_meta)
        footprint_meta.update({
            "enabled": True,
            "shell_atoms_before": int(np.count_nonzero(shell_z_mask)),
            "shell_atoms_after": int(np.count_nonzero(shell_mask)),
            "shell_atoms_removed": int(np.count_nonzero(shell_z_mask) - np.count_nonzero(shell_mask)),
        })
    core_kept_syms = [s for s, keep in zip(core_syms, core_mask) if keep]
    core_kept_pts = core_pts[core_mask]
    shell_kept_syms = [s for s, keep in zip(shell_syms, shell_mask) if keep]
    shell_kept_pts = shell_pts[shell_mask]
    syms = list(core_kept_syms) + list(shell_kept_syms)
    pts = np.vstack([core_kept_pts, shell_kept_pts]) if syms else np.zeros((0, 3), float)
    syms, pts, removed = _dedupe_by_distance(syms, pts, min_dist=min_separation)

    actual_core = _interface_layer_summary(
        core_kept_syms, core_kept_pts, charges, side="core", layer_tol=layer_tol
    )
    actual_shell = _interface_layer_summary(
        shell_kept_syms, shell_kept_pts, charges, side="shell", layer_tol=layer_tol
    )
    shift = pts.mean(axis=0) if len(pts) else np.zeros(3)
    pts = pts - shift if len(pts) else pts

    directions = []
    for n, _d in core_planes_cart:
        directions.append(core_frame.T @ np.asarray(n, float))
    for n, _d in shell_planes_cart:
        directions.append(shell_frame.T @ np.asarray(n, float))
    outer_planes = _planes_from_actual_hull(pts, directions, exclude_interface=False)

    meta = {
        "core_atoms": int(np.count_nonzero(core_mask)),
        "shell_atoms": int(np.count_nonzero(shell_mask)),
        "overlap_removed": int(removed),
        "interface_distance": float(interface_distance),
        "interface_mid_z": float(0.5 * interface_distance - shift[2]),
        "core_size_unit_cells": core_size_unit_cells,
        "shell_size_unit_cells": shell_size_unit_cells,
        "core_shape_mode": str(core_shape_mode).lower(),
        "shell_shape_mode": str(shell_shape_mode).lower(),
        "core_sphere_planes": int(core_sphere_planes),
        "shell_sphere_planes": int(shell_sphere_planes),
        "shell_footprint_match": footprint_meta,
        "actual_interface": {
            "core": actual_core,
            "shell": actual_shell,
            "charge_sum": int(actual_core["charge"] + actual_shell["charge"]),
        },
        "zsl_transform_applied": zsl_transform_meta,
    }
    return syms, pts, meta, outer_planes
