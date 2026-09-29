# src/builder/facet_reconstruction.py
"""
Polar {111} surface reconstruction for zinc-blende II-VI / III-V nanocrystals.

The step runs after charge-balance passivation and only when the facet seeds
activate both a cation-rich {111} family and an anion-rich {-1-1-1} family.
Facet polarity is read from the actual outer layer (outermost native species
along each <111> direction), not from the sign of the Miller index.

Algorithm:
  1. Anion-terminated {111} facets: remove sub-surface, fully coordinated
     cations with no two vacancies nearest neighbours in the cation network
     (uniform sqrt(3) x sqrt(3) pattern, maximum count).  Every anion left
     two-coordinated by a vacancy becomes the reconstruction ligand (Cl).
     Net charge per vacancy: -q_cat + n_conv * (q_lig - q_an)
     (CdSe: -2 + 3 = +1, InAs: -3 + 6 = +3).
     Surviving outer anions that are still nearest neighbours of each other
     (charge-delocalising runs) are broken by converting alternating ones to
     the ligand (minimum vertex cover), each adding q_lig - q_an.
  2. Cation-terminated {111} facets: strip the ligands that passivated them
     after the build (on-top, bridging or hollow, above the outer layer).
  3. Compensate the accumulated positive charge by removing non-adjacent outer
     cations on the cation-terminated facets, spread evenly (maximin).  Any
     remainder smaller than one cation charge is balanced by adding ligands on
     cation-facet sites: for II-VI mu3 hollows first, then mu2 bridges (on-top
     only as a last resort); for III-V terminal on-top ligands.

Vacancy acceptance: after removal, a native anion must keep CN >= 2 (CN == 2
converts it to the ligand) and a ligand must keep CN >= 1.  Cations on facet
edges (bonded to a ligand or to the outer layer of another {111} facet) are
avoided.  The same pattern is mapped onto all facets of a family with the
cluster's own proper rotations whenever the cluster is symmetric.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .analysis import PairCuts, _pair_cut_calibrated, derive_pair_cuts_from_cif
from .facets import detect_facets_from_nc
from .nc_types import Facet, FacetReconstructionSpec, SurfaceReconstructionSpec, Plane

CN_BULK = 4
LAYER_TOL = 0.4       # Å, half-thickness of one atomic (111) layer
MIN_FACET_ATOMS = 6   # outer-layer atoms needed to call a <111> direction a facet
SYM_TOL = 0.3         # Å, position tolerance when testing cluster rotations
NN_FACTOR = 1.15      # cation-cation nearest-neighbour conflict radius / d_nn


def _native_view(
    symbols: List[str],
    pts: NDArray[np.float64],
    ligand: str,
) -> Tuple[List[str], NDArray[np.float64], List[int]]:
    idx = [i for i, s in enumerate(symbols) if s != ligand]
    return [symbols[i] for i in idx], pts[idx], idx


def _native_facets_and_planes(
    symbols: List[str],
    pts: NDArray[np.float64],
    struct,
    charges: Dict[str, int],
    facet_seeds: List[Facet],
    ligand: str,
    surf_tol: float,
) -> Tuple[List[Facet], List[Plane]]:
    """Detect facets from native scaffold only (no ligands), for stable plane directions."""
    nat_syms, nat_pts, _ = _native_view(symbols, pts, ligand)
    if not nat_syms:
        return [], []
    return detect_facets_from_nc(nat_syms, nat_pts, struct.lattice, charges, facet_seeds, surf_tol)


def _total_q(symbols: List[str], charges: Dict[str, int]) -> int:
    return int(sum(int(charges.get(s, 0)) for s in symbols))


def _bulk_ideal_direction_sets(
    site_sym: str,
    bulk_struct,
    charges: Dict[str, int],
) -> List[List[np.ndarray]]:
    """
    Return all distinct first-shell opposite-charge direction sets for a species.

    Do not assume tetrahedral coordination or one crystallographic site per
    element.  If the CIF contains multiple local environments for the same
    species, each environment contributes one candidate direction set.
    """
    if bulk_struct is None or not hasattr(bulk_struct, "sites") or not hasattr(bulk_struct, "lattice"):
        return []

    site_q = int(charges.get(site_sym, 0))
    if site_q == 0:
        return []

    lattice = bulk_struct.lattice
    opp_sites = [
        s for s in bulk_struct.sites
        if int(charges.get(str(s.specie.symbol), 0)) * site_q < 0
    ]
    if not opp_sites:
        return []

    direction_sets: List[List[np.ndarray]] = []
    seen_keys: Set[Tuple[Tuple[float, float, float], ...]] = set()
    for ref_site in bulk_struct.sites:
        if str(ref_site.specie.symbol) != site_sym:
            continue

        ref_cart = np.asarray(ref_site.coords, float)
        candidates: List[Tuple[float, np.ndarray]] = []
        for opp in opp_sites:
            opp_cart = np.asarray(opp.coords, float)
            for ia in range(-1, 2):
                for ib in range(-1, 2):
                    for ic in range(-1, 2):
                        shift = ia * lattice.matrix[0] + ib * lattice.matrix[1] + ic * lattice.matrix[2]
                        vec = opp_cart + shift - ref_cart
                        dist = float(np.linalg.norm(vec))
                        if dist > 0.1:
                            candidates.append((dist, vec))

        if not candidates:
            continue
        candidates.sort(key=lambda rec: rec[0])
        d_min = candidates[0][0]
        dirs: List[np.ndarray] = []
        for dist, vec in candidates:
            if dist >= 1.2 * d_min:
                break
            unit = vec / np.linalg.norm(vec)
            if all(float(np.dot(unit, old)) < 0.99 for old in dirs):
                dirs.append(unit)

        if not dirs:
            continue
        key = tuple(sorted(tuple(np.round(v, 6)) for v in dirs))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        direction_sets.append(dirs)

    return direction_sets


def _bulk_cn_refs_from_struct(
    bulk_struct,
    charges: Dict[str, int],
    species: Set[str],
) -> Dict[str, int]:
    refs: Dict[str, int] = {}
    for sym in species:
        sets = _bulk_ideal_direction_sets(sym, bulk_struct, charges)
        if sets:
            refs[sym] = max(len(dirs) for dirs in sets)
        else:
            refs[sym] = CN_BULK
    return refs


def _greedy_independent_indices(points: NDArray[np.float64], min_separation: float) -> List[int]:
    pts = np.asarray(points, float)
    if len(pts) == 0:
        return []
    centroid = pts.mean(axis=0)
    order = sorted(
        range(len(pts)),
        key=lambda i: (-float(np.linalg.norm(pts[i] - centroid)), i),
    )
    selected: List[int] = []
    for i in order:
        if all(float(np.linalg.norm(pts[i] - pts[j])) >= min_separation for j in selected):
            selected.append(i)
    return selected


def _maximum_independent_indices(
    points: NDArray[np.float64],
    min_separation: float,
) -> List[int]:
    """
    Return the largest non-adjacent subset under the distance constraint.

    For the small per-facet candidate sets typical here, use exact branch and
    bound.  For very large facets, fall back to a deterministic maximal set so
    runtime stays bounded.
    """
    pts = np.asarray(points, float)
    n = len(pts)
    if n <= 1:
        return list(range(n))
    if n > 64:
        return _greedy_independent_indices(pts, min_separation)

    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    adj = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and d[i, j] < min_separation:
                mask |= 1 << j
        adj[i] = mask

    greedy = _greedy_independent_indices(pts, min_separation)
    best_mask = 0
    for i in greedy:
        best_mask |= 1 << i
    best_count = len(greedy)

    def branch(chosen_mask: int, remaining_mask: int) -> None:
        nonlocal best_mask, best_count
        if remaining_mask == 0:
            count = chosen_mask.bit_count()
            if count > best_count:
                best_count = count
                best_mask = chosen_mask
            return
        if chosen_mask.bit_count() + remaining_mask.bit_count() <= best_count:
            return

        rem_indices = [i for i in range(n) if (remaining_mask >> i) & 1]
        v = max(rem_indices, key=lambda i: (adj[i] & remaining_mask).bit_count())

        branch(chosen_mask | (1 << v), remaining_mask & ~(1 << v) & ~adj[v])
        branch(chosen_mask, remaining_mask & ~(1 << v))

    branch(0, (1 << n) - 1)
    return [i for i in range(n) if (best_mask >> i) & 1]


def _surface_outward_direction(idx: int, pts: NDArray[np.float64], planes: List[Plane], surf_tol: float) -> np.ndarray:
    nearest: Optional[Tuple[float, np.ndarray]] = None
    incident: List[np.ndarray] = []
    for n, d in planes:
        n = np.asarray(n, float)
        nn = np.linalg.norm(n)
        if nn > 1e-12:
            n = n / nn
        depth = float(d) - float(np.dot(pts[idx], n))
        if nearest is None or depth < nearest[0]:
            nearest = (depth, n)
        if depth < surf_tol:
            incident.append(n)
    if incident:
        vec = np.sum(incident, axis=0)
        nv = np.linalg.norm(vec)
        if nv > 1e-12:
            return vec / nv
    return nearest[1] if nearest is not None else np.array([0.0, 0.0, 1.0])


def _actual_opposite_bond_vectors(
    symbols: List[str],
    pts: NDArray[np.float64],
    host_idx: int,
    charges: Dict[str, int],
    pair_cuts: Optional[PairCuts],
) -> List[np.ndarray]:
    host_sym = symbols[host_idx]
    host_q = int(charges.get(host_sym, 0))
    if host_q == 0:
        return []
    vecs: List[np.ndarray] = []
    for j, sym_j in enumerate(symbols):
        if j == host_idx:
            continue
        if int(charges.get(sym_j, 0)) * host_q >= 0:
            continue
        cutoff = _pair_cut_calibrated(host_sym, sym_j, pair_cuts)
        vec = np.asarray(pts[j], float) - np.asarray(pts[host_idx], float)
        dist = float(np.linalg.norm(vec))
        if 0.1 < dist <= cutoff:
            vecs.append(vec / dist)
    return vecs


def _match_missing_ideal_dirs(
    actual_vecs: List[np.ndarray],
    ideal_dirs: List[np.ndarray],
    *,
    min_dot: float = 0.70,
) -> Tuple[List[np.ndarray], float]:
    assigned: Set[int] = set()
    score = 0.0
    for actual in actual_vecs:
        actual = np.asarray(actual, float)
        if np.linalg.norm(actual) < 1e-12:
            continue
        actual = actual / np.linalg.norm(actual)
        best_idx = -1
        best_dot = -2.0
        for k, ideal in enumerate(ideal_dirs):
            if k in assigned:
                continue
            dot = float(np.dot(actual, ideal))
            if dot > best_dot:
                best_idx = k
                best_dot = dot
        if best_idx >= 0 and best_dot >= min_dot:
            assigned.add(best_idx)
            score += best_dot
        else:
            score -= 1.0
    missing = [np.asarray(ideal_dirs[k], float) for k in range(len(ideal_dirs)) if k not in assigned]
    score -= 0.25 * abs(len(missing) - max(0, len(ideal_dirs) - len(actual_vecs)))
    return missing, score


def _strict_missing_vectors_for_hosts(
    symbols: List[str],
    pts: NDArray[np.float64],
    host_indices: List[int],
    charges: Dict[str, int],
    pair_cuts: Optional[PairCuts],
    bulk_struct,
    planes: List[Plane],
    surf_tol: float,
) -> Dict[int, List[np.ndarray]]:
    """
    Missing first-shell directions from the bulk coordination polyhedron.

    This intentionally has no radial/outward fallback and never flips a vector:
    if a crystallographic missing slot cannot be identified, the host is not
    used for reconstruction ligand compensation.
    """
    direction_cache: Dict[str, List[List[np.ndarray]]] = {}
    result: Dict[int, List[np.ndarray]] = {}
    for host_idx in host_indices:
        host_sym = symbols[host_idx]
        if host_sym not in direction_cache:
            direction_cache[host_sym] = _bulk_ideal_direction_sets(host_sym, bulk_struct, charges)
        direction_sets = direction_cache[host_sym]
        if not direction_sets:
            continue

        actual = _actual_opposite_bond_vectors(symbols, pts, host_idx, charges, pair_cuts)
        if not actual:
            continue

        best_missing: List[np.ndarray] = []
        best_score = -float("inf")
        for ideal_dirs in direction_sets:
            missing, score = _match_missing_ideal_dirs(actual, ideal_dirs)
            if score > best_score:
                best_score = score
                best_missing = missing

        if not best_missing:
            continue

        outward = _surface_outward_direction(host_idx, pts, planes, surf_tol)
        outward_slots = []
        for vec in best_missing:
            vec = np.asarray(vec, float)
            norm = np.linalg.norm(vec)
            if norm < 1e-12:
                continue
            vec = vec / norm
            if float(np.dot(vec, outward)) > 0.05:
                outward_slots.append(vec)
        if outward_slots:
            outward_slots.sort(key=lambda v: float(np.dot(v, outward)), reverse=True)
            result[host_idx] = outward_slots
    return result


def _ligand_add_positions_for_slots(
    symbols: List[str],
    pts: NDArray[np.float64],
    slots: List[Tuple[int, np.ndarray]],
    ligand: str,
    pair_cuts: Optional[PairCuts],
) -> List[np.ndarray]:
    positions: List[np.ndarray] = []
    for host_idx, vec in slots:
        vec = np.asarray(vec, float)
        if np.linalg.norm(vec) < 1e-12:
            continue
        vec = vec / np.linalg.norm(vec)
        host = symbols[host_idx]
        bond_len = 0.84 * _pair_cut_calibrated(host, ligand, pair_cuts)
        positions.append(np.asarray(pts[host_idx], float) + bond_len * vec)
    return positions


# --------------------------------------------------------------------------
# {111} reconstruction
# --------------------------------------------------------------------------

@dataclass
class _Polar111:
    hkl: Tuple[int, int, int]
    normal: NDArray[np.float64]
    kind: str              # "anion" | "cation" (outermost native species)
    outer: List[int]       # outer-layer native atom indices
    top: float             # projection of the outer layer on the normal


def _hkl_str(hkl: Tuple[int, int, int]) -> str:
    return "(" + " ".join(str(int(v)) for v in hkl) + ")"


def _zincblende_binary(struct, charges: Dict[str, int], ligand: str) -> Optional[Tuple[str, str]]:
    """Return (cation, anion) for a binary cubic, tetrahedral (zinc-blende) CIF, else None."""
    if struct is None or not hasattr(struct, "sites"):
        return None
    species = {str(site.specie.symbol) for site in struct.sites}
    species.discard(ligand)
    cations = sorted(s for s in species if int(charges.get(s, 0)) > 0)
    anions = sorted(s for s in species if int(charges.get(s, 0)) < 0)
    if len(cations) != 1 or len(anions) != 1:
        return None
    lat = struct.lattice
    if not (abs(lat.a - lat.b) < 1e-3 and abs(lat.a - lat.c) < 1e-3
            and all(abs(ang - 90.0) < 1e-2 for ang in lat.angles)):
        return None
    refs = _bulk_cn_refs_from_struct(struct, charges, {cations[0], anions[0]})
    if refs.get(cations[0]) != CN_BULK or refs.get(anions[0]) != CN_BULK:
        return None
    return cations[0], anions[0]


def _seeds_request_polar_111(facet_seeds: List[Facet]) -> bool:
    """True when the seeds activate both a cation-rich and an anion-rich {111} family."""
    cat = an = False
    for f in facet_seeds or []:
        if (abs(int(f.h)), abs(int(f.k)), abs(int(f.l))) != (1, 1, 1):
            continue
        term = str(f.termination or "").strip().lower().replace("-", "_")
        if term == "cation_rich":
            cat = True
        elif term == "anion_rich":
            an = True
    return cat and an


def _bond_graph(
    symbols: List[str],
    pts: NDArray[np.float64],
    charges: Dict[str, int],
    pair_cuts: Optional[PairCuts],
) -> List[Set[int]]:
    """Opposite-charge neighbour sets within the calibrated pair cutoffs."""
    nb: List[Set[int]] = [set() for _ in symbols]
    elems = sorted(set(symbols))
    cuts: Dict[Tuple[str, str], float] = {}
    for a in elems:
        for b in elems:
            if int(charges.get(a, 0)) * int(charges.get(b, 0)) < 0:
                cuts[(a, b)] = _pair_cut_calibrated(a, b, pair_cuts)
    if not cuts:
        return nb
    tree = cKDTree(pts)
    for i, j in tree.query_pairs(max(cuts.values())):
        cut = cuts.get((symbols[i], symbols[j]))
        if cut is not None and float(np.linalg.norm(pts[i] - pts[j])) <= cut:
            nb[i].add(j)
            nb[j].add(i)
    return nb


def _polar_111_facets(
    symbols: List[str],
    pts: NDArray[np.float64],
    struct,
    cation: str,
    anion: str,
) -> List[_Polar111]:
    recip = np.asarray(struct.lattice.reciprocal_lattice.matrix, float)
    native = np.array([i for i, s in enumerate(symbols) if s in (cation, anion)], dtype=int)
    facets: List[_Polar111] = []
    if len(native) == 0:
        return facets
    for hkl in itertools.product((1, -1), repeat=3):
        n = np.asarray(hkl, float) @ recip
        n /= np.linalg.norm(n)
        proj = pts[native] @ n
        top = float(proj.max())
        layer = [int(native[k]) for k in np.where(proj > top - LAYER_TOL)[0]]
        kinds = {symbols[i] for i in layer}
        if len(layer) < MIN_FACET_ATOMS or len(kinds) != 1:
            continue
        facets.append(_Polar111(
            hkl=tuple(int(v) for v in hkl),
            normal=n,
            kind="cation" if cation in kinds else "anion",
            outer=layer,
            top=top,
        ))
    return facets


def _cluster_rotations(
    symbols: List[str],
    pts: NDArray[np.float64],
    nb: List[Set[int]],
    struct,
    cation: str,
    anion: str,
    ligand: str,
) -> List[Tuple[NDArray[np.float64], Dict[int, int]]]:
    """
    Proper cubic rotations that map the cluster's ionic sites onto themselves.

    Sites are cations and anion positions (native anions plus ligands bridging
    at least two cations, i.e. anions already converted by passivation), so
    the test is insensitive to where charge balance put terminal ligands.
    Returns (R_cart, index map) pairs; the identity is always included.
    """
    site_cls: Dict[int, int] = {}
    for i, s in enumerate(symbols):
        if s == cation:
            site_cls[i] = 0
        elif s == anion:
            site_cls[i] = 1
        elif s == ligand and sum(1 for j in nb[i] if symbols[j] == cation) >= 2:
            site_cls[i] = 1
    idx = np.array(sorted(site_cls), dtype=int)
    cls = np.array([site_cls[i] for i in idx], dtype=int)
    P = pts[idx]
    center = P.mean(axis=0)
    tree = cKDTree(P)
    LT = np.asarray(struct.lattice.matrix, float).T   # columns = lattice vectors
    LT_inv = np.linalg.inv(LT)

    ops: List[Tuple[NDArray[np.float64], Dict[int, int]]] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            Rf = np.zeros((3, 3))
            for r, c in enumerate(perm):
                Rf[r, c] = signs[r]
            if np.linalg.det(Rf) < 0.5:
                continue
            R = LT @ Rf @ LT_inv
            if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
                continue
            Q = (P - center) @ R.T + center
            dist, hit = tree.query(Q)
            if np.all(dist < SYM_TOL) and np.all(cls[hit] == cls):
                ops.append((R, {int(idx[a]): int(idx[b]) for a, b in enumerate(hit)}))
    return ops


def _nn_cation_distance(struct) -> float:
    """Cation-cation nearest-neighbour distance of the fcc sublattice (a / sqrt 2)."""
    return float(struct.lattice.a) / np.sqrt(2.0)


def _in_plane_basis(struct, normal: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two fcc nearest-neighbour vectors at 60 degrees lying in the (111) plane."""
    L = np.asarray(struct.lattice.matrix, float)
    vecs = []
    for frac in itertools.permutations((0.5, 0.5, 0.0)):
        for s1, s2 in itertools.product((1, -1), repeat=2):
            f = np.array(frac, float)
            nz = np.nonzero(f)[0]
            f[nz[0]] *= s1
            f[nz[1]] *= s2
            v = f @ L
            if abs(float(np.dot(v, normal))) < 1e-3:
                vecs.append(v)
    a1 = vecs[0]
    d2 = float(np.dot(a1, a1))
    a2 = next(v for v in vecs if abs(float(np.dot(a1, v)) - 0.5 * d2) < 1e-3)
    return a1, a2


def _colour_classes(
    cands: List[int],
    pts: NDArray[np.float64],
    struct,
    normal: NDArray[np.float64],
) -> List[List[int]]:
    """
    Three-colour the triangular (111) cation layer: (i - j) mod 3 in the
    in-plane lattice basis.  No two members of a class are nearest
    neighbours.  Classes are returned largest first, ties broken by how
    central the class is.
    """
    a1, a2 = _in_plane_basis(struct, normal)
    basis = np.stack([a1, a2], axis=1)
    r0 = pts[cands[0]]
    classes: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    for c in cands:
        ij, *_ = np.linalg.lstsq(basis, pts[c] - r0, rcond=None)
        i, j = (int(round(v)) for v in ij)
        classes[(i - j) % 3].append(c)
    centroid = pts[cands].mean(axis=0)

    def _key(k: int):
        members = classes[k]
        spread = float(np.linalg.norm(pts[members].mean(axis=0) - centroid)) if members else np.inf
        return (-len(members), round(spread, 6), k)

    return [classes[k] for k in sorted(classes, key=_key)]


def _uniform_max_independent(
    cands: List[int],
    pts: NDArray[np.float64],
    struct,
    normal: NDArray[np.float64],
    d_nn: float,
) -> List[int]:
    """
    Maximum non-adjacent subset of cations in one (111) layer.

    The triangular layer is three-coloured; each colour class is a perfectly
    uniform sqrt(3) x sqrt(3) vacancy pattern with no nearest neighbours.  The
    largest class is used unless an exact maximum independent set is larger.
    """
    if not cands:
        return []
    best = _colour_classes(cands, pts, struct, normal)[0]
    exact_local = _maximum_independent_indices(pts[cands], NN_FACTOR * d_nn)
    if len(exact_local) > len(best):
        return [cands[k] for k in exact_local]
    return list(best)


def _fps_nonadjacent(
    cands: List[int],
    pts: NDArray[np.float64],
    k: int,
    d_nn: float,
    existing: Optional[List[int]] = None,
) -> List[int]:
    """Pick up to k maximin-spread cations, none nearest neighbours of each other or of `existing`."""
    existing = list(existing or [])
    chosen: List[int] = []
    if k <= 0 or not cands:
        return chosen
    centroid = pts[cands].mean(axis=0)
    min_sep = NN_FACTOR * d_nn
    while len(chosen) < k:
        ref = existing + chosen
        allowed = [
            c for c in cands
            if c not in chosen and c not in existing
            and all(float(np.linalg.norm(pts[c] - pts[x])) >= min_sep for x in ref)
        ]
        if not allowed:
            break
        if ref:
            ref_pts = pts[ref]
            best = max(allowed, key=lambda c: (
                round(float(np.min(np.linalg.norm(ref_pts - pts[c], axis=1))), 6),
                round(float(np.linalg.norm(pts[c] - centroid)), 6),
                -c,
            ))
        else:
            best = max(allowed, key=lambda c: (round(float(np.linalg.norm(pts[c] - centroid)), 6), -c))
        chosen.append(best)
    return chosen


def _symmetric_selection(
    facets: List[_Polar111],
    cands: Dict[int, List[int]],
    ops: List[Tuple[NDArray[np.float64], Dict[int, int]]],
    choose,
) -> Optional[Dict[int, List[int]]]:
    """
    Choose sites on one reference facet and map them onto every other facet
    of the group with cluster rotations.  Only reference candidates whose
    images are candidates on all facets are eligible.  Returns None when the
    cluster has no rotation relating the facets.
    """
    best_result: Optional[Dict[int, List[int]]] = None
    for ref in range(len(facets)):
        maps: Dict[int, Dict[int, int]] = {}
        ok = True
        for f in range(len(facets)):
            best_map = None
            best_overlap = -1
            target = set(cands[f])
            for R, mp in ops:
                if float(np.dot(R @ facets[ref].normal, facets[f].normal)) < 0.99:
                    continue
                overlap = sum(1 for c in cands[ref] if mp.get(c) in target)
                if overlap > best_overlap:
                    best_overlap, best_map = overlap, mp
            if best_map is None:
                ok = False
                break
            maps[f] = best_map
        if not ok:
            return None
        eligible = [
            c for c in cands[ref]
            if all(maps[f].get(c) in set(cands[f]) for f in range(len(facets)))
        ]
        picked = choose(ref, eligible)
        result = {f: [maps[f][c] for c in picked] for f in range(len(facets))}
        if best_result is None or len(picked) > len(best_result[0]):
            best_result = result
    return best_result


def _min_vertex_cover(
    nodes: List[int],
    pts: NDArray[np.float64],
    d_nn: float,
    anchors: Optional[Set[int]] = None,
) -> List[int]:
    """
    Smallest set of nodes to convert so no two remaining nodes are nearest
    neighbours (minimum vertex cover of the nearest-neighbour graph).  Exact
    per connected component (up to 20 nodes; greedy beyond).  Among equal
    covers, prefer converted nodes that are not adjacent to each other, which
    gives alternating Se/Cl along chains (Se-Se-Se -> Se-Cl-Se), then covers
    that convert more anchors.

    With `anchors`, only adjacencies involving at least one anchor are broken
    (runs touching the reconstructed facet), while non-anchor nodes may still
    be converted to break them.
    """
    cut = NN_FACTOR * d_nn
    adj: Dict[int, Set[int]] = {a: set() for a in nodes}
    for a, b in itertools.combinations(nodes, 2):
        if anchors is not None and a not in anchors and b not in anchors:
            continue
        if float(np.linalg.norm(pts[a] - pts[b])) < cut:
            adj[a].add(b)
            adj[b].add(a)

    cover: List[int] = []
    seen: Set[int] = set()
    for start in sorted(nodes):
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comp.sort()
        edges = [(a, b) for a in comp for b in adj[a] if a < b]
        if not edges:
            continue
        if len(comp) <= 20:
            best = None
            for size in range(1, len(comp) + 1):
                for subset in itertools.combinations(comp, size):
                    chosen = set(subset)
                    if all(a in chosen or b in chosen for a, b in edges):
                        inner = sum(1 for a, b in edges if a in chosen and b in chosen)
                        on_anchor = sum(1 for a in chosen if anchors is None or a in anchors)
                        key = (inner, -on_anchor)
                        if best is None or key < best[0]:
                            best = (key, list(subset))
                if best is not None:
                    break
            cover.extend(best[1])
        else:
            remaining = set(edges)
            while remaining:
                deg: Dict[int, int] = {}
                for a, b in remaining:
                    deg[a] = deg.get(a, 0) + 1
                    deg[b] = deg.get(b, 0) + 1
                v = max(sorted(deg), key=lambda x: deg[x])
                cover.append(v)
                remaining = {e for e in remaining if v not in e}
    return cover


def _interior_outer(outer: List[int], pts: NDArray[np.float64], d_nn: float) -> Set[int]:
    """
    Outer-layer atoms of a (111) facet with all six in-plane nearest
    neighbours present in the layer, i.e. not on a facet edge or vertex.
    Bridging (mu2/mu3) ligands are only placed over these.
    """
    if not outer:
        return set()
    tree = cKDTree(pts[outer])
    cut = NN_FACTOR * d_nn
    return {
        outer[k] for k in range(len(outer))
        if len(tree.query_ball_point(pts[outer[k]], cut)) - 1 >= 6
    }


def _bulk_bond_length(struct) -> float:
    """Zinc-blende cation-anion bond length, a * sqrt(3) / 4."""
    return float(struct.lattice.a) * np.sqrt(3.0) / 4.0


def _bridging_sites(
    host_normal: Dict[int, NDArray[np.float64]],
    pts: NDArray[np.float64],
    mu: int,
    d_nn: float,
    bond: float,
) -> List[Tuple[NDArray[np.float64], Tuple[int, ...]]]:
    """
    Candidate positions for a ligand bridging `mu` (3 or 2) mutually adjacent
    free cations of one cation-terminated (111) facet, at the bulk bond length
    from each host.  mu3 sits in a hollow; mu2 sits over the shared edge,
    tilted toward the side with the most room.  Positions closer than
    1.1 * bond to any non-host atom (e.g. the hollow above a sub-surface
    anion) are rejected.
    """
    hosts = sorted(host_normal)
    if len(hosts) < mu:
        return []
    tree = cKDTree(pts)
    min_clear = 1.1 * bond
    cut = NN_FACTOR * d_nn

    def clearance(pos: NDArray[np.float64], hs: Tuple[int, ...]) -> float:
        near = [j for j in tree.query_ball_point(pos, 2.0 * bond) if j not in hs]
        return min((float(np.linalg.norm(pts[j] - pos)) for j in near), default=np.inf)

    htree = cKDTree(pts[hosts])
    pairs = sorted((hosts[a], hosts[b]) for a, b in htree.query_pairs(cut))
    adj: Dict[int, Set[int]] = {h: set() for h in hosts}
    for a, b in pairs:
        if float(np.dot(host_normal[a], host_normal[b])) > 0.99:
            adj[a].add(b)
            adj[b].add(a)

    sites: List[Tuple[NDArray[np.float64], Tuple[int, ...]]] = []
    if mu == 3:
        for a in hosts:
            for b, c in itertools.combinations(sorted(x for x in adj[a] if x > a), 2):
                if c not in adj[b]:
                    continue
                n = host_normal[a]
                cen = (pts[a] + pts[b] + pts[c]) / 3.0
                h2 = bond * bond - float(np.sum((pts[a] - cen) ** 2))
                if h2 <= 0:
                    continue
                pos = cen + np.sqrt(h2) * n
                if clearance(pos, (a, b, c)) >= min_clear:
                    sites.append((pos, (a, b, c)))
    elif mu == 2:
        for a in hosts:
            for b in sorted(x for x in adj[a] if x > a):
                n = host_normal[a]
                mid = (pts[a] + pts[b]) / 2.0
                h2 = bond * bond - float(np.sum((pts[a] - mid) ** 2))
                if h2 <= 0:
                    continue
                e = (pts[b] - pts[a]) / np.linalg.norm(pts[b] - pts[a])
                t = np.cross(n, e)
                best = None
                for deg in range(-60, 61, 5):
                    th = np.radians(deg)
                    pos = mid + np.sqrt(h2) * (np.cos(th) * n + np.sin(th) * t)
                    clr = clearance(pos, (a, b))
                    if best is None or clr > best[0] + 1e-9:
                        best = (clr, pos)
                if best is not None and best[0] >= min_clear:
                    sites.append((best[1], (a, b)))
    return sites


def _pick_bridging_sites(
    sites: List[Tuple[NDArray[np.float64], Tuple[int, ...]]],
    n: int,
    used_hosts: Set[int],
    placed: List[NDArray[np.float64]],
    avoid: List[NDArray[np.float64]],
    bond: float,
    host_facet: Dict[int, int],
    facet_count: Dict[int, int],
) -> List[Tuple[NDArray[np.float64], Tuple[int, ...]]]:
    """
    Pick up to n sites with disjoint hosts, clear of placed ligands.  Each
    pick goes to the facet with the fewest ligands so far (`facet_count` is
    updated in place), maximin-spread within that facet.
    """
    chosen: List[Tuple[NDArray[np.float64], Tuple[int, ...]]] = []
    used = set(used_hosts)
    taken = list(placed)
    min_clear = 1.1 * bond
    while len(chosen) < n:
        allowed = [
            (pos, hs) for pos, hs in sites
            if not (set(hs) & used)
            and all(float(np.linalg.norm(pos - q)) >= min_clear for q in taken)
        ]
        if not allowed:
            break
        least = min(facet_count.get(host_facet[s[1][0]], 0) for s in allowed)
        allowed = [s for s in allowed if facet_count.get(host_facet[s[1][0]], 0) == least]
        ref = avoid + taken
        if ref:
            ref_arr = np.asarray(ref, float)
            pos, hs = max(allowed, key=lambda s: (
                round(float(np.min(np.linalg.norm(ref_arr - s[0], axis=1))), 6), tuple(-h for h in s[1])))
        else:
            pos, hs = allowed[0]
        chosen.append((pos, hs))
        used.update(hs)
        taken.append(pos)
        facet_count[host_facet[hs[0]]] = facet_count.get(host_facet[hs[0]], 0) + 1
    return chosen


def _anion_ok_after(
    symbols: List[str],
    cn_after: int,
    idx: int,
    anion: str,
    ligand: str,
    *,
    min_native: int,
) -> bool:
    if symbols[idx] == ligand:
        return cn_after >= 1
    if symbols[idx] == anion:
        return cn_after >= min_native
    return True


def reconstruct_polar_facets(
    symbols: List[str],
    pts: NDArray[np.float64],
    *,
    struct,
    facet_seeds: List[Facet],
    charges: Dict[str, int],
    ligand: str,
    surf_tol: float,
    cif_path: str,
    spec: FacetReconstructionSpec | SurfaceReconstructionSpec,
    charge_balance_fn=None,
    verbose: bool = False,
    write_all: bool = False,
    prefix: str = "nanocrystal",
    ledger: Optional[dict] = None,
) -> Tuple[List[str], NDArray[np.float64]]:
    """
    Polar {111} reconstruction of zinc-blende II-VI / III-V nanocrystals.

    See the module docstring for the algorithm.  `charge_balance_fn`,
    `write_all` and `prefix` are accepted for call-site compatibility.  When
    `ledger` is a dict it is filled with a summary of the reconstruction.
    """
    if not spec.enabled:
        return symbols, pts

    recon_ligand = getattr(spec, "ligand", None) or ligand
    info: dict = {"status": "skipped", "ligand": recon_ligand}
    if ledger is not None:
        ledger.clear()
        ledger.update(info)

    def _skip(msg: str):
        print(f"[recon] {msg}; skipping.")
        print("=" * 60)
        if ledger is not None:
            ledger["reason"] = msg
        return symbols, pts

    print(f"\n{'=' * 60}")
    print("[post-treatment:surface-reconstruction] Polar {111} reconstruction (zinc blende)")
    ignored = []
    if getattr(spec, "facets", ()):
        ignored.append("facets")
    if abs(float(getattr(spec, "target_reduction", 0.5)) - 0.5) > 1e-9:
        ignored.append("target_reduction")
    if getattr(spec, "min_separation", None) is not None:
        ignored.append("min_separation")
    if ignored:
        print(f"[recon] note: {', '.join(ignored)} no longer apply and are ignored.")

    ligand_charge = int(charges.get(recon_ligand, -1))
    if ligand_charge >= 0:
        return _skip(f"reconstruction ligand {recon_ligand!r} must be negatively charged")

    pair = _zincblende_binary(struct, charges, recon_ligand)
    if pair is None:
        return _skip("only binary zinc-blende II-VI / III-V structures are supported")
    cation, anion = pair
    if not _seeds_request_polar_111(facet_seeds):
        return _skip("needs both a cation_rich {111} and an anion_rich {-1-1-1} facet family")

    q_cat = int(charges[cation])
    q_an = int(charges[anion])
    pts = np.asarray(pts, float)
    symbols = list(symbols)
    q0 = _total_q(symbols, charges)
    print(f"[recon] {cation}{anion}: q_cat={q_cat:+d} q_an={q_an:+d} ligand={recon_ligand}({ligand_charge:+d})"
          f"  Q_total before = {q0:+d}")

    pair_cuts = derive_pair_cuts_from_cif(cif_path, charges, safety=1.00)
    nb = _bond_graph(symbols, pts, charges, pair_cuts)
    facets = _polar_111_facets(symbols, pts, struct, cation, anion)
    an_facets = [f for f in facets if f.kind == "anion"]
    cat_facets = [f for f in facets if f.kind == "cation"]
    if not an_facets or not cat_facets:
        return _skip(
            f"found {len(an_facets)} anion- and {len(cat_facets)} cation-terminated {{111}} facets; need both"
        )
    print(f"[recon] anion-terminated {{111}}: {', '.join(_hkl_str(f.hkl) for f in an_facets)}")
    print(f"[recon] cation-terminated {{111}}: {', '.join(_hkl_str(f.hkl) for f in cat_facets)}")

    ops = _cluster_rotations(symbols, pts, nb, struct, cation, anion, recon_ligand)
    print(f"[recon] cluster proper rotations: {len(ops)}")
    d_nn = _nn_cation_distance(struct)

    base_symbols = list(symbols)
    all_alive = np.ones(len(symbols), dtype=bool)
    outer_facets_of: Dict[int, Set[int]] = {}
    for fi, f in enumerate(facets):
        for i in f.outer:
            outer_facets_of.setdefault(i, set()).add(fi)

    def cn(i: int, alive: NDArray[np.bool_]) -> int:
        return sum(1 for j in nb[i] if alive[j])

    def touches_other_facet(c: int, own: int, alive: NDArray[np.bool_]) -> bool:
        if outer_facets_of.get(c, set()) - {own}:
            return True
        return any(outer_facets_of.get(a, set()) - {own} for a in nb[c] if alive[a])

    # ---- 1. anion-terminated facets: sub-surface cation vacancies ----------
    an_cands: Dict[int, List[int]] = {}
    for k, f in enumerate(an_facets):
        own = facets.index(f)
        below = f.top - LAYER_TOL
        subs = sorted({
            c for a in f.outer for c in nb[a]
            if symbols[c] == cation and float(pts[c] @ f.normal) < below
        })
        an_cands[k] = [
            c for c in subs
            if cn(c, all_alive) == CN_BULK
            and all(symbols[a] == anion for a in nb[c])
            and not touches_other_facet(c, own, all_alive)
            and all(_anion_ok_after(symbols, cn(a, all_alive) - 1, a, anion, recon_ligand, min_native=2)
                    for a in nb[c])
        ]

    def choose_anion(k: int, eligible: List[int]) -> List[int]:
        return _uniform_max_independent(eligible, pts, struct, an_facets[k].normal, d_nn)

    an_max = _symmetric_selection(an_facets, an_cands, ops, choose_anion) if len(an_facets) > 1 else None
    an_symmetric = an_max is not None
    if an_max is None:
        an_max = {k: choose_anion(k, an_cands[k]) for k in range(len(an_facets))}
    n_vac_max = min(len(v) for v in an_max.values())

    atom_tree = cKDTree(pts)
    an_outer_sets = [set(f.outer) for f in an_facets]

    def _break_runs(syms: List[str], alive: NDArray[np.bool_]) -> Dict[int, List[int]]:
        """
        Anions to convert so no surviving outer anion of an anion-terminated
        facet is a nearest neighbour of another under-coordinated anion, on
        the facet or across its edge.  Minimum alternating set per facet
        (Se-Se-Se -> Se-Cl-Se), mapped by symmetry when possible.
        """
        nodes: Dict[int, List[int]] = {}
        anchors: Dict[int, Set[int]] = {}
        for k, f in enumerate(an_facets):
            anc = [a for a in f.outer if alive[a] and syms[a] == anion]
            extra: Set[int] = set()
            for a in anc:
                for b in atom_tree.query_ball_point(pts[a], NN_FACTOR * d_nn):
                    if (b != a and b not in an_outer_sets[k] and alive[b] and syms[b] == anion
                            and cn(b, alive) < CN_BULK):
                        extra.add(b)
            nodes[k] = sorted(set(anc) | extra)
            anchors[k] = set(anc)

        def cover(k: int, eligible: List[int]) -> List[int]:
            return _min_vertex_cover(eligible, pts, d_nn, anchors=anchors[k] & set(eligible))

        def clean(k: int, conv: List[int]) -> bool:
            left = [x for x in nodes[k] if x not in set(conv)]
            return not _min_vertex_cover(left, pts, d_nn, anchors=anchors[k] - set(conv))

        res = _symmetric_selection(an_facets, nodes, ops, cover) if len(an_facets) > 1 else None
        if res is None or not all(clean(k, res[k]) for k in res):
            res = {k: cover(k, nodes[k]) for k in nodes}
        return res

    def _execute(n_vac: int) -> dict:
        syms = list(base_symbols)
        alive = all_alive.copy()
        an_sel = {
            k: v if len(v) == n_vac else _fps_nonadjacent(v, pts, n_vac, d_nn)
            for k, v in an_max.items()
        }

        # Global CN validation (edges shared between facets); drop offenders.
        removed_count: Dict[int, int] = {}
        vacancies: List[int] = []
        for k in range(len(an_facets)):
            kept = []
            for c in an_sel[k]:
                trial = dict(removed_count)
                for a in nb[c]:
                    trial[a] = trial.get(a, 0) + 1
                if all(_anion_ok_after(syms, cn(a, alive) - trial[a], a, anion, recon_ligand, min_native=2)
                       for a in nb[c]):
                    removed_count = trial
                    kept.append(c)
            an_sel[k] = kept
            vacancies.extend(kept)

        for c in vacancies:
            alive[c] = False
        converted = {k: 0 for k in range(len(an_facets))}
        facet_of_vac = {c: k for k in an_sel for c in an_sel[k]}
        dq_anion = -q_cat * len(vacancies)
        for c in vacancies:
            for a in nb[c]:
                if alive[a] and syms[a] == anion and cn(a, alive) == 2:
                    syms[a] = recon_ligand
                    converted[facet_of_vac[c]] += 1
                    dq_anion += ligand_charge - q_an

        # Break runs of adjacent surviving outer anions (charge delocalisation),
        # on the facet and across its edges, with a minimal alternating set.
        n_breaks = {k: 0 for k in range(len(an_facets))}
        for k, conv in _break_runs(syms, alive).items():
            for a in conv:
                if syms[a] == anion:
                    syms[a] = recon_ligand
                    dq_anion += ligand_charge - q_an
                    n_breaks[k] += 1
        q = q0 + dq_anion

        # ---- 2. strip one-bonded ligands from cation-terminated facets -----
        stripped = {k: 0 for k in range(len(cat_facets))}
        cat_outer = {i: k for k, f in enumerate(cat_facets) for i in f.outer}
        for li, s in enumerate(syms):
            if s != recon_ligand or not alive[li]:
                continue
            # Any ligand added on top of a cation-terminated facet (mu1 on-top,
            # mu2 bridge, mu3 hollow): all hosts on that facet's outer layer and
            # sitting above it.  Lattice-site ligands in the layer below stay.
            hosts = [j for j in nb[li] if alive[j] and syms[j] == cation]
            if not hosts or any(h not in cat_outer for h in hosts):
                continue
            k = cat_outer[hosts[0]]
            f = cat_facets[k]
            if any(cat_outer[h] != k for h in hosts) or float(pts[li] @ f.normal) < f.top + LAYER_TOL:
                continue
            alive[li] = False
            stripped[k] += 1
            q -= ligand_charge

        # ---- 3. outer-cation vacancies on cation-terminated facets ---------
        n_remove = q // q_cat if q > 0 else 0
        base = n_remove // len(cat_facets)

        # Surviving outer anions of the anion-terminated facets: a cation
        # removal must not leave a fresh under-coordinated anion next to them
        # (that would re-create a run and force another anion -> ligand swap).
        anchor_pts = [
            pts[a] for f in an_facets for a in f.outer if alive[a] and syms[a] == anion
        ]
        anchor_tree = cKDTree(np.asarray(anchor_pts)) if anchor_pts else None

        def joins_run(c: int) -> bool:
            if anchor_tree is None:
                return False
            for a in nb[c]:
                if alive[a] and syms[a] == anion and cn(a, alive) == CN_BULK:
                    if anchor_tree.query_ball_point(pts[a], NN_FACTOR * d_nn):
                        return True
            return False

        def cat_candidates(k: int, allow_ligand_nb: bool) -> List[int]:
            own = facets.index(cat_facets[k])
            out = []
            for c in cat_facets[k].outer:
                if not alive[c] or touches_other_facet(c, own, alive) or joins_run(c):
                    continue
                live_nb = [a for a in nb[c] if alive[a]]
                if not allow_ligand_nb and any(syms[a] == recon_ligand for a in live_nb):
                    continue
                if all(_anion_ok_after(syms, cn(a, alive) - 1, a, anion, recon_ligand, min_native=3)
                       for a in live_nb):
                    out.append(c)
            return out

        tier1 = {k: cat_candidates(k, False) for k in range(len(cat_facets))}
        tier2 = {k: cat_candidates(k, True) for k in range(len(cat_facets))}

        def choose_cation(k: int, eligible: List[int]) -> List[int]:
            first = [c for c in eligible if c in set(tier1[k])]
            if first:
                # Spread inside one sqrt(3) colour class: lattice-aligned and densest.
                for cls in _colour_classes(first, pts, struct, cat_facets[k].normal):
                    if len(cls) >= base:
                        return _fps_nonadjacent(cls, pts, base, d_nn)
            picked = _fps_nonadjacent(first, pts, base, d_nn)
            if len(picked) < base:
                picked += _fps_nonadjacent(eligible, pts, base - len(picked), d_nn, existing=picked)
            return picked

        cat_sel: Dict[int, List[int]] = {k: [] for k in range(len(cat_facets))}
        cat_symmetric = base == 0
        if base > 0:
            sym = _symmetric_selection(cat_facets, tier2, ops, choose_cation) if len(cat_facets) > 1 else None
            if sym is not None and all(len(v) == base for v in sym.values()):
                cat_sel, cat_symmetric = sym, True
            else:
                cat_sel = {k: choose_cation(k, tier2[k]) for k in range(len(cat_facets))}

        def add_one(k: int) -> bool:
            for pool in (tier1[k], tier2[k]):
                pick = _fps_nonadjacent(pool, pts, 1, d_nn, existing=cat_sel[k])
                if pick:
                    cat_sel[k].extend(pick)
                    return True
            return False

        # Remainder removals (and any shortfall) go to the facets with fewest so far.
        missing = n_remove - sum(len(v) for v in cat_sel.values())
        while missing > 0:
            order = sorted(range(len(cat_facets)), key=lambda k: (len(cat_sel[k]), k))
            if not any(add_one(k) for k in order):
                break
            missing -= 1

        removed_cations = [c for v in cat_sel.values() for c in v]
        for c in removed_cations:
            alive[c] = False
        q -= q_cat * len(removed_cations)

        # Cation removals can leave new under-coordinated anions next to an
        # anion-facet edge: break those runs too (compensated by ligands below).
        for k, conv in _break_runs(syms, alive).items():
            for a in conv:
                if syms[a] == anion:
                    syms[a] = recon_ligand
                    dq_anion += ligand_charge - q_an
                    q += ligand_charge - q_an
                    n_breaks[k] += 1

        # ---- 4. remainder: add ligands on cation-facet sites ---------------
        keep = np.where(alive)[0]
        remap = {int(old): new for new, old in enumerate(keep)}
        new_symbols = [syms[i] for i in keep]
        new_pts = pts[keep].copy()
        added = 0
        n_add = q // (-ligand_charge) if q > 0 else 0
        mu_added = {3: 0, 2: 0, 1: 0}
        facet_count: Dict[int, int] = {k: 0 for k in range(len(cat_facets))}
        if n_add > 0:
            free_hosts = {
                remap[i]: f.normal
                for f in cat_facets for i in f.outer
                if alive[i] and cn(i, alive) < CN_BULK
            }
            # Bridging sites only over interior cations (no facet edges).
            interior = set().union(*(_interior_outer(f.outer, pts, d_nn) for f in cat_facets))
            host_normal = {remap[i]: n for i, n in (
                (i, f.normal) for f in cat_facets for i in f.outer
                if alive[i] and cn(i, alive) < CN_BULK and i in interior
            )}
            host_facet = {
                remap[i]: k
                for k, f in enumerate(cat_facets) for i in f.outer
                if alive[i] and cn(i, alive) < CN_BULK
            }
            bond = _bulk_bond_length(struct)
            used_hosts: Set[int] = set()
            placed: List[NDArray[np.float64]] = []
            avoid = [pts[c] for c in removed_cations]

            # II-VI: mu3 hollows first, then mu2 bridges; each free cation hosts
            # one ligand.  III-V: terminal (mu1) ligands only.
            for mu in ((3, 2) if q_cat == 2 else ()):
                sites = _bridging_sites(host_normal, new_pts, mu, d_nn, bond)
                picked = _pick_bridging_sites(
                    sites, n_add - added, used_hosts, placed, avoid, bond, host_facet, facet_count
                )
                for pos, hs in picked:
                    new_symbols.append(recon_ligand)
                    new_pts = np.vstack([new_pts, pos])
                    placed.append(pos)
                    used_hosts.update(hs)
                    added += 1
                    mu_added[mu] += 1

            # Last resort: on-top (mu1) on the remaining free cations.
            if added < n_add:
                hosts = [h for h in free_hosts if h not in used_hosts]
                planes: List[Plane] = [(f.normal, f.top) for f in facets]
                missing_vecs = _strict_missing_vectors_for_hosts(
                    new_symbols, new_pts, hosts, charges, pair_cuts, struct, planes, surf_tol
                )
                slots = [(h, vecs[0]) for h, vecs in missing_vecs.items() if vecs]
                picked_slots: List[Tuple[int, np.ndarray]] = []
                while slots and len(picked_slots) < n_add - added:
                    ref = avoid + placed + [new_pts[h] for h, _ in picked_slots]
                    least = min(facet_count[host_facet[h]] for h, _ in slots)
                    pool = [sl for sl in slots if facet_count[host_facet[sl[0]]] == least]
                    if ref:
                        ref_arr = np.asarray(ref, float)
                        best = max(pool, key=lambda s: (
                            round(float(np.min(np.linalg.norm(ref_arr - new_pts[s[0]], axis=1))), 6), -s[0]))
                    else:
                        best = pool[0]
                    facet_count[host_facet[best[0]]] += 1
                    picked_slots.append(best)
                    slots = [s for s in slots if s[0] != best[0]]
                for pos in _ligand_add_positions_for_slots(new_symbols, new_pts, picked_slots, recon_ligand, pair_cuts):
                    new_symbols.append(recon_ligand)
                    new_pts = np.vstack([new_pts, np.asarray(pos, float)])
                    added += 1
                    mu_added[1] += 1

        return {
            "symbols": new_symbols,
            "pts": new_pts,
            "an_sel": an_sel,
            "converted": converted,
            "breaks": n_breaks,
            "dq_anion": dq_anion,
            "stripped": stripped,
            "tier2": tier2,
            "cat_sel": cat_sel,
            "cat_symmetric": cat_symmetric,
            "n_remove": n_remove,
            "removed": len(removed_cations),
            "n_add": n_add,
            "added": added,
            "mu_added": mu_added,
            "added_by_facet": facet_count,
            "q_final": _total_q(new_symbols, charges),
        }

    # Most vacancies per anion facet first; back off one per facet at a time
    # until the cation-terminated facets can compensate the charge exactly.
    # If no count balances, keep the attempt with the smallest residual charge
    # (most vacancies on ties).
    res = None
    for n_vac in range(n_vac_max, -1, -1):
        trial = _execute(n_vac)
        if res is None or abs(trial["q_final"]) < abs(res["q_final"]):
            res = trial
        if trial["q_final"] == 0:
            break
    n_vac_used = min(len(v) for v in res["an_sel"].values())

    print("\n=== {111} RECONSTRUCTION: anion-terminated facets ===")
    print("        hkl  candidates  vacancies  anion->ligand  chain breaks")
    for k, f in enumerate(an_facets):
        print(f"  {_hkl_str(f.hkl):>11s}  {len(an_cands[k]):10d}  {len(res['an_sel'][k]):9d}"
              f"  {res['converted'][k]:13d}  {res['breaks'][k]:12d}")
    print(f"[recon] pattern mapped by symmetry: {'yes' if an_symmetric else 'no'}"
          f" | ΔQ = {res['dq_anion']:+d}")
    if n_vac_used < n_vac_max:
        print(f"[recon] vacancies per facet reduced {n_vac_max} → {n_vac_used} so the "
              f"cation-terminated facets can compensate the charge.")

    print("\n=== {111} RECONSTRUCTION: cation-terminated facets ===")
    print("        hkl  stripped  candidates  removed  added")
    for k, f in enumerate(cat_facets):
        print(f"  {_hkl_str(f.hkl):>11s}  {res['stripped'][k]:8d}  {len(res['tier2'][k]):10d}"
              f"  {len(res['cat_sel'][k]):7d}  {res['added_by_facet'][k]:5d}")
    print(f"[recon] pattern mapped by symmetry: {'yes' if res['cat_symmetric'] else 'no'}"
          f" | {cation} removed = {res['removed']} (requested {res['n_remove']})"
          f" | {recon_ligand} added = {res['added']}"
          f" (mu3 {res['mu_added'][3]}, mu2 {res['mu_added'][2]}, mu1 {res['mu_added'][1]})")
    if res["added"] < res["n_add"]:
        print(f"[recon] warning: only {res['added']} of {res['n_add']} compensating {recon_ligand} sites found.")

    out_symbols, out_pts = res["symbols"], res["pts"]
    q_final = res["q_final"]
    rebalance: Optional[dict] = None
    if q_final != 0:
        # The {111} facets cannot absorb the whole charge: hand the leftover to
        # the regular charge balance (rebalancing mode, no structural prepass),
        # which can also use {100} facets and edges.
        from .passivation_iterative import charge_balance_iterative

        print(f"[recon] residual Q = {q_final:+d}; running charge balance on the leftover.")
        _cb_facets, cb_planes = _native_facets_and_planes(
            out_symbols, out_pts, struct, charges, facet_seeds, recon_ligand, surf_tol
        )
        before = {s: out_symbols.count(s) for s in (cation, anion, recon_ligand)}
        out_symbols, out_pts = charge_balance_iterative(
            list(out_symbols), np.asarray(out_pts, float), charges, recon_ligand,
            verbose, cb_planes, surf_tol, cif_path,
            positive_q_strategy="add",
            pair_cuts_override=pair_cuts,
            prepass_mode="none",
        )
        after = {s: out_symbols.count(s) for s in (cation, anion, recon_ligand)}
        rebalance = {
            "residual_charge": int(q_final),
            "ligands_added": int(after[recon_ligand] - before[recon_ligand]),
            "cations_removed": int(before[cation] - after[cation]),
            "anions_removed": int(before[anion] - after[anion]),
        }
        q_final = _total_q(out_symbols, charges)
        print(
            f"[recon] charge balance: {recon_ligand} {rebalance['ligands_added']:+d}, "
            f"{cation} {-rebalance['cations_removed']:+d}, {anion} {-rebalance['anions_removed']:+d}"
        )

    print(f"[recon] Done. Q_total = {q_final:+d}")
    if q_final != 0:
        print(f"[recon] warning: reconstruction left a net charge of {q_final:+d}.")
    print("=" * 60)

    if ledger is not None:
        ledger.update({
            "status": "applied",
            "cation": cation,
            "anion": anion,
            "anion_facets": [
                {"hkl": list(f.hkl), "vacancies": len(res["an_sel"][k]),
                 "anions_to_ligand": res["converted"][k], "chain_breaks": res["breaks"][k]}
                for k, f in enumerate(an_facets)
            ],
            "cation_facets": [
                {"hkl": list(f.hkl), "ligands_stripped": res["stripped"][k],
                 "cations_removed": len(res["cat_sel"][k]),
                 "ligands_added": res["added_by_facet"][k]}
                for k, f in enumerate(cat_facets)
            ],
            "anion_side_charge_delta": int(res["dq_anion"]),
            "ligands_stripped": int(sum(res["stripped"].values())),
            "cations_removed": int(res["removed"]),
            "ligands_added": int(res["added"]),
            "ligands_added_by_mu": {f"mu{k}": int(v) for k, v in res["mu_added"].items()},
            "symmetric": bool(an_symmetric and res["cat_symmetric"]),
            "total_charge_before": int(q0),
            "total_charge_after": int(q_final),
        })
        if rebalance is not None:
            ledger["charge_balance_fallback"] = rebalance
    return out_symbols, out_pts
