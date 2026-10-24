# src/builder/polyhedron.py
"""
Facet geometry of a Wulff-cut dot and the site role of every atom.

Two polyhedra share the facet normals of the construction:

- construction: the final Wulff planes n.x <= d (terminations resolved,
  symmetry expanded, auto-shifted off atom layers);
- atom-fitted: each plane moved to the outermost core atom along its normal,
  d = max over core atoms of n.r.

Both are clipped with scipy's HalfspaceIntersection, so a plane that only
touches a vertex or misses the solid has no face. Each face carries a family
label: the Miller family ("100", "110"), with A/B for a polar family whose
+n and -n faces are not symmetry-equivalent (zinc-blende {111}: "111A" when
the exposed layer is cation, "111B" when anion).

Site roles come from the clipped atom-fitted polyhedron: an atom within
`tol` of exactly one face lies on that facet, of two adjacent faces on their
edge, of three or more (or two meeting only at a corner) at a vertex.
Ligand atoms take the role of the core atom they bind to.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection, cKDTree

from .facets import unit_normal
from .nc_types import Facet, Plane

LAYER_TOL = 0.3          # Å; outermost atom layer of the cut (last-resort A/B rule)


# ---------------- facet family labels ----------------
def _point_group_rotations(struct) -> List[np.ndarray]:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    ops = SpacegroupAnalyzer(struct, symprec=1e-3).get_point_group_operations(cartesian=True)
    return [np.asarray(op.rotation_matrix, float) for op in ops]


def _hkl_orbit(struct, hkl: Tuple[int, int, int], rotations: Sequence[np.ndarray]) -> set:
    rec = struct.lattice.reciprocal_lattice
    recT = rec.matrix.T
    g0 = rec.get_cartesian_coords(hkl)
    out = set()
    for R in rotations:
        c = np.linalg.solve(recT, R @ g0)
        t = tuple(int(round(x)) for x in c)
        g = math.gcd(math.gcd(abs(t[0]), abs(t[1])), abs(t[2])) or 1
        out.add((t[0] // g, t[1] // g, t[2] // g))
    return out


def _family_digits(orbit: set) -> str:
    # The member with the most non-negative indices, then the largest: (100)
    # for hexagonal {100} rather than (-110), (111) for cubic {111}.
    rep = max(orbit, key=lambda t: (sum(x >= 0 for x in t), t))
    return "".join(str(abs(x)) for x in rep)


def _top_layer_charge(syms, pts, n, charges) -> float:
    proj = pts @ n
    top = float(proj.max())
    return float(sum(charges.get(syms[i], 0) for i in np.flatnonzero(proj >= top - LAYER_TOL)))


def facet_labels(
    struct,
    facets: Sequence[Facet],
    charges: Dict[str, int],
    syms: Optional[Sequence[str]] = None,
    pts: Optional[np.ndarray] = None,
) -> List[str]:
    """
    Family label per oriented facet: the Miller family of its point-group
    orbit, plus A (cation exposed) or B (anion exposed) for a polar family,
    i.e. when n and -n lie in different orbits (zinc-blende {111}, {211},
    {311}; wurtzite (001), {101}).

    The exposed species comes from the recipe's resolved termination when
    every member of the orbit has the same cation_rich/anion_rich one, else
    from the bulk bond rule the builder resolves terminations with (the
    species losing the fewest bonds across the plane), else from the
    outermost layer of the cut atoms `syms`/`pts`. The two orbits of a
    polar pair always get different letters.
    """
    from .main import _exposed_charge_for_wulff_hkl

    rotations = _point_group_rotations(struct)
    orbits: Dict[Tuple[int, int, int], set] = {}
    keys = []
    for f in facets:
        hkl = (int(f.h), int(f.k), int(f.l))
        key = next((k for k, orb in orbits.items() if hkl in orb), None)
        if key is None:
            orbits[hkl] = _hkl_orbit(struct, hkl, rotations)
            key = hkl
        keys.append(key)

    def neg(k):
        return tuple(-x for x in k)

    def partner(k):
        return next((j for j, orb in orbits.items() if neg(k) in orb), None)

    have_atoms = syms is not None and pts is not None and len(pts) > 0
    P = np.asarray(pts, float) if have_atoms else None
    score: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}
    for key, orb in orbits.items():
        if neg(key) in orb:
            continue
        terms = {getattr(f, "termination", None) for f, k in zip(facets, keys) if k == key}
        q_term = 1.0 if terms == {"cation_rich"} else -1.0 if terms == {"anion_rich"} else 0.0
        q_bond = float(_exposed_charge_for_wulff_hkl(struct, key, charges))
        q_cut = 0.0
        if have_atoms:
            q_cut = sum(_top_layer_charge(syms, P, unit_normal(struct, (f.h, f.k, f.l)), charges)
                        for f, k in zip(facets, keys) if k == key)
        score[key] = (q_term, q_bond, q_cut)

    suffix = {k: "" for k in orbits}
    for key, sc in score.items():
        q = next((x for x in sc if x != 0.0), 0.0)
        suffix[key] = "A" if q > 0 else "B" if q < 0 else ""
    # A polar pair present together: compare the two, rule by rule.
    for key in score:
        other = partner(key)
        if other is None or suffix[key] != suffix[other] and suffix[key]:
            continue
        for a, b in zip(score[key], score[other]):
            if a != b:
                suffix[key], suffix[other] = ("A", "B") if a > b else ("B", "A")
                break
    return [_family_digits(orbits[k]) + suffix[k] for k in keys]


# ---------------- clipped polyhedron ----------------
@dataclass
class Polyhedron:
    normals: np.ndarray                    # (P, 3) unit normals of all planes
    d: np.ndarray                          # (P,)
    labels: List[str]
    verts: np.ndarray                      # (V, 3)
    faces: Dict[int, List[int]] = field(default_factory=dict)   # plane -> ordered vertex ids
    areas: Dict[int, float] = field(default_factory=dict)       # plane -> Å^2
    edges: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (plane a, plane b, v0, v1)
    volume: float = 0.0

    def vertex_faces(self, v: int) -> List[int]:
        return [p for p, ids in self.faces.items() if v in ids]

    def vertex_type(self, v: int) -> str:
        return "+".join(sorted(self.labels[p] for p in self.vertex_faces(v)))

    def edge_type(self, a: int, b: int) -> str:
        return "|".join(sorted((self.labels[a], self.labels[b])))


def _interior_point(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """Chebyshev centre of A x <= b and its radius."""
    from scipy.optimize import linprog
    norm = np.linalg.norm(A, axis=1)
    res = linprog(c=[0, 0, 0, -1], A_ub=np.hstack([A, norm[:, None]]), b_ub=b,
                  bounds=[(None, None)] * 3 + [(0, None)], method="highs")
    if not res.success:
        raise ValueError("facet planes enclose no volume")
    return np.asarray(res.x[:3], float), float(res.x[3])


def clip(planes: Sequence[Plane], labels: Sequence[str], tol: float = 1e-6) -> Polyhedron:
    """Half-space intersection n.x <= d with its faces, edges and vertices."""
    N = np.array([np.asarray(n, float) / np.linalg.norm(n) for n, _d in planes])
    D = np.array([float(d) / np.linalg.norm(n) for n, d in planes])
    centre, radius = _interior_point(N, D)
    if radius < 1e-3:
        raise ValueError(f"facet planes enclose no volume (inradius {radius:.2e} Å)")
    raw = HalfspaceIntersection(np.hstack([N, -D[:, None]]), centre).intersections
    # Several planes through one corner give copies of it a few ulp apart.
    scale = max(1.0, float(np.abs(D).max()))
    merge = 1e-6 * scale
    keep: List[int] = []
    tree = cKDTree(raw)
    taken = np.zeros(len(raw), bool)
    for i in range(len(raw)):
        if taken[i]:
            continue
        group = tree.query_ball_point(raw[i], merge)
        taken[group] = True
        keep.append(i)
    verts = raw[keep]
    poly = Polyhedron(N, D, list(labels), verts)

    on_tol = 1e-5 * scale
    for p, (n, d) in enumerate(zip(N, D)):
        idx = np.flatnonzero(np.abs(verts @ n - d) < on_tol)
        if len(idx) < 3:
            continue
        pts = verts[idx]
        c = pts.mean(0)
        u = pts[0] - c
        u /= np.linalg.norm(u)
        w = np.cross(n, u)
        order = np.argsort(np.arctan2((pts - c) @ w, (pts - c) @ u))
        ring = idx[order]
        q = verts[ring]
        area = 0.5 * float(np.linalg.norm(np.cross(q, np.roll(q, -1, axis=0)).sum(0)))
        if area > tol * scale * scale:
            poly.faces[p] = [int(v) for v in ring]
            poly.areas[p] = area

    for a, b in itertools.combinations(sorted(poly.faces), 2):
        common = sorted(set(poly.faces[a]) & set(poly.faces[b]))
        if len(common) < 2:
            continue
        v0, v1 = max(itertools.combinations(common, 2),
                     key=lambda t: float(np.linalg.norm(verts[t[0]] - verts[t[1]])))
        if np.linalg.norm(verts[v0] - verts[v1]) > tol * scale:
            poly.edges.append((a, b, v0, v1))
    poly.volume = float(ConvexHull(verts).volume)
    return poly


def fitted_planes(planes: Sequence[Plane], core_pts: np.ndarray) -> List[Plane]:
    """The planes moved to the outermost core atom along each normal."""
    P = np.asarray(core_pts, float)
    out = []
    for n, _d in planes:
        n = np.asarray(n, float) / np.linalg.norm(n)
        out.append((n, float((P @ n).max())))
    return out


def summary(poly: Polyhedron, hkls: Optional[Sequence[Tuple[int, int, int]]] = None) -> dict:
    """
    JSON-ready geometry: areas by family label (Å^2), edges by facet-pair type
    (count, total length in Å), vertex count and types, volume (Å^3), and the
    plane of every oriented facet (d in Å from the construction origin).
    """
    areas: Dict[str, float] = defaultdict(float)
    for p, a in poly.areas.items():
        areas[poly.labels[p]] += a
    edges: Dict[str, List[float]] = defaultdict(lambda: [0, 0.0])
    for a, b, v0, v1 in poly.edges:
        rec = edges[poly.edge_type(a, b)]
        rec[0] += 1
        rec[1] += float(np.linalg.norm(poly.verts[v0] - poly.verts[v1]))
    used = sorted({v for ids in poly.faces.values() for v in ids})
    vtypes = Counter(poly.vertex_type(v) for v in used)
    planes = []
    for p, (n, d) in enumerate(zip(poly.normals, poly.d)):
        rec = {"label": poly.labels[p], "d_A": round(float(d), 4),
               "normal": [round(float(x), 6) for x in n],
               "area_A2": round(poly.areas.get(p, 0.0), 3)}
        if hkls is not None:
            rec = {"hkl": [int(x) for x in hkls[p]], **rec}
        planes.append(rec)
    return {
        "areas_A2": {k: round(v, 3) for k, v in sorted(areas.items())},
        "total_area_A2": round(float(sum(poly.areas.values())), 3),
        "edges": {k: {"n": int(v[0]), "length_A": round(v[1], 3)} for k, v in sorted(edges.items())},
        "n_faces": len(poly.faces),
        "n_edges": len(poly.edges),
        "n_vertices": len(used),
        "vertex_types": dict(sorted(vtypes.items())),
        "volume_A3": round(poly.volume, 3),
        "planes": planes,
    }


# ---------------- site roles ----------------
def _dist_to_face(x: np.ndarray, poly: Polyhedron, p: int) -> float:
    n, d = poly.normals[p], poly.d[p]
    s = float(d - x @ n)
    ring = poly.verts[poly.faces[p]]
    q = x + s * n                                   # projection on the plane
    inside = True
    best = math.inf
    for a, b in zip(ring, np.roll(ring, -1, axis=0)):
        e = b - a
        # Ring runs counter-clockwise about n: inside is to the left of each edge.
        if float(np.dot(np.cross(e, q - a), n)) < 0:
            inside = False
        t = min(1.0, max(0.0, float(np.dot(q - a, e) / max(np.dot(e, e), 1e-12))))
        best = min(best, float(np.linalg.norm(q - (a + t * e))))
    return abs(s) if inside else math.hypot(s, best)


def core_roles(poly: Polyhedron, pts: np.ndarray, tol: float = 1.0) -> List[str]:
    """
    "facet:<label>", "edge:<label>|<label>", "vertex:<labels joined by +>"
    or "interior" for every point, from its distance to the clipped faces.
    """
    adjacent = {(a, b) for a, b, _v0, _v1 in poly.edges}
    adjacent |= {(b, a) for a, b in adjacent}
    used = sorted({v for ids in poly.faces.values() for v in ids})
    vtree = cKDTree(poly.verts[used])
    P = np.asarray(pts, float)
    # Only atoms within tol of some face plane can be anything but interior;
    # screen in bulk (widened by 1e-6) and classify the rest one by one.
    faces = list(poly.faces)
    if faces:
        depth = poly.d[faces][None, :] - P @ poly.normals[faces].T
        maybe = (depth < tol + 1e-6).any(axis=1)
    else:
        maybe = np.zeros(len(P), dtype=bool)
    out = []
    for x, check in zip(P, maybe.tolist()):
        if not check:
            out.append("interior")
            continue
        near = []
        for p in poly.faces:
            s = float(poly.d[p] - x @ poly.normals[p])
            if s < tol and _dist_to_face(x, poly, p) < tol:
                near.append(p)
        if not near:
            out.append("interior")
        elif len(near) == 1:
            out.append(f"facet:{poly.labels[near[0]]}")
        elif len(near) == 2 and (near[0], near[1]) in adjacent:
            out.append(f"edge:{poly.edge_type(near[0], near[1])}")
        else:
            # Prefer the nearest corner shared by the faces found.
            cands = [v for v in used if set(near) <= set(poly.vertex_faces(v))] or \
                    [v for v in used if len(set(near) & set(poly.vertex_faces(v))) >= 2]
            if cands:
                v = min(cands, key=lambda v: float(np.linalg.norm(poly.verts[v] - x)))
            else:
                v = used[int(vtree.query(x)[1])]
            out.append(f"vertex:{poly.vertex_type(v)}")
    return out


def _covalent_radius(sym: str) -> float:
    from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
    return float(CovalentRadius.radius.get(sym, 0.8))


def ligand_hosts(
    syms: Sequence[str],
    pts: np.ndarray,
    core: np.ndarray,
    charges: Dict[str, int],
    bond_scale: float = 1.15,
) -> np.ndarray:
    """
    Host core atom of every non-core atom (-1 for core atoms). Non-core atoms
    are grouped into molecules by covalent radii; a molecule binds through
    its atom nearest the core, and its host is the nearest core cation (the
    nearest core anion when that binding atom is itself a cation, as in a
    Z-type MXn complex).
    """
    P = np.asarray(pts, float)
    hosts = np.full(len(syms), -1, dtype=int)
    lig = np.flatnonzero(~core)
    core_idx = np.flatnonzero(core)
    if lig.size == 0 or core_idx.size == 0:
        return hosts
    rad = {s: _covalent_radius(s) for s in {syms[i] for i in lig}}
    # union-find over the non-core atoms
    parent = {int(i): int(i) for i in lig}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = cKDTree(P[lig])
    rmax = max(rad.values())
    for a, b in tree.query_pairs(2.0 * bond_scale * rmax):
        i, j = int(lig[a]), int(lig[b])
        if np.linalg.norm(P[i] - P[j]) <= bond_scale * (rad[syms[i]] + rad[syms[j]]):
            parent[find(i)] = find(j)
    groups: Dict[int, List[int]] = defaultdict(list)
    for i in lig:
        groups[find(int(i))].append(int(i))

    cations = np.array([i for i in core_idx if charges.get(syms[i], 0) > 0], int)
    anions = np.array([i for i in core_idx if charges.get(syms[i], 0) <= 0], int)
    ctree = cKDTree(P[core_idx])
    cat_tree = cKDTree(P[cations]) if cations.size else None
    an_tree = cKDTree(P[anions]) if anions.size else None
    for members in groups.values():
        dist, _j = ctree.query(P[members])
        binder = members[int(np.argmin(np.atleast_1d(dist)))]
        use_anion = charges.get(syms[binder], 0) > 0 and an_tree is not None
        tree_h, pool = (an_tree, anions) if use_anion else (cat_tree, cations)
        if tree_h is None:
            tree_h, pool = ctree, core_idx
        host = int(pool[int(tree_h.query(P[binder])[1])])
        hosts[members] = host
    return hosts


def site_roles(
    poly: Polyhedron,
    syms: Sequence[str],
    pts: np.ndarray,
    core: np.ndarray,
    charges: Dict[str, int],
    tol: float = 1.0,
) -> Tuple[List[str], np.ndarray]:
    """Role per atom (core atoms by geometry, ligands from their host) and host index (-1 for core)."""
    core = np.asarray(core, bool)
    roles = ["interior"] * len(syms)
    for i, r in zip(np.flatnonzero(core), core_roles(poly, np.asarray(pts, float)[core], tol=tol)):
        roles[int(i)] = r
    hosts = ligand_hosts(syms, pts, core, charges)
    for i in np.flatnonzero(hosts >= 0):
        roles[int(i)] = roles[int(hosts[i])]
    return roles, hosts


def role_counts(roles: Sequence[str], syms: Sequence[str], core: np.ndarray) -> Dict[str, Dict[str, int]]:
    """Atoms per role and element, core and ligand separately."""
    out: Dict[str, Dict[str, int]] = {"core": {}, "ligand": {}}
    for part, mask in (("core", core), ("ligand", ~np.asarray(core, bool))):
        c = Counter(f"{r} {s}" for r, s, m in zip(roles, syms, mask) if m)
        out[part] = dict(sorted(c.items()))
    return out


def facet_geometry(
    construction_planes: Sequence[Plane],
    labels: Sequence[str],
    hkls: Sequence[Tuple[int, int, int]],
    syms: Sequence[str],
    pts: np.ndarray,
    core: np.ndarray,
    charges: Dict[str, int],
    *,
    role_tol: float = 1.0,
) -> Tuple[dict, Optional[dict]]:
    """
    The `facet_geometry` and `site_roles` manifest entries. Atoms in `pts`
    are in the construction frame (origin at the construction centre).
    """
    geo: dict = {"units": {"length": "Å", "area": "Å^2", "volume": "Å^3"}}
    try:
        geo["construction"] = summary(clip(construction_planes, labels), hkls)
    except Exception as exc:
        geo["construction"] = {"error": f"{type(exc).__name__}: {exc}"}
    core = np.asarray(core, bool)
    roles = None
    if not core.any():
        geo["atom_fitted"] = {"error": "no core atoms"}
        return geo, roles
    try:
        poly = clip(fitted_planes(construction_planes, np.asarray(pts, float)[core]), labels)
        geo["atom_fitted"] = summary(poly, hkls)
    except Exception as exc:
        geo["atom_fitted"] = {"error": f"{type(exc).__name__}: {exc}"}
        return geo, roles
    role_list, hosts = site_roles(poly, syms, pts, core, charges, tol=role_tol)
    roles = {
        "tolerance_A": float(role_tol),
        "roles": role_list,
        "host": [int(h) for h in hosts],
        "is_core": [bool(c) for c in core],
        "counts": role_counts(role_list, syms, core),
    }
    return geo, roles
