"""Zinc-blende occupation growth (move Z).

Parent identity is a Cd–Se occupation on the CIF tet lattice, not the
relaxed XYZ.  Children are made by vacating precursor Cd and filling
vacant cation+anion pairs.  After g-xTB the new core is kept only if it
still snaps onto zb (no Cd2Se2 diamonds).

Cl is placed from the 2p graph law around the *fixed* zb core; Cl does
not occupy CIF virtual sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..nc_types import NucleationSpec
from .lattice import (
    _atom_positions,
    _build_lattice_model,
    _cation_vacancies_on_anions,
    _make_core_graph,
    _partner_slots,
    _position_key,
    _position_occupied,
    _seed_state,
    _vacancies,
)
from .soft_rules import describe_structure
from .types import AtomRecord, FloatArray, _LatticeModel, _State

Edge = Tuple[int, int]
EdgeList = Tuple[Edge, ...]


@dataclass
class ZbOccupation:
    """One connected Cd–Se fragment sitting on zb sites."""

    k: int
    p: int
    symbols: Tuple[str, ...]
    coordinates: FloatArray
    core_edges: EdgeList
    parent_id: str = ""
    shed: int = 0
    p_m: int = 0
    notes: str = ""


@dataclass
class ZbGrowStats:
    """Counters for one growth step (log these, not every attempt)."""

    parents: int = 0
    snapped: int = 0
    snap_fail: int = 0
    n4_reject: int = 0
    children: int = 0
    attach_attempts: int = 0
    clash_skip: int = 0
    opt_keep: int = 0
    opt_reject_embed: int = 0
    opt_fail: int = 0

    def as_log(self) -> str:
        return (
            f"Z snap={self.snapped}/{self.parents} "
            f"snap_fail={self.snap_fail} n4_reject={self.n4_reject} "
            f"children={self.children} attach_try={self.attach_attempts} "
            f"clash_skip={self.clash_skip} "
            f"opt_keep={self.opt_keep} opt_not_zb={self.opt_reject_embed} "
            f"opt_fail={self.opt_fail}"
        )


def lattice_model(spec: NucleationSpec) -> _LatticeModel:
    return _build_lattice_model(spec)


def _species(spec: NucleationSpec) -> Tuple[str, str, str]:
    return spec.core.cation, spec.core.anion, spec.precursor.ligand


def seed_occupation(spec: NucleationSpec, model: _LatticeModel) -> ZbOccupation:
    """k=1 p=0 Cd–Se on zb."""

    state = _seed_state(model)
    occ = occupation_from_state(state, spec, k=1, p=0, parent_id="zb_seed")
    if occ is None:
        raise RuntimeError("zb seed is not a connected Cd–Se pair")
    return occ


def occupation_from_state(
    state: _State,
    spec: NucleationSpec,
    *,
    k: int,
    p: int,
    parent_id: str = "",
    shed: int = 0,
    p_m: int = 0,
    notes: str = "",
) -> Optional[ZbOccupation]:
    cation, anion, _ = _species(spec)
    se = [a for a in state.atoms if a.symbol == anion]
    cd = [a for a in state.atoms if a.symbol == cation]
    if len(se) != int(k) or len(cd) != int(k) + int(p):
        return None
    mapping: Dict[int, int] = {}
    symbols: List[str] = []
    coords: List[Tuple[float, float, float]] = []
    for i, atom in enumerate(se):
        mapping[int(atom.atom_id)] = i
        symbols.append(anion)
        coords.append(tuple(float(x) for x in atom.coordinates))
    for i, atom in enumerate(cd):
        mapping[int(atom.atom_id)] = int(k) + i
        symbols.append(cation)
        coords.append(tuple(float(x) for x in atom.coordinates))
    edges: List[Edge] = []
    for u, v in state.graph.edges:
        if u not in mapping or v not in mapping:
            continue
        a, b = mapping[int(u)], mapping[int(v)]
        if {symbols[a], symbols[b]} != {cation, anion}:
            continue
        edges.append((min(a, b), max(a, b)))
    if not edges:
        return None
    pts = np.asarray(coords, dtype=float)
    return ZbOccupation(
        k=int(k),
        p=int(p),
        symbols=tuple(symbols),
        coordinates=pts,
        core_edges=tuple(sorted(set(edges))),
        parent_id=parent_id,
        shed=int(shed),
        p_m=int(p_m),
        notes=notes,
    )


def state_from_occupation(
    occ: ZbOccupation, spec: NucleationSpec, model: _LatticeModel
) -> _State:
    cation, anion, _ = _species(spec)
    atoms = []
    for i, (sym, xyz) in enumerate(zip(occ.symbols, occ.coordinates)):
        if sym == anion:
            role = "core_anion"
        elif i < 2 * occ.k:
            role = "core_cation"
        else:
            role = "precursor_center"
        atoms.append(
            AtomRecord(
                i,
                sym,
                (float(xyz[0]), float(xyz[1]), float(xyz[2])),
                role,
            )
        )
    return _make_core_graph(atoms, model, spec)


def _rotation_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 rotation mapping unit vector a onto b."""

    ua = a / (float(np.linalg.norm(a)) + 1e-15)
    ub = b / (float(np.linalg.norm(b)) + 1e-15)
    v = np.cross(ua, ub)
    s = float(np.linalg.norm(v))
    c = float(np.dot(ua, ub))
    if s < 1e-10:
        if c >= 0.0:
            return np.eye(3)
        # 180° about any axis orthogonal to ua
        axis = np.array([1.0, 0.0, 0.0])
        if abs(ua[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - ua * float(np.dot(axis, ua))
        axis = axis / (float(np.linalg.norm(axis)) + 1e-15)
        k = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        return np.eye(3) + 2.0 * k @ k
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _twists_about(axis: np.ndarray) -> List[np.ndarray]:
    """Identity and ±120° about a tet bond (C3)."""

    u = axis / (float(np.linalg.norm(axis)) + 1e-15)
    out = [np.eye(3)]
    for ang in (2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0):
        c, s = float(np.cos(ang)), float(np.sin(ang))
        k = np.array(
            [[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]]
        )
        out.append(np.eye(3) * c + s * k + (1.0 - c) * np.outer(u, u))
    return out


def _zb_site_cloud(
    model: _LatticeModel,
    spec: NucleationSpec,
    center: np.ndarray,
    radius: float,
    *,
    max_sites: int = 400,
) -> _State:
    """BFS zb sites around ``center`` in the CIF seed frame (no recentering)."""

    cation, anion, _ = _species(spec)
    state = _seed_state(model)
    while len(state.atoms) < max_sites:
        occ = _atom_positions(state.atoms)
        seen = {
            _position_key(pt, model.site_tolerance) for pt in occ
        }
        additions: List[AtomRecord] = []
        for host_sym, target_sym, role in (
            (anion, cation, "core_cation"),
            (cation, anion, "core_anion"),
        ):
            for vac in _vacancies(
                state,
                host_symbol=host_sym,
                target_symbol=target_sym,
                model=model,
                spec=spec,
            ):
                if float(np.linalg.norm(vac.position - center)) > radius:
                    continue
                key = _position_key(vac.position, model.site_tolerance)
                if key in seen:
                    continue
                seen.add(key)
                additions.append(
                    AtomRecord(
                        len(state.atoms) + len(additions),
                        target_sym,
                        tuple(float(x) for x in vac.position),
                        role,
                    )
                )
        if not additions:
            break
        state = _make_core_graph(list(state.atoms) + additions, model, spec)
    return state


def zb_embeddable(
    symbols: Sequence[str],
    coords: np.ndarray,
    spec: NucleationSpec,
    model: _LatticeModel,
    *,
    parent_id: str = "",
    snap_tol: float = 0.85,
) -> Tuple[bool, Optional[ZbOccupation], str]:
    """True when the inorganic core is a zb fragment (no Cd2Se2)."""

    cation, anion, ligand = _species(spec)
    desc = describe_structure(symbols, coords, spec)
    if int(desc.n4) > 0:
        return False, None, f"n4={desc.n4}"
    core_idx = [i for i, s in enumerate(symbols) if s in {cation, anion}]
    if len(core_idx) < 2:
        return False, None, "too_few_core"
    core_xyz = np.asarray(coords, dtype=float)[core_idx]
    n_se = sum(1 for i in core_idx if symbols[i] == anion)
    n_cd = sum(1 for i in core_idx if symbols[i] == cation)
    k = n_se
    p = n_cd - k
    if k < 1 or p < 0:
        return False, None, "stoich"
    center = core_xyz.mean(axis=0)
    span = float(np.max(np.linalg.norm(core_xyz - center, axis=1)))
    radius = max(span, 3.0 * float(model.bond_length)) + 3.0 * float(
        model.bond_length
    )
    cloud = _zb_site_cloud(model, spec, np.zeros(3), radius)
    cat_sites = [
        np.asarray(a.coordinates, dtype=float)
        for a in cloud.atoms
        if a.symbol == cation
    ]
    an_sites = [
        np.asarray(a.coordinates, dtype=float)
        for a in cloud.atoms
        if a.symbol == anion
    ]
    pts = np.asarray(coords, dtype=float)
    se_idx = [i for i in core_idx if symbols[i] == anion]
    cd_idx = [i for i in core_idx if symbols[i] == cation]
    # Cd–Se contacts in the input (distance)
    r_bond = float(model.bond_length)
    mol_bonds: List[Tuple[int, int]] = []
    for i in se_idx:
        for j in cd_idx:
            if abs(float(np.linalg.norm(pts[i] - pts[j])) - r_bond) <= 0.55:
                mol_bonds.append((i, j))
    if not mol_bonds:
        return False, None, "no_cdse_bond"
    cloud_se_cd: List[Tuple[int, int]] = []
    for ia, se_pt in enumerate(an_sites):
        for ic, cd_pt in enumerate(cat_sites):
            if abs(float(np.linalg.norm(se_pt - cd_pt)) - r_bond) <= 0.35:
                cloud_se_cd.append((ia, ic))
    if not cloud_se_cd:
        return False, None, "empty_cloud"
    order = se_idx + cd_idx
    best: Optional[Tuple[float, List[np.ndarray], List[str]]] = None
    # Cap the search: a few mol bonds × a few cloud bonds × 3 twists.
    for mi, (mse, mcd) in enumerate(mol_bonds[:4]):
        va = pts[mcd] - pts[mse]
        for ia, ic in cloud_se_cd[:12]:
            vb = cat_sites[ic] - an_sites[ia]
            r0 = _rotation_a_to_b(va, vb)
            for tw in _twists_about(vb):
                rmat = tw @ r0
                shifted = (pts[order] - pts[mse]) @ rmat.T + an_sites[ia]
                used_cat: set = set()
                used_an: set = set()
                assigned: List[np.ndarray] = []
                worst = 0.0
                ok_ass = True
                for loc, idx in zip(shifted, order):
                    target = an_sites if symbols[idx] == anion else cat_sites
                    used = used_an if symbols[idx] == anion else used_cat
                    best_j = -1
                    best_d = snap_tol + 1.0
                    for j, site in enumerate(target):
                        if j in used:
                            continue
                        d = float(np.linalg.norm(loc - site))
                        if d < best_d:
                            best_d = d
                            best_j = j
                    if best_j < 0 or best_d > snap_tol:
                        ok_ass = False
                        break
                    used.add(best_j)
                    assigned.append(target[best_j])
                    worst = max(worst, best_d)
                if not ok_ass:
                    continue
                out_sym = [symbols[i] for i in order]
                if best is None or worst < best[0]:
                    best = (worst, assigned, out_sym)
    if best is None:
        return False, None, "snap_d"
    _worst, assigned, out_sym = best
    atoms = [
        AtomRecord(
            i,
            out_sym[i],
            tuple(float(x) for x in assigned[i]),
            "core_anion" if out_sym[i] == anion else "core_cation",
        )
        for i in range(len(out_sym))
    ]
    state = _make_core_graph(atoms, model, spec)
    import networkx as nx

    inorg = [
        i
        for i, a in enumerate(state.atoms)
        if a.symbol in {cation, anion}
    ]
    sub = state.graph.subgraph(inorg)
    if not inorg or not nx.is_connected(sub):
        return False, None, "disconnected"
    occ = occupation_from_state(
        state, spec, k=k, p=p, parent_id=parent_id, notes="snap"
    )
    if occ is None:
        return False, None, "canon"
    return True, occ, "ok"


def snap_parent(
    symbols: Sequence[str],
    coords: np.ndarray,
    spec: NucleationSpec,
    model: _LatticeModel,
    *,
    parent_id: str,
    k: int,
    p: int,
) -> Tuple[Optional[ZbOccupation], str]:
    ok, occ, why = zb_embeddable(
        symbols, coords, spec, model, parent_id=parent_id
    )
    if not ok or occ is None:
        return None, why
    if occ.k != int(k) or occ.p != int(p):
        return None, f"stoich_k{occ.k}p{occ.p}"
    return occ, "ok"


def lattice_k1_occupation(
    spec: NucleationSpec, model: _LatticeModel, p: int
) -> Optional[ZbOccupation]:
    """Build k=1 p on zb (seed + p precursor Cd).  Fallback if map XYZ won't snap."""

    occ = seed_occupation(spec, model)
    if int(p) <= 0:
        return occ
    added = _add_precursor_cd(occ, int(p), spec, model, cap=8)
    return added[0] if added else None


def _precursor_cd_ids(occ: ZbOccupation, spec: NucleationSpec) -> List[int]:
    cation, anion, _ = _species(spec)
    centroid = occ.coordinates.mean(axis=0)
    ranked: List[Tuple[int, float, int]] = []
    for i, sym in enumerate(occ.symbols):
        if sym != cation:
            continue
        deg = sum(
            1
            for a, b in occ.core_edges
            if i in (a, b) and occ.symbols[a + b - i] == anion
        )
        far = -float(np.linalg.norm(occ.coordinates[i] - centroid))
        ranked.append((deg, far, i))
    ranked.sort()
    return [i for _d, _f, i in ranked[: occ.p]]


def shed_occupation(
    occ: ZbOccupation,
    s: int,
    spec: NucleationSpec,
    model: _LatticeModel,
) -> Optional[ZbOccupation]:
    if s <= 0:
        return occ
    if s > occ.p:
        return None
    drop = set(_precursor_cd_ids(occ, spec)[:s])
    keep = [i for i in range(len(occ.symbols)) if i not in drop]
    mapping = {old: new for new, old in enumerate(keep)}
    atoms = [
        AtomRecord(
            mapping[i],
            occ.symbols[i],
            tuple(float(x) for x in occ.coordinates[i]),
            "core_anion"
            if occ.symbols[i] == spec.core.anion
            else "core_cation",
        )
        for i in keep
    ]
    state = _make_core_graph(atoms, model, spec)
    return occupation_from_state(
        state,
        spec,
        k=occ.k,
        p=occ.p - s,
        parent_id=occ.parent_id,
        shed=s,
        notes="shed",
    )


def _monomer_pairs(
    state: _State, model: _LatticeModel, spec: NucleationSpec
) -> List[Tuple[np.ndarray, np.ndarray]]:
    cation, anion, _ = _species(spec)
    cation_sites = _vacancies(
        state,
        host_symbol=anion,
        target_symbol=cation,
        model=model,
        spec=spec,
    )
    anion_sites = _vacancies(
        state,
        host_symbol=cation,
        target_symbol=anion,
        model=model,
        spec=spec,
    )
    occupied = _atom_positions(state.atoms)
    pairs: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], Tuple[np.ndarray, np.ndarray]] = {}
    for site in cation_sites:
        for anion_position in _partner_slots(
            site.position, cation, occupied, model
        ):
            pairs[
                (
                    _position_key(site.position, model.site_tolerance),
                    _position_key(anion_position, model.site_tolerance),
                )
            ] = (site.position, anion_position)
    for site in anion_sites:
        for cation_position in _partner_slots(
            site.position, anion, occupied, model
        ):
            pairs[
                (
                    _position_key(cation_position, model.site_tolerance),
                    _position_key(site.position, model.site_tolerance),
                )
            ] = (cation_position, site.position)
    return list(pairs.values())


def _place_monomer(
    state: _State,
    cation_position: np.ndarray,
    anion_position: np.ndarray,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> Optional[_State]:
    import networkx as nx

    cation, anion, _ = _species(spec)
    atoms = list(state.atoms)
    atoms.append(
        AtomRecord(
            len(atoms),
            cation,
            tuple(float(x) for x in cation_position),
            "core_cation",
        )
    )
    atoms.append(
        AtomRecord(
            len(atoms),
            anion,
            tuple(float(x) for x in anion_position),
            "core_anion",
        )
    )
    child = _make_core_graph(atoms, model, spec)
    inorg = [
        i for i, a in enumerate(child.atoms) if a.symbol in {cation, anion}
    ]
    if not inorg or not nx.is_connected(child.graph.subgraph(inorg)):
        return None
    return child


def attach_cdse(
    occ: ZbOccupation,
    spec: NucleationSpec,
    model: _LatticeModel,
    *,
    cap: int,
) -> List[ZbOccupation]:
    state = state_from_occupation(occ, spec, model)
    out: List[ZbOccupation] = []
    seen: set = set()
    for cd_pos, se_pos in _monomer_pairs(state, model, spec):
        child = _place_monomer(state, cd_pos, se_pos, model, spec)
        if child is None:
            continue
        new = occupation_from_state(
            child,
            spec,
            k=occ.k + 1,
            p=occ.p,
            parent_id=occ.parent_id,
            shed=occ.shed,
            notes="attach",
        )
        if new is None:
            continue
        key = tuple(sorted(new.core_edges))
        if key in seen:
            continue
        seen.add(key)
        out.append(new)
        if cap > 0 and len(out) >= cap:
            break
    return out


def _add_precursor_cd(
    occ: ZbOccupation,
    p_m: int,
    spec: NucleationSpec,
    model: _LatticeModel,
    *,
    cap: int,
) -> List[ZbOccupation]:
    if p_m <= 0:
        return [occ]
    cation, anion, _ = _species(spec)
    import networkx as nx

    frontier = [state_from_occupation(occ, spec, model)]
    for _ in range(int(p_m)):
        nxt: List[_State] = []
        seen: set = set()
        for src in frontier:
            for vac in _cation_vacancies_on_anions(src, model, spec):
                atoms = list(src.atoms)
                atoms.append(
                    AtomRecord(
                        len(atoms),
                        cation,
                        tuple(float(x) for x in vac.position),
                        "precursor_center",
                    )
                )
                child = _make_core_graph(atoms, model, spec)
                inorg = [
                    i
                    for i, a in enumerate(child.atoms)
                    if a.symbol in {cation, anion}
                ]
                if not inorg or not nx.is_connected(child.graph.subgraph(inorg)):
                    continue
                key = tuple(
                    sorted(
                        _position_key(
                            np.asarray(a.coordinates), model.site_tolerance
                        )
                        for a in child.atoms
                    )
                )
                if key in seen:
                    continue
                seen.add(key)
                nxt.append(child)
        if not nxt:
            return []
        frontier = nxt[: max(1, cap)]
    out: List[ZbOccupation] = []
    seen_e: set = set()
    for st in frontier:
        new = occupation_from_state(
            st,
            spec,
            k=occ.k,
            p=occ.p + int(p_m),
            parent_id=occ.parent_id,
            shed=occ.shed,
            p_m=int(p_m),
            notes="pm",
        )
        if new is None:
            continue
        key = tuple(sorted(new.core_edges))
        if key in seen_e:
            continue
        seen_e.add(key)
        out.append(new)
        if cap > 0 and len(out) >= cap:
            break
    return out


def grow_zb_children(
    occ: ZbOccupation,
    *,
    s: int,
    p_m: int,
    spec: NucleationSpec,
    model: _LatticeModel,
    cap: int,
    stats: Optional[ZbGrowStats] = None,
) -> List[ZbOccupation]:
    """Shed s extra Cd, attach one CdSe, add p_m precursor Cd."""

    after = shed_occupation(occ, s, spec, model)
    if after is None:
        return []
    attached = attach_cdse(after, spec, model, cap=max(cap, 1))
    if stats is not None:
        stats.attach_attempts += 1
    out: List[ZbOccupation] = []
    for core in attached:
        core.shed = s
        packed = _add_precursor_cd(core, p_m, spec, model, cap=cap)
        for child in packed:
            child.shed = s
            child.p_m = p_m
            child.parent_id = occ.parent_id
            out.append(child)
            if cap > 0 and len(out) >= cap:
                return out
    return out


def _cdcl_bond(pack: Any) -> float:
    if pack is None:
        return 2.50
    try:
        from .molecular_growth import _cdcl_bond_A

        return float(_cdcl_bond_A(pack))
    except Exception:
        return 2.50


def _perp_away_from(axis: np.ndarray, away: np.ndarray) -> np.ndarray:
    """Unit vector in the plane of ``away``, perpendicular to ``axis``."""

    a = axis / (float(np.linalg.norm(axis)) + 1e-15)
    v = away - a * float(np.dot(away, a))
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(tmp, a))) > 0.9:
            tmp = np.array([1.0, 0.0, 0.0])
        v = np.cross(a, tmp)
        n = float(np.linalg.norm(v))
    return v / (n + 1e-15)


def _too_close(
    pos: np.ndarray,
    xyz: np.ndarray,
    *,
    floor: float,
    ignore: Sequence[int] = (),
) -> bool:
    skip = set(ignore)
    for i, pt in enumerate(xyz):
        if i in skip:
            continue
        if float(np.linalg.norm(pos - pt)) < floor:
            return True
    return False


def place_cl_2p(
    occ: ZbOccupation,
    spec: NucleationSpec,
    pack: Any = None,
    model: Any = None,
) -> Optional[Tuple[Tuple[str, ...], np.ndarray, EdgeList]]:
    """Put 2p Cl on the fixed zb core (bridges first, then terminals).

    A bridge Cl sits *opposite* the shared Se of a Cd–Cd pair (Cd–Se–Cd–Cl
    rhombus).  A random lift used to land on that Se and trip the clash
    gate before g-xTB.  Terminals go along an unused tet hole of the host
    Cd (away from existing neighbours), not through the core COM.
    """

    cation, anion, ligand = _species(spec)
    r = _cdcl_bond(pack)
    n_cl = 2 * int(occ.p)
    symbols = list(occ.symbols)
    xyz = np.asarray(occ.coordinates, dtype=float).copy()
    edges = list(occ.core_edges)
    if n_cl <= 0:
        return tuple(symbols), xyz, tuple(sorted(edges))

    neigh: List[List[int]] = [[] for _ in symbols]
    for a, b in edges:
        neigh[a].append(b)
        neigh[b].append(a)
    se_ids = [i for i, s in enumerate(symbols) if s == anion]
    cd_ids = [i for i, s in enumerate(symbols) if s == cation]
    pairs: List[Tuple[float, int, int, int]] = []
    seen_pair = set()
    for se in se_ids:
        cds = [j for j in neigh[se] if symbols[j] == cation]
        for a, b in combinations(cds, 2):
            key = (min(a, b), max(a, b))
            if key in seen_pair:
                continue
            seen_pair.add(key)
            d = float(np.linalg.norm(xyz[a] - xyz[b]))
            pairs.append((d, a, b, se))
    pairs.sort()

    placed = 0
    for _d, a, b, se in pairs:
        if placed >= n_cl:
            break
        mid = 0.5 * (xyz[a] + xyz[b])
        axis = xyz[b] - xyz[a]
        nrm = float(np.linalg.norm(axis))
        if nrm < 1e-6:
            continue
        half = 0.5 * nrm
        # height so |Cd–Cl| ≈ r, Cl opposite Se
        if half >= r - 0.05:
            h = 0.80
        else:
            h = float(np.sqrt(max(r * r - half * half, 0.25)))
        away = _perp_away_from(axis, mid - xyz[se])
        pos = mid + away * h
        if _too_close(pos, xyz, floor=2.20, ignore=(a, b)):
            pos = mid - away * h
        if _too_close(pos, xyz, floor=2.20, ignore=(a, b)):
            continue
        idx = len(symbols)
        symbols.append(ligand)
        xyz = np.vstack([xyz, pos])
        edges.append((min(a, idx), max(a, idx)))
        edges.append((min(b, idx), max(b, idx)))
        neigh.append([a, b])
        neigh[a].append(idx)
        neigh[b].append(idx)
        placed += 1

    if placed < n_cl:
        cd_rank = sorted(cd_ids, key=lambda i: (len(neigh[i]), i))
        for cd in cd_rank:
            if placed >= n_cl:
                break
            # unused tet hole: away from current neighbours
            push = np.zeros(3)
            for nb in neigh[cd]:
                v = xyz[cd] - xyz[nb]
                n = float(np.linalg.norm(v))
                if n > 1e-8:
                    push += v / n
            n = float(np.linalg.norm(push))
            if n < 1e-8:
                continue
            pos = xyz[cd] + (push / n) * r
            if _too_close(pos, xyz, floor=2.20, ignore=(cd,)):
                continue
            idx = len(symbols)
            symbols.append(ligand)
            xyz = np.vstack([xyz, pos])
            edges.append((min(cd, idx), max(cd, idx)))
            neigh.append([cd])
            neigh[cd].append(idx)
            placed += 1

    if placed < n_cl:
        return None
    return tuple(symbols), xyz, tuple(sorted(set(edges)))


def construction_clash(
    symbols: Sequence[str],
    coords: np.ndarray,
    spec: NucleationSpec,
    *,
    floor: float = 2.20,
    bonded: Optional[Sequence[Edge]] = None,
) -> bool:
    """True if a pair that is not a Cd–Se / Cd–Cl contact is closer than floor."""

    pts = np.asarray(coords, dtype=float)
    n = len(symbols)
    allow = {
        tuple(sorted((spec.core.cation, spec.core.anion))),
        tuple(sorted((spec.precursor.center, spec.precursor.ligand))),
    }
    bonded_set = {tuple(sorted(e)) for e in (bonded or ())}
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            if d >= floor:
                continue
            pair = tuple(sorted((symbols[i], symbols[j])))
            if pair in allow and d >= 2.00:
                continue
            if bonded_set and (i, j) in bonded_set and d >= 2.00:
                continue
            return True
    return False
