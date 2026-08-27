"""Zinc-blende occupation growth (move Z).

Parent identity is a Cd–Se occupation on the CIF tet lattice, not the
relaxed XYZ.  Children are made by Z-type CdCl2 displacement on the
decorated parent, then filling a vacant cation+anion pair.  After g-xTB
the endpoint is retained in the growth lineage only when its Cd–Se
topology is preserved; changed minima are recorded off path rather than
snapped onto a new lattice fragment.

Cl is placed from the 2p graph law around the *fixed* zb core; Cl does
not occupy CIF virtual sites.  Shedding never removes a Cl-free Cd.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    site_ids: Tuple[str, ...] = ()
    occupation_id: str = ""
    parent_occupation_ids: Tuple[str, ...] = ()
    parent_structure_ids: Tuple[str, ...] = ()
    parent_id: str = ""
    shed: int = 0
    p_m: int = 0
    notes: str = ""
    #: Z-type leaving group that produced this child: geminal_terminal,
    #: geminal_bridge, borrowed, or empty when s = 0 / unrestricted.
    shed_kind: str = ""
    shed_se_wbo: float = 0.0


def _site_id(symbol: str, point: Sequence[float], tolerance: float) -> str:
    key = _position_key(np.asarray(point, dtype=float), tolerance)
    return f"{symbol}:{key[0]}:{key[1]}:{key[2]}"


def _occupation_shape_certificate(
    symbols: Sequence[str], coordinates: np.ndarray, tolerance: float
) -> str:
    """Rigid-motion and atom-order invariant certificate for a lattice site set.

    The complete, element-labelled distance graph distinguishes spatial lattice
    occupations that share the same nearest-neighbour Cd--Se graph.  It also
    merges translated, rotated, and reflected copies of the same finite ZB
    fragment, which is the symmetry reduction Move Z needs.
    """

    from itertools import permutations, product

    points = np.asarray(coordinates, dtype=float)
    scale = max(float(tolerance), 1.0e-6)
    quantized = np.rint(points / scale).astype(np.int64)
    canonical: Optional[Tuple[Tuple[Any, ...], ...]] = None
    # Cubic ZB symmetry is a subset of signed axis permutations.  Including
    # all 48 also merges mirror partners, which are energetically equivalent
    # in this achiral composition.  A deterministic occupied anchor removes
    # translation without throwing away the actual site arrangement.
    for axes in permutations(range(3)):
        permuted = quantized[:, axes]
        for signs in product((-1, 1), repeat=3):
            transformed = permuted * np.asarray(signs, dtype=np.int64)
            anchor_index = min(
                range(len(symbols)),
                key=lambda index: (
                    str(symbols[index]),
                    int(transformed[index, 0]),
                    int(transformed[index, 1]),
                    int(transformed[index, 2]),
                ),
            )
            anchor = transformed[anchor_index]
            rows = tuple(
                sorted(
                    (
                        str(symbol),
                        int(point[0] - anchor[0]),
                        int(point[1] - anchor[1]),
                        int(point[2] - anchor[2]),
                    )
                    for symbol, point in zip(symbols, transformed)
                )
            )
            if canonical is None or rows < canonical:
                canonical = rows
    payload = repr(canonical or ()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def ensure_occupation_identity(
    occupation: ZbOccupation,
    model: _LatticeModel,
) -> ZbOccupation:
    """Populate stable site and shape identities on an occupation in place."""

    if not occupation.site_ids:
        occupation.site_ids = tuple(
            _site_id(symbol, point, model.site_tolerance)
            for symbol, point in zip(occupation.symbols, occupation.coordinates)
        )
    if not occupation.occupation_id:
        cert = _occupation_shape_certificate(
            occupation.symbols, occupation.coordinates, model.site_tolerance
        )
        occupation.occupation_id = (
            f"zb_k{occupation.k:03d}_p{occupation.p:03d}_{cert}"
        )
    return occupation


def occupation_to_record(occupation: ZbOccupation) -> Dict[str, Any]:
    return {
        "occupation_id": occupation.occupation_id,
        "parent_occupation_ids": list(occupation.parent_occupation_ids),
        "parent_structure_ids": list(occupation.parent_structure_ids),
        "k": int(occupation.k),
        "p": int(occupation.p),
        "symbols": list(occupation.symbols),
        "lattice_coordinates": np.asarray(
            occupation.coordinates, dtype=float
        ).tolist(),
        "core_edges": [list(edge) for edge in occupation.core_edges],
        "site_ids": list(occupation.site_ids),
        "parent_id": occupation.parent_id,
        "shed": int(occupation.shed),
        "p_m": int(occupation.p_m),
        "notes": occupation.notes,
        "shed_kind": occupation.shed_kind,
        "shed_se_wbo": float(occupation.shed_se_wbo),
    }


def occupation_from_record(record: Dict[str, Any]) -> ZbOccupation:
    return ZbOccupation(
        k=int(record["k"]),
        p=int(record["p"]),
        symbols=tuple(str(value) for value in record["symbols"]),
        coordinates=np.asarray(record["lattice_coordinates"], dtype=float),
        core_edges=tuple(
            sorted((min(int(a), int(b)), max(int(a), int(b)))
                   for a, b in record["core_edges"])
        ),
        site_ids=tuple(str(value) for value in record.get("site_ids", ())),
        occupation_id=str(record.get("occupation_id") or ""),
        parent_occupation_ids=tuple(
            str(value) for value in record.get("parent_occupation_ids", ())
        ),
        parent_structure_ids=tuple(
            str(value) for value in record.get("parent_structure_ids", ())
        ),
        parent_id=str(record.get("parent_id") or ""),
        shed=int(record.get("shed", 0)),
        p_m=int(record.get("p_m", 0)),
        notes=str(record.get("notes") or ""),
        shed_kind=str(record.get("shed_kind") or ""),
        shed_se_wbo=float(record.get("shed_se_wbo") or 0.0),
    )


def load_occupation_manifest(path: Path) -> Dict[str, ZbOccupation]:
    """Return structure id -> stored lattice occupation.

    Occupations are the Move Z lineage, including routes whose unconstrained
    g-xTB minimum left the CIF (4-rings, topology=changed).  Those relaxed
    XYZ files are energetics only; the next k still grows from this lattice
    record.  Rows with no occupation payload are skipped.
    """

    manifest = Path(path)
    if manifest.is_dir():
        manifest = manifest / "zb_occupations.jsonl"
    out: Dict[str, ZbOccupation] = {}
    if not manifest.is_file():
        return out
    for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        structure_id = str(record.get("structure_id") or "")
        occupation_raw = record.get("occupation")
        if not structure_id or not isinstance(occupation_raw, dict):
            continue
        energy = record.get("energy_eV")
        chem = record.get("chemically_ok")
        if chem is False:
            continue
        if chem is None:
            violations = record.get("violations") or []
            if violations:
                continue
            if energy is None and not bool(
                record.get("propagation_eligible", False)
            ):
                continue
        out[structure_id] = occupation_from_record(occupation_raw)
    return out


def load_reference_occupation(
    path: Path,
    spec: NucleationSpec,
    model: _LatticeModel,
) -> ZbOccupation:
    """Load a core-only lattice reference used for endpoint diagnostics."""

    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    symbols = tuple(str(value) for value in raw.get("symbols", ()))
    coordinates = np.asarray(raw.get("coordinates", ()), dtype=float)
    if not symbols or coordinates.shape != (len(symbols), 3):
        raise ValueError(f"invalid ZB endpoint reference: {path}")
    atoms = [
        AtomRecord(
            index,
            symbol,
            tuple(float(value) for value in coordinates[index]),
            "core_anion" if symbol == spec.core.anion else "core_cation",
        )
        for index, symbol in enumerate(symbols)
    ]
    state = _make_core_graph(atoms, model, spec)
    k = symbols.count(spec.core.anion)
    p = symbols.count(spec.core.cation) - k
    occupation = occupation_from_state(
        state,
        spec,
        model=model,
        k=k,
        p=p,
        notes="endpoint_reference",
    )
    if occupation is None:
        raise ValueError(f"disconnected or invalid ZB endpoint reference: {path}")
    return occupation


def endpoint_similarity_diagnostic(
    occupation: ZbOccupation,
    reference: ZbOccupation,
    *,
    match_tolerance_A: float = 0.35,
) -> Dict[str, Any]:
    """Rotation-invariant shape comparison; never used as a growth filter.

    The best element-preserving assignment is evaluated over the 48 signed
    axis permutations of the cubic lattice.  This reports whether a k=13
    occupation approaches the known minimum Wulff-like Cd16Se13 core while
    leaving energetic selection entirely to g-xTB.
    """

    from itertools import permutations, product
    from scipy.optimize import linear_sum_assignment

    candidate_points = np.asarray(occupation.coordinates, dtype=float)
    reference_points = np.asarray(reference.coordinates, dtype=float)
    candidate_centered = candidate_points - candidate_points.mean(axis=0)
    reference_centered = reference_points - reference_points.mean(axis=0)
    candidate_symbols = np.asarray(occupation.symbols, dtype=object)
    reference_symbols = np.asarray(reference.symbols, dtype=object)

    best_matched = -1
    best_rmsd = float("inf")
    best_assignment_distances: List[float] = []
    for axes in permutations(range(3)):
        base = np.eye(3, dtype=float)[:, list(axes)]
        for signs in product((-1.0, 1.0), repeat=3):
            rotation = base @ np.diag(signs)
            rotated = candidate_centered @ rotation
            distances: List[float] = []
            for symbol in sorted(set(occupation.symbols) | set(reference.symbols)):
                ci = np.flatnonzero(candidate_symbols == symbol)
                ri = np.flatnonzero(reference_symbols == symbol)
                if not len(ci) or not len(ri):
                    continue
                matrix = np.linalg.norm(
                    rotated[ci, None, :] - reference_centered[None, ri, :],
                    axis=2,
                )
                rows, cols = linear_sum_assignment(matrix)
                distances.extend(float(matrix[row, col]) for row, col in zip(rows, cols))
            matched = sum(value <= float(match_tolerance_A) for value in distances)
            rmsd = (
                float(np.sqrt(np.mean(np.square(distances))))
                if distances
                else float("inf")
            )
            if matched > best_matched or (matched == best_matched and rmsd < best_rmsd):
                best_matched = matched
                best_rmsd = rmsd
                best_assignment_distances = distances

    def _degree_histogram(item: ZbOccupation) -> Dict[str, List[int]]:
        degrees = [0] * len(item.symbols)
        for left, right in item.core_edges:
            degrees[left] += 1
            degrees[right] += 1
        return {
            symbol: sorted(
                degrees[index]
                for index, value in enumerate(item.symbols)
                if value == symbol
            )
            for symbol in sorted(set(item.symbols))
        }

    def _extents(points: np.ndarray) -> List[float]:
        centered = points - points.mean(axis=0)
        return sorted(float(value) for value in np.ptp(centered, axis=0))

    denominator = max(len(occupation.symbols), len(reference.symbols), 1)
    return {
        "occupation_id": occupation.occupation_id,
        "k": int(occupation.k),
        "p": int(occupation.p),
        "reference_occupation_id": reference.occupation_id,
        "reference_k": int(reference.k),
        "reference_p": int(reference.p),
        "site_match_tolerance_A": float(match_tolerance_A),
        "matched_sites": int(max(0, best_matched)),
        "site_overlap_fraction": float(max(0, best_matched) / denominator),
        "assignment_rmsd_A": (
            None if not np.isfinite(best_rmsd) else float(best_rmsd)
        ),
        "assignment_max_A": (
            None
            if not best_assignment_distances
            else float(max(best_assignment_distances))
        ),
        "radius_gyration_A": _occupation_radius_of_gyration(occupation),
        "reference_radius_gyration_A": _occupation_radius_of_gyration(reference),
        "axis_extents_A": _extents(candidate_points),
        "reference_axis_extents_A": _extents(reference_points),
        "coordination_histogram": _degree_histogram(occupation),
        "reference_coordination_histogram": _degree_histogram(reference),
        "ranking_or_filtering_effect": "none",
    }


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
    ztype_geminal_terminal: int = 0
    ztype_geminal_bridge: int = 0
    ztype_borrowed: int = 0
    ztype_cl_free_cd: int = 0
    ztype_disjoint: int = 0

    def as_log(self) -> str:
        return (
            f"Z lineage={self.snapped}/{self.parents} "
            f"missing={self.snap_fail} n4_reject={self.n4_reject} "
            f"children={self.children} attach_try={self.attach_attempts} "
            f"clash_skip={self.clash_skip} "
            f"opt_keep={self.opt_keep} topology_changed={self.opt_reject_embed} "
            f"opt_fail={self.opt_fail} "
            f"ztype geminal_t={self.ztype_geminal_terminal} "
            f"geminal_b={self.ztype_geminal_bridge} "
            f"borrowed={self.ztype_borrowed} "
            f"cl_free={self.ztype_cl_free_cd} "
            f"disjoint_smax={self.ztype_disjoint}"
        )


def lattice_model(spec: NucleationSpec) -> _LatticeModel:
    return _build_lattice_model(spec)


def _species(spec: NucleationSpec) -> Tuple[str, str, str]:
    return spec.core.cation, spec.core.anion, spec.precursor.ligand


def seed_occupation(spec: NucleationSpec, model: _LatticeModel) -> ZbOccupation:
    """k=1 p=0 Cd–Se on zb."""

    state = _seed_state(model)
    occ = occupation_from_state(
        state, spec, model=model, k=1, p=0, parent_id="zb_seed"
    )
    if occ is None:
        raise RuntimeError("zb seed is not a connected Cd–Se pair")
    return occ


def occupation_from_state(
    state: _State,
    spec: NucleationSpec,
    *,
    model: _LatticeModel,
    k: int,
    p: int,
    parent_id: str = "",
    shed: int = 0,
    p_m: int = 0,
    notes: str = "",
    parent_occupation_ids: Sequence[str] = (),
    parent_structure_ids: Sequence[str] = (),
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
    occupation = ZbOccupation(
        k=int(k),
        p=int(p),
        symbols=tuple(symbols),
        coordinates=pts,
        core_edges=tuple(sorted(set(edges))),
        parent_occupation_ids=tuple(str(value) for value in parent_occupation_ids),
        parent_structure_ids=tuple(str(value) for value in parent_structure_ids),
        parent_id=parent_id,
        shed=int(shed),
        p_m=int(p_m),
        notes=notes,
    )
    return ensure_occupation_identity(occupation, model)


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
        state, spec, model=model, k=k, p=p, parent_id=parent_id, notes="snap"
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
        model=model,
        k=occ.k,
        p=occ.p - s,
        parent_id=occ.parent_id,
        shed=s,
        notes="shed",
        parent_occupation_ids=(occ.occupation_id,),
    )


def _occupation_without_cd(
    occ: ZbOccupation,
    drop: Sequence[int],
    spec: NucleationSpec,
    model: _LatticeModel,
    *,
    s: int,
    notes: str,
) -> Optional[ZbOccupation]:
    import networkx as nx

    drop_set = {int(i) for i in drop}
    keep = [i for i in range(len(occ.symbols)) if i not in drop_set]
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
    candidate = occupation_from_state(
        state,
        spec,
        model=model,
        k=occ.k,
        p=occ.p - int(s),
        parent_id=occ.parent_id,
        shed=int(s),
        notes=notes,
        parent_occupation_ids=(occ.occupation_id,),
    )
    if candidate is None:
        return None
    graph = nx.Graph()
    graph.add_nodes_from(range(len(candidate.symbols)))
    graph.add_edges_from(candidate.core_edges)
    if not nx.is_connected(graph):
        return None
    return candidate


def shed_occupations(
    occ: ZbOccupation,
    s: int,
    spec: NucleationSpec,
    model: _LatticeModel,
    *,
    remove_sets: Optional[Sequence[Sequence[int]]] = None,
) -> List[ZbOccupation]:
    """Enumerate unique connected removals of ``s`` excess Cd sites.

    By default every combination of cation sites is tried and symmetry
    canonicalization merges equivalent occupations.  Pass ``remove_sets``
    (Move Z Z-type displacement) to restrict the Cd that may leave.
    """

    if s <= 0:
        return [ensure_occupation_identity(occ, model)]
    if s > occ.p:
        return []
    cation = spec.core.cation
    if remove_sets is None:
        cd_ids = [i for i, symbol in enumerate(occ.symbols) if symbol == cation]
        trials: Sequence[Sequence[int]] = list(combinations(cd_ids, int(s)))
        notes = "shed_all"
    else:
        trials = [
            tuple(int(i) for i in combo)
            for combo in remove_sets
            if len(combo) == int(s)
        ]
        notes = "shed_ztype"
    unique: Dict[str, ZbOccupation] = {}
    for removed in trials:
        candidate = _occupation_without_cd(
            occ, removed, spec, model, s=int(s), notes=notes
        )
        if candidate is None:
            continue
        unique.setdefault(candidate.occupation_id, candidate)
    return list(unique.values())


ZTYPE_KIND_RANK = {
    "geminal_terminal": 0,
    "geminal_bridge": 1,
    "borrowed": 2,
}


@dataclass(frozen=True)
class ZtypeShedUnit:
    """One Z-type CdCl2 leaving group on a decorated parent.

    ``cd`` is an occupation index.  ``cl`` are indices into the parent XYZ.
    """

    cd: int
    cl: Tuple[int, int]
    kind: str
    kind_rank: int
    n_se: int
    se_wbo: float
    score: Tuple[Any, ...]


def _wbo_pair(
    wbo: Optional[Mapping[Tuple[int, int], float]], left: int, right: int
) -> float:
    if not wbo:
        return 0.0
    return abs(float(wbo.get((min(int(left), int(right)), max(int(left), int(right))), 0.0)))


def _map_occupation_to_parent(
    occupation: ZbOccupation, parent_symbols: Sequence[str]
) -> Optional[List[int]]:
    """Occupation index → parent XYZ index, or None if the cores differ."""

    n = len(occupation.symbols)
    if tuple(parent_symbols[:n]) == tuple(occupation.symbols):
        return list(range(n))
    used: set = set()
    mapping = [-1] * n
    for i, symbol in enumerate(occupation.symbols):
        for j, parent_symbol in enumerate(parent_symbols):
            if j in used or parent_symbol != symbol:
                continue
            mapping[i] = j
            used.add(j)
            break
        if mapping[i] < 0:
            return None
    return mapping


def ztype_shed_units(
    occupation: ZbOccupation,
    spec: NucleationSpec,
    *,
    parent_symbols: Sequence[str],
    parent_coordinates: np.ndarray,
    parent_edges: Sequence[Edge] = (),
    parent_wbo: Optional[Mapping[Tuple[int, int], float]] = None,
    ligand_bond_length: float = 0.0,
) -> List[ZtypeShedUnit]:
    """Legal Z-type CdCl2 units on a decorated parent.

    A Cd with no Cl is never a unit, including Cd–Se CN = 3.  Geminal
    Cl–Cd–Cl (two Cl already on the Cd) is preferred over CdCl plus a
    Cl borrowed from elsewhere.  Two terminal Cl outrank a pair that
    includes a μ2 bridge.  Ranking inside a kind is the Cd–Se Wiberg
    sum of the leaving cation (the bonds that break); without WBO it
    is the Cd–Se degree.
    """

    cation, anion, ligand = _species(spec)
    if ligand not in parent_symbols:
        return []
    cd_ids = [
        i for i, symbol in enumerate(occupation.symbols) if symbol == cation
    ]
    n_se = [0] * len(occupation.symbols)
    for left, right in occupation.core_edges:
        if occupation.symbols[left] == anion:
            n_se[right] += 1
        if occupation.symbols[right] == anion:
            n_se[left] += 1
    parent_xyz = np.asarray(parent_coordinates, dtype=float)
    cl_ids = [
        i for i, symbol in enumerate(parent_symbols) if symbol == ligand
    ]
    if not cl_ids:
        return []
    core_map = _map_occupation_to_parent(occupation, parent_symbols)
    neigh: Dict[int, List[int]] = {i: [] for i in range(len(parent_symbols))}
    for left, right in parent_edges or ():
        a, b = int(left), int(right)
        if a < len(parent_symbols) and b < len(parent_symbols):
            neigh[a].append(b)
            neigh[b].append(a)
    cutoff = float(ligand_bond_length) if ligand_bond_length > 0.0 else 2.90
    cl_of: Dict[int, List[int]] = {i: [] for i in cd_ids}
    cl_dent: Dict[int, int] = {j: 0 for j in cl_ids}
    aligned = core_map is not None
    has_cdcl = any(
        a < len(parent_symbols)
        and b < len(parent_symbols)
        and {parent_symbols[a], parent_symbols[b]} == {cation, ligand}
        for a, b in (parent_edges or ())
    )
    if aligned and has_cdcl:
        parent_of = {occ: parent for occ, parent in enumerate(core_map or [])}
        for occ_cd in cd_ids:
            parent_cd = parent_of[occ_cd]
            for j in neigh[parent_cd]:
                if parent_symbols[j] == ligand:
                    cl_of[occ_cd].append(j)
        for j in cl_ids:
            cl_dent[j] = sum(
                1 for n in neigh[j] if parent_symbols[n] == cation
            )
    else:
        for occ_cd in cd_ids:
            point = np.asarray(occupation.coordinates[occ_cd], dtype=float)
            if (
                aligned
                and core_map is not None
                and core_map[occ_cd] < len(parent_xyz)
            ):
                point = parent_xyz[core_map[occ_cd]]
            for j in cl_ids:
                if j >= len(parent_xyz):
                    continue
                if float(np.linalg.norm(parent_xyz[j] - point)) <= cutoff:
                    cl_of[occ_cd].append(j)
        for j in cl_ids:
            cl_dent[j] = sum(1 for occ_cd in cd_ids if j in cl_of[occ_cd])

    def se_wbo_of(occ_cd: int) -> float:
        if parent_wbo and aligned and core_map is not None:
            parent_cd = core_map[occ_cd]
            total = 0.0
            for left, right in occupation.core_edges:
                other = right if left == occ_cd else left if right == occ_cd else None
                if other is None:
                    continue
                if occupation.symbols[other] != anion:
                    continue
                total += _wbo_pair(parent_wbo, parent_cd, core_map[other])
            if total > 0.0:
                return total
        return float(n_se[occ_cd])

    def make_unit(occ_cd: int, cl_a: int, cl_b: int, kind: str) -> ZtypeShedUnit:
        pair = (min(int(cl_a), int(cl_b)), max(int(cl_a), int(cl_b)))
        rank = int(ZTYPE_KIND_RANK[kind])
        se = se_wbo_of(occ_cd)
        return ZtypeShedUnit(
            cd=int(occ_cd),
            cl=pair,
            kind=kind,
            kind_rank=rank,
            n_se=int(n_se[occ_cd]),
            se_wbo=float(se),
            score=(rank, round(float(se), 6), int(n_se[occ_cd]), int(occ_cd), pair),
        )

    units: List[ZtypeShedUnit] = []
    seen: set = set()
    for occ_cd in cd_ids:
        chlorides = sorted(set(cl_of[occ_cd]))
        if len(chlorides) >= 2:
            for cl_a, cl_b in combinations(chlorides, 2):
                terminal = cl_dent.get(cl_a, 1) <= 1 and cl_dent.get(cl_b, 1) <= 1
                kind = "geminal_terminal" if terminal else "geminal_bridge"
                unit = make_unit(occ_cd, cl_a, cl_b, kind)
                key = (unit.cd, unit.cl, unit.kind)
                if key in seen:
                    continue
                seen.add(key)
                units.append(unit)
        elif len(chlorides) == 1:
            own = chlorides[0]
            for other in cl_ids:
                if other == own:
                    continue
                unit = make_unit(occ_cd, own, other, "borrowed")
                key = (unit.cd, unit.cl, unit.kind)
                if key in seen:
                    continue
                seen.add(key)
                units.append(unit)
    units.sort(key=lambda unit: unit.score)
    return units


def _best_ztype_assignment(
    cds: Sequence[int],
    by_cd: Mapping[int, Sequence[ZtypeShedUnit]],
) -> Optional[List[ZtypeShedUnit]]:
    """Disjoint Cl assignment, one unit per Cd.  ``s`` is at most 2."""

    ordered = [int(cd) for cd in cds]
    best: Optional[List[ZtypeShedUnit]] = None
    best_key: Optional[Tuple[Any, ...]] = None

    def rec(index: int, used: set, picked: List[ZtypeShedUnit]) -> None:
        nonlocal best, best_key
        if index == len(ordered):
            key = (
                sum(unit.kind_rank for unit in picked),
                round(sum(unit.se_wbo for unit in picked), 6),
            )
            if best_key is None or key < best_key:
                best = list(picked)
                best_key = key
            return
        for unit in by_cd.get(ordered[index], ()):
            if unit.cl[0] in used or unit.cl[1] in used:
                continue
            rec(index + 1, used | {unit.cl[0], unit.cl[1]}, picked + [unit])

    rec(0, set(), [])
    return best


def ztype_removal_sets(
    units: Sequence[ZtypeShedUnit],
    s: int,
) -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], List[ZtypeShedUnit]]]:
    """Combinations of ``s`` Cd that can leave as disjoint CdCl2 units."""

    if s <= 0 or not units:
        return [], {}
    by_cd: Dict[int, List[ZtypeShedUnit]] = {}
    for unit in units:
        by_cd.setdefault(int(unit.cd), []).append(unit)
    cds = sorted(by_cd)
    if len(cds) < int(s):
        return [], {}
    sets: List[Tuple[int, ...]] = []
    meta: Dict[Tuple[int, ...], List[ZtypeShedUnit]] = {}
    for combo in combinations(cds, int(s)):
        assigned = _best_ztype_assignment(combo, by_cd)
        if assigned is None:
            continue
        key = tuple(sorted(combo))
        sets.append(key)
        meta[key] = assigned
    return sets, meta


def max_ztype_shed(units: Sequence[ZtypeShedUnit], *, cap: int) -> int:
    """Largest number of disjoint Z-type units, limited by ``cap``."""

    limit = min(int(cap), len({unit.cd for unit in units}))
    for s in range(limit, 0, -1):
        sets, _meta = ztype_removal_sets(units, s)
        if sets:
            return s
    return 0


def ztype_census(
    occupation: ZbOccupation,
    units: Sequence[ZtypeShedUnit],
    spec: NucleationSpec,
) -> Dict[str, int]:
    cation = spec.core.cation
    cd_ids = [
        i for i, symbol in enumerate(occupation.symbols) if symbol == cation
    ]
    with_cl = {unit.cd for unit in units}
    best: Dict[int, str] = {}
    for unit in units:
        prev = best.get(unit.cd)
        if prev is None or unit.kind_rank < ZTYPE_KIND_RANK.get(prev, 9):
            best[unit.cd] = unit.kind
    geminal_t = sum(1 for kind in best.values() if kind == "geminal_terminal")
    geminal_b = sum(1 for kind in best.values() if kind == "geminal_bridge")
    borrowed = sum(1 for kind in best.values() if kind == "borrowed")
    return {
        "geminal_terminal": geminal_t,
        "geminal_bridge": geminal_b,
        "borrowed": borrowed,
        "cl_free": int(len(cd_ids) - len(with_cl)),
        "disjoint": max_ztype_shed(units, cap=max(0, occupation.p)),
    }


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
            model=model,
            k=occ.k + 1,
            p=occ.p,
            parent_id=occ.parent_id,
            shed=occ.shed,
            notes="attach",
            parent_occupation_ids=(occ.occupation_id,),
        )
        if new is None:
            continue
        key = new.occupation_id
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
        frontier = nxt if cap <= 0 else nxt[: max(1, cap)]
    out: List[ZbOccupation] = []
    seen_e: set = set()
    for st in frontier:
        new = occupation_from_state(
            st,
            spec,
            model=model,
            k=occ.k,
            p=occ.p + int(p_m),
            parent_id=occ.parent_id,
            shed=occ.shed,
            p_m=int(p_m),
            notes="pm",
            parent_occupation_ids=(occ.occupation_id,),
        )
        if new is None:
            continue
        key = new.occupation_id
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
    relaxed_parent_coordinates: Optional[np.ndarray] = None,
    parent_wbo: Optional[Dict[Tuple[int, int], float]] = None,
    parent_ligand_coordinates: Optional[np.ndarray] = None,
    ligand_bond_length: float = 0.0,
    ztype_units: Optional[Sequence[ZtypeShedUnit]] = None,
) -> List[ZbOccupation]:
    """Z-type shed s CdCl2, attach one CdSe, add p_m precursor Cd.

    When ``ztype_units`` is provided (decorated parent), only Cl-bearing Cd
    leave, as disjoint CdCl2 packages.  ``None`` keeps the lattice-wide
    enumeration used by core-only tests.  An empty list means no legal
    Z-type unit and ``s > 0`` yields nothing.

    ``parent_ligand_coordinates`` are the relaxed positions of the parent's
    Cl, which the occupation itself does not carry.  They are what makes the
    ranking able to tell an open coordination slot from one a ligand already
    fills; without them the ordering falls back to the core-only form.
    """

    combo_meta: Dict[Tuple[int, ...], List[ZtypeShedUnit]] = {}
    remove_sets: Optional[List[Tuple[int, ...]]] = None
    if int(s) > 0 and ztype_units is not None:
        remove_sets, combo_meta = ztype_removal_sets(ztype_units, int(s))
        if not remove_sets:
            return []
    after_all = shed_occupations(
        occ, s, spec, model, remove_sets=remove_sets
    )
    if not after_all:
        return []
    if stats is not None:
        stats.attach_attempts += len(after_all)
    parent_index = {site_id: i for i, site_id in enumerate(occ.site_ids)}

    def _annotate(child: ZbOccupation, after: ZbOccupation) -> None:
        child.shed = s
        child.p_m = p_m
        child.parent_id = occ.parent_id
        child.parent_occupation_ids = (occ.occupation_id,)
        child.parent_structure_ids = (occ.parent_id,) if occ.parent_id else ()
        if not combo_meta or not occ.site_ids or not after.site_ids:
            return
        after_sites = set(after.site_ids)
        removed = tuple(
            sorted(
                parent_index[site_id]
                for site_id in occ.site_ids
                if site_id not in after_sites and site_id in parent_index
            )
        )
        assigned = combo_meta.get(removed)
        if not assigned:
            return
        worst = max(assigned, key=lambda unit: unit.kind_rank)
        child.shed_kind = worst.kind
        child.shed_se_wbo = float(sum(unit.se_wbo for unit in assigned))

    unique: Dict[str, ZbOccupation] = {}
    for after in after_all:
        attached = attach_cdse(after, spec, model, cap=0)
        for core in attached:
            core.shed = s
            packed = _add_precursor_cd(core, p_m, spec, model, cap=0)
            for child in packed:
                _annotate(child, after)
                existing = unique.get(child.occupation_id)
                if existing is None:
                    unique[child.occupation_id] = child
                    continue
                # Same lattice child from two Z-type removals: keep the easier one.
                old_rank = ZTYPE_KIND_RANK.get(existing.shed_kind, 9)
                new_rank = ZTYPE_KIND_RANK.get(child.shed_kind, 9)
                if (new_rank, child.shed_se_wbo) < (old_rank, existing.shed_se_wbo):
                    unique[child.occupation_id] = child
    ordered = sorted(
        unique.values(),
        key=lambda child: _growth_site_priority(
            child,
            occ,
            relaxed_parent_coordinates=relaxed_parent_coordinates,
            parent_wbo=parent_wbo,
            parent_ligand_coordinates=parent_ligand_coordinates,
            ligand_bond_length=ligand_bond_length,
        ),
    )
    return ordered if cap <= 0 else ordered[: int(cap)]


def _occupation_radius_of_gyration(occupation: ZbOccupation) -> float:
    points = np.asarray(occupation.coordinates, dtype=float)
    if not len(points):
        return 0.0
    centered = points - points.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def _growth_site_priority(
    child: ZbOccupation,
    parent: ZbOccupation,
    *,
    relaxed_parent_coordinates: Optional[np.ndarray],
    parent_wbo: Optional[Dict[Tuple[int, int], float]],
    parent_ligand_coordinates: Optional[np.ndarray] = None,
    ligand_bond_length: float = 0.0,
) -> Tuple[Any, ...]:
    """Soft ordering for a post-dedup child cap; never a legality filter.

    Two terms decide where the next CdSe monomer goes.

    ``shell_completion`` counts the host cations this child takes from three
    Cd-Se bonds to four.  That is the nucleation event: a cation with four Se
    is an interior atom, and it is the only move that makes one.  It ranks
    first because ``host_deficit`` below actively buries it -- a cation with
    three Se scores the *lowest* deficit precisely when it is one bond from
    closing, so completions landed at rank 66-281 of ~500 and the cap of 16
    discarded 726 of 726 of them over the 28 k=6 parents of growth_prod.

    ``open_valence`` replaces the core-only deficit with 4 - CN_Se - CN_Cl.
    A slot a chloride already fills is not a slot: measured on the relaxed
    k=6/k=7 endpoints, cations scoring deficit 3 carry 2.33 Cl on average and
    are coordinatively saturated, and the correlation between host_deficit
    and the real free valence is +0.05 -- the old term carries essentially no
    information about where a monomer can actually attach.  ``host_deficit``
    is kept as a last-resort tiebreak for the no-ligand-data path.
    """

    parent_sites = set(parent.site_ids)
    added = {
        index
        for index, site_id in enumerate(child.site_ids)
        if site_id not in parent_sites
    }
    parent_index = {site_id: i for i, site_id in enumerate(parent.site_ids)}
    child_sites = set(child.site_ids)
    removed_parent_indices = {
        index
        for index, site_id in enumerate(parent.site_ids)
        if site_id not in child_sites
    }
    host_parent_indices: set[int] = set()
    for left, right in child.core_edges:
        if left in added and child.site_ids[right] in parent_index:
            host_parent_indices.add(parent_index[child.site_ids[right]])
        if right in added and child.site_ids[left] in parent_index:
            host_parent_indices.add(parent_index[child.site_ids[left]])

    parent_degrees = [0] * len(parent.symbols)
    for left, right in parent.core_edges:
        parent_degrees[left] += 1
        parent_degrees[right] += 1
    host_deficit = sum(max(0, 4 - parent_degrees[index]) for index in host_parent_indices)

    # Cd-Se degree the hosts reach in the child, by site id.
    child_index = {site_id: i for i, site_id in enumerate(child.site_ids)}
    child_degrees: Dict[int, int] = {}
    for left, right in child.core_edges:
        child_degrees[left] = child_degrees.get(left, 0) + 1
        child_degrees[right] = child_degrees.get(right, 0) + 1
    shell_completion = 0
    for index in host_parent_indices:
        position = child_index.get(parent.site_ids[index])
        if position is None:
            continue
        if parent_degrees[index] == 3 and child_degrees.get(position, 0) >= 4:
            shell_completion += 1

    # Free valence of the hosts, chloride included.
    open_valence = 0
    if (
        parent_ligand_coordinates is not None
        and len(parent_ligand_coordinates)
        and float(ligand_bond_length) > 0.0
    ):
        ligands = np.asarray(parent_ligand_coordinates, dtype=float)
        reference = np.asarray(parent.coordinates, dtype=float)
        cutoff = float(ligand_bond_length)
        for index in host_parent_indices:
            if index >= len(reference):
                continue
            n_ligand = int(
                np.count_nonzero(
                    np.linalg.norm(ligands - reference[index], axis=1) <= cutoff
                )
            )
            open_valence += max(0, 4 - parent_degrees[index] - n_ligand)
    else:
        open_valence = host_deficit

    displacement = 0.0
    if relaxed_parent_coordinates is not None:
        relaxed = np.asarray(relaxed_parent_coordinates, dtype=float)
        reference = np.asarray(parent.coordinates, dtype=float)
        if relaxed.shape == reference.shape and len(reference) >= 2:
            rc = relaxed - relaxed.mean(axis=0)
            lc = reference - reference.mean(axis=0)
            u, _singular, vt = np.linalg.svd(rc.T @ lc)
            rotation = u @ vt
            if np.linalg.det(rotation) < 0.0:
                u[:, -1] *= -1.0
                rotation = u @ vt
            aligned = rc @ rotation + reference.mean(axis=0)
            displacement = sum(
                float(np.linalg.norm(aligned[index] - reference[index]))
                for index in host_parent_indices
            )
    # Z-type desorption coordinate: geminal CdCl2 before borrowed Cl, then
    # the Cd–Se Wiberg sum of the leaving cation (the bonds that break).
    # Cd–Cl WBO is not the reaction coordinate.  Ranking only; legality is
    # decided by ztype_shed_units.
    kind_rank = int(ZTYPE_KIND_RANK.get(getattr(child, "shed_kind", ""), 3))
    removal_wbo = float(getattr(child, "shed_se_wbo", 0.0) or 0.0)
    if removal_wbo <= 0.0 and parent_wbo and removed_parent_indices:
        n_core = len(parent.symbols)
        removal_wbo = sum(
            abs(float(order))
            for (left, right), order in parent_wbo.items()
            if (left in removed_parent_indices or right in removed_parent_indices)
            and left < n_core
            and right < n_core
            and parent.symbols[left] != parent.symbols[right]
        )
    # Ascending sort, so each term is negated to mean "more is better".
    #   kind_rank         geminal terminal, geminal bridge, borrowed
    #   removal_wbo       Cd–Se WBO of the leaving Cd (weaker first)
    #   shell_completion  hosts going 3 -> 4 Cd-Se: the nucleation event
    #   open_valence      free slots, 4 - CN_Se - CN_Cl, chloride included
    #   host_deficit      old core-only 4 - CN_Se, now only a tiebreak
    #   displacement      strain in the relaxed parent at the host
    #   radius_of_gyration / n_edges / id   compactness, then determinism
    return (
        kind_rank,
        round(removal_wbo, 6),
        -shell_completion,
        -open_valence,
        -host_deficit,
        -round(displacement, 6),
        _occupation_radius_of_gyration(child),
        -len(child.core_edges),
        child.occupation_id,
    )


def occupation_diversity_signature(
    occupation: ZbOccupation,
) -> Tuple[Any, ...]:
    """Coarse shape/coordination class used only for diversity retention."""

    graph_degrees = [0] * len(occupation.symbols)
    for left, right in occupation.core_edges:
        graph_degrees[left] += 1
        graph_degrees[right] += 1
    by_species = tuple(
        (
            symbol,
            tuple(
                sorted(
                    graph_degrees[index]
                    for index, candidate in enumerate(occupation.symbols)
                    if candidate == symbol
                )
            ),
        )
        for symbol in sorted(set(occupation.symbols))
    )
    points = np.asarray(occupation.coordinates, dtype=float)
    diameter = max(
        (
            float(np.linalg.norm(points[left] - points[right]))
            for left in range(len(points))
            for right in range(left + 1, len(points))
        ),
        default=0.0,
    )
    return (
        by_species,
        round(_occupation_radius_of_gyration(occupation) / 0.25),
        round(diameter / 0.25),
    )


def occupation_compactness_key(occupation: ZbOccupation) -> Tuple[Any, ...]:
    """Sort key for filling the parent cap with lattice-only routes.

    More 6-rings and branching first (compact ZB cuts), then smaller
    radius of gyration.  Collapsed g-xTB energy is not used.
    """

    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(len(occupation.symbols)))
    graph.add_edges_from(occupation.core_edges)
    n6 = sum(1 for cycle in nx.cycle_basis(graph) if len(cycle) == 6)
    degrees = [0] * len(occupation.symbols)
    for left, right in occupation.core_edges:
        degrees[left] += 1
        degrees[right] += 1
    n_cn3 = sum(1 for degree in degrees if degree >= 3)
    return (
        -int(n6),
        -int(n_cn3),
        round(_occupation_radius_of_gyration(occupation), 3),
        -len(occupation.core_edges),
        occupation.occupation_id,
    )


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


DEFAULT_BRIDGE_CD_CD_MIN_A = 3.20
DEFAULT_BRIDGE_CD_CD_MAX_A = 4.75
_SEGMENT_TOL_A = 0.40


def cation_on_segment(
    start: np.ndarray,
    end: np.ndarray,
    points: np.ndarray,
    *,
    tol_A: float = _SEGMENT_TOL_A,
) -> bool:
    """True if any point lies on the open segment start–end (occupied midpoint)."""

    axis = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length2 = float(np.dot(axis, axis))
    if length2 < 1.0e-12 or len(points) == 0:
        return False
    origin = np.asarray(start, dtype=float)
    for point in np.asarray(points, dtype=float):
        t = float(np.dot(point - origin, axis) / length2)
        if t <= 0.05 or t >= 0.95:
            continue
        if float(np.linalg.norm(point - (origin + t * axis))) <= float(tol_A):
            return True
    return False


def zb_metric_bridge_pairs(
    pair_list: Sequence[Edge],
    coordinates: np.ndarray,
    cd_indices: Sequence[int],
    spec: NucleationSpec,
    *,
    min_distance: float = DEFAULT_BRIDGE_CD_CD_MIN_A,
) -> List[Edge]:
    """Keep Cd–Cd pairs whose lattice length can host a μ2 Cl.

    Graph hop 2/4 is a chemical prefilter; this is the metric gate.  Pairs
    longer than ``bridge_cd_cd_max_distance`` (ZB NN is 4.34 Å; two 2.4 Å
    spheres stop meeting near 4.8 Å) or whose segment contains another
    occupied cation (collinear 8.68 Å through an interior Cd) are dropped.
    With no configured max, the list is unchanged so molecular growth is
    untouched.
    """

    configured = spec.graph_rules.bridge_cd_cd_max_distance
    if configured is None:
        return [tuple(sorted((int(a), int(b)))) for a, b in pair_list]
    max_d = float(configured)
    min_d = float(min_distance)
    pts = np.asarray(coordinates, dtype=float)
    cd = [int(i) for i in cd_indices]
    kept: List[Edge] = []
    for left, right in pair_list:
        a, b = int(left), int(right)
        if a >= len(pts) or b >= len(pts):
            continue
        distance = float(np.linalg.norm(pts[a] - pts[b]))
        if distance < min_d or distance > max_d:
            continue
        others = np.asarray(
            [pts[i] for i in cd if i not in (a, b) and i < len(pts)],
            dtype=float,
        )
        if len(others) and cation_on_segment(pts[a], pts[b], others):
            continue
        kept.append((min(a, b), max(a, b)))
    return kept


def _mu2_sites(
    host_a: np.ndarray,
    host_b: np.ndarray,
    radius: float,
) -> List[np.ndarray]:
    """Sphere-intersection points for a μ2 Cl, or empty if the hosts are too far."""

    a = np.asarray(host_a, dtype=float)
    b = np.asarray(host_b, dtype=float)
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < 1.0e-6 or length >= 2.0 * float(radius) - 0.02:
        return []
    unit = axis / length
    mid = a + 0.5 * axis
    height2 = float(radius) ** 2 - (0.5 * length) ** 2
    height = float(np.sqrt(max(height2, 0.25)))
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(tmp, unit))) > 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    perp = np.cross(unit, tmp)
    norm = float(np.linalg.norm(perp))
    if norm < 1.0e-8:
        return []
    perp = perp / norm
    return [mid + perp * height, mid - perp * height]


def place_cl_on_zb_core(
    state: _State,
    anchored: Mapping[int, Sequence[float]],
    spec: NucleationSpec,
    pack: Any = None,
    *,
    clash_floor: float = 2.20,
) -> Optional[np.ndarray]:
    """Put each graph Cl at a lattice-feasible 3D site on the frozen ZB core.

    μ2 Cl starts at the *outward* sphere intersection of its two hosts (away
    from the core COM, not on an occupied cation).  Terminals go along an
    unused tet direction of the host.  Returns None if a Cl has no legal site.
    """

    n_atoms = len(state.atoms)
    if n_atoms == 0 or not anchored:
        return None
    xyz = np.zeros((n_atoms, 3), dtype=float)
    for index, point in anchored.items():
        if 0 <= int(index) < n_atoms:
            xyz[int(index)] = np.asarray(point, dtype=float)
    cation, anion, ligand = _species(spec)
    radius = _cdcl_bond(pack)
    core_idx = [i for i in range(n_atoms) if state.atoms[i].symbol != ligand]
    if not core_idx:
        return xyz
    core_com = xyz[core_idx].mean(axis=0)
    cl_ids = [i for i, atom in enumerate(state.atoms) if atom.symbol == ligand]
    for cl in cl_ids:
        hosts = [
            int(j)
            for j in state.graph.neighbors(cl)
            if state.atoms[int(j)].symbol == cation
        ]
        pos: Optional[np.ndarray] = None
        ignore = tuple(hosts)
        if len(hosts) == 2:
            sites = _mu2_sites(xyz[hosts[0]], xyz[hosts[1]], radius)
            sites.sort(
                key=lambda p: -float(np.linalg.norm(p - core_com))
            )
            for candidate in sites:
                if not _too_close(candidate, xyz, floor=clash_floor, ignore=ignore):
                    pos = candidate
                    break
        elif len(hosts) == 1:
            host = hosts[0]
            push = np.zeros(3)
            for nb in state.graph.neighbors(host):
                if int(nb) == cl:
                    continue
                vec = xyz[host] - xyz[int(nb)]
                norm = float(np.linalg.norm(vec))
                if norm > 1.0e-8:
                    push += vec / norm
            norm = float(np.linalg.norm(push))
            if norm > 1.0e-8:
                candidate = xyz[host] + (push / norm) * radius
                if not _too_close(candidate, xyz, floor=clash_floor, ignore=(host,)):
                    pos = candidate
        elif len(hosts) >= 3:
            triangle = xyz[list(hosts[:3])]
            center = triangle.mean(axis=0)
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            nrm = float(np.linalg.norm(normal))
            if nrm > 1.0e-8:
                normal = normal / nrm
                if float(np.dot(center + normal - core_com, normal)) < 0.0:
                    normal = -normal
                candidate = center + normal * 1.10
                if not _too_close(candidate, xyz, floor=clash_floor, ignore=ignore):
                    pos = candidate
        if pos is None:
            return None
        xyz[cl] = pos
    from .molecular_rules import cl_on_cn4_cd_violations

    if cl_on_cn4_cd_violations(state, spec, xyz):
        return None
    return xyz


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
    """True if a pair that is not a Cd–Se / Cd–Cl contact is closer than floor.

    Also true if any Cl sits within the Cd–Cl bond of a cation that already
    has four Se neighbours — that is a fifth ligand in space, not a graph
    CN = 5.
    """

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
    from .molecular_rules import (
        DEFAULT_CD_CL_BOND_MAX,
        DEFAULT_CD_SE_BOND_MAX,
        _pair_bond_max,
    )

    cation = spec.core.cation
    anion = spec.core.anion
    ligand = spec.precursor.ligand
    se_cut = _pair_bond_max(spec, cation, anion, DEFAULT_CD_SE_BOND_MAX)
    cl_cut = _pair_bond_max(spec, cation, ligand, DEFAULT_CD_CL_BOND_MAX)
    cd = [i for i, s in enumerate(symbols) if s == cation]
    se = [i for i, s in enumerate(symbols) if s == anion]
    cl = [i for i, s in enumerate(symbols) if s == ligand]
    for i in cd:
        n_se = sum(
            1 for j in se if float(np.linalg.norm(pts[i] - pts[j])) <= se_cut
        )
        if n_se < 4:
            continue
        for j in cl:
            if float(np.linalg.norm(pts[i] - pts[j])) <= cl_cut:
                return True
    return False
