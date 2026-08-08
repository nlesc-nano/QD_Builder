"""Global, closure-aware reconstruction of anion-centred molecular motifs.

The graph owns atom identity and shared Cd atoms.  Motifs therefore contribute
geometry factors rather than independent rigid-body coordinates: a cycle is
closed by construction, and its strain is distributed over the empirical
bands instead of being left on one final spanning-tree edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import least_squares

from .molecular_rules import molecular_geometry_ok, pair_key
from .types import FloatArray, _State


@dataclass(frozen=True)
class MotifReconstructionCandidate:
    coordinates: Tuple[Tuple[float, float, float], ...]
    audit_violations: Tuple[str, ...]
    objective: float
    start_index: int
    max_bond_error_A: float


@dataclass(frozen=True)
class MotifReconstructionResult:
    candidates: Tuple[MotifReconstructionCandidate, ...]
    motif_violations: Tuple[str, ...] = ()
    starts_attempted: int = 0


def motif_vocabulary_violations(
    state: _State,
    *,
    cation: str = "Cd",
    anion: str = "Se",
    ligand: str = "Cl",
    motif_definitions: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Validate the five construction motifs requested for CdSe/CdCl2."""

    allowed: dict[str, set[int]] = {}
    if isinstance(motif_definitions, Mapping):
        for entry in motif_definitions.values():
            if not isinstance(entry, Mapping):
                continue
            center = entry.get("center")
            count = entry.get("linker_count")
            if center is not None and count is not None:
                allowed.setdefault(str(center), set()).add(int(count))
    if not allowed:
        allowed = {anion: {3, 4}, ligand: {1, 2, 3}}
    out: List[str] = []
    for atom in state.atoms:
        if atom.symbol not in allowed:
            continue
        degree = int(state.graph.degree[atom.atom_id])
        if degree not in allowed[atom.symbol]:
            out.append(
                f"unsupported_motif:{atom.symbol}-"
                f"{cation}{degree}:{atom.atom_id}"
            )
    return out


def _angle_deg(xyz: FloatArray, left: int, center: int, right: int) -> float:
    u = xyz[left] - xyz[center]
    v = xyz[right] - xyz[center]
    den = float(np.linalg.norm(u) * np.linalg.norm(v))
    if den < 1.0e-12:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(float(np.dot(u, v)) / den, -1.0, 1.0))))


def _improper_deg(
    xyz: FloatArray, center: int, first: int, second: int, third: int
) -> float:
    vectors = []
    for index in (first, second, third):
        vector = xyz[index] - xyz[center]
        vectors.append(vector / max(float(np.linalg.norm(vector)), 1.0e-12))
    sine = float(np.dot(vectors[0], np.cross(vectors[1], vectors[2])))
    return float(np.degrees(np.arcsin(np.clip(sine, -1.0, 1.0))))


def _periodic_delta(value: float, target: float) -> float:
    return float((value - target + 180.0) % 360.0 - 180.0)


def _junction_type(state: _State, index: int) -> str:
    return f"{state.atoms[index].symbol}_CN{int(state.graph.degree[index])}"


def _junction_cd_angle_terms(
    state: _State, junctions: Mapping[str, Any]
) -> List[Tuple[int, int, int, float, float, int]]:
    rows = junctions.get("cd_angle") or []
    terms: List[Tuple[int, int, int, float, float, int]] = []
    for atom in state.atoms:
        if atom.symbol != "Cd":
            continue
        center = atom.atom_id
        neighbors = [
            int(i) for i in state.graph.neighbors(center)
            if state.atoms[i].symbol in {"Se", "Cl"}
        ]
        for left, right in combinations(neighbors, 2):
            pair = sorted((_junction_type(state, left), _junction_type(state, right)))
            for row in rows:
                when = row.get("when") or {}
                expected = sorted((str(when.get("a1")), str(when.get("a2"))))
                if pair != expected or int(when.get("cd_cn", -1)) != state.graph.degree[center]:
                    continue
                terms.append((
                    left, center, right, float(row["deg"]),
                    max(float(row.get("tol_deg", 8.0)), 1.0),
                    int(row.get("n", 0)),
                ))
                break
    return terms


def _junction_improper_terms(
    state: _State, junctions: Mapping[str, Any]
) -> List[Tuple[int, int, int, int, float, float, int]]:
    rows = junctions.get("cd_improper_cn3") or []
    terms: List[Tuple[int, int, int, int, float, float, int]] = []
    for atom in state.atoms:
        center = atom.atom_id
        if atom.symbol != "Cd" or state.graph.degree[center] != 3:
            continue
        neighbors = sorted(int(i) for i in state.graph.neighbors(center))
        signature = sorted(_junction_type(state, i) for i in neighbors)
        for row in rows:
            when = row.get("when") or {}
            if signature != sorted(str(x) for x in when.get("neighbors", [])):
                continue
            terms.append((
                center, neighbors[0], neighbors[1], neighbors[2],
                float(row.get("abs_deg", 0.0)),
                max(float(row.get("tol_deg", 8.0)), 1.0),
                int(row.get("n", 0)),
            ))
            break
    return terms


def _junction_coplanar_terms(
    state: _State, junctions: Mapping[str, Any]
) -> List[Tuple[int, int, int, int, float, float]]:
    """Find shared anion/Cd2-Cl junctions that should be planar.

    The four atoms are ordered as ``Se-Cd-Cd-Cl``.  The two Cd atoms are the
    common linkers; the residual is the signed angle of Cl out of the plane
    defined by Se and that Cd pair.
    """

    rows = junctions.get("coplanar_shared_pair") or []
    terms: List[Tuple[int, int, int, int, float, float]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        when = row.get("when") or row
        anion = str(when.get("anion", "Se"))
        ligand = str(when.get("ligand", "Cl"))
        # Omitting anion_cn/ligand_cn (or giving -1) means "any coordination":
        # the Cd-Cl-Cd-anion four-ring is planar regardless of how coordinated
        # its anion and ligand are, so a rule can cover every such ring.
        anion_cn = int(when.get("anion_cn", when.get("se_cn", -1)))
        ligand_cn = int(when.get("ligand_cn", when.get("cl_cn", -1)))
        tolerance = max(float(row.get("tol_deg", row.get("tolerance_deg", 8.0))), 1.0)
        weight = float(row.get("weight", 1.0))
        for se in state.atoms:
            if se.symbol != anion:
                continue
            if anion_cn >= 0 and state.graph.degree[se.atom_id] != anion_cn:
                continue
            se_hosts = {
                int(i)
                for i in state.graph.neighbors(se.atom_id)
                if state.atoms[i].symbol == "Cd"
            }
            for cl in state.atoms:
                if cl.symbol != ligand:
                    continue
                if (
                    ligand_cn >= 0
                    and state.graph.degree[cl.atom_id] != ligand_cn
                ):
                    continue
                common = sorted(
                    se_hosts.intersection(
                        int(i)
                        for i in state.graph.neighbors(cl.atom_id)
                        if state.atoms[i].symbol == "Cd"
                    )
                )
                if len(common) != 2:
                    continue
                terms.append((se.atom_id, cl.atom_id, common[0], common[1], tolerance, weight))
    return terms


def _coplanarity_deg(
    xyz: FloatArray, se: int, cl: int, cd1: int, cd2: int
) -> float:
    """Signed angle (degrees) of Cl out of the Se-Cd-Cd plane."""

    axis = xyz[cd2] - xyz[cd1]
    se_vec = xyz[se] - xyz[cd1]
    cl_vec = xyz[cl] - xyz[cd1]
    normal = np.cross(axis, se_vec)
    normal_norm = float(np.linalg.norm(normal))
    cl_norm = float(np.linalg.norm(cl_vec))
    if normal_norm < 1.0e-12 or cl_norm < 1.0e-12:
        return 90.0
    sine = float(np.dot(cl_vec, normal) / (normal_norm * cl_norm))
    return float(np.degrees(np.arcsin(np.clip(sine, -1.0, 1.0))))


def _aligned_rmsd(left: FloatArray, right: FloatArray) -> float:
    a = np.asarray(left, dtype=float) - np.mean(left, axis=0)
    b = np.asarray(right, dtype=float) - np.mean(right, axis=0)
    u, _s, vt = np.linalg.svd(a.T @ b)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    delta = a @ rotation.T - b
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def reconstruct_motif_state(
    state: _State,
    pack: Any,
    spec: Any,
    *,
    starts: int = 12,
    keep: int = 3,
    seed: int = 1729,
    max_nfev: int = 40,
    overlap_min_A: float = 0.75,
    start_max_bond_error_A: float = 0.50,
) -> MotifReconstructionResult:
    """Return distinct whole-graph fits, including promising audit failures."""

    # Local import avoids a module cycle: molecular owns the authoritative
    # audit enumeration, while this module supplies an optional reconstruction.
    from . import molecular as M

    reconstruction = pack.raw.get("reconstruction") or {}
    audit_mode = str(reconstruction.get("audit", "exact")).strip().lower()
    clash_only = audit_mode in {"clash", "clashes", "clash_only", "clashes_only"}
    layout_iterations = max(20, int(reconstruction.get("layout_iterations", 40)))

    motif_errors = motif_vocabulary_violations(
        state,
        cation=spec.core.cation,
        anion=spec.core.anion,
        ligand=spec.precursor.ligand,
        motif_definitions=pack.raw.get("motifs"),
    )
    if motif_errors:
        return MotifReconstructionResult((), tuple(motif_errors), 0)

    n_atoms = len(state.atoms)
    degrees = [int(state.graph.degree[i]) for i in range(n_atoms)]
    bonds = [
        (int(a), int(b), float(M._molecular_bond_length(state, pack, spec, a, b, degrees)))
        for a, b in state.graph.edges
    ]
    audited_impropers, audited_angles = M._audited_local_terms(state, pack, spec)
    junctions = pack.raw.get("junctions") or {}
    junction_angles = _junction_cd_angle_terms(state, junctions)
    junction_impropers = _junction_improper_terms(state, junctions)
    junction_coplanar = _junction_coplanar_terms(state, junctions)
    bonded = {tuple(sorted((int(a), int(b)))) for a, b in state.graph.edges}
    pair_bands: List[Tuple[int, int, float]] = []
    for left in range(n_atoms):
        for right in range(left + 1, n_atoms):
            if (left, right) in bonded:
                continue
            rule = spec.graph_rules.pair_rules.get(
                pair_key(state.atoms[left].symbol, state.atoms[right].symbol)
            )
            if rule is None:
                continue
            floor = (
                float(rule.min_distance or 0.0)
                if not rule.bond_allowed
                else float(rule.bond_max_distance or 0.0)
            )
            if floor > 0.0:
                pair_bands.append((left, right, floor + 0.03))

    def residual(flat: FloatArray, tether: FloatArray) -> FloatArray:
        xyz = np.asarray(flat, dtype=float).reshape((n_atoms, 3))
        values: List[float] = []
        for left, right, target in bonds:
            values.append((float(np.linalg.norm(xyz[left] - xyz[right])) - target) / 0.025)
        for left, center, right, target, band in audited_angles:
            values.append(
                _periodic_delta(_angle_deg(xyz, left, center, right), target)
                / max(2.0, float(band) / 3.0)
            )
        for center, first, second, third, target in audited_impropers:
            values.append(
                _periodic_delta(
                    _improper_deg(xyz, center, first, second, third), target
                ) / 4.0
            )
        for left, right, floor in pair_bands:
            distance = float(np.linalg.norm(xyz[left] - xyz[right]))
            values.append(min(0.0, distance - floor) / 0.04)
        # Junction statistics guide closure but never override the exact audit.
        for left, center, right, target, width, _count in junction_angles:
            values.append(
                0.25 * _periodic_delta(
                    _angle_deg(xyz, left, center, right), target
                ) / width
            )
        for center, first, second, third, target, width, _count in junction_impropers:
            actual = abs(_improper_deg(xyz, center, first, second, third))
            values.append(0.20 * (actual - target) / width)
        for se, cl, cd1, cd2, width, weight in junction_coplanar:
            values.append(weight * _coplanarity_deg(xyz, se, cl, cd1, cd2) / width)
        values.extend((np.mean(xyz, axis=0) / 0.10).tolist())
        # Break exact rotational degeneracy without making a rigid motif.
        values.extend((0.001 * (xyz - tether)).reshape(-1).tolist())
        return np.asarray(values, dtype=float)

    accepted: List[MotifReconstructionCandidate] = []
    graph_seed = int(sum((a + 1) * (b + 7) for a, b in state.graph.edges))
    for start_index in range(max(1, int(starts))):
        layout = nx.spring_layout(
            state.graph,
            dim=3,
            seed=seed + graph_seed + start_index,
            iterations=layout_iterations,
            scale=3.5,
        )
        initial = np.asarray([layout[i] for i in range(n_atoms)], dtype=float)
        rng = np.random.default_rng(seed + 31 * graph_seed + start_index)
        initial += rng.normal(0.0, 0.18, initial.shape)
        initial -= np.mean(initial, axis=0)
        try:
            residual_count = len(residual(initial.reshape(-1), initial))
            method = "lm" if residual_count >= initial.size else "trf"
            fitted = least_squares(
                lambda flat: residual(flat, initial),
                initial.reshape(-1),
                method=method,
                max_nfev=max(1, int(max_nfev)),
                ftol=1.0e-8,
                xtol=1.0e-8,
                gtol=1.0e-8,
            )
        except Exception:  # one failed start must not discard the graph
            continue
        xyz = np.asarray(fitted.x, dtype=float).reshape((n_atoms, 3))
        if not np.all(np.isfinite(xyz)):
            continue
        bond_errors = [
            abs(float(np.linalg.norm(xyz[left] - xyz[right])) - target)
            for left, right, target in bonds
        ]
        max_bond_error = max(bond_errors, default=0.0)
        min_distance = min(
            (
                float(np.linalg.norm(xyz[left] - xyz[right]))
                for left in range(n_atoms)
                for right in range(left + 1, n_atoms)
            ),
            default=float("inf"),
        )
        # xTB can repair imperfect angles, but not an atom overlap or a graph
        # whose intended bonds were never made in the first place.
        if min_distance < float(overlap_min_A):
            continue
        if not clash_only and max_bond_error > float(start_max_bond_error_A):
            continue
        if clash_only:
            violations = M._motif_clash_violations(
                state, xyz, overlap_min_A=overlap_min_A
            )
        else:
            violations = M._exact_bond_violations(state, xyz, pack, spec)
            violations += M._exact_local_geometry_violations(state, xyz, pack, spec)
            if not violations:
                _ok, contact = molecular_geometry_ok(state, xyz, spec)
                violations += contact
        candidate = MotifReconstructionCandidate(
            coordinates=tuple(tuple(float(v) for v in row) for row in xyz),
            audit_violations=tuple(violations),
            objective=float(2.0 * fitted.cost),
            start_index=start_index,
            max_bond_error_A=max_bond_error,
        )
        if any(
            _aligned_rmsd(xyz, np.asarray(other.coordinates, dtype=float)) < 0.15
            for other in accepted
        ):
            continue
        accepted.append(candidate)
        accepted.sort(key=lambda item: (bool(item.audit_violations), len(item.audit_violations), item.objective))
        del accepted[max(1, int(keep)):]

    return MotifReconstructionResult(
        candidates=tuple(accepted),
        starts_attempted=max(1, int(starts)),
    )
