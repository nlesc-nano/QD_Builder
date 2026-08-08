from __future__ import annotations

from itertools import combinations, permutations

from .scoring import _coordination_score, _graph_coordination_score, _coordination_metadata, _target_cn

from .graph_ops import *  # private names via __all__

from .scoring import *  # private names via __all__

from .lattice import *  # private names via __all__

from .types import *  # private names via __all__

import math
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from ..nc_types import NucleationSpec
from .types import (
    AtomRecord,
    ClusterRecord,
    FloatArray,
    NucleationRegistry,
    _LatticeModel,
    _ProgressReporter,
)

# Half-angle for C2v terminal pair so ∠T–Cd–T = arccos(-1/3) ≈ 109.47°.
_TETRAHEDRAL_PAIR_HALF_ANGLE = 0.5 * math.acos(-1.0 / 3.0)


def _precondition_retained_registry(
    registry: NucleationRegistry,
    model: _LatticeModel,
    spec: NucleationSpec,
    progress: _ProgressReporter,
    discarded_counts: Optional[Mapping[int, Mapping[int, int]]] = None,
) -> None:
    """Attach a surface view without mutating construction-native coordinates."""

    total = sum(
        len(records) for bins in registry.values() for records in bins.values()
    )
    processed = 0
    started = time.monotonic()
    for k, bins in sorted(registry.items()):
        for p, records in sorted(bins.items()):
            if not records:
                continue
            progress.emit(
                f"surface geometry k={k} p={p}: "
                f"projecting {len(records)} retained structures"
            )
            for record in records:
                processed += 1
                _attach_surface_geometry(record, model, spec)
                progress.heartbeat(
                    f"surface geometry: processed={processed}/{total}, "
                    f"elapsed={time.monotonic() - started:.1f}s"
                )
    if discarded_counts is not None:
        for k, bins in sorted(discarded_counts.items()):
            for p, n_disc in sorted(bins.items()):
                if n_disc and not registry.get(k, {}).get(p):
                    progress.emit(
                        f"surface geometry k={k} p={p}: skip "
                        f"(0 retained, {int(n_disc)} discarded in bin)"
                    )
    progress.emit(
        f"surface geometry complete: processed={processed}, "
        f"elapsed={time.monotonic() - started:.1f}s"
    )


def _attach_surface_geometry(
    record: ClusterRecord,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> None:
    """Attach deterministic surface coordinates and their validity audit."""

    coordinates, metadata = _precondition_surface_geometry(record, model, spec)
    record.surface_coordinates_data = tuple(
        tuple(float(value) for value in point) for point in coordinates
    )
    record.metadata["surface_geometry"] = metadata


def _precondition_surface_geometry(
    record: ClusterRecord,
    model: _LatticeModel,
    spec: NucleationSpec,
    *,
    audit: bool = True,
) -> Tuple[FloatArray, Dict[str, object]]:
    """Construct retained-only coordinates with deterministic local rules."""

    native = record.coordinates
    surface = native.copy()
    applied: List[Dict[str, object]] = []
    unresolved: List[str] = []
    bridge_edges = [
        (left, right, data)
        for left, right, data in record.graph.edges(data=True)
        if data.get("kind") == "surface_bridge"
    ]
    bridged_hosts = {
        node
        for left, right, _data in bridge_edges
        for node in (left, right)
        if record.atoms[node].symbol != spec.precursor.ligand
    }

    # Anion positions remain fixed.  A cation whose neighbours are *all* anions
    # may move onto the line (CN 2) or plane (CN 3) they define -- DFT on k=2 p=0
    # confirms a CN-2 Cd relaxes to a pseudo-linear Se-Cd-Se, so the compressed
    # Cd-Se distance this produces is an acceptable pre-relaxation guess.
    #
    # Bridge hosts are deliberately excluded.  DFT on k=1 p=2 shows CN-3 Cd each
    # carrying two bridges keep their near-tetrahedral geometry rather than
    # flattening to trigonal planar, and moving such a cation also drags its
    # bridging Cl off the shared vacant CIF site it must occupy -- measured, that
    # collapsed a Cd-Cl-Cd angle from 109.47 to 60 degrees.  A CN-2 Cd carrying a
    # *terminal* Cl is handled further down by ``se_cd_cl_linear``, which reaches
    # the DFT geometry (180 degrees at the reference bond length) by moving the
    # ligand rather than the cation.
    for atom in record.atoms:
        if atom.symbol != spec.core.cation or atom.atom_id in bridged_hosts:
            continue
        neighbors = list(record.graph.neighbors(atom.atom_id))
        se_neighbors = [
            index
            for index in neighbors
            if record.atoms[index].symbol == spec.core.anion
        ]
        if len(neighbors) == 2 and len(se_neighbors) == 2:
            surface[atom.atom_id] = np.mean(surface[se_neighbors], axis=0)
            applied.append(
                {"rule": "cd_cn2_two_se_midpoint", "atom_id": atom.atom_id}
            )
        elif len(neighbors) == 3 and len(se_neighbors) == 3:
            projected = _project_point_to_plane(
                surface[atom.atom_id], surface[se_neighbors]
            )
            if projected is None:
                unresolved.append(
                    f"Cd {atom.atom_id}: three-Se plane is degenerate"
                )
            else:
                surface[atom.atom_id] = projected
                applied.append(
                    {"rule": "cd_cn3_three_se_plane", "atom_id": atom.atom_id}
                )

    # A bridge fixes its Cl before terminal ligands are rebuilt.  Rhombic
    # bridges use the DFT-motivated Cd-Se-Cd plane; exact-site bridges occupy
    # their common vacant CIF anion position directly.
    bridge_ligands: set[int] = set()
    for left, right, data in bridge_edges:
        ligand = (
            left
            if record.atoms[left].symbol == spec.precursor.ligand
            else right
        )
        second_host = right if ligand == left else left
        primary_hosts = [
            neighbor
            for neighbor in record.graph.neighbors(ligand)
            if neighbor != second_host
            and record.atoms[neighbor].symbol == spec.core.cation
        ]
        if len(primary_hosts) != 1:
            unresolved.append(f"Cl {ligand}: incomplete bridge metadata")
            continue
        primary = primary_hosts[0]
        bridge_mode = str(
            data.get("bridge_mode", "shared_occupied_neighbor")
        )
        shared = data.get("shared_neighbor")
        virtual_site = data.get("virtual_site")
        if bridge_mode == "shared_vacant_cif_site":
            if (
                not isinstance(virtual_site, (tuple, list))
                or len(virtual_site) != 3
            ):
                unresolved.append(f"Cl {ligand}: missing shared CIF site")
                continue
            position = np.asarray(virtual_site, dtype=float)
            rule_name = "shared_vacant_cif_site_bridge"
        else:
            if not isinstance(shared, int):
                unresolved.append(f"Cl {ligand}: missing shared Se")
                continue
            position = _symmetric_bridge_position(
                surface[primary],
                surface[second_host],
                surface[shared],
                float(data.get("surface_angle_deg", 90.0)),
                native[ligand],
            )
            if position is None:
                unresolved.append(f"Cl {ligand}: degenerate bridge plane")
                continue
            rule_name = "symmetric_cl_bridge"
        surface[ligand] = position
        bridge_ligands.add(ligand)
        applied.append(
            {
                "rule": rule_name,
                "atom_id": ligand,
                "host_atom_ids": [primary, second_host],
                "shared_neighbor_atom_id": shared,
                "bridge_mode": bridge_mode,
                "virtual_site_position": (
                    list(virtual_site) if virtual_site is not None else None
                ),
                "target_angle_deg": data.get("surface_angle_deg"),
                "out_of_plane_rotation_deg": 0.0,
            }
        )

    # Place terminal Cl atoms as a group around each Cd center.
    term_applied, term_unresolved = _apply_terminal_ligand_surface_rules(
        record=record,
        surface=surface,
        native=native,
        bridge_ligands=bridge_ligands,
        bridged_hosts=bridged_hosts,
        model=model,
        spec=spec,
        pass_label="primary",
    )
    applied.extend(term_applied)
    unresolved.extend(term_unresolved)

    # Final ligand pass: re-enforce preferred terminal geometries (same spirit as
    # CN2 linear / CN3 planar cation fixes) even when they leave CIF virtual
    # sites.  Critical for seed (1,2) where CN3 + two bridges + one terminal
    # must sit on the Cl–Cd–Cl bisector, not a skewed native CIF residual.
    final_applied, final_unresolved = _final_ligand_geometry_pass(
        record=record,
        surface=surface,
        native=native,
        bridge_ligands=bridge_ligands,
        bridged_hosts=bridged_hosts,
        model=model,
        spec=spec,
    )
    applied.extend(final_applied)
    unresolved.extend(final_unresolved)

    # Joint refine: if any terminal sits near any bridge Cl, reassign terminals
    # on that Cd maximizing separation from *all* bridge Cl (not only local).
    # Secondary only — does not replace the always-on final geometry pass.
    joint_refined = _joint_refine_terminals_near_bridges(
        record=record,
        surface=surface,
        native=native,
        bridge_ligands=bridge_ligands,
        model=model,
        spec=spec,
    )
    if joint_refined:
        applied.extend(joint_refined)

    if not np.all(np.isfinite(surface)):
        surface = native.copy()
        unresolved.append(
            "non-finite projection rejected; construction-native coordinates used"
        )

    if not audit:
        # Surface gate: only near-coincident atoms (0.75 Å), not full soft-clash.
        collisions, intrusions = _surface_conflicts(
            record, surface, spec, clash_radius=0.75
        )
        return surface, {
            "projection_valid": not collisions and not intrusions,
            "coordinate_collisions": collisions,
            "saturated_cd_intrusions": [
                {
                    "cd_atom_id": intrusion["cd_atom_id"],
                    "intruding_atom_id": intrusion["intruding_atom_id"],
                }
                for intrusion in intrusions
            ],
        }

    return surface, _surface_geometry_metadata(
        record,
        native,
        surface,
        spec,
        applied_rules=applied,
        unresolved_conflicts=unresolved,
    )


def _unit(vector: FloatArray) -> Optional[FloatArray]:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return None
    return vector / norm


def _project_point_to_plane(
    point: FloatArray,
    plane_points: FloatArray,
) -> Optional[FloatArray]:
    normal = _unit(
        np.cross(plane_points[1] - plane_points[0], plane_points[2] - plane_points[0])
    )
    if normal is None:
        return None
    return point - np.dot(point - plane_points[0], normal) * normal


def _terminal_position_two_fixed_bisector(
    center: FloatArray,
    fixed_a: FloatArray,
    fixed_b: FloatArray,
    *,
    bond_length: float,
    native_ligand: FloatArray,
    cluster_com: Optional[FloatArray] = None,
) -> Optional[FloatArray]:
    """Place a terminal Cl on the planar bisector completion of two fixed ligands.

    Used for CN=3 (or CN4 two-bridge) hosts where the preferred terminal site
    lies on the σ_d / σ_v-like plane that bisects ∠L₁–Cd–L₂ (H₂O-like),
    opposite the fixed wedge: direction ``-(û₁ + û₂)``.  Call sites:

    * two bridge Cl + one terminal
    * Se + one bridge Cl + one terminal

    May leave CIF virtual sites; that is intentional for the surface final-pass
    philosophy.  When the bisector is near-degenerate (fixed ligands nearly
    linear), fall back to the native projection and, if available, prefer the
    side away from the cluster centre of mass.
    """

    direction = _outward_planar_bisector(
        center, fixed_a, fixed_b, native_ligand
    )
    if direction is None:
        return None
    first = _unit(fixed_a - center)
    second = _unit(fixed_b - center)
    if (
        first is not None
        and second is not None
        and cluster_com is not None
        and _unit(first + second) is None
    ):
        # Degenerate linear fixed pair: choose the native-side direction that
        # points away from the cluster COM when possible.
        away = _unit(center - cluster_com)
        if away is not None and float(np.dot(direction, away)) < 0.0:
            direction = -direction
    return center + bond_length * direction


# Backward-compatible alias used by older tests / call sites.
_terminal_position_two_bridge_bisector = _terminal_position_two_fixed_bisector


def _terminal_pair_c2v_two_fixed(
    center: FloatArray,
    fixed_a: FloatArray,
    fixed_b: FloatArray,
    *,
    bond_length: float,
    native_ligands: Sequence[FloatArray],
) -> Optional[Tuple[FloatArray, FloatArray]]:
    """Place two terminal Cl with C₂ᵥ symmetry about two fixed ligands (P2).

    Terminals lie in the σ plane that bisects ∠A–Cd–B, symmetric above/below
    the A–Cd–B plane.  Closed-form neighborhood angles only (no LS / CIF fit).
    Handles off-lattice bridges (a·b ≠ −1/3): both terminals remain equivalent
    w.r.t. A and B by construction.

    ``native_ligands`` length 2 assigns which terminal gets which direction by
    minimal native displacement.
    """

    a = _unit(fixed_a - center)
    b = _unit(fixed_b - center)
    if a is None or b is None:
        return None
    m = _unit(-(a + b))
    n = _unit(np.cross(a, b))
    if m is None or n is None:
        return None
    cos_phi = math.cos(_TETRAHEDRAL_PAIR_HALF_ANGLE)
    sin_phi = math.sin(_TETRAHEDRAL_PAIR_HALF_ANGLE)
    t1 = _unit(m * cos_phi + n * sin_phi)
    t2 = _unit(m * cos_phi - n * sin_phi)
    if t1 is None or t2 is None:
        return None
    positions = (
        center + bond_length * t1,
        center + bond_length * t2,
    )
    if len(native_ligands) != 2:
        return positions
    # Assign directions to terminals by Hungarian-of-two: min total native SSQ.
    orderings = (
        (positions[0], positions[1]),
        (positions[1], positions[0]),
    )
    best = min(
        orderings,
        key=lambda ordered: sum(
            float(np.sum((pos - nat) ** 2))
            for pos, nat in zip(ordered, native_ligands)
        ),
    )
    return best


def _apply_terminal_ligand_surface_rules(
    *,
    record: ClusterRecord,
    surface: FloatArray,
    native: FloatArray,
    bridge_ligands: set[int],
    bridged_hosts: set[int],
    model: _LatticeModel,
    spec: NucleationSpec,
    pass_label: str,
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Place terminal (CN1) ligands by neighborhood angles around each cation.

    Unified final-pass table (surface only; bridges already fixed), enabled when
    ``spec.terminal_motifs == "zb_mx2"``:

    * **P3** CN2 anion + 1 T → linear
    * **P4** CN3 anion + 2 T (no bridge) → σ_d pair
    * **P1** 1 T + 2 fixed (2 Br, or anion+Br, or two non-bridge) → exterior bisector
    * **P2** 2 T + 2 fixed (anion+Br, or 2 Br) → C₂ᵥ pair about the fixed angle

    ``terminal_motifs: none`` skips this table.  Native lattice coords are not
    written here — construction occupation for next-k stays on virtual sites.
    """

    applied: List[Dict[str, object]] = []
    unresolved: List[str] = []
    motifs = str(getattr(spec, "terminal_motifs", "zb_mx2")).strip().lower()
    if motifs in {"", "none", "off", "false", "0"}:
        return applied, unresolved
    cation_symbols = {spec.core.cation, spec.precursor.center}
    cluster_com = np.mean(surface, axis=0)

    for center_atom in record.atoms:
        if center_atom.symbol not in cation_symbols:
            continue
        center = center_atom.atom_id
        neighbors = list(record.graph.neighbors(center))
        terminal = [
            index
            for index in neighbors
            if record.atoms[index].symbol == spec.precursor.ligand
            and index not in bridge_ligands
        ]
        if not terminal:
            continue
        se_neighbors = [
            index
            for index in neighbors
            if record.atoms[index].symbol == spec.core.anion
        ]
        fixed_neighbors = [
            index for index in neighbors if index not in terminal
        ]
        degree = len(neighbors)
        bridge_fixed = [
            index for index in fixed_neighbors if index in bridge_ligands
        ]
        n_se = len(se_neighbors)
        n_br = len(bridge_fixed)
        n_t = len(terminal)

        # --- P3: CN2 Se–Cd–Cl linear (no bridge) ---
        if degree == 2 and n_se == 1 and n_t == 1 and n_br == 0:
            direction = _unit(surface[center] - surface[se_neighbors[0]])
            if direction is None:
                unresolved.append(f"Cd {center}: zero Se-Cd direction")
                continue
            surface[terminal[0]] = surface[center] + model.bond_length * direction
            applied.append(
                {
                    "rule": "se_cd_cl_linear",
                    "atom_id": terminal[0],
                    "center_atom_id": center,
                    "pass": pass_label,
                }
            )
            continue

        # --- P4: CN3 Se + 2 terminals, no bridge ---
        if degree == 3 and n_se == 1 and n_t == 2 and n_br == 0:
            positions = _sigma_d_cl_pair(
                center,
                se_neighbors[0],
                terminal,
                native,
                surface,
                model,
                spec,
            )
            if positions is None:
                unresolved.append(f"Cd {center}: no sigma-d plane")
                continue
            for ligand, position in zip(terminal, positions):
                surface[ligand] = position
            applied.append(
                {
                    "rule": "se_cd_cl2_trigonal_sigma_d",
                    "atom_ids": list(terminal),
                    "center_atom_id": center,
                    "pass": pass_label,
                }
            )
            continue

        # --- P1: one terminal, two bridges ---
        if n_t == 1 and n_br == 2:
            position = _terminal_position_two_fixed_bisector(
                surface[center],
                surface[bridge_fixed[0]],
                surface[bridge_fixed[1]],
                bond_length=model.bond_length,
                native_ligand=native[terminal[0]],
                cluster_com=cluster_com,
            )
            if position is None:
                unresolved.append(
                    f"Cd {center}: two-bridge terminal bisector degenerate"
                )
                continue
            surface[terminal[0]] = position
            applied.append(
                {
                    "rule": "cd_two_bridge_terminal_bisector",
                    "atom_id": terminal[0],
                    "center_atom_id": center,
                    "bridge_ligand_ids": list(bridge_fixed),
                    "host_degree": degree,
                    "pass": pass_label,
                }
            )
            continue

        # --- P1: one terminal, Se + one bridge ---
        if n_t == 1 and n_br == 1 and n_se == 1:
            position = _terminal_position_two_fixed_bisector(
                surface[center],
                surface[se_neighbors[0]],
                surface[bridge_fixed[0]],
                bond_length=model.bond_length,
                native_ligand=native[terminal[0]],
                cluster_com=cluster_com,
            )
            if position is None:
                unresolved.append(
                    f"Cd {center}: Se–bridge terminal bisector degenerate"
                )
                continue
            surface[terminal[0]] = position
            applied.append(
                {
                    "rule": "cd_cn3_se_bridge_terminal_bisector",
                    "atom_id": terminal[0],
                    "center_atom_id": center,
                    "se_atom_id": se_neighbors[0],
                    "bridge_ligand_id": bridge_fixed[0],
                    "host_degree": degree,
                    "pass": pass_label,
                }
            )
            continue

        # --- P2: two terminals, Se + one bridge (Cd9 class) ---
        if n_t == 2 and n_br == 1 and n_se == 1:
            pair = _terminal_pair_c2v_two_fixed(
                surface[center],
                surface[se_neighbors[0]],
                surface[bridge_fixed[0]],
                bond_length=model.bond_length,
                native_ligands=(native[terminal[0]], native[terminal[1]]),
            )
            if pair is None:
                unresolved.append(
                    f"Cd {center}: Se–bridge C2v terminal pair degenerate"
                )
                continue
            for ligand, position in zip(terminal, pair):
                surface[ligand] = position
            applied.append(
                {
                    "rule": "cd_cn4_se_bridge_two_terminal_c2v",
                    "atom_ids": list(terminal),
                    "center_atom_id": center,
                    "se_atom_id": se_neighbors[0],
                    "bridge_ligand_id": bridge_fixed[0],
                    "host_degree": degree,
                    "pass": pass_label,
                }
            )
            continue

        # --- P2: two terminals, two bridges (rare) ---
        if n_t == 2 and n_br == 2:
            pair = _terminal_pair_c2v_two_fixed(
                surface[center],
                surface[bridge_fixed[0]],
                surface[bridge_fixed[1]],
                bond_length=model.bond_length,
                native_ligands=(native[terminal[0]], native[terminal[1]]),
            )
            if pair is None:
                unresolved.append(
                    f"Cd {center}: two-bridge C2v terminal pair degenerate"
                )
                continue
            for ligand, position in zip(terminal, pair):
                surface[ligand] = position
            applied.append(
                {
                    "rule": "cd_two_bridge_two_terminal_c2v",
                    "atom_ids": list(terminal),
                    "center_atom_id": center,
                    "bridge_ligand_ids": list(bridge_fixed),
                    "host_degree": degree,
                    "pass": pass_label,
                }
            )
            continue

        # --- P1: CN3 one terminal, two non-bridge fixed (or residual) ---
        if degree == 3 and n_t == 1 and len(fixed_neighbors) == 2:
            position = _terminal_position_two_fixed_bisector(
                surface[center],
                surface[fixed_neighbors[0]],
                surface[fixed_neighbors[1]],
                bond_length=model.bond_length,
                native_ligand=native[terminal[0]],
                cluster_com=cluster_com,
            )
            if position is None:
                unresolved.append(
                    f"Cd {center}: planar bisector is degenerate"
                )
                continue
            surface[terminal[0]] = position
            applied.append(
                {
                    "rule": "cd_cn3_terminal_cl_plane",
                    "atom_id": terminal[0],
                    "center_atom_id": center,
                    "pass": pass_label,
                }
            )
            continue

        # --- Defensive: unbridged CN4 (or other) lattice residual ---
        if degree == 4 and n_t + len(fixed_neighbors) == 4 and n_br == 0:
            positions, assignment_residual = _cif_tetrahedral_terminal_positions(
                center=center,
                terminal_ligands=terminal,
                fixed_neighbors=fixed_neighbors,
                surface=surface,
                native=native,
                model=model,
                spec=spec,
                repulsive_neighbors=(),
            )
            if positions is None:
                unresolved.append(
                    f"Cd {center}: CIF tetrahedral assignment failed"
                )
            else:
                for ligand, position in positions.items():
                    surface[ligand] = position
                applied.append(
                    {
                        "rule": "cd_cn4_terminal_cl_cif_tetrahedral",
                        "atom_ids": sorted(positions),
                        "center_atom_id": center,
                        "assignment_residual": assignment_residual,
                        "repulsive_bridge_ligands": [],
                        "pass": pass_label,
                    }
                )
            continue

        # Bridged patterns outside the table (should be rare).
        if n_br >= 1 and n_t >= 1:
            unresolved.append(
                f"Cd {center}: no neighborhood-angle rule for "
                f"CN{degree} Se={n_se} Br={n_br} T={n_t}"
            )
        elif degree >= 4 and center in bridged_hosts:
            unresolved.append(
                f"Cd {center}: no movable terminal Cl for CN{degree} "
                "tetrahedral adjustment"
            )

    return applied, unresolved


def _final_ligand_geometry_pass(
    *,
    record: ClusterRecord,
    surface: FloatArray,
    native: FloatArray,
    bridge_ligands: set[int],
    bridged_hosts: set[int],
    model: _LatticeModel,
    spec: NucleationSpec,
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Always re-apply terminal placement after the primary terminal loop.

    Mirrors the cation CN2/CN3 enforcement step: preferred local ligand
    geometries win even when they leave construction-native virtual sites.
    Soft-clash joint refine remains a separate secondary step.
    """

    return _apply_terminal_ligand_surface_rules(
        record=record,
        surface=surface,
        native=native,
        bridge_ligands=bridge_ligands,
        bridged_hosts=bridged_hosts,
        model=model,
        spec=spec,
        pass_label="final",
    )


def _joint_refine_terminals_near_bridges(
    *,
    record: ClusterRecord,
    surface: FloatArray,
    native: FloatArray,
    bridge_ligands: set[int],
    model: _LatticeModel,
    spec: NucleationSpec,
) -> List[Dict[str, object]]:
    """Reassign terminal ligands that sit too close to any bridge ligand.

    Uses global bridge positions as repulsors (not only bridges on the same host)
    so a terminal next to a multi-cation rhombus can move to a free tetrahedral
    slot on its host.  Skipped when ``terminal_motifs`` is ``none`` (the refine
    path assumes zincblende-like CIF residual slots).  Returns applied-rule
    dicts for the surface audit.
    """

    motifs = str(getattr(spec, "terminal_motifs", "zb_mx2")).strip().lower()
    if motifs in {"", "none", "off", "false", "0"}:
        return []
    if not bridge_ligands:
        return []
    clash = max(0.75, 0.45 * model.bond_length)
    bridge_positions = [surface[i] for i in sorted(bridge_ligands)]
    applied: List[Dict[str, object]] = []
    cation_symbols = {spec.core.cation, spec.precursor.center}
    for center_atom in record.atoms:
        if center_atom.symbol not in cation_symbols:
            continue
        center = center_atom.atom_id
        neighbors = list(record.graph.neighbors(center))
        terminal = [
            index
            for index in neighbors
            if record.atoms[index].symbol == spec.precursor.ligand
            and index not in bridge_ligands
        ]
        if not terminal:
            continue
        # Only re-place if at least one terminal is near some bridge Cl.
        needs_refine = any(
            float(np.linalg.norm(surface[term] - bridge_pos)) < clash
            for term in terminal
            for bridge_pos in bridge_positions
        )
        if not needs_refine:
            continue
        fixed_neighbors = [
            index for index in neighbors if index not in terminal
        ]
        if len(terminal) + len(fixed_neighbors) > 4:
            continue
        # All bridge ligands on the structure act as repulsors; also prefer
        # away from bridges bonded to this Cd.
        local_bridges = [i for i in fixed_neighbors if i in bridge_ligands]
        repulsive = sorted(set(local_bridges) | set(bridge_ligands))
        positions, residual = _cif_tetrahedral_terminal_positions(
            center=center,
            terminal_ligands=terminal,
            fixed_neighbors=fixed_neighbors,
            surface=surface,
            native=native,
            model=model,
            spec=spec,
            repulsive_neighbors=repulsive,
        )
        if positions is None:
            continue
        # Accept only if the worst terminal–bridge distance improves.
        old_min = min(
            float(np.linalg.norm(surface[term] - bridge_pos))
            for term in terminal
            for bridge_pos in bridge_positions
        )
        new_min = min(
            float(np.linalg.norm(positions[term] - bridge_pos))
            for term in terminal
            for bridge_pos in bridge_positions
        )
        if new_min + 1.0e-6 < old_min:
            continue
        for ligand, position in positions.items():
            surface[ligand] = position
        applied.append(
            {
                "rule": "joint_terminal_away_from_bridges",
                "center_atom_id": center,
                "atom_ids": sorted(terminal),
                "repulsive_bridge_ligands": repulsive,
                "assignment_residual": residual,
                "min_terminal_bridge_distance_before": round(old_min, 6),
                "min_terminal_bridge_distance_after": round(new_min, 6),
            }
        )
    return applied


def _symmetric_bridge_position(
    first_cd: FloatArray,
    second_cd: FloatArray,
    shared_se: FloatArray,
    angle_deg: float,
    native_cl: FloatArray,
) -> Optional[FloatArray]:
    """Place Cl opposite Se in the Cd-Se-Cd plane at the requested angle."""

    base = second_cd - first_cd
    base_unit = _unit(base)
    if base_unit is None:
        return None
    midpoint = 0.5 * (first_cd + second_cd)
    toward_se = shared_se - midpoint
    toward_se -= np.dot(toward_se, base_unit) * base_unit
    plane_direction = _unit(toward_se)
    if plane_direction is None:
        native_direction = native_cl - midpoint
        native_direction -= np.dot(native_direction, base_unit) * base_unit
        plane_direction = _unit(native_direction)
    if plane_direction is None:
        return None
    half_angle = math.radians(angle_deg) / 2.0
    height = 0.5 * float(np.linalg.norm(base)) / math.tan(half_angle)
    return midpoint - height * plane_direction


def _cif_tetrahedral_terminal_positions(
    *,
    center: int,
    terminal_ligands: Sequence[int],
    fixed_neighbors: Sequence[int],
    surface: FloatArray,
    native: FloatArray,
    model: _LatticeModel,
    spec: NucleationSpec,
    repulsive_neighbors: Sequence[int] = (),
) -> Tuple[Optional[Dict[int, FloatArray]], float]:
    """Assign terminal Cl to the unoccupied CIF tetrahedral directions.

    When ``repulsive_neighbors`` is set (typically bridge Cl on this Cd), the
    ranking is:

    1. low residual of fixed bonds on the CIF tetrahedron,
    2. **maximize** the closest approach of any terminal Cl to any repulsive
       neighbor (prefer the slot opposite a rhombic/CIF bridge),
    3. only then minimize native-coordinate displacement (weak tie-break).

    Without repulsors, behaviour is historical (residual, then native).
    """

    if (
        not terminal_ligands
        or len(terminal_ligands) + len(fixed_neighbors) > 4
    ):
        return None, math.inf
    fixed_directions = [
        _unit(surface[neighbor] - surface[center])
        for neighbor in fixed_neighbors
    ]
    if any(direction is None for direction in fixed_directions):
        return None, math.inf

    repulsive_positions = [
        surface[index]
        for index in repulsive_neighbors
        if 0 <= int(index) < len(surface)
    ]
    # Lexicographic key: lower is better.  Separation is stored negated so
    # larger min distance ranks first.
    best_key: Optional[
        Tuple[float, float, float, int, Tuple[int, ...], Tuple[int, ...]]
    ] = None
    best_positions: Optional[Dict[int, FloatArray]] = None
    for environment_index, environment in enumerate(
        model.environments[spec.core.cation]
    ):
        ideal_vectors = [np.asarray(vector, dtype=float) for vector in environment]
        ideal_directions = [_unit(vector) for vector in ideal_vectors]
        if any(direction is None for direction in ideal_directions):
            continue
        for fixed_slots in permutations(range(4), len(fixed_neighbors)):
            residual = sum(
                1.0 - float(np.dot(direction, ideal_directions[slot]))
                for direction, slot in zip(fixed_directions, fixed_slots)
                if direction is not None and ideal_directions[slot] is not None
            )
            remaining_slots = tuple(
                slot for slot in range(4) if slot not in fixed_slots
            )
            for terminal_slots in permutations(
                remaining_slots, len(terminal_ligands)
            ):
                positions = {
                    ligand: surface[center]
                    + model.bond_length * ideal_directions[slot]
                    for ligand, slot in zip(terminal_ligands, terminal_slots)
                    if ideal_directions[slot] is not None
                }
                if repulsive_positions:
                    min_sep = min(
                        float(np.linalg.norm(positions[ligand] - rep))
                        for ligand in terminal_ligands
                        for rep in repulsive_positions
                    )
                else:
                    min_sep = 0.0
                native_displacement = sum(
                    float(np.sum((positions[ligand] - native[ligand]) ** 2))
                    for ligand in terminal_ligands
                )
                key = (
                    residual,
                    -min_sep,
                    native_displacement,
                    environment_index,
                    fixed_slots,
                    terminal_slots,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_positions = positions
    return best_positions, math.inf if best_key is None else best_key[0]


def _sigma_d_cl_pair(
    center: int,
    se_neighbor: int,
    ligands: Sequence[int],
    native: FloatArray,
    surface: FloatArray,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> Optional[Tuple[FloatArray, FloatArray]]:
    """Choose the tetrahedral sigma-d plane requiring least Cl movement."""

    se_direction = _unit(surface[se_neighbor] - surface[center])
    if se_direction is None:
        return None
    best: Optional[Tuple[float, Tuple[FloatArray, FloatArray]]] = None
    for environment in model.environments[spec.core.cation]:
        ideal = [np.asarray(vector, dtype=float) for vector in environment]
        se_index = max(
            range(len(ideal)),
            key=lambda index: float(
                np.dot(se_direction, ideal[index] / np.linalg.norm(ideal[index]))
            ),
        )
        for index, vector in enumerate(ideal):
            if index == se_index:
                continue
            normal = _unit(np.cross(se_direction, vector))
            if normal is None:
                continue
            in_plane = _unit(np.cross(normal, se_direction))
            if in_plane is None:
                continue
            directions = (
                -0.5 * se_direction + (math.sqrt(3.0) / 2.0) * in_plane,
                -0.5 * se_direction - (math.sqrt(3.0) / 2.0) * in_plane,
            )
            positions = tuple(
                surface[center] + model.bond_length * direction
                for direction in directions
            )
            for ordered in (positions, (positions[1], positions[0])):
                cost = sum(
                    float(np.sum((position - native[ligand]) ** 2))
                    for ligand, position in zip(ligands, ordered)
                )
                if best is None or cost < best[0]:
                    best = (cost, ordered)
    return None if best is None else best[1]


def _outward_planar_bisector(
    center: FloatArray,
    first_neighbor: FloatArray,
    second_neighbor: FloatArray,
    native_ligand: FloatArray,
) -> Optional[FloatArray]:
    first = _unit(first_neighbor - center)
    second = _unit(second_neighbor - center)
    if first is None or second is None:
        return None
    outward = _unit(-(first + second))
    if outward is not None:
        return outward
    native_direction = native_ligand - center
    normal = _unit(np.cross(first, native_direction))
    if normal is None:
        return _unit(native_direction)
    projected = native_direction - np.dot(native_direction, normal) * normal
    return _unit(projected)


def _surface_conflicts(
    record: ClusterRecord,
    surface: FloatArray,
    spec: NucleationSpec,
    *,
    clash_radius: Optional[float] = None,
) -> Tuple[List[List[int]], List[Dict[str, object]]]:
    """Return the two hard surface faults: collisions and saturated intrusions.

    A saturated cation may not have a chemically compatible non-neighbour closer
    than its own farthest bond -- if it does, the projected geometry contradicts
    the graph and the structure fails the surface gate.

    Collisions flag near-coincident atoms on the **surface** view (same virtual
    slot).  Construction-native continuous decoration uses a stricter soft-clash
    radius separately; the surface gate must stay mild enough not to reject
    valid projected bridge/terminal geometries that baselines lock.

    Deliberately narrow on *intrusions*: it fires only for cations already at
    ``max_cn``.  A rhombic bridge puts its ligand about 3.07 A from each host,
    outside the bond shell, so widening intrusion checks is left alone until DFT
    demands otherwise; read bundles through the graph, not a cutoff.

    This used to exist as two independently maintained copies, one in the gate
    branch of ``_precondition_surface_geometry`` and one in
    ``_surface_geometry_metadata``, written with opposite comparison senses.
    They agreed, but nothing kept them agreeing.
    """

    count = len(record.atoms)
    distances = _pair_distances(surface)
    # Near-coincident only (~same site).  Default 0.75 Å catches stacked Cl
    # without failing normal terminal/bridge separations.
    if clash_radius is None:
        clash_radius = 0.75
    bonded = set()
    for left, right in record.graph.edges:
        bonded.add((min(left, right), max(left, right)))
    collisions: List[List[int]] = []
    for left, right in zip(*np.nonzero(np.triu(distances < clash_radius, 1))):
        pair = (int(left), int(right))
        if pair in bonded:
            if float(distances[left, right]) < 1.0e-6:
                collisions.append([pair[0], pair[1]])
            continue
        collisions.append([pair[0], pair[1]])

    allowed = _allowed_pair_matrix(
        [atom.symbol for atom in record.atoms], spec
    )
    intrusions: List[Dict[str, object]] = []
    for atom in record.atoms:
        center = atom.atom_id
        if (
            atom.symbol != spec.core.cation
            or record.graph.degree[center]
            != spec.graph_rules.max_cn[atom.symbol]
        ):
            continue
        bonded = list(record.graph.neighbors(center))
        farthest_bonded = float(np.max(distances[center, bonded]))
        excluded = np.zeros(count, dtype=bool)
        excluded[center] = True
        excluded[bonded] = True
        intruding = np.nonzero(
            allowed[center]
            & ~excluded
            & (distances[center] + 1.0e-8 < farthest_bonded)
        )[0]
        for neighbor in intruding:
            intrusions.append(
                {
                    "cd_atom_id": center,
                    "intruding_atom_id": int(neighbor),
                    "intruding_element": record.atoms[neighbor].symbol,
                    "distance_angstrom": float(distances[center, neighbor]),
                    "farthest_graph_neighbor_distance_angstrom": farthest_bonded,
                }
            )
    return collisions, intrusions


def _pauling_valences(
    record: ClusterRecord, spec: NucleationSpec
) -> Dict[str, object]:
    """Per-cation Pauling electrostatic valence sums -- reporting only.

    ``V(cation) = sum over bonded anions of |q(anion)| / CN(anion)``, so a
    cation is satisfied at ``V == |q(cation)|`` (2 for Cd here), oversaturated
    above it and undersaturated below.  Purely a function of the graph and the
    declared formal charges, so it costs one pass and needs no new input.

    **This does not rank anything, and the evidence says it should not.**  It
    looked promising -- across three k=2 p=3 relaxations the cation that shed a
    ligand was the most oversaturated one -- but two later checks undercut both
    halves of that:

    * It does not order isomers.  Relaxing the fourth k=2 p=3 member showed the
      one with the *highest* ``max V`` (2.67) is the *lowest* in energy, and at
      k=3 p=2 a 2.67-vs-2.17 pair came out 0.57 kcal/mol apart, inside DFT
      error.
    * The per-atom "which cation sheds" correlation is not reading-independent.
      Because a rhombic bridge sits ~3.07 A from its hosts (see
      ``_surface_conflicts``), a relaxed geometry admits several equally valid
      ``max_cn`` readings, and which cation counts as having shed a ligand --
      hence whether it was the ``max V`` one -- changes with the choice.

    So it is emitted purely to accumulate evidence.  Any future move to score on
    it needs a reading-independent formulation first.
    """

    charges = spec.charges
    valences: List[Dict[str, object]] = []
    for atom in record.atoms:
        charge = charges.get(atom.symbol)
        if charge is None or charge <= 0:
            continue
        total = 0.0
        for neighbor in record.graph.neighbors(atom.atom_id):
            anion = record.atoms[neighbor]
            anion_charge = charges.get(anion.symbol)
            anion_cn = record.graph.degree[neighbor]
            if anion_charge is None or anion_charge >= 0 or anion_cn == 0:
                continue
            total += abs(anion_charge) / anion_cn
        valences.append(
            {
                "atom_id": atom.atom_id,
                "symbol": atom.symbol,
                "target": float(charge),
                "pauling_valence": round(total, 6),
                "deviation": round(total - float(charge), 6),
            }
        )
    return {
        "cations": valences,
        "max_pauling_valence": (
            round(max(item["pauling_valence"] for item in valences), 6)
            if valences
            else None
        ),
        "max_abs_deviation": (
            round(max(abs(item["deviation"]) for item in valences), 6)
            if valences
            else None
        ),
    }


def _surface_geometry_metadata(
    record: ClusterRecord,
    native: FloatArray,
    surface: FloatArray,
    spec: NucleationSpec,
    *,
    applied_rules: Sequence[Mapping[str, object]],
    unresolved_conflicts: Sequence[str],
) -> Dict[str, object]:
    """Summarize local native/surface angles for JSON and the audit log."""

    environments: List[Dict[str, object]] = []
    angle_changes: List[float] = []
    for atom in record.atoms:
        neighbors = list(record.graph.neighbors(atom.atom_id))
        degree = len(neighbors)
        template = spec.geometry_rules.template_for(atom.symbol, degree)
        if degree < 2 or template is None:
            continue
        native_angles = _neighbor_angles(native, atom.atom_id, neighbors)
        surface_angles = _neighbor_angles(surface, atom.atom_id, neighbors)
        angle_changes.extend(
            abs(after - before)
            for before, after in zip(native_angles, surface_angles)
        )
        neighbor_symbols = sorted(record.atoms[index].symbol for index in neighbors)
        environments.append(
            {
                "atom_id": atom.atom_id,
                "symbol": atom.symbol,
                "cn": degree,
                "neighbors": neighbor_symbols,
                "template": template,
                "native_angles_deg": native_angles,
                "surface_angles_deg": surface_angles,
            }
        )
    edge_errors = [
        abs(
            float(np.linalg.norm(surface[left] - surface[right]))
            - float(np.linalg.norm(native[left] - native[right]))
        )
        for left, right in record.graph.edges
    ]
    displacements = np.linalg.norm(surface - native, axis=1)
    bridge_rules_by_ligand = {
        int(rule["atom_id"]): rule
        for rule in applied_rules
        if rule.get("rule") in {
            "symmetric_cl_bridge",
            "shared_vacant_cif_site_bridge",
        }
        and isinstance(rule.get("atom_id"), int)
    }
    bridge_geometry: List[Dict[str, object]] = []
    for left, right, data in record.graph.edges(data=True):
        if data.get("kind") != "surface_bridge":
            continue
        ligand = (
            left
            if record.atoms[left].symbol == spec.precursor.ligand
            else right
        )
        second_host = right if ligand == left else left
        primary_hosts = [
            neighbor
            for neighbor in record.graph.neighbors(ligand)
            if neighbor != second_host
            and record.atoms[neighbor].symbol == spec.core.cation
        ]
        shared = data.get("shared_neighbor")
        if len(primary_hosts) != 1:
            continue
        primary = primary_hosts[0]
        angle = _neighbor_angles(
            surface, ligand, [primary, second_host]
        )[0]
        if isinstance(shared, int):
            plane_normal = _unit(
                np.cross(
                    surface[primary] - surface[shared],
                    surface[second_host] - surface[shared],
                )
            )
            plane_distance: Optional[float] = (
                0.0
                if plane_normal is None
                else abs(
                    float(
                        np.dot(
                            surface[ligand] - surface[shared], plane_normal
                        )
                    )
                )
            )
        else:
            plane_distance = None
        bridge_mode = str(
            data.get("bridge_mode", "shared_occupied_neighbor")
        )
        bridge_geometry.append(
            {
                "ligand_atom_id": ligand,
                "host_atom_ids": [primary, second_host],
                "bridge_mode": bridge_mode,
                "shared_neighbor_atom_id": shared,
                "virtual_site_position": data.get("virtual_site"),
                "target_angle_deg": data.get("surface_angle_deg"),
                "surface_angle_deg": angle,
                "surface_cd_ligand_distances_angstrom": [
                    float(np.linalg.norm(surface[ligand] - surface[primary])),
                    float(
                        np.linalg.norm(surface[ligand] - surface[second_host])
                    ),
                ],
                "plane_distance_angstrom": plane_distance,
                "out_of_plane_rotation_deg": float(
                    bridge_rules_by_ligand.get(ligand, {}).get(
                        "out_of_plane_rotation_deg", 0.0
                    )
                ),
                "primary_cn_before_bridge": data.get(
                    "primary_cn_before_bridge"
                ),
                "secondary_cn_before_bridge": data.get(
                    "secondary_cn_before_bridge"
                ),
            }
        )
    collisions, saturated_cd_intrusions = _surface_conflicts(
        record, surface, spec, clash_radius=0.75
    )
    cn4_tetrahedral_rms: List[Dict[str, object]] = []
    tetrahedral_target = 109.47122063449069
    for atom in record.atoms:
        if (
            atom.symbol != spec.core.cation
            or record.graph.degree[atom.atom_id] != 4
        ):
            continue
        angles = _neighbor_angles(
            surface, atom.atom_id, list(record.graph.neighbors(atom.atom_id))
        )
        cn4_tetrahedral_rms.append(
            {
                "atom_id": atom.atom_id,
                "angular_rms_deg": float(
                    math.sqrt(
                        sum(
                            (angle - tetrahedral_target) ** 2
                            for angle in angles
                        )
                        / len(angles)
                    )
                ),
            }
        )
    projection_valid = not collisions and not saturated_cd_intrusions
    return {
        "mode": "retained_only_final_cn_geometry_projection",
        "converged": projection_valid,
        "projection_valid": projection_valid,
        "message": (
            "surface_projection_unresolved"
            if not projection_valid
            else (
                "final_cn_geometry_projection_applied"
                if applied_rules
                else "no_applicable_projection"
            )
        ),
        "evaluations": 0,
        "graph_preserved": True,
        "objective_rms": 0.0,
        "bond_length_rms_change_angstrom": float(
            np.sqrt(np.mean(np.square(edge_errors))) if edge_errors else 0.0
        ),
        "max_displacement_angstrom": float(
            np.max(displacements) if len(displacements) else 0.0
        ),
        "max_angle_change_deg": max(angle_changes, default=0.0),
        "coordinates_changed": bool(np.max(displacements) > 1.0e-10),
        "applied_rules": [dict(rule) for rule in applied_rules],
        "unresolved_conflicts": list(unresolved_conflicts),
        "bridge_geometry": bridge_geometry,
        "coordinate_collisions": collisions,
        "saturated_cd_intrusions": saturated_cd_intrusions,
        "cn4_tetrahedral_rms": cn4_tetrahedral_rms,
        # Reporting only -- never consulted by scoring or pruning.
        "pauling_valence": _pauling_valences(record, spec),
        "hard_surface_constraint": (
            "final graph Cd CN must not exceed max CN; when Cd is saturated, "
            "its graph neighbors must be its nearest chemically compatible "
            "neighbors"
        ),
        "environments": environments,
    }


def _neighbor_angles(
    coordinates: FloatArray,
    center: int,
    neighbors: Sequence[int],
) -> List[float]:
    """Return deterministic pair angles around one center in degrees."""

    vectors = [
        coordinates[neighbor] - coordinates[center] for neighbor in neighbors
    ]
    angles: List[float] = []
    for left, right in combinations(vectors, 2):
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        cosine = float(np.dot(left, right)) / max(denominator, 1.0e-12)
        angles.append(
            float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        )
    return [round(angle, 8) for angle in angles]

__all__ = [
    '_precondition_retained_registry',
    '_attach_surface_geometry',
    '_precondition_surface_geometry',
    '_unit',
    '_project_point_to_plane',
    '_terminal_position_two_fixed_bisector',
    '_terminal_pair_c2v_two_fixed',
    '_apply_terminal_ligand_surface_rules',
    '_final_ligand_geometry_pass',
    '_joint_refine_terminals_near_bridges',
    '_symmetric_bridge_position',
    '_cif_tetrahedral_terminal_positions',
    '_sigma_d_cl_pair',
    '_outward_planar_bisector',
    '_surface_conflicts',
    '_pauling_valences',
    '_surface_geometry_metadata',
    '_neighbor_angles',
    '_TETRAHEDRAL_PAIR_HALF_ANGLE',
    '_terminal_position_two_bridge_bisector',
]
