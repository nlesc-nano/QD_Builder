"""Lattice-free molecular isomer enumeration and embedding.

Builds ``(k, p)`` CdSe/CdCl2-like molecules from graph rules + a geometry pack,
without CIF virtual sites. Hard filters come from ``molecular_rules``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from itertools import combinations, permutations, product
from math import sqrt
import os
from pathlib import Path
import re
import time
from typing import (
    Callable,
    Dict,
    Iterable,
    Mapping,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import networkx as nx
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, lil_matrix

from ..graph_canon import canonical_form
from .xtb_relax import XtbSettings, relax_structures
from ..io_utils import write_xyz
from ..nc_types import NucleationSpec
from .geometry_pack import GeometryPack, load_geometry_pack
from .molecular_rules import (
    allowed_bond_pairs,
    inorganic_component_count,
    molecular_geometry_ok,
    molecular_graph_ok,
    molecular_graph_violations,
    pair_key,
    ring_size_violations,
)
from .molecular_motifs import coordination_motif_inventory
from .types import AtomRecord, ProgressCallback, _State

FloatArray = np.ndarray

# NOTE: ``EXACT_BOND_TOLERANCE`` and ``EXACT_ANGLE_TOLERANCE_DEG`` are *not*
# acceptance criteria.  They are the numerical epsilon that guards
# sphere-intersection discriminants during construction (``z_squared <
# -(EXACT_BOND_TOLERANCE**2)`` and friends, here and in molecular_sites /
# molecular_tet_sites / molecular_bridge_first) and the tightness the
# constructor aims for when it picks a terminal direction.  Raising them would
# silently change which bridges are considered geometrically constructible.
# Whether a *finished* molecule is accepted is decided by the AUDIT_* band
# below.
EXACT_BOND_TOLERANCE = 1.0e-3
# DFT-median Cd--Cl distances are local averages, while a bridge has to fit a
# triangle/tetrahedron of already-built hosts.  This bounded construction
# tolerance is used only by the deterministic bridge fallback; all inorganic
# bonds and all hard contact floors retain the exact checks above.
RELAXED_BRIDGE_BOND_TOLERANCE_A = 0.12
EXACT_ANGLE_TOLERANCE_DEG = 1.0e-5
# --- Acceptance band for a finished molecule -------------------------------
# A constructed structure has to be *reasonable*, not exact.  The constructor
# assembles a molecule from independent sources -- a rigid chair/boat ring
# template, CN-dependent bond tables, and free tetrahedral directions -- which
# cannot be mutually satisfied to machine precision.  The 6-ring template pins
# every ring bond at ``bond_cdse_A`` (2.69 A for the CdSe pack) while the CN
# table gives 2.649 A for Cd(cn4)-Se(cn3): a 0.041 A disagreement that is a
# property of the tables, not of any particular molecule.  Auditing that at
# 1e-3 A rejected every ring structure at k=4,p=3.
#
# These defaults follow the practice of established 3D structure generators
# (RDKit/ETKDG + force-field cleanup, CORINA, OMEGA), which validate geometry
# against roughly two standard deviations of the crystallographic distribution
# rather than against equality.  The angle band also matches the spread this
# project's own DFT tables record: the pack's ``by_role_signature`` entries
# carry p10/p90 windows 10-20 deg wide around each median.
#
# Packs may override any of these under a ``tolerances:`` block.
AUDIT_BOND_TOLERANCE_A = 0.05
AUDIT_ANGLE_TOLERANCE_DEG = 8.0
AUDIT_IMPROPER_TOLERANCE_DEG = 15.0

# --- Optimizer well shape -----------------------------------------------
# Restraints are flat-bottomed: zero inside [target-tol, target+tol], quadratic
# on the excess outside it.  A point target pulls even when the value is
# perfectly acceptable, so satisfying one restraint drags the others off theirs;
# with a band, only genuine violations exert force.  Scales below set the
# stiffness applied to that excess (k = 1/scale^2).
BOND_WELL_SCALE_A = 0.010
#: The optimizer's flat bottom is this fraction of the *audit* band, so it aims
#: inside the acceptance window rather than at its edge.  Measured at k=4,p=3:
#: with the full band (1.0) the fit parks bonds exactly on the threshold and
#: nothing survives the audit -- 0 accepted; at 0.5 it is 23.  Optimising to a
#: tighter tolerance than you validate against is the standard arrangement.
WELL_BAND_FRACTION = 0.5
#: Non-bonded repulsion is deliberately stiffer than bonding -- k_rep here is
#: (0.010/0.0071)^2 = 2x k_bond, in the 2-5x range that keeps an optimizer from
#: pushing atoms through one another to satisfy the covalent network.  It used
#: to be 0.25x, i.e. four times *weaker*, and contacts were the largest single
#: rejection category.
REPULSION_WELL_SCALE_A = 0.0071
ANGLE_WELL_SCALE_DEG = 4.0


def _band_excess(value: FloatArray, target: FloatArray, tol: FloatArray) -> FloatArray:
    """Signed distance outside ``target +/- tol``; zero inside the band."""

    deviation = value - target
    return np.where(
        deviation > tol,
        deviation - tol,
        np.where(deviation < -tol, deviation + tol, 0.0),
    )


#: Positional tether on the seed-ring atoms during the closure polish, in
#: angstrom.  The ring is restrained rather than frozen so that bonds from a
#: ring atom to the rest of the molecule can actually be met; weaker than the
#: bond term (0.01) so bonds win, strong enough that the chair does not drift.
RING_SEED_TETHER_A = 0.05

#: Worst single centre-angle error ``terminal_direction`` may leave behind
#: before declaring a terminal unplaceable.  Deliberately looser than
#: AUDIT_ANGLE_TOLERANCE_DEG: construction only has to get close enough for the
#: bounded repair to finish the job, and the audit still decides acceptance.
TERMINAL_DIRECTION_MAX_ANGLE_ERROR_DEG = 25.0

# A local repair is anchored to the deterministic frame.  If its initial
# normalized bond/contact residual is already this large, the frame is not a
# nearby repair and spending hundreds of numerical residual evaluations cannot
# produce an accepted molecule.
RELAX_INITIAL_COST_LIMIT = 1.0e8


def _molecular_relax_trace_enabled() -> bool:
    """Whether nonlinear geometry iterations should be printed.

    Relaxation is deliberately quiet in normal map generation because a
    single (k,p) bin can try many decorations.  Setting
    ``QD_MOLECULAR_RELAX_TRACE=1`` enables SciPy's per-iteration cost table
    plus one labelled start/end line for each local repair.
    """

    return os.environ.get("QD_MOLECULAR_RELAX_TRACE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


class ExactEmbeddingError(ValueError):
    """The fixed construction rules cannot realize a molecular graph."""

    def __init__(
        self,
        reasons: Sequence[str],
        *,
        coordinates: Optional[Sequence[Sequence[float]]] = None,
    ):
        self.reasons = tuple(str(reason) for reason in reasons)
        self.coordinates = (
            np.asarray(coordinates, dtype=float).copy()
            if coordinates is not None
            else None
        )
        super().__init__("; ".join(self.reasons))


class EnumerationLimitError(RuntimeError):
    """Exact enumeration reached a configured safety limit."""


# Distance (Å) used to flag a terminal Cl as "closable" onto a nonbonded Cd2.
# Slightly above the Cd–Cl bond cutoff so near-misses that DFT may bridge are
# counted; not a hard contact floor.
CLOSABLE_TERMINAL_CD2_A = 3.50


@dataclass(frozen=True)
class MolecularCollapseAnnotations:
    """Graph/geometry features that predict DFT bridge-closure risk.

    These do **not** change acceptance.  They label construction isomers so a
    later start→final graph comparison can promote YAML rules with evidence.
    """

    n_cd2: int
    n_cd3: int
    n_cd4: int
    mean_cd_cn: float
    n_terminal_cl: int
    n_mu2_cl: int
    n_mu3_cl: int
    #: terminal Cl on H + other Cd U with free valence (graph only)
    n_unsaturated_bridge_candidates: int
    #: terminal Cl within CLOSABLE_TERMINAL_CD2_A of a nonbonded Cd CN2
    n_closable_terminal_cd2: int
    #: Cl2Se1 host (two terminal Cl) with a closable contact to some Cd2
    n_cl2se1_near_cd2: int
    #: compact multiset of Cd–Se bonds as "cdCN-seCN:count,..."
    cdse_cn_pairs: str
    #: compact multiset of Cd CN2 neighbor signatures
    cd2_signatures: str

    def as_csv_row(self) -> Dict[str, object]:
        return {
            "n_cd2": self.n_cd2,
            "n_cd3": self.n_cd3,
            "n_cd4": self.n_cd4,
            "mean_cd_cn": f"{self.mean_cd_cn:.4f}",
            "n_terminal_cl": self.n_terminal_cl,
            "n_mu2_cl": self.n_mu2_cl,
            "n_mu3_cl": self.n_mu3_cl,
            "n_unsaturated_bridge_candidates": (
                self.n_unsaturated_bridge_candidates
            ),
            "n_closable_terminal_cd2": self.n_closable_terminal_cd2,
            "n_cl2se1_near_cd2": self.n_cl2se1_near_cd2,
            "cdse_cn_pairs": self.cdse_cn_pairs,
            "cd2_signatures": self.cd2_signatures,
        }


@dataclass(frozen=True)
class MolecularIsomer:
    """One unique molecular graph with optional embedded coordinates."""

    k: int
    p: int
    structure_id: str
    certificate: Tuple[object, ...]
    atoms: Tuple[AtomRecord, ...]
    graph: nx.Graph
    coordinates: Optional[Tuple[Tuple[float, float, float], ...]] = None
    #: Non-empty values are diagnostic audit warnings.  In the motif+xTB path
    #: an endpoint with coordinates is retained even when this post-relaxation
    #: audit reports a violation; an xTB failure with no coordinates is still
    #: not an isomer.
    violations: Tuple[str, ...] = ()
    annotations: Optional[MolecularCollapseAnnotations] = None
    #: Named final conformers. Ring graphs require chair; boat is optional.
    conformers: Tuple[
        Tuple[str, Tuple[Tuple[float, float, float], ...]], ...
    ] = ()
    #: Graph-only coordination motif counts, retained for downstream analysis.
    motif_inventory: Tuple[Tuple[str, int], ...] = ()
    #: GFN-xTB relaxation of the constructed geometry (unset when disabled).
    #: ``xtb_energy_eV`` is the first quantity that ranks isomers on physics
    #: rather than on a bond-count proxy.
    xtb_energy_eV: Optional[float] = None
    xtb_gap_eV: Optional[float] = None
    xtb_steps: int = 0
    xtb_converged: bool = False
    #: how many atom pairs changed bonded/not-bonded status during relaxation;
    #: nonzero means the relaxed structure is no longer the enumerated isomer
    xtb_connectivity_changed: int = 0
    xtb_coordinates: Optional[Tuple[Tuple[float, float, float], ...]] = None
    #: graph implied by the relaxed coordinates, using the pack's own bond
    #: distances -- so "did it stay the same molecule" is judged by the same
    #: criterion the enumerator used, not by a second bond-order convention
    xtb_relaxed_bonds: int = 0
    xtb_bonds_delta: int = 0
    xtb_relaxed_cl_motifs: Tuple[int, int, int] = (0, 0, 0)
    xtb_same_topology: bool = True
    #: structure_id of an *enumerated* isomer the relaxed graph turned into,
    #: empty when it is not one we generated
    xtb_matches: str = ""
    xtb_error: str = ""
    #: Motif-factor/xTB discovery provenance.  Empty means the ordinary
    #: deterministic construction path; a non-empty certificate identifies
    #: the graph whose relaxed start produced this structure.
    discovered_from: str = ""
    reconstruction_start: int = -1
    xtb_bond_orders: Optional[Tuple[Tuple[float, ...], ...]] = None
    source_edges: Tuple[Tuple[int, int], ...] = ()

    @property
    def symbols(self) -> List[str]:
        return [atom.symbol for atom in self.atoms]


@dataclass
class MolecularSkeletonRecord:
    """One Cd–Se skeleton for a (k, p) bin (before Cl decoration)."""

    skeleton_index: int
    n_edges: int
    cd_cn: Tuple[int, ...]  # sorted skeleton degrees of all Cd
    se_cn: Tuple[int, ...]  # sorted skeleton degrees of all Se
    status: str  # accepted | skipped_graph | skipped_frame
    reason: str = ""
    edges: Tuple[Tuple[int, int], ...] = ()
    coordinates: Optional[Tuple[Tuple[float, float, float], ...]] = None
    symbols: Tuple[str, ...] = ()  # only inorganic atoms for xyz dump
    #: Atom-id sets of the six-rings imposed by ring-first/fused construction.
    forced_rings: Tuple[Tuple[int, ...], ...] = ()


@dataclass
class MolecularFailureRecord:
    """Aggregated diagnostic for one representative reconstruction failure."""

    skeleton_index: int
    cd_cn: Tuple[int, ...]
    stage: str
    reason: str
    count: int = 0
    snapshot_kind: str = ""
    snapshot_symbols: Tuple[str, ...] = ()
    snapshot_coordinates: Optional[Tuple[Tuple[float, float, float], ...]] = None


@dataclass
class MolecularMotifTrial:
    """One inspectable motif-factor start and its optional xTB endpoint."""

    trial_id: str
    start_index: int
    symbols: Tuple[str, ...]
    source_edges: Tuple[Tuple[int, int], ...]
    initial_coordinates: Tuple[Tuple[float, float, float], ...]
    initial_violations: Tuple[str, ...] = ()
    xtb_ok: bool = False
    xtb_converged: bool = False
    xtb_energy_eV: Optional[float] = None
    xtb_error: str = ""
    xtb_coordinates: Optional[Tuple[Tuple[float, float, float], ...]] = None
    final_edges: Tuple[Tuple[int, int], ...] = ()
    final_violations: Tuple[str, ...] = ()


@dataclass
class MolecularBinResult:
    k: int
    p: int
    isomers: List[MolecularIsomer] = field(default_factory=list)
    raw_graphs: int = 0
    rejected: int = 0
    unique_graphs: int = 0
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    # Full rejection strings, including atom ids and measured distances when
    # the checker provides them.  ``rejection_reasons`` intentionally keeps
    # the compact category counts used by the existing summary CSV.
    rejection_details: Counter[str] = field(default_factory=Counter)
    failure_records: Dict[Tuple[object, ...], MolecularFailureRecord] = field(
        default_factory=dict
    )
    #: budgeted-embed bookkeeping (0 when the bin ran exhaustively)
    budget_pool: int = 0
    budget_embedded: int = 0
    ring_min_pattern_cd: Tuple[int, ...] = ()
    ring_min_pattern_se: Tuple[int, ...] = ()
    geometry_ring_pattern_cd: Tuple[int, ...] = ()
    geometry_ring_pattern_se: Tuple[int, ...] = ()
    incomplete: bool = False
    # Where candidates went *before* ``raw_graphs`` ever counted them.  Without
    # this the log can only show what survived long enough to be rejected.
    skeletons_total: int = 0
    skeletons_pruned_graph: int = 0
    skeletons_pruned_frame: int = 0
    modes_total: int = 0
    modes_kept: int = 0
    symmetry_pruned: int = 0
    revisited: int = 0
    infeasible_partials: int = 0
    over_capacity: int = 0
    geometry_pruned: int = 0
    frames_built: int = 0
    screened_before_embed: int = 0
    embedded: int = 0
    chair_refinements: int = 0
    boat_refinements: int = 0
    ring_refinement_failures: int = 0
    # Nonlinear repair diagnostics.  These are aggregate counters so normal
    # runs remain compact; per-iteration costs are available with
    # QD_MOLECULAR_RELAX_TRACE=1.
    optimizer_attempts: int = 0
    optimizer_successes: int = 0
    optimizer_nfev: int = 0
    motif_graphs_eligible: int = 0
    motif_reconstruction_attempts: int = 0
    motif_reconstruction_candidates: int = 0
    motif_pre_xtb_accepted: int = 0
    motif_xtb_attempts: int = 0
    motif_xtb_converged: int = 0
    motif_xtb_same_graph_rescues: int = 0
    motif_xtb_discovered: int = 0
    #: xTB endpoints that landed on a graph already represented by another
    #: enumerated source graph: (source structure, destination structure, detail).
    xtb_merge_records: List[Tuple[str, str, str]] = field(default_factory=list)
    #: source graphs removed by graph-certificate deduplication before xTB.
    graph_merge_records: List[Tuple[str, str, str]] = field(default_factory=list)
    motif_trials: List[MolecularMotifTrial] = field(default_factory=list)
    #: every enumerated skeleton (kept and skipped) with Cd/Se CN lists
    skeleton_records: List[MolecularSkeletonRecord] = field(default_factory=list)
    #: skeleton construction path: free | ring_first | fused2 | free_fallback | precomputed
    skeleton_mode_used: str = ""
    #: True if a ring-first (level≥1) pass produced ≥1 accepted isomer
    ring_first_proved: bool = False
    #: Highest structure level that produced ≥1 isomer (0=free, 1=1-ring, 2=fused2)
    proved_level: int = 0
    skeleton_generation_time_s: float = 0.0
    decoration_stream_time_s: float = 0.0
    decoration_generation_time_s: float = 0.0
    candidate_screen_time_s: float = 0.0

    def prefilter_summary(self) -> str:
        """One line describing everything dropped before enumeration."""

        return (
            f"skeletons {self.skeletons_total}"
            f"->{self.skeletons_total - self.skeletons_pruned_graph - self.skeletons_pruned_frame}"
            f" (graph rules {self.skeletons_pruned_graph}, "
            f"dead frame {self.skeletons_pruned_frame})"
            f" | host sets {self.modes_total}->{self.modes_kept}"
            f" | frames_built {self.frames_built}"
            f" | fused closure fits chair/boat {self.chair_refinements}/"
            f"{self.boat_refinements}"
            f" | partials pruned: symmetry {self.symmetry_pruned}, "
            f"revisited {self.revisited}, "
            f"coordination {self.infeasible_partials + self.over_capacity}, "
            f"geometry {self.geometry_pruned}"
            + (
                f" | optimizer attempts {self.optimizer_attempts}, "
                f"successes {self.optimizer_successes}, "
                f"nfev {self.optimizer_nfev}"
                if self.optimizer_attempts
                else ""
            )
        )


@dataclass
class MolecularMapResult:
    bins: Dict[Tuple[int, int], MolecularBinResult] = field(default_factory=dict)
    geometry_pack_name: str = ""


def molecular_stoichiometry_label(
    spec: NucleationSpec, k: int, p: int
) -> Tuple[str, str]:
    """Return building-block and expanded molecular stoichiometry labels."""

    ligand_suffix = (
        ""
        if spec.precursor.ligand_count == 1
        else str(spec.precursor.ligand_count)
    )
    blocks = (
        f"[{spec.core.cation}{spec.core.anion}]{k}"
        f"({spec.precursor.center}{spec.precursor.ligand}{ligand_suffix}){p}"
    )
    counts: Counter[str] = Counter()
    counts[spec.core.cation] += k
    counts[spec.core.anion] += k
    counts[spec.precursor.center] += p
    counts[spec.precursor.ligand] += p * spec.precursor.ligand_count
    order = []
    for symbol in (
        spec.core.cation,
        spec.core.anion,
        spec.precursor.center,
        spec.precursor.ligand,
    ):
        if symbol not in order:
            order.append(symbol)
    expanded = "".join(
        f"{symbol}{counts[symbol]}" for symbol in order if counts[symbol] > 0
    )
    return blocks, expanded


def molecular_max_p_from_se_capacity(spec: NucleationSpec, k: int) -> int:
    """Largest p allowed by Se coordination capacity for a connected skeleton.

    A connected bipartite skeleton with ``k`` core Cd, ``k`` Se, and ``p``
    precursor Cd has at least ``2k + p - 1`` Cd-Se edges. The Se side can
    accept at most ``k * max_cn(Se)`` edges, which gives this exact upper bound.
    """

    max_se_cn = int(spec.graph_rules.max_cn[spec.core.anion])
    return max(0, k * max_se_cn - (2 * k - 1))


@dataclass(frozen=True)
class SlotBasedPMax:
    """Automatic p ceiling for a k sweep (no user ``--pmax``).

    ``pmax`` is free Se on **accepted** p=0 cores (``max_free_slots``),
    capped by the global Se connectivity bound.  If no p=0 core is accepted,
    ``source`` is ``global_fallback`` and ``pmax`` equals that bound.

    No k-growth: each k is treated independently.
    """

    pmax: int
    global_bound: int
    n_p0_enumerated: int
    n_p0_accepted: int
    max_free_slots: int
    source: str  # "slots" | "global_fallback"


def molecular_max_p_from_accepted_p0_slots(
    spec: NucleationSpec,
    k: int,
    *,
    max_skeletons: int = 2000,
    extra_skeleton_edges: Optional[int] = None,
) -> SlotBasedPMax:
    """Slot-based pmax from free Se CN on accepted p=0 skeletons.

    For each legal bare [CdSe]_k core, free Se slots are
    ``sum(max_cn(Se) - deg(Se))``.  Adding precursor Cd uses those slots, so
    ``p ≲ max free`` is the saturation budget for that core family.
    """

    if k < 1:
        raise ValueError("k must be >= 1")
    check_spec = _molecular_check_spec(spec)
    global_bound = molecular_max_p_from_se_capacity(check_spec, k)
    max_se_cn = int(check_spec.graph_rules.max_cn[check_spec.core.anion])

    skeletons, _trunc = _enumerate_inorganic_edge_sets(
        k,
        0,
        check_spec,
        max_skeletons=max_skeletons,
        extra_skeleton_edges=extra_skeleton_edges,
    )
    if (
        not skeletons
        and max_structure_level_possible(k, 0, check_spec) > 0
        and bool(getattr(check_spec.graph_rules, "ring_first_fallback_to_open", True))
    ):
        skeletons, _trunc = _enumerate_inorganic_edge_sets(
            k,
            0,
            check_spec,
            max_skeletons=max_skeletons,
            extra_skeleton_edges=extra_skeleton_edges,
            mode="free",
        )
    symbols = _symbols_for_composition(check_spec, k, 0)
    roles = _roles_for_composition(check_spec, k, 0)
    atoms = _atoms_for_composition(symbols, roles)
    se_ids = [
        atom.atom_id
        for atom in atoms
        if atom.symbol == check_spec.core.anion
    ]

    max_free = 0
    n_accepted = 0
    for skel in skeletons:
        skeleton_graph = nx.Graph()
        skeleton_graph.add_nodes_from(range(len(atoms)))
        skeleton_graph.add_edges_from((int(a), int(b)) for a, b in skel)
        skel_state = _State(atoms=atoms, graph=skeleton_graph)
        if _skeleton_graph_violations(skel_state, check_spec):
            continue
        n_accepted += 1
        free = sum(
            max_se_cn - int(skeleton_graph.degree[s]) for s in se_ids
        )
        if free > max_free:
            max_free = free

    if n_accepted == 0:
        return SlotBasedPMax(
            pmax=int(global_bound),
            global_bound=int(global_bound),
            n_p0_enumerated=len(skeletons),
            n_p0_accepted=0,
            max_free_slots=0,
            source="global_fallback",
        )
    pmax = max(0, min(int(max_free), int(global_bound)))
    return SlotBasedPMax(
        pmax=pmax,
        global_bound=int(global_bound),
        n_p0_enumerated=len(skeletons),
        n_p0_accepted=n_accepted,
        max_free_slots=int(max_free),
        source="slots",
    )


def resolve_molecular_max_p(
    spec: NucleationSpec,
    k: int,
    pmax: Optional[int] = None,
    *,
    max_skeletons: int = 2000,
    extra_skeleton_edges: Optional[int] = None,
) -> Tuple[int, Optional[SlotBasedPMax]]:
    """Return the p ceiling for a k sweep and optional slot diagnostics.

    Explicit ``pmax`` wins (user ``--pmax``).  Otherwise use free Se slots on
    accepted p=0 cores (see :func:`molecular_max_p_from_accepted_p0_slots`).
    """

    if pmax is not None:
        return int(pmax), None
    info = molecular_max_p_from_accepted_p0_slots(
        spec,
        k,
        max_skeletons=max_skeletons,
        extra_skeleton_edges=extra_skeleton_edges,
    )
    return info.pmax, info


def _skeleton_se_capacity(
    k: int,
    p: int,
    inorganic_edges: Sequence[Tuple[int, int]],
    spec: NucleationSpec,
) -> Tuple[int, int]:
    """Return remaining Se coordination slots and this graph's p ceiling."""

    se_ids, _cd_ids, _cl_ids = _index_blocks(k, p)
    degrees = {se: 0 for se in se_ids}
    for left, right in inorganic_edges:
        if left in degrees:
            degrees[left] += 1
        if right in degrees:
            degrees[right] += 1
    max_se_cn = int(spec.graph_rules.max_cn[spec.core.anion])
    remaining = sum(max_se_cn - degree for degree in degrees.values())
    return remaining, p + remaining


def molecular_isomer_log_line(
    isomer: MolecularIsomer, spec: NucleationSpec
) -> str:
    """Human-readable graph composition for one accepted isomer."""

    cations = {spec.core.cation, spec.precursor.center}
    anion = spec.core.anion
    ligand = spec.precursor.ligand
    skeleton_bonds = sum(
        1
        for left, right in isomer.graph.edges
        if (
            isomer.atoms[left].symbol in cations
            and isomer.atoms[right].symbol == anion
        )
        or (
            isomer.atoms[right].symbol in cations
            and isomer.atoms[left].symbol == anion
        )
    )
    terminal_ligands = 0
    mu2_ligands = 0
    multi_ligands = 0
    terminal_bonds = 0
    mu2_bonds = 0
    multi_bonds = 0
    for atom in isomer.atoms:
        if atom.symbol != ligand:
            continue
        host_bonds = sum(
            1
            for neighbor in isomer.graph.neighbors(atom.atom_id)
            if isomer.atoms[neighbor].symbol in cations
        )
        if host_bonds <= 1:
            terminal_ligands += 1
            terminal_bonds += host_bonds
        elif host_bonds == 2:
            mu2_ligands += 1
            mu2_bonds += host_bonds
        else:
            multi_ligands += 1
            multi_bonds += host_bonds
    return (
        f"    ACCEPT {isomer.structure_id} | "
        f"bonds: skeleton={skeleton_bonds}, total={isomer.graph.number_of_edges()}, "
        f"{spec.precursor.center}-{ligand} terminal={terminal_bonds}, "
        f"bridge_mu2={mu2_bonds}, "
        f"multiple_mu3+={multi_bonds} | "
        f"Cl motifs: terminal={terminal_ligands}, mu2={mu2_ligands}, "
        f"mu3+={multi_ligands}"
    )


def _atoms_for_composition(
    symbols: Sequence[str],
    roles: Optional[Sequence[str]] = None,
) -> Tuple[AtomRecord, ...]:
    """Build the atom records, which are identical for every graph in a bin."""

    if roles is None:
        roles = ["atom"] * len(symbols)
    return tuple(
        AtomRecord(
            atom_id=i,
            symbol=str(sym),
            coordinates=(0.0, 0.0, 0.0),
            role=str(roles[i]),
        )
        for i, sym in enumerate(symbols)
    )


def _state_from_parts(
    symbols: Sequence[str],
    edges: Sequence[Tuple[int, int]],
    *,
    roles: Optional[Sequence[str]] = None,
) -> _State:
    atoms = _atoms_for_composition(symbols, roles)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(atoms)))
    graph.add_edges_from((int(a), int(b)) for a, b in edges)
    return _State(atoms=atoms, graph=graph)


def _skeleton_graph_violations(
    state: _State,
    spec: NucleationSpec,
) -> List[str]:
    """Graph violations that every decoration of one skeleton shares.

    A decoration only ever adds ligand-cation edges, so the cation-anion
    subgraph of a decorated graph *is* the skeleton.  Connectivity, the
    legality of the skeleton's own edges, and the anion coordination bounds are
    therefore fixed once the skeleton is chosen, and checking them per
    decoration re-derives the same answer thousands of times.
    """

    anion = spec.core.anion
    violations: List[str] = []

    allowed = allowed_bond_pairs(spec)
    for left, right in state.graph.edges:
        edge_symbols = tuple(
            sorted((state.atoms[left].symbol, state.atoms[right].symbol))
        )
        if edge_symbols not in allowed:
            violations.append(
                f"forbidden_edge:{edge_symbols[0]}-{edge_symbols[1]}:"
                f"{left}-{right}"
            )

    # The anion bonds only to cations, so its skeleton degree is already final.
    max_cn = spec.graph_rules.max_cn.get(anion)
    min_cn = spec.graph_rules.min_cn.get(anion, 0)
    for atom in state.atoms:
        if atom.symbol != anion:
            continue
        degree = state.graph.degree[atom.atom_id]
        if degree < 1:
            violations.append(f"cn0:{atom.symbol}:{atom.atom_id}")
        if max_cn is not None and degree > max_cn:
            violations.append(
                f"max_cn:{atom.symbol}:{atom.atom_id}:{degree}>{max_cn}"
            )
        if spec.enforce_min_cn and degree < min_cn:
            violations.append(
                f"min_cn:{atom.symbol}:{atom.atom_id}:{degree}<{min_cn}"
            )

    # A forbidden ring lives entirely in the cation-anion subgraph, so it is
    # decided by the skeleton alone -- no decoration can create or remove one.
    violations.extend(ring_size_violations(state, spec))

    if spec.require_inorganic_connected:
        components = inorganic_component_count(state, spec)
        if components > 1:
            violations.append(f"inorganic_disconnected:{components}")
        elif components == 0 and any(
            atom.symbol
            in {spec.core.cation, spec.core.anion, spec.precursor.center}
            for atom in state.atoms
        ):
            violations.append("inorganic_empty")

    return violations


def _graph_certificate(state: _State) -> Tuple[object, ...]:
    labels = [atom.symbol for atom in state.atoms]
    edges = [
        (u, v, "bond")
        for u, v in sorted(
            (min(a, b), max(a, b)) for a, b in state.graph.edges
        )
    ]
    return canonical_form(labels, edges).certificate


def _roles_for_composition(
    spec: NucleationSpec, k: int, p: int
) -> List[str]:
    """Assign construction roles: k core pairs + p precursor Cd + 2p Cl."""

    cation = spec.core.cation
    anion = spec.core.anion
    ligand = spec.precursor.ligand
    roles: List[str] = []
    # Se first (k), then core Cd (k), then excess Cd (p), then Cl (2p)
    for _ in range(k):
        roles.append("core_anion")
    for _ in range(k):
        roles.append("core_cation")
    for _ in range(p):
        roles.append("precursor_center")
    for _ in range(2 * p):
        roles.append("precursor_ligand")
    return roles


def _symbols_for_composition(spec: NucleationSpec, k: int, p: int) -> List[str]:
    return (
        [spec.core.anion] * k
        + [spec.core.cation] * k
        + [spec.precursor.center] * p
        + [spec.precursor.ligand] * (2 * p)
    )


def _index_blocks(k: int, p: int) -> Tuple[range, range, range]:
    """Return (se_ids, cd_ids, cl_ids)."""

    se = range(0, k)
    cd = range(k, k + k + p)
    cl = range(k + k + p, k + k + p + 2 * p)
    return se, cd, cl


def _cdse_min_ring(spec: NucleationSpec) -> Optional[int]:
    """Minimum allowed Cd–Se alternating cycle length, if configured."""

    rings = spec.graph_rules.min_ring_size or {}
    cation = spec.core.cation
    anion = spec.core.anion
    for key, minimum in rings.items():
        parts = set(str(key).split("-"))
        if parts == {cation, anion}:
            return int(minimum)
    return None


def _popcount(mask: int) -> int:
    return int(mask).bit_count()


def _bipartite_connected(rows: Sequence[int], n_cd: int, n_se: int) -> bool:
    """Whether the Cd–Se bipartite graph (row bitmasks) is connected."""

    n = n_cd + n_se
    if n == 0:
        return True
    adj: List[List[int]] = [[] for _ in range(n)]
    for c, mask in enumerate(rows):
        m = int(mask)
        s = 0
        while m:
            if m & 1:
                se_v = n_cd + s
                adj[c].append(se_v)
                adj[se_v].append(c)
            m >>= 1
            s += 1
    # Start from first vertex that has an edge (isolated vertices break
    # connectivity unless n==1).
    start = 0
    for v in range(n):
        if adj[v]:
            start = v
            break
    else:
        return n <= 1
    seen = {start}
    stack = [start]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == n


def _max_sorted_row_codes(
    rows: Sequence[int], n_se: int
) -> Tuple[int, ...]:
    """Lexicographic max of sorted-desc Cd row bitmasks over Se permutations.

    Used as an orderly canonicity key for 2-colored bipartite graphs (Cd and Se
    are different colors, so isomorphisms only permute within each part).
    """

    best: Optional[Tuple[int, ...]] = None
    se_idx = list(range(n_se))
    # n_se! is tiny for molecular k (k≤6 → ≤720).
    for perm in permutations(se_idx):
        key: List[int] = []
        for mask in rows:
            new_mask = 0
            for new_s, old_s in enumerate(perm):
                if int(mask) & (1 << old_s):
                    new_mask |= 1 << new_s
            key.append(new_mask)
        key.sort(reverse=True)
        t = tuple(key)
        if best is None or t > best:
            best = t
    return best if best is not None else tuple()


def ring_first_required(
    k: int,
    p: int,
    *,
    cd_pat: Sequence[int] = (3, 3, 4),
    se_pat: Sequence[int] = (3, 3, 3),
) -> bool:
    """True when composition can host min stable 6-ring pattern on a Cd3Se3.

    Pattern default ``Cd[3,3,4] Se[3,3,3]`` (full CN on the ring).  Bond-end
    balance: sum(cd)=10, sum(se)=9 → need Cl (p≥1) or extra Se beyond the
    three ring Se (k≥4).  DFT only named the pattern; this test is pure
    composition.
    """

    cd_t = tuple(int(x) for x in cd_pat)
    se_t = tuple(int(x) for x in se_pat)
    if k < len(se_t) or k < 3:
        return False
    if k + p < len(cd_t):
        return False
    extra_needed = sum(cd_t) - sum(se_t)
    if extra_needed <= 0:
        return True
    has_cl = p >= 1
    has_extra_se = k > len(se_t)
    return has_cl or has_extra_se


def ring_first_required_for_spec(
    k: int, p: int, spec: NucleationSpec
) -> bool:
    """Pack-gated :func:`ring_first_required`."""

    rules = spec.graph_rules
    if not bool(
        getattr(rules, "ring_first_when_pattern_possible", False)
    ):
        return False
    cd_pat = getattr(rules, "ring_min_pattern_cd", (3, 3, 4)) or (3, 3, 4)
    se_pat = getattr(rules, "ring_min_pattern_se", (3, 3, 3)) or (3, 3, 3)
    return ring_first_required(k, p, cd_pat=cd_pat, se_pat=se_pat)


def two_ring_possible(k: int, p: int) -> bool:
    """Composition can host two fused Cd–Se 6-rings (any fusion mode).

    Cheapest DFT fusion (path Cd1Se2) needs ≥5 Cd and ≥4 Se on the fused
    pair; face needs ≥4 Cd and ≥4 Se.  Also requires 1-ring pattern budget.
    """

    if not ring_first_required(k, p):
        return False
    n_cd = k + p
    # path: +2 Cd +1 Se → k≥4, n_cd≥5; face: +1 Cd +1 Se → k≥4, n_cd≥4
    return k >= 4 and n_cd >= 4


def two_ring_possible_for_spec(k: int, p: int, spec: NucleationSpec) -> bool:
    rules = spec.graph_rules
    if not bool(getattr(rules, "ring_first_when_pattern_possible", False)):
        return False
    if not bool(getattr(rules, "multi_ring_ladder", True)):
        return False
    return two_ring_possible(k, p)


def max_structure_level_possible(k: int, p: int, spec: NucleationSpec) -> int:
    """Highest structured skeleton level composition allows (0=free, 1=1-ring, 2=fused)."""

    if two_ring_possible_for_spec(k, p, spec):
        return 2
    if ring_first_required_for_spec(k, p, spec):
        return 1
    return 0


def structure_mode_for_level(level: int) -> str:
    """Map structure level → skeleton enum mode name."""

    if level >= 2:
        return "fused2"
    if level >= 1:
        return "ring_first"
    return "free"


def cdse_six_ring_sets(
    edges: Sequence[Tuple[int, int]],
    k: int,
    p: int,
) -> List[frozenset]:
    """Distinct Cd–Se alternating 6-cycles (each as a frozenset of atom ids)."""

    se_ids, cd_ids, _ = _index_blocks(k, p)
    se_set = set(se_ids)
    cd_list = list(cd_ids)
    g = nx.Graph()
    g.add_nodes_from(list(se_set) + cd_list)
    g.add_edges_from((int(a), int(b)) for a, b in edges)
    rings: Set[frozenset] = set()
    for i, c1 in enumerate(cd_list):
        for j in range(i + 1, len(cd_list)):
            c2 = cd_list[j]
            for m in range(j + 1, len(cd_list)):
                c3 = cd_list[m]
                n1 = set(g.neighbors(c1)) & se_set
                n2 = set(g.neighbors(c2)) & se_set
                n3 = set(g.neighbors(c3)) & se_set
                for a in n1 & n2:
                    for b in n2 & n3:
                        for c in n3 & n1:
                            if len({a, b, c}) == 3:
                                rings.add(frozenset((c1, c2, c3, a, b, c)))
    return list(rings)


def forced_ring_degree_profiles(
    edges: Sequence[Tuple[int, int]],
    k: int,
    p: int,
    spec: NucleationSpec,
    *,
    mode: str,
) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]:
    """Return forced rings and feasible per-Cd final-CN lower bounds.

    Se coordination is final in the skeleton because decoration only adds
    Cd--Cl edges.  It is therefore checked here.  Cd minima are returned as
    demand profiles for constraint propagation during Cl enumeration.

    Ring-first imposes one canonical six-ring and fused2 imposes two.  The
    seed growers canonicalise atom labels, so canonical ring ordering is used
    to retain stable identities in dumps and resumed decoration.
    """

    resolved = str(mode).strip().lower()
    n_forced = 2 if resolved == "fused2" else 1 if resolved == "ring_first" else 0
    if n_forced == 0:
        return (), ((),)
    rings = sorted(tuple(sorted(int(x) for x in ring)) for ring in cdse_six_ring_sets(edges, k, p))
    if len(rings) < n_forced:
        return (), ()
    forced = tuple(rings[:n_forced])
    se_ids, cd_ids, _ = _index_blocks(k, p)
    se_set, cd_set = set(se_ids), set(cd_ids)
    graph = nx.Graph()
    graph.add_edges_from((int(a), int(b)) for a, b in edges)
    se_pat = tuple(sorted(int(x) for x in spec.graph_rules.ring_min_pattern_se))
    cd_pat = tuple(sorted(int(x) for x in spec.graph_rules.ring_min_pattern_cd))
    for ring in forced:
        se_deg = tuple(sorted(int(graph.degree[x]) for x in ring if x in se_set))
        if len(se_deg) != len(se_pat) or any(a < b for a, b in zip(se_deg, se_pat)):
            return forced, ()

    cd_pos = {atom: i for i, atom in enumerate(cd_ids)}
    base = [int(graph.degree[atom]) for atom in cd_ids]
    max_cd = int(spec.graph_rules.max_cn[spec.core.cation])
    profiles: Set[Tuple[int, ...]] = set()
    # [3,3,4] has only three unique permutations, but support arbitrary packs.
    per_ring = [sorted(set(permutations(cd_pat))) for _ in forced]
    for assigned in product(*per_ring):
        lower = list(base)
        valid = True
        for ring, thresholds in zip(forced, assigned):
            ring_cd = sorted(x for x in ring if x in cd_set)
            if len(ring_cd) != len(thresholds):
                valid = False
                break
            for atom, threshold in zip(ring_cd, thresholds):
                pos = cd_pos[atom]
                lower[pos] = max(lower[pos], int(threshold))
                if lower[pos] > max_cd:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            deficits = [max(0, lower[i] - base[i]) for i in range(len(base))]
            n_ligands = 2 * p
            if any(value > n_ligands for value in deficits):
                continue
            if sum(deficits) > n_ligands * int(spec.graph_rules.max_cn[spec.precursor.ligand]):
                continue
            profiles.add(tuple(lower))
    return forced, tuple(sorted(profiles))


def count_cdse_six_rings(
    edges: Sequence[Tuple[int, int]],
    k: int,
    p: int,
) -> int:
    """Count distinct Cd–Se alternating 6-cycles in an inorganic edge set."""

    return len(cdse_six_ring_sets(edges, k, p))


def cdse_six_ring_atom_ids(
    edges: Sequence[Tuple[int, int]],
    k: int,
    p: int,
) -> Set[int]:
    """Atom ids that participate in at least one Cd–Se 6-ring."""

    atoms: Set[int] = set()
    for ring in cdse_six_ring_sets(edges, k, p):
        atoms.update(ring)
    return atoms


def ring_closure_log_label(
    n_rings: int,
    *,
    pattern_possible: Optional[bool] = None,
) -> str:
    """Human label for log lines.

    When ``pattern_possible`` is False, a graph 6-cycle is *not* the min
    full-CN ring target (Cd[3,3,4]/Se[3,3,3]) — label it accordingly.
    """

    if n_rings <= 0:
        return "0-ring (open/acyclic or no 6-cycle)"
    if pattern_possible is False:
        if n_rings == 1:
            return "1-cycle (not min-pattern ring; open path)"
        return f"{n_rings}-cycle (not min-pattern ring; open path)"
    if n_rings == 1:
        return "1-ring closed"
    return f"{n_rings}-ring closed"


def frame_degrees_for_skeleton(
    state: _State,
    inorganic_edges: Sequence[Tuple[int, int]],
    k: int,
    p: int,
    spec: NucleationSpec,
    *,
    ring_cd_cn: int = 4,
    ring_se_cn: int = 4,
    core_cd_min: int = 3,
    pendant_cd_min: int = 2,
) -> List[int]:
    """Target coordination numbers for **3D frame** angle/bond tables.

    Skeleton graph degrees are incomplete (Cl not placed yet).  The pack
    selects centre angles from ``degrees[i]``:

    - Cd CN2 → nearly linear (~176°) — **cannot close a 6-ring**
    - Cd CN3/4 → tetrahedral-like (~109.5°) — ring conformation

    Ring atoms must use the **target ring conformation** (Cd[3,3,4]-like /
    typically CN4 tables in the pack), not skeleton CN1/2.  This only chooses
    the **angle/bond template**; it does not invent bonds.  Decoration later
    fills free slots up to those target CNs.

    Rules (v1)::

        Cd in a 6-ring: ring_cd_cn (default 4 — pack has explicit tetrahedral)
        Cd skel ≥ 2 not in ring: max(skel, core_cd_min=3)
        Cd mono-Se pendant: max(skel, pendant_cd_min=2)
        Se in a 6-ring: ring_se_cn (default 4; cap at pack max, never force 5)
        Se otherwise: skeleton degree
        Cl: 0
    """

    n = len(state.atoms)
    degrees = [int(state.graph.degree[i]) for i in range(n)]
    se_ids, cd_ids, _cl_ids = _index_blocks(k, p)
    max_cd = int(spec.graph_rules.max_cn.get(spec.core.cation, 4))
    # Prefer tetrahedral-like Se ≤ 4 for frames even if graph max_cn allows 5.
    max_se = min(4, int(spec.graph_rules.max_cn.get(spec.core.anion, 4)))
    ring_atoms = cdse_six_ring_atom_ids(inorganic_edges, k, p)
    r_cd = min(max_cd, max(3, int(ring_cd_cn)))
    r_se = min(max_se, max(3, int(ring_se_cn)))

    for i in cd_ids:
        skel = degrees[i]
        if skel <= 0:
            continue
        if i in ring_atoms:
            # Ring Cd: tetrahedral template (Cd[3,3,4] world), never linear CN2.
            degrees[i] = r_cd
        elif skel >= 2:
            degrees[i] = min(max_cd, max(skel, int(core_cd_min)))
        else:
            degrees[i] = min(max_cd, max(skel, int(pendant_cd_min)))

    for i in se_ids:
        skel = degrees[i]
        if skel <= 0:
            continue
        if i in ring_atoms:
            degrees[i] = max(skel, r_se) if skel >= r_se else r_se
            degrees[i] = min(max_se, degrees[i])
        else:
            degrees[i] = min(max_se, skel)

    return degrees


def format_frame_cn_summary(
    degrees: Sequence[int],
    k: int,
    p: int,
) -> str:
    """Compact Cd/Se frame-CN lists for log lines."""

    se_ids, cd_ids, _ = _index_blocks(k, p)
    cd = sorted(int(degrees[i]) for i in cd_ids if int(degrees[i]) > 0)
    se = sorted(int(degrees[i]) for i in se_ids if int(degrees[i]) > 0)
    return f"frame Cd{cd} Se{se}"


def _alternating_six_cycle_order(
    edges: Sequence[Tuple[int, int]],
    ring: frozenset,
    se_set: Set[int],
    cd_set: Set[int],
) -> Optional[List[int]]:
    """Return Se-Cd-Se-Cd-Se-Cd order around one 6-ring, or None."""

    if len(ring) != 6:
        return None
    g = nx.Graph()
    g.add_nodes_from(ring)
    for a, b in edges:
        if a in ring and b in ring:
            g.add_edge(int(a), int(b))
    start = min(n for n in ring if n in se_set)
    order = [start]
    prev = -1
    cur = start
    for _ in range(5):
        nbrs = [n for n in g.neighbors(cur) if n != prev]
        if not nbrs:
            return None
        # Prefer alternating type.
        want_cd = cur in se_set
        typed = [
            n
            for n in nbrs
            if (n in cd_set) == want_cd and n not in order
        ]
        nxt = typed[0] if typed else (nbrs[0] if nbrs[0] not in order else None)
        if nxt is None:
            # last step may close to start
            close = [n for n in nbrs if n == start]
            if close and len(order) == 5:
                break
            return None
        order.append(nxt)
        prev, cur = cur, nxt
    if len(order) != 6:
        return None
    # Validate alternation
    for i, node in enumerate(order):
        if i % 2 == 0 and node not in se_set:
            return None
        if i % 2 == 1 and node not in cd_set:
            return None
    return order


#: The saturated limit every ring target interpolates toward.  Measured on
#: ``examples/cifs/CdSe_zb.cif``, bulk zinc-blende 6-rings sit at exactly this
#: angle at *both* Cd and Se, with exactly alternating +/-60 deg dihedrals.
TETRAHEDRAL_ANGLE_DEG = 109.4712206345
#: Ring coordination the pack's DFT ring averages describe (the
#: ``ring_min_pattern_*`` baseline), and the coordination at which a ring atom
#: is saturated and locally bulk-like.
RING_CN_REFERENCE = 3
RING_CN_SATURATED = 4


def _ring_saturation(coordination: int) -> float:
    """0 at the pack's reference ring CN, 1 once the atom is saturated.

    Clamped rather than extrapolated below the reference: the pack's ring
    averages were measured on a CN3-dominated pattern, so there is no ring data
    under it, and the ordinary CN2 tables describe a near-linear Cd (175.6 deg)
    that cannot occur inside a 6-ring.
    """

    span = RING_CN_SATURATED - RING_CN_REFERENCE
    if span <= 0:
        return 1.0
    fraction = (float(coordination) - RING_CN_REFERENCE) / float(span)
    return min(1.0, max(0.0, fraction))


def ring_cn_targets(
    pack: GeometryPack,
    order: Sequence[int],
    inorganic_edges: Sequence[Tuple[int, int]],
    se_set: Set[int],
    conformation: str,
) -> Tuple[List[float], Tuple[float, ...]]:
    """Per-atom ring angles and ring dihedrals, resolved by coordination number.

    The pack states one angle per element for the whole ring, which is a DFT
    average over a single CN pattern -- and very nearly the CN-weighted mean of
    the pack's own per-CN tables: for ``cd_cn [3,3,4]`` the tables give
    ``(117.831*2 + 109.5)/3 = 115.05`` against the stated 116.5, and for
    ``se_cn [3,3,3]`` they give 89.245 against the stated 89.5.  So the single
    number is an average, and averaging is the only thing wrong with it.

    An undercoordinated ring atom genuinely is compressed like that, but a
    *saturated* one is locally bulk-like and should be tetrahedral.  Each ring
    atom's angle is therefore interpolated linearly in its coordination number
    from the pack average at ``RING_CN_REFERENCE`` to
    ``TETRAHEDRAL_ANGLE_DEG`` at ``RING_CN_SATURATED``, so a fully saturated
    ring collapses onto the bulk ZB chair.

    This deliberately overrides the pack's own high-CN Se entries
    (``angles.Se.cn4 = 97.446``, 12 deg short of tetrahedral, and
    ``cn5 = 83.324``, below ``cn2``).  Those come from small strained clusters
    and are not monotonic in CN, so they are not a usable saturated limit.  A
    relaxation will close these angles up again; the point is to start on the
    correct side of the bulk limit rather than below it.

    Dihedrals are interpolated by the ring's *mean* saturation, since the
    torsion sequence is a whole-ring conformational property.  Only the chair
    is interpolated: zinc-blende is all-chair, so a boat has no bulk limit to
    collapse onto and keeps its DFT sequence.

    Saturation is measured on the **Cd-Se** coordination, not the total
    including Cl, for two reasons.  It is what the bulk limit means -- the
    ``[4,4,4]/[4,4,4]`` reference ring is one where every atom has four Cd-Se
    bonds, and bulk has no Cl -- so it is the honest measure of "how bulk-like
    is this ring".  And it is fixed by the skeleton, which keeps this function
    single-valued: ``embed_skeleton_frames`` builds the survey frame that
    enumerates bridge sites *before* any Cl exists, so a target that moved once
    Cl were added would give the survey and the final construction two
    different rings and silently lose candidates.
    """

    pattern = pack.cdse6_ring_pattern()
    angle_at_se = float(pattern.angle_at_se_deg)
    angle_at_cd = float(pattern.angle_at_cd_deg)
    inorganic_cn: Dict[int, int] = {}
    for left, right in inorganic_edges:
        inorganic_cn[int(left)] = inorganic_cn.get(int(left), 0) + 1
        inorganic_cn[int(right)] = inorganic_cn.get(int(right), 0) + 1
    angles: List[float] = []
    fractions: List[float] = []
    for atom in order:
        fraction = _ring_saturation(inorganic_cn.get(int(atom), 0))
        fractions.append(fraction)
        base = angle_at_se if atom in se_set else angle_at_cd
        angles.append(
            (1.0 - fraction) * base + fraction * TETRAHEDRAL_ANGLE_DEG
        )
    conf = str(conformation).strip().lower()
    observed = tuple(float(value) for value in pack.cdse6_dihedrals(conf))
    ideal = (
        (60.0, -60.0, 60.0, -60.0, 60.0, -60.0)
        if conf != "boat"
        else (0.0, 60.0, -60.0, 0.0, 60.0, -60.0)
    )
    if len(observed) < 6:
        observed = ideal
    if conf == "boat":
        return angles, observed
    ring_fraction = sum(fractions) / len(fractions) if fractions else 0.0
    dihedrals = tuple(
        (1.0 - ring_fraction) * observed[index]
        + ring_fraction * ideal[index]
        for index in range(6)
    )
    return angles, dihedrals


def _six_ring_template_positions(
    *,
    bond_length: float,
    angle_at_se_deg: float = 109.47,
    angle_at_cd_deg: float = 109.47,
    conformation: str = "chair",
    dihedrals_deg: Optional[Sequence[float]] = None,
    angles_deg: Optional[Sequence[float]] = None,
    bond_lengths: Optional[Sequence[float]] = None,
) -> List[FloatArray]:
    """Construct a **chair** or **boat** Cd₃Se₃ ring from pack targets.

    The hard-coded chair/boat arrays are initialization guesses only.  A deterministic
    constrained least-squares construction then fits the pack bond, alternating
    Se/Cd angles and all six phase-aligned dihedrals.

    ``bond_lengths[i]`` is the target for the bond from ring atom ``i`` to
    ``i+1``; when given it replaces the single ``bond_length`` so each ring bond
    can carry its own CN-resolved target and agree with the audit.

    Order: Se, Cd, Se, Cd, Se, Cd (indices 0..5).
    """

    conf = str(conformation).strip().lower()
    if conf not in {"chair", "boat"}:
        conf = "chair"
    per_edge = (
        tuple(float(v) for v in bond_lengths)
        if bond_lengths is not None and len(tuple(bond_lengths)) == 6
        else None
    )
    r = float(bond_length) if per_edge is None else sum(per_edge) / 6.0
    if r < 2.0:
        r = 2.635

    # Unit cyclohexane (C–C ≈ 1.54 Å in organic; we rescale to r).
    # Source: standard chair / boat coordinates (tetrahedral network).
    if conf == "boat":
        # Boat: atoms 0,3 flagpoles; 1–2–4–5 form the "hull"
        raw = np.array(
            [
                [1.070, -0.728, 0.510],
                [1.070, 0.728, 0.000],
                [0.000, 1.265, -0.510],
                [-1.070, 0.728, 0.510],
                [-1.070, -0.728, 0.000],
                [0.000, -1.265, -0.510],
            ],
            dtype=float,
        )
    else:
        # Chair: alternating up/down (ZB 6-ring motif)
        raw = np.array(
            [
                [1.070, -0.728, 0.255],
                [1.070, 0.728, -0.255],
                [0.000, 1.265, 0.255],
                [-1.070, 0.728, -0.255],
                [-1.070, -0.728, 0.255],
                [0.000, -1.265, -0.255],
            ],
            dtype=float,
        )

    # Scale so mean adjacent bond = r
    bonds = [
        float(np.linalg.norm(raw[(i + 1) % 6] - raw[i])) for i in range(6)
    ]
    mean_b = sum(bonds) / 6.0
    if mean_b < 1.0e-9:
        mean_b = 1.0
    scale = r / mean_b
    pts = [np.asarray(raw[i] * scale, dtype=float) for i in range(6)]

    # Exact bond equalisation (SHAKE on ring only — keeps chair/boat shape)
    for _ in range(40):
        max_err = 0.0
        for i in range(6):
            j = (i + 1) % 6
            dvec = pts[j] - pts[i]
            dn = float(np.linalg.norm(dvec))
            if dn < 1.0e-12:
                continue
            err = dn - (r if per_edge is None else per_edge[i])
            max_err = max(max_err, abs(err))
            corr = 0.5 * err * (dvec / dn)
            pts[i] = pts[i] + corr
            pts[j] = pts[j] - corr
        if max_err < 1.0e-4:
            break
    target_dih = tuple(float(x) for x in (dihedrals_deg or ()))
    if len(target_dih) < 6:
        defaults = (
            (60.0, -60.0, 60.0, -60.0, 60.0, -60.0)
            if conf == "chair"
            else (0.0, 60.0, -60.0, 0.0, 60.0, -60.0)
        )
        target_dih = defaults
    reference = np.vstack(pts)

    def angle_deg(a: FloatArray, b: FloatArray, c: FloatArray) -> float:
        u, v = a - b, c - b
        den = float(np.linalg.norm(u) * np.linalg.norm(v))
        if den < 1.0e-12:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(float(np.dot(u, v)) / den, -1.0, 1.0))))

    def torsion_deg(a: FloatArray, b: FloatArray, c: FloatArray, d: FloatArray) -> float:
        b0, b1, b2 = -(b - a), c - b, d - c
        norm = float(np.linalg.norm(b1))
        if norm < 1.0e-12:
            return 0.0
        b1 = b1 / norm
        v = b0 - float(np.dot(b0, b1)) * b1
        w = b2 - float(np.dot(b2, b1)) * b1
        return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))

    def wrapped_delta(value: float, target: float) -> float:
        return (value - target + 180.0) % 360.0 - 180.0

    # Per-position angle targets when the caller resolved them by coordination
    # number; otherwise the historical one-value-per-element behaviour.
    if angles_deg is not None and len(tuple(angles_deg)) == 6:
        angle_targets = tuple(float(value) for value in angles_deg)
    else:
        angle_targets = tuple(
            float(angle_at_se_deg if i % 2 == 0 else angle_at_cd_deg)
            for i in range(6)
        )

    def residual(flat: FloatArray) -> FloatArray:
        xyz = np.asarray(flat, dtype=float).reshape((6, 3))
        out: List[float] = []
        for i in range(6):
            target_r = r if per_edge is None else per_edge[i]
            out.append(
                (float(np.linalg.norm(xyz[(i + 1) % 6] - xyz[i])) - target_r) / 0.01
            )
        for i in range(6):
            target = angle_targets[i]
            out.append((angle_deg(xyz[i - 1], xyz[i], xyz[(i + 1) % 6]) - target) / 2.0)
        for i in range(6):
            value = torsion_deg(xyz[i - 1], xyz[i], xyz[(i + 1) % 6], xyz[(i + 2) % 6])
            out.append(wrapped_delta(value, target_dih[i]) / 3.0)
        # Remove rigid-body zero modes without changing internal geometry.
        out.extend(((xyz.mean(axis=0)) / 0.01).tolist())
        out.extend(((xyz[0] - reference[0]) / 0.05).tolist())
        out.extend(((xyz[1, 1:] - reference[1, 1:]) / 0.05).tolist())
        return np.asarray(out, dtype=float)

    fitted = least_squares(
        residual,
        reference.reshape(-1),
        method="trf",
        max_nfev=2000,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-11,
    ).x.reshape((6, 3))
    pts = [np.asarray(fitted[i], dtype=float) for i in range(6)]
    return pts


_RING_TEMPLATE_CACHE: Dict[
    Tuple[object, ...], Tuple[Tuple[float, float, float], ...]
] = {}


def _cached_six_ring_template_positions(
    *,
    bond_length: float,
    angle_at_se_deg: float,
    angle_at_cd_deg: float,
    conformation: str,
    dihedrals_deg: Sequence[float],
    angles_deg: Optional[Sequence[float]] = None,
    bond_lengths: Optional[Sequence[float]] = None,
) -> Tuple[FloatArray, ...]:
    """Return one immutable cached chair/boat template for a pack target set.

    ``angles_deg`` joins the key, so a ring whose coordination numbers resolve
    to different per-atom angles gets its own template.  There are only a
    handful of distinct ring CN patterns per bin, so the cache still absorbs
    the fit -- which is the expensive part.
    """

    key = (
        float(bond_length),
        float(angle_at_se_deg),
        float(angle_at_cd_deg),
        str(conformation).strip().lower(),
        tuple(float(value) for value in dihedrals_deg),
        tuple(float(value) for value in angles_deg) if angles_deg else None,
        tuple(float(value) for value in bond_lengths) if bond_lengths else None,
    )
    cached = _RING_TEMPLATE_CACHE.get(key)
    if cached is None:
        built = _six_ring_template_positions(
            bond_length=bond_length,
            angle_at_se_deg=angle_at_se_deg,
            angle_at_cd_deg=angle_at_cd_deg,
            conformation=conformation,
            dihedrals_deg=dihedrals_deg,
            angles_deg=angles_deg,
            bond_lengths=bond_lengths,
        )
        cached = tuple(
            (float(point[0]), float(point[1]), float(point[2]))
            for point in built
        )
        _RING_TEMPLATE_CACHE[key] = cached
    return tuple(np.asarray(point, dtype=float) for point in cached)


def _refine_completed_ring_graph(
    state: _State,
    coordinates: FloatArray,
    pack: GeometryPack,
    spec: NucleationSpec,
    k: int,
    p: int,
    conformation: str,
    *,
    fixed_ring_nodes: Optional[Set[int]] = None,
) -> FloatArray:
    """Close additional fused rings while keeping the seed ring fixed.

    This is only used for multi-ring graphs.  The first chair/boat seed is
    authoritative; only atoms outside ``fixed_ring_nodes`` are variables.
    """

    initial = np.asarray(coordinates, dtype=float)
    n = len(state.atoms)
    degrees = [int(state.graph.degree[i]) for i in range(n)]
    bonded = {
        (min(int(a), int(b)), max(int(a), int(b))) for a, b in state.graph.edges
    }
    ligand = spec.precursor.ligand
    inorganic_edges = tuple(
        (a, b)
        for a, b in bonded
        if state.atoms[a].symbol != ligand and state.atoms[b].symbol != ligand
    )
    se_ids, cd_ids, _ = _index_blocks(k, p)
    se_set, cd_set = set(se_ids), set(cd_ids)
    ring_orders = []
    for ring in cdse_six_ring_sets(inorganic_edges, k, p):
        order = _alternating_six_cycle_order(inorganic_edges, ring, se_set, cd_set)
        if order is not None:
            ring_orders.append(order)
    ring_orders.sort(key=lambda order: tuple(sorted(order)))
    n_forced = 2 if two_ring_possible_for_spec(k, p, spec) else 1
    ring_orders = ring_orders[:n_forced]
    # The seed ring is *restrained*, not frozen.  Freezing it made any bond
    # from a ring atom to the rest of the molecule unfixable: the template sets
    # the ring's internal spacing, and a neighbour that needs a different span
    # can only be stretched, never met.  That showed up as bonds long by
    # +0.1-0.2 A -- the dominant remaining rejection -- and no amount of moving
    # the *other* end could close it.  Letting the ring breathe under a stiff
    # tether is the standard restrained-cleanup arrangement (ETKDG/MMFF do the
    # same), and the ring angle/torsion terms still hold its shape.
    seed_nodes = set(fixed_ring_nodes or ())
    if not seed_nodes and ring_orders:
        seed_nodes = set(ring_orders[0])
    variable_nodes = list(range(n))
    seed_list = sorted(seed_nodes)
    if not variable_nodes:
        return initial.copy()
    variable_pos = {atom: offset for offset, atom in enumerate(variable_nodes)}
    pattern = pack.cdse6_ring_pattern()
    nonbond_floors: List[Tuple[int, int, float]] = []
    for a in range(n):
        for b in range(a + 1, n):
            if (a, b) in bonded:
                continue
            rule = spec.graph_rules.pair_rules.get(
                pair_key(state.atoms[a].symbol, state.atoms[b].symbol)
            )
            if rule is None:
                continue
            floor = float(rule.min_distance or 0.0)
            if floor > 0.0:
                floor += 0.02
            if rule.bond_allowed and rule.bond_max_distance is not None:
                floor = max(floor, float(rule.bond_max_distance) + 0.05)
            if floor > 0.0:
                nonbond_floors.append((a, b, floor))

    # Everything the residual needs that does not depend on the coordinates is
    # hoisted here.  The finite-difference Jacobian evaluates the residual once
    # per column group, so anything left inside it is paid tens of thousands of
    # times per solve -- ``_molecular_bond_length`` alone used to be called
    # ~200k times for a value that is fixed by the graph.
    bond_list = list(bonded)
    bond_left = np.fromiter((a for a, _ in bond_list), dtype=int, count=len(bond_list))
    bond_right = np.fromiter((b for _, b in bond_list), dtype=int, count=len(bond_list))
    # Ring bonds are no longer special-cased to the flat ``bond_cdse_A``: they
    # read the same CN-indexed table as every other bond, which is also what
    # the audit reads.
    bond_targets = np.array(
        [
            _molecular_bond_length(state, pack, spec, a, b, degrees)
            for a, b in bond_list
        ],
        dtype=float,
    )
    bond_tols = np.array(
        [
            WELL_BAND_FRACTION
            * _molecular_bond_tolerance(state, pack, spec, a, b, degrees)
            for a, b in bond_list
        ],
        dtype=float,
    )
    floor_left = np.fromiter(
        (a for a, _, _ in nonbond_floors), dtype=int, count=len(nonbond_floors)
    )
    floor_right = np.fromiter(
        (b for _, b, _ in nonbond_floors), dtype=int, count=len(nonbond_floors)
    )
    floor_values = np.fromiter(
        (f for _, _, f in nonbond_floors), dtype=float, count=len(nonbond_floors)
    )
    angle_i: List[int] = []
    angle_j: List[int] = []
    angle_k: List[int] = []
    angle_targets: List[float] = []
    tors_i: List[int] = []
    tors_j: List[int] = []
    tors_k: List[int] = []
    tors_l: List[int] = []
    tors_targets: List[float] = []
    for order in ring_orders:
        # Same coordination-resolved targets the seed template was built from,
        # so the closure polish is not pulling the ring away from its own
        # template.
        order_angles, order_dihedrals = ring_cn_targets(
            pack, order, inorganic_edges, se_set, conformation
        )
        for i, atom in enumerate(order):
            angle_i.append(order[i - 1])
            angle_j.append(atom)
            angle_k.append(order[(i + 1) % 6])
            angle_targets.append(order_angles[i])
            tors_i.append(order[i - 1])
            tors_j.append(atom)
            tors_k.append(order[(i + 1) % 6])
            tors_l.append(order[(i + 2) % 6])
            tors_targets.append(order_dihedrals[i])
    angle_i_a = np.array(angle_i, dtype=int)
    angle_j_a = np.array(angle_j, dtype=int)
    angle_k_a = np.array(angle_k, dtype=int)
    angle_target_a = np.array(angle_targets, dtype=float)
    tors_i_a = np.array(tors_i, dtype=int)
    tors_j_a = np.array(tors_j, dtype=int)
    tors_k_a = np.array(tors_k, dtype=int)
    tors_l_a = np.array(tors_l, dtype=int)
    tors_target_a = np.array(tors_targets, dtype=float)
    variable_index = np.array(variable_nodes, dtype=int)
    initial_centroid = initial.mean(axis=0)
    initial_variable = initial[variable_index]
    seed_index = np.array(seed_list, dtype=int)
    initial_seed = initial[seed_index] if seed_index.size else np.empty((0, 3))
    # Same reasoning as in ``_steric_relax_ligands``: the closure polish has to
    # move the quantities the audit rejects on, or completed CN3 centres stay
    # pyramidal and every ring candidate fails on ``improper:``.
    improper_terms, hard_angle_terms = _audited_local_terms(state, pack, spec)
    improper_index = np.array(
        [[c, a, b, d] for c, a, b, d, _ in improper_terms], dtype=int
    ).reshape(-1, 4)
    improper_target = np.array([t for *_, t in improper_terms], dtype=float)
    hard_index = np.array(
        [[l, c, r] for l, c, r, _t, _b, _g in hard_angle_terms], dtype=int
    ).reshape(-1, 3)
    hard_target = np.array(
        [t for _l, _c, _r, t, _b, _g in hard_angle_terms], dtype=float
    )
    hard_band = np.array(
        [b for _l, _c, _r, _t, b, _g in hard_angle_terms], dtype=float
    )
    # Alternative modes of one angle share a group id; the residual reduces
    # each group to its smallest excess, so the row count and the residual
    # length differ whenever a multi-modal centre is present.
    hard_group = np.array(
        [g for _l, _c, _r, _t, _b, g in hard_angle_terms], dtype=int
    )
    _grp = {}
    hard_group = np.array(
        [_grp.setdefault(g, len(_grp)) for g in hard_group], dtype=int
    )
    n_hard_rows = int(hard_group.max()) + 1 if hard_group.size else 0
    n_ring_terms = len(angle_i)
    stop_bond = len(bond_list)
    stop_floor = stop_bond + len(nonbond_floors)
    stop_ring = stop_floor + 2 * n_ring_terms
    stop_improper = stop_ring + len(improper_terms)
    stop_hard = stop_improper + n_hard_rows
    stop_seed = stop_hard + 3 * len(seed_index)
    out = np.empty(stop_seed + 3 + 3 * len(variable_index), dtype=float)
    work = initial.copy()

    def unpack(flat: FloatArray) -> FloatArray:
        xyz = initial.copy()
        xyz[variable_nodes] = np.asarray(flat, dtype=float).reshape(
            (len(variable_nodes), 3)
        )
        return xyz

    def residual(flat):
        work[variable_index] = np.asarray(flat, dtype=float).reshape(
            (len(variable_index), 3)
        )
        xyz = work
        # Weighted against the acceptance band, not against machine precision:
        # a bond at the edge of the audit band costs ~5 while an angle at the
        # edge of its band costs 1, so bonds stay stiffest without the fit
        # spending its whole budget chasing an unreachable 2e-4 A.
        out[:stop_bond] = _band_excess(
            np.linalg.norm(xyz[bond_right] - xyz[bond_left], axis=1),
            bond_targets,
            bond_tols,
        ) / BOND_WELL_SCALE_A
        if stop_floor > stop_bond:
            # Contacts steer the fit away from collapse, but must not compete
            # with the covalent graph strongly enough to stretch real bonds.
            out[stop_bond:stop_floor] = np.minimum(
                0.0,
                np.linalg.norm(xyz[floor_right] - xyz[floor_left], axis=1)
                - floor_values,
            ) / REPULSION_WELL_SCALE_A
        if n_ring_terms:
            left = xyz[angle_i_a] - xyz[angle_j_a]
            right = xyz[angle_k_a] - xyz[angle_j_a]
            den = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
            degenerate = den < 1.0e-12
            cosine = np.einsum("ij,ij->i", left, right) / np.where(degenerate, 1.0, den)
            angles = np.where(
                degenerate,
                0.0,
                np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))),
            )
            b0 = -(xyz[tors_j_a] - xyz[tors_i_a])
            b1 = xyz[tors_k_a] - xyz[tors_j_a]
            b2 = xyz[tors_l_a] - xyz[tors_k_a]
            b1_norm = np.linalg.norm(b1, axis=1)
            b1_unit = b1 / np.where(b1_norm < 1.0e-12, 1.0, b1_norm)[:, None]
            v = b0 - np.einsum("ij,ij->i", b0, b1_unit)[:, None] * b1_unit
            w = b2 - np.einsum("ij,ij->i", b2, b1_unit)[:, None] * b1_unit
            torsions = np.where(
                b1_norm < 1.0e-12,
                0.0,
                np.degrees(
                    np.arctan2(
                        np.einsum("ij,ij->i", np.cross(b1_unit, v), w),
                        np.einsum("ij,ij->i", v, w),
                    )
                ),
            )
            out[stop_floor:stop_ring:2] = (angles - angle_target_a) / 8.0
            out[stop_floor + 1 : stop_ring : 2] = (
                (torsions - tors_target_a + 180.0) % 360.0 - 180.0
            ) / 12.0
        improper_out, hard_out = _local_term_residuals(
            xyz,
            improper_index,
            improper_target,
            hard_index,
            hard_target,
            improper_scale=WELL_BAND_FRACTION * AUDIT_IMPROPER_TOLERANCE_DEG,
            hard_scale=WELL_BAND_FRACTION * hard_band,
            hard_group=hard_group,
        )
        out[stop_ring:stop_improper] = improper_out
        out[stop_improper:stop_hard] = hard_out
        if seed_index.size:
            # Stiff but finite: a 0.05 A shift of a seed-ring atom costs 1,
            # while a 0.05 A bond error costs 5, so bonds win and the ring
            # yields slightly instead of forcing the strain outward.
            out[stop_hard:stop_seed] = (
                (xyz[seed_index] - initial_seed).reshape(-1) / RING_SEED_TETHER_A
            )
        out[stop_seed : stop_seed + 3] = (xyz.mean(axis=0) - initial_centroid) / 0.05
        out[stop_seed + 3 :] = (xyz[variable_index] - initial_variable).reshape(-1) / 5.0
        return out.copy()

    n_variable = len(variable_nodes)
    n_rows = (
        len(bonded)
        + len(nonbond_floors)
        + 12 * len(ring_orders)
        + len(improper_terms)
        + len(hard_angle_terms)
        + 3 * len(seed_index)
        + 3
        + 3 * n_variable
    )
    # COO triplets: ``lil_matrix.__setitem__`` is the slow sparse path and this
    # pattern is rebuilt for every decorated graph.
    rows: List[int] = []
    cols: List[int] = []
    row = 0

    def mark(row_index: int, atom: int) -> None:
        offset = variable_pos.get(atom)
        if offset is not None:
            rows.extend((row_index, row_index, row_index))
            cols.extend((3 * offset, 3 * offset + 1, 3 * offset + 2))

    for a, b in bond_list:
        mark(row, a)
        mark(row, b)
        row += 1
    for a, b, _floor in nonbond_floors:
        mark(row, a)
        mark(row, b)
        row += 1
    for order in ring_orders:
        for i in range(6):
            for atom in (order[i - 1], order[i], order[(i + 1) % 6]):
                mark(row, atom)
            row += 1
            for atom in (order[i - 1], order[i], order[(i + 1) % 6], order[(i + 2) % 6]):
                mark(row, atom)
            row += 1
    for center, first, second, third, _target in improper_terms:
        for atom in (center, first, second, third):
            mark(row, atom)
        row += 1
    for left, center, right, _target, _band in hard_angle_terms:
        for atom in (left, center, right):
            mark(row, atom)
        row += 1
    for atom in seed_list:
        offset = variable_pos[atom]
        for axis in range(3):
            rows.append(row)
            cols.append(3 * offset + axis)
            row += 1
    for centroid_row in range(row, row + 3):
        rows.extend([centroid_row] * (3 * n_variable))
        cols.extend(range(3 * n_variable))
    row += 3
    for atom in variable_nodes:
        for axis in range(3):
            offset = variable_pos[atom]
            rows.append(row)
            cols.append(3 * offset + axis)
            row += 1
    sparsity = coo_matrix(
        (np.ones(len(rows), dtype=int), (rows, cols)),
        shape=(n_rows, 3 * n_variable),
    )

    result = least_squares(
        residual,
        initial[variable_nodes].reshape(-1),
        method="trf",
        jac_sparsity=sparsity.tocsr(),
        tr_solver="lsmr",
        # Ring templates are already deterministic; this fit is only a short
        # closure polish, not an unconstrained optimizer.  Keep the cap small
        # so every decorated graph does not pay a hundreds-step solve.  Raising
        # it to 60 was measured to buy nothing once the acceptance band became
        # reachable: identical accepted and rejected sets at k=4,p=3 for twice
        # the wall time.
        max_nfev=25,
        ftol=1.0e-7,
        xtol=1.0e-7,
        gtol=1.0e-7,
    )
    return unpack(result.x)


def _free_tetrahedral_direction(
    center: FloatArray,
    occupied: Sequence[FloatArray],
    *,
    angle_deg: float = 109.47,
) -> FloatArray:
    """Unit vector for a new bond at ``center`` avoiding occupied neighbour dirs."""

    center = np.asarray(center, dtype=float)
    dirs: List[FloatArray] = []
    for p in occupied:
        v = np.asarray(p, dtype=float) - center
        n = float(np.linalg.norm(v))
        if n > 1.0e-8:
            dirs.append(v / n)
    if not dirs:
        return np.array([1.0, 0.0, 0.0])
    if len(dirs) == 1:
        # any perpendicular
        a = dirs[0]
        tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = np.cross(a, tmp)
        perp = perp / (float(np.linalg.norm(perp)) + 1.0e-15)
        # angle from -a
        ang = np.radians(float(angle_deg))
        d = -a * np.cos(ang) + perp * np.sin(ang)
        return d / (float(np.linalg.norm(d)) + 1.0e-15)
    # Point away from the mean of occupied directions, then orthogonalise
    mean = sum(dirs)
    mean_n = float(np.linalg.norm(mean))
    if mean_n < 1.0e-8:
        d = np.cross(dirs[0], dirs[1] if len(dirs) > 1 else np.array([0.0, 0.0, 1.0]))
        dn = float(np.linalg.norm(d))
        if dn < 1.0e-8:
            return np.array([0.0, 0.0, 1.0])
        return d / dn
    d = -mean / mean_n
    # remove components along occupied to stay off existing bonds
    for u in dirs:
        d = d - float(np.dot(d, u)) * u
    dn = float(np.linalg.norm(d))
    if dn < 1.0e-8:
        return np.array([0.0, 0.0, 1.0])
    return d / dn


def _place_remaining_with_pack_rules(
    state: _State,
    pack: GeometryPack,
    spec: NucleationSpec,
    degrees: Sequence[int],
    coords: FloatArray,
    placed: List[bool],
    inorganic_edges: Sequence[Tuple[int, int]],
) -> bool:
    """Place non-ring inorganic atoms using pack bond lengths + free angles.

    Ring atoms must already be ``placed``.  Candidates for each new atom are
    scored by exact pack bond lengths, applicable preferred torsions, and soft
    non-bonded clearance — so precursors do not land on top of the ring.
    """

    n = len(state.atoms)
    cations = {spec.core.cation, spec.precursor.center}
    anion = spec.core.anion
    g = state.graph
    skel_set = {
        (min(int(a), int(b)), max(int(a), int(b))) for a, b in inorganic_edges
    }
    ring_endocyclic: Set[Tuple[int, int]] = set()
    inorganic_graph = nx.Graph()
    inorganic_graph.add_edges_from(skel_set)
    for cycle in nx.cycle_basis(inorganic_graph):
        if len(cycle) != 6:
            continue
        for left, right in zip(cycle, cycle[1:] + cycle[:1]):
            ring_endocyclic.add((min(int(left), int(right)), max(int(left), int(right))))
    inorg_els = cations | {anion}

    def bond_r(i: int, j: int) -> float:
        return float(_molecular_bond_length(state, pack, spec, i, j, degrees))

    def min_nonbond(i: int, pos: FloatArray) -> float:
        best = 1.0e9
        for t in range(n):
            if not placed[t] or t == i:
                continue
            if state.atoms[t].symbol not in inorg_els:
                continue
            key = (min(i, t), max(i, t))
            if key in skel_set:
                continue
            d = float(np.linalg.norm(pos - coords[t]))
            if d < best:
                best = d
        return best

    def direction_samples(
        host: int, occ_pts: Sequence[FloatArray], ang: float
    ) -> List[FloatArray]:
        base = _free_tetrahedral_direction(
            coords[host], occ_pts, angle_deg=float(ang)
        )
        samples = [base, -base]
        # rotate base around each occupied bond for more free slots
        for p in occ_pts[:3]:
            axis = np.asarray(p, dtype=float) - coords[host]
            an = float(np.linalg.norm(axis))
            if an < 1.0e-8:
                continue
            axis = axis / an
            for deg in (60.0, 120.0, 180.0, 240.0, 300.0):
                phi = np.radians(deg)
                # Rodrigues
                v = base
                samples.append(
                    v * np.cos(phi)
                    + np.cross(axis, v) * np.sin(phi)
                    + axis * float(np.dot(axis, v)) * (1.0 - np.cos(phi))
                )
        out: List[FloatArray] = []
        for d in samples:
            dn = float(np.linalg.norm(d))
            if dn > 1.0e-8:
                out.append(d / dn)
        return out or [np.array([1.0, 0.0, 0.0])]

    changed = True
    guard = 0
    while changed and guard < n + 10:
        changed = False
        guard += 1
        # place atoms with most placed neighbours first (more constrained)
        pending = []
        for i in range(n):
            if placed[i]:
                continue
            if state.atoms[i].symbol not in inorg_els:
                continue
            nbrs = [
                j
                for j in g.neighbors(i)
                if placed[j] and state.atoms[j].symbol in inorg_els
            ]
            if nbrs:
                pending.append((len(nbrs), i, nbrs))
        pending.sort(reverse=True)
        for _npl, i, nbrs in pending:
            if placed[i]:
                continue
            r_targets = [bond_r(i, j) for j in nbrs]
            best_pos = None
            best_rank: Optional[Tuple[float, float, float, float]] = None

            def consider(candidate: FloatArray, residual: float = 0.0) -> None:
                nonlocal best_pos, best_rank
                soft = _candidate_soft_clearance_penalty(
                    state, pack, i, candidate, coords, placed
                )
                preferred = _candidate_torsion_penalty(
                    state, pack, i, candidate, coords, placed, ring_endocyclic
                )
                clearance = min_nonbond(i, candidate)
                rank_key = (preferred, soft, -clearance, float(residual))
                if best_rank is None or rank_key < best_rank:
                    best_rank = rank_key
                    best_pos = candidate

            if len(nbrs) >= 2:
                # try multilateration first
                try:
                    pos = _multilaterated_position(
                        [coords[j] for j in nbrs[:3]],
                        r_targets[:3],
                        coords[nbrs[0]]
                        + r_targets[0] * np.array([0.0, 0.0, 1.0]),
                        [coords[t] for t in range(n) if placed[t]],
                    )
                    consider(pos)
                except ExactEmbeddingError:
                    pass
            # also try free directions from each host
            for j, rt in zip(nbrs, r_targets):
                occ = [
                    coords[k]
                    for k in g.neighbors(j)
                    if placed[k]
                    and k != i
                    and state.atoms[k].symbol in inorg_els
                ]
                ang = pack.center_angle_deg(
                    state.atoms[j].symbol,
                    int(degrees[j]),
                    default=109.47,
                ) or 109.47
                for direction in direction_samples(j, occ, float(ang)):
                    pos = coords[j] + rt * direction
                    residual = 0.0
                    # small penalty if far from other hosts when multi-bonded
                    if len(nbrs) > 1:
                        for j2, rt2 in zip(nbrs, r_targets):
                            if j2 == j:
                                continue
                            residual += 0.1 * abs(
                                float(np.linalg.norm(pos - coords[j2])) - rt2
                            )
                    consider(pos, residual)
            if best_pos is None:
                continue
            coords[i] = best_pos
            placed[i] = True
            changed = True

    skel = [(int(a), int(b)) for a, b in inorganic_edges]
    _enforce_inorganic_bond_lengths(
        coords, skel, bond_r, max_iter=40, tol=3.0e-3
    )
    # Mild non-bonded push (do not destroy bonds): separates stacked atoms
    inorg = [
        i
        for i in range(n)
        if placed[i] and state.atoms[i].symbol in inorg_els
    ]
    for _ in range(30):
        moved = False
        for ii, i in enumerate(inorg):
            for j in inorg[ii + 1 :]:
                key = (min(i, j), max(i, j))
                if key in skel_set:
                    continue
                dvec = coords[j] - coords[i]
                d = float(np.linalg.norm(dvec))
                # Cd–Cd / Se–Se floors from pack pair rules if present
                floor = 2.5
                if state.atoms[i].symbol == state.atoms[j].symbol:
                    if state.atoms[i].symbol == "Cd":
                        floor = 3.0
                    elif state.atoms[i].symbol == "Se":
                        floor = 3.2
                if d < floor and d > 1.0e-8:
                    push = 0.5 * (floor - d) * (dvec / d)
                    coords[i] = coords[i] - push
                    coords[j] = coords[j] + push
                    moved = True
        if not moved:
            break
        _enforce_inorganic_bond_lengths(
            coords, skel, bond_r, max_iter=15, tol=5.0e-3
        )
    return _min_edge_length(coords, skel) >= 2.2



def skeleton_has_only_large_rings(
    edges: Sequence[Tuple[int, int]],
    k: int,
    p: int,
) -> bool:
    """True if there is a ≥8 Cd–Se cycle but **no** 6-ring (reject pure C8+).

    Fused 6-rings may enclose an 8-cycle perimeter — those still have n6≥1 and
    are kept.  Standalone 8-ring seeds are not in the construction policy.
    """

    if count_cdse_six_rings(edges, k, p) > 0:
        return False
    se_ids, cd_ids, _ = _index_blocks(k, p)
    g = nx.Graph()
    g.add_nodes_from(list(se_ids) + list(cd_ids))
    g.add_edges_from((int(a), int(b)) for a, b in edges)
    if g.number_of_edges() == 0:
        return False
    try:
        basis = nx.cycle_basis(g)
    except Exception:  # noqa: BLE001
        return False
    for cyc in basis:
        if len(cyc) >= 8 and len(cyc) % 2 == 0:
            return True
    return False


def filter_skeletons_six_ring_policy(
    skeletons: Sequence[Tuple[Tuple[int, int], ...]],
    k: int,
    p: int,
    *,
    require_six: bool,
) -> List[Tuple[Tuple[int, int], ...]]:
    """Keep 6-ring seeds/fusions; drop pure ≥8-ring graphs."""

    out: List[Tuple[Tuple[int, int], ...]] = []
    for skel in skeletons:
        n6 = count_cdse_six_rings(skel, k, p)
        if require_six and n6 < 1:
            continue
        if skeleton_has_only_large_rings(skel, k, p):
            continue
        out.append(skel)
    return out


def _enforce_inorganic_bond_lengths(
    coords: FloatArray,
    edges: Sequence[Tuple[int, int]],
    target_fn,
    *,
    max_iter: int = 80,
    tol: float = 1.0e-3,
) -> float:
    """SHAKE-like projection so every edge length matches ``target_fn(a,b)``.

    Returns the maximum |d − target| after the last iteration.
    """

    coords = np.asarray(coords, dtype=float)
    edge_list = [(int(a), int(b)) for a, b in edges]
    max_err = 0.0
    for _ in range(int(max_iter)):
        max_err = 0.0
        for a, b in edge_list:
            target = float(target_fn(a, b))
            if target < 1.5:
                target = 2.635
            dvec = coords[b] - coords[a]
            dn = float(np.linalg.norm(dvec))
            if dn < 1.0e-12:
                coords[b] = coords[a] + np.array([target, 0.0, 0.0])
                max_err = max(max_err, target)
                continue
            err = dn - target
            max_err = max(max_err, abs(err))
            # move each atom half-way along the bond
            corr = 0.5 * err * (dvec / dn)
            coords[a] = coords[a] + corr
            coords[b] = coords[b] - corr
        if max_err < tol:
            break
    return max_err


def _min_edge_length(
    coords: FloatArray, edges: Sequence[Tuple[int, int]]
) -> float:
    coords = np.asarray(coords, dtype=float)
    best = float("inf")
    for a, b in edges:
        d = float(np.linalg.norm(coords[int(b)] - coords[int(a)]))
        if d < best:
            best = d
    return best if best < float("inf") else 0.0


def _try_pack_ring_frame(
    state: _State,
    inorganic_edges: Sequence[Tuple[int, int]],
    k: int,
    p: int,
    pack: GeometryPack,
    spec: NucleationSpec,
    degrees: Sequence[int],
    *,
    conformation: str = "chair",
) -> Optional[Tuple[FloatArray, List[bool]]]:
    """Seed one pack-driven 6-ring; place remaining inorganic atoms by pack rules.

    Reconstruction policy
    ---------------------
    1. Pick one Cd–Se 6-ring and fit the configured bond, alternating Cd/Se
       angles, and the selected chair/boat DFT-derived dihedral sequence.
    2. Place **all remaining** inorganic atoms (other fused rings, precursor
       Cd, …) with pack bond lengths and free tetrahedral directions at each
       host — the same idea as ``_construct_inorganic``.  No Kabsch fusion of
       a second ring (that stacked atoms onto the first) and no “ray from
       centroid”.

    Rejects frames with skeleton bonds < 2.2 Å or non-bonded contacts < 2.0 Å.
    """

    if not ring_first_required_for_spec(k, p, spec):
        return None

    se_ids, cd_ids, _ = _index_blocks(k, p)
    se_set = set(se_ids)
    cd_set = set(cd_ids)
    rings = cdse_six_ring_sets(inorganic_edges, k, p)
    if not rings:
        return None

    pattern = pack.cdse6_ring_pattern()
    r_bond = float(pattern.bond_cdse_A)
    if r_bond < 2.0:
        r_bond = 2.635

    n = len(state.atoms)
    coords = np.zeros((n, 3), dtype=float)
    placed = [False] * n
    cations = {spec.core.cation, spec.precursor.center}
    anion = spec.core.anion

    # Seed only the first 6-ring (stable seed).  Other rings grow with pack rules.
    # The template is built per ring, because its angle targets are resolved
    # from that ring's own coordination numbers.
    rings_sorted = sorted(rings, key=lambda r: (-len(r), sorted(r)))
    seeded = False
    for ring in rings_sorted:
        order = _alternating_six_cycle_order(
            inorganic_edges, frozenset(ring), se_set, cd_set
        )
        if order is None:
            continue
        ring_angles, ring_dihedrals = ring_cn_targets(
            pack, order, inorganic_edges, se_set, conformation
        )
        # Ring bonds come from the same CN-indexed table as every other bond,
        # instead of one flat ``bond_cdse_A`` for the whole ring.  That flat
        # value (2.69) disagreed with the table (2.602 at cn3/cn3 to 2.717 at
        # cn4/cn4) by up to 0.09 A, and since the audit reads the table, the
        # disagreement was charged to every ring bond.  It was the dominant
        # rejection for *dense* candidates -- the ones with the most bonds have
        # the most chances to hit it -- which biased acceptance against exactly
        # the compact structures a growing crystal should favour.
        ring_bond_lengths = [
            float(
                _molecular_bond_length(
                    state, pack, spec, order[i], order[(i + 1) % 6], degrees
                )
            )
            for i in range(6)
        ]
        template = _cached_six_ring_template_positions(
            bond_length=r_bond,
            angle_at_cd_deg=float(pattern.angle_at_cd_deg),
            angle_at_se_deg=float(pattern.angle_at_se_deg),
            conformation=conformation,
            dihedrals_deg=ring_dihedrals,
            angles_deg=ring_angles,
            bond_lengths=ring_bond_lengths,
        )
        for i, node in enumerate(order):
            coords[node] = template[i]
            placed[node] = True
        seeded = True
        break
    if not seeded:
        return None

    ok = _place_remaining_with_pack_rules(
        state,
        pack,
        spec,
        degrees,
        coords,
        placed,
        inorganic_edges,
    )
    if not ok:
        return None

    inorg = [
        i
        for i in range(n)
        if state.atoms[i].symbol in cations | {anion}
        and int(state.graph.degree[i]) > 0
    ]
    if not all(placed[i] for i in inorg):
        return None

    # Reject only true collapse (non-bonded < 1.5 Å after repair)
    skel = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in inorganic_edges}
    for i in inorg:
        for j in inorg:
            if j <= i:
                continue
            if (min(i, j), max(i, j)) in skel:
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < 1.5:
                return None
    return coords, placed



def embed_skeleton_frames(
    state: _State,
    inorganic_edges: Sequence[Tuple[int, int]],
    k: int,
    p: int,
    pack: GeometryPack,
    spec: NucleationSpec,
    *,
    limit: int = 1,
    max_slot_orders: int = 24,
    allow_ring_template_fallback: bool = True,
) -> Tuple[List[Tuple[FloatArray, List[bool]]], Optional[List[str]], List[int]]:
    """Build clean inorganic frames with ring-aware CN templates.

    Tries, in order:

    1. :func:`frame_degrees_for_skeleton` (ring Cd/Se → full-pattern CN tables)
    2. ring Cd/Se → CN3
    3. all bonded Cd ≥ 3, Se at skeleton (capped ≤ 4)
    4. Pack **chair** then **boat** seeds (only if min ring pattern is
       compositionally possible and 6-rings are present)

    Returns ``(frames, failure_reasons_or_None, degrees_used)``.
    """

    n = len(state.atoms)
    se_ids, cd_ids, _ = _index_blocks(k, p)
    # Prefer pack min-pattern CN (Cd[3,3,4]-like) for ring members.
    pat = pack.cdse6_ring_pattern()
    ring_cd = max(int(x) for x in pat.cd_cn)
    ring_se = max(int(x) for x in pat.se_cn)
    base = frame_degrees_for_skeleton(
        state,
        inorganic_edges,
        k,
        p,
        spec,
        ring_cd_cn=ring_cd,
        ring_se_cn=ring_se,
    )
    ring3 = frame_degrees_for_skeleton(
        state, inorganic_edges, k, p, spec, ring_cd_cn=3, ring_se_cn=3
    )
    candidates: List[List[int]] = [base]
    if ring3 != base:
        candidates.append(ring3)

    core3 = [int(state.graph.degree[i]) for i in range(n)]
    for i in cd_ids:
        if core3[i] > 0:
            core3[i] = max(core3[i], 3)
    for i in se_ids:
        if core3[i] > 0:
            core3[i] = min(4, max(core3[i], 2))
    if core3 not in candidates:
        candidates.append(core3)

    last_fail: Optional[List[str]] = None
    last_deg = base
    for deg in candidates:
        last_deg = deg
        built, fail = _clean_frames(
            state,
            pack,
            spec,
            deg,
            max_slot_orders=max_slot_orders,
            limit=limit,
        )
        if built:
            return built, None, deg
        last_fail = list(fail or ["frame_not_realisable"])

    # Chair/boat only when composition allows min full-CN ring pattern.
    if (
        allow_ring_template_fallback
        and ring_first_required_for_spec(k, p, spec)
        and cdse_six_ring_atom_ids(inorganic_edges, k, p)
    ):
        frames_out: List[Tuple[FloatArray, List[bool]]] = []
        for conf in pack.cdse6_conformations():
            built_ring = _try_pack_ring_frame(
                state,
                inorganic_edges,
                k,
                p,
                pack,
                spec,
                last_deg,
                conformation=conf,
            )
            if built_ring is not None:
                frames_out.append(built_ring)
                if 0 < limit <= len(frames_out):
                    return frames_out, None, last_deg
        if frames_out:
            return frames_out, None, last_deg

    return [], last_fail, last_deg




def _one_ring_base_masks(pack: Optional[GeometryPack] = None) -> List[int]:
    """Single-ring seed: Cd0(Se0,Se2), Cd1(Se0,Se1), Cd2(Se1,Se2).

    Read from ``motifs.ring6`` when the pack declares it, so which motif seeds
    a run is a pack decision rather than a code edit; the literal below is the
    fallback for packs written before that section existed.
    """

    if pack is not None:
        masks = pack.motif_masks("ring6")
        if masks:
            return masks
    return [
        (1 << 0) | (1 << 2),
        (1 << 0) | (1 << 1),
        (1 << 1) | (1 << 2),
    ]


def _two_ring_base_mask_sets(
    pack: Optional[GeometryPack] = None,
) -> List[Tuple[str, List[int], int]]:
    """Zinc-blende-compatible fusion modes as seed mask lists (no ranking).

    Each entry: (name, base_masks for Cd 0..n_seed-1, min_n_se).
    Ring1 always on Cd0–2 / Se0–2.  Second ring fused by mode.

    Which fusions a zinc-blende lattice actually contains was measured on
    ``examples/cifs/CdSe_zb.cif`` (3x3x3 supercell, 6-rings whose every member
    is CN4), counting shared atoms and shared bonds for each ring pair:

        corner / spiro   1 atom , 0 bonds   x12   present
        edge-fused       2 atoms, 1 bond    x36   present
        path-fused       3 atoms, 2 bonds   x30   present
        face-fused       4 atoms, 3 bonds   x 0   ABSENT

    So only ``path`` and ``edge`` are generated here.  Face-sharing two 6-rings
    across a 4-atom path builds a bicyclo[2.2.2] cage, whose two bridges force
    boat rings; zinc-blende is all-chair, and boats belong to wurtzite.  Note
    ``min_ring_size: Cd-Se: 6`` does not screen it out, because face-fusion
    creates no 4-ring -- it has to be excluded at the seed.
    """

    # ``min_n_se`` is not stated in the pack -- it is just how many Se the
    # motif references, so derive it and keep the two from drifting apart.
    if pack is not None:
        declared = pack.fusion_motifs()
        if declared:
            return [
                (name, masks, max(int(m).bit_length() for m in masks))
                for name, masks in declared
            ]

    r1 = _one_ring_base_masks(pack)
    # Path-share Cd1Se2: share Se0–Cd1–Se1; +Cd3,Cd4 +Se3
    # Ring2: Cd1-Se0-Cd3-Se3-Cd4-Se1-Cd1
    path = [
        r1[0],  # Cd0
        r1[1],  # Cd1 shared path
        r1[2],  # Cd2
        (1 << 0) | (1 << 3),  # Cd3: Se0, Se3
        (1 << 1) | (1 << 3),  # Cd4: Se1, Se3
    ]
    # Edge-share CdSe: share Cd0-Se0 only; +Cd3,Cd4 +Se3,Se4
    # Ring2: Cd0-Se0-Cd3-Se3-Cd4-Se4-Cd0
    edge = [
        (1 << 0) | (1 << 2) | (1 << 4),  # Cd0: Se0,Se2 + Se4
        r1[1],
        r1[2],
        (1 << 0) | (1 << 3),  # Cd3: Se0, Se3
        (1 << 3) | (1 << 4),  # Cd4: Se3, Se4
    ]
    return [
        ("path_Cd1Se2", path, 4),
        ("edge_CdSe", edge, 5),
    ]


def _skeleton_params(
    k: int,
    p: int,
    spec: NucleationSpec,
    *,
    extra_skeleton_edges: Optional[int],
) -> Optional[dict]:
    """Shared degree/edge bounds for free and ring-first skeleton enum."""

    se_ids, cd_ids, _cl_ids = _index_blocks(k, p)
    se_list = list(se_ids)
    cd_list = list(cd_ids)
    max_cd = int(spec.graph_rules.max_cn[spec.core.cation])
    max_se = int(spec.graph_rules.max_cn[spec.core.anion])
    min_cd = int(spec.graph_rules.min_cn.get(spec.core.cation, 0))
    min_se = int(spec.graph_rules.min_cn.get(spec.core.anion, 0))
    if not spec.enforce_min_cn:
        min_cd = max(1, min_cd) if min_cd else 1
        min_se = max(1, min_se) if min_se else 1
    min_cd = max(1, min(min_cd, 1))
    min_se = max(1, min_se)
    n_cd = len(cd_list)
    n_se = len(se_list)
    if n_cd == 0 or n_se == 0:
        return None
    min_edges = n_cd + n_se - 1
    hard_capacity = min(n_cd * n_se, n_se * max_se, n_cd * max_cd)
    if hard_capacity < min_edges:
        return {"empty": True}
    max_edges = (
        hard_capacity
        if extra_skeleton_edges is None
        else min(hard_capacity, min_edges + int(extra_skeleton_edges))
    )
    if max_edges < min_edges:
        return {"empty": True}
    min_ring = _cdse_min_ring(spec)
    forbid_c4 = min_ring is not None and min_ring > 4
    truncated_cap = max_edges < hard_capacity and (
        extra_skeleton_edges is not None
        and max_edges == min_edges + int(extra_skeleton_edges)
    )
    return {
        "empty": False,
        "se_list": se_list,
        "cd_list": cd_list,
        "max_cd": max_cd,
        "max_se": max_se,
        "min_cd": min_cd,
        "min_se": min_se,
        "n_cd": n_cd,
        "n_se": n_se,
        "min_edges": min_edges,
        "max_edges": max_edges,
        "hard_capacity": hard_capacity,
        "forbid_c4": forbid_c4,
        "truncated_cap": truncated_cap,
        "symbols": _symbols_for_composition(spec, k, p),
    }


def _rows_to_global_edges(
    rows: Sequence[int],
    cd_list: Sequence[int],
    se_list: Sequence[int],
) -> Tuple[Tuple[int, int], ...]:
    edges: List[Tuple[int, int]] = []
    for c_local, mask in enumerate(rows):
        m = int(mask)
        s_local = 0
        while m:
            if m & 1:
                a, b = cd_list[c_local], se_list[s_local]
                edges.append((min(a, b), max(a, b)))
            m >>= 1
            s_local += 1
    return tuple(sorted(edges))


def _enumerate_inorganic_edge_sets_free(
    k: int,
    p: int,
    spec: NucleationSpec,
    *,
    max_skeletons: int = 5000,
    extra_skeleton_edges: Optional[int] = None,
) -> Tuple[List[Tuple[Tuple[int, int], ...]], bool]:
    """Orderly C4-free bipartite enum (open skeletons allowed)."""

    params = _skeleton_params(
        k, p, spec, extra_skeleton_edges=extra_skeleton_edges
    )
    if params is None:
        return [()], False
    if params.get("empty"):
        return [], False

    se_list = params["se_list"]
    cd_list = params["cd_list"]
    max_cd = params["max_cd"]
    max_se = params["max_se"]
    min_cd = params["min_cd"]
    min_se = params["min_se"]
    n_cd = params["n_cd"]
    n_se = params["n_se"]
    min_edges = params["min_edges"]
    max_edges = params["max_edges"]
    forbid_c4 = params["forbid_c4"]
    truncated = bool(params["truncated_cap"])
    full_mask = (1 << n_se) - 1
    found: List[Tuple[Tuple[int, int], ...]] = []
    seen_cert: Set[Tuple[object, ...]] = set()
    symbols = params["symbols"]

    def emit_from_rows(rows: List[int]) -> None:
        nonlocal truncated
        if len(found) >= max_skeletons:
            truncated = True
            return
        # ``rec`` already emits Cd rows in non-increasing order, so the only
        # duplicates left are Se-column permutations.  The former orderly test
        # -- keep the row code only if it is already the max over all n_se!
        # column permutations -- was exact but cost ~712 us per candidate
        # against ~184 us for an exact certificate, and it degrades as n_se!
        # grows (89% of free-path time at k=5).  Deduplicating on the
        # certificate keeps the same one-representative-per-isomorphism-class
        # guarantee, and matches what the seeded path already does.
        inorg = list(cd_list) + list(se_list)
        remap = {old: index for index, old in enumerate(inorg)}
        edges = _rows_to_global_edges(rows, cd_list, se_list)
        cert = canonical_form(
            [symbols[index] for index in inorg],
            [(remap[a], remap[b], "bond") for a, b in edges],
        ).certificate
        if cert in seen_cert:
            return
        seen_cert.add(cert)
        found.append(edges)

    def rec(
        cd_idx: int,
        rows: List[int],
        se_deg: List[int],
        n_edges: int,
        prev_mask: int,
    ) -> None:
        nonlocal truncated
        if truncated or len(found) >= max_skeletons:
            truncated = truncated or len(found) >= max_skeletons
            return
        if cd_idx == n_cd:
            if n_edges < min_edges or n_edges > max_edges:
                return
            if any(d < min_se for d in se_deg):
                return
            if not _bipartite_connected(rows, n_cd, n_se):
                return
            emit_from_rows(rows)
            return
        remaining_cd = n_cd - cd_idx
        min_need = n_edges + remaining_cd * min_cd
        max_can = n_edges + remaining_cd * max_cd
        if min_need > max_edges or max_can < min_edges:
            return
        se_deficit = sum(max(0, min_se - d) for d in se_deg)
        if se_deficit > remaining_cd * max_cd:
            return
        mask_hi = prev_mask if cd_idx > 0 else full_mask
        max_this = min(
            max_cd,
            n_se,
            max_edges - n_edges - (remaining_cd - 1) * min_cd,
        )
        min_this = max(
            min_cd,
            min_edges - n_edges - (remaining_cd - 1) * max_cd,
        )
        if min_this > max_this:
            return
        for mask in range(mask_hi, -1, -1):
            d = _popcount(mask)
            if d < min_this or d > max_this:
                continue
            ok_cols = True
            s, mtmp = 0, mask
            while mtmp:
                if mtmp & 1 and se_deg[s] + 1 > max_se:
                    ok_cols = False
                    break
                mtmp >>= 1
                s += 1
            if not ok_cols:
                continue
            if forbid_c4 and any(
                _popcount(int(prev) & mask) > 1 for prev in rows
            ):
                continue
            s, mtmp = 0, mask
            while mtmp:
                if mtmp & 1:
                    se_deg[s] += 1
                mtmp >>= 1
                s += 1
            rows.append(mask)
            rec(cd_idx + 1, rows, se_deg, n_edges + d, mask)
            rows.pop()
            s, mtmp = 0, mask
            while mtmp:
                if mtmp & 1:
                    se_deg[s] -= 1
                mtmp >>= 1
                s += 1
            if truncated or len(found) >= max_skeletons:
                return

    rec(0, [], [0] * n_se, 0, full_mask)
    if len(found) >= max_skeletons:
        truncated = True
    return found, truncated


def _grow_skeletons_from_base_masks(
    k: int,
    p: int,
    spec: NucleationSpec,
    base_masks: Sequence[int],
    *,
    max_skeletons: int = 5000,
    extra_skeleton_edges: Optional[int] = None,
    found: Optional[List[Tuple[Tuple[int, int], ...]]] = None,
    seen_cert: Optional[Set[Tuple[object, ...]]] = None,
    truncated_in: bool = False,
) -> Tuple[List[Tuple[Tuple[int, int], ...]], bool]:
    """Grow Cd rows as supersets of ``base_masks``, then free remaining Cd."""

    params = _skeleton_params(
        k, p, spec, extra_skeleton_edges=extra_skeleton_edges
    )
    if params is None:
        return ([()] if found is None else found), False
    if params.get("empty"):
        return ([] if found is None else found), truncated_in

    se_list = params["se_list"]
    cd_list = params["cd_list"]
    max_cd = params["max_cd"]
    max_se = params["max_se"]
    min_cd = params["min_cd"]
    min_se = params["min_se"]
    n_cd = params["n_cd"]
    n_se = params["n_se"]
    min_edges = params["min_edges"]
    max_edges = params["max_edges"]
    forbid_c4 = params["forbid_c4"]
    truncated = bool(params["truncated_cap"]) or truncated_in
    symbols = params["symbols"]
    full_mask = (1 << n_se) - 1
    n_seed = len(base_masks)
    if n_cd < n_seed or n_se < 1:
        return ([] if found is None else found), truncated

    if found is None:
        found = []
    if seen_cert is None:
        seen_cert = set()
    seen_key: Set[Tuple[int, ...]] = set()
    forced_se_mask = 0
    for base_mask in base_masks:
        forced_se_mask |= int(base_mask)
    se_minimums = [
        max(min_se, 3) if forced_se_mask & (1 << index) else min_se
        for index in range(n_se)
    ]

    # The Se the seed pins are fixed, but the remaining ("free") Se columns are
    # interchangeable, and the row ordering imposed by ``rec_free`` says nothing
    # about them.  Permuting those columns is a graph isomorphism, so the max
    # over that subgroup of the sorted row masks is a genuine canonical form and
    # a sound key to deduplicate on -- and it costs a few table lookups where
    # ``canonical_form`` costs ~250 us.  At k=6,p=3 this collapses 131163 row
    # sets to 22369 before the exact check runs.
    #
    # It must be a *cache key*, not an orderly rejection test: requiring the
    # rows non-increasing and the columns non-increasing at the same time can
    # reject every representative of a class, because sorting the columns
    # unsorts the rows.
    free_se_positions = [
        index for index in range(n_se) if not (forced_se_mask & (1 << index))
    ]
    column_tables: List[List[int]] = []
    if 1 < len(free_se_positions) <= 6:
        for perm in permutations(free_se_positions):
            bit_of = dict(zip(free_se_positions, perm))
            table = [0] * (1 << n_se)
            for mask in range(1 << n_se):
                remapped = 0
                for bit in range(n_se):
                    if mask & (1 << bit):
                        remapped |= 1 << bit_of.get(bit, bit)
                table[mask] = remapped
            column_tables.append(table)

    def free_se_key(rows: Sequence[int]) -> Tuple[int, ...]:
        best: Optional[Tuple[int, ...]] = None
        for table in column_tables:
            candidate = sorted((table[mask] for mask in rows), reverse=True)
            as_tuple = tuple(candidate)
            if best is None or as_tuple > best:
                best = as_tuple
        return best if best is not None else tuple(sorted(rows, reverse=True))

    def emit(rows: List[int]) -> None:
        nonlocal truncated
        if len(found) >= max_skeletons:
            truncated = True
            return
        # Local to this seed set: a different base-mask set pins different Se,
        # so its keys are computed over a different subgroup and the two are not
        # comparable.  Cross-seed-set dedup stays with ``seen_cert``.
        key = free_se_key(rows)
        if key in seen_key:
            return
        seen_key.add(key)
        edges = _rows_to_global_edges(rows, cd_list, se_list)
        inorg = list(cd_list) + list(se_list)
        remap = {old: i for i, old in enumerate(inorg)}
        labels = [symbols[i] for i in inorg]
        cert_edges = [(remap[a], remap[b], "bond") for a, b in edges]
        cert = canonical_form(labels, cert_edges).certificate
        if cert in seen_cert:
            return
        seen_cert.add(cert)
        found.append(edges)

    def c4_ok(rows: Sequence[int], mask: int) -> bool:
        # ~2M calls per bin at k=6: call ``int.bit_count`` directly rather than
        # through ``_popcount``, whose wrapper costs ~2.8x the bare method.
        if not forbid_c4:
            return True
        return all((prev & mask).bit_count() <= 1 for prev in rows)

    def cols_ok(se_deg: Sequence[int], mask: int) -> bool:
        s, mtmp = 0, mask
        while mtmp:
            if mtmp & 1 and se_deg[s] + 1 > max_se:
                return False
            mtmp >>= 1
            s += 1
        return True

    def apply_mask(se_deg: List[int], mask: int, sign: int) -> None:
        s, mtmp = 0, mask
        while mtmp:
            if mtmp & 1:
                se_deg[s] += sign
            mtmp >>= 1
            s += 1

    def rec_free(
        cd_idx: int,
        rows: List[int],
        se_deg: List[int],
        n_edges: int,
        prev_mask: int,
    ) -> None:
        """Fill the Cd rows past the seed block, in non-increasing mask order.

        The Cd beyond the seed are mutually interchangeable -- every per-Cd
        bound (``min_cd``/``max_cd``, ``cols_ok``, ``c4_ok``) is symmetric in
        the Cd index, and ``se_minimums`` is indexed by Se -- so permuting
        their rows maps valid completions to valid completions.  Requiring the
        rows non-increasing therefore still reaches every isomorphism class
        while generating each one once instead of ``n_free!`` times.  Without
        it, k=5,p=3 emitted 170520 row sets for 118 distinct skeletons and
        k=6 did not finish at all.  This mirrors what ``rec`` already does in
        ``_enumerate_inorganic_edge_sets_free``.
        """

        nonlocal truncated
        if truncated or len(found) >= max_skeletons:
            truncated = True
            return
        remaining_cd = n_cd - cd_idx
        if any(
            se_deg[index] + remaining_cd < se_minimums[index]
            for index in range(n_se)
        ):
            return
        if cd_idx == n_cd:
            if n_edges < min_edges or n_edges > max_edges:
                return
            if any(d < se_minimums[i] for i, d in enumerate(se_deg)):
                return
            if not _bipartite_connected(rows, n_cd, n_se):
                return
            emit(rows)
            return
        min_need = n_edges + remaining_cd * min_cd
        max_can = n_edges + remaining_cd * max_cd
        if min_need > max_edges or max_can < min_edges:
            return
        se_deficit = sum(
            max(0, se_minimums[i] - d) for i, d in enumerate(se_deg)
        )
        if se_deficit > remaining_cd * max_cd:
            return
        max_this = min(
            max_cd,
            n_se,
            max_edges - n_edges - (remaining_cd - 1) * min_cd,
        )
        min_this = max(
            min_cd,
            min_edges - n_edges - (remaining_cd - 1) * max_cd,
        )
        if min_this > max_this:
            return
        for mask in range(min(prev_mask, full_mask), -1, -1):
            d = mask.bit_count()
            if d < min_this or d > max_this:
                continue
            if not cols_ok(se_deg, mask):
                continue
            if not c4_ok(rows, mask):
                continue
            apply_mask(se_deg, mask, +1)
            rows.append(mask)
            rec_free(cd_idx + 1, rows, se_deg, n_edges + d, mask)
            rows.pop()
            apply_mask(se_deg, mask, -1)
            if truncated or len(found) >= max_skeletons:
                return

    def rec_seed(
        seed_idx: int,
        rows: List[int],
        se_deg: List[int],
        n_edges: int,
    ) -> None:
        nonlocal truncated
        if truncated or len(found) >= max_skeletons:
            truncated = True
            return
        remaining_cd = n_cd - seed_idx
        if any(
            se_deg[index] + remaining_cd < se_minimums[index]
            for index in range(n_se)
        ):
            return
        if seed_idx == n_seed:
            # The free block starts unconstrained; only its own rows are
            # ordered relative to each other, never against the seed rows.
            rec_free(n_seed, rows, se_deg, n_edges, full_mask)
            return
        base = int(base_masks[seed_idx])
        # Base may reference Se indices ≥ n_se if seed needs more Se than available
        if base >= (1 << n_se):
            return
        free_bits = full_mask ^ base
        free_pos = [i for i in range(n_se) if free_bits & (1 << i)]
        n_free = len(free_pos)
        max_extra = max_cd - _popcount(base)
        if max_extra < 0:
            return

        def sub_rec(fi: int, extra_mask: int, extra_n: int) -> None:
            nonlocal truncated
            if truncated or len(found) >= max_skeletons:
                return
            if fi == n_free:
                mask = base | extra_mask
                d = _popcount(mask)
                if not cols_ok(se_deg, mask):
                    return
                if not c4_ok(rows, mask):
                    return
                apply_mask(se_deg, mask, +1)
                rows.append(mask)
                rec_seed(seed_idx + 1, rows, se_deg, n_edges + d)
                rows.pop()
                apply_mask(se_deg, mask, -1)
                return
            sub_rec(fi + 1, extra_mask, extra_n)
            if extra_n < max_extra:
                sub_rec(
                    fi + 1,
                    extra_mask | (1 << free_pos[fi]),
                    extra_n + 1,
                )

        sub_rec(0, 0, 0)

    rec_seed(0, [], [0] * n_se, 0)
    if len(found) >= max_skeletons:
        truncated = True
    return found, truncated


def _enumerate_inorganic_edge_sets_ring_first(
    k: int,
    p: int,
    spec: NucleationSpec,
    *,
    max_skeletons: int = 5000,
    extra_skeleton_edges: Optional[int] = None,
    pack: Optional[GeometryPack] = None,
) -> Tuple[List[Tuple[Tuple[int, int], ...]], bool]:
    """Build skeletons from a closed Cd3Se3 seed, then attach remaining atoms."""

    if k + p < 3 or k < 3:
        return _enumerate_inorganic_edge_sets_free(
            k,
            p,
            spec,
            max_skeletons=max_skeletons,
            extra_skeleton_edges=extra_skeleton_edges,
        )
    return _grow_skeletons_from_base_masks(
        k,
        p,
        spec,
        _one_ring_base_masks(pack),
        max_skeletons=max_skeletons,
        extra_skeleton_edges=extra_skeleton_edges,
    )


def _enumerate_inorganic_edge_sets_fused2(
    k: int,
    p: int,
    spec: NucleationSpec,
    *,
    max_skeletons: int = 5000,
    extra_skeleton_edges: Optional[int] = None,
    pack: Optional[GeometryPack] = None,
) -> Tuple[List[Tuple[Tuple[int, int], ...]], bool]:
    """Two fused 6-rings: union of ALL fusion modes (path, face, edge), deduped.

    No ranking among modes — passivation decides survivors.
    """

    if not two_ring_possible(k, p):
        return [], False
    found: List[Tuple[Tuple[int, int], ...]] = []
    seen: Set[Tuple[object, ...]] = set()
    truncated = False
    for _name, masks, min_se in _two_ring_base_mask_sets(pack):
        if k < min_se or (k + p) < len(masks):
            continue
        found, truncated = _grow_skeletons_from_base_masks(
            k,
            p,
            spec,
            masks,
            max_skeletons=max_skeletons,
            extra_skeleton_edges=extra_skeleton_edges,
            found=found,
            seen_cert=seen,
            truncated_in=truncated,
        )
        if truncated or len(found) >= max_skeletons:
            break
    return found, truncated


def _enumerate_inorganic_edge_sets(
    k: int,
    p: int,
    spec: NucleationSpec,
    *,
    max_skeletons: int = 5000,
    extra_skeleton_edges: Optional[int] = None,
    mode: Optional[str] = None,
    pack: Optional[GeometryPack] = None,
) -> Tuple[List[Tuple[Tuple[int, int], ...]], bool]:
    """Enumerate unique Cd–Se edge sets (connected bipartite, degree bounds).

    ``mode``:
      - ``free`` — orderly C4-free enum (open graphs allowed)
      - ``ring_first`` — one closed Cd3Se3 seed then grow
      - ``fused2`` — all 2-ring fusion modes then grow
      - ``auto`` — ring-first when pattern possible, else free
    """

    resolved = (mode or "auto").strip().lower()
    if resolved in {"", "auto"}:
        if ring_first_required_for_spec(k, p, spec):
            resolved = "ring_first"
        else:
            resolved = "free"
    if resolved == "fused2":
        found, truncated = _enumerate_inorganic_edge_sets_fused2(
            k,
            p,
            spec,
            max_skeletons=max_skeletons,
            extra_skeleton_edges=extra_skeleton_edges,
            pack=pack,
        )
        # 6-ring seeds/fusions only; drop pure ≥8-ring graphs (no 6-ring).
        found = filter_skeletons_six_ring_policy(
            found, k, p, require_six=True
        )
        return found, truncated
    if resolved == "ring_first":
        found, truncated = _enumerate_inorganic_edge_sets_ring_first(
            k,
            p,
            spec,
            max_skeletons=max_skeletons,
            extra_skeleton_edges=extra_skeleton_edges,
            pack=pack,
        )
        found = filter_skeletons_six_ring_policy(
            found, k, p, require_six=True
        )
        return found, truncated
    if resolved == "free":
        found, truncated = _enumerate_inorganic_edge_sets_free(
            k,
            p,
            spec,
            max_skeletons=max_skeletons,
            extra_skeleton_edges=extra_skeleton_edges,
        )
        # Free path policy:
        # - When min full-CN pattern is possible, free is usually a *fallback*
        #   after ring_first/fused2: prefer acyclic (true open) skeletons.
        # - When pattern is impossible (e.g. k=3 p=0), free is the primary
        #   mode.  min_cn≥2 may force a 6-cycle graph; that is NOT ring_first
        #   chemistry (no chair/boat pack tables) — DFT opens bare Cd2Se2.
        # Never keep pure ≥8-ring graphs (no 6-ring seed).
        found = filter_skeletons_six_ring_policy(
            found, k, p, require_six=False
        )
        open_only = [
            edges
            for edges in found
            if count_cdse_six_rings(edges, k, p) == 0
        ]
        if ring_first_required_for_spec(k, p, spec):
            return (open_only if open_only else found), truncated
        return found, truncated
    raise ValueError(
        f"unknown skeleton mode {mode!r}; use free|ring_first|fused2|auto"
    )


@dataclass
class _DecorationStatus:
    """Counters shared with a streaming decoration generator.

    The pruning happens deep inside the recursion, so the caller can only
    report where candidates went if the generator tallies it as it goes.
    """

    truncated: bool = False
    #: skeletons whose automorphism group exceeded
    #: ``bridge_first_max_automorphisms`` and were keyed by identity instead
    automorphism_cap_hits: int = 0
    #: partial assignments skipped because an equivalent one was already
    #: expanded under a symmetry of the skeleton
    symmetry_pruned: int = 0
    #: partial assignments reached again by a different insertion order
    revisited: int = 0
    #: partial assignments dropped because no completion could satisfy the
    #: coordination floors / exact surplus cover
    infeasible: int = 0
    #: partial assignments dropped on a coordination ceiling or bridge cap
    over_capacity: int = 0
    #: partial assignments dropped because no common frame realises the
    #: chosen bridges (sphere intersection conditioned on final CN)
    geometry_pruned: int = 0
    modes_total: int = 0
    modes_kept: int = 0
    automorphisms: int = 1
    degree_slices: int = 0
    frames_built: int = 0
    degree_vectors_total: int = 0
    degree_vectors_used: int = 0


@dataclass(frozen=True)
class _DegreeSlice:
    """One final Cd coordination vector and the modes/frames it admits.

    Decorations are enumerated *per slice* so bridge radii, mode alphabets and
    exact surplus covers all refer to the same final CN vector that the
    embedder will use.
    """

    degree: Tuple[int, ...]
    #: mode -> frame indices on which the mode is sphere-feasible
    mode_frames: Dict[Tuple[int, ...], frozenset]


def iter_cl_attachments(
    k: int,
    p: int,
    inorganic_edges: Sequence[Tuple[int, int]],
    spec: NucleationSpec,
    *,
    max_assignments: int = 0,
    status: Optional[_DecorationStatus] = None,
    mode_support: Optional[Dict[Tuple[int, ...], Set[Tuple[int, ...]]]] = None,
    degree_slices: Optional[Sequence[_DegreeSlice]] = None,
    degree_vectors: Optional[Sequence[Tuple[int, ...]]] = None,
    slice_builder: Optional[
        Callable[[Tuple[int, ...]], Optional[_DegreeSlice]]
    ] = None,
    required_degree_profiles: Optional[Sequence[Tuple[int, ...]]] = None,
    bridge_first_order: bool = False,
) -> Iterable[Tuple[Tuple[int, int], ...]]:
    """Stream symmetry-distinct Cl placements on one inorganic skeleton.

    Each of 2p Cl atoms is either:
    - terminal: one edge to a Cd
    - μ₂ bridge: two edges to a Cd–Cd pair (not necessarily bonded by Se)
    - μ₃ bridge: three edges to a Cd triplet

    Degree-vector-first geometry uses either:

    * ``degree_slices`` — prebuilt slices (tests / eager survey), or
    * ``degree_vectors`` + ``slice_builder`` — **lazy** frames: a CN vector's
      inorganic frames are built only when that orbit-minimum vector is entered,
      which avoids hundreds of unused frame constructions at k=2, p≥4.

    Each slice contributes only host modes that are sphere-feasible on some
    clean frame of that final CN vector, and every emitted decoration realises
    that vector exactly.  Mid-tree, the intersection of per-mode frame supports
    must stay nonempty so a single frame can still host the whole multiset.

    Set ``status.truncated`` when ``max_assignments`` binds.  Streaming matters:
    a single ``k=2, p=4`` skeleton yields many decorations, and materialising
    them costs memory the caller does not need.
    """

    if status is None:
        status = _DecorationStatus()
    if p == 0:
        yield ()
        return

    se_ids, cd_ids, cl_ids = _index_blocks(k, p)
    cd_list = list(cd_ids)
    cl_list = list(cl_ids)
    n_cl = len(cl_list)
    n_cd = len(cd_list)
    max_cd = int(spec.graph_rules.max_cn[spec.core.cation])
    max_cl = int(spec.graph_rules.max_cn[spec.precursor.ligand])
    bridge_cap = int(
        spec.graph_rules.max_shared_ligands_per_host_pair
        or spec.bridges_per_cd_pair
    )
    min_cd = (
        int(spec.graph_rules.min_cn.get(spec.core.cation, 0))
        if spec.enforce_min_cn
        else 0
    )

    position = {host: index for index, host in enumerate(cd_list)}
    base_degrees = [0] * n_cd
    for left, right in inorganic_edges:
        if left in position:
            base_degrees[position[left]] += 1
        if right in position:
            base_degrees[position[right]] += 1

    allowed_signatures = set(
        spec.graph_rules.allowed_neighbor_signatures.get(
            spec.precursor.ligand, ()
        )
    )

    def signature_allowed(host_count: int) -> bool:
        return (
            not allowed_signatures
            or f"{spec.core.cation}{host_count}" in allowed_signatures
        )

    # Full chemical alphabet before geometry.  Every mode stays available at
    # every depth within a fixed alphabet, so the state is a multiset and orbit
    # pruning under skeleton automorphisms is sound.
    all_modes: List[Tuple[int, ...]] = []
    for host_count in range(1, min(max_cl, n_cd) + 1):
        if signature_allowed(host_count):
            all_modes.extend(combinations(cd_list, host_count))
    if bridge_first_order:
        all_modes.sort(key=lambda mode: (-len(mode), mode))
    status.modes_total += len(all_modes)
    if not all_modes:
        return

    inorganic = nx.Graph()
    inorganic.add_nodes_from(
        (node, {"element": spec.core.anion}) for node in se_ids
    )
    inorganic.add_nodes_from(
        (node, {"element": spec.core.cation}) for node in cd_list
    )
    inorganic.add_edges_from(inorganic_edges)
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        inorganic,
        inorganic,
        node_match=nx.algorithms.isomorphism.categorical_node_match(
            "element", ""
        ),
    )
    host_maps = [
        {host: mapping[host] for host in cd_list}
        for mapping in matcher.isomorphisms_iter()
    ]

    def close_mode_set(
        modes: Sequence[Tuple[int, ...]],
    ) -> List[Tuple[int, ...]]:
        """Aut-close a mode set so orbit reduction stays well defined.

        Only *adds* modes.  For degree-first slices the extra modes are still
        subject to the exact surplus cover and (when present) frame-support
        intersection, so a mode that is not sphere-feasible on any remaining
        frame is dropped mid-tree rather than silently accepted.
        """

        closed = set(modes)
        for host_map in host_maps:
            for mode in list(closed):
                closed.add(tuple(sorted(host_map[host] for host in mode)))
        return [mode for mode in all_modes if mode in closed]

    emitted = 0

    def stream_alphabet(
        modes: List[Tuple[int, ...]],
        *,
        target_surplus: Optional[List[int]] = None,
        mode_frame_masks: Optional[List[int]] = None,
        n_frames: int = 0,
    ) -> Iterable[Tuple[Tuple[int, int], ...]]:
        """Enumerate multisets over a fixed mode alphabet.

        ``target_surplus`` forces an exact degree cover (degree-first path).
        ``mode_frame_masks`` is a per-mode bitset of frames; the active-frame
        intersection must stay nonempty (single-frame necessary condition).
        """

        nonlocal emitted
        if not modes:
            return
        n_modes = len(modes)
        mode_index = {mode: index for index, mode in enumerate(modes)}
        mode_permutations: set[Tuple[int, ...]] = set()
        for host_map in host_maps:
            try:
                mode_permutations.add(
                    tuple(
                        mode_index[
                            tuple(sorted(host_map[host] for host in mode))
                        ]
                        for mode in modes
                    )
                )
            except KeyError:
                # Host map sends some mode outside the alphabet; skip it for
                # symmetry reduction (safe: may only enumerate more).
                continue
        mode_permutations.discard(tuple(range(n_modes)))
        permutations = tuple(mode_permutations)
        status.automorphisms = max(
            status.automorphisms, len(permutations) + 1
        )

        mode_positions = [
            tuple(position[host] for host in mode) for mode in modes
        ]
        mode_pairs = [tuple(combinations(mode, 2)) for mode in modes]
        mode_mask = [
            sum(1 << slot for slot in slots) for slots in mode_positions
        ]
        reachable_mask = 0
        for mask in mode_mask:
            reachable_mask |= mask
        max_hosts_per_cl = max(len(mode) for mode in modes)
        all_frames_mask = (1 << n_frames) - 1 if n_frames > 0 else 0

        packed = n_modes <= 256
        if packed:
            empty_state: object = b""
            index_bytes = tuple(bytes((index,)) for index in range(n_modes))
            tables = tuple(
                bytes(
                    permutation[value] if value < n_modes else value
                    for value in range(256)
                )
                for permutation in permutations
            )

            def extend(chosen, index):
                child = chosen + index_bytes[index]
                if len(child) > 1 and index < child[-2]:
                    child = bytes(sorted(child))
                return child

            def canonical(chosen):
                best = chosen
                for table in tables:
                    image = bytes(sorted(chosen.translate(table)))
                    if image < best:
                        best = image
                return best

        else:
            empty_state = ()

            def extend(chosen, index):
                child = chosen + (index,)
                if len(child) > 1 and index < child[-2]:
                    child = tuple(sorted(child))
                return child

            def canonical(chosen):
                best = chosen
                for permutation in permutations:
                    image = tuple(
                        sorted(permutation[index] for index in chosen)
                    )
                    if image < best:
                        best = image
                return best

        exact = target_surplus is not None
        forbid_dual_term = bool(
            spec.graph_rules.forbid_mono_se_dual_terminal
        )
        mono_se_slots = (
            frozenset(
                slot
                for slot, skel in enumerate(base_degrees)
                if skel == 1
            )
            if forbid_dual_term
            else frozenset()
        )

        # --- Exact path: typed μ3 / μ2 / terminal counts (large-p win) -----
        # Free mixed multisets of 12 Cl explode; partitioning by role count
        # n1+n2+n3=L and n1+2n2+3n3=sum(surplus) cuts the search tree hard.
        if exact:
            assert target_surplus is not None
            surplus0 = list(target_surplus)
            total_s = sum(surplus0)
            extra = total_s - n_cl  # sum (size-1) over modes
            if extra < 0 or total_s > n_cl * max_hosts_per_cl:
                return
            by_size: Dict[int, List[int]] = {1: [], 2: [], 3: []}
            for index, mode in enumerate(modes):
                size = len(mode)
                if size in by_size:
                    by_size[size].append(index)
            host_term = [0] * n_cd
            host_bridge = [0] * n_cd
            pair_bridges: Dict[Tuple[int, int], int] = {}
            surplus = list(surplus0)
            seen_emit: set = set()

            def multiset_canonical(chosen_indices: Sequence[int]):
                best = tuple(sorted(chosen_indices))
                for permutation in permutations:
                    image = tuple(
                        sorted(permutation[index] for index in chosen_indices)
                    )
                    if image < best:
                        best = image
                return best

            def emit(chosen_indices: List[int]):
                nonlocal emitted
                if mono_se_slots and any(
                    host_term[slot] == 2 and host_bridge[slot] == 0
                    for slot in mono_se_slots
                ):
                    status.infeasible += 1
                    return
                key = (
                    multiset_canonical(chosen_indices)
                    if permutations
                    else tuple(sorted(chosen_indices))
                )
                if key in seen_emit:
                    status.symmetry_pruned += 1
                    return
                seen_emit.add(key)
                decoration: List[Tuple[int, int]] = []
                for offset, index in enumerate(chosen_indices):
                    ligand = cl_list[offset]
                    decoration.extend(
                        (ligand, host) for host in modes[index]
                    )
                emitted += 1
                yield tuple(sorted(decoration))

            def place_size(
                size: int,
                need: int,
                start: int,
                chosen: List[int],
                active_frames: int,
                pool: Sequence[int],
            ):
                nonlocal emitted
                if need == 0:
                    if size == 3:
                        yield from place_size(
                            2,
                            n2_target,
                            0,
                            chosen,
                            active_frames,
                            by_size[2],
                        )
                    elif size == 2:
                        yield from place_size(
                            1,
                            n1_target,
                            0,
                            chosen,
                            active_frames,
                            by_size[1],
                        )
                    else:
                        if any(surplus):
                            status.infeasible += 1
                            return
                        yield from emit(chosen)
                    return
                for pos in range(start, len(pool)):
                    if max_assignments > 0 and emitted >= max_assignments:
                        status.truncated = True
                        return
                    index = pool[pos]
                    slots = mode_positions[index]
                    if any(surplus[slot] <= 0 for slot in slots):
                        status.over_capacity += 1
                        continue
                    pairs = mode_pairs[index]
                    if bridge_cap > 0 and any(
                        pair_bridges.get(pair, 0) >= bridge_cap
                        for pair in pairs
                    ):
                        status.over_capacity += 1
                        continue
                    next_frames = active_frames
                    if mode_frame_masks is not None:
                        next_frames = (
                            active_frames & mode_frame_masks[index]
                        )
                        if not next_frames:
                            status.geometry_pruned += 1
                            continue
                    for slot in slots:
                        surplus[slot] -= 1
                    is_bridge = size >= 2
                    for slot in slots:
                        if slot in mono_se_slots:
                            if is_bridge:
                                host_bridge[slot] += 1
                            else:
                                host_term[slot] += 1
                    for pair in pairs:
                        pair_bridges[pair] = pair_bridges.get(pair, 0) + 1
                    # Mono-Se finished as dual-terminal → dead.
                    bad_dual = mono_se_slots and any(
                        surplus[slot] == 0
                        and host_term[slot] == 2
                        and host_bridge[slot] == 0
                        for slot in mono_se_slots
                    )
                    if bad_dual or any(s < 0 for s in surplus):
                        status.infeasible += 1
                    else:
                        chosen.append(index)
                        yield from place_size(
                            size,
                            need - 1,
                            pos,  # multiset: nondecreasing within type
                            chosen,
                            next_frames,
                            pool,
                        )
                        chosen.pop()
                    for pair in pairs:
                        pair_bridges[pair] -= 1
                        if pair_bridges[pair] == 0:
                            del pair_bridges[pair]
                    for slot in slots:
                        if slot in mono_se_slots:
                            if is_bridge:
                                host_bridge[slot] -= 1
                            else:
                                host_term[slot] -= 1
                    for slot in slots:
                        surplus[slot] += 1

            # n2 + 2*n3 = extra, n1 + n2 + n3 = n_cl
            for n3_target in range(0, extra // 2 + 1):
                n2_target = extra - 2 * n3_target
                n1_target = n_cl - n2_target - n3_target
                if n1_target < 0 or n2_target < 0:
                    continue
                if n3_target and not by_size[3]:
                    continue
                if n2_target and not by_size[2]:
                    continue
                if n1_target and not by_size[1]:
                    continue
                yield from place_size(
                    3,
                    n3_target,
                    0,
                    [],
                    all_frames_mask if mode_frame_masks is not None else 0,
                    by_size[3],
                )
                if status.truncated:
                    return
            return

        # --- Legacy non-exact path (min/max CN, no typed partition) --------
        degrees = list(base_degrees)
        demand_profiles = [
            tuple(int(x) for x in profile)
            for profile in (required_degree_profiles or ())
            if len(profile) == n_cd
        ]
        free_total = sum(max_cd - degree for degree in degrees)
        deficit_total = sum(
            max(0, min_cd - degree) for degree in degrees
        )
        deficit_mask = sum(
            1 << slot
            for slot, degree in enumerate(degrees)
            if degree < min_cd
        )
        host_term = [0] * n_cd
        host_bridge = [0] * n_cd

        def can_still_complete(remaining: int) -> bool:
            if free_total < remaining:
                return False
            if min_cd > 0 and deficit_mask:
                if deficit_mask & ~reachable_mask:
                    return False
                if deficit_total > remaining * max_hosts_per_cl:
                    return False
                if not all(
                min_cd - degree <= remaining
                for degree in degrees
                if degree < min_cd
                ):
                    return False
            if demand_profiles:
                alive = False
                for profile in demand_profiles:
                    deficits = [
                        max(0, profile[i] - degrees[i]) for i in range(n_cd)
                    ]
                    if any(value > remaining for value in deficits):
                        continue
                    if sum(deficits) > remaining * max_hosts_per_cl:
                        continue
                    alive = True
                    break
                if not alive:
                    return False
            return True

        def apply_bounds(slots: Tuple[int, ...], delta: int) -> None:
            nonlocal free_total, deficit_total, deficit_mask
            for slot in slots:
                before = degrees[slot]
                after = before + delta
                degrees[slot] = after
                free_total -= delta
                if before < min_cd:
                    deficit_total -= min_cd - before
                if after < min_cd:
                    deficit_total += min_cd - after
                    deficit_mask |= 1 << slot
                else:
                    deficit_mask &= ~(1 << slot)

        seen: set = set()
        pair_bridges = {}

        def rec(depth: int, chosen, active_frames: int):
            nonlocal emitted
            if depth == n_cl:
                if demand_profiles and not any(
                    all(degrees[i] >= profile[i] for i in range(n_cd))
                    for profile in demand_profiles
                ):
                    status.infeasible += 1
                    return
                decoration = []
                for offset, index in enumerate(chosen):
                    ligand = cl_list[offset]
                    decoration.extend(
                        (ligand, host) for host in modes[index]
                    )
                emitted += 1
                yield tuple(sorted(decoration))
                return
            remaining = n_cl - depth - 1
            for index in range(n_modes):
                if max_assignments > 0 and emitted >= max_assignments:
                    status.truncated = True
                    return
                slots = mode_positions[index]
                if any(degrees[slot] >= max_cd for slot in slots):
                    status.over_capacity += 1
                    continue
                pairs = mode_pairs[index]
                if bridge_cap > 0 and any(
                    pair_bridges.get(pair, 0) >= bridge_cap for pair in pairs
                ):
                    status.over_capacity += 1
                    continue
                next_frames = active_frames
                if mode_frame_masks is not None:
                    next_frames = active_frames & mode_frame_masks[index]
                    if not next_frames:
                        status.geometry_pruned += 1
                        continue
                child = extend(chosen, index)
                if child in seen:
                    status.revisited += 1
                    continue
                key = canonical(child) if permutations else child
                if key is not child and key in seen:
                    seen.add(child)
                    status.symmetry_pruned += 1
                    continue
                seen.add(child)
                seen.add(key)
                apply_bounds(slots, 1)
                if can_still_complete(remaining):
                    for pair in pairs:
                        pair_bridges[pair] = pair_bridges.get(pair, 0) + 1
                    yield from rec(depth + 1, child, next_frames)
                    for pair in pairs:
                        pair_bridges[pair] -= 1
                        if pair_bridges[pair] == 0:
                            del pair_bridges[pair]
                else:
                    status.infeasible += 1
                apply_bounds(slots, -1)

        if can_still_complete(n_cl):
            yield from rec(
                0,
                empty_state,
                all_frames_mask if mode_frame_masks is not None else 0,
            )

    # --- Geometry-conditioned degree-first path -----------------------------
    # CRITICAL: modes must be sphere-feasible on a clean frame for the target
    # final CN vector.  Combinatorial-only "production" without mid-tree
    # geometry floods the screen with bridges that never embed (embedded=0).
    lazy_degrees = (
        list(degree_vectors)
        if degree_vectors is not None and slice_builder is not None
        else None
    )
    eager_slices = list(degree_slices) if degree_slices is not None else None
    if lazy_degrees is not None or eager_slices is not None:
        min_bridge_host = int(spec.graph_rules.min_bridged_host_cn)

        def orbit_min_degree(degree: Sequence[int]) -> Tuple[int, ...]:
            best = tuple(int(value) for value in degree)
            for host_map in host_maps:
                image = [0] * n_cd
                for index in range(n_cd):
                    image[position[host_map[cd_list[index]]]] = int(
                        degree[index]
                    )
                candidate = tuple(image)
                if candidate < best:
                    best = candidate
            return best

        def iter_target_slices() -> Iterable[_DegreeSlice]:
            if eager_slices is not None:
                status.degree_vectors_total += len(eager_slices)
                for slice_ in eager_slices:
                    degree = slice_.degree
                    if len(degree) != n_cd:
                        continue
                    if tuple(int(v) for v in degree) != orbit_min_degree(
                        degree
                    ):
                        continue
                    surplus = [
                        int(degree[i]) - base_degrees[i] for i in range(n_cd)
                    ]
                    if any(s < 0 for s in surplus) or sum(surplus) < n_cl:
                        continue
                    if not _surplus_combinatorially_feasible(
                        surplus, base_degrees, degree, n_cl, spec
                    ):
                        status.infeasible += 1
                        continue
                    yield slice_
                return
            assert lazy_degrees is not None and slice_builder is not None
            status.degree_vectors_total += len(lazy_degrees)
            feasible = {
                tuple(int(v) for v in degree)
                for degree in lazy_degrees
                if len(degree) == n_cd
            }
            orbit_reps = sorted(
                {orbit_min_degree(d) for d in feasible}.intersection(feasible)
            )
            for degree in orbit_reps:
                surplus = [
                    int(degree[i]) - base_degrees[i] for i in range(n_cd)
                ]
                if any(s < 0 for s in surplus) or sum(surplus) < n_cl:
                    continue
                if not _surplus_combinatorially_feasible(
                    surplus, base_degrees, degree, n_cl, spec
                ):
                    status.infeasible += 1
                    continue
                built = slice_builder(degree)
                if built is None:
                    continue
                yield built

        for slice_ in iter_target_slices():
            degree = slice_.degree
            surplus = [
                int(degree[i]) - base_degrees[i] for i in range(n_cd)
            ]
            raw_modes = list(slice_.mode_frames)
            if not raw_modes:
                continue
            modes = close_mode_set(raw_modes)
            if not modes:
                continue
            if min_bridge_host > 1:
                final_cn = {
                    cd_list[i]: int(degree[i]) for i in range(n_cd)
                }
                modes = [
                    mode
                    for mode in modes
                    if len(mode) == 1
                    or all(final_cn[h] >= min_bridge_host for h in mode)
                ]
                if not modes:
                    continue
            n_frames = 0
            if slice_.mode_frames:
                n_frames = (
                    max(
                        (
                            max(frames)
                            for frames in slice_.mode_frames.values()
                            if frames
                        ),
                        default=-1,
                    )
                    + 1
                )
            mode_frame_masks = []
            for mode in modes:
                frames = slice_.mode_frames.get(mode)
                if frames is None:
                    mode_frame_masks.append(0)
                else:
                    mask = 0
                    for frame_index in frames:
                        mask |= 1 << frame_index
                    mode_frame_masks.append(mask)
            status.modes_kept = max(status.modes_kept, len(modes))
            status.degree_slices += 1
            status.degree_vectors_used += 1
            yield from stream_alphabet(
                modes,
                target_surplus=surplus,
                mode_frame_masks=mode_frame_masks,
                n_frames=n_frames,
            )
            if status.truncated:
                return
        return

    # --- Fully unconditioned legacy path ------------------------------------
    modes = list(all_modes)
    if mode_support is not None:
        support = {mode: set(degs) for mode, degs in mode_support.items()}
        for host_map in host_maps:
            for mode, degree_sets in list(support.items()):
                images = [host_map[host] for host in mode]
                order = sorted(range(len(mode)), key=lambda j: images[j])
                image_mode = tuple(images[j] for j in order)
                target = support.setdefault(image_mode, set())
                for degrees_tuple in degree_sets:
                    target.add(tuple(degrees_tuple[j] for j in order))
        modes = [mode for mode in modes if mode in support]
        if not modes:
            return
    status.modes_kept += len(modes)
    yield from stream_alphabet(modes)


def _bridge_first_production(
    *,
    k: int,
    p: int,
    cd_list: Sequence[int],
    cl_list: Sequence[int],
    base_degrees: Sequence[int],
    all_modes: Sequence[Tuple[int, ...]],
    host_maps: Sequence[Dict[int, int]],
    position: Dict[int, int],
    spec: NucleationSpec,
    max_cd: int,
    min_cd: int,
    bridge_cap: int,
    max_assignments: int,
    status: _DecorationStatus,
    close_mode_set: Callable[[Sequence[Tuple[int, ...]]], List[Tuple[int, ...]]],
) -> Iterable[Tuple[Tuple[int, int], ...]]:
    """Complete Cl decoration by bridge moves then terminal fill.

    Same pack legality as the finished-graph rules (min_bridged_host_cn,
    mono-Se dual-terminal ban, max_shared, CN bounds).  Does **not** add
    maximality (“must bridge if possible”); every still-legal end graph has a
    production sequence.  Geometry is left to the embed screen.
    """

    n_cl = len(cl_list)
    n_cd = len(cd_list)
    if n_cl == 0:
        yield ()
        return

    modes = close_mode_set(all_modes)
    if not modes:
        return
    status.modes_kept = max(status.modes_kept, len(modes))
    status.degree_slices = 1  # single production pass (not per CN vector)
    status.degree_vectors_used = 1

    min_bridge = int(spec.graph_rules.min_bridged_host_cn)
    forbid_dual = bool(spec.graph_rules.forbid_mono_se_dual_terminal)
    mono_se = {
        position[host]
        for host, skel in zip(cd_list, base_degrees)
        if skel == 1
    }

    mode_index = {mode: index for index, mode in enumerate(modes)}
    mode_positions = [
        tuple(position[host] for host in mode) for mode in modes
    ]
    mode_pairs = [tuple(combinations(mode, 2)) for mode in modes]
    mode_size = [len(mode) for mode in modes]

    # Aut action on mode indices (for emit-only orbit prune).
    mode_permutations: List[Tuple[int, ...]] = []
    for host_map in host_maps:
        try:
            mode_permutations.append(
                tuple(
                    mode_index[
                        tuple(sorted(host_map[host] for host in mode))
                    ]
                    for mode in modes
                )
            )
        except KeyError:
            continue
    identity = tuple(range(len(modes)))
    mode_permutations = [p for p in mode_permutations if p != identity]

    bridge_indices = [i for i, s in enumerate(mode_size) if s >= 2]
    # Bridge mode only if every host can reach min_bridged under max_cn.
    if min_bridge > 1:
        bridge_indices = [
            i
            for i in bridge_indices
            if all(
                max_cd >= min_bridge for _host in modes[i]
            )
        ]
    degrees = list(base_degrees)
    host_bridge = [0] * n_cd
    pair_bridges: Dict[Tuple[int, int], int] = {}
    chosen_bridges: List[int] = []
    seen_emit: set = set()
    emitted = 0

    def multiset_canonical(chosen: Sequence[int]) -> Tuple[int, ...]:
        best = tuple(sorted(chosen))
        for permutation in mode_permutations:
            image = tuple(sorted(permutation[index] for index in chosen))
            if image < best:
                best = image
        return best

    def finish_with_terminals(n_term: int) -> Iterable[Tuple[Tuple[int, int], ...]]:
        nonlocal emitted
        # Distribute n_term indistinguishable terminals under capacity + min_cn.
        counts = [0] * n_cd

        def dist(host: int, remaining: int) -> Iterable[Tuple[Tuple[int, int], ...]]:
            nonlocal emitted
            if host == n_cd:
                if remaining != 0:
                    return
                for slot in range(n_cd):
                    final = degrees[slot] + counts[slot]
                    if min_cd > 0 and final < min_cd:
                        status.infeasible += 1
                        return
                    if final > max_cd:
                        status.infeasible += 1
                        return
                    if host_bridge[slot] > 0 and final < min_bridge:
                        status.infeasible += 1
                        return
                    if (
                        forbid_dual
                        and slot in mono_se
                        and counts[slot] == 2
                        and host_bridge[slot] == 0
                    ):
                        status.infeasible += 1
                        return
                term_mode_list: List[int] = []
                for slot in range(n_cd):
                    host_id = cd_list[slot]
                    tmode = (host_id,)
                    if tmode not in mode_index:
                        status.infeasible += 1
                        return
                    t_idx = mode_index[tmode]
                    term_mode_list.extend([t_idx] * counts[slot])
                full = list(chosen_bridges) + term_mode_list
                key = multiset_canonical(full)
                if key in seen_emit:
                    status.symmetry_pruned += 1
                    return
                seen_emit.add(key)
                decoration: List[Tuple[int, int]] = []
                for offset, index in enumerate(full):
                    ligand = cl_list[offset]
                    decoration.extend(
                        (ligand, host) for host in modes[index]
                    )
                emitted += 1
                status.automorphisms = max(
                    status.automorphisms, len(mode_permutations) + 1
                )
                yield tuple(sorted(decoration))
                return

            cap = max_cd - degrees[host]
            later_min = sum(
                max(0, min_cd - degrees[j]) for j in range(host + 1, n_cd)
            )
            t_lo = max(0, min_cd - degrees[host])
            t_hi = min(cap, remaining - later_min)
            if t_hi < t_lo:
                return
            for t in range(t_lo, t_hi + 1):
                if max_assignments > 0 and emitted >= max_assignments:
                    status.truncated = True
                    return
                counts[host] = t
                yield from dist(host + 1, remaining - t)
                counts[host] = 0

        yield from dist(0, n_term)

    def rec_bridges(start: int) -> Iterable[Tuple[Tuple[int, int], ...]]:
        nonlocal emitted
        n_bridge = len(chosen_bridges)
        n_term = n_cl - n_bridge
        if n_term < 0:
            return
        cap_total = sum(max_cd - degrees[s] for s in range(n_cd))
        if n_term <= cap_total:
            ok_bridge_hosts = all(
                host_bridge[slot] <= 0
                or degrees[slot] + n_term >= min_bridge
                for slot in range(n_cd)
            )
            if ok_bridge_hosts:
                yield from finish_with_terminals(n_term)

        if n_bridge >= n_cl or status.truncated:
            return
        remaining_after = n_cl - n_bridge - 1
        for pos in range(start, len(bridge_indices)):
            if max_assignments > 0 and emitted >= max_assignments:
                status.truncated = True
                return
            index = bridge_indices[pos]
            slots = mode_positions[index]
            if any(degrees[slot] >= max_cd for slot in slots):
                status.over_capacity += 1
                continue
            pairs = mode_pairs[index]
            if bridge_cap > 0 and any(
                pair_bridges.get(pair, 0) >= bridge_cap for pair in pairs
            ):
                status.over_capacity += 1
                continue
            if min_bridge > 1 and any(
                degrees[slot] + 1 + remaining_after < min_bridge
                for slot in slots
            ):
                status.infeasible += 1
                continue
            for slot in slots:
                degrees[slot] += 1
                host_bridge[slot] += 1
            for pair in pairs:
                pair_bridges[pair] = pair_bridges.get(pair, 0) + 1
            chosen_bridges.append(index)
            yield from rec_bridges(pos)
            chosen_bridges.pop()
            for pair in pairs:
                pair_bridges[pair] -= 1
                if pair_bridges[pair] == 0:
                    del pair_bridges[pair]
            for slot in slots:
                degrees[slot] -= 1
                host_bridge[slot] -= 1

    yield from rec_bridges(0)


def _cl_attachments(
    k: int,
    p: int,
    inorganic_edges: Sequence[Tuple[int, int]],
    spec: NucleationSpec,
    *,
    max_assignments: int = 0,
) -> Tuple[List[Tuple[Tuple[int, int], ...]], bool]:
    """Materialise :func:`iter_cl_attachments` for callers that want a list."""

    status = _DecorationStatus()
    results = list(
        iter_cl_attachments(
            k,
            p,
            inorganic_edges,
            spec,
            max_assignments=max_assignments,
            status=status,
        )
    )
    return results, status.truncated


def _cation_degree_vectors(
    skeleton_degrees: Sequence[int],
    n_ligands: int,
    spec: NucleationSpec,
    *,
    limit: int,
) -> Optional[List[Tuple[int, ...]]]:
    """Final cation coordination numbers a decoration could realise.

    Each ligand contributes one to between one and ``max_cn`` hosts, so the
    total added coordination is bounded on both sides; that prunes the raw
    product of per-cation ranges hard.  Returns ``None`` when the enumeration
    would exceed ``limit``, which the caller must treat as "unknown", never as
    "none viable".

    Uses recursive generation with running-sum bounds so large cation counts
    do not materialise a full Cartesian product before filtering.
    """

    max_cn = int(spec.graph_rules.max_cn[spec.core.cation])
    min_cn = (
        int(spec.graph_rules.min_cn.get(spec.core.cation, 0))
        if spec.enforce_min_cn
        else 0
    )
    max_hosts = int(spec.graph_rules.max_cn[spec.precursor.ligand])
    skel = [int(degree) for degree in skeleton_degrees]
    n = len(skel)
    lows = [max(s, min_cn) for s in skel]
    highs = [max_cn] * n
    # Per-index min/max of (final - skel) for sum pruning.
    add_lo = [max(0, lows[i] - skel[i]) for i in range(n)]
    add_hi = [highs[i] - skel[i] for i in range(n)]
    if any(h < 0 for h in add_hi):
        return []
    sum_add_lo, sum_add_hi = n_ligands, n_ligands * max_hosts
    suffix_lo = [0] * (n + 1)
    suffix_hi = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_lo[i] = suffix_lo[i + 1] + add_lo[i]
        suffix_hi[i] = suffix_hi[i + 1] + add_hi[i]

    vectors: List[Tuple[int, ...]] = []
    partial = [0] * n

    def rec(index: int, added: int) -> bool:
        """Return False if the global vector limit is hit."""

        if index == n:
            if sum_add_lo <= added <= sum_add_hi:
                vectors.append(tuple(partial))
                if len(vectors) > limit:
                    return False
            return True
        # added + future in [sum_add_lo, sum_add_hi]
        lo_need = sum_add_lo - added - suffix_hi[index + 1]
        hi_need = sum_add_hi - added - suffix_lo[index + 1]
        a_min = max(add_lo[index], lo_need)
        a_max = min(add_hi[index], hi_need)
        for add in range(a_min, a_max + 1):
            partial[index] = skel[index] + add
            if not rec(index + 1, added + add):
                return False
        return True

    if not rec(0, 0):
        return None
    return vectors


def _surplus_combinatorially_feasible(
    surplus: Sequence[int],
    base_degrees: Sequence[int],
    final_degrees: Sequence[int],
    n_ligands: int,
    spec: NucleationSpec,
) -> bool:
    """Cheap necessary check before building inorganic frames for a CN vector.

    Avoids thousands of ``clean_frames`` calls at large p when the surplus
    cannot be realised under terminal/bridge mode sizes and pack rules.
    """

    if n_ligands == 0:
        return all(s == 0 for s in surplus)
    if any(s < 0 for s in surplus):
        return False
    total = sum(surplus)
    max_hosts = int(spec.graph_rules.max_cn[spec.precursor.ligand])
    if total < n_ligands or total > n_ligands * max_hosts:
        return False
    if any(s > n_ligands for s in surplus):
        # One Cl touches a host at most once.
        return False

    min_bridge = int(spec.graph_rules.min_bridged_host_cn)
    forbid_dual = bool(spec.graph_rules.forbid_mono_se_dual_terminal)

    # Hosts that cannot participate in any μ-mode (final CN too low).
    low = [
        i
        for i, final in enumerate(final_degrees)
        if final < min_bridge and surplus[i] > 0
    ]
    high = [
        i
        for i, final in enumerate(final_degrees)
        if final >= min_bridge and surplus[i] > 0
    ]
    s_low = sum(surplus[i] for i in low)
    s_high = sum(surplus[i] for i in high)
    # Low hosts only accept terminals → need ≥ s_low distinct terminal Cl.
    if s_low > n_ligands:
        return False
    remaining_cl = n_ligands - s_low
    remaining_need = s_high
    if remaining_cl < 0 or remaining_need < remaining_cl:
        return False
    if remaining_need > remaining_cl * max_hosts:
        return False

    # Mono-Se Cd with exactly two Cl slots cannot be two terminals.
    if forbid_dual:
        for i, base in enumerate(base_degrees):
            if base != 1 or surplus[i] != 2:
                continue
            # Needs ≥1 bridge → this host must be high-CN and some partner
            # must have remaining surplus for a multi-host mode.
            if final_degrees[i] < min_bridge:
                return False
            partners = sum(
                1
                for j, final in enumerate(final_degrees)
                if j != i and final >= min_bridge and surplus[j] > 0
            )
            if partners == 0 and n_ligands > 0:
                return False

    return True


def _reachable_host_sets(
    state: _State,
    frame: FloatArray,
    degrees: Sequence[int],
    cation_ids: Sequence[int],
    pack: GeometryPack,
    spec: NucleationSpec,
) -> Set[Tuple[int, ...]]:
    """Host sets a single ligand can actually bond, given one frame.

    A terminal ligand is always placeable.  A bridge has to sit on the
    intersection of spheres centred on hosts whose positions this frame already
    fixes, so the same arithmetic the embedder performs decides it here --
    before any decoration exists.
    """

    max_hosts = int(spec.graph_rules.max_cn[spec.precursor.ligand])
    allowed_signatures = set(
        spec.graph_rules.allowed_neighbor_signatures.get(
            spec.precursor.ligand, ()
        )
    )
    reachable: Set[Tuple[int, ...]] = set()
    for size in range(1, min(max_hosts, len(cation_ids)) + 1):
        if allowed_signatures and (
            f"{spec.core.cation}{size}" not in allowed_signatures
        ):
            continue
        for hosts in combinations(cation_ids, size):
            if size == 1:
                reachable.add(hosts)
                continue
            radii = [
                pack.bond_length(
                    "CdCl_bridge", degrees[host], size, default=2.40
                )
                for host in hosts
            ]
            if size == 2:
                separation = float(
                    np.linalg.norm(frame[hosts[0]] - frame[hosts[1]])
                )
                if separation < 1.0e-12:
                    continue
                axial = (
                    radii[0] * radii[0]
                    - radii[1] * radii[1]
                    + separation * separation
                ) / (2.0 * separation)
                if radii[0] * radii[0] - axial * axial < -EXACT_BOND_TOLERANCE:
                    continue
            else:
                try:
                    _three_sphere_intersections(
                        [frame[host] for host in hosts], radii
                    )
                except ExactEmbeddingError:
                    continue
            reachable.add(hosts)
    return reachable


@dataclass
class _SkeletonSurvey:
    """Geometry bookkeeping for one inorganic skeleton.

    Prefer the **lazy** path: ``pending_degrees`` lists feasible final CN
    vectors and :meth:`build_degree_slice` constructs frames only when the
    enumerator enters that vector.  Eager ``degree_slices`` remain available
    for tests that want the full table up front.
    """

    frames: Dict[
        Tuple[int, ...],
        Tuple[List[Tuple[FloatArray, List[bool]]], List[str]],
    ] = field(default_factory=dict)
    # mode -> host coordination numbers for which that mode is reachable
    # (union over degree vectors; filled as lazy slices are built).
    mode_support: Optional[Dict[Tuple[int, ...], Set[Tuple[int, ...]]]] = None
    # Eager path only.
    degree_slices: Optional[List[_DegreeSlice]] = None
    # Lazy path: CN vectors not yet framed.
    pending_degrees: Optional[List[Tuple[int, ...]]] = None
    frames_built: int = 0
    dead_reason: Optional[str] = None
    # Bound for lazy builders (not serialised state).
    _state: Optional[_State] = field(default=None, repr=False)
    _cation_ids: Optional[Tuple[int, ...]] = field(default=None, repr=False)
    _pack: Optional[GeometryPack] = field(default=None, repr=False)
    _spec: Optional[NucleationSpec] = field(default=None, repr=False)
    _frame_options: int = field(default=0, repr=False)
    _slice_cache: Dict[Tuple[int, ...], Optional[_DegreeSlice]] = field(
        default_factory=dict, repr=False
    )


def build_degree_slice(
    state: _State,
    cation_ids: Sequence[int],
    pack: GeometryPack,
    spec: NucleationSpec,
    vector: Sequence[int],
    *,
    frame_options: int = 0,
    frames_cache: Optional[
        Dict[
            Tuple[int, ...],
            Tuple[List[Tuple[FloatArray, List[bool]]], List[str]],
        ]
    ] = None,
) -> Tuple[Optional[_DegreeSlice], List[str], int]:
    """Build clean frames and mode→frame support for one final CN vector.

    Returns ``(slice_or_None, failure_reasons, frames_built_delta)``.
    """

    n = len(state.atoms)
    key = tuple(int(value) for value in vector)
    degrees = [state.graph.degree[i] for i in range(n)]
    for cation, degree in zip(cation_ids, key):
        degrees[cation] = degree
    built, failure = _clean_frames(
        state, pack, spec, degrees, limit=frame_options
    )
    if frames_cache is not None:
        frames_cache[key] = (built, list(failure or []))
    if not built:
        return None, list(failure or ["frame_not_realisable"]), 1
    mode_frames: Dict[Tuple[int, ...], Set[int]] = {}
    for frame_index, (frame, _placed) in enumerate(built):
        for mode in _reachable_host_sets(
            state, frame, degrees, cation_ids, pack, spec
        ):
            mode_frames.setdefault(mode, set()).add(frame_index)
    slice_ = _DegreeSlice(
        degree=key,
        mode_frames={
            mode: frozenset(indices) for mode, indices in mode_frames.items()
        },
    )
    return slice_, [], 1


def survey_skeleton_frames(
    state: _State,
    cation_ids: Sequence[int],
    pack: GeometryPack,
    spec: NucleationSpec,
    n_ligands: int,
    *,
    limit: int = 20000,
    frame_options: int = 0,
    eager: bool = False,
) -> _SkeletonSurvey:
    """Prepare geometry for one skeleton (lazy by default).

    The frame depends on the skeleton and the final cation coordination numbers
    and on nothing else.  Building every feasible CN vector up front is correct
    but catastrophic at k=2, p≥4 (hundreds of frame constructions per skeleton
    before any Cl is placed).  The default **lazy** path only lists the degree
    vectors; :meth:`_SkeletonSurvey` clients call :func:`build_degree_slice`
    (via ``slice_builder``) when a vector is first needed.

    ``eager=True`` restores the old behaviour (full table) for tests.
    """

    survey = _SkeletonSurvey(
        _state=state,
        _cation_ids=tuple(cation_ids),
        _pack=pack,
        _spec=spec,
        _frame_options=frame_options,
        mode_support={},
    )
    skeleton_degrees = [state.graph.degree[i] for i in cation_ids]
    vectors = _cation_degree_vectors(
        skeleton_degrees, n_ligands, spec, limit=limit
    )
    if vectors is None:
        # Too many coordination vectors to certify; fall back to deciding each
        # frame lazily without a pre-listed product (no degree-first alphabet).
        return survey
    if not vectors:
        survey.dead_reason = "frame_not_realisable"
        return survey

    pending = [tuple(int(value) for value in vector) for vector in vectors]
    survey.pending_degrees = pending
    if not eager:
        return survey

    reasons: Counter[str] = Counter()
    support: Dict[Tuple[int, ...], Set[Tuple[int, ...]]] = {}
    slices: List[_DegreeSlice] = []
    alive = False
    for vector in pending:
        slice_, failure, built = build_degree_slice(
            state,
            cation_ids,
            pack,
            spec,
            vector,
            frame_options=frame_options,
            frames_cache=survey.frames,
        )
        survey.frames_built += built
        if slice_ is None:
            if failure:
                reasons[failure[0]] += 1
            continue
        alive = True
        cation_list = list(cation_ids)
        for mode in slice_.mode_frames:
            host_degrees = tuple(
                int(vector[cation_list.index(host)]) for host in mode
            )
            support.setdefault(mode, set()).add(host_degrees)
        slices.append(slice_)
    if not alive:
        survey.dead_reason = (
            reasons.most_common(1)[0][0] if reasons else "frame_not_realisable"
        )
        return survey
    survey.mode_support = support
    survey.degree_slices = slices
    return survey


def _lazy_slice_builder(survey: _SkeletonSurvey) -> Callable[
    [Tuple[int, ...]], Optional[_DegreeSlice]
]:
    """Return a memoised builder that fills ``survey.frames`` on demand."""

    def builder(degree: Tuple[int, ...]) -> Optional[_DegreeSlice]:
        cached = survey._slice_cache.get(degree)
        if degree in survey._slice_cache:
            return cached
        assert survey._state is not None
        assert survey._cation_ids is not None
        assert survey._pack is not None
        assert survey._spec is not None
        slice_, _failure, built = build_degree_slice(
            survey._state,
            survey._cation_ids,
            survey._pack,
            survey._spec,
            degree,
            frame_options=survey._frame_options,
            frames_cache=survey.frames,
        )
        survey.frames_built += built
        if slice_ is not None and survey.mode_support is not None:
            cation_ids = survey._cation_ids
            for mode in slice_.mode_frames:
                host_degrees = tuple(
                    int(degree[cation_ids.index(host)]) for host in mode
                )
                survey.mode_support.setdefault(mode, set()).add(host_degrees)
        survey._slice_cache[degree] = slice_
        return slice_

    return builder


def bond_count_bands(
    pool: Sequence[Tuple[int, object]],
) -> Dict[int, List[int]]:
    """Group pool indices by total bond count.

    Bands are the *distinct total bond counts present in this pool* -- nothing
    is configured and nothing is normalised, because the comparison is only
    ever made inside one ``(k, p)``.  B_tot is an integer and its spread per
    composition is small (4 distinct values at k=4,p=3, 6 at k=5, 7 at k=6),
    so the observed values are already the bands.
    """

    bands: Dict[int, List[int]] = {}
    for index, (bond_total, _state) in enumerate(pool):
        bands.setdefault(int(bond_total), []).append(index)
    return bands


def _molecular_check_spec(spec: NucleationSpec) -> NucleationSpec:
    """Return the explicit spec; molecular policy has no hidden defaults."""

    return spec


@dataclass
class _CandidateScreen:
    """The accept/reject bar every candidate must clear, however it was made.

    Exact enumeration and guided growth differ only in *which* graphs they
    propose; what makes a graph acceptable is identical, so both drive this.
    Keeping it in one place is what lets guided coverage be compared against
    exact at all -- if the two pipelines could drift, a coverage number would
    mean nothing.
    """

    k: int
    p: int
    spec: NucleationSpec
    pack: Optional[GeometryPack]
    embed: bool
    cation_ids: Sequence[int]
    cation_min_cn: int
    bin_result: MolecularBinResult
    seen: Dict[Tuple[object, ...], MolecularIsomer]
    processed: Set[Tuple[object, ...]]
    progress: Optional[ProgressCallback] = None
    graph_checkpoint: Optional[Callable[[MolecularBinResult], None]] = None
    validate_every_graph: bool = False
    skeleton_index: int = 0
    ring_mode: str = "free"
    #: retain per-failure coordinate snapshots for the failures/*.xyz dump
    dump_failures: bool = False
    #: When set, ``offer`` stops after the graph-level checks and files the
    #: candidate in ``pool`` instead of embedding it.  The bin then chooses a
    #: bounded, density-balanced subset to embed, rather than embedding every
    #: candidate in whatever order the decorator happened to emit them.
    collect_only: bool = False
    pool: List[Tuple[int, _State]] = field(default_factory=list)
    #: how many sound frames to keep per coordination vector before calling a
    #: molecule unrealisable; 0 keeps every one of them
    frame_options: int = 0
    frame_cache: Dict[
        Tuple[int, ...],
        Tuple[List[Tuple[FloatArray, List[bool]]], List[str]],
    ] = field(default_factory=dict)
    # Ring seed construction depends only on the inorganic edge set, final
    # cation CN vector, and conformation—not on the particular Cl edge order.
    # Reusing it avoids rebuilding the same chair/boat and repeatedly running
    # _place_remaining_with_pack_rules for every decoration of one skeleton.
    ring_frame_cache: Dict[
        Tuple[object, ...], Optional[Tuple[FloatArray, List[bool]]]
    ] = field(default_factory=dict)
    # A single coordination vector can produce many decorated graphs.  The
    # bounded whole-graph repair is deliberately budgeted per skeleton so a
    # pathological family cannot turn decoration into an optimizer sweep.  The
    # budget was set when a repair could not succeed anyway (it was judged at
    # 1e-4 A) and the residual was a Python loop; vectorising it made each
    # attempt ~6x cheaper, so the same wall-clock now buys a usable number.
    relax_attempt_budget: int = 8
    relaxed_geometry_attempts: int = 0
    steric_relax_attempts: int = 0
    local_angle_relax_attempts: int = 0

    def _record_failure_details(
        self,
        state: _State,
        reasons: Sequence[str],
        *,
        stage: str,
        coordinates: Optional[Sequence[Sequence[float]]] = None,
        snapshot_kind: str = "",
    ) -> None:
        """Aggregate failure context and retain one representative snapshot."""

        cd_cn = tuple(int(state.graph.degree[cation]) for cation in self.cation_ids)
        snapshot_symbols: Tuple[str, ...] = ()
        snapshot_coordinates: Optional[Tuple[Tuple[float, float, float], ...]] = None
        # Reasons and counts are always aggregated; the coordinates behind them
        # are only retained when someone asked for the XYZ dump, since they are
        # held for the whole run and most of them are never looked at.
        if coordinates is not None and self.dump_failures:
            arr = np.asarray(coordinates, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 3:
                if snapshot_kind == "inorganic":
                    indices = [
                        atom.atom_id
                        for atom in state.atoms
                        if atom.symbol != self.spec.precursor.ligand
                    ]
                else:
                    indices = list(range(min(len(state.atoms), arr.shape[0])))
                if indices and max(indices) < arr.shape[0]:
                    snapshot_symbols = tuple(state.atoms[i].symbol for i in indices)
                    snapshot_coordinates = tuple(
                        tuple(float(value) for value in arr[i]) for i in indices
                    )
        for reason in reasons:
            reason_text = str(reason)
            key = (self.skeleton_index, cd_cn, stage, reason_text)
            record = self.bin_result.failure_records.get(key)
            if record is None:
                record = MolecularFailureRecord(
                    skeleton_index=self.skeleton_index,
                    cd_cn=cd_cn,
                    stage=stage,
                    reason=reason_text,
                )
                self.bin_result.failure_records[key] = record
            record.count += 1
            if (
                record.snapshot_coordinates is None
                and snapshot_coordinates is not None
            ):
                record.snapshot_kind = snapshot_kind or "molecule"
                record.snapshot_symbols = snapshot_symbols
                record.snapshot_coordinates = snapshot_coordinates

    def _reject(
        self,
        state: _State,
        reasons: Sequence[str],
        *,
        before_embed: bool = False,
        stage: str = "final_geometry",
        coordinates: Optional[Sequence[Sequence[float]]] = None,
        snapshot_kind: str = "",
    ) -> None:
        self.bin_result.rejected += 1
        if before_embed:
            self.bin_result.screened_before_embed += 1
        self.bin_result.rejection_reasons.update(
            reason.split(":", 1)[0] for reason in reasons
        )
        self.bin_result.rejection_details.update(str(reason) for reason in reasons)
        self._record_failure_details(
            state,
            reasons,
            stage=stage,
            coordinates=coordinates,
            snapshot_kind=snapshot_kind,
        )

    def _relax_log(self, message: str) -> None:
        """Emit an optimizer trace line only when explicitly requested."""

        if self.progress is not None and _molecular_relax_trace_enabled():
            self.progress(
                f"[molecular] relax k={self.k} p={self.p} "
                f"skeleton={self.skeleton_index} | {message}"
            )

    def _record_optimizer_stats(self, stats: Mapping[str, float]) -> None:
        self.bin_result.optimizer_attempts += 1
        self.bin_result.optimizer_nfev += int(stats.get("nfev", 0.0))
        if stats.get("success", 0.0) > 0.5:
            self.bin_result.optimizer_successes += 1

    def _motif_factor_enabled(self) -> bool:
        if self.pack is None:
            return False
        reconstruction = self.pack.raw.get("reconstruction") or {}
        return str(reconstruction.get("method", "")).strip().lower() == "motif_factor"

    def _offer_motif_factor(self, state: _State) -> Optional[MolecularIsomer]:
        """Reconstruct and xTB-test one graph without a skeleton-first frame."""

        from .molecular_motif_reconstruct import reconstruct_motif_state

        assert self.pack is not None
        source_certificate = _graph_certificate(state)
        if source_certificate in self.processed:
            existing = self.seen.get(source_certificate)
            if existing is not None:
                source_id = f"raw_graph{self.bin_result.raw_graphs:04d}"
                detail = (
                    f"k={self.k} p={self.p} graph merge before xTB: "
                    f"{source_id} -> {existing.structure_id} "
                    "(duplicate graph certificate; no reconstruction/xTB run)"
                )
                self.bin_result.graph_merge_records.append(
                    (source_id, existing.structure_id, detail)
                )
                if self.progress is not None:
                    self.progress(f"[molecular] {detail}")
            return None
        self.processed.add(source_certificate)
        settings_raw = self.pack.raw.get("reconstruction") or {}
        graph_number_hint = self.bin_result.motif_graphs_eligible + 1
        audit_mode = str(settings_raw.get("audit", "exact")).strip().lower()
        clash_only = audit_mode in {"clash", "clashes", "clash_only", "clashes_only"}
        starts = int(settings_raw.get("factor_starts_per_graph", 12))
        xtb_keep = int(settings_raw.get("xtb_starts_per_graph", 3))
        reconstruction_started = time.perf_counter()
        result = reconstruct_motif_state(
            state,
            self.pack,
            self.spec,
            starts=starts,
            keep=max(1, xtb_keep),
            max_nfev=int(settings_raw.get("max_nfev", 40)),
            overlap_min_A=float(settings_raw.get("overlap_min_A", 0.75)),
            start_max_bond_error_A=float(
                settings_raw.get("start_max_bond_error_A", 0.50)
            ),
        )
        self.bin_result.motif_reconstruction_attempts += result.starts_attempted
        reconstruction_elapsed = time.perf_counter() - reconstruction_started
        if self.progress is not None:
            self.progress(
                f"[molecular] motif graph graph{graph_number_hint:04d} | "
                f"reconstruction_starts={result.starts_attempted} | "
                f"candidates={len(result.candidates)} | "
                f"construction_audit_time_s={reconstruction_elapsed:.3f}"
            )
        if result.motif_violations:
            if self.progress is not None:
                self.progress(
                    f"[molecular] motif graph graph{graph_number_hint:04d} | "
                    "motif_audit=FAIL | "
                    f"violations={'|'.join(result.motif_violations)}"
                )
            self._reject(
                state,
                result.motif_violations,
                before_embed=True,
                stage="motif_vocabulary",
            )
            return None
        self.bin_result.motif_graphs_eligible += 1
        self.bin_result.unique_graphs += 1
        self.bin_result.embedded += 1
        self.bin_result.motif_reconstruction_candidates += len(result.candidates)
        if not result.candidates:
            self._reject(
                state,
                ["motif_reconstruction:no_nonoverlapping_start"],
                stage="motif_reconstruction",
            )
            return None

        graph_number = self.bin_result.motif_graphs_eligible
        trials: Dict[int, MolecularMotifTrial] = {}
        for candidate in result.candidates:
            trial = MolecularMotifTrial(
                trial_id=f"graph{graph_number:04d}_start{candidate.start_index:02d}",
                start_index=candidate.start_index,
                symbols=tuple(atom.symbol for atom in state.atoms),
                source_edges=tuple(
                    sorted(
                        (min(int(a), int(b)), max(int(a), int(b)))
                        for a, b in state.graph.edges
                    )
                ),
                initial_coordinates=candidate.coordinates,
                initial_violations=candidate.audit_violations,
            )
            self.bin_result.motif_trials.append(trial)
            trials[candidate.start_index] = trial

        xtb_settings = XtbSettings.from_pack(self.pack.raw.get("relaxation"))
        if xtb_settings.enabled and xtb_keep > 0:
            self.bin_result.motif_xtb_attempts += min(
                len(result.candidates), xtb_keep
            )

        if self.graph_checkpoint is not None:
            self.bin_result.isomers = list(self.seen.values())
            self.graph_checkpoint(self.bin_result)

        def add_isomer(
            final_state: _State,
            coordinates: Sequence[Sequence[float]],
            *,
            start_index: int,
            constructed: Optional[Sequence[Sequence[float]]] = None,
            xtb_result: Optional[object] = None,
        ) -> Optional[MolecularIsomer]:
            certificate = _graph_certificate(final_state)
            if certificate in self.seen:
                existing = self.seen[certificate]
                if xtb_result is None:
                    return existing
                if certificate != source_certificate:
                    source_iso = self.seen.get(source_certificate)
                    source_id = (
                        source_iso.structure_id
                        if source_iso is not None
                        else f"graph{graph_number:04d}"
                    )
                    detail = (
                        f"k={self.k} p={self.p} xTB merge: "
                        f"{source_id} -> {existing.structure_id} "
                        f"(graph{graph_number:04d}, final connectivity matches "
                        f"{existing.structure_id})"
                    )
                    self.bin_result.xtb_merge_records.append(
                        (source_id, existing.structure_id, detail)
                    )
                    if self.progress is not None:
                        self.progress(f"[molecular] {detail}")
                xr = xtb_result
                orders = getattr(xr, "bond_orders", None)
                updated = replace(
                    existing,
                    xtb_energy_eV=getattr(xr, "energy_eV", None),
                    xtb_gap_eV=getattr(xr, "gap_eV", None),
                    xtb_steps=int(getattr(xr, "steps", 0)),
                    xtb_converged=bool(getattr(xr, "converged", False)),
                    xtb_coordinates=tuple(
                        tuple(float(value) for value in row)
                        for row in coordinates
                    ),
                    xtb_connectivity_changed=len(
                        getattr(xr, "connectivity_changed", ())
                    ),
                    xtb_relaxed_bonds=final_state.graph.number_of_edges(),
                    xtb_bonds_delta=(
                        final_state.graph.number_of_edges()
                        - state.graph.number_of_edges()
                    ),
                    xtb_same_topology=(certificate == source_certificate),
                    xtb_bond_orders=(
                        None
                        if orders is None
                        else tuple(
                            tuple(float(value) for value in row)
                            for row in orders
                        )
                    ),
                )
                self.seen[certificate] = updated
                return updated
            packed = tuple(
                tuple(float(value) for value in row) for row in coordinates
            )
            same_graph = certificate == source_certificate
            structure_id = (
                f"k{self.k:03d}_p{self.p:03d}_mol{len(self.seen) + 1:04d}"
            )
            motif_inventory = tuple(
                coordination_motif_inventory(
                    final_state,
                    cation_symbols=(self.spec.core.cation, self.spec.precursor.center),
                    anion_symbols=(self.spec.core.anion,),
                    ligand_symbols=(self.spec.precursor.ligand,),
                ).items()
            )
            kwargs: Dict[str, object] = {}
            if xtb_result is not None:
                xr = xtb_result
                orders = getattr(xr, "bond_orders", None)
                kwargs.update(
                    xtb_energy_eV=getattr(xr, "energy_eV", None),
                    xtb_gap_eV=getattr(xr, "gap_eV", None),
                    xtb_steps=int(getattr(xr, "steps", 0)),
                    xtb_converged=bool(getattr(xr, "converged", False)),
                    xtb_coordinates=packed,
                    xtb_connectivity_changed=len(getattr(xr, "connectivity_changed", ())),
                    xtb_relaxed_bonds=final_state.graph.number_of_edges(),
                    xtb_bonds_delta=final_state.graph.number_of_edges() - state.graph.number_of_edges(),
                    xtb_same_topology=same_graph,
                    xtb_bond_orders=(
                        None
                        if orders is None
                        else tuple(tuple(float(v) for v in row) for row in orders)
                    ),
                )
            iso = MolecularIsomer(
                k=self.k,
                p=self.p,
                structure_id=structure_id,
                certificate=certificate,
                atoms=final_state.atoms,
                graph=final_state.graph.copy(),
                coordinates=(
                    tuple(tuple(float(v) for v in row) for row in constructed)
                    if constructed is not None
                    else packed
                ),
                annotations=annotate_molecular_state(final_state, self.spec, packed),
                motif_inventory=motif_inventory,
                discovered_from=("" if same_graph else repr(source_certificate)),
                reconstruction_start=start_index,
                source_edges=tuple(
                    sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in state.graph.edges)
                ),
                **kwargs,
            )
            self.seen[certificate] = iso
            return iso

        produced: List[MolecularIsomer] = []
        # A factor fit that already clears the audit is a valid result even if
        # the optional physical relaxation later changes topology.
        for candidate in result.candidates:
            if self.progress is not None:
                audit_status = "PASS" if not candidate.audit_violations else "FAIL"
                violation_text = (
                    "-" if not candidate.audit_violations
                    else "|".join(candidate.audit_violations)
                )
                self.progress(
                    f"[molecular] motif graph graph{graph_number:04d} "
                    f"start{candidate.start_index:02d} | "
                    f"construction_audit={audit_status} | "
                    f"max_bond_error_A={candidate.max_bond_error_A:.4f} | "
                    f"violations={violation_text} | "
                    f"file=graph{graph_number:04d}_start{candidate.start_index:02d}_initial.xyz"
                )
            if not candidate.audit_violations:
                iso = add_isomer(
                    state,
                    candidate.coordinates,
                    start_index=candidate.start_index,
                )
                if iso is not None:
                    produced.append(iso)
                    self.bin_result.motif_pre_xtb_accepted += 1

        if xtb_settings.enabled and xtb_keep > 0:
            selected = list(result.candidates[:xtb_keep])
            batch = [
                {
                    "id": f"motif-{index}",
                    "symbols": [atom.symbol for atom in state.atoms],
                    "positions": candidate.coordinates,
                    "edges": [(int(a), int(b)) for a, b in state.graph.edges],
                }
                for index, candidate in enumerate(selected)
            ]
            cutoffs = {
                tuple(sorted(rule.elements)): float(rule.bond_max_distance)
                for rule in self.spec.graph_rules.pair_rules.values()
                if rule.bond_allowed and rule.bond_max_distance
            }
            xtb_started = time.perf_counter()
            relaxed = relax_structures(batch, xtb_settings, cutoffs)
            xtb_elapsed = time.perf_counter() - xtb_started
            for candidate, xr in zip(selected, relaxed):
                trial = trials[candidate.start_index]
                trial.xtb_ok = bool(xr.ok)
                trial.xtb_converged = bool(xr.converged)
                trial.xtb_energy_eV = getattr(xr, "energy_eV", None)
                trial.xtb_error = str(xr.error)
                trial.xtb_coordinates = xr.coordinates
                trial.final_edges = tuple(xr.relaxed_edges)
                if not xr.ok or xr.coordinates is None:
                    if self.progress is not None:
                        self.progress(
                            f"[molecular] motif graph graph{graph_number:04d} "
                            f"start{candidate.start_index:02d} | "
                            f"xtb=FAIL | converged={str(xr.converged).lower()} | "
                            f"steps={xr.steps}/{xtb_settings.max_steps} | "
                            f"max_force={('-' if xr.max_force is None else f'{float(xr.max_force):.5f}')} | "
                            f"time_s={xtb_elapsed:.3f} | "
                            f"error={xr.error or 'no_coordinates'} | "
                            "file=none"
                        )
                    continue
                if xr.converged:
                    self.bin_result.motif_xtb_converged += 1
                graph = nx.Graph()
                graph.add_nodes_from(range(len(state.atoms)))
                graph.add_edges_from(xr.relaxed_edges)
                final_state = _State(atoms=state.atoms, graph=graph)
                post_audit_started = time.perf_counter()
                # Keep all graph/motif diagnostics.  A relaxed contact can
                # trigger more than one rule at once (for example, a marginal
                # Cd--Cl contact can make a Cl appear μ3 and also create a
                # second shared bridge on one Cd pair).  The post-xTB audit is
                # advisory, so hiding the later codes behind the first one
                # makes the warning misleading.
                violations = list(
                    molecular_graph_violations(final_state, self.spec)
                )
                violations.extend(
                    molecular_decoration_rule_violations(final_state, self.spec)
                )
                if not violations:
                    from .molecular_motif_reconstruct import motif_vocabulary_violations
                    violations = motif_vocabulary_violations(
                        final_state,
                        cation=self.spec.core.cation,
                        anion=self.spec.core.anion,
                        ligand=self.spec.precursor.ligand,
                        motif_definitions=self.pack.raw.get("motifs"),
                    )
                if not violations:
                    if clash_only:
                        violations = _motif_clash_violations(
                            final_state,
                            xr.coordinates,
                            overlap_min_A=float(
                                settings_raw.get("overlap_min_A", 0.75)
                            ),
                        )
                    else:
                        violations = _exact_bond_violations(
                            final_state, xr.coordinates, self.pack, self.spec
                        )
                        violations += _exact_local_geometry_violations(
                            final_state, xr.coordinates, self.pack, self.spec
                        )
                if not violations and not clash_only:
                    _ok, violations = molecular_geometry_ok(
                        final_state, xr.coordinates, self.spec
                    )
                if violations:
                    trial.final_violations = tuple(str(v) for v in violations)
                    post_audit_elapsed = time.perf_counter() - post_audit_started
                    if self.progress is not None:
                        energy = (
                            "-" if xr.energy_eV is None
                            else f"{float(xr.energy_eV):.6f} eV"
                        )
                        self.progress(
                            f"[molecular] motif graph graph{graph_number:04d} "
                            f"start{candidate.start_index:02d} | xtb=OK | "
                            f"converged={str(xr.converged).lower()} | "
                            f"steps={xr.steps}/{xtb_settings.max_steps} | "
                            f"max_force={('-' if xr.max_force is None else f'{float(xr.max_force):.5f}')} | "
                            f"time_s={xtb_elapsed:.3f} | energy={energy} | "
                            f"post_xtb_audit=FAIL(warning; stored) | audit_time_s={post_audit_elapsed:.3f} | "
                            f"violations={'|'.join(str(v) for v in violations)} | "
                            f"file=motif_trials/graph{graph_number:04d}_start{candidate.start_index:02d}_xtb.xyz "
                            f"diagnostic_file=graph{graph_number:04d}_start{candidate.start_index:02d}_xtb_audit_warning.xyz"
                        )
                    self._record_failure_details(
                        final_state,
                        violations,
                        stage="xtb_post_audit",
                        coordinates=xr.coordinates,
                        snapshot_kind="molecule",
                    )
                    warning_iso = add_isomer(
                        final_state,
                        xr.coordinates,
                        start_index=candidate.start_index,
                        constructed=candidate.coordinates,
                        xtb_result=xr,
                    )
                    if warning_iso is not None:
                        warning_iso = replace(
                            warning_iso,
                            violations=tuple(str(v) for v in violations),
                        )
                        self.seen[warning_iso.certificate] = warning_iso
                        produced.append(warning_iso)
                        if self.progress is not None:
                            self.progress(
                                f"[molecular] motif graph graph{graph_number:04d} "
                                f"start{candidate.start_index:02d} | files="
                                f"{trial.trial_id}_xtb.xyz,{warning_iso.structure_id}.xyz,"
                                f"{warning_iso.structure_id}_xtb.xyz | "
                                "audit_warning=stored_despite_failure"
                            )
                        if warning_iso.certificate == source_certificate:
                            self.bin_result.motif_xtb_same_graph_rescues += 1
                        else:
                            self.bin_result.motif_xtb_discovered += 1
                    continue
                trial.final_violations = ()
                post_audit_elapsed = time.perf_counter() - post_audit_started
                if self.progress is not None:
                    energy = (
                        "-" if xr.energy_eV is None
                        else f"{float(xr.energy_eV):.6f} eV"
                    )
                    self.progress(
                        f"[molecular] motif graph graph{graph_number:04d} "
                        f"start{candidate.start_index:02d} | xtb=OK | "
                        f"converged={str(xr.converged).lower()} | "
                        f"steps={xr.steps}/{xtb_settings.max_steps} | "
                        f"max_force={('-' if xr.max_force is None else f'{float(xr.max_force):.5f}')} | "
                        f"time_s={xtb_elapsed:.3f} | energy={energy} | "
                        f"post_xtb_audit=PASS | audit_time_s={post_audit_elapsed:.3f}"
                    )
                iso = add_isomer(
                    final_state,
                    xr.coordinates,
                    start_index=candidate.start_index,
                    constructed=candidate.coordinates,
                    xtb_result=xr,
                )
                if iso is not None:
                    produced.append(iso)
                    if self.progress is not None:
                        self.progress(
                            f"[molecular] motif graph graph{graph_number:04d} "
                            f"start{candidate.start_index:02d} | files="
                            f"{trial.trial_id}_xtb.xyz,{iso.structure_id}.xyz,"
                            f"{iso.structure_id}_xtb.xyz"
                        )
                    if iso.certificate == source_certificate:
                        self.bin_result.motif_xtb_same_graph_rescues += 1
                    else:
                        self.bin_result.motif_xtb_discovered += 1

        if produced:
            if self.graph_checkpoint is not None:
                self.bin_result.isomers = list(self.seen.values())
                self.graph_checkpoint(self.bin_result)
            if self.progress is not None:
                audited_here = sum(
                    1 for candidate in result.candidates
                    if not candidate.audit_violations
                )
                submitted_here = (
                    min(len(result.candidates), xtb_keep)
                    if xtb_settings.enabled and xtb_keep > 0
                    else 0
                )
                converged_here = sum(
                    1 for trial in trials.values() if trial.xtb_converged
                )
                self.progress(
                    f"[molecular] motif graph graph{graph_number:04d} | "
                    f"audited={audited_here} | "
                    f"xtb_submitted={submitted_here} | "
                    f"xtb_converged={converged_here}"
                )
            return produced[0]
        first_reasons = list(result.candidates[0].audit_violations)
        self._reject(
            state,
            first_reasons or ["motif_reconstruction:xtb_no_audited_product"],
            stage="motif_reconstruction",
            coordinates=result.candidates[0].coordinates,
            snapshot_kind="molecule",
        )
        return None

    def offer(self, state: _State) -> Optional[MolecularIsomer]:
        """Screen one candidate; return the accepted isomer or ``None``."""

        graph = state.graph
        spec = self.spec
        # Everything else in ``molecular_graph_violations`` is either a skeleton
        # invariant (checked once per skeleton) or already an invariant of the
        # generator; the cation floor is the only bound that both varies per
        # candidate and is not enforced during generation.
        violations = [
            f"min_cn:{state.atoms[cation].symbol}:{cation}:"
            f"{graph.degree[cation]}<{self.cation_min_cn}"
            for cation in self.cation_ids
            if graph.degree[cation] < self.cation_min_cn
        ]
        if self.validate_every_graph:
            violations = molecular_graph_violations(state, spec)
        if violations:
            self._reject(state, violations, stage="graph")
            return None

        # Completed-graph chemistry belongs before any coordinate work.  The
        # bridge-first generator already applies these rules while expanding
        # its beam, but the final check is still required for every decorator
        # (and for resumed/precomputed graphs).  Keeping it here prevents an
        # illegal local Cd/Cl motif from consuming a frame build or an embed
        # attempt.  No new k/p-dependent rule is introduced: this delegates to
        # the existing pack-controlled finished-graph validators.
        motif_violations = molecular_decoration_rule_violations(state, spec)
        if motif_violations:
            self._reject(
                state,
                motif_violations,
                before_embed=True,
                stage="graph_motifs",
            )
            return None

        if self.collect_only:
            # Graph-level screening is done; hand the survivor to the bin so it
            # can be ranked (bond-count bands, or compactness) before anything
            # is embedded.  This has to precede the motif-factor branch: that
            # path embeds immediately, so with it first the pool stayed empty
            # and every budget/selection setting was silently inert whenever
            # reconstruction.method was motif_factor.
            self.pool.append((int(graph.number_of_edges()), state))
            return None

        if self.embed and self._motif_factor_enabled():
            return self._offer_motif_factor(state)

        frames: List[Tuple[FloatArray, List[bool]]] = []
        frame_names: List[str] = []
        ring_construction = False
        # Only ring graphs get the unconditional closure polish.  Running it on
        # every candidate was tried and reverted: it bought no acceptance and
        # actively degraded geometry the constructor had already placed well
        # (median centre-angle error 2.0 -> 6.6 deg; one Cl-Cd-Cl went from the
        # DFT median 138.1 to 156.4).  Bonds outweigh angles by ~1e5 in the
        # objective, so a whole-graph solve spends good angles to buy bond
        # length.  Structures that need it still reach it through the bounded
        # repair after a failed audit.
        ring_refinement_needed = False
        fixed_ring_nodes: Set[int] = set()
        rank: Optional[Dict[int, int]] = None
        if self.embed and self.pack is not None:
            # One canonical ranking per candidate, shared by the screen and
            # every frame it tries; recomputing it per frame cost more than the
            # screening saved.
            rank = _canonical_ranks(
                state,
                list(range(len(state.atoms))),
                [graph.degree[i] for i in range(len(state.atoms))],
            )
            frame_key = tuple(
                graph.degree[cation] for cation in self.cation_ids
            )
            ligand = spec.precursor.ligand
            inorganic_edges = tuple(
                (int(a), int(b))
                for a, b in graph.edges
                if state.atoms[a].symbol != ligand and state.atoms[b].symbol != ligand
            )
            six_rings = cdse_six_ring_sets(inorganic_edges, self.k, self.p)
            # Incidental six-cycles in a free/open graph must not silently take
            # over the ring-template path.  Ring construction is enabled only
            # for a graph mode whose policy actually requires the configured
            # ring pattern; otherwise the ordinary open-frame builder remains
            # responsible for the skeleton.
            ring_construction = bool(six_rings) and self.ring_mode in {
                "ring_first",
                "fused2",
            }
            frame_dead: List[str] = []
            if ring_construction:
                degrees = [int(graph.degree[i]) for i in range(len(state.atoms))]
                se_ids, cd_ids, _ = _index_blocks(self.k, self.p)
                se_set, cd_set = set(se_ids), set(cd_ids)
                ring_orders = [
                    order
                    for ring in six_rings
                    if (order := _alternating_six_cycle_order(
                        inorganic_edges, ring, se_set, cd_set
                    )) is not None
                ]
                ring_orders.sort(key=lambda order: tuple(sorted(order)))
                # A single ring is already fully owned by its pack template;
                # only fused/multi-ring graphs need a closure polish for the
                # additional ring(s), and the seed ring remains fixed.
                # Every ring graph gets the closure polish, not only fused
                # ones.  A single-ring graph used to receive no whole-graph
                # relaxation at all, so a bond the constructor could only
                # stretch stayed stretched -- which is most of what
                # ``bond_geometry`` was reporting.
                ring_refinement_needed = bool(ring_orders)
                if ring_orders:
                    fixed_ring_nodes = set(ring_orders[0])
                # No bond override any more: the ring template is built from the
                # CN-indexed bond table, which is the same source the audit
                # reads, so overriding the audit to a flat ``bond_cdse_A`` would
                # reintroduce exactly the disagreement this removes.
                # Chair is the required acceptance conformation.  Boat is
                # fitted lazily only after chair succeeds below.
                for conf in ("chair",):
                    if conf not in self.pack.cdse6_conformations():
                        continue
                    cache_key = (
                        tuple(sorted(inorganic_edges)),
                        frame_key,
                        conf,
                    )
                    if cache_key not in self.ring_frame_cache:
                        self.ring_frame_cache[cache_key] = _try_pack_ring_frame(
                            state,
                            inorganic_edges,
                            self.k,
                            self.p,
                            self.pack,
                            spec,
                            degrees,
                            conformation=conf,
                        )
                    option = self.ring_frame_cache[cache_key]
                    if option is not None:
                        frames.append(option)
                        frame_names.append(conf)
                # A ring graph can still have a valid whole-graph frame even
                # when the selected ring template cannot be reconciled with
                # the final CN-dependent bond/angle table.  Keep the graph
                # candidate alive and try the ordinary frame builder before
                # the outer ring-policy fallback discards the family.
                if not frames:
                    fallback_frames, fallback_dead = _clean_frames(
                        state,
                        self.pack,
                        spec,
                        degrees,
                        limit=self.frame_options,
                    )
                    if fallback_frames:
                        frames = fallback_frames
                        frame_names = ["open"] * len(fallback_frames)
                        ring_construction = False
                    elif fallback_dead:
                        frame_dead.extend(fallback_dead)
            else:
                cached = self.frame_cache.get(frame_key)
                if cached is None:
                    built, dead = _clean_frames(
                        state, self.pack, spec, limit=self.frame_options
                    )
                    self.frame_cache[frame_key] = cached = (built, dead or [])
                frames, frame_dead = cached
                frame_names = [""] * len(frames)
            if not frames:
                self._reject(
                    state,
                    frame_dead or ["frame_not_realisable"],
                    before_embed=True,
                    stage="frame_build",
                )
                return None
            # The bridge span and the anion-anion angles depend only on where
            # the frame put the cations, so both can rule a frame out for a few
            # microseconds instead of a few milliseconds of embedding.  A
            # candidate is hopeless only if *no* sound frame survives them.
            survivors = []
            survivor_names: List[str] = []
            blocked: List[str] = []
            for name, option in zip(frame_names, frames):
                reasons = bridge_feasibility_violations(
                    state, option[0], self.pack, spec
                )
                if not reasons and not ring_construction:
                    reasons = local_angle_violations(
                        state, option[0], self.pack, spec, rank=rank
                    )
                if reasons:
                    if not blocked:
                        blocked = reasons
                    continue
                survivors.append(option)
                survivor_names.append(name)
            if not survivors:
                self._reject(
                    state,
                    blocked or ["frame_not_realisable"],
                    before_embed=True,
                    stage="frame_precheck",
                    coordinates=frames[0][0] if frames else None,
                    snapshot_kind="inorganic",
                )
                return None
            frames = survivors
            frame_names = survivor_names

        self.bin_result.unique_graphs += 1
        coords = None
        conformers: List[Tuple[str, Tuple[Tuple[float, float, float], ...]]] = []
        if self.embed and self.pack is not None:
            self.bin_result.embedded += 1
            first_reasons: List[str] = []

            def evaluate_option(
                name: str,
                option: Tuple[FloatArray, List[bool]],
            ) -> Tuple[
                Optional[FloatArray],
                List[str],
                Optional[FloatArray],
                bool,
            ]:
                """Embed, refine, and audit one already-screened frame."""

                relaxed = False
                try:
                    candidate = embed_molecular_state(
                        state, self.pack, spec, inorganic=option, rank=rank
                    )
                    if ring_refinement_needed:
                        if name == "chair":
                            self.bin_result.chair_refinements += 1
                        else:
                            self.bin_result.boat_refinements += 1
                        candidate = _refine_completed_ring_graph(
                            state,
                            candidate,
                            self.pack,
                            spec,
                            self.k,
                            self.p,
                            name or "chair",
                            fixed_ring_nodes=fixed_ring_nodes,
                        )
                except ExactEmbeddingError as exc:
                    # Only bridge motifs can be rescued by moving the host
                    # frame.  Keep terminal-only failures on the exact path;
                    # this avoids an expensive nonlinear solve for every
                    # ordinary rejected decoration.
                    _terminals, _bridges, _multi_host = _ligand_groups(
                        state, spec, rank
                    )
                    if (not _bridges and not _multi_host) or not any(
                        "sphere" in reason or "bridge_hosts" in reason
                        for reason in exc.reasons
                    ):
                        return None, list(exc.reasons), None, False
                    if self.relaxed_geometry_attempts >= self.relax_attempt_budget:
                        return None, list(exc.reasons), None, False
                    self.relaxed_geometry_attempts += 1
                    relax_stats: Dict[str, float] = {}
                    try:
                        candidate = _relaxed_complete_geometry(
                            state,
                            self.pack,
                            spec,
                            option,
                            log=self._relax_log,
                            stats=relax_stats,
                        )
                        self._record_optimizer_stats(relax_stats)
                        relaxed = True
                    except ExactEmbeddingError as fallback_exc:
                        self._record_optimizer_stats(relax_stats)
                        # If the broad bridge seed produced coordinates, feed
                        # them into the constrained angle/bond solve before
                        # declaring the bridge unrealisable.
                        if (
                            fallback_exc.coordinates is not None
                            and not any(
                                reason.startswith("bridge_hosts_too_far:")
                                for reason in fallback_exc.reasons
                            )
                        ):
                            relax_stats = {}
                            repaired = _steric_relax_ligands(
                                state,
                                self.pack,
                                spec,
                                fallback_exc.coordinates,
                                log=self._relax_log,
                                stats=relax_stats,
                            )
                            self._record_optimizer_stats(relax_stats)
                            if repaired is not None:
                                candidate = repaired
                                relaxed = False
                            else:
                                candidate = None
                        else:
                            candidate = None
                        if candidate is not None:
                            # Continue through the common hard audit below.
                            pass
                        else:
                            # Preserve the original sphere diagnosis, but
                            # expose the bounded repair reason as well so
                            # rejection manifests explain what failed.
                            fallback_reasons = list(fallback_exc.reasons)
                            for reason in exc.reasons:
                                if reason not in fallback_reasons:
                                    fallback_reasons.append(reason)
                            return (
                                None,
                                fallback_reasons,
                                fallback_exc.coordinates,
                                False,
                            )
                    except Exception as fallback_error:  # noqa: BLE001
                        self._record_optimizer_stats(relax_stats)
                        return (
                            None,
                            [
                                "relaxed_geometry_failed:"
                                f"{type(fallback_error).__name__}"
                            ],
                            None,
                            False,
                        )
                    if relaxed:
                        relax_stats = {}
                        repaired = _steric_relax_ligands(
                            state,
                            self.pack,
                            spec,
                            candidate,
                            log=self._relax_log,
                            stats=relax_stats,
                        )
                        self._record_optimizer_stats(relax_stats)
                        if repaired is not None:
                            candidate = repaired
                            relaxed = False
                    if ring_refinement_needed and not relaxed:
                        candidate = _refine_completed_ring_graph(
                            state,
                            candidate,
                            self.pack,
                            spec,
                            self.k,
                            self.p,
                            name or "chair",
                            fixed_ring_nodes=fixed_ring_nodes,
                        )
                def audit_geometry(
                    geometry: FloatArray,
                ) -> List[str]:
                    violations = _exact_bond_violations(
                        state,
                        geometry,
                        self.pack,
                        spec,
                        relaxed=relaxed,
                    )
                    # Ring endocyclic geometry is owned by the rigid template,
                    # but completed CN3/CN4 centres still have to satisfy the
                    # ordinary pack angle/improper audit.
                    violations += _exact_local_geometry_violations(
                        state, geometry, self.pack, spec
                    )
                    if not violations:
                        ok_geom, geom_viol = molecular_geometry_ok(
                            state, geometry, spec
                        )
                        if not ok_geom:
                            violations = geom_viol
                    return violations

                reasons = audit_geometry(candidate)
                # Contact and local CN3 angle/improper failures can be repaired
                # by moving ligands (and non-ring atoms) while the rigid ring
                # seed remains fixed.  Bond/frame failures stay deterministic.
                local_angle_failure = any(
                    reason.startswith(("improper:", "angle_geometry:"))
                    for reason in reasons
                )
                repairable_failure = any(
                    reason.startswith((
                        "contact:", "overlap:", "missing_edge:",
                        "frame_contact:", "improper:",
                        "angle_geometry:",
                    ))
                    for reason in reasons
                )
                budget_used = (
                    self.local_angle_relax_attempts
                    if local_angle_failure
                    else self.steric_relax_attempts
                )
                if (
                    reasons
                    and not relaxed
                    and budget_used < self.relax_attempt_budget
                    and repairable_failure
                ):
                    if local_angle_failure:
                        self.local_angle_relax_attempts += 1
                    else:
                        self.steric_relax_attempts += 1
                    relax_stats = {}
                    repaired = _steric_relax_ligands(
                        state,
                        self.pack,
                        spec,
                        candidate,
                        log=self._relax_log,
                        stats=relax_stats,
                    )
                    self._record_optimizer_stats(relax_stats)
                    if repaired is not None:
                        repaired_reasons = audit_geometry(repaired)
                        if not repaired_reasons:
                            return repaired, [], repaired, False
                        candidate = repaired
                        reasons = repaired_reasons
                if not reasons:
                    return candidate, [], candidate, relaxed
                return None, reasons, candidate, relaxed

            first_failed_coordinates: Optional[FloatArray] = None
            first_failed_kind = "inorganic"
            for name, option in zip(frame_names, frames):
                candidate, reasons, failed_candidate, _relaxed = evaluate_option(
                    name or "chair", option
                )
                if candidate is not None:
                    packed = tuple(
                        (float(x), float(y), float(z)) for x, y, z in candidate
                    )
                    if ring_construction:
                        conformers.append((name or "chair", packed))
                        coords = candidate
                        break
                    coords = candidate
                    break
                if not first_reasons:
                    first_reasons = list(reasons)
                    first_failed_coordinates = (
                        failed_candidate if failed_candidate is not None else option[0]
                    )
                    first_failed_kind = (
                        "molecule" if failed_candidate is not None else "inorganic"
                    )
            if ring_construction and not any(name == "chair" for name, _ in conformers):
                self.bin_result.ring_refinement_failures += 1
                self._reject(
                    state,
                    first_reasons or ["chair_not_realisable"],
                    stage="embed",
                    coordinates=first_failed_coordinates,
                    snapshot_kind=first_failed_kind,
                )
                return None
            if coords is None:
                self._reject(
                    state,
                    first_reasons or ["frame_not_realisable"],
                    stage="embed",
                    coordinates=first_failed_coordinates,
                    snapshot_kind=first_failed_kind,
                )
                return None

            # Boat is optional: only spend the nonlinear-fit cost after the
            # required chair has passed all final audits.
            if ring_construction and any(name == "chair" for name, _ in conformers):
                if "boat" in self.pack.cdse6_conformations():
                    boat_key = (
                        tuple(sorted(inorganic_edges)),
                        frame_key,
                        "boat",
                    )
                    if boat_key not in self.ring_frame_cache:
                        self.ring_frame_cache[boat_key] = _try_pack_ring_frame(
                            state,
                            inorganic_edges,
                            self.k,
                            self.p,
                            self.pack,
                            spec,
                            degrees,
                            conformation="boat",
                        )
                    boat_option = self.ring_frame_cache[boat_key]
                    if boat_option is not None:
                        boat_reasons = bridge_feasibility_violations(
                            state, boat_option[0], self.pack, spec
                        )
                        if not boat_reasons:
                            boat_candidate, _boat_failures, _boat_failed, _boat_relaxed = evaluate_option(
                                "boat", boat_option
                            )
                            if boat_candidate is not None:
                                conformers.append(
                                    (
                                        "boat",
                                        tuple(
                                            (float(x), float(y), float(z))
                                            for x, y, z in boat_candidate
                                        ),
                                    )
                                )

        # Construction defaults + optional collapse filters (pack-controlled).
        # Finished-graph decoration rules were checked before frame construction
        # above.  Geometry-dependent bridge-maximal and closable-terminal rules
        # remain here because they require coordinates.
        if (
            self.embed
            and self.pack is not None
            and spec.graph_rules.require_bridge_maximal
            and frames
        ):
            maximal = bridge_maximal_violations(
                state, self.pack, spec, frames
            )
            if maximal:
                self._reject(
                    state,
                    maximal,
                    stage="decoration_rules",
                    coordinates=coords,
                    snapshot_kind="molecule",
                )
                return None

        annotations = annotate_molecular_state(state, spec, coords)
        if (
            spec.graph_rules.reject_closable_terminal_cd2
            and annotations.n_closable_terminal_cd2 > 0
        ):
            self._reject(
                state,
                [
                    f"closable_terminal_cd2:"
                    f"{annotations.n_closable_terminal_cd2}"
                ],
                stage="decoration_rules",
                coordinates=coords,
                snapshot_kind="molecule",
            )
            return None

        certificate = _graph_certificate(state)
        if certificate in self.processed:
            return None
        self.processed.add(certificate)
        isomer = MolecularIsomer(
            k=self.k,
            p=self.p,
            structure_id=(
                f"k{self.k:03d}_p{self.p:03d}_mol{len(self.seen) + 1:04d}"
            ),
            certificate=certificate,
            atoms=state.atoms,
            graph=graph.copy(),
            coordinates=coords,
            annotations=annotations,
            conformers=tuple(conformers),
            motif_inventory=tuple(
                coordination_motif_inventory(
                    state,
                    cation_symbols=(spec.core.cation, spec.precursor.center),
                    anion_symbols=(spec.core.anion,),
                    ligand_symbols=(spec.precursor.ligand,),
                ).items()
            ),
        )
        self.seen[certificate] = isomer
        if self.progress is not None:
            self.progress(molecular_isomer_log_line(isomer, spec))
        return isomer


def _relax_bin_with_xtb(
    bin_result: MolecularBinResult,
    pack: Optional[GeometryPack],
    spec: NucleationSpec,
    known: Optional[Mapping[Tuple[object, ...], str]],
    progress: Optional[ProgressCallback],
) -> None:
    """Relax every accepted isomer in one xTB batch and record the results.

    Runs after acceptance, not instead of it: the construction audit decides
    which graphs are worth the quantum cost, and xTB then gives each survivor a
    geometry and a total energy.  A structure whose bonding drifts during
    relaxation is kept but flagged -- it is no longer the isomer that was
    enumerated, and that is the caller's decision to act on.
    """

    if pack is None or not bin_result.isomers:
        return
    reconstruction = pack.raw.get("reconstruction") or {}
    if str(reconstruction.get("method", "")).strip().lower() == "motif_factor":
        # The motif path deliberately relaxes viable pre-audit starts and has
        # already attached its xTB results.  Running this accepted-only batch
        # would duplicate the calculation and erase discovery provenance.
        return
    settings = XtbSettings.from_pack(pack.raw.get("relaxation"))
    if not settings.enabled:
        return
    batch = [
        {
            "id": iso.structure_id,
            "symbols": [atom.symbol for atom in iso.atoms],
            "positions": iso.coordinates,
            "edges": [(int(a), int(b)) for a, b in iso.graph.edges],
        }
        for iso in bin_result.isomers
        if iso.coordinates is not None
    ]
    if not batch:
        return
    if progress is not None:
        progress(
            f"    xtb: relaxing {len(batch)} accepted isomer(s) "
            f"with {settings.method}"
        )
    cutoffs = {
        tuple(sorted(rule.elements)): float(rule.bond_max_distance)
        for rule in spec.graph_rules.pair_rules.values()
        if rule.bond_allowed and rule.bond_max_distance
    }
    results = relax_structures(batch, settings, cutoffs)
    by_id = {entry["id"]: res for entry, res in zip(batch, results)}
    ok = 0
    updated: List[MolecularIsomer] = []
    for iso in bin_result.isomers:
        res = by_id.get(iso.structure_id)
        if res is None:
            updated.append(iso)
        elif not res.ok:
            updated.append(replace(iso, xtb_error=res.error))
        else:
            ok += 1
            after = nx.Graph()
            after.add_nodes_from(range(len(iso.atoms)))
            after.add_edges_from(res.relaxed_edges)
            ligand = spec.precursor.ligand
            motifs = Counter(
                after.degree[a.atom_id]
                for a in iso.atoms
                if a.symbol == ligand
            )
            relaxed_state = _State(atoms=iso.atoms, graph=after)
            cert = _graph_certificate(relaxed_state)
            updated.append(
                replace(
                    iso,
                    xtb_energy_eV=res.energy_eV,
                    xtb_gap_eV=res.gap_eV,
                    xtb_steps=res.steps,
                    xtb_converged=res.converged,
                    xtb_coordinates=res.coordinates,
                    xtb_connectivity_changed=len(res.connectivity_changed),
                    xtb_relaxed_bonds=after.number_of_edges(),
                    xtb_bonds_delta=(
                        after.number_of_edges() - iso.graph.number_of_edges()
                    ),
                    xtb_relaxed_cl_motifs=(
                        motifs.get(1, 0), motifs.get(2, 0),
                        sum(v for k_, v in motifs.items() if k_ >= 3),
                    ),
                    xtb_same_topology=(cert == iso.certificate),
                    xtb_matches=(
                        "" if cert == iso.certificate
                        else (known or {}).get(cert, "")
                    ),
                )
            )
    bin_result.isomers = updated
    if progress is not None and ok:
        energies = [
            iso.xtb_energy_eV
            for iso in bin_result.isomers
            if iso.xtb_energy_eV is not None
        ]
        drifted = sum(
            1 for iso in bin_result.isomers if iso.xtb_connectivity_changed
        )
        span = (max(energies) - min(energies)) if len(energies) > 1 else 0.0
        progress(
            f"    xtb: {ok}/{len(batch)} relaxed | spread {span:.2f} eV | "
            f"{drifted} with changed connectivity"
        )


def _log_xtb_energy_ranking(
    bin_result: MolecularBinResult,
    progress: Optional[ProgressCallback],
) -> None:
    """Print xTB energies ranked relative to the lowest-energy isomer."""

    if progress is None:
        return
    energies = [
        (iso.xtb_energy_eV, iso.structure_id)
        for iso in bin_result.isomers
        if iso.xtb_energy_eV is not None
    ]
    if not energies:
        return
    energies.sort(key=lambda item: float(item[0]))
    reference = float(energies[0][0])
    progress(
        f"    xTB energy ranking k={bin_result.k} p={bin_result.p} "
        "(relative kcal/mol; rank 1 = 0.000)"
    )
    for rank, (energy_eV, structure_id) in enumerate(energies, start=1):
        delta_kcal = (float(energy_eV) - reference) * 23.060548
        progress(
            f"      {rank:3d} {structure_id} "
            f"ΔE={delta_kcal: .3f} kcal/mol "
            f"E={float(energy_eV): .6f} eV"
        )


def enumerate_molecular_bin(
    k: int,
    p: int,
    spec: NucleationSpec,
    *,
    pack: Optional[GeometryPack] = None,
    embed: bool = True,
    max_skeletons: int = 2000,
    max_decoration_assignments: int = 0,
    extra_skeleton_edges: Optional[int] = None,
    allow_incomplete: bool = False,
    validate_every_graph: bool = False,
    frame_options: int = 0,
    dump_failures: bool = False,
    target_isomers: int = 0,
    progress: Optional[ProgressCallback] = None,
    graph_checkpoint: Optional[Callable[[MolecularBinResult], None]] = None,
    checkpoint: Optional[
        Callable[[MolecularBinResult, int, int], None]
    ] = None,
    precomputed_skeletons: Optional[
        Sequence[Tuple[Tuple[int, int], ...]]
    ] = None,
    skeleton_mode: Optional[str] = None,
    allow_ring_fallback: bool = True,
    min_structure_level: int = 0,
    _ring_fallback_depth: int = 0,
    _structure_mode_queue: Optional[Sequence[str]] = None,
) -> MolecularBinResult:
    """Enumerate unique legal molecular graphs for one (k, p).

    ``precomputed_skeletons`` skips inorganic re-enumeration and decorates the
    given Cd–Se edge sets (e.g. loaded from a prior skeleton dump).

    ``skeleton_mode``: ``free`` | ``ring_first`` | ``fused2`` | ``auto``.
    With adaptive fallback, structured levels that yield zero accepted isomers
    fall back to lower levels (fused2 → 1-ring → free), never below
    ``min_structure_level``.  Within fused2, all fusion modes are generated
    (no ranking).

    ``validate_every_graph`` re-runs the full ``molecular_graph_violations``
    on every candidate instead of the reduced per-decoration check.  It is a
    slow self-audit: the two must agree, and the test suite asserts they do.

    ``frame_options`` is how many sound cation-anion frames are kept per
    coordination vector before a molecule is called unrealisable; ``0``, the
    default, keeps every one and so saturates.  It buys structures directly --
    at ``k=2, p=3`` the accepted count runs 20, 34, 50, 55, 59 for 1, 2, 4, 8,
    16 frames and then stops -- because a frame with no defect of its own can
    still leave a cation unable to take its ligands while another slot order for
    the same coordination numbers can.  Lower it only to trade completeness for
    time; ``local_angle_violations`` is what keeps saturating affordable.
    """

    if k < 1:
        raise ValueError("k must be >= 1")
    if p < 0:
        raise ValueError("p must be >= 0")

    check_spec = _molecular_check_spec(spec)
    fallback_enabled = bool(
        getattr(check_spec.graph_rules, "ring_first_fallback_to_open", True)
    ) and bool(allow_ring_fallback)

    # Build ordered list of structure levels to try (high → low).
    level_to_mode = {2: "fused2", 1: "ring_first", 0: "free"}
    mode_to_level = {"fused2": 2, "ring_first": 1, "free": 0, "precomputed": -1}
    requested = (skeleton_mode or "auto").strip().lower()
    if _structure_mode_queue:
        modes_to_try = [str(m) for m in _structure_mode_queue]
    elif precomputed_skeletons is not None or requested == "precomputed":
        modes_to_try = ["precomputed"]
    elif requested in {"free", "ring_first", "fused2"}:
        modes_to_try = [requested]
    else:
        # auto ladder: prefer the strongest level already proved for this k;
        # lower levels are appended below when adaptive fallback is enabled.
        max_L = max_structure_level_possible(k, p, check_spec)
        min_L = max(0, int(min_structure_level))
        floor_L = min(min_L, max_L)
        modes_to_try = [
            level_to_mode[L]
            for L in range(max_L, floor_L - 1, -1)
            if L in level_to_mode
        ]
        # ``min_structure_level`` records the strongest structure previously
        # proved for this k and keeps that level preferred for later p.  It
        # must not disable the documented adaptive fallback: if the current
        # ring level produces no accepted molecules, lower levels (including
        # free/open) still need to be tried for this (k, p) bin.
        if fallback_enabled and floor_L > 0:
            modes_to_try.extend(
                level_to_mode[L]
                for L in range(floor_L - 1, -1, -1)
                if L in level_to_mode
            )
        if not modes_to_try:
            modes_to_try = (
                ["free"] if min_L == 0 else [level_to_mode.get(min_L, "free")]
            )

    # First mode for this call; further levels via recursive fallback.
    resolved_mode = modes_to_try[0]
    remaining_fallback_modes = (
        list(modes_to_try[1:]) if fallback_enabled else []
    )
    symbols = _symbols_for_composition(spec, k, p)
    roles = _roles_for_composition(spec, k, p)
    atoms = _atoms_for_composition(symbols, roles)
    bin_result = MolecularBinResult(k=k, p=p)
    bin_result.ring_min_pattern_cd = tuple(
        int(value) for value in check_spec.graph_rules.ring_min_pattern_cd
    )
    bin_result.ring_min_pattern_se = tuple(
        int(value) for value in check_spec.graph_rules.ring_min_pattern_se
    )
    if pack is not None:
        pack_pattern = pack.cdse6_ring_pattern()
        bin_result.geometry_ring_pattern_cd = tuple(
            int(value) for value in pack_pattern.cd_cn
        )
        bin_result.geometry_ring_pattern_se = tuple(
            int(value) for value in pack_pattern.se_cn
        )
    seen: Dict[Tuple[object, ...], MolecularIsomer] = {}
    processed: set[Tuple[object, ...]] = set()
    cation_ids = [
        atom.atom_id
        for atom in atoms
        if atom.symbol in {check_spec.core.cation, check_spec.precursor.center}
    ]
    cation_min_cn = (
        int(check_spec.graph_rules.min_cn.get(check_spec.core.cation, 0))
        if check_spec.enforce_min_cn
        else 0
    )
    screen = _CandidateScreen(
        k=k,
        p=p,
        spec=check_spec,
        pack=pack,
        embed=embed,
        cation_ids=cation_ids,
        cation_min_cn=cation_min_cn,
        bin_result=bin_result,
        seen=seen,
        processed=processed,
        progress=progress,
        graph_checkpoint=graph_checkpoint,
        validate_every_graph=validate_every_graph,
        frame_options=frame_options,
        dump_failures=dump_failures,
        collect_only=bool(
            target_isomers
            or float(
                getattr(spec.graph_rules, "selection_top_fraction", 0.0) or 0.0
            ) > 0.0
            or float(
                getattr(spec.graph_rules, "selection_max_wiener_excess", 0.0)
                or 0.0
            ) > 0.0
        ) and embed,
    )

    skeletons_truncated = False
    if precomputed_skeletons is not None or resolved_mode == "precomputed":
        skeletons = [
            tuple(sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in skel))
            for skel in (precomputed_skeletons or ())
        ]
        if progress is not None:
            progress(
                f"    inorganic skeletons={len(skeletons)} "
                f"(from saved dump; no re-enumeration)"
            )
    else:
        enum_mode = resolved_mode
        if progress is not None:
            if enum_mode == "fused2":
                progress(
                    "    fused-2 try → expect 2-ring closed skeletons "
                    "(all modes path∪face∪edge"
                    + (
                        "; fall back to 1-ring/free if 0 accepted)"
                        if remaining_fallback_modes
                        else ")"
                    )
                )
            elif enum_mode == "ring_first":
                progress(
                    "    1-ring closed try (Cd3Se3 seed"
                    + (
                        "; fallback if 0 accepted)"
                        if remaining_fallback_modes
                        else ")"
                    )
                )
            else:
                progress(
                    "    skeleton mode: free "
                    "(0-ring open allowed, still no 4-rings)"
                )
        skeleton_started = time.perf_counter()
        skeletons, skeletons_truncated = _enumerate_inorganic_edge_sets(
            k,
            p,
            check_spec,
            max_skeletons=max_skeletons,
            extra_skeleton_edges=extra_skeleton_edges,
            mode=enum_mode,
            pack=pack,
        )
        bin_result.skeleton_generation_time_s = time.perf_counter() - skeleton_started
        if skeletons_truncated:
            bin_result.incomplete = True
            if not allow_incomplete:
                raise EnumerationLimitError(
                    f"k={k} p={p}: skeleton enumeration hit a safety guard "
                    f"(max_skeletons={max_skeletons}, "
                    f"extra_skeleton_edges={extra_skeleton_edges}); rerun with a "
                    "larger --max-skeletons / --extra-skeleton-edges"
                )
        if progress is not None:
            # Summarize ring closure counts for this batch
            ring_hist: Dict[int, int] = {}
            for sk in skeletons:
                nr = count_cdse_six_rings(sk, k, p)
                ring_hist[nr] = ring_hist.get(nr, 0) + 1
            hist_txt = ", ".join(
                f"{ring_closure_log_label(nr, pattern_possible=ring_first_required_for_spec(k, p, check_spec))}×{cnt}"
                for nr, cnt in sorted(ring_hist.items())
            )
            progress(
                f"    inorganic skeletons={len(skeletons)}"
                + (" (TRUNCATED)" if skeletons_truncated else "")
                + f" [{enum_mode}]"
                + (f" | {hist_txt}" if hist_txt else "")
                + f" | generation_time_s={bin_result.skeleton_generation_time_s:.3f}"
            )
    decoration_started = time.perf_counter()
    decoration_generation_time = 0.0
    candidate_screen_time = 0.0
    bin_result.skeletons_total = len(skeletons)
    se_ids_list = [
        atom.atom_id
        for atom in atoms
        if atom.symbol == check_spec.core.anion
    ]
    for skeleton_index, skel in enumerate(skeletons, start=1):
        screen.skeleton_index = skeleton_index
        skeleton_graph = nx.Graph()
        skeleton_graph.add_nodes_from(range(len(atoms)))
        skeleton_graph.add_edges_from((int(a), int(b)) for a, b in skel)
        skel_state = _State(atoms=atoms, graph=skeleton_graph)
        cd_cn = tuple(
            sorted(int(skeleton_graph.degree[i]) for i in cation_ids)
        )
        se_cn = tuple(
            sorted(int(skeleton_graph.degree[i]) for i in se_ids_list)
        )
        skel_edges = tuple(
            (int(a), int(b)) for a, b in sorted(skel)
        )
        inorg_ids = list(cation_ids) + se_ids_list
        inorg_symbols = tuple(atoms[i].symbol for i in inorg_ids)

        def _record(
            status: str,
            reason: str = "",
            coords: Optional[Tuple[Tuple[float, float, float], ...]] = None,
        ) -> None:
            bin_result.skeleton_records.append(
                MolecularSkeletonRecord(
                    skeleton_index=skeleton_index,
                    n_edges=len(skel),
                    cd_cn=cd_cn,
                    se_cn=se_cn,
                    status=status,
                    reason=reason,
                    edges=skel_edges,
                    coordinates=coords,
                    symbols=inorg_symbols,
                )
            )

        skeleton_violations = _skeleton_graph_violations(
            skel_state, check_spec
        )
        if skeleton_violations:
            bin_result.skeletons_pruned_graph += 1
            reason = ", ".join(skeleton_violations)
            _record("skipped_graph", reason)
            if progress is not None:
                progress(
                    f"    skeleton {skeleton_index}/{len(skeletons)} | "
                    f"SKIP | Cd{list(cd_cn)} Se{list(se_cn)} | {reason}"
                )
            continue
        decoration_mode = str(
            getattr(check_spec.graph_rules, "decoration_mode", "graph_multiset")
            or "graph_multiset"
        ).strip().lower()
        profile_mode = resolved_mode
        if profile_mode == "precomputed":
            level = max_structure_level_possible(k, p, check_spec)
            profile_mode = structure_mode_for_level(level)
        screen.ring_mode = profile_mode
        forced_rings, ring_profiles = forced_ring_degree_profiles(
            skel, k, p, check_spec, mode=profile_mode
        )
        if profile_mode in {"ring_first", "fused2"} and not ring_profiles:
            bin_result.skeletons_pruned_graph += 1
            reason = "ring_min_pattern_not_feasible"
            _record("skipped_graph", reason)
            if progress is not None:
                progress(
                    f"    skeleton {skeleton_index}/{len(skeletons)} | SKIP | "
                    f"Cd{list(cd_cn)} Se{list(se_cn)} | {reason}"
                )
            continue
        survey = _SkeletonSurvey()
        lazy_builder: Optional[
            Callable[[Tuple[int, ...]], Optional[_DegreeSlice]]
        ] = None
        skel_coords: Optional[Tuple[Tuple[float, float, float], ...]] = None
        if (
            embed
            and pack is not None
            and decoration_mode not in {
                "skeleton_bridge_first", "motif_graph", "motif_bridge_first"
            }
        ):
            survey = survey_skeleton_frames(
                skel_state,
                cation_ids,
                pack,
                check_spec,
                2 * p,
                frame_options=frame_options,
                eager=False,
            )
            if survey.dead_reason is not None:
                bin_result.skeletons_pruned_frame += 1
                bin_result.rejection_reasons[
                    survey.dead_reason.split(":", 1)[0]
                ] += 1
                bin_result.rejection_details[survey.dead_reason] += 1
                _record("skipped_frame", survey.dead_reason)
                if progress is not None:
                    progress(
                        f"    skeleton {skeleton_index}/{len(skeletons)} | "
                        f"SKIP | Cd{list(cd_cn)} Se{list(se_cn)} | "
                        f"frame unrealisable | {survey.dead_reason}"
                    )
                continue
            if survey.pending_degrees is not None:
                lazy_builder = _lazy_slice_builder(survey)
            # Embed skeleton-only for XYZ dump (Cd/Se positions).
            # Use ring-aware frame CNs (Cd 3/4 angles), not low skeleton CN.
            try:
                frames, _dead, _fdeg = embed_skeleton_frames(
                    skel_state,
                    skel,
                    k,
                    p,
                    pack,
                    check_spec,
                    limit=1,
                )
                if frames:
                    full = np.asarray(frames[0][0], dtype=float)
                    skel_coords = tuple(
                        (float(full[i, 0]), float(full[i, 1]), float(full[i, 2]))
                        for i in inorg_ids
                    )
            except Exception:  # noqa: BLE001 — dump is best-effort
                skel_coords = None
        remaining_se_slots, graph_p_ceiling = _skeleton_se_capacity(
            k, p, skel, check_spec
        )
        bin_result.skeleton_records.append(
            MolecularSkeletonRecord(
                skeleton_index=skeleton_index,
                n_edges=len(skel),
                cd_cn=cd_cn,
                se_cn=se_cn,
                status="accepted",
                edges=skel_edges,
                coordinates=skel_coords,
                symbols=inorg_symbols,
                forced_rings=forced_rings,
            )
        )
        if progress is not None:
            progress(
                f"    skeleton {skeleton_index}/{len(skeletons)} | "
                f"Cd{list(cd_cn)} Se{list(se_cn)} | "
                f"Cd-Se bonds={len(skel)} | "
                f"open Se slots={remaining_se_slots}, "
                f"graph p ceiling={graph_p_ceiling} | "
                "enumerating symmetry-reduced chloride assignments..."
            )
        if max_decoration_assignments < 0:
            # Skeleton inventory only (dump tool); no Cl enumeration.
            if progress is not None:
                progress(
                    f"    skeleton {skeleton_index}/{len(skeletons)} | "
                    f"SKIP decoration (skeleton inventory mode)"
                )
            continue

        decoration_status = _DecorationStatus()
        if decoration_mode in {"skeleton_bridge_first", "motif_bridge_first"} or (
            decoration_mode in {"pack_sites", "tet_sites"}
            and embed and pack is not None
        ):
            state_skel = _State(atoms=atoms, graph=skeleton_graph)
            allowed_bridge_pairs: Optional[Set[Tuple[int, int]]] = None
            if decoration_mode == "skeleton_bridge_first" and embed and pack is not None:
                # The graph rules decide chemical legality, but a bridge also
                # needs a nearby Cd host pair.  Use a conservative skeleton
                # frame only to remove grossly impossible pairs; final CN
                # radii are checked again during exact embedding.  If the
                # advisory frame cannot be built, retain the graph-only path.
                try:
                    preview_frames, _preview_dead, _preview_degrees = (
                        embed_skeleton_frames(
                            state_skel,
                            skel,
                            k,
                            p,
                            pack,
                            check_spec,
                            limit=1,
                        )
                    )
                    if preview_frames:
                        preview = preview_frames[0][0]
                        cd_preview = list(cation_ids)
                        max_bridge_radius = max(
                            pack.bond_length(
                                "CdCl_bridge", cn_host, cn_cl, default=3.0
                            )
                            for cn_host in (2, 3, 4, 5)
                            for cn_cl in (2, 3)
                        )
                        configured_max = (
                            check_spec.graph_rules.bridge_cd_cd_max_distance
                        )
                        conservative_max = (
                            float(configured_max)
                            if configured_max is not None
                            else 2.0 * max_bridge_radius + 0.25
                        )
                        nearby = {
                            (min(left, right), max(left, right))
                            for offset, left in enumerate(cd_preview)
                            for right in cd_preview[offset + 1 :]
                            if float(np.linalg.norm(preview[right] - preview[left]))
                            <= conservative_max
                        }
                        if nearby or configured_max is not None:
                            allowed_bridge_pairs = nearby
                            if progress is not None:
                                span_label = (
                                    "configured bridge span"
                                    if configured_max is not None
                                    else "conservative bridge span"
                                )
                                progress(
                                    "      bridge host prefilter: "
                                    f"{len(nearby)}/{len(cation_ids) * (len(cation_ids) - 1) // 2} "
                                    f"Cd pairs within {span_label}"
                                )
                except Exception:  # noqa: BLE001 — advisory filter only
                    allowed_bridge_pairs = None
            if decoration_mode in {"skeleton_bridge_first", "motif_bridge_first"}:
                from .molecular_bridge_first import (
                    iter_cl_attachments_bridge_first,
                )

                decorations = iter_cl_attachments_bridge_first(
                    k,
                    p,
                    skel,
                    check_spec,
                    pack,
                    max_assignments=max_decoration_assignments,
                    status=decoration_status,
                    frame_options=frame_options,
                    state=state_skel,
                    cation_ids=cation_ids,
                    degree_vectors=survey.pending_degrees,
                    slice_builder=lazy_builder,
                    required_degree_profiles=ring_profiles,
                    allowed_bridge_pairs=allowed_bridge_pairs,
                    hard_max_bridge_per_cd=int(
                        check_spec.graph_rules.bridge_first_hard_max_bridges_per_cd
                    ),
                    strict_bridge_first=(decoration_mode == "motif_bridge_first"),
                )
                label = (
                    "motif bridge-first μ3/μ2 graph enumeration, then terminal fill"
                    if decoration_mode == "motif_bridge_first"
                    else "graph-only μ3/μ2/terminal enumeration with ring-CN pruning"
                )
            elif decoration_mode == "tet_sites":
                from .molecular_tet_sites import iter_cl_attachments_tet_sites

                decorations = iter_cl_attachments_tet_sites(
                    k,
                    p,
                    skel,
                    check_spec,
                    pack,
                    max_assignments=max_decoration_assignments,
                    status=decoration_status,
                    degree_vectors=survey.pending_degrees,
                    slice_builder=lazy_builder,
                    frame_options=frame_options,
                    state=state_skel,
                    cation_ids=cation_ids,
                )
                label = (
                    "tet slots (topology) then pack embed "
                    "(linear/trigonal/bridge tables may shift Cl)"
                )
            else:
                from .molecular_sites import iter_cl_attachments_pack_sites

                decorations = iter_cl_attachments_pack_sites(
                    k,
                    p,
                    skel,
                    check_spec,
                    pack,
                    max_assignments=max_decoration_assignments,
                    status=decoration_status,
                    degree_vectors=survey.pending_degrees,
                    slice_builder=lazy_builder,
                    frame_options=frame_options,
                    state=state_skel,
                    cation_ids=cation_ids,
                )
                label = "pack virtual sites; bridges rebuild acceptor slots"
            if progress is not None:
                n_vec = (
                    len(survey.pending_degrees)
                    if survey.pending_degrees is not None
                    else 0
                )
                progress(
                    f"      {decoration_mode} "
                    f"({n_vec} CN vectors listed; {label})..."
                )
        else:
            decorations = iter_cl_attachments(
                k,
                p,
                skel,
                check_spec,
                max_assignments=max_decoration_assignments,
                status=decoration_status,
                mode_support=survey.mode_support,
                degree_slices=survey.degree_slices,
                degree_vectors=survey.pending_degrees,
                slice_builder=lazy_builder,
            )
            if progress is not None:
                if decoration_mode == "motif_graph":
                    progress(
                        "      complete anion-motif multisets "
                        "(Se-Cd skeleton motifs + typed Cl-Cd1/2/3 motifs; "
                        "global closure reconstruction after graph dedup)..."
                    )
                elif survey.pending_degrees is not None:
                    progress(
                        "      degree-first + geometry modes "
                        f"({len(survey.pending_degrees)} CN vectors listed; "
                        "orbit/rule filter then typed μ3/μ2/terminal)..."
                    )
                else:
                    progress(
                        "      streaming symmetry-reduced chloride assignments..."
                    )
        # One inorganic frame per distinct set of final cation coordination
        # numbers, already built by the survey above.
        decoration_iterator = iter(decorations)
        while True:
            next_started = time.perf_counter()
            try:
                dec = next(decoration_iterator)
            except StopIteration:
                decoration_generation_time += time.perf_counter() - next_started
                break
            decoration_generation_time += time.perf_counter() - next_started
            bin_result.raw_graphs += 1
            if progress is not None and bin_result.raw_graphs % 5000 == 0:
                progress(
                    f"      processed graphs={bin_result.raw_graphs} | "
                    f"accepted={len(seen)} | rejected={bin_result.rejected} "
                    f"(screened before embedding="
                    f"{bin_result.screened_before_embed}, "
                    f"embedded={bin_result.embedded})"
                )
            graph = skeleton_graph.copy()
            graph.add_edges_from(dec)
            screen.frame_cache = survey.frames
            screen_started = time.perf_counter()
            screen.offer(_State(atoms=atoms, graph=graph))
            candidate_screen_time += time.perf_counter() - screen_started

        bin_result.symmetry_pruned += decoration_status.symmetry_pruned
        bin_result.revisited += decoration_status.revisited
        bin_result.infeasible_partials += decoration_status.infeasible
        bin_result.over_capacity += decoration_status.over_capacity
        bin_result.geometry_pruned += decoration_status.geometry_pruned
        bin_result.modes_total += decoration_status.modes_total
        bin_result.modes_kept += decoration_status.modes_kept
        bin_result.frames_built += survey.frames_built
        # Skeleton with listed CN vectors but no usable frame for any
        # orbit-minimum vector that the decorator tried: treat as frame-dead
        # only when nothing was emitted and every built slice failed.
        if (
            embed
            and pack is not None
            and survey.pending_degrees is not None
            and decoration_status.degree_vectors_used == 0
            and survey.frames_built > 0
            and bin_result.raw_graphs == 0
        ):
            # All orbit-min vectors that were probed failed frames or modes.
            pass
        if progress is not None:
            progress(
                f"      assignments streamed={bin_result.raw_graphs} | "
                f"pruned in generation: symmetry="
                f"{decoration_status.symmetry_pruned}, "
                f"revisited={decoration_status.revisited}, "
                f"coordination="
                f"{decoration_status.infeasible + decoration_status.over_capacity}, "
                f"geometry={decoration_status.geometry_pruned}"
                f" | frames_built={survey.frames_built}"
                f" | CN used={decoration_status.degree_vectors_used}/"
                f"{decoration_status.degree_vectors_total}"
                f" | |Aut|={decoration_status.automorphisms}"
                + (" (TRUNCATED)" if decoration_status.truncated else "")
            )
        if decoration_status.truncated:
            bin_result.incomplete = True
            if not allow_incomplete:
                raise EnumerationLimitError(
                    f"k={k} p={p}: skeleton {skeleton_index}/"
                    f"{len(skeletons)} reached "
                    "the decoration-assignment safety guard "
                    f"({max_decoration_assignments}); rerun with a larger "
                    "--max-decoration-assignments or 0 for unlimited"
                )

        bin_result.isomers = list(seen.values())
        if checkpoint is not None:
            checkpoint(bin_result, skeleton_index, len(skeletons))

    known_graphs: Optional[Dict[Tuple[object, ...], str]] = None
    if screen.collect_only:
        # Second pass: every candidate has cleared the graph-level screen and
        # is waiting in the pool.  Embed a density-balanced slice of it until
        # the bin has ``target_isomers`` accepted, instead of embedding the
        # whole pool in decoration order.  3D costs ~0.55 s per candidate and
        # is flat in k, so this is what bounds a sweep -- the pool itself is
        # cheap to build.
        pool = screen.pool
        screen.collect_only = False
        # A rank cut on graph compactness.  Wiener index (sum of shortest-path
        # lengths, in bonds) is computed on the finished graph before any
        # coordinates exist and tracks relative energy at rho +0.32 to +0.78
        # across bins; keeping the most compact 70% retains ~89% of each bin's
        # best decile.  It is applied as a *rank* because the raw score shifts
        # between bins -- an absolute cut empties some bins and leaves others
        # untouched.
        order_mode = str(
            getattr(spec.graph_rules, "selection_order", "bond_bands")
        ).strip().lower()
        fraction = float(
            getattr(spec.graph_rules, "selection_top_fraction", 0.0) or 0.0
        )
        excess_cap = float(
            getattr(spec.graph_rules, "selection_max_wiener_excess", 0.0) or 0.0
        )
        if order_mode == "compactness" and pool:
            def _compactness(entry):
                graph = entry[1].graph
                if graph.number_of_nodes() < 2 or not nx.is_connected(graph):
                    return float("inf")
                return float(nx.wiener_index(graph))

            scores = [_compactness(entry) for entry in pool]
            finite = [v for v in scores if v < float("inf")]
            # A relative cut against the bin's own most compact graph.  Unlike
            # a fixed rank it adapts to the spread: where the Wiener range is
            # narrow the descriptor carries no information and almost nothing
            # is dropped, and where it is wide the cut bites.
            if excess_cap > 0.0 and finite:
                best = min(finite)
                allowed = sum(
                    1 for v in scores if v <= best * (1.0 + excess_cap)
                )
                target_isomers = (
                    allowed if not target_isomers
                    else min(target_isomers, allowed)
                )
            if fraction > 0.0:
                budget = max(1, -(-len(pool) * 1000 // 1000))
                budget = max(1, int(len(pool) * min(fraction, 1.0) + 0.999))
                target_isomers = (
                    budget if not target_isomers else min(target_isomers, budget)
                )
            ranked = sorted(range(len(pool)), key=lambda i: scores[i])
            if progress is not None:
                progress(
                    f"    selection: compactness rank cut, pool={len(pool)} "
                    f"→ budget={target_isomers}"
                )
            for index in ranked:
                if len(seen) >= target_isomers:
                    break
                screen.offer(pool[index][1])
            bands = {}
            keys = []
        else:
            bands = bond_count_bands(pool)
            keys = sorted(bands)
        if progress is not None:
            progress(
                f"    budget: pool={len(pool)} graphs in {len(keys)} bond-count "
                f"band(s) {keys} → embedding for {target_isomers} accepted"
            )
        embedded_here = 0
        if keys:
            # Quota is on *accepted* per band, not on embeddings: bands differ
            # in accept rate by an order of magnitude (29.6% to 0.9% across
            # k=6,p=3), so an embedding quota just hands the budget to whichever
            # band converts best -- which is the sparse, elongated end.  Capping
            # what each band may *contribute* is what actually balances the set.
            quota = max(1, -(-target_isomers // len(keys)))
            cursor = {key: 0 for key in keys}
            won = {key: 0 for key in keys}
            while len(seen) < target_isomers:
                progressed = False
                for key in keys:
                    if len(seen) >= target_isomers:
                        break
                    if won[key] >= quota or cursor[key] >= len(bands[key]):
                        continue
                    before = len(seen)
                    screen.offer(pool[bands[key][cursor[key]]][1])
                    cursor[key] += 1
                    embedded_here += 1
                    if len(seen) > before:
                        won[key] += 1
                    progressed = True
                if progressed:
                    continue
                # Every band is either exhausted or has filled its share while
                # the bin is still short.  Release the caps a notch so the
                # bands that still have candidates can cover for the ones that
                # ran dry, rather than stopping under target.
                if all(cursor[key] >= len(bands[key]) for key in keys):
                    break
                quota += 1
        # Certificates of every graph-legal candidate this bin produced, so a
        # structure that relaxes into a *different* topology can be recognised
        # as one we already enumerated -- possibly one the construction audit
        # threw out -- rather than looking like a novel species.
        known_graphs = {}
        for _bond_total, cand in pool:
            known_graphs.setdefault(_graph_certificate(cand), "pool")
        for iso_id, iso_obj in seen.items():
            known_graphs[iso_id] = iso_obj.structure_id
        bin_result.isomers = list(seen.values())
        bin_result.budget_pool = len(pool)
        bin_result.budget_embedded = embedded_here
        if progress is not None:
            progress(
                f"    budget: embedded {embedded_here}/{len(pool)} → "
                f"accepted {len(seen)}"
            )
        if checkpoint is not None:
            checkpoint(bin_result, len(skeletons), len(skeletons))

    bin_result.decoration_stream_time_s = time.perf_counter() - decoration_started
    bin_result.decoration_generation_time_s = decoration_generation_time
    bin_result.candidate_screen_time_s = candidate_screen_time
    bin_result.isomers = list(seen.values())
    _relax_bin_with_xtb(bin_result, pack, check_spec, known_graphs, progress)
    _log_xtb_energy_ranking(bin_result, progress)
    if resolved_mode == "precomputed" or precomputed_skeletons is not None:
        bin_result.skeleton_mode_used = "precomputed"
        bin_result.proved_level = 0
    else:
        bin_result.skeleton_mode_used = resolved_mode
        lvl = mode_to_level.get(resolved_mode, 0)
        if len(bin_result.isomers) >= 1 and lvl >= 1:
            bin_result.ring_first_proved = True
            bin_result.proved_level = lvl
        elif len(bin_result.isomers) >= 1:
            bin_result.proved_level = 0
        else:
            bin_result.proved_level = 0

    # Adaptive fallback down the structure ladder (fused2 → 1-ring → free).
    if (
        remaining_fallback_modes
        and precomputed_skeletons is None
        and max_decoration_assignments >= 0
        and len(bin_result.isomers) == 0
        and _ring_fallback_depth < 4
    ):
        next_mode = remaining_fallback_modes[0]
        if progress is not None:
            progress(
                f"    {resolved_mode} → 0 accepted → fallback {next_mode}"
            )
        queue = list(remaining_fallback_modes)
        fb = enumerate_molecular_bin(
            k,
            p,
            spec,
            pack=pack,
            embed=embed,
            max_skeletons=max_skeletons,
            max_decoration_assignments=max_decoration_assignments,
            extra_skeleton_edges=extra_skeleton_edges,
            allow_incomplete=allow_incomplete,
            validate_every_graph=validate_every_graph,
            frame_options=frame_options,
            dump_failures=dump_failures,
            target_isomers=target_isomers,
            progress=progress,
            checkpoint=checkpoint,
            graph_checkpoint=graph_checkpoint,
            precomputed_skeletons=None,
            skeleton_mode=queue[0],
            allow_ring_fallback=True,
            min_structure_level=min_structure_level,
            _ring_fallback_depth=_ring_fallback_depth + 1,
            _structure_mode_queue=queue,
        )
        if fb.skeleton_mode_used == "free" and resolved_mode != "free":
            fb.skeleton_mode_used = "free_fallback"
        return fb
    return bin_result


def generate_molecular_map(
    spec: NucleationSpec,
    *,
    geometry_pack: str | Path | GeometryPack | None = None,
    kmin: int = 1,
    kmax: Optional[int] = None,
    pmin: int = 0,
    pmax: Optional[int] = None,
    embed: bool = True,
    allow_incomplete: bool = False,
    progress: Optional[ProgressCallback] = None,
    max_skeletons: int = 2000,
    max_decoration_assignments: int = 0,
    extra_skeleton_edges: Optional[int] = None,
    frame_options: int = 0,
    dump_failures: bool = False,
    target_isomers: int = 0,
    incremental_output: str | Path | None = None,
    skeleton_catalog: Optional[
        Dict[Tuple[int, int], Sequence[Tuple[Tuple[int, int], ...]]]
    ] = None,
    skeleton_catalog_dir: str | Path | None = None,
    require_catalog_skeletons: bool = False,
) -> MolecularMapResult:
    """Generate molecular isomers for k=1..kmax and feasible p.

    When ``incremental_output`` is set, the accumulated map is rewritten
    after every completed bin so long sweeps remain inspectable and retain
    all bins completed before a later interruption or enumeration limit.

    ``skeleton_catalog`` / ``skeleton_catalog_dir`` supply precomputed Cd–Se
    edge sets (from a prior dump).  For those ``(k, p)`` bins decoration runs
    without re-enumerating skeletons.  If ``require_catalog_skeletons`` is
    true, bins missing from the catalog are skipped instead of re-enumerated.
    """

    pack: Optional[GeometryPack] = None
    if isinstance(geometry_pack, GeometryPack):
        pack = geometry_pack
    elif geometry_pack is not None:
        pack = load_geometry_pack(geometry_pack)
    elif spec.geometry_pack is not None:
        pack = load_geometry_pack(spec.geometry_pack)
    elif embed:
        # Default pack next to repo if present
        default = (
            Path(__file__).resolve().parents[3]
            / "geometry_packs"
            / "cdse_cdcl2_molecular.yaml"
        )
        if default.is_file():
            pack = load_geometry_pack(default)

    k_lo = int(kmin)
    k_hi = int(kmax if kmax is not None else spec.kmax)
    p_lo = int(pmin)
    if k_lo < 1 or k_hi < k_lo:
        raise ValueError("require 1 <= kmin <= kmax")
    if p_lo < 0:
        raise ValueError("pmin must be >= 0")
    result = MolecularMapResult(
        geometry_pack_name=pack.name if pack else ""
    )

    catalog: Dict[Tuple[int, int], Sequence[Tuple[Tuple[int, int], ...]]] = {}
    if skeleton_catalog is not None:
        catalog.update(
            {
                key: list(val)
                for key, val in skeleton_catalog.items()
            }
        )
    if skeleton_catalog_dir is not None:
        loaded = load_skeleton_catalog(
            skeleton_catalog_dir, accepted_only=True, require_edges=True
        )
        for key, val in loaded.items():
            catalog.setdefault(key, val)
        if progress is not None:
            n_bin = len(loaded)
            n_sk = sum(len(v) for v in loaded.values())
            progress(
                f"[molecular] loaded skeleton catalog from "
                f"{skeleton_catalog_dir}: {n_sk} accepted skeleton(s) "
                f"in {n_bin} (k,p) bin(s)"
            )
            if n_sk == 0:
                progress(
                    "[molecular] WARNING: catalog has no edge lists "
                    "(old dump without edges column). Re-enumeration will "
                    "be used unless --require-saved-skeletons is set."
                )

    # Adaptive structure ladder: proved_level[k] = 0 free, 1 one-ring, 2 fused2.
    # Later p at same k never drop below proved level.
    proved_level: Dict[int, int] = {}

    for k in range(k_lo, k_hi + 1):
        max_p, slot_info = resolve_molecular_max_p(
            spec,
            k,
            pmax,
            max_skeletons=max_skeletons,
            extra_skeleton_edges=extra_skeleton_edges,
        )
        # If catalog constrains this k, also cover every p present in it
        # (even when auto pmax is lower / higher).
        catalog_ps = [pp for (kk, pp) in catalog if kk == k]
        if catalog_ps and pmax is None:
            max_p = max(int(max_p), max(catalog_ps))
        if progress is not None and slot_info is not None:
            if slot_info.source == "slots":
                progress(
                    f"[molecular] k={k} pmax={max_p} from accepted p=0 Se slots "
                    f"(max free={slot_info.max_free_slots}, "
                    f"accepted={slot_info.n_p0_accepted}/"
                    f"{slot_info.n_p0_enumerated}, "
                    f"global={slot_info.global_bound})"
                )
            else:
                progress(
                    f"[molecular] k={k} pmax={max_p} global Se bound "
                    f"(no accepted p=0; enumerated "
                    f"{slot_info.n_p0_enumerated})"
                )
        for p in range(p_lo, int(max_p) + 1):
            blocks, expanded = molecular_stoichiometry_label(spec, k, p)
            # Skip impossible under enforce_min_cn-like constraints early
            if k == 1 and p == 0:
                # CdSe monomer cannot satisfy min_cn 2 for both
                if progress is not None:
                    progress(
                        f"[molecular] SKIP k={k} p={p} | {blocks} | "
                        f"total={expanded} | cannot satisfy hard minimum CN"
                    )
                continue
            precomputed = catalog.get((k, p))
            if (
                require_catalog_skeletons
                and skeleton_catalog is None
                and skeleton_catalog_dir is None
            ):
                pass  # nothing to require
            if (
                require_catalog_skeletons
                and (k, p) not in catalog
                and max_structure_level_possible(k, p, spec) == 0
            ):
                if progress is not None:
                    progress(
                        f"[molecular] SKIP k={k} p={p} | {blocks} | "
                        f"not in saved skeleton catalog"
                    )
                continue

            min_L = int(proved_level.get(k, 0))
            if precomputed is not None:
                skel_mode: Optional[str] = "precomputed"
                allow_fb = False
                min_L = 0
            else:
                skel_mode = "auto"
                allow_fb = True

            if progress is not None:
                src = (
                    f" | saved skeletons={len(precomputed)}"
                    if precomputed is not None
                    else (
                        f" | structure ladder "
                        f"(min_level={min_L}, "
                        f"max_possible="
                        f"{max_structure_level_possible(k, p, spec)})"
                    )
                )
                progress(
                    f"[molecular] START k={k} p={p} | {blocks} | "
                    f"total={expanded}{src}"
                )

            def save_skeleton_checkpoint(
                partial: MolecularBinResult,
                skeleton_index: int,
                skeleton_count: int,
            ) -> None:
                if incremental_output is None:
                    return
                result.bins[(k, p)] = partial
                saved_root = write_molecular_map(
                    result,
                    incremental_output,
                    only_bin=(k, p),
                    dump_failures=dump_failures,
                )
                if progress is not None:
                    progress(
                        f"[molecular] CHECKPOINT k={k} p={p} | "
                        f"skeleton={skeleton_index}/{skeleton_count} | "
                        f"accepted={len(partial.isomers)} | {saved_root}"
                    )

            def save_graph_checkpoint(partial: MolecularBinResult) -> None:
                if incremental_output is None:
                    return
                result.bins[(k, p)] = partial
                saved_root = write_molecular_map(
                    result,
                    incremental_output,
                    only_bin=(k, p),
                    dump_failures=dump_failures,
                )
                if progress is not None:
                    progress(
                        f"[molecular] GRAPH CHECKPOINT k={k} p={p} | "
                        f"graphs={partial.motif_graphs_eligible} | {saved_root}"
                    )

            bin_res = enumerate_molecular_bin(
                k,
                p,
                spec,
                pack=pack,
                embed=embed and pack is not None,
                allow_incomplete=allow_incomplete,
                progress=progress,
                max_skeletons=max_skeletons,
                max_decoration_assignments=max_decoration_assignments,
                extra_skeleton_edges=extra_skeleton_edges,
                frame_options=frame_options,
                dump_failures=dump_failures,
                target_isomers=target_isomers,
                checkpoint=save_skeleton_checkpoint,
                graph_checkpoint=save_graph_checkpoint,
                precomputed_skeletons=precomputed,
                skeleton_mode=(
                    None if skel_mode == "precomputed" else skel_mode
                ),
                allow_ring_fallback=allow_fb,
                min_structure_level=min_L,
            )
            if bin_res.proved_level > proved_level.get(k, 0):
                if progress is not None:
                    progress(
                        f"[molecular] proved_level k={k} → "
                        f"{bin_res.proved_level} "
                        f"(mode={bin_res.skeleton_mode_used} at p={p}; "
                        f"later p stay ≥ this level)"
                    )
                proved_level[k] = bin_res.proved_level
            elif bin_res.ring_first_proved and proved_level.get(k, 0) < 1:
                proved_level[k] = 1
            result.bins[(k, p)] = bin_res
            if incremental_output is not None:
                saved_root = write_molecular_map(
                    result, incremental_output, dump_failures=dump_failures
                )
                if progress is not None:
                    progress(
                        f"[molecular] SAVED k={k} p={p} | {saved_root}"
                    )
            if progress is not None:
                progress(
                    f"[molecular] PRUNED k={k} p={p} | "
                    + bin_res.prefilter_summary()
                )
                progress(
                    f"[molecular] DONE  k={k} p={p} | "
                    f"accepted={len(bin_res.isomers)}, "
                    f"raw_graphs={bin_res.raw_graphs}, "
                    f"screened_before_embed={bin_res.screened_before_embed}, "
                    f"embedded={bin_res.embedded}, "
                    f"rejected={bin_res.rejected}, "
                    f"audited={bin_res.motif_pre_xtb_accepted}, "
                    f"xtb_submitted={bin_res.motif_xtb_attempts}, "
                    f"xtb_converged={bin_res.motif_xtb_converged}, "
                    f"skeleton_time_s={bin_res.skeleton_generation_time_s:.3f}, "
                    f"graph_generation_time_s={bin_res.decoration_generation_time_s:.3f}, "
                    f"candidate_screen_time_s={bin_res.candidate_screen_time_s:.3f}, "
                    f"decoration_stream_time_s={bin_res.decoration_stream_time_s:.3f}, "
                    f"incomplete={str(bin_res.incomplete).lower()}"
                )
    return result


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _best_unit_direction(
    matrix: FloatArray, targets: FloatArray
) -> Optional[FloatArray]:
    """``argmin ||M d - t||`` over unit vectors ``d``.

    The unconstrained least-squares solution normalised is not this vector, and
    the two diverge exactly when the targets are unreachable -- which is when
    it matters.  Written as a Lagrangian, the constrained minimiser solves
    ``(M'M + lam I) d = M't`` for whichever ``lam`` makes ``|d| = 1``.  Since
    ``|d(lam)|`` decreases monotonically once ``lam`` exceeds the smallest
    eigenvalue of ``M'M``, bisection finds it robustly; the system is 3x3, so
    this costs one eigendecomposition and a handful of divisions.
    """

    normal_matrix = matrix.T @ matrix
    projected = matrix.T @ targets
    eigenvalues, basis = np.linalg.eigh(normal_matrix)
    rotated = basis.T @ projected

    def squared_norm(shift: float) -> float:
        denominator = eigenvalues + shift
        denominator = np.where(np.abs(denominator) < 1.0e-14, 1.0e-14, denominator)
        return float(np.sum((rotated / denominator) ** 2))

    low = max(0.0, -float(eigenvalues.min())) + 1.0e-9
    high = low + 1.0
    for _ in range(200):
        if squared_norm(high) <= 1.0:
            break
        high *= 2.0
    else:
        return None
    for _ in range(100):
        middle = 0.5 * (low + high)
        if squared_norm(middle) > 1.0:
            low = middle
        else:
            high = middle
    shift = 0.5 * (low + high)
    direction = basis @ (rotated / (eigenvalues + shift))
    return _unit(direction)


def _cross3(a: FloatArray, b: FloatArray) -> FloatArray:
    """Cross product of two 3-vectors.

    ``np.cross`` is generic over trailing axes and spends most of its time in
    ``moveaxis``/``normalize_axis_tuple``; for the fixed 3-vectors here that
    dispatch dwarfs the six multiplications.
    """

    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    bx, by, bz = float(b[0]), float(b[1]), float(b[2])
    return np.array(
        [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx]
    )


def _unit(v: FloatArray) -> Optional[FloatArray]:
    # ``np.linalg.norm`` costs microseconds of dispatch on a 3-vector and this
    # runs hundreds of thousands of times per bin; the arithmetic is the same.
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    n = sqrt(x * x + y * y + z * z)
    if n < 1.0e-12:
        return None
    return v / n


def _dihedral_deg_points(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
    d: Sequence[float],
) -> float:
    """Return the signed four-point dihedral in degrees."""

    p0, p1, p2, p3 = (np.asarray(point, dtype=float) for point in (a, b, c, d))
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    norm = float(np.linalg.norm(b1))
    if norm < 1.0e-12:
        return 0.0
    b1 /= norm
    v = b0 - float(np.dot(b0, b1)) * b1
    w = b2 - float(np.dot(b2, b1)) * b1
    return float(
        np.degrees(
            np.arctan2(
                float(np.dot(_cross3(b1, v), w)),
                float(np.dot(v, w)),
            )
        )
    )


def _dihedral_delta_deg(value: float, target: float) -> float:
    """Shortest signed angular difference in degrees."""

    return (float(value) - float(target) + 180.0) % 360.0 - 180.0


def _candidate_soft_clearance_penalty(
    state: _State,
    pack: GeometryPack,
    atom: int,
    position: Sequence[float],
    coords: FloatArray,
    placed: Sequence[bool],
) -> float:
    """Score configured soft clearances against already placed atoms."""

    penalty = 0.0
    pos = np.asarray(position, dtype=float)
    for other, is_placed in enumerate(placed):
        if not is_placed or other == atom:
            continue
        edge = (min(int(atom), int(other)), max(int(atom), int(other)))
        if state.graph.has_edge(*edge):
            continue
        pair = pair_key(state.atoms[atom].symbol, state.atoms[other].symbol)
        penalty += pack.soft_contact_penalty(
            pair, float(np.linalg.norm(pos - coords[other]))
        )
    return penalty


def _candidate_torsion_penalty(
    state: _State,
    pack: GeometryPack,
    atom: int,
    position: Sequence[float],
    coords: FloatArray,
    placed: Sequence[bool],
    ring_edges: Set[Tuple[int, int]],
) -> float:
    """Return normalized preferred-torsion error for a candidate atom."""

    trial = np.asarray(coords, dtype=float).copy()
    trial[atom] = np.asarray(position, dtype=float)
    errors: List[float] = []
    for center in state.graph.neighbors(atom):
        if not placed[center]:
            continue
        for left in state.graph.neighbors(center):
            if left == atom or not placed[left]:
                continue
            for outer in state.graph.neighbors(left):
                if outer in {atom, center} or not placed[outer]:
                    continue
                path = tuple(
                    state.atoms[index].symbol
                    for index in (outer, left, center, atom)
                )
                if pack.dihedral_excluded(
                    path,
                    endocyclic=all(
                        (min(a, b), max(a, b)) in ring_edges
                        for a, b in ((outer, left), (left, center), (center, atom))
                    ),
                ):
                    continue
                preferred = pack.preferred_dihedral(path)
                if preferred is None:
                    continue
                target, tolerance = preferred
                value = _dihedral_deg_points(
                    trial[outer], trial[left], trial[center], trial[atom]
                )
                errors.append(abs(_dihedral_delta_deg(value, target)) / max(tolerance, 1.0e-6))
    return sum(errors)


def _regular_directions(n: int) -> List[FloatArray]:
    """Deterministic unit directions for placing n neighbors."""

    if n <= 0:
        return []
    if n == 1:
        return [np.array([1.0, 0.0, 0.0])]
    if n == 2:
        return [
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
        ]
    if n == 3:
        # 120° in xy plane
        return [
            np.array([1.0, 0.0, 0.0]),
            np.array([-0.5, np.sqrt(3) / 2, 0.0]),
            np.array([-0.5, -np.sqrt(3) / 2, 0.0]),
        ]
    if n == 4:
        # tetrahedron
        return [
            _unit(np.array([1.0, 1.0, 1.0])),
            _unit(np.array([1.0, -1.0, -1.0])),
            _unit(np.array([-1.0, 1.0, -1.0])),
            _unit(np.array([-1.0, -1.0, 1.0])),
        ]
    if n == 5:
        # Trigonal bipyramid: minimum pairwise angle 90 degrees.  The spiral
        # below would instead land points on *both* poles and put its polar
        # neighbours only 60 degrees apart, which for a CN-5 anion drives the
        # two cations to 2.78 A -- inside the forbidden Cd-Cd floor -- and made
        # every CN-5 skeleton unbuildable.
        return [
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([-0.5, np.sqrt(3) / 2, 0.0]),
            np.array([-0.5, -np.sqrt(3) / 2, 0.0]),
        ]
    if n == 6:
        # Octahedron: minimum pairwise angle 90 degrees.
        return [
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
        ]
    # Higher CN: Fibonacci sphere.  Sampling ``y`` at half-step offsets keeps
    # the points off the poles, so no two are closer than the spacing the
    # spiral is meant to deliver.
    dirs = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1.0 - 2.0 * (i + 0.5) / n
        radius = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        dirs.append(np.array([np.cos(theta) * radius, y, np.sin(theta) * radius]))
    return [d / np.linalg.norm(d) for d in dirs]


def _angle_directions(n: int, angle_deg: float) -> List[FloatArray]:
    """Directions with successive bond angle ``angle_deg`` for small n."""

    if n <= 2:
        if n == 1:
            return [np.array([1.0, 0.0, 0.0])]
        # place with given angle (for bent Se CN2)
        half = np.radians(angle_deg / 2.0)
        return [
            np.array([np.sin(half), np.cos(half), 0.0]),
            np.array([-np.sin(half), np.cos(half), 0.0]),
        ]
    if abs(angle_deg - 180.0) < 5.0 and n == 2:
        return [np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])]
    # For ~90° multi-neighbor: prefer orthogonal axes
    if 80.0 <= angle_deg <= 100.0 and n <= 3:
        axes = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        return axes[:n]
    return _regular_directions(n)


def _orthonormal_basis(axis: FloatArray) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """Return (axis_hat, u, v) forming a right-handed orthonormal frame."""

    a = _unit(axis)
    if a is None:
        a = np.array([1.0, 0.0, 0.0])
    # Pick a helper not parallel to a
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(a, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = _unit(_cross3(a, helper))
    assert u is not None
    v = _unit(_cross3(a, u))
    assert v is not None
    return a, u, v


def _three_sphere_intersections(
    centers: Sequence[FloatArray], radii: Sequence[float]
) -> Tuple[FloatArray, FloatArray]:
    """Return the two exact intersections of three non-collinear spheres."""

    p1, p2, p3 = (np.asarray(point, dtype=float) for point in centers)
    r1, r2, r3 = (float(radius) for radius in radii)
    ex = _unit(p2 - p1)
    if ex is None:
        raise ExactEmbeddingError(["mu3_hosts_coincident"])
    distance = float(np.linalg.norm(p2 - p1))
    p13 = p3 - p1
    projection = float(np.dot(ex, p13))
    ey = _unit(p13 - projection * ex)
    if ey is None:
        raise ExactEmbeddingError(["mu3_hosts_collinear"])
    transverse = float(np.dot(ey, p13))
    ez = _unit(_cross3(ex, ey))
    assert ez is not None
    x = (r1 * r1 - r2 * r2 + distance * distance) / (2.0 * distance)
    y = (
        r1 * r1
        - r3 * r3
        + projection * projection
        + transverse * transverse
        - 2.0 * projection * x
    ) / (2.0 * transverse)
    z_squared = r1 * r1 - x * x - y * y
    if z_squared < -(EXACT_BOND_TOLERANCE**2):
        raise ExactEmbeddingError(["mu3_spheres_do_not_intersect"])
    base = p1 + x * ex + y * ey
    offset = np.sqrt(max(0.0, z_squared)) * ez
    return base + offset, base - offset


def _relaxed_sphere_position(
    centers: Sequence[FloatArray],
    radii: Sequence[float],
    preferred: Optional[FloatArray] = None,
    *,
    tolerance: float = RELAXED_BRIDGE_BOND_TOLERANCE_A,
) -> FloatArray:
    """Find a deterministic near-intersection for a bridge motif.

    Exact sphere intersections are attempted first by the caller.  This
    fallback is only for ligand placement when DFT-median bridge radii and a
    fixed inorganic host frame miss by a small amount.  It solves the local
    distance equations, never chooses a torsion from a clearance score, and
    fails if the residual is too large.
    """

    points = [np.asarray(point, dtype=float) for point in centers]
    target = np.asarray([float(value) for value in radii], dtype=float)
    if not points or len(points) != len(target):
        raise ExactEmbeddingError(["invalid_bridge_sphere_constraints"])
    centroid = np.mean(np.vstack(points), axis=0)
    starts: List[FloatArray] = [
        np.asarray(preferred, dtype=float)
        if preferred is not None
        else centroid.copy(),
        centroid.copy(),
    ]
    if len(points) >= 2:
        axis = _unit(points[1] - points[0])
        if axis is not None:
            _axis, u, _v = _orthonormal_basis(axis)
            starts.extend(
                centroid + scale * u for scale in (1.0, -1.0)
            )

    def residual(position: FloatArray) -> FloatArray:
        return np.asarray(
            [float(np.linalg.norm(position - point)) - value
             for point, value in zip(points, target)],
            dtype=float,
        )

    best: Optional[Tuple[float, float, FloatArray]] = None
    for start in starts:
        result = least_squares(
            residual,
            np.asarray(start, dtype=float),
            max_nfev=20,
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
        )
        point = np.asarray(result.x, dtype=float)
        errors = np.abs(residual(point))
        score = (
            float(np.max(errors)),
            float(np.sum(errors)),
            point,
        )
        if best is None or score[:2] < best[:2]:
            best = score
    assert best is not None
    if best[0] > float(tolerance):
        raise ExactEmbeddingError(
            [f"bridge_spheres_incompatible:{best[0]:.4f}>{float(tolerance):.4f}"]
        )
    return np.asarray(best[2], dtype=float)


def _rotate_dirs_to_frame(
    dirs: Sequence[FloatArray], axis: FloatArray
) -> List[FloatArray]:
    """Rotate template dirs so dirs[0] maps onto ``axis`` (if possible)."""

    if not dirs:
        return []
    target = _unit(axis)
    source = _unit(np.asarray(dirs[0], dtype=float))
    if target is None or source is None:
        return [np.asarray(d, dtype=float) for d in dirs]
    # Rodrigues rotation source -> target
    v = _cross3(source, target)
    c = float(np.dot(source, target))
    if c < -0.999:
        # 180°: pick any perpendicular
        _, u, _ = _orthonormal_basis(source)
        rot = -np.eye(3) + 2.0 * np.outer(u, u)
    elif np.linalg.norm(v) < 1e-12:
        rot = np.eye(3)
    else:
        vx = np.array(
            [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
            dtype=float,
        )
        rot = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    out = []
    for d in dirs:
        rd = rot @ np.asarray(d, dtype=float)
        un = _unit(rd)
        out.append(un if un is not None else np.asarray(d, dtype=float))
    return out


def _align_template_to_placed(
    template: Sequence[FloatArray],
    placed_directions: Sequence[FloatArray],
) -> List[FloatArray]:
    """Orient a template so it best matches *every* already-placed neighbour.

    A BFS over a tree reaches each centre from one parent, so one reference
    direction determines the frame.  A cycle reaches a centre from two or more
    parents, and aligning on only the first silently ignores the rest -- which
    is how a bonded pair ended up 6.31 A apart against a 2.53 A target.

    Tries each way of matching the placed neighbours onto template slots,
    keeping the Kabsch-optimal rotation of the best match, and returns the
    unassigned slots for the neighbours still to place.
    """

    slots = [np.asarray(direction, dtype=float) for direction in template]
    fixed = [np.asarray(direction, dtype=float) for direction in placed_directions]
    best_error: Optional[float] = None
    best_free: List[FloatArray] = []
    for assignment in permutations(range(len(slots)), len(fixed)):
        matrix = np.zeros((3, 3), dtype=float)
        for slot, target in zip(assignment, fixed):
            matrix += np.outer(target, slots[slot])
        left, _singular, right = np.linalg.svd(matrix)
        correction = np.eye(3)
        # Reflections are not rotations; flip the least-significant axis.
        correction[2, 2] = np.sign(np.linalg.det(left @ right))
        rotation = left @ correction @ right
        error = sum(
            float(np.sum((rotation @ slots[slot] - target) ** 2))
            for slot, target in zip(assignment, fixed)
        )
        if best_error is None or error < best_error - 1.0e-12:
            best_error = error
            best_free = [
                rotation @ slots[slot]
                for slot in range(len(slots))
                if slot not in assignment
            ]
    return best_free


def _molecular_bond_length(
    state: _State,
    pack: GeometryPack,
    spec: NucleationSpec,
    left: int,
    right: int,
    degrees: Optional[Sequence[int]] = None,
) -> float:
    # Callers in a loop pass ``degrees`` in: rebuilding it here walked the
    # whole graph for every single bond length.
    if degrees is None:
        degrees = [
            state.graph.degree[index] for index in range(len(state.atoms))
        ]
    cations = {spec.core.cation, spec.precursor.center}
    anion = spec.core.anion
    left_symbol = state.atoms[left].symbol
    right_symbol = state.atoms[right].symbol
    if anion in (left_symbol, right_symbol) and (
        left_symbol in cations or right_symbol in cations
    ):
        cation = left if left_symbol in cations else right
        anion_id = right if cation == left else left
        return pack.bond_length(
            "CdSe", degrees[cation], degrees[anion_id], default=2.60
        )
    cation = left if left_symbol in cations else right
    ligand = right if cation == left else left
    ligand_cn = degrees[ligand]
    role = "CdCl_bridge" if ligand_cn >= 2 else "CdCl_terminal"
    return pack.bond_length(
        role,
        degrees[cation],
        ligand_cn,
        default=2.40 if ligand_cn >= 2 else 2.33,
    )


def _molecular_bond_tolerance(
    state: _State,
    pack: GeometryPack,
    spec: NucleationSpec,
    left: int,
    right: int,
    degrees: Optional[Sequence[int]] = None,
) -> float:
    """Band half-width for one bond -- the same lookup as its target length."""

    if degrees is None:
        degrees = [
            state.graph.degree[index] for index in range(len(state.atoms))
        ]
    cations = {spec.core.cation, spec.precursor.center}
    anion = spec.core.anion
    left_symbol = state.atoms[left].symbol
    right_symbol = state.atoms[right].symbol
    if anion in (left_symbol, right_symbol) and (
        left_symbol in cations or right_symbol in cations
    ):
        cation = left if left_symbol in cations else right
        anion_id = right if cation == left else left
        return pack.bond_tolerance_A("CdSe", degrees[cation], degrees[anion_id])
    cation = left if left_symbol in cations else right
    ligand = right if cation == left else left
    ligand_cn = degrees[ligand]
    role = "CdCl_bridge" if ligand_cn >= 2 else "CdCl_terminal"
    return pack.bond_tolerance_A(role, degrees[cation], ligand_cn)


def _neighbor_signature(state: _State, center: int) -> str:
    counts = Counter(
        state.atoms[neighbor].symbol
        for neighbor in state.graph.neighbors(center)
    )
    return "".join(
        f"{symbol}{counts[symbol]}" for symbol in sorted(counts)
    )


def _neighbor_geometry_role(state: _State, center: int, neighbor: int) -> str:
    """Local DFT geometry role for one neighbor of a center."""

    symbol = state.atoms[neighbor].symbol
    if symbol != "Cl":
        return symbol
    degree = state.graph.degree[neighbor]
    if degree <= 1:
        return "Cl_t"
    if degree == 2:
        other_hosts = [
            atom
            for atom in state.graph.neighbors(neighbor)
            if atom != center
            and state.atoms[atom].symbol == state.atoms[center].symbol
        ]
        shared = any(
            any(
                common != neighbor and state.atoms[common].symbol != symbol
                for common in nx.common_neighbors(state.graph, center, other_host)
            )
            for other_host in other_hosts
        )
        return "Cl_b2s" if shared else "Cl_b2n"
    return f"Cl_b{degree}"


def _role_environment(
    state: _State, center: int, left: int, right: int
) -> Tuple[str, str]:
    roles = [
        _neighbor_geometry_role(state, center, neighbor)
        for neighbor in state.graph.neighbors(center)
    ]
    pair_roles = [
        _neighbor_geometry_role(state, center, left),
        _neighbor_geometry_role(state, center, right),
    ]
    return "+".join(sorted(roles)), "-".join(sorted(pair_roles))


def _effective_center_angle_deg(
    state: _State,
    pack: GeometryPack,
    center: int,
    left: int,
    right: int,
    *,
    default: Optional[float] = None,
) -> Optional[float]:
    """Return an angle compatible with an explicitly planar CN3 template.

    DFT table entries are independent medians and need not sum to 360 degrees.
    For an A2B planar center, the repeated A-A angle is therefore kept exactly
    and the two equivalent A-B sectors share the remaining angle. This makes
    Cd(Cl2Se1) use the configured Cl-Cd-Cl angle rather than accidentally
    becoming linear.
    """

    center_symbol = state.atoms[center].symbol
    cn = state.graph.degree[center]
    signature = _neighbor_signature(state, center)
    pair = "-".join(
        sorted((state.atoms[left].symbol, state.atoms[right].symbol))
    )
    role_signature, role_pair = _role_environment(
        state, center, left, right
    )
    configured = pack.center_angle_deg(
        center_symbol,
        cn,
        neighbor_pair=pair,
        signature=signature,
        role_pair=role_pair,
        role_signature=role_signature,
        default=default,
    )
    if cn != 3 or pack.improper_angle_deg(center_symbol, cn, signature) != 0.0:
        return configured

    counts = Counter(
        state.atoms[neighbor].symbol
        for neighbor in state.graph.neighbors(center)
    )
    if len(counts) == 1:
        return configured
    repeated = next((symbol for symbol, count in counts.items() if count == 2), None)
    singleton = next((symbol for symbol, count in counts.items() if count == 1), None)
    if repeated is None or singleton is None:
        return configured
    repeated_pair = "-".join(sorted((repeated, repeated)))
    repeated_angle = pack.center_angle_deg(
        center_symbol,
        cn,
        neighbor_pair=repeated_pair,
        signature=signature,
        default=None,
    )
    if repeated_angle is None or not pack.center_angle_is_hard(
        center_symbol,
        cn,
        neighbor_pair=repeated_pair,
        signature=signature,
        role_pair=repeated_pair,
        role_signature=role_signature,
    ):
        return configured
    if state.atoms[left].symbol == repeated and state.atoms[right].symbol == repeated:
        return repeated_angle
    return (360.0 - repeated_angle) / 2.0


def _exact_bond_violations(
    state: _State,
    coordinates: Sequence[Sequence[float]],
    pack: GeometryPack,
    spec: NucleationSpec,
    target_overrides: Optional[Mapping[Tuple[int, int], float]] = None,
    relaxed: bool = False,
) -> List[str]:
    coords = np.asarray(coordinates, dtype=float)
    violations: List[str] = []
    degrees = [state.graph.degree[i] for i in range(len(state.atoms))]
    for left, right in state.graph.edges:
        edge = (min(int(left), int(right)), max(int(left), int(right)))
        target = (
            float(target_overrides[edge])
            if target_overrides is not None and edge in target_overrides
            else _molecular_bond_length(state, pack, spec, left, right, degrees)
        )
        distance = float(np.linalg.norm(coords[left] - coords[right]))
        error = abs(distance - target)
        ligand = spec.precursor.ligand
        bridge_tolerance = (
            RELAXED_BRIDGE_BOND_TOLERANCE_A
            if relaxed
            or (
                ligand in (state.atoms[left].symbol, state.atoms[right].symbol)
                and state.graph.degree[
                    left if state.atoms[left].symbol == ligand else right
                ] >= 2
            )
            else pack.audit_bond_tolerance_A
        )
        if error > bridge_tolerance:
            violations.append(
                f"bond_geometry:{left}-{right}:{distance:.6f}!={target:.6f}"
            )
    return violations


def _motif_clash_violations(
    state: _State,
    coordinates: Sequence[Sequence[float]],
    *,
    overlap_min_A: float = 0.75,
) -> List[str]:
    """Return only catastrophic Cartesian overlaps for motif starts.

    Motif reconstruction deliberately uses the detailed bridge-first pack as
    a *starting geometry* model.  Its bond lengths and angles are not a
    second acceptance contract: xTB is allowed to relax them.  The inexpensive
    pre-xTB gate therefore checks only finite coordinates and atom collapse.
    Graph legality is checked separately by the motif graph rules.
    """

    coords = np.asarray(coordinates, dtype=float)
    count = len(state.atoms)
    if coords.shape != (count, 3):
        return ["geometry:coordinate_shape"]
    if not np.all(np.isfinite(coords)):
        return ["geometry:non_finite"]
    wall = max(float(overlap_min_A), 0.0)
    violations: List[str] = []
    for left in range(count):
        for right in range(left + 1, count):
            distance = float(np.linalg.norm(coords[left] - coords[right]))
            if distance < wall:
                key = pair_key(state.atoms[left].symbol, state.atoms[right].symbol)
                violations.append(
                    f"overlap:{key}:{left}-{right}:{distance:.3f}<{wall:.3f}"
                )
    return violations


def _exact_local_geometry_violations(
    state: _State,
    coordinates: Sequence[Sequence[float]],
    pack: GeometryPack,
    spec: NucleationSpec,
) -> List[str]:
    """Validate hard local templates without changing constructed coordinates."""

    coords = np.asarray(coordinates, dtype=float)
    violations: List[str] = []
    improper_tolerance = pack.audit_improper_tolerance_deg
    audit_rank = _canonical_ranks(
        state,
        list(range(len(state.atoms))),
        [state.graph.degree[i] for i in range(len(state.atoms))],
    )
    for atom in state.atoms:
        center = atom.atom_id
        neighbors = sorted(
            state.graph.neighbors(center), key=audit_rank.__getitem__
        )
        # Angles are audited at every coordination, not only CN3: a CN2 Cd at
        # 148 deg against a 175 deg target used to pass simply because the loop
        # skipped it.  The improper stays CN3-only -- it is not defined
        # elsewhere.
        if len(neighbors) < 2:
            continue
        target = (
            pack.improper_angle_deg(
                atom.symbol, len(neighbors), _neighbor_signature(state, center)
            )
            if len(neighbors) == 3
            else None
        )
        # A missing (or disabled) improper target says nothing about the hard
        # centre angles, which are audited below for their own sake -- this
        # used to ``continue`` and skip them along with the improper.
        per_target_improper = (
            pack.improper_tolerance_deg(
                atom.symbol, len(neighbors), _neighbor_signature(state, center)
            )
            if len(neighbors) == 3
            else None
        )
        improper_band = (
            per_target_improper
            if per_target_improper is not None
            else improper_tolerance
        )
        if target is not None and improper_band is not None:
            vectors = []
            for neighbor in neighbors:
                direction = _unit(coords[neighbor] - coords[center])
                if direction is None:
                    violations.append(
                        f"nonplanar:{atom.symbol}:{center}:coincident"
                    )
                    vectors = []
                    break
                vectors.append(direction)
            if len(vectors) == 3:
                sine = float(np.dot(vectors[0], _cross3(vectors[1], vectors[2])))
                improper_deg = float(
                    np.degrees(np.arcsin(np.clip(sine, -1.0, 1.0)))
                )
                if abs(improper_deg - target) > improper_band:
                    violations.append(
                        f"improper:{atom.symbol}:{center}:"
                        f"{improper_deg:.8f}!={target:.8f}"
                    )
        for left, right in combinations(neighbors, 2):
            pair = "-".join(
                sorted((state.atoms[left].symbol, state.atoms[right].symbol))
            )
            role_signature, role_pair = _role_environment(
                state, center, left, right
            )
            if not pack.center_angle_is_hard(
                atom.symbol,
                len(neighbors),
                neighbor_pair=pair,
                signature=_neighbor_signature(state, center),
                role_pair=role_pair,
                role_signature=role_signature,
            ):
                continue
            desired = _effective_center_angle_deg(
                state, pack, center, left, right, default=None
            )
            if desired is None:
                continue
            left_vector = _unit(coords[left] - coords[center])
            right_vector = _unit(coords[right] - coords[center])
            assert left_vector is not None and right_vector is not None
            actual = float(
                np.degrees(
                    np.arccos(
                        np.clip(float(np.dot(left_vector, right_vector)), -1.0, 1.0)
                    )
                )
            )
            band = pack.center_angle_tolerance_deg(
                atom.symbol,
                len(neighbors),
                neighbor_pair=pair,
                signature=_neighbor_signature(state, center),
                role_pair=role_pair,
                role_signature=role_signature,
            )
            if band is None:
                band = pack.audit_angle_tolerance_deg
            if abs(actual - desired) > band:
                violations.append(
                    f"angle_geometry:{center}:{pair}:"
                    f"{actual:.8f}!={desired:.8f}"
                )
    return violations


def _audited_local_terms(
    state: _State,
    pack: GeometryPack,
    spec: NucleationSpec,
) -> Tuple[List[Tuple[int, int, int, int, float]], List[Tuple[int, int, int, float]]]:
    """Enumerate exactly what ``_exact_local_geometry_violations`` audits.

    Returns ``(impropers, hard_angles)`` where an improper is
    ``(center, n0, n1, n2, target_deg)`` over canonically ranked neighbours and
    a hard angle is ``(left, center, right, target_deg)``.

    The repair optimizers used to carry bond, contact and *soft* angle terms
    only, so a molecule rejected for an improper could never be repaired no
    matter how large the budget: nothing in the objective moved that quantity.
    Sharing this enumeration with the audit is what keeps the two definitions
    from drifting apart.
    """

    impropers: List[Tuple[int, int, int, int, float]] = []
    hard_angles: List[Tuple[int, int, int, float]] = []
    # When the pack disables the improper audit the optimizer must not chase it
    # either: an improper target that contradicts the pack's own angle medians
    # would drag the fit away from the geometry those medians describe.
    audit_impropers = pack.audit_improper_tolerance_deg is not None
    rank = _canonical_ranks(
        state,
        list(range(len(state.atoms))),
        [state.graph.degree[i] for i in range(len(state.atoms))],
    )
    for atom in state.atoms:
        center = atom.atom_id
        neighbors = sorted(state.graph.neighbors(center), key=rank.__getitem__)
        # Angles exist at every coordination and the audit checks them all, so
        # the optimizer has to restrain them all: restraining only CN3 while
        # auditing CN2 and CN4 guarantees a rejection the fit was never asked
        # to prevent.  The improper stays CN3-only -- it is undefined elsewhere.
        if len(neighbors) < 2:
            continue
        signature = _neighbor_signature(state, center)
        target = (
            pack.improper_angle_deg(atom.symbol, len(neighbors), signature)
            if audit_impropers and len(neighbors) == 3
            else None
        )
        if target is not None:
            impropers.append(
                (
                    center,
                    int(neighbors[0]),
                    int(neighbors[1]),
                    int(neighbors[2]),
                    float(target),
                )
            )
        for left, right in combinations(neighbors, 2):
            pair = "-".join(
                sorted((state.atoms[left].symbol, state.atoms[right].symbol))
            )
            role_signature, role_pair = _role_environment(state, center, left, right)
            if not pack.center_angle_is_hard(
                atom.symbol,
                len(neighbors),
                neighbor_pair=pair,
                signature=signature,
                role_pair=role_pair,
                role_signature=role_signature,
            ):
                continue
            desired = _effective_center_angle_deg(
                state, pack, center, left, right, default=None
            )
            if desired is None:
                continue
            band = pack.center_angle_tolerance_deg(
                atom.symbol,
                len(neighbors),
                neighbor_pair=pair,
                signature=signature,
                role_pair=role_pair,
                role_signature=role_signature,
            )
            modes = pack.center_angle_modes(
                atom.symbol,
                len(neighbors),
                neighbor_pair=pair,
                signature=signature,
                role_pair=role_pair,
                role_signature=role_signature,
            )
            width = float(
                band if band is not None else AUDIT_ANGLE_TOLERANCE_DEG
            )
            if modes:
                # A multi-modal centre is satisfied by the NEAREST mode, so one
                # row per mode sharing a group id; the residual takes the
                # minimum over the group.  A CN4 anion is cis (~85 deg) for four
                # of its six pairs and trans (~145 deg) for the other two, and a
                # single target sits in the empty valley between them.
                group = len(hard_angles)
                for deg_target, tol in modes:
                    hard_angles.append(
                        (int(left), center, int(right), float(deg_target),
                         float(tol), group)
                    )
            else:
                hard_angles.append(
                    (int(left), center, int(right), float(desired), width,
                     len(hard_angles))
                )
    return impropers, hard_angles


def _local_term_residuals(
    xyz: FloatArray,
    improper_index: FloatArray,
    improper_target: FloatArray,
    hard_index: FloatArray,
    hard_target: FloatArray,
    *,
    improper_scale: float,
    hard_scale: float,
    hard_group: Optional[FloatArray] = None,
) -> Tuple[FloatArray, FloatArray]:
    """Batched improper / hard-angle residuals shared by both optimizers."""

    if improper_index.size:
        center = xyz[improper_index[:, 0]]
        u0 = xyz[improper_index[:, 1]] - center
        u1 = xyz[improper_index[:, 2]] - center
        u2 = xyz[improper_index[:, 3]] - center
        u0 = u0 / np.maximum(np.linalg.norm(u0, axis=1), 1.0e-12)[:, None]
        u1 = u1 / np.maximum(np.linalg.norm(u1, axis=1), 1.0e-12)[:, None]
        u2 = u2 / np.maximum(np.linalg.norm(u2, axis=1), 1.0e-12)[:, None]
        sine = np.einsum("ij,ij->i", u0, np.cross(u1, u2))
        improper = np.degrees(np.arcsin(np.clip(sine, -1.0, 1.0)))
        improper_out = (
            _band_excess(improper, improper_target, improper_scale)
            / ANGLE_WELL_SCALE_DEG
        )
    else:
        improper_out = np.empty(0, dtype=float)
    if hard_index.size:
        center = xyz[hard_index[:, 1]]
        left = xyz[hard_index[:, 0]] - center
        right = xyz[hard_index[:, 2]] - center
        den = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        degenerate = den < 1.0e-12
        cosine = np.einsum("ij,ij->i", left, right) / np.where(degenerate, 1.0, den)
        actual = np.where(
            degenerate, 0.0, np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        )
        raw = _band_excess(actual, hard_target, hard_scale) / ANGLE_WELL_SCALE_DEG
        if hard_group is None:
            hard_out = raw
        else:
            # Rows sharing a group are alternative modes of ONE angle: the
            # angle is satisfied when it sits in any of them, so the group's
            # contribution is the smallest excess, not the sum.
            hard_out = np.full(int(hard_group.max()) + 1, np.inf, dtype=float)
            np.minimum.at(hard_out, hard_group, raw)
            hard_out = hard_out[np.isfinite(hard_out)]
    else:
        hard_out = np.empty(0, dtype=float)
    return improper_out, hard_out


def _canonical_ranks(
    state: _State,
    nodes: Sequence[int],
    degrees: Sequence[int],
) -> Dict[int, int]:
    """Position of each node in a canonical ordering of the induced subgraph.

    Construction order decides geometry: which anion seeds the breadth-first
    pass, which template slot each neighbour takes, which of two mirror
    solutions a bridge picks.  Driving those choices by atom id made the
    geometry -- and therefore acceptance -- depend on how the atoms happened to
    be numbered: at k=2, p=3, 22 of 32 accepted structures flipped to rejected
    when chemically identical atoms were merely permuted.  Ranking by a
    canonical labelling instead makes the constructed geometry a function of the
    molecule.

    Nodes are coloured by element *and* coordination number, because both
    already determine the local geometry the pack prescribes.
    """

    order = sorted(nodes)
    labels = [f"{state.atoms[node].symbol}:{degrees[node]}" for node in order]
    index = {node: position for position, node in enumerate(order)}
    edges = [
        (index[left], index[right], "bond")
        for left, right in state.graph.edges
        if left in index and right in index
    ]
    positions = canonical_form(
        labels, edges, compress_leaves=False
    ).positions
    return {node: positions[index[node]] for node in order}


def _ligand_groups(
    state: _State,
    spec: NucleationSpec,
    rank: Optional[Dict[int, int]] = None,
) -> Tuple[
    Dict[int, List[int]],
    Dict[Tuple[int, int], List[int]],
    List[Tuple[int, Tuple[int, ...]]],
]:
    """Split ligands into terminals, μ₂ bridges per host pair, and μ₃+ bridges.

    Shared by the embedder and by the pre-embedding feasibility screen so both
    visit the ligands in exactly the same order and therefore agree on which
    failure is reported first.
    """

    cations = {spec.core.cation, spec.precursor.center}
    ligand = spec.precursor.ligand
    order = (lambda i: i) if rank is None else rank.__getitem__

    terminals_on: Dict[int, List[int]] = {}
    bridges: List[Tuple[int, Tuple[int, ...]]] = []
    ligand_ids = sorted(
        (i for i, a in enumerate(state.atoms) if a.symbol == ligand),
        key=order,
    )
    for cl in ligand_ids:
        hosts = tuple(
            sorted(
                (
                    j
                    for j in state.graph.neighbors(cl)
                    if state.atoms[j].symbol in cations
                ),
                key=order,
            )
        )
        if len(hosts) == 1:
            terminals_on.setdefault(hosts[0], []).append(cl)
        else:
            bridges.append((cl, hosts))

    bridges_by_pair: Dict[Tuple[int, int], List[int]] = {}
    multi_host: List[Tuple[int, Tuple[int, ...]]] = []
    for cl, hosts in bridges:
        if len(hosts) == 2:
            bridges_by_pair.setdefault((hosts[0], hosts[1]), []).append(cl)
        else:
            multi_host.append((cl, hosts))
    return terminals_on, bridges_by_pair, multi_host


def frame_violations(
    state: _State,
    inorganic: FloatArray,
    pack: GeometryPack,
    spec: NucleationSpec,
    degrees: Optional[Sequence[int]] = None,
) -> List[str]:
    """Report defects the cation–anion frame carries on its own.

    A decoration only ever adds ligands, so anything already wrong *between
    frame atoms* is wrong for every decoration built on that frame: a skeleton
    bond the construction failed to realise, or two frame atoms closer than a
    forbidden-pair floor.  Deciding this once per frame replaces rediscovering
    it once per decoration -- at ``k=2, p=5`` one skeleton spent 2116 s and
    2,335,812 decorations to conclude exactly this.

    Returns at most one violation; the caller only needs the reason.
    """

    cations = {spec.core.cation, spec.precursor.center}
    anion = spec.core.anion
    frame = [
        atom.atom_id
        for atom in state.atoms
        if atom.symbol in cations or atom.symbol == anion
    ]
    if degrees is None:
        degrees = [state.graph.degree[i] for i in range(len(state.atoms))]
    coords = np.asarray(inorganic, dtype=float)

    for left, right in state.graph.edges:
        if state.atoms[left].symbol == spec.precursor.ligand:
            continue
        if state.atoms[right].symbol == spec.precursor.ligand:
            continue
        target = _molecular_bond_length(state, pack, spec, left, right, degrees)
        distance = float(np.linalg.norm(coords[left] - coords[right]))
        if abs(distance - target) > pack.audit_bond_tolerance_A:
            return [
                f"frame_bond_geometry:{left}-{right}:"
                f"{distance:.6f}!={target:.6f}"
            ]

    bonded = {
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in state.graph.edges
    }
    for index, left in enumerate(frame):
        for right in frame[index + 1 :]:
            key = pair_key(state.atoms[left].symbol, state.atoms[right].symbol)
            rule = spec.graph_rules.pair_rules.get(key)
            if rule is None or rule.bond_allowed:
                continue
            limit = float(rule.min_distance or 0.0)
            distance = float(np.linalg.norm(coords[left] - coords[right]))
            if distance < limit:
                return [
                    f"frame_contact:{key}:{left}-{right}:"
                    f"{distance:.3f}<{limit:.3f}"
                ]
    return []


def terminal_direction(
    state: _State,
    pack: GeometryPack,
    spec: NucleationSpec,
    degrees: Sequence[int],
    coords: FloatArray,
    placed: Sequence[bool],
    rank: Dict[int, int],
    host: int,
    terminal: int,
) -> FloatArray:
    """Direction for one terminal ligand, given everything already placed.

    Lifted out of the embedder so the pre-embed screen can replay it: on a host
    whose ligands are all terminal, every constraint comes from the frame, so
    whether a placement exists is decidable without building the molecule.
    """

    fixed = sorted(
        (
            neighbor
            for neighbor in state.graph.neighbors(host)
            if neighbor != terminal and placed[neighbor]
        ),
        key=rank.__getitem__,
    )
    if not fixed:
        return np.array([1.0, 0.0, 0.0])
    signature = _neighbor_signature(state, host)
    fixed_dirs: List[FloatArray] = []
    desired: List[float] = []
    host_symbol = state.atoms[host].symbol
    for neighbor in fixed:
        direction = _unit(coords[neighbor] - coords[host])
        if direction is None:
            raise ExactEmbeddingError(
                [f"coincident_bonded_atoms:{host}-{neighbor}"]
            )
        angle = _effective_center_angle_deg(
            state,
            pack,
            host,
            terminal,
            neighbor,
            default=None,
        )
        if angle is None:
            if degrees[host] == 2:
                angle = 180.0
            elif degrees[host] == 3:
                angle = 120.0
            else:
                angle = 109.471220634
        fixed_dirs.append(direction)
        desired.append(float(np.cos(np.radians(angle))))

    matrix = np.asarray(fixed_dirs, dtype=float)
    targets = np.asarray(desired, dtype=float)
    improper = pack.improper_angle_deg(
        host_symbol, degrees[host], signature
    )
    repeated_fixed = next(
        (
            index
            for index, neighbor in enumerate(fixed)
            if state.atoms[neighbor].symbol == state.atoms[terminal].symbol
        ),
        None,
    )
    repeated_pair = (
        None
        if repeated_fixed is None
        else "-".join(
            sorted(
                (
                    state.atoms[terminal].symbol,
                    state.atoms[fixed[repeated_fixed]].symbol,
                )
            )
        )
    )
    if (
        improper == 0.0
        and degrees[host] == 3
        and repeated_fixed is not None
        and repeated_pair is not None
        and len(fixed_dirs) >= 2
    ):
        anchor = fixed_dirs[repeated_fixed]
        other = fixed_dirs[0 if repeated_fixed != 0 else 1]
        normal = _unit(_cross3(anchor, other))
        if normal is None:
            _axis, normal, _other_axis = _orthonormal_basis(anchor)
        repeated_angle = _effective_center_angle_deg(
            state,
            pack,
            host,
            terminal,
            fixed[repeated_fixed],
            default=120.0,
        )
        assert repeated_angle is not None
        radians = np.radians(repeated_angle)
        candidates = [
            _unit(
                np.cos(radians) * anchor
                + sign * np.sin(radians) * _cross3(normal, anchor)
            )
            for sign in (-1.0, 1.0)
        ]
        valid_candidates = [
            candidate for candidate in candidates if candidate is not None
        ]
        if valid_candidates:
            return min(
                valid_candidates,
                key=lambda candidate: float(
                    np.sum(np.abs(matrix @ candidate - targets))
                ),
            )
    # ``matrix_rank``, not ``rank``: the enclosing scope already uses that
    # name for the canonical atom ordering.
    solution, _residuals, matrix_rank, _singular = np.linalg.lstsq(
        matrix, targets, rcond=None
    )
    norm_squared = float(np.dot(solution, solution))
    # ``|solution| > 1`` means the pack's angle targets to the already-placed
    # neighbours cannot all be met by any unit direction -- the placed geometry
    # (a rigid ring template) and the angle table are independent sources and
    # are not exactly reconcilable.  Demanding an exact unit solution here was
    # the same machine-precision mistake the audit tolerances used to make.
    #
    # Normalising the least-squares solution is *not* the best answer either:
    # ``argmin ||M d - t||`` over the unit sphere is a different vector from
    # the normalised unconstrained minimiser, and the gap grows exactly as the
    # targets become less reachable -- i.e. precisely in the cases that were
    # failing.  Solve the constrained problem instead, then judge the result by
    # the angle error it actually achieves rather than by a proxy on
    # ``|solution|``.
    if norm_squared > 1.0:
        direction = _best_unit_direction(matrix, targets)
        if direction is None:
            raise ExactEmbeddingError(
                [f"inconsistent_terminal_angles:{terminal}"]
            )
        achieved = np.degrees(
            np.arccos(np.clip(matrix @ direction, -1.0, 1.0))
        )
        wanted = np.degrees(np.arccos(np.clip(targets, -1.0, 1.0)))
        worst = float(np.max(np.abs(achieved - wanted))) if len(wanted) else 0.0
        # Construction is allowed to be looser than the audit: a placement a
        # few degrees out is exactly what the bounded repair exists to pull
        # back, and the audit still has the final say.  Only give up when no
        # unit direction comes close at all.
        if worst > TERMINAL_DIRECTION_MAX_ANGLE_ERROR_DEG:
            raise ExactEmbeddingError(
                [f"inconsistent_terminal_angles:{terminal}"]
            )
        return direction
    if improper == 0.0 and degrees[host] == 3 and matrix_rank >= 2:
        direction = _unit(solution)
        if direction is None:
            raise ExactEmbeddingError(
                [f"inconsistent_terminal_angles:{terminal}"]
            )
        return direction
    if matrix_rank < 3 and norm_squared <= 1.0:
        # Under-determined: use the spare null-space direction to reach unit
        # length exactly, which is the historical (and exact) behaviour.
        _u, _s, vh = np.linalg.svd(matrix, full_matrices=True)
        null_direction = vh[-1]
        solution = solution + np.sqrt(max(0.0, 1.0 - norm_squared)) * null_direction
    elif abs(norm_squared - 1.0) > 1.0e-6:
        direction = _unit(solution)
        if direction is None:
            raise ExactEmbeddingError(
                [f"inconsistent_terminal_angles:{terminal}"]
            )
        return direction
    direction = _unit(solution)
    if direction is None or np.max(np.abs(matrix @ direction - targets)) > 1.0e-6:
        raise ExactEmbeddingError(
            [f"inconsistent_terminal_angles:{terminal}"]
        )
    return direction


def terminal_direction_candidates(
    state: _State,
    pack: GeometryPack,
    spec: NucleationSpec,
    degrees: Sequence[int],
    coords: FloatArray,
    placed: Sequence[bool],
    rank: Dict[int, int],
    host: int,
    terminal: int,
) -> List[FloatArray]:
    """Return angle-compatible terminal directions ranked by local torsion/clearance."""

    base = terminal_direction(
        state, pack, spec, degrees, coords, placed, rank, host, terminal
    )
    candidates: List[FloatArray] = [base]
    fixed = sorted(
        (
            neighbor
            for neighbor in state.graph.neighbors(host)
            if neighbor != terminal and placed[neighbor]
        ),
        key=rank.__getitem__,
    )
    if not fixed:
        return candidates

    axes = [
        _unit(coords[neighbor] - coords[host])
        for neighbor in fixed
    ]
    for axis in (value for value in axes if value is not None):
        for angle_deg in (45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0):
            theta = np.radians(angle_deg)
            rotated = (
                base * np.cos(theta)
                + _cross3(axis, base) * np.sin(theta)
                + axis * float(np.dot(axis, base)) * (1.0 - np.cos(theta))
            )
            unit = _unit(rotated)
            if unit is None:
                continue
            hard_ok = True
            for neighbor in fixed:
                fixed_direction = _unit(coords[neighbor] - coords[host])
                assert fixed_direction is not None
                pair = "-".join(
                    sorted((state.atoms[terminal].symbol, state.atoms[neighbor].symbol))
                )
                role_signature, role_pair = _role_environment(
                    state, host, terminal, neighbor
                )
                signature = _neighbor_signature(state, host)
                if not pack.center_angle_is_hard(
                    state.atoms[host].symbol,
                    degrees[host],
                    neighbor_pair=pair,
                    signature=signature,
                    role_pair=role_pair,
                    role_signature=role_signature,
                ):
                    continue
                desired = _effective_center_angle_deg(
                    state, pack, host, terminal, neighbor, default=None
                )
                if desired is None:
                    continue
                actual = float(
                    np.degrees(
                        np.arccos(
                            np.clip(float(np.dot(unit, fixed_direction)), -1.0, 1.0)
                        )
                    )
                )
                if abs(actual - desired) > EXACT_ANGLE_TOLERANCE_DEG:
                    hard_ok = False
                    break
            if hard_ok:
                candidates.append(unit)

    unique: List[FloatArray] = []
    for candidate in candidates:
        if not any(float(np.linalg.norm(candidate - other)) < 1.0e-8 for other in unique):
            unique.append(candidate)
    return sorted(
        unique,
        key=lambda candidate: (
            _candidate_torsion_penalty(
                state, pack, terminal,
                coords[host] + _molecular_bond_length(
                    state, pack, spec, terminal, host, degrees
                ) * candidate,
                coords, placed, set(),
            ),
            _candidate_soft_clearance_penalty(
                state,
                pack,
                terminal,
                coords[host] + _molecular_bond_length(
                    state, pack, spec, terminal, host, degrees
                ) * candidate,
                coords,
                placed,
            ),
        ),
    )


def local_angle_violations(
    state: _State,
    frame: FloatArray,
    pack: GeometryPack,
    spec: NucleationSpec,
    degrees: Optional[Sequence[int]] = None,
    rank: Optional[Dict[int, int]] = None,
) -> List[str]:
    """Hard angles the frame already fixes and no ligand placement can repair.

    Between two anions on the same cation the angle is settled the moment the
    frame is built -- ligands attach elsewhere on that cation and cannot move
    it.  If the pack calls that angle hard and the frame misses it, this frame
    can never carry this molecule, and the several milliseconds of embedding it
    would take to discover that are wasted.

    A necessary condition, not a sufficient one: it says nothing about angles
    involving a ligand.  Its job is to let many frames be screened cheaply so a
    molecule is only called unrealisable after a wide search, not a token one.
    """

    if degrees is None:
        degrees = [state.graph.degree[i] for i in range(len(state.atoms))]
    cations = {spec.core.cation, spec.precursor.center}
    anion = spec.core.anion
    coords = np.asarray(frame, dtype=float)

    for atom in state.atoms:
        if atom.symbol not in cations:
            continue
        center = atom.atom_id
        fixed = [
            neighbor
            for neighbor in state.graph.neighbors(center)
            if state.atoms[neighbor].symbol == anion
        ]
        if len(fixed) < 2:
            continue
        signature = _neighbor_signature(state, center)
        for left, right in combinations(sorted(fixed), 2):
            pair = "-".join(
                sorted((state.atoms[left].symbol, state.atoms[right].symbol))
            )
            role_signature, role_pair = _role_environment(
                state, center, left, right
            )
            if not pack.center_angle_is_hard(
                atom.symbol,
                degrees[center],
                neighbor_pair=pair,
                signature=signature,
                role_pair=role_pair,
                role_signature=role_signature,
            ):
                continue
            desired = _effective_center_angle_deg(
                state, pack, center, left, right, default=None
            )
            if desired is None:
                continue
            first = _unit(coords[left] - coords[center])
            second = _unit(coords[right] - coords[center])
            if first is None or second is None:
                continue
            actual = float(
                np.degrees(
                    np.arccos(
                        np.clip(float(np.dot(first, second)), -1.0, 1.0)
                    )
                )
            )
            if abs(actual - desired) > EXACT_ANGLE_TOLERANCE_DEG:
                return [
                    f"local_angle:{center}:{pair}:"
                    f"{actual:.6f}!={desired:.6f}"
                ]

    # Hosts carrying only terminal ligands are fully decided by the frame: no
    # bridge can reach them, so every direction they are solved against is
    # already known here.  Replaying their placement costs microseconds and
    # settles the single largest embed-stage rejection
    # (``inconsistent_terminal_angles``) without building the molecule.
    ligand = spec.precursor.ligand
    if rank is None:
        rank = _canonical_ranks(state, list(range(len(state.atoms))), degrees)
    working = np.array(coords, dtype=float, copy=True)
    for atom in state.atoms:
        if atom.symbol not in cations:
            continue
        center = atom.atom_id
        attached = [
            neighbor
            for neighbor in state.graph.neighbors(center)
            if state.atoms[neighbor].symbol == ligand
        ]
        if not attached or any(degrees[cl] != 1 for cl in attached):
            continue
        local_placed = [
            state.atoms[i].symbol in cations or state.atoms[i].symbol == anion
            for i in range(len(state.atoms))
        ]
        for cl in sorted(attached, key=rank.__getitem__):
            try:
                direction = terminal_direction(
                    state, pack, spec, degrees, working, local_placed,
                    rank, center, cl,
                )
            except ExactEmbeddingError as exc:
                return list(exc.reasons)
            working[cl] = working[center] + _molecular_bond_length(
                state, pack, spec, cl, center, degrees
            ) * direction
            local_placed[cl] = True
    return []


def bridge_feasibility_violations(
    state: _State,
    inorganic: FloatArray,
    pack: GeometryPack,
    spec: NucleationSpec,
) -> List[str]:
    """Report bridge placements the fixed rules cannot realise.

    Every bridging ligand sits on the intersection of spheres centred on its
    hosts, and those host positions are already fixed by the inorganic frame.
    Exact sphere failures are deliberately deferred to the bounded complete
    graph solve in ``_CandidateScreen.offer``; otherwise a rigid partial frame
    would reject decorations that become realisable when the host triangle is
    allowed to move slightly.  Unsupported multi-host motifs remain hard
    failures here.

    Returns at most one violation, matching the embedder's fail-fast order.
    """

    _terminals, bridges_by_pair, multi_host = _ligand_groups(state, spec)
    degrees = [state.graph.degree[i] for i in range(len(state.atoms))]

    for (left, right), cl_list in bridges_by_pair.items():
        separation = float(
            np.linalg.norm(inorganic[right] - inorganic[left])
        )
        if separation < 1.0e-12:
            return [f"bridge_hosts_coincident:{left}-{right}"]
        configured_max = spec.graph_rules.bridge_cd_cd_max_distance
        if (
            configured_max is not None
            and separation > float(configured_max) + EXACT_BOND_TOLERANCE
        ):
            return [
                f"bridge_cd_cd_distance:{left}-{right}:"
                f"{separation:.4f}>{float(configured_max):.4f}"
            ]
        for cl in sorted(cl_list):
            radius_left = _molecular_bond_length(
                state, pack, spec, cl, left, degrees
            )
            radius_right = _molecular_bond_length(
                state, pack, spec, cl, right, degrees
            )
            axial = (
                radius_left * radius_left
                - radius_right * radius_right
                + separation * separation
            ) / (2.0 * separation)
            if radius_left * radius_left - axial * axial < -EXACT_BOND_TOLERANCE:
                # The complete-graph fallback in ``offer`` can move the
                # bridge hosts together while preserving the inorganic bond
                # network approximately.  Rejecting here would make a rigid
                # partial frame decide the chemistry before that solve runs.
                continue

    for cl, hosts in multi_host:
        if len(hosts) != 3:
            return [f"unsupported_multi_host_bridge:{cl}:{len(hosts)}"]
        try:
            _three_sphere_intersections(
                [inorganic[host] for host in hosts],
                [
                    _molecular_bond_length(
                        state, pack, spec, cl, host, degrees
                    )
                    for host in hosts
                ],
            )
        except ExactEmbeddingError as exc:
            # Defer sphere incompatibility to the bounded complete-graph
            # construction.  Structural errors such as an unsupported μ4
            # ligand remain hard failures.
            if any("collinear" in reason or "coincident" in reason for reason in exc.reasons):
                continue
            continue
    return []


def inorganic_coordinates(
    state: _State,
    pack: GeometryPack,
    spec: NucleationSpec,
    degrees: Optional[Sequence[int]] = None,
    *,
    max_slot_orders: int = 24,
) -> Tuple[FloatArray, List[bool]]:
    """Place the cation–anion frame; ligands are not touched.

    The result depends only on the skeleton and on the *final* coordination
    numbers (which set both the bond lengths and the centre angles), never on
    where the ligands end up.  Callers enumerating many decorations of one
    skeleton can therefore memoise this on the cation degree vector.

    Which template slot each first-shell neighbour takes decides whether a ring
    can close later: two cations in the axial slots of a CN-5 anion are 5.6 A
    apart, too far for a shared anion to bond both, while adjacent slots put
    them 3.9 A apart and close fine.  The default order is tried first, so an
    acyclic frame is built exactly as before.
    """

    frames, failure = _clean_frames(
        state, pack, spec, degrees, max_slot_orders=max_slot_orders, limit=1
    )
    if frames:
        return frames[0]
    if failure is not None:
        raise ExactEmbeddingError(failure)
    raise ExactEmbeddingError(["frame_not_realisable"])


def _clean_frames(
    state: _State,
    pack: GeometryPack,
    spec: NucleationSpec,
    degrees: Optional[Sequence[int]] = None,
    *,
    max_slot_orders: int = 24,
    limit: int = 0,
) -> Tuple[List[Tuple[FloatArray, List[bool]]], Optional[List[str]]]:
    """Frames with no defect of their own, in a deterministic order.

    More than one is worth keeping.  A frame that is perfectly sound on its own
    can still leave a cation's neighbours arranged so that no ligand placement
    satisfies the local angles, while a different slot order for the *same*
    coordination numbers admits one: every structure lost to a single-frame
    choice at k=2, p=3 turned out to be realisable under 6 to 12 of the orders.
    So "unrealisable" has to mean no frame works, not that the first one didn't.

    Returns the clean frames plus, when there are none, the best explanation.
    """

    if degrees is None:
        degrees = [state.graph.degree[i] for i in range(len(state.atoms))]
    found: List[Tuple[FloatArray, List[bool]]] = []
    best: Optional[Tuple[FloatArray, List[bool]]] = None
    best_error: Optional[Tuple[int, float]] = None
    best_reasons: Optional[List[str]] = None
    first_error: Optional[List[str]] = None
    failure_counts: Counter[str] = Counter()
    for order in _root_slot_orders(state, spec, degrees, max_slot_orders):
        try:
            coords, placed = _construct_inorganic(
                state, pack, spec, degrees, order
            )
        except ExactEmbeddingError as exc:
            failure_counts.update(str(reason) for reason in exc.reasons)
            if first_error is None:
                first_error = list(exc.reasons)
            continue
        violations = frame_violations(state, coords, pack, spec, degrees)
        if not violations:
            found.append((coords, placed))
            if 0 < limit <= len(found):
                return found, None
            continue
        failure_counts.update(str(reason) for reason in violations)
        score = (
            len(violations),
            _frame_bond_error(state, coords, pack, spec, degrees),
        )
        if best_error is None or score < best_error:
            best_error, best, best_reasons = score, (coords, placed), violations
    if found:
        return found, None
    if best is not None:
        return [], best_reasons
    if failure_counts:
        return [], [failure_counts.most_common(1)[0][0]]
    return [], first_error


def _frame_bond_error(
    state: _State,
    coords: FloatArray,
    pack: GeometryPack,
    spec: NucleationSpec,
    degrees: Sequence[int],
) -> float:
    ligand = spec.precursor.ligand
    worst = 0.0
    for left, right in state.graph.edges:
        if ligand in (state.atoms[left].symbol, state.atoms[right].symbol):
            continue
        target = _molecular_bond_length(state, pack, spec, left, right, degrees)
        distance = float(np.linalg.norm(coords[left] - coords[right]))
        worst = max(worst, abs(distance - target))
    return worst


def _root_slot_orders(
    state: _State,
    spec: NucleationSpec,
    degrees: Sequence[int],
    limit: int,
) -> Iterable[Optional[Tuple[int, ...]]]:
    """Template-slot orderings to try for the root anion, default first."""

    yield None
    anion = spec.core.anion
    roots = [i for i, a in enumerate(state.atoms) if a.symbol == anion]
    if not roots:
        return
    anion_nodes = [a.atom_id for a in state.atoms if a.symbol == anion]
    cation_nodes = [
        a.atom_id
        for a in state.atoms
        if a.symbol in {spec.core.cation, spec.precursor.center}
    ]
    root_rank = _canonical_ranks(state, anion_nodes + cation_nodes, degrees)
    roots.sort(key=lambda i: (-degrees[i], root_rank[i]))
    count = state.graph.degree[roots[0]]
    if count < 2:
        return
    emitted = 0
    for order in permutations(range(count)):
        if order == tuple(range(count)):
            continue
        emitted += 1
        if emitted > limit:
            return
        yield order


def _construct_inorganic(
    state: _State,
    pack: GeometryPack,
    spec: NucleationSpec,
    degrees: Sequence[int],
    root_slot_order: Optional[Tuple[int, ...]] = None,
) -> Tuple[FloatArray, List[bool]]:
    n = len(state.atoms)
    coords = np.zeros((n, 3), dtype=float)
    placed = [False] * n
    cations = {spec.core.cation, spec.precursor.center}
    anion = spec.core.anion
    frame_nodes = [
        atom.atom_id
        for atom in state.atoms
        if atom.symbol in cations or atom.symbol == anion
    ]
    rank = _canonical_ranks(state, frame_nodes, degrees)

    def bond_r(i: int, j: int) -> float:
        return _molecular_bond_length(state, pack, spec, i, j, degrees)

    def place_atom(idx: int, pos: FloatArray) -> None:
        coords[idx] = np.asarray(pos, dtype=float)
        placed[idx] = True

    # --- inorganic BFS from highest-CN Se ---
    se_nodes = [
        i for i, a in enumerate(state.atoms) if a.symbol == anion
    ]
    se_nodes.sort(key=lambda i: (-degrees[i], rank[i]))
    if not se_nodes:
        return coords, placed

    root = se_nodes[0]
    place_atom(root, np.zeros(3))
    queue = [root]

    while queue:
        center = queue.pop(0)
        neigh = sorted(
            (
                j
                for j in state.graph.neighbors(center)
                if state.atoms[j].symbol in cations | {anion}
            ),
            key=rank.__getitem__,
        )
        unplaced = [j for j in neigh if not placed[j]]
        if not unplaced:
            continue
        el = state.atoms[center].symbol
        cn = degrees[center]
        if el == anion:
            ang = pack.center_angle_deg(el, cn, default=90.0) or 90.0
        elif el in cations:
            if cn == 2:
                ang = pack.center_angle_deg(el, 2, default=176.0) or 176.0
            else:
                ang = pack.center_angle_deg(el, cn, default=109.5) or 109.5
        else:
            ang = 109.5

        placed_neigh = [j for j in neigh if placed[j]]
        n_total = len(unplaced) + len(placed_neigh)
        template = _angle_directions(max(n_total, len(unplaced)), ang)

        if len(placed_neigh) > 1:
            # A ring closure: honour every direction we are already committed
            # to, not just the first one.
            free_dirs = _align_template_to_placed(
                template,
                [coords[j] - coords[center] for j in placed_neigh],
            ) or _regular_directions(len(unplaced))
        elif placed_neigh:
            ref = coords[placed_neigh[0]] - coords[center]
            dirs = _rotate_dirs_to_frame(template, ref)
            # first template slot is occupied by placed neighbor
            free_dirs = dirs[1:] if len(dirs) > 1 else _regular_directions(len(unplaced))
        else:
            free_dirs = template if len(template) >= len(unplaced) else _regular_directions(len(unplaced))

        # Ensure enough distinct directions
        if len(free_dirs) < len(unplaced):
            free_dirs = list(free_dirs) + _regular_directions(len(unplaced))
        if (
            root_slot_order is not None
            and center == root
            and len(free_dirs) >= len(root_slot_order)
        ):
            free_dirs = [free_dirs[slot] for slot in root_slot_order]
        free_dirs = free_dirs[: len(unplaced)]

        for offset, j in enumerate(unplaced):
            if placed[j]:
                # Reached again from a second parent while this centre was
                # being expanded; its position is already fixed.
                continue
            direction = np.asarray(free_dirs[offset], dtype=float)
            un = _unit(direction)
            if un is None:
                un = np.array([1.0, 0.0, 0.0])
            r = bond_r(center, j)
            pos = coords[center] + r * un
            # A ring closure reaches this atom from more than one placed
            # neighbour.  Solving for a point that honours *all* of them is what
            # the plain template cannot do: the BFS realises only a spanning
            # tree, leaving every extra skeleton bond at whatever length falls
            # out -- measured as far off as 6.31 A against a 2.53 A target,
            # which killed every cyclic skeleton outright.
            anchors = [center] + sorted(
                (
                    other
                    for other in state.graph.neighbors(j)
                    if other != center
                    and placed[other]
                    and state.atoms[other].symbol in cations | {anion}
                ),
                key=rank.__getitem__,
            )
            if len(anchors) > 1:
                pos = _multilaterated_position(
                    [coords[anchor] for anchor in anchors],
                    [bond_r(j, anchor) for anchor in anchors],
                    pos,
                    [coords[i] for i in range(n) if placed[i]],
                )
            place_atom(j, pos)
            queue.append(j)

    # Fallback inorganic not reached
    for i in sorted(frame_nodes, key=rank.__getitem__):
        if placed[i]:
            continue
        if state.atoms[i].symbol not in cations | {anion}:
            continue
        neigh = sorted(
            (
                j
                for j in state.graph.neighbors(i)
                if placed[j] and j in rank
            ),
            key=rank.__getitem__,
        )
        if not neigh:
            raise ExactEmbeddingError([f"unreachable_inorganic:{i}"])
        j = neigh[0]
        r = bond_r(i, j)
        pos = coords[j] + r * _regular_directions(max(1, degrees[j]))[0]
        place_atom(i, pos)

    return coords, placed


def _multilaterated_position(
    anchors: Sequence[FloatArray],
    radii: Sequence[float],
    preferred: FloatArray,
    occupied: Sequence[FloatArray] = (),
) -> FloatArray:
    """Point at exactly the given distance from every anchor.

    Where the constraints leave freedom -- a circle for two anchors, a mirror
    pair for three -- the branch closest to ``preferred`` is taken, so the
    result stays near what the local angular template wanted and the choice
    stays deterministic.
    """

    if len(anchors) == 2:
        left, right = np.asarray(anchors[0]), np.asarray(anchors[1])
        radius_left, radius_right = float(radii[0]), float(radii[1])
        offset = right - left
        separation = float(np.linalg.norm(offset))
        if separation < 1.0e-12:
            raise ExactEmbeddingError(["frame_anchors_coincident"])
        axis = offset / separation
        axial = (
            radius_left * radius_left
            - radius_right * radius_right
            + separation * separation
        ) / (2.0 * separation)
        height_squared = radius_left * radius_left - axial * axial
        if height_squared < -EXACT_BOND_TOLERANCE:
            raise ExactEmbeddingError(["frame_spheres_do_not_intersect"])
        base = left + axial * axis
        toward = np.asarray(preferred, dtype=float) - base
        toward = toward - float(np.dot(toward, axis)) * axis
        direction = _unit(toward)
        if direction is None:
            _axis, direction, _other = _orthonormal_basis(axis)
        return base + np.sqrt(max(0.0, height_squared)) * direction

    candidates = _three_sphere_intersections(anchors[:3], radii[:3])
    extra = list(zip(anchors[3:], radii[3:]))

    def penalty(point: FloatArray) -> Tuple[int, float, float]:
        # When several cations are equidistant from two anions, one of the two
        # mirror solutions *is* the other anion's position.  Rank any candidate
        # that lands on an existing atom last rather than stacking them.
        collides = int(
            any(
                float(np.linalg.norm(point - np.asarray(other))) < 0.5
                for other in occupied
            )
        )
        residual = sum(
            abs(float(np.linalg.norm(point - center)) - radius)
            for center, radius in extra
        )
        return collides, residual, float(np.linalg.norm(point - preferred))

    return min(candidates, key=penalty)


def embed_molecular_state(
    state: _State,
    pack: GeometryPack,
    spec: NucleationSpec,
    *,
    inorganic: Optional[Tuple[FloatArray, List[bool]]] = None,
    rank: Optional[Dict[int, int]] = None,
) -> Tuple[Tuple[float, float, float], ...]:
    """Construct one fixed, deterministic geometry without coordinate repair.

    ``inorganic`` supplies a pre-computed cation–anion frame from
    :func:`inorganic_coordinates`; it is shared by every decoration with the
    same skeleton and coordination numbers, so an enumeration can build it once
    instead of once per candidate.
    """

    n = len(state.atoms)
    degrees = [state.graph.degree[i] for i in range(n)]
    cations = {spec.core.cation, spec.precursor.center}
    anion = spec.core.anion

    if inorganic is None:
        inorganic = inorganic_coordinates(state, pack, spec)
    frame, frame_placed = inorganic
    coords = np.array(frame, dtype=float, copy=True)
    placed = list(frame_placed)

    if not any(atom.symbol == anion for atom in state.atoms):
        return tuple((float(x), float(y), float(z)) for x, y, z in coords)

    # Ligand placement ranks over the *whole* molecule: which bridge is laid
    # down first, and which azimuth it takes, must not depend on atom ids.
    if rank is None:
        rank = _canonical_ranks(state, list(range(n)), degrees)

    def bond_r(i: int, j: int) -> float:
        return _molecular_bond_length(state, pack, spec, i, j, degrees)

    def place_atom(idx: int, pos: FloatArray) -> None:
        coords[idx] = np.asarray(pos, dtype=float)
        placed[idx] = True

    # --- ligands: group by placement mode so multi-Cl get distinct slots ---
    terminals_on, bridges_by_pair, multi_host = _ligand_groups(state, spec, rank)

    for (a, b), cl_list in bridges_by_pair.items():
        ab = coords[b] - coords[a]
        ab_u, u_perp, v_perp = _orthonormal_basis(ab)
        host_distance = float(np.linalg.norm(ab))
        if host_distance < 1.0e-12:
            raise ExactEmbeddingError([f"bridge_hosts_coincident:{a}-{b}"])
        # Distinct azimuthal slots around the Cd–Cd axis (never reuse a slot).
        for idx, cl in enumerate(sorted(cl_list, key=rank.__getitem__)):
            ra = bond_r(cl, a)
            rb = bond_r(cl, b)
            axial = (
                ra * ra - rb * rb + host_distance * host_distance
            ) / (2.0 * host_distance)
            height_squared = ra * ra - axial * axial
            if height_squared < -EXACT_BOND_TOLERANCE:
                place_atom(
                    cl,
                    _relaxed_sphere_position(
                        [coords[a], coords[b]],
                        [ra, rb],
                        preferred=(coords[a] + coords[b]) / 2.0,
                    ),
                )
                continue
            height = np.sqrt(max(0.0, height_squared))
            base = coords[a] + axial * ab_u
            shared = sorted(
                (
                    neighbor
                    for neighbor in nx.common_neighbors(state.graph, a, b)
                    if state.atoms[neighbor].symbol == spec.core.anion
                    and placed[neighbor]
                ),
                key=rank.__getitem__,
            )
            if shared:
                toward_shared = coords[shared[0]] - base
                toward_shared -= float(np.dot(toward_shared, ab_u)) * ab_u
                reference = _unit(-toward_shared)
            else:
                reference = u_perp
            if reference is None:
                reference = u_perp
            around = _unit(_cross3(ab_u, reference))
            if around is None:
                around = v_perp
            # The sphere intersection is a circle. Select its azimuth from the
            # *final* host environments, so adding this bridge changes the Cd
            # CN before its local angle table is queried.
            constraints: List[Tuple[int, int, float]] = []
            candidate_angles = [
                idx * (2.0 * np.pi / max(len(cl_list), 1)),
                np.pi + idx * (2.0 * np.pi / max(len(cl_list), 1)),
            ]
            # The circle is the remaining rotational degree of freedom.  A
            # deterministic fine sampling lets the DFT-peaked bridge torsion
            # participate without changing the exact sphere/bond constraints.
            candidate_angles.extend(
                np.radians(float(value)) for value in range(0, 360, 30)
            )
            for host in (a, b):
                radius = bond_r(cl, host)
                for neighbor in sorted(
                    state.graph.neighbors(host), key=rank.__getitem__
                ):
                    if neighbor == cl or not placed[neighbor]:
                        continue
                    desired_angle = _effective_center_angle_deg(
                        state, pack, host, cl, neighbor, default=None
                    )
                    if desired_angle is None:
                        continue
                    constraints.append((host, neighbor, desired_angle))
                    if height < 1.0e-12:
                        continue
                    fixed_direction = _unit(coords[neighbor] - coords[host])
                    assert fixed_direction is not None
                    coefficient_cos = float(np.dot(reference, fixed_direction))
                    coefficient_sin = float(np.dot(around, fixed_direction))
                    amplitude = float(
                        np.hypot(coefficient_cos, coefficient_sin)
                    )
                    if amplitude < 1.0e-12:
                        continue
                    rhs = (
                        np.cos(np.radians(desired_angle)) * radius
                        - float(np.dot(base - coords[host], fixed_direction))
                    ) / height
                    ratio = rhs / amplitude
                    if ratio < -1.0 - 1.0e-10 or ratio > 1.0 + 1.0e-10:
                        continue
                    phase = float(
                        np.arctan2(coefficient_sin, coefficient_cos)
                    )
                    delta = float(np.arccos(np.clip(ratio, -1.0, 1.0)))
                    candidate_angles.extend((phase + delta, phase - delta))

            def bridge_candidate_score(
                theta: float,
            ) -> Tuple[float, float, float, float, float, float, float]:
                point = base + height * (
                    np.cos(theta) * reference + np.sin(theta) * around
                )
                final_cn3_errors: List[float] = []
                other_errors: List[float] = []
                for host, neighbor, desired_angle in constraints:
                    bridge_direction = _unit(point - coords[host])
                    fixed_direction = _unit(coords[neighbor] - coords[host])
                    assert bridge_direction is not None
                    assert fixed_direction is not None
                    actual = float(
                        np.degrees(
                            np.arccos(
                                np.clip(
                                    float(
                                        np.dot(
                                            bridge_direction, fixed_direction
                                        )
                                    ),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    )
                    target_errors = (
                        final_cn3_errors
                        if degrees[host] == 3
                        else other_errors
                    )
                    target_errors.append(abs(actual - desired_angle))
                torsion_errors: List[float] = []
                for host, other_host in ((a, b), (b, a)):
                    for neighbor in sorted(
                        state.graph.neighbors(host), key=rank.__getitem__
                    ):
                        if (
                            not placed[neighbor]
                            or state.atoms[neighbor].symbol != spec.core.anion
                        ):
                            continue
                        path = tuple(
                            state.atoms[index].symbol
                            for index in (other_host, cl, host, neighbor)
                        )
                        preferred = pack.preferred_dihedral(path)
                        if preferred is None:
                            continue
                        target, tolerance = preferred
                        value = _dihedral_deg_points(
                            coords[other_host],
                            point,
                            coords[host],
                            coords[neighbor],
                        )
                        torsion_errors.append(
                            abs(_dihedral_delta_deg(value, target))
                            / max(tolerance, 1.0e-6)
                        )
                soft = _candidate_soft_clearance_penalty(
                    state, pack, cl, point, coords, placed
                )
                return (
                    max(final_cn3_errors, default=0.0),
                    sum(final_cn3_errors),
                    max(other_errors, default=0.0),
                    sum(other_errors),
                    sum(torsion_errors),
                    soft,
                    float(theta % (2.0 * np.pi)),
                )

            theta = min(candidate_angles, key=bridge_candidate_score)
            nrm = _unit(
                np.cos(theta) * reference + np.sin(theta) * around
            )
            if nrm is None:
                nrm = u_perp
            place_atom(cl, base + height * nrm)

    for cl, hosts in multi_host:
        if len(hosts) != 3:
            raise ExactEmbeddingError(
                [f"unsupported_multi_host_bridge:{cl}:{len(hosts)}"]
            )
        host_points = [coords[host] for host in hosts]
        host_radii = [bond_r(cl, host) for host in hosts]
        try:
            candidates = _three_sphere_intersections(host_points, host_radii)
        except ExactEmbeddingError:
            place_atom(
                cl,
                _relaxed_sphere_position(
                    host_points,
                    host_radii,
                    preferred=np.mean(host_points, axis=0),
                ),
            )
            continue
        shared_anions = sorted(
            set.intersection(
                *(
                    {
                        neighbor
                        for neighbor in state.graph.neighbors(host)
                        if state.atoms[neighbor].symbol == spec.core.anion
                        and placed[neighbor]
                    }
                    for host in hosts
                )
            ),
            key=rank.__getitem__,
        )
        if shared_anions:
            reference = np.mean(
                [coords[neighbor] for neighbor in shared_anions], axis=0
            )
            chosen = max(
                candidates,
                key=lambda point: float(np.linalg.norm(point - reference)),
            )
        else:
            chosen = candidates[0]
        place_atom(cl, chosen)

    # Terminal positions follow fixed local angles after bridge positions are
    # known.  No contact or clash information participates in this placement.
    for host, cl_list in sorted(
        terminals_on.items(), key=lambda item: rank[item[0]]
    ):
        for cl in sorted(cl_list, key=rank.__getitem__):
            direction = terminal_direction_candidates(
                state, pack, spec, degrees, coords, placed, rank, host, cl
            )[0]
            place_atom(cl, coords[host] + bond_r(cl, host) * direction)

    # Ensure every atom placed
    for i in range(n):
        if not placed[i]:
            raise ExactEmbeddingError([f"unplaced_atom:{i}"])

    return tuple((float(x), float(y), float(z)) for x, y, z in coords)


def _steric_relax_ligands(
    state: _State,
    pack: GeometryPack,
    spec: NucleationSpec,
    coordinates: Sequence[Sequence[float]],
    *,
    log: Optional[Callable[[str], None]] = None,
    stats: Optional[Dict[str, float]] = None,
) -> Optional[FloatArray]:
    """Reduce clashes with a constrained, regularized geometry solve.

    Ring atoms stay fixed.  Ligands and non-ring Cd/Se atoms are variables,
    every graph bond is a strongly weighted residual, and bridge/skeleton
    angles are regularized around their input values rather than held exactly
    fixed.  The residual form is intentional: unlike SLSQP's scalar objective
    plus separately finite-differenced equality constraints, sparse
    ``least_squares`` evaluates one local residual vector and can use a sparse
    finite-difference Jacobian.  Hard contact floors and exact bond tolerances
    are still audited by the caller.

    ``log`` receives labelled start/end messages.  With
    ``QD_MOLECULAR_RELAX_TRACE=1`` SciPy also prints its iteration cost table.
    """

    initial = np.asarray(coordinates, dtype=float)
    if initial.ndim != 2 or initial.shape != (len(state.atoms), 3):
        return None
    ligand = spec.precursor.ligand
    ligand_ids = [
        index for index, atom in enumerate(state.atoms)
        if atom.symbol == ligand
    ]
    if not ligand_ids:
        return None
    degrees = [state.graph.degree[i] for i in range(len(state.atoms))]
    cations = {spec.core.cation, spec.precursor.center}
    inorganic = nx.Graph()
    inorganic.add_edges_from(
        (left, right)
        for left, right in state.graph.edges
        if state.atoms[left].symbol != ligand
        and state.atoms[right].symbol != ligand
    )
    ring_nodes: Set[int] = set()
    for cycle in nx.cycle_basis(inorganic):
        if len(cycle) == 6:
            ring_nodes.update(int(index) for index in cycle)
    variable_ids = ligand_ids + [
        index for index, atom in enumerate(state.atoms)
        if atom.symbol in cations | {spec.core.anion}
        and index not in ring_nodes
    ]
    variable_ids = list(dict.fromkeys(variable_ids))
    variable = np.asarray([initial[index] for index in variable_ids], dtype=float)
    variable_pos = {atom: offset for offset, atom in enumerate(variable_ids)}

    def point(index: int, xyz: np.ndarray) -> FloatArray:
        offset = variable_pos.get(index)
        if offset is None:
            return initial[index]
        return xyz[offset]

    def angle_deg(a: FloatArray, b: FloatArray, c: FloatArray) -> float:
        left = a - b
        right = c - b
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator < 1.0e-12:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(
            float(np.dot(left, right)) / denominator, -1.0, 1.0
        ))))

    bond_constraints: List[Tuple[int, int, float]] = []
    for left, right in state.graph.edges:
        if left not in variable_pos and right not in variable_pos:
            continue
        bond_constraints.append((int(left), int(right), _molecular_bond_length(
            state, pack, spec, left, right, degrees
        )))

    # (left, center, right, reference angle, weight, angular scale).
    # Bridge angles are soft but receive a stronger regularizer than ordinary
    # open-chain angles.  CN2 cations remain the most flexible skeleton hinge.
    angle_terms: List[Tuple[int, int, int, float, float, float]] = []
    for center in range(len(state.atoms)):
        neighbors = list(state.graph.neighbors(center))
        if len(neighbors) < 2:
            continue
        for left, right in combinations(neighbors, 2):
            if center in ring_nodes and left in ring_nodes and right in ring_nodes:
                continue
            bridge_involved = (
                state.atoms[left].symbol == ligand
                or state.atoms[right].symbol == ligand
                or state.atoms[center].symbol == ligand
            )
            if bridge_involved:
                weight, scale = 8.0, 5.0
            elif state.atoms[center].symbol in cations and len(neighbors) == 2:
                weight, scale = 1.0, 12.0
            else:
                weight, scale = 0.5, 10.0
            angle_terms.append((
                int(left), int(center), int(right),
                angle_deg(initial[left], initial[center], initial[right]),
                weight, scale,
            ))

    bonded = {
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in state.graph.edges
    }
    repulsive_pairs: List[Tuple[int, int, float, float]] = []
    for left in range(len(state.atoms)):
        for right in range(left + 1, len(state.atoms)):
            if (left, right) in bonded:
                continue
            if left not in variable_pos and right not in variable_pos:
                continue
            pair = pair_key(state.atoms[left].symbol, state.atoms[right].symbol)
            rule = spec.graph_rules.pair_rules.get(pair)
            soft_rule = pack.one_four_rule(pair)
            hard_floor = float(rule.min_distance or 0.0) if rule else 0.0
            soft_floor = float(soft_rule.get("soft_min_A") or 0.0)
            # A modest buffer above the hard floor gives the optimizer a
            # repulsive shoulder without changing the final hard criterion.
            target = max(soft_floor, hard_floor + 0.20)
            if target > 0.0:
                repulsive_pairs.append((left, right, target, 0.12))

    # This is the same scalar energy used by the old SLSQP path, represented as
    # residuals so least_squares can form a sparse numerical Jacobian.  Bond
    # residuals are deliberately much stiffer than the soft angle/repulsion
    # terms; the final exact audit remains authoritative.
    variable_flat = variable.reshape(-1)
    trace = _molecular_relax_trace_enabled()
    evaluation = [0]

    def trace_cost(cost: float) -> None:
        if not trace:
            return
        message = f"eval={evaluation[0]} cost={cost:.6g}"
        if log is not None:
            log(message)
        else:
            print(f"[molecular-relax] {message}")

    # Loop invariants are hoisted out of ``residual``: the finite-difference
    # Jacobian re-evaluates it once per column group, so per-term Python loops
    # and scalar ``np.linalg.norm`` calls dominate the solve otherwise.
    variable_index = np.array(variable_ids, dtype=int)
    bond_left_a = np.array([l for l, _, _ in bond_constraints], dtype=int)
    bond_right_a = np.array([r for _, r, _ in bond_constraints], dtype=int)
    bond_target_a = np.array([t for _, _, t in bond_constraints], dtype=float)
    bond_tol_a = np.array(
        [
            WELL_BAND_FRACTION
            * _molecular_bond_tolerance(state, pack, spec, l, r, degrees)
            for l, r, _ in bond_constraints
        ],
        dtype=float,
    )
    rep_left_a = np.array([l for l, _, _, _ in repulsive_pairs], dtype=int)
    rep_right_a = np.array([r for _, r, _, _ in repulsive_pairs], dtype=int)
    rep_target_a = np.array([t for _, _, t, _ in repulsive_pairs], dtype=float)
    rep_width_a = np.array([w for _, _, _, w in repulsive_pairs], dtype=float)
    ang_left_a = np.array([l for l, _, _, _, _, _ in angle_terms], dtype=int)
    ang_center_a = np.array([c for _, c, _, _, _, _ in angle_terms], dtype=int)
    ang_right_a = np.array([r for _, _, r, _, _, _ in angle_terms], dtype=int)
    ang_target_a = np.array([t for _, _, _, t, _, _ in angle_terms], dtype=float)
    ang_gain_a = np.array(
        [sqrt(w) / s for _, _, _, _, w, s in angle_terms], dtype=float
    )
    # The audit rejects on impropers and hard centre angles, so the repair has
    # to be able to move them; without these rows an ``improper:`` failure was
    # unrepairable at any budget.
    improper_terms, hard_angle_terms = _audited_local_terms(state, pack, spec)
    improper_index = np.array(
        [[c, a, b, d] for c, a, b, d, _ in improper_terms], dtype=int
    ).reshape(-1, 4)
    improper_target = np.array([t for *_, t in improper_terms], dtype=float)
    hard_index = np.array(
        [[l, c, r] for l, c, r, _t, _b, _g in hard_angle_terms], dtype=int
    ).reshape(-1, 3)
    hard_target = np.array(
        [t for _l, _c, _r, t, _b, _g in hard_angle_terms], dtype=float
    )
    hard_band = np.array(
        [b for _l, _c, _r, _t, b, _g in hard_angle_terms], dtype=float
    )
    # Alternative modes of one angle share a group id; the residual reduces
    # each group to its smallest excess, so the row count and the residual
    # length differ whenever a multi-modal centre is present.
    hard_group = np.array(
        [g for _l, _c, _r, _t, _b, g in hard_angle_terms], dtype=int
    )
    _grp = {}
    hard_group = np.array(
        [_grp.setdefault(g, len(_grp)) for g in hard_group], dtype=int
    )
    n_hard_rows = int(hard_group.max()) + 1 if hard_group.size else 0
    n_tether = len(variable_flat)
    stop_bond = n_tether + len(bond_constraints)
    stop_rep = stop_bond + len(repulsive_pairs)
    stop_angle = stop_rep + len(angle_terms)
    stop_improper = stop_angle + len(improper_terms)
    row_count = stop_improper + n_hard_rows
    values_buf = np.empty(row_count, dtype=float)
    work_full = initial.copy()

    def residual(flat: np.ndarray) -> np.ndarray:
        work_full[variable_index] = flat.reshape((-1, 3))
        xyz = work_full
        values_buf[:n_tether] = 0.01 * (flat - variable_flat)
        if bond_left_a.size:
            values_buf[n_tether:stop_bond] = _band_excess(
                np.linalg.norm(xyz[bond_left_a] - xyz[bond_right_a], axis=1),
                bond_target_a,
                bond_tol_a,
            ) / BOND_WELL_SCALE_A
        if rep_left_a.size:
            distance = np.linalg.norm(xyz[rep_left_a] - xyz[rep_right_a], axis=1)
            values_buf[stop_bond:stop_rep] = np.expm1(
                np.clip((rep_target_a - distance) / rep_width_a, 0.0, 30.0)
            )
        if ang_left_a.size:
            left_v = xyz[ang_left_a] - xyz[ang_center_a]
            right_v = xyz[ang_right_a] - xyz[ang_center_a]
            den = np.linalg.norm(left_v, axis=1) * np.linalg.norm(right_v, axis=1)
            degenerate = den < 1.0e-12
            cosine = np.einsum("ij,ij->i", left_v, right_v) / np.where(
                degenerate, 1.0, den
            )
            actual = np.where(
                degenerate,
                0.0,
                np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))),
            )
            values_buf[stop_rep:stop_angle] = ang_gain_a * (actual - ang_target_a)
        improper_out, hard_out = _local_term_residuals(
            xyz,
            improper_index,
            improper_target,
            hard_index,
            hard_target,
            improper_scale=WELL_BAND_FRACTION * AUDIT_IMPROPER_TOLERANCE_DEG,
            hard_scale=WELL_BAND_FRACTION * hard_band,
            hard_group=hard_group,
        )
        values_buf[stop_angle:stop_improper] = improper_out
        values_buf[stop_improper:] = hard_out
        result = values_buf.copy()
        evaluation[0] += 1
        if evaluation[0] == 1 or evaluation[0] % 25 == 0:
            trace_cost(0.5 * float(np.dot(result, result)))
        return result

    # Each residual depends on only the variables belonging to its local atom
    # set.  Supplying this pattern avoids perturbing every Cartesian coordinate
    # for every finite-difference column, which is the main speedup over the
    # former dense SLSQP objective/constraint pair.  ``lm`` is a dense solver
    # and is passed no ``jac_sparsity``, so for small systems the pattern is
    # not built at all.
    solver_name = "lm" if len(variable_flat) <= 96 else "trf-sparse"
    sparsity = None
    if solver_name != "lm":
        sparse_rows: List[int] = list(range(len(variable_flat)))
        sparse_cols: List[int] = list(range(len(variable_flat)))
        row = len(variable_flat)

        def mark(row_index: int, atom_ids: Iterable[int]) -> None:
            for atom in atom_ids:
                offset = variable_pos.get(int(atom))
                if offset is not None:
                    sparse_rows.extend((row_index, row_index, row_index))
                    sparse_cols.extend(
                        (3 * offset, 3 * offset + 1, 3 * offset + 2)
                    )

        for left, right, _target in bond_constraints:
            mark(row, (left, right))
            row += 1
        for left, right, _target, _width in repulsive_pairs:
            mark(row, (left, right))
            row += 1
        for left, center, right, _target, _weight, _scale in angle_terms:
            mark(row, (left, center, right))
            row += 1
        for center, first, second, third, _target in improper_terms:
            mark(row, (center, first, second, third))
            row += 1
        for left, center, right, _target, _band in hard_angle_terms:
            mark(row, (left, center, right))
            row += 1
        sparsity = coo_matrix(
            (np.ones(len(sparse_rows), dtype=int), (sparse_rows, sparse_cols)),
            shape=(row_count, len(variable_flat)),
        )

    initial_residual = residual(variable_flat)
    initial_cost = 0.5 * float(np.dot(initial_residual, initial_residual))
    if log is not None:
        log(
            f"start solver={solver_name} vars={len(variable_ids)} "
            f"residuals={len(initial_residual)} cost={initial_cost:.6g}"
        )
    if initial_cost > RELAX_INITIAL_COST_LIMIT:
        if stats is not None:
            stats["nfev"] = 0.0
            stats["initial_cost"] = initial_cost
            stats["final_cost"] = initial_cost
            stats["max_bond_error"] = float("inf")
            stats["success"] = 0.0
        if log is not None:
            log(
                f"skip solver={solver_name} initial_cost={initial_cost:.6g} "
                f">{RELAX_INITIAL_COST_LIMIT:.6g}"
            )
        return None
    try:
        if solver_name == "lm":
            result = least_squares(
                residual,
                variable_flat,
                x_scale="jac",
                method="lm",
                max_nfev=40,
                ftol=1.0e-7,
                xtol=1.0e-7,
                gtol=1.0e-7,
                verbose=2 if trace else 0,
            )
        else:
            result = least_squares(
                residual,
                variable_flat,
                jac_sparsity=sparsity.tocsr(),
                x_scale="jac",
                method="trf",
                tr_solver="lsmr",
                max_nfev=40,
                ftol=1.0e-7,
                xtol=1.0e-7,
                gtol=1.0e-7,
                verbose=2 if trace else 0,
            )
    except Exception as exc:  # noqa: BLE001
        if log is not None:
            log(
                f"failed solver={solver_name} error={type(exc).__name__}:"
                f"{exc}"
            )
        return None
    if not np.all(np.isfinite(result.x)):
        if log is not None:
            log(f"failed solver={solver_name} nonfinite_result")
        return None
    fitted = initial.copy()
    fitted[variable_ids] = result.x.reshape((-1, 3))
    max_bond_error = max(
        (
            abs(
                float(np.linalg.norm(fitted[right] - fitted[left]))
                - target
            )
            for left, right, target in bond_constraints
        ),
        default=0.0,
    )
    final_cost = float(result.cost)
    if stats is not None:
        stats["nfev"] = float(
            getattr(result, "nfev", getattr(result, "nit", 0))
        )
        stats["initial_cost"] = initial_cost
        stats["final_cost"] = final_cost
        stats["max_bond_error"] = max_bond_error
        stats["success"] = float(bool(getattr(result, "success", False)))
    if log is not None:
        log(
            f"done solver={solver_name} status={result.status} "
            f"nfev={getattr(result, 'nfev', 0)} cost={final_cost:.6g} "
            f"bond_err={max_bond_error:.6g}"
        )
    # The repair is only useful if its result can actually be accepted, so this
    # gate tracks the audit band rather than sitting orders of magnitude below
    # it -- at 1e-4 A the solver discarded every result it produced.
    if max_bond_error > pack.audit_bond_tolerance_A:
        return None
    return fitted


def _relaxed_complete_geometry(
    state: _State,
    pack: GeometryPack,
    spec: NucleationSpec,
    inorganic: Tuple[FloatArray, List[bool]],
    *,
    log: Optional[Callable[[str], None]] = None,
    stats: Optional[Dict[str, float]] = None,
) -> FloatArray:
    """Construct a complete graph when a frozen bridge frame is impossible.

    This is a bounded local distance-geometry repair, not a conformer search:
    the graph and all hard contacts stay fixed, while the initial inorganic
    frame is used as an anchor and every graph bond is fitted simultaneously.
    It is intentionally invoked only after the deterministic exact placer has
    failed on a bridge motif.
    """

    frame, placed = inorganic
    n = len(state.atoms)
    degrees = [state.graph.degree[i] for i in range(n)]
    cations = {spec.core.cation, spec.precursor.center}
    anion = spec.core.anion
    ligand = spec.precursor.ligand
    rank = _canonical_ranks(state, list(range(n)), degrees)
    coords = np.zeros((n, 3), dtype=float)
    coords[:] = np.asarray(frame, dtype=float)
    frame_indices = [
        i
        for i, atom in enumerate(state.atoms)
        if atom.symbol in cations or atom.symbol == anion
    ]
    if any(not placed[i] for i in frame_indices):
        raise ExactEmbeddingError(["relaxed_frame_missing_inorganic"])

    def bond_r(i: int, j: int) -> float:
        return _molecular_bond_length(state, pack, spec, i, j, degrees)

    _terminals, bridges_by_pair, multi_host = _ligand_groups(
        state, spec, rank
    )
    # Do not spend a nonlinear solve on a bridge whose host separation is well
    # outside the maximum movement allowed by the frame anchor.  This is a
    # conservative *solver guard*, not a new graph rule: a pair that is only a
    # little too long can still be repaired below, while a grossly long pair
    # is guaranteed to fail the Cd--Cl sphere constraints.
    for (left, right), cl_list in bridges_by_pair.items():
        if not cl_list:
            continue
        required = max(
            bond_r(cl, left) + bond_r(cl, right) for cl in cl_list
        )
        separation = float(np.linalg.norm(coords[left] - coords[right]))
        if separation > required + 0.90:
            raise ExactEmbeddingError(
                [
                    f"bridge_hosts_too_far:{left}-{right}:"
                    f"{separation:.3f}>{required + 0.90:.3f}"
                ],
                coordinates=coords,
            )
    # Deterministic bridge seeds.  The optimizer below is allowed to move both
    # hosts and ligands, so a centroid is preferable to an arbitrary torsion.
    for (left, right), cl_list in bridges_by_pair.items():
        midpoint = (coords[left] + coords[right]) / 2.0
        for cl in sorted(cl_list, key=rank.__getitem__):
            try:
                coords[cl] = _relaxed_sphere_position(
                    [coords[left], coords[right]],
                    [bond_r(cl, left), bond_r(cl, right)],
                    preferred=midpoint,
                    tolerance=1.0e9,
                )
            except ExactEmbeddingError:
                coords[cl] = midpoint

    for cl, hosts in multi_host:
        if len(hosts) != 3:
            raise ExactEmbeddingError(
                [f"unsupported_multi_host_bridge:{cl}:{len(hosts)}"]
            )
        midpoint = np.mean([coords[host] for host in hosts], axis=0)
        try:
            coords[cl] = _relaxed_sphere_position(
                [coords[host] for host in hosts],
                [bond_r(cl, host) for host in hosts],
                preferred=midpoint,
                tolerance=1.0e9,
            )
        except ExactEmbeddingError:
            coords[cl] = midpoint

    occupied = [coords[host] for host in frame_indices]
    for host, cl_list in _terminals.items():
        for cl in sorted(cl_list, key=rank.__getitem__):
            direction = _free_tetrahedral_direction(
                coords[host], occupied, angle_deg=109.5
            )
            coords[cl] = coords[host] + bond_r(cl, host) * direction
            occupied.append(coords[cl])

    bonded = {
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in state.graph.edges
    }
    bond_targets = {
        (left, right): bond_r(left, right)
        for left, right in bonded
    }
    nonbond_floors: List[Tuple[int, int, float]] = []
    for left in range(n):
        for right in range(left + 1, n):
            if (left, right) in bonded:
                continue
            rule = spec.graph_rules.pair_rules.get(
                pair_key(state.atoms[left].symbol, state.atoms[right].symbol)
            )
            if rule is not None and rule.min_distance is not None:
                nonbond_floors.append((left, right, float(rule.min_distance)))

    trace = _molecular_relax_trace_enabled()
    evaluation = [0]

    def trace_cost(cost: float) -> None:
        if not trace:
            return
        message = f"eval={evaluation[0]} cost={cost:.6g}"
        if log is not None:
            log(message)
        else:
            print(f"[molecular-relax] {message}")

    def residual(flat: FloatArray) -> FloatArray:
        xyz = np.asarray(flat, dtype=float).reshape((n, 3))
        values: List[float] = []
        for left, right in bonded:
            values.append(
                (
                    float(np.linalg.norm(xyz[right] - xyz[left]))
                    - bond_targets[(left, right)]
                )
                / 0.025
            )
        # Keep the inorganic frame recognizable, but allow bridge host
        # triangles to move enough to become realizable.
        for index in frame_indices:
            values.extend(((xyz[index] - frame[index]) / 0.45).tolist())
        for left, right, floor in nonbond_floors:
            values.append(
                min(
                    0.0,
                    float(np.linalg.norm(xyz[right] - xyz[left]))
                    - (floor + 0.02),
                )
                / 0.01
            )
        result = np.asarray(values, dtype=float)
        evaluation[0] += 1
        if evaluation[0] == 1 or evaluation[0] % 25 == 0:
            trace_cost(0.5 * float(np.dot(result, result)))
        return result

    variable_flat = coords.reshape(-1)
    initial_residual = residual(variable_flat)
    initial_cost = 0.5 * float(np.dot(initial_residual, initial_residual))
    # Locality of each distance residual makes finite-difference Jacobians
    # sparse for larger molecules.  Small bridge repairs use dense LM below;
    # building this matrix is then unnecessary overhead, but keeping it here
    # is cheap and lets the large path reuse the same residual definition.
    sparsity = lil_matrix(
        (len(bonded) + 3 * len(frame_indices) + len(nonbond_floors), 3 * n),
        dtype=int,
    )
    row = 0

    def mark(row_index: int, atom_ids: Iterable[int]) -> None:
        for atom in atom_ids:
            sparsity[row_index, 3 * int(atom):3 * int(atom) + 3] = 1

    for left, right in bonded:
        mark(row, (left, right))
        row += 1
    for index in frame_indices:
        for component in range(3):
            sparsity[row + component, 3 * int(index) + component] = 1
        row += 3
    for left, right, _floor in nonbond_floors:
        mark(row, (left, right))
        row += 1
    if log is not None:
        log(
            f"start solver=least_squares_bridge vars={n} "
            f"residuals={len(initial_residual)} "
            f"cost={initial_cost:.6g}"
        )
    use_lm = len(variable_flat) <= 96
    solver_name = "lm" if use_lm else "trf-sparse"
    solver_kwargs: Dict[str, object] = {
        "x_scale": "jac",
        "method": "lm" if use_lm else "trf",
        "max_nfev": 8 if use_lm else 8,
        "ftol": 1.0e-7,
        "xtol": 1.0e-7,
        "gtol": 1.0e-7,
        "verbose": 2 if trace else 0,
    }
    if not use_lm:
        solver_kwargs.update(
            jac_sparsity=sparsity.tocsr(),
            tr_solver="lsmr",
        )
    result = least_squares(residual, variable_flat, **solver_kwargs)
    fitted = np.asarray(result.x, dtype=float).reshape((n, 3))
    max_bond_error = max(
        (
            abs(
                float(np.linalg.norm(fitted[right] - fitted[left]))
                - bond_r(left, right)
            )
            for left, right in bonded
        ),
        default=0.0,
    )
    if stats is not None:
        stats["nfev"] = float(getattr(result, "nfev", 0))
        stats["initial_cost"] = initial_cost
        stats["final_cost"] = float(result.cost)
        stats["max_bond_error"] = max_bond_error
        stats["success"] = float(bool(getattr(result, "success", False)))
    if log is not None:
        log(
            f"done solver={solver_name} status={result.status} "
            f"nfev={getattr(result, 'nfev', 0)} cost={float(result.cost):.6g} "
            f"bond_err={max_bond_error:.6g}"
        )
    if max_bond_error > RELAXED_BRIDGE_BOND_TOLERANCE_A:
        raise ExactEmbeddingError(
            [
                "relaxed_geometry_bond_residual:"
                f"{max_bond_error:.4f}>{RELAXED_BRIDGE_BOND_TOLERANCE_A:.4f}"
            ],
            coordinates=fitted,
        )
    return fitted


def _forbidden_cdse_cn_pair_violations(
    state: _State, spec: NucleationSpec
) -> List[str]:
    """Reject graphs whose final Cd–Se CN pairs are pack-forbidden."""

    forbidden = set(spec.graph_rules.forbid_cdse_cn_pairs)
    if not forbidden:
        return []
    cations = {spec.core.cation, spec.precursor.center}
    anion = spec.core.anion
    degrees = [state.graph.degree[i] for i in range(len(state.atoms))]
    for left, right in state.graph.edges:
        left_symbol = state.atoms[left].symbol
        right_symbol = state.atoms[right].symbol
        if left_symbol in cations and right_symbol == anion:
            cd, se = left, right
        elif right_symbol in cations and left_symbol == anion:
            cd, se = right, left
        else:
            continue
        pair = (degrees[cd], degrees[se])
        if pair in forbidden:
            return [f"forbidden_cdse_cn_pair:Cd{pair[0]}-Se{pair[1]}"]
    return []


def min_bridged_host_cn_violations(
    state: _State, spec: NucleationSpec
) -> List[str]:
    """μ-Cl hosts must meet ``graph_rules.min_bridged_host_cn`` (final CN)."""

    minimum = int(spec.graph_rules.min_bridged_host_cn)
    if minimum <= 1:
        return []
    cations = {spec.core.cation, spec.precursor.center}
    ligand = spec.precursor.ligand
    degrees = [state.graph.degree[i] for i in range(len(state.atoms))]
    for atom in state.atoms:
        if atom.symbol != ligand or degrees[atom.atom_id] < 2:
            continue
        for host in state.graph.neighbors(atom.atom_id):
            if state.atoms[host].symbol not in cations:
                continue
            if degrees[host] < minimum:
                return [
                    f"min_bridged_host_cn:{host}:"
                    f"{degrees[host]}<{minimum}"
                ]
    return []


def mu3_host_bridge_overlap_violations(
    state: _State, spec: NucleationSpec
) -> List[str]:
    """A μ3 chloride cap may not share a Cd with another Cl bridge.

    Terminal chloride is allowed on the same Cd.  A second chloride with at
    least two Cd neighbours is a μ2/μ3 bridge and is forbidden by the hard
    motif rule.
    """

    if not getattr(spec.graph_rules, "forbid_mu3_host_bridge_overlap", False):
        return []
    cations = {spec.core.cation, spec.precursor.center}
    ligand = spec.precursor.ligand
    for cap in state.atoms:
        if cap.symbol != ligand:
            continue
        cap_hosts = [
            host
            for host in state.graph.neighbors(cap.atom_id)
            if state.atoms[host].symbol in cations
        ]
        if len(cap_hosts) < 3:
            continue
        for host in cap_hosts:
            for other in state.graph.neighbors(host):
                if other == cap.atom_id or state.atoms[other].symbol != ligand:
                    continue
                other_hosts = sum(
                    1
                    for neighbor in state.graph.neighbors(other)
                    if state.atoms[neighbor].symbol in cations
                )
                if other_hosts >= 2:
                    return [
                        f"mu3_host_bridge_overlap:{cap.atom_id}-"
                        f"{host}-{other}"
                    ]
    return []


def mono_se_dual_terminal_violations(
    state: _State, spec: NucleationSpec
) -> List[str]:
    """Mono-Se Cd must not carry two terminal Cl and zero bridges.

    Skeleton-aware construction default: Cd with exactly one Se and two Cl
    should be (1μ+1t) or (2μ), matching clean-DFT Cl2Se1 environments.
    """

    if not spec.graph_rules.forbid_mono_se_dual_terminal:
        return []
    cations = {spec.core.cation, spec.precursor.center}
    anion = spec.core.anion
    ligand = spec.precursor.ligand
    degrees = [state.graph.degree[i] for i in range(len(state.atoms))]
    for atom in state.atoms:
        if atom.symbol not in cations:
            continue
        center = atom.atom_id
        se_n = sum(
            1
            for j in state.graph.neighbors(center)
            if state.atoms[j].symbol == anion
        )
        if se_n != 1:
            continue
        cl_n = [
            j
            for j in state.graph.neighbors(center)
            if state.atoms[j].symbol == ligand
        ]
        if len(cl_n) != 2:
            continue
        if all(degrees[j] == 1 for j in cl_n):
            return [f"mono_se_dual_terminal:{center}"]
    return []


def shared_cd_pair_violations(
    state: _State, spec: NucleationSpec
) -> List[str]:
    """Reject two anion motifs sharing a cation pair they cannot both fit.

    An anion holds its cations at a fixed angle, so it *demands* a specific
    cation-cation separation.  Two anions bonded to the same two cations both
    demand one, and if the two cannot be reconciled inside their angle bands
    the graph has no 3D realisation -- a tetrahedral Se wants ~4.4 A between
    its Cd while a mu2 Cl bridge wants ~3.6 A.  The offending pairs are derived
    from the pack's own tables, so this tracks the angles rather than naming
    them.  Checked on the graph, before any coordinate work.
    """

    forbidden = set(spec.graph_rules.forbid_shared_cd_pair)
    if not forbidden:
        return []
    cations = {spec.core.cation, spec.precursor.center}
    anions = [
        atom.atom_id
        for atom in state.atoms
        if atom.symbol not in cations and state.graph.degree[atom.atom_id] >= 2
    ]
    for left, right in combinations(anions, 2):
        shared = [
            host
            for host in nx.common_neighbors(state.graph, left, right)
            if state.atoms[host].symbol in cations
        ]
        if len(shared) < 2:
            continue
        a = (state.atoms[left].symbol, int(state.graph.degree[left]))
        b = (state.atoms[right].symbol, int(state.graph.degree[right]))
        for first, second in ((a, b), (b, a)):
            if (first[0], first[1], second[0], second[1]) in forbidden:
                return [
                    f"incompatible_shared_hosts:"
                    f"{first[0]}{first[1]}-{second[0]}{second[1]}:"
                    f"{left}-{right}"
                ]
    return []


def molecular_decoration_rule_violations(
    state: _State, spec: NucleationSpec
) -> List[str]:
    """Pack construction defaults checked on the finished decorated graph."""

    from .molecular_rules import required_ring_violations

    return (
        mu3_host_bridge_overlap_violations(state, spec)
        or
        min_bridged_host_cn_violations(state, spec)
        or mono_se_dual_terminal_violations(state, spec)
        or _forbidden_cdse_cn_pair_violations(state, spec)
        or shared_cd_pair_violations(state, spec)
        # Rings a graph must contain are a whole-graph property, so unlike
        # ``min_ring_size`` (a skeleton invariant) this cannot be hoisted to
        # the skeleton stage -- ligand-containing macrocycles count.
        or required_ring_violations(state, spec)
    )


def _mu2_sphere_ok(
    pack: GeometryPack,
    host_a: FloatArray,
    host_b: FloatArray,
    radius_a: float,
    radius_b: float,
) -> bool:
    separation = float(np.linalg.norm(host_b - host_a))
    if separation < 1.0e-12:
        return False
    axial = (
        radius_a * radius_a - radius_b * radius_b + separation * separation
    ) / (2.0 * separation)
    return radius_a * radius_a - axial * axial >= -EXACT_BOND_TOLERANCE


def bridge_maximal_violations(
    state: _State,
    pack: GeometryPack,
    spec: NucleationSpec,
    frames: Sequence[Tuple[FloatArray, List[bool]]],
) -> List[str]:
    """Non-empty if a terminal can still form a sphere-feasible μ2 upgrade.

    A graph is bridge-maximal when no terminal Cl on host H admits a μ2 onto
    another cation U with free valence, under tabulated radii and some clean
    frame of the *current* coordination numbers (after the upgrade on U).
    """

    if not frames:
        return []
    cations = {spec.core.cation, spec.precursor.center}
    ligand = spec.precursor.ligand
    max_cd = int(spec.graph_rules.max_cn[spec.core.cation])
    bridge_cap = int(
        spec.graph_rules.max_shared_ligands_per_host_pair
        or spec.bridges_per_cd_pair
        or 0
    )
    degrees = [state.graph.degree[i] for i in range(len(state.atoms))]
    cd_ids = [
        atom.atom_id for atom in state.atoms if atom.symbol in cations
    ]
    cl_ids = [
        atom.atom_id for atom in state.atoms if atom.symbol == ligand
    ]

    pair_shared: Dict[Tuple[int, int], int] = {}
    for cl in cl_ids:
        hosts = sorted(
            j
            for j in state.graph.neighbors(cl)
            if state.atoms[j].symbol in cations
        )
        for left, right in combinations(hosts, 2):
            key = (left, right)
            pair_shared[key] = pair_shared.get(key, 0) + 1

    for cl in cl_ids:
        if degrees[cl] != 1:
            continue
        hosts = [
            j
            for j in state.graph.neighbors(cl)
            if state.atoms[j].symbol in cations
        ]
        if len(hosts) != 1:
            continue
        host = hosts[0]
        for other in cd_ids:
            if other == host or degrees[other] >= max_cd:
                continue
            pair = (min(host, other), max(host, other))
            if bridge_cap > 0 and pair_shared.get(pair, 0) >= bridge_cap:
                continue
            # After upgrade: Cl CN=2, other CN+1, host CN unchanged.
            cn_host = degrees[host]
            cn_other = degrees[other] + 1
            radius_host = pack.bond_length(
                "CdCl_bridge", cn_host, 2, default=2.40
            )
            radius_other = pack.bond_length(
                "CdCl_bridge", cn_other, 2, default=2.40
            )
            for frame, _placed in frames:
                if _mu2_sphere_ok(
                    pack,
                    frame[host],
                    frame[other],
                    radius_host,
                    radius_other,
                ):
                    return [
                        f"not_bridge_maximal:Cl{cl}:Cd{host}-Cd{other}"
                    ]
    return []


def annotate_molecular_state(
    state: _State,
    spec: NucleationSpec,
    coordinates: Optional[Sequence[Sequence[float]]] = None,
    *,
    closable_distance: Optional[float] = None,
) -> MolecularCollapseAnnotations:
    """Label a graph (and optional coords) with DFT-collapse risk features."""

    if closable_distance is None:
        closable_distance = float(
            spec.graph_rules.closable_terminal_cd2_distance
            or CLOSABLE_TERMINAL_CD2_A
        )
    graph = state.graph
    cations = {spec.core.cation, spec.precursor.center}
    anion = spec.core.anion
    ligand = spec.precursor.ligand
    max_cd = int(spec.graph_rules.max_cn[spec.core.cation])
    bridge_cap = int(
        spec.graph_rules.max_shared_ligands_per_host_pair
        or spec.bridges_per_cd_pair
        or 0
    )

    cd_ids = [
        atom.atom_id for atom in state.atoms if atom.symbol in cations
    ]
    cl_ids = [
        atom.atom_id for atom in state.atoms if atom.symbol == ligand
    ]
    degrees = [graph.degree[i] for i in range(len(state.atoms))]

    n_cd2 = sum(1 for i in cd_ids if degrees[i] == 2)
    n_cd3 = sum(1 for i in cd_ids if degrees[i] == 3)
    n_cd4 = sum(1 for i in cd_ids if degrees[i] == 4)
    mean_cd_cn = (
        float(sum(degrees[i] for i in cd_ids) / len(cd_ids)) if cd_ids else 0.0
    )

    n_terminal = sum(1 for i in cl_ids if degrees[i] == 1)
    n_mu2 = sum(1 for i in cl_ids if degrees[i] == 2)
    n_mu3 = sum(1 for i in cl_ids if degrees[i] >= 3)

    # Shared ligand counts per host pair (for cap checks).
    pair_shared: Dict[Tuple[int, int], int] = {}
    for cl in cl_ids:
        hosts = sorted(
            j
            for j in graph.neighbors(cl)
            if state.atoms[j].symbol in cations
        )
        for left, right in combinations(hosts, 2):
            key = (left, right)
            pair_shared[key] = pair_shared.get(key, 0) + 1

    unsaturated = 0
    for cl in cl_ids:
        if degrees[cl] != 1:
            continue
        hosts = [
            j
            for j in graph.neighbors(cl)
            if state.atoms[j].symbol in cations
        ]
        if len(hosts) != 1:
            continue
        host = hosts[0]
        for other in cd_ids:
            if other == host or degrees[other] >= max_cd:
                continue
            pair = (min(host, other), max(host, other))
            if bridge_cap > 0 and pair_shared.get(pair, 0) >= bridge_cap:
                continue
            unsaturated += 1

    cdse_pairs: Counter[str] = Counter()
    for left, right in graph.edges:
        left_symbol = state.atoms[left].symbol
        right_symbol = state.atoms[right].symbol
        if left_symbol in cations and right_symbol == anion:
            cd, se = left, right
        elif right_symbol in cations and left_symbol == anion:
            cd, se = right, left
        else:
            continue
        cdse_pairs[f"{degrees[cd]}-{degrees[se]}"] += 1
    cdse_text = ",".join(
        f"{key}:{count}" for key, count in sorted(cdse_pairs.items())
    )

    cd2_sigs: Counter[str] = Counter()
    for cd in cd_ids:
        if degrees[cd] == 2:
            cd2_sigs[_neighbor_signature(state, cd)] += 1
    cd2_sig_text = ",".join(
        f"{key}:{count}" for key, count in sorted(cd2_sigs.items())
    )

    closable = 0
    cl2se1_near = 0
    if coordinates is not None:
        coords = np.asarray(coordinates, dtype=float)
        if coords.shape == (len(state.atoms), 3):
            # Hosts that are Cl2Se1 with two terminal Cl.
            cl2se1_hosts = set()
            for cd in cd_ids:
                if degrees[cd] != 3:
                    continue
                if _neighbor_signature(state, cd) != "Cl2Se1":
                    continue
                cl_n = [
                    j
                    for j in graph.neighbors(cd)
                    if state.atoms[j].symbol == ligand
                ]
                if len(cl_n) == 2 and all(degrees[j] == 1 for j in cl_n):
                    cl2se1_hosts.add(cd)

            cd2_set = {i for i in cd_ids if degrees[i] == 2}
            for cl in cl_ids:
                if degrees[cl] != 1:
                    continue
                hosts = [
                    j
                    for j in graph.neighbors(cl)
                    if state.atoms[j].symbol in cations
                ]
                if len(hosts) != 1:
                    continue
                host = hosts[0]
                for other in cd2_set:
                    if other == host or graph.has_edge(cl, other):
                        continue
                    dist = float(np.linalg.norm(coords[cl] - coords[other]))
                    if dist > closable_distance:
                        continue
                    closable += 1
                    if host in cl2se1_hosts:
                        cl2se1_near += 1

    return MolecularCollapseAnnotations(
        n_cd2=n_cd2,
        n_cd3=n_cd3,
        n_cd4=n_cd4,
        mean_cd_cn=mean_cd_cn,
        n_terminal_cl=n_terminal,
        n_mu2_cl=n_mu2,
        n_mu3_cl=n_mu3,
        n_unsaturated_bridge_candidates=unsaturated,
        n_closable_terminal_cd2=closable,
        n_cl2se1_near_cd2=cl2se1_near,
        cdse_cn_pairs=cdse_text,
        cd2_signatures=cd2_sig_text,
    )


def _format_cn_list(values: Sequence[int]) -> str:
    return "[" + ",".join(str(int(v)) for v in values) + "]"


def _failure_slug(value: str, *, limit: int = 80) -> str:
    """Filesystem-safe short label for a failure reason."""

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return (slug or "failure")[:limit]


def skeleton_xyz_basename(
    k: int,
    p: int,
    skeleton_index: int,
    *,
    conformation: Optional[str] = None,
) -> str:
    """``skeleton_k2_p6_001.xyz`` or ``…_001_chair.xyz`` for ring dumps."""

    base = f"skeleton_k{k}_p{p}_{skeleton_index:03d}"
    conf = (conformation or "").strip().lower()
    if conf in {"chair", "boat"}:
        return f"{base}_{conf}.xyz"
    return f"{base}.xyz"


def format_skeleton_edges(edges: Sequence[Tuple[int, int]]) -> str:
    """Serialize Cd–Se edges for ``skeletons.csv`` (``0-5;1-5;2-6``)."""

    pairs = sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in edges)
    return ";".join(f"{a}-{b}" for a, b in pairs)


def parse_skeleton_edges(text: str) -> Tuple[Tuple[int, int], ...]:
    """Parse ``format_skeleton_edges`` output."""

    raw = (text or "").strip().strip('"')
    if not raw:
        return ()
    out: List[Tuple[int, int]] = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        left, right = part.split("-", 1)
        a, b = int(left), int(right)
        out.append((min(a, b), max(a, b)))
    return tuple(sorted(out))


def load_skeleton_catalog(
    output_dir: str | Path,
    *,
    accepted_only: bool = True,
    require_edges: bool = True,
) -> Dict[Tuple[int, int], List[Tuple[Tuple[int, int], ...]]]:
    """Load inorganic edge sets from a prior skeleton dump.

    Reads ``output_dir/skeletons.csv`` (preferred) and falls back to
    ``k###/skeletons/skeletons.tsv`` rows that include an ``edges`` column.
    Returns ``{(k, p): [edge_tuple, ...]}`` in dump order.

    Older dumps without an ``edges`` column yield empty lists for those rows
    (re-enumeration is required for decoration).
    """

    root = Path(output_dir)
    catalog: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], ...]]] = {}
    sources: List[Path] = []
    global_csv = root / "skeletons.csv"
    if global_csv.is_file():
        sources.append(global_csv)
    for tsv in sorted(root.glob("k*/skeletons/skeletons.tsv")):
        sources.append(tsv)
    if not sources:
        return catalog

    import csv as _csv

    for path in sources:
        with path.open(encoding="utf-8", newline="") as fh:
            reader = _csv.DictReader(fh)
            if reader.fieldnames is None:
                continue
            fields = {name.strip() for name in reader.fieldnames}
            if "k" not in fields or "p" not in fields:
                continue
            has_edges = "edges" in fields
            for row in reader:
                try:
                    k = int(row["k"])
                    p = int(row["p"])
                except (KeyError, ValueError):
                    continue
                status = str(row.get("status", "")).strip().lower()
                if accepted_only and status and status != "accepted":
                    continue
                try:
                    has_ring = int(str(row.get("n_six_rings", "0") or "0")) > 0
                except ValueError:
                    has_ring = False
                if has_ring and not str(row.get("forced_rings", "") or "").strip():
                    continue
                edges_raw = row.get("edges", "") if has_edges else ""
                edges = parse_skeleton_edges(edges_raw) if edges_raw else ()
                if require_edges and not edges:
                    continue
                catalog.setdefault((k, p), []).append(edges)
    return catalog


def bins_present_in_skeleton_dump(
    output_dir: str | Path,
) -> Set[Tuple[int, int]]:
    """Return ``(k, p)`` pairs that already appear in a skeleton dump table."""

    root = Path(output_dir)
    found: Set[Tuple[int, int]] = set()
    import csv as _csv

    paths = [root / "skeletons.csv"] + sorted(
        root.glob("k*/skeletons/skeletons.tsv")
    )
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as fh:
            reader = _csv.DictReader(fh)
            if not reader.fieldnames or "k" not in reader.fieldnames:
                continue
            for row in reader:
                try:
                    found.add((int(row["k"]), int(row["p"])))
                except (KeyError, ValueError):
                    continue
    return found


def write_skeleton_xyz(
    path: Path,
    *,
    k: int,
    p: int,
    skeleton_index: int,
    status: str,
    cd_cn: Sequence[int],
    se_cn: Sequence[int],
    n_edges: int,
    symbols: Sequence[str],
    coordinates: Sequence[Sequence[float]],
    reason: str = "",
) -> None:
    """Write one skeleton XYZ with CN metadata in the comment line."""

    path.parent.mkdir(parents=True, exist_ok=True)
    write_xyz(str(path), list(symbols), np.asarray(coordinates, dtype=float))
    cd_txt = _format_cn_list(cd_cn)
    se_txt = _format_cn_list(se_cn)
    try:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if len(lines) >= 2:
            comment = (
                f"k={k} p={p} skeleton={skeleton_index} status={status} "
                f"Cd{cd_txt} Se{se_txt} n_edges={n_edges}"
            )
            if reason:
                comment += f" reason={reason.replace(' ', '_')}"
            lines[1] = comment
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass


def write_skeleton_inventory(
    bin_res: MolecularBinResult,
    bin_dir: Path,
) -> None:
    """Write skeleton XYZ + tables under ``bin_dir/skeletons/`` (legacy layout)."""

    skel_dir = bin_dir / "skeletons"
    skel_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "skeleton_index,status,n_edges,cd_cn,se_cn,reason,xyz"
    ]
    md = [
        f"# Skeletons k={bin_res.k} p={bin_res.p}\n\n",
        "| # | status | n_edges | Cd CN | Se CN | reason |\n",
        "|--:|--------|-------:|-------|-------|--------|\n",
    ]
    k, p = bin_res.k, bin_res.p
    for rec in bin_res.skeleton_records:
        cd_txt = _format_cn_list(rec.cd_cn)
        se_txt = _format_cn_list(rec.se_cn)
        xyz_name = ""
        if rec.coordinates is not None and rec.symbols:
            xyz_name = skeleton_xyz_basename(k, p, rec.skeleton_index)
            write_skeleton_xyz(
                skel_dir / xyz_name,
                k=k,
                p=p,
                skeleton_index=rec.skeleton_index,
                status=rec.status,
                cd_cn=rec.cd_cn,
                se_cn=rec.se_cn,
                n_edges=rec.n_edges,
                symbols=rec.symbols,
                coordinates=rec.coordinates,
                reason=rec.reason,
            )
        reason = rec.reason.replace(",", ";")
        lines.append(
            f"{rec.skeleton_index},{rec.status},{rec.n_edges},"
            f"\"{cd_txt}\",\"{se_txt}\",{reason},{xyz_name}"
        )
        md.append(
            f"| {rec.skeleton_index} | {rec.status} | {rec.n_edges} | "
            f"`{cd_txt}` | `{se_txt}` | {reason or '—'} |\n"
        )
    (skel_dir / "skeletons.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (skel_dir / "skeletons.md").write_text("".join(md), encoding="utf-8")


_SKELETON_CSV_HEADER = (
    "k,p,skeleton_index,status,n_edges,cd_cn,se_cn,reason,xyz,edges,"
    "n_six_rings,skeleton_mode,forced_rings"
)


def format_forced_rings(rings: Sequence[Sequence[int]]) -> str:
    return "|".join("-".join(str(int(atom)) for atom in ring) for ring in rings)


def dump_skeletons_upfront(
    spec: NucleationSpec,
    output_dir: str | Path,
    *,
    pack: Optional[GeometryPack] = None,
    kmin: int = 1,
    kmax: Optional[int] = None,
    pmin: int = 0,
    pmax: Optional[int] = None,
    max_skeletons: int = 2000,
    extra_skeleton_edges: Optional[int] = None,
    embed: bool = True,
    progress: Optional[ProgressCallback] = None,
    resume: bool = False,
) -> Path:
    """Enumerate and write the graph-only skeleton catalog before decoration.

    Layout (grouped by k)::

        output_dir/
          skeletons.csv          # includes ``edges`` for resume/decoration
          k002/
            skeletons/
              skeletons.tsv
              skeletons.md

    When ``resume`` is true, ``(k, p)`` bins already present in
    ``skeletons.csv`` (or per-k tsv) are skipped so a killed dump can continue.
    The global CSV is rewritten after every ``(k, p)`` so mid-run kills keep
    partial progress.  ``embed`` is retained for CLI compatibility and is not
    used at this stage.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    check_spec = _molecular_check_spec(spec)

    k_lo = int(kmin)
    k_hi = int(kmax if kmax is not None else spec.kmax)
    p_lo = int(pmin)
    global_lines: List[str] = [_SKELETON_CSV_HEADER]
    # Preserve prior rows when resuming so we do not drop finished bins.
    prior_bins = bins_present_in_skeleton_dump(root) if resume else set()
    if resume and (root / "skeletons.csv").is_file():
        # Keep existing lines (including header swap to new schema if needed).
        import csv as _csv

        with (root / "skeletons.csv").open(encoding="utf-8", newline="") as fh:
            reader = _csv.DictReader(fh)
            if reader.fieldnames and "k" in reader.fieldnames:
                for row in reader:
                    try:
                        k_r, p_r = int(row["k"]), int(row["p"])
                    except (KeyError, ValueError):
                        continue
                    edges = row.get("edges", "") or ""
                    n_six = row.get("n_six_rings", "") or ""
                    row_out = (
                        f"{k_r},{p_r},{row.get('skeleton_index', '')},"
                        f"{row.get('status', '')},{row.get('n_edges', '')},"
                        f"\"{row.get('cd_cn', '').strip(chr(34))}\","
                        f"\"{row.get('se_cn', '').strip(chr(34))}\","
                        f"{row.get('reason', '').replace(',', ';')},"
                        f"{row.get('xyz', '')},{edges},{n_six},"
                        f"{row.get('skeleton_mode', '')},"
                        f"{row.get('forced_rings', '')}"
                    )
                    global_lines.append(row_out)

    by_k: Dict[int, List[str]] = {}
    by_k_md: Dict[int, List[str]] = {}

    def flush_global() -> None:
        (root / "skeletons.csv").write_text(
            "\n".join(global_lines) + "\n", encoding="utf-8"
        )

    if progress is not None:
        progress(
            f"[molecular] SKELETON DUMP | writing all Cd–Se skeletons to "
            f"{root} before decoration..."
            + (" (resume: skip finished bins)" if resume else "")
        )

    for k in range(k_lo, k_hi + 1):
        max_p, slot_info = resolve_molecular_max_p(
            check_spec,
            k,
            pmax,
            max_skeletons=max_skeletons,
            extra_skeleton_edges=extra_skeleton_edges,
        )
        if progress is not None:
            if slot_info is not None and slot_info.source == "slots":
                progress(
                    f"  k={k}: slot-based pmax={max_p} "
                    f"(max free Se on accepted p=0="
                    f"{slot_info.max_free_slots}, "
                    f"accepted={slot_info.n_p0_accepted}/"
                    f"{slot_info.n_p0_enumerated}, "
                    f"global={slot_info.global_bound})"
                )
            elif slot_info is not None:
                progress(
                    f"  k={k}: pmax={max_p} from global Se bound "
                    f"(no accepted p=0; enumerated "
                    f"{slot_info.n_p0_enumerated})"
                )
            else:
                progress(f"  k={k}: pmax={max_p} (user override)")
        k_dir = root / f"k{k:03d}" / "skeletons"
        k_dir.mkdir(parents=True, exist_ok=True)
        if not resume:
            for stale_xyz in k_dir.glob("skeleton_k*_p*_*.xyz"):
                stale_xyz.unlink()
        by_k[k] = [_SKELETON_CSV_HEADER]
        by_k_md[k] = [
            f"# Skeletons k={k}\n\n",
            "| p | # | status | n_edges | Cd CN | Se CN | rings | reason | xyz |\n",
            "|--:|--:|--------|-------:|-------|-------|-------|--------|-----|\n",
        ]
        # Reload prior rows for this k into by_k so tsv stays complete on resume.
        if resume:
            for line in global_lines[1:]:
                parts = line.split(",", 2)
                if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == k:
                    by_k[k].append(line)

        for p in range(p_lo, int(max_p) + 1):
            if k == 1 and p == 0:
                continue
            if resume and (k, p) in prior_bins:
                if progress is not None:
                    progress(
                        f"  k={k} p={p}: skip (already in skeleton dump)"
                    )
                continue
            symbols = _symbols_for_composition(check_spec, k, p)
            roles = _roles_for_composition(check_spec, k, p)
            atoms = _atoms_for_composition(symbols, roles)
            cation_ids = [
                atom.atom_id
                for atom in atoms
                if atom.symbol
                in {check_spec.core.cation, check_spec.precursor.center}
            ]
            se_ids_list = [
                atom.atom_id
                for atom in atoms
                if atom.symbol == check_spec.core.anion
            ]
            # Same highest structure level as decoration ladder (not 1-ring only).
            dump_level = max_structure_level_possible(k, p, check_spec)
            dump_mode = structure_mode_for_level(dump_level)
            if progress is not None:
                if dump_mode == "fused2":
                    progress(
                        f"  k={k} p={p}: structure level 2 "
                        f"(fused-2 seed: all modes path∪face∪edge)"
                    )
                elif dump_mode == "ring_first":
                    progress(
                        f"  k={k} p={p}: structure level 1 "
                        f"(1-ring closed Cd3Se3 seed)"
                    )
                else:
                    progress(
                        f"  k={k} p={p}: structure level 0 "
                        f"(free skeletons, open allowed)"
                    )
            skeletons, _trunc = _enumerate_inorganic_edge_sets(
                k,
                p,
                check_spec,
                max_skeletons=max_skeletons,
                extra_skeleton_edges=extra_skeleton_edges,
                mode=dump_mode,
                pack=pack,
            )
            # Skeleton dumps are graph-only and apply the same structured
            # feasibility ladder as decoration.  A composition label alone is
            # not proof that the hard Se/Cd ring pattern can be completed.
            while dump_mode in {"fused2", "ring_first"}:
                viable = [
                    sk
                    for sk in skeletons
                    if forced_ring_degree_profiles(
                        sk, k, p, check_spec, mode=dump_mode
                    )[1]
                ]
                if viable:
                    skeletons = viable
                    break
                dump_mode = "ring_first" if dump_mode == "fused2" else "free"
                skeletons, _trunc = _enumerate_inorganic_edge_sets(
                    k,
                    p,
                    check_spec,
                    max_skeletons=max_skeletons,
                    extra_skeleton_edges=extra_skeleton_edges,
                    mode=dump_mode,
                    pack=pack,
                )
            if progress is not None:
                progress(
                    f"  k={k} p={p}: enumerated {len(skeletons)} skeleton(s) "
                    f"[{dump_mode}]"
                )
            for skeleton_index, skel in enumerate(skeletons, start=1):
                skeleton_graph = nx.Graph()
                skeleton_graph.add_nodes_from(range(len(atoms)))
                skeleton_graph.add_edges_from(
                    (int(a), int(b)) for a, b in skel
                )
                skel_state = _State(atoms=atoms, graph=skeleton_graph)
                cd_cn = tuple(
                    sorted(
                        int(skeleton_graph.degree[i]) for i in cation_ids
                    )
                )
                se_cn = tuple(
                    sorted(
                        int(skeleton_graph.degree[i]) for i in se_ids_list
                    )
                )
                n_six = count_cdse_six_rings(skel, k, p)
                forced_rings, _ring_profiles = forced_ring_degree_profiles(
                    skel, k, p, check_spec, mode=dump_mode
                )
                ring_lab = ring_closure_log_label(
                    n_six,
                    pattern_possible=ring_first_required_for_spec(
                        k, p, check_spec
                    ),
                )
                cd_txt = _format_cn_list(cd_cn)
                se_txt = _format_cn_list(se_cn)
                status = "accepted"
                reason = ""
                viol = _skeleton_graph_violations(skel_state, check_spec)
                if viol:
                    status = "skipped_graph"
                    reason = ";".join(viol)
                xyz_name = ""
                # ``embed`` remains accepted for CLI compatibility, but the
                # skeleton stage is deliberately graph-only.  Coordinates are
                # constructed only after a completed decorated graph exists.
                edges_txt = format_skeleton_edges(skel)
                row = (
                    f"{k},{p},{skeleton_index},{status},{len(skel)},"
                    f"\"{cd_txt}\",\"{se_txt}\","
                    f"{reason.replace(',', ';')},{xyz_name},{edges_txt},"
                    f"{n_six},{dump_mode},{format_forced_rings(forced_rings)}"
                )
                global_lines.append(row)
                by_k[k].append(row)
                by_k_md[k].append(
                    f"| {p} | {skeleton_index} | {status} | {len(skel)} | "
                    f"`{cd_txt}` | `{se_txt}` | {ring_lab} | "
                    f"{reason or '—'} | {xyz_name or '—'} |\n"
                )
                if progress is not None:
                    progress(
                        f"    skeleton_k{k}_p{p}_{skeleton_index:03d} "
                        f"{status} {ring_lab} Cd{cd_txt} Se{se_txt}"
                        + (f" | {reason}" if reason else "")
                    )
            # Flush after every (k,p) so a kill mid-sweep keeps progress.
            flush_global()
            (k_dir / "skeletons.tsv").write_text(
                "\n".join(by_k[k]) + "\n", encoding="utf-8"
            )

        (k_dir / "skeletons.tsv").write_text(
            "\n".join(by_k[k]) + "\n", encoding="utf-8"
        )
        (k_dir / "skeletons.md").write_text(
            "".join(by_k_md[k]), encoding="utf-8"
        )

    flush_global()
    if progress is not None:
        progress(
            f"[molecular] SKELETON DUMP DONE | {root / 'skeletons.csv'} | "
            "graph-only catalog; edges/forced_rings enable decoration resume"
        )
    return root


def _csv_cell(value: object) -> str:
    """Quote a CSV field when it contains a separator.

    Several annotation values are themselves comma-joined lists
    (``cdse_cn_pairs`` renders as ``2-2:1,2-4:1,...``), which silently shifted
    every column to their right.
    """

    text = "" if value is None else str(value)
    if any(ch in text for ch in (",", '"', "\n")):
        return '"' + text.replace('"', '""') + '"'
    return text


def write_molecular_map(
    result: MolecularMapResult,
    output_dir: str | Path,
    *,
    only_bin: Optional[Tuple[int, int]] = None,
    dump_failures: bool = False,
) -> Path:
    """Write XYZ, graph annotations, and coordination-motif inventories.

    ``dump_failures`` additionally writes one representative geometry per
    distinct failure under ``k***/p***/failures/``.  It is off by default: a
    run rejects far more candidates than it accepts, so the dump dominates the
    output tree while most of it is never looked at.  ``failure_manifest.csv``
    still records the stage, CN vector, reason and count of every failure
    either way -- only the coordinates are withheld.

    ``only_bin`` restricts the write to a single ``(k, p)`` directory and skips
    the run-level CSVs, which aggregate every bin accumulated so far.  The
    per-skeleton checkpoint uses it: rewriting the whole map once per skeleton
    made checkpoint cost grow with the size of the run (measured 0.004 s ->
    0.100 s per call over a k=1..3 sweep as the isomer count went 1 -> 27, i.e.
    quadratic in a full sweep).  The run-level CSVs are still regenerated in
    full at every bin boundary and at the end of the run.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    annotation_fields = [
        "n_cd2",
        "n_cd3",
        "n_cd4",
        "mean_cd_cn",
        "n_terminal_cl",
        "n_mu2_cl",
        "n_mu3_cl",
        "n_unsaturated_bridge_candidates",
        "n_closable_terminal_cd2",
        "n_cl2se1_near_cd2",
        "cdse_cn_pairs",
        "cd2_signatures",
    ]
    xtb_fields = [
        "xtb_energy_eV",
        "xtb_gap_eV",
        "xtb_steps",
        "xtb_converged",
        "xtb_relaxed_bonds",
        "xtb_bonds_delta",
        "xtb_cl_terminal",
        "xtb_cl_mu2",
        "xtb_cl_mu3",
        "xtb_same_topology",
        "xtb_matches",
        "xtb_xyz",
        "xtb_error",
    ]
    index_header = (
        "k,p,structure_id,conformation,n_atoms,n_edges,xyz,"
        + ",".join(annotation_fields)
        + ","
        + ",".join(xtb_fields)
    )
    index_lines = [index_header]
    annotation_lines = ["k,p,structure_id," + ",".join(annotation_fields)]
    motif_lines = ["k,p,structure_id,motif,count"]
    rejection_lines = ["k,p,reason,count"]
    rejection_detail_lines = ["k,p,detail,count"]
    failure_manifest_lines = [
        "k,p,skeleton_index,cd_cn,stage,reason,count,snapshot_kind,xyz"
    ]
    xtb_transition_lines = [
        "k,p,structure_id,start,source_edges,final_edges,gained_edges,lost_edges,"
        "same_topology,source_motifs,final_motifs"
    ]
    xtb_merge_lines = [
        "k,p,source_structure,destination_structure,detail"
    ]
    graph_merge_lines = [
        "k,p,source_graph,destination_structure,detail"
    ]
    xtb_geometry_lines = [
        "k,p,structure_id,start,all_atom_rmsd_A,inorganic_rmsd_A,max_displacement_A"
    ]
    xtb_bond_order_lines = ["k,p,structure_id,left,right,wiberg"]
    motif_trial_lines = [
        "k,p,trial_id,start,initial_xyz,xtb_xyz,xtb_ok,xtb_converged,"
        "source_edges,final_edges,initial_violations,final_violations,xtb_error"
    ]
    all_skel_lines = [
        "k,p,skeleton_index,status,n_edges,cd_cn,se_cn,reason,xyz"
    ]
    for (k, p), bin_res in sorted(result.bins.items()):
        if only_bin is not None and (k, p) != only_bin:
            continue
        bin_dir = root / f"k{k:03d}" / f"p{p:03d}"
        bin_dir.mkdir(parents=True, exist_ok=True)
        output_isomers = bin_res.isomers
        if bin_res.motif_xtb_attempts:
            # Motif runs keep graph-audited starts in memory for diagnostics,
            # but the main p00* directory should contain only structures with
            # an xTB endpoint.  A post-xTB audit failure is retained as a
            # diagnostic warning; only attempts with no coordinates are
            # excluded from the main p00* directory.
            output_isomers = [
                iso for iso in bin_res.isomers if iso.xtb_coordinates is not None
            ]
        if bin_res.skeleton_records:
            write_skeleton_inventory(bin_res, bin_dir)
            for rec in bin_res.skeleton_records:
                cd_txt = "[" + ",".join(str(x) for x in rec.cd_cn) + "]"
                se_txt = "[" + ",".join(str(x) for x in rec.se_cn) + "]"
                xyz_rel = (
                    f"k{k:03d}/skeletons/"
                    f"{skeleton_xyz_basename(k, p, rec.skeleton_index)}"
                    if rec.coordinates is not None
                    else ""
                )
                all_skel_lines.append(
                    f"{k},{p},{rec.skeleton_index},{rec.status},"
                    f"{rec.n_edges},\"{cd_txt}\",\"{se_txt}\","
                    f"{rec.reason.replace(',', ';')},{xyz_rel}"
                )
        if bin_res.motif_trials:
            trial_dir = bin_dir / "motif_trials"
            trial_dir.mkdir(parents=True, exist_ok=True)
            bin_trial_lines = [motif_trial_lines[0]]

            def trial_edge_text(edges: Iterable[Tuple[int, int]]) -> str:
                return "|".join(f"{left}-{right}" for left, right in edges)

            for trial in bin_res.motif_trials:
                initial_path = trial_dir / f"{trial.trial_id}_initial.xyz"
                write_xyz(
                    str(initial_path),
                    list(trial.symbols),
                    np.asarray(trial.initial_coordinates, dtype=float),
                )
                xtb_path = ""
                if trial.xtb_coordinates is not None:
                    relaxed_path = trial_dir / f"{trial.trial_id}_xtb.xyz"
                    write_xyz(
                        str(relaxed_path),
                        list(trial.symbols),
                        np.asarray(trial.xtb_coordinates, dtype=float),
                        comment=(
                            f"{trial.trial_id} energy_eV={trial.xtb_energy_eV:.6f}"
                            if trial.xtb_energy_eV is not None
                            else trial.trial_id
                        ),
                    )
                    xtb_path = str(relaxed_path.relative_to(root))
                    if trial.final_violations:
                        warning_path = bin_dir / f"{trial.trial_id}_xtb_audit_warning.xyz"
                        write_xyz(
                            str(warning_path),
                            list(trial.symbols),
                            np.asarray(trial.xtb_coordinates, dtype=float),
                            comment=(
                                f"{trial.trial_id} xTB_audit_warning "
                                f"energy_eV={trial.xtb_energy_eV:.6f} "
                                f"violations={'|'.join(trial.final_violations)}"
                                if trial.xtb_energy_eV is not None
                                else f"{trial.trial_id} xTB_audit_warning "
                                f"violations={'|'.join(trial.final_violations)}"
                            ),
                        )
                cells = (
                    k,
                    p,
                    trial.trial_id,
                    trial.start_index,
                    str(initial_path.relative_to(root)),
                    xtb_path,
                    str(trial.xtb_ok).lower(),
                    str(trial.xtb_converged).lower(),
                    trial_edge_text(trial.source_edges),
                    trial_edge_text(trial.final_edges),
                    "|".join(trial.initial_violations),
                    "|".join(trial.final_violations),
                    trial.xtb_error,
                )
                line = ",".join(_csv_cell(value) for value in cells)
                motif_trial_lines.append(line)
                bin_trial_lines.append(line)
            (bin_dir / "motif_trials.csv").write_text(
                "\n".join(bin_trial_lines) + "\n", encoding="utf-8"
            )
        for iso in output_isomers:
            symbols = [a.symbol for a in iso.atoms]
            ann = iso.annotations
            if ann is None:
                ann_vals = {field: "" for field in annotation_fields}
            else:
                ann_vals = ann.as_csv_row()
            ann_csv = ",".join(
                _csv_cell(ann_vals[field]) for field in annotation_fields
            )
            outputs = list(iso.conformers)
            if not outputs:
                # ``coordinates`` is a numpy array on the embedded path, so it
                # cannot be truth-tested; only an un-embedded isomer has none.
                packed = (
                    iso.coordinates
                    if iso.coordinates is not None
                    else tuple((0.0, 0.0, 0.0) for _ in symbols)
                )
                outputs = [("", packed)]
            # Relaxed geometry is written once per isomer, next to the
            # constructed one, so both are inspectable side by side.
            xtb_path = ""
            if iso.xtb_coordinates is not None:
                relaxed = bin_dir / f"{iso.structure_id}_xtb.xyz"
                write_xyz(
                    str(relaxed),
                    symbols,
                    np.asarray(iso.xtb_coordinates, dtype=float),
                    comment=(
                        f"{iso.structure_id} energy_eV={iso.xtb_energy_eV:.6f} "
                        f"xtb_converged={str(iso.xtb_converged).lower()} "
                        "motifs="
                        + "|".join(f"{name}:{count}" for name, count in iso.motif_inventory)
                        + (
                            " audit_warning=" + "|".join(str(v) for v in iso.violations)
                            if iso.violations else ""
                        )
                    ),
                )
                xtb_path = str(relaxed)
            cl_t, cl_2, cl_3 = iso.xtb_relaxed_cl_motifs
            xtb_csv = ",".join(
                _csv_cell(value)
                for value in (
                    "" if iso.xtb_energy_eV is None else f"{iso.xtb_energy_eV:.6f}",
                    "" if iso.xtb_gap_eV is None else f"{iso.xtb_gap_eV:.4f}",
                    iso.xtb_steps,
                    str(iso.xtb_converged).lower(),
                    iso.xtb_relaxed_bonds,
                    iso.xtb_bonds_delta,
                    cl_t,
                    cl_2,
                    cl_3,
                    str(iso.xtb_same_topology).lower(),
                    iso.xtb_matches,
                    xtb_path,
                    iso.xtb_error.replace(",", ";"),
                )
            )
            for conformation, packed in outputs:
                suffix = f"_{conformation}" if conformation else ""
                path = bin_dir / f"{iso.structure_id}{suffix}.xyz"
                write_xyz(str(path), symbols, np.asarray(packed, dtype=float))
                index_lines.append(
                    f"{k},{p},{iso.structure_id},{conformation},{len(symbols)},"
                    f"{iso.graph.number_of_edges()},{path},{ann_csv},{xtb_csv}"
                )
            if iso.source_edges:
                source = set(iso.source_edges)
                final = {
                    (min(int(a), int(b)), max(int(a), int(b)))
                    for a, b in iso.graph.edges
                }
                gained = tuple(sorted(final - source))
                lost = tuple(sorted(source - final))

                def motif_summary(edges: Set[Tuple[int, int]]) -> str:
                    degree = Counter(i for edge in edges for i in edge)
                    counts = Counter(
                        f"{atom.symbol}-Cd{degree.get(atom.atom_id, 0)}"
                        for atom in iso.atoms
                        if atom.symbol in {"Se", "Cl"}
                    )
                    return "|".join(
                        f"{name}:{count}" for name, count in sorted(counts.items())
                    )

                def edge_text(edges: Iterable[Tuple[int, int]]) -> str:
                    return "|".join(f"{a}-{b}" for a, b in sorted(edges))

                xtb_transition_lines.append(
                    f'{k},{p},{iso.structure_id},{iso.reconstruction_start},'
                    f'"{edge_text(source)}","{edge_text(final)}",'
                    f'"{edge_text(gained)}","{edge_text(lost)}",'
                    f'{str(not gained and not lost).lower()},'
                    f'"{motif_summary(source)}","{motif_summary(final)}"'
                )
            if iso.coordinates is not None and iso.xtb_coordinates is not None:
                before = np.asarray(iso.coordinates, dtype=float)
                after = np.asarray(iso.xtb_coordinates, dtype=float)

                def aligned_delta(indices: Sequence[int]) -> FloatArray:
                    a = before[list(indices)]
                    b = after[list(indices)]
                    ac = a - np.mean(a, axis=0)
                    bc = b - np.mean(b, axis=0)
                    u, _s, vt = np.linalg.svd(ac.T @ bc)
                    rotation = vt.T @ u.T
                    if np.linalg.det(rotation) < 0:
                        vt[-1] *= -1
                        rotation = vt.T @ u.T
                    return ac @ rotation.T - bc

                all_ids = list(range(len(iso.atoms)))
                inorganic_ids = [
                    atom.atom_id for atom in iso.atoms
                    if atom.symbol != "Cl"
                ]
                delta_all = aligned_delta(all_ids)
                delta_inorganic = aligned_delta(inorganic_ids)
                rms_all = float(np.sqrt(np.mean(np.sum(delta_all * delta_all, axis=1))))
                rms_inorganic = float(np.sqrt(np.mean(np.sum(delta_inorganic * delta_inorganic, axis=1))))
                max_move = float(np.max(np.linalg.norm(delta_all, axis=1)))
                xtb_geometry_lines.append(
                    f"{k},{p},{iso.structure_id},{iso.reconstruction_start},"
                    f"{rms_all:.6f},{rms_inorganic:.6f},{max_move:.6f}"
                )
            if iso.xtb_bond_orders is not None:
                orders = np.asarray(iso.xtb_bond_orders, dtype=float)
                for left in range(len(iso.atoms)):
                    for right in range(left + 1, len(iso.atoms)):
                        xtb_bond_order_lines.append(
                            f"{k},{p},{iso.structure_id},{left},{right},"
                            f"{orders[left, right]:.8f}"
                        )
            annotation_lines.append(
                f"{k},{p},{iso.structure_id},{ann_csv}"
            )
            for motif, count in iso.motif_inventory:
                motif_lines.append(
                    f"{k},{p},{iso.structure_id},{motif},{int(count)}"
                )
        n_acc_skel = sum(
            1 for r in bin_res.skeleton_records if r.status == "accepted"
        )
        meta = bin_dir / "bin_meta.txt"
        meta.write_text(
            f"k={k} p={p}\n"
            f"isomers={len(bin_res.isomers)}\n"
            f"isomers_written={len(output_isomers)}\n"
            f"raw_graphs={bin_res.raw_graphs}\n"
            f"unique_graphs={bin_res.unique_graphs}\n"
            f"rejected={bin_res.rejected}\n"
            f"incomplete={str(bin_res.incomplete).lower()}\n"
            f"skeletons_total={bin_res.skeletons_total}\n"
            f"skeletons_accepted={n_acc_skel}\n"
            f"skeletons_pruned_graph={bin_res.skeletons_pruned_graph}\n"
            f"skeletons_pruned_frame={bin_res.skeletons_pruned_frame}\n"
            f"ring_min_pattern_cd={_format_cn_list(bin_res.ring_min_pattern_cd)}\n"
            f"ring_min_pattern_se={_format_cn_list(bin_res.ring_min_pattern_se)}\n"
            f"geometry_ring_pattern_cd={_format_cn_list(bin_res.geometry_ring_pattern_cd)}\n"
            f"geometry_ring_pattern_se={_format_cn_list(bin_res.geometry_ring_pattern_se)}\n"
            f"motif_graphs_eligible={bin_res.motif_graphs_eligible}\n"
            f"motif_reconstruction_attempts={bin_res.motif_reconstruction_attempts}\n"
            f"motif_reconstruction_candidates={bin_res.motif_reconstruction_candidates}\n"
            f"motif_pre_xtb_accepted={bin_res.motif_pre_xtb_accepted}\n"
            f"motif_xtb_attempts={bin_res.motif_xtb_attempts}\n"
            f"motif_xtb_converged={bin_res.motif_xtb_converged}\n"
            f"motif_xtb_same_graph_rescues={bin_res.motif_xtb_same_graph_rescues}\n"
            f"motif_xtb_discovered={bin_res.motif_xtb_discovered}\n"
            f"xtb_merges={len(bin_res.xtb_merge_records)}\n"
            f"graph_merges_before_xtb={len(bin_res.graph_merge_records)}\n"
            f"skeleton_generation_time_s={bin_res.skeleton_generation_time_s:.3f}\n"
            f"decoration_generation_time_s={bin_res.decoration_generation_time_s:.3f}\n"
            f"candidate_screen_time_s={bin_res.candidate_screen_time_s:.3f}\n"
            f"decoration_stream_time_s={bin_res.decoration_stream_time_s:.3f}\n"
            f"Note: raw_graphs are Cl decorations streamed for accepted "
            f"skeletons; most can fail at embed (e.g. terminal angles).\n",
            encoding="utf-8",
        )
        for reason, count in sorted(bin_res.rejection_reasons.items()):
            rejection_lines.append(f"{k},{p},{reason},{count}")
        for detail, count in sorted(bin_res.rejection_details.items()):
            escaped = str(detail).replace('"', '""')
            rejection_detail_lines.append(f'{k},{p},"{escaped}",{count}')
        failure_dir = bin_dir / "failures"
        for record in sorted(
            bin_res.failure_records.values(),
            key=lambda item: (
                item.skeleton_index,
                item.cd_cn,
                item.stage,
                item.reason,
            ),
        ):
            xyz_rel = ""
            if (
                dump_failures
                and record.snapshot_coordinates
                and record.snapshot_symbols
            ):
                failure_dir.mkdir(parents=True, exist_ok=True)
                filename = (
                    f"failure_skel{record.skeleton_index:03d}_"
                    f"cn{_failure_slug(_format_cn_list(record.cd_cn))}_"
                    f"{_failure_slug(record.stage)}_"
                    f"{_failure_slug(record.reason)}.xyz"
                )
                xyz_path = failure_dir / filename
                write_xyz(
                    str(xyz_path),
                    list(record.snapshot_symbols),
                    np.asarray(record.snapshot_coordinates, dtype=float),
                )
                xyz_rel = str(xyz_path.relative_to(root))
            cells = [
                str(k),
                str(p),
                str(record.skeleton_index),
                _format_cn_list(record.cd_cn),
                record.stage,
                record.reason,
                str(record.count),
                record.snapshot_kind,
                xyz_rel,
            ]
            failure_manifest_lines.append(",".join(
                '"' + str(cell).replace('"', '""') + '"' for cell in cells
            ))
        for source_id, destination_id, detail in bin_res.xtb_merge_records:
            xtb_merge_lines.append(
                ",".join(
                    _csv_cell(value)
                    for value in (k, p, source_id, destination_id, detail)
                )
            )
        for source_id, destination_id, detail in bin_res.graph_merge_records:
            graph_merge_lines.append(
                ",".join(
                    _csv_cell(value)
                    for value in (k, p, source_id, destination_id, detail)
                )
            )
    if only_bin is not None:
        # Run-level CSVs aggregate every bin; a single-bin checkpoint would
        # truncate them.  They are rewritten in full at the next bin boundary.
        return root
    (root / "index.csv").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    (root / "annotations.csv").write_text(
        "\n".join(annotation_lines) + "\n", encoding="utf-8"
    )
    (root / "motifs.csv").write_text(
        "\n".join(motif_lines) + "\n", encoding="utf-8"
    )
    (root / "rejections.csv").write_text(
        "\n".join(rejection_lines) + "\n", encoding="utf-8"
    )
    (root / "rejection_details.csv").write_text(
        "\n".join(rejection_detail_lines) + "\n", encoding="utf-8"
    )
    (root / "failure_manifest.csv").write_text(
        "\n".join(failure_manifest_lines) + "\n", encoding="utf-8"
    )
    (root / "xtb_transitions.csv").write_text(
        "\n".join(xtb_transition_lines) + "\n", encoding="utf-8"
    )
    (root / "xtb_merges.csv").write_text(
        "\n".join(xtb_merge_lines) + "\n", encoding="utf-8"
    )
    (root / "graph_merges.csv").write_text(
        "\n".join(graph_merge_lines) + "\n", encoding="utf-8"
    )
    (root / "xtb_geometry_changes.csv").write_text(
        "\n".join(xtb_geometry_lines) + "\n", encoding="utf-8"
    )
    (root / "xtb_bond_orders.csv").write_text(
        "\n".join(xtb_bond_order_lines) + "\n", encoding="utf-8"
    )
    (root / "motif_trials.csv").write_text(
        "\n".join(motif_trial_lines) + "\n", encoding="utf-8"
    )
    (root / "skeletons.csv").write_text(
        "\n".join(all_skel_lines) + "\n", encoding="utf-8"
    )
    (root / "README.txt").write_text(
        "Molecular map (lattice-free).\n"
        f"geometry_pack={result.geometry_pack_name}\n"
        "Coordinates are constructed only after graph decoration and graph "
        "deduplication; final contacts can reject a completed graph.\n"
        "rejections.csv contains category counts; rejection_details.csv "
        "preserves full rejection strings with atom ids/distances when "
        "available. failure_manifest.csv contains stage/CN/skeleton context "
        "and at most one representative failed XYZ per failure class.\n"
        "Graphs are filtered by pair legality, inorganic connectivity, CN "
        "bounds, bridges_per_Cd_pair, and isomorphism dedup.\n"
        "annotations.csv labels collapse-risk features (n_cd2, closable "
        "terminal→Cd2 contacts, unsaturated bridge candidates) for DFT "
        "start→final graph comparison; they do not change acceptance.\n"
        "xtb_merges.csv lists motif source structures whose relaxed connectivity "
        "matches another enumerated structure.\n"
        "graph_merges.csv lists source graphs removed by graph deduplication "
        "before reconstruction/xTB.\n"
        "Upfront (run_molecular_map): skeletons.csv is the graph-only global "
        "catalog and records edges plus forced ring identities.\n"
        "raw_graphs count Cl decorations *before* embed; high raw + zero "
        "accepted usually means embed rejects (angles/contacts), not empty "
        "skeleton search.\n",
        encoding="utf-8",
    )
    return root
