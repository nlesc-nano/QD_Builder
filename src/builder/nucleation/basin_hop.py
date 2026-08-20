"""Basin hopping over relaxed minima at fixed ``(k, p)`` composition.

Why this exists
---------------

Move Z finds isomers by enumerating zinc-blende occupations, decorating them
with graph rules, embedding with distance geometry, and relaxing.  Nothing in
that chain *searches*: every endpoint is whatever basin the embedding start
happened to fall into.  Three measurements from the k=1..4 run say the endpoints
may be artifacts of the seeding rather than real minima --- the lowest off-path
endpoint beat the lowest on-path one in 10 of 12 bins (by 0.05-0.92 eV), the two
embedding starts of one decorated graph disagree by a median 0.65 eV, and 2066
of 4084 structures fragment during plain minimisation.

Basin hopping (Wales & Doye) answers "is the bin minimum real?" using *only*
local optimisation, which is the single thing the g-xTB backend exposes:
``_cli_command`` in :mod:`.xtb_relax` emits ``--opt`` unconditionally and there
is no single-point or gradient path.  The transformed surface E'(x) = E(min(x))
is a staircase, so barriers stop mattering and no thermostat, trajectory parser
or confining wall is needed --- all of which MD would require, at a system size
(15-35 atoms, in vacuum, with labile CdCl2) where ps-scale dynamics mostly
samples evaporation.

Nothing here is wired into the growth pipeline.  It reads structures that a
finished run already wrote and reports what it finds.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from ..nc_types import NucleationSpec
from .types import AtomRecord, _State
from .xtb_relax import XtbSettings, relax_structures, relaxed_edges

__all__ = [
    "BasinHopMinimum",
    "BasinHopResult",
    "MOVES",
    "basin_hop",
    "describe_minimum",
]

#: Perturbation kinds.  ``surface_swap`` is the classic move for ionic clusters:
#: exchanging a cation and a ligand keeps the composition exact while changing
#: which sites are surface-terminated, which is precisely the degree of freedom
#: the graph enumerator fixes when it chooses a decoration.
MOVES = ("shake", "single_atom", "surface_swap")


@dataclass
class BasinHopMinimum:
    """One distinct relaxed basin discovered by a walker."""

    structure_id: str
    seed_id: str
    step: int
    energy_eV: float
    symbols: Tuple[str, ...]
    coordinates: np.ndarray
    edges: Tuple[Tuple[int, int], ...]
    core_edges: Tuple[Tuple[int, int], ...]
    #: Did the Cd-Se core survive?  Compared against the walker's seed, which
    #: is the same test move Z applies (``final_core == expected_core``).
    core_preserved: bool = True
    #: Secondary, and weaker than it sounds: a *relaxed* core drifts ~1 A off
    #: ideal lattice sites, so this is False even for endpoints the growth
    #: pipeline accepted.  Reported, never used to filter.
    zb_embeddable: bool = False
    zb_reason: str = ""
    violations: Tuple[str, ...] = ()
    n_components: int = 1
    converged: bool = False

    @property
    def clean(self) -> bool:
        """Would the growth pipeline's post-relaxation audit accept this?"""

        return self.converged and not self.violations and self.n_components == 1


@dataclass
class BasinHopResult:
    """Outcome of one walker."""

    seed_id: str
    k: int
    p: int
    seed_energy_eV: float
    steps_run: int = 0
    n_relaxations: int = 0
    n_accepted: int = 0
    n_rejected_pre_qc: int = 0
    n_failed: int = 0
    minima: List[BasinHopMinimum] = field(default_factory=list)
    trajectory: List[Tuple[int, float, bool]] = field(default_factory=list)

    @property
    def best(self) -> Optional[BasinHopMinimum]:
        if not self.minima:
            return None
        return min(self.minima, key=lambda item: item.energy_eV)

    @property
    def gain_eV(self) -> float:
        """How far below the seed the best discovered basin sits (<= 0 is a win)."""

        best = self.best
        if best is None:
            return 0.0
        return float(best.energy_eV) - float(self.seed_energy_eV)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def _overlaps(coordinates: np.ndarray, overlap_min_A: float) -> bool:
    """Catastrophic-overlap guard, so a doomed proposal costs no g-xTB call.

    This deliberately uses the pack's ``reconstruction.overlap_min_A`` (0.75 A)
    and *not* ``relaxation.artifact_min_distance``.  The artifact floors are a
    post-relaxation test for collapsed geometries -- Cd-Cd 2.80 A, against a
    relaxed seed whose tightest Cd-Cd is 3.06 A.  Applying them to a *proposal*
    rejects almost every trial move, because a 0.35 A shake closes a 0.26 A
    margin.  A proposal at Cd-Cd 2.7 A is a perfectly good starting geometry;
    the optimiser is what decides whether it is a real minimum.
    """

    if overlap_min_A <= 0.0:
        return False
    points = np.asarray(coordinates, dtype=float)
    deltas = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(distances, np.inf)
    return bool(distances.min() < float(overlap_min_A))


def _perturb(
    symbols: Sequence[str],
    coordinates: np.ndarray,
    *,
    move: str,
    amplitude_A: float,
    cation: str,
    ligand: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """One trial displacement of an already-relaxed geometry."""

    points = np.array(coordinates, dtype=float, copy=True)
    if move == "shake":
        points += rng.normal(0.0, amplitude_A, size=points.shape)
    elif move == "single_atom":
        # A larger kick on one atom explores a different neighbourhood than a
        # uniform shake of the same total magnitude: it moves one coordination
        # environment rather than softening all of them.
        index = int(rng.integers(0, len(points)))
        points[index] += rng.normal(0.0, 3.0 * amplitude_A, size=3)
    elif move == "surface_swap":
        cations = [i for i, s in enumerate(symbols) if s == cation]
        ligands = [i for i, s in enumerate(symbols) if s == ligand]
        if cations and ligands:
            a = int(rng.choice(cations))
            b = int(rng.choice(ligands))
            points[[a, b]] = points[[b, a]]
            points += rng.normal(0.0, 0.25 * amplitude_A, size=points.shape)
        else:
            points += rng.normal(0.0, amplitude_A, size=points.shape)
    else:
        raise ValueError(f"unknown move {move!r}; expected one of {MOVES}")
    return points


# ---------------------------------------------------------------------------
# audits, reusing the pipeline's own definitions
# ---------------------------------------------------------------------------


def describe_minimum(
    structure_id: str,
    seed_id: str,
    step: int,
    *,
    k: int,
    p: int,
    symbols: Sequence[str],
    coordinates: np.ndarray,
    energy_eV: float,
    converged: bool,
    spec: NucleationSpec,
    cutoffs: Mapping[Tuple[str, str], float],
    zb_model: Optional[Any] = None,
    reference_core_edges: Optional[Sequence[Tuple[int, int]]] = None,
    motif_definitions: Optional[Mapping[str, Any]] = None,
    artifact_floors: Optional[Mapping[str, float]] = None,
) -> BasinHopMinimum:
    """Label a relaxed geometry with the same audits the growth path applies."""

    import networkx as nx

    from .molecular import (
        molecular_decoration_rule_violations,
        molecular_graph_violations,
    )
    from .molecular_motif_reconstruct import motif_vocabulary_violations
    from .molecular_rules import forbidden_pair_contact_violations

    points = np.asarray(coordinates, dtype=float)
    edges = tuple(
        sorted(
            (min(int(a), int(b)), max(int(a), int(b)))
            for a, b in relaxed_edges(symbols, points, cutoffs)
        )
    )
    graph = nx.Graph()
    graph.add_nodes_from(range(len(symbols)))
    graph.add_edges_from(edges)
    atoms = [
        AtomRecord(i, str(sym), tuple(float(x) for x in points[i]), "relaxed")
        for i, sym in enumerate(symbols)
    ]
    state = _State(atoms=atoms, graph=graph)

    violations: List[str] = []
    try:
        violations.extend(molecular_graph_violations(state, spec))
        violations.extend(molecular_decoration_rule_violations(state, spec))
        # The two below are what move Z also checks before it calls an
        # endpoint propagation-eligible.  Without them a structure can look
        # "clean" here and still be rejected by the growth pipeline.
        violations.extend(
            motif_vocabulary_violations(
                state,
                cation=spec.core.cation,
                anion=spec.core.anion,
                ligand=spec.precursor.ligand,
                motif_definitions=motif_definitions,
            )
        )
        violations.extend(
            forbidden_pair_contact_violations(
                list(symbols), points, spec, floors=artifact_floors or None
            )
        )
    except Exception as exc:  # noqa: BLE001 — a label, never a control path
        violations.append(f"audit_failed:{type(exc).__name__}")

    core_pair = {spec.core.cation, spec.core.anion}
    core_edges = tuple(
        edge for edge in edges if {symbols[edge[0]], symbols[edge[1]]} == core_pair
    )

    embeddable, reason = False, "not_checked"
    if zb_model is not None:
        from .molecular_zb_growth import zb_embeddable

        try:
            embeddable, _occ, reason = zb_embeddable(
                list(symbols), points, spec, zb_model, parent_id=structure_id
            )
            reason = str(reason or "")
        except Exception as exc:  # noqa: BLE001 — diagnostic only
            embeddable, reason = False, f"zb_check_failed:{type(exc).__name__}"

    core_preserved = (
        True
        if reference_core_edges is None
        else core_edges == tuple(sorted(tuple(e) for e in reference_core_edges))
    )

    return BasinHopMinimum(
        structure_id=structure_id,
        seed_id=seed_id,
        step=int(step),
        energy_eV=float(energy_eV),
        symbols=tuple(str(s) for s in symbols),
        coordinates=points,
        edges=edges,
        core_edges=core_edges,
        core_preserved=bool(core_preserved),
        zb_embeddable=bool(embeddable),
        zb_reason=reason,
        violations=tuple(dict.fromkeys(str(v) for v in violations)),
        n_components=int(nx.number_connected_components(graph)) if len(graph) else 0,
        converged=bool(converged),
    )


def _as_parent(minimum: BasinHopMinimum, k: int, p: int) -> Any:
    """Wrap a discovered basin so the consolidation test can compare it.

    ``relaxed_minimum_similarity`` is already the pipeline's definition of "same
    basin" -- coloured-graph mapping, internal pair distances, then permuted
    Kabsch RMSD -- so the walker's history reuses it instead of inventing a
    second, differently calibrated notion of identity.
    """

    from .molecular_growth import ParentStructure

    return ParentStructure(
        k=int(k),
        p=int(p),
        structure_id=minimum.structure_id,
        symbols=minimum.symbols,
        coordinates=minimum.coordinates,
        energy_eV=float(minimum.energy_eV),
        edges=list(minimum.edges),
        core_edges=list(minimum.core_edges),
    )


# ---------------------------------------------------------------------------
# the walker
# ---------------------------------------------------------------------------


def basin_hop(
    seed: Mapping[str, Any],
    settings: XtbSettings,
    spec: NucleationSpec,
    *,
    k: int,
    p: int,
    steps: int = 200,
    temperature_eV: float = 0.15,
    moves: Sequence[str] = MOVES,
    amplitude_A: float = 0.35,
    rng_seed: int = 1729,
    cutoffs: Optional[Mapping[Tuple[str, str], float]] = None,
    zb_model: Optional[Any] = None,
    consolidation: Optional[Any] = None,
    overlap_min_A: float = 0.75,
    motif_definitions: Optional[Mapping[str, Any]] = None,
    progress: Optional[Callable[[str], None]] = None,
    on_minimum: Optional[Callable[[BasinHopMinimum], None]] = None,
) -> BasinHopResult:
    """Run one basin-hopping walker from an already-relaxed structure.

    ``seed`` needs ``id``, ``symbols``, ``positions`` and ``energy_eV``.  The
    seed is taken as a relaxed minimum and is never re-optimised, so a walker
    costs exactly ``steps`` g-xTB optimisations minus the proposals killed by
    the distance floors before they are submitted.

    Acceptance is Metropolis on the *minimised* energies with ``temperature_eV``
    as kT.  0.15 eV is deliberately generous for a 15-35 atom cluster: the point
    is to leave the seed basin, not to sample a canonical ensemble.
    """

    from .molecular_growth import (
        MinimumConsolidation,
        bond_cutoffs_from_spec,
        relaxed_minimum_similarity,
    )

    for move in moves:
        if move not in MOVES:
            raise ValueError(f"unknown move {move!r}; expected from {MOVES}")
    if not moves:
        raise ValueError("basin_hop needs at least one move kind")

    if cutoffs is None:
        cutoffs = bond_cutoffs_from_spec(spec)
    if consolidation is None:
        # Same thresholds the growth path uses to decide two endpoints are the
        # same basin, with one deliberate difference: reflection is allowed.
        # The growth path forbids it because a mirror image is a *different
        # lattice route* and must keep its own lineage.  A search has no
        # lineage -- an enantiomorph is the same isomer at the same energy, and
        # counting it twice inflates "distinct minima found".  Measured: two
        # records identical to 0.1 meV were kept as separate basins without
        # this.
        consolidation = MinimumConsolidation(enabled=True, allow_reflection=True)

    symbols = [str(s) for s in seed["symbols"]]
    current = np.asarray(seed["positions"], dtype=float)
    current_energy = float(seed["energy_eV"])
    seed_id = str(seed["id"])
    rng = np.random.default_rng(int(rng_seed))
    cation = spec.core.cation
    ligand = spec.precursor.ligand

    result = BasinHopResult(
        seed_id=seed_id, k=int(k), p=int(p), seed_energy_eV=current_energy
    )

    # The seed itself is basin zero, so a walker that never escapes still
    # reports something and the rediscovery count is well defined.
    seed_minimum = describe_minimum(
        f"{seed_id}_bh000",
        seed_id,
        0,
        k=k,
        p=p,
        symbols=symbols,
        coordinates=current,
        energy_eV=current_energy,
        converged=True,
        spec=spec,
        cutoffs=cutoffs,
        zb_model=zb_model,
        motif_definitions=motif_definitions,
        artifact_floors=settings.artifact_min_distance,
    )
    result.minima.append(seed_minimum)
    reference_core = seed_minimum.core_edges
    if on_minimum is not None:
        on_minimum(seed_minimum)
    if progress is not None:
        progress(
            f"start  E={current_energy:.6f}  {len(symbols)} atoms  "
            f"{steps} steps  kT={temperature_eV} amp={amplitude_A}"
        )
    started = time.perf_counter()
    known: List[Any] = [_as_parent(seed_minimum, k, p)]

    for step in range(1, int(steps) + 1):
        move = str(moves[int(rng.integers(0, len(moves)))])
        proposal = None
        for _attempt in range(8):
            trial = _perturb(
                symbols,
                current,
                move=move,
                amplitude_A=amplitude_A,
                cation=cation,
                ligand=ligand,
                rng=rng,
            )
            if not _overlaps(trial, overlap_min_A):
                proposal = trial
                break
        if proposal is None:
            result.n_rejected_pre_qc += 1
            continue

        structure_id = f"{seed_id}_bh{step:03d}"
        payload = {
            "id": structure_id,
            "symbols": list(symbols),
            "positions": [list(point) for point in proposal],
            "edges": list(seed_minimum.edges),
        }
        relaxed = relax_structures([payload], settings, cutoffs)[0]
        result.n_relaxations += 1
        result.steps_run = step

        if not relaxed.ok or relaxed.coordinates is None or relaxed.energy_eV is None:
            result.n_failed += 1
            if progress is not None:
                progress(
                    f"{step:4d}/{steps} {move:12s} FAILED "
                    f"({relaxed.error or 'no energy'})"
                )
            continue

        energy = float(relaxed.energy_eV)
        delta = energy - current_energy
        if delta <= 0.0:
            accepted = True
        elif temperature_eV > 0.0:
            accepted = bool(rng.random() < math.exp(-delta / temperature_eV))
        else:
            accepted = False

        candidate = describe_minimum(
            structure_id,
            seed_id,
            step,
            k=k,
            p=p,
            symbols=symbols,
            coordinates=np.asarray(relaxed.coordinates, dtype=float),
            energy_eV=energy,
            converged=bool(relaxed.converged),
            spec=spec,
            cutoffs=cutoffs,
            zb_model=zb_model,
            reference_core_edges=reference_core,
            motif_definitions=motif_definitions,
            artifact_floors=settings.artifact_min_distance,
        )
        parent = _as_parent(candidate, k, p)
        is_new = all(
            relaxed_minimum_similarity(other, parent, consolidation, spec) is None
            for other in known
        )
        if is_new:
            result.minima.append(candidate)
            known.append(parent)
            if on_minimum is not None:
                on_minimum(candidate)

        result.trajectory.append((step, energy, accepted))
        if accepted:
            result.n_accepted += 1
            current = candidate.coordinates
            current_energy = energy

        if progress is not None:
            best = result.best
            progress(
                f"{step:4d}/{steps} {move:12s} E={energy:.6f} "
                f"dE={delta:+7.3f} {'acc' if accepted else 'rej'}"
                f"{' NEW' if is_new else '    '}"
                f"{'' if candidate.core_preserved else ' core!'}"
                f"{'' if candidate.clean else ' dirty'}"
                f"  best={0.0 if best is None else best.energy_eV:.6f}"
                f"  esc={len(result.minima) - 1}"
                f"  {time.perf_counter() - started:.0f}s"
            )

    result.minima.sort(key=lambda item: item.energy_eV)
    return result
