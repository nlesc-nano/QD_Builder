from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Structure

from ..nc_types import (
    CoreMonomerSpec,
    NucleationBridgeRule,
    NucleationGeometryRules,
    NucleationGraphRules,
    NucleationSpec,
    PrecursorUnitSpec,
)

FloatArray = NDArray[np.float64]
NucleationRegistry = Dict[int, Dict[int, List["ClusterRecord"]]]
ProgressCallback = Callable[[str], None]
_StateEnumerationCache = MutableMapping[
    Tuple[object, ...],
    Tuple[Tuple["_State", ...], int, Dict[str, int]],
]

@dataclass(frozen=True)
class AtomRecord:
    """One unrelaxed atom and its historical construction role."""

    atom_id: int
    symbol: str
    coordinates: Tuple[float, float, float]
    role: str
    unit_id: Optional[int] = None


@dataclass
class ClusterRecord:
    """One valid, symmetry-distinct candidate in a physical ``(k, p)`` bin."""

    structure_id: str
    k: int
    p: int
    atoms: List[AtomRecord]
    graph: nx.Graph
    selection_status: str = "unranked"
    selection_reason: str = ""
    coordination_score: Tuple[int, ...] = ()
    source_operations: Tuple[str, ...] = ()
    source_structure_ids: Tuple[str, ...] = ()
    metadata: Dict[str, object] = field(default_factory=dict)
    surface_coordinates_data: Optional[
        Tuple[Tuple[float, float, float], ...]
    ] = None

    @property
    def symbols(self) -> List[str]:
        return [atom.symbol for atom in self.atoms]

    @property
    def coordinates(self) -> FloatArray:
        return np.asarray([atom.coordinates for atom in self.atoms], dtype=float)

    @property
    def surface_coordinates(self) -> FloatArray:
        """Retained-only projected coordinates, or construction coordinates."""

        if self.surface_coordinates_data is None:
            return self.coordinates.copy()
        return np.asarray(self.surface_coordinates_data, dtype=float)

    # Compatibility accessors for early users of the experimental API.
    @property
    def core_isomer_id(self) -> str:
        return f"k{self.k:03d}_p{self.p:03d}"

    @property
    def canonical_lineage(self) -> Tuple[int, int]:
        return (self.k, self.p)

    @property
    def lineage_aliases(self) -> Tuple[Tuple[int, int], ...]:
        return ((self.k, self.p),)

    @property
    def growth_status(self) -> str:
        return str(self.metadata.get("growth_status", "unknown"))

    @property
    def open_growth_sites(self) -> int:
        return int(self.metadata.get("open_growth_sites", 0))


@dataclass
class SweepAudit:
    """Accounting for one generation edge between physical bins."""

    k: int
    operation: str
    p_from: int
    p_to: int
    source_count: int
    raw_count: int
    valid_count: int
    symmetry_duplicate_count: int
    invalid_reasons: Dict[str, int] = field(default_factory=dict)
    stage_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class NucleationResult:
    """Retained and discarded registries plus complete sweep accounting."""

    registry: NucleationRegistry
    discarded_registry: NucleationRegistry = field(default_factory=dict)
    discarded_counts: Dict[int, Dict[int, int]] = field(default_factory=dict)
    sweep_audit: List[SweepAudit] = field(default_factory=list)
    graph_rules: Dict[str, object] = field(default_factory=dict)
    geometry_rules: Dict[str, object] = field(default_factory=dict)
    reference_bond_length: float = 0.0
    completeness: Dict[str, object] = field(default_factory=dict)
    """What this run does and does not claim to have enumerated.

    Serialized into ``registry.json`` and summarised in ``nucleation.log`` so a
    consumer never has to infer completeness from the absence of a warning.
    """

    @property
    def growth_audit(self) -> List[SweepAudit]:
        return [
            audit for audit in self.sweep_audit
            if audit.operation in {"core_growth", "core_skeleton_growth"}
        ]

    @property
    def deduplication_audit(self) -> Tuple[()]:
        return ()

    @property
    def collapse_audit(self) -> Tuple[()]:
        return ()


@dataclass(frozen=True)
class _LatticeModel:
    structure: Structure
    core: CoreMonomerSpec
    environments: Mapping[str, Tuple[Tuple[Tuple[float, float, float], ...], ...]]
    bond_length: float
    site_tolerance: float
    # Hard lower bounds for same-species contacts in construction-native
    # coordinates.  Core homonuclear bounds come from the first same-species
    # shell in the CIF; ligand bounds use a conservative vdW-based fallback.
    same_species_min_distance: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class _State:
    atoms: Tuple[AtomRecord, ...]
    graph: nx.Graph
    geometry_residual: float = 0.0


@dataclass
class _EnumerationCache:
    """Reusable exact-topology work for one nucleation-map generation."""

    states: _StateEnumerationCache = field(default_factory=dict)
    automorphisms: Dict[
        Tuple[Tuple[str, ...], Tuple[Tuple[int, int, str], ...]],
        Tuple[Tuple[int, ...], ...],
    ] = field(default_factory=dict)
    reachable_scores: Dict[
        Tuple[object, ...], Tuple[int, ...]
    ] = field(default_factory=dict)

    def get(
        self, key: Tuple[object, ...]
    ) -> Optional[Tuple[Tuple["_State", ...], int, Dict[str, int]]]:
        return self.states.get(key)

    def __setitem__(
        self,
        key: Tuple[object, ...],
        value: Tuple[Tuple["_State", ...], int, Dict[str, int]],
    ) -> None:
        self.states[key] = value


@dataclass(frozen=True)
class _Vacancy:
    species: str
    position: FloatArray
    hosts: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class _BridgeCandidate:
    """One terminal-ligand reassignment that can create a Cd--Cl--Cd edge."""

    primary: int
    host: int
    rule: NucleationBridgeRule
    mode: str
    shared_neighbor: Optional[int] = None
    virtual_site: Optional[Tuple[float, float, float]] = None
    virtual_hosts: Tuple[int, ...] = ()


@dataclass
class _Generation:
    records: List[ClusterRecord]
    raw_count: int
    invalid_reasons: Dict[str, int]
    stage_counts: Dict[str, int] = field(default_factory=dict)
    greedy_incumbent_score: Optional[Tuple[int, ...]] = None


@dataclass
class _ProgressReporter:
    """Flush-friendly progress reporting shared by nested generation loops."""

    callback: Optional[ProgressCallback] = None
    verbose: bool = False
    interval_seconds: float = 5.0
    _last_heartbeat: float = field(default_factory=time.monotonic)

    def emit(self, message: str, *, verbose_only: bool = False) -> None:
        if self.callback is None or (verbose_only and not self.verbose):
            return
        self.callback(f"[nucleation] {message}")

    def heartbeat(self, message: str) -> None:
        if self.callback is None:
            return
        now = time.monotonic()
        if (
            self.interval_seconds <= 0
            or now - self._last_heartbeat >= self.interval_seconds
        ):
            self._last_heartbeat = now
            self.callback(f"[nucleation] {message}")

__all__ = [
    'AtomRecord',
    'ClusterRecord',
    'SweepAudit',
    'NucleationResult',
    '_LatticeModel',
    '_State',
    '_EnumerationCache',
    '_Vacancy',
    '_BridgeCandidate',
    '_Generation',
    '_ProgressReporter',
    'FloatArray',
    'NucleationRegistry',
    'ProgressCallback',
    '_StateEnumerationCache',
]
