"""Package growth for the lattice-free molecular map.

Two complementary moves (both when ``geometry.start_from: relaxed_coords``)::

  **A graph** — combinatorial precursor-Cd shed on the core graph → monomer
  attach + p_m inflate → Cl redecorate → motif_factor 3D rebuild → full g-xTB.

  **B coord** — parent XYZ → WBO package shed (least-bound CdCl2 first) →
  place CdSe + p_m CdCl2 (embed distances) → optional short cleanup → full
  g-xTB of the carried geometry.

  p_child = p_parent − s + p_m

Chemical-potential / grand-potential numbers are report-only
(see ``formation.py``); they never filter parents or channels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import networkx as nx
import numpy as np
import yaml

from ..nc_types import NucleationSpec
from .formation import (
    MonomerReferences,
    format_bin_ranking,
    format_package_growth_profile,
    load_monomer_references,
)
from .geometry_pack import GeometryPack, load_geometry_pack
from .soft_rules import (
    INDEX_FIELDS as SOFT_INDEX_FIELDS,
    SoftRulesConfig,
    apply_soft_columns,
    describe_graph,
    describe_structure,
)
from .molecular import (
    _index_blocks,
    _symbols_for_composition,
    enumerate_molecular_bin,
    generate_molecular_map,
)
from .molecular_lineage import (
    core_certificate,
    shed_and_grow,
)
from .spec import load_nucleation_spec
from .xtb_relax import relaxed_edges
from .xtb_relax import write_wbo_file as _write_wbo_file

FloatArray = np.ndarray
Edge = Tuple[int, int]
EdgeList = Tuple[Edge, ...]
Vec3 = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def p_surf_capacity(k: int, beta: float) -> int:
    """Quasi-spherical surface excess: floor(β · k^{2/3}).

    Same algebra as ``engine._p_surf`` (lattice nucleation).  ``beta <= 0``
    disables the surface law (return 0).
    """

    if int(k) <= 0 or float(beta) <= 0.0:
        return 0
    return int(math.floor(float(beta) * (float(k) ** (2.0 / 3.0))))


def effective_s_max(
    k: int,
    p: int,
    *,
    beta: float,
    alpha: float,
    hard: int,
) -> int:
    """Packages removable at parent size k.

    ``s_max = min(p, floor(α · p_surf(k)), hard)`` when β > 0; otherwise
    ``min(p, hard)``.  ``hard`` is the YAML ``shed.max_shed`` cap.
    """

    p = max(0, int(p))
    hard = max(0, int(hard))
    if p <= 0 or hard <= 0:
        return 0
    if float(beta) <= 0.0:
        return min(p, hard)
    surface = p_surf_capacity(k, beta)
    s_max = min(p, int(math.floor(max(0.0, float(alpha)) * surface)), hard)
    return max(0, s_max)


@dataclass(frozen=True)
class LatticeSwitch:
    """When a parent has *local* tetrahedral holes, decorate with tet_sites.

    tet_sites ≠ zinc-blende.  A magic-size cluster can be locally tetrahedral
    (vacant tet directions on Cd, CN3/CN4 Se) without long-range zb order.
    Zinc-blende / Wulff is a later hop to the CIF lattice engine (k≈13+).
    """

    enabled: bool = False
    from_k: int = 7
    decoration_mode: str = "tet_sites"
    fallback: str = "motif_bridge_target"
    min_cn4_fraction: Optional[float] = None
    min_six_rings: Optional[int] = None
    max_core_rmsd_to_zb_A: Optional[float] = None

    @classmethod
    def from_raw(cls, raw: Any) -> "LatticeSwitch":
        if not isinstance(raw, dict) or not raw:
            return cls()
        cn4 = six = rmsd = None
        require = raw.get("require_any") or []
        if isinstance(require, dict):
            require = [require]
        for item in require:
            if not isinstance(item, dict):
                continue
            if "min_cn4_fraction" in item:
                cn4 = float(item["min_cn4_fraction"])
            if "min_six_rings" in item:
                six = int(item["min_six_rings"])
            if "max_core_rmsd_to_zb_A" in item:
                rmsd = float(item["max_core_rmsd_to_zb_A"])
        if "min_cn4_fraction" in raw:
            cn4 = float(raw["min_cn4_fraction"])
        if "min_six_rings" in raw:
            six = int(raw["min_six_rings"])
        if "max_core_rmsd_to_zb_A" in raw:
            rmsd = float(raw["max_core_rmsd_to_zb_A"])
        return cls(
            enabled=bool(raw.get("enabled", False)),
            from_k=int(raw.get("from_k", 7)),
            decoration_mode=str(raw.get("decoration_mode", "tet_sites")),
            fallback=str(raw.get("fallback", "motif_bridge_target")),
            min_cn4_fraction=cn4,
            min_six_rings=six,
            max_core_rmsd_to_zb_A=rmsd,
        )


@dataclass(frozen=True)
class MinimumConsolidation:
    """Composite definition of one relaxed structural basin.

    Cartesian RMSD is evaluated only after graph-constrained permutation and
    optimal proper rotation.  Internal distances, final topology, energy, and
    optional WBO similarity are independent guards against false merging.
    """

    enabled: bool = False
    energy_tolerance_eV: float = 0.01
    pair_distance_rms_A: float = 0.05
    core_rmsd_A: float = 0.10
    full_rmsd_A: float = 0.15
    max_displacement_A: float = 0.25
    wbo_rms_tolerance: float = 0.15
    require_wbo_when_available: bool = False
    allow_reflection: bool = False
    max_graph_mappings: int = 256
    max_minima_per_occupation: int = 2
    max_occupations_per_minimum: int = 0

    @classmethod
    def from_raw(cls, raw: Any) -> "MinimumConsolidation":
        if not isinstance(raw, dict) or not raw:
            return cls()
        return cls(
            enabled=bool(raw.get("enabled", False)),
            energy_tolerance_eV=float(raw.get("energy_tolerance_eV", 0.01)),
            pair_distance_rms_A=float(raw.get("pair_distance_rms_A", 0.05)),
            core_rmsd_A=float(raw.get("core_rmsd_A", 0.10)),
            full_rmsd_A=float(raw.get("full_rmsd_A", 0.15)),
            max_displacement_A=float(raw.get("max_displacement_A", 0.25)),
            wbo_rms_tolerance=float(raw.get("wbo_rms_tolerance", 0.15)),
            require_wbo_when_available=bool(
                raw.get("require_wbo_when_available", False)
            ),
            allow_reflection=bool(raw.get("allow_reflection", False)),
            max_graph_mappings=max(1, int(raw.get("max_graph_mappings", 256))),
            max_minima_per_occupation=max(
                1, int(raw.get("max_minima_per_occupation", 2))
            ),
            max_occupations_per_minimum=max(
                0, int(raw.get("max_occupations_per_minimum", 0))
            ),
        )


@dataclass(frozen=True)
class GrowthWindow:
    """Resolved envelope for one parent k (top-level YAML + matching ``by_k``)."""

    k_from: int
    monomer_p_values: Tuple[int, ...]
    max_shed: int
    attach: str
    energy_window_eV: float
    max_skeletons_frac: float
    max_skeletons_cap: int
    decorations_per_skeleton: int
    max_children_per_channel: int
    move_graph: bool
    move_coord: bool
    move_zb_sites: bool
    child_redecorate: bool
    child_redecorate_slack: bool
    selection_max_per_skeleton: int
    surface_beta: float
    surface_alpha: float
    p_slack: int
    persist_wbo: bool
    shed_mode: str
    prefer_low_shed: bool
    max_opts_per_k: int
    min_p_parent: int = 1
    lattice: LatticeSwitch = field(default_factory=LatticeSwitch)
    soft_rules: SoftRulesConfig = field(default_factory=SoftRulesConfig)

    def p_surf(self, k: int) -> int:
        return p_surf_capacity(k, self.surface_beta)

    def s_max_for(self, k: int, p: int) -> int:
        return effective_s_max(
            k,
            p,
            beta=self.surface_beta,
            alpha=self.surface_alpha,
            hard=self.max_shed,
        )

    def allow_p_child(self, k_child: int, p_child: int) -> bool:
        if self.surface_beta <= 0.0:
            return True
        return int(p_child) <= self.p_surf(k_child) + int(self.p_slack)

    def allow_redecorate(self, k_child: int, p_child: int) -> bool:
        """Move A on this (k, p)?  Slack bins (p > p_surf) can stay B-only."""

        if not self.child_redecorate:
            return False
        if self.child_redecorate_slack or self.surface_beta <= 0.0:
            return True
        return int(p_child) <= self.p_surf(int(k_child))

    def describe(self) -> str:
        cap = (
            f"p_surf({self.k_from})={self.p_surf(self.k_from)} "
            f"p_child≤p_surf({self.k_from + 1})+{self.p_slack}"
            f"={self.p_surf(self.k_from + 1) + self.p_slack}"
            if self.surface_beta > 0.0
            else "p_surf=off"
        )
        return (
            f"k={self.k_from}: p_m={list(self.monomer_p_values)} "
            f"max_shed={self.max_shed} attach={self.attach} "
            f"β={self.surface_beta} α={self.surface_alpha} {cap} "
            f"moves A={self.move_graph} B={self.move_coord} "
            f"Z={self.move_zb_sites} "
            f"redecorate={self.child_redecorate} "
            f"A_slack={'on' if self.child_redecorate_slack else 'off'} "
            f"shed={self.shed_mode} "
            f"min_p={self.min_p_parent} "
            f"soft={'on' if self.soft_rules.enabled else 'off'}"
        )


@dataclass
class GrowthConfig:
    """Parsed ``growth.yaml`` (growth path only)."""

    raw: Dict[str, Any]
    monomer_p_values: Tuple[int, ...] = (1, 2, 3)
    references: Optional[MonomerReferences] = None
    energy_window_eV: float = 1.0
    rank_by: str = "energy"
    max_skeletons_frac: float = 0.30
    max_skeletons_cap: int = 25
    decorations_per_skeleton: int = 2
    shed_mode: str = "distance"
    max_shed: int = 2
    prefer_low_shed: bool = True
    persist_wbo: bool = True
    max_children_per_channel: int = 500
    attach: str = "enumerate"
    move_graph: bool = True
    move_coord: bool = True
    move_zb_sites: bool = False
    surface_beta: float = 0.0
    surface_alpha: float = 1.0
    p_slack: int = 1
    selection_max_per_skeleton: int = 0
    max_opts_per_k: int = 0
    lattice: LatticeSwitch = field(default_factory=LatticeSwitch)
    by_k: Tuple[Dict[str, Any], ...] = ()
    # geometry / move B (coordinate carry-over)
    start_from: str = "relaxed_coords"  # relaxed_coords | graph_only
    place_monomer: str = "embed_tables"
    clash_policy: str = "soft"
    local_cleanup_enabled: bool = True
    local_cleanup_method: str = "g-xTB"
    local_cleanup_cycles: int = 20
    require_charge_neutral_for_cleanup: bool = True
    child_redecorate: bool = True
    child_redecorate_slack: bool = True
    child_full_opt: str = "g-xTB"
    child_full_opt_cycles: int = 150
    delta_mu_cdcl2_eV: Tuple[float, ...] = ()
    min_p_parent: int = 1
    soft_rules: SoftRulesConfig = field(default_factory=SoftRulesConfig)
    endpoint_diagnostic_k: int = 0
    endpoint_reference: Optional[Path] = None
    endpoint_match_tolerance_A: float = 0.35
    minimum_consolidation: MinimumConsolidation = field(
        default_factory=MinimumConsolidation
    )

    @property
    def use_coord_carry(self) -> bool:
        """True when move B (3D carry-over) should run."""

        if not self.move_coord:
            return False
        return str(self.start_from).lower() in {
            "relaxed_coords",
            "relaxed",
            "coords",
            "coordinate",
            "coordinates",
        }

    def window_for(self, k: int) -> GrowthWindow:
        """Top-level settings overlaid with the first matching ``by_k`` block."""

        k = int(k)
        overlay: Dict[str, Any] = {}
        for block in self.by_k:
            k_min = int(block.get("k_min", 1))
            k_max = int(block.get("k_max", k_min))
            if k_min <= k <= k_max:
                overlay = dict(block)
                break
        p_vals = overlay.get("monomer_p_values", self.monomer_p_values)
        moves = overlay.get("moves")
        move_graph = self.move_graph
        move_coord = self.move_coord
        move_zb_sites = self.move_zb_sites
        if isinstance(moves, dict):
            move_graph = bool(moves.get("graph", move_graph))
            move_coord = bool(moves.get("coord", move_coord))
            move_zb_sites = bool(moves.get("zb_sites", move_zb_sites))
        elif "move_graph" in overlay:
            move_graph = bool(overlay["move_graph"])
        elif "move_coord" in overlay:
            move_coord = bool(overlay["move_coord"])
        if "move_zb_sites" in overlay:
            move_zb_sites = bool(overlay["move_zb_sites"])
        child = overlay.get("child")
        redecorate = self.child_redecorate
        redecorate_slack = self.child_redecorate_slack
        if isinstance(child, dict):
            if "redecorate" in child:
                redecorate = bool(child["redecorate"])
            if "redecorate_slack" in child:
                redecorate_slack = bool(child["redecorate_slack"])
        if "redecorate" in overlay:
            redecorate = bool(overlay["redecorate"])
        if "redecorate_slack" in overlay:
            redecorate_slack = bool(overlay["redecorate_slack"])
        soft = self.soft_rules.merged_with(overlay.get("soft_rules"))
        return GrowthWindow(
            k_from=k,
            monomer_p_values=tuple(int(x) for x in p_vals),
            max_shed=int(overlay.get("max_shed", self.max_shed)),
            attach=str(overlay.get("attach", self.attach)).lower(),
            energy_window_eV=float(
                overlay.get("energy_window_eV", self.energy_window_eV)
            ),
            max_skeletons_frac=float(
                overlay.get("max_skeletons_frac", self.max_skeletons_frac)
            ),
            max_skeletons_cap=int(
                overlay.get("max_skeletons_cap", self.max_skeletons_cap)
            ),
            decorations_per_skeleton=int(
                overlay.get(
                    "decorations_per_skeleton", self.decorations_per_skeleton
                )
            ),
            max_children_per_channel=int(
                overlay.get(
                    "max_children_per_channel", self.max_children_per_channel
                )
            ),
            move_graph=move_graph,
            move_coord=move_coord,
            move_zb_sites=move_zb_sites,
            child_redecorate=redecorate,
            child_redecorate_slack=redecorate_slack,
            selection_max_per_skeleton=int(
                overlay.get(
                    "selection_max_per_skeleton",
                    self.selection_max_per_skeleton,
                )
            ),
            surface_beta=float(overlay.get("surface_beta", self.surface_beta)),
            surface_alpha=float(
                overlay.get("surface_alpha", self.surface_alpha)
            ),
            p_slack=int(overlay.get("p_slack", self.p_slack)),
            persist_wbo=bool(overlay.get("persist_wbo", self.persist_wbo)),
            shed_mode=str(overlay.get("shed_mode", self.shed_mode)).lower(),
            prefer_low_shed=bool(
                overlay.get("prefer_low_shed", self.prefer_low_shed)
            ),
            max_opts_per_k=int(
                overlay.get("max_opts_per_k", self.max_opts_per_k)
            ),
            min_p_parent=int(
                overlay.get("min_p", overlay.get("min_p_parent", self.min_p_parent))
            ),
            lattice=self.lattice,
            soft_rules=soft,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GrowthConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"growth.yaml must be a mapping: {path}")
        parents = raw.get("parents") or {}
        shed = raw.get("shed") or {}
        geom = raw.get("geometry") or {}
        if not isinstance(geom, dict):
            geom = {}
        cleanup = geom.get("local_cleanup") or {}
        if not isinstance(cleanup, dict):
            cleanup = {}
        child = raw.get("child") or {}
        if not isinstance(child, dict):
            child = {}
        mu = raw.get("chemical_potential") or {}
        surface = raw.get("surface") or {}
        if not isinstance(surface, dict):
            surface = {}
        moves = raw.get("moves") or {}
        if not isinstance(moves, dict):
            moves = {}
        budget = raw.get("budget") or {}
        if not isinstance(budget, dict):
            budget = {}
        endpoint = raw.get("endpoint_diagnostic") or {}
        if not isinstance(endpoint, dict):
            endpoint = {}
        consolidation = MinimumConsolidation.from_raw(
            raw.get("minimum_consolidation")
        )
        endpoint_reference = None
        if endpoint.get("enabled", False) and endpoint.get("reference"):
            endpoint_reference = Path(str(endpoint["reference"]))
            if not endpoint_reference.is_absolute():
                endpoint_reference = (path.parent / endpoint_reference).resolve()
        refs = None
        if raw.get("references"):
            try:
                refs = load_monomer_references(raw["references"])
            except ValueError:
                refs = None
        p_vals = raw.get("monomer_p_values") or [1, 2, 3]
        dmu = ()
        if mu.get("enabled", True):
            dmu = tuple(float(x) for x in (mu.get("delta_mu_cdcl2_eV") or ()))
        by_k_raw = raw.get("by_k") or []
        if not isinstance(by_k_raw, list):
            raise ValueError("growth.yaml by_k must be a list of mappings")
        by_k = tuple(dict(b) for b in by_k_raw if isinstance(b, dict))
        attach = str(raw.get("attach", "enumerate")).lower()
        if attach not in {"local", "enumerate"}:
            raise ValueError(
                f"growth.yaml attach must be 'local' or 'enumerate', got {attach!r}"
            )
        shed_mode = str(shed.get("mode", "distance")).lower()
        if shed_mode in {"wbo", "enumerate"}:
            pass
        elif shed_mode not in {"distance", "gfn1_wbo"}:
            raise ValueError(
                f"growth.yaml shed.mode must be distance|wbo|enumerate|gfn1_wbo, "
                f"got {shed_mode!r}"
            )
        return cls(
            raw=raw,
            monomer_p_values=tuple(int(x) for x in p_vals),
            references=refs,
            energy_window_eV=float(parents.get("energy_window_eV", 1.0)),
            rank_by=str(parents.get("rank_by", "energy")).lower(),
            max_skeletons_frac=float(parents.get("max_skeletons_frac", 0.30)),
            max_skeletons_cap=int(parents.get("max_skeletons_cap", 25)),
            decorations_per_skeleton=int(
                parents.get("decorations_per_skeleton", 2)
            ),
            min_p_parent=int(parents.get("min_p", parents.get("min_p_parent", 1))),
            shed_mode=shed_mode,
            max_shed=int(shed.get("max_shed", 2)),
            prefer_low_shed=bool(shed.get("prefer_low_shed", True)),
            persist_wbo=bool(shed.get("persist_wbo", True)),
            max_children_per_channel=int(
                raw.get("max_children_per_channel", 500)
            ),
            attach=attach,
            move_graph=bool(moves.get("graph", True)),
            move_coord=bool(moves.get("coord", True)),
            move_zb_sites=bool(moves.get("zb_sites", False)),
            surface_beta=float(surface.get("beta", 0.0)),
            surface_alpha=float(surface.get("alpha", 1.0)),
            p_slack=int(surface.get("p_slack", 1)),
            selection_max_per_skeleton=int(
                raw.get("selection_max_per_skeleton", 0)
            ),
            max_opts_per_k=int(budget.get("max_opts_per_k", 0)),
            lattice=LatticeSwitch.from_raw(raw.get("lattice_switch")),
            by_k=by_k,
            start_from=str(geom.get("start_from", "relaxed_coords")),
            place_monomer=str(geom.get("place_monomer", "embed_tables")),
            clash_policy=str(geom.get("clash_policy", "soft")),
            local_cleanup_enabled=bool(cleanup.get("enabled", True)),
            local_cleanup_method=str(cleanup.get("method", "g-xTB")),
            local_cleanup_cycles=int(cleanup.get("max_cycles", 20)),
            require_charge_neutral_for_cleanup=bool(
                geom.get("require_charge_neutral_for_cleanup", True)
            ),
            child_redecorate=bool(child.get("redecorate", True)),
            child_redecorate_slack=bool(child.get("redecorate_slack", True)),
            child_full_opt=str(child.get("full_opt", "g-xTB")),
            child_full_opt_cycles=int(
                child.get(
                    "full_opt_cycles",
                    child.get("max_cycles", 150),
                )
            ),
            delta_mu_cdcl2_eV=dmu,
            soft_rules=SoftRulesConfig.from_raw(raw.get("soft_rules")),
            endpoint_diagnostic_k=(
                int(endpoint.get("k", 13))
                if endpoint.get("enabled", False)
                else 0
            ),
            endpoint_reference=endpoint_reference,
            endpoint_match_tolerance_A=float(
                endpoint.get("site_match_tolerance_A", 0.35)
            ),
            minimum_consolidation=consolidation,
        )


# ---------------------------------------------------------------------------
# Parent structures
# ---------------------------------------------------------------------------


@dataclass
class ParentStructure:
    """One relaxed parent isomer used as a growth source."""

    k: int
    p: int
    structure_id: str
    symbols: Tuple[str, ...]
    coordinates: FloatArray  # (n, 3)
    energy_eV: float
    edges: EdgeList  # distance-inferred full graph
    core_edges: EdgeList  # Cd–Se only
    wbo: Optional[Dict[Tuple[int, int], float]] = None
    wbo_source: str = "none"  # wbo_file | csv | distance | none
    source_path: str = ""
    # Move-Z dual representation.  The relaxed XYZ supplies energy/feedback;
    # this stored occupation is the sole source of lattice lineage.
    zb_occupation: Optional[Any] = None
    # Relaxed-basin consolidation metadata.  A selected basin may yield more
    # than one ParentStructure when distinct ZB occupations reached the same
    # minimum; each route keeps its own atom correspondence and WBO matrix.
    minimum_id: str = ""
    minimum_representative_id: str = ""
    minimum_member_ids: Tuple[str, ...] = ()
    minimum_occupation_ids: Tuple[str, ...] = ()
    minimum_multiplicity: int = 1

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)


@dataclass
class GrowthChannelResult:
    """One (parent, s, p_m) channel → child cores at (k+1, p_child)."""

    parent_id: str
    k_parent: int
    p_parent: int
    shed: int
    p_m: int
    k_child: int
    p_child: int
    n_cores: int
    core_edges: List[EdgeList] = field(default_factory=list)
    move: str = "graph"  # graph | coord


@dataclass
class CoordSeed:
    """One coordinate-carried child ready for cleanup / full opt (move B)."""

    k: int
    p: int
    structure_id: str
    parent_id: str
    shed: int
    p_m: int
    symbols: Tuple[str, ...]
    coordinates: FloatArray
    core_edges: EdgeList
    wbo_scores: Tuple[float, ...] = ()  # scores of packages shed (ascending)
    cleanup_s: float = 0.0
    cleanup_ok: bool = False
    notes: str = ""


@dataclass
class RankedIsomer:
    """Lightweight energy row for raw or consolidated bin rankings."""

    structure_id: str
    xtb_energy_eV: float
    seed_skeleton: str = "------"
    growth_move: str = "?"  # A | B
    parent_id: str = ""
    minimum_id: str = ""
    minimum_multiplicity: int = 1


@dataclass
class GrowthStepResult:
    """Outcome of growing all parents from k → k+1."""

    k_from: int
    k_to: int
    parents_selected: int
    channels: List[GrowthChannelResult]
    skeleton_catalog: Dict[Tuple[int, int], List[EdgeList]]
    parent_records: List[Dict[str, Any]] = field(default_factory=list)
    #: move-B coordinate seeds keyed by (k_child, p_child)
    coord_seeds: Dict[Tuple[int, int], List[CoordSeed]] = field(
        default_factory=dict
    )
    #: move-Z zb occupations keyed by (k_child, p_child)
    zb_seeds: Dict[Tuple[int, int], List[Any]] = field(default_factory=dict)
    zb_stats: Any = None


def bond_cutoffs_from_spec(spec: NucleationSpec) -> Dict[Tuple[str, str], float]:
    """Cd–Se / Cd–Cl bond_max_distance pairs for distance graphs."""

    cut: Dict[Tuple[str, str], float] = {}
    for key, rule in (spec.graph_rules.pair_rules or {}).items():
        if not getattr(rule, "bond_allowed", False):
            continue
        dmax = getattr(rule, "bond_max_distance", None)
        if dmax is None:
            continue
        elems = tuple(sorted(rule.elements))
        cut[elems] = float(dmax)
    # fallbacks matching production pack
    cut.setdefault(tuple(sorted((spec.core.cation, spec.core.anion))), 3.25)
    cut.setdefault(
        tuple(sorted((spec.precursor.center, spec.precursor.ligand))), 2.90
    )
    return cut


def parse_xyz(path: Path) -> Tuple[List[str], FloatArray, Dict[str, str]]:
    """Read XYZ; comment line key=value pairs into meta."""

    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    comment = lines[1] if len(lines) > 1 else ""
    meta: Dict[str, str] = {}
    for m in re.finditer(r"(\w+)=([^\s]+)", comment):
        meta[m.group(1)] = m.group(2)
    meta["_comment"] = comment
    symbols: List[str] = []
    coords: List[List[float]] = []
    for row in lines[2 : 2 + n]:
        parts = row.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.asarray(coords, dtype=float), meta


def parse_wbo(path: Path) -> Dict[Tuple[int, int], float]:
    """Parse xtb ``wbo`` file (1-based indices)."""

    out: Dict[Tuple[int, int], float] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            i, j, w = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
        except ValueError:
            continue
        out[(min(i, j), max(i, j))] = w
    return out


def write_wbo_file(
    path: Path,
    bond_orders: Sequence[Sequence[float]],
    *,
    threshold: float = 0.05,
) -> None:
    """Write an xtb-style 1-based ``wbo`` file next to a relaxed XYZ."""

    _write_wbo_file(path, bond_orders, threshold=threshold)


def _wbo_from_bond_orders_csv(
    csv_path: Path, structure_id: str
) -> Dict[Tuple[int, int], float]:
    """Parse ``xtb_bond_orders.csv`` rows for one structure id."""

    out: Dict[Tuple[int, int], float] = {}
    if not csv_path.is_file():
        return out
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("structure_id") or "") != structure_id:
                continue
            try:
                i, j = int(row["left"]), int(row["right"])
                w = float(row.get("wiberg") or row.get("order") or "")
            except (KeyError, ValueError, TypeError):
                continue
            out[(min(i, j), max(i, j))] = w
    return out


def load_parent_wbo(
    xyz_path: Path,
    *,
    structure_id: str,
    run_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[Tuple[int, int], float]], str]:
    """WBO for a parent: ``*.wbo`` next to XYZ, then map CSV, else none.

    g-xTB does not write Wiberg files; GFN-xTB ``wbo`` and the map dump
    ``xtb_bond_orders.csv`` are the two real sources.
    """

    candidates = [
        xyz_path.with_suffix(".wbo"),
        xyz_path.with_name(xyz_path.stem + ".wbo"),
        xyz_path.with_name("wbo"),
        xyz_path.parent / "wbo",
    ]
    for cand in candidates:
        parsed = parse_wbo(cand)
        if parsed:
            return parsed, "wbo_file"
    search_dirs = []
    if run_dir is not None:
        search_dirs.append(Path(run_dir))
    search_dirs.append(xyz_path.parent)
    if xyz_path.parent.parent != xyz_path.parent:
        search_dirs.append(xyz_path.parent.parent)
        if xyz_path.parent.parent.parent != xyz_path.parent.parent:
            search_dirs.append(xyz_path.parent.parent.parent)
    seen: set = set()
    for folder in search_dirs:
        key = str(folder)
        if key in seen:
            continue
        seen.add(key)
        parsed = _wbo_from_bond_orders_csv(folder / "xtb_bond_orders.csv", structure_id)
        if parsed:
            return parsed, "csv"
    return None, "none"


def core_edges_from_full(
    symbols: Sequence[str],
    edges: Sequence[Edge],
    *,
    cation: str,
    anion: str,
) -> EdgeList:
    """Keep only cation–anion bonds."""

    out = []
    for a, b in edges:
        pair = {symbols[a], symbols[b]}
        if pair == {cation, anion}:
            out.append((min(a, b), max(a, b)))
    return tuple(sorted(out))


@dataclass
class EnergyRecord:
    """Lightweight isomer energy for ranking / profile (no coordinates)."""

    structure_id: str
    xtb_energy_eV: float
    k: int
    p: int
    xtb_converged: bool = True


def load_energy_index(
    run_dir: Path,
    *,
    k_values: Optional[Sequence[int]] = None,
    require_converged: bool = True,
) -> List[EnergyRecord]:
    """Load g-xTB energies from a map/growth ``index.csv`` (all or selected k)."""

    run_dir = Path(run_dir)
    index_path = run_dir / "index.csv"
    if not index_path.is_file():
        return []
    k_set = None if k_values is None else {int(x) for x in k_values}
    out: List[EnergyRecord] = []
    with index_path.open() as handle:
        for row in csv.DictReader(handle):
            try:
                rk, rp = int(row["k"]), int(row["p"])
            except (KeyError, ValueError):
                continue
            if k_set is not None and rk not in k_set:
                continue
            conv = str(row.get("xtb_converged", "")).lower().strip()
            if require_converged and conv in ("false", "0", "no"):
                continue
            try:
                energy = float(row["xtb_energy_eV"])
            except (KeyError, ValueError, TypeError):
                continue
            if not math.isfinite(energy):
                continue
            sid = str(row.get("structure_id") or f"k{rk:03d}_p{rp:03d}")
            out.append(
                EnergyRecord(
                    structure_id=sid,
                    xtb_energy_eV=energy,
                    k=rk,
                    p=rp,
                    xtb_converged=conv not in ("false", "0", "no"),
                )
            )
    return out


def bin_minima_from_records(
    records: Sequence[EnergyRecord],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Map (k, p) → {energy_eV, structure_id} for the lowest-E isomer."""

    best: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for rec in records:
        key = (int(rec.k), int(rec.p))
        e = float(rec.xtb_energy_eV)
        cur = best.get(key)
        if cur is None or e < float(cur["energy_eV"]):
            best[key] = {
                "energy_eV": e,
                "structure_id": rec.structure_id,
            }
    return best


def merge_bin_minima(
    *maps: Mapping[Tuple[int, int], Mapping[str, Any]],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Merge several (k,p)→min maps; keep the lower energy when both present.

    Near-degenerate energies prefer a real structure id over ``ref:…``.
    """

    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for m in maps:
        for key, row in m.items():
            e = row.get("energy_eV")
            if e is None:
                continue
            e = float(e)
            sid = str(row.get("structure_id") or "")
            cur = out.get(key)
            if cur is None:
                out[key] = {"energy_eV": e, "structure_id": sid}
                continue
            cur_e = float(cur["energy_eV"])
            if e < cur_e - 1.0e-9:
                out[key] = {"energy_eV": e, "structure_id": sid}
            elif abs(e - cur_e) <= 1.0e-9:
                cur_sid = str(cur.get("structure_id") or "")
                if cur_sid.startswith("ref:") and not sid.startswith("ref:"):
                    out[key] = {"energy_eV": e, "structure_id": sid}
    return out


def seed_package_minima_from_refs(
    refs: Optional[MonomerReferences],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """k=1 package energies from growth.yaml ``package_cluster_eV``."""

    if refs is None:
        return {}
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for pm, e in refs.package_cluster_eV.items():
        out[(1, int(pm))] = {
            "energy_eV": float(e),
            "structure_id": f"ref:E(1,{int(pm)})",
        }
    return out


def format_prior_map_rankings(
    run_dir: Path,
    *,
    growth: GrowthConfig,
    k_max: int,
) -> str:
    """Full bin rankings for parent map sizes k=1 … k_max (same log format)."""

    records = load_energy_index(run_dir, k_values=range(1, int(k_max) + 1))
    if not records:
        return f"  (no index energies in {run_dir} for k≤{k_max})"
    by_bin: Dict[Tuple[int, int], List[EnergyRecord]] = {}
    for rec in records:
        by_bin.setdefault((rec.k, rec.p), []).append(rec)
    pms = tuple(growth.monomer_p_values) or (1, 2, 3)
    dmu = tuple(growth.delta_mu_cdcl2_eV) or (-1.0, 0.0, 1.0)
    chunks = [
        f"  ══ prior map rankings from {run_dir.name}  "
        f"(k=1…{k_max}; g-xTB already relaxed) ══"
    ]
    for (k, p) in sorted(by_bin):
        chunks.append(
            format_bin_ranking(
                by_bin[(k, p)],
                k=k,
                p=p,
                refs=growth.references,
                package_p_m=pms,
                delta_mu=dmu,
            )
        )
    return "\n\n".join(chunks)


def parent_k_inventory(run_dir: Path) -> Dict[int, int]:
    """Converged parent counts by k in a finished run (index.csv, else xyz)."""

    run_dir = Path(run_dir)
    counts: Dict[int, int] = {}
    index_path = run_dir / "index.csv"
    if index_path.is_file():
        with index_path.open() as handle:
            for row in csv.DictReader(handle):
                try:
                    rk = int(row["k"])
                except (KeyError, ValueError):
                    continue
                conv = str(row.get("xtb_converged", "true")).lower().strip()
                if conv in ("false", "0", "no"):
                    continue
                counts[rk] = counts.get(rk, 0) + 1
        if counts:
            return dict(sorted(counts.items()))
    for kdir in sorted(run_dir.glob("k[0-9][0-9][0-9]")):
        try:
            rk = int(kdir.name[1:])
        except ValueError:
            continue
        n = sum(1 for _ in kdir.glob("p*/*_xtb.xyz"))
        if n:
            counts[rk] = n
    return dict(sorted(counts.items()))


def load_parents_from_run(
    run_dir: Path,
    *,
    k: int,
    spec: NucleationSpec,
    p_values: Optional[Sequence[int]] = None,
) -> List[ParentStructure]:
    """Load converged relaxed parents for fixed k from a map run directory."""

    run_dir = Path(run_dir)
    cutoffs = bond_cutoffs_from_spec(spec)
    cation = spec.core.cation
    anion = spec.core.anion
    parents: List[ParentStructure] = []
    try:
        from .molecular_zb_growth import load_occupation_manifest

        zb_by_structure = load_occupation_manifest(run_dir)
    except Exception:
        zb_by_structure = {}

    index_path = run_dir / "index.csv"
    rows: List[Dict[str, str]] = []
    if index_path.is_file():
        with index_path.open() as handle:
            rows = list(csv.DictReader(handle))

    if rows:
        for row in rows:
            try:
                rk, rp = int(row["k"]), int(row["p"])
            except (KeyError, ValueError):
                continue
            if rk != k:
                continue
            if p_values is not None and rp not in set(p_values):
                continue
            if str(row.get("xtb_converged", "")).lower() != "true":
                continue
            try:
                energy = float(row["xtb_energy_eV"])
            except (KeyError, ValueError):
                continue
            sid = row.get("structure_id") or ""
            xyz_rel = row.get("xtb_xyz") or row.get("xyz") or ""
            candidates = []
            if xyz_rel:
                candidates.append(run_dir / xyz_rel)
                candidates.append(Path(xyz_rel))
            candidates.append(
                run_dir / f"k{k:03d}" / f"p{rp:03d}" / f"{sid}_xtb.xyz"
            )
            xyz_path = next((p for p in candidates if p.is_file()), None)
            if xyz_path is None:
                continue
            symbols, coords, _meta = parse_xyz(xyz_path)
            edges = tuple(
                sorted(
                    (min(a, b), max(a, b))
                    for a, b in relaxed_edges(symbols, coords, cutoffs)
                )
            )
            core = core_edges_from_full(
                symbols, edges, cation=cation, anion=anion
            )
            sid_use = sid or xyz_path.stem
            wbo, wbo_src = load_parent_wbo(
                xyz_path, structure_id=sid_use, run_dir=run_dir
            )
            parents.append(
                ParentStructure(
                    k=k,
                    p=rp,
                    structure_id=sid_use,
                    symbols=tuple(symbols),
                    coordinates=coords,
                    energy_eV=energy,
                    edges=edges,
                    core_edges=core,
                    wbo=wbo,
                    wbo_source=wbo_src if wbo else "none",
                    source_path=str(xyz_path),
                    zb_occupation=zb_by_structure.get(sid_use),
                )
            )
        return parents

    # Fallback: scan k###/p###/*_xtb.xyz
    kdir = run_dir / f"k{k:03d}"
    if not kdir.is_dir():
        return []
    for pdir in sorted(kdir.glob("p*")):
        try:
            rp = int(pdir.name[1:])
        except ValueError:
            continue
        if p_values is not None and rp not in set(p_values):
            continue
        for xyz_path in sorted(pdir.glob("*_xtb.xyz")):
            symbols, coords, meta = parse_xyz(xyz_path)
            if meta.get("xtb_converged", "true").lower() == "false":
                continue
            try:
                energy = float(meta.get("energy_eV", "nan"))
            except ValueError:
                continue
            if not math.isfinite(energy):
                continue
            edges = tuple(
                sorted(
                    (min(a, b), max(a, b))
                    for a, b in relaxed_edges(symbols, coords, cutoffs)
                )
            )
            core = core_edges_from_full(
                symbols, edges, cation=cation, anion=anion
            )
            sid_use = xyz_path.stem.replace("_xtb", "")
            wbo, wbo_src = load_parent_wbo(
                xyz_path, structure_id=sid_use, run_dir=run_dir
            )
            parents.append(
                ParentStructure(
                    k=k,
                    p=rp,
                    structure_id=sid_use,
                    symbols=tuple(symbols),
                    coordinates=coords,
                    energy_eV=energy,
                    edges=edges,
                    core_edges=core,
                    wbo=wbo,
                    wbo_source=wbo_src if wbo else "none",
                    source_path=str(xyz_path),
                    zb_occupation=zb_by_structure.get(sid_use),
                )
            )
    return parents


def _core_fingerprint(
    parent: ParentStructure, spec: NucleationSpec
) -> Tuple[object, ...]:
    """Isomorphism class of the relaxed Cd–Se core."""

    occupation = getattr(parent, "zb_occupation", None)
    occupation_id = str(getattr(occupation, "occupation_id", "") or "")
    if occupation_id:
        # The ID is based on the complete coloured lattice site set modulo
        # cubic symmetry and translation, so it preserves spatial occupation
        # diversity that a Cd--Se edge graph alone cannot distinguish.
        return ("zb", occupation_id, parent.k, parent.p)

    se_ids, cd_ids, _ = _index_blocks(parent.k, parent.p)
    # Map only inorganic nodes present in core edges
    nodes = sorted({n for e in parent.core_edges for n in e})
    if not nodes:
        return ("empty", parent.k, parent.p)
    # Relabel to dense 0..n for certificate via networkx
    g = nx.Graph()
    for i, node in enumerate(nodes):
        sym = parent.symbols[node]
        g.add_node(i, element=sym)
    idx = {node: i for i, node in enumerate(nodes)}
    for a, b in parent.core_edges:
        if a in idx and b in idx:
            g.add_edge(idx[a], idx[b])
    # Weisfeiler-Lehman style: sorted degree+element signature + edge multiset
    labels = [
        (g.nodes[i]["element"], g.degree[i]) for i in range(len(nodes))
    ]
    labels.sort()
    edge_sig = tuple(
        sorted(
            (
                tuple(sorted((g.nodes[u]["element"], g.nodes[v]["element"]))),
            )
            for u, v in g.edges
        )
    )
    return (tuple(labels), edge_sig, parent.k, parent.p)


@dataclass(frozen=True)
class MinimumSimilarity:
    """Invariant comparison metrics for two relaxed endpoints."""

    pair_distance_rms_A: float
    core_rmsd_A: float
    full_rmsd_A: float
    max_displacement_A: float
    energy_delta_eV: float
    wbo_rms: Optional[float]
    graph_mapping: Tuple[Tuple[int, int], ...]


@dataclass
class RelaxedMinimumCluster:
    """One relaxed basin and every raw endpoint/lineage route reaching it."""

    minimum_id: str
    representative: ParentStructure
    members: List[ParentStructure]
    member_metrics: Dict[str, MinimumSimilarity]

    @property
    def occupation_ids(self) -> Tuple[str, ...]:
        return tuple(
            sorted(
                {
                    str(member.zb_occupation.occupation_id)
                    for member in self.members
                    if getattr(member, "zb_occupation", None) is not None
                    and str(member.zb_occupation.occupation_id)
                }
            )
        )

    def route_representatives(
        self,
        *,
        max_occupations: int = 0,
    ) -> List[ParentStructure]:
        """Lowest-energy endpoint for every distinct ZB occupation route."""

        by_occupation: Dict[str, ParentStructure] = {}
        ordered_members = sorted(
            self.members, key=lambda item: (item.energy_eV, item.structure_id)
        )
        has_lattice_routes = any(
            getattr(member, "zb_occupation", None) is not None
            for member in ordered_members
        )
        for member in ordered_members:
            occupation = getattr(member, "zb_occupation", None)
            occupation_id = str(getattr(occupation, "occupation_id", "") or "")
            # Before Move Z is initialized (normally k=1), every relaxed
            # endpoint maps to the same composition-defined lattice seed.
            # Keep one basin representative rather than recreating duplicate
            # identical routes.  Once manifests exist, every distinct stored
            # occupation is preserved.
            key = occupation_id or (
                f"endpoint:{member.structure_id}"
                if has_lattice_routes
                else "uninitialized_zb_route"
            )
            by_occupation.setdefault(key, member)
        routes = list(by_occupation.values())
        if max_occupations > 0:
            routes = routes[: int(max_occupations)]
        member_ids = tuple(sorted(member.structure_id for member in self.members))
        occupation_ids = self.occupation_ids
        return [
            replace(
                member,
                minimum_id=self.minimum_id,
                minimum_representative_id=self.representative.structure_id,
                minimum_member_ids=member_ids,
                minimum_occupation_ids=occupation_ids,
                minimum_multiplicity=len(self.members),
            )
            for member in routes
        ]


def _parent_coloured_graph(parent: ParentStructure) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(
        (index, {"element": str(symbol)})
        for index, symbol in enumerate(parent.symbols)
    )
    graph.add_edges_from(
        (int(left), int(right)) for left, right in parent.edges
    )
    return graph


def _node_environment_signature(
    graph: nx.Graph,
    symbols: Sequence[str],
    node: int,
) -> Tuple[Any, ...]:
    distances = nx.single_source_shortest_path_length(graph, int(node))
    shells = tuple(
        sorted(
            (
                int(distance),
                str(symbols[index]),
                int(graph.degree[index]),
            )
            for index, distance in distances.items()
        )
    )
    neighbours = tuple(
        sorted(str(symbols[index]) for index in graph.neighbors(int(node)))
    )
    return (
        str(symbols[node]),
        int(graph.degree[node]),
        neighbours,
        shells,
    )


def _internal_atom_features(parent: ParentStructure) -> np.ndarray:
    """Rotation/translation-invariant distance environment of every atom."""

    coordinates = np.asarray(parent.coordinates, dtype=float)
    elements = sorted(set(parent.symbols))
    rows: List[List[float]] = []
    for index in range(len(parent.symbols)):
        feature: List[float] = []
        for element in elements:
            feature.extend(
                sorted(
                    float(np.linalg.norm(coordinates[index] - coordinates[other]))
                    for other, symbol in enumerate(parent.symbols)
                    if symbol == element
                )
            )
        rows.append(feature)
    return np.asarray(rows, dtype=float)


def _mapping_preserves_graph(
    left: nx.Graph,
    right: nx.Graph,
    mapping: Mapping[int, int],
) -> bool:
    if len(mapping) != left.number_of_nodes():
        return False
    mapped_edges = {
        tuple(sorted((int(mapping[a]), int(mapping[b]))))
        for a, b in left.edges
    }
    right_edges = {tuple(sorted((int(a), int(b)))) for a, b in right.edges}
    return mapped_edges == right_edges


def _candidate_graph_mappings(
    left: ParentStructure,
    right: ParentStructure,
    *,
    max_mappings: int,
) -> List[Dict[int, int]]:
    """Graph-constrained permutations, with an internal-distance fast path."""

    from scipy.optimize import linear_sum_assignment

    left_graph = _parent_coloured_graph(left)
    right_graph = _parent_coloured_graph(right)
    node_match = lambda a, b: a.get("element") == b.get("element")
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        left_graph, right_graph, node_match=node_match
    )
    if not matcher.is_isomorphic():
        return []

    left_features = _internal_atom_features(left)
    right_features = _internal_atom_features(right)
    left_groups: Dict[Tuple[Any, ...], List[int]] = defaultdict(list)
    right_groups: Dict[Tuple[Any, ...], List[int]] = defaultdict(list)
    for index in left_graph.nodes:
        left_groups[
            _node_environment_signature(left_graph, left.symbols, int(index))
        ].append(int(index))
    for index in right_graph.nodes:
        right_groups[
            _node_environment_signature(right_graph, right.symbols, int(index))
        ].append(int(index))

    proposed: Dict[int, int] = {}
    if set(left_groups) == set(right_groups):
        for signature in sorted(left_groups, key=repr):
            left_ids = left_groups[signature]
            right_ids = right_groups[signature]
            if len(left_ids) != len(right_ids):
                proposed = {}
                break
            cost = np.linalg.norm(
                left_features[left_ids, None, :]
                - right_features[None, right_ids, :],
                axis=2,
            )
            rows, columns = linear_sum_assignment(cost)
            proposed.update(
                {
                    left_ids[int(row)]: right_ids[int(column)]
                    for row, column in zip(rows, columns)
                }
            )
    mappings: List[Dict[int, int]] = []
    if proposed and _mapping_preserves_graph(left_graph, right_graph, proposed):
        mappings.append(proposed)

    # Symmetric graphs can defeat an independent Hungarian assignment.  VF2
    # supplies exact graph permutations; the cap bounds highly symmetric Cl
    # shells, while the internal-distance fast path handles the common case.
    for mapping in matcher.isomorphisms_iter():
        candidate = {int(a): int(b) for a, b in mapping.items()}
        if proposed and candidate == proposed:
            continue
        mappings.append(candidate)
        if len(mappings) >= max(1, int(max_mappings)):
            break
    return mappings


def _proper_aligned_displacements(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    allow_reflection: bool,
) -> np.ndarray:
    left = np.asarray(reference, dtype=float)
    right = np.asarray(candidate, dtype=float)
    left_centered = left - left.mean(axis=0)
    right_centered = right - right.mean(axis=0)
    u, _singular, vt = np.linalg.svd(right_centered.T @ left_centered)
    rotation = u @ vt
    if not allow_reflection and np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    aligned = right_centered @ rotation
    return np.linalg.norm(aligned - left_centered, axis=1)


def _wbo_mapping_rms(
    left: ParentStructure,
    right: ParentStructure,
    mapping: Mapping[int, int],
) -> Optional[float]:
    if not left.wbo or not right.wbo:
        return None
    pairs = {
        tuple(sorted((int(a), int(b)))) for a, b in left.edges
    }
    if not pairs:
        return 0.0
    delta = []
    for a, b in sorted(pairs):
        mapped = tuple(sorted((int(mapping[a]), int(mapping[b]))))
        delta.append(
            float(left.wbo.get((a, b), 0.0))
            - float(right.wbo.get(mapped, 0.0))
        )
    return float(np.sqrt(np.mean(np.square(delta))))


def relaxed_minimum_similarity(
    left: ParentStructure,
    right: ParentStructure,
    config: MinimumConsolidation,
    spec: NucleationSpec,
) -> Optional[MinimumSimilarity]:
    """Composite invariant same-basin test; return best metrics or ``None``."""

    if (left.k, left.p, tuple(sorted(left.symbols))) != (
        right.k,
        right.p,
        tuple(sorted(right.symbols)),
    ):
        return None
    energy_delta = abs(float(left.energy_eV) - float(right.energy_eV))
    if energy_delta > float(config.energy_tolerance_eV):
        return None

    mappings = _candidate_graph_mappings(
        left, right, max_mappings=config.max_graph_mappings
    )
    if not mappings:
        return None
    left_coordinates = np.asarray(left.coordinates, dtype=float)
    right_coordinates = np.asarray(right.coordinates, dtype=float)
    left_distances = np.linalg.norm(
        left_coordinates[:, None, :] - left_coordinates[None, :, :], axis=2
    )
    core_ids = [
        index
        for index, symbol in enumerate(left.symbols)
        if symbol in {spec.core.cation, spec.core.anion}
    ]
    best: Optional[MinimumSimilarity] = None
    for mapping in mappings:
        order = [int(mapping[index]) for index in range(len(left.symbols))]
        ordered = right_coordinates[order]
        ordered_distances = np.linalg.norm(
            ordered[:, None, :] - ordered[None, :, :], axis=2
        )
        triangle = np.triu_indices(len(left.symbols), k=1)
        pair_rms = float(
            np.sqrt(
                np.mean(
                    np.square(
                        left_distances[triangle] - ordered_distances[triangle]
                    )
                )
            )
        )
        if pair_rms > float(config.pair_distance_rms_A):
            continue
        full_displacements = _proper_aligned_displacements(
            left_coordinates,
            ordered,
            allow_reflection=config.allow_reflection,
        )
        core_displacements = _proper_aligned_displacements(
            left_coordinates[core_ids],
            ordered[core_ids],
            allow_reflection=config.allow_reflection,
        )
        full_rms = float(np.sqrt(np.mean(np.square(full_displacements))))
        core_rms = float(np.sqrt(np.mean(np.square(core_displacements))))
        max_displacement = float(max(full_displacements, default=0.0))
        if full_rms > float(config.full_rmsd_A):
            continue
        if core_rms > float(config.core_rmsd_A):
            continue
        if max_displacement > float(config.max_displacement_A):
            continue
        wbo_rms = _wbo_mapping_rms(left, right, mapping)
        if (
            config.require_wbo_when_available
            and wbo_rms is not None
            and wbo_rms > float(config.wbo_rms_tolerance)
        ):
            continue
        metrics = MinimumSimilarity(
            pair_distance_rms_A=pair_rms,
            core_rmsd_A=core_rms,
            full_rmsd_A=full_rms,
            max_displacement_A=max_displacement,
            energy_delta_eV=energy_delta,
            wbo_rms=wbo_rms,
            graph_mapping=tuple(sorted(mapping.items())),
        )
        if best is None or (
            metrics.full_rmsd_A,
            metrics.pair_distance_rms_A,
            metrics.max_displacement_A,
        ) < (
            best.full_rmsd_A,
            best.pair_distance_rms_A,
            best.max_displacement_A,
        ):
            best = metrics
    return best


def _minimum_geometry_id(parent: ParentStructure) -> str:
    graph = _parent_coloured_graph(parent)
    graph_hash = nx.weisfeiler_lehman_graph_hash(
        graph, node_attr="element", iterations=6
    )
    coordinates = np.asarray(parent.coordinates, dtype=float)
    distances: List[Tuple[str, str, int]] = []
    for left in range(len(parent.symbols)):
        for right in range(left + 1, len(parent.symbols)):
            a, b = sorted((parent.symbols[left], parent.symbols[right]))
            distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
            distances.append((a, b, int(round(distance / 0.01))))
    payload = repr((graph_hash, tuple(sorted(distances)))).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"min_k{parent.k:03d}_p{parent.p:03d}_{digest}"


def consolidate_relaxed_minima(
    parents: Sequence[ParentStructure],
    config: MinimumConsolidation,
    spec: NucleationSpec,
) -> List[RelaxedMinimumCluster]:
    """Complete-linkage relaxed-basin clustering, lowest energy first."""

    ordered = sorted(
        parents, key=lambda item: (float(item.energy_eV), item.structure_id)
    )
    clusters: List[RelaxedMinimumCluster] = []
    for candidate in ordered:
        best_cluster = None
        best_metrics: Optional[MinimumSimilarity] = None
        for cluster in clusters:
            comparisons: List[MinimumSimilarity] = []
            for member in cluster.members:
                metrics = relaxed_minimum_similarity(
                    member, candidate, config, spec
                )
                if metrics is None:
                    comparisons = []
                    break
                comparisons.append(metrics)
            if not comparisons:
                continue
            representative_metrics = relaxed_minimum_similarity(
                cluster.representative, candidate, config, spec
            )
            if representative_metrics is None:
                continue
            if (
                best_metrics is None
                or representative_metrics.full_rmsd_A
                < best_metrics.full_rmsd_A
            ):
                best_cluster = cluster
                best_metrics = representative_metrics
        if best_cluster is None or best_metrics is None:
            minimum_id = _minimum_geometry_id(candidate)
            zero_mapping = tuple((index, index) for index in range(len(candidate.symbols)))
            zero = MinimumSimilarity(
                pair_distance_rms_A=0.0,
                core_rmsd_A=0.0,
                full_rmsd_A=0.0,
                max_displacement_A=0.0,
                energy_delta_eV=0.0,
                wbo_rms=0.0 if candidate.wbo else None,
                graph_mapping=zero_mapping,
            )
            clusters.append(
                RelaxedMinimumCluster(
                    minimum_id=minimum_id,
                    representative=candidate,
                    members=[candidate],
                    member_metrics={candidate.structure_id: zero},
                )
            )
            continue
        best_cluster.members.append(candidate)
        best_cluster.member_metrics[candidate.structure_id] = best_metrics
    return clusters


def _minimum_cluster_record(
    cluster: RelaxedMinimumCluster,
) -> Dict[str, Any]:
    representative = cluster.representative
    members = []
    for member in sorted(
        cluster.members, key=lambda item: (item.energy_eV, item.structure_id)
    ):
        metrics = cluster.member_metrics[member.structure_id]
        occupation = getattr(member, "zb_occupation", None)
        members.append(
            {
                "structure_id": member.structure_id,
                "energy_eV": float(member.energy_eV),
                "occupation_id": str(
                    getattr(occupation, "occupation_id", "") or ""
                ),
                "pair_distance_rms_A": metrics.pair_distance_rms_A,
                "core_rmsd_A": metrics.core_rmsd_A,
                "full_rmsd_A": metrics.full_rmsd_A,
                "max_displacement_A": metrics.max_displacement_A,
                "energy_delta_eV": metrics.energy_delta_eV,
                "wbo_rms": metrics.wbo_rms,
            }
        )
    return {
        "schema_version": 1,
        "minimum_id": cluster.minimum_id,
        "k": int(representative.k),
        "p": int(representative.p),
        "representative_structure_id": representative.structure_id,
        "representative_energy_eV": float(representative.energy_eV),
        "multiplicity": len(cluster.members),
        "occupation_ids": list(cluster.occupation_ids),
        "members": members,
    }


def write_minimum_clusters(
    output_dir: Path,
    k: int,
    p: int,
    clusters: Sequence[RelaxedMinimumCluster],
    *,
    config: Optional[MinimumConsolidation] = None,
) -> Path:
    """Write one restart-inspectable basin inventory without deleting XYZ."""

    directory = Path(output_dir) / f"k{int(k):03d}" / f"p{int(p):03d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "minimum_clusters.json"
    payload = {
        "schema_version": 1,
        "algorithm": "coloured_graph+internal_distances+permuted_kabsch",
        "k": int(k),
        "p": int(p),
        "raw_endpoint_count": sum(len(cluster.members) for cluster in clusters),
        "minimum_count": len(clusters),
        "clusters": [_minimum_cluster_record(cluster) for cluster in clusters],
    }
    if config is not None:
        payload["criteria"] = {
            "energy_tolerance_eV": config.energy_tolerance_eV,
            "pair_distance_rms_A": config.pair_distance_rms_A,
            "core_rmsd_A": config.core_rmsd_A,
            "full_rmsd_A": config.full_rmsd_A,
            "max_displacement_A": config.max_displacement_A,
            "wbo_rms_tolerance": config.wbo_rms_tolerance,
            "require_wbo_when_available": config.require_wbo_when_available,
            "allow_reflection": config.allow_reflection,
            "max_graph_mappings": config.max_graph_mappings,
        }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _relaxed_minimum_diversity_signature(
    cluster: RelaxedMinimumCluster,
) -> Tuple[Any, ...]:
    representative = cluster.representative
    graph = _parent_coloured_graph(representative)
    degree_signature = tuple(
        (
            element,
            tuple(
                sorted(
                    int(graph.degree[index])
                    for index, symbol in enumerate(representative.symbols)
                    if symbol == element
                )
            ),
        )
        for element in sorted(set(representative.symbols))
    )
    coordinates = np.asarray(representative.coordinates, dtype=float)
    centered = coordinates - coordinates.mean(axis=0)
    moments = np.linalg.eigvalsh(centered.T @ centered / max(len(centered), 1))
    shape = tuple(int(round(float(value) / 0.10)) for value in moments)
    return degree_signature, shape


def select_parents(
    parents: Sequence[ParentStructure],
    growth: GrowthConfig,
    spec: NucleationSpec,
) -> List[ParentStructure]:
    """Energy-window and diversity selection over relaxed minima or cores."""

    if not parents:
        return []
    # Group by (k, p) then by core fingerprint
    by_bin: Dict[Tuple[int, int], List[ParentStructure]] = {}
    for p in parents:
        by_bin.setdefault((p.k, p.p), []).append(p)

    selected: List[ParentStructure] = []
    for (k, p), group in sorted(by_bin.items()):
        win = growth.window_for(k)
        if int(p) < int(win.min_p_parent):
            continue
        consolidation = growth.minimum_consolidation

        if consolidation.enabled:
            clusters = consolidate_relaxed_minima(group, consolidation, spec)

            def _cluster_score(cluster: RelaxedMinimumCluster) -> float:
                representative = cluster.representative
                if not win.soft_rules.enabled:
                    return float(representative.energy_eV)
                desc = describe_structure(
                    representative.symbols, representative.coordinates, spec
                )
                return win.soft_rules.rank_score_eV(
                    representative.energy_eV, desc, k
                )

            minimum_energy = min(
                float(cluster.representative.energy_eV) for cluster in clusters
            )
            clusters = [
                cluster
                for cluster in clusters
                if float(cluster.representative.energy_eV)
                <= minimum_energy + win.energy_window_eV
            ]
            clusters.sort(
                key=lambda cluster: (
                    _cluster_score(cluster),
                    cluster.minimum_id,
                )
            )
            n_keep = max(
                1,
                min(
                    win.max_skeletons_cap,
                    int(
                        math.ceil(
                            win.max_skeletons_frac * max(1, len(clusters))
                        )
                    ),
                ),
            )
            n_keep = min(n_keep, len(clusters))

            retained: List[RelaxedMinimumCluster] = []
            retained_ids: set[str] = set()
            seen_shapes: set[Tuple[Any, ...]] = set()
            occupation_counts: Counter[str] = Counter()

            def _occupation_quota_allows(
                cluster: RelaxedMinimumCluster,
            ) -> bool:
                occupation_ids = cluster.occupation_ids
                if not occupation_ids:
                    return True
                limit = int(consolidation.max_minima_per_occupation)
                return any(occupation_counts[value] < limit for value in occupation_ids)

            def _retain(cluster: RelaxedMinimumCluster) -> None:
                retained.append(cluster)
                retained_ids.add(cluster.minimum_id)
                seen_shapes.add(_relaxed_minimum_diversity_signature(cluster))
                occupation_counts.update(cluster.occupation_ids)

            if clusters:
                _retain(clusters[0])
            # First spend the basin budget on distinct relaxed shapes.
            for cluster in clusters[1:]:
                if len(retained) >= n_keep:
                    break
                signature = _relaxed_minimum_diversity_signature(cluster)
                if signature in seen_shapes or not _occupation_quota_allows(cluster):
                    continue
                _retain(cluster)
            # Then fill by energy while respecting per-occupation basin caps.
            for cluster in clusters:
                if len(retained) >= n_keep:
                    break
                if cluster.minimum_id in retained_ids:
                    continue
                if not _occupation_quota_allows(cluster):
                    continue
                _retain(cluster)
            # Do not leave capacity empty merely because every route reached
            # its soft quota; the global energy/diversity cap remains primary.
            for cluster in clusters:
                if len(retained) >= n_keep:
                    break
                if cluster.minimum_id not in retained_ids:
                    _retain(cluster)

            for cluster in retained:
                selected.extend(
                    cluster.route_representatives(
                        max_occupations=(
                            consolidation.max_occupations_per_minimum
                        )
                    )
                )
            continue

        emin = min(x.energy_eV for x in group)
        windowed = [
            x
            for x in group
            if x.energy_eV <= emin + win.energy_window_eV
        ]
        if not windowed:
            windowed = [min(group, key=lambda x: x.energy_eV)]

        # skeleton buckets
        buckets: Dict[Tuple[object, ...], List[ParentStructure]] = {}
        for x in windowed:
            fp = _core_fingerprint(x, spec)
            buckets.setdefault(fp, []).append(x)
        def _score(x: ParentStructure) -> float:
            if not win.soft_rules.enabled:
                return float(x.energy_eV)
            desc = describe_structure(x.symbols, x.coordinates, spec)
            return win.soft_rules.rank_score_eV(x.energy_eV, desc, k)

        # rank skeletons by best energy (or E + soft penalty)
        skel_ranked = sorted(
            buckets.items(),
            key=lambda kv: min(_score(y) for y in kv[1]),
        )
        n_keep = max(
            1,
            min(
                win.max_skeletons_cap,
                int(math.ceil(win.max_skeletons_frac * max(1, len(skel_ranked)))),
            ),
        )
        n_keep = min(n_keep, len(skel_ranked))
        retained_buckets = skel_ranked[:n_keep]
        if skel_ranked and all(
            getattr(member, "zb_occupation", None) is not None
            for _fingerprint, members in skel_ranked
            for member in members
        ):
            from .molecular_zb_growth import occupation_diversity_signature

            retained_buckets = [skel_ranked[0]]
            retained_keys = {skel_ranked[0][0]}
            seen_classes = {
                occupation_diversity_signature(
                    skel_ranked[0][1][0].zb_occupation
                )
            }
            for fingerprint, members in skel_ranked[1:]:
                signature = occupation_diversity_signature(
                    members[0].zb_occupation
                )
                if signature in seen_classes:
                    continue
                retained_buckets.append((fingerprint, members))
                retained_keys.add(fingerprint)
                seen_classes.add(signature)
                if len(retained_buckets) >= n_keep:
                    break
            if len(retained_buckets) < n_keep:
                retained_buckets.extend(
                    item
                    for item in skel_ranked
                    if item[0] not in retained_keys
                )
                retained_buckets = retained_buckets[:n_keep]
        for _fp, members in retained_buckets:
            members_sorted = sorted(members, key=_score)
            selected.extend(members_sorted[: win.decorations_per_skeleton])
    return selected


# ---------------------------------------------------------------------------
# Packages and shedding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CdCl2Package:
    """One complete precursor unit: Cd + two Cl."""

    cd: int
    cl: Tuple[int, int]
    score: float  # lower = less bound (shed first)


def identify_packages(
    parent: ParentStructure,
    spec: NucleationSpec,
) -> List[CdCl2Package]:
    """Find **non-overlapping** CdCl2 packages on the relaxed graph.

    Each package is one Cd + two Cl.  Candidates are scored (WBO sum if
    available, else longer Cd–Cl → weaker → shed first), then greedily kept
    in score order so that **no two packages share a Cd or Cl**.

    Overlapping assignment was a real bug: shedding s packages could remove
    s Cd but fewer than 2s Cl (shared Cl double-counted), leaving the wrong
    composition (extra Cl ≈ −12 keV on g-xTB) while still labelling
    ``[CdSe]_k(CdCl2)_p``.
    """

    ligand = spec.precursor.ligand
    cation = spec.precursor.center
    anion = spec.core.anion
    symbols = parent.symbols
    coords = parent.coordinates
    # adjacency from edges
    neigh: Dict[int, List[int]] = {i: [] for i in range(len(symbols))}
    for a, b in parent.edges:
        neigh[a].append(b)
        neigh[b].append(a)

    candidates: List[CdCl2Package] = []
    for i, sym in enumerate(symbols):
        if sym != cation:
            continue
        cl_n = [j for j in neigh[i] if symbols[j] == ligand]
        if len(cl_n) < 2:
            continue
        # Prefer true precursor-like Cd: at least one Cl; optional Se still ok
        # (surface core Cd with two Cl can appear — non-overlap + score handles).
        cl_n = sorted(
            cl_n,
            key=lambda j: float(np.linalg.norm(coords[i] - coords[j])),
        )[:2]
        c1, c2 = cl_n[0], cl_n[1]
        if parent.wbo:
            w1 = parent.wbo.get((min(i, c1), max(i, c1)), 0.0)
            w2 = parent.wbo.get((min(i, c2), max(i, c2)), 0.0)
            score = w1 + w2
        else:
            d1 = float(np.linalg.norm(coords[i] - coords[c1]))
            d2 = float(np.linalg.norm(coords[i] - coords[c2]))
            # longer mean distance → weaker → smaller score
            score = -0.5 * (d1 + d2)
        # slight preference to shed Cd with fewer Se bonds (more precursor-like)
        n_se = sum(1 for j in neigh[i] if symbols[j] == anion)
        score = score + 0.01 * n_se
        candidates.append(CdCl2Package(cd=i, cl=(c1, c2), score=score))
    candidates.sort(key=lambda p: (p.score, p.cd))

    # Greedy non-overlapping selection (least-bound first for shed order)
    used: set = set()
    packages: List[CdCl2Package] = []
    for pkg in candidates:
        atoms = {pkg.cd, pkg.cl[0], pkg.cl[1]}
        if atoms & used:
            continue
        used |= atoms
        packages.append(pkg)
    return packages


def expected_composition(
    k: int, p: int, *, cation: str = "Cd", anion: str = "Se", ligand: str = "Cl"
) -> Dict[str, int]:
    """Atom counts for neutral ``k CdSe + p CdCl2``."""

    return {anion: int(k), cation: int(k + p), ligand: int(2 * p)}


def composition_counts(symbols: Sequence[str]) -> Dict[str, int]:
    from collections import Counter

    return dict(Counter(symbols))


def core_edges_after_removing_cd(
    core_edges: EdgeList,
    remove_cd: Sequence[int],
) -> EdgeList:
    """Drop edges incident to removed precursor Cd indices."""

    drop = set(remove_cd)
    return tuple(
        sorted(e for e in core_edges if e[0] not in drop and e[1] not in drop)
    )


def remap_core_edges_to_blocks(
    symbols: Sequence[str],
    core_edges: EdgeList,
    *,
    k: int,
    p: int,
    cation: str,
    anion: str,
) -> Optional[EdgeList]:
    """Map arbitrary-index core edges onto canonical ``_index_blocks(k, p)``."""

    se_ids, cd_ids, _ = _index_blocks(k, p)
    used = {n for e in core_edges for n in e}
    se_keep = [i for i, s in enumerate(symbols) if s == anion][:k]
    cd_all = [i for i, s in enumerate(symbols) if s == cation]
    if len(se_keep) != k or len(cd_all) < k + p:
        return None
    cd_ranked = sorted(cd_all, key=lambda i: (0 if i in used else 1, i))
    cd_keep = cd_ranked[: k + p]
    mapping = {}
    for new, old in zip(se_ids, sorted(se_keep)):
        mapping[old] = new
    for new, old in zip(cd_ids, sorted(cd_keep)):
        mapping[old] = new
    out = []
    for a, b in core_edges:
        if a not in mapping or b not in mapping:
            continue
        na, nb = mapping[a], mapping[b]
        out.append((min(na, nb), max(na, nb)))
    return tuple(sorted(set(out)))


def parent_core_in_blocks(
    parent: ParentStructure, spec: NucleationSpec
) -> Optional[EdgeList]:
    """Cd–Se edges of the parent in canonical (k, p) index layout."""

    k, p = parent.k, parent.p
    symbols = parent.symbols
    exp = (
        [spec.core.anion] * k
        + [spec.core.cation] * (k + p)
        + [spec.precursor.ligand] * (2 * p)
    )
    if list(symbols) == exp and len(symbols) == len(exp):
        return tuple(sorted(parent.core_edges))
    return remap_core_edges_to_blocks(
        symbols,
        parent.core_edges,
        k=k,
        p=p,
        cation=spec.core.cation,
        anion=spec.core.anion,
    )


def parent_cn4_fraction(parent: ParentStructure, spec: NucleationSpec) -> float:
    """Fraction of inorganic (Cd/Se) atoms with degree ≥ 4 on the parent graph."""

    cation, anion = spec.core.cation, spec.core.anion
    inorganic = [
        i
        for i, sym in enumerate(parent.symbols)
        if sym in {cation, anion}
    ]
    if not inorganic:
        return 0.0
    deg: Dict[int, int] = {i: 0 for i in inorganic}
    inorg = set(inorganic)
    for a, b in parent.core_edges:
        if a in inorg and b in inorg:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
    n4 = sum(1 for i in inorganic if deg.get(i, 0) >= 4)
    return n4 / float(len(inorganic))


def parent_six_ring_count(parent: ParentStructure, spec: NucleationSpec) -> int:
    """Cd–Se 6-rings on the remapped parent core, or 0 if remap fails."""

    from .molecular import count_cdse_six_rings

    core = parent_core_in_blocks(parent, spec)
    if not core:
        return 0
    return int(count_cdse_six_rings(core, parent.k, parent.p))


def parent_rmsd_to_zb_A(
    parent: ParentStructure, spec: NucleationSpec
) -> Optional[float]:
    """Kabsch RMSD of the inorganic core vs a zb fragment from ``spec.cif``.

    Returns None if the CIF cannot be read or the fragment is too small.
    """

    cif = getattr(spec, "cif", None)
    if not cif or not Path(str(cif)).is_file():
        return None
    cation, anion = spec.core.cation, spec.core.anion
    core_idx = [
        i
        for i, sym in enumerate(parent.symbols)
        if sym in {cation, anion}
    ]
    if len(core_idx) < 4:
        return None
    try:
        from pymatgen.core import Structure

        struct = Structure.from_file(str(cif))
        struct.make_supercell((3, 3, 3))
    except Exception:
        return None
    sites = [
        site
        for site in struct.sites
        if str(site.specie.symbol) in {cation, anion}
    ]
    if len(sites) < len(core_idx):
        return None
    # Take the first N zb sites of matching element counts (compact origin cube).
    want_cat = sum(1 for i in core_idx if parent.symbols[i] == cation)
    want_an = len(core_idx) - want_cat
    picked: List[Any] = []
    n_cat = n_an = 0
    origin = sites[0].coords
    ordered = sorted(sites, key=lambda s: float(np.linalg.norm(s.coords - origin)))
    for site in ordered:
        sym = str(site.specie.symbol)
        if sym == cation and n_cat < want_cat:
            picked.append(site)
            n_cat += 1
        elif sym == anion and n_an < want_an:
            picked.append(site)
            n_an += 1
        if n_cat == want_cat and n_an == want_an:
            break
    if len(picked) != len(core_idx):
        return None
    # Align by element-sorted coordinates (not a true matching — diagnostic only).
    def _sorted_xyz(symbols: Sequence[str], coords: FloatArray) -> FloatArray:
        order = sorted(range(len(symbols)), key=lambda i: (symbols[i], i))
        return np.asarray(coords, dtype=float)[order]

    parent_xyz = _sorted_xyz(
        [parent.symbols[i] for i in core_idx],
        parent.coordinates[core_idx],
    )
    zb_xyz = _sorted_xyz(
        [str(s.specie.symbol) for s in picked],
        np.asarray([s.coords for s in picked], dtype=float),
    )
    parent_xyz = parent_xyz - parent_xyz.mean(axis=0)
    zb_xyz = zb_xyz - zb_xyz.mean(axis=0)
    u, _s, vt = np.linalg.svd(parent_xyz.T @ zb_xyz)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        rot = vt.T @ u.T
    aligned = parent_xyz @ rot.T
    delta = aligned - zb_xyz
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def parent_has_local_tet(
    parent: ParentStructure,
    spec: NucleationSpec,
    switch: LatticeSwitch,
) -> Tuple[bool, str]:
    """Local tetrahedral character (MSC), not zinc-blende long-range order.

    CN4 fraction and six-rings mean “Cd has tet holes / rings exist”.
    Optional RMSD-to-zb is a *crystal* diagnostic and should not be the
    reason to turn on tet_sites; keep it off in production YAML.
    """

    reasons: List[str] = []
    if switch.min_cn4_fraction is not None:
        frac = parent_cn4_fraction(parent, spec)
        reasons.append(f"cn4={frac:.2f}")
        if frac >= float(switch.min_cn4_fraction):
            return True, f"cn4_fraction={frac:.2f}"
    if switch.min_six_rings is not None:
        n6 = parent_six_ring_count(parent, spec)
        reasons.append(f"r6={n6}")
        if n6 >= int(switch.min_six_rings):
            return True, f"six_rings={n6}"
    if switch.max_core_rmsd_to_zb_A is not None:
        rmsd = parent_rmsd_to_zb_A(parent, spec)
        if rmsd is not None:
            reasons.append(f"rmsd={rmsd:.2f}")
            if rmsd <= float(switch.max_core_rmsd_to_zb_A):
                return True, f"zb_rmsd={rmsd:.2f}A"
    return False, "no_criterion (" + ",".join(reasons) + ")"


def choose_decoration_mode(
    parents: Sequence[ParentStructure],
    *,
    k_child: int,
    spec: NucleationSpec,
    switch: LatticeSwitch,
) -> Tuple[str, str]:
    """Return (mode, note) for decorating children at ``k_child``.

    Empty mode means “leave the pack graph_rules alone”.
    """

    if not switch.enabled or int(k_child) < int(switch.from_k):
        return "", "lattice_switch off or k < from_k"
    for parent in parents:
        ok, why = parent_has_local_tet(parent, spec, switch)
        if ok:
            return str(switch.decoration_mode), f"parent {parent.structure_id}: {why}"
    fallback = str(switch.fallback or "")
    if fallback:
        return fallback, "no parent met local-tet tests"
    return "", "no parent met local-tet tests"


def parent_looks_zb_like(
    parent: ParentStructure,
    spec: NucleationSpec,
    switch: LatticeSwitch,
) -> Tuple[bool, str]:
    """Deprecated name: local tet, not zb.  Use ``parent_has_local_tet``."""

    return parent_has_local_tet(parent, spec, switch)


def spec_with_decoration_mode(spec: NucleationSpec, mode: str) -> NucleationSpec:
    """Copy of ``spec`` with ``graph_rules.decoration_mode`` forced to ``mode``."""

    from dataclasses import replace

    rules = replace(
        spec.graph_rules,
        decoration_mode=str(mode),
        decoration_mode_from_k=0,
        decoration_mode_at_or_above="",
    )
    return replace(spec, graph_rules=rules)


# ---------------------------------------------------------------------------
# Move B: coordinate carry-over (WBO shed + place monomer + cleanup)
# ---------------------------------------------------------------------------

# Default zb-like distances when pack tables are unavailable
_DEFAULT_CDSE_A = 2.62
_DEFAULT_CDCL_A = 2.45


def _cdse_bond_A(pack: Optional[GeometryPack]) -> float:
    if pack is None:
        return _DEFAULT_CDSE_A
    raw = pack.raw or {}
    embed = raw.get("embed") or raw
    # common keys in production packs
    for key in ("cd_se", "Cd-Se", "cdse"):
        block = embed.get(key) if isinstance(embed, dict) else None
        if isinstance(block, dict):
            for kk in ("r0_A", "r_A", "length_A", "median_A"):
                if kk in block:
                    return float(block[kk])
    return _DEFAULT_CDSE_A


def _cdcl_bond_A(pack: Optional[GeometryPack]) -> float:
    if pack is None:
        return _DEFAULT_CDCL_A
    raw = pack.raw or {}
    embed = raw.get("embed") or raw
    for key in ("cd_cl", "Cd-Cl", "cdcl"):
        block = embed.get(key) if isinstance(embed, dict) else None
        if isinstance(block, dict):
            for kk in ("r0_A", "r_A", "length_A", "median_A"):
                if kk in block:
                    return float(block[kk])
    return _DEFAULT_CDCL_A


def _formal_charge(symbols: Sequence[str], spec: NucleationSpec) -> int:
    """Formal charge from pack oxidation states (CdSe/CdCl2 → 0 when complete)."""

    ch = {
        spec.core.cation: 2,
        spec.core.anion: -2,
        spec.precursor.ligand: -1,
    }
    # precursor center is same element as core cation usually
    ch.setdefault(spec.precursor.center, 2)
    return int(sum(ch.get(s, 0) for s in symbols))


def shed_packages_coords(
    parent: ParentStructure,
    *,
    s: int,
    packages: Sequence[CdCl2Package],
) -> Tuple[Tuple[str, ...], FloatArray, EdgeList, Tuple[float, ...]]:
    """Remove the ``s`` least-bound packages (already sorted ascending score).

    Returns (symbols, coords, remaining_edges, shed_scores).
    """

    if s <= 0:
        return (
            parent.symbols,
            np.asarray(parent.coordinates, dtype=float).copy(),
            parent.edges,
            (),
        )
    drop: set = set()
    scores: List[float] = []
    for pkg in list(packages)[:s]:
        drop.add(pkg.cd)
        drop.add(pkg.cl[0])
        drop.add(pkg.cl[1])
        scores.append(float(pkg.score))
    keep = [i for i in range(len(parent.symbols)) if i not in drop]
    if not keep:
        raise ValueError("shed removed all atoms")
    old_to_new = {old: new for new, old in enumerate(keep)}
    symbols = tuple(parent.symbols[i] for i in keep)
    coords = np.asarray(parent.coordinates, dtype=float)[keep].copy()
    edges = tuple(
        sorted(
            (old_to_new[a], old_to_new[b])
            for a, b in parent.edges
            if a in old_to_new and b in old_to_new
        )
    )
    return symbols, coords, edges, tuple(scores)


def _outward_direction(
    coords: FloatArray,
    anchor: int,
    neigh: Sequence[int],
) -> np.ndarray:
    """Unit vector pointing away from neighbours / COM."""

    com = coords.mean(axis=0)
    if neigh:
        mean_n = coords[list(neigh)].mean(axis=0)
        d = coords[anchor] - mean_n
    else:
        d = coords[anchor] - com
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        # arbitrary
        d = np.array([1.0, 0.0, 0.0])
        n = 1.0
    return d / n


def _orthonormal_pair(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = axis / (float(np.linalg.norm(axis)) + 1e-15)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, tmp)
    e1 = e1 / (float(np.linalg.norm(e1)) + 1e-15)
    e2 = np.cross(a, e1)
    e2 = e2 / (float(np.linalg.norm(e2)) + 1e-15)
    return e1, e2


def place_monomer_and_packages(
    symbols: Sequence[str],
    coords: FloatArray,
    edges: EdgeList,
    *,
    k_parent: int,
    p_after_shed: int,
    p_m: int,
    spec: NucleationSpec,
    pack: Optional[GeometryPack] = None,
) -> Tuple[Tuple[str, ...], FloatArray, EdgeList]:
    """Add CdSe monomer + ``p_m`` CdCl2 packages in 3D (embed-table distances).

    Keeps all remaining parent atoms (including un-shed Cl).  Places:
      * new Se outward from a low-CN core Cd
      * new core Cd bonded to that Se
      * ``p_m`` precursor Cd each with two Cl (package)
    """

    cation = spec.core.cation
    anion = spec.core.anion
    ligand = spec.precursor.ligand
    r_cdse = _cdse_bond_A(pack)
    r_cdcl = _cdcl_bond_A(pack)

    sym = list(symbols)
    xyz = np.asarray(coords, dtype=float).copy()
    edge_list = list(edges)
    neigh: Dict[int, List[int]] = {i: [] for i in range(len(sym))}
    for a, b in edge_list:
        neigh[a].append(b)
        neigh[b].append(a)

    # Prefer a core Cd (bonded to Se) with free valence as monomer attach site
    se_ids = [i for i, s in enumerate(sym) if s == anion]
    cd_ids = [i for i, s in enumerate(sym) if s == cation]
    max_cd = int(spec.graph_rules.max_cn.get(cation, 4))
    max_se = int(spec.graph_rules.max_cn.get(anion, 4))
    core_cd = [
        i
        for i in cd_ids
        if any(sym[j] == anion for j in neigh[i]) and len(neigh[i]) < max_cd
    ]
    if not core_cd:
        core_cd = [i for i in cd_ids if len(neigh[i]) < max_cd] or cd_ids
    if not core_cd:
        raise ValueError("no Cd to attach monomer")
    host = min(core_cd, key=lambda i: len(neigh[i]))

    # new Se along outward direction from host
    d = _outward_direction(xyz, host, neigh[host])
    new_se = len(sym)
    sym.append(anion)
    xyz = np.vstack([xyz, xyz[host] + d * r_cdse])
    edge_list.append((host, new_se))
    neigh[host].append(new_se)
    neigh[new_se] = [host]

    # new core Cd opposite to host relative to new Se
    d2 = _outward_direction(xyz, new_se, neigh[new_se])
    new_cd = len(sym)
    sym.append(cation)
    xyz = np.vstack([xyz, xyz[new_se] + d2 * r_cdse])
    edge_list.append((new_se, new_cd))
    neigh[new_se].append(new_cd)
    neigh[new_cd] = [new_se]

    # p_m precursor packages: attach to open Se (prefer original Se)
    open_se = [
        i
        for i in se_ids + [new_se]
        if len(neigh.get(i, [])) < max_se
    ]
    e1, e2 = _orthonormal_pair(d2)
    for m in range(int(p_m)):
        if not open_se:
            # fall back: attach to new_se even if overloaded (clash soft)
            attach_se = new_se
        else:
            attach_se = open_se[m % len(open_se)]
        dse = _outward_direction(xyz, attach_se, neigh.get(attach_se, []))
        # slight azimuthal offset per package
        ang = (2.0 * math.pi * m) / max(1, p_m)
        direction = dse * 0.7 + (math.cos(ang) * e1 + math.sin(ang) * e2) * 0.3
        direction = direction / (float(np.linalg.norm(direction)) + 1e-15)
        pre_cd = len(sym)
        sym.append(cation)
        xyz = np.vstack([xyz, xyz[attach_se] + direction * r_cdse])
        edge_list.append((attach_se, pre_cd))
        neigh.setdefault(attach_se, []).append(pre_cd)
        neigh[pre_cd] = [attach_se]
        # two Cl tetrahedral-ish around precursor Cd
        axis = xyz[pre_cd] - xyz[attach_se]
        axis = axis / (float(np.linalg.norm(axis)) + 1e-15)
        u, v = _orthonormal_pair(axis)
        for sign in (+1.0, -1.0):
            cl_dir = (axis * 0.3 + sign * u * 0.95)
            cl_dir = cl_dir / (float(np.linalg.norm(cl_dir)) + 1e-15)
            cl_i = len(sym)
            sym.append(ligand)
            xyz = np.vstack([xyz, xyz[pre_cd] + cl_dir * r_cdcl])
            edge_list.append((pre_cd, cl_i))
            neigh[pre_cd].append(cl_i)
            neigh[cl_i] = [pre_cd]

    edges_out = tuple(
        sorted((min(a, b), max(a, b)) for a, b in edge_list)
    )
    return tuple(sym), xyz, edges_out


def core_edges_from_coords(
    symbols: Sequence[str],
    coords: FloatArray,
    *,
    spec: NucleationSpec,
    cutoffs: Mapping[Tuple[str, str], float],
) -> EdgeList:
    """Cd–Se edges only from distance graph."""

    edges = relaxed_edges(list(symbols), coords, cutoffs)
    cation, anion = spec.core.cation, spec.core.anion
    core = []
    for a, b in edges:
        pair = {symbols[a], symbols[b]}
        if pair == {cation, anion}:
            core.append((min(a, b), max(a, b)))
    return tuple(sorted(core))


def full_opt_relaxation_raw(
    pack: Optional[GeometryPack],
    growth: GrowthConfig,
) -> Dict[str, Any]:
    """Pack ``relaxation:`` overlay for move A/B full geo-opt.

    Caps ``max_steps`` at ``growth.child_full_opt_cycles`` (default 150).
    Move-B pre-cleanup uses ``local_cleanup_structure`` instead and keeps
    its own 20-cycle cap.
    """

    base: Dict[str, Any] = {}
    if pack is not None and isinstance(pack.raw, dict):
        base = dict(pack.raw.get("relaxation") or {})
    base["enabled"] = True
    if growth.child_full_opt:
        base["method"] = growth.child_full_opt
    cycles = int(growth.child_full_opt_cycles)
    if cycles > 0:
        base["max_steps"] = cycles
    return base


def overlay_pack_full_opt(
    pack: Optional[GeometryPack],
    growth: GrowthConfig,
) -> None:
    """Write the A/B maxcycle into ``pack.raw['relaxation']`` (move A reads it)."""

    if pack is None or not isinstance(getattr(pack, "raw", None), dict):
        return
    pack.raw["relaxation"] = full_opt_relaxation_raw(pack, growth)


def local_cleanup_structure(
    symbols: Sequence[str],
    coords: FloatArray,
    *,
    growth: GrowthConfig,
    pack: Optional[GeometryPack],
    structure_id: str,
) -> Tuple[FloatArray, float, bool, str]:
    """Short g-xTB / GFN cleanup.  Returns (coords, time_s, ok, note).

    Important: the pack full-opt ``timeout_s`` (often 1800 s) must **not** be
    inherited here — otherwise a bad placed seed hangs the whole growth step.
    ``max_steps`` is written to an xtb ``xcontrol`` for the CLI path.
    """

    if not growth.local_cleanup_enabled or growth.local_cleanup_cycles <= 0:
        return np.asarray(coords, dtype=float), 0.0, False, "cleanup_disabled"

    from .xtb_relax import XtbSettings, relax_structures

    method = growth.local_cleanup_method
    # Prefer pack relaxation binary/env when method is g-xTB, but cap wall time
    base: Dict[str, Any] = {}
    if pack is not None and isinstance(pack.raw, dict):
        base = dict(pack.raw.get("relaxation") or {})
    base["enabled"] = True
    base["method"] = method
    base["max_steps"] = int(growth.local_cleanup_cycles)
    # Always override pack full-opt timeout (setdefault is wrong here)
    base["timeout_s"] = float(
        min(float(base.get("timeout_s") or 60.0), 60.0)
    )
    # do not re-check connectivity during a short prelax
    base["check_connectivity"] = False
    settings = XtbSettings.from_pack(base)
    t0 = time.perf_counter()
    try:
        results = relax_structures(
            [
                {
                    "id": f"cleanup-{structure_id}",
                    "symbols": list(symbols),
                    "positions": np.asarray(coords, dtype=float).tolist(),
                }
            ],
            settings,
            None,
        )
    except Exception as exc:  # noqa: BLE001
        dt = time.perf_counter() - t0
        return (
            np.asarray(coords, dtype=float),
            dt,
            False,
            f"cleanup_exc:{exc}",
        )
    dt = time.perf_counter() - t0
    xr = results[0]
    if not xr.ok or xr.coordinates is None:
        return (
            np.asarray(coords, dtype=float),
            dt,
            False,
            f"cleanup_fail:{xr.error or 'no_coords'}",
        )
    return (
        np.asarray(xr.coordinates, dtype=float),
        dt,
        True,
        f"cleanup_ok maxcycle={growth.local_cleanup_cycles} t={dt:.1f}s",
    )


def compact_growth_id(k: int, p: int, move: str, serial: int) -> str:
    """Short, non-nested child id.  k/p live in the path; lineage is metadata.

    Move B used to embed the full parent ``structure_id``, so names grew
    one generation per step and hit filesystem limits around k ~ 8–10.
    """

    raw = str(move).lower()
    if raw in {"b", "coord", "coordinate"}:
        letter = "B"
    elif raw in {"z", "zb", "zb_sites"}:
        letter = "Z"
    else:
        letter = "A"
    return f"k{int(k):03d}_p{int(p):03d}_{letter}{int(serial):04d}"


def parse_compact_serial(structure_id: str, *, move: str = "B") -> Optional[int]:
    """Return the serial in ``k003_p002_B0007``, or None if not that scheme."""

    letter = "B" if str(move).lower() in {"b", "coord", "coordinate"} else "A"
    m = re.fullmatch(
        rf"k(\d+)_p(\d+)_{letter}(\d+)",
        str(structure_id),
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return int(m.group(3))


def _lineage_key(
    k: int, p: int, parent_id: str, shed: int, p_m: int
) -> Tuple[int, int, str, int, int]:
    return (int(k), int(p), str(parent_id), int(shed), int(p_m))


def _lineage_energy_map(
    output_dir: Optional[Path],
) -> Dict[Tuple[int, int, str, int, int], Tuple[str, float]]:
    """(k, p, parent_id, shed, p_m) → (structure_id, energy_eV) from index."""

    out: Dict[Tuple[int, int, str, int, int], Tuple[str, float]] = {}
    if output_dir is None:
        return out
    path = Path(output_dir) / "index.csv"
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            move = (row.get("move") or "").strip().lower()
            if move not in {"coord", "b"}:
                continue
            parent_id = (row.get("parent_id") or "").strip()
            sid = (row.get("structure_id") or "").strip()
            if not parent_id or not sid:
                continue
            try:
                k = int(row["k"])
                p = int(row["p"])
                shed = int(row.get("shed") or 0)
                p_m = int(row.get("p_m") or 0)
                energy = float(row.get("xtb_energy_eV") or "")
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(energy):
                continue
            out[_lineage_key(k, p, parent_id, shed, p_m)] = (sid, energy)
    return out


def _used_b_serials(output_dir: Optional[Path], k: int, p: int) -> set:
    """Serials already taken by ``k###_p###_B####`` in this bin."""

    used: set = set()
    if output_dir is None:
        return used
    bdir = Path(output_dir) / f"k{int(k):03d}" / f"p{int(p):03d}"
    if bdir.is_dir():
        for path in bdir.glob("*_xtb*.xyz"):
            stem = path.name.split("_xtb", 1)[0]
            serial = parse_compact_serial(stem, move="B")
            if serial is not None:
                used.add(serial)
    index_path = Path(output_dir) / "index.csv"
    if index_path.is_file():
        with index_path.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                try:
                    if int(row.get("k") or -1) != int(k):
                        continue
                    if int(row.get("p") or -1) != int(p):
                        continue
                except ValueError:
                    continue
                serial = parse_compact_serial(
                    str(row.get("structure_id") or ""), move="B"
                )
                if serial is not None:
                    used.add(serial)
    return used


def assign_compact_b_ids(
    seeds: Sequence[CoordSeed],
    output_dir: Optional[Path] = None,
) -> None:
    """Give each B seed a short id; reuse a finished lineage on restart.

    New files are ``k###_p###_B0001_xtb.xyz``.  Parent, shed and p_m stay
    in ``index.csv`` and the XYZ comment — not in the filename.
    """

    if not seeds:
        return
    lineage = _lineage_energy_map(output_dir)
    by_bin: Dict[Tuple[int, int], List[CoordSeed]] = defaultdict(list)
    for seed in seeds:
        by_bin[(int(seed.k), int(seed.p))].append(seed)
    for (k, p), group in by_bin.items():
        used = _used_b_serials(output_dir, k, p)
        occupied = {
            hit[0]
            for s in group
            if (hit := lineage.get(
                _lineage_key(s.k, s.p, s.parent_id, s.shed, s.p_m)
            ))
        }
        next_serial = (max(used) + 1) if used else 1
        for seed in group:
            hit = lineage.get(
                _lineage_key(seed.k, seed.p, seed.parent_id, seed.shed, seed.p_m)
            )
            if hit is not None:
                seed.structure_id = hit[0]
                continue
            current = parse_compact_serial(seed.structure_id, move="B")
            if (
                current is not None
                and seed.structure_id.startswith(f"k{k:03d}_p{p:03d}_")
                and current not in used
                and seed.structure_id not in occupied
            ):
                used.add(current)
                occupied.add(seed.structure_id)
                continue
            while next_serial in used:
                next_serial += 1
            seed.structure_id = compact_growth_id(k, p, "B", next_serial)
            used.add(next_serial)
            occupied.add(seed.structure_id)
            next_serial += 1


def _reject_new_cdse_4rings(spec: NucleationSpec) -> bool:
    return bool(getattr(spec.graph_rules, "reject_new_cdse_4rings", False))


def _parent_cdse_n4(parent: ParentStructure, spec: NucleationSpec) -> int:
    """Cd–Se 4-ring count on the parent (relaxed XYZ when present)."""

    coords = getattr(parent, "coordinates", None)
    if coords is None:
        return 0
    try:
        return int(describe_structure(parent.symbols, coords, spec).n4)
    except Exception:
        return 0


def _graph_cdse_n4(
    edges: Sequence[Edge],
    *,
    k: int,
    p: int,
    spec: NucleationSpec,
) -> int:
    """Cd–Se 4-ring count on a construction core (no coordinates)."""

    anion, cation = spec.core.anion, spec.core.cation
    symbols = [anion] * int(k) + [cation] * (int(k) + int(p))
    try:
        return int(describe_graph(symbols, edges, spec).n4)
    except Exception:
        return 0


def _gained_cdse_4rings_3d(
    parent: ParentStructure,
    symbols: Sequence[str],
    coords: Any,
    spec: NucleationSpec,
) -> bool:
    """True when a 3D child has more Cd–Se diamonds than its parent."""

    if not _reject_new_cdse_4rings(spec):
        return False
    try:
        child_n4 = int(describe_structure(symbols, coords, spec).n4)
    except Exception:
        return False
    return child_n4 > _parent_cdse_n4(parent, spec)


def _drop_cores_with_new_4rings(
    cores: Sequence[EdgeList],
    *,
    parent: ParentStructure,
    k_child: int,
    p_child: int,
    spec: NucleationSpec,
) -> List[EdgeList]:
    """Move A: drop child cores that *gained* a Cd–Se 4-ring vs the parent.

    Same rule as Move B.  Cd–Se–Cd–Cl rhombi are not n4.  A child that
    merely *keeps* a parent diamond is kept.
    """

    if not _reject_new_cdse_4rings(spec) or not cores:
        return list(cores)
    parent_n4 = _parent_cdse_n4(parent, spec)
    kept: List[EdgeList] = []
    for edges in cores:
        if _graph_cdse_n4(edges, k=k_child, p=p_child, spec=spec) > parent_n4:
            continue
        kept.append(edges)
    return kept


def build_coord_seed(
    parent: ParentStructure,
    *,
    s: int,
    p_m: int,
    growth: GrowthConfig,
    spec: NucleationSpec,
    pack: Optional[GeometryPack],
    cutoffs: Mapping[Tuple[str, str], float],
    serial: int,
) -> Optional[CoordSeed]:
    """Move B for one (parent, s, p_m): WBO shed → place monomer → cleanup."""

    packages = identify_packages(parent, spec)
    if s > len(packages):
        return None
    try:
        symbols, coords, edges, shed_scores = shed_packages_coords(
            parent, s=s, packages=packages
        )
        # After shed: expect k Se, k+(p-s) Cd, 2(p-s) Cl
        p_left = parent.p - s
        exp_after_shed = expected_composition(
            parent.k,
            p_left,
            cation=spec.core.cation,
            anion=spec.core.anion,
            ligand=spec.precursor.ligand,
        )
        got_after = composition_counts(symbols)
        if got_after != exp_after_shed:
            # Overlapping packages or mis-tagged parent — refuse this channel
            return None

        symbols, coords, edges = place_monomer_and_packages(
            symbols,
            coords,
            edges,
            k_parent=parent.k,
            p_after_shed=p_left,
            p_m=p_m,
            spec=spec,
            pack=pack,
        )
    except (ValueError, IndexError) as exc:
        return None

    k_child = parent.k + 1
    p_child = parent.p - s + p_m
    exp = expected_composition(
        k_child,
        p_child,
        cation=spec.core.cation,
        anion=spec.core.anion,
        ligand=spec.precursor.ligand,
    )
    got = composition_counts(symbols)
    if got != exp:
        # Do not full-opt a misbuilt stoichiometry (would look like a "weird"
        # isomer energy for the same [CdSe]_k(CdCl2)_p label).
        return None

    # Short per-bin serial.  Parent / s / p_m stay in index.csv + XYZ comment
    # so names do not nest the parent id (that grew past NAME_MAX).
    sid = compact_growth_id(k_child, p_child, "B", serial)

    if _gained_cdse_4rings_3d(parent, symbols, coords, spec):
        return None

    q = _formal_charge(symbols, spec)
    cleanup_s = 0.0
    cleanup_ok = False
    notes = f"shed={s} p_m={p_m} charge={q} atoms={got}"
    if growth.local_cleanup_enabled:
        if growth.require_charge_neutral_for_cleanup and q != 0:
            notes += " cleanup=skipped(non-neutral)"
        else:
            coords, cleanup_s, cleanup_ok, cnote = local_cleanup_structure(
                symbols,
                coords,
                growth=growth,
                pack=pack,
                structure_id=sid,
            )
            notes += f" {cnote}"
            # cleanup can in principle change little; re-check counts unchanged
            if composition_counts(
                [symbols[i] for i in range(len(symbols))]
            ) != exp:
                pass  # symbols unchanged by cleanup
            if _gained_cdse_4rings_3d(parent, symbols, coords, spec):
                return None

    core = core_edges_from_coords(symbols, coords, spec=spec, cutoffs=cutoffs)
    if not core:
        # fall back: edges from placement that are Cd–Se
        cation, anion = spec.core.cation, spec.core.anion
        core = tuple(
            sorted(
                (min(a, b), max(a, b))
                for a, b in edges
                if {symbols[a], symbols[b]} == {cation, anion}
            )
        )
    return CoordSeed(
        k=k_child,
        p=p_child,
        structure_id=sid,
        parent_id=parent.structure_id,
        shed=s,
        p_m=p_m,
        symbols=symbols,
        coordinates=coords,
        core_edges=core,
        wbo_scores=shed_scores,
        cleanup_s=cleanup_s,
        cleanup_ok=cleanup_ok,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Growth step
# ---------------------------------------------------------------------------


def grow_cores_from_parents(
    parents: Sequence[ParentStructure],
    *,
    growth: GrowthConfig,
    spec: NucleationSpec,
    pack: Optional[GeometryPack] = None,
) -> GrowthStepResult:
    """Build child catalogs from parents using both growth moves.

    **Move A (graph):** combinatorial precursor-Cd shed on the core graph +
    monomer attach + p_m inflation → unique cores for redecorate/motif_factor.

    **Move B (coord):** when ``geometry.start_from: relaxed_coords``, WBO
    package shed on parent 3D → place CdSe + p_m CdCl2 → optional short
    cleanup → seed frames for full opt (and core added to catalog).

    **Move Z (zb_sites):** load the persistent zinc-blende occupation,
    shed extra Cd, and fill a vacant CdSe pair (+ p_m precursor Cd).  The
    relaxed parent supplies only soft local feedback.  Pack graph rules place
    Cl, anchored multi-start reconstruction builds the 3D starts, and only a
    topology-preserving converged g-xTB endpoint can propagate.
    """

    if not parents:
        return GrowthStepResult(
            k_from=0,
            k_to=1,
            parents_selected=0,
            channels=[],
            skeleton_catalog={},
        )
    k = parents[0].k
    window = growth.window_for(k)
    catalog: Dict[Tuple[int, int], Dict[EdgeList, None]] = {}
    channels: List[GrowthChannelResult] = []
    parent_records: List[Dict[str, Any]] = []
    coord_seeds: Dict[Tuple[int, int], List[CoordSeed]] = defaultdict(list)
    cutoffs = bond_cutoffs_from_spec(spec)
    seed_serial = 0
    bin_serial: Dict[Tuple[int, int], int] = defaultdict(int)
    use_coord = bool(window.move_coord) and growth.use_coord_carry
    use_graph = bool(window.move_graph)
    use_zb = bool(getattr(window, "move_zb_sites", False))
    zb_seeds: Dict[Tuple[int, int], List[Any]] = defaultdict(list)
    zb_seen: Dict[str, Any] = {}
    zb_model = None
    zb_stats = None
    if use_zb:
        from .molecular_zb_growth import (
            ZbGrowStats,
            ensure_occupation_identity,
            grow_zb_children,
            lattice_k1_occupation,
            lattice_model,
        )

        zb_model = lattice_model(spec)
        zb_stats = ZbGrowStats()

    for parent in parents:
        core = parent_core_in_blocks(parent, spec)
        packages = identify_packages(parent, spec)
        n_pkg = len(packages) if packages else parent.p
        max_s = min(
            window.s_max_for(parent.k, parent.p),
            parent.p,
            n_pkg,
        )
        if window.prefer_low_shed:
            s_order = list(range(0, max_s + 1))
        else:
            s_order = list(range(max_s, -1, -1))

        parent_records.append(
            {
                "structure_id": parent.structure_id,
                "k": parent.k,
                "p": parent.p,
                "energy_eV": parent.energy_eV,
                "n_packages": len(packages),
                "has_wbo": parent.wbo is not None,
                "wbo_source": parent.wbo_source,
                "source": parent.source_path,
                "minimum_id": parent.minimum_id,
                "minimum_representative_id": parent.minimum_representative_id,
                "minimum_member_ids": list(parent.minimum_member_ids),
                "minimum_occupation_ids": list(parent.minimum_occupation_ids),
                "minimum_multiplicity": int(parent.minimum_multiplicity),
            }
        )

        zb_occ = None
        if use_zb and zb_model is not None and zb_stats is not None:
            zb_stats.parents += 1
            stored = getattr(parent, "zb_occupation", None)
            why = "missing_occupation_manifest"
            if stored is not None:
                zb_occ = ensure_occupation_identity(stored, zb_model)
                if zb_occ.k != parent.k or zb_occ.p != parent.p:
                    why = (
                        f"stored_stoich_k{zb_occ.k}p{zb_occ.p}"
                    )
                    zb_occ = None
                else:
                    zb_occ.parent_id = parent.structure_id
                    why = "stored_occupation"
            # A legacy parent tree can initialize Move Z only at k=1.  Every
            # later step must load the occupation written by the prior step;
            # relaxed-coordinate snapping is intentionally not a lineage path.
            if zb_occ is None and parent.k == 1:
                zb_occ = lattice_k1_occupation(spec, zb_model, parent.p)
                if zb_occ is not None:
                    zb_occ.parent_id = parent.structure_id
                    why = "lattice_k1"
            if zb_occ is None:
                zb_stats.snap_fail += 1
                if str(why).startswith("n4"):
                    zb_stats.n4_reject += 1
            else:
                zb_stats.snapped += 1

        for s in s_order:
            p_out = parent.p - s
            if p_out < 0:
                continue

            # ---- Move A: graph catalog ----
            children_base: List[EdgeList] = []
            if use_graph and core is not None:
                children_base = shed_and_grow(
                    core,
                    k=parent.k,
                    p=parent.p,
                    p_out=p_out,
                    spec=spec,
                    max_children=window.max_children_per_channel,
                    attach=window.attach,
                )
                children_base = _drop_cores_with_new_4rings(
                    children_base,
                    parent=parent,
                    k_child=parent.k + 1,
                    p_child=p_out,
                    spec=spec,
                )
                children_base = _rank_child_cores(
                    children_base,
                    k=parent.k + 1,
                    p=p_out,
                    spec=spec,
                    cap=window.max_children_per_channel,
                )
            for p_m in window.monomer_p_values:
                p_child = p_out + p_m
                if not window.allow_p_child(parent.k + 1, p_child):
                    continue
                children_pm: List[EdgeList] = []
                if use_graph:
                    children_pm = _inflate_cores_with_precursor(
                        children_base,
                        k_child=parent.k + 1,
                        p_from=p_out,
                        p_to=p_child,
                        spec=spec,
                    )
                    children_pm = _drop_cores_with_new_4rings(
                        children_pm,
                        parent=parent,
                        k_child=parent.k + 1,
                        p_child=p_child,
                        spec=spec,
                    )
                    children_pm = _rank_child_cores(
                        children_pm,
                        k=parent.k + 1,
                        p=p_child,
                        spec=spec,
                        cap=window.max_children_per_channel,
                    )
                    _store_channel(
                        catalog,
                        channels,
                        parent,
                        s,
                        p_m,
                        parent.k + 1,
                        p_child,
                        children_pm,
                        move="graph",
                    )

                # ---- Move B: coordinate carry-over ----
                if use_coord:
                    seed_serial += 1
                    if seed_serial == 1 or seed_serial % 5 == 0:
                        # visible progress: cleanup can take tens of seconds each
                        print(
                            f"[growth] move B seed {seed_serial}: "
                            f"parent={parent.structure_id} s={s} p_m={p_m} "
                            f"(WBO shed + place"
                            f"{'+cleanup' if growth.local_cleanup_enabled else ''})",
                            flush=True,
                        )
                    p_child_guess = parent.p - s + p_m
                    seed = build_coord_seed(
                        parent,
                        s=s,
                        p_m=p_m,
                        growth=growth,
                        spec=spec,
                        pack=pack,
                        cutoffs=cutoffs,
                        serial=bin_serial[(parent.k + 1, p_child_guess)] + 1,
                    )
                    if seed is not None:
                        bin_serial[(seed.k, seed.p)] += 1
                        seed.structure_id = compact_growth_id(
                            seed.k, seed.p, "B", bin_serial[(seed.k, seed.p)]
                        )
                        coord_seeds[(seed.k, seed.p)].append(seed)
                        # also register core for redecorate diversity
                        if seed.core_edges:
                            bucket = catalog.setdefault((seed.k, seed.p), {})
                            if seed.core_edges not in bucket:
                                bucket[seed.core_edges] = None
                        channels.append(
                            GrowthChannelResult(
                                parent_id=parent.structure_id,
                                k_parent=parent.k,
                                p_parent=parent.p,
                                shed=s,
                                p_m=p_m,
                                k_child=seed.k,
                                p_child=seed.p,
                                n_cores=1 if seed.core_edges else 0,
                                core_edges=(
                                    [seed.core_edges] if seed.core_edges else []
                                ),
                                move="coord",
                            )
                        )
                        if seed_serial == 1 or seed_serial % 5 == 0:
                            print(
                                f"[growth]   -> {seed.structure_id}  "
                                f"cleanup={seed.cleanup_ok} "
                                f"t={seed.cleanup_s:.1f}s  {seed.notes}",
                                flush=True,
                            )

                if use_zb and zb_occ is not None and zb_model is not None:
                    kids = grow_zb_children(
                        zb_occ,
                        s=s,
                        p_m=p_m,
                        spec=spec,
                        model=zb_model,
                        cap=int(window.max_children_per_channel),
                        stats=zb_stats,
                        relaxed_parent_coordinates=np.asarray(
                            parent.coordinates, dtype=float
                        )[: len(zb_occ.symbols)],
                        parent_wbo=(
                            parent.wbo
                            if tuple(parent.symbols[: len(zb_occ.symbols)])
                            == tuple(zb_occ.symbols)
                            else None
                        ),
                    )
                    kept_kids = []
                    for kid in kids:
                        uniq = str(kid.occupation_id)
                        if uniq in zb_seen:
                            existing = zb_seen[uniq]
                            existing.parent_occupation_ids = tuple(
                                sorted(
                                    set(existing.parent_occupation_ids)
                                    | set(kid.parent_occupation_ids)
                                )
                            )
                            existing.parent_structure_ids = tuple(
                                sorted(
                                    set(existing.parent_structure_ids)
                                    | set(kid.parent_structure_ids)
                                )
                            )
                            continue
                        zb_seen[uniq] = kid
                        kept_kids.append(kid)
                        zb_seeds[(kid.k, kid.p)].append(kid)
                        bucket = catalog.setdefault((kid.k, kid.p), {})
                        if kid.core_edges not in bucket:
                            bucket[kid.core_edges] = None
                    if zb_stats is not None:
                        zb_stats.children += len(kept_kids)
                    if kept_kids:
                        channels.append(
                            GrowthChannelResult(
                                parent_id=parent.structure_id,
                                k_parent=parent.k,
                                p_parent=parent.p,
                                shed=s,
                                p_m=p_m,
                                k_child=parent.k + 1,
                                p_child=p_child,
                                n_cores=len(kept_kids),
                                core_edges=[kid.core_edges for kid in kept_kids],
                                move="zb_sites",
                            )
                        )

    skeleton_catalog = {
        key: list(edges_map.keys()) for key, edges_map in catalog.items()
    }
    return GrowthStepResult(
        k_from=k,
        k_to=k + 1,
        parents_selected=len(parents),
        channels=channels,
        skeleton_catalog=skeleton_catalog,
        parent_records=parent_records,
        coord_seeds=dict(coord_seeds),
        zb_seeds=dict(zb_seeds),
        zb_stats=zb_stats,
    )


def _rank_child_cores(
    cores: Sequence[EdgeList],
    *,
    k: int,
    p: int,
    spec: NucleationSpec,
    cap: int,
) -> List[EdgeList]:
    """Keep the lowest-construction-cost cores (F6 / no diamonds)."""

    from .soft_rules import construction_score

    items = list(cores)
    if not items:
        return []
    if not bool(getattr(spec.graph_rules, "rank_cores_by_fusion", False)):
        return items
    anion, cation = spec.core.anion, spec.core.cation
    symbols = [anion] * int(k) + [cation] * (int(k) + int(p))
    scored = [
        (
            construction_score(describe_graph(symbols, edges, spec), spec),
            edges,
        )
        for edges in items
    ]
    scored.sort(key=lambda kv: kv[0])
    keep = scored[: max(0, int(cap))] if cap > 0 else scored
    return [edges for _cost, edges in keep]


def _remap_after_drop(
    se_ids: List[int], remaining_cd: List[int], k: int, p_new: int
) -> Dict[int, int]:
    se_out, cd_out = range(0, k), range(k, k + k + p_new)
    mapping = {}
    for new, old in zip(se_out, sorted(se_ids)):
        mapping[old] = new
    for new, old in zip(cd_out, sorted(remaining_cd)):
        mapping[old] = new
    return mapping


def _inflate_cores_with_precursor(
    cores: Sequence[EdgeList],
    *,
    k_child: int,
    p_from: int,
    p_to: int,
    spec: NucleationSpec,
) -> List[EdgeList]:
    """Add (p_to - p_from) precursor Cd slots to cores at (k_child, p_from).

    Extra Cd are appended in the canonical block with **no** Se edges; they
    become Cl hosts on redecoration.  If p_to == p_from, return cores as-is.
    Connectivity of the bonded component is preserved; isolated precursor Cd
    are allowed in the block layout used by decoration.
    """

    if p_to < p_from:
        return []
    if p_to == p_from:
        return list(cores)
    se_from = range(0, k_child)
    cd_from = range(k_child, k_child + k_child + p_from)
    out: List[EdgeList] = []
    seen: set = set()
    from .molecular_lineage import _core_is_legal

    for edges in cores:
        mapped = []
        ok = True
        for a, b in edges:
            def new_id(x: int) -> Optional[int]:
                if x in se_from:
                    return x
                if x in cd_from:
                    return k_child + (x - k_child)
                return None

            na, nb = new_id(a), new_id(b)
            if na is None or nb is None:
                ok = False
                break
            mapped.append((min(na, nb), max(na, nb)))
        if not ok:
            continue
        key = tuple(sorted(mapped))
        # Always bond extra precursor Cd so inorganic stays connected.
        key2 = _bond_extra_precursors(key, k_child, p_from, p_to, spec)
        if key2 is None:
            continue
        if not _core_is_legal(key2, k_child, p_to, spec):
            continue
        if key2 in seen:
            continue
        seen.add(key2)
        out.append(key2)
    return out


def _bond_extra_precursors(
    edges: EdgeList,
    k: int,
    p_from: int,
    p_to: int,
    spec: NucleationSpec,
) -> Optional[EdgeList]:
    """Attach new precursor Cd to lowest-degree Se (keeps graph connected)."""

    se_ids = list(range(0, k))
    cd_old = list(range(k, k + k + p_from))
    cd_new = list(range(k + k + p_from, k + k + p_to))
    g = nx.Graph()
    g.add_nodes_from(se_ids + cd_old + cd_new)
    g.add_edges_from(edges)
    max_se = int(spec.graph_rules.max_cn.get(spec.core.anion, 4))
    e_list = list(edges)
    for cd in cd_new:
        # pick Se with free valence and lowest degree
        candidates = [
            se for se in se_ids if g.degree(se) < max_se
        ]
        if not candidates:
            return None
        se = min(candidates, key=lambda s: g.degree(s))
        g.add_edge(cd, se)
        e_list.append((min(cd, se), max(cd, se)))
    return tuple(sorted(set(e_list)))


def _apply_opt_budget(
    result: GrowthStepResult, max_opts: int
) -> GrowthStepResult:
    """Cap coordinate seeds plus unique cores so a survey stays a few hundred.

    Seeds are kept first (one opt each); remaining slots become cores that
    will be redecorated.  Catalog keys and channel lists are left intact.
    """

    if max_opts <= 0:
        return result
    kept_seeds: Dict[Tuple[int, int], List[CoordSeed]] = {}
    used = 0
    for key in sorted(result.coord_seeds):
        bucket: List[CoordSeed] = []
        for seed in result.coord_seeds[key]:
            if used >= max_opts:
                break
            bucket.append(seed)
            used += 1
        if bucket:
            kept_seeds[key] = bucket
        if used >= max_opts:
            break
    remain = max(0, max_opts - used)
    kept_cat: Dict[Tuple[int, int], List[EdgeList]] = {}
    for key in sorted(result.skeleton_catalog):
        cores = list(result.skeleton_catalog[key])
        if remain <= 0:
            break
        take = cores[:remain]
        kept_cat[key] = take
        remain -= len(take)
    result.coord_seeds = kept_seeds
    result.skeleton_catalog = kept_cat
    return result


def _store_channel(
    catalog: Dict[Tuple[int, int], Dict[EdgeList, None]],
    channels: List[GrowthChannelResult],
    parent: ParentStructure,
    s: int,
    p_m: int,
    k_child: int,
    p_child: int,
    children: Sequence[EdgeList],
    *,
    move: str = "graph",
) -> None:
    if not children:
        channels.append(
            GrowthChannelResult(
                parent_id=parent.structure_id,
                k_parent=parent.k,
                p_parent=parent.p,
                shed=s,
                p_m=p_m,
                k_child=k_child,
                p_child=p_child,
                n_cores=0,
                core_edges=[],
                move=move,
            )
        )
        return
    bucket = catalog.setdefault((k_child, p_child), {})
    kept = []
    for e in children:
        if e not in bucket:
            bucket[e] = None
            kept.append(e)
    channels.append(
        GrowthChannelResult(
            parent_id=parent.structure_id,
            k_parent=parent.k,
            p_parent=parent.p,
            shed=s,
            p_m=p_m,
            k_child=k_child,
            p_child=p_child,
            n_cores=len(kept),
            core_edges=kept,
            move=move,
        )
    )


class GrowthLog:
    """Compact growth logger: stages + job lines; hides verbose molecular spam.

    Always prints lines starting with ``[growth]`` or ``[growth-job]``.
    With ``verbose=True``, also prints the rest (motif details, etc.).

    Optional ``log_path``: append-only file so Slurm resubmits continue the
    same log (also print to stdout unless ``quiet``).
    """

    def __init__(
        self,
        *,
        verbose: bool = False,
        quiet: bool = False,
        log_path: Optional[Path] = None,
    ) -> None:
        self.verbose = bool(verbose)
        self.quiet = bool(quiet)
        self._job_i = 0
        self._log_path = Path(log_path) if log_path is not None else None
        self._log_fh = None
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = self._log_path.open("a", encoding="utf-8")
            self._emit(
                f"\n######## growth log resume/append  "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}  ########\n"
            )
        # global tallies (all bins in this growth step)
        self.n_gxtb = 0
        self.n_merge = 0
        self.n_fail = 0
        # current bin context
        self._bin_k: Optional[int] = None
        self._bin_p: Optional[int] = None
        self._bin_cores: int = 0
        self._bin_gxtb: int = 0
        self._bin_merge: int = 0
        self._bin_fail: int = 0
        self._cores_done: int = 0
        self._cores_total: int = 0
        self._xtb_starts: int = 1
        # current optimization block (move-B (k,p) group or move-A bin)
        self._block_i: int = 0
        self._block_n: int = 0
        self._block_seq: int = 0
        self._n_blocks: int = 0
        self._block_label: str = ""
        # wall clock
        self._t_mark: float = time.perf_counter()
        self._t_bin0: float = self._t_mark
        self._t_step0: float = self._t_mark
        self._sum_opt_s: float = 0.0
        self._sum_recon_s: float = 0.0
        self._bin_opt_s: float = 0.0
        self._bin_recon_s: float = 0.0

    def _emit(self, text: str) -> None:
        """Write to stdout (unless quiet) and append-only log file."""

        line = text if text.endswith("\n") else text + "\n"
        if not self.quiet:
            print(line, end="", flush=True)
        if self._log_fh is not None:
            self._log_fh.write(line)
            self._log_fh.flush()

    def close(self) -> None:
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def _tick(self) -> float:
        """Seconds since last mark; advances the mark."""

        now = time.perf_counter()
        dt = now - self._t_mark
        self._t_mark = now
        return dt

    def stage(self, n: int, total: int, title: str, **fields: Any) -> None:
        if self.quiet and self._log_fh is None:
            return
        self._tick()
        self._emit(f"\n=== STAGE {n}/{total}: {title} ===\n")
        for key, value in fields.items():
            self._emit(f"  {key}: {value}\n")

    def line(self, msg: str) -> None:
        if self.quiet and self._log_fh is None:
            return
        self._emit(f"[growth] {msg}\n")

    def block(self, text: str) -> None:
        """Write a multi-line block (bin rankings) to stdout and the log file.

        ``print()`` alone never reaches ``growth_run.log``.
        """

        if self.quiet and self._log_fh is None:
            return
        if not text:
            return
        self._emit(text if text.endswith("\n") else text + "\n")


    def pipeline_blurb(self, growth: "GrowthConfig") -> None:
        """One-time note: both growth moves and timing keys."""

        if self.quiet and self._log_fh is None:
            return
        self.line("growth moves:")
        self.line(
            "  A graph: combinatorial precursor-Cd shed on core graph -> "
            f"p_m={list(growth.monomer_p_values)} inflate -> Cl redecorate -> "
            "motif_factor rebuild -> full g-xTB"
        )
        if growth.use_coord_carry:
            self.line(
                "  B coord: parent XYZ -> WBO package shed (s least-bound "
                f"CdCl2) -> place CdSe+p_m CdCl2 (embed distances) -> "
                f"cleanup {'ON' if growth.local_cleanup_enabled else 'OFF'}"
                f"({growth.local_cleanup_method}, <={growth.local_cleanup_cycles} steps"
                f"{', neutral-only' if growth.require_charge_neutral_for_cleanup else ''}"
                f") -> full g-xTB"
            )
        else:
            self.line("  B coord: OFF")
        if getattr(growth, "move_zb_sites", False):
            self.line(
                "  Z zb_sites: snap parent onto zinc-blende -> "
                "shed extra Cd -> fill vacant CdSe pair + p_m Cd -> "
                "pack decorate (graph_rules 2p) -> embed.yaml 3D -> "
                "full g-xTB -> keep only if the relaxed core still "
                "embeds on zb.  Genealogy is the occupation, not the XYZ."
            )
        else:
            self.line("  Z zb_sites: OFF")
        self.line(
            f"  shed: mode={growth.shed_mode} max_shed={growth.max_shed} "
            f"prefer_low_shed={growth.prefer_low_shed} "
            f"attach={growth.attach} "
            f"p_surf β={growth.surface_beta} α={growth.surface_alpha} "
            f"slack={growth.p_slack}"
        )
        self.line(
            f"  full opt: method={growth.child_full_opt} "
            f"maxcycle={int(growth.child_full_opt_cycles)}  "
            f"(A+B geo-opt; cleanup stays "
            f"{int(growth.local_cleanup_cycles)} cycles)"
        )
        self.line(
            "  timing keys: opt= full g-xTB; recon= motif_factor (A only); "
            "cleanup= short prelax (B); +dt= wall since previous line"
        )

    def set_block_plan(self, n_blocks: int) -> None:
        """How many optimization blocks this step will open (B groups + A bins)."""

        self._n_blocks = max(0, int(n_blocks))
        self._block_seq = 0
        self._block_i = 0
        self._block_n = 0
        self._block_label = ""

    def begin_block(self, n_jobs: int, *, label: str = "") -> int:
        """Start a running ``i/N`` counter for the next group of opts.

        ``n_jobs`` is the planned number of structures to relax in this
        block (exact for move B; cores, or cores×starts, for move A).
        If more jobs arrive than planned, the denominator grows with ``i``.
        """

        self._block_seq += 1
        self._block_i = 0
        self._block_n = max(0, int(n_jobs))
        self._block_label = str(label or "")
        return self._block_seq

    def _block_tag(self) -> str:
        """``14/32`` (or ``14/?`` if the planned count is still unknown)."""

        n = int(self._block_n)
        i = int(self._block_i)
        if n <= 0:
            return f"{i:3d}/?"
        width = max(2, len(str(n)))
        return f"{i:{width}d}/{n}"

    def _advance_block_job(self) -> str:
        """Count one relax in the current block; return ``i/N`` tag."""

        self._block_i += 1
        if self._block_n > 0 and self._block_i > self._block_n:
            self._block_n = self._block_i
        return self._block_tag()

    def _block_header_bit(self) -> str:
        if self._n_blocks > 0:
            return f"block {self._block_seq}/{self._n_blocks}"
        if self._block_seq > 0:
            return f"block {self._block_seq}"
        return ""

    def set_work_plan(
        self,
        *,
        cores_total: int,
        bin_plan: Mapping[str, int],
        xtb_starts_per_graph: int = 1,
    ) -> None:
        """Print decorate/opt plan (cores known; g-xTB count not yet)."""

        self._cores_total = int(cores_total)
        self._xtb_starts = max(1, int(xtb_starts_per_graph))
        if self.quiet and self._log_fh is None:
            return
        self.line(
            f"work plan: {self._cores_total} unique child cores "
            f"across {len(bin_plan)} bins"
        )
        self.line(f"  by bin: {dict(bin_plan)}")
        self.line(
            "  note: #g-xTB calcs != #cores - each core yields 0+ Cl "
            "decorations; each *unique* graph runs "
            f"<={self._xtb_starts} g-xTB opt(s); merges skip g-xTB"
        )

    def log_channel_summary(
        self,
        channels: Sequence["GrowthChannelResult"],
        parent_records: Sequence[Mapping[str, Any]],
    ) -> None:
        """Compact shed / package accounting after core growth."""

        if self.quiet and self._log_fh is None:
            return
        by_s: Counter = Counter()
        by_pm: Counter = Counter()
        by_s_cores: Counter = Counter()
        by_pm_cores: Counter = Counter()
        by_move: Counter = Counter()
        n_ch = 0
        for ch in channels:
            n_ch += 1
            by_s[ch.shed] += 1
            by_pm[ch.p_m] += 1
            by_s_cores[ch.shed] += ch.n_cores
            by_pm_cores[ch.p_m] += ch.n_cores
            by_move[getattr(ch, "move", "graph")] += 1
        self.line(
            f"channels: {n_ch}  "
            f"(graph={by_move.get('graph', 0)}  coord={by_move.get('coord', 0)})"
        )
        s_bits = [
            f"s={s}: ch={by_s[s]} cores+={by_s_cores[s]}"
            for s in sorted(by_s)
        ]
        pm_bits = [
            f"p_m={pm}: ch={by_pm[pm]} cores+={by_pm_cores[pm]}"
            for pm in sorted(by_pm)
        ]
        self.line(f"  by shed s:  {',  '.join(s_bits) or '(none)'}")
        self.line(f"  by p_m:     {',  '.join(pm_bits) or '(none)'}")
        n_pkg = sum(int(r.get("n_packages") or 0) for r in parent_records)
        n_wbo = sum(1 for r in parent_records if r.get("has_wbo"))
        sources = Counter(
            str(r.get("wbo_source") or "none") for r in parent_records
        )
        src_txt = ", ".join(f"{k}={v}" for k, v in sorted(sources.items()))
        self.line(
            f"  parent packages: {n_pkg} total; "
            f"{n_wbo}/{len(parent_records)} parents have WBO "
            f"({src_txt})"
        )
        if n_wbo == 0 and parent_records:
            self.line(
                "  NOTE: no Wiberg files on parents (typical for g-xTB). "
                "Package ranking falls back to Cd–Cl distance "
                "(longer = weaker = shed first)."
            )

    def begin_bin(
        self,
        *,
        k: int,
        p: int,
        cores: int,
        cores_done: int,
        cores_total: int,
        jobs: Optional[int] = None,
    ) -> None:
        """Open a bin header with core accounting and a fresh ``i/N`` block."""

        self._bin_k = int(k)
        self._bin_p = int(p)
        self._bin_cores = int(cores)
        self._bin_gxtb = 0
        self._bin_merge = 0
        self._bin_fail = 0
        self._bin_opt_s = 0.0
        self._bin_recon_s = 0.0
        self._cores_done = int(cores_done)
        self._cores_total = int(cores_total)
        planned = int(jobs) if jobs is not None else int(cores)
        self.begin_block(planned, label=f"A k={k} p={p}")
        self._t_bin0 = time.perf_counter()
        self._tick()
        if self.quiet and self._log_fh is None:
            return
        after = cores_done + cores
        blk = self._block_header_bit()
        title = f"--- bin k={k} p={p} ---"
        if blk:
            title = f"{title}  {blk}"
        self.line(title)
        self.line(
            f"  cores in this bin: {cores}   "
            f"(overall cores {cores_done} done -> {after}/{cores_total} after bin)"
        )
        self.line(
            f"  opts in this block: {planned}   "
            f"(job lines show global_n  i/{planned})"
        )
        self.line(
            f"  path: core graph -> Cl decorate -> motif_factor -> g-xTB "
            f"(<={self._xtb_starts} opt/unique graph); merges free"
        )
        if self.n_gxtb or self.n_merge:
            self.line(
                f"  so far (all bins): gxtb={self.n_gxtb}  "
                f"merge={self.n_merge}  fail={self.n_fail}  "
                f"optSum={self._sum_opt_s:.0f}s reconSum={self._sum_recon_s:.0f}s"
            )

    def end_bin(
        self,
        *,
        k: int,
        p: int,
        n_iso: int,
        n_ok: int,
        n_fail: int,
        n_merged: int,
        raw_graphs: Optional[int] = None,
    ) -> None:
        """Close a bin with final job accounting."""

        if self.quiet and self._log_fh is None:
            return
        wall = time.perf_counter() - self._t_bin0
        extra = ""
        if raw_graphs is not None:
            extra = f"  raw_graphs={raw_graphs}"
        self.line(
            f"bin k={k} p={p} done: isomers={n_iso}  "
            f"with_E={n_ok}  no_E={n_fail}  graph_merges={n_merged}"
            f"{extra}"
        )
        self.line(
            f"  this bin: gxtb={self._bin_gxtb}  "
            f"merge={self._bin_merge}  fail={self._bin_fail}  "
            f"wall={wall:.1f}s  optSum={self._bin_opt_s:.1f}s  "
            f"reconSum={self._bin_recon_s:.1f}s  "
            f"(other~{max(0.0, wall - self._bin_opt_s - self._bin_recon_s):.1f}s "
            f"decorate/merge/overhead)"
        )
        self.line(
            f"  global: gxtb={self.n_gxtb}  merge={self.n_merge}  "
            f"fail={self.n_fail}"
        )
        self._tick()

    @staticmethod
    def _formula_kp(k: object, p: object) -> str:
        """ASCII formula [CdSe]_k(CdCl2)_p for log lines."""

        try:
            return f"[CdSe]_{int(k)}(CdCl2)_{int(p)}"
        except (TypeError, ValueError):
            return f"[CdSe]_{k}(CdCl2)_{p}"

    @staticmethod
    def _parent_kp_from_id(parent_id: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse k###_p### from a structure id like k002_p003_mol0010."""

        m = re.search(r"k(\d+)_p(\d+)", str(parent_id), flags=re.IGNORECASE)
        if not m:
            return None, None
        return int(m.group(1)), int(m.group(2))

    def __call__(self, msg: str) -> None:
        """ProgressCallback-compatible."""

        if self.quiet and self._log_fh is None:
            return
        text = str(msg)
        if text.startswith("[growth-job]"):
            self._job_i += 1
            dt = self._tick()
            parts = text.replace("[growth-job]", "").strip().split()
            kv: Dict[str, str] = {}
            for part in parts:
                if "=" in part:
                    key, val = part.split("=", 1)
                    kv[key] = val
            sid = kv.get("id", "?")
            e = kv.get("E_eV", "n/a")
            t = kv.get("t_s", "0")
            recon = kv.get("recon_s", "0")
            rel = kv.get("relax", "?")
            k = kv.get("k", "?")
            p = kv.get("p", "?")
            err = kv.get("err", "")
            into = kv.get("into", "")
            move = kv.get("move", "A")
            shed = kv.get("s", kv.get("shed", "-"))
            p_m = kv.get("p_m", kv.get("pm", "-"))
            parent = kv.get("parent", kv.get("parent_id", ""))
            k_par = kv.get("k_parent", kv.get("k_par", ""))
            p_par = kv.get("p_parent", kv.get("p_par", ""))
            try:
                opt_s = float(t)
            except ValueError:
                opt_s = 0.0
            try:
                recon_s = float(recon)
            except ValueError:
                recon_s = 0.0

            if rel == "merged" or e == "merged":
                self.n_merge += 1
                self._bin_merge += 1
                target = into or "?"
                self._emit(
                    f"  {self.n_merge:4d}  dup  k={k} p={p}  "
                    f"{self._formula_kp(k, p)}  "
                    f"already={target}  +dt={dt:5.1f}s\n"
                )
                return

            # resume skip: still count as completed work, not a new opt
            if err == "resume_skip":
                self.n_gxtb += 1
                self._bin_gxtb += 1
                blk = self._advance_block_job()
                child_f = self._formula_kp(k, p)
                if k_par == "" or p_par == "":
                    pk, pp = self._parent_kp_from_id(parent)
                else:
                    try:
                        pk, pp = int(k_par), int(p_par)
                    except ValueError:
                        pk, pp = self._parent_kp_from_id(parent)
                if pk is not None and pp is not None:
                    stoich = (
                        f"{self._formula_kp(pk, pp)}  -s={shed} +p_m={p_m} "
                        f"-> {child_f}"
                    )
                else:
                    stoich = f"-s={shed} +p_m={p_m} -> {child_f}"
                parent_bit = f"  parent={parent}" if parent else ""
                self._emit(
                    f"  {self.n_gxtb:4d}  {blk}  {stoich}  "
                    f"E={e:>14s}  "
                    f"opt=  0.0s clean={recon_s:4.1f}s "
                    f"+dt={dt:5.1f}s  resume_skip"
                    f"{parent_bit}\n"
                )
                return

            self.n_gxtb += 1
            self._bin_gxtb += 1
            blk = self._advance_block_job()
            self._sum_opt_s += opt_s
            self._sum_recon_s += recon_s
            self._bin_opt_s += opt_s
            self._bin_recon_s += recon_s
            if rel == "fail":
                self.n_fail += 1
                self._bin_fail += 1
            extra = f" ({err})" if err and rel == "fail" else ""
            recon_lab = "clean" if move in ("B", "coord", "b") else "recon"
            child_f = self._formula_kp(k, p)
            steps = kv.get("steps", "")
            max_steps = kv.get("max_steps", "")
            if steps or max_steps:
                cyc_bit = f" cyc={steps or '?'}/{max_steps or '?'}"
            else:
                cyc_bit = ""

            if move in ("B", "coord", "b"):
                if k_par == "" or p_par == "":
                    pk, pp = self._parent_kp_from_id(parent)
                else:
                    try:
                        pk, pp = int(k_par), int(p_par)
                    except ValueError:
                        pk, pp = self._parent_kp_from_id(parent)
                if pk is not None and pp is not None:
                    parent_f = self._formula_kp(pk, pp)
                    stoich = (
                        f"{parent_f}  -s={shed} +p_m={p_m} -> {child_f}"
                    )
                else:
                    stoich = f"-s={shed} +p_m={p_m} -> {child_f}"
                parent_bit = f"  parent={parent}" if parent else ""
                self._emit(
                    f"  {self.n_gxtb:4d}  {blk}  {stoich}  "
                    f"E={e:>14s}  "
                    f"opt={opt_s:5.1f}s{cyc_bit} {recon_lab}={recon_s:4.1f}s "
                    f"+dt={dt:5.1f}s  {rel}{extra}"
                    f"{parent_bit}\n"
                )
            else:
                self._emit(
                    f"  {self.n_gxtb:4d}  {blk}  k={k} p={p}  {child_f}  "
                    f"redecorate  "
                    f"E={e:>14s}  "
                    f"opt={opt_s:5.1f}s{cyc_bit} {recon_lab}={recon_s:4.1f}s "
                    f"+dt={dt:5.1f}s  {rel}{extra}\n"
                )
            return
        if text.startswith("[growth]"):
            self._emit(text if text.endswith("\n") else text + "\n")
            return
        if self.verbose:
            self._emit(text if text.endswith("\n") else text + "\n")



def run_growth_step(
    *,
    run_dir: Path,
    k_from: int,
    growth: GrowthConfig,
    map_spec: NucleationSpec,
    pack: Optional[GeometryPack] = None,
    p_parents: Optional[Sequence[int]] = None,
    decorate: bool = True,
    embed: bool = True,
    output_dir: Optional[Path] = None,
    progress: Optional[Any] = None,
    resume: bool = True,
) -> GrowthStepResult:
    """Select parents at ``k_from``, grow cores to k+1, optionally decorate.

    When ``decorate`` is true, each child (k, p) bin is built with
    ``enumerate_molecular_bin(..., precomputed_skeletons=...)`` and written
    under ``output_dir`` if given.

    Restart (``resume=True``, default):
      * skip finished k-step if ``.step_kXXX_complete`` exists
      * skip move-B opts already in ``index.csv`` / ``*_xtb.xyz``
      * skip move-A bins with ``.bin_A_complete`` marker (reload ranks from index)
    """

    log = progress if isinstance(progress, GrowthLog) else None
    out_path = Path(output_dir) if output_dir else None

    # Entire step already finished
    if (
        resume
        and out_path is not None
        and step_complete_marker(out_path, k_from).is_file()
    ):
        if log:
            log.line(
                f"RESUME: step k={k_from}->{k_from + 1} already complete "
                f"({step_complete_marker(out_path, k_from).name}); skipping work"
            )
        # Minimal result so multi-step CLI can continue
        return GrowthStepResult(
            k_from=k_from,
            k_to=k_from + 1,
            parents_selected=0,
            channels=[],
            skeleton_catalog={},
            parent_records=[],
            coord_seeds={},
        )

    def _p(msg: str) -> None:
        if progress is None:
            return
        if log is not None:
            if msg.startswith("[growth]") or msg.startswith("==="):
                # stage/line helpers use print directly on GrowthLog
                pass
            progress(msg if msg.startswith("[") else f"[growth] {msg}")
        else:
            progress(msg)

    # load, grow, [B opt], [A decorate], [merged rank], write
    n_stages = 5 if decorate else 3

    if log:
        log.stage(
            1,
            n_stages,
            "load parents",
            parents_dir=str(run_dir),
            k_from=k_from,
            p_filter=list(p_parents) if p_parents else "all",
        )
    parents_all = load_parents_from_run(
        run_dir, k=k_from, spec=map_spec, p_values=p_parents
    )
    parents = select_parents(parents_all, growth, map_spec)
    window = growth.window_for(k_from)
    if not parents_all:
        have = parent_k_inventory(run_dir)
        have_txt = (
            ", ".join(f"k={k} (n={n})" for k, n in have.items())
            if have
            else "none"
        )
        hint = ""
        if have and k_from not in have:
            first = min(have)
            last = max(have)
            hint = (
                f"  --k-from is the *parent* size already in --parents, "
                f"not the first child.  This run has {have_txt}.  "
                f"To grow onward use --k-from {last} "
                f"(e.g. --k-from {last} --k-to {last + 2})."
            )
        msg = (
            f"no converged parents at k={k_from} in {run_dir} "
            f"(available: {have_txt})"
        )
        if log:
            log.line(msg)
            if hint:
                log.line(hint)
        raise FileNotFoundError(msg + (("\n" + hint) if hint else ""))
    if log:
        selected_minima = len(
            {
                parent.minimum_id or f"endpoint:{parent.structure_id}"
                for parent in parents
            }
        )
        log.line(
            f"loaded {len(parents_all)} converged parents → "
            f"selected {selected_minima} relaxed minima / "
            f"{len(parents)} ZB routes "
            f"(window={window.energy_window_eV} eV, "
            f"cap={window.max_skeletons_cap} skel, "
            f"≤{window.decorations_per_skeleton} dec/core, "
            f"min_p={window.min_p_parent}"
            f"{', soft rank' if window.soft_rules.enabled else ''})"
        )
        log.line(f"envelope: {window.describe()}")
    elif progress:
        progress(
            f"[growth] k={k_from}: loaded {len(parents_all)} parents, "
            f"selected {len(parents)}"
        )

    if log:
        log.pipeline_blurb(growth)
        log.stage(
            2,
            n_stages,
            f"grow cores  k={k_from} → k={k_from + 1}",
            packages=list(window.monomer_p_values),
            max_shed=window.max_shed,
            attach=window.attach,
            p_surf=window.p_surf(k_from) if window.surface_beta > 0 else "off",
            move_A="graph" if window.move_graph else "off",
            move_B=(
                "coord WBO/distance shed + place + cleanup"
                if window.move_coord and growth.use_coord_carry
                else "off"
            ),
            move_Z="zb_sites" if window.move_zb_sites else "off",
        )
    # load pack early so move B can use embed distances / cleanup settings
    if pack is None and map_spec.geometry_pack:
        try:
            pack = load_geometry_pack(map_spec.geometry_pack)
        except Exception:
            pack = None
    result = grow_cores_from_parents(
        parents, growth=growth, spec=map_spec, pack=pack
    )
    n_cores = sum(len(v) for v in result.skeleton_catalog.values())
    bin_plan = {
        f"k{k}p{p}": len(cores)
        for (k, p), cores in sorted(result.skeleton_catalog.items())
    }
    if log:
        log.log_channel_summary(result.channels, result.parent_records)
    # xtb_starts_per_graph from pack embed/reconstruction (default 1)
    xtb_starts = 1
    if pack is not None:
        recon = (pack.raw or {}).get("reconstruction") or {}
        try:
            xtb_starts = int(recon.get("xtb_starts_per_graph", 1))
        except (TypeError, ValueError):
            xtb_starts = 1
    elif map_spec is not None and getattr(map_spec, "geometry_pack", None):
        try:
            _tmp_pack = load_geometry_pack(map_spec.geometry_pack)
            recon = (_tmp_pack.raw or {}).get("reconstruction") or {}
            xtb_starts = int(recon.get("xtb_starts_per_graph", 1))
        except Exception:
            xtb_starts = 1

    n_seeds = sum(len(v) for v in result.coord_seeds.values())
    if log:
        n_zb = sum(len(v) for v in (result.zb_seeds or {}).values())
        log.line(
            f"channels={len(result.channels)}  unique_cores={n_cores}  "
            f"coord_seeds={n_seeds}  zb_occupations={n_zb}"
        )
        if result.zb_stats is not None:
            log.line(result.zb_stats.as_log())
        log.set_work_plan(
            cores_total=n_cores,
            bin_plan=bin_plan,
            xtb_starts_per_graph=xtb_starts,
        )
    elif progress:
        progress(
            f"[growth] k={k_from}->{k_from + 1}: "
            f"{len(result.channels)} channels, {n_cores} unique child cores, "
            f"bins={sorted(result.skeleton_catalog)}"
        )

    # Report-only: print parent-map rankings (k=1 … k_from) before child opts
    if log:
        prior = format_prior_map_rankings(
            run_dir, growth=growth, k_max=k_from
        )
        log.block(prior)

    child_minima: Dict[Tuple[int, int], Dict[str, Any]] = {}
    # Raw A/B/Z energy rows per (k,p); optionally consolidated before ranking.
    bin_ranks: Dict[Tuple[int, int], List[RankedIsomer]] = defaultdict(list)
    do_redecorate = bool(decorate and window.child_redecorate)
    if window.move_zb_sites and decorate:
        # Move Z has its own occupation-aware decorator/embedder below.  The
        # generic bin path stores only edge lists and would lose lattice-site
        # identity.  In a mixed A+Z run it remains enabled for Move A cores.
        do_redecorate = bool(window.move_graph)
    if window.max_opts_per_k > 0:
        result = _apply_opt_budget(result, window.max_opts_per_k)
        n_cores = sum(len(v) for v in result.skeleton_catalog.values())
        n_seeds = sum(len(v) for v in result.coord_seeds.values())
        if log:
            log.line(
                f"budget: max_opts_per_k={window.max_opts_per_k} → "
                f"cores={n_cores} coord_seeds={n_seeds}"
            )
    if decorate and (result.skeleton_catalog or result.coord_seeds):
        if pack is None and map_spec.geometry_pack:
            try:
                pack = load_geometry_pack(map_spec.geometry_pack)
            except Exception:
                pack = None
        if pack is not None and log is not None:
            recon = (pack.raw or {}).get("reconstruction") or {}
            try:
                xtb_starts = int(recon.get("xtb_starts_per_graph", xtb_starts))
            except (TypeError, ValueError):
                pass
            log._xtb_starts = max(1, int(xtb_starts))
        # After move-B cleanup (20 cycles).  A/B full opts read this cap.
        overlay_pack_full_opt(pack, growth)
        out = Path(output_dir) if output_dir else None
        if out:
            out.mkdir(parents=True, exist_ok=True)

        n_b_blocks = (
            sum(1 for v in result.coord_seeds.values() if v)
            if result.coord_seeds and embed
            else 0
        )
        n_a_blocks = len(result.skeleton_catalog) if do_redecorate else 0
        n_z_blocks = (
            sum(1 for v in (result.zb_seeds or {}).values() if v)
            if result.zb_seeds and embed
            else 0
        )
        if log and (n_b_blocks or n_a_blocks or n_z_blocks):
            log.line(
                f"opt blocks this step: B groups={n_b_blocks}, "
                f"A bins={n_a_blocks}, Z groups={n_z_blocks}; "
                f"job lines show  global_n  i/N_block"
            )

        # ---- Move B first: full opt of coordinate-carried seeds ----
        if result.coord_seeds and embed:
            if log:
                log.set_block_plan(n_b_blocks)
                log.stage(
                    3,
                    n_stages,
                    "move B: full opt of coord-carried seeds",
                    n_seeds=n_seeds,
                    note="WBO shed + placed monomer; ranking deferred",
                )
            _opt_coord_seeds(
                result.coord_seeds,
                growth=growth,
                map_spec=map_spec,
                pack=pack,
                output_dir=out,
                progress=log if log else progress,
                child_minima=child_minima,
                bin_ranks=bin_ranks,
            )

        # ---- Move Z: persistent occupation -> graph decoration -> anchored
        # embed -> unconstrained g-xTB.  This is intentionally separate from
        # generic Move A because the latter owns no lattice-site metadata.
        if result.zb_seeds and embed:
            _opt_zb_occupations(
                result.zb_seeds,
                growth=growth,
                map_spec=map_spec,
                pack=pack,
                output_dir=out,
                progress=log if log else progress,
                child_minima=child_minima,
                bin_ranks=bin_ranks,
                zb_stats=result.zb_stats,
            )

        if log and window.move_zb_sites:
            log.line(
                "move Z cores -> pack decorate (2p / bridge-target) -> "
                "anchored embed.yaml motif_factor -> unconstrained g-xTB; "
                "propagate only converged, topology-preserving endpoints"
            )

        if log and result.skeleton_catalog and do_redecorate:
            log.set_block_plan(n_a_blocks)
            log.stage(
                3,
                n_stages,
                "move A: decorate cores + motif_factor + opt",
                total_cores=n_cores,
                note=(
                    "graph cores -> Cl redecorate -> motif_factor -> g-xTB; "
                    "dup graphs skip opt; ranking after all bins"
                ),
            )
        elif log and result.skeleton_catalog and not do_redecorate:
            log.line(
                "move A redecorate: OFF (child.redecorate=false or --cores-only)"
            )
        done_cores = 0
        cation = map_spec.core.cation
        anion = map_spec.core.anion
        dec_mode, dec_note = choose_decoration_mode(
            parents,
            k_child=k_from + 1,
            spec=map_spec,
            switch=window.lattice,
        )
        dec_spec = (
            spec_with_decoration_mode(map_spec, dec_mode)
            if dec_mode
            else map_spec
        )
        if window.selection_max_per_skeleton > 0:
            from dataclasses import replace as _replace

            dec_spec = _replace(
                dec_spec,
                graph_rules=_replace(
                    dec_spec.graph_rules,
                    selection_max_per_skeleton=int(
                        window.selection_max_per_skeleton
                    ),
                    selection_per_skeleton_from_k=0,
                ),
            )
        if log and do_redecorate:
            log.line(f"decoration mode: {dec_mode}  ({dec_note})")
        for (k, p), cores in sorted(
            result.skeleton_catalog.items() if do_redecorate else ()
        ):
            n_coord = len(result.coord_seeds.get((k, p), ()))
            if not window.allow_redecorate(k, p):
                cap = window.p_surf(k) if window.surface_beta > 0 else "?"
                if log:
                    log.line(
                        f"skip move A k={k} p={p}: slack bin "
                        f"(p > p_surf={cap}); B-only, rhombi still allowed"
                    )
                done_cores += len(cores)
                continue
            if log:
                log.begin_bin(
                    k=k,
                    p=p,
                    cores=len(cores),
                    cores_done=done_cores,
                    cores_total=n_cores,
                    jobs=len(cores),
                )
                log.line(
                    f"  bin: Z_zb_cores={len(cores)}  "
                    f"(decorate+embed via pack; rank after bin)"
                    if window.move_zb_sites
                    else (
                        f"  bin: A_graph_cores={len(cores)}  "
                        f"B_coord_seeds={n_coord} (B already opted; "
                        f"rank merged at end)"
                    )
                )
            elif progress:
                progress(
                    f"[growth] decorate k={k} p={p} cores={len(cores)}"
                )

            # ---- resume: skip finished move-A bins ----
            if (
                resume
                and out is not None
                and bin_A_complete_marker(out, k, p).is_file()
            ):
                if log:
                    log.line(
                        f"  RESUME: skip move A for k={k} p={p} "
                        f"({bin_A_complete_marker(out, k, p).name})"
                    )
                # reload A rows from index; B rows already in bin_ranks
                existing_a = _load_ranked_from_disk(
                    out, k, p, move_filter="A"
                )
                # if move column missing, load all non-B
                if not existing_a:
                    all_rows = _load_ranked_from_disk(out, k, p)
                    existing_a = [
                        r for r in all_rows if r.growth_move != "B"
                    ]
                bin_ranks[(k, p)].extend(existing_a)
                for r in existing_a:
                    prev = child_minima.get((k, p))
                    if prev is None or r.xtb_energy_eV < float(
                        prev["energy_eV"]
                    ):
                        child_minima[(k, p)] = {
                            "energy_eV": r.xtb_energy_eV,
                            "structure_id": r.structure_id,
                        }
                done_cores += len(cores)
                if log:
                    n_b = sum(
                        1 for r in bin_ranks[(k, p)] if r.growth_move == "B"
                    )
                    log.line(
                        f"  resumed rank pool k={k} p={p}: "
                        f"A={len(existing_a)} B={n_b}"
                    )
                continue

            bin_res = enumerate_molecular_bin(
                k,
                p,
                dec_spec,
                pack=pack,
                embed=embed,
                precomputed_skeletons=cores,
                progress=progress,
            )
            if window.move_zb_sites:
                n_before = len(bin_res.isomers)
                bin_res = _filter_bin_zb_embeddable(
                    bin_res, map_spec, zb_stats=result.zb_stats
                )
                if log:
                    log.line(
                        f"  zb filter k={k} p={p}: "
                        f"{len(bin_res.isomers)}/{n_before} still embed on zb"
                    )
            done_cores += len(cores)
            # track bin minimum for package profile
            with_e = [
                iso
                for iso in bin_res.isomers
                if iso.xtb_energy_eV is not None
            ]
            if with_e:
                best = min(with_e, key=lambda iso: float(iso.xtb_energy_eV))
                prev = child_minima.get((k, p))
                if prev is None or float(best.xtb_energy_eV) < float(
                    prev["energy_eV"]
                ):
                    child_minima[(k, p)] = {
                        "energy_eV": float(best.xtb_energy_eV),
                        "structure_id": best.structure_id,
                    }
            # accumulate move-A rows for final merged ranking
            bin_ranks[(k, p)].extend(
                _ranked_from_molecular_isomers(
                    bin_res.isomers, move="A", cation=cation, anion=anion
                )
            )
            if log:
                n_iso = len(bin_res.isomers)
                n_ok = len(with_e)
                n_merged = sum(
                    1
                    for rec in getattr(bin_res, "graph_merge_records", []) or []
                )
                n_fail = max(0, n_iso - n_ok)
                n_b = sum(
                    1 for r in bin_ranks[(k, p)] if r.growth_move == "B"
                )
                log.end_bin(
                    k=k,
                    p=p,
                    n_iso=n_iso,
                    n_ok=n_ok,
                    n_fail=n_fail,
                    n_merged=n_merged,
                    raw_graphs=getattr(bin_res, "raw_graphs", None),
                )
                log.line(
                    f"  deferred rank pool k={k} p={p}: "
                    f"A={n_ok} B={n_b} (print after all bins)"
                )
            if out is not None:
                _write_growth_bin(out, bin_res, growth, spec=map_spec)
                # mark bin A finished for restart
                marker = bin_A_complete_marker(out, k, p)
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(
                    f"k={k} p={p} isomers={len(bin_res.isomers)} "
                    f"with_E={len(with_e)}\n",
                    encoding="utf-8",
                )

        # ---- Consolidate relaxed basins before ranking / next-k selection ----
        consolidation_counts: Dict[Tuple[int, int], Tuple[int, int]] = {}
        if (
            growth.minimum_consolidation.enabled
            and out is not None
            and bin_ranks
        ):
            loaded_by_k: Dict[int, List[ParentStructure]] = {}
            for k, p in sorted(bin_ranks):
                if k not in loaded_by_k:
                    loaded_by_k[k] = load_parents_from_run(
                        out, k=k, spec=map_spec
                    )
                endpoints = [
                    parent for parent in loaded_by_k[k] if parent.p == p
                ]
                if not endpoints:
                    continue
                clusters = consolidate_relaxed_minima(
                    endpoints,
                    growth.minimum_consolidation,
                    map_spec,
                )
                write_minimum_clusters(
                    out,
                    k,
                    p,
                    clusters,
                    config=growth.minimum_consolidation,
                )
                raw_moves = {
                    row.structure_id: row.growth_move
                    for row in bin_ranks[(k, p)]
                }
                consolidated_rows: List[RankedIsomer] = []
                for cluster in clusters:
                    representative = cluster.representative
                    occupation = getattr(
                        representative, "zb_occupation", None
                    )
                    occupation_id = str(
                        getattr(occupation, "occupation_id", "") or ""
                    )
                    consolidated_rows.append(
                        RankedIsomer(
                            structure_id=representative.structure_id,
                            xtb_energy_eV=float(representative.energy_eV),
                            seed_skeleton=(
                                occupation_id[-6:]
                                if occupation_id
                                else cluster.minimum_id[-6:]
                            ),
                            growth_move=raw_moves.get(
                                representative.structure_id, "Z"
                            ),
                            parent_id="",
                            minimum_id=cluster.minimum_id,
                            minimum_multiplicity=len(cluster.members),
                        )
                    )
                consolidation_counts[(k, p)] = (
                    len(endpoints),
                    len(consolidated_rows),
                )
                bin_ranks[(k, p)] = consolidated_rows

        # ---- Final rankings for every child (k,p) ----
        if log and bin_ranks:
            log.stage(
                4,
                n_stages,
                "consolidated relaxed-minimum rankings per (k,p)",
                note=(
                    "same coloured graph + internal distances + invariant "
                    "core/full RMSD; raw endpoints retained"
                ),
            )
            for (k, p) in sorted(bin_ranks):
                rows = bin_ranks[(k, p)]
                if not rows:
                    continue
                n_a = sum(1 for r in rows if r.growth_move == "A")
                n_b = sum(1 for r in rows if r.growth_move == "B")
                n_z = sum(1 for r in rows if r.growth_move == "Z")
                raw_count, minimum_count = consolidation_counts.get(
                    (k, p), (len(rows), len(rows))
                )
                ranking = format_bin_ranking(
                    rows,
                    k=k,
                    p=p,
                    refs=growth.references,
                    package_p_m=tuple(growth.monomer_p_values) or (1, 2, 3),
                    delta_mu=tuple(growth.delta_mu_cdcl2_eV)
                    or (-1.0, 0.0, 1.0),
                    title_note=(
                        f"minima={minimum_count} from raw={raw_count}; "
                        f"A={n_a} B={n_b} Z={n_z}"
                    ),
                )
                log.block(ranking)
                # refresh child_minima from merged pool
                best = min(rows, key=lambda r: r.xtb_energy_eV)
                child_minima[(k, p)] = {
                    "energy_eV": float(best.xtb_energy_eV),
                    "structure_id": best.structure_id,
                }

        # mark whole k->k+1 step complete for multi-step / resubmit
        if out is not None and decorate:
            if not child_minima and not bin_ranks:
                if log:
                    log.line(
                        "no kept children with energy; "
                        "not writing step-complete marker "
                        f"(delete {step_complete_marker(out, k_from).name} "
                        "if a previous empty run wrote one)"
                    )
            else:
                step_complete_marker(out, k_from).write_text(
                    f"k_from={k_from} k_to={k_from + 1} complete\n",
                    encoding="utf-8",
                )
                if log:
                    log.line(
                        f"checkpoint: wrote {step_complete_marker(out, k_from).name}"
                    )

    # Package growth profile: k=1…k_child along p = k·p_m for each package
    if log:
        parent_minima = bin_minima_from_records(
            load_energy_index(run_dir, k_values=range(1, k_from + 1))
        )
        out_minima: Dict[Tuple[int, int], Dict[str, Any]] = {}
        if output_dir is not None:
            out_minima = bin_minima_from_records(
                load_energy_index(
                    Path(output_dir),
                    k_values=range(1, k_from + 2),
                    require_converged=False,
                )
            )
        minima = merge_bin_minima(
            seed_package_minima_from_refs(growth.references),
            parent_minima,
            out_minima,
            child_minima,
        )
        profile = format_package_growth_profile(
            minima,
            refs=growth.references,
            package_p_m=tuple(growth.monomer_p_values) or (1, 2, 3),
            k_values=tuple(range(1, k_from + 2)),
            delta_mu=tuple(growth.delta_mu_cdcl2_eV) or (-1.0, 0.0, 1.0),
        )
        print(profile, flush=True)

    if log:
        log.stage(
            n_stages,
            n_stages,
            "write outputs",
            output=str(output_dir) if output_dir else "(none)",
            parents_selected=result.parents_selected,
            child_cores=n_cores,
        )
    return result


# ---------------------------------------------------------------------------
# Restart / checkpoint helpers
# ---------------------------------------------------------------------------


def step_complete_marker(output_dir: Path, k_from: int) -> Path:
    return Path(output_dir) / f".step_k{int(k_from):03d}_complete"


def bin_A_complete_marker(output_dir: Path, k: int, p: int) -> Path:
    return Path(output_dir) / f"k{int(k):03d}" / f"p{int(p):03d}" / ".bin_A_complete"


def _parse_energy_from_xyz_comment(path: Path) -> Optional[float]:
    """Read energy_eV=... from the second line of an XYZ if present."""

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    if len(lines) < 2:
        return None
    m = re.search(r"energy_eV\s*=\s*([-+0-9.eE]+)", lines[1])
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _index_rows_by_id(output_dir: Path) -> Dict[str, Dict[str, str]]:
    """Map structure_id -> last index.csv row."""

    path = Path(output_dir) / "index.csv"
    out: Dict[str, Dict[str, str]] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            sid = (row.get("structure_id") or "").strip()
            if sid:
                out[sid] = row
    return out


def _index_has_energy(output_dir: Path, structure_id: str) -> Optional[float]:
    rows = _index_rows_by_id(output_dir)
    row = rows.get(structure_id)
    if not row:
        return None
    try:
        e = float(row.get("xtb_energy_eV") or "")
    except ValueError:
        return None
    return e if math.isfinite(e) else None


def _append_index_row_unique(
    output_dir: Path,
    row: Dict[str, Any],
    fieldnames: Sequence[str],
) -> bool:
    """Append one index row if structure_id not already present. Return True if written."""

    sid = str(row.get("structure_id") or "")
    if not sid:
        return False
    if _index_has_energy(output_dir, sid) is not None:
        return False
    index_path = Path(output_dir) / "index.csv"
    write_header = not index_path.is_file()
    with index_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    return True


def _load_ranked_from_disk(
    output_dir: Path,
    k: int,
    p: int,
    *,
    move_filter: Optional[str] = None,
) -> List[RankedIsomer]:
    """Rebuild ranking rows for (k,p) from index.csv (+ xyz energy fallback)."""

    rows: List[RankedIsomer] = []
    index_path = Path(output_dir) / "index.csv"
    if not index_path.is_file():
        return rows
    with index_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            try:
                rk, rp = int(row["k"]), int(row["p"])
            except (KeyError, ValueError):
                continue
            if rk != k or rp != p:
                continue
            move = (row.get("move") or "?").strip()
            if move_filter is not None and move != move_filter:
                # move column: "coord" for B, often empty/1 for A
                if move_filter == "B" and move not in ("coord", "B", "b"):
                    continue
                if move_filter == "A" and move in ("coord", "B", "b"):
                    continue
            sid = (row.get("structure_id") or "").strip()
            try:
                e = float(row.get("xtb_energy_eV") or "")
            except ValueError:
                continue
            if not sid or not math.isfinite(e):
                continue
            gmove = "B" if move in ("coord", "B", "b") else "A"
            rows.append(
                RankedIsomer(
                    structure_id=sid,
                    xtb_energy_eV=e,
                    seed_skeleton="------",
                    growth_move=gmove,
                    parent_id=str(row.get("parent_id") or ""),
                )
            )
    return rows


def _core_skeleton_fp(
    core_edges: EdgeList,
    *,
    symbols: Optional[Sequence[str]] = None,
    cation: str = "Cd",
    anion: str = "Se",
) -> str:
    """Short fingerprint of a Cd-Se core edge list (for ranking colours)."""

    if not core_edges:
        return "------"
    g = nx.Graph()
    nodes = {n for e in core_edges for n in e}
    for n in nodes:
        el = anion if symbols is not None and n < len(symbols) and symbols[n] == anion else cation
        # Without symbols, parity of index is unreliable; mark both as 'X'
        if symbols is None:
            el = "X"
        g.add_node(n, el=el)
    g.add_edges_from(core_edges)
    try:
        return nx.weisfeiler_lehman_graph_hash(
            g, node_attr="el", iterations=4
        )[:6]
    except Exception:
        return "------"


def _ranked_from_molecular_isomers(
    isomers: Sequence[Any],
    *,
    move: str = "A",
    cation: str = "Cd",
    anion: str = "Se",
) -> List[RankedIsomer]:
    """Collect energy rows from MolecularIsomer-like objects."""

    from .molecular import skeleton_fingerprint

    out: List[RankedIsomer] = []
    for iso in isomers:
        e = getattr(iso, "xtb_energy_eV", None)
        if e is None:
            continue
        skel = getattr(iso, "seed_skeleton", None) or ""
        if not skel and getattr(iso, "atoms", None) is not None:
            try:
                skel = skeleton_fingerprint(
                    iso.atoms, iso.graph, cation, anion
                )
            except Exception:
                skel = "------"
        out.append(
            RankedIsomer(
                structure_id=str(iso.structure_id),
                xtb_energy_eV=float(e),
                seed_skeleton=str(skel or "------")[:6],
                growth_move=move,
            )
        )
    return out


def _opt_coord_seeds(
    coord_seeds: Mapping[Tuple[int, int], Sequence[CoordSeed]],
    *,
    growth: GrowthConfig,
    map_spec: NucleationSpec,
    pack: Optional[GeometryPack],
    output_dir: Optional[Path],
    progress: Optional[Any],
    child_minima: Dict[Tuple[int, int], Dict[str, Any]],
    bin_ranks: Dict[Tuple[int, int], List[RankedIsomer]],
) -> None:
    """Full g-xTB opt for move-B coordinate seeds; write XYZ + index rows."""

    from .xtb_relax import XtbSettings, relax_structures

    if not coord_seeds:
        return
    settings = XtbSettings.from_pack(full_opt_relaxation_raw(pack, growth))
    cutoffs = bond_cutoffs_from_spec(map_spec)
    log = progress if isinstance(progress, GrowthLog) else None
    cation = map_spec.core.cation
    anion = map_spec.core.anion

    groups: List[Tuple[Tuple[int, int], List[CoordSeed]]] = [
        (key, list(coord_seeds[key]))
        for key in sorted(coord_seeds)
        if coord_seeds[key]
    ]
    for _, seeds in groups:
        assign_compact_b_ids(seeds, output_dir)
    n_flat = sum(len(seeds) for _, seeds in groups)
    if log:
        log.line(f"move B: full g-xTB opt of {n_flat} coord-carried children")
        log.line(
            "p_child = p_parent - s + p_m;  "
            "row shows parent formula -s +p_m -> child formula"
        )
        log.line(
            "row: global  i/N_block  [CdSe]_kpar(CdCl2)_ppar  -s=.. +p_m=.. -> "
            "[CdSe]_k(CdCl2)_p  E  opt clean +dt  status  parent=.."
        )
        log.line(
            "NOTE: per-(k,p) ranking is deferred until move A finishes "
            "(merge A redecorate + B coord into each bin)"
        )
    elif progress:
        progress(f"[growth] move B: opt {n_flat} coord seeds")

    n_skip = 0
    n_run = 0
    for kp, seeds in groups:
        if log:
            idx = log.begin_block(
                len(seeds), label=f"B k={kp[0]} p={kp[1]}"
            )
            blk = log._block_header_bit() or f"block {idx}"
            log.line(f"--- move B k={kp[0]} p={kp[1]} ---  {blk}")
            log.line(
                f"  opts in this block: {len(seeds)}   "
                f"(job lines show global_n  i/{len(seeds)})"
            )
        for seed in seeds:
            pk, pp = GrowthLog._parent_kp_from_id(seed.parent_id)
            k_par = "" if pk is None else str(pk)
            p_par = "" if pp is None else str(pp)
            recon_s = float(seed.cleanup_s)
            bdir = (
                None
                if output_dir is None
                else Path(output_dir) / f"k{seed.k:03d}" / f"p{seed.p:03d}"
            )
            xyz_path = (
                None
                if bdir is None
                else bdir / f"{seed.structure_id}_xtb.xyz"
            )

            # ---- restart: skip if energy already on disk ----
            e_existing: Optional[float] = None
            if output_dir is not None:
                e_existing = _index_has_energy(output_dir, seed.structure_id)
                if e_existing is None and xyz_path is not None and xyz_path.is_file():
                    e_existing = _parse_energy_from_xyz_comment(xyz_path)
            if e_existing is not None:
                n_skip += 1
                e = float(e_existing)
                if progress is not None:
                    progress(
                        f"[growth-job] k={seed.k} p={seed.p} move=B "
                        f"s={seed.shed} p_m={seed.p_m} "
                        f"k_parent={k_par} p_parent={p_par} "
                        f"parent={seed.parent_id} "
                        f"id={seed.structure_id} "
                        f"t_s=0.0 recon_s={recon_s:.1f} "
                        f"E_eV={e:.6f} relax=ok err=resume_skip"
                    )
                prev = child_minima.get((seed.k, seed.p))
                if prev is None or e < float(prev["energy_eV"]):
                    child_minima[(seed.k, seed.p)] = {
                        "energy_eV": e,
                        "structure_id": seed.structure_id,
                    }
                skel = _core_skeleton_fp(
                    seed.core_edges,
                    symbols=seed.symbols,
                    cation=cation,
                    anion=anion,
                )
                bin_ranks.setdefault((seed.k, seed.p), []).append(
                    RankedIsomer(
                        structure_id=seed.structure_id,
                        xtb_energy_eV=e,
                        seed_skeleton=str(skel or "------")[:6],
                        growth_move="B",
                        parent_id=seed.parent_id,
                    )
                )
                continue

            t0 = time.perf_counter()
            batch = [
                {
                    "id": seed.structure_id,
                    "symbols": list(seed.symbols),
                    "positions": np.asarray(seed.coordinates, dtype=float).tolist(),
                    "edges": list(seed.core_edges),
                }
            ]
            results = relax_structures(batch, settings, cutoffs)
            xr = results[0]
            opt_s = time.perf_counter() - t0
            n_run += 1
            # Prune g-xTB artifact endpoints (Se–Cl / Se–Se / Cd–Cd contacts).
            artifact_codes: List[str] = []
            if xr.ok and xr.coordinates is not None:
                from .molecular_rules import forbidden_pair_contact_violations

                artifact_codes = forbidden_pair_contact_violations(
                    list(seed.symbols),
                    xr.coordinates,
                    map_spec,
                    floors=settings.artifact_min_distance or None,
                )
            relax_tag = (
                "artifact"
                if artifact_codes
                else (
                    "ok"
                    if xr.converged
                    else (
                        "maxcycle"
                        if str(getattr(xr, "status", "") or "") == "maxcycle"
                        or (
                            not xr.converged
                            and int(xr.steps) >= int(settings.max_steps) > 0
                        )
                        else "unconv"
                    )
                )
            )
            if progress is not None:
                base = (
                    f"[growth-job] k={seed.k} p={seed.p} move=B "
                    f"s={seed.shed} p_m={seed.p_m} "
                    f"k_parent={k_par} p_parent={p_par} "
                    f"parent={seed.parent_id} "
                    f"id={seed.structure_id} "
                    f"t_s={opt_s:.1f} recon_s={recon_s:.1f} "
                    f"steps={int(getattr(xr, 'steps', 0))} "
                    f"max_steps={int(settings.max_steps)} "
                )
                if artifact_codes:
                    progress(
                        base
                        + "E_eV=n/a "
                        + f"relax=artifact err={'|'.join(artifact_codes[:3])}"
                    )
                elif xr.ok and xr.energy_eV is not None:
                    progress(
                        base
                        + f"E_eV={float(xr.energy_eV):.6f} "
                        + f"relax={relax_tag}"
                    )
                else:
                    progress(
                        base
                        + "E_eV=n/a "
                        + f"relax=fail err={xr.error or 'coord_opt'}"
                    )
            if artifact_codes:
                # Still dump diagnostic XYZ (energy tagged as artifact) but do
                # not enter ranking / child_minima.
                if output_dir is not None and xr.coordinates is not None:
                    bdir = Path(output_dir) / f"k{seed.k:03d}" / f"p{seed.p:03d}"
                    bdir.mkdir(parents=True, exist_ok=True)
                    xyz = bdir / f"{seed.structure_id}_xtb_artifact.xyz"
                    e_note = (
                        "n/a"
                        if xr.energy_eV is None
                        else f"{float(xr.energy_eV):.6f}"
                    )
                    lines = [
                        str(len(seed.symbols)),
                        f"{seed.structure_id} ARTIFACT energy_eV={e_note} "
                        f"violations={'|'.join(artifact_codes)} "
                        f"parent={seed.parent_id}",
                    ]
                    for sym, pos in zip(seed.symbols, xr.coordinates):
                        lines.append(
                            f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"
                        )
                    xyz.write_text("\n".join(lines) + "\n")
                continue
            if not xr.ok or xr.energy_eV is None:
                continue
            e = float(xr.energy_eV)
            prev = child_minima.get((seed.k, seed.p))
            if prev is None or e < float(prev["energy_eV"]):
                child_minima[(seed.k, seed.p)] = {
                    "energy_eV": e,
                    "structure_id": seed.structure_id,
                }
            # Prefer fingerprint of the *relaxed* Cd-Se core when available
            skel = _core_skeleton_fp(
                seed.core_edges, symbols=seed.symbols, cation=cation, anion=anion
            )
            if xr.coordinates is not None and cutoffs:
                try:
                    from .molecular import skeleton_fingerprint
                    from .xtb_relax import relaxed_edges as _re

                    full_e = _re(list(seed.symbols), xr.coordinates, cutoffs)
                    g = nx.Graph()
                    for i, sym in enumerate(seed.symbols):
                        if sym in (cation, anion):
                            g.add_node(i)
                    for a, b in full_e:
                        if a in g and b in g:
                            g.add_edge(a, b)
                    atoms = [
                        type("Atom", (), {"symbol": seed.symbols[i]})()
                        for i in range(len(seed.symbols))
                    ]
                    skel = skeleton_fingerprint(
                        atoms, g, cation, anion
                    )
                except Exception:
                    pass
            bin_ranks.setdefault((seed.k, seed.p), []).append(
                RankedIsomer(
                    structure_id=seed.structure_id,
                    xtb_energy_eV=e,
                    seed_skeleton=str(skel or "------")[:6],
                    growth_move="B",
                    parent_id=seed.parent_id,
                )
            )
            if output_dir is None:
                continue
            bdir = Path(output_dir) / f"k{seed.k:03d}" / f"p{seed.p:03d}"
            bdir.mkdir(parents=True, exist_ok=True)
            coords = xr.coordinates or tuple(
                map(tuple, np.asarray(seed.coordinates))
            )
            xyz = bdir / f"{seed.structure_id}_xtb.xyz"
            lines = [
                str(len(seed.symbols)),
                f"{seed.structure_id} energy_eV={e} move=coord "
                f"shed={seed.shed} p_m={seed.p_m} parent={seed.parent_id} "
                f"{seed.notes}",
            ]
            for sym, pos in zip(seed.symbols, coords):
                lines.append(f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
            xyz.write_text("\n".join(lines) + "\n")
            if growth.persist_wbo and getattr(xr, "bond_orders", None):
                write_wbo_file(xyz.with_suffix(".wbo"), xr.bond_orders)
            refs = growth.references
            fields = _growth_index_fields(growth)
            row = {
                "k": seed.k,
                "p": seed.p,
                "structure_id": seed.structure_id,
                "xtb_energy_eV": f"{e:.8f}",
                "xtb_converged": bool(xr.converged),
                "dE_f_eV": "",
                "growth": "1",
                "move": "coord",
                "shed": seed.shed,
                "p_m": seed.p_m,
                "parent_id": seed.parent_id,
            }
            apply_soft_columns(
                row,
                symbols=list(seed.symbols),
                coords=np.asarray(coords, dtype=float),
                energy_eV=e,
                k=seed.k,
                rules=growth.window_for(seed.k).soft_rules,
                spec=map_spec,
            )
            if refs is not None:
                de_f = refs.formation_eV(e, seed.k, seed.p)
                row["dE_f_eV"] = f"{de_f:.8f}"
                for dmu in growth.delta_mu_cdcl2_eV:
                    key = f"Omega_dmu_{dmu:+.2f}"
                    row[key] = f"{refs.grand_potential_eV(e, seed.k, seed.p, dmu):.8f}"
            _append_index_row_unique(Path(output_dir), row, fields)

    if log:
        log.line(
            f"move B opts: ran={n_run}  resumed/skipped={n_skip}  "
            f"total_seeds={n_flat}"
        )


def _opt_zb_seeds(
    zb_seeds: Mapping[Tuple[int, int], Sequence[Any]],
    *,
    growth: GrowthConfig,
    map_spec: NucleationSpec,
    pack: Optional[GeometryPack],
    output_dir: Optional[Path],
    progress: Optional[Any],
    child_minima: Dict[Tuple[int, int], Dict[str, Any]],
    bin_ranks: Dict[Tuple[int, int], List[RankedIsomer]],
    zb_stats: Any = None,
) -> None:
    """2p-decorate zb cores, full opt, keep only zb-embeddable endpoints."""

    from .molecular_zb_growth import (
        construction_clash,
        lattice_model,
        place_cl_2p,
        zb_embeddable,
    )
    from .xtb_relax import XtbSettings, relax_structures

    if not zb_seeds:
        return
    settings = XtbSettings.from_pack(full_opt_relaxation_raw(pack, growth))
    cutoffs = bond_cutoffs_from_spec(map_spec)
    log = progress if isinstance(progress, GrowthLog) else None
    try:
        model = lattice_model(map_spec)
    except Exception as exc:
        if log:
            log.line(f"move Z: cannot load zb CIF ({exc}); skip opt")
        return

    groups = [
        (key, list(zb_seeds[key]))
        for key in sorted(zb_seeds)
        if zb_seeds[key]
    ]
    n_flat = sum(len(v) for _, v in groups)
    if log:
        log.line(f"move Z: decorate+opt {n_flat} zb occupations")
        log.line(
            "row: keep only endpoints that snap back onto zb "
            "(n4 or snap fail are dropped, no XYZ)"
        )

    serial = 0
    for (k, p), occs in groups:
        if log:
            log.begin_block(len(occs), label=f"Z k={k} p={p}")
            log.line(f"--- move Z k={k} p={p} ---  {len(occs)} occupations")
        for occ in occs:
            serial += 1
            sid = compact_growth_id(occ.k, occ.p, "Z", serial)
            try:
                placed = place_cl_2p(occ, map_spec, pack, model=model)
            except Exception:
                placed = None
            if placed is None:
                if zb_stats is not None:
                    zb_stats.clash_skip += 1
                if progress is not None:
                    progress(
                        f"[growth-job] k={occ.k} p={occ.p} move=Z "
                        f"id={sid} E_eV=n/a relax=clash "
                        f"parent={occ.parent_id}"
                    )
                continue
            symbols, coords, cl_edges = placed
            if construction_clash(
                symbols, coords, map_spec, bonded=cl_edges
            ):
                if zb_stats is not None:
                    zb_stats.clash_skip += 1
                if progress is not None:
                    progress(
                        f"[growth-job] k={occ.k} p={occ.p} move=Z "
                        f"id={sid} E_eV=n/a relax=clash "
                        f"parent={occ.parent_id}"
                    )
                continue
            t0 = time.perf_counter()
            batch = [
                {
                    "id": sid,
                    "symbols": list(symbols),
                    "positions": np.asarray(coords, dtype=float).tolist(),
                    "edges": list(occ.core_edges),
                }
            ]
            results = relax_structures(batch, settings, cutoffs)
            xr = results[0]
            opt_s = time.perf_counter() - t0
            if not xr.ok or xr.energy_eV is None or xr.coordinates is None:
                if zb_stats is not None:
                    zb_stats.opt_fail += 1
                if progress is not None:
                    progress(
                        f"[growth-job] k={occ.k} p={occ.p} move=Z "
                        f"id={sid} t_s={opt_s:.1f} E_eV=n/a relax=fail "
                        f"err={getattr(xr, 'error', None) or 'zb_opt'} "
                        f"parent={occ.parent_id}"
                    )
                continue
            ok, _emb, why = zb_embeddable(
                list(symbols),
                np.asarray(xr.coordinates, dtype=float),
                map_spec,
                model,
                parent_id=occ.parent_id,
            )
            if not ok:
                if zb_stats is not None:
                    zb_stats.opt_reject_embed += 1
                if progress is not None:
                    progress(
                        f"[growth-job] k={occ.k} p={occ.p} move=Z "
                        f"id={sid} t_s={opt_s:.1f} "
                        f"E_eV={float(xr.energy_eV):.6f} "
                        f"relax=not_zb err={why} parent={occ.parent_id}"
                    )
                continue
            if zb_stats is not None:
                zb_stats.opt_keep += 1
            e = float(xr.energy_eV)
            if progress is not None:
                progress(
                    f"[growth-job] k={occ.k} p={occ.p} move=Z "
                    f"s={occ.shed} p_m={occ.p_m} "
                    f"parent={occ.parent_id} id={sid} "
                    f"t_s={opt_s:.1f} E_eV={e:.6f} relax=ok"
                )
            prev = child_minima.get((occ.k, occ.p))
            if prev is None or e < float(prev["energy_eV"]):
                child_minima[(occ.k, occ.p)] = {
                    "energy_eV": e,
                    "structure_id": sid,
                }
            bin_ranks.setdefault((occ.k, occ.p), []).append(
                RankedIsomer(
                    structure_id=sid,
                    xtb_energy_eV=e,
                    seed_skeleton="zb----",
                    growth_move="Z",
                    parent_id=occ.parent_id,
                )
            )
            if output_dir is None:
                continue
            bdir = Path(output_dir) / f"k{occ.k:03d}" / f"p{occ.p:03d}"
            bdir.mkdir(parents=True, exist_ok=True)
            xyz = bdir / f"{sid}_xtb.xyz"
            lines = [
                str(len(symbols)),
                f"{sid} energy_eV={e} move=Z shed={occ.shed} "
                f"p_m={occ.p_m} parent={occ.parent_id}",
            ]
            for sym, pos in zip(symbols, xr.coordinates):
                lines.append(f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
            xyz.write_text("\n".join(lines) + "\n")
            refs = growth.references
            fields = _growth_index_fields(growth)
            row = {
                "k": occ.k,
                "p": occ.p,
                "structure_id": sid,
                "xtb_energy_eV": f"{e:.8f}",
                "xtb_converged": bool(xr.converged),
                "dE_f_eV": "",
                "growth": "1",
                "move": "Z",
                "shed": occ.shed,
                "p_m": occ.p_m,
                "parent_id": occ.parent_id,
            }
            apply_soft_columns(
                row,
                symbols=list(symbols),
                coords=np.asarray(xr.coordinates, dtype=float),
                energy_eV=e,
                k=occ.k,
                rules=growth.window_for(occ.k).soft_rules,
                spec=map_spec,
            )
            if refs is not None:
                de_f = refs.formation_eV(e, occ.k, occ.p)
                row["dE_f_eV"] = f"{de_f:.8f}"
                for dmu in growth.delta_mu_cdcl2_eV:
                    key = f"Omega_dmu_{dmu:+.2f}"
                    row[key] = (
                        f"{refs.grand_potential_eV(e, occ.k, occ.p, dmu):.8f}"
                    )
            _append_index_row_unique(Path(output_dir), row, fields)

    if log and zb_stats is not None:
        log.line(zb_stats.as_log())


def _filter_bin_zb_embeddable(
    bin_res: Any,
    spec: NucleationSpec,
    *,
    zb_stats: Any = None,
) -> Any:
    """Drop decorated/opted isomers whose Cd–Se core is not a zb fragment."""

    from .molecular_zb_growth import lattice_model, zb_embeddable

    try:
        model = lattice_model(spec)
    except Exception:
        return bin_res
    kept = []
    for iso in bin_res.isomers:
        coords = iso.xtb_coordinates or iso.coordinates
        if coords is None or iso.xtb_energy_eV is None:
            if zb_stats is not None:
                zb_stats.opt_fail += 1
            continue
        symbols = [a.symbol for a in iso.atoms]
        ok, _occ, _why = zb_embeddable(
            symbols,
            np.asarray(coords, dtype=float),
            spec,
            model,
            parent_id=getattr(iso, "structure_id", ""),
        )
        if not ok:
            if zb_stats is not None:
                zb_stats.opt_reject_embed += 1
            continue
        if zb_stats is not None:
            zb_stats.opt_keep += 1
        kept.append(iso)
    bin_res.isomers = kept
    return bin_res


def _aligned_core_rmsd(
    reference: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
) -> float:
    left = np.asarray(reference, dtype=float)
    right = np.asarray(candidate, dtype=float)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        return float("nan")
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    u, _singular, vt = np.linalg.svd(right.T @ left)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    delta = right @ rotation - left
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _select_zb_decorations(isomers: Sequence[Any], limit: int) -> List[Any]:
    """Select graph-distinct ligand shells without an energy proxy.

    One representative of every Cl-degree/Cd-coordination class is taken
    before a second member of a class.  This keeps terminal/bridge alternatives
    in the small pre-g-xTB budget instead of relying on generator order.
    """

    if limit <= 0 or len(isomers) <= limit:
        return list(isomers)
    buckets: Dict[Tuple[Any, ...], List[Any]] = defaultdict(list)
    for isomer in isomers:
        cl_degrees = Counter(
            int(isomer.graph.degree[atom.atom_id])
            for atom in isomer.atoms
            if atom.symbol == "Cl"
        )
        cd_degrees = tuple(
            sorted(
                int(isomer.graph.degree[atom.atom_id])
                for atom in isomer.atoms
                if atom.symbol == "Cd"
            )
        )
        signature = (
            tuple(sorted(cl_degrees.items())),
            cd_degrees,
            int(isomer.graph.number_of_edges()),
        )
        buckets[signature].append(isomer)
    ordered_keys = sorted(buckets, key=repr)
    selected: List[Any] = []
    depth = 0
    while len(selected) < limit:
        progressed = False
        for key in ordered_keys:
            if depth < len(buckets[key]):
                selected.append(buckets[key][depth])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        depth += 1
    return selected


def _append_zb_manifest(output_dir: Path, record: Mapping[str, Any]) -> None:
    path = Path(output_dir) / "zb_occupations.jsonl"
    structure_id = str(record.get("structure_id") or "")
    replacement = json.dumps(dict(record), sort_keys=True)
    if structure_id and path.is_file():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            try:
                if str(json.loads(line).get("structure_id") or "") == structure_id:
                    lines[index] = replacement
                    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    return
            except (json.JSONDecodeError, AttributeError):
                continue
    with path.open("a", encoding="utf-8") as handle:
        handle.write(replacement + "\n")


def _append_jsonl_unique(
    path: Path,
    record: Mapping[str, Any],
    *,
    key: str,
) -> None:
    value = str(record.get(key) or "")
    if value and path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                if str(json.loads(line).get(key) or "") == value:
                    return
            except (json.JSONDecodeError, AttributeError):
                continue
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


def _opt_zb_occupations(
    zb_seeds: Mapping[Tuple[int, int], Sequence[Any]],
    *,
    growth: GrowthConfig,
    map_spec: NucleationSpec,
    pack: Optional[GeometryPack],
    output_dir: Optional[Path],
    progress: Optional[Any],
    child_minima: Dict[Tuple[int, int], Dict[str, Any]],
    bin_ranks: Dict[Tuple[int, int], List[RankedIsomer]],
    zb_stats: Any = None,
) -> None:
    """Decorate stored ZB occupations, anchor-embed, relax, and audit.

    The occupation is never reconstructed from the relaxed XYZ.  Graph rules
    generate ligand shells, the geometry pack places them around the exact
    lattice core, and only a converged topology-preserving endpoint is allowed
    to seed the next ZB growth step.
    """

    if not zb_seeds or pack is None:
        return

    from dataclasses import replace as dc_replace

    from .molecular import (
        _graph_certificate,
        enumerate_molecular_bin,
        molecular_decoration_rule_violations,
        molecular_graph_violations,
    )
    from .molecular_motif_reconstruct import (
        motif_vocabulary_violations,
        reconstruct_motif_state,
    )
    from .molecular_rules import forbidden_pair_contact_violations
    from .molecular_zb_growth import (
        endpoint_similarity_diagnostic,
        lattice_model,
        load_reference_occupation,
        occupation_to_record,
    )
    from .types import _State
    from .xtb_relax import XtbSettings, relax_structures

    settings = XtbSettings.from_pack(full_opt_relaxation_raw(pack, growth))
    cutoffs = bond_cutoffs_from_spec(map_spec)
    recon = (pack.raw or {}).get("reconstruction") or {}
    factor_starts = int(recon.get("factor_starts_per_graph", 3))
    xtb_starts = int(recon.get("xtb_starts_per_graph", 2))
    max_nfev = int(recon.get("max_nfev", 40))
    decoration_limit = int(
        getattr(map_spec.graph_rules, "selection_max_per_skeleton", 3) or 3
    )
    log = progress if isinstance(progress, GrowthLog) else None
    endpoint_reference = None
    if growth.endpoint_diagnostic_k > 0 and growth.endpoint_reference is not None:
        endpoint_reference = load_reference_occupation(
            growth.endpoint_reference,
            map_spec,
            lattice_model(map_spec),
        )
    for (k, p), occupations in sorted(zb_seeds.items()):
        if log:
            log.begin_block(len(occupations), label=f"Z k={k} p={p}")
            log.line(
                f"--- move Z k={k} p={p}: occupations={len(occupations)} "
                f"decorations<={decoration_limit} embed={factor_starts}->{xtb_starts} ---"
            )
        for occupation in occupations:
            if (
                endpoint_reference is not None
                and k == growth.endpoint_diagnostic_k
            ):
                diagnostic = endpoint_similarity_diagnostic(
                    occupation,
                    endpoint_reference,
                    match_tolerance_A=growth.endpoint_match_tolerance_A,
                )
                if output_dir is not None:
                    _append_jsonl_unique(
                        Path(output_dir) / f"k{k:03d}_endpoint_diagnostics.jsonl",
                        diagnostic,
                        key="occupation_id",
                    )
                if log:
                    log.line(
                        f"endpoint diagnostic {occupation.occupation_id}: "
                        f"Wulff-site overlap="
                        f"{diagnostic['site_overlap_fraction']:.3f} "
                        f"assignment_rmsd_A={diagnostic['assignment_rmsd_A']} "
                        "(report only)"
                    )
            graph_bin = enumerate_molecular_bin(
                k,
                p,
                map_spec,
                pack=pack,
                embed=False,
                precomputed_skeletons=[occupation.core_edges],
                progress=None,
            )
            decorations = _select_zb_decorations(
                graph_bin.isomers, decoration_limit
            )
            if not decorations:
                if zb_stats is not None:
                    zb_stats.clash_skip += 1
                continue

            for decoration_index, isomer in enumerate(decorations, start=1):
                n_core = len(occupation.symbols)
                if tuple(isomer.symbols[:n_core]) != tuple(occupation.symbols):
                    if zb_stats is not None:
                        zb_stats.opt_fail += 1
                    continue
                state = _State(atoms=isomer.atoms, graph=isomer.graph.copy())
                anchored = {
                    index: np.asarray(occupation.coordinates[index], dtype=float)
                    for index in range(n_core)
                }
                rebuilt = reconstruct_motif_state(
                    state,
                    pack,
                    map_spec,
                    starts=max(1, factor_starts),
                    keep=max(1, xtb_starts),
                    max_nfev=max_nfev,
                    overlap_min_A=float(recon.get("overlap_min_A", 0.75)),
                    start_max_bond_error_A=float(
                        recon.get("start_max_bond_error_A", 0.50)
                    ),
                    core_coordinates=anchored,
                )
                candidates = [
                    candidate
                    for candidate in rebuilt.candidates
                    if not candidate.audit_violations
                ][: max(1, xtb_starts)]
                if not candidates:
                    if zb_stats is not None:
                        zb_stats.clash_skip += 1
                    continue

                for candidate in candidates:
                    sid = (
                        f"k{k:03d}_p{p:03d}_Z"
                        f"{occupation.occupation_id[-8:]}"
                        f"d{decoration_index:02d}s{int(candidate.start_index):02d}"
                    )
                    source_edges = tuple(
                        sorted(
                            (min(int(a), int(b)), max(int(a), int(b)))
                            for a, b in state.graph.edges
                        )
                    )
                    payload = {
                        "id": sid,
                        "symbols": list(isomer.symbols),
                        "positions": [list(point) for point in candidate.coordinates],
                        "edges": list(source_edges),
                    }
                    t0 = time.perf_counter()
                    xr = relax_structures([payload], settings, cutoffs)[0]
                    # A max-cycle endpoint is evidence for a second start, not a
                    # rankable minimum.  Restart once from the endpoint.
                    if (
                        xr.ok
                        and not xr.converged
                        and xr.coordinates is not None
                    ):
                        retry_payload = dict(payload)
                        retry_payload["positions"] = [
                            list(point) for point in xr.coordinates
                        ]
                        retry = relax_structures(
                            [retry_payload], dc_replace(settings, accept_maxcycle=False), cutoffs
                        )[0]
                        if retry.ok and retry.coordinates is not None:
                            xr = retry
                    elapsed = time.perf_counter() - t0

                    violations: List[str] = []
                    topology_status = "unavailable"
                    core_rmsd = float("nan")
                    final_edges: Tuple[Tuple[int, int], ...] = ()
                    propagation_eligible = False
                    if xr.ok and xr.coordinates is not None:
                        final_edges = tuple(
                            sorted(
                                (min(int(a), int(b)), max(int(a), int(b)))
                                for a, b in xr.relaxed_edges
                            )
                        )
                        final_graph = nx.Graph()
                        final_graph.add_nodes_from(range(len(isomer.atoms)))
                        final_graph.add_edges_from(final_edges)
                        final_state = _State(
                            atoms=isomer.atoms, graph=final_graph
                        )
                        violations.extend(
                            molecular_graph_violations(final_state, map_spec)
                        )
                        violations.extend(
                            molecular_decoration_rule_violations(
                                final_state, map_spec
                            )
                        )
                        violations.extend(
                            motif_vocabulary_violations(
                                final_state,
                                cation=map_spec.core.cation,
                                anion=map_spec.core.anion,
                                ligand=map_spec.precursor.ligand,
                                motif_definitions=pack.raw.get("motifs"),
                            )
                        )
                        violations.extend(
                            forbidden_pair_contact_violations(
                                list(isomer.symbols),
                                xr.coordinates,
                                map_spec,
                                floors=settings.artifact_min_distance or None,
                            )
                        )
                        final_core = tuple(
                            edge
                            for edge in final_edges
                            if {
                                isomer.symbols[edge[0]],
                                isomer.symbols[edge[1]],
                            }
                            == {map_spec.core.cation, map_spec.core.anion}
                        )
                        expected_core = tuple(sorted(occupation.core_edges))
                        topology_status = (
                            "preserved"
                            if final_core == expected_core
                            else "changed"
                        )
                        core_rmsd = _aligned_core_rmsd(
                            occupation.coordinates,
                            np.asarray(xr.coordinates, dtype=float)[:n_core],
                        )
                        propagation_eligible = bool(
                            xr.converged
                            and xr.energy_eV is not None
                            and topology_status == "preserved"
                            and not violations
                        )
                    else:
                        violations.append(str(xr.error or "gxtb_failed"))

                    status = (
                        "propagate"
                        if propagation_eligible
                        else (
                            "off_path"
                            if topology_status == "changed" and xr.energy_eV is not None
                            else "failed"
                        )
                    )
                    if propagation_eligible and zb_stats is not None:
                        zb_stats.opt_keep += 1
                    elif topology_status == "changed" and zb_stats is not None:
                        zb_stats.opt_reject_embed += 1
                    elif zb_stats is not None:
                        zb_stats.opt_fail += 1

                    energy = None if xr.energy_eV is None else float(xr.energy_eV)
                    if progress is not None:
                        progress(
                            f"[growth-job] k={k} p={p} move=Z id={sid} "
                            f"occ={occupation.occupation_id} dec={decoration_index} "
                            f"start={candidate.start_index} "
                            f"E_eV={'n/a' if energy is None else f'{energy:.6f}'} "
                            f"t_s={elapsed:.1f} topology={topology_status} "
                            f"status={status} parent={occupation.parent_id}"
                        )

                    if output_dir is not None:
                        bdir = Path(output_dir) / f"k{k:03d}" / f"p{p:03d}"
                        bdir.mkdir(parents=True, exist_ok=True)
                        coords_out = (
                            xr.coordinates
                            if xr.coordinates is not None
                            else candidate.coordinates
                        )
                        suffix = "_xtb.xyz" if propagation_eligible else "_offpath.xyz"
                        xyz_path = bdir / f"{sid}{suffix}"
                        lines = [
                            str(len(isomer.symbols)),
                            f"{sid} energy_eV={energy} move=Z "
                            f"occupation_id={occupation.occupation_id} "
                            f"propagation_eligible={str(propagation_eligible).lower()} "
                            f"topology={topology_status}",
                        ]
                        lines.extend(
                            f"{symbol} {point[0]:.8f} {point[1]:.8f} {point[2]:.8f}"
                            for symbol, point in zip(isomer.symbols, coords_out)
                        )
                        xyz_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                        if xr.bond_orders is not None:
                            write_wbo_file(
                                xyz_path.with_suffix(".wbo"), xr.bond_orders
                            )
                        manifest_record = {
                            "schema_version": 2,
                            "structure_id": sid,
                            "occupation": occupation_to_record(occupation),
                            "parent_structure_id": occupation.parent_id,
                            "parent_structure_ids": list(
                                occupation.parent_structure_ids
                            ),
                            "decoration_id": f"dec{decoration_index:03d}",
                            "decoration_certificate": repr(
                                _graph_certificate(state)
                            ),
                            "embedding_start": int(candidate.start_index),
                            "source_edges": [list(edge) for edge in source_edges],
                            "final_edges": [list(edge) for edge in final_edges],
                            "energy_eV": energy,
                            "xtb_converged": bool(xr.converged),
                            "topology_status": topology_status,
                            "propagation_eligible": propagation_eligible,
                            "core_rmsd_A": core_rmsd,
                            "violations": list(dict.fromkeys(str(v) for v in violations)),
                            "xyz": str(xyz_path.relative_to(output_dir)),
                        }
                        _append_zb_manifest(Path(output_dir), manifest_record)

                    if not propagation_eligible or energy is None:
                        continue
                    previous = child_minima.get((k, p))
                    if previous is None or energy < float(previous["energy_eV"]):
                        child_minima[(k, p)] = {
                            "energy_eV": energy,
                            "structure_id": sid,
                        }
                    bin_ranks.setdefault((k, p), []).append(
                        RankedIsomer(
                            structure_id=sid,
                            xtb_energy_eV=energy,
                            seed_skeleton=occupation.occupation_id[-6:],
                            growth_move="Z",
                            parent_id=occupation.parent_id,
                        )
                    )
                    if output_dir is not None:
                        row = {
                            "k": k,
                            "p": p,
                            "structure_id": sid,
                            "xtb_energy_eV": f"{energy:.8f}",
                            "xtb_converged": True,
                            "dE_f_eV": "",
                            "growth": "1",
                            "move": "Z",
                            "shed": occupation.shed,
                            "p_m": occupation.p_m,
                            "parent_id": occupation.parent_id,
                            "occupation_id": occupation.occupation_id,
                            "parent_occupation_ids": "|".join(
                                occupation.parent_occupation_ids
                            ),
                            "parent_structure_ids": "|".join(
                                occupation.parent_structure_ids
                            ),
                            "decoration_id": f"dec{decoration_index:03d}",
                            "embedding_start": candidate.start_index,
                            "topology_status": topology_status,
                            "propagation_eligible": True,
                            "core_rmsd_A": f"{core_rmsd:.8f}",
                        }
                        if growth.references is not None:
                            row["dE_f_eV"] = f"{growth.references.formation_eV(energy, k, p):.8f}"
                            for dmu in growth.delta_mu_cdcl2_eV:
                                row[f"Omega_dmu_{dmu:+.2f}"] = (
                                    f"{growth.references.grand_potential_eV(energy, k, p, dmu):.8f}"
                                )
                        _append_index_row_unique(
                            Path(output_dir), row, _growth_index_fields(growth)
                        )


def _growth_index_fields(growth: GrowthConfig) -> List[str]:
    """Column order shared by Move A and Move B index writers.

    A used to omit ``shed`` / ``p_m`` / ``parent_id``, so appending A rows
    into a B-started ``index.csv`` shifted every soft-rule column.
    """

    fields = [
        "k",
        "p",
        "structure_id",
        "xtb_energy_eV",
        "xtb_converged",
        "dE_f_eV",
        "growth",
        "move",
        "shed",
        "p_m",
        "parent_id",
        "occupation_id",
        "parent_occupation_ids",
        "parent_structure_ids",
        "decoration_id",
        "embedding_start",
        "topology_status",
        "propagation_eligible",
        "core_rmsd_A",
    ]
    for dmu in growth.delta_mu_cdcl2_eV:
        fields.append(f"Omega_dmu_{dmu:+.2f}")
    fields.extend(SOFT_INDEX_FIELDS)
    return fields


def _write_growth_bin(
    out_dir: Path,
    bin_res: Any,
    growth: GrowthConfig,
    spec: Optional[NucleationSpec] = None,
) -> None:
    """Write a lightweight index + formation columns for one bin.

    Index rows are skipped if ``structure_id`` is already present (restart-safe).
    Failed / artifact A opts (``xtb_energy_eV is None``) are not written to
    disk — they are not parents and they used to flood the bin with
    ``energy_eV=None`` XYZ.
    """

    k, p = bin_res.k, bin_res.p
    bdir = out_dir / f"k{k:03d}" / f"p{p:03d}"
    bdir.mkdir(parents=True, exist_ok=True)
    refs = growth.references
    fields = _growth_index_fields(growth)

    for iso in bin_res.isomers:
        e = iso.xtb_energy_eV
        if e is None:
            continue
        row = {
            "k": k,
            "p": p,
            "structure_id": iso.structure_id,
            "xtb_energy_eV": f"{e:.8f}",
            "xtb_converged": iso.xtb_converged,
            "dE_f_eV": "",
            "growth": "1",
            "move": "A",
            "shed": getattr(iso, "shed", ""),
            "p_m": getattr(iso, "p_m", ""),
            "parent_id": getattr(iso, "parent_id", ""),
        }
        coords = iso.xtb_coordinates or iso.coordinates
        if coords is not None:
            apply_soft_columns(
                row,
                symbols=[a.symbol for a in iso.atoms],
                coords=np.asarray(coords, dtype=float),
                energy_eV=e,
                k=k,
                rules=growth.window_for(k).soft_rules,
                spec=spec,
            )
        if refs is not None:
            de_f = refs.formation_eV(e, k, p)
            row["dE_f_eV"] = f"{de_f:.8f}"
            for dmu in growth.delta_mu_cdcl2_eV:
                key = f"Omega_dmu_{dmu:+.2f}"
                row[key] = f"{refs.grand_potential_eV(e, k, p, dmu):.8f}"
        _append_index_row_unique(out_dir, row, fields)
        if coords is None:
            continue
        xyz = bdir / f"{iso.structure_id}_xtb.xyz"
        # do not overwrite a finished xyz on restart unless missing
        if not xyz.is_file():
            symbols = [a.symbol for a in iso.atoms]
            lines = [
                str(len(symbols)),
                f"{iso.structure_id} energy_eV={e} move=A",
            ]
            for sym, pos in zip(symbols, coords):
                lines.append(
                    f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"
                )
            xyz.write_text("\n".join(lines) + "\n")


def write_growth_summary(result: GrowthStepResult, path: Path) -> None:
    """CSV of channels + parent list."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "parent_id",
                "k_parent",
                "p_parent",
                "shed",
                "p_m",
                "k_child",
                "p_child",
                "n_cores",
            ],
        )
        writer.writeheader()
        for ch in result.channels:
            writer.writerow(
                {
                    "parent_id": ch.parent_id,
                    "k_parent": ch.k_parent,
                    "p_parent": ch.p_parent,
                    "shed": ch.shed,
                    "p_m": ch.p_m,
                    "k_child": ch.k_child,
                    "p_child": ch.p_child,
                    "n_cores": ch.n_cores,
                }
            )


__all__ = [
    "GrowthConfig",
    "GrowthLog",
    "ParentStructure",
    "EnergyRecord",
    "GrowthChannelResult",
    "CoordSeed",
    "GrowthStepResult",
    "bond_cutoffs_from_spec",
    "load_energy_index",
    "bin_minima_from_records",
    "merge_bin_minima",
    "seed_package_minima_from_refs",
    "format_prior_map_rankings",
    "load_parents_from_run",
    "select_parents",
    "identify_packages",
    "shed_packages_coords",
    "place_monomer_and_packages",
    "build_coord_seed",
    "compact_growth_id",
    "assign_compact_b_ids",
    "SoftRulesConfig",
    "grow_cores_from_parents",
    "run_growth_step",
    "write_growth_summary",
    "full_opt_relaxation_raw",
    "overlay_pack_full_opt",
]
