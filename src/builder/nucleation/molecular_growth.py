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
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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
    """Lightweight energy row for merged A+B bin rankings."""

    structure_id: str
    xtb_energy_eV: float
    seed_skeleton: str = "------"
    growth_move: str = "?"  # A | B
    parent_id: str = ""


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
                )
            )
    return parents


def _core_fingerprint(
    parent: ParentStructure, spec: NucleationSpec
) -> Tuple[object, ...]:
    """Isomorphism class of the relaxed Cd–Se core."""

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


def select_parents(
    parents: Sequence[ParentStructure],
    growth: GrowthConfig,
    spec: NucleationSpec,
) -> List[ParentStructure]:
    """Energy window + skeleton diversity + decorations per skeleton."""

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
        for _fp, members in skel_ranked[:n_keep]:
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

    **Move Z (zb_sites):** snap the parent core onto zinc-blende sites,
    shed extra Cd, fill a vacant CdSe pair (+ p_m precursor Cd).  Cl is
    placed by the 2p law around that core at opt time.  The relaxed
    child is kept only if it still embeds on zb.
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
    zb_seen: set = set()
    zb_model = None
    zb_stats = None
    if use_zb:
        from .molecular_zb_growth import (
            ZbGrowStats,
            grow_zb_children,
            lattice_k1_occupation,
            lattice_model,
            snap_parent,
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
            }
        )

        zb_occ = None
        if use_zb and zb_model is not None and zb_stats is not None:
            zb_stats.parents += 1
            zb_occ, why = snap_parent(
                parent.symbols,
                parent.coordinates,
                spec,
                zb_model,
                parent_id=parent.structure_id,
                k=parent.k,
                p=parent.p,
            )
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
                    )
                    kept_kids = []
                    for kid in kids:
                        uniq = (kid.k, kid.p, tuple(sorted(kid.core_edges)))
                        if uniq in zb_seen:
                            continue
                        zb_seen.add(uniq)
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
                "  Z zb_sites: snap parent core onto zinc-blende -> "
                "shed extra Cd -> fill vacant CdSe pair + p_m Cd -> "
                "2p Cl on that core -> clash check -> full g-xTB -> "
                "keep only if relaxed core still embeds on zb.  "
                "Genealogy is the occupation, not the XYZ."
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
        log.line(
            f"loaded {len(parents_all)} converged parents → "
            f"selected {len(parents)} "
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
    # Merged A+B energy rows per (k,p); ranked once after all bins finish
    bin_ranks: Dict[Tuple[int, int], List[RankedIsomer]] = defaultdict(list)
    do_redecorate = bool(decorate and window.child_redecorate)
    if window.move_zb_sites:
        # Z owns decorate+opt from CIF core coords; motif_factor would
        # throw the occupation away.
        do_redecorate = False
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

        if result.zb_seeds and embed:
            if log:
                log.set_block_plan(n_z_blocks)
                log.stage(
                    3,
                    n_stages,
                    "move Z: 2p Cl on zb core + full opt",
                    n_occupations=sum(
                        len(v) for v in result.zb_seeds.values()
                    ),
                    note="keep only zb-embeddable endpoints; no energy=None XYZ",
                )
            _opt_zb_seeds(
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
                    f"  bin: A_graph_cores={len(cores)}  "
                    f"B_coord_seeds={n_coord} (B already opted; "
                    f"rank merged at end)"
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

        # ---- Final merged rankings for every child (k,p) ----
        if log and bin_ranks:
            log.stage(
                4,
                n_stages,
                "merged rankings A+B per (k,p)",
                note="all sheddings/packages/moves for each composition",
            )
            for (k, p) in sorted(bin_ranks):
                rows = bin_ranks[(k, p)]
                if not rows:
                    continue
                n_a = sum(1 for r in rows if r.growth_move == "A")
                n_b = sum(1 for r in rows if r.growth_move == "B")
                ranking = format_bin_ranking(
                    rows,
                    k=k,
                    p=p,
                    refs=growth.references,
                    package_p_m=tuple(growth.monomer_p_values) or (1, 2, 3),
                    delta_mu=tuple(growth.delta_mu_cdcl2_eV)
                    or (-1.0, 0.0, 1.0),
                    title_note=f"merged A={n_a} B={n_b}",
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
                symbols, coords, _edges = place_cl_2p(occ, map_spec, pack)
            except Exception:
                if zb_stats is not None:
                    zb_stats.clash_skip += 1
                continue
            if construction_clash(symbols, coords, map_spec):
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
