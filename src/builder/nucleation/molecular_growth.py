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

FloatArray = np.ndarray
Edge = Tuple[int, int]
EdgeList = Tuple[Edge, ...]
Vec3 = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


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
    shed_mode: str = "wbo"
    max_shed: int = 2
    prefer_low_shed: bool = True
    max_children_per_channel: int = 500
    # geometry / move B (coordinate carry-over)
    start_from: str = "relaxed_coords"  # relaxed_coords | graph_only
    place_monomer: str = "embed_tables"
    clash_policy: str = "soft"
    local_cleanup_enabled: bool = True
    local_cleanup_method: str = "g-xTB"
    local_cleanup_cycles: int = 20
    require_charge_neutral_for_cleanup: bool = True
    child_redecorate: bool = True
    child_full_opt: str = "g-xTB"
    delta_mu_cdcl2_eV: Tuple[float, ...] = ()

    @property
    def use_coord_carry(self) -> bool:
        """True when move B (3D carry-over) should run."""

        return str(self.start_from).lower() in {
            "relaxed_coords",
            "relaxed",
            "coords",
            "coordinate",
            "coordinates",
        }

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
            shed_mode=str(shed.get("mode", "wbo")).lower(),
            max_shed=int(shed.get("max_shed", 2)),
            prefer_low_shed=bool(shed.get("prefer_low_shed", True)),
            max_children_per_channel=int(
                raw.get("max_children_per_channel", 500)
            ),
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
            child_full_opt=str(child.get("full_opt", "g-xTB")),
            delta_mu_cdcl2_eV=dmu,
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
            wbo_path = xyz_path.parent / "wbo"
            if not wbo_path.is_file():
                wbo_path = xyz_path.with_name("wbo")
            parents.append(
                ParentStructure(
                    k=k,
                    p=rp,
                    structure_id=sid or xyz_path.stem,
                    symbols=tuple(symbols),
                    coordinates=coords,
                    energy_eV=energy,
                    edges=edges,
                    core_edges=core,
                    wbo=parse_wbo(wbo_path) or None,
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
            parents.append(
                ParentStructure(
                    k=k,
                    p=rp,
                    structure_id=xyz_path.stem.replace("_xtb", ""),
                    symbols=tuple(symbols),
                    coordinates=coords,
                    energy_eV=energy,
                    edges=edges,
                    core_edges=core,
                    wbo=None,
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
        emin = min(x.energy_eV for x in group)
        windowed = [
            x
            for x in group
            if x.energy_eV <= emin + growth.energy_window_eV
        ]
        if not windowed:
            windowed = [min(group, key=lambda x: x.energy_eV)]

        # skeleton buckets
        buckets: Dict[Tuple[object, ...], List[ParentStructure]] = {}
        for x in windowed:
            fp = _core_fingerprint(x, spec)
            buckets.setdefault(fp, []).append(x)
        # rank skeletons by best energy
        skel_ranked = sorted(
            buckets.items(),
            key=lambda kv: min(y.energy_eV for y in kv[1]),
        )
        n_keep = max(
            1,
            min(
                growth.max_skeletons_cap,
                int(math.ceil(growth.max_skeletons_frac * max(1, len(skel_ranked)))),
            ),
        )
        n_keep = min(n_keep, len(skel_ranked))
        for _fp, members in skel_ranked[:n_keep]:
            members_sorted = sorted(members, key=lambda x: x.energy_eV)
            selected.extend(members_sorted[: growth.decorations_per_skeleton])
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
    """Find CdCl2-like packages on the relaxed graph.

    Prefers Cd with exactly two Cl neighbours.  Score: sum of WBO(Cd–Cl) if
    available, else inverse mean distance (longer bonds → lower score → shed
    first).
    """

    ligand = spec.precursor.ligand
    cation = spec.precursor.center
    symbols = parent.symbols
    coords = parent.coordinates
    # adjacency from edges
    neigh: Dict[int, List[int]] = {i: [] for i in range(len(symbols))}
    for a, b in parent.edges:
        neigh[a].append(b)
        neigh[b].append(a)

    packages: List[CdCl2Package] = []
    for i, sym in enumerate(symbols):
        if sym != cation:
            continue
        cl_n = [j for j in neigh[i] if symbols[j] == ligand]
        if len(cl_n) < 2:
            continue
        # take the two closest Cl as the package
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
        packages.append(CdCl2Package(cd=i, cl=(c1, c2), score=score))
    packages.sort(key=lambda p: (p.score, p.cd))
    return packages


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


def local_cleanup_structure(
    symbols: Sequence[str],
    coords: FloatArray,
    *,
    growth: GrowthConfig,
    pack: Optional[GeometryPack],
    structure_id: str,
) -> Tuple[FloatArray, float, bool, str]:
    """Short g-xTB / GFN cleanup.  Returns (coords, time_s, ok, note)."""

    if not growth.local_cleanup_enabled or growth.local_cleanup_cycles <= 0:
        return np.asarray(coords, dtype=float), 0.0, False, "cleanup_disabled"

    from .xtb_relax import XtbSettings, relax_structures

    method = growth.local_cleanup_method
    # Prefer pack relaxation binary/env when method is g-xTB
    base = {}
    if pack is not None and isinstance(pack.raw, dict):
        base = dict(pack.raw.get("relaxation") or {})
    base["enabled"] = True
    base["method"] = method
    base["max_steps"] = int(growth.local_cleanup_cycles)
    # short wall for cleanup so it cannot dominate
    base.setdefault("timeout_s", 120.0)
    settings = XtbSettings.from_pack(base)
    t0 = time.perf_counter()
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
        f"cleanup_ok steps≤{growth.local_cleanup_cycles}",
    )


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
        symbols, coords, edges = place_monomer_and_packages(
            symbols,
            coords,
            edges,
            k_parent=parent.k,
            p_after_shed=parent.p - s,
            p_m=p_m,
            spec=spec,
            pack=pack,
        )
    except (ValueError, IndexError) as exc:
        return None

    k_child = parent.k + 1
    p_child = parent.p - s + p_m
    sid = (
        f"coord_k{k_child:03d}_p{p_child:03d}_"
        f"from_{parent.structure_id}_s{s}_pm{p_m}_{serial:04d}"
    )

    q = _formal_charge(symbols, spec)
    cleanup_s = 0.0
    cleanup_ok = False
    notes = f"shed={s} p_m={p_m} charge={q}"
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
    catalog: Dict[Tuple[int, int], Dict[EdgeList, None]] = {}
    channels: List[GrowthChannelResult] = []
    parent_records: List[Dict[str, Any]] = []
    coord_seeds: Dict[Tuple[int, int], List[CoordSeed]] = defaultdict(list)
    cutoffs = bond_cutoffs_from_spec(spec)
    seed_serial = 0

    for parent in parents:
        core = parent_core_in_blocks(parent, spec)
        packages = identify_packages(parent, spec)
        max_s = min(growth.max_shed, parent.p, len(packages) if packages else parent.p)
        if growth.prefer_low_shed:
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
                "source": parent.source_path,
            }
        )

        for s in s_order:
            p_out = parent.p - s
            if p_out < 0:
                continue

            # ---- Move A: graph catalog ----
            children_base: List[EdgeList] = []
            if core is not None:
                children_base = shed_and_grow(
                    core,
                    k=parent.k,
                    p=parent.p,
                    p_out=p_out,
                    spec=spec,
                    max_children=growth.max_children_per_channel,
                )
            for p_m in growth.monomer_p_values:
                p_child = p_out + p_m
                children_pm = _inflate_cores_with_precursor(
                    children_base,
                    k_child=parent.k + 1,
                    p_from=p_out,
                    p_to=p_child,
                    spec=spec,
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
                if growth.use_coord_carry:
                    seed_serial += 1
                    seed = build_coord_seed(
                        parent,
                        s=s,
                        p_m=p_m,
                        growth=growth,
                        spec=spec,
                        pack=pack,
                        cutoffs=cutoffs,
                        serial=seed_serial,
                    )
                    if seed is not None:
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
    )


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

    Distinguishes:
      * **cores** — unique child Cd–Se skeletons (known up front)
      * **gxtb** — real g-xTB opt calculations (known only as graphs appear)
      * **merge** — duplicate graphs that skip g-xTB (free)

    Timing on job lines:
      * ``opt=``   wall time of the g-xTB binary (or 0 for merges)
      * ``recon=`` motif_factor 3D rebuild for that unique graph (0 if merge)
      * ``+Δ=``    wall clock since the previous job/bin event — this is what
                   you actually wait, including Cl decoration between graphs
    """

    def __init__(self, *, verbose: bool = False, quiet: bool = False) -> None:
        self.verbose = bool(verbose)
        self.quiet = bool(quiet)
        self._job_i = 0
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
        # wall clock
        self._t_mark: float = time.perf_counter()
        self._t_bin0: float = self._t_mark
        self._t_step0: float = self._t_mark
        self._sum_opt_s: float = 0.0
        self._sum_recon_s: float = 0.0
        self._bin_opt_s: float = 0.0
        self._bin_recon_s: float = 0.0

    def _tick(self) -> float:
        """Seconds since last mark; advances the mark."""

        now = time.perf_counter()
        dt = now - self._t_mark
        self._t_mark = now
        return dt

    def stage(self, n: int, total: int, title: str, **fields: Any) -> None:
        if self.quiet:
            return
        self._tick()
        print(f"\n=== STAGE {n}/{total}: {title} ===", flush=True)
        for key, value in fields.items():
            print(f"  {key}: {value}", flush=True)

    def line(self, msg: str) -> None:
        if self.quiet:
            return
        print(f"[growth] {msg}", flush=True)

    def pipeline_blurb(self, growth: "GrowthConfig") -> None:
        """One-time note: both growth moves and timing keys."""

        if self.quiet:
            return
        print("[growth] two growth moves (both when start_from=relaxed_coords):", flush=True)
        print(
            "[growth]   A graph: combinatorial precursor-Cd shed on core graph → "
            f"p_m={list(growth.monomer_p_values)} inflate → Cl redecorate → "
            "motif_factor rebuild → full g-xTB",
            flush=True,
        )
        if growth.use_coord_carry:
            print(
                "[growth]   B coord: parent XYZ → WBO package shed (s least-bound "
                f"CdCl2) → place CdSe+p_m CdCl2 (embed distances) → "
                f"cleanup {'ON' if growth.local_cleanup_enabled else 'OFF'}"
                f"({growth.local_cleanup_method}, ≤{growth.local_cleanup_cycles} steps"
                f"{', neutral-only' if growth.require_charge_neutral_for_cleanup else ''}"
                f") → full g-xTB",
                flush=True,
            )
        else:
            print(
                "[growth]   B coord: OFF (geometry.start_from != relaxed_coords)",
                flush=True,
            )
        print(
            f"[growth]   shed: mode={growth.shed_mode} max_shed={growth.max_shed} "
            f"prefer_low_shed={growth.prefer_low_shed}",
            flush=True,
        )
        print(
            "[growth]   timing keys: opt= full g-xTB; recon= motif_factor (A only); "
            "cleanup= short prelax (B); +dt= wall since previous line",
            flush=True,
        )

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
        if self.quiet:
            return
        print(
            f"[growth] work plan: {self._cores_total} unique child cores "
            f"across {len(bin_plan)} bins",
            flush=True,
        )
        print(f"[growth]   by bin: {dict(bin_plan)}", flush=True)
        print(
            "[growth]   note: #g-xTB calcs ≠ #cores — each core yields 0+ Cl "
            "decorations; each *unique* graph runs "
            f"≤{self._xtb_starts} g-xTB opt(s); merges skip g-xTB",
            flush=True,
        )

    def log_channel_summary(
        self,
        channels: Sequence["GrowthChannelResult"],
        parent_records: Sequence[Mapping[str, Any]],
    ) -> None:
        """Compact shed / package accounting after core growth."""

        if self.quiet:
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
        print(
            f"[growth] channels: {n_ch}  "
            f"(graph={by_move.get('graph', 0)}  coord={by_move.get('coord', 0)})",
            flush=True,
        )
        s_bits = [
            f"s={s}: ch={by_s[s]} cores+={by_s_cores[s]}"
            for s in sorted(by_s)
        ]
        pm_bits = [
            f"p_m={pm}: ch={by_pm[pm]} cores+={by_pm_cores[pm]}"
            for pm in sorted(by_pm)
        ]
        print(f"[growth]   by shed s:  {',  '.join(s_bits) or '(none)'}", flush=True)
        print(f"[growth]   by p_m:     {',  '.join(pm_bits) or '(none)'}", flush=True)
        n_pkg = sum(int(r.get("n_packages") or 0) for r in parent_records)
        n_wbo = sum(1 for r in parent_records if r.get("has_wbo"))
        print(
            f"[growth]   parent packages: {n_pkg} total; "
            f"{n_wbo}/{len(parent_records)} parents have WBO "
            f"(coord shed uses WBO/distance rank; graph shed is combinatorial)",
            flush=True,
        )

    def begin_bin(
        self,
        *,
        k: int,
        p: int,
        cores: int,
        cores_done: int,
        cores_total: int,
    ) -> None:
        """Open a bin header with core accounting."""

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
        self._t_bin0 = time.perf_counter()
        self._tick()
        if self.quiet:
            return
        after = cores_done + cores
        print(
            f"[growth] --- bin k={k} p={p} ---",
            flush=True,
        )
        print(
            f"[growth]   cores in this bin: {cores}   "
            f"(overall cores {cores_done} done → {after}/{cores_total} after bin)",
            flush=True,
        )
        print(
            f"[growth]   path: core graph → Cl decorate → motif_factor → g-xTB "
            f"(≤{self._xtb_starts} opt/unique graph); merges free",
            flush=True,
        )
        if self.n_gxtb or self.n_merge:
            print(
                f"[growth]   so far (all bins): gxtb={self.n_gxtb}  "
                f"merge={self.n_merge}  fail={self.n_fail}  "
                f"optΣ={self._sum_opt_s:.0f}s reconΣ={self._sum_recon_s:.0f}s",
                flush=True,
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

        if self.quiet:
            return
        wall = time.perf_counter() - self._t_bin0
        extra = ""
        if raw_graphs is not None:
            extra = f"  raw_graphs={raw_graphs}"
        print(
            f"[growth] bin k={k} p={p} done: isomers={n_iso}  "
            f"with_E={n_ok}  no_E={n_fail}  graph_merges={n_merged}"
            f"{extra}",
            flush=True,
        )
        print(
            f"[growth]   this bin: gxtb={self._bin_gxtb}  "
            f"merge={self._bin_merge}  fail={self._bin_fail}  "
            f"wall={wall:.1f}s  optΣ={self._bin_opt_s:.1f}s  "
            f"reconΣ={self._bin_recon_s:.1f}s  "
            f"(other≈{max(0.0, wall - self._bin_opt_s - self._bin_recon_s):.1f}s "
            f"decorate/merge/overhead)",
            flush=True,
        )
        print(
            f"[growth]   global: gxtb={self.n_gxtb}  merge={self.n_merge}  "
            f"fail={self.n_fail}",
            flush=True,
        )
        self._tick()

    def __call__(self, msg: str) -> None:
        """ProgressCallback-compatible."""

        if self.quiet:
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
                print(
                    f"  merge {self.n_merge:4d}  k={k} p={p}  {sid:22s}  "
                    f"-> {target:22s}  "
                    f"+dt={dt:5.1f}s  "
                    f"[bin m{self._bin_merge} | g{self.n_gxtb} m{self.n_merge}]",
                    flush=True,
                )
                return

            self.n_gxtb += 1
            self._bin_gxtb += 1
            self._sum_opt_s += opt_s
            self._sum_recon_s += recon_s
            self._bin_opt_s += opt_s
            self._bin_recon_s += recon_s
            if rel == "fail":
                self.n_fail += 1
                self._bin_fail += 1
            extra = f" ({err})" if err and rel == "fail" else ""
            # recon column: motif_factor (A) or cleanup prelax (B)
            recon_lab = "clean" if move in ("B", "coord", "b") else "recon"
            print(
                f"  gxtb {self.n_gxtb:4d}  k={k} p={p}  move={move}  "
                f"{sid:22s}  E={e:>14s}  "
                f"opt={opt_s:5.1f}s {recon_lab}={recon_s:4.1f}s +dt={dt:5.1f}s  "
                f"relax={rel}{extra}  "
                f"[bin g{self._bin_gxtb} | g{self.n_gxtb} m{self.n_merge}]",
                flush=True,
            )
            return
        if text.startswith("[growth]"):
            print(text, flush=True)
            return
        if self.verbose:
            print(text, flush=True)


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
) -> GrowthStepResult:
    """Select parents at ``k_from``, grow cores to k+1, optionally decorate.

    When ``decorate`` is true, each child (k, p) bin is built with
    ``enumerate_molecular_bin(..., precomputed_skeletons=...)`` and written
    under ``output_dir`` if given.
    """

    log = progress if isinstance(progress, GrowthLog) else None

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

    n_stages = 4 if decorate else 3

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
    if log:
        log.line(
            f"loaded {len(parents_all)} converged parents → "
            f"selected {len(parents)} "
            f"(window={growth.energy_window_eV} eV, "
            f"≤{growth.decorations_per_skeleton} dec/core)"
        )
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
            packages=list(growth.monomer_p_values),
            max_shed=growth.max_shed,
            move_A="graph combinatorial shed",
            move_B=(
                "coord WBO shed + place + cleanup"
                if growth.use_coord_carry
                else "off"
            ),
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
        log.line(
            f"channels={len(result.channels)}  unique_cores={n_cores}  "
            f"coord_seeds={n_seeds}  "
            f"(A→redecorate; B→full opt of carried geometry)"
        )
        log.set_work_plan(
            cores_total=n_cores,
            bin_plan=bin_plan,
            xtb_starts_per_graph=xtb_starts,
        )
    elif progress:
        progress(
            f"[growth] k={k_from}→{k_from + 1}: "
            f"{len(result.channels)} channels, {n_cores} unique child cores, "
            f"bins={sorted(result.skeleton_catalog)}"
        )

    # Report-only: print parent-map rankings (k=1 … k_from) before child opts
    if log:
        prior = format_prior_map_rankings(
            run_dir, growth=growth, k_max=k_from
        )
        print(prior, flush=True)

    child_minima: Dict[Tuple[int, int], Dict[str, Any]] = {}
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
        out = Path(output_dir) if output_dir else None
        if out:
            out.mkdir(parents=True, exist_ok=True)

        # ---- Move B first: full opt of coordinate-carried seeds ----
        if result.coord_seeds and embed:
            if log:
                log.stage(
                    3,
                    n_stages,
                    "move B: full opt of coord-carried seeds",
                    n_seeds=n_seeds,
                    note="WBO shed + placed monomer; already cleaned if enabled",
                )
            _opt_coord_seeds(
                result.coord_seeds,
                growth=growth,
                map_spec=map_spec,
                pack=pack,
                output_dir=out,
                progress=log if log else progress,
                child_minima=child_minima,
            )

        if log and result.skeleton_catalog:
            log.stage(
                3 if not result.coord_seeds else 3,
                n_stages,
                "move A: decorate cores + motif_factor + opt",
                total_cores=n_cores,
                note=(
                    "graph cores → Cl redecorate → motif_factor → g-xTB; "
                    "merges free"
                ),
            )
        done_cores = 0
        for (k, p), cores in sorted(result.skeleton_catalog.items()):
            n_coord = len(result.coord_seeds.get((k, p), ()))
            if log:
                log.begin_bin(
                    k=k,
                    p=p,
                    cores=len(cores),
                    cores_done=done_cores,
                    cores_total=n_cores,
                )
                log.line(
                    f"  bin moves: A_graph_cores={len(cores)}  "
                    f"B_coord_seeds={n_coord} (B already opted if present)"
                )
            elif progress:
                progress(
                    f"[growth] decorate k={k} p={p} cores={len(cores)}"
                )
            bin_res = enumerate_molecular_bin(
                k,
                p,
                map_spec,
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
            if log:
                n_iso = len(bin_res.isomers)
                n_ok = len(with_e)
                n_merged = sum(
                    1
                    for rec in getattr(bin_res, "graph_merge_records", []) or []
                )
                n_fail = max(0, n_iso - n_ok)
                log.end_bin(
                    k=k,
                    p=p,
                    n_iso=n_iso,
                    n_ok=n_ok,
                    n_fail=n_fail,
                    n_merged=n_merged,
                    raw_graphs=getattr(bin_res, "raw_graphs", None),
                )
                ranking = format_bin_ranking(
                    bin_res.isomers,
                    k=k,
                    p=p,
                    refs=growth.references,
                    package_p_m=tuple(growth.monomer_p_values) or (1, 2, 3),
                    delta_mu=tuple(growth.delta_mu_cdcl2_eV)
                    or (-1.0, 0.0, 1.0),
                )
                print(ranking, flush=True)
            if out is not None:
                _write_growth_bin(out, bin_res, growth)

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


def _opt_coord_seeds(
    coord_seeds: Mapping[Tuple[int, int], Sequence[CoordSeed]],
    *,
    growth: GrowthConfig,
    map_spec: NucleationSpec,
    pack: Optional[GeometryPack],
    output_dir: Optional[Path],
    progress: Optional[Any],
    child_minima: Dict[Tuple[int, int], Dict[str, Any]],
) -> None:
    """Full g-xTB opt for move-B coordinate seeds; write XYZ + index rows."""

    from .xtb_relax import XtbSettings, relax_structures

    if not coord_seeds:
        return
    base: Dict[str, Any] = {}
    if pack is not None and isinstance(pack.raw, dict):
        base = dict(pack.raw.get("relaxation") or {})
    base["enabled"] = True
    if growth.child_full_opt:
        base["method"] = growth.child_full_opt
    settings = XtbSettings.from_pack(base)
    cutoffs = bond_cutoffs_from_spec(map_spec)
    log = progress if isinstance(progress, GrowthLog) else None

    flat: List[CoordSeed] = []
    for key in sorted(coord_seeds):
        flat.extend(coord_seeds[key])
    if log:
        log.line(f"move B: optimizing {len(flat)} coord-carried children")
    elif progress:
        progress(f"[growth] move B: opt {len(flat)} coord seeds")

    for seed in flat:
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
        recon_s = float(seed.cleanup_s)  # report cleanup time under recon slot
        if progress is not None:
            if xr.ok and xr.energy_eV is not None:
                progress(
                    f"[growth-job] k={seed.k} p={seed.p} move=B "
                    f"id={seed.structure_id} "
                    f"E_eV={float(xr.energy_eV):.6f} "
                    f"t_s={opt_s:.1f} recon_s={recon_s:.1f} "
                    f"relax={'ok' if xr.converged else 'fail'}"
                )
            else:
                progress(
                    f"[growth-job] k={seed.k} p={seed.p} move=B "
                    f"id={seed.structure_id} "
                    f"E_eV=n/a t_s={opt_s:.1f} recon_s={recon_s:.1f} "
                    f"relax=fail err={xr.error or 'coord_opt'}"
                )
        if not xr.ok or xr.energy_eV is None:
            continue
        e = float(xr.energy_eV)
        prev = child_minima.get((seed.k, seed.p))
        if prev is None or e < float(prev["energy_eV"]):
            child_minima[(seed.k, seed.p)] = {
                "energy_eV": e,
                "structure_id": seed.structure_id,
            }
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
        # append index.csv
        index_path = Path(output_dir) / "index.csv"
        write_header = not index_path.is_file()
        refs = growth.references
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
        with index_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()
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
            if refs is not None:
                de_f = refs.formation_eV(e, seed.k, seed.p)
                row["dE_f_eV"] = f"{de_f:.8f}"
                for dmu in growth.delta_mu_cdcl2_eV:
                    key = f"Omega_dmu_{dmu:+.2f}"
                    row[key] = f"{refs.grand_potential_eV(e, seed.k, seed.p, dmu):.8f}"
            writer.writerow(row)


def _write_growth_bin(
    out_dir: Path,
    bin_res: Any,
    growth: GrowthConfig,
) -> None:
    """Write a lightweight index + formation columns for one bin."""

    k, p = bin_res.k, bin_res.p
    bdir = out_dir / f"k{k:03d}" / f"p{p:03d}"
    bdir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.csv"
    write_header = not index_path.is_file()
    refs = growth.references
    fields = [
        "k",
        "p",
        "structure_id",
        "xtb_energy_eV",
        "xtb_converged",
        "dE_f_eV",
        "growth",
    ]
    for dmu in growth.delta_mu_cdcl2_eV:
        fields.append(f"Omega_dmu_{dmu:+.2f}")

    with index_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for iso in bin_res.isomers:
            e = iso.xtb_energy_eV
            row = {
                "k": k,
                "p": p,
                "structure_id": iso.structure_id,
                "xtb_energy_eV": "" if e is None else f"{e:.8f}",
                "xtb_converged": iso.xtb_converged,
                "dE_f_eV": "",
                "growth": "1",
            }
            if e is not None and refs is not None:
                de_f = refs.formation_eV(e, k, p)
                row["dE_f_eV"] = f"{de_f:.8f}"
                for dmu in growth.delta_mu_cdcl2_eV:
                    key = f"Omega_dmu_{dmu:+.2f}"
                    row[key] = f"{refs.grand_potential_eV(e, k, p, dmu):.8f}"
            writer.writerow(row)
            if iso.xtb_coordinates is not None or iso.coordinates is not None:
                coords = iso.xtb_coordinates or iso.coordinates
                xyz = bdir / f"{iso.structure_id}_xtb.xyz"
                symbols = [a.symbol for a in iso.atoms]
                lines = [str(len(symbols)), f"{iso.structure_id} energy_eV={e}"]
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
    "grow_cores_from_parents",
    "run_growth_step",
    "write_growth_summary",
]
