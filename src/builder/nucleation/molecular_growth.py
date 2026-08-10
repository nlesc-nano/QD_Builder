"""Package growth for the lattice-free molecular map.

Mirrors lattice nucleation building blocks:

    parent (k, p)
      -> shed s complete CdCl2 packages (WBO: least-bound first)
      -> add CdSe + p_m CdCl2  (monomer_p_values)
      -> p_child = p - s + p_m
      -> redecorate + full g-xTB (via existing map pipeline)

Parents use **relaxed** coordinates and distance-inferred bonds (pack
``bond_max_distance``).  Chemical-potential / grand-potential numbers are
report-only (see ``formation.py``); they never filter parents or channels.

Coordinate carry-over: child core starts from stripped parent XYZ + placed
CdSe; decoration still uses the map embedder unless a future path injects
the frame.
"""

from __future__ import annotations

import csv
import math
import re
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
    local_cleanup_cycles: int = 20
    delta_mu_cdcl2_eV: Tuple[float, ...] = ()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GrowthConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"growth.yaml must be a mapping: {path}")
        parents = raw.get("parents") or {}
        shed = raw.get("shed") or {}
        geom = raw.get("geometry") or {}
        cleanup = (geom.get("local_cleanup") or {}) if isinstance(geom, dict) else {}
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
            local_cleanup_cycles=int(cleanup.get("max_cycles", 20)),
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


@dataclass
class GrowthStepResult:
    """Outcome of growing all parents from k → k+1."""

    k_from: int
    k_to: int
    parents_selected: int
    channels: List[GrowthChannelResult]
    skeleton_catalog: Dict[Tuple[int, int], List[EdgeList]]
    parent_records: List[Dict[str, Any]] = field(default_factory=list)


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
# Growth step
# ---------------------------------------------------------------------------


def grow_cores_from_parents(
    parents: Sequence[ParentStructure],
    *,
    growth: GrowthConfig,
    spec: NucleationSpec,
) -> GrowthStepResult:
    """Build child Cd–Se core catalogs for k+1 from selected parents.

    For each parent, shed ``s`` packages (0…max_shed) then add a monomer
    package with ``p_m`` CdCl2 units::

        p_child = p_parent - s + p_m

    Core graphs come from ``shed_and_grow`` at ``p_out = p - s``, then
    precursor slots are inflated when ``p_m > 0``.  Package WBO scores are
    recorded on parents for reporting; combinatorial shed in
    ``shed_and_grow`` enumerates legal cores.
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

    for parent in parents:
        core = parent_core_in_blocks(parent, spec)
        if core is None:
            continue
        packages = identify_packages(parent, spec)
        max_s = min(growth.max_shed, parent.p)
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
                "source": parent.source_path,
            }
        )

        for s in s_order:
            p_out = parent.p - s
            if p_out < 0:
                continue
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
        )
    )


class GrowthLog:
    """Compact growth logger: stages + job lines; hides verbose molecular spam.

    Always prints lines starting with ``[growth]`` or ``[growth-job]``.
    With ``verbose=True``, also prints the rest (motif details, etc.).
    """

    def __init__(self, *, verbose: bool = False, quiet: bool = False) -> None:
        self.verbose = bool(verbose)
        self.quiet = bool(quiet)
        self._job_i = 0

    def stage(self, n: int, total: int, title: str, **fields: Any) -> None:
        if self.quiet:
            return
        print(f"\n=== STAGE {n}/{total}: {title} ===", flush=True)
        for key, value in fields.items():
            print(f"  {key}: {value}", flush=True)

    def line(self, msg: str) -> None:
        if self.quiet:
            return
        print(f"[growth] {msg}", flush=True)

    def __call__(self, msg: str) -> None:
        """ProgressCallback-compatible."""

        if self.quiet:
            return
        text = str(msg)
        if text.startswith("[growth-job]"):
            self._job_i += 1
            parts = text.replace("[growth-job]", "").strip().split()
            kv = {}
            for part in parts:
                if "=" in part:
                    key, val = part.split("=", 1)
                    kv[key] = val
            sid = kv.get("id", "?")
            e = kv.get("E_eV", "n/a")
            t = kv.get("t_s", "?")
            rel = kv.get("relax", "?")
            k = kv.get("k", "?")
            p = kv.get("p", "?")
            err = kv.get("err", "")
            into = kv.get("into", "")
            if rel == "merged" or e == "merged":
                target = into or "?"
                print(
                    f"  job {self._job_i:4d}  k={k} p={p}  {sid:28s}  "
                    f"E={'merged':>16s}      t={t:>6s}s  status=merged→{target}",
                    flush=True,
                )
                return
            extra = f"  ({err})" if err and rel == "fail" else ""
            print(
                f"  job {self._job_i:4d}  k={k} p={p}  {sid:28s}  "
                f"E={e:>16s} eV  t={t:>6s}s  relax={rel}{extra}",
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
        log.stage(
            2,
            n_stages,
            f"grow cores  k={k_from} → k={k_from + 1}",
            packages=list(growth.monomer_p_values),
            max_shed=growth.max_shed,
        )
    result = grow_cores_from_parents(parents, growth=growth, spec=map_spec)
    n_cores = sum(len(v) for v in result.skeleton_catalog.values())
    bin_plan = {
        f"k{k}p{p}": len(cores)
        for (k, p), cores in sorted(result.skeleton_catalog.items())
    }
    if log:
        log.line(
            f"channels={len(result.channels)}  unique_cores={n_cores}  "
            f"(these cores will be decorated)"
        )
        log.line(f"plan by bin: {bin_plan}")
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
    if decorate and result.skeleton_catalog:
        if pack is None and map_spec.geometry_pack:
            pack = load_geometry_pack(map_spec.geometry_pack)
        out = Path(output_dir) if output_dir else None
        if out:
            out.mkdir(parents=True, exist_ok=True)
        if log:
            log.stage(
                3,
                n_stages,
                "decorate + embed + opt",
                total_cores=n_cores,
                note="one relax job per accepted decorated graph",
            )
        done_cores = 0
        for (k, p), cores in sorted(result.skeleton_catalog.items()):
            if log:
                log.line(
                    f"--- bin k={k} p={p}  cores={len(cores)}  "
                    f"(running total cores {done_cores}/{n_cores}) ---"
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
                log.line(
                    f"bin k={k} p={p} done: isomers={n_iso}  "
                    f"with_E={n_ok}  no_E={n_fail}  "
                    f"graph_merges={n_merged}"
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
    "grow_cores_from_parents",
    "run_growth_step",
    "write_growth_summary",
]
