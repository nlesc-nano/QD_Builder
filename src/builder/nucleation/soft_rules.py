"""Soft ranking terms for molecular growth (YAML ``soft_rules``).

After the energy window, parents / children can be ranked by

    score = E + penalty

where ``penalty`` is a sum of local graph terms (eV).  None of these
force zinc-blende, a sphere, or a Wulff count.  They only prefer
motifs that zb, wz and polytwistane share and down-rank the B sausage.

  * diamond      — Cd–Se–Cd–Se 4-rings (not Cd–Se–Cd–Cl rhombi)
  * f6           — 6-ring fusion quality: share 2 or 3 atoms is clean;
                   share ≥ 4 is a diamond collapsed into two chairs
  * terminal_se3cl — Cd with 3 Se + a Cl that sees only that Cd
  * se1cl3       — precursor-like CdCl3 on one Se (soft, high k)
  * asphericity  — off by default; optional, and can apply only when
                   f6_dirty > 0 so rods/platelets with clean fusion pass
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..nc_types import NucleationSpec

Edge = Tuple[int, int]


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def _cutoffs_from_spec(
    spec: Optional[NucleationSpec],
) -> Dict[Tuple[str, str], float]:
    cation, anion, ligand = "Cd", "Se", "Cl"
    if spec is not None:
        cation = str(spec.core.cation)
        anion = str(spec.core.anion)
        ligand = str(spec.precursor.ligand)
    cut: Dict[Tuple[str, str], float] = {
        _pair_key(cation, anion): 3.25,
        _pair_key(cation, ligand): 2.90,
    }
    if spec is None:
        return cut
    rules = getattr(getattr(spec, "graph_rules", None), "pair_rules", None) or {}
    for rule in rules.values():
        if not getattr(rule, "bond_allowed", False):
            continue
        dmax = getattr(rule, "bond_max_distance", None)
        if dmax is None:
            continue
        elems = tuple(sorted(rule.elements))
        cut[elems] = float(dmax)
    return cut


def _species(spec: Optional[NucleationSpec]) -> Tuple[str, str, str]:
    if spec is None:
        return "Cd", "Se", "Cl"
    return (
        str(spec.core.cation),
        str(spec.core.anion),
        str(spec.precursor.ligand),
    )


def _neighbour_lists(
    symbols: Sequence[str],
    coords: np.ndarray,
    cutoffs: Mapping[Tuple[str, str], float],
) -> List[List[int]]:
    n = len(symbols)
    neigh: List[List[int]] = [[] for _ in range(n)]
    if n < 2 or coords.size == 0:
        return neigh
    pts = np.asarray(coords, dtype=float)
    if pts.shape != (n, 3):
        return neigh
    span = float(np.max(pts) - np.min(pts))
    if not math.isfinite(span) or span < 1e-4:
        return neigh
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    for i in range(n):
        for j in range(i + 1, n):
            limit = cutoffs.get(_pair_key(symbols[i], symbols[j]))
            if limit is None:
                continue
            if d[i, j] <= float(limit):
                neigh[i].append(j)
                neigh[j].append(i)
    return neigh


def _cdse_four_rings(
    symbols: Sequence[str],
    neigh: Sequence[Sequence[int]],
    *,
    cation: str,
    anion: str,
) -> List[frozenset]:
    """Cd–Se–Cd–Se diamonds only (two Se sharing two cation neighbours)."""

    se = [i for i, s in enumerate(symbols) if s == anion]
    rings: List[frozenset] = []
    for a in range(len(se)):
        for b in range(a + 1, len(se)):
            i, j = se[a], se[b]
            shared = sorted(
                x
                for x in set(neigh[i]) & set(neigh[j])
                if symbols[x] == cation
            )
            for u in range(len(shared)):
                for v in range(u + 1, len(shared)):
                    rings.append(frozenset((i, j, shared[u], shared[v])))
    return rings


def _cdse_six_rings(
    symbols: Sequence[str],
    neigh: Sequence[Sequence[int]],
    *,
    cation: str,
    anion: str,
) -> List[frozenset]:
    nodes = [i for i, s in enumerate(symbols) if s in (cation, anion)]
    found: set = set()
    allowed = {cation, anion}
    for start in nodes:
        stack: List[Tuple[int, Tuple[int, ...]]] = [(start, (start,))]
        while stack:
            cur, path = stack.pop()
            if len(path) == 6:
                if start in neigh[cur] and start == min(path):
                    els = [symbols[i] for i in path]
                    if els.count(cation) == 3 and els.count(anion) == 3:
                        found.add(frozenset(path))
                continue
            for nb in neigh[cur]:
                if nb in path or nb < start:
                    continue
                if {symbols[cur], symbols[nb]} != allowed:
                    continue
                stack.append((nb, path + (nb,)))
    return list(found)


def _share_hist(cycles: Sequence[frozenset]) -> Dict[int, int]:
    hist: Dict[int, int] = {}
    n = len(cycles)
    for i in range(n):
        for j in range(i + 1, n):
            s = len(cycles[i] & cycles[j])
            if s >= 2:
                hist[s] = hist.get(s, 0) + 1
    return hist


def _asphericity(symbols: Sequence[str], coords: np.ndarray, core: Sequence[str]) -> float:
    """Relative asphericity of the inorganic cloud.  0 = isotropic (sphere *or* tet)."""

    pts = np.asarray(coords, dtype=float)
    mask = np.array([s in core for s in symbols])
    if int(mask.sum()) < 4:
        return 0.0
    x = pts[mask] - pts[mask].mean(axis=0)
    tensor = x.T @ x / float(mask.sum())
    ev = np.sort(np.linalg.eigvalsh(tensor))
    tot = float(ev.sum())
    if tot <= 1e-12:
        return 0.0
    return float(ev[2] - 0.5 * (ev[0] + ev[1])) / tot


@dataclass(frozen=True)
class SoftDescriptors:
    """Local counts written to ``index.csv`` and used for ranking."""

    n4: int = 0
    n4_fused: int = 0
    f6_clean: int = 0
    f6_dirty: int = 0
    n6: int = 0
    n_term_se3cl: int = 0
    n_se1cl3: int = 0
    asphericity: float = 0.0

    def as_index_row(self) -> Dict[str, str]:
        return {
            "n4": str(self.n4),
            "n4_fused": str(self.n4_fused),
            "f6_clean": str(self.f6_clean),
            "f6_dirty": str(self.f6_dirty),
            "n6": str(self.n6),
            "n_term_se3cl": str(self.n_term_se3cl),
            "n_se1cl3": str(self.n_se1cl3),
            "asphericity": f"{self.asphericity:.4f}",
        }


# Dimensionless construction score (graph_rules.construction_score).
# Same motif family as the post-relax eV rank, but this is *graph* cost.
DEFAULT_CONSTRUCTION_SCORE: Dict[str, float] = {
    "n4": 15.0,
    "n4_fused": 30.0,
    "f6_clean": -4.0,
    "f6_dirty": 20.0,
    "terminal_se3cl": 20.0,
    "se1cl3": 5.0,
}


def construction_score(
    desc: SoftDescriptors,
    spec: Optional[NucleationSpec] = None,
) -> float:
    """Lower is better.  Used when choosing cores / decorations."""

    weights = dict(DEFAULT_CONSTRUCTION_SCORE)
    raw = {}
    if spec is not None:
        raw = getattr(spec.graph_rules, "construction_score", None) or {}
    if isinstance(raw, dict):
        for key, val in raw.items():
            try:
                weights[str(key)] = float(val)
            except (TypeError, ValueError):
                continue
    return float(
        weights.get("n4", 0.0) * desc.n4
        + weights.get("n4_fused", 0.0) * desc.n4_fused
        + weights.get("f6_clean", 0.0) * desc.f6_clean
        + weights.get("f6_dirty", 0.0) * desc.f6_dirty
        + weights.get("terminal_se3cl", 0.0) * desc.n_term_se3cl
        + weights.get("se1cl3", 0.0) * desc.n_se1cl3
    )


INDEX_FIELDS: Tuple[str, ...] = (
    "n4",
    "n4_fused",
    "n6",
    "f6_clean",
    "f6_dirty",
    "n_term_se3cl",
    "n_se1cl3",
    "asphericity",
    "soft_penalty_eV",
    "rank_score_eV",
)


def describe_from_neigh(
    symbols: Sequence[str],
    neigh: Sequence[Sequence[int]],
    spec: Optional[NucleationSpec] = None,
    *,
    asphericity: float = 0.0,
) -> SoftDescriptors:
    """Same counts as ``describe_structure``, from an explicit neighbour list."""

    cation, anion, ligand = _species(spec)
    fours = _cdse_four_rings(symbols, neigh, cation=cation, anion=anion)
    sixes = _cdse_six_rings(symbols, neigh, cation=cation, anion=anion)
    n4_fused = 0
    for i, ring in enumerate(fours):
        if any(len(ring & other) >= 2 for j, other in enumerate(fours) if i != j):
            n4_fused += 1
    share = _share_hist(sixes)
    f6_clean = int(share.get(2, 0) + share.get(3, 0))
    f6_dirty = int(sum(c for s, c in share.items() if s >= 4))
    n_term = 0
    n_se1cl3 = 0
    for i, sym in enumerate(symbols):
        if sym != cation:
            continue
        n_se = sum(1 for x in neigh[i] if symbols[x] == anion)
        cl_hosts = [x for x in neigh[i] if symbols[x] == ligand]
        n_cl = len(cl_hosts)
        if n_se == 1 and n_cl >= 3:
            n_se1cl3 += 1
        if n_se == 3 and n_cl >= 1:
            if any(
                sum(1 for y in neigh[cl] if symbols[y] == cation) <= 1
                for cl in cl_hosts
            ):
                n_term += 1
    return SoftDescriptors(
        n4=len(fours),
        n4_fused=n4_fused,
        f6_clean=f6_clean,
        f6_dirty=f6_dirty,
        n6=len(sixes),
        n_term_se3cl=n_term,
        n_se1cl3=n_se1cl3,
        asphericity=float(asphericity),
    )


def describe_structure(
    symbols: Sequence[str],
    coords: np.ndarray,
    spec: Optional[NucleationSpec] = None,
) -> SoftDescriptors:
    """Bond-graph descriptors of a relaxed XYZ."""

    cation, anion, _ligand = _species(spec)
    neigh = _neighbour_lists(symbols, coords, _cutoffs_from_spec(spec))
    asp = _asphericity(symbols, coords, (cation, anion))
    return describe_from_neigh(symbols, neigh, spec, asphericity=asp)


def describe_graph(
    symbols: Sequence[str],
    edges: Sequence[Tuple[int, int]],
    spec: Optional[NucleationSpec] = None,
) -> SoftDescriptors:
    """Descriptors of a construction graph (no coordinates)."""

    n = len(symbols)
    neigh: List[List[int]] = [[] for _ in range(n)]
    for a, b in edges:
        ia, ib = int(a), int(b)
        if 0 <= ia < n and 0 <= ib < n:
            neigh[ia].append(ib)
            neigh[ib].append(ia)
    return describe_from_neigh(symbols, neigh, spec, asphericity=0.0)


@dataclass(frozen=True)
class _Term:
    enabled: bool = True
    weight_eV: float = 0.0
    from_k: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

    def active(self, k: int) -> bool:
        return bool(self.enabled) and int(k) >= int(self.from_k)


def _term_from_raw(raw: Any, *, default_weight: float, default_from: int) -> _Term:
    if not isinstance(raw, dict):
        raw = {}
    extra = {
        key: raw[key]
        for key in raw
        if key not in {"enabled", "weight_eV", "from_k"}
    }
    return _Term(
        enabled=bool(raw.get("enabled", True)),
        weight_eV=float(raw.get("weight_eV", default_weight)),
        from_k=int(raw.get("from_k", default_from)),
        extra=extra,
    )


@dataclass(frozen=True)
class SoftRulesConfig:
    """Parsed ``soft_rules:`` block.  Code default is off; YAML turns it on."""

    enabled: bool = False
    diamond: _Term = field(
        default_factory=lambda: _Term(True, 0.15, 2, {"fused_weight_eV": 0.30})
    )
    f6: _Term = field(
        default_factory=lambda: _Term(
            True, -0.04, 3, {"dirty_weight_eV": 0.20}
        )
    )
    terminal_se3cl: _Term = field(
        default_factory=lambda: _Term(True, 0.20, 4)
    )
    se1cl3: _Term = field(default_factory=lambda: _Term(True, 0.05, 6))
    asphericity: _Term = field(
        default_factory=lambda: _Term(
            False, 1.0, 6, {"only_if_dirty_f6": True}
        )
    )

    @classmethod
    def from_raw(cls, raw: Any) -> "SoftRulesConfig":
        if not isinstance(raw, dict):
            return cls()
        diamond = _term_from_raw(
            raw.get("diamond"), default_weight=0.15, default_from=2
        )
        if "fused_weight_eV" not in diamond.extra:
            diamond = _Term(
                diamond.enabled,
                diamond.weight_eV,
                diamond.from_k,
                {**diamond.extra, "fused_weight_eV": 0.30},
            )
        f6 = _term_from_raw(raw.get("f6"), default_weight=-0.04, default_from=3)
        if "dirty_weight_eV" not in f6.extra:
            f6 = _Term(
                f6.enabled,
                f6.weight_eV,
                f6.from_k,
                {**f6.extra, "dirty_weight_eV": 0.20},
            )
        asp = _term_from_raw(
            raw.get("asphericity"), default_weight=1.0, default_from=6
        )
        if "only_if_dirty_f6" not in asp.extra:
            asp = _Term(
                asp.enabled,
                asp.weight_eV,
                asp.from_k,
                {**asp.extra, "only_if_dirty_f6": True},
            )
        return cls(
            enabled=bool(raw.get("enabled", False)),
            diamond=diamond,
            f6=f6,
            terminal_se3cl=_term_from_raw(
                raw.get("terminal_se3cl"), default_weight=0.20, default_from=4
            ),
            se1cl3=_term_from_raw(
                raw.get("se1cl3"), default_weight=0.05, default_from=6
            ),
            asphericity=asp,
        )

    def merged_with(self, overlay: Any) -> "SoftRulesConfig":
        if not isinstance(overlay, dict) or not overlay:
            return self
        base = self.to_raw()
        _deep_update(base, overlay)
        return SoftRulesConfig.from_raw(base)

    def to_raw(self) -> Dict[str, Any]:
        def pack(term: _Term) -> Dict[str, Any]:
            out = {
                "enabled": term.enabled,
                "weight_eV": term.weight_eV,
                "from_k": term.from_k,
            }
            out.update(term.extra)
            return out

        return {
            "enabled": self.enabled,
            "diamond": pack(self.diamond),
            "f6": pack(self.f6),
            "terminal_se3cl": pack(self.terminal_se3cl),
            "se1cl3": pack(self.se1cl3),
            "asphericity": pack(self.asphericity),
        }

    def penalty_eV(self, desc: SoftDescriptors, k: int) -> float:
        """Additive eV.  Lower (more negative) is better."""

        if not self.enabled:
            return 0.0
        k = int(k)
        pen = 0.0
        if self.diamond.active(k):
            fused_w = float(self.diamond.extra.get("fused_weight_eV", 0.30))
            pen += self.diamond.weight_eV * desc.n4
            pen += fused_w * desc.n4_fused
        if self.f6.active(k):
            dirty_w = float(self.f6.extra.get("dirty_weight_eV", 0.20))
            pen += self.f6.weight_eV * desc.f6_clean
            pen += dirty_w * desc.f6_dirty
        if self.terminal_se3cl.active(k):
            pen += self.terminal_se3cl.weight_eV * desc.n_term_se3cl
        if self.se1cl3.active(k):
            pen += self.se1cl3.weight_eV * desc.n_se1cl3
        if self.asphericity.active(k):
            only_dirty = bool(self.asphericity.extra.get("only_if_dirty_f6", True))
            if (not only_dirty) or desc.f6_dirty > 0:
                pen += self.asphericity.weight_eV * float(desc.asphericity)
        return float(pen)

    def rank_score_eV(
        self, energy_eV: float, desc: SoftDescriptors, k: int
    ) -> float:
        return float(energy_eV) + self.penalty_eV(desc, k)

def _deep_update(dst: Dict[str, Any], src: Mapping[str, Any]) -> None:
    for key, val in src.items():
        if isinstance(val, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], val)
        else:
            dst[key] = val


def apply_soft_columns(
    row: Dict[str, Any],
    *,
    symbols: Sequence[str],
    coords: np.ndarray,
    energy_eV: Optional[float],
    k: int,
    rules: SoftRulesConfig,
    spec: Optional[NucleationSpec] = None,
) -> SoftDescriptors:
    """Fill ``row`` with descriptor + score columns."""

    desc = describe_structure(symbols, coords, spec)
    row.update(desc.as_index_row())
    if energy_eV is None or not math.isfinite(float(energy_eV)):
        row["soft_penalty_eV"] = ""
        row["rank_score_eV"] = ""
        return desc
    pen = rules.penalty_eV(desc, k)
    row["soft_penalty_eV"] = f"{pen:.6f}"
    row["rank_score_eV"] = f"{float(energy_eV) + pen:.6f}"
    return desc
