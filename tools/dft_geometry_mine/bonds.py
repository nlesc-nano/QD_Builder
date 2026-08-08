"""Distance matrix, chemical graph, and homonuclear quarantine."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .xyz_io import Frame


def distance_matrix(coordinates: Sequence[Sequence[float]]) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def neighbor_signature(neighbor_symbols: Sequence[str]) -> str:
    """Canonical multiset label, e.g. Cl1Se2."""

    counts = Counter(neighbor_symbols)
    if not counts:
        return "empty"
    return "".join(f"{el}{counts[el]}" for el in sorted(counts))


@dataclass
class BondCutoffs:
    cd_se: float = 3.25
    cd_cl: float = 3.10
    cd_cd: float = 3.20
    se_se: float = 3.80
    cl_cl: float = 2.70


@dataclass
class GraphAnalysis:
    """Chemical graph + homonuclear flags for one frame."""

    symbols: Tuple[str, ...]
    distances: np.ndarray
    edges: List[Tuple[int, int, str, float]]  # i, j, pair_type, length
    neighbors: List[List[int]]
    degrees: List[int]
    n_cd_cd: int = 0
    n_se_se: int = 0
    n_cl_cl: int = 0
    cd_cd_pairs: List[Tuple[int, int, float]] = field(default_factory=list)
    se_se_pairs: List[Tuple[int, int, float]] = field(default_factory=list)
    cl_cl_pairs: List[Tuple[int, int, float]] = field(default_factory=list)
    same_species_samples: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def has_homonuclear_contact(self) -> bool:
        return self.n_cd_cd > 0 or self.n_se_se > 0 or self.n_cl_cl > 0

    def formula_counts(self) -> Dict[str, int]:
        return dict(Counter(self.symbols))


def pair_type(sym_a: str, sym_b: str) -> Optional[str]:
    pair = tuple(sorted((sym_a, sym_b)))
    if pair == ("Cd", "Se"):
        return "CdSe"
    if pair == ("Cd", "Cl"):
        return "CdCl"
    return None


def analyze_frame(frame: Frame, cutoffs: BondCutoffs) -> GraphAnalysis:
    """Build heteronuclear graph and scan same-species contacts."""

    symbols = frame.symbols
    n = len(symbols)
    dist = distance_matrix(frame.coordinates)
    neighbors: List[List[int]] = [[] for _ in range(n)]
    edges: List[Tuple[int, int, str, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            ptype = pair_type(symbols[i], symbols[j])
            if ptype is None:
                continue
            d = float(dist[i, j])
            limit = cutoffs.cd_se if ptype == "CdSe" else cutoffs.cd_cl
            if d <= limit:
                edges.append((i, j, ptype, d))
                neighbors[i].append(j)
                neighbors[j].append(i)

    degrees = [len(neigh) for neigh in neighbors]

    cd_cd: List[Tuple[int, int, float]] = []
    se_se: List[Tuple[int, int, float]] = []
    cl_cl: List[Tuple[int, int, float]] = []
    same_samples: List[Tuple[str, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            if symbols[i] != symbols[j]:
                continue
            d = float(dist[i, j])
            el = symbols[i]
            same_samples.append((el, d))
            if el == "Cd" and d < cutoffs.cd_cd:
                cd_cd.append((i, j, d))
            elif el == "Se" and d < cutoffs.se_se:
                se_se.append((i, j, d))
            elif el == "Cl" and d < cutoffs.cl_cl:
                cl_cl.append((i, j, d))

    return GraphAnalysis(
        symbols=symbols,
        distances=dist,
        edges=edges,
        neighbors=neighbors,
        degrees=degrees,
        n_cd_cd=len(cd_cd),
        n_se_se=len(se_se),
        n_cl_cl=len(cl_cl),
        cd_cd_pairs=cd_cd,
        se_se_pairs=se_se,
        cl_cl_pairs=cl_cl,
        same_species_samples=same_samples,
    )


def formula_matches_kp(
    counts: Mapping[str, int],
    k: Optional[int],
    p: Optional[int],
) -> Tuple[bool, str]:
    """Check Cd_(k+p) Se_k Cl_(2p) when k,p known."""

    if k is None or p is None:
        return True, "kp_unknown"
    n_cd = int(counts.get("Cd", 0))
    n_se = int(counts.get("Se", 0))
    n_cl = int(counts.get("Cl", 0))
    expect_cd = k + p
    expect_se = k
    expect_cl = 2 * p
    if n_cd == expect_cd and n_se == expect_se and n_cl == expect_cl:
        return True, "ok"
    return (
        False,
        f"got Cd{n_cd}Se{n_se}Cl{n_cl} expected Cd{expect_cd}Se{expect_se}Cl{expect_cl}",
    )
