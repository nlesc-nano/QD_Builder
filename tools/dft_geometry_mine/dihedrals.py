"""Improper planarity and ordinary proper-torsion evidence from DFT frames."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Sequence

import numpy as np

from .bonds import GraphAnalysis, neighbor_signature
from .xyz_io import Frame


@dataclass(frozen=True)
class ImproperDihedralSample:
    element: str
    cn: int
    neighbor_signature: str
    improper_deg: float


@dataclass(frozen=True)
class ProperDihedralSample:
    atom_signature: str
    dihedral_deg: float


def _signed_dihedral_deg(
    p0: Sequence[float],
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
) -> Optional[float]:
    points = [np.asarray(point, dtype=float) for point in (p0, p1, p2, p3)]
    b0 = points[0] - points[1]
    b1 = points[2] - points[1]
    b2 = points[3] - points[2]
    norm = float(np.linalg.norm(b1))
    if norm < 1.0e-12:
        return None
    axis = b1 / norm
    v = b0 - float(np.dot(b0, axis)) * axis
    w = b2 - float(np.dot(b2, axis)) * axis
    if np.linalg.norm(v) < 1.0e-12 or np.linalg.norm(w) < 1.0e-12:
        return None
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(axis, v), w))
    return float(np.degrees(np.arctan2(y, x)))


def collect_improper_dihedrals(
    frame: Frame, graph: GraphAnalysis
) -> List[ImproperDihedralSample]:
    """Collect one permutation-stable planarity deviation for every CN3 center."""

    coords = np.asarray(frame.coordinates, dtype=float)
    samples: List[ImproperDihedralSample] = []
    for center, neighbors in enumerate(graph.neighbors):
        if len(neighbors) != 3:
            continue
        ordered = sorted(neighbors)
        value = _signed_dihedral_deg(
            coords[ordered[0]],
            coords[center],
            coords[ordered[1]],
            coords[ordered[2]],
        )
        if value is None:
            continue
        # Both 0 and 180 degrees are coplanar in an ordinary torsion convention.
        deviation = min(abs(value), abs(180.0 - abs(value)))
        samples.append(
            ImproperDihedralSample(
                element=graph.symbols[center],
                cn=3,
                neighbor_signature=neighbor_signature(
                    graph.symbols[index] for index in ordered
                ),
                improper_deg=float(deviation),
            )
        )
    return samples


def collect_proper_dihedrals(
    frame: Frame, graph: GraphAnalysis
) -> List[ProperDihedralSample]:
    """Collect every unique bonded path of four atoms, modulo path reversal."""

    coords = np.asarray(frame.coordinates, dtype=float)
    samples: List[ProperDihedralSample] = []
    seen = set()
    for middle_left, middle_right, _pair_type, _length in graph.edges:
        for outer_left in graph.neighbors[middle_left]:
            if outer_left == middle_right:
                continue
            for outer_right in graph.neighbors[middle_right]:
                if outer_right in {middle_left, outer_left}:
                    continue
                path = (outer_left, middle_left, middle_right, outer_right)
                canonical = min(path, tuple(reversed(path)))
                if canonical in seen:
                    continue
                seen.add(canonical)
                value = _signed_dihedral_deg(*(coords[index] for index in path))
                if value is None:
                    continue
                symbols = tuple(graph.symbols[index] for index in path)
                reverse_symbols = tuple(reversed(symbols))
                signature = "-".join(min(symbols, reverse_symbols))
                samples.append(
                    ProperDihedralSample(
                        atom_signature=signature,
                        dihedral_deg=float(value),
                    )
                )
    return samples
