"""Local bond angles and motif classification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bonds import GraphAnalysis, neighbor_signature
from .xyz_io import Frame


def _angle_deg(
    origin: np.ndarray, a: np.ndarray, b: np.ndarray
) -> Optional[float]:
    va = a - origin
    vb = b - origin
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1.0e-12 or nb < 1.0e-12:
        return None
    cos = float(np.dot(va, vb) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


@dataclass
class AngleSample:
    element: str
    cn: int
    neighbor_signature: str
    angle_deg: float
    neighbor_pair: str  # e.g. Se-Se, Se-Cl, Cd-Cd
    role_signature: str = ""
    neighbor_role_pair: str = ""


@dataclass
class BondSample:
    pair_type: str  # CdSe | CdCl_terminal | CdCl_bridge
    length: float
    cn_cd: int
    cn_other: int  # Se or Cl degree
    k_window: str


def k_window(k: int | None) -> str:
    if k is None:
        return "k_unknown"
    if k <= 2:
        return "k_le_2"
    if k <= 4:
        return "k_3_4"
    return "k_ge_5"


def collect_angles(
    frame: Frame, graph: GraphAnalysis
) -> List[AngleSample]:
    samples: List[AngleSample] = []
    coords = np.asarray(frame.coordinates, dtype=float)
    for center, neigh in enumerate(graph.neighbors):
        if len(neigh) < 2:
            continue
        el = graph.symbols[center]
        cn = graph.degrees[center]
        sig = neighbor_signature(graph.symbols[j] for j in neigh)
        roles = [_neighbor_role(graph, center, neighbor) for neighbor in neigh]
        role_signature = "+".join(sorted(roles))
        for a_idx in range(len(neigh)):
            for b_idx in range(a_idx + 1, len(neigh)):
                i = neigh[a_idx]
                j = neigh[b_idx]
                ang = _angle_deg(coords[center], coords[i], coords[j])
                if ang is None:
                    continue
                pair = "-".join(
                    sorted((graph.symbols[i], graph.symbols[j]))
                )
                role_pair = "-".join(
                    sorted(
                        (
                            _neighbor_role(graph, center, i),
                            _neighbor_role(graph, center, j),
                        )
                    )
                )
                samples.append(
                    AngleSample(
                        element=el,
                        cn=cn,
                        neighbor_signature=sig,
                        angle_deg=ang,
                        neighbor_pair=pair,
                        role_signature=role_signature,
                        neighbor_role_pair=role_pair,
                    )
                )
    return samples


def _neighbor_role(graph: GraphAnalysis, center: int, neighbor: int) -> str:
    """Role label used by the molecular geometry pack."""

    symbol = graph.symbols[neighbor]
    if symbol != "Cl":
        return symbol
    degree = graph.degrees[neighbor]
    if degree <= 1:
        return "Cl_t"
    if degree == 2:
        other_hosts = [
            atom
            for atom in graph.neighbors[neighbor]
            if atom != center and graph.symbols[atom] == graph.symbols[center]
        ]
        shared = any(
            any(
                common != neighbor and graph.symbols[common] != symbol
                for common in set(graph.neighbors[center]).intersection(
                    graph.neighbors[other_host]
                )
            )
            for other_host in other_hosts
        )
        return "Cl_b2s" if shared else "Cl_b2n"
    return f"Cl_b{degree}"


def collect_bonds(
    graph: GraphAnalysis, *, k: Optional[int]
) -> List[BondSample]:
    window = k_window(k)
    samples: List[BondSample] = []
    for i, j, ptype, length in graph.edges:
        if ptype == "CdSe":
            cd = i if graph.symbols[i] == "Cd" else j
            se = j if cd == i else i
            samples.append(
                BondSample(
                    pair_type="CdSe",
                    length=length,
                    cn_cd=graph.degrees[cd],
                    cn_other=graph.degrees[se],
                    k_window=window,
                )
            )
        elif ptype == "CdCl":
            cd = i if graph.symbols[i] == "Cd" else j
            cl = j if cd == i else i
            role = (
                "CdCl_bridge"
                if graph.degrees[cl] >= 2
                else "CdCl_terminal"
            )
            samples.append(
                BondSample(
                    pair_type=role,
                    length=length,
                    cn_cd=graph.degrees[cd],
                    cn_other=graph.degrees[cl],
                    k_window=window,
                )
            )
    return samples


def motif_flags(
    frame: Frame,
    graph: GraphAnalysis,
    *,
    linear_threshold_deg: float = 160.0,
) -> Dict[str, object]:
    """Boolean / scalar motif summary for one structure."""

    coords = np.asarray(frame.coordinates, dtype=float)
    cd_cn2_angles: List[float] = []
    se_cn2_angles: List[float] = []
    cd_cl_cd_angles: List[float] = []

    for center, neigh in enumerate(graph.neighbors):
        el = graph.symbols[center]
        cn = graph.degrees[center]
        if cn < 2:
            continue
        if el == "Cd" and cn == 2 and len(neigh) == 2:
            ang = _angle_deg(
                coords[center], coords[neigh[0]], coords[neigh[1]]
            )
            if ang is not None:
                cd_cn2_angles.append(ang)
        if el == "Se" and cn == 2 and len(neigh) == 2:
            ang = _angle_deg(
                coords[center], coords[neigh[0]], coords[neigh[1]]
            )
            if ang is not None:
                se_cn2_angles.append(ang)
        if el == "Cl" and cn == 2 and len(neigh) == 2:
            # Cd–Cl–Cd bridge angle
            if all(graph.symbols[n] == "Cd" for n in neigh):
                ang = _angle_deg(
                    coords[center], coords[neigh[0]], coords[neigh[1]]
                )
                if ang is not None:
                    cd_cl_cd_angles.append(ang)

    max_cn: Dict[str, int] = {"Cd": 0, "Se": 0, "Cl": 0}
    min_cn: Dict[str, int] = {"Cd": 99, "Se": 99, "Cl": 99}
    for i, el in enumerate(graph.symbols):
        if el not in max_cn:
            continue
        d = graph.degrees[i]
        max_cn[el] = max(max_cn[el], d)
        min_cn[el] = min(min_cn[el], d)
    for el in list(min_cn):
        if min_cn[el] == 99:
            min_cn[el] = 0

    n_bridge_cl = sum(
        1
        for i, el in enumerate(graph.symbols)
        if el == "Cl" and graph.degrees[i] >= 2
    )
    n_terminal_cl = sum(
        1
        for i, el in enumerate(graph.symbols)
        if el == "Cl" and graph.degrees[i] == 1
    )

    cd_cn2_linear = (
        all(a >= linear_threshold_deg for a in cd_cn2_angles)
        if cd_cn2_angles
        else None
    )

    return {
        "n_cd_cn2": len(cd_cn2_angles),
        "cd_cn2_mean_angle": _mean(cd_cn2_angles),
        "cd_cn2_min_angle": min(cd_cn2_angles) if cd_cn2_angles else None,
        "cd_cn2_all_linear": cd_cn2_linear,
        "n_se_cn2": len(se_cn2_angles),
        "se_cn2_mean_angle": _mean(se_cn2_angles),
        "se_cn2_min_angle": min(se_cn2_angles) if se_cn2_angles else None,
        "n_cl_bridge_angles": len(cd_cl_cd_angles),
        "cd_cl_cd_mean_angle": _mean(cd_cl_cd_angles),
        "cd_cl_cd_min_angle": min(cd_cl_cd_angles) if cd_cl_cd_angles else None,
        "cd_cl_cd_max_angle": max(cd_cl_cd_angles) if cd_cl_cd_angles else None,
        "n_bridge_cl": n_bridge_cl,
        "n_terminal_cl": n_terminal_cl,
        "max_cn_cd": max_cn["Cd"],
        "max_cn_se": max_cn["Se"],
        "max_cn_cl": max_cn["Cl"],
        "min_cn_cd": min_cn["Cd"],
        "min_cn_se": min_cn["Se"],
        "min_cn_cl": min_cn["Cl"],
        "n_cdse_bonds": sum(1 for e in graph.edges if e[2] == "CdSe"),
        "n_cdcl_bonds": sum(1 for e in graph.edges if e[2] == "CdCl"),
    }


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))
