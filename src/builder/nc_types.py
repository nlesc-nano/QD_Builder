# nanocrystal_builder/types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class Facet:
    h: int
    k: int
    l: int
    gamma: float

Plane = Tuple[NDArray[np.float64], float]  # (normal n̂ in Cartesian, offset d)


