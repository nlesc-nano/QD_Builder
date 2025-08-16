# src/builder/nc_types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np

# Basic
@dataclass(frozen=True)
class Facet:
    h: int
    k: int
    l: int
    gamma: float

Plane = Tuple[np.ndarray, float]

# Global passivation spec
@dataclass(frozen=True)
class PassivationSpec:
    ligand: str
    surf_tol: float

# Stack building specs
@dataclass(frozen=True)
class BuildSpec:
    radius: Optional[float] = None          # core: absolute Å
    radius_scale: Optional[float] = None    # shell: multiplier on core radius
    interface_clearance: float = 1.6        # Å

@dataclass(frozen=True)
class StrainPolicy:
    type: str = "none"      # "none" | "uniform" | "biaxial"
    max_percent: float = 3.0

@dataclass(frozen=True)
class AlignSpec:
    core_facet: Optional[Tuple[int,int,int]] = None
    shell_facet: Optional[Tuple[int,int,int]] = None
    core_dir:   Optional[Tuple[int,int,int]] = None
    shell_dir:  Optional[Tuple[int,int,int]] = None
    strain:     StrainPolicy = StrainPolicy()

@dataclass(frozen=True)
class MaterialSpec:
    name: str
    cif: str
    seeds: List[Facet]
    aspect: Tuple[float,float,float]
    build: BuildSpec
    align: Optional[AlignSpec] = None

# Unified config returned by parse_yaml_config
@dataclass(frozen=True)
class Config:
    mode: str                               # "single" | "stack"
    seeds: List[Facet]
    aspect: Tuple[float,float,float]
    proper_only: bool
    pair_opposites: bool
    passivation: PassivationSpec
    charges: Dict[str,int]
    materials: List[MaterialSpec]


