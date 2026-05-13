# src/builder/nc_types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Any
import numpy as np

# Basic
@dataclass(frozen=True)
class Facet:
    h: int
    k: int
    l: int
    gamma: float
    termination: Optional[str] = None

Plane = Tuple[np.ndarray, float]

# Global passivation spec
@dataclass(frozen=True)
class PassivationSpec:
    ligand: str                 # anion ligand (legacy)
    surf_tol: float = 1.0
    cation_ligand: Optional[str] = None


@dataclass(frozen=True)
class FacetReconstructionSpec:
    """
    Minimal spec for polar-facet Lannoo reconstruction.

    Algorithm (Option C): strip all selected-facet ligands simultaneously →
    reconstruct each facet greedily (most-charged-first) → one final global
    charge-balance pass.

    YAML keys:
      facets:               list of {hkl: ...} entries to reconstruct
      cation_ligand:        optional symbol for cationic passivant (e.g. "NH3")
      cation_ligand_charge: formal charge of cation_ligand (required when set)
    """
    enabled: bool = False
    facets: Tuple[Tuple[int, int, int], ...] = ()
    cation_ligand: Optional[str] = None
    cation_ligand_charge: Optional[int] = None

# Stack building specs
@dataclass(frozen=True)
class BuildSpec:
    radius: Optional[float] = None          # core: absolute Å
    radius_scale: Optional[float] = None    # shell: multiplier on core radius
    size_unit_cells: Optional[Tuple[float, float, float]] = None
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
    shape_mode: str = "wulff"
    sphere_planes: int = 192
    align: Optional[AlignSpec] = None

# Unified config returned by parse_yaml_config
@dataclass(frozen=True)
class Config:
    mode: str                               # "single" | "stack"
    seeds: List[Facet]
    aspect: Tuple[float,float,float]
    shape_mode: str
    sphere_planes: int
    size_unit_cells: Optional[Tuple[float, float, float]]
    proper_only: bool
    pair_opposites: bool
    passivation: PassivationSpec
    charges: Dict[str,int]
    materials: List[MaterialSpec]
    twins: Optional[List[Dict[str, Any]]] = None
    construction_origin: Optional[Dict[str, Any]] = None
    facet_reconstruction: FacetReconstructionSpec = field(default_factory=FacetReconstructionSpec)
    experimental: Dict[str, Any] = field(default_factory=dict)
