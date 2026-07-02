# src/builder/nc_types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Any
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Neutral-ligand post-treatment types (declared first; used by PassivationSpec)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NeutralLigandPass:
    """
    A single pass in the neutral-ligand post-treatment.

    target      : 'cation' | 'anion' | 'both'
    smiles      : SMILES string of the neutral molecule (kept in neutral form).
    distribution: 'random' | 'segmented' | 'uniform'
    ratio       : fraction of eligible sites to passivate (0.0 – 1.0).
    """
    target: str
    smiles: str
    distribution: str = "random"
    ratio: float = 1.0
    target_count: int = 0


@dataclass(frozen=True)
class NeutralLigandPostTreatSpec:
    """
    Configuration for the optional neutral-ligand post-treatment step.
    Executed after charge-balance (Q=0) to passivate remaining structurally
    undercoordinated surface sites with neutral organic/inorganic ligands.
    """
    enabled: bool = False
    passes: Tuple[NeutralLigandPass, ...] = ()
    ff: str = "uff"               # RDKit force-field for 3-D embedding
    refinement_passes: int = 2     # steric rotational-scan passes
    sterics_mode: str = "vdw"      # heavy | all | vdw
    offset_out: float = 0.5        # extra Å along anchor direction
    seed: int = 1337


@dataclass(frozen=True)
class LigandExchangePass:
    """
    Replace native charge-balance ligands with charged ligands built from SMILES.

    replace     : native ligand symbol to replace, e.g. "Cl".
    charge      : final molecular ligand charge; currently supports -1 and +1.
    smiles      : one or more neutral precursor SMILES.
    distribution: 'random' | 'segmented' | 'uniform'
    ratio       : fraction of eligible native ligands to exchange.
    """
    replace: str
    charge: int
    smiles: Tuple[str, ...]
    distribution: str = "random"
    ratio: float = 1.0
    target_count: int = 0


@dataclass(frozen=True)
class AlloyingPass:
    """Replace inorganic core/surface atoms with another ion before ligand treatments."""
    replace: str
    replacement: str
    replacement_charge: int
    region: str = "both"              # surface | core | both
    distribution: str = "random"      # random | segmented | uniform
    ratio: float = 1.0
    target_count: int = 0


@dataclass(frozen=True)
class AlloyingPostTreatSpec:
    enabled: bool = False
    passes: Tuple[AlloyingPass, ...] = ()
    seed: int = 1337


@dataclass(frozen=True)
class LigandExchangePostTreatSpec:
    """
    Optional post-treatment ligand exchange.
    Runs after coordination-based post-treatments and substitutes already placed
    native charge-balance ligands without changing their validated virtual site.
    """
    enabled: bool = False
    passes: Tuple[LigandExchangePass, ...] = ()
    ff: str = "uff"
    refinement_passes: int = 2
    sterics_mode: str = "vdw"
    seed: int = 1337


@dataclass(frozen=True)
class ZTypeDisplacementPass:
    """
    Remove neutral Z-type surface units, e.g. CdCl2, CdSe, CsBr, PbBr2.

    cation     : positive surface species used as the group center.
    anion      : negative ligand/native species removed nearest to the cation.
    anion_count: number of anions removed per cation; if omitted by YAML/API,
                 it is derived from formal charges.
    target_count: exact number of groups to remove; if <= 0, ratio is used.
    distribution: 'random' | 'segmented' | 'uniform'
    ratio      : fraction of eligible cation-centered groups to displace.
    """
    cation: str
    anion: str
    anion_count: int = 0
    target_count: int = 0
    distribution: str = "random"
    ratio: float = 1.0


@dataclass(frozen=True)
class ZTypeDisplacementPostTreatSpec:
    """Optional post-treatment for removing neutral inorganic Z-type units."""
    enabled: bool = False
    passes: Tuple[ZTypeDisplacementPass, ...] = ()
    seed: int = 1337


@dataclass(frozen=True)
class SurfaceReconstructionSpec:
    """
    Simplified polar-surface reconstruction post-treatment.

    The step computes residual Lannoo-like facet charges after charge-balance
    passivation, sparsely swaps native anions on negative polar facets to the
    reconstruction ligand, and compensates each swap by adding one ligand to an
    available cation-rich polar site.
    """
    enabled: bool = False
    ligand: Optional[str] = None
    facets: Tuple[Tuple[int, int, int], ...] = ()
    auto_facets: bool = True
    target_reduction: float = 0.5
    min_separation: Optional[float] = None
    distribution: str = "fps"
    seed: int = 1337


@dataclass(frozen=True)
class PostTreatmentSpec:
    surface_reconstruction: SurfaceReconstructionSpec = field(default_factory=SurfaceReconstructionSpec)
    alloying: AlloyingPostTreatSpec = field(default_factory=AlloyingPostTreatSpec)
    z_type_displacement: ZTypeDisplacementPostTreatSpec = field(default_factory=ZTypeDisplacementPostTreatSpec)
    neutral_ligands: NeutralLigandPostTreatSpec = field(default_factory=NeutralLigandPostTreatSpec)
    ligand_exchange: LigandExchangePostTreatSpec = field(default_factory=LigandExchangePostTreatSpec)

# Basic
@dataclass(frozen=True)
class Facet:
    h: int
    k: int
    l: int
    gamma: float
    termination: Optional[str] = None
    scope: str = "family"

Plane = Tuple[np.ndarray, float]

# Global passivation spec
@dataclass(frozen=True)
class PassivationSpec:
    ligand: str                 # anion ligand (legacy)
    surf_tol: float = 1.0
    cation_ligand: Optional[str] = None
    prepass_mode: str = "standard"
    prepass_min_cn_terrace: int = 3
    prepass_min_cn_edge: int = 3
    prepass_min_cn_vertex: int = 3
    include_sublayer: bool = False
    neutral_ligands: NeutralLigandPostTreatSpec = field(
        default_factory=NeutralLigandPostTreatSpec
    )


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
    interface: Optional[Dict] = None

@dataclass(frozen=True)
class StackSpec:
    """Stack-mode options (core-shell builds)."""
    geometry_reference: str = "core"  # core | shortest | shell
    interface: str = "abrupt"          # abrupt | mixed
    mixing_width: float = 3.0          # in Angstroms

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
    post_treatment: PostTreatmentSpec = field(default_factory=PostTreatmentSpec)
    experimental: Dict[str, Any] = field(default_factory=dict)
    stack: StackSpec = field(default_factory=StackSpec)
