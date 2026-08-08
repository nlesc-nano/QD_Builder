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
    target_symbol: Optional[str] = None


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
    smiles: Tuple[str, ...]
    charge: Optional[int] = None
    replace_charge: Optional[int] = None
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
    exchange_type: str = "displacement"
    smiles: Optional[str] = None


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
class NeutralExchangePass:
    cation: str
    anion: str
    anion_count: int = 0
    exchange_type: str = "mxn"   # "mxn" | "zwitterion" | "l_type"; "salt" accepted as legacy alias
    smiles: str = ""
    distribution: str = "random"
    ratio: float = 1.0
    target_count: int = 0


@dataclass(frozen=True)
class NeutralExchangePostTreatSpec:
    enabled: bool = False
    passes: Tuple[NeutralExchangePass, ...] = ()
    seed: int = 1337


@dataclass(frozen=True)
class PostTreatmentSpec:
    surface_reconstruction: SurfaceReconstructionSpec = field(default_factory=SurfaceReconstructionSpec)
    alloying: AlloyingPostTreatSpec = field(default_factory=AlloyingPostTreatSpec)
    z_type_displacement: ZTypeDisplacementPostTreatSpec = field(default_factory=ZTypeDisplacementPostTreatSpec)
    neutral_ligands: NeutralLigandPostTreatSpec = field(default_factory=NeutralLigandPostTreatSpec)
    ligand_exchange: LigandExchangePostTreatSpec = field(default_factory=LigandExchangePostTreatSpec)
    neutral_exchange: NeutralExchangePostTreatSpec = field(default_factory=NeutralExchangePostTreatSpec)

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


# Nucleation-map specifications
@dataclass(frozen=True)
class CoreMonomerSpec:
    """The two oppositely charged species that make one core monomer."""

    cation: str
    anion: str


@dataclass(frozen=True)
class PrecursorUnitSpec:
    """
    A surface-excess precursor unit.

    ``center`` occupies a vacant core-cation lattice site and ``ligand_count``
    atoms of ``ligand`` are placed on its remaining CIF-derived coordination
    slots.  For example, CdCl2 is represented by center="Cd", ligand="Cl",
    ligand_count=2.
    """

    center: str
    ligand: str
    ligand_count: int


@dataclass(frozen=True)
class NucleationBridgeRule:
    """One optional ligand bridge motif evaluated during graph screening."""

    ligand: str
    host: str
    shared_neighbor: str
    surface_angle_deg: float = 90.0
    min_bridged_host_cn: int = 1
    """Lowest final coordination both cations of a bridge may have.

    ``1`` imposes no restriction, the historical behaviour.  ``3`` encodes the
    observation that a two-coordinate cation does not participate in a bridge: it
    prefers to stay linear with its own ligand.

    Evidence on CdSe/CdCl2, from three DFT relaxations:

    * ``k=1, p=1`` -- the only bridge available leaves one Cd at CN 2, and the
      relaxed structure instead keeps both Cd at CN 2 with terminal Cl and a
      linear Se-Cd-Cl.  The bridge does not form.
    * ``k=1, p=2`` -- three bridges whose cations are all at CN 3 or 4 hold, and
      the structure stays close to its construction geometry.
    * ``k=2, p=0`` -- a CN-2 Cd relaxes to a pseudo-linear Se-Cd-Se.

    **This is deliberately a constraint on the finished structure, not on the
    construction step.** An earlier attempt phrased it as a minimum coordination
    for the *donor* -- the cation lending the ligand -- and it silently did
    nothing: the same structure is reachable from a different ligand arrangement
    whose donor is three-coordinate, and the route-merging DAG rebuilt it there.
    Any rule expressed on how a structure was built leaks in the same way, because
    isomorphic results from different routes are merged by design.

    All three observations are at small ``k`` on one system, so the default leaves
    the rule off.  Above ``k=2`` every currently retained structure contains a
    bridge with a CN-2 cation, so enabling it replaces those bins wholesale --
    more than the present evidence supports.
    """


@dataclass(frozen=True)
class NucleationPairRule:
    """Bond permission and geometric limits for one unordered element pair."""

    elements: Tuple[str, str]
    bond_allowed: bool
    bond_max_distance: Optional[float] = None
    min_distance: Optional[float] = None


@dataclass(frozen=True)
class NucleationGraphRules:
    """Chemical graph constraints used only during nucleation enumeration.

    ``allowed_bonds`` contains canonical, unordered element pairs. Native
    enumeration uses the core monomer's rigid nearest-neighbour distance and
    CIF-derived coordination-site directions; retained-only surface templates
    are configured separately.
    """

    min_cn: Dict[str, int]
    max_cn: Dict[str, int]
    allowed_bonds: Tuple[Tuple[str, str], ...]
    bridge_rules: Tuple[NucleationBridgeRule, ...] = ()
    pair_rules: Dict[str, NucleationPairRule] = field(default_factory=dict)
    allowed_neighbor_signatures: Dict[str, Tuple[str, ...]] = field(
        default_factory=dict
    )
    max_shared_ligands_per_host_pair: int = 0
    # Optional hard Cd--Cd host separation cap for a bridging ligand in
    # skeleton_bridge_first.  ``None`` keeps the historical conservative
    # span derived from the largest Cd--Cl bridge radii.
    bridge_cd_cd_max_distance: Optional[float] = None
    # Shortest cycle allowed in the alternating subgraph of an element pair,
    # keyed like ``"Cd-Se"``.  A Cd2Se2 four-ring is not a coordination or
    # distance violation -- it is simply a motif the reference chemistry does
    # not contain -- so nothing else in these rules can exclude it.
    min_ring_size: Dict[str, int] = field(default_factory=dict)
    # Molecular decoration defaults (finished graph, not construction route).
    # min_bridged_host_cn: both hosts of any μ2/μ3 Cl must have final CN ≥ this
    # (nucleation default 3; molecular pack enables it explicitly).
    min_bridged_host_cn: int = 1
    # Mono-Se Cd with exactly two Cl must not keep both as terminals (DFT:
    # Cl2Se1 is (1μ+1t) or (2μ), never (2t)).
    forbid_mono_se_dual_terminal: bool = False
    # Optional DFT-collapse filters (default off).  Promote only after
    # start→final graph statistics justify them; they do not affect the
    # exhaustive construction path unless explicitly enabled in the pack.
    reject_closable_terminal_cd2: bool = False
    closable_terminal_cd2_distance: float = 3.50
    require_bridge_maximal: bool = False
    # Forbidden final Cd–Se coordination pairs as (cn_cd, cn_se), e.g. (2, 5).
    forbid_cdse_cn_pairs: Tuple[Tuple[int, int], ...] = ()
    #: anion motif pairs, as (element, cn, element, cn), that may not
    #: share two cations: the cation-cation separation each one demands
    #: cannot be reconciled inside their own angle bands
    forbid_shared_cd_pair: Tuple[Tuple[str, int, str, int], ...] = ()
    # Molecular Cl decoration strategy:
    #   "graph_multiset" — abstract μ1/μ2/μ3 host multisets (default)
    #   "skeleton_bridge_first" — 3D skeleton → distance-gated bridges → terminals
    #   "motif_bridge_first" — graph-only bridges → residual terminal fill → motif 3D
    #   "tet_sites" / "pack_sites" — experimental site enumerators
    decoration_mode: str = "graph_multiset"
    # For skeleton_bridge_first only: p=1 may place terminals on any legal
    # Cd after bridge candidates are considered, without lowest-CN ordering.
    bridge_first_p1_terminal_policy: str = "unrestricted"
    # Experimental motif bridge-first controls.  Bridges are enumerated before
    # terminal fills and the empirical Cd bridge load is bounded.
    bridge_first_hard_max_bridges_per_cd: int = 2
    #: Soft bridge load per Cd.  Loads above this are penalised in the beam
    #: ranking (``over_pref``), which is ordered *above* the bridge-count term
    #: -- so this, not the hard cap, is what actually limits bridge load.
    #: Relaxed structures show 3 bridging Cl on one Cd is common, so raising
    #: this to 3 is the knob for reaching them.
    bridge_first_prefer_bridges_per_cd: int = 2
    #: Rings a graph MUST contain, e.g.
    #: ``[{"size": 8, "min_count": 1, "from_k": 4}]``.  The inverse of
    #: ``min_ring_size``: this gate is what bounds the combinatorics at large
    #: k, where exhaustive enumeration stops being affordable.
    required_rings: Tuple[Dict[str, int], ...] = ()
    # When true, a Cd attached to a μ3 Cl cap may not participate in another
    # μ2/μ3 Cl bridge.  Terminal Cl remains allowed on that Cd.
    forbid_mu3_host_bridge_overlap: bool = False
    # Passivation: if k >= passivate_min_cd_cn_k_ge and p >= passivate_min_cd_cn_p_ge,
    # require every Cd final CN >= passivate_min_cd_cn (kill leftover CN2 at high p).
    passivate_min_cd_cn_p_ge: int = 100  # 100 = disabled; fixed-geom needs Cd2
    passivate_min_cd_cn_k_ge: int = 2
    passivate_min_cd_cn: int = 3
    # When composition can host min stable 6-ring pattern Cd[3,3,4] Se[3,3,3],
    # try closed Cd3Se3 seed first. If passivation accepts nothing, optionally
    # fall back to free (open) skeletons (still no Cd–Se 4-rings).
    ring_first_when_pattern_possible: bool = False
    ring_first_fallback_to_open: bool = True
    # When true, try fused-2 seeds (all fusion modes) before 1-ring / free.
    multi_ring_ladder: bool = True
    ring_min_pattern_cd: Tuple[int, ...] = (3, 3, 4)
    ring_min_pattern_se: Tuple[int, ...] = (3, 3, 3)


@dataclass(frozen=True)
class NucleationGeometryRules:
    """Retained-only local geometry templates keyed by element and CN.

    ``by_cn`` stores explicit coordination-number templates such as
    ``{"Cd": {2: "linear", 3: "trigonal_planar", 4: "tetrahedral"}}``.
    ``all_cn`` supplies a species-wide template, used for tetrahedral anions
    and ligands in the CdSe/CdCl2 example.
    """

    by_cn: Dict[str, Dict[int, str]]
    all_cn: Dict[str, str]

    def template_for(self, symbol: str, coordination: int) -> Optional[str]:
        """Return the configured template for one local environment."""

        return self.by_cn.get(symbol, {}).get(
            coordination, self.all_cn.get(symbol)
        )


@dataclass(frozen=True)
class NucleationSpec:
    """Configuration for bottom-up ``(k, p)`` nucleation-map generation."""

    cif: str
    charges: Dict[str, int]
    core: CoreMonomerSpec
    precursor: PrecursorUnitSpec
    kmax: int
    graph_rules: NucleationGraphRules
    geometry_rules: NucleationGeometryRules
    geometry_pack: Optional[str] = None
    site_tolerance: float = 0.20
    bond_count_scope: str = "all"
    """Which bonds the selection score counts at its bond-count component.

    ``"all"`` counts every bond, the historical behaviour, so forming a bridge
    raises a structure's rank.  ``"skeleton"`` counts only cation-anion bonds, so
    ligand bonds and bridges stop buying rank and structures are ranked by the
    inorganic framework alone.

    Measured on CdSe/CdCl2 at k<=2, switching to ``"skeleton"`` barely moves the
    total retained count (18 -> 19) but does change individual bins, because it
    changes which score layer wins and layers differ in how many structures tie
    at them.
    """

    mode: str = "exact"
    """``"exact"`` enumerates every symmetry-distinct ligand arrangement per bin.

    ``"guided"`` places ligand shells per skeleton using passivation ordering
    (bridging sites first, then undercoordinated cations).  By default that is
    **one** shell; raise ``shells_per_skeleton`` / ``shell_score_layers`` to keep
    a coordination-score band of passivation isomers on the **same** skeleton
    without switching to full exact mode for the whole map.
    """

    shells_per_skeleton: int = 1
    """Max distinct ligand shells kept **per inorganic skeleton** in guided mode.

    ``1`` is historical guided (one passivation-ordered shell + its bridge
    variants that survive cross-skeleton merging).  ``N>1`` ranks candidate
    shells on that skeleton by coordination score and keeps up to ``N`` after
    applying ``shell_score_layers`` / ``shells_per_score_layer``.  Ignored when
    ``mode="exact"`` (exact still enumerates all shells).
    """

    shell_score_layers: int = 1
    """How many top **distinct** coordination scores to walk for guided shells.

    Scores are sorted high→low.  ``1`` = only the best score (e.g. max 20).
    ``2`` = best and next distinct score (20 then 19), ``3`` = 20, 19, 18, …
    How many shells are taken **at each** of those scores is
    ``shells_per_score_layer``.  Global hard cap remains ``shells_per_skeleton``.
    """

    shells_per_score_layer: int = 0
    """How many shells to keep **within each** score layer (per skeleton).

    ``0`` (default) = keep **all** candidates that share that coordination
    score (historical multi-shell band: whole layer).

    ``1`` = keep only the **top** shell at that score (stable tie-break), then
    the top at the next score, etc.  Example with ``shell_score_layers: 3`` and
    ``shells_per_score_layer: 1``: best among score 20, best among 19, best
    among 18.

    ``N>1`` = top N within each visited score layer (then ``shells_per_skeleton``
    still caps the total).
    """

    shell_enum_max_assignments: int = 10000
    """Safety bound for guided multi-shell: if the theoretical site-subset
    count ``C(n_sites, n_ligands)`` exceeds this, only the greedy guided shell
    (plus bridges) is used for that skeleton instead of full orbit enumeration.

    Prevents accidental combinatorial blow-ups at large free-site counts while
    still allowing multi-shell when the shell space is small.
    """

    exact_through_k: int = 3
    """Last ``k`` whose row is grown from *every* unique skeleton.

    Distinct skeletons grow exponentially in ``k`` -- measured 4, 14, 243 for
    ``k`` = 1, 2, 3 -- because the count is essentially the number of distinct
    lattice animals on the cation sublattice, and most of them are open or
    branched arrangements rather than compact dots.  Retaining those is right for
    a small nucleus and wrong once the interior has become bulk-like, so above
    this threshold only the cores of *retained* structures propagate to ``k+1``.

    The selection score already ranks on minimum-CN compliance and then bond
    count, and for fixed ``(k, p)`` maximising bond count is maximising
    compactness -- so this is the existing ranking applied one level earlier, not
    a new chemical parameter.  It does reintroduce path dependence: a structure
    reachable only through an intermediate that ranked poorly is lost, and no
    later work recovers it.  ``registry.json`` discloses that.
    """

    core_growth_policy: str = "all"
    """How candidate core monomers are filtered when growing ``k -> k+1``.

    ``"all"`` keeps every connected monomer placement (historical default).

    ``"max_bonds"`` keeps, per parent skeleton, only children that maximise the
    number of new chemical edges created by the added cation-anion pair
    (steepest-ascent compact growth).

    ``"compact_ring"`` applies ``max_bonds`` first, then among survivors prefers
    children that create at least one new 6-cycle (smallest ring on the
    zincblende nearest-neighbour graph).  If any ring-closing child exists at
    the max-bond tier, non-closing ones are dropped.

    Non-``all`` policies reintroduce path dependence and are disclosed in the
    completeness block.  They apply only when the destination ``k`` is at least
    ``compact_from_k`` so small-nucleus maps stay complete by default.
    """

    compact_from_k: int = 3
    """First destination ``k`` at which ``core_growth_policy`` is applied.

    Growth steps that build skeletons with ``k_new < compact_from_k`` always use
    full ``all`` placement.  Default 3 leaves ``k<=2`` rows complete (baseline
    lock) while compact growth can tame ``k>=3``.
    """

    inorganic_ring_length: int = 6
    """Preferred inorganic cycle length for compact / fused growth and beams.

    Zincblende and wurtzite tetrahedral lattices use 6 (chairs).  Rock-salt
    packs typically set 4.  Used by ``core_growth_policy: compact_ring``,
    ``fused_chair_*``, and ``p_beam_rank`` ring modes.
    """

    geometry_defaults: str = "zb_tetrahedral"
    """Whether YAML geometry_rules are seeded with zincblende-like defaults.

    ``"zb_tetrahedral"`` (historical): cation CN2 linear / CN3 trigonal /
    CN4 tetrahedral and anion/ligand tetrahedral when omitted.
    ``"none"``: only templates listed under ``geometry_rules`` apply (use for
    rock-salt and other non-ZB packs).
    """

    terminal_motifs: str = "zb_mx2"
    """Post-passivation terminal-ligand placement table for surface coords.

    ``"zb_mx2"`` (historical): CdSe/CdCl2-style linear, σ_d, bisector, C2v
    patterns after bridges are fixed.  Construction for the next ``k`` still
    uses lattice virtual sites.
    ``"none"``: skip that table (recommended for rock-salt and uncalibrated
    systems until a motif pack exists).
    """

    passivation_ring_policy: str = "none"
    """How ligand-containing rings influence bin retention after placement.

    Placement itself is unchanged (exact orbits or guided shell, then latent
    bridges under max_cn and ``min_bridged_host_cn``).  Ring policy only ranks
    or filters among chemically legal finished graphs.

    ``"none"`` -- historical behaviour; ring counts are still reported.

    ``"prefer_cl_rings"`` / ``"prefer_ligand_rings"`` -- within the winning
    coordination/surface layer, keep only structures that maximise
    ligand-containing rings of lengths in ``ring_lengths``.

    ``"require_cl_rings"`` / ``"require_ligand_rings"`` -- if any surface-valid
    winner has at least one targeted ligand ring, drop those with zero.
    Stronger; disclosed as incomplete.
    """

    ring_lengths: Tuple[int, ...] = (4, 6)
    """Cycle lengths counted for ``passivation_ring_policy`` (Cl-containing).

    Length 4 captures the Cd–Se–Cd–Cl rhombus from latent bridges (still subject
    to ``min_bridged_host_cn``).  Length 6 captures larger Cl-including loops.
    """

    discarded_through_k: int = 2
    """Highest ``k`` for which full discarded isomers are kept and written.

    Above this only ``discarded_counts`` are stored (saves disk).  Raise to 3+
    when exporting open-ring DFT candidates: retained winners are almost always
    compact (many Cd–Se 6-rings), so truly open skeletons live in the discarded
    set.
    """

    p_skeleton_beam: int = 0
    """Max unique skeletons kept at each (k, p) after merge (0 = unlimited).

    Caps the p-axis explosion: ligand/bridge work and growth to p+1 only see the
    top-B skeletons by ``p_beam_rank``.  Disclosed when it drops candidates.
    """

    p_beam_from_k: int = 4
    """Apply ``p_skeleton_beam`` only for rows with ``k >= p_beam_from_k``."""

    p_beam_rank: str = "fused_rings"
    """How to rank skeletons for the p-beam.

    ``bonds`` -- inorganic edge count
    ``six_rings`` -- number of inorganic 6-cycles (chairs)
    ``fused_rings`` -- edge-sharing pairs of 6-cycles, then six_rings, then bonds
    """

    fused_chair_from_k: int = 6
    """Destination k from which fused-chair bias applies on core growth (0=off)."""

    fused_chair_mode: str = "off"
    """``off`` | ``rank`` | ``prefer_positive``.

    After compact_ring / max_bonds, among remaining k→k+1 children:
    ``rank`` keeps the highest fused-chair count tier;
    ``prefer_positive`` drops zero-fused children when any fused child exists.
    """

    retain_score_layers: int = 1
    """How many top distinct coordination-score layers are retained per bin.

    ``1`` is historical winners-only.  ``2+`` keeps the next-best layers as
    retained so their cores can feed k→k+1 growth (soft lineage band).  Ranking
    still prefers higher scores; this only widens what is labelled retained and
    what propagates under ``exact_through_k`` narrowing.
    """

    retain_max_per_bin: int = 0
    """Hard cap on retained structures **per ``(k, p)`` bin** after score layers.

    Applied independently at every ``k`` and every ``p`` (not a global budget
    for the whole map).  So ``32`` means up to 32 retained isomers in
    ``structures/k003/p005/retained/``, and separately up to 32 in
    ``k003/p006/retained/``, etc.  Intermediate ``p`` can hit the cap more often
    because ligand isomer counts peak there; early ``k`` often stays well under.

    ``0`` means unlimited.  Combined with ``retain_score_layers`` this bounds
    propagation cost when the soft band would otherwise explode.

    Continuous decorated growth cost scales roughly as
    ``(# retained parents) × (# packages) × (# attachments) × (1+max_shed)``,
    so prefer smaller caps (e.g. 6–12) once continuous_decoration is on.
    """

    growth_max_parents_per_bin: int = 0
    """Max retained parents used for k→k+1 growth from each parent p-bin.

    ``0`` = use every retained structure in the bin.  ``N>0`` keeps only the
    top-N by coordination score as growth sources.  Export still keeps the full
    retained set; this only throttles the growth fan-out (critical for continuous
    decoration at k≥3).
    """

    core_growth_occupation: str = "bare"
    """How tetrahedral sites are treated when growing ``k → k+1`` from retained.

    ``"bare"`` (historical): strip ligands, then place the monomer on any
    vacant CIF direction of the ligand-free skeleton.  Cl that occupied a
    virtual site no longer blocks attachment.

    ``"decorated"``: place the monomer on the passivated parent.  Ligands and
    precursor atoms occupy sites and block anion/cation attachment there.
    With ``continuous_decoration`` (default when decorated in guided maps), the
    ligand shell is **kept** across k and only Δp ligands are placed on free
    sites; without it, children are stripped and re-passivated (historical).
    Applies when growth is narrowed to retained sources
    (``k >= exact_through_k``).
    """

    continuous_decoration: bool = False
    """Keep the ligand shell across k→k+1 when using decorated growth.

    ``False`` (historical): strip Cl after monomer attach; rebuild shell at k+1.
    ``True``: shed only requested CdCl2 units, attach CdSe on free sites, add
    monomer-package ligands only on free sites, and ladder p by free-site
    precursor adds.  Soft-retained ligand isomers act as ligand-diffusion
    samples on the same skeleton family.
    """

    monomer_p_values: Tuple[int, ...] = ()
    """Precursor counts ``p_m`` on the **added** monomer package at k→k+1.

    Empty ``()`` means a single bare core add with ``p_m = 0`` (historical:
    inject children at parent p minus shed only).  Non-empty, e.g. ``(0, 1, 2)``,
    means each retained parent is grown once per package: nominal product
    ``p0 = p_parent - shed + p_m``.  This is the building-block picture:
    solution seeds are ``(1, p_m)``, not a permanent parent-p filter.
    """

    seed_p: Optional[int] = None
    """Optional centre used to **derive** ``monomer_p_values`` when that list is empty.

    With ``parent_p_mode: all_retained`` (default), ``seed_p`` does **not**
    restrict which clusters may grow.  With ``parent_p_mode: seed_band``, only
    parents with ``|p - seed_p| <= seed_p_window`` feed core growth (legacy).
    """

    seed_p_window: int = 0
    """Half-width around ``seed_p`` for package derivation and optional seed_band parents."""

    parent_p_mode: str = "all_retained"
    """Which parent p-bins feed k→k+1 core growth.

    ``all_retained`` -- every retained (or skeleton) p-bin may grow (building-block).
    ``seed_band`` -- only ``|p - seed_p| <= seed_p_window`` (requires seed_p).
    """

    p_ladder_mode: str = "inherited_plus"
    """How far the p-ladder at destination k may extend.

    ``inherited_plus`` -- historical: ``p_max = max(inherited p) + k_growth_max_add``
    (``max_add < 0`` → full capacity).  Ladder starts from low p as before.

    ``product_window`` -- building-block: after core growth injects keys ``P_inj``,
    only process ``p`` in
    ``[min(P_inj) - max_shed, max(P_inj) + max_add]`` (capacity still binds).
    Use with non-empty ``monomer_p_values`` so ``p0 = p + p_m`` is meaningful.
    """

    k_growth_max_shed: int = 0
    """Max precursor units removable when preparing a growth parent (free sites).

    Also, under ``p_ladder_mode: product_window``, how far the ladder may extend
    **below** the lowest injected ``p0``.  ``0`` = no fixed shed (unless
    ``p_surf_beta > 0`` supplies a surface-scaled shed law).  With the default
    no-surface configuration, zero means **unlimited up to the parent p**:
    shedding is constrained by Se capacity rather than by an arbitrary fixed
    count.  Set a positive value to restore an explicit fixed shedding cap.

    When ``p_surf_beta > 0``, effective shed is
    ``min(p, floor(shed_alpha * p_surf(k)), k_growth_max_shed or ∞)`` with
    ``p_surf(k) = floor(p_surf_beta * k^(2/3))`` (scenario A).
    """

    k_growth_max_add: int = -1
    """How far the p-ladder may extend **above** injected product p.

    ``-1``: unlimited up to anion capacity (or ``p_surf`` when ``p_surf_beta > 0``).
    ``N>=0`` with ``inherited_plus``: ``p_max = max(inherited) + N``.
    ``N>=0`` with ``product_window``: ``p_max = max(injected p0) + N``.

    When ``p_surf_beta > 0``, the ladder is also capped by spherical surface
    capacity ``p_surf(k)`` so redecoration does not run to the CN ceiling ``3k``.
    """

    p_surf_beta: float = 0.0
    """Optional spherical surface prefactor for map bounds.

    ``0`` (default) disables the surface law; shedding then explores all
    complete CdCl2 packages allowed by the Se-capacity filter.  A positive
    ``β`` explicitly opts into the surface-scaled shedding/lattice ceiling,
    with ``p_surf(k)=floor(β k^(2/3))``.  ``β ≈ 3`` matches scenario A
    calibrated to faceted NC headroom (between conservative 1.5 and large-NC
    ~4.5).
    """

    shed_alpha: float = 1.0
    """With ``p_surf_beta > 0``: ``s_max = min(p, floor(shed_alpha * p_surf(k)))``.

    ``1.0`` = shed at most one surface-worth of excess (scenario A).
    """

    require_inorganic_connected: bool = False
    """If true, the cation–anion (Cd–Se) subgraph must be a single component.

    Forbids halide-only links between separate inorganic fragments (H1).
    Default false keeps historical lattice maps unchanged; set true for
    molecular construction.
    """

    bridges_per_cd_pair: int = 0
    """Max ligand bridges between the same unordered host–host pair (H6).

    ``0`` = unlimited (historical).  ``2`` matches clean DFT (double bridges
    allowed, triple never seen).  Keyword in YAML: ``bridges_per_Cd_pair``.
    """

    enforce_min_cn: bool = False
    """If true, every atom degree must be >= graph_rules.min_cn[symbol] (H4/H7).

    Default false: min_cn remains a ranking preference during lattice growth
    (seed CdSe is CN1).  Set true for finished molecular graphs so Cd/Se
    cannot sit at CN1 when min_cn is 2.
    """

    contact_min_distance: Dict[str, float] = field(default_factory=dict)
    """Minimum allowed distances (Å) for **non-bonded / forbidden** pairs.

    Keys are sorted element pairs ``"Cd-Cd"``, ``"Se-Se"``, ``"Cl-Cl"``,
    ``"Cl-Se"``.  After molecular embedding, any pair closer than the cutoff
    causes the isomer to be discarded.  Empty dict uses built-in molecular
    defaults (see ``molecular_rules.DEFAULT_CONTACT_MIN_DISTANCE``).
    """

    @property
    def ligand_max_cn(self) -> int:
        """Compatibility view of the ligand limit in ``graph_rules``."""

        return self.graph_rules.max_cn[self.precursor.ligand]
