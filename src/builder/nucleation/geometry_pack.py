"""Load and query DFT-derived molecular geometry packs."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml


@dataclass(frozen=True)
class BondLengthEntry:
    pair: str
    cn_cd: int
    cn_other: int
    r_A: float
    #: half-width of the acceptable band; ``None`` falls back to the pack-wide
    #: audit tolerance, so packs written before per-bond bands still load.
    tol_A: Optional[float] = None


@dataclass(frozen=True)
class CdSe6RingPattern:
    """DFT-style endocyclic geometry for a full-CN Cd₃Se₃ 6-ring (not bare Cd2Se2)."""

    name: str
    bond_cdse_A: float
    angle_at_cd_deg: float
    angle_at_se_deg: float
    # Frame CN templates for the three ring Cd / three ring Se (full pattern).
    cd_cn: Tuple[int, int, int] = (3, 3, 4)
    se_cn: Tuple[int, int, int] = (3, 3, 3)


@dataclass
class GeometryPack:
    """Executable graph and geometry rules for lattice-free embedding."""

    schema_version: int
    name: str
    bonds: Tuple[BondLengthEntry, ...]
    graph_rules: Dict[str, Any]
    angles: Dict[str, Any]
    dihedrals: Dict[str, Any]
    raw: Dict[str, Any] = field(default_factory=dict)
    #: Cd–Se 6-ring templates: chair/boat only; no planar / no bare Cd2Se2.
    rings: Dict[str, Any] = field(default_factory=dict)
    #: Acceptance band for a finished molecule.  Absent keys fall back to the
    #: module defaults in ``molecular``, so existing packs keep their behaviour.
    tolerances: Dict[str, Any] = field(default_factory=dict)
    #: Seed motifs the skeleton enumerator grows from.  Absent -> the built-in
    #: Cd3Se3 ring and its path/edge fusions.
    motifs: Dict[str, Any] = field(default_factory=dict)
    # A pack is immutable in practice and its tables are tiny, but the lookups
    # below run millions of times per enumeration: ``bond_length`` alone scans
    # and sorts the whole bond table on every call.  Memoise per instance
    # rather than with ``lru_cache``, which would need the pack to be hashable.
    _bond_cache: Dict[Any, float] = field(
        default_factory=dict, repr=False, compare=False
    )
    _angle_cache: Dict[Any, Optional[float]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _hard_cache: Dict[Any, bool] = field(
        default_factory=dict, repr=False, compare=False
    )
    _improper_cache: Dict[Any, Optional[float]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _proper_cache: Dict[Any, Optional[Mapping[str, Any]]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _one_four_cache: Dict[Any, Mapping[str, Any]] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def require_inorganic_connected(self) -> bool:
        return bool(self.graph_rules.get("require_inorganic_connected", False))

    @property
    def enforce_min_cn(self) -> bool:
        return bool(self.graph_rules.get("enforce_min_cn", False))

    @property
    def max_shared_ligands_per_host_pair(self) -> int:
        return int(self.graph_rules.get("max_shared_ligands_per_host_pair", 0))

    @staticmethod
    def _edges_to_masks(edges: Sequence[Sequence[int]]) -> List[int]:
        """Turn ``[[cd, se], ...]`` local edges into one Se bitmask per Cd.

        The enumerator seeds a motif as a row bitmask per seeded cation, which
        is compact but unreadable; the pack states the same thing as an edge
        list over local indices ``Cd0..`` / ``Se0..``.
        """

        pairs = [(int(cd), int(se)) for cd, se in edges]
        if not pairs:
            return []
        masks = [0] * (max(cd for cd, _ in pairs) + 1)
        for cd, se in pairs:
            masks[cd] |= 1 << se
        return masks

    def motif_masks(self, name: str) -> Optional[List[int]]:
        """Seed masks for one named motif, or ``None`` when the pack omits it."""

        entry = (self.motifs or {}).get(name)
        if not isinstance(entry, Mapping) or not entry.get("edges"):
            return None
        return self._edges_to_masks(entry["edges"])

    def fusion_motifs(self) -> Optional[List[Tuple[str, List[int]]]]:
        """Named two-ring fusion seeds, or ``None`` when the pack omits them.

        Which fusions a lattice actually contains is chemistry, not code: bulk
        zinc-blende has corner, edge and path fusions but no face fusion, so a
        ZB pack lists only path and edge while a wurtzite pack may add more.
        """

        entries = (self.motifs or {}).get("fusions")
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            return None
        out: List[Tuple[str, List[int]]] = []
        for entry in entries:
            if not isinstance(entry, Mapping) or not entry.get("edges"):
                continue
            out.append(
                (str(entry.get("name", "fusion")), self._edges_to_masks(entry["edges"]))
            )
        return out or None

    def nonbonded_floor(
        self, pair: Tuple[str, str], path_length: Optional[int] = None
    ) -> Optional[Tuple[float, float]]:
        """``(min_A, hard_A)`` for a pair with **no** graph edge.

        ``nonbonded_1_4`` covers graph path length exactly 3, where a cis
        torsion legitimately brings atoms closer; everything further apart, or
        disconnected, uses ``nonbonded``.  Restraints are one-sided: below
        ``min_A`` soft, at or below ``min_A - tol_A`` a hard reject.
        """

        key = "-".join(sorted(pair))
        blocks: List[str] = []
        if path_length == 3:
            blocks.append("nonbonded_1_4")
        blocks.append("nonbonded")
        for name in blocks:
            table = self.raw.get(name)
            if not isinstance(table, Mapping):
                continue
            entry = table.get(key)
            if isinstance(entry, Mapping) and entry.get("min_A") is not None:
                low = float(entry["min_A"])
                return low, low - float(entry.get("tol_A", 0.0))
        return None

    def angle_sum_cn3(self, element: str) -> Optional[Tuple[float, float]]:
        """``(target_deg, tol_deg)`` for the CN3 angle sum, if the pack states it."""

        table = self.raw.get("angle_sum_cn3")
        if not isinstance(table, Mapping):
            return None
        entry = table.get(element)
        if not isinstance(entry, Mapping) or entry.get("target_deg") is None:
            return None
        return float(entry["target_deg"]), float(entry.get("tol_deg", 0.0))

    def _tolerance(self, key: str, default: float) -> float:
        value = self.tolerances.get(key)
        return default if value is None else float(value)

    @property
    def audit_bond_tolerance_A(self) -> float:
        """How far a finished bond may sit from its table value, in angstrom."""

        from .molecular import AUDIT_BOND_TOLERANCE_A

        return self._tolerance("audit_bond_A", AUDIT_BOND_TOLERANCE_A)

    def bond_tolerance_A(
        self, pair: str, cn_cd: int, cn_other: int
    ) -> float:
        """Half-width of the acceptable band for one bond.

        Per-entry ``tol_A`` wins; otherwise the pack-wide audit band.  This is
        what turns the optimizer's bond term into a flat-bottomed well: inside
        ``r_A +/- tol_A`` nothing pulls, so satisfying one constraint no longer
        drags the others off their targets.
        """

        key = (str(pair), int(cn_cd), int(cn_other))
        for entry in self.bonds:
            if (entry.pair, entry.cn_cd, entry.cn_other) == key:
                if entry.tol_A is not None:
                    return float(entry.tol_A)
                break
        return self.audit_bond_tolerance_A

    @property
    def audit_angle_tolerance_deg(self) -> float:
        """How far a finished hard centre angle may sit from its target."""

        from .molecular import AUDIT_ANGLE_TOLERANCE_DEG

        return self._tolerance("audit_angle_deg", AUDIT_ANGLE_TOLERANCE_DEG)

    @property
    def audit_improper_tolerance_deg(self) -> Optional[float]:
        """Band for the improper audit, or ``None`` when the pack disables it.

        The improper at a CN3 centre is *fully determined* by the three
        pairwise angles at that centre -- the only extra information it carries
        is the sign, i.e. which side the pyramid points, which is a mirror
        image with no chemistry attached for these species.  Auditing both the
        angles and the improper therefore over-constrains the centre, and does
        real damage when the two disagree: a pack whose angle medians sum to
        less than 360 deg describes a pyramidal centre, and cannot also require
        a planar improper.  Such a pack should set ``audit_improper_deg: null``
        and let its angles govern.
        """

        from .molecular import AUDIT_IMPROPER_TOLERANCE_DEG

        if "audit_improper_deg" not in self.tolerances:
            return AUDIT_IMPROPER_TOLERANCE_DEG
        value = self.tolerances["audit_improper_deg"]
        return None if value is None else float(value)

    def _resolved_pair_rules(self) -> Dict[str, Any]:
        """Pair rules with their distances filled in from the tables.

        A pack may state ``Cd-Se: allowed`` and nothing else, keeping every
        distance in ``bonds`` / ``nonbonded`` so no number is written twice.
        The downstream spec still wants a distance per pair, so it is derived
        here: the hard non-bonded floor is what decides whether two unbonded
        atoms are too close, which is exactly what these fields mean.
        """

        out: Dict[str, Any] = {}
        for key, value in (self.graph_rules.get("pair_rules") or {}).items():
            entry = {"bond": value} if isinstance(value, str) else dict(value)
            elements = tuple(
                part.strip() for part in str(key).replace("_", "-").split("-")
            )
            floor = self.nonbonded_floor(elements) if len(elements) == 2 else None
            allowed = str(entry.get("bond", "")).strip().lower() == "allowed" or (
                entry.get("bond") is True
            )
            if allowed and entry.get("bond_max_distance") is None and floor:
                entry["bond_max_distance"] = floor[1]
            if not allowed and entry.get("min_distance") is None and floor:
                entry["min_distance"] = floor[1]
            out[key] = entry
        return out

    def _motif_span(
        self, anion: str, anion_cn: int, cd_cn: int, bond_key: str
    ) -> Optional[Tuple[float, float, float]]:
        """``(span, r, half_angle_cos)`` for one anion-centred motif.

        An anion of coordination ``n`` holds its Cd at a fixed pairwise angle,
        so it *demands* a Cd...Cd separation of ``2 r sin(theta/2)``.  Two
        motifs sharing two Cd both demand one, and they have to agree.
        """

        theta = self.center_angle_deg(anion, anion_cn, neighbor_pair="Cd-Cd")
        if theta is None:
            return None
        # Only compare where the table actually has a row: a missing entry
        # falls back to a generic default, and comparing two defaults says
        # nothing about whether the two motifs really conflict.
        if not any(
            (e.pair, e.cn_cd, e.cn_other) == (bond_key, cd_cn, anion_cn)
            for e in self.bonds
        ):
            return None
        r = self.bond_length(bond_key, cd_cn, anion_cn)
        half = math.radians(theta) / 2.0
        return 2.0 * r * math.sin(half), r, math.cos(half)

    def incompatible_shared_anion_pairs(
        self, cation: str = "Cd", anion: str = "Se", ligand: str = "Cl"
    ) -> List[Tuple[str, int, str, int]]:
        """Anion motif pairs that cannot share two cations.

        Derived from the tables rather than declared, so it tracks the angles:
        each motif fixes a cation-cation separation, and if two motifs share a
        cation *pair* the two separations must be reconcilable inside their own
        angle bands.  A tetrahedral (109.5 deg) anion wants ~4.4 A while every
        90 deg bridge wants ~3.6 A -- 0.8 A apart, which no band absorbs.

        Only pairs incompatible at *every* cation CN are reported, so this
        never rejects something merely tight.
        """

        keys = {anion: f"{cation}{anion}", ligand: f"{cation}Cl_bridge"}
        motifs = [
            (el, cn)
            for el, cn_max in ((anion, 6), (ligand, 3))
            for cn in range(2, cn_max + 1)
            if self.center_angle_deg(el, cn, neighbor_pair=f"{cation}-{cation}")
            is not None
        ]
        out: List[Tuple[str, int, str, int]] = []
        for (ea, na), (eb, nb) in itertools.combinations(motifs, 2):
            if ea == eb:
                continue  # two of the same anion on one pair: a ring rule
            bad_everywhere = None
            for cd_cn in (2, 3, 4):
                a = self._motif_span(ea, na, cd_cn, keys[ea])
                b = self._motif_span(eb, nb, cd_cn, keys[eb])
                if a is None or b is None:
                    continue          # table has no row here; no evidence
                if bad_everywhere is None:
                    bad_everywhere = True
                gap = abs(a[0] - b[0])
                budget = 0.0
                for el, cn, (span, r, cos_half) in ((ea, na, a), (eb, nb, b)):
                    band = self.center_angle_tolerance_deg(
                        el, cn, neighbor_pair=f"{cation}-{cation}"
                    )
                    if band is None:
                        band = self.audit_angle_tolerance_deg
                    budget += math.radians(band) * r * cos_half
                if gap <= budget:
                    bad_everywhere = False
                    break
            if bad_everywhere:
                out.append((ea, na, eb, nb))
        return out

    def nucleation_graph_rules_mapping(self) -> Dict[str, Any]:
        """Translate the pack schema to the existing nucleation spec schema."""

        coordination = self.graph_rules.get("coordination") or {}
        forbid_raw = self.graph_rules.get("forbid_cdse_cn_pairs") or []
        forbid_pairs: List[Tuple[int, int]] = []
        if isinstance(forbid_raw, list):
            for item in forbid_raw:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    forbid_pairs.append((int(item[0]), int(item[1])))
        decoration_mode = str(self.graph_rules.get("decoration_mode") or "")
        if "forbid_shared_cd_pair" in self.graph_rules:
            forbidden_shared_pairs = [
                tuple(item)
                for item in (self.graph_rules.get("forbid_shared_cd_pair") or [])
            ]
        elif decoration_mode in {"motif_graph", "motif_bridge_first"}:
            # Motif assembly deliberately keeps the incidence graph and lets
            # the motif/junction reconstruction resolve shared-host geometry.
            # The span heuristic is too aggressive here (notably Se-Cd4 with
            # Cl-Cd2), so it must not silently remove a motif graph.
            forbidden_shared_pairs = []
        else:
            forbidden_shared_pairs = self.incompatible_shared_anion_pairs()
        return {
            "min_cn": {
                str(symbol): int(bounds["min"])
                for symbol, bounds in coordination.items()
            },
            "max_cn": {
                str(symbol): int(bounds["max"])
                for symbol, bounds in coordination.items()
            },
            "pair_rules": self._resolved_pair_rules(),
            "forbid_shared_cd_pair": forbidden_shared_pairs,
            "allowed_neighbor_signatures": dict(
                self.graph_rules.get("allowed_neighbor_signatures") or {}
            ),
            "max_shared_ligands_per_host_pair": (
                self.max_shared_ligands_per_host_pair
            ),
            "bridge_cd_cd_max_distance": (
                None
                if self.graph_rules.get("bridge_cd_cd_max_distance") is None
                else float(self.graph_rules["bridge_cd_cd_max_distance"])
            ),
            "min_ring_size": dict(self.graph_rules.get("min_ring_size") or {}),
            "min_bridged_host_cn": int(
                self.graph_rules.get("min_bridged_host_cn", 1)
            ),
            "forbid_mono_se_dual_terminal": bool(
                self.graph_rules.get("forbid_mono_se_dual_terminal", False)
            ),
            "reject_closable_terminal_cd2": bool(
                self.graph_rules.get("reject_closable_terminal_cd2", False)
            ),
            "closable_terminal_cd2_distance": float(
                self.graph_rules.get(
                    "closable_terminal_cd2_distance", 3.50
                )
            ),
            "require_bridge_maximal": bool(
                self.graph_rules.get("require_bridge_maximal", False)
            ),
            "forbid_cdse_cn_pairs": forbid_pairs,
            "decoration_mode": str(
                self.graph_rules.get("decoration_mode", "graph_multiset")
            ).strip().lower()
            or "graph_multiset",
            "bridge_first_p1_terminal_policy": str(
                self.graph_rules.get(
                    "bridge_first_p1_terminal_policy", "unrestricted"
                )
            ).strip().lower()
            or "unrestricted",
            "passivate_min_cd_cn_p_ge": int(
                self.graph_rules.get("passivate_min_cd_cn_p_ge", 100)
            ),
            "passivate_min_cd_cn_k_ge": int(
                self.graph_rules.get("passivate_min_cd_cn_k_ge", 2)
            ),
            "passivate_min_cd_cn": int(
                self.graph_rules.get("passivate_min_cd_cn", 3)
            ),
            "bridge_first_hard_max_bridges_per_cd": int(
                self.graph_rules.get("bridge_first_hard_max_bridges_per_cd", 2)
            ),
            "bridge_first_prefer_bridges_per_cd": int(
                self.graph_rules.get("bridge_first_prefer_bridges_per_cd", 2)
            ),
            "required_rings": self.graph_rules.get("required_rings") or [],
            "min_core_edge_fraction": float(
                self.graph_rules.get("min_core_edge_fraction", 0.0) or 0.0
            ),
            "max_core_cut_edges": int(
                self.graph_rules.get("max_core_cut_edges", -1)
            ),
            "max_excess_cn1_cations": int(
                self.graph_rules.get("max_excess_cn1_cations", -1)
            ),
            "bridge_first_max_automorphisms": int(
                self.graph_rules.get("bridge_first_max_automorphisms", 64)
            ),
            "bridge_first_target_bridge_fraction": float(
                self.graph_rules.get(
                    "bridge_first_target_bridge_fraction", 0.0
                ) or 0.0
            ),
            "bridge_first_maximize_bridged_pairs": bool(
                self.graph_rules.get(
                    "bridge_first_maximize_bridged_pairs", False
                )
            ),
            "selection_order": str(
                self.graph_rules.get("selection_order", "bond_bands")
            ),
            "selection_top_fraction": float(
                self.graph_rules.get("selection_top_fraction", 0.0) or 0.0
            ),
            "selection_max_wiener_excess": float(
                self.graph_rules.get("selection_max_wiener_excess", 0.0) or 0.0
            ),
            "forbid_mu3_host_bridge_overlap": bool(
                self.graph_rules.get("forbid_mu3_host_bridge_overlap", False)
            ),
            "ring_first_when_pattern_possible": bool(
                self.graph_rules.get(
                    "ring_first_when_pattern_possible", False
                )
            ),
            "ring_first_fallback_to_open": bool(
                self.graph_rules.get("ring_first_fallback_to_open", True)
            ),
            "multi_ring_ladder": bool(
                self.graph_rules.get("multi_ring_ladder", True)
            ),
            "ring_min_pattern_cd": tuple(
                int(x)
                for x in (
                    self.graph_rules.get("ring_min_pattern_cd") or (3, 3, 4)
                )
            ),
            "ring_min_pattern_se": tuple(
                int(x)
                for x in (
                    self.graph_rules.get("ring_min_pattern_se") or (3, 3, 3)
                )
            ),
        }

    def bond_length(
        self,
        pair: str,
        cn_cd: int,
        cn_other: int,
        *,
        default: float = 2.55,
    ) -> float:
        """Return tabulated r, with nearest-CN fallback then default."""

        key = (pair, cn_cd, cn_other, default)
        cached = self._bond_cache.get(key)
        if cached is not None:
            return cached
        self._bond_cache[key] = value = self._bond_length(
            pair, cn_cd, cn_other, default=default
        )
        return value

    def _bond_length(
        self,
        pair: str,
        cn_cd: int,
        cn_other: int,
        *,
        default: float = 2.55,
    ) -> float:
        exact = [
            e
            for e in self.bonds
            if e.pair == pair and e.cn_cd == cn_cd and e.cn_other == cn_other
        ]
        if exact:
            return float(exact[0].r_A)
        # Fallback: same pair, minimize |cn_cd-diff| + |cn_other-diff|
        scored = []
        for e in self.bonds:
            if e.pair != pair:
                continue
            score = abs(e.cn_cd - cn_cd) + abs(e.cn_other - cn_other)
            scored.append((score, e.r_A))
        if scored:
            scored.sort()
            return float(scored[0][1])
        # Broad pair family fallback
        family = pair.split("_")[0] if "_" in pair else pair
        for e in self.bonds:
            if e.pair.startswith(family) or e.pair == family:
                scored.append((0, e.r_A))
        if scored:
            scored.sort()
            return float(scored[0][1])
        return float(default)

    def center_angle_deg(
        self,
        element: str,
        cn: int,
        *,
        neighbor_pair: Optional[str] = None,
        signature: Optional[str] = None,
        role_pair: Optional[str] = None,
        role_signature: Optional[str] = None,
        default: Optional[float] = None,
    ) -> Optional[float]:
        key = (element, cn, neighbor_pair, signature, role_pair,
               role_signature, default)
        if key in self._angle_cache:
            return self._angle_cache[key]
        self._angle_cache[key] = value = self._center_angle_deg(
            element,
            cn,
            neighbor_pair=neighbor_pair,
            signature=signature,
            role_pair=role_pair,
            role_signature=role_signature,
            default=default,
        )
        return value

    @staticmethod
    def _pair_keys(role_pair: Optional[str], neighbor_pair: Optional[str]) -> List[str]:
        """Keys to try, most specific first.

        ``_role_environment`` already emits exactly the names the table uses
        (``Cl_b2s-Cl_t``, sorted), so this is a direct match -- role key first,
        then the bare element pair for rows written without roles.
        """

        return [key for key in (role_pair, neighbor_pair) if key]

    @staticmethod
    def _angle_value(entry: object, want: str) -> Optional[float]:
        """Read ``deg``/``angle_deg`` (or ``tol_deg``) from a table entry."""

        if isinstance(entry, Mapping):
            for key in (("deg", "angle_deg") if want == "deg" else ("tol_deg",)):
                if entry.get(key) is not None:
                    return float(entry[key])
            return None
        if want == "deg" and isinstance(entry, (int, float)):
            return float(entry)
        return None

    def _center_angle_entry(
        self,
        element: str,
        cn: int,
        *,
        neighbor_pair: Optional[str] = None,
        signature: Optional[str] = None,
        role_pair: Optional[str] = None,
        role_signature: Optional[str] = None,
    ) -> object:
        """Most specific matching table entry, or ``None``.

        Order: role signature -> signature -> by_pair -> bare pair -> default.
        """

        block = self.angles.get(element, {})
        conf = block.get(f"cn{cn}") or block.get(str(cn)) or {}
        if not isinstance(conf, Mapping):
            return None
        pairs = self._pair_keys(role_pair, neighbor_pair)

        by_role = conf.get("by_role_signature")
        if role_signature and isinstance(by_role, Mapping):
            role_block = by_role.get(role_signature)
            if isinstance(role_block, Mapping):
                for candidate in pairs:
                    if candidate in role_block:
                        return role_block[candidate]

        by_sig = conf.get("by_signature")
        if signature and isinstance(by_sig, Mapping):
            sig_block = by_sig.get(signature) or {}
            if isinstance(sig_block, Mapping):
                nested = sig_block.get("by_role_signature")
                if role_signature and isinstance(nested, Mapping):
                    role_block = nested.get(role_signature) or {}
                    if isinstance(role_block, Mapping):
                        for candidate in pairs:
                            if candidate in role_block:
                                return role_block[candidate]
                for candidate in pairs:
                    if candidate in sig_block:
                        return sig_block[candidate]

        by_pair = conf.get("by_pair")
        if isinstance(by_pair, Mapping):
            for candidate in pairs:
                if candidate in by_pair:
                    return by_pair[candidate]
        for candidate in pairs:
            if candidate in conf:
                return conf[candidate]
        if conf.get("default") is not None:
            return conf["default"]
        if conf.get("default_deg") is not None:
            return conf["default_deg"]
        return None

    def _center_angle_deg(
        self,
        element: str,
        cn: int,
        *,
        neighbor_pair: Optional[str] = None,
        signature: Optional[str] = None,
        role_pair: Optional[str] = None,
        role_signature: Optional[str] = None,
        default: Optional[float] = None,
    ) -> Optional[float]:
        entry = self._center_angle_entry(
            element, cn, neighbor_pair=neighbor_pair, signature=signature,
            role_pair=role_pair, role_signature=role_signature,
        )
        value = self._angle_value(entry, "deg")
        return default if value is None else value

    def center_angle_modes(
        self,
        element: str,
        cn: int,
        *,
        neighbor_pair: Optional[str] = None,
        signature: Optional[str] = None,
        role_pair: Optional[str] = None,
        role_signature: Optional[str] = None,
    ) -> Optional[Tuple[Tuple[float, float], ...]]:
        """Multi-modal angle target as ``((deg, tol_deg), ...)``, or ``None``.

        A four-coordinate anion is *not* described by one angle: the measured
        Se-Cd4 distribution is bimodal, ~85 deg for the four cis pairs and
        ~160 deg for the two trans pairs, with nothing in between.  A single
        ``deg:`` entry has to sit in the empty valley and matches neither.
        Declare both explicitly instead::

            angles:
              Se:
                cn4:
                  default: {deg: 109.5, tol_deg: 12}   # fallback
                  modes:
                    - {deg: 85,  tol_deg: 15}          # cis
                    - {deg: 160, tol_deg: 20}          # trans

        An angle satisfies a multi-modal entry when it lies in *any* mode's
        band; consumers take the minimum band excess over the modes.  The
        ``default`` is retained so callers that do not understand ``modes``
        keep working.
        """

        entry = self._center_angle_entry(
            element,
            cn,
            neighbor_pair=neighbor_pair,
            signature=signature,
            role_pair=role_pair,
            role_signature=role_signature,
        )
        block = self.angles.get(element, {})
        conf = block.get(f"cn{cn}") or block.get(str(cn)) or {}
        raw = None
        if isinstance(entry, Mapping) and entry.get("modes") is not None:
            raw = entry.get("modes")
        elif isinstance(conf, Mapping):
            raw = conf.get("modes")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return None
        modes: List[Tuple[float, float]] = []
        for row in raw:
            if not isinstance(row, Mapping):
                continue
            deg = row.get("deg")
            tol = row.get("tol_deg", row.get("tolerance_deg"))
            # A mode without a band cannot be enforced, and an unenforced mode
            # would silently absorb every angle in a min-over-modes residual.
            if deg is None or tol is None:
                continue
            modes.append((float(deg), float(tol)))
        return tuple(modes) or None

    def center_angle_tolerance_deg(
        self,
        element: str,
        cn: int,
        *,
        neighbor_pair: Optional[str] = None,
        signature: Optional[str] = None,
        role_pair: Optional[str] = None,
        role_signature: Optional[str] = None,
    ) -> Optional[float]:
        """Acceptance band for this angle, or ``None`` when the table has none.

        ``None`` means the pack states a target but no tolerance, which is how
        an older pack reads -- such an angle stays unenforced, as before.
        """

        entry = self._center_angle_entry(
            element, cn, neighbor_pair=neighbor_pair, signature=signature,
            role_pair=role_pair, role_signature=role_signature,
        )
        return self._angle_value(entry, "tol_deg")

    def center_angle_is_hard(
        self,
        element: str,
        cn: int,
        *,
        neighbor_pair: str,
        signature: str,
        role_pair: Optional[str] = None,
        role_signature: Optional[str] = None,
    ) -> bool:
        key = (element, cn, neighbor_pair, signature, role_pair, role_signature)
        cached = self._hard_cache.get(key)
        if cached is not None:
            return cached
        self._hard_cache[key] = value = self._center_angle_is_hard(
            element,
            cn,
            neighbor_pair=neighbor_pair,
            signature=signature,
            role_pair=role_pair,
            role_signature=role_signature,
        )
        return value

    def _center_angle_is_hard(
        self,
        element: str,
        cn: int,
        *,
        neighbor_pair: Optional[str] = None,
        signature: Optional[str] = None,
        role_pair: Optional[str] = None,
        role_signature: Optional[str] = None,
    ) -> bool:
        """An angle is enforced exactly when the table states a tolerance.

        Previously this consulted a separate ``hard``/``status`` flag that no
        pack ever set, so every angle was advisory and the audit checked none
        of them.  Stating ``tol_deg`` is the pack saying how far the angle may
        stray, which is the same thing as saying it is checked.
        """

        return (
            self.center_angle_tolerance_deg(
                element,
                cn,
                neighbor_pair=neighbor_pair,
                signature=signature,
                role_pair=role_pair,
                role_signature=role_signature,
            )
            is not None
        )

    def improper_angle_deg(
        self, element: str, cn: int, signature: str
    ) -> Optional[float]:
        """Return an explicitly configured hard improper, if one exists."""

        key = (element, cn, signature)
        if key in self._improper_cache:
            return self._improper_cache[key]
        improper = self.dihedrals.get("improper") or {}
        element_rules = improper.get(element) or {}
        cn_rules = element_rules.get(f"cn{cn}") or element_rules.get(str(cn)) or {}
        value = None
        if isinstance(cn_rules, Mapping):
            # Signature row first, then a ``default`` row covering every
            # signature -- planarity is often one statement, not four.
            value = cn_rules.get(signature)
            if value is None:
                value = cn_rules.get("default")
        result = self._angle_value(value, "deg")
        self._improper_cache[key] = result
        return result

    def improper_tolerance_deg(
        self, element: str, cn: int, signature: str
    ) -> Optional[float]:
        """Band for this improper, or ``None`` when the table states none."""

        improper = self.dihedrals.get("improper") or {}
        element_rules = improper.get(element) or {}
        cn_rules = element_rules.get(f"cn{cn}") or element_rules.get(str(cn)) or {}
        if not isinstance(cn_rules, Mapping):
            return None
        entry = cn_rules.get(signature)
        if entry is None:
            entry = cn_rules.get("default")
        return self._angle_value(entry, "tol_deg")

    @staticmethod
    def _dihedral_path(path_symbols: Sequence[str]) -> Tuple[str, ...]:
        """Canonicalise a four-atom path so reverse paths are equivalent."""

        path = tuple(str(symbol) for symbol in path_symbols)
        if len(path) != 4:
            return path
        reverse = tuple(reversed(path))
        return min(path, reverse)

    def _proper_rule(self, path_symbols: Sequence[str]) -> Optional[Mapping[str, Any]]:
        """Return the executable proper-dihedral rule for a path, if present."""

        canonical = self._dihedral_path(path_symbols)
        if canonical in self._proper_cache:
            return self._proper_cache[canonical]
        proper = self.dihedrals.get("proper") or {}
        if not isinstance(proper, Sequence) or isinstance(proper, (str, bytes)):
            self._proper_cache[canonical] = None
            return None
        for raw_rule in proper:
            if not isinstance(raw_rule, Mapping):
                continue
            raw_path = raw_rule.get("path")
            if isinstance(raw_path, str):
                candidate = tuple(part.strip() for part in raw_path.split("-"))
            elif isinstance(raw_path, Sequence):
                candidate = tuple(str(part) for part in raw_path)
            else:
                continue
            if self._dihedral_path(candidate) == canonical:
                self._proper_cache[canonical] = raw_rule
                return raw_rule
        self._proper_cache[canonical] = None
        return None

    def preferred_dihedral(
        self, path_symbols: Sequence[str]
    ) -> Optional[Tuple[float, float]]:
        """Return ``(target_deg, tolerance_deg)`` for preferred torsions."""

        rule = self._proper_rule(path_symbols)
        if rule is None or str(rule.get("weight", "preferred")).lower() != "preferred":
            return None
        target = rule.get("target_deg")
        if target is None:
            return None
        return float(target), float(rule.get("tolerance_deg", 30.0))

    def dihedral_weight(self, path_symbols: Sequence[str]) -> str:
        """Return the placement policy for a proper torsion path."""

        rule = self._proper_rule(path_symbols)
        if rule is None:
            return "ignore"
        return str(rule.get("weight", "preferred")).strip().lower()

    def dihedral_excluded(
        self,
        path_symbols: Sequence[str],
        *,
        endocyclic: bool = False,
    ) -> bool:
        """Whether a proper rule explicitly excludes this local path."""

        rule = self._proper_rule(path_symbols)
        if rule is None:
            return False
        excluded = rule.get("exclude_if")
        return bool(endocyclic and str(excluded).strip().lower() == "endocyclic_ring_edge")

    def one_four_rule(self, pair: str) -> Mapping[str, Any]:
        """Return the configured soft/hard clearance rule for a pair."""

        rules = self.dihedrals.get("one_four") or {}
        if not isinstance(rules, Mapping):
            return {}
        key = "-".join(sorted(str(part) for part in str(pair).split("-")))
        if key in self._one_four_cache:
            return self._one_four_cache[key]
        for raw_key, value in rules.items():
            canonical = "-".join(sorted(str(part) for part in str(raw_key).split("-")))
            if canonical == key and isinstance(value, Mapping):
                self._one_four_cache[key] = value
                return value
        self._one_four_cache[key] = {}
        return {}

    def soft_contact_penalty(self, pair: str, distance_A: float) -> float:
        """Return a non-negative soft penalty below the configured soft minimum."""

        rule = self.one_four_rule(pair)
        soft_min = rule.get("soft_min_A")
        if soft_min is None:
            return 0.0
        return max(0.0, float(soft_min) - float(distance_A))

    def one_four_hard_min(self, pair: str) -> Optional[float]:
        """Return the documented hard minimum without overriding pair_rules."""

        value = self.one_four_rule(pair).get("hard_min_A")
        return None if value is None else float(value)

    def cdse6_conformations(self) -> Tuple[str, ...]:
        """Allowed 6-ring shapes: chair and/or boat (never planar)."""

        block = (
            self.rings.get("cdse_6") if isinstance(self.rings, Mapping) else None
        )
        if not isinstance(block, Mapping):
            return ("chair", "boat")
        conf = block.get("conformations") or ["chair", "boat"]
        out: List[str] = []
        for item in conf:
            name = str(item).strip().lower()
            if name in {"chair", "boat"} and name not in out:
                out.append(name)
        return tuple(out) if out else ("chair", "boat")

    def cdse6_dihedrals(self, conformation: str) -> Tuple[float, ...]:
        """Endocyclic dihedral sequence (deg) that defines chair vs boat.

        Six successive torsions around Se–Cd–Se–Cd–Se–Cd.  Defaults:
        chair ±60° alternating; boat 0/±60 pattern.
        """

        conf = str(conformation).strip().lower()
        defaults = {
            "chair": (60.0, -60.0, 60.0, -60.0, 60.0, -60.0),
            "boat": (0.0, 60.0, -60.0, 0.0, 60.0, -60.0),
        }
        block = (
            self.rings.get("cdse_6") if isinstance(self.rings, Mapping) else None
        )
        if not isinstance(block, Mapping):
            return defaults.get(conf, defaults["chair"])
        dih = block.get("dihedrals") or {}
        if not isinstance(dih, Mapping):
            return defaults.get(conf, defaults["chair"])
        raw = dih.get(conf)
        if raw is None:
            return defaults.get(conf, defaults["chair"])
        vals = tuple(float(x) for x in raw)
        if len(vals) < 3:
            return defaults.get(conf, defaults["chair"])
        if len(vals) < 6:
            # repeat to length 6
            reps = list(vals)
            while len(reps) < 6:
                reps.append(reps[len(reps) % len(vals)])
            return tuple(reps[:6])
        return vals[:6]

    def cdse6_ring_pattern(
        self, name: Optional[str] = None
    ) -> CdSe6RingPattern:
        """Full-CN ring pattern (default min stable Cd[3,3,4]/Se[3,3,3])."""

        block = (
            self.rings.get("cdse_6") if isinstance(self.rings, Mapping) else {}
        )
        if not isinstance(block, Mapping):
            block = {}
        patterns = block.get("patterns") or {}
        if not isinstance(patterns, Mapping):
            patterns = {}
        # Single construction pattern (averaged DFT endocyclic values).
        key = name or str(block.get("default_pattern") or "default")
        raw = patterns.get(key) if isinstance(patterns, Mapping) else None
        if not isinstance(raw, Mapping) and patterns:
            # Fall back to the only / first pattern entry if default name missing.
            first = next(iter(patterns.values()), None)
            if isinstance(first, Mapping):
                raw = first
                key = str(next(iter(patterns.keys())))
        if not isinstance(raw, Mapping):
            return CdSe6RingPattern(
                name="default",
                bond_cdse_A=2.635,
                angle_at_cd_deg=109.47,
                angle_at_se_deg=109.47,
                cd_cn=(3, 3, 4),
                se_cn=(3, 3, 3),
            )
        cd_cn_raw = list(raw.get("cd_cn") or (3, 3, 4))
        se_cn_raw = list(raw.get("se_cn") or (3, 3, 3))
        while len(cd_cn_raw) < 3:
            cd_cn_raw.append(cd_cn_raw[-1] if cd_cn_raw else 3)
        while len(se_cn_raw) < 3:
            se_cn_raw.append(se_cn_raw[-1] if se_cn_raw else 3)
        return CdSe6RingPattern(
            name=str(key),
            bond_cdse_A=float(raw.get("bond_cdse_A", 2.635)),
            angle_at_cd_deg=float(raw.get("angle_at_cd_deg", 109.47)),
            angle_at_se_deg=float(raw.get("angle_at_se_deg", 109.47)),
            cd_cn=(int(cd_cn_raw[0]), int(cd_cn_raw[1]), int(cd_cn_raw[2])),
            se_cn=(int(se_cn_raw[0]), int(se_cn_raw[1]), int(se_cn_raw[2])),
        )


def load_geometry_pack(path: str | Path) -> GeometryPack:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"geometry pack must be a mapping: {path}")
    # A compact motif pack may borrow the detailed executable geometry tables
    # from the current-builder pack while keeping its own graph rules,
    # motifs, reconstruction policy, and xTB settings.
    reference = raw.get("geometry_reference")
    if reference:
        reference_path = Path(str(reference))
        if not reference_path.is_absolute():
            reference_path = (path.parent / reference_path).resolve()
        reference_pack = load_geometry_pack(reference_path)
        merged = dict(reference_pack.raw)
        merged.update(dict(raw))
        for key in ("bonds", "angles", "dihedrals", "rings", "tolerances"):
            if key not in raw and key in reference_pack.raw:
                merged[key] = reference_pack.raw[key]
        raw = merged
    schema_version = int(raw.get("schema_version", 1))
    if schema_version != 2:
        raise ValueError(
            f"molecular geometry pack {path} requires schema_version: 2"
        )
    obsolete = {"centers", "hard_rules_defaults", "local_geometry", "bridging"}
    present = sorted(obsolete.intersection(raw))
    if present:
        raise ValueError(
            f"obsolete molecular geometry-pack keys in {path}: {', '.join(present)}"
        )
    graph_rules = raw.get("graph_rules")
    if not isinstance(graph_rules, Mapping):
        raise KeyError(f"geometry pack {path} requires graph_rules")
    coordination = graph_rules.get("coordination")
    if not isinstance(coordination, Mapping):
        raise KeyError(f"geometry pack {path} requires graph_rules.coordination")
    obsolete_graph = sorted(
        {"allowed_bonds", "bridging", "local_geometry"}.intersection(graph_rules)
    )
    if obsolete_graph:
        raise ValueError(
            f"obsolete graph_rules keys in {path}: {', '.join(obsolete_graph)}"
        )

    def metadata_paths(value: object, prefix: str = "") -> List[str]:
        found: List[str] = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{prefix}.{key}" if prefix else str(key)
                # Junction sample counts and modality notes are executable:
                # counts rank reconstruction starts and modes, while the
                # reported shape selects the appropriate circular residual.
                # They therefore belong in the self-contained motif pack.
                if not prefix and str(key) in {"junctions", "geometry_reference"}:
                    continue
                if str(key) in {"n", "shape", "note"}:
                    found.append(child_path)
                found.extend(metadata_paths(child, child_path))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                found.extend(metadata_paths(child, f"{prefix}[{index}]"))
        return found

    metadata = metadata_paths(raw)
    if metadata:
        raise ValueError(
            f"non-executable metadata keys in {path}: {', '.join(metadata)}"
        )
    bonds_raw = raw.get("bonds") or []
    angles_raw = raw.get("angles") or {}
    dihedrals_raw = raw.get("dihedrals") or {}
    # The motif builder declares local geometry once, beside the motif it
    # describes.  Compile that concise schema to the legacy executable tables
    # in memory; do not force users to duplicate motifs as separate bond,
    # angle and improper sections merely to satisfy the current builder API.
    motif_defs = raw.get("motifs") or {}
    linker_geometry = raw.get("linker_geometry") or {}
    if not bonds_raw and isinstance(motif_defs, Mapping):
        compiled_bonds: List[Dict[str, Any]] = []
        compiled_angles: Dict[str, Dict[str, Any]] = {}
        compiled_improper: Dict[str, Dict[str, Any]] = {}
        for motif in motif_defs.values():
            if not isinstance(motif, Mapping):
                continue
            center = str(motif.get("center", ""))
            linker = str(motif.get("linker", ""))
            count = int(motif.get("linker_count", 0))
            lengths = motif.get("bond_A_by_linker_cn") or {}
            if center == "Cl":
                pair = "CdCl_terminal" if count == 1 else "CdCl_bridge"
            elif center == "Se":
                pair = "CdSe"
            else:
                pair = f"{linker}{center}"
            for cd_cn, distance in lengths.items():
                compiled_bonds.append({
                    "pair": pair,
                    "cn_cd": int(cd_cn),
                    "cn_other": count,
                    "r_A": float(distance),
                    "tol_A": float(motif.get("bond_tol_A", 0.05)),
                })
            if motif.get("angle_deg") is not None and count >= 2:
                compiled_angles.setdefault(center, {})[f"cn{count}"] = {
                    "default": {
                        "deg": float(motif["angle_deg"]),
                        "tol_deg": float(motif.get("angle_tol_deg", 8.0)),
                    }
                }
            if motif.get("improper_deg") is not None and count == 3:
                compiled_improper.setdefault(center, {})["cn3"] = {
                    f"{linker}3": {
                        "deg": float(motif["improper_deg"]),
                        "tol_deg": float(motif.get("improper_tol_deg", 15.0)),
                    }
                }
        if isinstance(linker_geometry, Mapping):
            for element, by_cn in linker_geometry.items():
                if not isinstance(by_cn, Mapping):
                    continue
                for cn_key, entry in by_cn.items():
                    if not isinstance(entry, Mapping):
                        continue
                    cn_name = str(cn_key)
                    compiled_angles.setdefault(str(element), {})[cn_name] = {
                        "default": {
                            "deg": float(entry.get("angle_deg", 109.5)),
                            "tol_deg": float(entry.get("angle_tol_deg", 8.0)),
                        }
                    }
                    if entry.get("improper_deg") is not None and cn_name == "cn3":
                        compiled_improper.setdefault(str(element), {})["cn3"] = {
                            "default": {
                                "deg": float(entry["improper_deg"]),
                                "tol_deg": float(entry.get("improper_tol_deg", 15.0)),
                            }
                        }
        bonds_raw = compiled_bonds
        angles_raw = compiled_angles
        dihedrals_raw = {"improper": compiled_improper}

    bonds: List[BondLengthEntry] = []
    for item in bonds_raw:
        if not isinstance(item, Mapping):
            continue
        bonds.append(
            BondLengthEntry(
                pair=str(item["pair"]),
                cn_cd=int(item["cn_cd"]),
                cn_other=int(item["cn_other"]),
                r_A=float(item["r_A"]),
                tol_A=(
                    None if item.get("tol_A") is None else float(item["tol_A"])
                ),
            )
        )
    rings_raw = raw.get("rings") or {}
    if rings_raw is None:
        rings_raw = {}
    if not isinstance(rings_raw, Mapping):
        raise TypeError(f"geometry pack rings must be a mapping: {path}")
    skeleton_motifs = raw.get("skeleton_motifs")
    if skeleton_motifs is None:
        legacy_motifs = raw.get("motifs") or {}
        # Before motif assembly existed, ``motifs`` meant inorganic ring seed
        # masks.  New anion-centred motifs have center/linker/linker_count and
        # must not be reinterpreted as special skeleton/ring seeds.
        if isinstance(legacy_motifs, Mapping) and any(
            isinstance(value, Mapping)
            and (value.get("edges") or key == "fusions")
            for key, value in legacy_motifs.items()
        ):
            skeleton_motifs = legacy_motifs
        else:
            skeleton_motifs = {}
    return GeometryPack(
        schema_version=schema_version,
        name=str(raw.get("name", path.stem)),
        bonds=tuple(bonds),
        graph_rules=dict(graph_rules),
        angles=dict(angles_raw),
        dihedrals=dict(dihedrals_raw),
        raw=dict(raw),
        rings=dict(rings_raw),
        tolerances=dict(raw.get("tolerances") or {}),
        motifs=dict(skeleton_motifs or {}),
    )
