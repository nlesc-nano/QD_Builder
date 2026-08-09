from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from ..nc_types import (
    CoreMonomerSpec,
    NucleationBridgeRule,
    NucleationGeometryRules,
    NucleationGraphRules,
    NucleationPairRule,
    NucleationSpec,
    PrecursorUnitSpec,
)

_GEOMETRY_TEMPLATES = {"linear", "trigonal_planar", "tetrahedral", "octahedral"}

# "exact" enumerates every symmetry-distinct ligand arrangement; "guided" places
# one passivation-ordered shell per skeleton and does not enumerate isomers.
_NUCLEATION_MODES = {"exact", "guided"}

# "all" counts every bond at the score's bond-count component;
# "skeleton" counts only cation-anion bonds, so bridges stop buying rank.
_BOND_COUNT_SCOPES = {"all", "skeleton"}

# Core monomer growth filters (see NucleationSpec.core_growth_policy).
_CORE_GROWTH_POLICIES = {"all", "max_bonds", "compact_ring"}

# Post-passivation Cl-ring ranking (placement unchanged; see NucleationSpec).
_PASSIVATION_RING_POLICIES = {
    "none",
    "prefer_cl_rings",
    "require_cl_rings",
    "prefer_ligand_rings",
    "require_ligand_rings",
}
_GEOMETRY_DEFAULTS = {"zb_tetrahedral", "none"}
_TERMINAL_MOTIFS = {"zb_mx2", "none"}

_P_BEAM_RANKS = {"bonds", "six_rings", "fused_rings"}
_FUSED_CHAIR_MODES = {"off", "rank", "prefer_positive"}
_CORE_GROWTH_OCCUPATIONS = {"bare", "decorated"}
_PARENT_P_MODES = {"all_retained", "seed_band"}
_P_LADDER_MODES = {"inherited_plus", "product_window"}

# Checkpoint schema for --restart (per finished k-row).
# The default growth-shedding semantics changed: with no explicit surface law
# and no positive hard cap, all complete CdCl2 packages may be shed.  Bump the
# checkpoint schema so an older partial run cannot be resumed with different
# growth channels.
_CHECKPOINT_SCHEMA_VERSION = 2

# Above this many bridge opportunities the sub-maximum fallback is skipped and
# the bin records that the maximum-cardinality restriction went undischarged.
_SUB_MAXIMUM_FALLBACK_LIMIT = 20



def is_nucleation_yaml(path: str | Path) -> bool:
    """Return whether a YAML document contains a nucleation section."""

    raw = yaml.safe_load(Path(path).read_text()) or {}
    return isinstance(raw, Mapping) and isinstance(raw.get("nucleation"), Mapping)


def _bond_key(left: str, right: str) -> Tuple[str, str]:
    """Return the canonical representation of an unordered element pair."""

    first, second = str(left), str(right)
    return (first, second) if first <= second else (second, first)


def _parse_allowed_bonds(raw: Sequence[object]) -> Tuple[Tuple[str, str], ...]:
    """Parse and canonicalize strict unordered bond pairs from YAML."""

    pairs: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for index, item in enumerate(raw):
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) != 2
        ):
            raise TypeError(
                "nucleation.graph_rules.allowed_bonds"
                f"[{index}] must be a two-element sequence"
            )
        left, right = (str(item[0]).strip(), str(item[1]).strip())
        if not left or not right:
            raise ValueError("allowed bond species names must not be empty")
        pair = _bond_key(left, right)
        if pair in seen:
            raise ValueError(
                "duplicate unordered allowed bond: "
                f"{pair[0]}-{pair[1]}"
            )
        seen.add(pair)
        pairs.append(pair)
    if not pairs:
        raise ValueError("nucleation.graph_rules.allowed_bonds must not be empty")
    return tuple(sorted(pairs))


def _pair_rule_key(left: str, right: str) -> str:
    first, second = _bond_key(str(left).strip(), str(right).strip())
    return f"{first}-{second}"


def _split_pair_rule_key(raw: object) -> Tuple[str, str]:
    parts = str(raw).replace("_", "-").split("-")
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise ValueError(
            "nucleation.graph_rules.pair_rules keys must look like 'Cd-Se'"
        )
    return _bond_key(parts[0].strip(), parts[1].strip())


def _parse_required_rings(raw: object) -> Tuple[Dict[str, int], ...]:
    """Parse ``graph_rules.required_rings`` -- rings a graph must contain."""

    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError(
            "nucleation.graph_rules.required_rings must be a list of mappings"
        )
    out: List[Dict[str, int]] = []
    for row in raw:
        if not isinstance(row, Mapping):
            raise TypeError("required_rings entries must be mappings")
        size = int(row.get("size", 0))
        if size < 3:
            raise ValueError(f"required_rings size must be >= 3, got {size}")
        out.append(
            {
                "size": size,
                "min_count": max(1, int(row.get("min_count", 1))),
                "from_k": max(0, int(row.get("from_k", 0))),
            }
        )
    return tuple(out)


def _parse_min_ring_size(raw: object) -> Dict[str, int]:
    """Parse ``graph_rules.min_ring_size`` -- ``{"Cd-Se": 6}``."""

    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("nucleation.graph_rules.min_ring_size must be a mapping")
    parsed: Dict[str, int] = {}
    for key, value in raw.items():
        parts = str(key).replace("_", "-").split("-")
        if len(parts) != 2:
            raise ValueError(
                "nucleation.graph_rules.min_ring_size keys must look like 'Cd-Se'"
            )
        size = int(value)
        if size < 3:
            raise ValueError(
                f"min_ring_size[{key}] must be at least 3, got {size}"
            )
        parsed["-".join(sorted(parts))] = size
    return parsed


def _parse_pair_rules(raw: object) -> Dict[str, NucleationPairRule]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("nucleation.graph_rules.pair_rules must be a mapping")
    rules: Dict[str, NucleationPairRule] = {}
    for key_raw, value_raw in raw.items():
        if isinstance(value_raw, str):
            # Topology-only form: distances live in the bonds / nonbonded
            # tables, so a pair rule need only say whether the edge may exist.
            value_raw = {"bond": value_raw}
        if not isinstance(value_raw, Mapping):
            raise TypeError(
                f"nucleation.graph_rules.pair_rules.{key_raw} must be a mapping "
                "or the string 'allowed' / 'forbidden'"
            )
        elements = _split_pair_rule_key(key_raw)
        key = _pair_rule_key(*elements)
        if key in rules:
            raise ValueError(f"duplicate unordered pair rule: {key}")
        bond_raw = value_raw.get("bond")
        if isinstance(bond_raw, bool):
            bond_allowed = bond_raw
        else:
            bond_text = str(bond_raw).strip().lower()
            if bond_text not in {"allowed", "forbidden"}:
                raise ValueError(
                    f"pair rule {key} bond must be 'allowed' or 'forbidden'"
                )
            bond_allowed = bond_text == "allowed"
        bond_max = value_raw.get("bond_max_distance")
        min_distance = value_raw.get("min_distance")
        rules[key] = NucleationPairRule(
            elements=elements,
            bond_allowed=bond_allowed,
            bond_max_distance=None if bond_max is None else float(bond_max),
            min_distance=(
                None if min_distance is None else float(min_distance)
            ),
        )
    return dict(sorted(rules.items()))


def _parse_contact_min_distance(raw: object) -> Dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("nucleation.contact_min_distance must be a mapping")
    parsed: Dict[str, float] = {}
    for key_raw, value_raw in raw.items():
        elements = _split_pair_rule_key(key_raw)
        key = _pair_rule_key(*elements)
        if key in parsed:
            raise ValueError(f"duplicate unordered contact pair: {key}")
        parsed[key] = float(value_raw)
    return parsed


def _policy_lookup(
    section: Mapping[str, object],
    key: str,
    default: object,
    *,
    nested: Optional[str] = None,
) -> object:
    """Flat ``section[key]`` with optional nested map override."""

    if nested is not None:
        block = section.get(nested)
        if isinstance(block, Mapping) and key in block:
            return block[key]
    if key in section:
        return section[key]
    return default


def _parse_geometry_rules(
    raw: object,
    *,
    core: CoreMonomerSpec,
    precursor: PrecursorUnitSpec,
    geometry_defaults: str = "zb_tetrahedral",
) -> NucleationGeometryRules:
    """Parse optional retained-only templates, optionally merged with defaults."""

    defaults_mode = str(geometry_defaults).strip().lower()
    if defaults_mode not in _GEOMETRY_DEFAULTS:
        raise ValueError(
            "nucleation.geometry_defaults must be one of "
            + ", ".join(sorted(_GEOMETRY_DEFAULTS))
            + f"; got {geometry_defaults!r}"
        )
    by_cn: Dict[str, Dict[int, str]] = {}
    all_cn: Dict[str, str] = {}
    if defaults_mode == "zb_tetrahedral":
        by_cn = {
            core.cation: {
                2: "linear",
                3: "trigonal_planar",
                4: "tetrahedral",
            }
        }
        by_cn.setdefault(
            precursor.center,
            {
                2: "linear",
                3: "trigonal_planar",
                4: "tetrahedral",
            },
        )
        all_cn = {
            core.anion: "tetrahedral",
            precursor.ligand: "tetrahedral",
        }
    if raw is None:
        return NucleationGeometryRules(by_cn=by_cn, all_cn=all_cn)
    if not isinstance(raw, Mapping):
        raise TypeError("nucleation.geometry_rules must be a mapping")
    for symbol_raw, rules_raw in raw.items():
        symbol = str(symbol_raw)
        if not isinstance(rules_raw, Mapping):
            raise TypeError(
                f"nucleation.geometry_rules.{symbol} must be a mapping"
            )
        explicit: Dict[int, str] = {}
        for key_raw, template_raw in rules_raw.items():
            key = str(key_raw).strip().lower()
            template = str(template_raw).strip().lower()
            if template not in _GEOMETRY_TEMPLATES:
                raise ValueError(
                    f"unsupported geometry template {template!r} for {symbol}"
                )
            if key == "all":
                all_cn[symbol] = template
                continue
            if not key.startswith("cn") or not key[2:].isdigit():
                raise ValueError(
                    f"geometry key for {symbol} must be 'all' or 'cnN': {key}"
                )
            coordination = int(key[2:])
            if coordination < 2:
                raise ValueError(
                    f"geometry templates require CN >= 2: {symbol}.{key}"
                )
            explicit[coordination] = template
        if explicit:
            by_cn.setdefault(symbol, {}).update(explicit)
    return NucleationGeometryRules(by_cn=by_cn, all_cn=all_cn)


def _parse_ring_lengths(raw: object) -> Tuple[int, ...]:
    """Parse passivation ring lengths; default (4, 6) for Cl 4- and 6-cycles."""

    if raw is None:
        return (4, 6)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("nucleation.ring_lengths must be a sequence of integers")
    lengths: List[int] = []
    for index, item in enumerate(raw):
        value = int(item)
        if value < 3:
            raise ValueError(
                f"nucleation.ring_lengths[{index}] must be >= 3; got {value}"
            )
        if value not in lengths:
            lengths.append(value)
    if not lengths:
        raise ValueError("nucleation.ring_lengths must not be empty")
    return tuple(sorted(lengths))


def _parse_monomer_p_values(raw: object) -> Tuple[int, ...]:
    """Parse optional monomer package p list; empty means p_m=0 only at runtime."""

    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError(
            "nucleation.monomer_p_values must be a sequence of nonnegative ints"
        )
    values: List[int] = []
    for index, item in enumerate(raw):
        value = int(item)
        if value < 0:
            raise ValueError(
                f"nucleation.monomer_p_values[{index}] must be >= 0; got {value}"
            )
        if value not in values:
            values.append(value)
    return tuple(sorted(values))


def _parse_bridge_rules(raw: object) -> Tuple[NucleationBridgeRule, ...]:
    """Parse optional graph-level latent ligand bridge motifs."""

    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise TypeError("nucleation.graph_rules.bridging must be a mapping")
    rules: List[NucleationBridgeRule] = []
    for ligand_raw, rule_raw in raw.items():
        ligand = str(ligand_raw).strip()
        if not ligand or not isinstance(rule_raw, Mapping):
            raise TypeError(
                "each nucleation.graph_rules.bridging entry must be a mapping"
            )
        try:
            rules.append(
                NucleationBridgeRule(
                    ligand=ligand,
                    host=str(rule_raw["host"]).strip(),
                    shared_neighbor=str(
                        rule_raw["shared_neighbor"]
                    ).strip(),
                    surface_angle_deg=float(
                        rule_raw.get("surface_angle_deg", 90.0)
                    ),
                    min_bridged_host_cn=int(
                        rule_raw.get("min_bridged_host_cn", 1)
                    ),
                )
            )
        except KeyError as exc:
            raise KeyError(
                f"bridging rule for {ligand} requires {exc.args[0]}"
            ) from exc
    return tuple(sorted(rules, key=lambda rule: rule.ligand))


def _parse_neighbor_signatures(raw: object) -> Dict[str, Tuple[str, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(
            "nucleation.graph_rules.allowed_neighbor_signatures must be a mapping"
        )
    parsed: Dict[str, Tuple[str, ...]] = {}
    for symbol, values in raw.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(
                "allowed_neighbor_signatures values must be sequences"
            )
        parsed[str(symbol)] = tuple(str(value) for value in values)
    return parsed


def load_nucleation_spec(
    path: str | Path, geometry_pack: str | Path | None = None
) -> NucleationSpec:
    """Load a nucleation-only YAML file and resolve its CIF beside the YAML."""

    yaml_path = Path(path)
    raw = yaml.safe_load(yaml_path.read_text()) or {}
    section_raw = raw.get("nucleation") or {}
    if not isinstance(section_raw, Mapping):
        raise TypeError("nucleation must be a mapping")
    section = dict(section_raw)
    pack_ref = (
        geometry_pack
        if geometry_pack is not None
        else section.get("geometry_pack")
    )
    # A molecular builder file may be both the run specification and its
    # geometry pack.  This keeps the chemistry, construction policy and
    # relaxation settings in one directly-runnable YAML instead of requiring
    # a small wrapper YAML whose only purpose is to point back here.
    # A driver that composes its stages with ``include:`` carries no
    # ``graph_rules`` of its own -- they arrive from an included file -- so it
    # must be recognised as an inline pack by the include list as well.
    inline_pack = (
        pack_ref is None
        and raw.get("schema_version") is not None
        and (
            isinstance(raw.get("graph_rules"), Mapping)
            or bool(raw.get("include"))
        )
    )
    if inline_pack:
        pack_ref = yaml_path
    pack_path: Optional[Path] = None
    if pack_ref is not None:
        from .geometry_pack import load_geometry_pack

        pack_path = Path(str(pack_ref))
        if not pack_path.is_absolute():
            if inline_pack:
                pack_path = yaml_path.resolve()
            else:
                base = Path.cwd() if geometry_pack is not None else yaml_path.parent
                pack_path = (base / pack_path).resolve()
        duplicate_keys = sorted(
            key
            for key in (
                "graph_rules",
                "geometry_rules",
                "require_inorganic_connected",
                "enforce_min_cn",
                "bridges_per_Cd_pair",
                "bridges_per_cd_pair",
                "contact_min_distance",
            )
            if key in section
        )
        if duplicate_keys:
            raise ValueError(
                "molecular rules belong in geometry_pack, not nucleation YAML: "
                + ", ".join(duplicate_keys)
            )
        pack = load_geometry_pack(pack_path)
        section["graph_rules"] = pack.nucleation_graph_rules_mapping()
        section["require_inorganic_connected"] = (
            pack.require_inorganic_connected
        )
        section["enforce_min_cn"] = pack.enforce_min_cn
        section["bridges_per_Cd_pair"] = (
            pack.max_shared_ligands_per_host_pair
        )
        section["geometry_defaults"] = "none"
    core_raw = section.get("core_monomer") or {}
    precursor_raw = section.get("precursor") or {}
    if not isinstance(core_raw, Mapping) or not isinstance(precursor_raw, Mapping):
        raise TypeError("core_monomer and precursor must be mappings")
    graph_rules_raw = section.get("graph_rules")
    if not isinstance(graph_rules_raw, Mapping):
        raise KeyError("nucleation YAML requires graph_rules")
    min_cn_raw = graph_rules_raw.get("min_cn")
    max_cn_raw = graph_rules_raw.get("max_cn")
    allowed_bonds_raw = graph_rules_raw.get("allowed_bonds")
    pair_rules = _parse_pair_rules(graph_rules_raw.get("pair_rules"))
    if not isinstance(min_cn_raw, Mapping):
        raise KeyError("nucleation.graph_rules requires min_cn")
    if not isinstance(max_cn_raw, Mapping):
        raise KeyError("nucleation.graph_rules requires max_cn")
    if allowed_bonds_raw is None:
        if not pair_rules:
            raise KeyError(
                "nucleation.graph_rules requires allowed_bonds or pair_rules"
            )
        allowed_bonds = tuple(
            sorted(
                rule.elements
                for rule in pair_rules.values()
                if rule.bond_allowed
            )
        )
    elif not isinstance(allowed_bonds_raw, Sequence) or isinstance(
        allowed_bonds_raw, (str, bytes)
    ):
        raise TypeError("nucleation.graph_rules.allowed_bonds must be a sequence")
    else:
        allowed_bonds = _parse_allowed_bonds(allowed_bonds_raw)
        if pair_rules:
            pair_allowed = tuple(
                sorted(
                    rule.elements
                    for rule in pair_rules.values()
                    if rule.bond_allowed
                )
            )
            if allowed_bonds != pair_allowed:
                raise ValueError(
                    "allowed_bonds conflicts with graph_rules.pair_rules"
                )
    cif_raw = raw.get("cif", section.get("cif"))
    if cif_raw is None:
        raise KeyError("nucleation YAML requires cif")
    cif_path = Path(str(cif_raw))
    if not cif_path.is_absolute():
        cif_path = (yaml_path.parent / cif_path).resolve()
    charges_raw = raw.get("charges", section.get("charges"))
    if not isinstance(charges_raw, Mapping):
        raise KeyError("nucleation YAML requires charges")
    try:
        core = CoreMonomerSpec(
            cation=str(core_raw["cation"]),
            anion=str(core_raw["anion"]),
        )
        precursor = PrecursorUnitSpec(
            center=str(precursor_raw["center"]),
            ligand=str(precursor_raw["ligand"]),
            ligand_count=int(precursor_raw["ligand_count"]),
        )
        spec = NucleationSpec(
            cif=str(cif_path),
            charges={str(key): int(value) for key, value in charges_raw.items()},
            core=core,
            precursor=precursor,
            kmax=int(section["kmax"]),
            geometry_pack=None if pack_path is None else str(pack_path),
            graph_rules=NucleationGraphRules(
                min_cn={
                    str(symbol): int(value)
                    for symbol, value in min_cn_raw.items()
                },
                max_cn={
                    str(symbol): int(value)
                    for symbol, value in max_cn_raw.items()
                },
                allowed_bonds=allowed_bonds,
                bridge_rules=_parse_bridge_rules(
                    graph_rules_raw.get("bridging")
                ),
                pair_rules=pair_rules,
                allowed_neighbor_signatures=_parse_neighbor_signatures(
                    graph_rules_raw.get("allowed_neighbor_signatures")
                ),
                max_shared_ligands_per_host_pair=int(
                    graph_rules_raw.get(
                        "max_shared_ligands_per_host_pair", 0
                    )
                ),
                bridge_cd_cd_max_distance=(
                    None
                    if graph_rules_raw.get("bridge_cd_cd_max_distance") is None
                    else float(graph_rules_raw["bridge_cd_cd_max_distance"])
                ),
                min_ring_size=_parse_min_ring_size(
                    graph_rules_raw.get("min_ring_size")
                ),
                min_bridged_host_cn=int(
                    graph_rules_raw.get("min_bridged_host_cn", 1)
                ),
                forbid_mono_se_dual_terminal=bool(
                    graph_rules_raw.get(
                        "forbid_mono_se_dual_terminal", False
                    )
                ),
                reject_closable_terminal_cd2=bool(
                    graph_rules_raw.get(
                        "reject_closable_terminal_cd2", False
                    )
                ),
                closable_terminal_cd2_distance=float(
                    graph_rules_raw.get(
                        "closable_terminal_cd2_distance", 3.50
                    )
                ),
                require_bridge_maximal=bool(
                    graph_rules_raw.get("require_bridge_maximal", False)
                ),
                forbid_cdse_cn_pairs=tuple(
                    (int(item[0]), int(item[1]))
                    for item in (
                        graph_rules_raw.get("forbid_cdse_cn_pairs") or ()
                    )
                    if isinstance(item, (list, tuple)) and len(item) == 2
                ),
                forbid_shared_cd_pair=tuple(
                    (str(x[0]), int(x[1]), str(x[2]), int(x[3]))
                    for x in (graph_rules_raw.get("forbid_shared_cd_pair") or ())
                    if len(x) == 4
                ),
                decoration_mode=str(
                    graph_rules_raw.get("decoration_mode", "graph_multiset")
                ).strip().lower()
                or "graph_multiset",
                bridge_first_p1_terminal_policy=str(
                    graph_rules_raw.get(
                        "bridge_first_p1_terminal_policy", "unrestricted"
                    )
                ).strip().lower()
                or "unrestricted",
                bridge_first_hard_max_bridges_per_cd=int(
                    graph_rules_raw.get("bridge_first_hard_max_bridges_per_cd", 2)
                ),
                bridge_first_prefer_bridges_per_cd=int(
                    graph_rules_raw.get("bridge_first_prefer_bridges_per_cd", 2)
                ),
                bridge_target_ring_closing_only=bool(
                    graph_rules_raw.get(
                        "bridge_target_ring_closing_only", True
                    )
                ),
                bridge_target_max_shared_per_pair=int(
                    graph_rules_raw.get(
                        "bridge_target_max_shared_per_pair", 2
                    )
                ),
                bridge_target_min_host_cn_cap=int(
                    graph_rules_raw.get("bridge_target_min_host_cn_cap", 2)
                ),
                bridge_target_avoid_triangles=bool(
                    graph_rules_raw.get("bridge_target_avoid_triangles", True)
                ),
                bridge_target_max_automorphisms=int(
                    graph_rules_raw.get(
                        "bridge_target_max_automorphisms", 4096
                    )
                ),
                bridge_first_max_automorphisms=int(
                    graph_rules_raw.get("bridge_first_max_automorphisms", 64)
                ),
                bridge_first_target_bridge_fraction=float(
                    graph_rules_raw.get(
                        "bridge_first_target_bridge_fraction", 0.0
                    ) or 0.0
                ),
                bridge_first_maximize_bridged_pairs=bool(
                    graph_rules_raw.get(
                        "bridge_first_maximize_bridged_pairs", False
                    )
                ),
                selection_order=str(
                    graph_rules_raw.get("selection_order", "bond_bands")
                ).strip().lower() or "bond_bands",
                selection_top_fraction=float(
                    graph_rules_raw.get("selection_top_fraction", 0.0) or 0.0
                ),
                selection_max_wiener_excess=float(
                    graph_rules_raw.get("selection_max_wiener_excess", 0.0)
                    or 0.0
                ),
                required_rings=_parse_required_rings(
                    graph_rules_raw.get("required_rings")
                ),
                min_core_edge_fraction=float(
                    graph_rules_raw.get("min_core_edge_fraction", 0.0) or 0.0
                ),
                max_core_cut_edges=int(
                    graph_rules_raw.get("max_core_cut_edges", -1)
                ),
                max_excess_cn1_cations=int(
                    graph_rules_raw.get("max_excess_cn1_cations", -1)
                ),
                forbid_mu3_host_bridge_overlap=bool(
                    graph_rules_raw.get("forbid_mu3_host_bridge_overlap", False)
                ),
                passivate_min_cd_cn_p_ge=int(
                    graph_rules_raw.get("passivate_min_cd_cn_p_ge", 100)
                ),
                passivate_min_cd_cn_k_ge=int(
                    graph_rules_raw.get("passivate_min_cd_cn_k_ge", 2)
                ),
                passivate_min_cd_cn=int(
                    graph_rules_raw.get("passivate_min_cd_cn", 3)
                ),
                ring_first_when_pattern_possible=bool(
                    graph_rules_raw.get(
                        "ring_first_when_pattern_possible", False
                    )
                ),
                ring_first_fallback_to_open=bool(
                    graph_rules_raw.get("ring_first_fallback_to_open", True)
                ),
                multi_ring_ladder=bool(
                    graph_rules_raw.get("multi_ring_ladder", True)
                ),
                ring_min_pattern_cd=tuple(
                    int(x)
                    for x in (
                        graph_rules_raw.get("ring_min_pattern_cd")
                        or (3, 3, 4)
                    )
                ),
                ring_min_pattern_se=tuple(
                    int(x)
                    for x in (
                        graph_rules_raw.get("ring_min_pattern_se")
                        or (3, 3, 3)
                    )
                ),
            ),

            geometry_rules=_parse_geometry_rules(
                section.get("geometry_rules"),
                core=core,
                precursor=precursor,
                geometry_defaults=str(
                    _policy_lookup(
                        section,
                        "geometry_defaults",
                        "zb_tetrahedral",
                        nested="lattice_policy",
                    )
                ).strip().lower(),
            ),
            inorganic_ring_length=int(
                _policy_lookup(
                    section,
                    "inorganic_ring_length",
                    6,
                    nested="lattice_policy",
                )
            ),
            geometry_defaults=str(
                _policy_lookup(
                    section,
                    "geometry_defaults",
                    "zb_tetrahedral",
                    nested="lattice_policy",
                )
            ).strip().lower(),
            terminal_motifs=str(
                _policy_lookup(
                    section,
                    "terminal_motifs",
                    "zb_mx2",
                    nested="surface_geometry",
                )
            ).strip().lower(),
            site_tolerance=float(section.get("site_tolerance", 0.20)),
            mode=str(section.get("mode", "exact")),
            shells_per_skeleton=int(section.get("shells_per_skeleton", 1)),
            shell_score_layers=int(section.get("shell_score_layers", 1)),
            shells_per_score_layer=int(
                section.get("shells_per_score_layer", 0)
            ),
            shell_enum_max_assignments=int(
                section.get("shell_enum_max_assignments", 10000)
            ),
            bond_count_scope=str(
                section.get("bond_count_scope", "all")
            ),
            exact_through_k=int(section.get("exact_through_k", 3)),
            core_growth_policy=str(
                section.get("core_growth_policy", "all")
            ).strip().lower(),
            compact_from_k=int(section.get("compact_from_k", 3)),
            passivation_ring_policy=str(
                section.get("passivation_ring_policy", "none")
            ).strip().lower(),
            ring_lengths=_parse_ring_lengths(section.get("ring_lengths")),
            discarded_through_k=int(
                section.get("discarded_through_k", 2)
            ),
            p_skeleton_beam=int(section.get("p_skeleton_beam", 0)),
            p_beam_from_k=int(section.get("p_beam_from_k", 4)),
            p_beam_rank=str(
                section.get("p_beam_rank", "fused_rings")
            ).strip().lower(),
            fused_chair_from_k=int(
                section.get("fused_chair_from_k", 6)
            ),
            fused_chair_mode=str(
                section.get("fused_chair_mode", "off")
            ).strip().lower(),
            retain_score_layers=int(
                section.get("retain_score_layers", 1)
            ),
            retain_max_per_bin=int(
                section.get("retain_max_per_bin", 0)
            ),
            growth_max_parents_per_bin=int(
                section.get("growth_max_parents_per_bin", 0)
            ),
            core_growth_occupation=str(
                section.get("core_growth_occupation", "bare")
            ).strip().lower(),
            continuous_decoration=bool(
                section.get(
                    "continuous_decoration",
                    # Default on when decorated growth is requested.
                    str(section.get("core_growth_occupation", "bare"))
                    .strip()
                    .lower()
                    == "decorated",
                )
            ),
            monomer_p_values=_parse_monomer_p_values(
                section.get("monomer_p_values")
            ),
            seed_p=(
                None
                if section.get("seed_p") is None
                else int(section["seed_p"])
            ),
            seed_p_window=int(section.get("seed_p_window", 0)),
            parent_p_mode=str(
                section.get("parent_p_mode", "all_retained")
            ).strip().lower(),
            p_ladder_mode=str(
                section.get("p_ladder_mode", "inherited_plus")
            ).strip().lower(),
            k_growth_max_shed=int(
                section.get("k_growth_max_shed", 0)
            ),
            k_growth_max_add=int(
                section.get("k_growth_max_add", -1)
            ),
            p_surf_beta=float(section.get("p_surf_beta", 0.0)),
            shed_alpha=float(section.get("shed_alpha", 1.0)),
            require_inorganic_connected=bool(
                section.get("require_inorganic_connected", False)
            ),
            bridges_per_cd_pair=int(
                section.get(
                    "bridges_per_Cd_pair",
                    section.get("bridges_per_cd_pair", 0),
                )
            ),
            enforce_min_cn=bool(section.get("enforce_min_cn", False)),
            contact_min_distance=_parse_contact_min_distance(
                section.get("contact_min_distance")
            ),
        )
    except KeyError as exc:
        raise KeyError(f"missing nucleation setting: {exc.args[0]}") from exc
    _validate_spec(spec)
    return spec


def _spec_run_fingerprint(spec: NucleationSpec) -> Dict[str, object]:
    """Stable identity of a nucleation run for checkpoint resume.

    ``kmax`` is intentionally omitted so a finished map can be extended by
    raising kmax and restarting.  Everything that changes the (k,p) chemistry
    or ranking must match.
    """

    cif_path = Path(spec.cif)
    cif_digest = ""
    if cif_path.is_file():
        cif_digest = hashlib.sha256(cif_path.read_bytes()).hexdigest()
    return {
        "checkpoint_schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "cif": str(cif_path.resolve()) if cif_path.exists() else str(cif_path),
        "cif_sha256": cif_digest,
        "charges": dict(sorted(spec.charges.items())),
        "core": {
            "cation": spec.core.cation,
            "anion": spec.core.anion,
        },
        "precursor": {
            "center": spec.precursor.center,
            "ligand": spec.precursor.ligand,
            "ligand_count": spec.precursor.ligand_count,
        },
        "mode": spec.mode,
        "shells_per_skeleton": int(spec.shells_per_skeleton),
        "shell_score_layers": int(spec.shell_score_layers),
        "shells_per_score_layer": int(spec.shells_per_score_layer),
        "shell_enum_max_assignments": int(spec.shell_enum_max_assignments),
        "bond_count_scope": spec.bond_count_scope,
        "exact_through_k": spec.exact_through_k,
        "core_growth_policy": spec.core_growth_policy,
        "compact_from_k": spec.compact_from_k,
        "inorganic_ring_length": spec.inorganic_ring_length,
        "geometry_defaults": spec.geometry_defaults,
        "terminal_motifs": spec.terminal_motifs,
        "passivation_ring_policy": spec.passivation_ring_policy,
        "ring_lengths": list(spec.ring_lengths),
        "discarded_through_k": spec.discarded_through_k,
        "p_skeleton_beam": spec.p_skeleton_beam,
        "p_beam_from_k": spec.p_beam_from_k,
        "p_beam_rank": spec.p_beam_rank,
        "fused_chair_from_k": spec.fused_chair_from_k,
        "fused_chair_mode": spec.fused_chair_mode,
        "retain_score_layers": spec.retain_score_layers,
        "retain_max_per_bin": spec.retain_max_per_bin,
        "growth_max_parents_per_bin": spec.growth_max_parents_per_bin,
        "core_growth_occupation": spec.core_growth_occupation,
        "continuous_decoration": bool(spec.continuous_decoration),
        "monomer_p_values": list(spec.monomer_p_values),
        "seed_p": spec.seed_p,
        "seed_p_window": spec.seed_p_window,
        "parent_p_mode": spec.parent_p_mode,
        "p_ladder_mode": spec.p_ladder_mode,
        "k_growth_max_shed": spec.k_growth_max_shed,
        "k_growth_max_add": spec.k_growth_max_add,
        "p_surf_beta": float(spec.p_surf_beta),
        "shed_alpha": float(spec.shed_alpha),
        "require_inorganic_connected": bool(spec.require_inorganic_connected),
        "bridges_per_Cd_pair": int(spec.bridges_per_cd_pair),
        "enforce_min_cn": bool(spec.enforce_min_cn),
        "contact_min_distance": dict(
            sorted(spec.contact_min_distance.items())
        ),
        "site_tolerance": spec.site_tolerance,
        "geometry_pack": spec.geometry_pack,
        "graph_rules": {
            "min_cn": dict(sorted(spec.graph_rules.min_cn.items())),
            "max_cn": dict(sorted(spec.graph_rules.max_cn.items())),
            "allowed_bonds": [list(pair) for pair in spec.graph_rules.allowed_bonds],
            "pair_rules": {
                key: {
                    "bond": "allowed" if rule.bond_allowed else "forbidden",
                    "bond_max_distance": rule.bond_max_distance,
                    "min_distance": rule.min_distance,
                }
                for key, rule in sorted(spec.graph_rules.pair_rules.items())
            },
            "bridging": {
                rule.ligand: {
                    "host": rule.host,
                    "shared_neighbor": rule.shared_neighbor,
                    "surface_angle_deg": rule.surface_angle_deg,
                    "min_bridged_host_cn": rule.min_bridged_host_cn,
                }
                for rule in spec.graph_rules.bridge_rules
            },
            "allowed_neighbor_signatures": {
                symbol: list(signatures)
                for symbol, signatures in sorted(
                    spec.graph_rules.allowed_neighbor_signatures.items()
                )
            },
            "max_shared_ligands_per_host_pair": int(
                spec.graph_rules.max_shared_ligands_per_host_pair
            ),
            "bridge_cd_cd_max_distance": (
                None
                if spec.graph_rules.bridge_cd_cd_max_distance is None
                else float(spec.graph_rules.bridge_cd_cd_max_distance)
            ),
            "min_ring_size": dict(
                sorted(spec.graph_rules.min_ring_size.items())
            ),
            "min_bridged_host_cn": int(spec.graph_rules.min_bridged_host_cn),
            "forbid_mono_se_dual_terminal": bool(
                spec.graph_rules.forbid_mono_se_dual_terminal
            ),
            "reject_closable_terminal_cd2": bool(
                spec.graph_rules.reject_closable_terminal_cd2
            ),
            "closable_terminal_cd2_distance": float(
                spec.graph_rules.closable_terminal_cd2_distance
            ),
            "require_bridge_maximal": bool(
                spec.graph_rules.require_bridge_maximal
            ),
            "forbid_cdse_cn_pairs": [
                list(pair) for pair in spec.graph_rules.forbid_cdse_cn_pairs
            ],
            "bridge_first_p1_terminal_policy": (
                spec.graph_rules.bridge_first_p1_terminal_policy
            ),
            "bridge_first_hard_max_bridges_per_cd": int(
                spec.graph_rules.bridge_first_hard_max_bridges_per_cd
            ),
            "bridge_first_prefer_bridges_per_cd": int(
                spec.graph_rules.bridge_first_prefer_bridges_per_cd
            ),
            "forbid_mu3_host_bridge_overlap": bool(
                spec.graph_rules.forbid_mu3_host_bridge_overlap
            ),
        },
    }


def _validate_spec(spec: NucleationSpec) -> None:
    if spec.kmax < 1:
        raise ValueError("kmax must be at least 1")
    bridge_span = spec.graph_rules.bridge_cd_cd_max_distance
    if bridge_span is not None and bridge_span <= 0.0:
        raise ValueError(
            "nucleation.graph_rules.bridge_cd_cd_max_distance must be positive"
        )
    if spec.site_tolerance <= 0:
        raise ValueError("site_tolerance must be positive")
    if spec.bond_count_scope not in _BOND_COUNT_SCOPES:
        raise ValueError(
            "nucleation.bond_count_scope must be one of "
            + ", ".join(sorted(_BOND_COUNT_SCOPES))
            + f"; got {spec.bond_count_scope!r}"
        )
    if spec.mode not in _NUCLEATION_MODES:
        raise ValueError(
            "nucleation.mode must be one of "
            + ", ".join(sorted(_NUCLEATION_MODES))
            + f"; got {spec.mode!r}"
        )
    if spec.exact_through_k < 1:
        raise ValueError("nucleation.exact_through_k must be at least 1")
    if spec.core_growth_policy not in _CORE_GROWTH_POLICIES:
        raise ValueError(
            "nucleation.core_growth_policy must be one of "
            + ", ".join(sorted(_CORE_GROWTH_POLICIES))
            + f"; got {spec.core_growth_policy!r}"
        )
    if spec.compact_from_k < 2:
        raise ValueError(
            "nucleation.compact_from_k must be at least 2 "
            "(destination k=1 is the seed, not a growth product)"
        )
    if int(spec.inorganic_ring_length) < 3:
        raise ValueError(
            "nucleation.inorganic_ring_length must be >= 3; "
            f"got {spec.inorganic_ring_length!r}"
        )
    if spec.geometry_defaults not in _GEOMETRY_DEFAULTS:
        raise ValueError(
            "nucleation.geometry_defaults must be one of "
            + ", ".join(sorted(_GEOMETRY_DEFAULTS))
            + f"; got {spec.geometry_defaults!r}"
        )
    if spec.terminal_motifs not in _TERMINAL_MOTIFS:
        raise ValueError(
            "nucleation.terminal_motifs must be one of "
            + ", ".join(sorted(_TERMINAL_MOTIFS))
            + f"; got {spec.terminal_motifs!r}"
        )
    if spec.passivation_ring_policy not in _PASSIVATION_RING_POLICIES:
        raise ValueError(
            "nucleation.passivation_ring_policy must be one of "
            + ", ".join(sorted(_PASSIVATION_RING_POLICIES))
            + f"; got {spec.passivation_ring_policy!r}"
        )
    if not spec.ring_lengths:
        raise ValueError("nucleation.ring_lengths must not be empty")
    if any(length < 3 for length in spec.ring_lengths):
        raise ValueError("nucleation.ring_lengths entries must be >= 3")
    if spec.discarded_through_k < 0:
        raise ValueError("nucleation.discarded_through_k must be nonnegative")
    if spec.p_skeleton_beam < 0:
        raise ValueError("nucleation.p_skeleton_beam must be nonnegative (0 = off)")
    if spec.p_beam_from_k < 1:
        raise ValueError("nucleation.p_beam_from_k must be at least 1")
    if spec.p_beam_rank not in _P_BEAM_RANKS:
        raise ValueError(
            "nucleation.p_beam_rank must be one of "
            + ", ".join(sorted(_P_BEAM_RANKS))
            + f"; got {spec.p_beam_rank!r}"
        )
    if spec.fused_chair_from_k < 0:
        raise ValueError("nucleation.fused_chair_from_k must be nonnegative")
    if spec.fused_chair_mode not in _FUSED_CHAIR_MODES:
        raise ValueError(
            "nucleation.fused_chair_mode must be one of "
            + ", ".join(sorted(_FUSED_CHAIR_MODES))
            + f"; got {spec.fused_chair_mode!r}"
        )
    if spec.retain_score_layers < 1:
        raise ValueError("nucleation.retain_score_layers must be at least 1")
    if spec.shells_per_skeleton < 1:
        raise ValueError("nucleation.shells_per_skeleton must be at least 1")
    if spec.shell_score_layers < 1:
        raise ValueError("nucleation.shell_score_layers must be at least 1")
    if spec.shells_per_score_layer < 0:
        raise ValueError(
            "nucleation.shells_per_score_layer must be nonnegative "
            "(0 = keep entire score layer)"
        )
    if spec.shell_enum_max_assignments < 0:
        raise ValueError(
            "nucleation.shell_enum_max_assignments must be nonnegative "
            "(0 = never orbit-enumerate in guided multi-shell)"
        )
    if spec.retain_max_per_bin < 0:
        raise ValueError(
            "nucleation.retain_max_per_bin must be nonnegative (0 = unlimited)"
        )
    if spec.growth_max_parents_per_bin < 0:
        raise ValueError(
            "nucleation.growth_max_parents_per_bin must be nonnegative "
            "(0 = use all retained)"
        )
    if spec.core_growth_occupation not in _CORE_GROWTH_OCCUPATIONS:
        raise ValueError(
            "nucleation.core_growth_occupation must be one of "
            + ", ".join(sorted(_CORE_GROWTH_OCCUPATIONS))
            + f"; got {spec.core_growth_occupation!r}"
        )
    if spec.seed_p is not None and spec.seed_p < 0:
        raise ValueError("nucleation.seed_p must be nonnegative when set")
    if spec.seed_p_window < 0:
        raise ValueError("nucleation.seed_p_window must be nonnegative")
    if spec.parent_p_mode not in _PARENT_P_MODES:
        raise ValueError(
            "nucleation.parent_p_mode must be one of "
            + ", ".join(sorted(_PARENT_P_MODES))
            + f"; got {spec.parent_p_mode!r}"
        )
    if spec.parent_p_mode == "seed_band" and spec.seed_p is None:
        raise ValueError(
            "nucleation.parent_p_mode=seed_band requires seed_p to be set"
        )
    if spec.p_ladder_mode not in _P_LADDER_MODES:
        raise ValueError(
            "nucleation.p_ladder_mode must be one of "
            + ", ".join(sorted(_P_LADDER_MODES))
            + f"; got {spec.p_ladder_mode!r}"
        )
    if any(v < 0 for v in spec.monomer_p_values):
        raise ValueError("nucleation.monomer_p_values entries must be nonnegative")
    if spec.k_growth_max_shed < 0:
        raise ValueError("nucleation.k_growth_max_shed must be nonnegative")
    if float(spec.p_surf_beta) < 0.0:
        raise ValueError("nucleation.p_surf_beta must be nonnegative (0 = off)")
    if float(spec.shed_alpha) < 0.0:
        raise ValueError("nucleation.shed_alpha must be nonnegative")
    if int(spec.bridges_per_cd_pair) < 0:
        raise ValueError(
            "nucleation.bridges_per_Cd_pair must be nonnegative "
            "(0 = unlimited)"
        )
    if spec.graph_rules.max_shared_ligands_per_host_pair < 0:
        raise ValueError(
            "graph_rules.max_shared_ligands_per_host_pair must be nonnegative"
        )
    if any(float(value) <= 0.0 for value in spec.contact_min_distance.values()):
        raise ValueError("nucleation.contact_min_distance values must be positive")
    if spec.graph_rules.pair_rules:
        active = sorted(
            {
                spec.core.cation,
                spec.core.anion,
                spec.precursor.center,
                spec.precursor.ligand,
            }
        )
        expected = {
            _pair_rule_key(left, right)
            for index, left in enumerate(active)
            for right in active[index:]
        }
        missing_pairs = sorted(expected.difference(spec.graph_rules.pair_rules))
        if missing_pairs:
            raise ValueError(
                "nucleation.graph_rules.pair_rules is incomplete; missing: "
                + ", ".join(missing_pairs)
            )
        for key, rule in spec.graph_rules.pair_rules.items():
            if rule.bond_allowed:
                if rule.bond_max_distance is None or rule.bond_max_distance <= 0.0:
                    raise ValueError(
                        f"allowed pair {key} requires positive bond_max_distance"
                    )
            elif rule.min_distance is None or rule.min_distance <= 0.0:
                raise ValueError(
                    f"forbidden pair {key} requires positive min_distance"
                )
        overlap = set(spec.contact_min_distance).intersection(
            spec.graph_rules.pair_rules
        )
        if overlap:
            raise ValueError(
                "contact_min_distance conflicts with pair_rules for: "
                + ", ".join(sorted(overlap))
            )
    if spec.k_growth_max_add < -1:
        raise ValueError(
            "nucleation.k_growth_max_add must be >= -1 (-1 = unlimited ladder)"
        )
    if spec.precursor.ligand_count < 1:
        raise ValueError("precursor.ligand_count must be at least 1")
    required = {
        spec.core.cation,
        spec.core.anion,
        spec.precursor.center,
        spec.precursor.ligand,
    }
    for symbol, signatures in spec.graph_rules.allowed_neighbor_signatures.items():
        if symbol not in required:
            raise ValueError(
                f"allowed_neighbor_signatures references unused species: {symbol}"
            )
        if not signatures:
            raise ValueError(
                f"allowed_neighbor_signatures for {symbol} must not be empty"
            )
    missing = sorted(required.difference(spec.charges))
    if missing:
        raise ValueError(f"missing formal charges for: {', '.join(missing)}")
    core_charge = (
        spec.charges[spec.core.cation] + spec.charges[spec.core.anion]
    )
    if core_charge != 0:
        raise ValueError(
            "declared core monomer must be charge neutral: "
            f"q({spec.core.cation}) + q({spec.core.anion}) = {core_charge}"
        )
    precursor_charge = (
        spec.charges[spec.precursor.center]
        + int(spec.precursor.ligand_count)
        * spec.charges[spec.precursor.ligand]
    )
    if precursor_charge != 0:
        raise ValueError(
            "declared precursor package must be charge neutral: "
            f"q({spec.precursor.center}) + "
            f"{spec.precursor.ligand_count} q({spec.precursor.ligand}) = "
            f"{precursor_charge}"
        )
    missing_cn = sorted(required.difference(spec.graph_rules.max_cn))
    if missing_cn:
        raise ValueError(
            "missing nucleation.graph_rules.max_cn for: "
            + ", ".join(missing_cn)
        )
    missing_min_cn = sorted(required.difference(spec.graph_rules.min_cn))
    if missing_min_cn:
        raise ValueError(
            "missing nucleation.graph_rules.min_cn for: "
            + ", ".join(missing_min_cn)
        )
    unknown_cn = sorted(
        (
            set(spec.graph_rules.max_cn)
            | set(spec.graph_rules.min_cn)
        ).difference(required)
    )
    if unknown_cn:
        raise ValueError(
            "nucleation graph CN rules contain unused species: "
            + ", ".join(unknown_cn)
        )
    geometry_species = (
        set(spec.geometry_rules.by_cn) | set(spec.geometry_rules.all_cn)
    )
    unknown_geometry = sorted(geometry_species.difference(required))
    if unknown_geometry:
        raise ValueError(
            "nucleation geometry rules contain unused species: "
            + ", ".join(unknown_geometry)
        )
    for symbol, rules in spec.geometry_rules.by_cn.items():
        too_large = sorted(
            coordination
            for coordination in rules
            if coordination > spec.graph_rules.max_cn[symbol]
        )
        if too_large:
            raise ValueError(
                f"geometry CN exceeds maximum CN for {symbol}: "
                + ", ".join(str(value) for value in too_large)
            )
    nonpositive = sorted(
        symbol
        for symbol, maximum in spec.graph_rules.max_cn.items()
        if maximum < 1
    )
    if nonpositive:
        raise ValueError(
            "nucleation maximum CN must be positive for: "
            + ", ".join(nonpositive)
        )
    negative_minimum = sorted(
        symbol
        for symbol, minimum in spec.graph_rules.min_cn.items()
        if minimum < 0
    )
    if negative_minimum:
        raise ValueError(
            "nucleation minimum CN must be nonnegative for: "
            + ", ".join(negative_minimum)
        )
    inverted = sorted(
        symbol
        for symbol in required
        if spec.graph_rules.min_cn[symbol]
        > spec.graph_rules.max_cn[symbol]
    )
    if inverted:
        raise ValueError(
            "nucleation minimum CN exceeds maximum CN for: "
            + ", ".join(inverted)
        )
    allowed = set(spec.graph_rules.allowed_bonds)
    for pair in allowed:
        unknown = set(pair).difference(required)
        if unknown:
            raise ValueError(
                "allowed bond references unused species: "
                + ", ".join(sorted(unknown))
            )
    core_pair = _bond_key(spec.core.cation, spec.core.anion)
    if core_pair not in allowed:
        raise ValueError(
            "allowed_bonds must include the core monomer pair "
            f"{spec.core.cation}-{spec.core.anion}"
        )
    precursor_pair = _bond_key(
        spec.precursor.center, spec.precursor.ligand
    )
    if precursor_pair not in allowed:
        raise ValueError(
            "allowed_bonds must include the precursor pair "
            f"{spec.precursor.center}-{spec.precursor.ligand}"
        )
    for rule in spec.graph_rules.bridge_rules:
        bridge_species = {rule.ligand, rule.host, rule.shared_neighbor}
        unknown = sorted(bridge_species.difference(required))
        if unknown:
            raise ValueError(
                "bridging rule references unused species: "
                + ", ".join(unknown)
            )
        if rule.ligand != spec.precursor.ligand:
            raise ValueError(
                "bridging rule ligand must match precursor.ligand: "
                f"{rule.ligand}"
            )
        if _bond_key(rule.ligand, rule.host) not in allowed:
            raise ValueError(
                "bridging host-ligand pair must be allowed: "
                f"{rule.host}-{rule.ligand}"
            )
        if _bond_key(rule.host, rule.shared_neighbor) not in allowed:
            raise ValueError(
                "bridging host-shared-neighbor pair must be allowed: "
                f"{rule.host}-{rule.shared_neighbor}"
            )
        if spec.graph_rules.max_cn[rule.ligand] < 2:
            raise ValueError(
                f"bridging ligand {rule.ligand} requires maximum CN >= 2"
            )
        if rule.min_bridged_host_cn < 1:
            raise ValueError(
                f"bridging min_bridged_host_cn for {rule.ligand} must be at "
                "least 1"
            )
        if rule.min_bridged_host_cn > spec.graph_rules.max_cn[rule.host]:
            raise ValueError(
                f"bridging min_bridged_host_cn for {rule.ligand} exceeds "
                f"maximum CN of host {rule.host}, so no bridge could ever form"
            )
        if not 0.0 < rule.surface_angle_deg < 180.0:
            raise ValueError(
                "bridging surface_angle_deg must be between 0 and 180"
            )
    if not Path(spec.cif).is_file():
        raise FileNotFoundError(spec.cif)

__all__ = [
    'is_nucleation_yaml',
    '_bond_key',
    '_parse_allowed_bonds',
    '_parse_geometry_rules',
    '_parse_ring_lengths',
    '_parse_monomer_p_values',
    '_parse_bridge_rules',
    'load_nucleation_spec',
    '_spec_run_fingerprint',
    '_validate_spec',
    '_GEOMETRY_TEMPLATES',
    '_NUCLEATION_MODES',
    '_BOND_COUNT_SCOPES',
    '_CORE_GROWTH_POLICIES',
    '_PASSIVATION_RING_POLICIES',
    '_P_BEAM_RANKS',
    '_FUSED_CHAIR_MODES',
    '_CORE_GROWTH_OCCUPATIONS',
    '_PARENT_P_MODES',
    '_P_LADDER_MODES',
    '_CHECKPOINT_SCHEMA_VERSION',
    '_SUB_MAXIMUM_FALLBACK_LIMIT',
]
