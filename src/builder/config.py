# src/builder/config.py
from __future__ import annotations
import argparse
import re
import sys
from typing import Tuple, List, Dict

try:
    import yaml
except ImportError:
    sys.exit("pip install pyyaml")

from .nc_types import (
    Config, Facet,
    PassivationSpec, MaterialSpec, BuildSpec, AlignSpec, StrainPolicy,
    FacetReconstructionSpec, StackSpec,
    NeutralLigandPass, NeutralLigandPostTreatSpec,
    LigandExchangePass, LigandExchangePostTreatSpec,
    SurfaceReconstructionSpec, PostTreatmentSpec,
)

# -------------------- CLI --------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nc-builder",
        description="Coordination-aware Wulff-cut nanocrystal builder with surface passivation."
    )
    p.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="Stack mode: YAML recipe only. Single-material: CIF file then YAML recipe.",
    )
    p.add_argument("-r", "--radius", type=float, default=None,
                   help="Target outer Wulff radius (Å)")
    p.add_argument("-size-unit-cells", "--size-unit-cells", type=float, default=None,
                   help="Material-scaled size: set Wulff radius to this many shortest lattice vectors; floats like 1.5 are allowed.")

    p.add_argument("-o", "--out", default="nanocrystal.xyz", help="Output XYZ path (final)")
    p.add_argument("--write-all", action="store_true", help="Also write *_cut.xyz")
    p.add_argument("--center", action="store_true", help="Center the particle at the COM before writing")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")

    p.add_argument(
        "--positive-q-mode",
        choices=["remove", "add"],
        default="remove",
        help="Strategy for neutralizing Q > 0: "
             "'remove' cations (default; cation-deficient surface). "
             "'add' more anions (ligand-rich surface)."
    )

    pre = p.add_argument_group("preprocessing")
    pre.add_argument(
        "--prune-min-cn", type=int, default=2,
        help="Remove atoms with CN < this before facet detection (default: 2 → removes CN=1)."
    )
    pre.add_argument(
        "--prune-passes", type=int, default=10,
        help="Maximum pruning passes to reach stability (default: 10)."
    )
    pre.add_argument(
        "--no-prune-mono", dest="prune_mono", action="store_false",
        help="Disable the pre-pass pruning step."
    )
    pre.add_argument(
        "--prepass-mode",
        choices=["standard", "role-aware"],
        default=None,
        help="Prepass cleaning mode: 'standard' (original CN<=2 pruning, default) or 'role-aware'."
    )
    pre.add_argument(
        "--prepass-min-cn-terrace",
        type=int,
        default=None,
        help="Minimum coordination number to keep for terrace atoms in role-aware mode (default: 3)."
    )
    pre.add_argument(
        "--prepass-min-cn-edge",
        type=int,
        default=None,
        help="Minimum coordination number to keep for edge atoms in role-aware mode (default: 3)."
    )
    pre.add_argument(
        "--prepass-min-cn-vertex",
        type=int,
        default=None,
        help="Minimum coordination number to keep for vertex atoms in role-aware mode (default: 1)."
    )
    p.set_defaults(prune_mono=True)

    # Wulff / symmetry
    w = p.add_argument_group("wulff / symmetry")
    # Tri-state via None so YAML can decide if CLI not set
    try:
        import argparse as _ap
        w.add_argument(
            "--proper-rotations-only",
            action=_ap.BooleanOptionalAction,
            default=None,
            help="Use only proper rotations (det=+1) for seed expansion; if omitted, YAML decides."
        )
    except Exception:
        w.add_argument("--proper-rotations-only", action="store_true", default=True,
                       help="Use only proper rotations (det=+1) for seed expansion (default on).")

    # Shape / anisotropy (global default for single-material; ignored in stack if YAML provides per-material)
    shape = p.add_argument_group("shape")
    shape.add_argument(
        "--aspect", type=float, nargs=3, metavar=("AX", "AY", "AZ"),
        default=None,
        help="Anisotropy along lattice a,b,c axes (default from YAML or 1 1 1). "
             "Examples: platelet 1 1 0.3; rod 0.7 0.7 2.0"
    )

    # Core lattice fit / interface strain (optional)
    strain = p.add_argument_group("core lattice fit / interface strain")
    strain.add_argument(
        "--core-lattice-fit",
        action="store_true",
        help="After relabeling, shrink inner core atoms with an affine map to the core CIF lattice, "
             "and apply a smooth strain blend near the core boundary. "
             "Enabled by default in stack mode unless --no-core-lattice-fit is set."
    )
    strain.add_argument(
        "--no-core-lattice-fit",
        action="store_true",
        help="Disable core lattice affine fit in stack mode (fit is on by default for core-shell)."
    )
    strain.add_argument(
        "--core-strain-width",
        type=float,
        default=2.0,
        help="Width (Å) of the blending zone inside the core boundary (default: 2.0 Å)."
    )
    strain.add_argument(
        "--core-center",
        choices=["origin", "com"],
        default="com",
        help="Reference point for the affine map center: 'com' (inner-core COM) or 'origin' (0,0,0)."
    )

    # Facet scan (pre-run diagnostic)
    scan = p.add_argument_group("facet scan")
    scan.add_argument("--scan-facets", action="store_true",
                      help="Scan symmetry-distinct facets and classify polar vs non-polar using user charges.")
    scan.add_argument("--scan-max-index", type=int, default=2,
                      help="Max |h|,|k|,|l| for facet scan (default: 2).")
    scan.add_argument("--scan-slab-size", type=float, default=18.0,
                      help="Min slab thickness in Å for scan (default: 18).")
    scan.add_argument("--scan-vacuum-size", type=float, default=20.0,
                      help="Min vacuum thickness in Å for scan (default: 20).")
    scan.add_argument("--scan-shifts", type=int, default=8,
                      help="Number of evenly-spaced terminations to sample with unsymmetrized slabs (default: 8).")

    return p

# -------------------- YAML helpers --------------------

def _parse_hkl(val) -> tuple[int, int, int]:
    """
    Parse a Miller index (hkl) from many notations, including compact signed per-digit:
      "111", "100", "-1-1-1", "-100", "1-10", "11-1",
      "1 1 1", "1,1,1", "(1 1 1)", "[1, 0, 0]",
      111, [1,1,1], (-1,-1,-1)
    """
    # list/tuple
    if isinstance(val, (list, tuple)) and len(val) == 3:
        h, k, l = (int(val[0]), int(val[1]), int(val[2]))
        if (h, k, l) == (0, 0, 0):
            raise ValueError("hkl cannot be (0,0,0)")
        return (h, k, l)

    # plain int like 111 or 100
    if isinstance(val, int):
        s = f"{abs(val)}"
        if not s.isdigit() or len(s) != 3:
            raise ValueError(f"Ambiguous integer Miller index: {val!r} (expect 3 digits)")
        sign = -1 if val < 0 else 1
        return (sign * int(s[0]), int(s[1]), int(s[2]))

    # string
    if isinstance(val, str):
        s = val.strip()
        # 1) Try three full signed integers when separated by non-sign delimiters.
        # Compact strings like "-1-1-1" are handled by the per-digit branch below.
        nums = re.findall(r'(?<![\d+-])[+-]?\d+(?!\d)', s)
        if len(nums) == 3 and re.search(r'[\s,;\[\]()]', s):
            h, k, l = (int(nums[0]), int(nums[1]), int(nums[2]))
            if (h, k, l) == (0, 0, 0):
                raise ValueError("hkl cannot be (0,0,0)")
            return (h, k, l)

        # 2) Compact per-digit with optional signs before each digit ("-100", "1-10", "11-1", "-1-1-1")
        s2 = re.sub(r'[^0-9+-]', '', s)
        tokens = re.findall(r'[+-]?\d', s2)
        if len(tokens) == 3 and all(len(t.replace('+','').replace('-','')) == 1 for t in tokens):
            h, k, l = (int(tokens[0]), int(tokens[1]), int(tokens[2]))
            if (h, k, l) == (0, 0, 0):
                raise ValueError("hkl cannot be (0,0,0)")
            return (h, k, l)

        # 3) Plain compact unsigned
        s3 = re.sub(r'[()\[\]\s,;]', '', s)
        if s3.isdigit() and len(s3) == 3:
            return (int(s3[0]), int(s3[1]), int(s3[2]))

        raise ValueError(f"Invalid hkl format: {val!r}")

    raise TypeError(f"Unsupported hkl type: {type(val).__name__}")

def _parse_aspect(val) -> Tuple[float, float, float]:
    """
    Parse aspect multipliers along lattice a,b,c axes.
    Accepts list/tuple, mapping {a,b,c} or {x,y,z}, or "ax ay az".
    """
    if val is None:
        return (1.0, 1.0, 1.0)
    if isinstance(val, (list, tuple)) and len(val) == 3:
        ax, ay, az = (float(val[0]), float(val[1]), float(val[2]))
        return (ax, ay, az)
    if isinstance(val, dict):
        keys = {k.lower(): float(v) for k, v in val.items()}
        def get3(a, b, c): return (keys[a], keys[b], keys[c])
        if all(k in keys for k in ("a", "b", "c")): return get3("a", "b", "c")
        if all(k in keys for k in ("x", "y", "z")): return get3("x", "y", "z")
        raise ValueError(f"shape.aspect mapping must have a/b/c or x/y/z: {val!r}")
    if isinstance(val, str):
        toks = [t for t in re.split(r"[,\s]+", val.strip()) if t]
        if len(toks) == 3:
            return (float(toks[0]), float(toks[1]), float(toks[2]))
        raise ValueError(f"Cannot parse shape.aspect string: {val!r}")
    raise TypeError(f"Unsupported shape.aspect type: {type(val).__name__}")


def _parse_size_unit_cells(val) -> Tuple[float, float, float]:
    if val is None:
        raise ValueError("size_unit_cells cannot be None")
    if isinstance(val, (int, float)):
        x = float(val)
        return (x, x, x)
    if isinstance(val, (list, tuple)) and len(val) == 3:
        return (float(val[0]), float(val[1]), float(val[2]))
    if isinstance(val, str):
        toks = [t for t in re.split(r"[,\s]+", val.strip("[]() ")) if t]
        if len(toks) == 1:
            x = float(toks[0])
            return (x, x, x)
        if len(toks) == 3:
            return (float(toks[0]), float(toks[1]), float(toks[2]))
    raise TypeError("size_unit_cells must be a number or three values")

# -------------------- YAML → Config --------------------
def _normalize_twins(raw):
    """
    Accept twins: [ {...}, {...} ]  or twins: {...}  or missing/None.
    Return list[dict] or None.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        return [dict(x) for x in raw]
    return [dict(raw)]


def _parse_optional_bool(raw, *, field: str) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        val = raw.strip().lower()
        if val in {"true", "yes", "on", "1"}:
            return True
        if val in {"false", "no", "off", "0"}:
            return False
    raise TypeError(f"{field} must be a boolean")


def _parse_facet_reconstruction(raw) -> FacetReconstructionSpec:
    if raw is None:
        return FacetReconstructionSpec()
    if not isinstance(raw, dict):
        raise TypeError("facet_reconstruction must be a mapping")

    enabled = _parse_optional_bool(raw.get("enabled"), field="facet_reconstruction.enabled")
    if enabled is False:
        return FacetReconstructionSpec()

    facets_raw = raw.get("facets") or []
    if not isinstance(facets_raw, list):
        raise TypeError("facet_reconstruction.facets must be a list")

    facets: List[tuple] = []
    for entry in facets_raw:
        if isinstance(entry, dict) and "hkl" in entry:
            hkl = _parse_hkl(entry["hkl"])
        elif isinstance(entry, (list, tuple)) and len(entry) == 3:
            hkl = (int(entry[0]), int(entry[1]), int(entry[2]))
        else:
            raise TypeError(
                f"facet_reconstruction.facets entry must be a dict with 'hkl' key: {entry!r}"
            )
        facets.append(hkl)

    cation_ligand = raw.get("cation_ligand")
    cation_ligand_charge = raw.get("cation_ligand_charge")
    if cation_ligand is not None and cation_ligand_charge is None:
        raise ValueError(
            "facet_reconstruction.cation_ligand_charge is required when cation_ligand is set"
        )

    return FacetReconstructionSpec(
        enabled=enabled if enabled is not None else bool(facets),
        facets=tuple(facets),
        cation_ligand=str(cation_ligand) if cation_ligand else None,
        cation_ligand_charge=int(cation_ligand_charge) if cation_ligand_charge is not None else None,
    )


def _parse_neutral_ligands(raw) -> NeutralLigandPostTreatSpec:
    """Parse the optional neutral_ligands: block from a YAML passivation dict."""
    if raw is None:
        return NeutralLigandPostTreatSpec()  # disabled by default
    if not isinstance(raw, dict):
        raise TypeError("neutral_ligands must be a mapping")

    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return NeutralLigandPostTreatSpec(enabled=False)

    # -- Parse passes list --
    passes_raw = raw.get("passes") or []
    if not isinstance(passes_raw, list):
        raise TypeError("neutral_ligands.passes must be a list")

    valid_targets = {"anion", "cation", "both"}
    valid_distributions = {"random", "segmented", "uniform"}
    passes = []
    for i, entry in enumerate(passes_raw):
        if not isinstance(entry, dict):
            raise TypeError(f"neutral_ligands.passes[{i}] must be a mapping")
        target = str(entry.get("target", "")).strip().lower()
        if target not in valid_targets:
            raise ValueError(
                f"neutral_ligands.passes[{i}].target must be one of {valid_targets!r}, got {target!r}"
            )
        smiles = str(entry.get("smiles", "")).strip()
        if not smiles:
            raise ValueError(f"neutral_ligands.passes[{i}].smiles is required")
        distribution = str(entry.get("distribution", "random")).strip().lower()
        if distribution not in valid_distributions:
            raise ValueError(
                f"neutral_ligands.passes[{i}].distribution must be one of {valid_distributions!r}"
            )
        ratio = float(entry.get("ratio", 1.0))
        if not (0.0 < ratio <= 1.0):
            raise ValueError(
                f"neutral_ligands.passes[{i}].ratio must be in (0, 1], got {ratio}"
            )
        passes.append(NeutralLigandPass(
            target=target,
            smiles=smiles,
            distribution=distribution,
            ratio=ratio,
        ))

    # -- Shared placement options --
    ff = str(raw.get("ff", "uff")).strip().lower()
    if ff not in {"uff", "mmff"}:
        raise ValueError(f"neutral_ligands.ff must be 'uff' or 'mmff', got {ff!r}")
    refinement_passes = int(raw.get("refinement_passes", 2))
    sterics_mode = str(raw.get("sterics_mode", "vdw")).strip().lower()
    if sterics_mode not in {"heavy", "all", "vdw"}:
        raise ValueError(f"neutral_ligands.sterics_mode must be 'heavy', 'all', or 'vdw'")
    offset_out = float(raw.get("offset_out", 0.5))
    seed = int(raw.get("seed", 1337))

    return NeutralLigandPostTreatSpec(
        enabled=True,
        passes=tuple(passes),
        ff=ff,
        refinement_passes=refinement_passes,
        sterics_mode=sterics_mode,
        offset_out=offset_out,
        seed=seed,
    )


def _parse_ligand_exchange(raw) -> LigandExchangePostTreatSpec:
    if raw is None:
        return LigandExchangePostTreatSpec()
    if not isinstance(raw, dict):
        raise TypeError("post_treatment.ligand_exchange must be a mapping")

    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return LigandExchangePostTreatSpec(enabled=False)

    passes_raw = raw.get("passes") or []
    if not isinstance(passes_raw, list):
        raise TypeError("post_treatment.ligand_exchange.passes must be a list")

    valid_distributions = {"random", "segmented", "uniform"}
    passes = []
    for i, entry in enumerate(passes_raw):
        if not isinstance(entry, dict):
            raise TypeError(f"ligand_exchange.passes[{i}] must be a mapping")
        replace = str(entry.get("replace", "")).strip()
        if not replace:
            raise ValueError(f"ligand_exchange.passes[{i}].replace is required")
        charge = int(entry.get("charge", 0))
        if charge not in {-1, +1}:
            raise ValueError(f"ligand_exchange.passes[{i}].charge currently supports -1 or +1")
        smiles_raw = entry.get("smiles")
        if isinstance(smiles_raw, str):
            smiles = (smiles_raw.strip(),)
        elif isinstance(smiles_raw, (list, tuple)):
            smiles = tuple(str(s).strip() for s in smiles_raw)
        else:
            raise TypeError(f"ligand_exchange.passes[{i}].smiles must be a string or list")
        smiles = tuple(s for s in smiles if s)
        if not smiles:
            raise ValueError(f"ligand_exchange.passes[{i}].smiles is required")
        distribution = str(entry.get("distribution", "random")).strip().lower()
        if distribution not in valid_distributions:
            raise ValueError(
                f"ligand_exchange.passes[{i}].distribution must be one of {valid_distributions!r}"
            )
        ratio = float(entry.get("ratio", 1.0))
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ligand_exchange.passes[{i}].ratio must be in (0, 1], got {ratio}")
        passes.append(LigandExchangePass(
            replace=replace,
            charge=charge,
            smiles=smiles,
            distribution=distribution,
            ratio=ratio,
        ))

    ff = str(raw.get("ff", "uff")).strip().lower()
    if ff not in {"uff", "mmff"}:
        raise ValueError(f"ligand_exchange.ff must be 'uff' or 'mmff', got {ff!r}")
    sterics_mode = str(raw.get("sterics_mode", "vdw")).strip().lower()
    if sterics_mode not in {"heavy", "all", "vdw"}:
        raise ValueError("ligand_exchange.sterics_mode must be 'heavy', 'all', or 'vdw'")

    return LigandExchangePostTreatSpec(
        enabled=True,
        passes=tuple(passes),
        ff=ff,
        refinement_passes=int(raw.get("refinement_passes", 2)),
        sterics_mode=sterics_mode,
        seed=int(raw.get("seed", 1337)),
    )


def _parse_surface_reconstruction(raw, *, default_ligand: str) -> SurfaceReconstructionSpec:
    if raw is None:
        return SurfaceReconstructionSpec()
    if not isinstance(raw, dict):
        raise TypeError("post_treatment.surface_reconstruction must be a mapping")

    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return SurfaceReconstructionSpec(enabled=False)

    facets_raw = raw.get("facets", "auto")
    auto_facets = facets_raw is None or (isinstance(facets_raw, str) and facets_raw.strip().lower() == "auto")
    facets: List[tuple[int, int, int]] = []
    if not auto_facets:
        if not isinstance(facets_raw, list):
            raise TypeError("post_treatment.surface_reconstruction.facets must be 'auto' or a list")
        for entry in facets_raw:
            if isinstance(entry, dict) and "hkl" in entry:
                facets.append(_parse_hkl(entry["hkl"]))
            else:
                facets.append(_parse_hkl(entry))

    target_reduction = float(raw.get("target_reduction", 0.5))
    if not (0.0 < target_reduction <= 1.0):
        raise ValueError("post_treatment.surface_reconstruction.target_reduction must be in (0, 1]")

    min_sep_raw = raw.get("min_separation", None)
    min_separation = None
    if min_sep_raw is not None and not (isinstance(min_sep_raw, str) and min_sep_raw.strip().lower() == "auto"):
        min_separation = float(min_sep_raw)
        if min_separation <= 0.0:
            raise ValueError("post_treatment.surface_reconstruction.min_separation must be positive or 'auto'")

    distribution = str(raw.get("distribution", "fps")).strip().lower()
    if distribution != "fps":
        raise ValueError("post_treatment.surface_reconstruction.distribution currently supports only 'fps'")

    return SurfaceReconstructionSpec(
        enabled=True,
        ligand=str(raw.get("ligand", default_ligand)),
        facets=tuple(facets),
        auto_facets=auto_facets,
        target_reduction=target_reduction,
        min_separation=min_separation,
        distribution=distribution,
        seed=int(raw.get("seed", 1337)),
    )


def _surface_reconstruction_from_legacy(raw, *, default_ligand: str) -> SurfaceReconstructionSpec:
    """Compatibility shim for the old top-level facet_reconstruction block."""
    if raw is None:
        return SurfaceReconstructionSpec()
    if not isinstance(raw, dict):
        return SurfaceReconstructionSpec()
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return SurfaceReconstructionSpec()
    sr_raw = {
        "enabled": True,
        "ligand": raw.get("ligand", default_ligand),
        "facets": raw.get("facets", "auto"),
        "target_reduction": raw.get("target_reduction", 0.5),
        "min_separation": raw.get("min_separation", None),
        "distribution": raw.get("distribution", "fps"),
        "seed": raw.get("seed", 1337),
    }
    return _parse_surface_reconstruction(sr_raw, default_ligand=default_ligand)


def _parse_post_treatment(raw, *, pass_cfg: dict, default_ligand: str, legacy_recon_raw) -> PostTreatmentSpec:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("post_treatment must be a mapping")

    neutral_raw = raw.get("neutral_ligands", raw.get("neutral-ligands"))
    if neutral_raw is None:
        neutral_raw = pass_cfg.get("neutral_ligands")
    neutral_ligands = _parse_neutral_ligands(neutral_raw)

    exchange_raw = raw.get("ligand_exchange", raw.get("ligand-exchange"))
    ligand_exchange = _parse_ligand_exchange(exchange_raw)

    surface_raw = raw.get("surface_reconstruction", raw.get("surface-reconstruction"))
    surface_reconstruction = _parse_surface_reconstruction(surface_raw, default_ligand=default_ligand)
    if not surface_reconstruction.enabled:
        surface_reconstruction = _surface_reconstruction_from_legacy(
            legacy_recon_raw, default_ligand=default_ligand
        )

    return PostTreatmentSpec(
        surface_reconstruction=surface_reconstruction,
        neutral_ligands=neutral_ligands,
        ligand_exchange=ligand_exchange,
    )


def parse_yaml_config(path: str) -> Config:
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    # ---- Global passivation + charges ----
    pass_cfg = cfg.get("passivation") or {}
    lig_old = pass_cfg.get("ligand")           # legacy key
    lig_new = pass_cfg.get("anion_ligand")     # new key
    cat_new = pass_cfg.get("cation_ligand")    # optional new key
    surf_tol = float(pass_cfg.get("surf_tol", 1.0))

    # Back-compat shim: accept either 'ligand' or 'anion_ligand'
    if not lig_old:
        if lig_new:
            pass_cfg["ligand"] = lig_new
            cfg["passivation"] = pass_cfg
            lig_old = lig_new
        else:
            raise KeyError("YAML: need passivation.ligand or passivation.anion_ligand")

    # Charges are required
    if "charges" not in cfg:
        raise KeyError("YAML: need 'charges' (global)")
    # keep them numeric but allow +2/-1 in YAML
    charges: Dict[str, int] = {str(k): int(v) for k, v in cfg["charges"].items()}

    # Ensure ligand charges exist (harmless if already provided)
    if lig_old not in charges:
        charges[lig_old] = -1
    if cat_new and (cat_new not in charges):
        charges[cat_new] = +1

    # Build the passivation spec.
    prepass_mode = str(pass_cfg.get("prepass_mode", "standard")).strip().lower()
    prepass_min_cn_terrace = int(pass_cfg.get("prepass_min_cn_terrace", 3))
    prepass_min_cn_edge = int(pass_cfg.get("prepass_min_cn_edge", 3))
    prepass_min_cn_vertex = int(pass_cfg.get("prepass_min_cn_vertex", 1 if prepass_mode == "role-aware" else 3))
    include_sublayer = bool(pass_cfg.get("include_sublayer", False))

    legacy_neutral_ligands = _parse_neutral_ligands(pass_cfg.get("neutral_ligands"))
    passiv_spec = PassivationSpec(
        ligand=str(lig_old),                # anion ligand (legacy field)
        surf_tol=surf_tol,
        cation_ligand=str(cat_new) if cat_new else None,
        prepass_mode=prepass_mode,
        prepass_min_cn_terrace=prepass_min_cn_terrace,
        prepass_min_cn_edge=prepass_min_cn_edge,
        prepass_min_cn_vertex=prepass_min_cn_vertex,
        include_sublayer=include_sublayer,
        neutral_ligands=legacy_neutral_ligands,
    )

    # ---- Global options ----
    proper_only = bool(cfg.get("symmetry", {}).get("proper_rotations_only", True))
    pair_opposites = bool(cfg.get("facet_options", {}).get("pair_opposites", True))

    # ---- twins (top-level) ----
    twins = _normalize_twins(cfg.get("twins"))
    construction_origin = cfg.get("construction_origin")
    if construction_origin is not None and not isinstance(construction_origin, dict):
        raise TypeError("construction_origin must be a mapping, e.g. {center_on_species: In}")
    facet_reconstruction = _parse_facet_reconstruction(cfg.get("facet_reconstruction"))
    post_treatment = _parse_post_treatment(
        cfg.get("post_treatment", cfg.get("post-treatment")),
        pass_cfg=pass_cfg,
        default_ligand=str(lig_old),
        legacy_recon_raw=cfg.get("facet_reconstruction"),
    )
    experimental = cfg.get("experimental") or {}
    if not isinstance(experimental, dict):
        raise TypeError("experimental must be a mapping")
    stack_raw = cfg.get("stack") or {}
    if not isinstance(stack_raw, dict):
        raise TypeError("stack must be a mapping")
    geometry_reference = str(stack_raw.get("geometry_reference", "core")).strip().lower()
    if geometry_reference not in {"core", "shortest", "shell"}:
        raise ValueError("stack.geometry_reference must be 'core', 'shortest', or 'shell'")
    interface = str(stack_raw.get("interface", "abrupt")).strip().lower()
    if interface not in {"abrupt", "mixed"}:
        raise ValueError("stack.interface must be 'abrupt' or 'mixed'")
    mixing_width = float(stack_raw.get("mixing_width", 3.0))
    stack_spec = StackSpec(
        geometry_reference=geometry_reference,
        interface=interface,
        mixing_width=mixing_width
    )

    # Register cation_ligand charge if provided
    if facet_reconstruction.cation_ligand and facet_reconstruction.cation_ligand not in charges:
        charges[facet_reconstruction.cation_ligand] = facet_reconstruction.cation_ligand_charge or +1
    if post_treatment.surface_reconstruction.ligand and post_treatment.surface_reconstruction.ligand not in charges:
        charges[post_treatment.surface_reconstruction.ligand] = -1

    # ---- Helper: parse facets list/mapping → List[Facet] ----
    def _parse_facets(raw) -> List[Facet]:
        if raw is None:
            raise KeyError("Missing 'facets' (or 'seeds') section")

        # Accept mapping {hkl: gamma} or list of {"hkl": ..., "gamma": ...}
        if isinstance(raw, dict):
            items = [{"hkl": k, "gamma": v} for k, v in raw.items()]
        elif isinstance(raw, list):
            # Also accept compact [ [hkl, gamma], ... ]
            items = []
            for it in raw:
                if isinstance(it, dict) and "hkl" in it and "gamma" in it:
                    items.append(it)
                elif isinstance(it, dict) and "family" in it:
                    raise ValueError("Use hkl with scope: family instead of family")
                elif isinstance(it, (list, tuple)) and len(it) == 2:
                    items.append({"hkl": it[0], "gamma": it[1]})
                else:
                    raise TypeError(
                        "facets/seeds list items must be dicts with keys {hkl,gamma}, "
                        "or 2-tuples [hkl, gamma]"
                    )
        else:
            raise TypeError("facets/seeds must be a list or a mapping of hkl->gamma")

        g_by: Dict[tuple[int, int, int], float] = {}
        term_by: Dict[tuple[int, int, int], str | None] = {}
        scope_by: Dict[tuple[int, int, int], str] = {}
        for f in items:
            if "family" in f:
                raise ValueError("Use hkl with scope: family instead of family")
            hkl_raw = f.get("hkl")
            h, k, l = _parse_hkl(hkl_raw)
            g_by[(h, k, l)] = float(f["gamma"])
            scope = str(f.get("scope", "family")).strip().lower()
            if scope not in {"family", "facet"}:
                raise ValueError("facet scope must be 'family' or 'facet'")
            scope_by[(h, k, l)] = scope
            term = f.get("termination")
            if term is not None:
                term_s = str(term).strip().lower()
                if term_s not in {"cation_rich", "anion_rich"}:
                    raise ValueError("facet termination must be 'cation_rich' or 'anion_rich'")
                term_by[(h, k, l)] = term_s
            else:
                term_by[(h, k, l)] = None

        if pair_opposites:
            for (h, k, l), g in list(g_by.items()):
                if scope_by.get((h, k, l)) == "facet":
                    continue
                if term_by.get((h, k, l)) is not None:
                    continue
                opp = (-h, -k, -l)
                if opp not in g_by:
                    g_by[opp] = g
                    term_by[opp] = None
                    scope_by[opp] = scope_by.get((h, k, l), "family")

        return [
            Facet(h=h, k=k, l=l, gamma=g, termination=term_by.get((h, k, l)), scope=scope_by.get((h, k, l), "family"))
            for (h, k, l), g in sorted(g_by.items())
        ]

    def _parse_shape(raw) -> tuple[Tuple[float, float, float], str, int]:
        aspect = (1.0, 1.0, 1.0)
        mode = "wulff"
        sphere_planes = 192
        if isinstance(raw, dict):
            if "aspect" in raw:
                aspect = _parse_aspect(raw.get("aspect"))
            mode = str(raw.get("mode", mode)).strip().lower()
            sphere_planes = int(raw.get("sphere_planes", sphere_planes))
        if mode not in {"wulff", "sphere"}:
            raise ValueError("shape.mode must be 'wulff' or 'sphere'")
        if sphere_planes < 12:
            raise ValueError("shape.sphere_planes must be at least 12")
        return aspect, mode, sphere_planes

    # ---- STACK MODE (multi-material) ----
    if "materials" in cfg:
        mats: List[MaterialSpec] = []
        for m in cfg["materials"]:
            if not isinstance(m, dict):
                raise TypeError("Each entry in 'materials' must be a mapping")

            name = str(m.get("name", "material"))
            if "cif" not in m:
                raise KeyError("materials[]: missing 'cif'")
            cif = str(m["cif"])

            # facets or seeds (accept either key)
            aspect, shape_mode, sphere_planes = _parse_shape(m.get("shape"))

            raw_facets = m.get("facets", m.get("seeds"))
            if raw_facets is None and shape_mode == "sphere":
                seeds = []
            else:
                seeds = _parse_facets(raw_facets)

            # build (optional; provide safe defaults so YAML can omit it)
            b = m.get("build", {}) or {}
            size_raw = m.get("size_unit_cells", b.get("size_unit_cells"))
            build = BuildSpec(
                radius=float(b["radius"]) if "radius" in b else None,
                radius_scale=float(b["radius_scale"]) if "radius_scale" in b else None,
                size_unit_cells=_parse_size_unit_cells(size_raw) if size_raw is not None else None,
                interface_clearance=float(b.get("interface_clearance", 1.6)),
            )

            # optional alignment (kept for future use)
            align = None
            if "align" in m and isinstance(m["align"], dict):
                a = m["align"]
                strain = a.get("strain_policy", {}) or {}
                align = AlignSpec(
                    core_facet=_parse_hkl(a["core_facet"]) if "core_facet" in a else None,
                    shell_facet=_parse_hkl(a["shell_facet"]) if "shell_facet" in a else None,
                    core_dir=tuple(int(x) for x in a["core_dir"]) if "core_dir" in a else None,
                    shell_dir=tuple(int(x) for x in a["shell_dir"]) if "shell_dir" in a else None,
                    strain=StrainPolicy(
                        type=str(strain.get("type", "none")).lower(),
                        max_percent=float(strain.get("max_percent", 3.0)),
                    ),
                )

            interface = m.get("interface")
            if interface is not None and not isinstance(interface, dict):
                raise TypeError(f"Material {name} 'interface' must be a mapping")

            mats.append(MaterialSpec(
                name=name,
                cif=cif,
                seeds=seeds,
                aspect=aspect,
                build=build,
                shape_mode=shape_mode,
                sphere_planes=sphere_planes,
                align=align,
                interface=interface,
            ))

        return Config(
            mode="stack",
            seeds=[], aspect=(1.0, 1.0, 1.0), shape_mode="wulff", sphere_planes=192,
            size_unit_cells=None,
            proper_only=proper_only, pair_opposites=pair_opposites,
            passivation=passiv_spec, charges=charges, materials=mats,
            twins=twins,
            construction_origin=construction_origin,
            facet_reconstruction=facet_reconstruction,
            post_treatment=post_treatment,
            experimental=experimental,
            stack=stack_spec,
        )

    # ---- SINGLE MODE (legacy) ----
    # top-level: accept 'facets' or 'seeds'
    aspect, shape_mode, sphere_planes = _parse_shape(cfg.get("shape"))
    size_unit_cells = (
        _parse_size_unit_cells(cfg.get("size_unit_cells"))
        if cfg.get("size_unit_cells") is not None
        else None
    )

    top_facets = cfg.get("facets", cfg.get("seeds"))
    if top_facets is None and shape_mode != "sphere":
        raise KeyError("YAML: need 'facets' (or 'seeds') for single mode")

    seeds = [] if top_facets is None else _parse_facets(top_facets)

    return Config(
        mode="single",
        seeds=seeds, aspect=aspect, shape_mode=shape_mode, sphere_planes=sphere_planes,
        size_unit_cells=size_unit_cells,
        proper_only=proper_only, pair_opposites=pair_opposites,
        passivation=passiv_spec, charges=charges, materials=[],
        twins=twins,
        construction_origin=construction_origin,
        facet_reconstruction=facet_reconstruction,
        post_treatment=post_treatment,
        experimental=experimental,
        stack=stack_spec,
    )
