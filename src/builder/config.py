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
    PassivationSpec, MaterialSpec, BuildSpec, AlignSpec, StrainPolicy
)

# -------------------- CLI --------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nc-builder",
        description="Coordination-aware Wulff-cut nanocrystal builder with surface passivation."
    )
    # NOTE: In stack mode, --cif / --radius are ignored (sizes & CIFs come from YAML)
    p.add_argument("cif", help="Input bulk CIF file (ignored in stack mode)")
    p.add_argument("yaml", help="YAML recipe file (single or multi-material)")
    p.add_argument("-r", "--radius", type=float, required=True,
                   help="Target Wulff radius (Å) for single-material mode")

    p.add_argument("-o", "--out", default="nanocrystal.xyz", help="Output XYZ path (final)")
    p.add_argument("--write-all", action="store_true", help="Also write *_cut.xyz")
    p.add_argument("--center", action="store_true", help="Center the particle at the COM before writing")
    p.add_argument("--seed", type=int, default=42, help="Random seed for tie-breaks and placement")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")

    p.add_argument(
        "--parity",
        choices=["remove", "add"],
        default="remove",
        help="Odd-parity resolution after outer swaps: "
             "'remove' stops at Q=-1 then removes one ligand (default). "
             "'add' overshoots to Q=+1 then adds one ligand."
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
    p.set_defaults(prune_mono=True)

    # Wulff / symmetry
    w = p.add_argument_group("wulff / symmetry")
    w.add_argument("--pair-opposites", action="store_true", default=True,
                   help="If a signed facet (hkl) is provided without its antipode (-h -k -l), "
                        "auto-add the opposite with the same gamma (default: on).")
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

    p.add_argument(
        "--core-by-labeling",
        action="store_true",
        help="Build shell NC at -r, carve inner core using core.aspect, and relabel cation/anion inside."
    )

    # Core lattice fit / interface strain (optional)
    strain = p.add_argument_group("core lattice fit / interface strain")
    strain.add_argument(
        "--core-lattice-fit",
        action="store_true",
        help="After relabeling, shrink inner core atoms with an affine map to the core CIF lattice, "
             "and apply a smooth strain blend near the core boundary."
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

    diag = p.add_argument_group("diagnostics")
    diag.add_argument(
        "--bond-stats",
        action="store_true",
        help="Print cation–anion nearest-neighbor distance stats for core/interface/shell (pre/post shrink)."
    )
    
 
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
        # 1) Try three full signed integers anywhere
        nums = re.findall(r'(?<!\d)[+-]?\d+(?!\d)', s)
        if len(nums) == 3:
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

# -------------------- YAML → Config --------------------
# -------------------- YAML → Config --------------------

def parse_yaml_config(path: str) -> Config:
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    # ---- Global passivation + charges ----
    passiv = cfg.get("passivation", {}) or {}
    if "ligand" not in passiv:
        raise KeyError("YAML: need passivation.ligand (global)")
    passiv_spec = PassivationSpec(
        ligand=str(passiv["ligand"]),
        surf_tol=float(passiv.get("surf_tol", 1.0)),
    )

    if "charges" not in cfg:
        raise KeyError("YAML: need 'charges' (global)")
    charges: Dict[str, int] = {str(k): int(v) for k, v in cfg["charges"].items()}

    # ---- Global options ----
    proper_only = bool(cfg.get("symmetry", {}).get("proper_rotations_only", True))
    pair_opposites = bool(cfg.get("facet_options", {}).get("pair_opposites", True))

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
                elif isinstance(it, (list, tuple)) and len(it) == 2:
                    items.append({"hkl": it[0], "gamma": it[1]})
                else:
                    raise TypeError(
                        "facets/seeds list items must be dicts with keys {hkl,gamma} "
                        "or 2-tuples [hkl, gamma]"
                    )
        else:
            raise TypeError("facets/seeds must be a list or a mapping of hkl->gamma")

        g_by: Dict[tuple[int, int, int], float] = {}
        for f in items:
            h, k, l = _parse_hkl(f["hkl"])
            g_by[(h, k, l)] = float(f["gamma"])

        if pair_opposites:
            for (h, k, l), g in list(g_by.items()):
                opp = (-h, -k, -l)
                if opp not in g_by:
                    g_by[opp] = g

        return [Facet(h=h, k=k, l=l, gamma=g) for (h, k, l), g in sorted(g_by.items())]

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
            raw_facets = m.get("facets", m.get("seeds"))
            seeds = _parse_facets(raw_facets)

            # aspect (per-material); accept m.shape.aspect or default 1,1,1
            aspect = (1.0, 1.0, 1.0)
            if isinstance(m.get("shape"), dict):
                aspect = _parse_aspect(m["shape"].get("aspect"))

            # build (optional; provide safe defaults so YAML can omit it)
            b = m.get("build", {}) or {}
            build = BuildSpec(
                radius=float(b["radius"]) if "radius" in b else None,
                radius_scale=float(b["radius_scale"]) if "radius_scale" in b else None,
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

            mats.append(MaterialSpec(
                name=name, cif=cif, seeds=seeds, aspect=aspect, build=build, align=align
            ))

        return Config(
            mode="stack",
            seeds=[], aspect=(1.0, 1.0, 1.0),
            proper_only=proper_only, pair_opposites=pair_opposites,
            passivation=passiv_spec, charges=charges, materials=mats,
        )

    # ---- SINGLE MODE (legacy) ----
    # top-level: accept 'facets' or 'seeds'
    top_facets = cfg.get("facets", cfg.get("seeds"))
    if top_facets is None:
        raise KeyError("YAML: need 'facets' (or 'seeds') for single mode")

    seeds = _parse_facets(top_facets)

    aspect = (1.0, 1.0, 1.0)
    if isinstance(cfg.get("shape"), dict):
        aspect = _parse_aspect(cfg["shape"].get("aspect"))

    return Config(
        mode="single",
        seeds=seeds, aspect=aspect,
        proper_only=proper_only, pair_opposites=pair_opposites,
        passivation=passiv_spec, charges=charges, materials=[],
    )

