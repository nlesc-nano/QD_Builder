# nanocrystal_builder/config.py
from __future__ import annotations
import argparse, json, sys, re
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

try:
    import yaml
except ImportError:
    sys.exit("pip install pyyaml")

from .nc_types import Facet

# -------------------- CLI --------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nanocrystal_builder",
        description="Coordination-aware Wulff-cut nanocrystal builder with surface passivation."
    )
    p.add_argument("cif", help="Input bulk CIF file")
    p.add_argument("yaml", help="YAML recipe file with facets, charges, passivation settings")
    p.add_argument("-r", "--radius", type=float, required=True, help="Target Wulff radius (Å)")

    p.add_argument("-o", "--out", default="nanocrystal.xyz", help="Output XYZ path (final)")
    p.add_argument("--write-all", action="store_true", help="Also write *_cut.xyz and *_passivated.xyz")
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
        help="Remove atoms with coordination number < this before facet detection (default: 2 → removes CN=1)."
    )
    pre.add_argument(
        "--prune-passes", type=int, default=10,
        help="Maximum number of pruning passes to reach stability (default: 10)."
    )
    pre.add_argument(
        "--no-prune-mono", dest="prune_mono", action="store_false",
        help="Disable the pre-pass pruning step."
    )
    p.set_defaults(prune_mono=True)
    # in build_parser()
    w = p.add_argument_group("wulff / symmetry")
    w.add_argument("--pair-opposites", action="store_true", default=True,
                   help="If a signed facet (hkl) is provided without its antipode (-h -k -l), "
                        "auto-add the opposite with the same gamma (default: on).")
    w.add_argument("--proper-rotations-only", action="store_true", default=True,
                   help="Use only proper rotations (det=+1) when expanding seed facets; "
                        "prevents folding +hkl into -h-k-l via inversion (default: on).")
    return p

# -------------------- YAML --------------------

_HKL_RX = re.compile(r"^\s*([+-]?\d+)\s*[, ]\s*([+-]?\d+)\s*[, ]\s*([+-]?\d+)\s*$")

# config.py

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
        # 1) Try three full signed integers anywhere (handles "1 1 1", "1, -1, 0", "(1 1 1)")
        nums = re.findall(r'(?<!\d)[+-]?\d+(?!\d)', s)
        if len(nums) == 3:
            h, k, l = (int(nums[0]), int(nums[1]), int(nums[2]))
            if (h, k, l) == (0, 0, 0):
                raise ValueError("hkl cannot be (0,0,0)")
            return (h, k, l)

        # 2) Compact per-digit with optional signs before each digit (handles "-100", "1-10", "11-1", "-1-1-1")
        # Strip brackets/commas/spaces; keep only digits and signs
        s2 = re.sub(r'[^0-9+-]', '', s)
        # Tokenize as signed single digits
        tokens = re.findall(r'[+-]?\d', s2)
        if len(tokens) == 3 and all(len(t.replace('+','').replace('-','')) == 1 for t in tokens):
            h, k, l = (int(tokens[0]), int(tokens[1]), int(tokens[2]))
            if (h, k, l) == (0, 0, 0):
                raise ValueError("hkl cannot be (0,0,0)")
            return (h, k, l)

        # 3) Plain compact unsigned "111" / "100"
        s3 = re.sub(r'[()\[\]\s,;]', '', s)
        if s3.isdigit() and len(s3) == 3:
            return (int(s3[0]), int(s3[1]), int(s3[2]))

        raise ValueError(f"Invalid hkl format: {val!r}")

    raise TypeError(f"Unsupported hkl type: {type(val).__name__}")


@dataclass(frozen=True)
class Config:
    seeds: List[Facet]
    expand_symmetry: bool
    ligand: str
    surf_tol: float
    charges: Dict[str, int]

def parse_yaml_config(path: str):
    cfg = yaml.safe_load(open(path))

    if 'facets' not in cfg:
        raise KeyError("YAML: need 'facets'")
    if 'passivation' not in cfg or 'ligand' not in cfg['passivation']:
        raise KeyError("YAML: need passivation.ligand")
    if 'charges' not in cfg:
        raise KeyError("YAML: need 'charges'")

    # facet entries: allow
    #  - list:  - {hkl: "111", gamma: 1.0}  - {hkl:"-1-1-1", gamma:0.8}
    #  - map:   facets: {"111": 1.0, "-1-1-1": 0.8}
    raw_facets = cfg['facets']
    items = []
    if isinstance(raw_facets, dict):
        for k, v in raw_facets.items():
            items.append({"hkl": str(k), "gamma": float(v)})
    elif isinstance(raw_facets, list):
        items.extend(raw_facets)
    else:
        raise TypeError("YAML facets must be a list or a mapping of hkl->gamma")

    # parse into a dict to allow overriding same signed hkl
    gamma_by_hkl: dict[tuple[int,int,int], float] = {}
    for f in items:
        h, k, l = _parse_hkl(f['hkl'])   # your existing helper supports "-1-1-1"
        gamma_by_hkl[(h, k, l)] = float(f['gamma'])

    pair_opposites = bool(cfg.get('facet_options', {}).get('pair_opposites', True))
    # CLI override takes precedence if present
    try:
        # when called from main, args are attached to cfg via Namespace-like object; skip if not
        from argparse import Namespace
        if isinstance(cfg.get('_args'), Namespace):
            if hasattr(cfg['_args'], 'pair_opposites'):
                pair_opposites = cfg['_args'].pair_opposites
    except Exception:
        pass

    # auto-pair: add missing antipodes with same gamma
    if pair_opposites:
        to_add = {}
        for (h, k, l), g in list(gamma_by_hkl.items()):
            opp = (-h, -k, -l)
            if opp not in gamma_by_hkl:
                to_add[opp] = g
        gamma_by_hkl.update(to_add)

    seeds = [Facet(h=h, k=k, l=l, gamma=g) for (h, k, l), g in sorted(gamma_by_hkl.items())]

    ligand = cfg['passivation']['ligand']
    surf_tol = float(cfg['passivation'].get('surf_tol', 1.0))
    charges = {k: int(v) for k, v in cfg['charges'].items()}
    if ligand not in charges:
        raise KeyError(f"YAML: missing charge for {ligand}")

    return seeds, ligand, surf_tol, charges, pair_opposites


