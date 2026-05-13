#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

try:
    import yaml
    from pymatgen.core import Structure
except ImportError:
    raise SystemExit("Install pyyaml and pymatgen to run the Janus builder.")

from builder.heterointerface import (
    analyze_terminations,
    build_janus_candidate,
    build_janus_candidate_cells,
    build_janus_candidate_wulff,
    counts_label,
    enumerate_interface_candidates,
    filter_lattice_matched_candidates,
    hkl_label,
    unique_family_terminations,
)
from builder.io_utils import write_xyz
from builder.analysis import merge_pair_cuts_from_cifs
from builder.passivation_iterative import charge_balance_iterative
from builder.nc_types import Facet


def _parse_charge_token(token: str) -> tuple[str, int]:
    m = re.fullmatch(r"\s*([A-Z][a-z]?)(?:\s*=\s*|\s*:\s*)?([+-]?\d+)\s*", token)
    if not m:
        raise argparse.ArgumentTypeError(f"Invalid charge token: {token!r}; use Cd=+2 or Cd:+2")
    return m.group(1), int(m.group(2))


def _charges_from_yaml(path: Path) -> dict[str, int]:
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    raw = data.get("charges")
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} does not contain a top-level charges: mapping")
    return {str(k): int(v) for k, v in raw.items()}


def _load_yaml_input(path: str | None) -> dict:
    if path is None:
        return {}
    with Path(path).open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a YAML mapping")
    return data


def _pick(cfg: dict, dotted: str, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _assign_from_yaml(args, cfg: dict, attr: str, dotted: str) -> None:
    val = _pick(cfg, dotted, None)
    if val is not None:
        setattr(args, attr, val)


def _resolve_path(value: str, base_dir: Path) -> str:
    path = Path(str(value))
    if path.is_absolute() or path.exists():
        return str(path)
    rel = base_dir / path
    return str(rel if rel.exists() else path)


def _parse_hkl(value):
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    s = str(value).strip()
    nums = re.findall(r"[+-]?\d", re.sub(r"[^0-9+-]", "", s))
    if len(nums) == 3:
        hkl = tuple(int(x) for x in nums)
        if hkl != (0, 0, 0):
            return hkl
    raise ValueError(f"Invalid hkl/family value: {value!r}")


def _parse_facets(raw) -> list[Facet]:
    if not isinstance(raw, list):
        raise SystemExit("build.core.facets and build.shell.facets must be lists")
    out = []
    for item in raw:
        if not isinstance(item, dict):
            raise SystemExit("facet entries must be mappings")
        if "family" in item:
            raise SystemExit("Use hkl with scope: family instead of family")
        hkl = _parse_hkl(item.get("hkl"))
        term = item.get("termination")
        if term is not None:
            term = str(term).strip().lower()
        scope = str(item.get("scope", "family")).strip().lower()
        if scope not in {"family", "facet"}:
            raise SystemExit("facet scope must be 'family' or 'facet'")
        out.append(Facet(hkl[0], hkl[1], hkl[2], float(item.get("gamma", 1.0)), termination=term, scope=scope))
    return out


def _shape_mode_from_yaml(cfg: dict, path: str) -> str:
    shape = _pick(cfg, path, {}) or {}
    if not isinstance(shape, dict):
        return "wulff"
    mode = str(shape.get("mode", "wulff")).strip().lower()
    if mode not in {"wulff", "sphere"}:
        raise SystemExit(f"{path}.mode must be 'wulff' or 'sphere'")
    return mode


def _sphere_planes_from_yaml(cfg: dict, path: str) -> int:
    shape = _pick(cfg, path, {}) or {}
    if not isinstance(shape, dict):
        return 192
    n = int(shape.get("sphere_planes", 192))
    if n < 12:
        raise SystemExit(f"{path}.sphere_planes must be at least 12")
    return n


def _apply_yaml_config(args, cfg: dict, base_dir: Path):
    _assign_from_yaml(args, cfg, "core_cif", "materials.core.cif")
    _assign_from_yaml(args, cfg, "shell_cif", "materials.shell.cif")
    _assign_from_yaml(args, cfg, "core_name", "materials.core.name")
    _assign_from_yaml(args, cfg, "shell_name", "materials.shell.name")

    if args.core_cif is not None:
        args.core_cif = _resolve_path(args.core_cif, base_dir)
    if args.shell_cif is not None:
        args.shell_cif = _resolve_path(args.shell_cif, base_dir)

    _assign_from_yaml(args, cfg, "top", "candidates.top")
    _assign_from_yaml(args, cfg, "candidate_match", "candidates.match")
    _assign_from_yaml(args, cfg, "candidate_core_family", "candidates.core_family")
    _assign_from_yaml(args, cfg, "candidate_shell_family", "candidates.shell_family")
    _assign_from_yaml(args, cfg, "candidate_core_hkl", "candidates.core_hkl")
    _assign_from_yaml(args, cfg, "candidate_shell_hkl", "candidates.shell_hkl")
    _assign_from_yaml(args, cfg, "max_index", "scan.max_index")
    _assign_from_yaml(args, cfg, "layer_tol", "scan.layer_tol")
    _assign_from_yaml(args, cfg, "allow_charged_neutral", "scan.allow_charged_neutral")
    _assign_from_yaml(args, cfg, "signed", "scan.signed")
    all_rot = _pick(cfg, "scan.all_rotations", None)
    if all_rot is not None:
        args.all_rotations = bool(all_rot)

    _assign_from_yaml(args, cfg, "radius", "build.radius")
    _assign_from_yaml(args, cfg, "build_mode", "build.mode")
    _assign_from_yaml(args, cfg, "lateral_cells", "build.lateral_cells")
    _assign_from_yaml(args, cfg, "core_layers", "build.core_layers")
    _assign_from_yaml(args, cfg, "shell_layers", "build.shell_layers")
    _assign_from_yaml(args, cfg, "interface_distance", "build.interface_distance")
    _assign_from_yaml(args, cfg, "min_separation", "build.min_separation")
    _assign_from_yaml(args, cfg, "match_core_footprint", "build.match_core_footprint")
    _assign_from_yaml(args, cfg, "footprint_margin", "build.footprint_margin")
    _assign_from_yaml(args, cfg, "footprint_shape", "build.footprint_shape")
    _assign_from_yaml(args, cfg, "mushroom_overhang", "build.mushroom_overhang")
    _assign_from_yaml(args, cfg, "core_size_unit_cells", "build.core.size_unit_cells")
    _assign_from_yaml(args, cfg, "shell_size_unit_cells", "build.shell.size_unit_cells")
    args.core_shape_mode = _shape_mode_from_yaml(cfg, "build.core.shape")
    args.shell_shape_mode = _shape_mode_from_yaml(cfg, "build.shell.shape")
    args.core_sphere_planes = _sphere_planes_from_yaml(cfg, "build.core.shape")
    args.shell_sphere_planes = _sphere_planes_from_yaml(cfg, "build.shell.shape")
    if _pick(cfg, "build.core.facets", None) is not None:
        args.core_facets = _parse_facets(_pick(cfg, "build.core.facets"))
    if _pick(cfg, "build.shell.facets", None) is not None:
        args.shell_facets = _parse_facets(_pick(cfg, "build.shell.facets"))
    _assign_from_yaml(args, cfg, "out_dir", "output.out_dir")
    _assign_from_yaml(args, cfg, "prefix", "output.prefix")

    pass_cfg = cfg.get("passivation", {}) or {}
    if isinstance(pass_cfg, dict):
        if bool(pass_cfg.get("enabled", "ligand" in pass_cfg)):
            _assign_from_yaml(args, cfg, "passivate_ligand", "passivation.ligand")
        _assign_from_yaml(args, cfg, "surf_tol", "passivation.surf_tol")
        _assign_from_yaml(args, cfg, "positive_q_mode", "passivation.positive_q_mode")
        _assign_from_yaml(args, cfg, "positive_q_mode_core", "passivation.positive_q_mode_core")
        _assign_from_yaml(args, cfg, "positive_q_mode_shell", "passivation.positive_q_mode_shell")

    match_cfg = cfg.get("matching", {}) or {}
    if isinstance(match_cfg, dict):
        method = str(match_cfg.get("method", "zsl")).lower()
        args.no_zsl = method in {"none", "off", "false"}
        _assign_from_yaml(args, cfg, "zsl_max_area", "matching.max_area")
        _assign_from_yaml(args, cfg, "zsl_max_length_tol", "matching.max_length_tol")
        _assign_from_yaml(args, cfg, "zsl_max_angle_tol", "matching.max_angle_tol")
        _assign_from_yaml(args, cfg, "zsl_max_area_ratio_tol", "matching.max_area_ratio_tol")

    if "charges" in cfg and isinstance(cfg["charges"], dict):
        args.config_charges = {str(k): int(v) for k, v in cfg["charges"].items()}

    if args.lateral_cells is not None:
        args.lateral_cells = [float(args.lateral_cells[0]), float(args.lateral_cells[1])]
    return args


def _read_charges(args) -> dict[str, int]:
    charges: dict[str, int] = {}
    charges.update(getattr(args, "config_charges", {}) or {})
    if args.yaml:
        charges.update(_charges_from_yaml(Path(args.yaml)))
    for token in args.charges or []:
        el, q = _parse_charge_token(token)
        charges[el] = q
    if not charges:
        raise SystemExit("Provide charges either via --yaml input.yaml or --charges Cs=+1 Pb=+2 Br=-1")
    return charges


def _family_filter_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    return "{" + text + "}"


def _filter_candidates(candidates, args):
    out = list(candidates)
    if getattr(args, "candidate_match", None):
        want = str(args.candidate_match).strip()
        out = [cand for cand in out if cand.compatibility == want]
    core_family = _family_filter_value(getattr(args, "candidate_core_family", None))
    if core_family:
        out = [cand for cand in out if cand.core.family == core_family]
    shell_family = _family_filter_value(getattr(args, "candidate_shell_family", None))
    if shell_family:
        out = [cand for cand in out if cand.shell.family == shell_family]
    if getattr(args, "candidate_core_hkl", None):
        want = _parse_hkl(args.candidate_core_hkl)
        out = [cand for cand in out if cand.core.hkl == want]
    if getattr(args, "candidate_shell_hkl", None):
        want = _parse_hkl(args.candidate_shell_hkl)
        out = [cand for cand in out if cand.shell.hkl == want]
    return out


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "x"


def _candidate_label(idx: int, cand) -> str:
    return (
        f"cand{idx:02d}_"
        f"core{_safe(cand.core.family.strip('{}'))}{_safe(hkl_label(cand.core.hkl))}_"
        f"shell{_safe(cand.shell.family.strip('{}'))}{_safe(hkl_label(cand.shell.hkl))}"
    )


def _describe_candidate(idx: int, cand) -> str:
    lm = cand.lattice_match
    zsl = "no ZSL data"
    if lm is not None:
        zsl = (
            f"ZSL area={lm.area:.2f} A^2, "
            f"length mismatch={100.0 * lm.max_length_mismatch:.2f}%, "
            f"angle mismatch={lm.angle_mismatch_deg:.2f} deg"
        )
    return (
        f"[candidate {idx}] "
        f"core {cand.core.family} {hkl_label(cand.core.hkl)} "
        f"Q={cand.core.charge:+d} {cand.core.richness} ({counts_label(cand.core.counts)})  |  "
        f"shell {cand.shell.family} {hkl_label(cand.shell.hkl)} "
        f"Q={cand.shell.charge:+d} {cand.shell.richness} ({counts_label(cand.shell.counts)})  |  "
        f"Qsum={cand.core.charge + cand.shell.charge:+d}, {cand.compatibility}, {zsl}"
    )


def _fmt_counts_dict(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return " ".join(f"{el}:{n}" for el, n in counts.items())


def _write_summary(path: Path, rows: list[dict]) -> None:
    headers = [
        "rank", "file", "atoms", "Q", "core_atoms", "shell_atoms", "removed",
        "core_hkl", "core_Q", "core_term", "shell_hkl", "shell_Q", "shell_term",
        "Qsum_interface", "match", "zsl_area", "zsl_len_pct", "zsl_angle_deg",
        "actual_core_interface", "actual_core_Q", "actual_shell_interface",
        "actual_shell_Q", "actual_interface_Qsum",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_counts(symbols: list[str], charges: dict[str, int]) -> str:
    cnt = Counter(symbols)
    q = int(sum(charges.get(el, 0) * n for el, n in cnt.items()))
    parts = [f"{el}:{cnt[el]}" for el in sorted(cnt)]
    return " ".join(parts) + f" | Q={q:+d}"


def _ordered_xyz(symbols: list[str], pts, charges: dict[str, int], ligand: str | None):
    order = sorted(
        range(len(symbols)),
        key=lambda i: (
            1 if ligand is not None and symbols[i] == ligand else 0,
            0 if charges.get(symbols[i], 0) > 0 else 1,
            str(symbols[i]),
            i,
        ),
    )
    return [symbols[i] for i in order], pts[order]


def _warn_charge_cif_mismatch(cif_path: str, charges: dict[str, int]) -> None:
    struct = Structure.from_file(cif_path)
    seen = set()
    for site in struct:
        sym = str(site.specie.symbol)
        if sym in seen or sym not in charges:
            continue
        seen.add(sym)
        oxi = getattr(site.specie, "oxi_state", None)
        if oxi is None:
            continue
        if int(round(float(oxi))) != int(charges[sym]):
            print(
                f"    [warn] {Path(cif_path).name}: CIF oxidation state for {sym} "
                f"is {float(oxi):+.0f}, but input charge is {charges[sym]:+d}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Experimental Janus heterostructure builder from charge/ZSL-ranked facet candidates."
    )
    parser.add_argument("input", nargs="?", help="YAML input file, or core CIF in legacy CLI mode")
    parser.add_argument("shell_cif", nargs="?", help="Shell CIF in legacy CLI mode")
    parser.add_argument("--core-name", default="core")
    parser.add_argument("--shell-name", default="shell")
    parser.add_argument("--yaml", help="Builder YAML from which to read top-level charges")
    parser.add_argument("--charges", nargs="*", help="Charges, e.g. Cs=+1 Pb=+2 Br=-1 S=-2")
    parser.add_argument("--max-index", type=int, default=1)
    parser.add_argument("--all-rotations", action="store_true")
    parser.add_argument("--layer-tol", type=float, default=0.08)
    parser.add_argument("--allow-charged-neutral", action="store_true")
    parser.add_argument("--signed", action="store_true", help="Keep every signed symmetry-equivalent orientation")
    parser.add_argument("--top", type=int, default=5, help="Build this many top-ranked candidates")
    parser.add_argument("--radius", type=float, default=18.0, help="Final Janus particle radius in Angstrom")
    parser.add_argument("--build-mode", choices=["radius", "interface_cell", "wulff_janus"], default="radius")
    parser.add_argument("--lateral-cells", type=float, nargs=2, metavar=("NX", "NY"),
                        help="Use interface-cell mode with this x/y footprint in surface-cell repetitions")
    parser.add_argument("--core-layers", type=int, default=6, help="Core depth in interface-normal layers for cell mode")
    parser.add_argument("--shell-layers", type=int, default=6, help="Shell depth in interface-normal layers for cell mode")
    parser.add_argument("--interface-distance", type=float, default=2.8, help="Initial separation between interface layers")
    parser.add_argument("--min-separation", type=float, default=1.2, help="Remove duplicate/overlapping sites below this distance")
    parser.add_argument("--no-match-core-footprint", dest="match_core_footprint", action="store_false",
                        help="In Wulff-Janus mode, keep the independent shell lateral footprint")
    parser.add_argument("--footprint-margin", type=float, default=1.0,
                        help="Extra Angstrom margin around the core in-plane footprint for Wulff-Janus shell clipping")
    parser.add_argument("--footprint-shape", choices=["bbox", "convex_hull", "mushroom"], default="bbox",
                        help="Wulff-Janus shell clipping footprint; mushroom permits a spherical shell cap overhang")
    parser.add_argument("--mushroom-overhang", type=float, default=None,
                        help="Max lateral overhang in Angstrom for footprint_shape=mushroom; default auto-fits shell cap")
    parser.add_argument("--passivate-ligand", help="Run existing charge-balancing/passivation on outer surfaces only")
    parser.add_argument("--surf-tol", type=float, default=2.0, help="Surface tolerance for passivation")
    parser.add_argument("--positive-q-mode", choices=["remove", "add"], default="add")
    parser.add_argument("--positive-q-mode-core", choices=["remove", "add", "skip", "none"],
                        help="Janus/interface-cell passivation: Q>0 strategy for atoms on the core side")
    parser.add_argument("--positive-q-mode-shell", choices=["remove", "add", "skip", "none"],
                        help="Janus/interface-cell passivation: Q>0 strategy for atoms on the shell side")
    parser.add_argument("--out-dir", default="janus_candidates")
    parser.add_argument("--prefix", default="janus")
    parser.add_argument("--no-zsl", action="store_true", help="Disable ZSL filtering; normally not recommended")
    parser.add_argument("--zsl-max-area", type=float, default=400.0)
    parser.add_argument("--zsl-max-length-tol", type=float, default=0.03)
    parser.add_argument("--zsl-max-angle-tol", type=float, default=0.01)
    parser.add_argument("--zsl-max-area-ratio-tol", type=float, default=0.09)
    parser.set_defaults(
        core_size_unit_cells=None,
        shell_size_unit_cells=None,
        core_facets=None,
        shell_facets=None,
        core_shape_mode="wulff",
        shell_shape_mode="wulff",
        core_sphere_planes=192,
        shell_sphere_planes=192,
        mushroom_overhang=None,
        candidate_match=None,
        candidate_core_family=None,
        candidate_shell_family=None,
        candidate_core_hkl=None,
        candidate_shell_hkl=None,
        positive_q_mode_core=None,
        positive_q_mode_shell=None,
        match_core_footprint=True,
    )
    args = parser.parse_args(argv)

    cfg = {}
    if args.input and args.input.lower().endswith((".yaml", ".yml")):
        cfg = _load_yaml_input(args.input)
        args.core_cif = None
        args.shell_cif = None
        args = _apply_yaml_config(args, cfg, Path(args.input).resolve().parent)
    else:
        args.core_cif = args.input

    if not args.core_cif or not args.shell_cif:
        raise SystemExit(
            "Provide either a YAML input file or legacy positional CIFs: "
            "build_janus_heterostructures.py input.yaml"
        )

    charges = _read_charges(args)
    proper_only = not args.all_rotations
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1] Reading inputs")
    print(f"    core CIF : {args.core_cif} ({args.core_name})")
    print(f"    shell CIF: {args.shell_cif} ({args.shell_name})")
    print("    charges  : " + ", ".join(f"{el}={q:+d}" for el, q in sorted(charges.items())))
    print(f"    max-index: {args.max_index}")
    _warn_charge_cif_mismatch(args.core_cif, charges)
    _warn_charge_cif_mismatch(args.shell_cif, charges)

    print("\n[2] Scanning facet terminations from both CIFs")
    core_terms = analyze_terminations(
        args.core_cif,
        charges,
        material_name=args.core_name,
        max_index=args.max_index,
        proper_only=proper_only,
        layer_tol=args.layer_tol,
    )
    shell_terms = analyze_terminations(
        args.shell_cif,
        charges,
        material_name=args.shell_name,
        max_index=args.max_index,
        proper_only=proper_only,
        layer_tol=args.layer_tol,
    )
    print(f"    raw terminations: core={len(core_terms)}, shell={len(shell_terms)}")
    if not args.signed:
        core_terms = unique_family_terminations(core_terms)
        shell_terms = unique_family_terminations(shell_terms)
        print(f"    compact terminations: core={len(core_terms)}, shell={len(shell_terms)}")

    print("\n[3] Filtering by interface charge compatibility")
    candidates = enumerate_interface_candidates(
        core_terms,
        shell_terms,
        allow_charged_neutral=args.allow_charged_neutral,
    )
    print(f"    charge-compatible candidates: {len(candidates)}")
    if not candidates:
        raise SystemExit("No charge-compatible interface candidates found.")

    if args.no_zsl:
        print("\n[4] ZSL lattice matching disabled by --no-zsl")
    else:
        print("\n[4] Filtering by ZSL / Zur-McGill 2D lattice matching")
        print(
            f"    max_area={args.zsl_max_area:g} A^2, "
            f"length_tol={args.zsl_max_length_tol:g}, angle_tol={args.zsl_max_angle_tol:g}"
        )
        candidates = filter_lattice_matched_candidates(
            candidates,
            max_area_ratio_tol=args.zsl_max_area_ratio_tol,
            max_area=args.zsl_max_area,
            max_length_tol=args.zsl_max_length_tol,
            max_angle_tol=args.zsl_max_angle_tol,
        )
        print(f"    ZSL-matched candidates: {len(candidates)}")
        if not candidates:
            raise SystemExit("No ZSL-matched candidates found. Relax ZSL tolerances or use --no-zsl.")

    candidates = _filter_candidates(candidates, args)
    if any(getattr(args, name, None) for name in (
        "candidate_match",
        "candidate_core_family",
        "candidate_shell_family",
        "candidate_core_hkl",
        "candidate_shell_hkl",
    )):
        print(f"\n[4b] Candidate filters applied; remaining candidates: {len(candidates)}")
    if not candidates:
        raise SystemExit("No candidates remain after candidate filters.")

    build_count = min(int(args.top), len(candidates))
    mode = str(args.build_mode).replace("-", "_")
    if mode == "radius" and args.lateral_cells is not None:
        mode = "interface_cell"
    print(f"\n[5] Building top {build_count} Janus QD candidate(s)")
    if mode == "wulff_janus":
        print(
            f"    mode=wulff_janus, core_size_unit_cells={args.core_size_unit_cells}, "
            f"shell_size_unit_cells={args.shell_size_unit_cells}, "
            f"interface_distance={args.interface_distance:.2f} A, "
            f"match_core_footprint={bool(args.match_core_footprint)}"
        )
        print(
            f"    outer shapes: core={args.core_shape_mode}, shell={args.shell_shape_mode}"
        )
    elif mode == "interface_cell":
        print(
            f"    mode=interface-cell, lateral_cells=({args.lateral_cells[0]:g}, {args.lateral_cells[1]:g}), "
            f"core_layers={args.core_layers}, shell_layers={args.shell_layers}, "
            f"interface_distance={args.interface_distance:.2f} A"
        )
    else:
        print(f"    mode=radius, radius={args.radius:.2f} A, interface_distance={args.interface_distance:.2f} A")
    if args.passivate_ligand:
        if args.passivate_ligand not in charges:
            raise SystemExit(f"--passivate-ligand {args.passivate_ligand!r} requires a charge entry, e.g. Cl=-1")
        if mode not in {"interface_cell", "wulff_janus"}:
            raise SystemExit("Passivation currently requires build.mode interface_cell or wulff_janus.")
        print(
            f"    passivation enabled: ligand={args.passivate_ligand}, "
            f"surf_tol={args.surf_tol:g} A, positive_q_mode={args.positive_q_mode}"
        )
        if args.positive_q_mode_core or args.positive_q_mode_shell:
            print(
                "    region-aware Q>0 modes: "
                f"core={args.positive_q_mode_core or args.positive_q_mode}, "
                f"shell={args.positive_q_mode_shell or args.positive_q_mode}"
            )
        pair_cuts = merge_pair_cuts_from_cifs([args.core_cif, args.shell_cif], charges, safety=1.00)
    else:
        pair_cuts = None
    rows: list[dict] = []
    for idx, cand in enumerate(candidates[:build_count], start=1):
        print("\n" + _describe_candidate(idx, cand))
        if mode == "wulff_janus":
            if args.core_shape_mode != "sphere" and args.core_facets is None:
                raise SystemExit("build.mode wulff_janus requires build.core.facets unless build.core.shape.mode is sphere")
            if args.shell_shape_mode != "sphere" and args.shell_facets is None:
                raise SystemExit("build.mode wulff_janus requires build.shell.facets unless build.shell.shape.mode is sphere")
            if args.core_size_unit_cells is None or args.shell_size_unit_cells is None:
                raise SystemExit("build.mode wulff_janus requires build.core.size_unit_cells and build.shell.size_unit_cells")
            syms, pts, meta, outer_planes = build_janus_candidate_wulff(
                cand,
                charges,
                core_facets=args.core_facets or [],
                shell_facets=args.shell_facets or [],
                core_size_unit_cells=args.core_size_unit_cells,
                shell_size_unit_cells=args.shell_size_unit_cells,
                proper_only=proper_only,
                interface_distance=args.interface_distance,
                layer_tol=args.layer_tol,
                min_separation=args.min_separation,
                match_core_footprint=bool(args.match_core_footprint),
                footprint_margin=args.footprint_margin,
                footprint_shape=args.footprint_shape,
                mushroom_overhang=args.mushroom_overhang,
                core_shape_mode=args.core_shape_mode,
                shell_shape_mode=args.shell_shape_mode,
                core_sphere_planes=args.core_sphere_planes,
                shell_sphere_planes=args.shell_sphere_planes,
            )
        elif mode == "interface_cell":
            syms, pts, meta, outer_planes = build_janus_candidate_cells(
                cand,
                charges,
                lateral_cells=(args.lateral_cells[0], args.lateral_cells[1]),
                core_layers=args.core_layers,
                shell_layers=args.shell_layers,
                interface_distance=args.interface_distance,
                layer_tol=args.layer_tol,
                min_separation=args.min_separation,
            )
        else:
            syms, pts, meta = build_janus_candidate(
                cand,
                charges,
                radius=args.radius,
                interface_distance=args.interface_distance,
                layer_tol=args.layer_tol,
                min_separation=args.min_separation,
            )
            outer_planes = []
        label = _candidate_label(idx, cand)
        xyz_path = out_dir / f"{args.prefix}_{label}.xyz"
        json_path = out_dir / f"{args.prefix}_{label}.json"
        if args.passivate_ligand:
            print("    running outer-surface passivation; buried interface planes are excluded")
            positive_q_by_z = None
            if args.positive_q_mode_core or args.positive_q_mode_shell:
                if "interface_mid_z" not in meta:
                    raise SystemExit("Region-aware positive_q_mode requires an interface_mid_z from the builder.")
                positive_q_by_z = (
                    float(meta["interface_mid_z"]),
                    args.positive_q_mode_core or args.positive_q_mode,
                    args.positive_q_mode_shell or args.positive_q_mode,
                )
            syms, pts = charge_balance_iterative(
                syms,
                pts,
                charges,
                args.passivate_ligand,
                verbose=True,
                planes=outer_planes,
                surf_tol=args.surf_tol,
                cif_path=args.shell_cif,
                positive_q_strategy=args.positive_q_mode,
                write_all=False,
                prefix=str(out_dir / f"{args.prefix}_{label}"),
                pair_cuts_override=pair_cuts,
                positive_q_strategy_by_z=positive_q_by_z,
            )
        out_syms, out_pts = _ordered_xyz(syms, pts, charges, args.passivate_ligand)
        write_xyz(str(xyz_path), out_syms, out_pts)
        counts_line = _print_counts(syms, charges)
        print(f"    atoms written: {len(syms)} ({counts_line})")
        print(
            f"    region atoms before overlap cleanup: core={meta['core_atoms']}, "
            f"shell={meta['shell_atoms']}; removed_close_sites={meta['overlap_removed']}"
        )
        footprint = meta.get("shell_footprint_match")
        if footprint and footprint.get("enabled"):
            print(
                "    shell footprint clipped to core: "
                f"{footprint.get('method', 'unknown')}, "
                f"margin={float(footprint.get('margin', 0.0)):.2f} A, "
                f"shell z-half {footprint.get('shell_atoms_before', 0)} -> "
                f"{footprint.get('shell_atoms_after', 0)} atoms"
            )
        actual = meta["actual_interface"]
        print(
            "    actual interface layers: "
            f"core {_fmt_counts_dict(actual['core']['counts'])} "
            f"(Q={actual['core']['charge']:+d})  |  "
            f"shell {_fmt_counts_dict(actual['shell']['counts'])} "
            f"(Q={actual['shell']['charge']:+d})  |  "
            f"Qsum={actual['charge_sum']:+d}"
        )
        if meta.get("zsl_transform_applied"):
            print("    ZSL in-plane transform applied to shell coordinates")
        else:
            print("    ZSL in-plane transform was not applied")
        print(f"    wrote {xyz_path}")

        lm = cand.lattice_match
        q_total = int(sum(charges.get(el, 0) * n for el, n in Counter(syms).items()))
        payload = {
            "candidate_rank": idx,
            "core": {
                "cif": args.core_cif,
                "family": cand.core.family,
                "hkl": cand.core.hkl,
                "charge": cand.core.charge,
                "richness": cand.core.richness,
                "counts": cand.core.counts,
            },
            "shell": {
                "cif": args.shell_cif,
                "family": cand.shell.family,
                "hkl": cand.shell.hkl,
                "charge": cand.shell.charge,
                "richness": cand.shell.richness,
                "counts": cand.shell.counts,
            },
            "compatibility": cand.compatibility,
            "interface_charge_sum": cand.core.charge + cand.shell.charge,
            "lattice_match": None if lm is None else {
                "area": lm.area,
                "max_length_mismatch": lm.max_length_mismatch,
                "angle_mismatch_deg": lm.angle_mismatch_deg,
                "film_transformation": lm.film_transformation,
                "substrate_transformation": lm.substrate_transformation,
            },
            "build": meta,
            "build_mode": mode,
            "passivation": None if not args.passivate_ligand else {
                "ligand": args.passivate_ligand,
                "surf_tol": args.surf_tol,
                "positive_q_mode": args.positive_q_mode,
                "interface_excluded": True,
            },
            "counts": dict(Counter(syms)),
            "total_charge": q_total,
        }
        with json_path.open("w") as fh:
            json.dump(payload, fh, indent=2)

        rows.append({
            "rank": idx,
            "file": xyz_path.name,
            "atoms": len(syms),
            "Q": q_total,
            "core_atoms": meta["core_atoms"],
            "shell_atoms": meta["shell_atoms"],
            "removed": meta["overlap_removed"],
            "core_hkl": hkl_label(cand.core.hkl),
            "core_Q": cand.core.charge,
            "core_term": cand.core.richness,
            "shell_hkl": hkl_label(cand.shell.hkl),
            "shell_Q": cand.shell.charge,
            "shell_term": cand.shell.richness,
            "Qsum_interface": cand.core.charge + cand.shell.charge,
            "match": cand.compatibility,
            "zsl_area": "" if lm is None else f"{lm.area:.4f}",
            "zsl_len_pct": "" if lm is None else f"{100.0 * lm.max_length_mismatch:.4f}",
            "zsl_angle_deg": "" if lm is None else f"{lm.angle_mismatch_deg:.4f}",
            "actual_core_interface": _fmt_counts_dict(actual["core"]["counts"]),
            "actual_core_Q": actual["core"]["charge"],
            "actual_shell_interface": _fmt_counts_dict(actual["shell"]["counts"]),
            "actual_shell_Q": actual["shell"]["charge"],
            "actual_interface_Qsum": actual["charge_sum"],
        })

    summary_path = out_dir / f"{args.prefix}_summary.csv"
    _write_summary(summary_path, rows)
    print(f"\n[6] Wrote summary table: {summary_path}")
    if args.passivate_ligand:
        print("[note] Outer-surface ligand passivation was applied with the buried interface excluded.")
    else:
        print("[note] These are bare inorganic Janus candidates. The buried interface is intentionally unpassivated.")
        print("[note] Add --passivate-ligand <X> to run outer-only ligand passivation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
