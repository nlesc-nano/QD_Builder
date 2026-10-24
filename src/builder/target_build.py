# src/builder/target_build.py
"""
Target mode: the neutral build closest to a set of facet distances h_f and a
target size m.

The recipe gives `facet_distances` (Å from the construction origin) and
`target: {m, w, centres, roundings, max_builds}`. Over the construction
centres c (every species and the empty high-symmetry points of the CIF) and
the layer roundings h_f + r delta_f (r in `roundings`, delta_f the spacing
of the cation layers along the family's normal), every distinct cut is
built, and among the neutral builds the one minimising

    J = ((m_b - m) / m)^2 + w * sum_f (A_f,b / A_b - A_f(h) / A(h))^2

is rebuilt with the full recipe (post-treatments included) and written.
m_b counts formula units of the core, A_f,b are the atom-fitted facet areas
of the build, and A_f(h) those of the target polyhedron P(h). Without
`target.m`, m = V(h) / v_fu (volume per formula unit of the CIF).
"""
from __future__ import annotations

import contextlib
import copy
import io
import itertools
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from .facets import expand_facets, halfspaces, unit_normal
from .nc_types import Facet


# ---------------- lattice helpers ----------------
def layer_spacing(struct, hkl: Tuple[int, int, int]) -> float:
    """
    Spacing of the cation layers along the normal of (hkl): the plane
    spacing of the primitive lattice divided by the distinct cation layers
    per period (zinc blende {111} a/sqrt3, {100} a/2, {110} a/(2 sqrt2);
    wurtzite (001) c/2).
    """
    prim = struct.get_primitive_structure()
    n = unit_normal(struct, hkl)
    rec = prim.lattice.reciprocal_lattice_crystallographic
    c = np.linalg.solve(rec.matrix.T, n)
    c = c / np.abs(c[np.abs(c) > 1e-8]).min()
    for mult in range(1, 49):
        v = c * mult
        if np.allclose(v, np.round(v), atol=1e-3):
            break
    else:
        raise ValueError(f"no lattice plane along ({hkl})")
    ints = np.round(v).astype(int)
    g = math.gcd(math.gcd(abs(int(ints[0])), abs(int(ints[1]))), abs(int(ints[2])))
    ints = tuple(int(x) // g for x in ints)
    period = float(prim.lattice.d_hkl(ints))
    # Distinct layers of one sublattice (the species with fewest sites) per period.
    counts = {}
    for site in prim.sites:
        counts[site.specie.symbol] = counts.get(site.specie.symbol, 0) + 1
    sub = min(counts, key=lambda el: (counts[el], el))
    layers = set()
    for site in prim.sites:
        if site.specie.symbol != sub:
            continue
        x = float(site.coords @ n) % period
        layers.add(0.0 if period - x < 1e-3 else round(x, 3))
    return period / max(1, len(layers))


def _stabiliser_order(ops, f: np.ndarray) -> int:
    return sum(1 for op in ops if np.allclose((op.operate(f) - f + 0.5) % 1 - 0.5, 0, atol=1e-4))


def auto_centres(struct) -> List[Tuple[str, dict]]:
    """
    Construction centres: every species, then the empty points of a 1/4 grid
    of the CIF cell whose site symmetry is as high as the lowest of the
    occupied sites, one per symmetry orbit (zinc blende: the tetrahedral and
    octahedral holes).
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    ops = SpacegroupAnalyzer(struct, symprec=1e-3).get_symmetry_operations(cartesian=False)
    out = [(sp, {"center_on_species": sp}) for sp in dict.fromkeys(s.specie.symbol for s in struct.sites)]
    occ = np.asarray([s.frac_coords for s in struct.sites], float)
    min_order = min(_stabiliser_order(ops, f) for f in occ)
    seen: List[np.ndarray] = [np.asarray(op.operate(f)) % 1 for f in occ for op in ops]
    for f in itertools.product(np.arange(4) / 4.0, repeat=3):
        f = np.asarray(f, float)
        d = struct.lattice.get_cartesian_coords((occ - f + 0.5) % 1 - 0.5)
        if np.min(np.linalg.norm(d, axis=1)) < 0.5:
            continue
        if any(np.allclose((f - s + 0.5) % 1 - 0.5, 0, atol=1e-4) for s in seen):
            continue
        if _stabiliser_order(ops, f) < min_order:
            continue
        seen.extend(np.asarray(op.operate(f)) % 1 for op in ops)
        label = "frac(" + " ".join(f"{x:g}" for x in f) + ")"
        out.append((label, {"fractional_shift": [float(x) for x in f]}))
    return out


def _centre_specs(struct, spec) -> List[Tuple[str, dict]]:
    if spec == "auto":
        return auto_centres(struct)
    out = []
    for c in spec:
        if isinstance(c, str):
            out.append((c, {"center_on_species": c}))
        elif isinstance(c, dict):
            label = ",".join(f"{k}({' '.join(f'{x:g}' for x in v)})" if isinstance(v, list) else f"{k}={v}"
                             for k, v in c.items())
            out.append((label, dict(c)))
        else:
            raise TypeError(f"target.centres entry {c!r} must be a species or a construction_origin mapping")
    return out


# ---------------- target polyhedron ----------------
def _family_seeds(cfg_seeds: Sequence[Facet], families: Dict[str, float]) -> Dict[str, List[Facet]]:
    """Seeds of every facet_distances family (gamma = h_f identifies them)."""
    from .config import _parse_facet_distances, _parse_hkl
    out: Dict[str, List[Facet]] = {}
    for key, item in zip(families, _parse_facet_distances(families)):
        hkl = _parse_hkl(item["hkl"])
        term = item.get("termination")
        seeds = [s for s in cfg_seeds if (s.h, s.k, s.l) == hkl and s.termination == term]
        if term is None:
            seeds += [s for s in cfg_seeds if (s.h, s.k, s.l) == tuple(-x for x in hkl) and s.termination is None]
        out[key] = seeds
    return out


def _wulff_by_family(struct, cfg, families):
    from .main import _resolve_facet_terminations
    out = {}
    for key, seeds in _family_seeds(cfg.seeds, families).items():
        resolved = _resolve_facet_terminations(struct, seeds, cfg.charges)
        out[key] = [f for s in resolved for f in expand_facets(struct, [s], proper_only=cfg.proper_only)]
    return out


def _wulff_for(wulff_by_family, h: Dict[str, float]) -> List[Facet]:
    out, seen = [], set()
    for key, fs in wulff_by_family.items():
        for f in fs:
            if (f.h, f.k, f.l) in seen:
                continue
            seen.add((f.h, f.k, f.l))
            out.append(Facet(f.h, f.k, f.l, float(h[key]), termination=f.termination, scope=f.scope))
    return out


def _area_fractions(areas: Dict[str, float]) -> Dict[str, float]:
    tot = sum(areas.values())
    return {k: v / tot for k, v in areas.items()} if tot > 0 else {}


# ---------------- search ----------------
def _subargv(args, cif: str, yaml_path: str, out: str, *, final: bool) -> List[str]:
    argv = [cif, yaml_path, "-o", out, "--positive-q-mode", args.positive_q_mode,
            "--prune-min-cn", str(args.prune_min_cn), "--prune-passes", str(args.prune_passes)]
    if not getattr(args, "prune_mono", True):
        argv.append("--no-prune-mono")
    for opt in ("prepass_mode", "prepass_min_cn_terrace", "prepass_min_cn_edge", "prepass_min_cn_vertex"):
        val = getattr(args, opt, None)
        if val is not None:
            argv += ["--" + opt.replace("_", "-"), str(val)]
    po = getattr(args, "proper_rotations_only", None)
    if po is not None:
        argv.append("--proper-rotations-only" if po else "--no-proper-rotations-only")
    if final:
        for flag in ("write_all", "center", "verbose"):
            if getattr(args, flag, False):
                argv.append("--" + flag.replace("_", "-"))
    return argv


def _cut_key(struct, wulff, h, origin_spec) -> Tuple:
    from .geometry import build_nanocrystal, dedupe_points
    from .main import construction_origin_shift
    with contextlib.redirect_stdout(io.StringIO()):
        syms, pts, _planes = build_nanocrystal(
            struct, wulff, min(h.values()), aspect=(1.0, 1.0, 1.0),
            origin_shift=construction_origin_shift(struct, origin_spec),
        )
    syms, pts = dedupe_points(syms, pts, tol=1e-3)
    return tuple(sorted(zip(syms, map(tuple, np.round(pts, 2).tolist()))))


def _realised(geo: dict) -> Dict[str, dict]:
    """Atom-fitted plane distances per family label: mean, min, max (Å)."""
    by: Dict[str, List[float]] = {}
    for pl in geo.get("planes", []):
        if pl["area_A2"] > 0:
            by.setdefault(pl["label"], []).append(pl["d_A"])
    return {k: {"mean": round(float(np.mean(v)), 3), "min": round(min(v), 3), "max": round(max(v), 3)}
            for k, v in sorted(by.items())}


def run_target_mode(args, cfg, struct) -> int:
    """Search centres and layer roundings, write the best neutral build to args.out."""
    from .main import main as builder_main
    from .polyhedron import clip, facet_labels, summary

    raw = yaml.safe_load(Path(args.yaml).read_text())
    target = cfg.target
    families = dict(cfg.facet_distances)
    w = float(target["w"])

    # The target polyhedron P(h): the planes at h_f, no shift off atom layers.
    wbf = _wulff_by_family(struct, cfg, families)
    wulff0 = _wulff_for(wbf, families)
    planes0 = halfspaces(struct, wulff0, min(families.values()), aspect=(1.0, 1.0, 1.0))
    labels0 = facet_labels(struct, wulff0, cfg.charges)
    poly0 = summary(clip(planes0, labels0), [(f.h, f.k, f.l) for f in wulff0])
    frac0 = _area_fractions(poly0["areas_A2"])
    comp = struct.composition.element_composition
    n_fu = comp.num_atoms / comp.reduced_composition.num_atoms
    v_fu = struct.volume / n_fu
    m_target = target["m"] if target["m"] is not None else poly0["volume_A3"] / v_fu

    from .config import _parse_facet_distances, _parse_hkl
    deltas = {key: layer_spacing(struct, _parse_hkl(item["hkl"]))
              for key, item in zip(families, _parse_facet_distances(families))}
    centres = _centre_specs(struct, target["centres"])
    combos = sorted(itertools.product(target["roundings"], repeat=len(families)),
                    key=lambda r: (sum(abs(x) for x in r), r))
    print(f"[target] h = {families} Å, m = {m_target:.1f}, w = {w:g}; delta_f = "
          + ", ".join(f"{k}: {v:.3f}" for k, v in deltas.items())
          + f"; {len(centres)} centres x {len(combos)} roundings")

    cand_raw = copy.deepcopy(raw)
    cand_raw.pop("target", None)
    post = cand_raw.get("post_treatment") or cand_raw.get("post-treatment")
    if isinstance(post, dict):            # ligand-only steps do not change the core
        for k in ("z_type_ligands", "neutral_ligands"):
            post.pop(k, None)

    candidates: List[dict] = []
    seen: Dict[Tuple, int] = {}
    n_builds = 0
    best = None
    truncated = False
    with tempfile.TemporaryDirectory() as tmp:
        for label, origin in centres:
            for r in combos:
                h = {k: families[k] + ri * deltas[k] for k, ri in zip(families, r)}
                rec = {"centre": label, "roundings": dict(zip(families, r)),
                       "h_A": {k: round(v, 4) for k, v in h.items()}}
                if min(h.values()) <= 0:
                    candidates.append({**rec, "status": "skipped: non-positive distance"})
                    continue
                key = (_cut_key(struct, _wulff_for(wbf, h), h, origin),)
                if key in seen:
                    candidates.append({**rec, "status": f"same cut as candidate {seen[key]}", "same_as": seen[key]})
                    continue
                seen[key] = len(candidates)
                if n_builds >= target["max_builds"]:
                    truncated = True
                    candidates.append({**rec, "status": "not built: max_builds reached"})
                    continue
                n_builds += 1
                r_yaml = copy.deepcopy(cand_raw)
                r_yaml["facet_distances"] = rec["h_A"]
                r_yaml["construction_origin"] = origin
                yp = os.path.join(tmp, f"c{len(candidates)}.yaml")
                op = os.path.join(tmp, f"c{len(candidates)}.xyz")
                Path(yp).write_text(yaml.safe_dump(r_yaml, sort_keys=False))
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        builder_main(_subargv(args, args.cif, yp, op, final=False))
                    man = json.loads(Path(op).with_suffix(".json").read_text())
                except (Exception, SystemExit) as exc:
                    candidates.append({**rec, "status": f"failed: {type(exc).__name__}: {str(exc)[:120]}"})
                    continue
                q = man.get("total_charge")
                geo = (man.get("facet_geometry") or {}).get("atom_fitted") or {}
                m_b = (man.get("composition") or {}).get("m")
                rec.update(total_charge=q, m_b=m_b, counts=dict(man.get("counts", {})))
                if q != 0 or m_b is None or "areas_A2" not in geo:
                    rec["status"] = "rejected: " + ("charge %+d" % q if q != 0 else "no geometry")
                    candidates.append(rec)
                    continue
                frac_b = _area_fractions(geo["areas_A2"])
                j_m = ((m_b - m_target) / m_target) ** 2
                j_s = sum((frac_b.get(k, 0.0) - frac0.get(k, 0.0)) ** 2 for k in set(frac_b) | set(frac0))
                rec.update(status="neutral", J=round(j_m + w * j_s, 6), J_m=round(j_m, 6), J_shape=round(j_s, 6),
                           areas_A2=geo["areas_A2"], h_fitted_A=_realised(geo))
                candidates.append(rec)
                if best is None or (rec["J"], abs(m_b - m_target)) < (best[0]["J"], abs(best[0]["m_b"] - m_target)):
                    best = (rec, r_yaml)

    if best is None:
        raise ValueError(f"target mode: no neutral build among {n_builds} distinct cuts "
                         f"({len(candidates)} candidates)")
    rec, r_yaml = best
    final = copy.deepcopy(raw)
    final.pop("target", None)
    final["facet_distances"] = r_yaml["facet_distances"]
    final["construction_origin"] = r_yaml["construction_origin"]
    out_yaml = os.path.splitext(args.out)[0] + "_target.yaml"
    Path(out_yaml).write_text(yaml.safe_dump(final, sort_keys=False))
    rc = builder_main(_subargv(args, args.cif, out_yaml, args.out, final=True))
    man_path = Path(os.path.splitext(args.out)[0] + ".json")
    man = json.loads(man_path.read_text())
    man["target_search"] = {
        "target": {"facet_distances_A": families, "m": round(float(m_target), 3), "w": w,
                   "m_from_volume": target["m"] is None, "volume_A3": poly0["volume_A3"],
                   "areas_A2": poly0["areas_A2"], "area_fractions": {k: round(v, 5) for k, v in frac0.items()}},
        "layer_spacing_A": {k: round(v, 4) for k, v in deltas.items()},
        "best": {k: rec[k] for k in ("centre", "roundings", "h_A", "m_b", "J", "J_m", "J_shape",
                                     "areas_A2", "h_fitted_A")},
        "recipe": os.path.basename(out_yaml),
        "n_candidates": len(candidates),
        "n_builds": n_builds,
        "truncated": truncated,
        "candidates": candidates,
    }
    man_path.write_text(json.dumps(man, indent=2, default=int))
    print(f"[target] best: centre {rec['centre']}, roundings {rec['roundings']}, m_b = {rec['m_b']} "
          f"(target {m_target:.1f}), J = {rec['J']:.4g}; {n_builds} builds of {len(candidates)} candidates")
    return int(rc or 0)
