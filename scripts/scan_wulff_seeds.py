#!/usr/bin/env python3
"""Scan continuum Wulff cuts for discrete seed sizes (composition staircase).

Uses the same half-space Wulff construction as QD_builder
(``geometry.build_nanocrystal`` / ``facets.halfspaces``), but sweeps:

  - isotropic ``size_unit_cells`` (or an explicit list)
  - construction origins (COM, species-centred, …)
  - ``auto_shift_planes`` on/off

Output (under ``-o`` / current directory by default)::

  wulff_seed_search/
    seed_search.csv
    seed_search.json
    xyz/   optional cut XYZ for selected rows

Example::

  python scripts/scan_wulff_seeds.py \\
    examples/core-only/cdse_wulff_seed_search.yaml \\
    -o /tmp/cdse_seed_scan

This documents why Se-centred size≈1 lands near k≈13 and which smaller
(often non-stoichiometric) plateaus exist at sub-cell radii.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from pymatgen.core import Structure

# Allow running from repo root without install
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from builder.facets import expand_facets, halfspaces  # noqa: E402
from builder.geometry import (  # noqa: E402
    PLANE_EPS,
    auto_shift_planes,
    dedupe_points,
    inside,
    rep_ranges,
)
from builder.io_utils import write_xyz  # noqa: E402
from builder.main import construction_origin_shift  # noqa: E402
from builder.nc_types import Facet  # noqa: E402
from builder.stack import size_unit_cells_to_radius_aspect  # noqa: E402


def _parse_hkl(raw: Any) -> Tuple[int, int, int]:
    if isinstance(raw, str):
        s = raw.strip().replace(" ", "")
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1]
        # support "111", "1,1,1", "1 1 1", "-1-1-1"
        if "," in s:
            parts = s.split(",")
        elif " " in s:
            parts = s.split()
        else:
            import re

            # signed packed: -1-1-1 or 1-10
            parts = re.findall(r"-?\d+", s)
            if len(parts) == 1 and re.fullmatch(r"-?\d{3}", s):
                # "100", "111", "-11" invalid; three single digits with optional leading -
                m = re.fullmatch(r"(-?\d)(-?\d)(-?\d)", s)
                if m:
                    parts = list(m.groups())
                elif re.fullmatch(r"\d{3}", s):
                    parts = list(s)
        if len(parts) != 3:
            raise ValueError(f"cannot parse hkl {raw!r}")
        return int(parts[0]), int(parts[1]), int(parts[2])
    if isinstance(raw, Sequence) and len(raw) == 3:
        return int(raw[0]), int(raw[1]), int(raw[2])
    raise ValueError(f"cannot parse hkl {raw!r}")


def _load_facets(data: Mapping[str, Any]) -> List[Facet]:
    facets: List[Facet] = []
    for item in data.get("facets") or []:
        h, k, l = _parse_hkl(item["hkl"])
        facets.append(
            Facet(
                h=h,
                k=k,
                l=l,
                gamma=float(item.get("gamma", 1.0)),
                termination=item.get("termination"),
                scope=str(item.get("scope", "family")),
            )
        )
    if not facets:
        raise ValueError("YAML must define at least one facet")
    return facets


def _size_grid(search: Mapping[str, Any]) -> List[float]:
    if search.get("sizes"):
        return [float(x) for x in search["sizes"]]
    start = float(search.get("size_start", 0.4))
    stop = float(search.get("size_stop", 1.5))
    step = float(search.get("size_step", 0.05))
    if step <= 0:
        raise ValueError("size_step must be > 0")
    out: List[float] = []
    x = start
    while x <= stop + 1e-12:
        out.append(round(x, 10))
        x += step
    return out


def _origins(
    struct: Structure,
    search: Mapping[str, Any],
) -> List[Tuple[str, np.ndarray]]:
    """Return (label, cartesian origin used as construction shift)."""

    raw = search.get("origins")
    if not raw:
        raw = ["com", {"species": "Se"}, {"species": "Cd"}]
    out: List[Tuple[str, np.ndarray]] = []
    for item in raw:
        if item == "com" or (isinstance(item, Mapping) and item.get("type") == "com"):
            out.append(("com", struct.cart_coords.mean(axis=0).copy()))
            continue
        if isinstance(item, str) and item.lower() in {"com", "centroid"}:
            out.append(("com", struct.cart_coords.mean(axis=0).copy()))
            continue
        if isinstance(item, Mapping):
            if "species" in item or "center_on_species" in item:
                sp = str(item.get("species", item.get("center_on_species")))
                shift = construction_origin_shift(
                    struct, {"center_on_species": sp}
                )
                if shift is None:
                    raise ValueError(f"no sites for species {sp}")
                out.append((f"species:{sp}", np.asarray(shift, float)))
                continue
            if "cartesian" in item:
                out.append(
                    (
                        "cartesian",
                        np.asarray(item["cartesian"], float),
                    )
                )
                continue
        raise ValueError(f"unsupported origin entry: {item!r}")
    return out


def _build_cut(
    struct: Structure,
    facets: List[Facet],
    radius: float,
    aspect: Tuple[float, float, float],
    origin: np.ndarray,
    *,
    do_auto_shift: bool,
) -> Tuple[List[str], np.ndarray, List, Dict[str, Any]]:
    planes = halfspaces(struct, facets, radius, aspect=aspect)
    base = struct.frac_coords @ struct.lattice.matrix - np.asarray(origin, float)
    site_symbols = [site.specie.symbol for site in struct.sites]
    maxd = max(d for _, d in planes)
    rx, ry, rz = rep_ranges(struct.lattice, maxd)

    all_coords = []
    for i, j, k in product(rx, ry, rz):
        shift = (
            i * struct.lattice.matrix[0]
            + j * struct.lattice.matrix[1]
            + k * struct.lattice.matrix[2]
        )
        all_coords.append(base + shift)
    all_coords = np.vstack(all_coords)

    n_shifted = 0
    if do_auto_shift:
        before = list(planes)
        # Suppress noisy prints from auto_shift by temporarily redirecting
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            planes = auto_shift_planes(
                all_coords, planes, threshold=0.05, shift_amount=0.25
            )
        n_shifted = sum(
            1
            for (_n0, d0), (_n1, d1) in zip(before, planes)
            if abs(d1 - d0) > 1e-12
        )
        maxd = max(d for _, d in planes)
        rx, ry, rz = rep_ranges(struct.lattice, maxd)

    syms: List[str] = []
    pts: List[np.ndarray] = []
    for i, j, k in product(rx, ry, rz):
        shift = (
            i * struct.lattice.matrix[0]
            + j * struct.lattice.matrix[1]
            + k * struct.lattice.matrix[2]
        )
        coords = base + shift
        mask = inside(coords, planes)
        idxs = np.where(mask)[0]
        if idxs.size:
            syms.extend(site_symbols[idx] for idx in idxs.tolist())
            pts.extend(coords[idxs])
    syms, pts_arr = dedupe_points(syms, np.asarray(pts, float), tol=1e-3)
    meta = {
        "n_planes": len(planes),
        "n_planes_auto_shifted": n_shifted,
        "max_plane_d": float(max(d for _, d in planes)) if planes else 0.0,
    }
    return syms, pts_arr, planes, meta


def _cation_anion(charges: Mapping[str, int]) -> Tuple[Optional[str], Optional[str]]:
    cats = [el for el, q in charges.items() if q > 0]
    ans = [el for el, q in charges.items() if q < 0]
    # Prefer inorganic: ignore Cl-like if multiple
    ligand_like = {"Cl", "Br", "I", "F", "H"}
    cats = [c for c in cats if c not in ligand_like] or cats
    ans = [a for a in ans if a not in ligand_like] or ans
    return (cats[0] if cats else None, ans[0] if ans else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "yaml",
        type=Path,
        help="Builder-style YAML with facets/charges and optional seed_search block",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("wulff_seed_search"),
        help="Output directory (default: ./wulff_seed_search)",
    )
    parser.add_argument(
        "--write-xyz",
        action="store_true",
        help="Write cut XYZ for rows that pass write_xyz_min_k",
    )
    args = parser.parse_args(argv)

    yaml_path = args.yaml.resolve()
    data = yaml.safe_load(yaml_path.read_text()) or {}
    search = data.get("seed_search") or {}
    cif_raw = data.get("cif")
    if not cif_raw:
        raise SystemExit("YAML must define cif:")
    cif_path = Path(cif_raw)
    if not cif_path.is_absolute():
        cif_path = (yaml_path.parent / cif_path).resolve()

    struct = Structure.from_file(str(cif_path))
    charges = {str(k): int(v) for k, v in (data.get("charges") or {}).items()}
    cation, anion = _cation_anion(charges)
    proper = bool((data.get("symmetry") or {}).get("proper_rotations_only", True))
    seed_facets = _load_facets(data)
    wulff_facets = expand_facets(struct, seed_facets, proper_only=proper)

    sizes = _size_grid(search)
    origins = _origins(struct, search)
    auto_flags = search.get("auto_shift", [False, True])
    auto_flags = [bool(x) for x in auto_flags]
    write_min_k = int(search.get("write_xyz_min_k", 7))
    max_imbalance = search.get("max_stoich_imbalance", None)
    if max_imbalance is not None:
        max_imbalance = int(max_imbalance)

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz_dir = out_dir / "xyz"
    if args.write_xyz:
        xyz_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    print(
        f"CIF={cif_path.name} facets={len(wulff_facets)} "
        f"sizes={len(sizes)} origins={len(origins)} auto_shift={auto_flags}"
    )
    print(
        f"{'size':>5} {'R':>7} {'origin':>12} {'shift':>5} "
        f"{'N':>4} {'k':>4} {'imbal':>5} counts"
    )

    for size, (olabel, origin), do_shift in product(sizes, origins, auto_flags):
        R, aspect = size_unit_cells_to_radius_aspect(
            struct, (size, size, size)
        )
        syms, pts, _planes, meta = _build_cut(
            struct,
            wulff_facets,
            R,
            aspect,
            origin,
            do_auto_shift=do_shift,
        )
        counts = dict(Counter(syms))
        n_cat = int(counts.get(cation or "", 0))
        n_an = int(counts.get(anion or "", 0))
        k_core = min(n_cat, n_an) if (cation and anion) else 0
        imbalance = abs(n_cat - n_an)
        if max_imbalance is not None and imbalance > max_imbalance:
            continue

        row = {
            "size_unit_cells": size,
            "radius_angstrom": round(R, 6),
            "origin": olabel,
            "auto_shift": do_shift,
            "n_atoms": len(syms),
            "k_core": k_core,
            "n_cation": n_cat,
            "n_anion": n_an,
            "stoich_imbalance": imbalance,
            "counts": counts,
            "cation": cation,
            "anion": anion,
            **meta,
        }

        if args.write_xyz and k_core >= write_min_k and len(syms) > 0:
            fname = (
                f"size{size:g}_R{R:.3f}_{olabel.replace(':', '-')}"
                f"_shift{int(do_shift)}_k{k_core}.xyz"
            )
            path = xyz_dir / fname
            write_xyz(
                str(path),
                syms,
                pts,
                comment=(
                    f"wulff_seed size={size} R={R:.4f} origin={olabel} "
                    f"auto_shift={do_shift} k_core={k_core}"
                ),
            )
            row["xyz"] = str(path.relative_to(out_dir))

        rows.append(row)
        print(
            f"{size:5.2f} {R:7.3f} {olabel:>12} {str(do_shift):>5} "
            f"{len(syms):4d} {k_core:4d} {imbalance:5d} {counts}"
        )

    # CSV
    csv_path = out_dir / "seed_search.csv"
    fieldnames = [
        "size_unit_cells",
        "radius_angstrom",
        "origin",
        "auto_shift",
        "n_atoms",
        "k_core",
        "n_cation",
        "n_anion",
        "stoich_imbalance",
        "counts",
        "n_planes",
        "n_planes_auto_shifted",
        "max_plane_d",
        "xyz",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            r = dict(row)
            r["counts"] = json.dumps(row["counts"], sort_keys=True)
            writer.writerow(r)

    # JSON + plateau summary
    plateaus: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            f"origin={row['origin']}|shift={row['auto_shift']}|"
            f"k={row['k_core']}|N={row['n_atoms']}|{json.dumps(row['counts'], sort_keys=True)}"
        )
        plateaus.setdefault(key, []).append(
            {
                "size_min": row["size_unit_cells"],
                "size_max": row["size_unit_cells"],
                "R_min": row["radius_angstrom"],
                "R_max": row["radius_angstrom"],
            }
        )
    # merge contiguous sizes per key
    plateau_list = []
    for key, chunks in plateaus.items():
        sizes_k = sorted(c["size_min"] for c in chunks)
        # re-derive from rows
        matching = [
            r
            for r in rows
            if (
                f"origin={r['origin']}|shift={r['auto_shift']}|"
                f"k={r['k_core']}|N={r['n_atoms']}|{json.dumps(r['counts'], sort_keys=True)}"
            )
            == key
        ]
        matching.sort(key=lambda r: r["size_unit_cells"])
        plateau_list.append(
            {
                "origin": matching[0]["origin"],
                "auto_shift": matching[0]["auto_shift"],
                "k_core": matching[0]["k_core"],
                "n_atoms": matching[0]["n_atoms"],
                "counts": matching[0]["counts"],
                "size_min": matching[0]["size_unit_cells"],
                "size_max": matching[-1]["size_unit_cells"],
                "R_min": matching[0]["radius_angstrom"],
                "R_max": matching[-1]["radius_angstrom"],
                "n_samples": len(matching),
            }
        )
    plateau_list.sort(
        key=lambda p: (p["origin"], p["auto_shift"], p["size_min"], p["k_core"])
    )

    summary = {
        "cif": str(cif_path),
        "yaml": str(yaml_path),
        "cation": cation,
        "anion": anion,
        "n_facets": len(wulff_facets),
        "lattice_a_angstrom": float(struct.lattice.a),
        "n_rows": len(rows),
        "rows": rows,
        "plateaus": plateau_list,
        "note": (
            "k_core = min(n_cation, n_anion) on the raw Wulff cut before "
            "charge balance/passivation. Plateaus show discrete magic sizes."
        ),
    }
    json_path = out_dir / "seed_search.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"\nwrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"{len(plateau_list)} composition plateaus")
    # Highlight near-stoichiometric plateaus with k>=3
    print("\nNear-stoichiometric plateaus (imbalance<=2, k_core>=3):")
    for p in plateau_list:
        imb = abs(p["counts"].get(cation or "", 0) - p["counts"].get(anion or "", 0))
        if p["k_core"] >= 3 and imb <= 2:
            print(
                f"  k={p['k_core']:3d} N={p['n_atoms']:3d} "
                f"size=[{p['size_min']:g},{p['size_max']:g}] "
                f"origin={p['origin']} shift={p['auto_shift']} "
                f"{p['counts']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
