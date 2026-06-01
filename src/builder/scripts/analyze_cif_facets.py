#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
    from pymatgen.core import Structure
    from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:
    raise SystemExit("Install runtime deps first, e.g. pymatgen and pyyaml.")


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


def _read_charges(args) -> dict[str, int]:
    charges: dict[str, int] = {}
    if args.yaml:
        charges.update(_charges_from_yaml(Path(args.yaml)))
    for token in args.charges or []:
        el, q = _parse_charge_token(token)
        charges[el] = q
    if not charges:
        raise SystemExit("Provide charges either via YAML or --charges Cd=+2 Se=-2")
    return charges


def _gcd3(hkl: tuple[int, int, int]) -> int:
    return math.gcd(math.gcd(abs(hkl[0]), abs(hkl[1])), abs(hkl[2])) or 1


def _primitive_hkl(hkl) -> tuple[int, int, int]:
    hkl = tuple(int(round(x)) for x in hkl)
    g = _gcd3(hkl)
    return (hkl[0] // g, hkl[1] // g, hkl[2] // g)


def _hkl_label(hkl: tuple[int, int, int]) -> str:
    return "(" + " ".join(f"{x:+d}" if x < 0 else str(x) for x in hkl) + ")"


def _hkl_compact(hkl: tuple[int, int, int]) -> str:
    return "(" + "".join(str(x) for x in hkl) + ")"


def _family_label(hkl: tuple[int, int, int]) -> str:
    return "{" + "".join(str(x) for x in hkl) + "}"


def _unit_normal(struct: Structure, hkl: tuple[int, int, int]) -> np.ndarray:
    v = struct.lattice.reciprocal_lattice.get_cartesian_coords(hkl)
    return v / np.linalg.norm(v)


def _signed_equivalents(
    struct: Structure,
    hkl: tuple[int, int, int],
    *,
    proper_only: bool,
    include_opposites: bool = True,
    ops: list | None = None,
) -> list[tuple[int, int, int]]:
    if ops is None:
        ops = SpacegroupAnalyzer(struct, symprec=1e-3).get_symmetry_operations(cartesian=True)
    rec_t = struct.lattice.reciprocal_lattice.matrix.T
    g0 = struct.lattice.reciprocal_lattice.get_cartesian_coords(hkl)
    out: set[tuple[int, int, int]] = set()
    for op in ops:
        if proper_only and np.linalg.det(op.rotation_matrix) < 0.999:
            continue
        g_rot = op.rotation_matrix @ g0
        coeff = np.linalg.solve(rec_t, g_rot)
        hkl_i = _primitive_hkl(coeff)
        if hkl_i != (0, 0, 0):
            out.add(hkl_i)
    out.add(_primitive_hkl(hkl))
    if include_opposites:
        out.update(tuple(-x for x in item) for item in list(out))
    return sorted(out, key=lambda t: (abs(t[0]) + abs(t[1]) + abs(t[2]), t))


def _bulk_formula_counts(struct: Structure, charges: dict[str, int]) -> Counter:
    counts = Counter(str(site.specie.symbol) for site in struct.sites if str(site.specie.symbol) in charges)
    if not counts:
        counts = Counter(str(site.specie.symbol) for site in struct.sites)
    g = 0
    for v in counts.values():
        g = math.gcd(g, int(v))
    if g > 1:
        return Counter({k: v // g for k, v in counts.items()})
    return counts


def _is_stoich_multiple(counts: Counter, formula: Counter) -> bool:
    if not counts:
        return False
    elems = set(formula)
    if set(counts) != elems:
        return False
    ratios = []
    for el in sorted(elems):
        if formula[el] == 0:
            return False
        ratios.append(counts[el] / formula[el])
    return max(ratios) - min(ratios) < 1e-8


def _layer_groups(
    struct: Structure,
    hkl: tuple[int, int, int],
    charges: dict[str, int],
    *,
    supercell: int,
    layer_tol: float,
) -> list[dict[str, Any]]:
    gvec = struct.lattice.reciprocal_lattice.get_cartesian_coords(hkl)
    period = 1.0 / np.linalg.norm(gvec)
    phase_tol = layer_tol / period

    frac = np.asarray([site.frac_coords for site in struct.sites], dtype=float)
    phases = (frac @ np.asarray(hkl, dtype=float)) % 1.0
    order = np.argsort(phases)

    raw_groups: list[list[int]] = []
    current: list[int] = []
    last = None
    for idx in order:
        p = float(phases[idx])
        if last is None or abs(p - last) <= phase_tol:
            current.append(int(idx))
        else:
            raw_groups.append(current)
            current = [int(idx)]
        last = p
    if current:
        raw_groups.append(current)

    if len(raw_groups) > 1:
        first = raw_groups[0]
        last_group = raw_groups[-1]
        gap_wrap = (float(phases[first[0]]) + 1.0) - float(phases[last_group[-1]])
        if gap_wrap <= phase_tol:
            raw_groups = [last_group + first] + raw_groups[1:-1]

    formula = _bulk_formula_counts(struct, charges)
    layers = []
    for group in raw_groups:
        counts = Counter(str(struct.sites[i].specie.symbol) for i in group)
        q = int(sum(charges.get(el, 0) * n_el for el, n_el in counts.items()))
        pos = float(np.mean([phases[i] for i in group]) * period)
        layers.append({
            "position_A": pos,
            "counts": dict(sorted(counts.items())),
            "charge": q,
            "stoich_multiple": _is_stoich_multiple(counts, formula),
        })

    layers.sort(key=lambda row: row["position_A"])
    return layers


def _dedupe_layer_patterns(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for layer in layers:
        key = (tuple(sorted(layer["counts"].items())), layer["charge"])
        if key in seen:
            continue
        seen.add(key)
        out.append(layer)
    return out


def _classify_layers(patterns: list[dict[str, Any]]) -> str:
    if not patterns:
        return "unknown"
    charges = [int(row["charge"]) for row in patterns]
    if all(q == 0 for q in charges):
        if all(bool(row["stoich_multiple"]) for row in patterns):
            return "non-polar stoichiometric"
        return "non-polar mixed-neutral"
    if any(q > 0 for q in charges) and any(q < 0 for q in charges):
        return "polar"
    return "charged/mixed termination"


def _has_charged_opposite_layers(patterns: list[dict[str, Any]]) -> bool:
    charges = [int(row["charge"]) for row in patterns]
    return any(q > 0 for q in charges) and any(q < 0 for q in charges)


def _richness(layer: dict[str, Any], charges: dict[str, int]) -> str:
    q = int(layer["charge"])
    if q > 0:
        positives = [el for el in layer["counts"] if charges.get(el, 0) > 0]
        return "cation-rich" if positives else "positive"
    if q < 0:
        negatives = [el for el in layer["counts"] if charges.get(el, 0) < 0]
        return "anion-rich" if negatives else "negative"
    if layer["stoich_multiple"]:
        return "stoichiometric"
    return "neutral mixed"


def _fmt_counts(counts: dict[str, int]) -> str:
    return " ".join(f"{el}:{n}" for el, n in counts.items())


def _equivalent_hkls(row: dict[str, Any]) -> str:
    return ", ".join(_hkl_compact(item["hkl"]) for item in row["signed"])


def _yaml_hint(row: dict[str, Any]) -> str:
    family = row["family"].strip("{}")
    status = row["family_status"]
    if status == "polar":
        return (
            f'family: "{family}", termination: cation_rich or anion_rich'
        )
    if status == "termination-sensitive":
        return (
            f'family: "{family}", termination: cation_rich or anion_rich'
        )
    if status == "non-polar":
        return f'family: "{family}"'
    return f'family: "{family}" plus manual inspection'


def _simple_classification(row: dict[str, Any]) -> str:
    status = row["family_status"]
    if status == "polar":
        return "polar; cation-rich and anion-rich signs are symmetry-distinct"
    if status == "termination-sensitive":
        return "termination-sensitive; alternating charged terminations are symmetry-equivalent"
    if status == "non-polar":
        return "non-polar / stoichiometric; no termination keyword needed"
    if status == "charged/mixed termination":
        return "charged or mixed; inspect terminations"
    return status


def _termination_summary(row: dict[str, Any], charges: dict[str, int]) -> str:
    if row["family_status"] not in {"polar", "termination-sensitive"}:
        return ""
    cation_terms = []
    anion_terms = []
    for signed in row["signed"]:
        for pattern in signed["patterns"]:
            rich = _richness(pattern, charges)
            if rich == "cation-rich":
                cation_terms.append(_hkl_compact(signed["hkl"]))
            elif rich == "anion-rich":
                anion_terms.append(_hkl_compact(signed["hkl"]))
    cation_terms = sorted(set(cation_terms))
    anion_terms = sorted(set(anion_terms))
    if cation_terms and anion_terms:
        if row["family_status"] == "polar":
            return (
                "Both cation-rich and anion-rich terminations are available and symmetry-distinct. "
                "Use the termination keyword rather than hard-coding the CIF sign."
            )
        return (
            "This family has alternating charged terminations even though opposite signs are symmetry-equivalent. "
            "Use the termination keyword when you need a specific cation- or anion-terminated cut."
        )
    return ""


def _analyze(
    cif: Path,
    charges: dict[str, int],
    *,
    max_index: int,
    proper_only: bool,
    supercell: int,
    layer_tol: float,
) -> list[dict[str, Any]]:
    struct = Structure.from_file(str(cif))
    missing = sorted({str(site.specie.symbol) for site in struct.sites} - set(charges))
    if missing:
        print(f"[warn] charges missing for CIF species {missing}; they contribute Q=0", file=sys.stderr)

    ops = SpacegroupAnalyzer(struct, symprec=1e-3).get_symmetry_operations(cartesian=True)
    reps = get_symmetrically_distinct_miller_indices(struct, max_index=max_index)
    rows = []
    for rep in sorted(reps, key=lambda t: (abs(t[0]) + abs(t[1]) + abs(t[2]), t)):
        symmetry_signed = set(_signed_equivalents(
            struct,
            tuple(rep),
            proper_only=proper_only,
            include_opposites=False,
            ops=ops,
        ))
        signed = _signed_equivalents(
            struct,
            tuple(rep),
            proper_only=proper_only,
            include_opposites=True,
            ops=ops,
        )
        signed_rows = []
        classifications = []
        has_polar_split = False
        has_charged_terminations = False
        for hkl in signed:
            layers = _layer_groups(
                struct,
                hkl,
                charges,
                supercell=supercell,
                layer_tol=layer_tol,
            )
            patterns = _dedupe_layer_patterns(layers)
            status = _classify_layers(patterns)
            if _has_charged_opposite_layers(patterns):
                has_charged_terminations = True
            opposite = tuple(-x for x in hkl)
            if (
                _has_charged_opposite_layers(patterns)
                and hkl in symmetry_signed
                and opposite not in symmetry_signed
            ):
                has_polar_split = True
            classifications.append(status)
            signed_rows.append({
                "hkl": hkl,
                "status": status,
                "patterns": patterns,
            })

        if has_polar_split:
            family_status = "polar"
        elif has_charged_terminations:
            family_status = "termination-sensitive"
        elif all(c.startswith("non-polar") for c in classifications):
            family_status = "non-polar"
        else:
            family_status = "mixed/ambiguous"

        rows.append({
            "family": _family_label(tuple(rep)),
            "representative": tuple(rep),
            "family_status": family_status,
            "multiplicity": len(signed),
            "signed": signed_rows,
        })
    return rows


def _write_markdown(
    path: Path,
    cif: Path,
    charges: dict[str, int],
    rows: list[dict[str, Any]],
    *,
    details: bool,
) -> None:
    with path.open("w") as fh:
        fh.write("# CIF Facet Analysis\n\n")
        fh.write(f"- CIF: `{cif}`\n")
        fh.write("- Charges: " + ", ".join(f"{el}={q:+d}" for el, q in sorted(charges.items())) + "\n\n")

        fh.write("## Summary\n\n")
        fh.write("| family | equivalent signed facets | classification | YAML hint |\n")
        fh.write("| --- | --- | --- | --- |\n")
        for row in rows:
            fh.write(
                f"| {row['family']} | {_equivalent_hkls(row)} | "
                f"{_simple_classification(row)} | `{_yaml_hint(row)}` |\n"
            )
            term_note = _termination_summary(row, charges)
            if term_note:
                fh.write(f"\n> {row['family']}: {term_note}\n\n")

        if not details:
            return

        fh.write("\n## Details\n\n")
        for row in rows:
            fh.write(f"### {row['family']} representative {_hkl_label(row['representative'])}\n\n")
            fh.write(f"Family classification: **{row['family_status']}**\n\n")
            for signed in row["signed"]:
                fh.write(f"- `{_hkl_label(signed['hkl'])}`: {signed['status']}\n")
                for pattern in signed["patterns"]:
                    fh.write(
                        f"  - Q={pattern['charge']:+d}, "
                        f"{_richness(pattern, charges)}, "
                        f"{_fmt_counts(pattern['counts'])}\n"
                    )
                fh.write("\n")


def _print_text(cif: Path, charges: dict[str, int], rows: list[dict[str, Any]], *, details: bool) -> None:
    print(f"CIF: {cif}")
    print("Charges: " + ", ".join(f"{el}={q:+d}" for el, q in sorted(charges.items())))
    print("\nFACET SUMMARY")
    for row in rows:
        print(f"\n{row['family']}")
        print(f"  equivalent facets: {_equivalent_hkls(row)}")
        print(f"  classification:    {_simple_classification(row)}")
        print(f"  YAML hint:         {_yaml_hint(row)}")
        term_note = _termination_summary(row, charges)
        if term_note:
            print(f"  note:              {term_note}")

    if not details:
        return

    print("\nDETAILS")
    for row in rows:
        print(f"\n{row['family']} representative {_hkl_label(row['representative'])}: {row['family_status']}")
        for signed in row["signed"]:
            print(f"  {_hkl_label(signed['hkl']):>12s}: {signed['status']}")
            for pattern in signed["patterns"]:
                print(
                    f"      Q={pattern['charge']:+d}  "
                    f"{_richness(pattern, charges):18s}  "
                    f"{_fmt_counts(pattern['counts'])}"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze CIF facet families and periodic layer polarity without building a nanocrystal."
    )
    parser.add_argument("cif", help="Input CIF")
    parser.add_argument("yaml", nargs="?", help="Optional builder YAML from which to read charges")
    parser.add_argument("--charges", nargs="*", help="Charges, e.g. Cd=+2 Se=-2 Cl=-1")
    parser.add_argument("--max-index", type=int, default=2, help="Max Miller index to scan")
    parser.add_argument("--all-rotations", action="store_true", help="Include improper symmetry operations")
    parser.add_argument("--supercell", type=int, default=3, help="Bulk supercell used for layer detection")
    parser.add_argument("--layer-tol", type=float, default=0.08, help="Layer grouping tolerance in Angstrom")
    parser.add_argument("--details", action="store_true", help="Also print/write layer charge details")
    parser.add_argument("--out", help="Optional Markdown output path")
    args = parser.parse_args(argv)

    cif = Path(args.cif)
    charges = _read_charges(args)
    rows = _analyze(
        cif,
        charges,
        max_index=args.max_index,
        proper_only=not args.all_rotations,
        supercell=args.supercell,
        layer_tol=args.layer_tol,
    )
    _print_text(cif, charges, rows, details=args.details)
    if args.out:
        _write_markdown(Path(args.out), cif, charges, rows, details=args.details)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
