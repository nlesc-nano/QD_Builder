#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

try:
    import yaml
except ImportError:
    raise SystemExit("Install pyyaml to read builder YAML charge files.")

from builder.heterointerface import (
    analyze_terminations,
    charge_class,
    counts_label,
    enumerate_interface_candidates,
    filter_lattice_matched_candidates,
    hkl_label,
    unique_family_terminations,
)


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
        raise SystemExit("Provide charges either via --yaml input.yaml or --charges Cd=+2 Se=-2")
    return charges


def _format_row(candidate) -> list[str]:
    core = candidate.core
    shell = candidate.shell
    lm = candidate.lattice_match
    return [
        core.family,
        hkl_label(core.hkl),
        f"{core.charge:+d}",
        charge_class(core.charge),
        core.richness,
        counts_label(core.counts),
        shell.family,
        hkl_label(shell.hkl),
        f"{shell.charge:+d}",
        charge_class(shell.charge),
        shell.richness,
        counts_label(shell.counts),
        f"{core.charge + shell.charge:+d}",
        candidate.compatibility,
        "-" if lm is None else f"{lm.area:.2f}",
        "-" if lm is None else f"{100.0 * lm.max_length_mismatch:.2f}",
        "-" if lm is None else f"{lm.angle_mismatch_deg:.2f}",
    ]


def _print_table(candidates, *, limit: int | None) -> None:
    rows = [_format_row(cand) for cand in candidates[:limit]]
    headers = [
        "core fam", "core hkl", "Qc", "sc", "core term", "core layer",
        "shell fam", "shell hkl", "Qs", "ss", "shell term", "shell layer",
        "Qsum", "match", "ZSL area", "len %", "ang deg",
    ]
    all_rows = [headers] + rows
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def _write_markdown(path: Path, candidates, *, core_cif: Path, shell_cif: Path, charges: dict[str, int]) -> None:
    headers = [
        "core family", "core hkl", "Qc", "core term", "core layer",
        "shell family", "shell hkl", "Qs", "shell term", "shell layer",
        "Qsum", "match", "ZSL area", "length mismatch %", "angle mismatch deg",
    ]
    with path.open("w") as fh:
        fh.write("# Heterointerface Candidate Scan\n\n")
        fh.write(f"- Core CIF: `{core_cif}`\n")
        fh.write(f"- Shell CIF: `{shell_cif}`\n")
        fh.write("- Charges: " + ", ".join(f"{el}={q:+d}" for el, q in sorted(charges.items())) + "\n\n")
        fh.write("| " + " | ".join(headers) + " |\n")
        fh.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for cand in candidates:
            core = cand.core
            shell = cand.shell
            lm = cand.lattice_match
            row = [
                core.family,
                f"`{hkl_label(core.hkl)}`",
                f"{core.charge:+d}",
                core.richness,
                counts_label(core.counts),
                shell.family,
                f"`{hkl_label(shell.hkl)}`",
                f"{shell.charge:+d}",
                shell.richness,
                counts_label(shell.counts),
                f"{core.charge + shell.charge:+d}",
                cand.compatibility,
                "-" if lm is None else f"{lm.area:.2f}",
                "-" if lm is None else f"{100.0 * lm.max_length_mismatch:.2f}",
                "-" if lm is None else f"{lm.angle_mismatch_deg:.2f}",
            ]
            fh.write("| " + " | ".join(row) + " |\n")


def _write_csv(path: Path, candidates) -> None:
    headers = [
        "core_family", "core_hkl", "core_charge", "core_charge_sign", "core_richness", "core_counts",
        "shell_family", "shell_hkl", "shell_charge", "shell_charge_sign", "shell_richness", "shell_counts",
        "residual_charge", "compatibility", "zsl_area", "zsl_max_length_mismatch_percent",
        "zsl_angle_mismatch_deg",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for cand in candidates:
            writer.writerow(_format_row(cand))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Experimental charge-based scan of possible Janus heterointerfaces from two CIFs."
    )
    parser.add_argument("core_cif")
    parser.add_argument("shell_cif")
    parser.add_argument("--core-name", default="core")
    parser.add_argument("--shell-name", default="shell")
    parser.add_argument("--yaml", help="Builder YAML from which to read top-level charges")
    parser.add_argument("--charges", nargs="*", help="Charges, e.g. Cs=+1 Pb=+2 Br=-1 S=-2")
    parser.add_argument("--max-index", type=int, default=1, help="Max Miller index to scan; default keeps n=1")
    parser.add_argument("--all-rotations", action="store_true", help="Include improper symmetry operations")
    parser.add_argument("--layer-tol", type=float, default=0.08, help="Layer grouping tolerance in Angstrom")
    parser.add_argument("--allow-charged-neutral", action="store_true", help="Also keep charged/neutral interface pairs")
    parser.add_argument("--signed", action="store_true", help="Keep every signed symmetry-equivalent orientation")
    parser.add_argument("--zsl", action="store_true", help="Filter charge-compatible pairs through ZSL lattice matching")
    parser.add_argument("--zsl-max-area", type=float, default=400.0, help="Maximum ZSL superlattice area in Angstrom^2")
    parser.add_argument("--zsl-max-length-tol", type=float, default=0.03, help="Maximum relative vector length mismatch")
    parser.add_argument("--zsl-max-angle-tol", type=float, default=0.01, help="Maximum relative angle tolerance used by ZSL")
    parser.add_argument("--zsl-max-area-ratio-tol", type=float, default=0.09, help="Maximum ZSL area ratio tolerance")
    parser.add_argument("--limit", type=int, default=80, help="Rows to print to terminal; use 0 for all")
    parser.add_argument("--out", help="Optional Markdown output path")
    parser.add_argument("--csv", help="Optional CSV output path")
    args = parser.parse_args(argv)

    charges = _read_charges(args)
    proper_only = not args.all_rotations
    core_cif = Path(args.core_cif)
    shell_cif = Path(args.shell_cif)

    print(f"[scan] core={core_cif} shell={shell_cif} max_index={args.max_index}")
    core_terms = analyze_terminations(
        str(core_cif),
        charges,
        material_name=args.core_name,
        max_index=args.max_index,
        proper_only=proper_only,
        layer_tol=args.layer_tol,
    )
    shell_terms = analyze_terminations(
        str(shell_cif),
        charges,
        material_name=args.shell_name,
        max_index=args.max_index,
        proper_only=proper_only,
        layer_tol=args.layer_tol,
    )
    if not args.signed:
        core_terms = unique_family_terminations(core_terms)
        shell_terms = unique_family_terminations(shell_terms)
    candidates = enumerate_interface_candidates(
        core_terms,
        shell_terms,
        allow_charged_neutral=args.allow_charged_neutral,
    )
    n_charge_candidates = len(candidates)
    if args.zsl:
        print(
            "[zsl] filtering compatible pairs "
            f"(max_area={args.zsl_max_area:g} Å^2, "
            f"length_tol={args.zsl_max_length_tol:g}, angle_tol={args.zsl_max_angle_tol:g})"
        )
        candidates = filter_lattice_matched_candidates(
            candidates,
            max_area_ratio_tol=args.zsl_max_area_ratio_tol,
            max_area=args.zsl_max_area,
            max_length_tol=args.zsl_max_length_tol,
            max_angle_tol=args.zsl_max_angle_tol,
        )
    print(
        f"[scan] terminations: core={len(core_terms)} shell={len(shell_terms)} | "
        f"compatible_pairs={n_charge_candidates}"
        + (f" | zsl_matched={len(candidates)}" if args.zsl else "")
    )

    limit = None if args.limit == 0 else args.limit
    _print_table(candidates, limit=limit)

    if args.out:
        _write_markdown(Path(args.out), candidates, core_cif=core_cif, shell_cif=shell_cif, charges=charges)
        print(f"[write] {args.out}")
    if args.csv:
        _write_csv(Path(args.csv), candidates)
        print(f"[write] {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
