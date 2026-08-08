#!/usr/bin/env python3
"""Compare full coordination signatures and global shape descriptors by bin.

The comparison is performed within each ``(k,p)`` bin.  Coordination numbers
are inferred from the final relaxed XYZ using the same Cd--Se and Cd--Cl
cutoffs used by ``analyze_cdcl2_shedding.py``.  A Cd signature is the sorted
multiset of Cd coordination numbers; for example, ``[4,2,3]`` is written in
canonical form as ``[4,3,2]``.

Example::

    python scripts/analyze_cn_global_descriptors.py \
      --analysis-root runs/cdse_map/analysis_all_dft \
      --shedding-root runs/cdse_map/analysis_all_dft/cdcl2_shedding \
      --output runs/cdse_map/analysis_all_dft/cn_global_analysis \
      --quality-status ready
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Mapping, Sequence

HARTREE_TO_KCAL_MOL = 627.5094740631
DEFAULT_CD_SE_CUTOFF = 3.25
DEFAULT_CD_CL_CUTOFF = 3.10


def f(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def read_xyz(path: Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        count_line = handle.readline()
        if not count_line:
            raise ValueError("empty XYZ")
        count = int(count_line.strip())
        if not handle.readline():
            raise ValueError("missing XYZ comment")
        symbols: list[str] = []
        coordinates: list[tuple[float, float, float]] = []
        for _ in range(count):
            fields = handle.readline().split()
            if len(fields) < 4:
                raise ValueError("truncated XYZ")
            symbols.append(fields[0])
            coordinates.append(tuple(float(x.replace("D", "E").replace("d", "e")) for x in fields[1:4]))
    return symbols, coordinates


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def coordination(
    symbols: Sequence[str], coordinates: Sequence[Sequence[float]],
    cd_se_cutoff: float, cd_cl_cutoff: float,
) -> tuple[list[int], list[tuple[int, int, str]]]:
    degrees = [0] * len(symbols)
    edges: list[tuple[int, int, str]] = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            pair = {symbols[i], symbols[j]}
            if pair == {"Cd", "Se"}:
                cutoff, kind = cd_se_cutoff, "Cd-Se"
            elif pair == {"Cd", "Cl"}:
                cutoff, kind = cd_cl_cutoff, "Cd-Cl"
            else:
                continue
            if distance(coordinates[i], coordinates[j]) <= cutoff:
                degrees[i] += 1
                degrees[j] += 1
                edges.append((i, j, kind))
    return degrees, edges


def signature(values: Sequence[int]) -> str:
    return "[" + ",".join(str(v) for v in sorted(values, reverse=True)) + "]"


def multiset_triplets(values: Sequence[int]) -> tuple[int, int]:
    """Return counts of Cd CN triplets [3,3,3] and [4,3,2].

    The order of atoms is irrelevant.  The second count is therefore the
    number of possible choices of one CN=4, one CN=3 and one CN=2 Cd atom.
    """

    counts = Counter(values)
    n333 = counts.get(3, 0) * (counts.get(3, 0) - 1) * (counts.get(3, 0) - 2) // 6
    n423 = counts.get(4, 0) * counts.get(3, 0) * counts.get(2, 0)
    return n333, n423


def prepare(
    analysis_root: Path, shedding_root: Path, statuses: set[str] | None,
    cd_se_cutoff: float, cd_cl_cutoff: float,
) -> list[dict[str, Any]]:
    originals = read_csv(analysis_root / "structures.csv")
    summary = {row["structure_id"]: row for row in read_csv(shedding_root / "cdcl2_structure_summary.csv")}
    rows: list[dict[str, Any]] = []
    for original in originals:
        sid = original.get("structure_id", "")
        srow = summary.get(sid)
        if srow is None:
            continue
        quality = srow.get("quality_status", original.get("quality_status", ""))
        if statuses and quality not in statuses:
            continue
        energy = f(srow.get("energy_hartree")) or f(original.get("energy_hartree"))
        if energy is None:
            continue
        xyz_value = srow.get("relaxed_xyz") or original.get("relaxed_xyz", "")
        xyz = Path(xyz_value)
        if not xyz.is_absolute():
            xyz = analysis_root / xyz
        try:
            symbols, coordinates = read_xyz(xyz)
            degrees, edges = coordination(symbols, coordinates, cd_se_cutoff, cd_cl_cutoff)
        except (OSError, ValueError, IndexError):
            continue
        cd = [degrees[i] for i, symbol in enumerate(symbols) if symbol == "Cd"]
        se = [degrees[i] for i, symbol in enumerate(symbols) if symbol == "Se"]
        cl = [degrees[i] for i, symbol in enumerate(symbols) if symbol == "Cl"]
        cd_se = sum(kind == "Cd-Se" for _, _, kind in edges)
        cd_cl = sum(kind == "Cd-Cl" for _, _, kind in edges)
        cd_se_degrees = [0] * len(symbols)
        cd_cl_degrees = [0] * len(symbols)
        for left, right, kind in edges:
            target = cd_se_degrees if kind == "Cd-Se" else cd_cl_degrees
            target[left] += 1
            target[right] += 1
        cd_se_cn = [cd_se_degrees[i] for i, symbol in enumerate(symbols) if symbol == "Cd"]
        cd_cl_cn = [cd_cl_degrees[i] for i, symbol in enumerate(symbols) if symbol == "Cd"]
        se_cn = [cd_se_degrees[i] for i, symbol in enumerate(symbols) if symbol == "Se"]
        cd_triplet_333_count, cd_triplet_423_count = multiset_triplets(cd)
        row: dict[str, Any] = dict(original)
        row.update({
            "structure_id": sid, "k": int(srow.get("k", original.get("k", 0))),
            "p": int(srow.get("p", original.get("p", 0))),
            "quality_status": quality, "energy_hartree": energy,
            "relaxed_xyz_used": str(xyz),
            "cd_cn_signature": signature(cd), "se_cn_signature": signature(se),
            "cd_se_cn_signature": signature(cd_se_cn),
            "cd_cl_cn_signature": signature(cd_cl_cn),
            "cd_cn_mean": mean(cd) if cd else None,
            "cd_cn_std": pstdev(cd) if len(cd) > 1 else 0.0,
            "cd_cn_min": min(cd) if cd else None, "cd_cn_max": max(cd) if cd else None,
            "se_cn_mean": mean(se) if se else None,
            "se_cn_std": pstdev(se) if len(se) > 1 else 0.0,
            "se_cn_min": min(se) if se else None, "se_cn_max": max(se) if se else None,
            "cl_cn_mean": mean(cl) if cl else None,
            "skeleton_bonds_recomputed": cd_se,
            "ligand_bonds_recomputed": cd_cl,
            "total_bonds_recomputed": len(edges),
            "cd_se_bonds_recomputed": cd_se, "cd_cl_bonds_recomputed": cd_cl,
            "cd_se_cn_mean": mean(cd_se_cn) if cd_se_cn else None,
            "cd_se_cn_std": pstdev(cd_se_cn) if len(cd_se_cn) > 1 else 0.0,
            "cd_se_cn_min": min(cd_se_cn) if cd_se_cn else None,
            "cd_se_cn_max": max(cd_se_cn) if cd_se_cn else None,
            "cd_cl_cn_mean": mean(cd_cl_cn) if cd_cl_cn else None,
            "cd_cl_cn_std": pstdev(cd_cl_cn) if len(cd_cl_cn) > 1 else 0.0,
            "se_cn_host_mean": mean(se_cn) if se_cn else None,
            "se_cn_host_std": pstdev(se_cn) if len(se_cn) > 1 else 0.0,
            "se_cn_host_min": min(se_cn) if se_cn else None,
            "se_cn_host_max": max(se_cn) if se_cn else None,
            "cd_cn1_count": cd.count(1), "cd_cn2_count": cd.count(2),
            "cd_cn3_count": cd.count(3), "cd_cn4_count": cd.count(4),
            "cd_cn5plus_count": sum(v >= 5 for v in cd),
            "cd_triplet_333_count": cd_triplet_333_count,
            "cd_triplet_423_count": cd_triplet_423_count,
            "cd_has_triplet_333": int(cd_triplet_333_count > 0),
            "cd_has_triplet_423": int(cd_triplet_423_count > 0),
            "se_cn1_count": se.count(1), "se_cn2_count": se.count(2),
            "se_cn3_count": se.count(3), "se_cn4_count": se.count(4),
            "se_cn5plus_count": sum(v >= 5 for v in se),
        })
        rows.append(row)
    bins: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bins[(row["k"], row["p"])].append(row)
    ranked: list[dict[str, Any]] = []
    for records in bins.values():
        minimum = min(row["energy_hartree"] for row in records)
        for row in records:
            gap = (row["energy_hartree"] - minimum) * HARTREE_TO_KCAL_MOL
            if gap <= 3:
                group = "winner_window"
            elif gap >= 30:
                group = "extreme_unstable"
            elif gap >= 20:
                group = "high_gap"
            else:
                group = "intermediate"
            ranked.append({**row, "bin_min_energy_hartree": minimum,
                           "relative_energy_kcal_mol": gap,
                           "stability_group": group, "bin_size": len(records)})
    return sorted(ranked, key=lambda row: (row["k"], row["p"], row["relative_energy_kcal_mol"]))


def numeric_descriptors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names = [
        "skeleton_bonds_recomputed", "ligand_bonds_recomputed",
        "total_bonds_recomputed", "cd_se_bonds_recomputed", "cd_cl_bonds_recomputed",
        "cd_se_cn_mean", "cd_se_cn_std", "cd_se_cn_min", "cd_se_cn_max",
        "cd_cl_cn_mean", "cd_cl_cn_std", "se_cn_host_mean", "se_cn_host_std",
        "se_cn_host_min", "se_cn_host_max",
        "cd_cn_mean", "cd_cn_std", "cd_cn_min", "cd_cn_max", "se_cn_mean", "se_cn_std",
        "se_cn_min", "se_cn_max", "final_total_bonds", "final_radius_of_gyration",
        "final_max_span", "final_max_pair_distance", "final_mean_CN_Cd", "final_mean_CN_Se",
        "final_min_CN_Cd", "final_min_CN_Se", "final_CN_deficit_Cd", "final_CN_deficit_Se",
        "cd_triplet_333_count", "cd_triplet_423_count", "cd_has_triplet_333", "cd_has_triplet_423",
    ]
    return [n for n in names if any(f(row.get(n)) is not None for row in rows)]


def comparisons(rows: Sequence[Mapping[str, Any]], names: Sequence[str], unstable: str) -> list[dict[str, Any]]:
    bins: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        bins[(int(row["k"]), int(row["p"]))].append(row)
    out: list[dict[str, Any]] = []
    for name in names:
        deltas: list[float] = []; stable_values: list[float] = []; unstable_values: list[float] = []; matched = 0
        for records in bins.values():
            stable = [f(r.get(name)) for r in records if r["stability_group"] == "winner_window"]
            bad = [f(r.get(name)) for r in records if r["stability_group"] == unstable]
            stable = [v for v in stable if v is not None]; bad = [v for v in bad if v is not None]
            if not stable or not bad: continue
            matched += 1; stable_values.extend(stable); unstable_values.extend(bad)
            deltas.append(mean(bad) - mean(stable))
        if deltas:
            out.append({"descriptor": name, "matched_bins": matched,
                        "mean_within_bin_delta": mean(deltas),
                        "median_within_bin_delta": median(deltas),
                        "mean_unstable_minus_winner": mean(deltas),
                        "pooled_winner_mean": mean(stable_values),
                        "pooled_unstable_mean": mean(unstable_values),
                        "pooled_difference": mean(unstable_values) - mean(stable_values)})
    return sorted(out, key=lambda r: abs(r["pooled_difference"]), reverse=True)


def signature_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = ["winner_window", "high_gap", "extreme_unstable"]
    signatures = sorted({r["cd_cn_signature"] for r in rows} | {r["se_cn_signature"] for r in rows})
    out: list[dict[str, Any]] = []
    for kind, field in (("Cd", "cd_cn_signature"), ("Se", "se_cn_signature")):
        values = sorted({r[field] for r in rows})
        for sig in values:
            for group in groups:
                selected = [r for r in rows if r["stability_group"] == group]
                count = sum(r[field] == sig for r in selected)
                out.append({"species": kind, "signature": sig, "group": group,
                            "count": count, "fraction": count / len(selected) if selected else None})
    return out


def bin_bond_cn_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize skeleton/ligand bond counts and host CN distributions per bin."""

    bins: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        bins[(int(row["k"]), int(row["p"]))].append(row)
    output: list[dict[str, Any]] = []
    for (k, p), records in sorted(bins.items()):
        for group in ("winner_window", "loser"):
            selected = (
                [r for r in records if r["stability_group"] == "winner_window"]
                if group == "winner_window"
                else [r for r in records if r["stability_group"] != "winner_window"]
            )
            if not selected:
                continue
            row: dict[str, Any] = {"k": k, "p": p, "group": group, "count": len(selected)}
            for field in (
                "skeleton_bonds_recomputed", "ligand_bonds_recomputed", "total_bonds_recomputed",
                "cd_se_cn_mean", "cd_se_cn_std", "cd_se_cn_min", "cd_se_cn_max",
                "cd_cl_cn_mean", "cd_cl_cn_std",
                "se_cn_host_mean", "se_cn_host_std", "se_cn_host_min", "se_cn_host_max",
            ):
                values = [f(r.get(field)) for r in selected]
                values = [v for v in values if v is not None]
                row[f"{field}_mean"] = mean(values) if values else None
                row[f"{field}_median"] = median(values) if values else None
                row[f"{field}_min"] = min(values) if values else None
                row[f"{field}_max"] = max(values) if values else None
            # Within this group, equal bond counts can still carry different
            # host CN distributions.  This is the distinction of interest for
            # cases such as [3,3,3] versus [4,3,2].
            for bond_field, label in (("skeleton_bonds_recomputed", "skeleton"), ("total_bonds_recomputed", "total")):
                distributions: dict[Any, set[tuple[str, str]]] = defaultdict(set)
                for r in selected:
                    distributions[r.get(bond_field)].add((str(r.get("cd_se_cn_signature")), str(r.get("se_cn_signature"))))
                row[f"{label}_bond_values"] = ";".join(str(v) for v in sorted(distributions, key=str))
                row[f"{label}_bond_values_with_multiple_CN_distributions"] = sum(len(values) > 1 for values in distributions.values())
            output.append(row)
    return output


def equal_bond_comparisons(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pair winners and losers with identical skeleton/ligand bond counts."""

    bins: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        bins[(int(row["k"]), int(row["p"]))].append(row)
    output: list[dict[str, Any]] = []
    for (k, p), records in sorted(bins.items()):
        winners = [r for r in records if r["stability_group"] == "winner_window"]
        losers = [r for r in records if r["stability_group"] != "winner_window"]
        if not winners or not losers:
            continue
        pairs = []
        for winner in winners:
            for loser in losers:
                if (winner.get("skeleton_bonds_recomputed"), winner.get("ligand_bonds_recomputed")) == (loser.get("skeleton_bonds_recomputed"), loser.get("ligand_bonds_recomputed")):
                    pairs.append((winner, loser))
        if not pairs:
            continue
        different_cd = sum(w.get("cd_se_cn_signature") != l.get("cd_se_cn_signature") for w, l in pairs)
        different_se = sum(w.get("se_cn_signature") != l.get("se_cn_signature") for w, l in pairs)
        output.append({
            "k": k, "p": p, "equal_bond_pairs": len(pairs),
            "different_Cd_host_CN_pairs": different_cd,
            "different_Se_CN_pairs": different_se,
            "fraction_different_Cd_host_CN": different_cd / len(pairs),
            "fraction_different_Se_CN": different_se / len(pairs),
            "mean_energy_gap_loser_minus_winner_kcal": mean(
                (l["energy_hartree"] - w["energy_hartree"]) * HARTREE_TO_KCAL_MOL for w, l in pairs
            ),
            "bond_count_examples": ";".join(
                f"skel={w['skeleton_bonds_recomputed']},lig={w['ligand_bonds_recomputed']}"
                for w, _ in pairs[:10]
            ),
        })
    return output


def fixed_p_cn_trends(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Winner host-CN trend versus k for each fixed ligand count p."""

    winners = [r for r in rows if r["stability_group"] == "winner_window"]
    groups: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in winners:
        groups[(int(row["p"]), int(row["k"]))].append(row)
    output: list[dict[str, Any]] = []
    for (p, k), selected in sorted(groups.items()):
        def vals(field: str) -> list[float]:
            return [v for r in selected if (v := f(r.get(field))) is not None]
        for field, label in (("cd_se_cn_mean", "winner_mean_Cd_Se_CN"),
                             ("cd_cl_cn_mean", "winner_mean_Cd_Cl_CN"),
                             ("se_cn_host_mean", "winner_mean_Se_CN")):
            values = vals(field)
            if not values:
                continue
            output.append({"p": p, "k": k, "descriptor": label, "winner_count": len(values),
                           "mean": mean(values), "median": median(values), "min": min(values), "max": max(values)})
    return output


def within_winner_equal_bonds(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Find winner/winner pairs with equal bonds but different CN allocation."""

    bins: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["stability_group"] == "winner_window":
            bins[(int(row["k"]), int(row["p"]))].append(row)
    output: list[dict[str, Any]] = []
    for (k, p), winners in sorted(bins.items()):
        for mode, fields in (
            ("total", ("total_bonds_recomputed",)),
            ("skeleton_and_ligand", ("skeleton_bonds_recomputed", "ligand_bonds_recomputed")),
        ):
            pairs = []
            for left_index, left in enumerate(winners):
                for right in winners[left_index + 1:]:
                    if all(left.get(field) == right.get(field) for field in fields):
                        different = (left.get("cd_se_cn_signature"), left.get("se_cn_signature")) != (right.get("cd_se_cn_signature"), right.get("se_cn_signature"))
                        if different:
                            pairs.append((left, right))
            if pairs:
                output.append({
                    "k": k, "p": p, "mode": mode, "winner_pairs": len(pairs),
                    "mean_absolute_energy_difference_kcal": mean(
                        abs(left["energy_hartree"] - right["energy_hartree"]) * HARTREE_TO_KCAL_MOL
                        for left, right in pairs
                    ),
                    "max_absolute_energy_difference_kcal": max(
                        abs(left["energy_hartree"] - right["energy_hartree"]) * HARTREE_TO_KCAL_MOL
                        for left, right in pairs
                    ),
                })
    return output


def write_report(path: Path, rows: Sequence[Mapping[str, Any]], comp: Sequence[Mapping[str, Any]], sigs: Sequence[Mapping[str, Any]], equal_pairs: Sequence[Mapping[str, Any]]) -> None:
    groups = Counter(r["stability_group"] for r in rows)
    lines = ["# Coordination-number and global-descriptor analysis", "",
             f"Structures: **{len(rows)}**; groups: `{dict(groups)}`", "",
             "All energies and comparisons are evaluated within the same `(k,p)` bin. "
             "Cd/Se signatures are sorted descending, so `[4,2,3]` is represented as `[4,3,2]`.", "",
             "## Descriptor comparison", "",
             "Positive values mean the unstable group has the larger value than the winner window.", "",
             "| descriptor | matched bins | winner mean | unstable mean | unstable - winner |",
             "|---|---:|---:|---:|---:|"]
    for r in comp[:30]:
        lines.append(f"| {r['descriptor']} | {r['matched_bins']} | {r['pooled_winner_mean']:.4g} | {r['pooled_unstable_mean']:.4g} | {r['mean_within_bin_delta']:+.4g} |")
    lines += ["", "## Coordination signatures", "",
              "Fractions are global group fractions; inspect the CSV for per-signature counts.", "",
              "| species | signature | winner | high-gap | extreme |", "|---|---|---:|---:|---:|"]
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in sigs:
        if r["fraction"] is not None: grouped[(r["species"], r["signature"])][r["group"]] = r["fraction"]
    # Avoid flooding the human-readable report with one-off signatures.  The
    # complete table remains available in cn_signature_commonality.csv.
    for (species, sig), vals in sorted(grouped.items()):
        if max(vals.get("winner_window", 0), vals.get("high_gap", 0), vals.get("extreme_unstable", 0)) < 0.10:
            continue
        lines.append(f"| {species} | `{sig}` | {vals.get('winner_window', 0):.3f} | {vals.get('high_gap', 0):.3f} | {vals.get('extreme_unstable', 0):.3f} |")
    lines += ["", "## Requested Cd CN triplets", "",
              "A triplet is present when the structure contains at least three Cd atoms with the indicated coordination numbers; atom order is ignored.", "",
              "| Cd triplet | winner fraction | high-gap fraction | extreme fraction |", "|---|---:|---:|---:|"]
    for label, field in (("[3,3,3]", "cd_has_triplet_333"), ("[4,2,3]", "cd_has_triplet_423")):
        fractions = []
        for group in ("winner_window", "high_gap", "extreme_unstable"):
            selected = [r for r in rows if r["stability_group"] == group]
            fractions.append(sum(int(r.get(field, 0)) > 0 for r in selected) / len(selected) if selected else 0.0)
        lines.append(f"| `{label}` | {fractions[0]:.3f} | {fractions[1]:.3f} | {fractions[2]:.3f} |")
    lines += ["", "## Equal bond counts, different CN distributions", "",
              "These are winner/loser pairs with identical numbers of skeleton Cd--Se bonds and ligand Cd--Cl bonds. The coordination distributions can nevertheless differ.", "",
              "| k | p | equal-bond pairs | different Cd host-CN | different Se CN | mean loser-winner gap (kcal/mol) |", "|---:|---:|---:|---:|---:|---:|"]
    for r in equal_pairs:
        lines.append(f"| {r['k']} | {r['p']} | {r['equal_bond_pairs']} | {r['different_Cd_host_CN_pairs']} | {r['different_Se_CN_pairs']} | {r['mean_energy_gap_loser_minus_winner_kcal']:.3g} |")
    lines += ["", "## Interpretation", "",
              "Use exact signatures as retention/ranking evidence, not as universal hard constraints. "
              "Total bond count and radius of gyration must be compared within `(k,p)` because both scale with composition and size.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--shedding-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-status", action="append", default=None)
    parser.add_argument("--cd-se-cutoff", type=float, default=DEFAULT_CD_SE_CUTOFF)
    parser.add_argument("--cd-cl-cutoff", type=float, default=DEFAULT_CD_CL_CUTOFF)
    args = parser.parse_args()
    statuses = set(args.quality_status) if args.quality_status else None
    rows = prepare(args.analysis_root.expanduser().resolve(), args.shedding_root.expanduser().resolve(), statuses, args.cd_se_cutoff, args.cd_cl_cutoff)
    if not rows: parser.error("no structures with readable relaxed XYZ and energy")
    comp = comparisons(rows, numeric_descriptors(rows), "high_gap")
    extreme = comparisons(rows, numeric_descriptors(rows), "extreme_unstable")
    sigs = signature_rows(rows)
    bin_summary = bin_bond_cn_summary(rows)
    equal_pairs = equal_bond_comparisons(rows)
    trends = fixed_p_cn_trends(rows)
    winner_equal = within_winner_equal_bonds(rows)
    out = args.output.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "cn_global_ranked_structures.csv", rows)
    write_csv(out / "cn_global_comparisons_high_gap.csv", comp)
    write_csv(out / "cn_global_comparisons_extreme.csv", extreme)
    write_csv(out / "cn_signature_commonality.csv", sigs)
    write_csv(out / "cn_bin_bond_cn_summary.csv", bin_summary)
    write_csv(out / "cn_equal_bond_cn_comparisons.csv", equal_pairs)
    write_csv(out / "cn_fixed_p_winner_trends.csv", trends)
    write_csv(out / "cn_winner_equal_bond_distributions.csv", winner_equal)
    write_report(out / "cn_global_analysis_report.md", rows, extreme, sigs, equal_pairs)
    print(f"Included structures: {len(rows)}")
    print(f"Groups: {dict(Counter(r['stability_group'] for r in rows))}")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
