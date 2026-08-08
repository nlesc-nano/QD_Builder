#!/usr/bin/env python3
"""Select a diverse, deduplicated CdSe/CdCl2 DFT calibration set.

The script merges one or more completed nucleation bundles, removes structures
that are equivalent as DFT inputs, samples retained structures and informative
soft-rejected controls, and creates CP2K-ready calculation directories.

Deduplication is invariant to atom ordering, translation, rotation, reflection,
and bundle-specific structure IDs.  It uses element-resolved interatomic
distance multisets within each ``(k, p)`` composition.  This is the appropriate
equivalence for isolated, field-free DFT calculations: structures with the same
elements and all the same pair distances have the same starting geometry up to
a rigid transformation or mirror operation.

Rejected controls are soft by default: they must satisfy the minimum-CN rules
and have been discarded only for a lower coordination rank or lower Cl-ring
rank.  Hard min-CN failures can be included explicitly, but are normally a poor
use of the DFT budget.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Iterable, Mapping, Optional, Sequence


SOFT_REJECT_REASONS = {"lower_coordination_rank", "lower_cl_ring_rank"}
EXPECTED_CHARGES = {"Cd": 2, "Se": -2, "Cl": -1}


@dataclass(frozen=True)
class XYZ:
    symbols: tuple[str, ...]
    coordinates: tuple[tuple[float, float, float], ...]


@dataclass
class Candidate:
    bundle_label: str
    bundle_root: Path
    status: str
    record: dict[str, object]
    xyz_path: Optional[Path]
    xyz_kind: str
    xyz: XYZ
    geometry_key: tuple[object, ...]
    fingerprint: str

    @property
    def k(self) -> int:
        return int(self.record["k"])

    @property
    def p(self) -> int:
        return int(self.record["p"])

    @property
    def structure_id(self) -> str:
        fallback = self.xyz_path.stem if self.xyz_path is not None else "candidate"
        return str(self.record.get("structure_id", fallback))

    @property
    def source_reference(self) -> str:
        if self.xyz_path is not None:
            return str(self.xyz_path)
        return f"{self.bundle_root / 'registry.json'}#{self.structure_id}"

    @property
    def metadata(self) -> Mapping[str, object]:
        raw = self.record.get("metadata", {})
        return raw if isinstance(raw, Mapping) else {}

    @property
    def selection(self) -> Mapping[str, object]:
        raw = self.record.get("selection", {})
        return raw if isinstance(raw, Mapping) else {}

    @property
    def score(self) -> tuple[int, ...]:
        raw = self.selection.get("coordination_score", ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return ()
        return tuple(int(value) for value in raw)


@dataclass
class MergedCandidate:
    k: int
    p: int
    geometry_key: tuple[object, ...]
    fingerprint: str
    aliases: list[Candidate] = field(default_factory=list)
    score_layer_rank: int = 0

    @property
    def representative(self) -> Candidate:
        """Prefer a retained surface XYZ, then the best deterministic alias."""

        return sorted(
            self.aliases,
            key=lambda item: (
                item.status == "retained",
                item.xyz_kind == "surface",
                item.score,
                item.bundle_label,
                item.structure_id,
                item.source_reference,
            ),
            reverse=True,
        )[0]

    @property
    def selection_class(self) -> str:
        if any(item.status == "retained" for item in self.aliases):
            return "retained"
        return "soft_rejected"

    @property
    def source_labels(self) -> tuple[str, ...]:
        return tuple(sorted({item.bundle_label for item in self.aliases}))

    @property
    def source_statuses(self) -> tuple[str, ...]:
        return tuple(sorted({item.status for item in self.aliases}))

    @property
    def family_ids(self) -> tuple[str, ...]:
        values = {
            str(
                item.record.get("skeleton_family_id")
                or item.metadata.get("skeleton_family_id")
                or ""
            )
            for item in self.aliases
        }
        return tuple(sorted(value for value in values if value))

    @property
    def ligand_hashes(self) -> tuple[str, ...]:
        values = {
            str(
                item.record.get("ligand_shell_hash")
                or item.metadata.get("ligand_shell_hash")
                or ""
            )
            for item in self.aliases
        }
        return tuple(sorted(value for value in values if value))

    @property
    def reason(self) -> str:
        reasons = {
            str(item.selection.get("reason", "")) for item in self.aliases
        }
        reasons.discard("")
        return ";".join(sorted(reasons))

    @property
    def best_score(self) -> tuple[int, ...]:
        return max((item.score for item in self.aliases), default=())

    def numeric_descriptors(self) -> tuple[float, ...]:
        item = self.representative
        meta = item.metadata
        rings_raw = meta.get("rings", {})
        rings = rings_raw if isinstance(rings_raw, Mapping) else {}
        return (
            float(meta.get("bridge_count", 0) or 0),
            float(meta.get("bond_count", 0) or 0),
            float(meta.get("total_cn", 0) or 0),
            float(meta.get("min_cn_total_shortfall", 0) or 0),
            float(meta.get("geometry_residual", 0.0) or 0.0),
            float(rings.get("cl_ring_total", 0) or 0),
            float(rings.get("inorganic_six_rings", 0) or 0),
            float(self.score_layer_rank),
        )


def read_xyz(path: Path) -> XYZ:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"empty or malformed XYZ: {path}")
    count = int(lines[0].strip())
    atom_lines = lines[2 : 2 + count]
    if len(atom_lines) != count:
        raise ValueError(f"XYZ atom count mismatch: {path}")
    symbols: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
    for line in atom_lines:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"malformed XYZ atom line in {path}: {line!r}")
        symbols.append(fields[0])
        coordinates.append(tuple(float(value) for value in fields[1:4]))
    return XYZ(tuple(symbols), tuple(coordinates))


def xyz_from_record(record: Mapping[str, object]) -> tuple[XYZ, str]:
    symbols_raw = record.get("symbols", ())
    if not isinstance(symbols_raw, Sequence) or isinstance(symbols_raw, (str, bytes)):
        raise ValueError("record has no symbol sequence")
    symbols = tuple(str(value) for value in symbols_raw)
    surface_raw = record.get("surface_coordinates", ())
    native_raw = record.get("coordinates", ())
    use_surface = (
        isinstance(surface_raw, Sequence)
        and not isinstance(surface_raw, (str, bytes))
        and len(surface_raw) == len(symbols)
        and len(symbols) > 0
    )
    coordinates_raw = surface_raw if use_surface else native_raw
    if not isinstance(coordinates_raw, Sequence) or isinstance(
        coordinates_raw, (str, bytes)
    ):
        raise ValueError("record has no coordinate sequence")
    coordinates: list[tuple[float, float, float]] = []
    for row in coordinates_raw:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) < 3:
            raise ValueError("record contains malformed coordinates")
        coordinates.append(tuple(float(value) for value in row[:3]))
    if len(coordinates) != len(symbols):
        raise ValueError("record symbol/coordinate count mismatch")
    kind = "registry_surface" if use_surface else "registry_construction_native"
    return XYZ(symbols, tuple(coordinates)), kind


def write_xyz(path: Path, xyz: XYZ, comment: str) -> None:
    lines = [str(len(xyz.symbols)), comment]
    for symbol, coordinates in zip(xyz.symbols, xyz.coordinates):
        lines.append(
            f"{symbol:<2s} {coordinates[0]: .12f} {coordinates[1]: .12f} "
            f"{coordinates[2]: .12f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def geometry_key(xyz: XYZ, tolerance: float) -> tuple[object, ...]:
    """Return an element- and rigid-motion-invariant geometry signature."""

    if tolerance <= 0:
        raise ValueError("deduplication tolerance must be positive")
    counts = tuple(sorted(Counter(xyz.symbols).items()))
    distances: dict[tuple[str, str], list[int]] = defaultdict(list)
    for left in range(len(xyz.symbols)):
        for right in range(left + 1, len(xyz.symbols)):
            pair = tuple(sorted((xyz.symbols[left], xyz.symbols[right])))
            distance = math.dist(xyz.coordinates[left], xyz.coordinates[right])
            distances[pair].append(int(round(distance / tolerance)))
    signature = tuple(
        (pair, tuple(sorted(values))) for pair, values in sorted(distances.items())
    )
    return counts, signature


def fingerprint_for(key: tuple[object, ...]) -> str:
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()


def expected_formula(k: int, p: int) -> Counter[str]:
    formula = Counter({"Cd": k + p, "Se": k})
    if p > 0:
        formula["Cl"] = 2 * p
    return formula


def validate_neutral_formula(candidate: Candidate) -> None:
    actual = Counter(candidate.xyz.symbols)
    expected = expected_formula(candidate.k, candidate.p)
    if actual != expected:
        raise ValueError(
            f"nonmatching formula for {candidate.xyz_path}: got {dict(actual)}, "
            f"expected {dict(expected)} for k={candidate.k}, p={candidate.p}"
        )
    charge = sum(EXPECTED_CHARGES[symbol] * count for symbol, count in actual.items())
    record_charge = int(candidate.metadata.get("formal_charge", charge) or 0)
    if charge != 0 or record_charge != 0:
        raise ValueError(
            f"non-neutral candidate {candidate.xyz_path}: formula charge={charge}, "
            f"record formal_charge={record_charge}"
        )


def unique_labels(paths: Sequence[Path]) -> list[str]:
    base = [path.resolve().name for path in paths]
    totals = Counter(base)
    seen: Counter[str] = Counter()
    labels: list[str] = []
    for name in base:
        seen[name] += 1
        labels.append(name if totals[name] == 1 else f"{name}_{seen[name]}")
    return labels


def iter_records(
    registry: Mapping[str, object], key: str
) -> Iterable[dict[str, object]]:
    rows = registry.get(key, {})
    if not isinstance(rows, Mapping):
        return
    for k_bins in rows.values():
        if not isinstance(k_bins, Mapping):
            continue
        for records in k_bins.values():
            if not isinstance(records, list):
                continue
            for record in records:
                if isinstance(record, dict):
                    yield record


def record_xyz_path(
    root: Path, record: Mapping[str, object], status: str
) -> tuple[Path, str]:
    meta_raw = record.get("metadata", {})
    meta = meta_raw if isinstance(meta_raw, Mapping) else {}
    if status == "retained" and meta.get("surface_xyz_path"):
        return root / str(meta["surface_xyz_path"]), "surface"
    if meta.get("construction_native_xyz_path"):
        return root / str(meta["construction_native_xyz_path"]), "construction_native"
    structure_id = str(record.get("structure_id", ""))
    k, p = int(record["k"]), int(record["p"])
    suffix = "surface" if status == "retained" else "construction_native"
    path = (
        root
        / "structures"
        / f"k{k:03d}"
        / f"p{p:03d}"
        / status
        / f"{structure_id}_{suffix}.xyz"
    )
    return path, suffix


def load_bundle(
    root: Path,
    label: str,
    *,
    tolerance: float,
    k_min: int,
    k_max: int,
    require_kmax: int,
    include_hard_rejected: bool,
    allow_incomplete: bool,
) -> tuple[list[Candidate], Counter[str]]:
    registry_path = root / "registry.json"
    registry: Optional[dict[str, object]] = None
    if registry_path.is_file():
        loaded = json.loads(registry_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            registry = loaded

    # During a long k=6 run, registry.json is written only after the complete
    # map finishes.  A DONE checkpoint contains the same record dictionaries
    # for every finished k-row, so it is safe to use those immutable rows while
    # ignoring the active/incomplete row.
    checkpoint_registry: Optional[dict[str, object]] = None
    loaded_from_checkpoints = False
    if allow_incomplete:
        checkpoint_rows: dict[str, dict[str, list[dict[str, object]]]] = {}
        checkpoint_discarded: dict[str, dict[str, list[dict[str, object]]]] = {}
        checkpoint_root = root / "checkpoint"
        if checkpoint_root.is_dir():
            for done in sorted(checkpoint_root.glob("k*/DONE")):
                k_dir = done.parent
                try:
                    k_value = int(k_dir.name[1:])
                except ValueError:
                    continue
                retained_path = k_dir / "retained.json"
                skeleton_path = k_dir / "skeletons.json"
                if not retained_path.is_file() or not skeleton_path.is_file():
                    continue
                retained = json.loads(retained_path.read_text(encoding="utf-8"))
                discarded_path = k_dir / "discarded.json"
                discarded = (
                    json.loads(discarded_path.read_text(encoding="utf-8"))
                    if discarded_path.is_file()
                    else {}
                )
                if not isinstance(retained, dict) or not isinstance(discarded, dict):
                    continue
                checkpoint_rows[str(k_value)] = retained
                checkpoint_discarded[str(k_value)] = discarded
        if checkpoint_rows:
            checkpoint_registry = {
                "registry": checkpoint_rows,
                "discarded_registry": checkpoint_discarded,
            }
            if registry is None:
                registry = checkpoint_registry
                loaded_from_checkpoints = True
            else:
                final_k = max(
                    (int(value) for value in registry.get("registry", {}) or {}),
                    default=0,
                )
                checkpoint_k = max((int(value) for value in checkpoint_rows), default=0)
                if checkpoint_k > final_k:
                    registry = checkpoint_registry

    if registry is None:
        raise FileNotFoundError(
            f"completed bundle registry not found: {registry_path}; "
            "use --allow-incomplete-source for DONE checkpoints"
        )
    rows_raw = registry.get("registry", {})
    rows = rows_raw if isinstance(rows_raw, Mapping) else {}
    reached = max((int(value) for value in rows), default=0)
    if require_kmax > 0 and reached < require_kmax:
        raise RuntimeError(
            f"bundle {root} reached only k={reached}; required k={require_kmax}"
        )
    counts: Counter[str] = Counter()
    candidates: list[Candidate] = []
    for status, registry_key in (
        ("retained", "registry"),
        ("discarded", "discarded_registry"),
    ):
        for record in iter_records(registry, registry_key):
            k, p = int(record["k"]), int(record["p"])
            if not (k_min <= k <= k_max):
                counts[f"{status}_outside_k_range"] += 1
                continue
            selection_raw = record.get("selection", {})
            selection = selection_raw if isinstance(selection_raw, Mapping) else {}
            meta_raw = record.get("metadata", {})
            meta = meta_raw if isinstance(meta_raw, Mapping) else {}
            if status == "discarded" and not include_hard_rejected:
                reason = str(selection.get("reason", ""))
                compliant = bool(meta.get("min_cn_compliant", False))
                if not compliant or reason not in SOFT_REJECT_REASONS:
                    counts["discarded_hard_or_uninformative"] += 1
                    continue
            xyz_path, xyz_kind = record_xyz_path(root, record, status)
            if xyz_path.is_file():
                xyz = read_xyz(xyz_path)
                resolved_xyz_path: Optional[Path] = xyz_path.resolve()
            else:
                xyz, xyz_kind = xyz_from_record(record)
                resolved_xyz_path = None
                counts[f"{status}_xyz_reconstructed_from_registry"] += 1
            key = geometry_key(xyz, tolerance)
            candidate = Candidate(
                bundle_label=label,
                bundle_root=root,
                status=status,
                record=record,
                xyz_path=resolved_xyz_path,
                xyz_kind=xyz_kind,
                xyz=xyz,
                geometry_key=key,
                fingerprint=fingerprint_for(key),
            )
            validate_neutral_formula(candidate)
            candidates.append(candidate)
            counts[f"{status}_loaded"] += 1
    if loaded_from_checkpoints:
        counts["loaded_from_done_checkpoints"] += 1
    return candidates, counts


def load_excluded_geometries(
    manifests: Sequence[Path], tolerance: float
) -> tuple[set[tuple[int, int, tuple[object, ...]]], Counter[str]]:
    excluded: set[tuple[int, int, tuple[object, ...]]] = set()
    counts: Counter[str] = Counter()
    for manifest in manifests:
        manifest = manifest.resolve()
        if not manifest.is_file():
            raise FileNotFoundError(f"exclusion manifest not found: {manifest}")
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                k, p = int(row["k"]), int(row["p"])
                start = manifest.parent / row["run_dir"] / "start.xyz"
                if not start.is_file():
                    source = Path(row.get("source_xyz", ""))
                    start = source if source.is_absolute() else manifest.parent / source
                if not start.is_file():
                    counts["missing_exclusion_xyz"] += 1
                    continue
                key = geometry_key(read_xyz(start), tolerance)
                excluded.add((k, p, key))
                counts["excluded_manifest_rows"] += 1
    return excluded, counts


def merge_candidates(
    candidates: Sequence[Candidate],
    excluded: set[tuple[int, int, tuple[object, ...]]],
) -> tuple[list[MergedCandidate], Counter[str]]:
    merged: dict[tuple[int, int, tuple[object, ...]], MergedCandidate] = {}
    counts: Counter[str] = Counter()
    for candidate in candidates:
        key = (candidate.k, candidate.p, candidate.geometry_key)
        if key in excluded:
            counts["already_in_exclusion_manifest"] += 1
            continue
        if key not in merged:
            merged[key] = MergedCandidate(
                k=candidate.k,
                p=candidate.p,
                geometry_key=candidate.geometry_key,
                fingerprint=candidate.fingerprint,
            )
        else:
            counts["duplicate_aliases_merged"] += 1
        merged[key].aliases.append(candidate)

    by_bin: dict[tuple[int, int], list[MergedCandidate]] = defaultdict(list)
    for item in merged.values():
        by_bin[(item.k, item.p)].append(item)
    for items in by_bin.values():
        scores = sorted({item.best_score for item in items}, reverse=True)
        rank = {score: index + 1 for index, score in enumerate(scores)}
        for item in items:
            item.score_layer_rank = rank[item.best_score]
    return list(merged.values()), counts


def allocate_quotas(
    groups: Mapping[tuple[int, int], Sequence[MergedCandidate]],
    target: int,
    minimum: int,
    maximum: int,
) -> dict[tuple[int, int], int]:
    bins = sorted(groups)
    cap = {
        key: min(len(groups[key]), maximum if maximum > 0 else len(groups[key]))
        for key in bins
    }
    target = min(target, sum(cap.values()))
    quota = {key: 0 for key in bins}
    remaining = target

    # First guarantee a small representation from every populated bin.
    for level in range(max(0, minimum)):
        for key in bins:
            if remaining <= 0:
                break
            if quota[key] <= level and quota[key] < cap[key]:
                quota[key] += 1
                remaining -= 1
        if remaining <= 0:
            break

    # Then equalize counts across bins, favoring larger k on exact ties.
    while remaining > 0:
        available = [key for key in bins if quota[key] < cap[key]]
        if not available:
            break
        key = min(available, key=lambda value: (quota[value], -value[0], value[1]))
        quota[key] += 1
        remaining -= 1
    return quota


def normalize_vectors(
    items: Sequence[MergedCandidate],
) -> dict[str, tuple[float, ...]]:
    raw = [item.numeric_descriptors() for item in items]
    if not raw:
        return {}
    columns = list(zip(*raw))
    bounds = [(min(values), max(values)) for values in columns]
    result: dict[str, tuple[float, ...]] = {}
    for item, vector in zip(items, raw):
        normalized = tuple(
            0.0 if high == low else (value - low) / (high - low)
            for value, (low, high) in zip(vector, bounds)
        )
        result[item.fingerprint] = normalized
    return result


def candidate_distance(
    left: MergedCandidate,
    right: MergedCandidate,
    normalized: Mapping[str, tuple[float, ...]],
) -> float:
    lv = normalized[left.fingerprint]
    rv = normalized[right.fingerprint]
    numeric = math.sqrt(sum((a - b) ** 2 for a, b in zip(lv, rv)) / len(lv))
    family = 1.0 if set(left.family_ids).isdisjoint(right.family_ids) else 0.0
    ligand = 1.0 if set(left.ligand_hashes).isdisjoint(right.ligand_hashes) else 0.0
    source = 1.0 if set(left.source_labels).isdisjoint(right.source_labels) else 0.0
    reason = 1.0 if left.reason != right.reason else 0.0
    return numeric + 0.85 * family + 0.25 * ligand + 0.15 * source + 0.10 * reason


def priority_key(item: MergedCandidate) -> tuple[object, ...]:
    rep = item.representative
    meta = rep.metadata
    ring_control = "lower_cl_ring_rank" in item.reason
    return (
        ring_control,
        -item.score_layer_rank,
        item.best_score,
        -float(meta.get("geometry_residual", 0.0) or 0.0),
        len(item.source_labels),
        item.fingerprint,
    )


def choose_diverse(
    items: Sequence[MergedCandidate],
    count: int,
    *,
    priority_fraction: float = 0.0,
) -> list[MergedCandidate]:
    if count <= 0:
        return []
    remaining = sorted(items, key=priority_key, reverse=True)
    if len(remaining) <= count:
        return remaining
    normalized = normalize_vectors(remaining)
    anchor_count = max(1, int(math.ceil(count * priority_fraction)))
    selected = remaining[:anchor_count]
    del remaining[:anchor_count]
    while remaining and len(selected) < count:
        represented_sources = set().union(*(set(item.source_labels) for item in selected))
        represented_families = set().union(*(set(item.family_ids) for item in selected))

        def next_key(item: MergedCandidate) -> tuple[object, ...]:
            min_distance = min(
                candidate_distance(item, chosen, normalized) for chosen in selected
            )
            new_source = bool(set(item.source_labels) - represented_sources)
            new_family = bool(set(item.family_ids) - represented_families)
            return (new_source, new_family, min_distance, priority_key(item))

        chosen = max(remaining, key=next_key)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def stratified_select(
    items: Sequence[MergedCandidate],
    target: int,
    minimum_per_bin: int,
    maximum_per_bin: int,
    priority_fraction: float = 0.0,
) -> list[MergedCandidate]:
    groups: dict[tuple[int, int], list[MergedCandidate]] = defaultdict(list)
    for item in items:
        groups[(item.k, item.p)].append(item)
    quotas = allocate_quotas(
        groups, target, minimum=minimum_per_bin, maximum=maximum_per_bin
    )
    selected: list[MergedCandidate] = []
    for key in sorted(groups):
        selected.extend(
            choose_diverse(
                groups[key],
                quotas[key],
                priority_fraction=priority_fraction,
            )
        )
    return selected


def capped_capacity(items: Sequence[MergedCandidate], maximum_per_bin: int) -> int:
    counts = Counter((item.k, item.p) for item in items)
    return sum(min(count, maximum_per_bin) for count in counts.values())


def required_side(coordinates: Sequence[Sequence[float]], padding: float) -> float:
    spans = [
        max(point[axis] for point in coordinates)
        - min(point[axis] for point in coordinates)
        for axis in range(3)
    ]
    return max(spans) + float(padding)


def rounded_side(value: float, minimum: float, quantum: float) -> float:
    value = max(float(value), float(minimum))
    return math.ceil(value / quantum - 1.0e-12) * quantum


def safe_structure_id(item: MergedCandidate) -> str:
    base = item.representative.structure_id
    return f"{base}__{item.fingerprint[:8]}"


def write_outputs(
    selected: Sequence[MergedCandidate],
    output: Path,
    template_path: Path,
    *,
    padding: float,
    min_box: float,
    round_box: float,
    summary: dict[str, object],
) -> None:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output}; choose a new directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    template = template_path.resolve().read_text(encoding="utf-8")
    for placeholder in ("@XYZ_FILE@", "@BOX_SIZE@"):
        if placeholder not in template:
            raise ValueError(f"CP2K template is missing {placeholder}")

    bin_side: dict[tuple[int, int], float] = {}
    for item in selected:
        xyz = item.representative.xyz
        key = (item.k, item.p)
        bin_side[key] = max(
            bin_side.get(key, 0.0), required_side(xyz.coordinates, padding)
        )
    bin_side = {
        key: rounded_side(value, min_box, round_box)
        for key, value in bin_side.items()
    }

    ordered = sorted(
        selected,
        key=lambda item: (
            item.k,
            item.p,
            item.selection_class,
            item.representative.structure_id,
            item.fingerprint,
        ),
    )
    manifest_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for index, item in enumerate(ordered):
        rep = item.representative
        structure_id = safe_structure_id(item)
        relative_dir = Path(f"k{item.k:03d}") / f"p{item.p:03d}" / structure_id
        run_dir = output / relative_dir
        run_dir.mkdir(parents=True, exist_ok=False)
        if rep.xyz_path is not None:
            shutil.copy2(rep.xyz_path, run_dir / "start.xyz")
        else:
            write_xyz(
                run_dir / "start.xyz",
                rep.xyz,
                f"reconstructed from {rep.source_reference}; {rep.xyz_kind}",
            )
        box = bin_side[(item.k, item.p)]
        box_text = f"{box:.6f}".rstrip("0").rstrip(".")
        rendered = template.replace("@XYZ_FILE@", "start.xyz").replace(
            "@BOX_SIZE@", box_text
        )
        (run_dir / "cp2k_job.in").write_text(rendered, encoding="utf-8")
        manifest_rows.append(
            {
                "index": index,
                "k": item.k,
                "p": item.p,
                "structure_id": structure_id,
                "box_angstrom": box_text,
                "run_dir": relative_dir.as_posix(),
                "source_xyz": rep.source_reference,
            }
        )
        meta = rep.metadata
        rings_raw = meta.get("rings", {})
        rings = rings_raw if isinstance(rings_raw, Mapping) else {}
        detail_rows.append(
            {
                "index": index,
                "k": item.k,
                "p": item.p,
                "structure_id": structure_id,
                "selection_class": item.selection_class,
                "selection_reason": item.reason,
                "score_layer_rank": item.score_layer_rank,
                "coordination_score": json.dumps(item.best_score),
                "skeleton_family_ids": ";".join(item.family_ids),
                "ligand_shell_hashes": ";".join(item.ligand_hashes),
                "bridge_count": meta.get("bridge_count", 0),
                "cl_ring_total": rings.get("cl_ring_total", 0),
                "inorganic_six_rings": rings.get("inorganic_six_rings", 0),
                "min_cn_compliant": bool(meta.get("min_cn_compliant", False)),
                "min_cn_total_shortfall": meta.get("min_cn_total_shortfall", 0),
                "geometry_residual": meta.get("geometry_residual", 0.0),
                "source_bundles": ";".join(item.source_labels),
                "source_statuses": ";".join(item.source_statuses),
                "source_xyz_kind": rep.xyz_kind,
                "bundle_structure_ids": ";".join(
                    sorted({alias.structure_id for alias in item.aliases})
                ),
                "source_operations": ";".join(
                    sorted(
                        {
                            str(value)
                            for alias in item.aliases
                            for value in alias.record.get("source_operations", [])
                        }
                    )
                ),
                "parent_structure_ids": ";".join(
                    sorted(
                        {
                            str(value)
                            for alias in item.aliases
                            for value in alias.record.get("source_structure_ids", [])
                        }
                    )
                ),
                "retain_bands": ";".join(
                    sorted(
                        {
                            str(alias.metadata.get("retain_band", ""))
                            for alias in item.aliases
                            if alias.metadata.get("retain_band")
                        }
                    )
                ),
                "formal_charge": meta.get("formal_charge", 0),
                "dedup_alias_count": len(item.aliases),
                "geometry_fingerprint": item.fingerprint,
                "run_dir": relative_dir.as_posix(),
                "source_xyz": rep.source_reference,
            }
        )

    def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    write_tsv(output / "manifest.tsv", manifest_rows)
    write_tsv(output / "selection.tsv", detail_rows)
    with (output / "box_sizes.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("k", "p", "box_angstrom"))
        for (k, p), box in sorted(bin_side.items()):
            writer.writerow((k, p, f"{box:g}"))
    summary["selected_total"] = len(ordered)
    summary["selected_by_class"] = dict(
        Counter(item.selection_class for item in ordered)
    )
    summary["selected_by_k"] = dict(
        sorted(Counter(str(item.k) for item in ordered).items())
    )
    summary["selected_with_source_bundle"] = {
        label: sum(label in item.source_labels for item in ordered)
        for label in sorted({value for item in ordered for value in item.source_labels})
    }
    summary["selected_shared_between_bundles"] = sum(
        len(item.source_labels) > 1 for item in ordered
    )
    summary["selected_by_bin"] = {
        f"k{k:03d}_p{p:03d}": count
        for (k, p), count in sorted(
            Counter((item.k, item.p) for item in ordered).items()
        )
    }
    summary["box_sizes_angstrom"] = {
        f"k{k:03d}_p{p:03d}": box for (k, p), box in sorted(bin_side.items())
    }
    (output / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="Completed nucleation bundle; repeat to merge search channels",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=240)
    parser.add_argument(
        "--discard-fraction",
        type=float,
        default=0.20,
        help="Target fraction of soft-rejected controls (default: 0.20)",
    )
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument(
        "--require-kmax",
        type=int,
        default=6,
        help="Refuse incomplete source bundles; 0 disables the check",
    )
    parser.add_argument(
        "--allow-incomplete-source",
        action="store_true",
        help=(
            "Read only checkpoint rows marked DONE when registry.json is not "
            "yet available; active k rows are ignored"
        ),
    )
    parser.add_argument("--min-retained-per-bin", type=int, default=3)
    parser.add_argument("--max-retained-per-bin", type=int, default=8)
    parser.add_argument("--max-rejected-per-bin", type=int, default=8)
    parser.add_argument(
        "--include-hard-rejected",
        action="store_true",
        help="Also allow min-CN failures (off by default)",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Existing CP2K manifest whose starting geometries must be excluded",
    )
    parser.add_argument(
        "--dedup-tolerance",
        type=float,
        default=1.0e-4,
        help="Angstrom quantization for element-resolved distance signatures",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).with_name("template.in"),
    )
    parser.add_argument("--padding", type=float, default=12.0)
    parser.add_argument("--min-box", type=float, default=20.0)
    parser.add_argument("--round-box", type=float, default=1.0)
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("--budget must be positive")
    if not 0.0 <= args.discard_fraction <= 1.0:
        parser.error("--discard-fraction must lie in [0, 1]")
    if args.k_min < 1 or args.k_max < args.k_min:
        parser.error("invalid k range")
    if args.require_kmax < 0:
        parser.error("--require-kmax must be nonnegative")
    if args.min_retained_per_bin < 0:
        parser.error("--min-retained-per-bin must be nonnegative")
    if args.max_retained_per_bin < 1 or args.max_rejected_per_bin < 1:
        parser.error("per-bin maxima must be positive")
    if args.dedup_tolerance <= 0:
        parser.error("--dedup-tolerance must be positive")
    if args.padding <= 0 or args.min_box <= 0 or args.round_box <= 0:
        parser.error("padding and box dimensions must be positive")
    return args


def main() -> int:
    args = parse_args()
    source_roots = [path.resolve() for path in args.source]
    labels = unique_labels(source_roots)
    all_candidates: list[Candidate] = []
    load_summary: dict[str, dict[str, int]] = {}
    for root, label in zip(source_roots, labels):
        candidates, counts = load_bundle(
            root,
            label,
            tolerance=args.dedup_tolerance,
            k_min=args.k_min,
            k_max=args.k_max,
            require_kmax=args.require_kmax,
            include_hard_rejected=args.include_hard_rejected,
            allow_incomplete=args.allow_incomplete_source,
        )
        all_candidates.extend(candidates)
        load_summary[label] = dict(counts)
        print(f"Loaded {len(candidates)} eligible records from {label}")

    excluded, exclusion_counts = load_excluded_geometries(
        args.exclude_manifest, args.dedup_tolerance
    )
    merged, merge_counts = merge_candidates(all_candidates, excluded)
    retained = [item for item in merged if item.selection_class == "retained"]
    rejected = [item for item in merged if item.selection_class == "soft_rejected"]
    reject_capacity = capped_capacity(rejected, args.max_rejected_per_bin)
    retain_capacity = capped_capacity(retained, args.max_retained_per_bin)
    reject_target = min(
        reject_capacity, int(round(args.budget * args.discard_fraction))
    )
    retain_target = min(retain_capacity, args.budget - reject_target)
    if retain_target + reject_target < args.budget:
        spare = args.budget - retain_target - reject_target
        add_retained = min(spare, retain_capacity - retain_target)
        retain_target += add_retained
        spare -= add_retained
        reject_target += min(spare, reject_capacity - reject_target)

    selected_retained = stratified_select(
        retained,
        retain_target,
        minimum_per_bin=args.min_retained_per_bin,
        maximum_per_bin=args.max_retained_per_bin,
        priority_fraction=0.0,
    )
    selected_rejected = stratified_select(
        rejected,
        reject_target,
        minimum_per_bin=0,
        maximum_per_bin=args.max_rejected_per_bin,
        # Half of every rejected-bin quota remains close to the retained
        # score boundary; the rest maximizes motif and source diversity.
        priority_fraction=0.5,
    )
    selected = selected_retained + selected_rejected
    summary: dict[str, object] = {
        "sources": [str(path) for path in source_roots],
        "load_counts": load_summary,
        "exclusion_counts": dict(exclusion_counts),
        "merge_counts": dict(merge_counts),
        "configuration": {
            "budget": args.budget,
            "discard_fraction": args.discard_fraction,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "require_kmax": args.require_kmax,
            "allow_incomplete_source": args.allow_incomplete_source,
            "min_retained_per_bin": args.min_retained_per_bin,
            "max_retained_per_bin": args.max_retained_per_bin,
            "max_rejected_per_bin": args.max_rejected_per_bin,
            "include_hard_rejected": args.include_hard_rejected,
            "dedup_tolerance_angstrom": args.dedup_tolerance,
            "padding_angstrom": args.padding,
            "min_box_angstrom": args.min_box,
            "round_box_angstrom": args.round_box,
            "exclude_manifests": [str(path.resolve()) for path in args.exclude_manifest],
        },
        "eligible_after_merge": {
            "retained": len(retained),
            "soft_rejected": len(rejected),
        },
        "selection_targets": {
            "retained": retain_target,
            "soft_rejected": reject_target,
        },
    }
    write_outputs(
        selected,
        args.output,
        args.template,
        padding=args.padding,
        min_box=args.min_box,
        round_box=args.round_box,
        summary=summary,
    )
    print(
        f"Selected {len(selected_retained)} retained + "
        f"{len(selected_rejected)} soft-rejected = {len(selected)} structures"
    )
    print(f"CP2K manifest: {args.output.resolve() / 'manifest.tsv'}")
    print(f"Selection metadata: {args.output.resolve() / 'selection.tsv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
