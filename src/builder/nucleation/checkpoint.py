from __future__ import annotations

from .bundle import *  # private names via __all__

from .graph_ops import *  # private names via __all__

from .surface import *  # private names via __all__

from .scoring import *  # private names via __all__

from .lattice import *  # private names via __all__

from .spec import *  # private names via __all__

from .types import *  # private names via __all__

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import yaml

from ..io_utils import write_xyz
from ..nc_types import NucleationSpec
from .types import (
    AtomRecord,
    ClusterRecord,
    NucleationRegistry,
    NucleationResult,
    SweepAudit,
    _State,
)

def _state_to_checkpoint(state: _State) -> Dict[str, object]:
    return {
        "atoms": [
            {
                "id": atom.atom_id,
                "symbol": atom.symbol,
                "coordinates": list(atom.coordinates),
                "role": atom.role,
                "unit_id": atom.unit_id,
            }
            for atom in state.atoms
        ],
        "edges": [
            {
                "source": int(left),
                "target": int(right),
                **{
                    key: _json_value(value)
                    for key, value in data.items()
                },
            }
            for left, right, data in state.graph.edges(data=True)
        ],
        "geometry_residual": float(state.geometry_residual),
    }


def _state_from_checkpoint(payload: Mapping[str, object]) -> _State:
    atoms = tuple(
        AtomRecord(
            int(item["id"]),
            str(item["symbol"]),
            tuple(float(v) for v in item["coordinates"]),
            str(item["role"]),
            item.get("unit_id"),
        )
        for item in payload["atoms"]  # type: ignore[index]
    )
    graph = nx.Graph()
    for atom in atoms:
        graph.add_node(
            atom.atom_id,
            element=atom.symbol,
            role=atom.role,
            unit_id=atom.unit_id,
        )
    for edge in payload.get("edges", []):  # type: ignore[union-attr]
        data = {
            key: value
            for key, value in edge.items()
            if key not in {"source", "target"}
        }
        graph.add_edge(int(edge["source"]), int(edge["target"]), **data)
    return _State(
        atoms,
        graph,
        float(payload.get("geometry_residual", 0.0)),
    )


def _record_from_dict(payload: Mapping[str, object]) -> ClusterRecord:
    """Rebuild a ClusterRecord from ``_record_to_dict`` / checkpoint JSON."""

    atoms = [
        AtomRecord(
            int(item["id"]),
            str(item["symbol"]),
            tuple(float(v) for v in item["coordinates"]),
            str(item["role"]),
            item.get("unit_id"),
        )
        for item in payload["atoms"]  # type: ignore[index]
    ]
    graph_payload = payload.get("graph")
    if isinstance(graph_payload, Mapping):
        graph = nx.node_link_graph(graph_payload, edges="edges")
    else:
        graph = nx.Graph()
        for atom in atoms:
            graph.add_node(
                atom.atom_id,
                element=atom.symbol,
                role=atom.role,
                unit_id=atom.unit_id,
            )
    surface_data = payload.get("surface_coordinates")
    surface_tuple = None
    if isinstance(surface_data, list) and surface_data:
        surface_tuple = tuple(
            tuple(float(v) for v in point) for point in surface_data
        )
    selection = payload.get("selection") or {}
    if not isinstance(selection, Mapping):
        selection = {}
    score_raw = selection.get("coordination_score") or ()
    return ClusterRecord(
        structure_id=str(payload.get("structure_id", "")),
        k=int(payload["k"]),
        p=int(payload["p"]),
        atoms=atoms,
        graph=graph,
        selection_status=str(selection.get("status", "unranked")),
        selection_reason=str(selection.get("reason", "")),
        coordination_score=tuple(int(v) for v in score_raw),
        source_operations=tuple(
            str(v) for v in (payload.get("source_operations") or ())
        ),
        source_structure_ids=tuple(
            str(v) for v in (payload.get("source_structure_ids") or ())
        ),
        metadata=dict(payload.get("metadata") or {}),
        surface_coordinates_data=surface_tuple,
    )


def _skeletons_to_checkpoint(
    skeletons: Mapping[int, Sequence[Tuple[_State, Tuple[str, ...]]]],
) -> Dict[str, object]:
    return {
        str(p): [
            {
                "state": _state_to_checkpoint(state),
                "routes": list(routes),
            }
            for state, routes in entries
        ]
        for p, entries in sorted(skeletons.items())
    }


def _skeletons_from_checkpoint(
    payload: Mapping[str, object],
) -> Dict[int, List[Tuple[_State, Tuple[str, ...]]]]:
    result: Dict[int, List[Tuple[_State, Tuple[str, ...]]]] = {}
    for p_raw, entries in payload.items():
        rows: List[Tuple[_State, Tuple[str, ...]]] = []
        for item in entries:  # type: ignore[union-attr]
            state = _state_from_checkpoint(item["state"])
            routes = tuple(str(r) for r in item.get("routes") or ())
            rows.append((state, routes))
        result[int(p_raw)] = rows
    return result


def write_nucleation_checkpoint(
    *,
    root: str | Path,
    spec: NucleationSpec,
    k: int,
    retained: Mapping[int, Sequence[ClusterRecord]],
    discarded: Mapping[int, Sequence[ClusterRecord]],
    skeletons: Mapping[int, Sequence[Tuple[_State, Tuple[str, ...]]]],
    discarded_counts: Mapping[int, int],
    mark_done: bool = True,
    last_completed_p: Optional[int] = None,
    p_cap: Optional[int] = None,
    max_inherited: Optional[int] = None,
    inherited: Optional[
        Mapping[int, Sequence[Tuple[_State, Tuple[str, ...]]]]
    ] = None,
) -> Path:
    """Persist a k-row checkpoint for ``--restart``.

    With ``mark_done=False`` this is a *partial* row (p-ladder still running):
    retained/skeletons are rewritten after each finished ``(k, p)`` bin so a
    crash does not lose completed bins.  ``DONE`` is written only when
    ``mark_done=True`` (full k finished).
    """

    root_path = Path(root)
    checkpoint = root_path / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    fingerprint_path = checkpoint / "run_fingerprint.json"
    fingerprint_path.write_text(
        json.dumps(_spec_run_fingerprint(spec), indent=2, sort_keys=True) + "\n"
    )
    k_dir = checkpoint / f"k{k:03d}"
    k_dir.mkdir(parents=True, exist_ok=True)
    (k_dir / "retained.json").write_text(
        json.dumps(
            {
                str(p): [_record_to_dict(record) for record in records]
                for p, records in sorted(retained.items())
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (k_dir / "discarded.json").write_text(
        json.dumps(
            {
                str(p): [_record_to_dict(record) for record in records]
                for p, records in sorted(discarded.items())
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (k_dir / "discarded_counts.json").write_text(
        json.dumps(
            {str(p): int(count) for p, count in sorted(discarded_counts.items())},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (k_dir / "skeletons.json").write_text(
        json.dumps(_skeletons_to_checkpoint(skeletons), indent=2, sort_keys=True)
        + "\n"
    )
    if inherited is not None:
        (k_dir / "inherited.json").write_text(
            json.dumps(
                _skeletons_to_checkpoint(inherited), indent=2, sort_keys=True
            )
            + "\n"
        )
    completed_p = (
        int(last_completed_p)
        if last_completed_p is not None
        else (max(retained) if retained else -1)
    )
    progress_payload = {
        "status": "done" if mark_done else "in_progress",
        "k": int(k),
        "last_completed_p": completed_p,
        "p_cap": None if p_cap is None else int(p_cap),
        "max_inherited": None if max_inherited is None else int(max_inherited),
        "retained_bins": sorted(int(p) for p in retained),
        "skeleton_bins": sorted(int(p) for p in skeletons),
    }
    (k_dir / "progress.json").write_text(
        json.dumps(progress_payload, indent=2, sort_keys=True) + "\n"
    )
    done_path = k_dir / "DONE"
    if mark_done:
        done_path.write_text(f"k={k}\n")
    elif done_path.is_file():
        done_path.unlink()
    return k_dir


def write_nucleation_bin_structures(
    *,
    root: str | Path,
    k: int,
    p: int,
    retained: Sequence[ClusterRecord],
    discarded: Sequence[ClusterRecord],
    write_discarded: bool,
) -> None:
    """Write XYZ for one finished ``(k, p)`` bin (retained always; discarded optional).

    Layout: ``structures/k{k:03d}/p{p:03d}/{retained|discarded}/``.
    Called as soon as a bin is selected so restart / inspection does not wait
    for the full k-row.
    """

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    targets: List[Tuple[str, Sequence[ClusterRecord]]] = [
        ("retained", retained),
    ]
    if write_discarded:
        targets.append(("discarded", discarded))
    for status, records in targets:
        directory = (
            root_path / "structures" / f"k{k:03d}" / f"p{p:03d}" / status
        )
        if directory.exists():
            shutil.rmtree(directory)
        if not records:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        for record in records:
            construction_path = (
                directory / f"{record.structure_id}_construction_native.xyz"
            )
            record.metadata["construction_native_xyz_path"] = str(
                construction_path.relative_to(root_path)
            )
            write_xyz(
                str(construction_path),
                record.symbols,
                record.coordinates,
                comment=(
                    f"{_formula(record.symbols)}_construction_native_"
                    f"graph_ranked_bridges_{record.metadata.get('bridge_count', 0)}"
                ),
            )
            if (
                status == "retained"
                or record.metadata.get("surface_selection_rejected", False)
            ):
                projection_valid = bool(
                    record.metadata.get("surface_geometry", {}).get(
                        "projection_valid", False
                    )
                )
                surface_suffix = (
                    "surface" if projection_valid else "surface_rejected"
                )
                surface_path = (
                    directory / f"{record.structure_id}_{surface_suffix}.xyz"
                )
                record.metadata["surface_xyz_path"] = str(
                    surface_path.relative_to(root_path)
                )
                write_xyz(
                    str(surface_path),
                    record.symbols,
                    record.surface_coordinates,
                    comment=(
                        f"{_formula(record.symbols)}_surface_projected_"
                        f"valid_{str(projection_valid).lower()}"
                    ),
                )


def detect_checkpoint_k_done(root: str | Path) -> int:
    """Return the highest k with a complete checkpoint, or 0 if none."""

    checkpoint = Path(root) / "checkpoint"
    if not checkpoint.is_dir():
        return 0
    done = 0
    for path in checkpoint.glob("k*/DONE"):
        try:
            k = int(path.parent.name[1:])
        except ValueError:
            continue
        if (path.parent / "retained.json").is_file() and (
            path.parent / "skeletons.json"
        ).is_file():
            done = max(done, k)
    return done


def detect_checkpoint_partial_k(
    root: str | Path,
) -> Optional[Tuple[int, int]]:
    """Return ``(k, last_completed_p)`` for an in-progress k-row, if any.

    A partial row has ``progress.json`` with ``status=in_progress`` and no
    ``DONE`` file.  Used to resume the p-ladder inside that k.
    """

    checkpoint = Path(root) / "checkpoint"
    if not checkpoint.is_dir():
        return None
    partial: Optional[Tuple[int, int]] = None
    for path in checkpoint.glob("k*/progress.json"):
        k_dir = path.parent
        if (k_dir / "DONE").is_file():
            continue
        try:
            k = int(k_dir.name[1:])
        except ValueError:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("status", "")) != "in_progress":
            continue
        if not (k_dir / "retained.json").is_file():
            continue
        if not (k_dir / "skeletons.json").is_file():
            continue
        last_p = int(payload.get("last_completed_p", -1))
        if partial is None or k > partial[0]:
            partial = (k, last_p)
    return partial


def load_nucleation_checkpoint(
    root: str | Path,
    spec: NucleationSpec,
    *,
    force: bool = False,
) -> Tuple[int, NucleationResult, Dict[int, Dict[int, List[Tuple[_State, Tuple[str, ...]]]]]]:
    """Load finished k-rows from ``root/checkpoint`` for resume.

    Returns ``(k_done, partial_result, skeleton_rows)``.
    Incomplete in-progress k rows are *not* loaded here; use
    :func:`load_nucleation_partial_k` after this.
    """

    root_path = Path(root)
    checkpoint = root_path / "checkpoint"
    fingerprint_path = checkpoint / "run_fingerprint.json"
    if not fingerprint_path.is_file():
        raise FileNotFoundError(
            f"no checkpoint fingerprint at {fingerprint_path}; "
            "cannot restart (use a clean output directory for a new run)"
        )
    stored = json.loads(fingerprint_path.read_text())
    expected = _spec_run_fingerprint(spec)
    if stored != expected and not force:
        mismatched = sorted(
            key
            for key in set(stored) | set(expected)
            if stored.get(key) != expected.get(key)
        )
        raise ValueError(
            "checkpoint fingerprint does not match this YAML/CIF/rules; "
            "refusing to restart. Mismatched keys: "
            + ", ".join(mismatched)
            + ". Pass force_restart to override."
        )
    k_done = detect_checkpoint_k_done(root_path)
    partial = detect_checkpoint_partial_k(root_path)
    if k_done < 1 and partial is None:
        raise FileNotFoundError(
            f"no complete or in-progress k checkpoint under {checkpoint}"
        )
    result = NucleationResult(
        registry={},
        discarded_registry={},
        discarded_counts={},
        graph_rules=expected["graph_rules"],  # type: ignore[arg-type]
        geometry_rules={},
        reference_bond_length=0.0,
    )
    skeleton_rows: Dict[
        int, Dict[int, List[Tuple[_State, Tuple[str, ...]]]]
    ] = {}
    for k in range(1, k_done + 1):
        k_dir = checkpoint / f"k{k:03d}"
        retained_raw = json.loads((k_dir / "retained.json").read_text())
        discarded_raw = json.loads((k_dir / "discarded.json").read_text())
        counts_raw = json.loads((k_dir / "discarded_counts.json").read_text())
        skeletons_raw = json.loads((k_dir / "skeletons.json").read_text())
        result.registry[k] = {
            int(p): [_record_from_dict(item) for item in records]
            for p, records in retained_raw.items()
        }
        if k <= spec.discarded_through_k:
            result.discarded_registry[k] = {
                int(p): [_record_from_dict(item) for item in records]
                for p, records in discarded_raw.items()
            }
        result.discarded_counts[k] = {
            int(p): int(count) for p, count in counts_raw.items()
        }
        skeleton_rows[k] = _skeletons_from_checkpoint(skeletons_raw)
    return k_done, result, skeleton_rows


def load_nucleation_partial_k(
    root: str | Path,
    spec: NucleationSpec,
    k: int,
) -> Dict[str, object]:
    """Load in-progress state for one unfinished k-row (p-ladder resume)."""

    k_dir = Path(root) / "checkpoint" / f"k{k:03d}"
    progress = json.loads((k_dir / "progress.json").read_text())
    retained_raw = json.loads((k_dir / "retained.json").read_text())
    discarded_raw = json.loads((k_dir / "discarded.json").read_text())
    counts_raw = json.loads((k_dir / "discarded_counts.json").read_text())
    skeletons_raw = json.loads((k_dir / "skeletons.json").read_text())
    inherited_path = k_dir / "inherited.json"
    inherited: Dict[int, List[Tuple[_State, Tuple[str, ...]]]] = {}
    if inherited_path.is_file():
        inherited = _skeletons_from_checkpoint(
            json.loads(inherited_path.read_text())
        )
    return {
        "progress": progress,
        "retained": {
            int(p): [_record_from_dict(item) for item in records]
            for p, records in retained_raw.items()
        },
        "discarded": {
            int(p): [_record_from_dict(item) for item in records]
            for p, records in discarded_raw.items()
        },
        "discarded_counts": {
            int(p): int(count) for p, count in counts_raw.items()
        },
        "skeletons": _skeletons_from_checkpoint(skeletons_raw),
        "inherited": inherited,
        "last_completed_p": int(progress.get("last_completed_p", -1)),
        "p_cap": progress.get("p_cap"),
        "max_inherited": progress.get("max_inherited"),
    }


def write_nucleation_k_structures(
    result: NucleationResult,
    output_directory: str | Path,
    k: int,
) -> None:
    """Write XYZ trees for one k without deleting other k folders."""

    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    retained_bins = result.registry.get(k, {})
    discarded_bins = result.discarded_registry.get(k, {})
    all_p = sorted(set(retained_bins) | set(discarded_bins))
    for p in all_p:
        write_nucleation_bin_structures(
            root=root,
            k=k,
            p=p,
            retained=retained_bins.get(p, ()),
            discarded=discarded_bins.get(p, ()),
            write_discarded=bool(discarded_bins.get(p)),
        )

__all__ = [
    '_state_to_checkpoint',
    '_state_from_checkpoint',
    '_record_from_dict',
    '_skeletons_to_checkpoint',
    '_skeletons_from_checkpoint',
    'write_nucleation_checkpoint',
    'write_nucleation_bin_structures',
    'detect_checkpoint_k_done',
    'detect_checkpoint_partial_k',
    'load_nucleation_checkpoint',
    'load_nucleation_partial_k',
    'write_nucleation_k_structures',
]
