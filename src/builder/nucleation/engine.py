from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
from itertools import combinations, permutations
import json
import math
from pathlib import Path
import shutil
import textwrap
import time
from typing import Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Structure
from scipy.spatial import ConvexHull, QhullError
import yaml

from ..analysis import cif_first_shell_vector_sets, derive_pair_cuts_from_cif
from ..graph_canon import canonical_form, compress_leaves
from ..io_utils import write_xyz
from ..nc_types import (
    CoreMonomerSpec,
    NucleationBridgeRule,
    NucleationGeometryRules,
    NucleationGraphRules,
    NucleationSpec,
    PrecursorUnitSpec,
)

from .types import *
from .spec import *
from .graph_ops import *
from .lattice import *
from .scoring import *
from .surface import *
from .checkpoint import *
from .bundle import *

def generate_nucleation_map(spec: NucleationSpec) -> NucleationRegistry:
    """Return the retained coordination-optimal registry."""

    return generate_nucleation_result(spec).registry


def generate_nucleation_result(
    spec: NucleationSpec,
    *,
    progress: Optional[ProgressCallback] = None,
    verbose: bool = False,
    progress_interval: float = 5.0,
    checkpoint_dir: Optional[str | Path] = None,
    restart: bool = False,
    force_restart: bool = False,
) -> NucleationResult:
    """Generate coordination-ranked ``(k,p)`` maps through ``spec.kmax``.

    Library calls remain silent unless ``progress`` is supplied.  The callback
    receives complete, prefixed lines.  ``progress_interval`` controls
    heartbeats from long combinatorial loops and may be set to zero in tests.

    When ``checkpoint_dir`` is set, each finished ``k`` is written under
    ``checkpoint/kXXX/`` so a later call with ``restart=True`` can continue from
    the last complete row.  ``force_restart`` ignores fingerprint mismatches.
    """

    _validate_spec(spec)
    model = _build_lattice_model(spec)
    reporter = _ProgressReporter(
        callback=progress,
        verbose=verbose,
        interval_seconds=max(0.0, float(progress_interval)),
    )
    enumeration_cache = _EnumerationCache()
    reporter.emit(
        f"starting map: kmax={spec.kmax}, "
        f"reference_bond_length={model.bond_length:.6f} A"
    )
    result = NucleationResult(
        registry={},
        discarded_registry={},
        discarded_counts={},
        graph_rules={
            "min_cn": dict(sorted(spec.graph_rules.min_cn.items())),
            "max_cn": dict(sorted(spec.graph_rules.max_cn.items())),
            "allowed_bonds": [
                list(pair) for pair in spec.graph_rules.allowed_bonds
            ],
            "bridging": {
                rule.ligand: {
                    "host": rule.host,
                    "shared_neighbor": rule.shared_neighbor,
                    "surface_angle_deg": rule.surface_angle_deg,
                    "min_bridged_host_cn": rule.min_bridged_host_cn,
                }
                for rule in spec.graph_rules.bridge_rules
            },
        },
        geometry_rules={
            "by_cn": {
                symbol: {
                    f"cn{coordination}": template
                    for coordination, template in sorted(rules.items())
                }
                for symbol, rules in sorted(spec.geometry_rules.by_cn.items())
            },
            "all": dict(sorted(spec.geometry_rules.all_cn.items())),
        },
        reference_bond_length=model.bond_length,
    )

    skeleton_rows: Dict[
        int, Dict[int, List[Tuple[_State, Tuple[str, ...]]]]
    ] = {}
    narrowing_audit: Dict[str, object] = {}
    start_k = 1
    partial_resume: Optional[Dict[str, object]] = None
    if restart:
        if checkpoint_dir is None:
            raise ValueError("restart=True requires checkpoint_dir")
        k_done, loaded, skeleton_rows = load_nucleation_checkpoint(
            checkpoint_dir, spec, force=force_restart
        )
        partial_info = detect_checkpoint_partial_k(checkpoint_dir)
        if k_done >= spec.kmax and partial_info is None:
            reporter.emit(
                f"restart: checkpoint already complete through k={k_done} "
                f"(kmax={spec.kmax}); nothing to do"
            )
            loaded.reference_bond_length = model.bond_length
            loaded.geometry_rules = result.geometry_rules
            loaded.graph_rules = result.graph_rules
            loaded.completeness = _completeness_report(
                spec, loaded, narrowing_audit, reporter
            )
            return loaded
        reporter.emit(
            f"restart: loaded complete rows through k={k_done}; "
            f"continuing to kmax={spec.kmax}"
        )
        result.registry = loaded.registry
        result.discarded_registry = loaded.discarded_registry
        result.discarded_counts = loaded.discarded_counts
        result.reference_bond_length = model.bond_length
        start_k = max(k_done, 1)
        if partial_info is not None:
            pk, last_p = partial_info
            if pk > k_done and pk <= spec.kmax:
                partial_resume = load_nucleation_partial_k(
                    checkpoint_dir, spec, pk
                )
                # Prefer stored inherited when present.
                stored_inh = partial_resume.get("inherited") or {}
                reporter.emit(
                    f"restart: found in-progress k={pk} "
                    f"(last completed p={last_p}); resuming p-ladder"
                )
                resume_inherited = stored_inh
                if not resume_inherited and pk == 1:
                    resume_inherited = {
                        0: [(_seed_state(model), ("seed",))]
                    }
                partial_valid = bool(resume_inherited) or pk == 1
                partial_error = "stored inherited set is empty"
                if partial_valid:
                    try:
                        for p_key, entries in resume_inherited.items():  # type: ignore[union-attr]
                            for state, _routes in entries:
                                has_ligands = any(
                                    atom.role == "precursor_ligand"
                                    for atom in state.atoms
                                )
                                if has_ligands:
                                    _assert_atoms_match_bin(
                                        state.atoms,
                                        k=pk,
                                        p=int(p_key),
                                        spec=spec,
                                        context="restart inherited checkpoint",
                                    )
                                else:
                                    _assert_bare_skeleton_matches_bin(
                                        state.atoms,
                                        k=pk,
                                        p=int(p_key),
                                        spec=spec,
                                        context="restart inherited checkpoint",
                                    )
                    except AssertionError as exc:
                        partial_valid = False
                        partial_error = str(exc)
                if not partial_valid:
                    # A failed run may have checkpointed channel products from
                    # buggy/older bookkeeping.  Finished rows remain sound;
                    # ignore only the partial destination row and regenerate
                    # its inherited growth from the last DONE row.
                    reporter.emit(
                        f"restart: ignoring incompatible partial k={pk} "
                        f"checkpoint ({partial_error}); recomputing from "
                        f"complete k={k_done}"
                    )
                    partial_resume = None
                else:
                    retained, discarded, skeletons, audits = _complete_k_dag(
                        k=pk,
                        inherited=resume_inherited,  # type: ignore[arg-type]
                        model=model,
                        spec=spec,
                        progress=reporter,
                        cache=enumeration_cache,
                        checkpoint_dir=checkpoint_dir,
                        resume_state=partial_resume,
                    )
                    skeleton_rows[pk] = skeletons
                    result.registry[pk] = retained
                    if pk <= spec.discarded_through_k:
                        result.discarded_registry[pk] = discarded
                    result.discarded_counts[pk] = {
                        p: len(records) for p, records in discarded.items()
                    }
                    result.sweep_audit.extend(audits)
                    for audit in audits:
                        dropped = int(
                            audit.stage_counts.get("p_skeleton_beam_dropped", 0)
                        )
                        if dropped:
                            narrowing_audit["p_beam_dropped"] = (
                                int(narrowing_audit.get("p_beam_dropped", 0))
                                + dropped
                            )
                    start_k = pk
                    partial_resume = None
    else:
        retained, discarded, skeletons, audits = _complete_k_dag(
            k=1,
            inherited={0: [(_seed_state(model), ("seed",))]},
            model=model,
            spec=spec,
            progress=reporter,
            cache=enumeration_cache,
            checkpoint_dir=checkpoint_dir,
        )
        skeleton_rows[1] = skeletons
        result.registry[1] = {
            p: recs for p, recs in retained.items() if recs
        }
        result.discarded_registry[1] = discarded
        result.discarded_counts[1] = {
            p: len(records) for p, records in discarded.items()
        }
        result.sweep_audit.extend(audits)
        for audit in audits:
            dropped = int(
                audit.stage_counts.get("p_skeleton_beam_dropped", 0)
            )
            if dropped:
                narrowing_audit["p_beam_dropped"] = (
                    int(narrowing_audit.get("p_beam_dropped", 0)) + dropped
                )
        if checkpoint_dir is not None:
            reporter.emit(
                f"checkpoint: k=1 complete under {checkpoint_dir}"
            )

    for k in range(start_k, spec.kmax):
        next_initial: Dict[int, List[Tuple[_State, Tuple[str, ...]]]] = {}
        growth_audits: List[SweepAudit] = []
        # Distinct skeletons grow exponentially in k while the retained set stays
        # flat, so above the exact threshold only the cores of retained
        # structures carry forward.  The row itself is still enumerated in full;
        # this narrows only what *leaves* it.
        narrowed = k >= spec.exact_through_k
        use_decorated = (
            narrowed and spec.core_growth_occupation == "decorated"
        )
        # Parents: all retained/skeleton p by default; optional legacy seed_band.
        parent_p_values = sorted(skeleton_rows[k])
        if narrowed:
            parent_p_values = sorted(result.registry.get(k, {}))
        growth_p_list = [
            p for p in parent_p_values if _p_allowed_for_k_growth(p, spec)
        ]
        packages = _monomer_packages(spec)
        if not growth_p_list:
            reporter.emit(
                f"k={k} -> k={k + 1}: no parent p-bins under "
                f"parent_p_mode={spec.parent_p_mode}; core growth closed"
            )
            break
        reporter.emit(
            f"k={k} -> k={k + 1}: parent_p_mode={spec.parent_p_mode} "
            f"parents_p={growth_p_list}; monomer packages p_m={list(packages)}"
        )
        narrowing_audit["parent_p_mode"] = spec.parent_p_mode
        narrowing_audit["monomer_p_values"] = list(packages)
        if spec.parent_p_mode == "seed_band":
            narrowing_audit["seed_p"] = (
                None if spec.seed_p is None else int(spec.seed_p)
            )
            narrowing_audit["seed_p_window"] = int(spec.seed_p_window)
            narrowing_audit["seed_p_filtered"] = 1
        growth_sources: Dict[int, List[Tuple[_State, Tuple[str, ...]]]] = {}
        if not use_decorated:
            for p in growth_p_list:
                if narrowed:
                    growth_sources[p] = _retained_core_sources(
                        result.registry.get(k, {}).get(p, ()), model, spec
                    )
                else:
                    growth_sources[p] = list(skeleton_rows[k].get(p, ()))
        if narrowed and not use_decorated:
            before = sum(len(v) for v in skeleton_rows[k].values())
            after = sum(len(v) for v in growth_sources.values())
            reporter.emit(
                f"k={k} -> k={k + 1}: NARROWED to retained cores "
                f"(exact_through_k={spec.exact_through_k}): "
                f"skeletons={before} -> cores={after}; "
                "enumeration above this k is no longer complete"
            )
            narrowing_audit["skeletons_before"] = (
                narrowing_audit.get("skeletons_before", 0) + before
            )
            narrowing_audit["cores_after"] = (
                narrowing_audit.get("cores_after", 0) + after
            )
            # Only a step that actually drops skeletons can lose a structure.
            # The rule being switched on is not itself a loss of completeness.
            if after < before:
                narrowing_audit["first_binding_k"] = min(
                    narrowing_audit.get("first_binding_k", k), k
                )
                narrowing_audit["binding_steps"] = (
                    narrowing_audit.get("binding_steps", 0) + 1
                )
        if use_decorated:
            before = sum(len(v) for v in skeleton_rows[k].values())
            retained_n = sum(
                len(result.registry.get(k, {}).get(p, ()))
                for p in growth_p_list
            )
            reporter.emit(
                f"k={k} -> k={k + 1}: NARROWED to decorated retained "
                f"(exact_through_k={spec.exact_through_k}, "
                f"core_growth_occupation=decorated): "
                f"skeletons={before} -> decorated_parents={retained_n}; "
                "Cl-occupied tetrahedral sites block monomer attachment"
            )
            narrowing_audit["skeletons_before"] = (
                narrowing_audit.get("skeletons_before", 0) + before
            )
            narrowing_audit["cores_after"] = (
                narrowing_audit.get("cores_after", 0) + retained_n
            )
            narrowing_audit["decorated_growth"] = 1
            if retained_n < before:
                narrowing_audit["first_binding_k"] = min(
                    narrowing_audit.get("first_binding_k", k), k
                )
                narrowing_audit["binding_steps"] = (
                    narrowing_audit.get("binding_steps", 0) + 1
                )
        active_growth_policy = (
            spec.core_growth_policy
            if (
                spec.core_growth_policy != "all"
                and (k + 1) >= spec.compact_from_k
            )
            else "all"
        )
        if active_growth_policy != "all":
            reporter.emit(
                f"k={k} -> k={k + 1}: core growth policy="
                f"{active_growth_policy} "
                f"(compact_from_k={spec.compact_from_k}); "
                "open/branched monomer attachments may be dropped"
            )
            narrowing_audit["core_growth_policy"] = active_growth_policy
            narrowing_audit["compact_from_k"] = spec.compact_from_k
            narrowing_audit["core_growth_first_k"] = min(
                int(narrowing_audit.get("core_growth_first_k", k + 1)),
                k + 1,
            )
        next_decorated: Dict[
            int, List[Tuple[_State, Tuple[str, ...]]]
        ] = {}
        use_continuous = (
            use_decorated and bool(spec.continuous_decoration)
        )
        if use_decorated:
            if float(spec.p_surf_beta) > 0.0:
                grow_msg = (
                    f"k={k} -> k={k + 1}: growing from decorated retained "
                    f"(p_surf_beta={spec.p_surf_beta}, "
                    f"s_max_law~min(p,floor({spec.shed_alpha}*p_surf(k))), "
                    f"p_surf(k+1)={_p_surf(k + 1, spec.p_surf_beta)}, "
                    f"packages={list(packages)}, continuous={use_continuous})"
                )
            else:
                grow_msg = (
                    f"k={k} -> k={k + 1}: growing from decorated retained "
                    f"(max_shed={'all' if spec.k_growth_max_shed == 0 else spec.k_growth_max_shed}, "
                    f"packages={list(packages)}, continuous={use_continuous})"
                )
            reporter.emit(grow_msg)
            if use_continuous:
                narrowing_audit["continuous_decoration"] = 1
            for p in growth_p_list:
                records = list(result.registry.get(k, {}).get(p, ()))
                if (
                    spec.growth_max_parents_per_bin > 0
                    and len(records) > spec.growth_max_parents_per_bin
                ):
                    records = sorted(
                        records,
                        key=lambda rec: (
                            rec.coordination_score,
                            rec.structure_id or "",
                        ),
                        reverse=True,
                    )[: spec.growth_max_parents_per_bin]
                    reporter.emit(
                        f"k={k} -> k={k + 1} p={p}: growth_max_parents_per_bin="
                        f"{spec.growth_max_parents_per_bin} "
                        f"(truncated parents for continuous/decorated growth)"
                    )
                for p_m in packages:
                    raw_by_p, attempted, growth_stats = (
                        _decorated_core_children_by_p(
                            records,
                            k_from=k,
                            p=p,
                            model=model,
                            spec=spec,
                            p_m=p_m,
                        )
                    )
                    total_unique = 0
                    for p_to, raw_children in sorted(raw_by_p.items()):
                        if use_continuous:
                            children = _unique_decorated_with_routes(
                                raw_children,
                                progress=reporter,
                                context=(
                                    f"k={k}->{k + 1} p={p}+pm{p_m}->p0={p_to} "
                                    "continuous merge"
                                ),
                            )
                            if children:
                                next_decorated.setdefault(p_to, []).extend(
                                    children
                                )
                        else:
                            children = _unique_skeleton_candidates(
                                raw_children,
                                model,
                                spec,
                                reporter,
                                context=(
                                    f"k={k}->{k + 1} p={p}+pm{p_m}->p0={p_to} "
                                    "decorated merge"
                                ),
                            )
                            if children:
                                next_initial.setdefault(p_to, []).extend(
                                    children
                                )
                        total_unique += len(children)
                    policy_pruned = int(
                        growth_stats.get("core_growth_policy_pruned", 0)
                    )
                    reporter.emit(
                        f"k={k} -> k={k + 1} p={p} + p_m={p_m} (decorated"
                        f"{', continuous' if use_continuous else ''}): "
                        f"raw={attempted}, "
                        f"connected={growth_stats.get('core_growth_raw_connected', 0)}, "
                        f"after_policy={growth_stats.get('core_growth_after_policy', 0)}, "
                        f"policy_pruned={policy_pruned}, "
                        f"unique_total={total_unique}, "
                        f"p0_targets={sorted(raw_by_p)}"
                    )
                    if policy_pruned:
                        narrowing_audit["core_growth_policy_pruned"] = (
                            int(
                                narrowing_audit.get(
                                    "core_growth_policy_pruned", 0
                                )
                            )
                            + policy_pruned
                        )
                    growth_audits.append(
                        SweepAudit(
                            k=k,
                            operation="core_skeleton_growth",
                            p_from=p,
                            p_to=p,
                            source_count=len(records),
                            raw_count=attempted,
                            valid_count=total_unique,
                            symmetry_duplicate_count=0,
                            stage_counts={
                                "ligand_enumerations": 0,
                                "growth_narrowed_to_retained_cores": 1,
                                "growth_occupation": "decorated",
                                "continuous_decoration": int(use_continuous),
                                "monomer_p_m": int(p_m),
                                "growth_sources_available": len(
                                    skeleton_rows[k].get(p, ())
                                ),
                                **{
                                    key: value
                                    for key, value in growth_stats.items()
                                    if key
                                    not in {
                                        "monomer_p_m",
                                        "continuous_decoration",
                                    }
                                },
                            },
                        )
                    )
        else:
            reporter.emit(
                f"k={k} -> k={k + 1}: growing merged inorganic skeletons "
                f"(packages={list(packages)})"
            )
            for p, sources in sorted(growth_sources.items()):
                for p_m in packages:
                    # Building-block bare path: all packages + shed via one helper.
                    if (
                        packages != (0,)
                        or p > 0
                        or spec.k_growth_max_shed > 0
                        or float(spec.p_surf_beta) > 0.0
                    ):
                        raw_by_p, attempted, growth_stats = (
                            _bare_package_core_children(
                                sources,
                                k_from=k,
                                p=p,
                                model=model,
                                spec=spec,
                                p_m=p_m,
                            )
                        )
                        total_unique = 0
                        for p_to, raw_children in sorted(raw_by_p.items()):
                            children = _unique_skeleton_candidates(
                                raw_children,
                                model,
                                spec,
                                reporter,
                                context=(
                                    f"k={k}->{k + 1} p={p}+pm{p_m}->p0={p_to} "
                                    "bare package merge"
                                ),
                            )
                            if children:
                                next_initial.setdefault(p_to, []).extend(
                                    children
                                )
                            total_unique += len(children)
                        policy_pruned = int(
                            growth_stats.get("core_growth_policy_pruned", 0)
                        )
                        reporter.emit(
                            f"k={k} -> k={k + 1} p={p} + p_m={p_m} (bare): "
                            f"raw={attempted}, "
                            f"connected={growth_stats.get('core_growth_raw_connected', 0)}, "
                            f"after_policy={growth_stats.get('core_growth_after_policy', 0)}, "
                            f"policy_pruned={policy_pruned}, "
                            f"unique_total={total_unique}, "
                            f"p0_targets={sorted(raw_by_p)}"
                        )
                        if policy_pruned:
                            narrowing_audit["core_growth_policy_pruned"] = (
                                int(
                                    narrowing_audit.get(
                                        "core_growth_policy_pruned", 0
                                    )
                                )
                                + policy_pruned
                            )
                        growth_audits.append(
                            SweepAudit(
                                k=k,
                                operation="core_skeleton_growth",
                                p_from=p,
                                p_to=p,
                                source_count=len(sources),
                                raw_count=attempted,
                                valid_count=total_unique,
                                symmetry_duplicate_count=0,
                                stage_counts={
                                    "ligand_enumerations": 0,
                                    "growth_narrowed_to_retained_cores": int(
                                        narrowed
                                    ),
                                    "growth_occupation": "bare",
                                    "monomer_p_m": int(p_m),
                                    "growth_sources_available": len(
                                        skeleton_rows[k].get(p, ())
                                    ),
                                    **{
                                        key: value
                                        for key, value in growth_stats.items()
                                        if key != "monomer_p_m"
                                    },
                                },
                            )
                        )
                    else:
                        # Historical path: p_m=0, no shed, inject at same p.
                        raw_children, attempted, growth_stats = (
                            _core_skeleton_children(
                                sources,
                                k_from=k,
                                p=p,
                                model=model,
                                spec=spec,
                            )
                        )
                        children = _unique_skeleton_candidates(
                            raw_children,
                            model,
                            spec,
                            reporter,
                            context=f"k={k}->{k + 1} p={p} core merge",
                        )
                        if children:
                            next_initial.setdefault(p, []).extend(children)
                        policy_pruned = int(
                            growth_stats.get("core_growth_policy_pruned", 0)
                        )
                        reporter.emit(
                            f"k={k} -> k={k + 1} p={p}: "
                            f"raw_skeletons={attempted}, "
                            f"connected={growth_stats.get('core_growth_raw_connected', 0)}, "
                            f"after_policy={growth_stats.get('core_growth_after_policy', 0)}, "
                            f"policy_pruned={policy_pruned}, "
                            f"unique={len(children)}, "
                            f"duplicates={max(0, len(raw_children) - len(children))}"
                        )
                        if policy_pruned:
                            narrowing_audit["core_growth_policy_pruned"] = (
                                int(
                                    narrowing_audit.get(
                                        "core_growth_policy_pruned", 0
                                    )
                                )
                                + policy_pruned
                            )
                        growth_audits.append(
                            SweepAudit(
                                k=k,
                                operation="core_skeleton_growth",
                                p_from=p,
                                p_to=p,
                                source_count=len(sources),
                                raw_count=attempted,
                                valid_count=len(children),
                                symmetry_duplicate_count=max(
                                    0, len(raw_children) - len(children)
                                ),
                                stage_counts={
                                    "ligand_enumerations": 0,
                                    "growth_narrowed_to_retained_cores": int(
                                        narrowed
                                    ),
                                    "growth_occupation": "bare",
                                    "growth_sources_available": len(
                                        skeleton_rows[k].get(p, ())
                                    ),
                                    **growth_stats,
                                },
                            )
                        )
        # Deduplicate per p after multi-source extends.
        for p_to, pool in list(next_initial.items()):
            next_initial[p_to] = _unique_skeleton_candidates(
                pool,
                model,
                spec,
                reporter,
                context=f"k={k}->{k + 1} p'={p_to} final merge",
            )
        for p_to, pool in list(next_decorated.items()):
            next_decorated[p_to] = _unique_decorated_with_routes(
                pool,
                progress=reporter,
                context=f"k={k}->{k + 1} p'={p_to} continuous final merge",
            )
        result.sweep_audit.extend(growth_audits)
        if not next_initial and not next_decorated:
            reporter.emit(f"k={k} -> k={k + 1}: core growth closed")
            break
        (
            next_retained,
            next_discarded,
            next_skeletons,
            next_audits,
        ) = _complete_k_dag(
            k=k + 1,
            inherited=next_initial,
            model=model,
            spec=spec,
            progress=reporter,
            cache=enumeration_cache,
            checkpoint_dir=checkpoint_dir,
            inherited_decorated=next_decorated if next_decorated else None,
        )
        skeleton_rows[k + 1] = next_skeletons
        result.registry[k + 1] = {
            p: recs for p, recs in next_retained.items() if recs
        }
        result.discarded_counts[k + 1] = {
            p: len(records) for p, records in next_discarded.items()
        }
        if k + 1 <= spec.discarded_through_k:
            result.discarded_registry[k + 1] = next_discarded
        result.sweep_audit.extend(next_audits)
        for audit in next_audits:
            dropped = int(
                audit.stage_counts.get("p_skeleton_beam_dropped", 0)
            )
            if dropped:
                narrowing_audit["p_beam_dropped"] = (
                    int(narrowing_audit.get("p_beam_dropped", 0)) + dropped
                )
        if checkpoint_dir is not None:
            reporter.emit(
                f"checkpoint: k={k + 1} complete under {checkpoint_dir}"
            )
    result.completeness = _completeness_report(
        spec, result, narrowing_audit, reporter
    )
    # Drop empty retained bins (p-ladder may visit stoichiometries with no winner).
    _prune_empty_retained_bins(result.registry)
    reporter.emit("surface geometry: projecting retained structures")
    _precondition_retained_registry(
        result.registry,
        model,
        spec,
        reporter,
        discarded_counts=result.discarded_counts,
    )
    retained_total = sum(
        len(records)
        for bins in result.registry.values()
        for records in bins.values()
    )
    discarded_total = sum(
        count
        for bins in result.discarded_counts.values()
        for count in bins.values()
    )
    empty_bins = sum(
        1
        for k_key, bins in result.discarded_counts.items()
        for p_key, count in bins.items()
        if count > 0 and not result.registry.get(k_key, {}).get(p_key)
    )
    reporter.emit(
        f"map complete: retained={retained_total}, "
        f"discarded={discarded_total}"
        + (
            f", bins_with_discarded_only={empty_bins}"
            if empty_bins
            else ""
        )
    )
    return result


def _prune_empty_retained_bins(registry: NucleationRegistry) -> None:
    """Remove (k, p) keys that hold no retained structures."""

    for k in list(registry):
        registry[k] = {
            p: records for p, records in registry[k].items() if records
        }
        if not registry[k]:
            del registry[k]


def _precursor_skeleton_children(
    sources: Sequence[Tuple[_State, Tuple[str, ...]]],
    *,
    k: int,
    p: int,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> Tuple[List[Tuple[_State, Tuple[str, ...]]], int]:
    """Add one precursor center without constructing any ligand shell."""

    children: List[Tuple[_State, Tuple[str, ...]]] = []
    attempted = 0
    for source_index, (source, _source_routes) in enumerate(sources, start=1):
        if _se_coordination_capacity(source, spec) < 1:
            continue
        for site in _cation_vacancies_on_anions(source, model, spec):
            attempted += 1
            atoms = list(source.atoms)
            atoms.append(
                AtomRecord(
                    len(atoms),
                    spec.precursor.center,
                    tuple(float(value) for value in site.position),
                    "precursor_center",
                    p + 1,
                )
            )
            child = _make_core_graph(tuple(atoms), model, spec)
            if not _base_coordination_valid(child, model, spec):
                continue
            route = f"k{k:03d}_p{p:03d}_add_p_source{source_index:04d}"
            children.append((child, (route,)))
    return children, attempted


def _completeness_report(
    spec: NucleationSpec,
    result: NucleationResult,
    narrowing: Mapping[str, object],
    progress: _ProgressReporter,
) -> Dict[str, object]:
    """State exactly what the run enumerated, and what it did not.

    Independent approximations can be active.  ``mode="guided"`` places one
    ligand shell per skeleton and so never enumerates isomers at all.
    ``exact_through_k`` narrows growth above some ``k`` to the cores of retained
    structures.  ``core_growth_policy`` drops open/branched monomer attachments
    when growing into ``k >= compact_from_k``.  Once a branch is cut, later work
    cannot recover it.  A consumer must be able to see that without inferring it
    from the absence of a warning.
    """

    guided = spec.mode == "guided"
    highest_k = max(result.registry, default=0)
    # The narrowing rule being switched on is not itself a loss; only a step that
    # actually dropped skeletons can have lost a structure.
    narrowed_from = narrowing.get("first_binding_k")
    core_growth_pruned = int(narrowing.get("core_growth_policy_pruned", 0))
    core_growth_first = narrowing.get("core_growth_first_k")

    guarantees: List[str] = []
    approximations: List[Dict[str, object]] = []

    if guided:
        multi = (
            int(spec.shells_per_skeleton) > 1
            or int(spec.shell_score_layers) > 1
            or int(spec.shells_per_score_layer) > 0
        )
        if multi:
            approximations.append({
                "stage": "ligand_placement",
                "method": "guided_multi_shell_score_band",
                "applies_from_k": 1,
                "shells_per_skeleton": int(spec.shells_per_skeleton),
                "shell_score_layers": int(spec.shell_score_layers),
                "shells_per_score_layer": int(spec.shells_per_score_layer),
                "shell_enum_max_assignments": int(
                    spec.shell_enum_max_assignments
                ),
                "effect": (
                    "per skeleton: greedy passivation shell plus orbit "
                    "enumeration when C(sites,ligands) <= "
                    "shell_enum_max_assignments; walk top shell_score_layers "
                    "distinct scores (e.g. 20,19,18); keep "
                    "shells_per_score_layer winners per score "
                    "(0=whole layer); hard cap shells_per_skeleton. "
                    "Not full exact enumeration when assignment space is huge."
                ),
                "completeness": "not_guaranteed",
            })
        else:
            approximations.append({
                "stage": "ligand_placement",
                "method": "guided_passivation_order",
                "applies_from_k": 1,
                "effect": (
                    "one ligand shell per skeleton; symmetry-distinct ligand "
                    "arrangements on the same skeleton are never generated"
                ),
                "completeness": "not_guaranteed",
            })
    else:
        guarantees.append(
            "all symmetry-distinct ligand arrangements enumerated per bin"
        )
        guarantees.append("dominance and bound pruning provably safe")

    if narrowed_from is not None:
        approximations.append({
            "stage": "skeleton_growth",
            "method": "retained_cores_only",
            "applies_from_k": int(narrowed_from),
            "skeletons_available": int(narrowing.get("skeletons_before", 0)),
            "cores_propagated": int(narrowing.get("cores_after", 0)),
            "effect": (
                "skeleton branches whose structures were not retained are cut; "
                "a structure reachable only through such a branch is lost and "
                "cannot be recovered at higher k"
            ),
            "completeness": "not_guaranteed",
        })
    elif narrowing.get("skeletons_before"):
        # The rule was active but never dropped anything, so nothing was lost.
        guarantees.append(
            "retained-core growth was active from k="
            f"{spec.exact_through_k} but dropped no skeleton, so no branch "
            "was cut"
        )
    elif spec.exact_through_k > highest_k:
        guarantees.append(
            f"every unique skeleton propagated (exact_through_k="
            f"{spec.exact_through_k} was never reached)"
        )

    if core_growth_pruned > 0 and core_growth_first is not None:
        policy_name = str(
            narrowing.get("core_growth_policy", spec.core_growth_policy)
        )
        approximations.append({
            "stage": "core_monomer_growth",
            "method": policy_name,
            "applies_from_k": int(core_growth_first),
            "compact_from_k": int(
                narrowing.get("compact_from_k", spec.compact_from_k)
            ),
            "children_pruned": core_growth_pruned,
            "effect": (
                "per parent, only steepest-ascent compact monomer attachments "
                "are kept"
                + (
                    " and, when available, those that close a new 6-ring"
                    if policy_name == "compact_ring"
                    else ""
                )
                + "; open/branched lattice animals reachable only through a "
                "lower-Δbond (or non-ring) attachment are lost"
            ),
            "completeness": "not_guaranteed",
        })
    elif (
        spec.core_growth_policy != "all"
        and highest_k >= spec.compact_from_k
    ):
        guarantees.append(
            f"core_growth_policy={spec.core_growth_policy} was configured from "
            f"k>={spec.compact_from_k} but pruned no children"
        )
    elif spec.core_growth_policy == "all" or highest_k < spec.compact_from_k:
        guarantees.append(
            "every connected core-monomer placement kept "
            f"(core_growth_policy={spec.core_growth_policy}, "
            f"compact_from_k={spec.compact_from_k})"
        )

    if spec.passivation_ring_policy != "none":
        approximations.append({
            "stage": "passivation_ring_selection",
            "method": spec.passivation_ring_policy,
            "ring_lengths": list(spec.ring_lengths),
            "effect": (
                "within the winning coordination/surface layer, structures "
                "are ranked or filtered by Cl-containing rings of the listed "
                "lengths; ligand placement is unchanged and Cl 4-rings from "
                "bridges still require min_bridged_host_cn on both hosts"
            ),
            "completeness": "not_guaranteed",
        })
    else:
        guarantees.append(
            "passivation_ring_policy=none (ring counts reported only)"
        )

    beam_dropped_total = int(narrowing.get("p_beam_dropped", 0))
    if beam_dropped_total > 0 or (
        spec.p_skeleton_beam > 0 and highest_k >= spec.p_beam_from_k
    ):
        if beam_dropped_total > 0:
            approximations.append({
                "stage": "p_skeleton_beam",
                "method": spec.p_beam_rank,
                "beam_width": spec.p_skeleton_beam,
                "applies_from_k": spec.p_beam_from_k,
                "skeletons_dropped": beam_dropped_total,
                "effect": (
                    "at each (k,p) with k>=p_beam_from_k, only the top "
                    "p_skeleton_beam unique skeletons (by inorganic ring/bond "
                    "rank) receive ligand placement and grow to p+1"
                ),
                "completeness": "not_guaranteed",
            })
        else:
            guarantees.append(
                f"p_skeleton_beam={spec.p_skeleton_beam} from k>="
                f"{spec.p_beam_from_k} never dropped a skeleton"
            )
    elif spec.p_skeleton_beam <= 0:
        guarantees.append("p_skeleton_beam off (unlimited skeletons per p)")

    if (
        spec.fused_chair_mode != "off"
        and spec.fused_chair_from_k > 0
        and highest_k >= spec.fused_chair_from_k
    ):
        approximations.append({
            "stage": "fused_chair_growth",
            "method": spec.fused_chair_mode,
            "applies_from_k": spec.fused_chair_from_k,
            "effect": (
                "on k→k+1 core growth, after compact_ring/max_bonds, "
                "children are ranked or filtered by edge-fused 6-ring pairs"
            ),
            "completeness": "not_guaranteed",
        })
    elif spec.fused_chair_mode == "off":
        guarantees.append("fused_chair_mode=off")

    if spec.retain_score_layers > 1 or spec.retain_max_per_bin > 0:
        approximations.append({
            "stage": "soft_retain_band",
            "method": "score_layers_and_cap",
            "retain_score_layers": spec.retain_score_layers,
            "retain_max_per_bin": spec.retain_max_per_bin,
            "effect": (
                "more than the single top coordination-score layer may be "
                "retained per bin (lineage band), optionally capped by "
                "retain_max_per_bin; winners still rank first"
            ),
            "completeness": "not_guaranteed",
        })
    else:
        guarantees.append(
            "retain_score_layers=1 (single top coordination layer per bin)"
        )

    if spec.core_growth_occupation == "decorated" and narrowing.get(
        "decorated_growth"
    ):
        approximations.append({
            "stage": "core_growth_occupation",
            "method": "decorated",
            "effect": (
                "k→k+1 monomer attachment uses passivated parents; ligands "
                "occupy tetrahedral sites and block those directions. Child "
                "skeletons are still stripped for the p-DAG."
            ),
            "completeness": "not_guaranteed",
        })
    elif spec.core_growth_occupation == "decorated":
        guarantees.append(
            "core_growth_occupation=decorated configured but not applied "
            "(exact skeleton growth still active)"
        )
    else:
        guarantees.append(
            "core_growth_occupation=bare (ligands stripped before k-growth)"
        )

    packages = list(_monomer_packages(spec))
    if packages != [0] or spec.monomer_p_values:
        approximations.append({
            "stage": "monomer_packages",
            "method": "building_block",
            "monomer_p_values": packages,
            "effect": (
                "each k→k+1 step attaches a monomer package with precursor "
                "count p_m from this list; product p0 = p_parent - shed + p_m"
            ),
            "completeness": "not_guaranteed",
        })
    else:
        guarantees.append(
            "monomer package p_m=0 only (bare core add; p conserved except shed)"
        )

    if narrowing.get("seed_p_filtered") or spec.parent_p_mode == "seed_band":
        approximations.append({
            "stage": "parent_p_filter",
            "method": spec.parent_p_mode,
            "seed_p": narrowing.get("seed_p", spec.seed_p),
            "seed_p_window": narrowing.get(
                "seed_p_window", spec.seed_p_window
            ),
            "effect": (
                "only parent p-bins near seed_p feed k→k+1 growth "
                "(legacy seed_band mode)"
            ),
            "completeness": "not_guaranteed",
        })
    else:
        guarantees.append(
            "parent_p_mode=all_retained "
            "(every retained p-bin may feed core growth)"
        )

    capacity_only_shedding = (
        float(spec.p_surf_beta) <= 0.0
        and int(spec.k_growth_max_shed) <= 0
    )
    if capacity_only_shedding:
        if highest_k >= 2:
            guarantees.append(
                "Se-capacity-only shedding (all complete package shed counts "
                "are allowed; child p is limited by remaining Se "
                "coordination slots)"
            )
        else:
            # Preserve the compact seed-row report: no k-growth edge has been
            # traversed yet, so shedding semantics have not affected output.
            guarantees.append(
                "Δp unrestricted (no shed variants; full p-ladder at each k)"
            )
    elif (
        spec.k_growth_max_shed > 0
        or float(spec.p_surf_beta) > 0.0
        or spec.k_growth_max_add >= 0
        or spec.p_ladder_mode == "product_window"
    ):
        if float(spec.p_surf_beta) > 0.0:
            delta_effect = (
                "scenario A surface bounds: "
                "p_surf(k)=floor(β k^(2/3)); "
                "s_max=min(p, floor(α p_surf(k))); "
                "channel p_child≤min(p+p_m, p_surf(k+1)) "
                "(residual shell p−s kept; re-adsorb ≤ s+p_m); "
                "ladder p_cap≤p_surf(k) (not CN ceiling 3k)"
            )
        else:
            delta_effect = (
                "shed before attach and/or limit the destination p-ladder: "
                "inherited_plus uses max(inherited)+max_add; product_window "
                "uses [min(p0)-max_shed, max(p0)+max_add]"
            )
        approximations.append({
            "stage": "delta_p_window",
            "method": (
                "surface_inventory"
                if float(spec.p_surf_beta) > 0.0
                else spec.p_ladder_mode
            ),
            "k_growth_max_shed": spec.k_growth_max_shed,
            "k_growth_max_add": spec.k_growth_max_add,
            "p_surf_beta": float(spec.p_surf_beta),
            "shed_alpha": float(spec.shed_alpha),
            "effect": delta_effect,
            "completeness": "not_guaranteed",
        })
    else:
        guarantees.append(
            "Δp unrestricted (no fixed shedding or surface cap; full p-ladder "
            "subject to Se capacity)"
        )

    exact_through = highest_k
    if narrowed_from is not None:
        exact_through = min(exact_through, int(narrowed_from))
    if core_growth_pruned > 0 and core_growth_first is not None:
        # Completeness ends at the last fully unrestricted destination k.
        exact_through = min(exact_through, int(core_growth_first) - 1)
    if guided:
        exact_through = 0
    if spec.passivation_ring_policy != "none":
        exact_through = 0
    if beam_dropped_total > 0:
        exact_through = min(
            exact_through, max(0, spec.p_beam_from_k - 1)
        )
    if (
        spec.fused_chair_mode != "off"
        and highest_k >= spec.fused_chair_from_k > 0
    ):
        exact_through = min(
            exact_through, max(0, spec.fused_chair_from_k - 1)
        )
    if spec.retain_score_layers > 1 or spec.retain_max_per_bin > 0:
        exact_through = 0
    if narrowing.get("decorated_growth") or narrowing.get("seed_p_filtered"):
        exact_through = 0
    if (
        spec.k_growth_max_shed > 0
        or float(spec.p_surf_beta) > 0.0
        or (spec.k_growth_max_add >= 0 and highest_k >= 2)
    ):
        exact_through = 0

    report: Dict[str, object] = {
        "mode": spec.mode,
        "exact_through_k": spec.exact_through_k,
        "core_growth_policy": spec.core_growth_policy,
        "compact_from_k": spec.compact_from_k,
        "passivation_ring_policy": spec.passivation_ring_policy,
        "ring_lengths": list(spec.ring_lengths),
        "p_skeleton_beam": spec.p_skeleton_beam,
        "p_beam_from_k": spec.p_beam_from_k,
        "p_beam_rank": spec.p_beam_rank,
        "fused_chair_from_k": spec.fused_chair_from_k,
        "fused_chair_mode": spec.fused_chair_mode,
        "retain_score_layers": spec.retain_score_layers,
        "retain_max_per_bin": spec.retain_max_per_bin,
        "growth_max_parents_per_bin": spec.growth_max_parents_per_bin,
        "core_growth_occupation": spec.core_growth_occupation,
        "continuous_decoration": bool(spec.continuous_decoration),
        "monomer_p_values": list(_monomer_packages(spec)),
        "seed_p": spec.seed_p,
        "seed_p_window": spec.seed_p_window,
        "parent_p_mode": spec.parent_p_mode,
        "p_ladder_mode": spec.p_ladder_mode,
        "k_growth_max_shed": spec.k_growth_max_shed,
        "k_growth_max_add": spec.k_growth_max_add,
        "p_surf_beta": float(spec.p_surf_beta),
        "shed_alpha": float(spec.shed_alpha),
        "kmax_reached": highest_k,
        "enumeration_complete_through_k": exact_through,
        "guarantees": guarantees,
        "approximations": approximations,
        "bridge_scope": "lexicographic_score_optimum",
        "discarded_counts_semantics": (
            "exact for k<=2; a LOWER BOUND for k>2, where bases provably unable "
            "to win their bin are pruned before becoming records"
        ),
    }

    if approximations:
        progress.emit(
            "WARNING: enumeration is NOT complete above k="
            f"{exact_through} -- "
            + "; ".join(
                f"{item['stage']}={item['method']}" for item in approximations
            )
            + ". See the completeness block in registry.json."
        )
    else:
        progress.emit(
            f"enumeration complete through k={exact_through} "
            "(no approximations active)"
        )
    return report


def _select_shells_by_score_band(
    scored: Sequence[Tuple[Tuple[int, ...], _State]],
    *,
    score_layers: int,
    max_shells: int,
    per_score_layer: int = 0,
) -> List[_State]:
    """Walk top coordination scores and keep winners per layer.

    ``scored`` entries are ``(coordination_score, state)``; higher scores win.

    * ``score_layers`` -- how many **distinct** scores to visit (best, then
      next, … e.g. 20 then 19 then 18).
    * ``per_score_layer`` -- how many shells to take **within** each of those
      scores.  ``0`` = entire layer; ``1`` = only the top shell at that score
      (stable tie-break by graph hash / coordinates).
    * ``max_shells`` -- hard cap on the total list after layer walks.
    """

    if not scored:
        return []
    layers_wanted = max(1, int(score_layers))
    cap = max(1, int(max_shells))
    per_layer = int(per_score_layer)
    unique_scores = sorted({score for score, _state in scored}, reverse=True)
    kept: List[Tuple[Tuple[int, ...], _State]] = []
    for score in unique_scores[:layers_wanted]:
        layer = [(sc, st) for sc, st in scored if sc == score]
        # Prefer higher score already fixed; within layer, stable secondary key.
        # (All share the same coordination score, so hash/coords only order ties.)
        layer.sort(
            key=lambda item: (
                _graph_hash(item[1].graph),
                tuple(atom.coordinates for atom in item[1].atoms),
            )
        )
        if per_layer > 0:
            layer = layer[:per_layer]
        kept.extend(layer)
    # Preserve layer order (best score first); do not re-sort away the band.
    return [state for _score, state in kept[:cap]]


def _guided_skeleton_bin(
    skeletons: Sequence[Tuple[_State, Tuple[str, ...]]],
    *,
    k: int,
    p: int,
    model: _LatticeModel,
    spec: NucleationSpec,
    progress: _ProgressReporter,
    cache: _EnumerationCache,
) -> _Generation:
    """Build guided ligand shell(s) per skeleton (optional multi-shell band).

    Default (``shells_per_skeleton=1``, ``shell_score_layers=1``): one
    passivation-ordered shell per skeleton plus latent bridges -- historical
    guided.

    With ``shells_per_skeleton>1`` or ``shell_score_layers>1``: also orbit-
    enumerate ligand placements on that skeleton when the theoretical
    assignment count is at most ``shell_enum_max_assignments``, score shells by
    coordination, keep the top score layers and cap at ``shells_per_skeleton``.
    Same skeleton, different passivation isomers.  Not full-map exact mode.
    """

    records: List[ClusterRecord] = []
    invalid: Dict[str, int] = {}
    shells_cap = max(1, int(spec.shells_per_skeleton))
    shell_layers = max(1, int(spec.shell_score_layers))
    per_layer = max(0, int(spec.shells_per_score_layer))
    multi_shell = shells_cap > 1 or shell_layers > 1 or per_layer > 0
    stages: Dict[str, int] = {
        "guided_skeletons": len(skeletons),
        "shells_per_skeleton": shells_cap,
        "shell_score_layers": shell_layers,
        "shells_per_score_layer": per_layer,
        "guided_multi_shell": int(multi_shell),
    }
    accumulator = _CandidateAccumulator()
    started = time.monotonic()
    if multi_shell:
        progress.emit(
            f"k={k} p={p}: GUIDED multi-shell placement, "
            f"skeletons={len(skeletons)}, "
            f"shells_per_skeleton={shells_cap}, "
            f"shell_score_layers={shell_layers}, "
            f"shells_per_score_layer={per_layer}"
        )
    else:
        progress.emit(
            f"k={k} p={p}: GUIDED ligand placement, skeletons={len(skeletons)} "
            "(one shell per skeleton; isomers are not enumerated)"
        )

    ligand_count = spec.precursor.ligand_count * p
    dead_ends = 0
    multi_enum_used = 0
    multi_enum_skipped_cap = 0
    shells_selected = 0
    for index, (skeleton, routes) in enumerate(skeletons, start=1):
        bases: List[_State] = []
        seeded = _greedy_incumbent_state(skeleton, ligand_count, model, spec)
        if seeded is None:
            dead_ends += 1
            _increment(invalid, "guided_placement_dead_end")
            continue
        bases.append(seeded)

        if multi_shell and ligand_count > 0:
            sites = _all_outward_ligand_sites(skeleton, model, spec)
            assignment_count = (
                math.comb(len(sites), ligand_count)
                if len(sites) >= ligand_count
                else 0
            )
            max_assign = max(0, int(spec.shell_enum_max_assignments))
            if assignment_count == 0:
                _increment(invalid, "guided_multi_shell_no_sites")
            elif max_assign > 0 and assignment_count > max_assign:
                multi_enum_skipped_cap += 1
                _increment(invalid, "guided_multi_shell_assignment_cap")
            else:
                multi_enum_used += 1
                states, _attempted, reasons, _stage = _enumerate_ligand_states(
                    skeleton.atoms,
                    ligand_count,
                    model,
                    spec,
                    progress,
                    cache,
                    context=(
                        f"k={k} p={p} guided multi-shell "
                        f"skeleton={index}/{len(skeletons)}"
                    ),
                )
                _merge_reason_counts(invalid, reasons)
                bases.extend(states)

        # Symmetry-merge bases on this skeleton, expand latent bridges, score.
        local_acc = _CandidateAccumulator(model, spec, comparison="bridges")
        for base in bases:
            for variant in _latent_bridge_variants(
                base, model, spec, prune_dominated=True, cache=cache
            ):
                local_acc.add(variant, routes)
        scored: List[Tuple[Tuple[int, ...], _State]] = []
        for state, _routes in local_acc.result():
            score = _graph_coordination_score(state.atoms, state.graph, spec)
            scored.append((score, state))
        selected = _select_shells_by_score_band(
            scored,
            score_layers=shell_layers,
            max_shells=shells_cap,
            per_score_layer=per_layer,
        )
        shells_selected += len(selected)
        for state in selected:
            accumulator.add(state, routes)
        progress.heartbeat(
            f"k={k} p={p}: guided skeleton={index}/{len(skeletons)}, "
            f"bases={len(bases)}, selected_shells={len(selected)}, "
            f"classes={len(accumulator.classes)}, dead_ends={dead_ends}, "
            f"elapsed={time.monotonic() - started:.1f}s"
        )

    for state, routes in accumulator.result():
        records.append(
            _record_from_state(
                state, k=k, p=p, spec=spec,
                operation="guided_placement", source_ids=routes,
            )
        )
    stages["guided_dead_ends"] = dead_ends
    stages["guided_records"] = len(records)
    stages["guided_multi_shell_enum_skeletons"] = multi_enum_used
    stages["guided_multi_shell_cap_skipped"] = multi_enum_skipped_cap
    stages["guided_shells_selected"] = shells_selected
    progress.emit(
        f"k={k} p={p}: guided placement complete, records={len(records)}, "
        f"dead_ends={dead_ends}, multi_shell_enum={multi_enum_used}, "
        f"cap_skipped={multi_enum_skipped_cap}, "
        f"elapsed={time.monotonic() - started:.1f}s"
    )
    return _Generation(records, len(skeletons), invalid, stages)


def _bridge_opportunity_graph(
    state: _State,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> nx.Graph:
    """Augment a decorated base with the bridge options its geometry allows.

    Bases used to be merged on graph isomorphism alone, which silently discarded
    structures: a ``shared_vacant_cif_site`` bridge needs two cations to share a
    vacant anion site, and that is a property of the *coordinates*.  Two bases
    can therefore be graph-isomorphic while offering different bridges, and
    keeping one representative threw the other's options away -- measured at
    k=2 p=1, three classes held members with CIF-site arc counts [0,2], [0,1],
    [0,1] and in every case the kept representative had none.

    Adding one labelled node per opportunity, joined to its donor, acceptor and
    shared anion, folds that geometry into the certificate.  Same technique as
    :func:`_skeleton_frontier_graph`, which augments a skeleton with its growth
    sites for exactly the same reason.
    """

    graph = state.graph.copy()
    _terminal, arcs = _bridge_candidate_arcs(state, model, spec)
    next_node = len(state.atoms)
    for arc in sorted(
        arcs,
        key=lambda item: (
            item.mode,
            item.primary,
            item.host,
            item.shared_neighbor if item.shared_neighbor is not None else -1,
            item.virtual_site or (),
        ),
    ):
        graph.add_node(
            next_node,
            element=f"bridge_option_{arc.mode}",
            role="bridge_option",
        )
        # Distinct bond_order strings colour the roles apart; the fingerprint
        # reads bond_order, so donor/acceptor/shared stay distinguishable.
        graph.add_edge(next_node, arc.primary, bond_order="bridge_donor")
        graph.add_edge(next_node, arc.host, bond_order="bridge_acceptor")
        if arc.shared_neighbor is not None:
            graph.add_edge(
                next_node, arc.shared_neighbor, bond_order="bridge_shared"
            )
        next_node += 1
    return graph


def _retained_core_sources(
    records: Sequence[ClusterRecord],
    model: _LatticeModel,
    spec: NucleationSpec,
) -> List[Tuple[_State, Tuple[str, ...]]]:
    """Reduce retained records to the deduplicated skeletons underneath them.

    A record's skeleton is its atoms with the ligand species removed; growth to
    ``k+1`` then proceeds from those cores instead of from every distinct
    skeleton in the row.  Several retained records usually share one core -- 14
    records collapsed onto 6 cores at k=2 and 9 at k=3 -- so this is where the
    reduction comes from.

    Deduplication uses the frontier-augmented certificate, the same equivalence
    the DAG already uses to merge skeletons reached by different routes, so two
    cores are only merged when their growth options agree as well as their
    topology.

    Lineage is carried through: each core keeps the ``structure_id``s of the
    retained records that produced it, so a survivor at high ``k`` can still be
    traced back through the narrowing.
    """

    by_certificate: Dict[Tuple[object, ...], Tuple[_State, set[str]]] = {}
    for record in records:
        core = _make_core_graph(
            _without_ligands(record.atoms, spec), model, spec
        )
        if not core.atoms or not nx.is_connected(core.graph):
            continue
        certificate = _graph_certificate(
            _skeleton_frontier_graph(core, model, spec)
        )
        route = record.structure_id or ""
        existing = by_certificate.get(certificate)
        if existing is None:
            by_certificate[certificate] = (core, {route} if route else set())
            continue
        kept, routes = existing
        if route:
            routes.add(route)
        # Keep the lexicographically first core so the choice cannot depend on
        # the order records happen to arrive in.
        if _CandidateAccumulator._state_key(
            core
        ) < _CandidateAccumulator._state_key(kept):
            by_certificate[certificate] = (core, routes)
    return [
        (core, tuple(sorted(routes)))
        for core, routes in by_certificate.values()
    ]


def _p_beam_sort_key(
    state: _State, rank: str, *, ring_length: int = 6
) -> Tuple[int, ...]:
    """Higher is better for p-skeleton beam ranking."""

    bonds, rings, fused = _skeleton_ring_metrics(state, ring_length)
    if rank == "bonds":
        return (bonds, rings, fused)
    if rank == "six_rings":
        return (rings, fused, bonds)
    # fused_rings (default)
    return (fused, rings, bonds)


def _apply_p_skeleton_beam(
    skeletons: Sequence[Tuple[_State, Tuple[str, ...]]],
    *,
    k: int,
    p: int,
    spec: NucleationSpec,
    progress: _ProgressReporter,
) -> Tuple[List[Tuple[_State, Tuple[str, ...]]], int]:
    """Keep top-B skeletons for ligand work and p→p+1 growth."""

    beam = spec.p_skeleton_beam
    if (
        beam <= 0
        or k < spec.p_beam_from_k
        or len(skeletons) <= beam
    ):
        return list(skeletons), 0
    ring_length = int(spec.inorganic_ring_length)
    ranked = sorted(
        skeletons,
        key=lambda item: (
            _p_beam_sort_key(
                item[0], spec.p_beam_rank, ring_length=ring_length
            ),
            _graph_hash(item[0].graph),
        ),
        reverse=True,
    )
    kept = ranked[:beam]
    dropped = len(skeletons) - len(kept)
    progress.emit(
        f"k={k} p={p}: p-skeleton beam={beam} rank={spec.p_beam_rank}: "
        f"skeletons {len(skeletons)} -> {len(kept)} (dropped {dropped})"
    )
    return kept, dropped


def _ring_counts_for_record(
    record: ClusterRecord,
    spec: NucleationSpec,
) -> Dict[str, object]:
    """Count simple cycles, including Cl-containing and inorganic Cd–Se rings.

    Bridge formation (and thus Cl 4-rings) remains gated by max_cn and
    ``min_bridged_host_cn`` during latent bridge search; this only *reports*
    and optionally *ranks* finished legal graphs.

    ``inorganic_rings_by_length`` counts cycles on the ligand-free core graph
    (Cd–Se only).  That is the right quantity for open vs closed *core*
    topology: a structure can be labelled open only when this 6-count is low.
    """

    ligand = spec.precursor.ligand
    lengths = tuple(spec.ring_lengths) if spec.ring_lengths else (4, 6)
    ligand_nodes = {
        atom.atom_id
        for atom in record.atoms
        if atom.symbol == ligand or atom.role == "precursor_ligand"
    }
    by_length: Dict[str, int] = {}
    cl_by_length: Dict[str, int] = {}
    for length in lengths:
        total, with_cl = _count_cycles_on_graph(
            record.graph, length, ligand_nodes=ligand_nodes
        )
        by_length[str(length)] = total
        cl_by_length[str(length)] = with_cl

    # Inorganic (Cd–Se) subgraph: drop ligand nodes, keep core + precursor centers.
    inorganic = record.graph.copy()
    inorganic.remove_nodes_from(ligand_nodes)
    inorganic_by_length: Dict[str, int] = {}
    for length in lengths:
        total, _with_cl = _count_cycles_on_graph(inorganic, length)
        inorganic_by_length[str(length)] = total
    # Always report pure 6-ring chair count even if ring_lengths omits 6.
    if "6" not in inorganic_by_length:
        inorganic_by_length["6"] = _count_cycles_on_graph(inorganic, 6)[0]

    cl_ring_total = sum(cl_by_length.values())
    return {
        "ring_lengths": list(lengths),
        "rings_by_length": by_length,
        "cl_rings_by_length": cl_by_length,
        "cl_ring_total": cl_ring_total,
        "inorganic_rings_by_length": inorganic_by_length,
        "inorganic_six_rings": int(inorganic_by_length.get("6", 0)),
        # Lex key for prefer/require policies (higher is better).
        "cl_ring_rank": tuple(
            cl_by_length[str(length)] for length in lengths
        ),
        # Bridges that form Cl 4-rings still obey min_bridged_host_cn.
        "min_bridged_host_cn_note": (
            "Cl 4-rings arise from latent bridges; host CN floor is "
            "graph_rules.bridging.*.min_bridged_host_cn (default 3)"
        ),
    }


def _apply_passivation_ring_policy(
    retained: List[ClusterRecord],
    spec: NucleationSpec,
) -> List[ClusterRecord]:
    """Filter a surface-valid score layer by ligand-ring preference."""

    policy = str(spec.passivation_ring_policy).strip().lower()
    if policy == "none" or len(retained) <= 1:
        return retained
    ranked = [
        (
            tuple(record.metadata.get("rings", {}).get("cl_ring_rank", ())),
            record,
        )
        for record in retained
    ]
    if policy in {"prefer_cl_rings", "prefer_ligand_rings"}:
        best = max(item[0] for item in ranked)
        return [record for key, record in ranked if key == best]
    # require_cl_rings / require_ligand_rings
    if policy in {"require_cl_rings", "require_ligand_rings"}:
        if any(sum(key) > 0 for key, _record in ranked):
            kept = [record for key, record in ranked if sum(key) > 0]
            return kept if kept else retained
    return retained


def _filter_core_children_by_policy(
    parent: _State,
    candidates: Sequence[Tuple[_State, Tuple[str, ...]]],
    policy: str,
    *,
    k_to: int = 0,
    fused_mode: str = "off",
    fused_from_k: int = 0,
    ring_length: int = 6,
) -> List[Tuple[_State, Tuple[str, ...]]]:
    """Apply max_bonds / compact_ring / fused-ring filtering per parent."""

    if policy == "all" or len(candidates) <= 1:
        pool = list(candidates)
    else:
        parent_edges = parent.graph.number_of_edges()
        bond_scored = [
            (child.graph.number_of_edges() - parent_edges, child, routes)
            for child, routes in candidates
        ]
        if not bond_scored:
            return []
        best_bonds = max(item[0] for item in bond_scored)
        bond_winners = [
            (child, routes)
            for delta, child, routes in bond_scored
            if delta == best_bonds
        ]
        if policy == "max_bonds" or len(bond_winners) <= 1:
            pool = bond_winners
        else:
            # compact_ring: among max-bond children, prefer new preferred rings.
            ring_scored = [
                (
                    _new_six_ring_count(parent, child, ring_length),
                    child,
                    routes,
                )
                for child, routes in bond_winners
            ]
            if any(delta > 0 for delta, _child, _routes in ring_scored):
                pool = [
                    (child, routes)
                    for delta, child, routes in ring_scored
                    if delta > 0
                ]
            else:
                pool = bond_winners

    # Optional fused-ring bias on the surviving set (destination k large enough).
    if (
        fused_mode != "off"
        and fused_from_k > 0
        and k_to >= fused_from_k
        and len(pool) > 1
    ):
        scored = [
            (
                _fused_chair_metrics(child.graph, ring_length)[1],
                child,
                routes,
            )
            for child, routes in pool
        ]
        if fused_mode == "prefer_positive":
            if any(fused > 0 for fused, _c, _r in scored):
                pool = [
                    (child, routes)
                    for fused, child, routes in scored
                    if fused > 0
                ]
        elif fused_mode == "rank":
            best_fused = max(fused for fused, _c, _r in scored)
            pool = [
                (child, routes)
                for fused, child, routes in scored
                if fused == best_fused
            ]
    return pool


def _monomer_packages(spec: NucleationSpec) -> Tuple[int, ...]:
    """Resolved p_m values for the added monomer unit at each k→k+1 step.

    Explicit ``monomer_p_values`` win.  Else if ``seed_p`` is set, packages are
    every nonnegative integer in ``[seed_p - window, seed_p + window]``.
    Else only ``p_m = 0`` (historical bare core add).
    """

    if spec.monomer_p_values:
        return tuple(sorted(set(int(v) for v in spec.monomer_p_values)))
    if spec.seed_p is not None:
        lo = max(0, int(spec.seed_p) - int(spec.seed_p_window))
        hi = int(spec.seed_p) + int(spec.seed_p_window)
        return tuple(range(lo, hi + 1))
    return (0,)


def _p_allowed_for_k_growth(p: int, spec: NucleationSpec) -> bool:
    """Whether a parent p-bin may feed k→k+1 growth."""

    if spec.parent_p_mode == "all_retained":
        return True
    # seed_band
    if spec.seed_p is None:
        return True
    return abs(int(p) - int(spec.seed_p)) <= int(spec.seed_p_window)


def _product_p0(parent_p: int, shed: int, p_m: int) -> int:
    """Nominal product precursor count after attach + optional shed."""

    return max(0, int(parent_p) - int(shed) + int(p_m))


def _p_surf(k: int, beta: float) -> int:
    """Quasi-spherical surface excess capacity: floor(β · k^(2/3))."""

    if k <= 0 or beta <= 0.0:
        return 0
    return int(math.floor(float(beta) * (float(k) ** (2.0 / 3.0))))


def _se_coordination_capacity(
    state: _State,
    spec: NucleationSpec,
) -> int:
    """Return the number of remaining Cd--Se coordination slots.

    A precursor CdCl2 package must bind at least one Se atom.  The capacity is
    therefore the sum of the unused Se coordination slots, using the declared
    graph-rule maximum (four for the CdSe maps).  Ligands are deliberately
    ignored: Cl never bonds to Se, and this helper is also used on decorated
    states after their ligand shell is stripped.
    """

    anion = spec.core.anion
    cation_symbols = {spec.core.cation, spec.precursor.center}
    max_cn = int(spec.graph_rules.max_cn[anion])
    capacity = 0
    for atom in state.atoms:
        if atom.symbol != anion:
            continue
        cd_neighbors = sum(
            1
            for neighbor in state.graph.neighbors(atom.atom_id)
            if state.atoms[neighbor].symbol in cation_symbols
        )
        capacity += max(0, max_cn - cd_neighbors)
    return int(capacity)


def _se_capacity_allows(
    state: _State,
    current_p: int,
    requested_p: int,
    spec: NucleationSpec,
) -> bool:
    """Whether ``requested_p`` fits in the Se capacity of ``state``.

    ``state`` is the ligand-free child skeleton before any newly requested
    precursor packages are placed.  Existing packages are already counted in
    ``current_p``; each remaining Se slot can accommodate one additional
    package at minimum.
    """

    return int(requested_p) <= int(current_p) + _se_coordination_capacity(
        state, spec
    )


def _core_formula_k(state: _State, spec: NucleationSpec) -> int:
    """Se (anion) count = nucleation k for a decorated or bare state."""

    anion = spec.core.anion
    return sum(1 for atom in state.atoms if atom.symbol == anion)


def _effective_max_shed(
    k: int,
    p: int,
    spec: NucleationSpec,
) -> int:
    """Max CdCl2-like packages removable at parent size k.

    Scenario A (``p_surf_beta > 0``)::

        s_max = min(p, floor(shed_alpha * p_surf(k)), hard_cap?)

    Without a surface law, a positive ``k_growth_max_shed`` remains an
    explicit fixed cap.  The default zero means no artificial shedding cap:
    all complete packages in the parent may be considered, up to ``p``.
    """

    p = max(0, int(p))
    if p <= 0:
        return 0
    hard = int(spec.k_growth_max_shed)
    beta = float(spec.p_surf_beta)
    if beta > 0.0:
        alpha = max(0.0, float(spec.shed_alpha))
        surface = _p_surf(k, beta)
        s_max = min(p, int(math.floor(alpha * surface)))
        if hard > 0:
            s_max = min(s_max, hard)
        return max(0, s_max)
    if hard > 0:
        return min(p, hard)
    return p


def _effective_p_cap(
    k: int,
    *,
    capacity_cap: int,
    max_inherited: int,
    spec: NucleationSpec,
) -> int:
    """Upper p for the destination-k ladder (redecoration ceiling).

    With ``p_surf_beta > 0`` (scenario A): cap by spherical surface
    ``p_surf(k)`` so the map does not redecorate toward the CN ceiling ``3k``.
    Optional ``k_growth_max_add`` still tightens relative to injected products.
    """

    cap = int(capacity_cap)
    beta = float(spec.p_surf_beta)
    if beta > 0.0:
        cap = min(cap, _p_surf(k, beta))
    if k > 1 and int(spec.k_growth_max_add) >= 0:
        cap = min(cap, int(max_inherited) + int(spec.k_growth_max_add))
    return max(0, cap)


def _channel_p_child_max(
    parent_p: int,
    shed: int,
    p_m: int,
    k_child: int,
    spec: NucleationSpec,
) -> int:
    """Upper product p for one (shed, p_m) channel under scenario A.

    Continuous decoration keeps the **unshed** residual shell ``(p - s)`` and
    may re-adsorb at most the local inventory ``M = s + p_m`` (shed CdCl2 still
    nearby + monomer package).  So::

        p_child ≤ min(p_surf(k+1), (p - s) + (s + p_m)) = min(p_surf, p + p_m)

    Equivalently residual floor ``p0 = p - s + p_m`` is never *lowered* by the
    inventory (that would erase an already-bound shell when s=0).  The inventory
    only limits **climbing above** residual via free-site redecoration::

        p_child ∈ [0, min(p_surf, max(p0, M))] with M = s + p_m
        and typically p0 ≤ p_child ≤ min(p_surf, p + p_m)
    """

    residual = max(0, int(parent_p) - int(shed))
    pool = max(0, int(shed) + int(p_m))  # re-adsorbable local inventory
    # Residual shell + re-adsorbed pool, without double-counting residual.
    inventory_ceiling = residual + pool  # = parent_p + p_m
    beta = float(spec.p_surf_beta)
    if beta <= 0.0:
        return inventory_ceiling
    return min(inventory_ceiling, _p_surf(k_child, beta))


def _unique_decorated_with_routes(
    items: Sequence[Tuple[_State, Tuple[str, ...]]],
    *,
    progress: Optional[_ProgressReporter] = None,
    context: str = "continuous merge",
) -> List[Tuple[_State, Tuple[str, ...]]]:
    """Deduplicate fully decorated states by full-graph certificate; merge routes."""

    by_cert: Dict[Tuple[object, ...], Tuple[_State, set[str]]] = {}
    for state, routes in items:
        cert = _graph_certificate(state.graph)
        existing = by_cert.get(cert)
        if existing is None:
            by_cert[cert] = (state, set(routes))
            continue
        kept, route_set = existing
        route_set.update(routes)
        # Deterministic representative.
        if _CandidateAccumulator._state_key(state) < _CandidateAccumulator._state_key(
            kept
        ):
            by_cert[cert] = (state, route_set)
    if progress is not None and len(items) >= 10:
        progress.emit(
            f"{context}: continuous unique {len(items)} -> {len(by_cert)}"
        )
    return [
        (state, tuple(sorted(routes)))
        for state, routes in by_cert.values()
    ]


def _monomer_pair_placements(
    source: _State,
    model: _LatticeModel,
    spec: NucleationSpec,
    *,
    count_ligands_as_neighbors: bool,
) -> List[Tuple[FloatArray, FloatArray]]:
    """Enumerate connected cation+anion monomer attachment sites on ``source``."""

    cation_sites = _vacancies(
        source,
        host_symbol=spec.core.anion,
        target_symbol=spec.core.cation,
        model=model,
        spec=spec,
        count_ligands_as_neighbors=count_ligands_as_neighbors,
    )
    anion_sites = _vacancies(
        source,
        host_symbol=spec.core.cation,
        target_symbol=spec.core.anion,
        model=model,
        spec=spec,
        count_ligands_as_neighbors=count_ligands_as_neighbors,
    )
    occupied = _atom_positions(source.atoms)
    pairs: Dict[
        Tuple[Tuple[int, int, int], Tuple[int, int, int]],
        Tuple[FloatArray, FloatArray],
    ] = {}
    for site in cation_sites:
        for anion_position in _partner_slots(
            site.position, spec.core.cation, occupied, model
        ):
            pairs[
                (
                    _position_key(site.position, model.site_tolerance),
                    _position_key(anion_position, model.site_tolerance),
                )
            ] = (site.position, anion_position)
    for site in anion_sites:
        for cation_position in _partner_slots(
            site.position, spec.core.anion, occupied, model
        ):
            pairs[
                (
                    _position_key(cation_position, model.site_tolerance),
                    _position_key(site.position, model.site_tolerance),
                )
            ] = (cation_position, site.position)
    return list(pairs.values())


def _place_monomer_on_source(
    source: _State,
    cation_position: FloatArray,
    anion_position: FloatArray,
    model: _LatticeModel,
    spec: NucleationSpec,
    *,
    keep_ligands: bool = False,
) -> Optional[_State]:
    """Attach one core monomer.

    ``keep_ligands=False`` (historical): return the ligand-stripped connected
    core for the bare p-DAG.  ``keep_ligands=True``: keep the Cl shell so
    continuous decoration can only fill free sites for Δp.
    """

    if keep_ligands:
        occupied = _atom_positions(source.atoms)
        radius = _soft_clash_radius(model)
        if _position_clashes(cation_position, occupied, radius):
            return None
        if _position_clashes(anion_position, occupied, radius):
            return None
        # New cation/anion must also not land on each other.
        if float(np.linalg.norm(cation_position - anion_position)) < radius:
            return None

    atoms = list(source.atoms)
    atoms.extend(
        (
            AtomRecord(
                len(atoms),
                spec.core.cation,
                tuple(float(value) for value in cation_position),
                "core_cation",
            ),
            AtomRecord(
                len(atoms) + 1,
                spec.core.anion,
                tuple(float(value) for value in anion_position),
                "core_anion",
            ),
        )
    )
    child = _make_core_graph(tuple(atoms), model, spec)
    bare_atoms = _without_ligands(child.atoms, spec)
    bare = _make_core_graph(bare_atoms, model, spec)
    if not bare.atoms or not nx.is_connected(bare.graph):
        return None
    if keep_ligands:
        if not _state_valid(child, model, spec):
            return None
        if _state_has_soft_clashes(child, model):
            return None
        return child
    return bare


def _skeleton_family_id(
    atoms: Sequence[AtomRecord],
    model: _LatticeModel,
    spec: NucleationSpec,
) -> str:
    """Stable id for the ligand-free inorganic core (lineage / ligand diffusion)."""

    bare_atoms = _without_ligands(atoms, spec)
    if not bare_atoms:
        return "fam_empty"
    core = _make_core_graph(bare_atoms, model, spec)
    if not core.atoms:
        return "fam_empty"
    digest = hashlib.sha1(
        repr(_graph_certificate(core.graph)).encode("utf-8")
    ).hexdigest()[:12]
    return f"fam_{digest}"


def _attach_lineage_metadata(
    record: ClusterRecord,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> None:
    """Tag skeleton family vs full-graph shell for lineage / ligand diffusion."""

    record.metadata["skeleton_family_id"] = _skeleton_family_id(
        record.atoms, model, spec
    )
    record.metadata["ligand_shell_hash"] = _graph_hash(record.graph)


def _place_n_ligands_free_sites(
    state: _State,
    n_ligands: int,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> Optional[_State]:
    """Place n terminal ligands only on free outward sites (Cl occupies space)."""

    if n_ligands <= 0:
        return state
    current = state
    radius = _soft_clash_radius(model)
    for _ in range(n_ligands):
        sites = _all_outward_ligand_sites(current, model, spec)
        # Reject sites that clash with ANY existing atom (not only ~0.2 Å).
        positions = _atom_positions(current.atoms)
        best_site: Optional[_Vacancy] = None
        best_key: Optional[Tuple[int, int, Tuple[int, int, int]]] = None
        for site in sites:
            if _position_clashes(site.position, positions, radius):
                continue
            if (
                not site.hosts
                or len(site.hosts)
                > spec.graph_rules.max_cn[spec.precursor.ligand]
            ):
                continue
            if any(
                current.graph.degree[host] + 1
                > spec.graph_rules.max_cn[current.atoms[host].symbol]
                for host in site.hosts
            ):
                continue
            deficit = max(
                spec.graph_rules.max_cn[current.atoms[host].symbol]
                - current.graph.degree[host]
                for host in site.hosts
            )
            key = (
                -len(site.hosts),
                -deficit,
                _position_key(site.position, model.site_tolerance),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_site = site
        if best_site is None:
            return None
        current = _extend_core_graph(
            current,
            [
                AtomRecord(
                    atom_id=len(current.atoms),
                    symbol=spec.precursor.ligand,
                    coordinates=tuple(
                        float(value) for value in best_site.position
                    ),
                    role="precursor_ligand",
                    unit_id=None,
                )
            ],
            model,
            spec,
        )
        # Fail fast after each ligand (do not stack near-duplicates then check).
        if _state_has_soft_clashes(current, model):
            return None
    if not _state_valid(current, model, spec):
        return None
    if _state_has_soft_clashes(current, model):
        return None
    return current


def _add_precursor_packages_free_sites(
    state: _State,
    p_m: int,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> Optional[_State]:
    """Add p_m CdCl2-like packages using only free tetrahedral sites."""

    if p_m <= 0:
        return state
    # A package consumes at least one Se coordination slot.  Check the whole
    # requested package before entering the per-site placement loop; this is a
    # cheap hard feasibility filter for continuous decorated growth.
    bare_state = _make_core_graph(
        _without_ligands(state.atoms, spec), model, spec
    )
    if _se_coordination_capacity(bare_state, spec) < int(p_m):
        return None
    current = state
    radius = _soft_clash_radius(model)
    for unit in range(p_m):
        # Free cation vacancies on anions; Cl blocks directions.
        sites = _vacancies(
            current,
            host_symbol=spec.core.anion,
            target_symbol=spec.precursor.center,
            model=model,
            spec=spec,
            count_ligands_as_neighbors=True,
        )
        # Also allow vacancies on existing cations if needed for precursor Cd.
        if not sites:
            sites = _vacancies(
                current,
                host_symbol=spec.core.anion,
                target_symbol=spec.core.cation,
                model=model,
                spec=spec,
                count_ligands_as_neighbors=True,
            )
        if not sites:
            return None
        positions = _atom_positions(current.atoms)
        free_sites = [
            site
            for site in sites
            if not _position_clashes(site.position, positions, radius)
        ]
        if not free_sites:
            return None
        site = min(
            free_sites,
            key=lambda item: _position_key(item.position, model.site_tolerance),
        )
        unit_id = (
            max(
                (
                    atom.unit_id
                    for atom in current.atoms
                    if atom.unit_id is not None
                ),
                default=0,
            )
            + 1
        )
        current = _extend_core_graph(
            current,
            [
                AtomRecord(
                    atom_id=len(current.atoms),
                    symbol=spec.precursor.center,
                    coordinates=tuple(
                        float(value) for value in site.position
                    ),
                    role="precursor_center",
                    unit_id=unit_id,
                )
            ],
            model,
            spec,
        )
        if not _base_coordination_valid(current, model, spec):
            return None
        if _state_has_soft_clashes(current, model):
            return None
        current = _place_n_ligands_free_sites(
            current, spec.precursor.ligand_count, model, spec
        )
        if current is None:
            return None
    if not _state_valid(current, model, spec):
        return None
    return current


def _precursor_center_ids(state: _State) -> List[int]:
    return sorted(
        atom.atom_id
        for atom in state.atoms
        if atom.role == "precursor_center"
    )


def _remove_precursor_centers(
    state: _State,
    center_ids: Sequence[int],
    model: _LatticeModel,
    spec: NucleationSpec,
) -> Optional[_State]:
    """Drop complete neutral precursor packages.

    A surface ligand has no persistent ``unit_id`` after bridge formation, so
    ownership cannot be recovered unambiguously.  Remove exactly the package
    stoichiometry instead: for each selected center, remove
    ``precursor.ligand_count`` ligands.  Ligands most strongly associated with
    the dropped centers are selected first, with terminal ligands preferred to
    ligands that remain bonded to retained hosts.  This keeps shedding bounded
    to one deterministic choice per center combination while preserving charge
    and the declared ``(k, p)`` composition.
    """

    centers = {
        int(i)
        for i in center_ids
        if 0 <= int(i) < len(state.atoms)
        and state.atoms[int(i)].role == "precursor_center"
    }
    ligand_ids = [
        atom.atom_id
        for atom in state.atoms
        if atom.role == "precursor_ligand"
    ]
    ligand_target = len(centers) * int(spec.precursor.ligand_count)
    if len(ligand_ids) < ligand_target:
        return None

    def ligand_removal_key(ligand_id: int) -> Tuple[int, int, int, int]:
        hosts = list(state.graph.neighbors(ligand_id))
        dropped_bonds = sum(host in centers for host in hosts)
        retained_bonds = len(hosts) - dropped_bonds
        return (-dropped_bonds, retained_bonds, len(hosts), ligand_id)

    removed_ligands = set(
        sorted(ligand_ids, key=ligand_removal_key)[:ligand_target]
    )
    drop = centers | removed_ligands
    kept = [
        atom for atom in state.atoms if atom.atom_id not in drop
    ]
    return _make_core_graph(kept, model, spec)


def _shed_parent_variants(
    state: _State,
    p: int,
    model: _LatticeModel,
    spec: NucleationSpec,
    *,
    k: Optional[int] = None,
) -> List[Tuple[_State, int, int]]:
    """Return ``(parent_state, p_out, shed)`` including shed=0 and up to max_shed.

    Combinations of which centers to remove are bounded; when too many, the
    least-connected precursor centers are removed greedily.

    With ``p_surf_beta > 0``, max shed follows scenario A:
    ``min(p, floor(shed_alpha * β k^{2/3}))`` (optional hard cap
    ``k_growth_max_shed``).  With no surface law, zero
    ``k_growth_max_shed`` means all complete packages up to ``p`` are
    considered; a positive value is an explicit fixed cap.
    """

    variants: List[Tuple[_State, int, int]] = [(state, p, 0)]
    k_eff = int(k) if k is not None else _core_formula_k(state, spec)
    max_shed = _effective_max_shed(k_eff, p, spec)
    if max_shed <= 0 or p <= 0:
        return variants
    centers = _precursor_center_ids(state)
    if not centers:
        return variants
    for shed in range(1, min(max_shed, len(centers), p) + 1):
        p_out = p - shed
        if p_out < 0:
            break
        combos = list(combinations(centers, shed))
        if len(combos) > 16:
            # Greedy: drop centers with fewest bonds to core anions first.
            def center_key(cid: int) -> Tuple[int, int]:
                anion_bonds = sum(
                    1
                    for nb in state.graph.neighbors(cid)
                    if state.atoms[nb].symbol == spec.core.anion
                )
                return (anion_bonds, cid)

            chosen = tuple(sorted(centers, key=center_key)[:shed])
            combos = [chosen]
        seen: set[Tuple[object, ...]] = set()
        for combo in combos:
            stripped = _remove_precursor_centers(state, combo, model, spec)
            if stripped is None or not stripped.atoms:
                continue
            _assert_atoms_match_bin(
                stripped.atoms,
                k=k_eff,
                p=p_out,
                spec=spec,
                context=f"shedding {shed} precursor package(s)",
            )
            cert = _graph_certificate(stripped.graph)
            if cert in seen:
                continue
            seen.add(cert)
            variants.append((stripped, p_out, shed))
    return variants


def _core_skeleton_children(
    sources: Sequence[Tuple[_State, Tuple[str, ...]]],
    *,
    k_from: int,
    p: int,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> Tuple[List[Tuple[_State, Tuple[str, ...]]], int, Dict[str, int]]:
    """Add one core monomer to inorganic skeletons only.

    Returns ``(children, attempted_placements, filter_stats)``.  When
    ``core_growth_policy`` is active for this destination ``k``, each parent's
    children are reduced to the steepest-ascent compact set before they enter
    the global uniqueness merge.
    """

    k_to = k_from + 1
    policy = spec.core_growth_policy
    if policy != "all" and k_to < spec.compact_from_k:
        policy = "all"
    children: List[Tuple[_State, Tuple[str, ...]]] = []
    attempted = 0
    raw_connected = 0
    after_policy = 0
    for source_index, (source, _source_routes) in enumerate(sources, start=1):
        pairs = _monomer_pair_placements(
            source, model, spec, count_ligands_as_neighbors=False
        )
        parent_children: List[Tuple[_State, Tuple[str, ...]]] = []
        for cation_position, anion_position in pairs:
            attempted += 1
            child = _place_monomer_on_source(
                source, cation_position, anion_position, model, spec
            )
            if child is None:
                continue
            route = (
                f"k{k_from:03d}_p{p:03d}_add_k_source"
                f"{source_index:04d}"
            )
            parent_children.append((child, (route,)))
        raw_connected += len(parent_children)
        kept = _filter_core_children_by_policy(
            source,
            parent_children,
            policy,
            k_to=k_to,
            fused_mode=spec.fused_chair_mode,
            fused_from_k=spec.fused_chair_from_k,
            ring_length=int(spec.inorganic_ring_length),
        )
        after_policy += len(kept)
        children.extend(kept)
    stats = {
        "core_growth_policy": 0 if policy == "all" else 1,
        "core_growth_raw_connected": raw_connected,
        "core_growth_after_policy": after_policy,
        "core_growth_policy_pruned": max(0, raw_connected - after_policy),
    }
    return children, attempted, stats


def _add_bare_precursor_centers_variants(
    state: _State,
    count: int,
    *,
    p_start: int,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> Tuple[List[_State], int]:
    """Add the center atoms of ``count`` ligand-free precursor packages.

    Bare growth postpones ligand-shell construction until the destination bin,
    but the package's precursor centers are part of the skeleton and must be
    present before that bin is labeled.  Enumerate available center placements
    one package at a time and deduplicate each layer to limit branching.
    """

    frontier: List[Tuple[_State, Tuple[str, ...]]] = [(state, ())]
    attempted = 0
    for offset in range(max(0, int(count))):
        raw: List[Tuple[_State, Tuple[str, ...]]] = []
        for source, routes in frontier:
            # Every newly added precursor Cd must consume at least one
            # previously unused Se coordination slot.  Apply this before site
            # enumeration so a chemically impossible package count cannot
            # trigger a large placement search.
            remaining = max(0, int(count) - offset)
            if _se_coordination_capacity(source, spec) < remaining:
                continue
            for site in _cation_vacancies_on_anions(source, model, spec):
                attempted += 1
                atoms = list(source.atoms)
                atoms.append(
                    AtomRecord(
                        atom_id=len(atoms),
                        symbol=spec.precursor.center,
                        coordinates=tuple(
                            float(value) for value in site.position
                        ),
                        role="precursor_center",
                        unit_id=int(p_start) + offset + 1,
                    )
                )
                child = _make_core_graph(atoms, model, spec)
                if not _base_coordination_valid(child, model, spec):
                    continue
                raw.append((child, routes))
        if not raw:
            return [], attempted
        frontier = _unique_skeleton_candidates(raw, model, spec)
    return [item[0] for item in frontier], attempted


def _bare_package_core_children(
    sources: Sequence[Tuple[_State, Tuple[str, ...]]],
    *,
    k_from: int,
    p: int,
    model: _LatticeModel,
    spec: NucleationSpec,
    p_m: int = 0,
) -> Tuple[
    Dict[int, List[Tuple[_State, Tuple[str, ...]]]],
    int,
    Dict[str, int],
]:
    """Bare-skeleton core growth with monomer package p_m and optional shed.

    Landing key is ``p0 = p - shed + p_m`` (building-block bookkeeping).
    """

    by_p: Dict[int, List[Tuple[_State, Tuple[str, ...]]]] = {}
    attempted = 0
    raw_connected = 0
    after_policy = 0
    channel_cap_pruned = 0
    se_capacity_pruned = 0
    package_center_attempted = 0
    k_to = k_from + 1
    policy = spec.core_growth_policy
    if policy != "all" and k_to < spec.compact_from_k:
        policy = "all"
    for source_index, (source, routes) in enumerate(sources, start=1):
        for parent, _p_after_shed, shed in _shed_parent_variants(
            source, p, model, spec, k=k_from
        ):
            p0 = _product_p0(p, shed, p_m)
            # A cap is a feasibility condition, never permission to relabel a
            # structure whose actual composition is still p0.
            if float(spec.p_surf_beta) > 0.0:
                channel_cap = _channel_p_child_max(
                    p, shed, p_m, k_to, spec
                )
                if p0 > channel_cap:
                    channel_cap_pruned += 1
                    continue
            pairs = _monomer_pair_placements(
                parent, model, spec, count_ligands_as_neighbors=False
            )
            parent_children: List[Tuple[_State, Tuple[str, ...]]] = []
            for cation_position, anion_position in pairs:
                attempted += 1
                child = _place_monomer_on_source(
                    parent, cation_position, anion_position, model, spec
                )
                if child is None:
                    continue
                p_parent_after_shed = max(0, int(p) - int(shed))
                if not _se_capacity_allows(
                    child,
                    p_parent_after_shed,
                    p0,
                    spec,
                ):
                    se_capacity_pruned += 1
                    continue
                packaged, center_attempted = (
                    _add_bare_precursor_centers_variants(
                        child,
                        p_m,
                        p_start=p - shed,
                        model=model,
                        spec=spec,
                    )
                )
                package_center_attempted += center_attempted
                route = (
                    f"k{k_from:03d}_p{p:03d}_pm{p_m:02d}_shed{shed}_"
                    f"p0{p0:03d}_src{source_index:04d}"
                )
                for packaged_child in packaged:
                    parent_children.append(
                        (
                            packaged_child,
                            (route, *routes) if routes else (route,),
                        )
                    )
            raw_connected += len(parent_children)
            kept = _filter_core_children_by_policy(
                parent,
                parent_children,
                policy,
                k_to=k_to,
                fused_mode=spec.fused_chair_mode,
                fused_from_k=spec.fused_chair_from_k,
                ring_length=int(spec.inorganic_ring_length),
            )
            after_policy += len(kept)
            by_p.setdefault(p0, []).extend(kept)
    stats = {
        "core_growth_raw_connected": raw_connected,
        "core_growth_after_policy": after_policy,
        "core_growth_policy_pruned": max(0, raw_connected - after_policy),
        "surface_channel_cap_pruned": channel_cap_pruned,
        "se_capacity_pruned": se_capacity_pruned,
        "package_center_placements_attempted": package_center_attempted,
        "monomer_p_m": int(p_m),
    }
    return by_p, attempted, stats


def _bare_shed_core_children(
    sources: Sequence[Tuple[_State, Tuple[str, ...]]],
    *,
    k_from: int,
    p: int,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> Tuple[
    Dict[int, List[Tuple[_State, Tuple[str, ...]]]],
    int,
    Dict[str, int],
]:
    """Backward-compatible shed-only bare growth (p_m = 0). """

    return _bare_package_core_children(
        sources,
        k_from=k_from,
        p=p,
        model=model,
        spec=spec,
        p_m=0,
    )


def _decorated_core_children_by_p(
    records: Sequence[ClusterRecord],
    *,
    k_from: int,
    p: int,
    model: _LatticeModel,
    spec: NucleationSpec,
    p_m: int = 0,
) -> Tuple[
    Dict[int, List[Tuple[_State, Tuple[str, ...]]]],
    int,
    Dict[str, int],
]:
    """Place monomer on free sites of passivated parents; Cl blocks directions.

    Returns children grouped by nominal product ``p0 = p - shed + p_m``.

    With ``continuous_decoration``: children keep the ligand shell and receive
    only free-site monomer packages (``p_m``).  Otherwise children are
    ligand-stripped skeletons for the historical re-passivation DAG.
    """

    by_p: Dict[int, List[Tuple[_State, Tuple[str, ...]]]] = {}
    attempted = 0
    raw_connected = 0
    after_policy = 0
    blocked_parents = 0
    channel_cap_pruned = 0
    se_capacity_pruned = 0
    package_center_attempted = 0
    continuous = bool(spec.continuous_decoration)
    k_to = k_from + 1
    policy = spec.core_growth_policy
    if policy != "all" and k_to < spec.compact_from_k:
        policy = "all"
    for source_index, record in enumerate(records, start=1):
        decorated = _make_core_graph(record.atoms, model, spec)
        if not decorated.atoms:
            continue
        variants = _shed_parent_variants(
            decorated, p, model, spec, k=k_from
        )
        for parent, _p_after_shed, shed in variants:
            p0 = _product_p0(p, shed, p_m)
            # The surface law prunes an infeasible channel.  It must not change
            # the bin label without removing a neutral precursor package.
            if float(spec.p_surf_beta) > 0.0:
                channel_cap = _channel_p_child_max(
                    p, shed, p_m, k_to, spec
                )
                if p0 > channel_cap:
                    channel_cap_pruned += 1
                    continue
            pairs = _monomer_pair_placements(
                parent,
                model,
                spec,
                count_ligands_as_neighbors=True,
            )
            if not pairs and shed == 0:
                blocked_parents += 1
            parent_children: List[Tuple[_State, Tuple[str, ...]]] = []
            # Policy filter needs a ligand-free parent graph for bond deltas.
            parent_bare = _make_core_graph(
                _without_ligands(parent.atoms, spec), model, spec
            )
            for cation_position, anion_position in pairs:
                attempted += 1
                child = _place_monomer_on_source(
                    parent,
                    cation_position,
                    anion_position,
                    model,
                    spec,
                    keep_ligands=continuous,
                )
                if child is None:
                    continue
                child_bare = _make_core_graph(
                    _without_ligands(child.atoms, spec), model, spec
                )
                p_parent_after_shed = max(0, int(p) - int(shed))
                if not _se_capacity_allows(
                    child_bare,
                    p_parent_after_shed,
                    p0,
                    spec,
                ):
                    se_capacity_pruned += 1
                    continue
                if continuous:
                    dressed = _add_precursor_packages_free_sites(
                        child, p_m, model, spec
                    )
                    if dressed is None:
                        continue
                    # Always re-validate full continuous assembly (no stacked Cl).
                    if not _state_valid(dressed, model, spec):
                        continue
                    if _state_has_soft_clashes(dressed, model):
                        continue
                    # Greedy max bridges (one shell): full enumeration explodes
                    # continuous attach (connected ≫ raw).
                    for variant in _latent_bridge_greedy_max(
                        dressed, model, spec
                    ):
                        child_bare = _make_core_graph(
                            _without_ligands(variant.atoms, spec), model, spec
                        )
                        route = (
                            f"k{k_from:03d}_p{p:03d}_pm{p_m:02d}_dec_shed{shed}_"
                            f"p0{p0:03d}_src{source_index:04d}_cont"
                        )
                        sid = record.structure_id or ""
                        routes = (route,) if not sid else (route, sid)
                        parent_children.append((variant, routes, child_bare))
                else:
                    # ``keep_ligands=False`` preserves the unshed parent
                    # precursor centers but does not materialize the p_m
                    # centers carried by the incoming monomer package.  The
                    # destination p-DAG rebuilds Cl later, yet its ligand-free
                    # skeleton must already contain exactly p0 Cd centers.
                    packaged, center_attempted = (
                        _add_bare_precursor_centers_variants(
                            child,
                            p_m,
                            p_start=p - shed,
                            model=model,
                            spec=spec,
                        )
                    )
                    package_center_attempted += center_attempted
                    route = (
                        f"k{k_from:03d}_p{p:03d}_pm{p_m:02d}_dec_shed{shed}_"
                        f"p0{p0:03d}_src{source_index:04d}"
                    )
                    sid = record.structure_id or ""
                    routes = (route,) if not sid else (route, sid)
                    for packaged_child in packaged:
                        _assert_bare_skeleton_matches_bin(
                            packaged_child.atoms,
                            k=k_to,
                            p=p0,
                            spec=spec,
                            context="decorated re-passivation growth",
                        )
                        parent_children.append(
                            (packaged_child, routes, packaged_child)
                        )
            raw_connected += len(parent_children)
            # Filter by bare-core policy, then keep full states.
            bare_pool = [
                (bare, routes) for _full, routes, bare in parent_children
            ]
            kept_bare = _filter_core_children_by_policy(
                parent_bare,
                bare_pool,
                policy,
                k_to=k_to,
                fused_mode=spec.fused_chair_mode,
                fused_from_k=spec.fused_chair_from_k,
                ring_length=int(spec.inorganic_ring_length),
            )
            kept_certs = {
                _graph_certificate(bare.graph) for bare, _r in kept_bare
            }
            kept_full: List[Tuple[_State, Tuple[str, ...]]] = []
            for full, routes, bare in parent_children:
                if _graph_certificate(bare.graph) in kept_certs:
                    kept_full.append((full, routes))
            after_policy += len(kept_full)
            by_p.setdefault(p0, []).extend(kept_full)
    stats = {
        "core_growth_policy": 0 if policy == "all" else 1,
        "core_growth_raw_connected": raw_connected,
        "core_growth_after_policy": after_policy,
        "core_growth_policy_pruned": max(0, raw_connected - after_policy),
        "surface_channel_cap_pruned": channel_cap_pruned,
        "se_capacity_pruned": se_capacity_pruned,
        "package_center_placements_attempted": package_center_attempted,
        "decorated_blocked_parents": blocked_parents,
        "growth_occupation": 1,
        "continuous_decoration": int(continuous),
        "monomer_p_m": int(p_m),
    }
    return by_p, attempted, stats


def _enumerate_skeleton_bin(
    skeletons: Sequence[Tuple[_State, Tuple[str, ...]]],
    *,
    k: int,
    p: int,
    model: _LatticeModel,
    spec: NucleationSpec,
    progress: _ProgressReporter,
    cache: _EnumerationCache,
) -> _Generation:
    """Construct ligand graphs once for every merged inorganic skeleton."""

    records: List[ClusterRecord] = []
    # Bases must agree on their bridge options, not merely on topology: a
    # shared-vacant-CIF-site bridge depends on the coordinates, so merging
    # graph-isomorphic bases discarded reachable structures.
    base_accumulator = _CandidateAccumulator(
        model, spec, comparison="bridges"
    )
    invalid: Dict[str, int] = {}
    stages: Dict[str, int] = {}
    raw = 0
    cross_started = time.monotonic()

    if spec.mode == "guided":
        return _guided_skeleton_bin(
            skeletons, k=k, p=p, model=model, spec=spec,
            progress=progress, cache=cache,
        )

    progress.emit(
        f"k={k} p={p}: ligand enumeration and streaming "
        f"cross-skeleton filtering started, skeletons={len(skeletons)}"
    )
    for index, (skeleton, routes) in enumerate(skeletons, start=1):
        states, attempted, reasons, stage_counts = _enumerate_ligand_states(
            skeleton.atoms,
            spec.precursor.ligand_count * p,
            model,
            spec,
            progress,
            cache,
            context=(
                f"k={k} p={p} DAG skeleton={index}/{len(skeletons)}"
            ),
        )
        raw += attempted
        _merge_reason_counts(invalid, reasons)
        _merge_reason_counts(stages, stage_counts)
        for state in states:
            base_accumulator.add(state, routes)
        progress.heartbeat(
            f"k={k} p={p}: cross-skeleton filtering "
            f"skeleton={index}/{len(skeletons)}, "
            f"candidates={base_accumulator.candidate_count}, "
            f"classes={len(base_accumulator.classes)}, "
            f"isomorphism_checks={base_accumulator.isomorphism_checks}, "
            f"elapsed={time.monotonic() - cross_started:.1f}s"
        )
    unique_bases = base_accumulator.result()
    stages["base_embeddings_before_cross_skeleton_symmetry"] = (
        base_accumulator.candidate_count
    )
    stages["base_embeddings_after_cross_skeleton_symmetry"] = len(
        unique_bases
    )
    stages["cross_skeleton_base_duplicates"] = max(
        0, base_accumulator.candidate_count - len(unique_bases)
    )
    stages["cross_skeleton_isomorphism_checks"] = (
        base_accumulator.isomorphism_checks
    )
    progress.emit(
        f"k={k} p={p}: cross-skeleton filtering complete, "
        f"candidates={base_accumulator.candidate_count}, "
        f"unique_bases={len(unique_bases)}, "
        f"duplicates={base_accumulator.candidate_count - len(unique_bases)}, "
        f"isomorphism_checks={base_accumulator.isomorphism_checks}, "
        f"elapsed={time.monotonic() - cross_started:.1f}s"
    )
    bridge_search_states = 0
    dominated_bridge_variants_pruned = 0
    bridge_variants = 0
    bridge_bases_bound_pruned = 0
    bridge_bases_dp_pruned = 0
    bridge_symmetry_bases = 0
    bridge_identity_fallback_bases = 0
    bridge_symmetry_pruned = 0
    bridge_orbit_representatives = 0
    bridge_raw_extensions = 0
    bridge_automorphism_cache_hits = 0
    bridge_exactness_certified = 0
    bridge_sub_maximum_fallbacks = 0
    bridge_sub_maximum_undischarged = 0
    bridge_sub_maximum_contenders = 0
    bridge_started = time.monotonic()
    # The optimistic bound only gates the k>2 branch below, so do not pay for it
    # at k<=2 where every base is searched regardless.
    if k > 2:
        ranked_bases = [
            (_optimistic_bridge_score(base, spec), base, routes)
            for base, routes in unique_bases
        ]
        ranked_bases.sort(
            key=lambda item: (
                item[0],
                _graph_hash(item[1].graph),
                tuple(atom.coordinates for atom in item[1].atoms),
            ),
            reverse=True,
        )
    else:
        ranked_bases = [(None, base, routes) for base, routes in unique_bases]

    # Build one guided structure per skeleton the way the passivation module
    # would.  Its score is an achieved, surface-valid incumbent: at k>2 it lets
    # the pruning gates fire from the very first base, and at every k the
    # greedy-vs-selected comparison is the audit that will one day justify a
    # guided large-k mode.  It is never added to the records -- the exact
    # enumeration regenerates its isomorphism class on its own.
    greedy_incumbent: Optional[Tuple[int, ...]] = None
    for skeleton, _routes in skeletons:
        seeded = _greedy_incumbent_state(
            skeleton, spec.precursor.ligand_count * p, model, spec
        )
        if seeded is None:
            continue
        for variant in _latent_bridge_variants(
            seeded, model, spec, prune_dominated=True, cache=cache
        ):
            gated = variant.graph.graph.get("surface_gate_valid")
            if gated is None:
                probe = _record_from_state(
                    variant, k=k, p=p, spec=spec, operation="greedy_incumbent"
                )
                _surface, geometry = _precondition_surface_geometry(
                    probe, model, spec, audit=False
                )
                gated = geometry.get("projection_valid", False)
            if gated:
                score = _graph_coordination_score(
                    variant.atoms, variant.graph, spec
                )
                if greedy_incumbent is None or score > greedy_incumbent:
                    greedy_incumbent = score
    stages["greedy_incumbent_found"] = int(greedy_incumbent is not None)

    best_surface_score: Optional[Tuple[int, ...]] = (
        greedy_incumbent if k > 2 else None
    )
    progress.emit(
        f"k={k} p={p}: bridge search started, "
        f"bases={len(unique_bases)}, "
        f"greedy_incumbent={'yes' if greedy_incumbent is not None else 'no'}"
    )
    for base_index, (score_bound, base, routes) in enumerate(
        ranked_bases, start=1
    ):
        if k > 2 and best_surface_score is not None:
            # Cheap componentwise bound first; the joint decision procedure
            # only for survivors.  Both prune exactly: they dominate every
            # score reachable by any bridge arrangement on this base.
            pruned_by = None
            if score_bound < best_surface_score:
                bridge_bases_bound_pruned += 1
                pruned_by = "bound"
            elif (
                _reachable_bridge_score_max(base, model, spec, cache)
                < best_surface_score
            ):
                bridge_bases_dp_pruned += 1
                pruned_by = "dp"
            if pruned_by is not None:
                progress.heartbeat(
                    f"k={k} p={p}: bridge base={base_index}/{len(unique_bases)}, "
                    f"bound_pruned={bridge_bases_bound_pruned}, "
                    f"dp_pruned={bridge_bases_dp_pruned}, "
                    f"search_states={bridge_search_states}, "
                    f"variants={bridge_variants}, records={len(records)}, "
                    f"elapsed={time.monotonic() - bridge_started:.1f}s"
                )
                continue
        base_bridge_stats: Dict[str, int] = {}
        variants = _latent_bridge_variants(
            base,
            model,
            spec,
            prune_dominated=k > 2,
            cache=cache,
            stats_out=base_bridge_stats,
        )
        bridge_variants += max(0, len(variants) - 1)
        bridge_search_states += int(
            base_bridge_stats.get("bridge_search_states", 0)
        )
        dominated_bridge_variants_pruned += int(
            base_bridge_stats.get("dominated_bridge_variants_pruned", 0)
        )
        bridge_used_symmetry = int(
            base_bridge_stats.get("bridge_symmetry_used", 0)
        )
        bridge_has_search_metadata = "bridge_symmetry_used" in base_bridge_stats
        bridge_symmetry_bases += bridge_used_symmetry
        bridge_identity_fallback_bases += int(
            bridge_has_search_metadata and not bridge_used_symmetry
        )
        bridge_symmetry_pruned += int(
            base_bridge_stats.get("bridge_symmetry_pruned", 0)
        )
        bridge_orbit_representatives += int(
            base_bridge_stats.get("bridge_orbit_representatives", 0)
        )
        bridge_raw_extensions += int(
            base_bridge_stats.get("bridge_raw_extensions", 0)
        )
        bridge_exactness_certified += int(
            base_bridge_stats.get("bridge_exactness_certified", 0)
        )
        fallback_flag = int(base_bridge_stats.get("bridge_sub_maximum_fallback", 0))
        if fallback_flag > 0:
            bridge_sub_maximum_fallbacks += 1
        elif fallback_flag < 0:
            bridge_sub_maximum_undischarged += 1
        bridge_sub_maximum_contenders += int(
            base_bridge_stats.get("bridge_sub_maximum_contenders", 0)
        )
        bridge_automorphism_cache_hits += int(
            base_bridge_stats.get("bridge_automorphism_cache_hits", 0)
        )
        for state in variants:
            record = _record_from_state(
                state,
                k=k,
                p=p,
                spec=spec,
                operation="skeleton_dag",
                source_ids=routes,
            )
            records.append(record)
            if k > 2:
                # The bridge search gates the layer it returns, so reuse that
                # verdict rather than projecting the same geometry again.
                gated = state.graph.graph.get("surface_gate_valid")
                if gated is None:
                    _surface, geometry = _precondition_surface_geometry(
                        record, model, spec, audit=False
                    )
                    gated = geometry.get("projection_valid", False)
                if gated:
                    score = _graph_coordination_score(
                        state.atoms, state.graph, spec
                    )
                    if best_surface_score is None or score > best_surface_score:
                        best_surface_score = score
        progress.heartbeat(
            f"k={k} p={p}: bridge base={base_index}/{len(unique_bases)}, "
            f"bound_pruned={bridge_bases_bound_pruned}, "
            f"symmetric_bases={bridge_symmetry_bases}, "
            f"symmetry_pruned={bridge_symmetry_pruned}, "
            f"orbit_representatives={bridge_orbit_representatives}, "
            f"search_states={bridge_search_states}, "
            f"variants={bridge_variants}, records={len(records)}, "
            f"elapsed={time.monotonic() - bridge_started:.1f}s"
        )
    progress.emit(
        f"k={k} p={p}: bridge search complete, "
        f"bases={len(unique_bases)}, bound_pruned={bridge_bases_bound_pruned}, "
        f"symmetric_bases={bridge_symmetry_bases}, "
        f"identity_fallbacks={bridge_identity_fallback_bases}, "
        f"symmetry_pruned={bridge_symmetry_pruned}, "
        f"orbit_representatives={bridge_orbit_representatives}, "
        f"search_states={bridge_search_states}, "
        f"variants={bridge_variants}, records={len(records)}, "
        f"elapsed={time.monotonic() - bridge_started:.1f}s"
    )
    stages["bridge_variants"] = bridge_variants
    stages["bridge_search_states"] = bridge_search_states
    stages["dominated_bridge_variants_pruned"] = (
        dominated_bridge_variants_pruned
    )
    stages["bridge_bases_bound_pruned"] = bridge_bases_bound_pruned
    stages["bridge_bases_dp_pruned"] = bridge_bases_dp_pruned
    stages["bridge_exactness_certified"] = bridge_exactness_certified
    stages["bridge_sub_maximum_fallbacks"] = bridge_sub_maximum_fallbacks
    stages["bridge_sub_maximum_contenders"] = bridge_sub_maximum_contenders
    # Bases where the maximum-cardinality restriction could be neither proved
    # optimal nor cheaply checked.  Non-zero means this bin carries an
    # undischarged assumption and must say so.
    stages["bridge_sub_maximum_undischarged"] = bridge_sub_maximum_undischarged
    stages["bridge_symmetry_bases"] = bridge_symmetry_bases
    stages["bridge_identity_fallback_bases"] = (
        bridge_identity_fallback_bases
    )
    stages["bridge_symmetry_pruned"] = bridge_symmetry_pruned
    stages["bridge_orbit_representatives"] = bridge_orbit_representatives
    stages["bridge_raw_extensions"] = bridge_raw_extensions
    stages["bridge_automorphism_cache_hits"] = (
        bridge_automorphism_cache_hits
    )
    stages["merged_skeletons"] = len(skeletons)
    return _Generation(
        records,
        raw,
        invalid,
        stages,
        greedy_incumbent_score=greedy_incumbent,
    )


def _validate_strip_parents(
    skeletons: Mapping[int, Sequence[Tuple[_State, Tuple[str, ...]]]],
    *,
    k: int,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> List[SweepAudit]:
    """Verify downward parentage without regenerating ligand configurations."""

    audits: List[SweepAudit] = []
    for p in sorted(value for value in skeletons if value > 0):
        parents = skeletons.get(p - 1, ())
        parent_graphs = [
            _skeleton_frontier_graph(state, model, spec)
            for state, _routes in parents
        ]
        attempted = 0
        matched = 0
        invalid: Dict[str, int] = {}
        for state, _routes in skeletons[p]:
            centers = [
                atom for atom in state.atoms
                if atom.role == "precursor_center"
            ]
            state_has_parent = False
            for center in centers:
                attempted += 1
                remaining = _normalize_atoms(
                    atom for atom in state.atoms
                    if atom.atom_id != center.atom_id
                )
                candidate = _make_core_graph(remaining, model, spec)
                if not nx.is_connected(candidate.graph):
                    _increment(invalid, "disconnected_after_strip")
                    continue
                graph = _skeleton_frontier_graph(candidate, model, spec)
                if any(
                    _graphs_isomorphic(graph, parent_graph)
                    for parent_graph in parent_graphs
                ):
                    matched += 1
                    state_has_parent = True
                else:
                    _increment(invalid, "unregistered_strip_parent")
            if not state_has_parent:
                _increment(invalid, "skeleton_without_strip_parent")
        audits.append(
            SweepAudit(
                k=k,
                operation="strip_validation",
                p_from=p,
                p_to=p - 1,
                source_count=len(skeletons[p]),
                raw_count=attempted,
                valid_count=matched,
                symmetry_duplicate_count=0,
                invalid_reasons=invalid,
                stage_counts={"ligand_enumerations": 0},
            )
        )
    return audits


def _complete_k_dag(
    *,
    k: int,
    inherited: Mapping[int, Sequence[Tuple[_State, Tuple[str, ...]]]],
    model: _LatticeModel,
    spec: NucleationSpec,
    progress: _ProgressReporter,
    cache: _EnumerationCache,
    checkpoint_dir: Optional[str | Path] = None,
    resume_state: Optional[Mapping[str, object]] = None,
    inherited_decorated: Optional[
        Mapping[int, Sequence[Tuple[_State, Tuple[str, ...]]]]
    ] = None,
) -> Tuple[
    Dict[int, List[ClusterRecord]],
    Dict[int, List[ClusterRecord]],
    Dict[int, List[Tuple[_State, Tuple[str, ...]]]],
    List[SweepAudit],
]:
    """Complete one k row of the path-independent skeleton DAG.

    ``inherited`` holds ligand-free skeletons (historical path).
    ``inherited_decorated`` holds fully passivated continuous-decoration
    children: those bins skip full ligand rebuild and only free-site Δp ladder.

    When ``checkpoint_dir`` is set, each finished ``(k, p)`` bin is flushed to
    disk immediately after retention.
    """

    retained_registry: Dict[int, List[ClusterRecord]] = {}
    discarded_registry: Dict[int, List[ClusterRecord]] = {}
    skeleton_registry: Dict[int, List[Tuple[_State, Tuple[str, ...]]]] = {}
    audits: List[SweepAudit] = []
    generated: List[Tuple[_State, Tuple[str, ...]]] = []
    generated_decorated: List[Tuple[_State, Tuple[str, ...]]] = []
    decorated_map: Dict[int, List[Tuple[_State, Tuple[str, ...]]]] = {
        int(p_key): list(entries)
        for p_key, entries in (inherited_decorated or {}).items()
    }
    inject_keys = set(inherited) | set(decorated_map)
    max_inherited = max(inject_keys, default=0)
    min_inherited = min(inject_keys) if inject_keys else 0
    # Derived capacity bound: every cation needs at least one anion neighbour,
    # the only anions are the k core ones, and each holds at most max_cn bonds,
    # so k + p <= max_cn[anion] * k.  Growth must close at or before this p.
    capacity_cap = (spec.graph_rules.max_cn[spec.core.anion] - 1) * k
    p_cap = capacity_cap
    p_min = 0
    # Optional Δp window after k-growth.  Never apply on the k=1 seed row.
    if k > 1 and inject_keys:
        # Shed bound at parent size for product_window down-window.
        parent_k = max(1, k - 1)
        shed_bound = _effective_max_shed(
            parent_k, max(max_inherited, 1), spec
        )
        if spec.p_ladder_mode == "product_window":
            # Explore only around injected product p0 values.
            p_min = max(0, min_inherited - shed_bound)
            if float(spec.p_surf_beta) > 0.0:
                # Scenario A: redecoration ceiling = spherical surface, not 3k.
                p_cap = _effective_p_cap(
                    k,
                    capacity_cap=capacity_cap,
                    max_inherited=max_inherited,
                    spec=spec,
                )
            elif spec.k_growth_max_add >= 0:
                p_cap = min(
                    capacity_cap,
                    max_inherited + int(spec.k_growth_max_add),
                )
            else:
                p_cap = capacity_cap
        elif float(spec.p_surf_beta) > 0.0:
            p_cap = _effective_p_cap(
                k,
                capacity_cap=capacity_cap,
                max_inherited=max_inherited,
                spec=spec,
            )
        elif spec.k_growth_max_add >= 0:
            # inherited_plus (historical)
            p_cap = min(capacity_cap, max_inherited + int(spec.k_growth_max_add))
    p = p_min
    if resume_state is not None:
        retained_registry = dict(resume_state["retained"])  # type: ignore[arg-type]
        discarded_registry = dict(resume_state["discarded"])  # type: ignore[arg-type]
        skeleton_registry = dict(resume_state["skeletons"])  # type: ignore[arg-type]
        last_p = int(resume_state["last_completed_p"])  # type: ignore[arg-type]
        stored_cap = resume_state.get("p_cap")
        if stored_cap is not None:
            p_cap = int(stored_cap)
        stored_max_inh = resume_state.get("max_inherited")
        if stored_max_inh is not None:
            max_inherited = max(max_inherited, int(stored_max_inh))
        # Resume at the next p after the last fully completed bin.
        if last_p >= 0 and last_p in skeleton_registry:
            parents = skeleton_registry[last_p]
            raw_generated, _attempted = _precursor_skeleton_children(
                parents, k=k, p=last_p, model=model, spec=spec
            )
            generated = _unique_skeleton_candidates(
                raw_generated,
                model,
                spec,
                progress,
                context=f"k={k} p={last_p}->p={last_p + 1} resume merge",
            )
            p = last_p + 1
            progress.emit(
                f"k={k}: resuming p-ladder after p={last_p} "
                f"(next p={p}, skeletons_so_far={len(skeleton_registry)})"
            )
        elif last_p >= 0:
            p = last_p + 1
            progress.emit(
                f"k={k}: resuming p-ladder at p={p} "
                f"(no skeletons at last_completed_p={last_p})"
            )
    progress.emit(
        f"k={k}: starting merged skeleton DAG "
        f"(p_ladder_mode={spec.p_ladder_mode}, p in [{p_min}, {p_cap}]"
        + (
            f", max_add={spec.k_growth_max_add}, max_shed={spec.k_growth_max_shed}"
            + (
                f", p_surf_beta={spec.p_surf_beta}, "
                f"p_surf(k)={_p_surf(k, spec.p_surf_beta)}"
                if float(spec.p_surf_beta) > 0.0
                else ""
            )
            if k > 1 and inject_keys
            else ""
        )
        + (
            f", continuous_decorated_bins={sorted(decorated_map)}"
            if decorated_map
            else ""
        )
        + f", capacity={capacity_cap})"
    )
    if checkpoint_dir is not None and resume_state is None:
        # Seed partial checkpoint with inherited so mid-k restart can continue.
        write_nucleation_checkpoint(
            root=checkpoint_dir,
            spec=spec,
            k=k,
            retained=retained_registry,
            discarded=(
                discarded_registry
                if k <= spec.discarded_through_k
                else {}
            ),
            skeletons=skeleton_registry,
            discarded_counts={
                p_key: len(recs)
                for p_key, recs in discarded_registry.items()
            },
            mark_done=False,
            last_completed_p=-1,
            p_cap=p_cap,
            max_inherited=max_inherited,
            inherited=inherited,
        )

    def _flush_bin_checkpoint(completed_p: int) -> None:
        if checkpoint_dir is None:
            return
        write_discarded = k <= spec.discarded_through_k
        write_nucleation_bin_structures(
            root=checkpoint_dir,
            k=k,
            p=completed_p,
            retained=retained_registry.get(completed_p, ()),
            discarded=(
                discarded_registry.get(completed_p, ())
                if write_discarded
                else ()
            ),
            write_discarded=write_discarded,
        )
        write_nucleation_checkpoint(
            root=checkpoint_dir,
            spec=spec,
            k=k,
            retained=retained_registry,
            discarded=(
                discarded_registry if write_discarded else {}
            ),
            skeletons=skeleton_registry,
            discarded_counts={
                p_key: len(recs)
                for p_key, recs in discarded_registry.items()
            },
            mark_done=False,
            last_completed_p=completed_p,
            p_cap=p_cap,
            max_inherited=max_inherited,
            inherited=inherited,
        )
        progress.emit(
            f"checkpoint: k={k} p={completed_p} retained="
            f"{len(retained_registry.get(completed_p, ()))} "
            f"flushed under {checkpoint_dir}"
        )

    while True:
        pool = [*inherited.get(p, ()), *generated]
        dec_pool = [*decorated_map.get(p, ()), *generated_decorated]
        if not pool and not dec_pool:
            if p < max_inherited:
                p += 1
                generated = []
                generated_decorated = []
                continue
            break
        assert p <= p_cap, (
            f"k={k} p={p} exceeds the anion-capacity bound p <= "
            f"{p_cap}; a coordination rule is being violated"
        )
        # Already completed this p on a prior partial run: skip rebuild, but
        # keep the precursor-grown pool for p+1 if we must walk past it.
        if (
            resume_state is not None
            and p in retained_registry
            and p in skeleton_registry
        ):
            if p >= p_cap:
                generated = []
                generated_decorated = []
            else:
                raw_skip, _att = _precursor_skeleton_children(
                    skeleton_registry[p],
                    k=k,
                    p=p,
                    model=model,
                    spec=spec,
                )
                generated = _unique_skeleton_candidates(
                    raw_skip,
                    model,
                    spec,
                    progress,
                    context=f"k={k} p={p}->p={p + 1} skip-resume merge",
                )
                generated_decorated = []
            p += 1
            continue
        if len(pool) >= 20 or len(dec_pool) >= 20:
            progress.emit(
                f"k={k} p={p}/{p_cap}: merging pool "
                f"(bare={len(pool)}, continuous={len(dec_pool)})"
            )
        unique: List[Tuple[_State, Tuple[str, ...]]] = []
        merged_routes = 0
        beam_dropped = 0
        generation_records: List[ClusterRecord] = []
        generation_raw = 0
        generation_invalid: Dict[str, int] = {}
        generation_stages: Dict[str, int] = {}
        greedy_incumbent_score = None
        continuous_unique: List[Tuple[_State, Tuple[str, ...]]] = []

        if pool:
            unique = _unique_skeleton_candidates(
                pool,
                model,
                spec,
                progress,
                context=f"k={k} p={p}/{p_cap} skeleton pool",
            )
            merged_routes = max(0, len(pool) - len(unique))
            unique, beam_dropped = _apply_p_skeleton_beam(
                unique, k=k, p=p, spec=spec, progress=progress
            )
            skeleton_registry[p] = unique
            progress.emit(
                f"k={k} p={p}/{p_cap}: skeletons={len(unique)}, "
                f"parent_routes_merged={merged_routes}"
                + (
                    f", beam_dropped={beam_dropped}"
                    if beam_dropped
                    else ""
                )
            )
            generation = _enumerate_skeleton_bin(
                unique,
                k=k,
                p=p,
                model=model,
                spec=spec,
                progress=progress,
                cache=cache,
            )
            generation_records.extend(generation.records)
            generation_raw += generation.raw_count
            for key, value in generation.invalid_reasons.items():
                generation_invalid[key] = (
                    generation_invalid.get(key, 0) + value
                )
            generation_stages.update(generation.stage_counts)
            greedy_incumbent_score = generation.greedy_incumbent_score

        if dec_pool:
            continuous_unique = _unique_decorated_with_routes(
                dec_pool,
                progress=progress,
                context=f"k={k} p={p}/{p_cap} continuous pool",
            )
            # Cap continuous shells with the same beam width (full-graph rank
            # by bare skeleton metrics).
            if (
                spec.p_skeleton_beam > 0
                and k >= spec.p_beam_from_k
                and len(continuous_unique) > spec.p_skeleton_beam
            ):
                scored = [
                    (
                        _p_beam_sort_key(
                            _make_core_graph(
                                _without_ligands(state.atoms, spec),
                                model,
                                spec,
                            ),
                            spec.p_beam_rank,
                            ring_length=int(spec.inorganic_ring_length),
                        ),
                        state,
                        routes,
                    )
                    for state, routes in continuous_unique
                ]
                scored.sort(key=lambda item: item[0], reverse=True)
                continuous_unique = [
                    (state, routes)
                    for _key, state, routes in scored[: spec.p_skeleton_beam]
                ]
            progress.emit(
                f"k={k} p={p}/{p_cap}: CONTINUOUS shells={len(continuous_unique)} "
                f"(free-site Δp; maximizing Cl bridges)"
            )
            cont_bridge_variants = 0
            for state, routes in continuous_unique:
                # Greedy max-bridge fill (one variant per shell). Full subset
                # enumeration is too slow for continuous k≥3.
                variants = _latent_bridge_greedy_max(state, model, spec)
                cont_bridge_variants += len(variants)
                for variant in variants:
                    record = _record_from_state(
                        variant,
                        k=k,
                        p=p,
                        spec=spec,
                        operation="continuous_decoration",
                        source_ids=routes,
                    )
                    generation_records.append(record)
            generation_raw += cont_bridge_variants
            generation_stages["continuous_shells"] = len(continuous_unique)
            generation_stages["continuous_bridge_variants"] = cont_bridge_variants
            generation_stages["continuous_bridge_mode"] = "greedy_max"
            progress.emit(
                f"k={k} p={p}: continuous greedy bridges "
                f"shells={len(continuous_unique)} -> "
                f"states={cont_bridge_variants}"
            )
            # Also register bare cores for audits / non-continuous consumers.
            bare_from_cont = [
                (
                    _make_core_graph(
                        _without_ligands(state.atoms, spec), model, spec
                    ),
                    routes,
                )
                for state, routes in continuous_unique
            ]
            if p not in skeleton_registry:
                skeleton_registry[p] = _unique_skeleton_candidates(
                    bare_from_cont,
                    model,
                    spec,
                    progress,
                    context=f"k={k} p={p} continuous bare extract",
                )
            else:
                skeleton_registry[p] = _unique_skeleton_candidates(
                    [*skeleton_registry[p], *bare_from_cont],
                    model,
                    spec,
                    progress,
                    context=f"k={k} p={p} merge continuous bare",
                )

        if not generation_records and not unique and not continuous_unique:
            p += 1
            generated = []
            generated_decorated = []
            continue

        retained, discarded, duplicates = _select_bin(
            generation_records, k, p, model, spec, progress
        )
        for record in [*retained, *discarded]:
            _attach_lineage_metadata(record, model, spec)
        _assign_structure_ids(retained, discarded, k, p)
        # Keep discarded accounting for empty bins; omit empty retained keys
        # so the map and surface pass only show real winners.
        if retained:
            retained_registry[p] = retained
        elif p in retained_registry:
            del retained_registry[p]
        discarded_registry[p] = discarded
        if not retained and discarded:
            progress.emit(
                f"k={k} p={p}: retained=0, discarded={len(discarded)} "
                f"(no surface-valid winner; discarded counts only)"
                + (
                    f", continuous={len(continuous_unique)}"
                    if continuous_unique
                    else ""
                )
            )
        else:
            progress.emit(
                f"k={k} p={p}: retained={len(retained)}, "
                f"discarded={len(discarded)}, symmetry_duplicates={duplicates}"
                + (
                    f", continuous={len(continuous_unique)}"
                    if continuous_unique
                    else ""
                )
            )
        _flush_bin_checkpoint(p)
        stage_counts = {
            **generation_stages,
            "parent_routes_merged": merged_routes,
            "p_skeleton_beam": int(spec.p_skeleton_beam),
            "p_skeleton_beam_dropped": int(beam_dropped),
            "p_skeleton_beam_kept": len(unique) + len(continuous_unique),
        }
        if greedy_incumbent_score is not None and retained:
            stage_counts["greedy_incumbent_matches_selection"] = int(
                greedy_incumbent_score == retained[0].coordination_score
            )
        audits.append(
            SweepAudit(
                k=k,
                operation="dag_bin",
                p_from=p,
                p_to=p,
                source_count=len(unique) + len(continuous_unique),
                raw_count=generation_raw,
                valid_count=len(generation_records),
                symmetry_duplicate_count=duplicates,
                invalid_reasons=generation_invalid,
                stage_counts=stage_counts,
            )
        )
        # Next lines used to run with no log line; at high p they can take
        # minutes (frontier certificates on many children) and look "hung".
        if p >= p_cap:
            # Terminal empty step: capacity (or Δp add window) forbids p+1.
            progress.emit(
                f"k={k} p={p}: p-ladder closed at capacity bound {p_cap}"
            )
            audits.append(
                SweepAudit(
                    k=k,
                    operation="skeleton_passivation",
                    p_from=p,
                    p_to=p + 1,
                    source_count=len(unique) + len(continuous_unique),
                    raw_count=0,
                    valid_count=0,
                    symmetry_duplicate_count=0,
                    stage_counts={"ligand_enumerations": 0},
                )
            )
            generated = []
            generated_decorated = []
            p += 1
            continue
        # Bare ladder (historical).
        if unique:
            progress.emit(
                f"k={k} p={p}: growing precursor skeletons to p={p + 1} "
                f"(sources={len(unique)})"
            )
            raw_generated, attempted = _precursor_skeleton_children(
                unique,
                k=k,
                p=p,
                model=model,
                spec=spec,
            )
            progress.emit(
                f"k={k} p={p}: precursor placements raw={attempted}, "
                f"connected={len(raw_generated)}; merging unique skeletons"
            )
            generated = _unique_skeleton_candidates(
                raw_generated,
                model,
                spec,
                progress,
                context=f"k={k} p={p}->p={p + 1} skeleton merge",
            )
            progress.emit(
                f"k={k} p={p} -> p={p + 1}: unique_skeletons={len(generated)}, "
                f"duplicates={max(0, len(raw_generated) - len(generated))}"
            )
            audits.append(
                SweepAudit(
                    k=k,
                    operation="skeleton_passivation",
                    p_from=p,
                    p_to=p + 1,
                    source_count=len(unique),
                    raw_count=attempted,
                    valid_count=len(generated),
                    symmetry_duplicate_count=max(
                        0, len(raw_generated) - len(generated)
                    ),
                    stage_counts={"ligand_enumerations": 0},
                )
            )
        else:
            generated = []
        # Continuous free-site ladder: add one CdCl2 package on free sites only.
        if continuous_unique:
            cont_children: List[Tuple[_State, Tuple[str, ...]]] = []
            cont_attempted = 0
            for state, routes in continuous_unique:
                cont_attempted += 1
                child = _add_precursor_packages_free_sites(
                    state, 1, model, spec
                )
                if child is None:
                    continue
                # Greedy re-bridge after free-site package add.
                for variant in _latent_bridge_greedy_max(child, model, spec):
                    route = f"k{k:03d}_p{p:03d}_cont_add_p"
                    cont_children.append(
                        (
                            variant,
                            (route, *routes) if routes else (route,),
                        )
                    )
            generated_decorated = _unique_decorated_with_routes(
                cont_children,
                progress=progress,
                context=f"k={k} p={p}->p={p + 1} continuous ladder",
            )
            progress.emit(
                f"k={k} p={p} -> p={p + 1}: continuous free-site packages "
                f"raw={cont_attempted}, unique={len(generated_decorated)}"
            )
            audits.append(
                SweepAudit(
                    k=k,
                    operation="continuous_passivation",
                    p_from=p,
                    p_to=p + 1,
                    source_count=len(continuous_unique),
                    raw_count=cont_attempted,
                    valid_count=len(generated_decorated),
                    symmetry_duplicate_count=max(
                        0, cont_attempted - len(generated_decorated)
                    ),
                    stage_counts={"ligand_enumerations": 0, "free_site_only": 1},
                )
            )
        else:
            generated_decorated = []
        p += 1
    # Strip-parent validation is an audit of the skeleton DAG, not required to
    # produce retained structures.  At large k it walks every p-bin with
    # frontier certificates and can dominate wall time after the last bin
    # (exactly the "hung after retained=..." symptom).  Skip for guided maps
    # and for k > 2; exact small-k still gets the audit.
    if spec.mode == "exact" and k <= 2:
        progress.emit(f"k={k}: strip-parent validation started")
        audits.extend(
            _validate_strip_parents(
                skeleton_registry,
                k=k,
                model=model,
                spec=spec,
            )
        )
        progress.emit(f"k={k}: strip-parent validation complete")
    else:
        progress.emit(
            f"k={k}: strip-parent validation skipped "
            f"(mode={spec.mode}, k={k}; audit-only and expensive at high p)"
        )
    progress.emit(
        f"k={k}: DAG complete, bins={len(retained_registry)}, "
        f"skeletons={sum(len(items) for items in skeleton_registry.values())}, "
        f"retained={sum(len(items) for items in retained_registry.values())}"
    )
    if checkpoint_dir is not None:
        write_discarded = k <= spec.discarded_through_k
        write_nucleation_checkpoint(
            root=checkpoint_dir,
            spec=spec,
            k=k,
            retained=retained_registry,
            discarded=discarded_registry if write_discarded else {},
            skeletons=skeleton_registry,
            discarded_counts={
                p_key: len(recs)
                for p_key, recs in discarded_registry.items()
            },
            mark_done=True,
            last_completed_p=(
                max(retained_registry) if retained_registry else -1
            ),
            p_cap=p_cap,
            max_inherited=max_inherited,
            inherited=inherited,
        )
        progress.emit(
            f"checkpoint: k={k} complete (DONE) under {checkpoint_dir}"
        )
    return retained_registry, discarded_registry, skeleton_registry, audits


def _select_bin(
    records: Sequence[ClusterRecord],
    k: int,
    p: int,
    model: _LatticeModel,
    spec: NucleationSpec,
    progress: _ProgressReporter,
) -> Tuple[List[ClusterRecord], List[ClusterRecord], int]:
    for record in records:
        _assert_atoms_match_bin(
            record.atoms,
            k=k,
            p=p,
            spec=spec,
            context="bin selection",
        )
    progress.emit(
        f"k={k} p={p}: bin symmetry filtering started, "
        f"candidates={len(records)}"
    )
    unique, duplicates = _unique_records(
        records,
        progress=progress,
        context=f"k={k} p={p} bin selection",
    )
    progress.emit(
        f"k={k} p={p}: bin symmetry filtering complete, "
        f"unique={len(unique)}, duplicates={duplicates}"
    )
    need_rings_for_policy = spec.passivation_ring_policy != "none"
    for record in unique:
        record.coordination_score = _coordination_score(record, model, spec)
        # Defer expensive ring DFS until we know who is retained (or need rings
        # for passivation_ring_policy on the winning surface layer).
        record.metadata.update(
            _coordination_metadata(
                record, model, spec, include_rings=False
            )
        )
    if not unique:
        return [], [], duplicates
    scores = sorted(
        {record.coordination_score for record in unique}, reverse=True
    )
    retained: List[ClusterRecord] = []
    selected_score: Optional[Tuple[int, ...]] = None
    retained_scores: List[Tuple[int, ...]] = []
    surface_rejected: set[int] = set()
    surface_processed = 0
    surface_started = time.monotonic()
    layers_wanted = max(1, int(spec.retain_score_layers))
    progress.emit(
        f"k={k} p={p}: ranked surface screening started, "
        f"unique={len(unique)}, score_layers={len(scores)}, "
        f"retain_score_layers={layers_wanted}"
    )
    for score in scores:
        if len(retained_scores) >= layers_wanted:
            break
        score_group = [
            record for record in unique if record.coordination_score == score
        ]
        valid_group: List[ClusterRecord] = []
        for record in score_group:
            _attach_surface_geometry(record, model, spec)
            surface_processed += 1
            if record.metadata["surface_geometry"].get(
                "projection_valid", False
            ):
                valid_group.append(record)
            else:
                surface_rejected.add(id(record))
            progress.heartbeat(
                f"k={k} p={p}: surface processed={surface_processed}/"
                f"{len(unique)}, valid_in_layer={len(valid_group)}, "
                f"rejected={len(surface_rejected)}, "
                f"elapsed={time.monotonic() - surface_started:.1f}s"
            )
        if not valid_group:
            continue
        if need_rings_for_policy:
            for record in valid_group:
                record.metadata["rings"] = _ring_counts_for_record(
                    record, spec
                )
        before_rings = len(valid_group)
        # Ring policy ranks within a score layer; only the first (best) layer
        # is filtered by passivation_ring_policy so lower layers stay a pure
        # coordination band for lineage.
        if not retained_scores and spec.passivation_ring_policy != "none":
            layer_kept = _apply_passivation_ring_policy(valid_group, spec)
            if len(layer_kept) < before_rings:
                progress.emit(
                    f"k={k} p={p}: passivation_ring_policy="
                    f"{spec.passivation_ring_policy} "
                    f"kept {len(layer_kept)}/{before_rings} "
                    f"(Cl rings {list(spec.ring_lengths)}; "
                    "bridges still use min_bridged_host_cn)"
                )
        else:
            layer_kept = valid_group
        if not layer_kept:
            continue
        if selected_score is None:
            selected_score = score
        retained_scores.append(score)
        retained.extend(layer_kept)
    if retained and spec.retain_max_per_bin > 0:
        # Prefer higher score, then stable structure key, then cap.
        retained.sort(
            key=lambda record: (
                record.coordination_score,
                _record_sort_key(record),
            ),
            reverse=True,
        )
        if len(retained) > spec.retain_max_per_bin:
            progress.emit(
                f"k={k} p={p}: retain_max_per_bin="
                f"{spec.retain_max_per_bin} truncated "
                f"{len(retained)} -> {spec.retain_max_per_bin}"
            )
            retained = retained[: spec.retain_max_per_bin]
    progress.emit(
        f"k={k} p={p}: ranked surface screening complete, "
        f"processed={surface_processed}/{len(unique)}, "
        f"retained={len(retained)}, layers={len(retained_scores)}, "
        f"rejected={len(surface_rejected)}, "
        f"elapsed={time.monotonic() - surface_started:.1f}s"
    )
    retained_ids = {id(record) for record in retained}
    discarded = [record for record in unique if id(record) not in retained_ids]
    if surface_rejected:
        progress.emit(
            f"k={k} p={p}: surface slot gate rejected "
            f"{len(surface_rejected)} candidates"
        )
    compliant_exists = any(
        bool(record.metadata.get("min_cn_compliant", False))
        for record in unique
    )
    for record in retained:
        record.selection_status = "retained"
        record.selection_reason = (
            "min_cn_compliant"
            if record.metadata.get("min_cn_compliant", False)
            else "minimum_cn_shortfall"
        )
        if not record.metadata.get("rings"):
            record.metadata["rings"] = _ring_counts_for_record(record, spec)
    # Cheap rings only for discarded that are kept in the registry (k small).
    if k <= spec.discarded_through_k:
        for record in discarded:
            if not record.metadata.get("rings"):
                record.metadata["rings"] = _ring_counts_for_record(
                    record, spec
                )
    for record in discarded:
        record.selection_status = "discarded"
        if id(record) in surface_rejected:
            record.selection_reason = "surface_slot_conflict"
            record.metadata["surface_selection_rejected"] = True
        else:
            if (
                id(record) not in surface_rejected
                and spec.passivation_ring_policy != "none"
                and selected_score is not None
                and record.coordination_score == selected_score
            ):
                record.selection_reason = "lower_cl_ring_rank"
            else:
                record.selection_reason = (
                    "min_cn_violation"
                    if compliant_exists
                    and not record.metadata.get("min_cn_compliant", False)
                    else "lower_coordination_rank"
                )
        reference_score = selected_score or scores[0]
        record.metadata["best_score_in_bin"] = list(reference_score)
    for record in retained:
        if (
            selected_score is not None
            and record.coordination_score != selected_score
        ):
            record.metadata["retain_band"] = "score_layer_band"
        elif selected_score is not None:
            record.metadata["retain_band"] = "top_score"
    for record in [*retained, *discarded]:
        _attach_lineage_metadata(record, model, spec)
    retained.sort(key=_record_sort_key)
    discarded.sort(key=_record_sort_key)
    return retained, discarded, duplicates


def _assign_structure_ids(
    retained: Sequence[ClusterRecord],
    discarded: Sequence[ClusterRecord],
    k: int,
    p: int,
) -> None:
    for index, record in enumerate(retained, start=1):
        record.structure_id = f"k{k:03d}_p{p:03d}_retained_iso{index:04d}"
    for index, record in enumerate(discarded, start=1):
        record.structure_id = f"k{k:03d}_p{p:03d}_discarded_iso{index:04d}"



def _enumerate_ligand_states(
    atoms_without_ligands: Sequence[AtomRecord],
    ligand_count: int,
    model: _LatticeModel,
    spec: NucleationSpec,
    progress: _ProgressReporter,
    cache: _EnumerationCache,
    *,
    context: str,
) -> Tuple[List[_State], int, Dict[str, int], Dict[str, int]]:
    """Distribute indistinguishable ligands over all exact outward host sites.

    The precursor units determine stoichiometry, not persistent ligand
    ownership.  Once the required number of centers exists, their ligands may
    occupy any compatible cation-sublattice virtual site.  Every geometrically
    present allowed bond is formed automatically, so bridging occurs only when
    two hosts share one exact lattice site.

    Latent bridges are *not* applied here.  ``_enumerate_skeleton_bin`` runs the
    bridge search once per symmetry-unique base after cross-skeleton merging,
    which is strictly less work than running it per skeleton.
    """

    base_atoms = _normalize_atoms(atoms_without_ligands)
    base = _make_core_graph(base_atoms, model, spec)
    reasons: Dict[str, int] = {}
    if not _base_coordination_valid(base, model, spec):
        return [], 0, {"base_overcoordination_or_missing_anion": 1}, {}
    cache_key: Tuple[object, ...] = (
        ligand_count,
        tuple(
            (
                atom.symbol,
                atom.role,
                _position_key(
                    np.asarray(atom.coordinates, dtype=float),
                    model.site_tolerance,
                ),
            )
            for atom in base.atoms
        ),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        cached_states, attempted, cached_reasons = cached
        progress.emit(
            f"{context}: cache hit, reused={len(cached_states)}, "
            f"raw_assignments={attempted}"
        )
        return list(cached_states), attempted, dict(cached_reasons), {
            "capacity_pruned": 0,
            "symmetry_pruned_before_embedding": 0,
            "embedded": 0,
            "chemically_valid": len(cached_states),
            "cache_hits": 1,
            "cached_representatives": len(cached_states),
        }
    if ligand_count == 0:
        valid = [base] if nx.is_connected(base.graph) else []
        cache[cache_key] = (tuple(valid), 1, dict(reasons))
        return valid, 1, reasons, {
            "capacity_pruned": 0,
            "symmetry_pruned_before_embedding": 0,
            "embedded": 1,
            "chemically_valid": len(valid),
        }

    center_count = sum(
        atom.role == "precursor_center" for atom in base.atoms
    )
    expected = center_count * spec.precursor.ligand_count
    if ligand_count != expected:
        return [], 0, {"wrong_ligand_count": 1}, {}

    sites = _all_outward_ligand_sites(base, model, spec)
    assignment_count = (
        math.comb(len(sites), ligand_count)
        if len(sites) >= ligand_count
        else 0
    )
    progress.emit(
        f"{context}: sites={len(sites)}, ligands={ligand_count}, "
        f"assignments={assignment_count}"
    )
    if assignment_count == 0:
        return [], 0, {"insufficient_outward_ligand_sites": 1}, {}
    states: List[_State] = []
    accepted_certificates: set[Tuple[object, ...]] = set()
    node_order = sorted(base.graph.nodes)
    node_indices = {node: index for index, node in enumerate(node_order)}
    automorphism_permutations, automorphism_cache_hit = _graph_automorphisms(
        base.graph, cache
    )
    capacity_pruned = 0
    identical_host_pruned = 0
    symmetry_pruned = 0
    embedded_count = 0
    started = time.monotonic()

    mapped_site_hosts: List[List[Tuple[int, ...]]] = [
        [
            tuple(
                sorted(
                    permutation[node_indices[host]]
                    for host in vacancy.hosts
                )
            )
            for vacancy in sites
        ]
        for permutation in automorphism_permutations
    ]

    def canonical_subset(
        subset: Sequence[int],
    ) -> Tuple[Tuple[int, ...], ...]:
        return min(
            tuple(sorted(mapped[index] for index in subset))
            for mapped in mapped_site_hosts
        )

    # Generate one representative per orbit at every ligand-count level.
    # This avoids walking C(N,L) raw site subsets merely to discover that
    # almost all of them have the same host-incidence pattern.
    empty_additions = (0,) * len(base.atoms)
    orbit_level: Dict[
        Tuple[Tuple[int, ...], ...],
        Tuple[Tuple[int, ...], Tuple[int, ...]],
    ] = {(): ((), empty_additions)}
    orbit_extensions = 0
    duplicate_extensions_pruned = 0
    orbit_capacity_pruned = 0
    for depth in range(1, ligand_count + 1):
        next_level: Dict[
            Tuple[Tuple[int, ...], ...],
            Tuple[Tuple[int, ...], Tuple[int, ...]],
        ] = {}
        raw_candidates_seen: set[Tuple[int, ...]] = set()
        for subset, previous_additions in orbit_level.values():
            used = set(subset)
            for site_index in range(len(sites)):
                if site_index in used:
                    continue
                orbit_extensions += 1
                candidate = tuple(sorted((*subset, site_index)))
                if candidate in raw_candidates_seen:
                    duplicate_extensions_pruned += 1
                    continue
                raw_candidates_seen.add(candidate)
                vacancy = sites[site_index]
                if (
                    not vacancy.hosts
                    or len(vacancy.hosts)
                    > spec.graph_rules.max_cn[spec.precursor.ligand]
                ):
                    orbit_capacity_pruned += 1
                    continue
                additions = list(previous_additions)
                invalid = False
                for host in vacancy.hosts:
                    additions[host] += 1
                    if (
                        base.graph.degree[host] + additions[host]
                        > spec.graph_rules.max_cn[base.atoms[host].symbol]
                    ):
                        invalid = True
                        break
                if invalid:
                    orbit_capacity_pruned += 1
                    continue
                signature = canonical_subset(candidate)
                old = next_level.get(signature)
                if old is None or candidate < old[0]:
                    next_level[signature] = (candidate, tuple(additions))
        orbit_level = next_level
        progress.emit(
            f"{context}: orbit level={depth}/{ligand_count}, "
            f"representatives={len(orbit_level)}, "
            f"extensions={orbit_extensions}, "
            f"duplicate_extensions_pruned={duplicate_extensions_pruned}, "
            f"capacity_pruned={orbit_capacity_pruned}",
            verbose_only=True,
        )
        if not orbit_level:
            break

    def report_assignment_progress(processed: int) -> None:
        if progress.callback is None:
            return
        elapsed = time.monotonic() - started
        progress.heartbeat(
            f"{context}: orbit processed={processed}/{len(orbit_assignments)} "
            f"({100.0 * processed / max(1, len(orbit_assignments)):.1f}%), "
            f"theoretical={assignment_count}, "
            f"capacity_pruned={capacity_pruned}, "
            f"identical_host_pruned={identical_host_pruned}, "
            f"isomorphic_pruned={symmetry_pruned}, "
            f"embedded={embedded_count}, "
            f"valid={len(states)}, "
            f"elapsed={elapsed:.1f}s"
        )

    orbit_assignments = [
        tuple(sites[index] for index in subset)
        for subset, _additions in orbit_level.values()
    ]
    identical_host_pruned = max(0, assignment_count - len(orbit_assignments))
    capacity_pruned = orbit_capacity_pruned
    for processed, assignment in enumerate(orbit_assignments, start=1):
        host_additions: Dict[int, int] = {}
        for vacancy in assignment:
            for host in vacancy.hosts:
                host_additions[host] = host_additions.get(host, 0) + 1
        if any(
            base.graph.degree[host] + addition
            > spec.graph_rules.max_cn[base.atoms[host].symbol]
            for host, addition in host_additions.items()
        ) or any(
            not vacancy.hosts
            or len(vacancy.hosts)
            > spec.graph_rules.max_cn[spec.precursor.ligand]
            for vacancy in assignment
        ):
            capacity_pruned += 1
            report_assignment_progress(processed)
            continue

        # Derive the embedded graph once and deduplicate on that same graph.
        # Hashing a separately hand-built graph risked keying a bucket on a
        # connectivity that differed from the one stored in it.
        embedded = _extend_core_graph(
            base,
            [
                AtomRecord(
                    atom_id=len(base.atoms) + offset,
                    symbol=spec.precursor.ligand,
                    coordinates=tuple(
                        float(value) for value in vacancy.position
                    ),
                    role="precursor_ligand",
                    unit_id=None,
                )
                for offset, vacancy in enumerate(assignment)
            ],
            model,
            spec,
        )
        certificate = _graph_certificate(embedded.graph)
        if certificate in accepted_certificates:
            symmetry_pruned += 1
            report_assignment_progress(processed)
            continue

        embedded_count += 1
        if not _state_valid(embedded, model, spec):
            _increment(reasons, "invalid_coordination_or_geometry")
        else:
            accepted_certificates.add(certificate)
            states.append(embedded)
        report_assignment_progress(processed)
    unique_states, duplicates = _unique_states(
        states,
        progress=progress,
        context=f"{context} embeddings",
    )
    if duplicates:
        reasons["embedding_symmetry_duplicates"] = duplicates
    progress.emit(
        f"{context}: enumeration complete, raw_assignments={assignment_count}, "
        f"capacity_pruned={capacity_pruned}, "
        f"identical_host_pruned={identical_host_pruned}, "
        f"isomorphic_pruned={symmetry_pruned}, "
        f"embedded={embedded_count}, "
        f"orbit_representatives={len(orbit_assignments)}, "
        f"valid_unique={len(unique_states)}, "
        f"post_embedding_duplicates={duplicates}"
    )
    cache[cache_key] = (
        tuple(unique_states),
        assignment_count,
        dict(reasons),
    )
    return unique_states, assignment_count, reasons, {
        "capacity_pruned": capacity_pruned,
        "automorphism_cache_hits": automorphism_cache_hit,
        "theoretical_assignments": assignment_count,
        "orbit_extensions": orbit_extensions,
        "duplicate_orbit_extensions_pruned": duplicate_extensions_pruned,
        "orbit_representatives": len(orbit_assignments),
        "identical_host_pruned": identical_host_pruned,
        "symmetry_pruned_before_embedding": (
            identical_host_pruned + symmetry_pruned
        ),
        "isomorphic_pruned": symmetry_pruned,
        "embedded": embedded_count,
        "chemically_valid": len(states),
    }


def _all_outward_ligand_sites(
    base: _State,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> List[_Vacancy]:
    """Merge exact ligand vacancies from every cation-like atom."""

    vacancies: List[_Vacancy] = []
    for atom in base.atoms:
        if atom.role not in {"core_cation", "precursor_center"}:
            continue
        for position in _exact_outward_ligand_sites(
            atom.atom_id, base, model, spec
        ):
            _merge_vacancy(
                vacancies,
                spec.precursor.ligand,
                position,
                atom.atom_id,
                model.site_tolerance,
            )
    vacancies.sort(
        key=lambda vacancy: _position_key(
            vacancy.position, model.site_tolerance
        )
    )
    return vacancies


def _exact_outward_ligand_sites(
    host: int,
    base: _State,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> List[FloatArray]:
    """Return unoccupied cation-template sites pointing out of the cluster."""

    positions = _atom_positions(base.atoms)
    actual = [
        positions[neighbor] - positions[host]
        for neighbor in base.graph.neighbors(host)
    ]
    candidates: List[FloatArray] = []
    for environment in _best_environments(
        model.environments[model.core.cation], actual
    ):
        for vector in environment:
            candidate = positions[host] + np.asarray(vector, dtype=float)
            if _position_occupied(candidate, positions, model.site_tolerance):
                continue
            if not _is_outward_site(
                positions[host], candidate, positions, model.site_tolerance
            ):
                continue
            if not any(
                np.linalg.norm(candidate - old) < model.site_tolerance
                for old in candidates
            ):
                candidates.append(candidate)
    candidates.sort(key=lambda point: _position_key(point, model.site_tolerance))
    return candidates


def _is_outward_site(
    host: FloatArray,
    candidate: FloatArray,
    inorganic_positions: FloatArray,
    tolerance: float,
) -> bool:
    """Apply a convex-envelope test with a radial fallback for small clusters."""

    centroid = inorganic_positions.mean(axis=0)
    radial = host - centroid
    direction = candidate - host
    if np.linalg.norm(radial) > tolerance and float(np.dot(direction, radial)) <= 0:
        return False

    centered = inorganic_positions - centroid
    if len(inorganic_positions) < 4 or np.linalg.matrix_rank(centered) < 3:
        return True
    try:
        hull = ConvexHull(inorganic_positions)
    except QhullError:
        return True
    inside_or_on_hull = all(
        float(np.dot(equation[:3], candidate) + equation[3]) <= tolerance
        for equation in hull.equations
    )
    return not inside_or_on_hull


def _bridge_candidate_topology_key(
    candidate: _BridgeCandidate,
    atom_mapping: Optional[Mapping[int, int]] = None,
) -> Tuple[object, ...]:
    """Describe one bridge opportunity without using absolute coordinates."""

    mapping = atom_mapping or {}

    def mapped(atom_id: int) -> int:
        return mapping.get(atom_id, atom_id)

    return (
        candidate.rule.ligand,
        candidate.rule.host,
        candidate.rule.shared_neighbor,
        candidate.mode,
        mapped(candidate.primary),
        mapped(candidate.host),
        (
            mapped(candidate.shared_neighbor)
            if candidate.shared_neighbor is not None
            else -1
        ),
        tuple(sorted(mapped(host) for host in candidate.virtual_hosts)),
    )


def _bridge_candidate_permutations(
    state: _State,
    candidates: Sequence[_BridgeCandidate],
    terminal_by_primary: Mapping[int, Sequence[int]],
    cache: Optional[_EnumerationCache],
) -> Tuple[Tuple[Tuple[int, ...], ...], int]:
    """Induce safe bridge-candidate permutations from base automorphisms.

    Terminal ligand leaves are compressed into a donor-supply node label.
    This prevents factorial permutations of chemically indistinguishable Cl
    leaves from dominating the automorphism calculation.  An atom
    automorphism is retained only when it maps the complete bridge-candidate
    set bijectively onto itself.
    """

    identity = tuple(range(len(candidates)))
    if len(candidates) < 4:
        return (identity,), 0
    keys: Dict[Tuple[object, ...], int] = {}
    for index, candidate in enumerate(candidates):
        key = _bridge_candidate_topology_key(candidate)
        if key in keys:
            # Multiple geometrically distinct opportunities with identical
            # topology cannot be mapped safely without lattice operations.
            return (identity,), 0
        keys[key] = index

    terminal_ids = {
        ligand
        for ligands in terminal_by_primary.values()
        for ligand in ligands
    }
    environment = nx.Graph()
    for atom in state.atoms:
        if atom.atom_id in terminal_ids:
            continue
        terminal_supply = len(terminal_by_primary.get(atom.atom_id, ()))
        environment.add_node(
            atom.atom_id,
            element=f"{atom.symbol}|terminal_supply={terminal_supply}",
        )
    for left, right, data in state.graph.edges(data=True):
        if left not in environment or right not in environment:
            continue
        environment.add_edge(
            left,
            right,
            bond_order=data.get("bond_order", 1),
        )
    local_signatures = [
        (
            environment.nodes[node]["element"],
            environment.degree[node],
            tuple(
                sorted(
                    environment.nodes[neighbor]["element"]
                    for neighbor in environment.neighbors(node)
                )
            ),
        )
        for node in environment.nodes
    ]
    if len(set(local_signatures)) == len(local_signatures):
        return (identity,), 0
    node_order = sorted(environment.nodes)
    node_indices = {node: index for index, node in enumerate(node_order)}
    atom_permutations, cache_hit = _graph_automorphisms(environment, cache)

    induced: set[Tuple[int, ...]] = {identity}
    for permutation in atom_permutations:
        atom_mapping = {
            node: node_order[permutation[node_indices[node]]]
            for node in node_order
        }
        mapped_indices: List[int] = []
        for candidate in candidates:
            mapped_index = keys.get(
                _bridge_candidate_topology_key(candidate, atom_mapping)
            )
            if mapped_index is None:
                break
            mapped_indices.append(mapped_index)
        else:
            if len(set(mapped_indices)) == len(candidates):
                induced.add(tuple(mapped_indices))
    return tuple(sorted(induced)), cache_hit


def _greedy_incumbent_state(
    skeleton: _State,
    ligand_count: int,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> Optional[_State]:
    """Place the ligand shell by the passivation module's construction order.

    One structure, built the way a chemist passivates a faceted surface and the
    way ``_priority3_balance_positive_q_add`` repairs a Wulff cut: prefer a
    bridging site (one Cl, two bonds), then the most starved host, ties broken
    deterministically by lattice position.  The result is used two ways --

    * its score seeds the k>2 incumbent so the reachable-score gate can prune
      from the first base (exact: the incumbent is an achieved structure and
      pruning is strict, so nothing tied-or-better is ever lost);
    * per bin, whether this single guided structure already attains the
      selected score is recorded as the *greedy gap* audit -- the k at which
      the gap closes is where a guided mode becomes empirically sufficient.

    Returns the undecorated-plus-ligands state, or ``None`` when the greedy
    order dead-ends before placing every ligand (the exact enumeration is then
    the only authority).
    """

    state = skeleton
    for _placement in range(ligand_count):
        sites = _all_outward_ligand_sites(state, model, spec)
        best_site: Optional[_Vacancy] = None
        best_key: Optional[Tuple[int, int, Tuple[int, int, int]]] = None
        for site in sites:
            if (
                not site.hosts
                or len(site.hosts)
                > spec.graph_rules.max_cn[spec.precursor.ligand]
            ):
                continue
            if any(
                state.graph.degree[host] + 1
                > spec.graph_rules.max_cn[state.atoms[host].symbol]
                for host in site.hosts
            ):
                continue
            deficit = max(
                spec.graph_rules.max_cn[state.atoms[host].symbol]
                - state.graph.degree[host]
                for host in site.hosts
            )
            key = (
                -len(site.hosts),
                -deficit,
                _position_key(site.position, model.site_tolerance),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_site = site
        if best_site is None:
            return None
        state = _extend_core_graph(
            state,
            [
                AtomRecord(
                    atom_id=len(state.atoms),
                    symbol=spec.precursor.ligand,
                    coordinates=tuple(
                        float(value) for value in best_site.position
                    ),
                    role="precursor_ligand",
                    unit_id=None,
                )
            ],
            model,
            spec,
        )
    if not _state_valid(state, model, spec):
        return None
    return state


def _bridge_candidate_arcs(
    state: _State,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> Tuple[Dict[int, List[int]], List[_BridgeCandidate]]:
    """Derive every latent-bridge opportunity of one base.

    Returns ``(terminal_by_primary, candidates)``: the movable terminal ligands
    per donor cation, and each possible donor-to-acceptor arc in both motifs
    (rhombus over a shared occupied anion, and exact shared vacant CIF site).
    Shared by the bridge enumerator, the reachable-score gate and the greedy
    incumbent so all three see the same opportunity set.
    """

    rules = {rule.ligand: rule for rule in spec.graph_rules.bridge_rules}
    terminal_by_primary: Dict[int, List[int]] = {}
    candidates: List[_BridgeCandidate] = []
    if not rules:
        return terminal_by_primary, candidates
    for primary_atom in state.atoms:
        matching_rules = [
            rule for rule in rules.values()
            if primary_atom.symbol == rule.host
        ]
        if not matching_rules:
            continue
        rule = matching_rules[0]
        primary = primary_atom.atom_id
        terminal_ligands = sorted(
            neighbor
            for neighbor in state.graph.neighbors(primary)
            if state.atoms[neighbor].symbol == rule.ligand
            and state.graph.degree[neighbor]
            < spec.graph_rules.max_cn[rule.ligand]
        )
        if not terminal_ligands:
            continue
        terminal_by_primary[primary] = terminal_ligands
        pair_candidates: Dict[int, int] = {}
        for shared in state.graph.neighbors(primary):
            if state.atoms[shared].symbol != rule.shared_neighbor:
                continue
            for second_host in state.graph.neighbors(shared):
                if second_host == primary:
                    continue
                if state.atoms[second_host].symbol != rule.host:
                    continue
                if (
                    state.graph.degree[second_host]
                    >= spec.graph_rules.max_cn[rule.host]
                ):
                    continue
                pair_candidates.setdefault(second_host, shared)
        for second_host, shared in sorted(pair_candidates.items()):
            candidates.append(
                _BridgeCandidate(
                    primary=primary,
                    host=second_host,
                    rule=rule,
                    mode="shared_occupied_neighbor",
                    shared_neighbor=shared,
                )
            )

    # A common vacant anion site is an exact CIF bridge position.  Generate
    # both donor directions because only the donor must supply a terminal Cl;
    # the receiving Cd merely needs one free coordination slot.
    rule_by_host = {rule.host: rule for rule in rules.values()}
    for vacancy in _anion_vacancies_on_cations(state, model, spec):
        if len(vacancy.hosts) < 2:
            continue
        site = tuple(float(value) for value in vacancy.position)
        for first, second in combinations(sorted(vacancy.hosts), 2):
            for primary, host in ((first, second), (second, first)):
                rule = rule_by_host.get(state.atoms[primary].symbol)
                if rule is None or primary not in terminal_by_primary:
                    continue
                if state.atoms[host].symbol != rule.host:
                    continue
                if state.graph.degree[host] >= spec.graph_rules.max_cn[rule.host]:
                    continue
                candidates.append(
                    _BridgeCandidate(
                        primary=primary,
                        host=host,
                        rule=rule,
                        mode="shared_vacant_cif_site",
                        virtual_site=site,
                        virtual_hosts=tuple(sorted(vacancy.hosts)),
                    )
                )
    return terminal_by_primary, candidates


def _latent_bridge_greedy_max(
    state: _State,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> List[_State]:
    """Single max-bridge shell via greedy fill (for continuous growth).

    Full ``_latent_bridge_variants`` enumerates *all* maximum-cardinality bridge
    sets and can explode after continuous attach (connected ≫ raw).  Continuous
    decoration only needs one highly bridged shell: sort opportunities by
    preference and add every arc that still fits capacity, pair uniqueness and
    ``min_bridged_host_cn``.
    """

    rules = {rule.ligand: rule for rule in spec.graph_rules.bridge_rules}
    if not rules:
        return [state]
    terminal_by_primary, candidates = _bridge_candidate_arcs(state, model, spec)
    if not candidates:
        return [state]

    min_host_cn = max(
        (rule.min_bridged_host_cn for rule in rules.values()), default=1
    )
    atom_count = len(state.atoms)
    base_degree = [state.graph.degree[i] for i in range(atom_count)]
    used_supply = [0] * atom_count
    added_degree = [0] * atom_count
    pair_used: set[Tuple[int, int]] = set()
    # Existing bridges already occupy host pairs.
    for atom in state.atoms:
        if atom.symbol not in rules:
            continue
        hosts = [
            neighbor
            for neighbor in state.graph.neighbors(atom.atom_id)
            if state.atoms[neighbor].symbol == rules[atom.symbol].host
        ]
        if len(hosts) == 2:
            pair_used.add((min(hosts), max(hosts)))

    # Prefer exact CIF bridges, then undercoordinated acceptors, then donors
    # that already meet min_host_cn (so the rule does not block them).
    def sort_key(cand: _BridgeCandidate) -> Tuple[int, int, int, int, int]:
        host_def = max(
            0,
            spec.graph_rules.max_cn[state.atoms[cand.host].symbol]
            - base_degree[cand.host],
        )
        primary_ok = 0 if base_degree[cand.primary] >= min_host_cn else 1
        return (
            0 if cand.mode == "shared_vacant_cif_site" else 1,
            primary_ok,
            -host_def,
            cand.primary,
            cand.host,
        )

    ordered = sorted(candidates, key=sort_key)
    selected: List[_BridgeCandidate] = []
    supply_limit = {
        primary: len(ligands)
        for primary, ligands in terminal_by_primary.items()
    }
    for cand in ordered:
        pair = (
            (cand.primary, cand.host)
            if cand.primary < cand.host
            else (cand.host, cand.primary)
        )
        if pair in pair_used:
            continue
        if used_supply[cand.primary] >= supply_limit.get(cand.primary, 0):
            continue
        host_cap = (
            spec.graph_rules.max_cn[state.atoms[cand.host].symbol]
            - base_degree[cand.host]
        )
        if added_degree[cand.host] >= host_cap:
            continue
        # Finished-structure min CN: donor does not rise; acceptor +1.
        donor_final = base_degree[cand.primary]
        acceptor_final = base_degree[cand.host] + added_degree[cand.host] + 1
        if min_host_cn > 1 and (
            donor_final < min_host_cn or acceptor_final < min_host_cn
        ):
            continue
        selected.append(cand)
        used_supply[cand.primary] += 1
        added_degree[cand.host] += 1
        pair_used.add(pair)

    if not selected:
        return [state]

    positions = _atom_positions(state.atoms)
    graph = state.graph.copy()
    selected_by_primary: Dict[int, List[_BridgeCandidate]] = {}
    for cand in selected:
        selected_by_primary.setdefault(cand.primary, []).append(cand)
    for primary, choices in sorted(selected_by_primary.items()):
        choices.sort(
            key=lambda item: (
                0 if item.mode == "shared_vacant_cif_site" else 1,
                item.host,
                item.shared_neighbor if item.shared_neighbor is not None else -1,
                item.virtual_site or (),
            )
        )
        terminals = list(terminal_by_primary[primary])
        for ligand, candidate in zip(terminals, choices):
            graph.add_edge(
                ligand,
                candidate.host,
                kind="surface_bridge",
                bond_order=1,
                bridge_mode=candidate.mode,
                shared_neighbor=candidate.shared_neighbor,
                virtual_site=candidate.virtual_site,
                surface_angle_deg=(
                    candidate.rule.surface_angle_deg
                    if candidate.mode == "shared_occupied_neighbor"
                    else None
                ),
                native_distance=float(
                    np.linalg.norm(positions[ligand] - positions[candidate.host])
                ),
                primary_host=primary,
                primary_cn_before_bridge=state.graph.degree[primary],
                secondary_cn_before_bridge=state.graph.degree[candidate.host],
            )
    if any(
        graph.degree[atom.atom_id] > spec.graph_rules.max_cn[atom.symbol]
        for atom in state.atoms
    ):
        return [state]
    return [_State(state.atoms, graph, state.geometry_residual)]


def _latent_bridge_variants(
    state: _State,
    model: _LatticeModel,
    spec: NucleationSpec,
    *,
    prune_dominated: bool = False,
    cache: Optional[_EnumerationCache] = None,
    stats_out: Optional[MutableMapping[str, int]] = None,
) -> List[_State]:
    """Return the baseline and maximum valid latent bridge graphs.

    Two bridge motifs are enumerated.  An ``occupied_shared_neighbor`` bridge
    closes the DFT-motivated Cd--Se--Cd--Cl rhombus and is projected off the
    construction lattice.  A ``shared_vacant_cif_site`` bridge instead moves a
    terminal ligand onto an exact anion site shared by two Cd tetrahedra.  The
    latter closes longer Cd--Se lattice paths without requiring the Cd pair to
    share an already occupied Se.
    """

    rules = {rule.ligand: rule for rule in spec.graph_rules.bridge_rules}
    if not rules:
        return [state]
    terminal_by_primary, candidates = _bridge_candidate_arcs(state, model, spec)
    if not candidates:
        return [state]

    candidates.sort(
        key=lambda item: (
            0 if item.mode == "shared_vacant_cif_site" else 1,
            item.primary,
            item.host,
            item.shared_neighbor if item.shared_neighbor is not None else -1,
            item.virtual_site or (),
            item.virtual_hosts,
        )
    )
    # Flatten every per-candidate constant into an indexable array once.  The
    # search below touches these millions of times, and the original attribute
    # lookups plus ``tuple(sorted(...))`` pair keys dominated the bridge stage.
    candidate_count = len(candidates)
    atom_count = len(state.atoms)
    cand_primary = [candidate.primary for candidate in candidates]
    cand_host = [candidate.host for candidate in candidates]
    cand_exact = [
        candidate.mode == "shared_vacant_cif_site" for candidate in candidates
    ]
    pair_ids: Dict[Tuple[int, int], int] = {}
    cand_pair: List[int] = []
    for primary, host in zip(cand_primary, cand_host):
        key = (primary, host) if primary < host else (host, primary)
        cand_pair.append(pair_ids.setdefault(key, len(pair_ids)))
    # Donor supply and acceptor capacity are per atom, so index them per
    # candidate to spare the search a second indirection.
    cand_supply_limit = [
        len(terminal_by_primary[primary]) for primary in cand_primary
    ]
    cand_host_capacity = [
        spec.graph_rules.max_cn[state.atoms[host].symbol]
        - state.graph.degree[host]
        for host in cand_host
    ]

    used_supply = [0] * atom_count
    added_degree = [0] * atom_count
    pair_used = [False] * len(pair_ids)
    for atom in state.atoms:
        if atom.symbol not in rules:
            continue
        hosts = [
            neighbor
            for neighbor in state.graph.neighbors(atom.atom_id)
            if state.atoms[neighbor].symbol == rules[atom.symbol].host
        ]
        if len(hosts) != 2:
            continue
        key = (min(hosts), max(hosts))
        if key in pair_ids:
            pair_used[pair_ids[key]] = True

    best_size = 0

    def can_include(index: int) -> bool:
        return (
            not pair_used[cand_pair[index]]
            and used_supply[cand_primary[index]] < cand_supply_limit[index]
            and added_degree[cand_host[index]] < cand_host_capacity[index]
        )

    def push(index: int) -> None:
        used_supply[cand_primary[index]] += 1
        added_degree[cand_host[index]] += 1
        pair_used[cand_pair[index]] = True

    def pop(index: int) -> None:
        used_supply[cand_primary[index]] -= 1
        added_degree[cand_host[index]] -= 1
        pair_used[cand_pair[index]] = False

    # Epoch markers stand in for the three Python sets the original bound
    # allocated on every call.  The returned value is unchanged.
    seen_primary = [0] * atom_count
    seen_host = [0] * atom_count
    seen_pair = [0] * len(pair_ids)
    epoch = 0

    def remaining_bound(index: int) -> int:
        nonlocal epoch
        epoch += 1
        feasible_count = 0
        primary_slots = 0
        host_slots = 0
        pair_count = 0
        for candidate_index in range(index, candidate_count):
            pair = cand_pair[candidate_index]
            if pair_used[pair]:
                continue
            primary = cand_primary[candidate_index]
            supply_left = cand_supply_limit[candidate_index] - used_supply[primary]
            if supply_left <= 0:
                continue
            host = cand_host[candidate_index]
            capacity_left = (
                cand_host_capacity[candidate_index] - added_degree[host]
            )
            if capacity_left <= 0:
                continue
            feasible_count += 1
            if seen_primary[primary] != epoch:
                seen_primary[primary] = epoch
                primary_slots += supply_left
            if seen_host[host] != epoch:
                seen_host[host] = epoch
                host_slots += capacity_left
            if seen_pair[pair] != epoch:
                seen_pair[pair] = epoch
                pair_count += 1
        return min(feasible_count, primary_slots, pair_count, host_slots)

    def maximize(index: int, selected_count: int) -> None:
        nonlocal best_size
        if selected_count + remaining_bound(index) <= best_size:
            best_size = max(best_size, selected_count)
            return
        if index == candidate_count:
            best_size = max(best_size, selected_count)
            return
        if can_include(index):
            push(index)
            maximize(index + 1, selected_count + 1)
            pop(index)
        maximize(index + 1, selected_count)

    maximize(0, 0)
    if best_size == 0:
        return [state]
    candidate_permutations, automorphism_cache_hit = (
        _bridge_candidate_permutations(
            state, candidates, terminal_by_primary, cache
        )
    )
    symmetry_used = len(candidate_permutations) > 1
    symmetry_stats = {
        "bridge_raw_extensions": 0,
        "bridge_symmetry_pruned": 0,
        "bridge_orbit_representatives": 0,
    }
    positions = _atom_positions(state.atoms)

    def build_graph(
        selected: Sequence[_BridgeCandidate],
    ) -> nx.Graph:
        graph = state.graph.copy()
        selected_by_primary: Dict[
            int, List[_BridgeCandidate]
        ] = {}
        for candidate in selected:
            selected_by_primary.setdefault(candidate.primary, []).append(candidate)
        for primary, choices in sorted(selected_by_primary.items()):
            choices.sort(
                key=lambda item: (
                    0 if item.mode == "shared_vacant_cif_site" else 1,
                    item.host,
                    item.shared_neighbor if item.shared_neighbor is not None else -1,
                    item.virtual_site or (),
                )
            )
            for ligand, candidate in zip(
                terminal_by_primary[primary], choices
            ):
                graph.add_edge(
                    ligand,
                    candidate.host,
                    kind="surface_bridge",
                    bond_order=1,
                    bridge_mode=candidate.mode,
                    shared_neighbor=candidate.shared_neighbor,
                    virtual_site=candidate.virtual_site,
                    surface_angle_deg=(
                        candidate.rule.surface_angle_deg
                        if candidate.mode == "shared_occupied_neighbor"
                        else None
                    ),
                    native_distance=float(
                        np.linalg.norm(
                            positions[ligand] - positions[candidate.host]
                        )
                    ),
                    primary_host=primary,
                    primary_cn_before_bridge=state.graph.degree[primary],
                    secondary_cn_before_bridge=state.graph.degree[candidate.host],
                )
        if any(
            graph.degree[atom.atom_id] > spec.graph_rules.max_cn[atom.symbol]
            for atom in state.atoms
        ):
            raise RuntimeError(
                "internal bridge enumeration error: final CN exceeds maximum"
            )
        return graph

    variants: List[_State] = []
    variant_of: Dict[Tuple[object, ...], int] = {}

    def reset_variants() -> None:
        variants.clear()
        variant_of.clear()

    def store_graph(graph: nx.Graph) -> None:
        # Bridge mode is part of the certificate, so two graphs in the same
        # class really are the same structure and the first is as good as any.
        # (This used to have to prefer the exact-CIF-site member of a class,
        # compensating for an equivalence relation that could not tell the two
        # motifs apart.)
        certificate = _graph_certificate(graph)
        if certificate in variant_of:
            return
        variant_of[certificate] = len(variants)
        variants.append(_State(state.atoms, graph, state.geometry_residual))

    selected: List[int] = []

    def enumerate_maximum(sink: List[Tuple[int, ...]]) -> None:
        def walk(index: int) -> None:
            if len(selected) == best_size:
                sink.append(tuple(selected))
                return
            if index == candidate_count:
                return
            if len(selected) + remaining_bound(index) < best_size:
                return
            if can_include(index):
                push(index)
                selected.append(index)
                walk(index + 1)
                selected.pop()
                pop(index)
            walk(index + 1)

        walk(0)

    # Enumerate the maximum sets exactly once.  The original walked this search
    # again for every score layer, which cost O(layers) full traversals per base
    # on the k>2 path.
    raw_maximum_subsets: List[Tuple[int, ...]] = []
    enumerate_maximum(raw_maximum_subsets)
    if symmetry_used:
        representatives: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
        for subset in raw_maximum_subsets:
            signature = min(
                tuple(sorted(permutation[index] for index in subset))
                for permutation in candidate_permutations
            )
            previous = representatives.get(signature)
            if previous is None or subset < previous:
                representatives[signature] = subset
        maximum_subsets = sorted(representatives.values())
        symmetry_stats["bridge_raw_extensions"] = len(raw_maximum_subsets)
        symmetry_stats["bridge_symmetry_pruned"] = max(
            0, len(raw_maximum_subsets) - len(maximum_subsets)
        )
        symmetry_stats["bridge_orbit_representatives"] = len(maximum_subsets)
    else:
        maximum_subsets = raw_maximum_subsets

    # Enumerating only maximum-cardinality sets was an assumption, not a proof.
    # Two facts make it checkable rather than assumed:
    #
    #   Lemma 1.  For feasible S subset S', score(S) < score(S').  Adding a
    #   bridge raises an acceptor cation and a terminal ligand, so no minimum-CN
    #   shortfall can grow: components 1-3 weakly improve, and the bond count
    #   (component 4) strictly increases.  Lexicographically S' therefore wins.
    #   Hence the optimum is always attained at an inclusion-*maximal* set.
    #
    #   Lemma 2.  If some maximum-cardinality set leaves no minimum-CN violation,
    #   its components 1-3 sit at their absolute optimum (1, 0, 0), so nothing can
    #   beat it there, and it has strictly more bonds than any smaller set.  It is
    #   then provably optimal and maximum-cardinality enumeration is exact.
    #
    # Lemma 2 discharges the assumption outright on most bases; only where a
    # violation survives can a smaller maximal set win, and only there is the
    # fallback below paid for.
    score_context = _BridgeScoreContext(
        state,
        spec,
        cand_primary,
        cand_host,
        cand_exact,
        terminal_by_primary,
    )
    score_for = score_context.score

    # A bridge may be required to leave both of its cations at or above a minimum
    # final coordination.  This is checked on the *finished* arc set, never on how
    # it was built: the donor's coordination is fixed before the bridge exists, so
    # a donor-side rule is trivially evaded by reaching the same structure from a
    # ligand arrangement with a better-coordinated donor -- and the route-merging
    # DAG does exactly that.  The acceptor's coordination, by contrast, depends on
    # how many bridges it accepts, so this cannot be decided until the whole set
    # is known and is not a valid branch prune.
    min_host_cn = max(
        (rule.min_bridged_host_cn for rule in rules.values()), default=1
    )

    def bridged_hosts_meet_min_cn(subset: Sequence[int]) -> bool:
        if min_host_cn <= 1 or not subset:
            return True
        degrees = score_context.degrees(subset)
        return all(
            degrees[cand_primary[index]] >= min_host_cn
            and degrees[cand_host[index]] >= min_host_cn
            for index in subset
        )

    def sub_maximum_contenders(
        best_maximum: Optional[Tuple[int, ...]]
    ) -> List[Tuple[int, ...]]:
        """Feasible sets below maximum cardinality worth keeping.

        Reached when Lemma 2's certificate fails, or when the minimum-bridged-host
        rule rejects the maximum-cardinality sets -- that rule is not monotone in
        the arc set, so a smaller set can satisfy it where every larger one fails,
        and the search cannot prune towards it.  ``best_maximum`` of ``None`` means
        no maximum set survived, so every admissible set is a contender.
        """

        winners: List[Tuple[int, ...]] = []
        chosen: List[int] = []

        def walk(index: int) -> None:
            if index == candidate_count:
                if chosen and len(chosen) < best_size:
                    subset = tuple(chosen)
                    if not bridged_hosts_meet_min_cn(subset):
                        return
                    if best_maximum is None or score_for(subset) >= best_maximum:
                        winners.append(subset)
                return
            if can_include(index):
                push(index)
                chosen.append(index)
                walk(index + 1)
                chosen.pop()
                pop(index)
            walk(index + 1)

        walk(0)
        return winners

    # Drop maximum sets that leave a bridge cation below the minimum coordination.
    rejected_by_min_host_cn = 0
    if min_host_cn > 1:
        admissible = [s for s in maximum_subsets if bridged_hosts_meet_min_cn(s)]
        rejected_by_min_host_cn = len(maximum_subsets) - len(admissible)
        maximum_subsets = admissible

    # Lemma 2's certificate: no surviving minimum-CN violation in the best
    # maximum-cardinality set means maximum-cardinality enumeration is exact.
    maximum_scores = [score_for(subset) for subset in maximum_subsets]
    best_maximum_score = max(maximum_scores) if maximum_scores else None
    exactness_certified = (
        best_maximum_score is not None and best_maximum_score[1] == 0
    )
    # When the minimum-bridged-host rule removed every maximum set, a smaller set
    # may still be admissible -- the rule is not monotone, so the search cannot
    # find those by pruning and the fallback has to enumerate them.
    needs_fallback = best_maximum_score is None and rejected_by_min_host_cn > 0
    sub_maximum_subsets: List[Tuple[int, ...]] = []
    fallback_ran = 0
    if (
        (needs_fallback or (best_maximum_score is not None and not exactness_certified))
        and candidate_count <= _SUB_MAXIMUM_FALLBACK_LIMIT
    ):
        fallback_ran = 1
        sub_maximum_subsets = sub_maximum_contenders(best_maximum_score)
    elif needs_fallback or (
        best_maximum_score is not None and not exactness_certified
    ):
        # Too many opportunities to discharge the assumption here; say so
        # rather than let an unproven restriction pass silently.
        fallback_ran = -1
    if sub_maximum_subsets:
        maximum_subsets = [*maximum_subsets, *sub_maximum_subsets]
    symmetry_stats["bridge_exactness_certified"] = int(exactness_certified)
    symmetry_stats["bridge_sub_maximum_fallback"] = fallback_ran
    symmetry_stats["bridge_sub_maximum_contenders"] = len(sub_maximum_subsets)
    symmetry_stats["bridge_rejected_by_min_host_cn"] = rejected_by_min_host_cn

    def publish_stats(
        *,
        search_count: int,
        dominated_count: int = 0,
    ) -> None:
        if stats_out is None:
            return
        stats_out.update(symmetry_stats)
        stats_out.update(
            {
                "bridge_search_states": search_count,
                "dominated_bridge_variants_pruned": dominated_count,
                "bridge_symmetry_used": int(symmetry_used),
                "bridge_automorphism_count": len(candidate_permutations),
                "bridge_automorphism_cache_hits": automorphism_cache_hit,
            }
        )

    bridge_search_count = len(maximum_subsets)

    if not prune_dominated:
        store_graph(state.graph)
        for subset in maximum_subsets:
            store_graph(build_graph([candidates[index] for index in subset]))
        publish_stats(search_count=bridge_search_count)
        return variants

    # Above k=2 discarded candidates are not exported.  Score every maximum set
    # without building it, then materialize and symmetry-filter only the best
    # layer containing at least one surface-valid graph.  A lower local layer
    # cannot win the global (k,p) selection while that better graph exists.
    subsets_by_score: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
    for subset in maximum_subsets:
        subsets_by_score.setdefault(score_for(subset), []).append(subset)
    baseline_score = score_for(())
    for target_score in sorted(
        {baseline_score, *subsets_by_score}, reverse=True
    ):
        reset_variants()
        if baseline_score == target_score:
            store_graph(state.graph)
        for subset in subsets_by_score.get(target_score, ()):
            store_graph(build_graph([candidates[index] for index in subset]))
        surface_valid: List[_State] = []
        for variant in variants:
            record = _record_from_state(
                variant,
                k=0,
                p=0,
                spec=spec,
                operation="bridge_local_screen",
            )
            _surface, geometry = _precondition_surface_geometry(
                record, model, spec, audit=False
            )
            if geometry.get("projection_valid", False):
                surface_valid.append(variant)
        if surface_valid:
            dominated_count = max(
                0, bridge_search_count + 1 - len(surface_valid)
            )
            publish_stats(
                search_count=bridge_search_count,
                dominated_count=dominated_count,
            )
            for variant in surface_valid:
                variant.graph.graph["bridge_search_states"] = (
                    bridge_search_count
                )
                variant.graph.graph["dominated_bridge_variants_pruned"] = (
                    dominated_count
                )
                # These variants have just passed the gate; record that so the
                # caller need not project the same geometry a second time.  The
                # fallback return below is deliberately left unstamped because
                # it has not been gated.
                variant.graph.graph["surface_gate_valid"] = True
            return surface_valid

    publish_stats(
        search_count=bridge_search_count,
        dominated_count=bridge_search_count,
    )
    return [state]


def _record_from_state(
    state: _State,
    *,
    k: int,
    p: int,
    spec: NucleationSpec,
    operation: str,
    source_ids: Tuple[str, ...] = (),
) -> ClusterRecord:
    graph = state.graph.copy()
    graph.graph.update({"k": k, "p": p})
    bridge_edges = [
        {
            "ligand_atom_id": left
            if state.atoms[left].role == "precursor_ligand"
            else right,
            "host_atom_id": right
            if state.atoms[left].role == "precursor_ligand"
            else left,
            "shared_neighbor_atom_id": data.get("shared_neighbor"),
            "bridge_mode": data.get(
                "bridge_mode", "shared_occupied_neighbor"
            ),
            "virtual_site_position": data.get("virtual_site"),
            "surface_angle_deg": data.get("surface_angle_deg"),
            "native_distance_angstrom": data.get("native_distance"),
            "primary_host_atom_id": data.get("primary_host"),
            "primary_cn_before_bridge": data.get(
                "primary_cn_before_bridge"
            ),
            "secondary_cn_before_bridge": data.get(
                "secondary_cn_before_bridge"
            ),
        }
        for left, right, data in graph.edges(data=True)
        if data.get("kind") == "surface_bridge"
    ]
    return ClusterRecord(
        structure_id="",
        k=k,
        p=p,
        atoms=list(state.atoms),
        graph=graph,
        source_operations=(operation,),
        source_structure_ids=source_ids,
        metadata={
            "formal_charge": _formal_charge(state.atoms, spec.charges),
            "geometry_residual": state.geometry_residual,
            "bridge_edges": bridge_edges,
            "bridge_count": len(bridge_edges),
        },
    )


def _unique_records(
    records: Sequence[ClusterRecord],
    *,
    progress: Optional[_ProgressReporter] = None,
    context: str = "structures",
) -> Tuple[List[ClusterRecord], int]:
    if progress is not None and records:
        progress.emit(
            f"{context}: symmetry filtering {len(records)} candidates",
            verbose_only=True,
        )
    groups: List[List[ClusterRecord]] = []
    group_of: Dict[Tuple[object, ...], int] = {}
    started = time.monotonic()
    for processed, record in enumerate(records, start=1):
        certificate = _graph_certificate(record.graph)
        index = group_of.get(certificate)
        if index is None:
            group_of[certificate] = len(groups)
            groups.append([record])
        else:
            groups[index].append(record)
        if progress is not None and progress.callback is not None:
            progress.heartbeat(
                f"{context}: symmetry processed={processed}/{len(records)} "
                f"({100.0 * processed / len(records):.1f}%), "
                f"classes={len(groups)}, "
                f"elapsed={time.monotonic() - started:.1f}s"
            )
    representatives: List[ClusterRecord] = []
    for group in groups:
        representative = min(group, key=_record_sort_key)
        representative.source_operations = tuple(
            sorted({op for record in group for op in record.source_operations})
        )
        representative.source_structure_ids = tuple(
            sorted(
                {
                    source
                    for record in group
                    for source in record.source_structure_ids
                    if source
                }
            )
        )
        representative.metadata["symmetry_copy_count"] = len(group) - 1
        representatives.append(representative)
    representatives.sort(key=_record_sort_key)
    if progress is not None and records:
        progress.emit(
            f"{context}: symmetry complete, "
            f"classes={len(representatives)}, "
            f"duplicates={len(records) - len(representatives)}",
            verbose_only=True,
        )
    return representatives, len(records) - len(representatives)


def _unique_states(
    states: Sequence[_State],
    *,
    progress: Optional[_ProgressReporter] = None,
    context: str = "embeddings",
) -> Tuple[List[_State], int]:
    if progress is not None and states:
        progress.emit(
            f"{context}: symmetry filtering {len(states)} candidates",
            verbose_only=True,
        )
    unique: List[_State] = []
    seen: set[Tuple[object, ...]] = set()
    started = time.monotonic()
    for processed, state in enumerate(states, start=1):
        certificate = _graph_certificate(state.graph)
        if certificate not in seen:
            seen.add(certificate)
            unique.append(state)
        if progress is not None and progress.callback is not None:
            progress.heartbeat(
                f"{context}: symmetry processed={processed}/{len(states)} "
                f"({100.0 * processed / len(states):.1f}%), "
                f"classes={len(unique)}, "
                f"elapsed={time.monotonic() - started:.1f}s"
            )
    unique.sort(key=lambda state: (_graph_hash(state.graph), state.geometry_residual))
    if progress is not None and states:
        progress.emit(
            f"{context}: symmetry complete, classes={len(unique)}, "
            f"duplicates={len(states) - len(unique)}",
            verbose_only=True,
        )
    return unique, len(states) - len(unique)


@dataclass
class _CandidateClass:
    state: _State
    comparison_graph: nx.Graph
    sources: set[str]


@dataclass
class _CandidateAccumulator:
    """Incrementally merge isomorphic states and their DAG source routes."""

    def __init__(
        self,
        model: Optional[_LatticeModel] = None,
        spec: Optional[NucleationSpec] = None,
        *,
        comparison: str = "frontier",
    ) -> None:
        self.model = model
        self.spec = spec
        # What two candidates must agree on to be considered the same.
        #   "graph"    -- topology alone
        #   "frontier" -- topology plus growth sites (skeleton merging)
        #   "bridges"  -- topology plus the bridge options the geometry allows,
        #                 required for ligand-decorated bases because bridge
        #                 availability is a property of coordinates, not of the
        #                 graph.  Merging on topology alone lost structures.
        self.comparison = comparison
        self.classes: List[_CandidateClass] = []
        self.class_of: Dict[Tuple[object, ...], int] = {}
        self.candidate_count = 0
        self.isomorphism_checks = 0

    @staticmethod
    def _state_key(state: _State) -> Tuple[object, ...]:
        return (
            _graph_hash(state.graph),
            state.geometry_residual,
            tuple(atom.coordinates for atom in state.atoms),
        )

    def add(self, state: _State, sources: Sequence[str]) -> bool:
        """Add a candidate; return whether it created a new symmetry class."""

        self.candidate_count += 1
        if self.model is None or self.spec is None:
            comparison_graph = state.graph
        elif self.comparison == "bridges":
            comparison_graph = _bridge_opportunity_graph(
                state, self.model, self.spec
            )
        elif self.comparison == "frontier":
            comparison_graph = _skeleton_frontier_graph(
                state, self.model, self.spec
            )
        else:
            comparison_graph = state.graph
        certificate = _graph_certificate(comparison_graph)
        index = self.class_of.get(certificate)
        if index is not None:
            self.isomorphism_checks += 1
            group = self.classes[index]
            group.sources.update(source for source in sources if source)
            if self._state_key(state) < self._state_key(group.state):
                group.state = state
                group.comparison_graph = comparison_graph
            return False
        self.class_of[certificate] = len(self.classes)
        self.classes.append(
            _CandidateClass(
                state=state,
                comparison_graph=comparison_graph,
                sources={source for source in sources if source},
            )
        )
        return True

    def result(self) -> List[Tuple[_State, Tuple[str, ...]]]:
        result = [
            (group.state, tuple(sorted(group.sources)))
            for group in self.classes
        ]
        result.sort(key=lambda item: (_graph_hash(item[0].graph), item[1]))
        return result


def _unique_skeleton_candidates(
    candidates: Sequence[Tuple[_State, Tuple[str, ...]]],
    model: Optional[_LatticeModel] = None,
    spec: Optional[NucleationSpec] = None,
    progress: Optional[_ProgressReporter] = None,
    *,
    context: str = "skeleton merge",
) -> List[Tuple[_State, Tuple[str, ...]]]:
    """Collapse skeletons only when topology and growth frontiers agree."""

    accumulator = _CandidateAccumulator(model, spec)
    total = len(candidates)
    started = time.monotonic()
    for index, (state, sources) in enumerate(candidates, start=1):
        accumulator.add(state, sources)
        if progress is not None and total >= 20 and (
            index == total or index % max(1, total // 10) == 0
        ):
            progress.heartbeat(
                f"{context}: merging {index}/{total}, "
                f"classes={len(accumulator.classes)}, "
                f"elapsed={time.monotonic() - started:.1f}s"
            )
    return accumulator.result()


_FRONTIER_CACHE: Dict[Tuple[object, ...], nx.Graph] = {}
_FRONTIER_CACHE_LIMIT = 4096


def _skeleton_frontier_graph(
    state: _State,
    model: _LatticeModel,
    spec: NucleationSpec,
) -> nx.Graph:
    """Augment a skeleton with its growth-relevant vacant lattice sites.

    Deriving the frontier means recomputing both vacancy sets, which is the
    expensive part.  The same skeleton is presented more than once -- once while
    merging candidates and again while validating downward parentage -- so the
    result is cached on the skeleton's exact content.  Keying on content rather
    than on object identity keeps a recycled ``id()`` from serving a stale graph.

    The returned graph is shared, so treat it as read-only.  Both call sites only
    hash it and test it for isomorphism.
    """

    cache_key = (state.atoms, tuple(sorted(state.graph.edges)))
    cached = _FRONTIER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    graph = state.graph.copy()
    next_node = len(state.atoms)
    vacancy_sets = (
        _cation_vacancies_on_anions(state, model, spec),
        _anion_vacancies_on_cations(state, model, spec),
    )
    for vacancies in vacancy_sets:
        for vacancy in vacancies:
            graph.add_node(
                next_node,
                element=f"vacancy_{vacancy.species}",
                role="growth_vacancy",
            )
            for host in sorted(vacancy.hosts):
                graph.add_edge(
                    host,
                    next_node,
                    kind="vacancy_incidence",
                    bond_order=1,
                )
            next_node += 1
    if len(_FRONTIER_CACHE) >= _FRONTIER_CACHE_LIMIT:
        _FRONTIER_CACHE.clear()
    _FRONTIER_CACHE[cache_key] = graph
    return graph


def _assert_bare_skeleton_matches_bin(
    atoms: Sequence[AtomRecord],
    *,
    k: int,
    p: int,
    spec: NucleationSpec,
    context: str,
) -> None:
    """Validate a ligand-free skeleton before destination-shell rebuilding.

    A bare ``(k,p)`` skeleton is intentionally cationic because its ``2p`` Cl
    atoms have not been placed yet.  Its role inventory must nevertheless be
    exact: ``k`` core Cd, ``k`` core Se, and ``p`` precursor Cd centers, with no
    precursor ligands or unknown roles.  This catches channel-label drift at
    the growth boundary rather than much later during bin selection.
    """

    role_counts = Counter(atom.role for atom in atoms)
    expected = {
        "core_cation": int(k),
        "core_anion": int(k),
        "precursor_center": int(p),
        "precursor_ligand": 0,
    }
    mismatches = {
        role: (role_counts.get(role, 0), count)
        for role, count in expected.items()
        if role_counts.get(role, 0) != count
    }
    unexpected = {
        role: count
        for role, count in role_counts.items()
        if role not in expected and count
    }
    expected_charge = -int(p) * int(spec.precursor.ligand_count) * int(
        spec.charges[spec.precursor.ligand]
    )
    charge = _formal_charge(atoms, spec.charges)
    if mismatches or unexpected or charge != expected_charge:
        details: List[str] = []
        if mismatches:
            details.append(f"role counts actual/expected={mismatches}")
        if unexpected:
            details.append(f"unexpected roles={unexpected}")
        if charge != expected_charge:
            details.append(
                f"bare formal charge actual/expected={charge}/{expected_charge}"
            )
        raise AssertionError(
            f"{context}: invalid bare k={k}, p={p} skeleton ("
            + "; ".join(details)
            + ")"
        )


def _assert_atoms_match_bin(
    atoms: Sequence[AtomRecord],
    *,
    k: int,
    p: int,
    spec: NucleationSpec,
    context: str,
) -> None:
    """Fail fast if atoms do not represent the declared neutral ``(k, p)``.

    The role counts make this stricter than checking charge alone: for example,
    exchanging two Cl for one Cd could otherwise remain charge neutral while
    no longer representing ``Cd_(k+p) Se_k Cl_(2p)``.
    """

    role_counts = Counter(atom.role for atom in atoms)
    expected = {
        "core_cation": int(k),
        "core_anion": int(k),
        "precursor_center": int(p),
        "precursor_ligand": int(p) * int(spec.precursor.ligand_count),
    }
    mismatches = {
        role: (role_counts.get(role, 0), count)
        for role, count in expected.items()
        if role_counts.get(role, 0) != count
    }
    unexpected = {
        role: count
        for role, count in role_counts.items()
        if role not in expected and count
    }
    charge = _formal_charge(atoms, spec.charges)
    if mismatches or unexpected or charge != 0:
        details: List[str] = []
        if mismatches:
            details.append(f"role counts actual/expected={mismatches}")
        if unexpected:
            details.append(f"unexpected roles={unexpected}")
        if charge != 0:
            details.append(f"formal charge={charge}")
        raise AssertionError(
            f"{context}: invalid k={k}, p={p} composition ("
            + "; ".join(details)
            + ")"
        )


def _increment(counts: Dict[str, int], key: str, value: int = 1) -> None:
    counts[key] = counts.get(key, 0) + value


def _merge_reason_counts(target: Dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        _increment(target, key, int(value))

__all__ = [
    'generate_nucleation_map',
    'generate_nucleation_result',
    '_prune_empty_retained_bins',
    '_precursor_skeleton_children',
    '_completeness_report',
    '_select_shells_by_score_band',
    '_guided_skeleton_bin',
    '_bridge_opportunity_graph',
    '_retained_core_sources',
    '_p_beam_sort_key',
    '_apply_p_skeleton_beam',
    '_ring_counts_for_record',
    '_apply_passivation_ring_policy',
    '_filter_core_children_by_policy',
    '_monomer_packages',
    '_p_allowed_for_k_growth',
    '_product_p0',
    '_p_surf',
    '_se_coordination_capacity',
    '_se_capacity_allows',
    '_core_formula_k',
    '_effective_max_shed',
    '_effective_p_cap',
    '_channel_p_child_max',
    '_unique_decorated_with_routes',
    '_monomer_pair_placements',
    '_place_monomer_on_source',
    '_skeleton_family_id',
    '_attach_lineage_metadata',
    '_place_n_ligands_free_sites',
    '_add_precursor_packages_free_sites',
    '_precursor_center_ids',
    '_remove_precursor_centers',
    '_shed_parent_variants',
    '_core_skeleton_children',
    '_add_bare_precursor_centers_variants',
    '_bare_package_core_children',
    '_bare_shed_core_children',
    '_decorated_core_children_by_p',
    '_enumerate_skeleton_bin',
    '_validate_strip_parents',
    '_complete_k_dag',
    '_select_bin',
    '_assign_structure_ids',
    '_enumerate_ligand_states',
    '_all_outward_ligand_sites',
    '_exact_outward_ligand_sites',
    '_is_outward_site',
    '_bridge_candidate_topology_key',
    '_bridge_candidate_permutations',
    '_greedy_incumbent_state',
    '_bridge_candidate_arcs',
    '_latent_bridge_greedy_max',
    '_latent_bridge_variants',
    '_record_from_state',
    '_unique_records',
    '_unique_states',
    '_CandidateClass',
    '_CandidateAccumulator',
    '_unique_skeleton_candidates',
    '_skeleton_frontier_graph',
    '_assert_bare_skeleton_matches_bin',
    '_assert_atoms_match_bin',
    '_increment',
    '_merge_reason_counts',
    '_FRONTIER_CACHE_LIMIT',
]
