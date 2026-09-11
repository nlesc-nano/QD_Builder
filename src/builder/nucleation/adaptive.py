"""Adaptive A->B->C lineage search built on the durable pilot journal.

The orchestration in this module is chemistry-agnostic: phases, family
coverage, budgets, convergence and resume are protocol concerns.  Proposal
construction is delegated to an adapter.  The built-in adapter intentionally
wraps the existing CdSe/CdCl2 graph and embedding machinery.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import importlib
import inspect
import json
from pathlib import Path

import yaml

from .pilot import Pilot, PilotConfig, round_robin
from .pilot_proposals import Proposal, local_proposals, validate_composition
from .search_analysis import MinimumArchive, digest


@dataclass
class AdaptiveConfig(PilotConfig):
    """System-independent search policy; chemistry lives in the adapter."""

    version: int = 3
    k_max: int = 8
    max_calls: int = 18_000
    seed_calls: int = 0
    p_max: dict = field(
        default_factory=lambda: {1: 3, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11}
    )
    stage_calls: dict = field(
        default_factory=lambda: {4: 1_600, 5: 2_400, 6: 3_200, 7: 5_000, 8: 5_000}
    )
    operation_calls: dict = field(default_factory=dict)
    family_slots: dict = field(
        default_factory=lambda: {4: 80, 5: 120, 6: 150, 7: 160, 8: 160}
    )
    parent_p_by_k: dict = field(default_factory=dict)
    proposal_p_by_k: dict = field(default_factory=dict)
    cohort_min_primary_families_by_p: dict = field(default_factory=dict)
    required_primary_families_by_p: dict = field(default_factory=dict)
    p_states_per_family: int = 3
    geometries_per_family: int = 2
    subfamilies_per_family: int = 2
    admission_fraction: float = 0.15
    admission_fraction_by_k: dict = field(default_factory=dict)
    source_novelty_fraction: float = 0.0
    minimum_launched_cycles: int = 2
    import_source_cohorts: bool = False
    phase_cycle_start: dict = field(default_factory=dict)
    enabled_phases: list = field(default_factory=lambda: ["A", "B", "C"])
    phase_a_k: list = field(default_factory=lambda: [4, 5, 6])
    phase_b_k: int = 7
    phase_c_k: int = 8
    min_cycles: int = 2
    max_cycles: int = 8
    convergence_patience: int = 2
    new_families_per_100_calls: float = 5.0
    convergence_energy_window_eV: float | None = None
    energy_improvement_eV: float = 0.05
    minimum_endpoint_fraction: float = 0.25
    proposals_per_family_fixed: int = 3
    proposals_per_family_growth: int = 6
    chemistry_adapter: str = "cdse_cdcl2"

    @classmethod
    def load(cls, path):
        raw = yaml.safe_load(Path(path).read_text()) or {}
        value = cls(**raw)
        if value.version not in {2, 3}:
            raise ValueError("adaptive configuration requires version 2 or 3")
        if value.chemistry_adapter != "cdse_cdcl2" and ":" not in value.chemistry_adapter:
            raise ValueError("chemistry_adapter must be a built-in name or module:Class")
        if not (1 <= value.workers <= 24 and 0 < value.max_calls <= 50_000):
            raise ValueError("invalid adaptive worker/call budget")
        if not (0 < value.launch_hours <= 46 and 0 < value.timeout_s <= 1800):
            raise ValueError("invalid adaptive deadline")
        if not 1 <= value.max_steps <= 150:
            raise ValueError("invalid optimization cycle limit")
        if value.progress_interval_batches < 1:
            raise ValueError("progress_interval_batches must be positive")
        value.p_max = {int(k): int(v) for k, v in value.p_max.items()}
        value.stage_calls = {int(k): int(v) for k, v in value.stage_calls.items()}
        value.operation_calls = {
            int(k): {str(operation): int(limit) for operation, limit in limits.items()}
            for k, limits in value.operation_calls.items()
        }
        value.family_slots = {int(k): int(v) for k, v in value.family_slots.items()}
        value.parent_p_by_k = {
            int(k): [int(p) for p in values]
            for k, values in value.parent_p_by_k.items()
        }
        value.proposal_p_by_k = {
            int(k): [int(p) for p in values]
            for k, values in value.proposal_p_by_k.items()
        }
        value.cohort_min_primary_families_by_p = {
            int(k): {int(p): int(count) for p, count in requirements.items()}
            for k, requirements in value.cohort_min_primary_families_by_p.items()
        }
        value.required_primary_families_by_p = {
            int(k): {int(p): int(count) for p, count in requirements.items()}
            for k, requirements in value.required_primary_families_by_p.items()
        }
        value.admission_fraction_by_k = {
            int(k): float(fraction)
            for k, fraction in value.admission_fraction_by_k.items()
        }
        value.phase_cycle_start = {
            str(phase): int(cycle)
            for phase, cycle in value.phase_cycle_start.items()
        }
        expected = list(range(1, value.phase_c_k + 1))
        if sorted(value.p_max) != expected:
            raise ValueError("p_max must cover every k through Phase C")
        if value.phase_a_k != list(range(value.phase_a_k[0], value.phase_a_k[-1] + 1)):
            raise ValueError("phase_a_k must be a contiguous increasing range")
        if not value.enabled_phases or any(
            phase not in {"A", "B", "C"} for phase in value.enabled_phases
        ):
            raise ValueError("enabled_phases must contain A, B and/or C")
        if any(
            phase not in {"A", "B", "C"} or cycle < 0 or cycle >= value.max_cycles
            for phase, cycle in value.phase_cycle_start.items()
        ):
            raise ValueError("phase_cycle_start requires valid phases and cycle indices")
        searched = set(value.phase_a_k if "A" in value.enabled_phases else [])
        if "B" in value.enabled_phases:
            searched.add(value.phase_b_k)
        if "C" in value.enabled_phases:
            searched.add(value.phase_c_k)
        if not searched <= set(value.stage_limits()) or not searched <= set(value.family_slots):
            raise ValueError("stage_calls and family_slots must cover all phase k values")
        if sum(value.stage_limits().values()) > value.max_calls:
            raise ValueError("per-k stage_calls exceed max_calls")
        if value.operation_calls and any(
            operation not in {"fixed", "growth"} or limit <= 0
            for limits in value.operation_calls.values()
            for operation, limit in limits.items()
        ):
            raise ValueError("operation_calls accepts positive fixed/growth limits")
        if not (1 <= value.p_states_per_family <= 4):
            raise ValueError("invalid p_states_per_family")
        if not (1 <= value.geometries_per_family <= 2):
            raise ValueError("invalid geometries_per_family")
        if not (1 <= value.subfamilies_per_family <= 4):
            raise ValueError("invalid subfamilies_per_family")
        if not 0 <= value.admission_fraction < 0.5:
            raise ValueError("admission_fraction must be in [0, 0.5)")
        if any(
            k not in value.family_slots or not 0 <= fraction < 0.5
            for k, fraction in value.admission_fraction_by_k.items()
        ):
            raise ValueError("admission_fraction_by_k has an invalid k or fraction")
        for label, selection in (
            ("parent_p_by_k", value.parent_p_by_k),
            ("proposal_p_by_k", value.proposal_p_by_k),
        ):
            if any(
                k not in value.p_max
                or not values
                or len(values) != len(set(values))
                or any(p < 1 or p > value.p_max[k] for p in values)
                for k, values in selection.items()
            ):
                raise ValueError(f"{label} has an invalid k or p selection")
        for label, requirements in (
            (
                "cohort_min_primary_families_by_p",
                value.cohort_min_primary_families_by_p,
            ),
            ("required_primary_families_by_p", value.required_primary_families_by_p),
        ):
            if any(
                k not in value.p_max
                or any(
                    p < 1 or p > value.p_max[k] or count < 1
                    for p, count in req.items()
                )
                for k, req in requirements.items()
            ):
                raise ValueError(f"{label} has an invalid requirement")
        if not 0 <= value.source_novelty_fraction < 0.5:
            raise ValueError("source_novelty_fraction must be in [0, 0.5)")
        if (
            value.convergence_energy_window_eV is not None
            and value.convergence_energy_window_eV <= 0
        ):
            raise ValueError("convergence_energy_window_eV must be positive or null")
        if not (1 <= value.minimum_launched_cycles <= value.max_cycles):
            raise ValueError("invalid minimum_launched_cycles")
        if not (1 <= value.min_cycles <= value.max_cycles <= 20):
            raise ValueError("invalid adaptive cycle bounds")
        if not (1 <= value.convergence_patience <= value.max_cycles):
            raise ValueError("invalid convergence patience")
        if not 0 < value.minimum_endpoint_fraction <= 1:
            raise ValueError("minimum_endpoint_fraction must be in (0, 1]")
        if min(value.stage_calls.values()) <= 0 or min(value.family_slots.values()) <= 0:
            raise ValueError("stage and family budgets must be positive")
        if value.proposals_per_family_fixed < 3:
            raise ValueError("fixed-family budget must cover all fixed-k move classes")
        return value

    def stage_limits(self):
        if self.operation_calls:
            return {
                k: sum(limits.values()) for k, limits in self.operation_calls.items()
            }
        return dict(self.stage_calls)

    def operation_limit(self, k, operation):
        if not self.operation_calls:
            return self.stage_limits().get(k, 0)
        return self.operation_calls.get(k, {}).get(operation, 0)

    def family_admission_fraction(self, k):
        return self.admission_fraction_by_k.get(k, self.admission_fraction)


def lineage_family(row):
    """Stable relaxed inorganic-core family used as the survival unit."""

    final = row["final"]
    return "family_" + digest(
        [
            row["k"],
            final["core_hash"],
            final.get("se_cn", {}),
            final.get("n4", 0),
            final.get("n6", 0),
        ]
    )[:20]


def coarse_lineage_family(row):
    """Stable discrete coordination/ring family used for lineage survival.

    Exact graph topology remains available through :func:`lineage_family` as a
    subfamily.  The coarser key prevents every graph edit from consuming a new
    beam slot.  Continuous radius and tetrahedral-order descriptors are not
    hashed: tiny relaxation changes near a bin boundary must not rename a
    frozen family.  Geometry diversity is retained separately through
    ``core_geometry_hash`` representatives in :meth:`adaptive_parents`.
    """

    final = row["final"]
    cd_se_cn = Counter()
    for environment, count in final.get("cd_environments", {}).items():
        cd_se_cn[int(environment.split(",", 1)[0])] += count

    def ring_class(value):
        return 0 if value == 0 else 1 if value == 1 else 2

    return "coarse_" + digest(
        [
            row["k"],
            tuple(sorted((int(cn), count) for cn, count in final["se_cn"].items())),
            tuple(sorted(cd_se_cn.items())),
            ring_class(final.get("n4", 0)),
            ring_class(final.get("n6", 0)),
        ]
    )[:20]


def _legacy_coarse_lineage_family(row):
    """Version-3 q4/radius hash, used only to translate old journals."""

    final = row["final"]
    cd_se_cn = Counter()
    for environment, count in final.get("cd_environments", {}).items():
        cd_se_cn[int(environment.split(",", 1)[0])] += count

    def ring_class(value):
        return 0 if value == 0 else 1 if value == 1 else 2

    q4 = final.get("tetrahedral_q4", [])
    mean_q4 = sum(q4) / len(q4) if q4 else -2.0
    return "coarse_" + digest(
        [
            row["k"],
            tuple(sorted((int(cn), count) for cn, count in final["se_cn"].items())),
            tuple(sorted(cd_se_cn.items())),
            ring_class(final.get("n4", 0)),
            ring_class(final.get("n6", 0)),
            round(final.get("radius_A", 0.0) / 0.25),
            round(mean_q4 / 0.2),
        ]
    )[:20]


class CdSeCdClAdapter:
    """Chemistry hook for the present Cd_(k+p)Se_kCl_(2p) model."""

    name = "cdse_cdcl2"
    fixed_channels = ("topology", "exchange", "reconstruction")

    @staticmethod
    def family(row):
        return coarse_lineage_family(row)

    @staticmethod
    def subfamily(row):
        return lineage_family(row)

    @staticmethod
    def validate(proposal):
        return validate_composition(proposal)

    @staticmethod
    def classify(pilot, proposal, result, arm):
        return Pilot.classify(pilot, proposal, result, arm)

    @staticmethod
    def archive(spec):
        return MinimumArchive(spec)

    @staticmethod
    def bootstrap(pilot):
        return pilot.seeds()

    def fixed(self, pilot, parent, cycle, limit):
        jobs = []
        # Exchange was the productive fixed-k move in v1.  Topology remains as
        # a minority exploration channel; reconstruction is sampled every
        # third cycle because it mostly reconverged existing basins.
        schedule = [("exchange", 2), ("topology", 1)]
        if cycle % 3 == 0:
            schedule.append(("reconstruction", 1))
        remaining = limit
        for channel, requested in schedule:
            count = min(requested, remaining)
            if count <= 0:
                break
            rng = pilot.rng("adaptive", "fixed", cycle, channel, parent["minimum_id"])
            made = local_proposals(
                parent,
                rng,
                pilot.spec,
                pilot.pack,
                channel=channel,
                limit=count,
            )
            jobs.extend(made)
            remaining -= len(made)
        return jobs[:limit]

    def grow(self, pilot, parent, target_k, cycle, limit):
        rng = pilot.rng("adaptive", "growth", cycle, parent["minimum_id"])
        jobs = local_proposals(
            parent,
            rng,
            pilot.spec,
            pilot.pack,
            channel="growth",
            limit=max(2, limit - 2),
        )
        # Preserve a bounded lattice/occupation route where the relaxed parent
        # can be snapped.  It complements, rather than dominates, direct growth.
        jobs.extend(pilot.lattice_proposals(parent, target_k, limit=2))
        return jobs[:limit]


def load_adapter(name):
    """Load a built-in or ``module:Class`` chemistry plugin.

    A plugin supplies ``family``, ``validate``, ``classify``, ``archive``,
    ``bootstrap``, ``fixed`` and ``grow``.  This keeps system chemistry out of
    phase control.
    """

    if name == "cdse_cdcl2":
        adapter = CdSeCdClAdapter()
    else:
        module_name, class_name = name.split(":", 1)
        adapter = getattr(importlib.import_module(module_name), class_name)()
    required = (
        "family",
        "validate",
        "classify",
        "archive",
        "bootstrap",
        "fixed",
        "grow",
        "subfamily",
    )
    missing = [method for method in required if not callable(getattr(adapter, method, None))]
    if missing:
        raise TypeError(f"chemistry adapter is missing methods: {', '.join(missing)}")
    if not hasattr(adapter, "name"):
        adapter.name = name
    return adapter


class AdaptivePilot(Pilot):
    """Resumable family-complete backfill followed by gated k extension."""

    def __init__(self, config, map_path, growth_path, source, output, *, backend=None):
        super().__init__(config, map_path, growth_path, source, output, backend=backend)
        self.adapter = load_adapter(config.chemistry_adapter)
        self.archive = self.adapter.archive(self.spec)
        self.operation_calls_used = Counter()
        self._source_novel_families_cache = None

    def setup(self):
        super().setup()
        self.operation_calls_used = Counter()
        for event in self.events:
            if event["event"] != "launch":
                continue
            proposal = event["proposal"]
            operation = proposal.get("search_operation", "")
            if operation:
                self.operation_calls_used[event["stage"], operation] += 1

    def event(self, data):
        super().event(data)
        if data["event"] == "launch":
            operation = data["proposal"].get("search_operation", "")
            if operation:
                self.operation_calls_used[data["stage"], operation] += 1

    def extra_protocol_sources(self):
        path = inspect.getsourcefile(type(self.adapter))
        if not path:
            return {"chemistry_adapter": digest(self.config.chemistry_adapter)}
        source = Path(path)
        return {f"chemistry_adapter:{source.name}": digest(source.read_text())}

    def proposal_valid(self, proposal):
        return self.adapter.validate(proposal)

    def classify(self, proposal, result, arm):
        return self.adapter.classify(self, proposal, result, arm)

    def allowed(self, arm, stage, proposal=None):
        base = (
            arm == "experimental"
            and stage in self.config.stage_limits()
            and self.calls[arm, stage] < self.config.stage_limits()[stage]
            and sum(self.calls.values()) < self.config.max_calls
            and not self.expired()
        )
        if not base or proposal is None or not proposal.search_operation:
            return base
        limit = self.config.operation_limit(stage, proposal.search_operation)
        return (
            limit > 0
            and self.operation_calls_used[stage, proposal.search_operation] < limit
        )

    def extra_status(self):
        return {
            "operation_calls": {
                f"{k}:{operation}": count
                for (k, operation), count in sorted(self.operation_calls_used.items())
            }
        }

    def initialize_source(self):
        imported = any(event["event"] == "archive_import" for event in self.events)
        if not imported:
            path = self.seed_dir / "minima.json"
            if not path.is_file():
                if not (self.seed_dir / "index.csv").is_file():
                    raise ValueError("adaptive source requires minima.json or index.csv")
                if 1 not in self.config.stage_limits() or self.config.seed_calls <= 0:
                    raise ValueError(
                        "bootstrap sources require seed_calls > 0 and a stage_calls entry for k=1"
                    )
                tag = "bootstrap"
                if ("experimental", 1, tag) not in self.stage_done:
                    queue = self.prepare_queue(
                        "experimental", 1, tag, lambda: self.adapter.bootstrap(self)
                    )
                    self.evaluate(
                        [("experimental", proposal) for proposal in queue], 1
                    )
                    if not self.expired():
                        self.mark_stage("experimental", 1, tag)
                return
            data = json.loads(path.read_text())
            source_rows = data.get("experimental") or data.get("control")
            if not isinstance(source_rows, dict) or not source_rows:
                raise ValueError("source minima.json has no usable minimum archive")
            rows = []
            for original_id, source in sorted(source_rows.items()):
                row = dict(source)
                row["minimum_id"] = original_id
                row["source_protocol"] = row.get("protocol")
                row["protocol"] = self.protocol["fingerprint"]
                row["source"] = f"import:{original_id}"
                row["routes"] = sorted(set(row.get("routes", [])) - {original_id})
                rows.append(row)
            event = dict(event="archive_import", arm="experimental", rows=rows)
            self.event(event)
            for row in rows:
                self.store(
                    "experimental", row, preferred_id=row["minimum_id"]
                )
            self.checkpoint()
            print(f"[adaptive] imported {len(rows)} source minima", flush=True)
        self._import_source_cohorts()

    def _import_source_cohorts(self):
        """Freeze selected family cohorts from an earlier adaptive archive.

        Minima remain the authoritative structural source.  Only cohort
        membership is transferred; calls, queues, plateau counters and phase
        completion state deliberately start afresh in the new output.
        """

        if not self.config.import_source_cohorts or any(
            event["event"] == "source_cohorts_imported" for event in self.events
        ):
            return
        journal = self.seed_dir / "events.jsonl"
        if not journal.is_file():
            raise ValueError("import_source_cohorts requires source events.jsonl")
        states = {}
        representatives = defaultdict(list)
        with journal.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid source events.jsonl line {line_number}"
                    ) from exc
                phase = event.get("phase")
                k = event.get("k")
                if event.get("event") == "cohort_initialized":
                    states[phase, k] = list(event.get("families", []))
                elif event.get("event") == "cohort_admission" and (phase, k) in states:
                    states[phase, k].extend(event.get("families", []))
                elif event.get("event") == "family_plan":
                    representatives[
                        phase, event.get("source_k"), event.get("family_id")
                    ].append(event.get("parent_id"))

        imported = {}
        for phase in self.config.enabled_phases:
            phase_ks = {
                "A": self.config.phase_a_k,
                "B": [self.config.phase_b_k],
                "C": [self.config.phase_c_k],
            }[phase]
            for k in phase_ks:
                if self._cohort_state(phase, k)[0] is not None:
                    continue
                families = list(dict.fromkeys(states.get((phase, k), [])))
                if not families:
                    raise ValueError(f"source has no {phase} k={k} cohort to import")
                available = set(self._ranked_families(k)[0])
                legacy_translation = {
                    _legacy_coarse_lineage_family(row): self.adapter.family(row)
                    for row in self.rows["experimental"].values()
                    if row["k"] == k
                }
                # Version-3 archives may contain family hashes made with the
                # retired continuous q4/radius bins.  Translate them through
                # their journaled parent representatives instead of rejecting
                # an otherwise valid continuation archive.
                translated = []
                unresolved = []
                for family in families:
                    if family in available:
                        translated.append(family)
                        continue
                    for parent_id in representatives.get((phase, k, family), []):
                        row = self.rows["experimental"].get(parent_id)
                        if row is not None:
                            translated.append(self.adapter.family(row))
                            break
                    else:
                        translated_family = legacy_translation.get(family)
                        if translated_family is None:
                            unresolved.append(family)
                        else:
                            translated.append(translated_family)
                if unresolved:
                    raise ValueError(
                        f"source {phase} k={k} cohort has {len(unresolved)} "
                        "families with no current or journaled representative"
                    )
                families = list(dict.fromkeys(translated))
                missing = [family for family in families if family not in available]
                if missing:
                    raise ValueError(
                        f"source {phase} k={k} cohort has {len(missing)} families "
                        "absent from imported minima"
                    )
                capacity = self.config.family_slots[k]
                if len(families) > capacity:
                    raise ValueError(
                        f"source {phase} k={k} cohort size {len(families)} exceeds "
                        f"configured capacity {capacity}"
                    )
                self.event(
                    dict(
                        event="cohort_initialized",
                        phase=phase,
                        k=k,
                        cycle=self.config.phase_cycle_start.get(phase, 0),
                        capacity=capacity,
                        families=families,
                        known_families=sorted(available),
                        imported_from=str(self.seed_dir),
                    )
                )
                imported[f"{phase}:k{k}"] = len(families)
        self.event(dict(event="source_cohorts_imported", cohorts=imported))
        self.checkpoint()
        print(f"[adaptive] imported frozen source cohorts: {imported}", flush=True)

    def _plans(self):
        return Counter(
            (event["phase"], event["operation"], event["family_id"])
            for event in self.events
            if event["event"] == "family_plan"
        )

    def _ranked_families(self, k):
        parent_p = set(self.config.parent_p_by_k.get(k, []))
        rows = [
            r
            for r in self.rows["experimental"].values()
            if r["k"] == k and (not parent_p or r["p"] in parent_p)
        ]
        by_family = defaultdict(list)
        for row in rows:
            by_family[self.adapter.family(row)].append(row)

        energy_rank = {}
        for p in sorted({r["p"] for r in rows}):
            ordered = sorted(
                (r for r in rows if r["p"] == p and r["role"] == "primary"),
                key=lambda r: (r["energy_eV"], r["minimum_id"]),
            )
            for rank, row in enumerate(ordered):
                energy_rank[row["minimum_id"]] = rank / max(1, len(ordered) - 1)

        ranked = sorted(
            by_family,
            key=lambda family_id: (
                min(
                    energy_rank.get(r["minimum_id"], 2.0)
                    + (0.5 if r["role"] == "audit" else 0.0)
                    + (0.5 if r.get("audit_derived", False) else 0.0)
                    for r in by_family[family_id]
                ),
                family_id,
            ),
        )
        return ranked, by_family

    def _source_novel_families(self, k):
        """Families first discovered by the immediately preceding archive.

        This supports a bounded novelty reserve when extending k.  It compares
        primary result rows with the archive-import baseline in the source
        journal using the *current* stable family definition.  Missing legacy
        journals simply provide no reserve candidates.
        """

        if self._source_novel_families_cache is None:
            discovered = defaultdict(list)
            seen = defaultdict(set)
            journal = self.seed_dir / "events.jsonl"
            if journal.is_file():
                with journal.open() as handle:
                    for line_number, line in enumerate(handle, start=1):
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"invalid source events.jsonl line {line_number}"
                            ) from exc
                        if event.get("event") == "archive_import":
                            for row in event.get("rows", []):
                                if row.get("role") != "primary":
                                    continue
                                family = self.adapter.family(row)
                                seen[row["k"]].add(family)
                        elif event.get("event") == "result":
                            row = event.get("row")
                            if not row or row.get("role") != "primary":
                                continue
                            family = self.adapter.family(row)
                            if family not in seen[row["k"]]:
                                seen[row["k"]].add(family)
                                discovered[row["k"]].append(family)
            self._source_novel_families_cache = {
                size: list(dict.fromkeys(families))
                for size, families in discovered.items()
            }
        return self._source_novel_families_cache.get(k, [])

    def _cohort_state(self, phase, k):
        initialized = None
        admitted = []
        for event in self.events:
            if event.get("phase") != phase or event.get("k") != k:
                continue
            if event["event"] == "cohort_initialized":
                initialized = event
            elif event["event"] == "cohort_admission":
                admitted.extend(event["families"])
        if initialized is None:
            return None, []
        return initialized, list(dict.fromkeys(initialized["families"] + admitted))

    def _update_cohort(self, phase, k, cycle):
        ranked, by_family = self._ranked_families(k)
        if not ranked:
            return []
        initialized, cohort = self._cohort_state(phase, k)
        capacity = self.config.family_slots[k]
        if initialized is None:
            admission_fraction = self.config.family_admission_fraction(k)
            initial_capacity = max(
                1, int(capacity * (1.0 - admission_fraction))
            )
            source_novel = (
                set(self._source_novel_families(k))
                if self.config.source_novelty_fraction > 0
                else set()
            )
            composition_reserved = []
            for p, required in sorted(
                self.config.cohort_min_primary_families_by_p.get(k, {}).items()
            ):
                candidates = [
                    family
                    for family in ranked
                    if any(
                        row["p"] == p and row["role"] == "primary"
                        for row in by_family[family]
                    )
                ]
                if len(candidates) < required:
                    raise ValueError(
                        f"k={k} parent cohort requires {required} primary families "
                        f"at p={p}, but the source has {len(candidates)}"
                    )
                for family in candidates[:required]:
                    if family not in composition_reserved:
                        composition_reserved.append(family)
            if len(composition_reserved) > initial_capacity:
                raise ValueError(
                    f"k={k} composition reservations exceed initial capacity "
                    f"{initial_capacity}"
                )
            novelty_candidates = [
                family
                for family in ranked
                if family in source_novel and family not in composition_reserved
            ]
            novelty_slots = min(
                len(novelty_candidates),
                initial_capacity - len(composition_reserved),
                int(round(capacity * self.config.source_novelty_fraction)),
            )
            reserved_novelty = novelty_candidates[:novelty_slots]
            reserved = composition_reserved + reserved_novelty
            if len(reserved) > initial_capacity:
                raise ValueError(
                    f"k={k} cohort reservations exceed initial capacity "
                    f"{initial_capacity}"
                )
            general = [
                family
                for family in ranked
                if family not in source_novel and family not in composition_reserved
            ]
            cohort = general[: initial_capacity - len(reserved)] + reserved
            # If fewer established families exist than expected, fill without
            # exceeding the fixed initial capacity.
            if len(cohort) < initial_capacity:
                cohort.extend(
                    family
                    for family in ranked
                    if family not in cohort
                )
                cohort = cohort[:initial_capacity]
            self.event(
                dict(
                    event="cohort_initialized",
                    phase=phase,
                    k=k,
                    cycle=cycle,
                    capacity=capacity,
                    families=cohort,
                    known_families=ranked,
                    composition_reserved_families=composition_reserved,
                    source_novelty_families=reserved_novelty,
                )
            )
            return cohort

        # An imported continuation cohort is an experimental control: archive
        # new families, but do not let discoveries alter the population whose
        # representative rotation is being completed.
        if initialized.get("imported_from"):
            return cohort
        if len(cohort) >= capacity:
            return cohort
        known = set(initialized.get("known_families", initialized["families"]))
        candidates = [
            family_id
            for family_id in ranked
            if family_id not in known and family_id not in cohort
        ]
        admitted = candidates[: capacity - len(cohort)]
        if admitted:
            self.event(
                dict(
                    event="cohort_admission",
                    phase=phase,
                    k=k,
                    cycle=cycle,
                    families=admitted,
                )
            )
            cohort.extend(admitted)
        return cohort

    def _coverage_cycle_ledger(self):
        launched = {
            (event["id"], event["proposal"].get("search_cycle", -1))
            for event in self.events
            if event["event"] == "launch"
        }
        ledger = defaultdict(set)
        for event in self.events:
            if event["event"] != "family_plan":
                continue
            if any(
                (proposal_id, event["cycle"]) in launched
                for proposal_id in event.get("proposal_ids", [])
            ) or not event.get("novel_proposal_ids", event.get("proposal_ids", [])):
                ledger[
                    event["phase"], event["operation"], event["family_id"]
                ].add(event["cycle"])
        return ledger

    def _covered_cycles(self, phase, operation, family_id, ledger=None):
        ledger = ledger if ledger is not None else self._coverage_cycle_ledger()
        return ledger[phase, operation, family_id]

    def adaptive_parents(self, k, phase, operation, cycle=0):
        _, by_family = self._ranked_families(k)
        _, families = self._cohort_state(phase, k)
        families = [family_id for family_id in families if family_id in by_family]
        coverage_ledger = self._coverage_cycle_ledger()
        families.sort(
            key=lambda family_id: (
                len(
                    self._covered_cycles(
                        phase, operation, family_id, coverage_ledger
                    )
                ),
                family_id,
            )
        )
        selected = []
        for family_id in families:
            candidates = by_family[family_id]
            primary_candidates = [r for r in candidates if r["role"] == "primary"]
            if primary_candidates:
                candidates = primary_candidates
            clean_lineage = [r for r in candidates if not r.get("audit_derived", False)]
            if clean_lineage:
                candidates = clean_lineage
            exact_groups = defaultdict(list)
            for row in candidates:
                exact_groups[row["p"], self.adapter.subfamily(row)].append(row)
            exact_by_p = defaultdict(deque)
            for (p, exact), group in exact_groups.items():
                exact_by_p[p].append(
                    (min(row["energy_eV"] for row in group), exact)
                )
            for p in exact_by_p:
                exact_by_p[p] = deque(sorted(exact_by_p[p]))
            exact_ranked = []
            while any(exact_by_p.values()):
                for p in sorted(exact_by_p):
                    if exact_by_p[p]:
                        _, exact = exact_by_p[p].popleft()
                        exact_ranked.append((p, exact))
                        if len(exact_ranked) >= self.config.subfamilies_per_family:
                            break
                if len(exact_ranked) >= self.config.subfamilies_per_family:
                    break
            candidates = [row for exact in exact_ranked for row in exact_groups[exact]]
            representatives_by_p = defaultdict(list)
            seen_geometries = defaultdict(set)
            for row in sorted(
                candidates,
                key=lambda r: (
                    r["role"] != "primary",
                    r["energy_eV"],
                    r["minimum_id"],
                ),
            ):
                geometry = row["final"].get(
                    "core_geometry_hash", row["minimum_id"]
                )
                if (
                    geometry not in seen_geometries[row["p"]]
                    and len(representatives_by_p[row["p"]])
                    < self.config.geometries_per_family
                ):
                    seen_geometries[row["p"]].add(geometry)
                    representatives_by_p[row["p"]].append(row)
            p_values = sorted(representatives_by_p)
            # Preserve lean, central and passivated representatives where they
            # exist; this prevents one arbitrary p from owning the lineage.
            indices = [0, len(p_values) // 2, len(p_values) - 1]
            chosen_p = []
            for index in indices + list(range(len(p_values))):
                p = p_values[index]
                if p not in chosen_p:
                    chosen_p.append(p)
                if len(chosen_p) >= self.config.p_states_per_family:
                    break
            # One composition representative per family and cycle prevents p
            # multiplicity from consuming the lineage budget.  Successive
            # cycles rotate through lean/central/passivated states.
            p_index = cycle % len(chosen_p)
            p = chosen_p[p_index]
            geometries = representatives_by_p[p]
            geometry_cycle = cycle // len(chosen_p)
            selected.append((family_id, geometries[geometry_cycle % len(geometries)]))
        return selected

    @staticmethod
    def _fair_jobs(jobs):
        unique = {}
        for job in jobs:
            if job.id in unique:
                unique[job.id].parents = sorted(
                    set(unique[job.id].parents + job.parents)
                )
            else:
                unique[job.id] = job
        by_parent = defaultdict(list)
        for job in unique.values():
            by_parent[tuple(job.parents)].append(job)
        queues = [
            deque(round_robin(sorted(group, key=lambda p: (p.channel, p.id))))
            for group in by_parent.values()
        ]
        ordered = []
        while any(queues):
            for queue in queues:
                if queue:
                    ordered.append(queue.popleft())
        # Interleave one audit descendant after nine clean proposals.  This
        # satisfies the conservative running 10% quota while giving admitted
        # audit lineages a real chance to launch.
        clean = deque(job for job in ordered if not job.audit_derived)
        audit = deque(job for job in ordered if job.audit_derived)
        scheduled = []
        while clean:
            for _ in range(9):
                if clean:
                    scheduled.append(clean.popleft())
            if audit:
                scheduled.append(audit.popleft())
        scheduled.extend(audit)
        return scheduled

    def adaptive_proposals(self, phase, operation, source_k, target_k, cycle):
        jobs = []
        for family_id, parent in self.adaptive_parents(
            source_k, phase, operation, cycle
        ):
            if self.expired():
                break
            if operation == "fixed":
                made = self.adapter.fixed(
                    self,
                    parent,
                    cycle,
                    self.config.proposals_per_family_fixed,
                )
            else:
                made = self.adapter.grow(
                    self,
                    parent,
                    target_k,
                    cycle,
                    self.config.proposals_per_family_growth,
                )
            made = [
                proposal
                for proposal in made
                if proposal.k == target_k
                and 1 <= proposal.p <= self.config.p_max[target_k]
                and (
                    not self.config.proposal_p_by_k.get(target_k)
                    or proposal.p in self.config.proposal_p_by_k[target_k]
                )
                and self.proposal_valid(proposal)
            ]
            for proposal in made:
                proposal.search_phase = phase
                proposal.search_cycle = cycle
                proposal.search_operation = operation
                proposal.source_family = family_id
            reserved_ids = {
                proposal_id
                for reserved_arm, proposal_id, _ in self.reserved
                if reserved_arm == "experimental"
            }
            self.event(
                dict(
                    event="family_plan",
                    phase=phase,
                    cycle=cycle,
                    operation=operation,
                    source_k=source_k,
                    target_k=target_k,
                    family_id=family_id,
                    parent_id=parent["minimum_id"],
                    proposals=len(made),
                    proposal_ids=[proposal.id for proposal in made],
                    novel_proposal_ids=[
                        proposal.id for proposal in made if proposal.id not in reserved_ids
                    ],
                )
            )
            jobs.extend(made)
        return self._fair_jobs(jobs)

    def _snapshot(self, ks):
        snapshot = {}
        for k in ks:
            target_p = set(self.config.proposal_p_by_k.get(k, []))
            rows = [
                row
                for row in self.rows["experimental"].values()
                if row["k"] == k
                and row["role"] == "primary"
                and (not target_p or row["p"] in target_p)
            ]
            best = {}
            for row in rows:
                best[row["p"]] = min(
                    best.get(row["p"], float("inf")), row["energy_eV"]
                )
            families = {self.adapter.family(row) for row in rows}
            window = self.config.convergence_energy_window_eV
            relevant_families = (
                families
                if window is None
                else {
                    self.adapter.family(row)
                    for row in rows
                    if row["energy_eV"] <= best[row["p"]] + window
                }
            )
            snapshot[k] = dict(
                families=families,
                relevant_families=relevant_families,
                best=best,
                calls=self.calls["experimental", k],
                primary_families_by_p={
                    p: len(
                        {
                            self.adapter.family(row)
                            for row in rows
                            if row["p"] == p
                        }
                    )
                    for p in sorted({row["p"] for row in rows})
                },
            )
        return snapshot

    def _cycle_metrics(self, phase, cycle, before, after):
        outcomes = defaultdict(Counter)
        launches = {
            (event["arm"], event["id"], event["attempt"]): event
            for event in self.events
            if event["event"] == "launch"
        }
        for event in self.events:
            if event["event"] != "result":
                continue
            launch = launches.get(
                (event["arm"], event["id"], event["attempt"])
            )
            if launch is None:
                continue
            proposal = launch["proposal"]
            if (
                proposal.get("search_phase") == phase
                and proposal.get("search_cycle") == cycle
            ):
                outcomes[event["stage"]][
                    event["row"]["role"] if event.get("row") else "rejected"
                ] += 1
        per_k = {}
        for k in sorted(before):
            calls = after[k]["calls"] - before[k]["calls"]
            discovered = len(after[k]["families"] - before[k]["families"])
            rate = 100.0 * discovered / max(1, calls)
            relevant_discovered = len(
                after[k]["relevant_families"] - before[k]["families"]
            )
            relevant_rate = 100.0 * relevant_discovered / max(1, calls)
            improvements = [
                before[k]["best"][p] - energy
                for p, energy in after[k]["best"].items()
                if p in before[k]["best"] and energy < before[k]["best"][p]
            ]
            improvement = max(improvements, default=0.0)
            usable = outcomes[k]["primary"] + outcomes[k]["audit"]
            endpoint_fraction = usable / max(1, calls)
            per_k[str(k)] = dict(
                calls=calls,
                new_primary_families=discovered,
                new_families_per_100_calls=rate,
                new_energy_window_families=relevant_discovered,
                new_energy_window_families_per_100_calls=relevant_rate,
                convergence_energy_window_eV=(
                    self.config.convergence_energy_window_eV
                ),
                primary_families_by_p=after[k]["primary_families_by_p"],
                maximum_within_composition_improvement_eV=improvement,
                usable_endpoints=usable,
                endpoint_fraction=endpoint_fraction,
                stagnant=calls > 0
                and endpoint_fraction >= self.config.minimum_endpoint_fraction
                and relevant_rate < self.config.new_families_per_100_calls
                and improvement < self.config.energy_improvement_eV,
            )
        calls = sum(metrics["calls"] for metrics in per_k.values())
        discovered = sum(
            metrics["new_primary_families"] for metrics in per_k.values()
        )
        relevant_discovered = sum(
            metrics["new_energy_window_families"] for metrics in per_k.values()
        )
        improvement = max(
            (
                metrics["maximum_within_composition_improvement_eV"]
                for metrics in per_k.values()
            ),
            default=0.0,
        )
        event = dict(
            event="adaptive_cycle",
            phase=phase,
            cycle=cycle,
            calls=calls,
            new_primary_families=discovered,
            new_families_per_100_calls=100.0 * discovered / max(1, calls),
            new_energy_window_families=relevant_discovered,
            new_energy_window_families_per_100_calls=(
                100.0 * relevant_discovered / max(1, calls)
            ),
            convergence_energy_window_eV=self.config.convergence_energy_window_eV,
            maximum_within_composition_improvement_eV=improvement,
            stagnant=bool(per_k) and all(m["stagnant"] for m in per_k.values()),
            per_k=per_k,
        )
        self.event(event)
        return event

    def _coverage_debt(self, phase, requirements):
        coverage_ledger = self._coverage_cycle_ledger()
        debt = {}
        for operation, source_k in requirements:
            _, retained = self._cohort_state(phase, source_k)
            missing = []
            for family_id in retained:
                covered_cycles = coverage_ledger[phase, operation, family_id]
                if len(covered_cycles) < self.config.minimum_launched_cycles:
                    missing.append(family_id)
            debt[f"{operation}:k{source_k}"] = missing
        return debt

    def _coverage_complete(self, phase, requirements):
        return not any(self._coverage_debt(phase, requirements).values())

    def _composition_coverage_debt(self, ks):
        debt = {}
        for k in ks:
            requirements = self.config.required_primary_families_by_p.get(k, {})
            if not requirements:
                continue
            rows = [
                row
                for row in self.rows["experimental"].values()
                if row["k"] == k and row["role"] == "primary"
            ]
            by_p = defaultdict(set)
            for row in rows:
                by_p[row["p"]].add(self.adapter.family(row))
            for p, required in sorted(requirements.items()):
                missing = max(0, required - len(by_p[p]))
                if missing:
                    debt[f"k{k}:p{p}"] = missing
        return debt

    def _composition_coverage_complete(self, ks):
        return not self._composition_coverage_debt(ks)

    def _run_queue(self, phase, cycle, operation, source_k, target_k):
        tag = f"{phase}:{cycle}:{operation}:{source_k}:{target_k}"
        if ("experimental", target_k, tag) in self.stage_done:
            print(
                f"[adaptive] skip phase={phase} cycle={cycle} operation={operation} "
                f"k={source_k}->{target_k} reason=already_complete",
                flush=True,
            )
            return
        limit = self.config.operation_limit(target_k, operation)
        if limit <= 0 or self.operation_calls_used[target_k, operation] >= limit:
            self.event(
                dict(
                    event="operation_budget_exhausted",
                    phase=phase,
                    cycle=cycle,
                    operation=operation,
                    source_k=source_k,
                    target_k=target_k,
                )
            )
            self.mark_stage("experimental", target_k, tag)
            print(
                f"[adaptive] skip phase={phase} cycle={cycle} operation={operation} "
                f"k={source_k}->{target_k} reason=budget_exhausted "
                f"calls={self.operation_calls_used[target_k, operation]}/{limit}",
                flush=True,
            )
            return
        self._update_cohort(phase, source_k, cycle)
        queue = self.prepare_queue(
            "experimental",
            target_k,
            tag,
            lambda: self.adaptive_proposals(
                phase, operation, source_k, target_k, cycle
            ),
        )
        pending = [
            proposal
            for proposal in queue
            if ("experimental", proposal.id, proposal.attempt) not in self.reserved
        ]
        compositions = Counter(proposal.p for proposal in pending)
        channels = Counter(proposal.channel for proposal in pending)
        cohort_size = len(self._cohort_state(phase, source_k)[1])
        p_text = ",".join(
            f"p{p}:{count}" for p, count in sorted(compositions.items())
        )
        channel_text = ",".join(
            f"{channel}:{count}" for channel, count in sorted(channels.items())
        )
        used_before = self.operation_calls_used[target_k, operation]
        print(
            f"[adaptive] queue phase={phase} cycle={cycle} operation={operation} "
            f"k={source_k}->{target_k} cohort={cohort_size} total={len(queue)} "
            f"pending={len(pending)} channels={channel_text or '-'} "
            f"compositions={p_text or '-'} "
            f"operation_calls={used_before}/{limit}",
            flush=True,
        )
        self.evaluate([("experimental", proposal) for proposal in queue], target_k)
        used_after = self.operation_calls_used[target_k, operation]
        print(
            f"[adaptive] operation_done phase={phase} cycle={cycle} "
            f"operation={operation} k={source_k}->{target_k} "
            f"launched={used_after - used_before} operation_calls={used_after}/{limit} "
            f"calls_total={sum(self.calls.values())}/{self.config.max_calls}",
            flush=True,
        )
        if not self.expired():
            self.mark_stage("experimental", target_k, tag)

    def _phase_complete(self, phase):
        return any(
            event["event"] == "adaptive_phase_done" and event["phase"] == phase
            for event in self.events
        )

    def _run_phase(self, phase, ks, *, extend_from=None):
        if self._phase_complete(phase):
            return True
        old_cycles = [
            event
            for event in self.events
            if event["event"] == "adaptive_cycle" and event["phase"] == phase
        ]
        stagnant_runs = {k: 0 for k in ks}
        for k in ks:
            for event in reversed(old_cycles):
                metrics = event.get("per_k", {}).get(str(k))
                if not metrics or not metrics["stagnant"]:
                    break
                stagnant_runs[k] += 1
        start = max(
            max((event["cycle"] for event in old_cycles), default=-1) + 1,
            self.config.phase_cycle_start.get(phase, 0),
        )
        requirements = [
            ("fixed", k)
            for k in ks
            if self.config.operation_limit(k, "fixed") > 0
        ]
        requirements.extend(
            ("growth", k)
            for k in ks[:-1]
            if self.config.operation_limit(k + 1, "growth") > 0
        )
        if (
            extend_from is not None
            and self.config.operation_limit(ks[0], "growth") > 0
        ):
            requirements.append(("growth", extend_from))
        budget_text = ",".join(
            f"k{target_k}.{operation}="
            f"{self.operation_calls_used[target_k, operation]}/"
            f"{self.config.operation_limit(target_k, operation)}"
            for target_k in ks
            for operation in ("growth", "fixed")
            if self.config.operation_limit(target_k, operation) > 0
        )
        print(
            f"[adaptive] phase={phase} ready cycles={start}..{self.config.max_cycles - 1} "
            f"workers={self.config.workers} budgets={budget_text or '-'} "
            f"calls_total={sum(self.calls.values())}/{self.config.max_calls}",
            flush=True,
        )
        for cycle in range(start, self.config.max_cycles):
            if self.expired() or sum(self.calls.values()) >= self.config.max_calls:
                return False
            print(f"[adaptive] phase={phase} cycle={cycle} started", flush=True)
            before = self._snapshot(ks)
            if (
                extend_from is not None
                and self.config.operation_limit(ks[0], "growth") > 0
            ):
                self._run_queue(phase, cycle, "growth", extend_from, ks[0])
            for index, k in enumerate(ks):
                if self.config.operation_limit(k, "fixed") > 0:
                    self._run_queue(phase, cycle, "fixed", k, k)
                if (
                    index + 1 < len(ks)
                    and self.config.operation_limit(ks[index + 1], "growth") > 0
                ):
                    self._run_queue(phase, cycle, "growth", k, ks[index + 1])
            after = self._snapshot(ks)
            metrics = self._cycle_metrics(phase, cycle, before, after)
            for k in ks:
                km = metrics["per_k"][str(k)]
                print(
                    f"[adaptive] phase={phase} cycle={cycle} k={k} "
                    f"calls={km['calls']} new_families={km['new_primary_families']} "
                    f"rate_per_100={km['new_families_per_100_calls']:.2f} "
                    f"window_rate={km['new_energy_window_families_per_100_calls']:.2f} "
                    f"best_dE={km['maximum_within_composition_improvement_eV']:.4f} "
                    f"endpoints={km['usable_endpoints']}/{km['calls']} "
                    f"stagnant={str(km['stagnant']).lower()} "
                    f"families_by_p="
                    + (
                        ",".join(
                            f"p{p}:{count}"
                            for p, count in sorted(
                                km["primary_families_by_p"].items(),
                                key=lambda item: int(item[0]),
                            )
                        )
                        or "-"
                    ),
                    flush=True,
                )
                stagnant_runs[k] = (
                    stagnant_runs[k] + 1 if km["stagnant"] else 0
                )
            self.checkpoint()
            lineage_debt = self._coverage_debt(phase, requirements)
            composition_debt = self._composition_coverage_debt(ks)
            lineage_text = ",".join(
                f"{key}:{len(missing)}"
                for key, missing in sorted(lineage_debt.items())
            )
            composition_text = ",".join(
                f"{key}:{missing}"
                for key, missing in sorted(composition_debt.items())
            )
            print(
                f"[adaptive] coverage phase={phase} cycle={cycle} "
                f"lineage_debt={lineage_text or 'none'} "
                f"composition_debt={composition_text or 'none'}",
                flush=True,
            )
            exhausted_without_plateau = []
            target_operations = defaultdict(set)
            for operation, source_k in requirements:
                target_k = source_k if operation == "fixed" else source_k + 1
                target_operations[target_k].add(operation)
            for k in ks:
                if metrics["per_k"][str(k)]["calls"]:
                    continue
                if target_operations[k] and all(
                    self.operation_calls_used[k, operation]
                    >= self.config.operation_limit(k, operation)
                    for operation in target_operations[k]
                ):
                    exhausted_without_plateau.append(k)
            if exhausted_without_plateau:
                self.event(
                    dict(
                        event="adaptive_phase_halted",
                        phase=phase,
                        reason="operation_budget_exhausted_before_per_k_plateau",
                        cycle=cycle,
                        k=exhausted_without_plateau,
                        coverage_debt=self._coverage_debt(phase, requirements),
                        composition_coverage_debt=self._composition_coverage_debt(ks),
                    )
                )
                print(f"[adaptive] phase={phase} halted: stage budget", flush=True)
                return False
            if (
                cycle + 1 >= self.config.min_cycles
                and all(
                    stagnant_runs[k] >= self.config.convergence_patience for k in ks
                )
                and self._coverage_complete(phase, requirements)
                and self._composition_coverage_complete(ks)
            ):
                self.event(
                    dict(
                        event="adaptive_phase_done",
                        phase=phase,
                        reason="discovery_and_energy_plateau",
                        cycle=cycle,
                        composition_coverage_debt={},
                    )
                )
                print(f"[adaptive] phase={phase} complete: plateau", flush=True)
                return True
        self.event(
            dict(
                event="adaptive_phase_halted",
                phase=phase,
                reason="maximum_cycles_without_plateau",
            coverage_debt=self._coverage_debt(phase, requirements),
            composition_coverage_debt=self._composition_coverage_debt(ks),
            stagnant_runs=stagnant_runs,
            )
        )
        print(f"[adaptive] phase={phase} halted: maximum cycles", flush=True)
        return False

    def _adaptive_report(self):
        plans = self._plans()
        rows = []
        for k in sorted(self.config.family_slots):
            coarse_families = {
                self.adapter.family(row)
                for row in self.rows["experimental"].values()
                if row["k"] == k and row["role"] == "primary"
            }
            exact_families = {
                self.adapter.subfamily(row)
                for row in self.rows["experimental"].values()
                if row["k"] == k and row["role"] == "primary"
            }
            cohorts = {
                phase: len(self._cohort_state(phase, k)[1])
                for phase in self.config.enabled_phases
            }
            rows.append(
                dict(
                    k=k,
                    primary_families=len(coarse_families),
                    exact_graph_families=len(exact_families),
                    primary_minima=sum(
                        row["k"] == k and row["role"] == "primary"
                        for row in self.rows["experimental"].values()
                    ),
                    calls=self.calls["experimental", k],
                    operation_calls={
                        operation: self.operation_calls_used[k, operation]
                        for operation in ("fixed", "growth")
                    },
                    cohorts=cohorts,
                )
            )
        cycles = [e for e in self.events if e["event"] == "adaptive_cycle"]
        phases = [
            e
            for e in self.events
            if e["event"] in {"adaptive_phase_done", "adaptive_phase_halted"}
        ]
        self.atomic_json(
            self.output / "adaptive_assessment.json",
            dict(
                chemistry_adapter=self.adapter.name,
                stages=rows,
                cycles=cycles,
                phases=phases,
                family_plans=sum(plans.values()),
                enabled_phases=self.config.enabled_phases,
            ),
        )

    def run(self):
        self.setup()
        self.recover()
        self.initialize_source()
        phase_a = True
        if "A" in self.config.enabled_phases:
            phase_a = self._run_phase("A", list(self.config.phase_a_k))
        phase_b = phase_a
        if phase_a and "B" in self.config.enabled_phases:
            phase_b = self._run_phase(
                "B", [self.config.phase_b_k], extend_from=self.config.phase_b_k - 1
            )
        if phase_b and "C" in self.config.enabled_phases:
            self._run_phase(
                "C", [self.config.phase_c_k], extend_from=self.config.phase_c_k - 1
            )
        self.checkpoint()
        self.report()
        self._adaptive_report()
        print(f"[adaptive] finished; results={self.output}", flush=True)


__all__ = [
    "AdaptiveConfig",
    "AdaptivePilot",
    "CdSeCdClAdapter",
    "lineage_family",
    "load_adapter",
]
