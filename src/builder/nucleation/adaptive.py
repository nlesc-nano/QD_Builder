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
    p_states_per_family: int = 3
    geometries_per_family: int = 2
    subfamilies_per_family: int = 2
    admission_fraction: float = 0.15
    minimum_launched_cycles: int = 2
    enabled_phases: list = field(default_factory=lambda: ["A", "B", "C"])
    phase_a_k: list = field(default_factory=lambda: [4, 5, 6])
    phase_b_k: int = 7
    phase_c_k: int = 8
    min_cycles: int = 2
    max_cycles: int = 8
    convergence_patience: int = 2
    new_families_per_100_calls: float = 5.0
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
        value.p_max = {int(k): int(v) for k, v in value.p_max.items()}
        value.stage_calls = {int(k): int(v) for k, v in value.stage_calls.items()}
        value.operation_calls = {
            int(k): {str(operation): int(limit) for operation, limit in limits.items()}
            for k, limits in value.operation_calls.items()
        }
        value.family_slots = {int(k): int(v) for k, v in value.family_slots.items()}
        expected = list(range(1, value.phase_c_k + 1))
        if sorted(value.p_max) != expected:
            raise ValueError("p_max must cover every k through Phase C")
        if value.phase_a_k != list(range(value.phase_a_k[0], value.phase_a_k[-1] + 1)):
            raise ValueError("phase_a_k must be a contiguous increasing range")
        if not value.enabled_phases or any(
            phase not in {"A", "B", "C"} for phase in value.enabled_phases
        ):
            raise ValueError("enabled_phases must contain A, B and/or C")
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
    """Coordination/ring/geometry family used for bounded lineage survival.

    Exact graph topology remains available through :func:`lineage_family` as a
    subfamily.  The coarser key prevents every graph edit from consuming a new
    beam slot while still separating compact geometrically distinct motifs.
    """

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
        if any(event["event"] == "archive_import" for event in self.events):
            return
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

    def _plans(self):
        return Counter(
            (event["phase"], event["operation"], event["family_id"])
            for event in self.events
            if event["event"] == "family_plan"
        )

    def _ranked_families(self, k):
        rows = [r for r in self.rows["experimental"].values() if r["k"] == k]
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
        ranked, _ = self._ranked_families(k)
        if not ranked:
            return []
        initialized, cohort = self._cohort_state(phase, k)
        capacity = self.config.family_slots[k]
        if initialized is None:
            initial_capacity = max(
                1, int(capacity * (1.0 - self.config.admission_fraction))
            )
            cohort = ranked[:initial_capacity]
            self.event(
                dict(
                    event="cohort_initialized",
                    phase=phase,
                    k=k,
                    cycle=cycle,
                    capacity=capacity,
                    families=cohort,
                    known_families=ranked,
                )
            )
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
            rows = [
                row
                for row in self.rows["experimental"].values()
                if row["k"] == k and row["role"] == "primary"
            ]
            best = {}
            for row in rows:
                best[row["p"]] = min(
                    best.get(row["p"], float("inf")), row["energy_eV"]
                )
            snapshot[k] = dict(
                families={self.adapter.family(row) for row in rows},
                best=best,
                calls=self.calls["experimental", k],
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
                maximum_within_composition_improvement_eV=improvement,
                usable_endpoints=usable,
                endpoint_fraction=endpoint_fraction,
                stagnant=calls > 0
                and endpoint_fraction >= self.config.minimum_endpoint_fraction
                and rate < self.config.new_families_per_100_calls
                and improvement < self.config.energy_improvement_eV,
            )
        calls = sum(metrics["calls"] for metrics in per_k.values())
        discovered = sum(
            metrics["new_primary_families"] for metrics in per_k.values()
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

    def _run_queue(self, phase, cycle, operation, source_k, target_k):
        tag = f"{phase}:{cycle}:{operation}:{source_k}:{target_k}"
        if ("experimental", target_k, tag) in self.stage_done:
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
        self.evaluate([("experimental", proposal) for proposal in queue], target_k)
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
        start = max((event["cycle"] for event in old_cycles), default=-1) + 1
        requirements = [("fixed", k) for k in ks]
        requirements.extend(("growth", k) for k in ks[:-1])
        if extend_from is not None:
            requirements.append(("growth", extend_from))
        for cycle in range(start, self.config.max_cycles):
            if self.expired() or sum(self.calls.values()) >= self.config.max_calls:
                return False
            print(f"[adaptive] phase={phase} cycle={cycle} started", flush=True)
            before = self._snapshot(ks)
            if extend_from is not None:
                self._run_queue(phase, cycle, "growth", extend_from, ks[0])
            for index, k in enumerate(ks):
                self._run_queue(phase, cycle, "fixed", k, k)
                if index + 1 < len(ks):
                    self._run_queue(phase, cycle, "growth", k, ks[index + 1])
            after = self._snapshot(ks)
            metrics = self._cycle_metrics(phase, cycle, before, after)
            for k in ks:
                km = metrics["per_k"][str(k)]
                print(
                    f"[adaptive] phase={phase} cycle={cycle} k={k} "
                    f"calls={km['calls']} new_families={km['new_primary_families']} "
                    f"rate_per_100={km['new_families_per_100_calls']:.2f} "
                    f"best_dE={km['maximum_within_composition_improvement_eV']:.4f}",
                    flush=True,
                )
                stagnant_runs[k] = (
                    stagnant_runs[k] + 1 if km["stagnant"] else 0
                )
            self.checkpoint()
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
            ):
                self.event(
                    dict(
                        event="adaptive_phase_done",
                        phase=phase,
                        reason="discovery_and_energy_plateau",
                        cycle=cycle,
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
