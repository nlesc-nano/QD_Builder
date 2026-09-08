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

    version: int = 2
    k_max: int = 8
    max_calls: int = 18_000
    seed_calls: int = 0
    p_max: dict = field(
        default_factory=lambda: {1: 3, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11}
    )
    stage_calls: dict = field(
        default_factory=lambda: {4: 1_600, 5: 2_400, 6: 3_200, 7: 5_000, 8: 5_000}
    )
    family_slots: dict = field(
        default_factory=lambda: {4: 80, 5: 120, 6: 150, 7: 160, 8: 160}
    )
    p_states_per_family: int = 3
    geometries_per_family: int = 2
    phase_a_k: list = field(default_factory=lambda: [4, 5, 6])
    phase_b_k: int = 7
    phase_c_k: int = 8
    min_cycles: int = 2
    max_cycles: int = 8
    convergence_patience: int = 2
    new_families_per_100_calls: float = 5.0
    energy_improvement_eV: float = 0.05
    proposals_per_family_fixed: int = 3
    proposals_per_family_growth: int = 6
    chemistry_adapter: str = "cdse_cdcl2"

    @classmethod
    def load(cls, path):
        raw = yaml.safe_load(Path(path).read_text()) or {}
        value = cls(**raw)
        if value.version != 2:
            raise ValueError("adaptive configuration requires version 2")
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
        value.family_slots = {int(k): int(v) for k, v in value.family_slots.items()}
        expected = list(range(1, value.phase_c_k + 1))
        if sorted(value.p_max) != expected:
            raise ValueError("p_max must cover every k through Phase C")
        if value.phase_a_k != list(range(value.phase_a_k[0], value.phase_a_k[-1] + 1)):
            raise ValueError("phase_a_k must be a contiguous increasing range")
        searched = set(value.phase_a_k + [value.phase_b_k, value.phase_c_k])
        if not searched <= set(value.stage_calls) or not searched <= set(value.family_slots):
            raise ValueError("stage_calls and family_slots must cover all phase k values")
        if sum(value.stage_calls.values()) > value.max_calls:
            raise ValueError("per-k stage_calls exceed max_calls")
        if not (1 <= value.p_states_per_family <= 4):
            raise ValueError("invalid p_states_per_family")
        if not (1 <= value.geometries_per_family <= 2):
            raise ValueError("invalid geometries_per_family")
        if not (1 <= value.min_cycles <= value.max_cycles <= 20):
            raise ValueError("invalid adaptive cycle bounds")
        if not (1 <= value.convergence_patience <= value.max_cycles):
            raise ValueError("invalid convergence patience")
        if min(value.stage_calls.values()) <= 0 or min(value.family_slots.values()) <= 0:
            raise ValueError("stage and family budgets must be positive")
        if value.proposals_per_family_fixed < 3:
            raise ValueError("fixed-family budget must cover all fixed-k move classes")
        return value

    def stage_limits(self):
        return dict(self.stage_calls)


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


class CdSeCdClAdapter:
    """Chemistry hook for the present Cd_(k+p)Se_kCl_(2p) model."""

    name = "cdse_cdcl2"
    fixed_channels = ("topology", "exchange", "reconstruction")

    @staticmethod
    def family(row):
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
        each = max(1, limit // len(self.fixed_channels))
        for channel in self.fixed_channels:
            rng = pilot.rng("adaptive", "fixed", cycle, channel, parent["minimum_id"])
            jobs.extend(
                local_proposals(
                    parent,
                    rng,
                    pilot.spec,
                    pilot.pack,
                    channel=channel,
                    limit=each,
                )
            )
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

    def allowed(self, arm, stage):
        return (
            arm == "experimental"
            and stage in self.config.stage_limits()
            and self.calls[arm, stage] < self.config.stage_limits()[stage]
            and sum(self.calls.values()) < self.config.max_calls
            and not self.expired()
        )

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

    def adaptive_parents(self, k, phase, operation, cycle=0):
        rows = [r for r in self.rows["experimental"].values() if r["k"] == k]
        by_family = defaultdict(list)
        for row in rows:
            by_family[self.adapter.family(row)].append(row)
        plans = self._plans()

        # Rank energies only within fixed composition.  Energies at different p
        # are not compared without ligand/cation reservoirs.
        energy_rank = {}
        for p in sorted({r["p"] for r in rows}):
            ordered = sorted(
                (r for r in rows if r["p"] == p and r["role"] == "primary"),
                key=lambda r: (r["energy_eV"], r["minimum_id"]),
            )
            for rank, row in enumerate(ordered):
                energy_rank[row["minimum_id"]] = rank / max(1, len(ordered) - 1)

        # Retention is stable and scientifically ranked.  The ledger orders
        # work *within* that retained set; it must not rotate through an
        # unbounded tail and thereby defeat the family cap.
        families = sorted(
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
        )[: self.config.family_slots[k]]
        families.sort(key=lambda family_id: (plans[phase, operation, family_id], family_id))
        selected = []
        for family_id in families:
            candidates = by_family[family_id]
            primary_candidates = [r for r in candidates if r["role"] == "primary"]
            if primary_candidates:
                candidates = primary_candidates
            clean_lineage = [r for r in candidates if not r.get("audit_derived", False)]
            if clean_lineage:
                candidates = clean_lineage
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
            for _, group in sorted(by_parent.items())
        ]
        ordered = []
        while any(queues):
            for queue in queues:
                if queue:
                    ordered.append(queue.popleft())
        # The running audit quota cannot accept an audit-derived proposal at
        # call zero.  Put clean lineages first so audit work is not silently
        # discarded merely because of queue order.
        return [job for job in ordered if not job.audit_derived] + [
            job for job in ordered if job.audit_derived
        ]

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
                )
            )
            jobs.extend(made)
        return self._fair_jobs(
            proposal
            for proposal in jobs
            if proposal.k == target_k
            and 1 <= proposal.p <= self.config.p_max[target_k]
            and self.proposal_valid(proposal)
        )

    def _snapshot(self, ks):
        families = {
            self.adapter.family(row)
            for row in self.rows["experimental"].values()
            if row["k"] in ks and row["role"] == "primary"
        }
        best = {}
        for row in self.rows["experimental"].values():
            key = (row["k"], row["p"])
            if row["k"] in ks and row["role"] == "primary":
                best[key] = min(best.get(key, float("inf")), row["energy_eV"])
        return families, best, sum(self.calls.values())

    def _cycle_metrics(self, phase, cycle, before, after):
        old_families, old_best, old_calls = before
        new_families, new_best, new_calls = after
        calls = new_calls - old_calls
        discovered = len(new_families - old_families)
        rate = 100.0 * discovered / max(1, calls)
        improvements = [
            old_best[key] - energy
            for key, energy in new_best.items()
            if key in old_best and energy < old_best[key]
        ]
        improvement = max(improvements, default=0.0)
        stagnant = calls > 0 and (
            rate < self.config.new_families_per_100_calls
            and improvement < self.config.energy_improvement_eV
        )
        event = dict(
            event="adaptive_cycle",
            phase=phase,
            cycle=cycle,
            calls=calls,
            new_primary_families=discovered,
            new_families_per_100_calls=rate,
            maximum_within_composition_improvement_eV=improvement,
            stagnant=stagnant,
        )
        self.event(event)
        return event

    def _coverage_complete(self, phase, requirements, cycle):
        plans = self._plans()
        launched = {
            (event["arm"], event["id"])
            for event in self.events
            if event["event"] == "launch"
        }
        for operation, source_k in requirements:
            retained = {
                family_id
                for family_id, _ in self.adaptive_parents(
                    source_k, phase, operation, cycle
                )
            }
            minimum_plans = min(
                self.config.min_cycles, self.config.p_states_per_family
            )
            if any(
                plans[phase, operation, family_id] < minimum_plans
                for family_id in retained
            ):
                return False
            for family_id in retained:
                family_events = [
                    event
                    for event in self.events
                    if event["event"] == "family_plan"
                    and event["phase"] == phase
                    and event["operation"] == operation
                    and event["family_id"] == family_id
                ]
                proposed = {
                    proposal_id
                    for event in family_events
                    for proposal_id in event.get("proposal_ids", [])
                }
                if proposed and not any(
                    ("experimental", proposal_id) in launched for proposal_id in proposed
                ):
                    return False
        return True

    def _run_queue(self, phase, cycle, operation, source_k, target_k):
        tag = f"{phase}:{cycle}:{operation}:{source_k}:{target_k}"
        if ("experimental", target_k, tag) in self.stage_done:
            return
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
        stagnant_run = 0
        for event in reversed(old_cycles):
            if not event["stagnant"]:
                break
            stagnant_run += 1
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
            print(
                f"[adaptive] phase={phase} cycle={cycle} calls={metrics['calls']} "
                f"new_families={metrics['new_primary_families']} "
                f"rate_per_100={metrics['new_families_per_100_calls']:.2f} "
                f"best_dE={metrics['maximum_within_composition_improvement_eV']:.4f}",
                flush=True,
            )
            stagnant_run = stagnant_run + 1 if metrics["stagnant"] else 0
            self.checkpoint()
            if metrics["calls"] == 0 and any(
                self.calls["experimental", k] >= self.config.stage_limits()[k]
                for k in ks
            ):
                self.event(
                    dict(
                        event="adaptive_phase_halted",
                        phase=phase,
                        reason="stage_budget_exhausted_before_plateau",
                        cycle=cycle,
                    )
                )
                print(f"[adaptive] phase={phase} halted: stage budget", flush=True)
                return False
            if (
                cycle + 1 >= self.config.min_cycles
                and stagnant_run >= self.config.convergence_patience
                and self._coverage_complete(phase, requirements, cycle)
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
            )
        )
        print(f"[adaptive] phase={phase} halted: maximum cycles", flush=True)
        return False

    def _adaptive_report(self):
        plans = self._plans()
        rows = []
        for k in sorted(self.config.family_slots):
            families = {
                self.adapter.family(row)
                for row in self.rows["experimental"].values()
                if row["k"] == k and row["role"] == "primary"
            }
            rows.append(
                dict(
                    k=k,
                    primary_families=len(families),
                    primary_minima=sum(
                        row["k"] == k and row["role"] == "primary"
                        for row in self.rows["experimental"].values()
                    ),
                    calls=self.calls["experimental", k],
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
            ),
        )

    def run(self):
        self.setup()
        self.recover()
        self.initialize_source()
        phase_a = self._run_phase("A", list(self.config.phase_a_k))
        phase_b = phase_a and self._run_phase(
            "B", [self.config.phase_b_k], extend_from=self.config.phase_b_k - 1
        )
        if phase_b:
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
