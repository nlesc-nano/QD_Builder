"""Opt-in, resumable, budgeted comparison of lattice and molecular searches.

Only Pilot.evaluate may invoke the electronic-structure backend. Every call is
reserved durably before submission. An interrupted reservation remains charged.
"""
from __future__ import annotations
import json
import hashlib
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import yaml

from .basin_hop import describe_minimum
from .geometry_pack import load_geometry_pack
from .molecular_growth import (
    GrowthConfig,
    MinimumConsolidation,
    ParentStructure,
    _select_zb_occupation_parents,
    bond_cutoffs_from_spec,
    load_parents_from_run,
)
from .molecular_zb_growth import (
    lattice_model,
    lattice_k1_occupation,
    occupation_from_record,
    occupation_to_record,
    grow_zb_children,
    ztype_shed_units,
)
from .pilot_proposals import (
    Proposal,
    embed_core,
    fresh_cores,
    local_proposals,
    validate_composition,
    graph_state,
)
from .search_analysis import MinimumArchive, corrected_occupation, descriptors, digest
from .spec import load_nucleation_spec
from .xtb_relax import XtbSettings, relax_structures


@dataclass
class PilotConfig:
    version: int = 1
    seed: int = 1729
    k_max: int = 6
    workers: int = 24
    max_calls: int = 4000
    seed_calls: int = 128
    launch_hours: float = 46
    timeout_s: int = 1800
    max_steps: int = 150
    p_max: dict = field(default_factory=lambda: {1: 3, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9})
    shares: dict = field(
        default_factory=lambda: dict(
            growth=0.5, exchange=0.2, reconstruction=0.15, fresh=0.15
        )
    )
    bridge_prior: float = 0.8
    primary_slots: int = 12
    audit_slots: int = 2
    audit_fraction: float = 0.1
    proposals_per_move: int = 8
    proposals_per_parent: int = 32
    fixed_rounds: int = 2
    checkpoint_interval_batches: int = 1
    progress_interval_batches: int = 1

    @classmethod
    def load(cls, path):
        value = cls(**(yaml.safe_load(Path(path).read_text()) or {}))
        if value.version != 1:
            raise ValueError("unsupported pilot configuration version")
        if not (1 <= value.primary_slots <= 12 and 0 <= value.audit_slots <= 2):
            raise ValueError("invalid population limits")
        if not (
            1 <= value.proposals_per_move <= 8 and 1 <= value.proposals_per_parent <= 32
        ):
            raise ValueError("proposal limits exceed bounded pilot")
        if not 1 <= value.max_steps <= 150:
            raise ValueError("invalid optimization cycle limit")
        if not (2 <= value.k_max <= 6 and 1 <= value.workers <= 24):
            raise ValueError("pilot supports k=2..6 and 1..24 workers")
        if not (0 < value.seed_calls < value.max_calls and value.max_calls <= 4000):
            raise ValueError("invalid call budget")
        if not (0 < value.launch_hours <= 46 and 0 < value.timeout_s <= 1800):
            raise ValueError("invalid deadline")
        if (
            set(value.shares) != {"growth", "exchange", "reconstruction", "fresh"}
            or abs(sum(value.shares.values()) - 1) > 1e-9
        ):
            raise ValueError("channel shares must be nonnegative and sum to one")
        if min(value.shares.values()) < 0 or not 0 <= value.bridge_prior <= 1:
            raise ValueError("invalid fraction")
        if not 0 <= value.audit_fraction <= 0.1 or value.audit_slots > 2:
            raise ValueError("audit allowance exceeds pilot limit")
        if value.fixed_rounds > 2 or value.fixed_rounds < 0:
            raise ValueError("at most two fixed-k rounds")
        if value.checkpoint_interval_batches < 1:
            raise ValueError("checkpoint_interval_batches must be positive")
        if value.progress_interval_batches < 1:
            raise ValueError("progress_interval_batches must be positive")
        value.p_max = {int(k): int(v) for k, v in value.p_max.items()}
        if any(
            k not in value.p_max or value.p_max[k] < 1
            for k in range(1, value.k_max + 1)
        ):
            raise ValueError("missing composition bounds")
        return value

    def stage_limits(self):
        total = (self.max_calls - self.seed_calls) // 2
        weights = np.array([10, 15, 20, 25, 30][: self.k_max - 1], float)
        exact = weights / weights.sum() * total
        limits = np.floor(exact).astype(int)
        for i in sorted(range(len(limits)), key=lambda i: (-(exact[i] - limits[i]), i))[
            : total - int(limits.sum())
        ]:
            limits[i] += 1
        return {k: int(n) for k, n in zip(range(2, self.k_max + 1), limits)}


def select_population(rows, rng, primary_slots=12, audit_slots=2):
    """Six energy, four family coverage, two exploration; two shells/core."""
    primary = sorted(
        [r for r in rows if r["role"] == "primary"],
        key=lambda r: (r["energy_eV"], r["minimum_id"]),
    )
    selected = []
    cores = Counter()
    ids = set()

    def take(r):
        core = r["final"].get("core_geometry_hash", r["final"]["core_hash"])
        if r["minimum_id"] in ids or cores[core] >= 2:
            return False
        selected.append(r)
        ids.add(r["minimum_id"])
        cores[core] += 1
        return True

    for r in primary:
        if len(selected) >= min(6, primary_slots):
            break
        take(r)
    signatures = {family(r) for r in selected}
    for r in primary:
        if len(selected) >= min(10, primary_slots):
            break
        if family(r) not in signatures and take(r):
            signatures.add(family(r))
    remaining = [r for r in primary if r["minimum_id"] not in ids]
    rng.shuffle(remaining)
    explored = 0
    for r in remaining:
        if len(selected) >= primary_slots or explored >= 2:
            break
        explored += take(r)
    for r in primary:
        if len(selected) >= primary_slots:
            break
        take(r)
    audit = sorted(
        [r for r in rows if r["role"] == "audit"],
        key=lambda r: (r["energy_eV"], r["minimum_id"]),
    )
    return selected + audit[:audit_slots]


def family(row):
    f = row["final"]
    return (
        f["core_hash"],
        tuple(sorted(f["cd_environments"].items())),
        f["mu2"],
        f["mu3"],
    )


def classification_observations(row):
    """Return the distinct classification decisions represented by ``row``.

    A relaxed geometry can be evaluated under more than one graph-rule policy.
    Those decisions describe eligibility, not distinct potential-energy minima,
    and must therefore survive geometric basin consolidation.
    """

    observations = row.get("classification_history")
    if not observations:
        observations = [
            dict(
                role=row.get("role", "unknown"),
                violations=sorted(set(row.get("violations", []))),
                classification_mode=row.get("classification_mode", "unknown"),
                energy_eV=row.get("energy_eV"),
                structure_id=row.get("structure_id"),
                source=row.get("source"),
            )
        ]
    unique = {}
    for observation in observations:
        value = dict(observation)
        value["violations"] = sorted(set(value.get("violations", [])))
        key = json.dumps(value, sort_keys=True, default=str, allow_nan=False)
        unique[key] = value
    return [unique[key] for key in sorted(unique)]


def add_classification_provenance(row, *others):
    """Attach non-destructive multi-policy classification provenance."""

    observations = classification_observations(row)
    for other in others:
        observations.extend(classification_observations(other))
    unique = {}
    for observation in observations:
        key = json.dumps(observation, sort_keys=True, default=str, allow_nan=False)
        unique[key] = observation
    history = [unique[key] for key in sorted(unique)]
    modes = defaultdict(set)
    for observation in history:
        modes[observation.get("role", "unknown")].add(
            observation.get("classification_mode", "unknown")
        )
    row["classification_history"] = history
    row["classification_eligibility"] = {
        role: sorted(values) for role, values in sorted(modes.items())
    }
    return row


def round_robin(items):
    bins = defaultdict(deque)
    for item in items:
        bins[item.p].append(item)
    out = []
    while any(bins.values()):
        for p in sorted(bins):
            if bins[p]:
                out.append(bins[p].popleft())
    return out


class Pilot:
    def __init__(
        self, config, map_path, growth_path, seed_dir, output, *, backend=None
    ):
        self.config = config
        self.map_path = Path(map_path)
        self.seed_dir = Path(seed_dir)
        self.output = Path(output)
        self.spec = load_nucleation_spec(self.map_path)
        self.pack = load_geometry_pack(self.map_path)
        # A validation branch may explicitly soften a named graph rule while
        # retaining the original pack on disk.  Apply the override to both
        # executable representations so proposal generation and final
        # post-relax classification use the same chemistry.  The override is
        # part of the adaptive config/protocol fingerprint and therefore
        # cannot be mixed with a strict continuation.
        overrides = config.chemistry_options.get("graph_rule_overrides", {})
        if overrides:
            if not isinstance(overrides, dict):
                raise ValueError("graph_rule_overrides must be a mapping")
            unknown = sorted(
                key for key in overrides
                if not hasattr(self.spec.graph_rules, str(key))
            )
            if unknown:
                raise ValueError(
                    f"unknown nucleation graph-rule override(s): {unknown}"
                )
            self.spec = replace(
                self.spec,
                graph_rules=replace(self.spec.graph_rules, **overrides),
            )
            self.pack.graph_rules.update(overrides)
            self.pack.raw.setdefault("graph_rules", {}).update(overrides)
        self.growth = GrowthConfig.from_yaml(Path(growth_path))
        self.model = lattice_model(self.spec)
        self.cutoffs = bond_cutoffs_from_spec(self.spec)
        raw = dict(self.pack.raw.get("relaxation", {}))
        self.settings = replace(
            XtbSettings.from_pack(raw),
            enabled=True,
            workers=1,
            threads_per_worker=1,
            max_steps=config.max_steps,
            timeout_s=config.timeout_s,
            accept_maxcycle=True,
        )
        self.backend = backend or relax_structures
        self._real_backend = backend is None
        self.calls = Counter()
        self.worker_seconds = Counter()
        self.channel_calls = Counter()
        self.audit_calls = Counter()
        self.rows = {"control": {}, "experimental": {}}
        self.archive = MinimumArchive(self.spec)
        self.completed = set()
        self.reserved = set()
        self.events = []
        self.stage_done = set()
        self.proposal_routes = defaultdict(set)
        self.queues = {}
        self.started = time.time()
        self.elapsed_before = 0.0
        self.protocol = None
        self._batches_since_checkpoint = 0

    def extra_protocol_sources(self):
        """Additional source hashes supplied by a protocol subclass."""

        return {}

    def setup(self):
        binary = shutil.which(self.settings.binary or "gxtb")
        if self._real_backend and not binary:
            raise RuntimeError(f"g-xTB executable not found: {self.settings.binary}")
        repo = Path(__file__).resolve().parents[3]
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
        )
        # Include working source content: a dirty tree must not resume as its HEAD.
        sources = {
            str(p.relative_to(repo)): digest(p.read_text())
            for p in sorted((repo / "src/builder/nucleation").glob("*.py"))
        }
        sources.update(self.extra_protocol_sources())
        seed_index = self.seed_dir / "index.csv"

        def file_hash(path):
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            return h.hexdigest()

        seed_hashes = {
            str(p.relative_to(self.seed_dir)): file_hash(p)
            for p in sorted(self.seed_dir.rglob("*.xyz"))
        }
        # Continuation protocols may import a completed pilot archive rather
        # than a legacy XYZ/index corpus.  Its scientific contents are part of
        # the immutable protocol fingerprint too.
        for name in ("minima.json", "protocol.json", "status.json"):
            candidate = self.seed_dir / name
            if candidate.is_file():
                seed_hashes[name] = file_hash(candidate)
        fingerprint = digest(
            dict(
                config=asdict(self.config),
                pack=self.pack.raw,
                growth=asdict(self.growth),
                sources=sources,
                seed_index=digest(seed_index.read_text())
                if seed_index.exists()
                else None,
                seed_hashes=seed_hashes,
                binary=file_hash(Path(binary)) if binary else "test-backend",
                xtbpath=os.environ.get("XTBPATH", ""),
            )
        )
        self.protocol = dict(
            fingerprint=fingerprint,
            config=asdict(self.config),
            pack=self.pack.raw,
            source_revision=git.stdout.strip(),
            source_hashes=sources,
            seed_hashes=seed_hashes,
            binary=binary,
            seed_dir=str(self.seed_dir.resolve()),
            created=self.started,
        )
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "protocol.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["fingerprint"] != fingerprint:
                raise ValueError(
                    "protocol fingerprint differs; use a new output directory"
                )
            self.protocol = previous
            status = self.output / "status.json"
            if status.exists():
                self.elapsed_before = (
                    float(json.loads(status.read_text()).get("elapsed_hours", 0)) * 3600
                )
        else:
            self.atomic_json(path, self.protocol)
        self.replay()

    @staticmethod
    def atomic_json(path, value):
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w") as fh:
            json.dump(value, fh, sort_keys=True, default=str, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        temp.replace(path)

    def event(self, data):
        with (self.output / "events.jsonl").open("a") as fh:
            fh.write(json.dumps(data, sort_keys=True, allow_nan=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.events.append(data)

    def replay(self):
        path = self.output / "events.jsonl"
        if not path.exists():
            return
        valid = []
        with path.open() as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except ValueError:
                    raise ValueError(
                        "incomplete event journal; preserve it and recover before resume"
                    )
                valid.append(ev)
        for ev in valid:
            kind = ev["event"]
            if kind == "launch":
                arm, k = ev["arm"], ev["stage"]
                self.calls[arm, k] += 1
                self.reserved.add((arm, ev["id"], ev["attempt"]))
                self.channel_calls[arm, k, ev["proposal"]["channel"]] += 1
                self.audit_calls[arm, k] += ev["proposal"].get("audit_derived", False)
            elif kind == "result":
                self.completed.add((ev["arm"], ev["id"], ev["attempt"]))
                self.worker_seconds[ev["arm"]] += ev["seconds"]
                if ev.get("row"):
                    self.store(ev["arm"], ev["row"], replay=True)
            elif kind == "stage_done":
                self.stage_done.add((ev["arm"], ev["stage"], ev["round"]))
            elif kind == "queue":
                self.queues[ev["arm"], ev["stage"], ev["round"]] = ev["proposals"]
            elif kind == "archive_import":
                for row in ev["rows"]:
                    self.store(
                        ev.get("arm", "experimental"),
                        row,
                        replay=True,
                        preferred_id=row.get("minimum_id"),
                    )
        self.events = valid

    def prepare_queue(self, arm, k, round_number, build):
        key = (arm, k, round_number)
        if key not in self.queues:
            records = [p.record() for p in build()]
            self.event(
                dict(
                    event="queue",
                    arm=arm,
                    stage=k,
                    round=round_number,
                    proposals=records,
                )
            )
            self.queues[key] = records
        return [Proposal(**r) for r in self.queues[key]]

    def allowed(self, arm, stage, proposal=None):
        limit = (
            self.config.seed_calls
            if arm == "shared"
            else self.config.stage_limits()[stage]
        )
        return (
            self.calls[arm, stage] < limit
            and sum(self.calls.values()) < self.config.max_calls
            and not self.expired()
        )

    def expired(self):
        return (
            self.elapsed_before + time.time() - self.started
            >= self.config.launch_hours * 3600
        )

    def store(self, arm, row, replay=False, preferred_id=None):
        row = dict(row)
        mid, fresh = self.archive.add(row, row["positions"], preferred_id=preferred_id)
        row["minimum_id"] = mid
        for target in ["control", "experimental"] if arm == "shared" else [arm]:
            if target == "control" and row["role"] == "audit":
                continue
            old = self.rows[target].get(mid)
            if old:
                old["routes"] = sorted(
                    set(old.get("routes", []) + row.get("routes", [])) - {mid}
                )
                # Preserve every ideal occupation mapping into this basin.
                occupations = {
                    o["occupation_id"]: o
                    for o in old.get("occupations", []) + row.get("occupations", [])
                }
                old["occupations"] = list(occupations.values())
                origins = {
                    o["occupation"]["occupation_id"]: o
                    for o in old.get("occupation_origins", [])
                    + row.get("occupation_origins", [])
                }
                old["occupation_origins"] = list(origins.values())
                # Role is a graph-policy eligibility decision.  Never let a
                # slightly lower-energy audit realization make an established
                # primary basin disappear.  Conversely, a primary realization
                # upgrades an audit basin even when it is marginally higher in
                # energy.  Preserve both decisions as provenance.
                role_rank = {"primary": 0, "audit": 1}
                replace_row = (
                    role_rank.get(row.get("role"), 2), row["energy_eV"]
                ) < (
                    role_rank.get(old.get("role"), 2), old["energy_eV"]
                )
                classification_differs = (
                    row.get("role") != old.get("role")
                    or row.get("classification_mode")
                    != old.get("classification_mode")
                    or sorted(row.get("violations", []))
                    != sorted(old.get("violations", []))
                    or "classification_history" in row
                    or "classification_history" in old
                )
                selected = row.copy() if replace_row else old
                selected["routes"] = old["routes"]
                selected["occupations"] = old["occupations"]
                selected["occupation_origins"] = old["occupation_origins"]
                if classification_differs:
                    add_classification_provenance(selected, old, row)
                self.rows[target][mid] = selected.copy()
            else:
                row["routes"] = sorted(set(row.get("routes", [])) - {mid})
                self.rows[target][mid] = row.copy()
        return mid, fresh

    def classify(self, proposal, xr, arm):
        if (
            not xr.ok
            or xr.coordinates is None
            or xr.energy_eV is None
            or not np.isfinite(xr.energy_eV)
        ):
            return None
        if not validate_composition(proposal):
            return None
        minimum = describe_minimum(
            proposal.id,
            proposal.id,
            0,
            k=proposal.k,
            p=proposal.p,
            symbols=proposal.symbols,
            coordinates=np.asarray(xr.coordinates),
            energy_eV=xr.energy_eV,
            converged=xr.converged,
            spec=self.spec,
            cutoffs=self.cutoffs,
            reference_core_edges=[
                e
                for e in proposal.edges
                if {proposal.symbols[e[0]], proposal.symbols[e[1]]} == {"Cd", "Se"}
            ],
            motif_definitions=self.pack.raw.get("motifs"),
            artifact_floors=self.settings.artifact_min_distance,
        )
        from .molecular_rules import cl_on_cn4_cd_violations

        violations = list(minimum.violations) + cl_on_cn4_cd_violations(
            graph_state(proposal.symbols, minimum.edges, xr.coordinates),
            self.spec,
            np.asarray(xr.coordinates),
        )
        self.last_audit = dict(
            violations=violations,
            n_components=minimum.n_components,
            connectivity_preserved=minimum.core_preserved,
            descriptors=descriptors(proposal.symbols, minimum.edges, xr.coordinates),
        )
        if not xr.converged or minimum.n_components != 1:
            return None
        hard = ("artifact:", "audit_failed:", "forbidden_pair:", "cn0:")
        if any(v.startswith(hard) for v in violations):
            return None
        role = "audit" if violations else "primary"
        if arm == "control" and role == "audit":
            return None
        f = descriptors(proposal.symbols, minimum.edges, xr.coordinates)
        return dict(
            structure_id=proposal.id,
            k=proposal.k,
            p=proposal.p,
            symbols=proposal.symbols,
            positions=np.asarray(xr.coordinates).tolist(),
            edges=[list(e) for e in minimum.edges],
            energy_eV=float(xr.energy_eV),
            role=role,
            violations=violations,
            final=f,
            protocol=self.protocol["fingerprint"],
            source=f"{arm}:{proposal.id}",
            routes=proposal.parents,
            occupations=[proposal.occupation] if proposal.occupation else [],
            occupation_origins=[
                dict(
                    occupation=proposal.occupation,
                    symbols=proposal.symbols,
                    positions=np.asarray(xr.coordinates).tolist(),
                    edges=[list(e) for e in minimum.edges],
                    energy_eV=float(xr.energy_eV),
                    connectivity_preserved=minimum.core_preserved,
                )
            ]
            if proposal.occupation
            else [],
            connectivity_preserved=minimum.core_preserved,
            audit_derived=proposal.audit_derived,
            classification_mode=self.config.chemistry_options.get(
                "classification_mode", "strict"
            ),
            construction=descriptors(proposal.symbols, proposal.edges),
            converged=True,
        )

    def proposal_valid(self, proposal):
        """Chemistry hook used before any backend call."""

        return validate_composition(proposal)

    def evaluate(self, tasks, stage):
        """Bounded deterministic batches; backend timing includes failed calls."""
        tasks = deque(tasks)
        batch_number = 0
        while tasks and not self.expired():
            batch = []
            while tasks and len(batch) < self.config.workers:
                arm, p = tasks.popleft()
                key = (arm, p.id, p.attempt)
                if key in self.reserved or not self.allowed(arm, stage, p):
                    continue
                if p.audit_derived and arm != "shared":
                    if self.audit_calls[arm, stage] + 1 > (
                        (self.calls[arm, stage] + 1) * self.config.audit_fraction
                        + 1e-12
                    ):
                        continue
                if not self.proposal_valid(p):
                    raise ValueError("proposal composition mismatch")
                self.event(
                    dict(
                        event="launch",
                        arm=arm,
                        stage=stage,
                        id=p.id,
                        attempt=p.attempt,
                        proposal=p.record(),
                    )
                )
                self.reserved.add(key)
                self.calls[arm, stage] += 1
                self.channel_calls[arm, stage, p.channel] += 1
                self.audit_calls[arm, stage] += p.audit_derived
                batch.append((arm, p))
            if not batch:
                break

            batch_number += 1
            show_progress = (
                batch_number == 1
                or batch_number % self.config.progress_interval_batches == 0
                or not tasks
            )
            if show_progress:
                channels = Counter(p.channel for _, p in batch)
                compositions = Counter(p.p for _, p in batch)
                channel_text = ",".join(
                    f"{name}:{count}" for name, count in sorted(channels.items())
                )
                p_text = ",".join(
                    f"p{p}:{count}" for p, count in sorted(compositions.items())
                )
                print(
                    f"[pilot] launch k={stage} batch={batch_number} size={len(batch)} "
                    f"queue_left={len(tasks)} channels={channel_text or '-'} "
                    f"compositions={p_text or '-'} "
                    f"calls_k={sum(n for (arm, k), n in self.calls.items() if k == stage)} "
                    f"calls_total={sum(self.calls.values())}/{self.config.max_calls}",
                    flush=True,
                )

            def one(task):
                arm, p = task
                start = time.monotonic()
                try:
                    result = self.backend([p.payload()], self.settings, self.cutoffs)[0]
                except Exception as exc:
                    from .xtb_relax import XtbResult

                    result = XtbResult(ok=False, error=f"{type(exc).__name__}: {exc}")
                return result, time.monotonic() - start

            batch_started = time.monotonic()
            with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
                results = list(pool.map(one, batch))
            batch_wall_seconds = time.monotonic() - batch_started
            outcomes = Counter()
            retries = 0
            for (arm, p), (xr, seconds) in zip(batch, results):
                self.last_audit = {}
                row = self.classify(p, xr, arm)
                if row:
                    row["minimum_id"], _ = self.store(arm, row)
                    outcomes[row["role"]] += 1
                elif not xr.ok:
                    outcomes["failed"] += 1
                elif not xr.converged:
                    outcomes["unconverged"] += 1
                else:
                    outcomes["rejected"] += 1
                self.event(
                    dict(
                        event="result",
                        arm=arm,
                        stage=stage,
                        id=p.id,
                        attempt=p.attempt,
                        seconds=seconds,
                        error=xr.error,
                        converged=bool(xr.converged),
                        row=row,
                        audit=self.last_audit,
                        energy_eV=float(xr.energy_eV)
                        if xr.energy_eV is not None and np.isfinite(xr.energy_eV)
                        else None,
                        positions=np.asarray(xr.coordinates).tolist()
                        if xr.coordinates is not None
                        else None,
                    )
                )
                self.completed.add((arm, p.id, p.attempt))
                self.worker_seconds[arm] += seconds
                if (
                    xr.ok
                    and not xr.converged
                    and xr.coordinates is not None
                    and p.attempt == 0
                ):
                    # Same proposal id must remain stable across retry coordinates.
                    retry = replace(
                        p,
                        positions=np.asarray(xr.coordinates).tolist(),
                        attempt=1,
                        origin_id=p.id,
                    )
                    tasks.appendleft((arm, retry))
                    retries += 1
            if show_progress:
                outcome_text = ",".join(
                    f"{name}:{outcomes[name]}"
                    for name in ("primary", "audit", "rejected", "unconverged", "failed")
                    if outcomes[name]
                )
                worker_seconds = sum(seconds for _, seconds in results)
                print(
                    f"[pilot] finish k={stage} batch={batch_number} "
                    f"outcomes={outcome_text or '-'} retries={retries} "
                    f"queue_left={len(tasks)} wall_s={batch_wall_seconds:.1f} "
                    f"worker_s={worker_seconds:.1f} "
                    f"calls_total={sum(self.calls.values())}/{self.config.max_calls}",
                    flush=True,
                )
            self._batches_since_checkpoint += 1
            if (
                self._batches_since_checkpoint
                >= self.config.checkpoint_interval_batches
            ):
                self.checkpoint()
                self._batches_since_checkpoint = 0

    def checkpoint(self):
        self.atomic_json(self.output / "minima.json", self.rows)
        status = dict(
            calls={f"{a}:{k}": n for (a, k), n in self.calls.items()},
            worker_seconds=dict(self.worker_seconds),
            elapsed_hours=(self.elapsed_before + time.time() - self.started) / 3600,
            completed_stages=sorted(self.stage_done),
            reserved_unfinished=len(self.reserved - self.completed),
        )
        status.update(self.extra_status())
        self.atomic_json(
            self.output / "status.json",
            status,
        )

    def extra_status(self):
        return {}

    def rng(self, *parts):
        return np.random.default_rng(int(digest([self.config.seed, *parts])[:16], 16))

    def seeds(self):
        proposals = []
        for k in [1, 2]:
            for parent in load_parents_from_run(self.seed_dir, k=k, spec=self.spec):
                if parent.p > self.config.p_max[k]:
                    continue
                proposals.append(
                    Proposal(
                        k,
                        parent.p,
                        list(parent.symbols),
                        parent.coordinates.tolist(),
                        [list(e) for e in parent.edges],
                        [f"seed:{parent.structure_id}"],
                        "seed",
                        occupation_to_record(parent.zb_occupation)
                        if parent.zb_occupation is not None
                        else None,
                    )
                )
        # Include lattice roots explicitly as construction roots, not historical pathways.
        for p in range(1, self.config.p_max[1] + 1):
            occ = lattice_k1_occupation(self.spec, self.model, p)
            if occ:
                jobs, _ = embed_core(
                    1,
                    p,
                    list(occ.symbols),
                    occ.core_edges,
                    occ.coordinates,
                    self.rng("seed", p),
                    self.spec,
                    self.pack,
                    ["seed:lattice"],
                    "seed",
                    occupation_to_record(occ),
                    limit=2,
                )
                proposals.extend(jobs)
        for k in [1, 2]:
            for p in range(1, self.config.p_max[k] + 1):
                if self.expired():
                    break
                rng = self.rng("fresh-seed", k, p)
                for sy, edges in fresh_cores(k, p, rng, limit=4):
                    jobs, _ = embed_core(
                        k,
                        p,
                        sy,
                        edges,
                        None,
                        rng,
                        self.spec,
                        self.pack,
                        ["seed:graph"],
                        "seed",
                        limit=1,
                    )
                    proposals.extend(jobs)
        # Round-robin over (k,p), not just p, with lattice roots ahead of duplicate imported starts.
        unique = {p.id: p for p in proposals}
        groups = defaultdict(deque)
        for p in sorted(unique.values(), key=lambda p: (p.occupation is None, p.id)):
            groups[p.k, p.p].append(p)
        ordered = []
        while any(groups.values()):
            for key in sorted(groups):
                if groups[key]:
                    ordered.append(groups[key].popleft())
        return ordered[: self.config.seed_calls]

    def parents(self, arm, k):
        bins = defaultdict(list)
        for row in self.rows[arm].values():
            if row["k"] == k:
                bins[row["p"]].append(row)
        out = []
        for p, rows in sorted(bins.items()):
            if arm == "control":
                candidates = []
                lookup = {}
                for row in rows:
                    for index, origin in enumerate(row.get("occupation_origins", [])):
                        sy = tuple(origin["symbols"])
                        edges = tuple(tuple(e) for e in origin["edges"])
                        pid = f"{row['minimum_id']}:route{index}"
                        occ = occupation_from_record(
                            corrected_occupation(origin["occupation"])
                        )
                        candidates.append(
                            ParentStructure(
                                k,
                                p,
                                pid,
                                sy,
                                np.asarray(origin["positions"]),
                                origin["energy_eV"],
                                edges,
                                tuple(
                                    e
                                    for e in edges
                                    if {sy[e[0]], sy[e[1]]} == {"Cd", "Se"}
                                ),
                                zb_occupation=occ,
                                propagation_eligible=origin["connectivity_preserved"],
                                topology_status="preserved"
                                if origin["connectivity_preserved"]
                                else "changed",
                            )
                        )
                        lookup[pid] = dict(
                            row,
                            **{
                                n: origin[n]
                                for n in ["symbols", "positions", "edges", "energy_eV"]
                            },
                            occupations=[occupation_to_record(occ)],
                        )
                for parent in (
                    _select_zb_occupation_parents(
                        candidates, self.growth.window_for(k), self.spec
                    )
                    if candidates
                    else []
                ):
                    out.append(lookup[parent.structure_id])
            else:
                out.extend(
                    select_population(
                        rows,
                        self.rng("select", arm, k, p),
                        self.config.primary_slots,
                        self.config.audit_slots,
                    )
                )
        return out

    def lattice_proposals(self, parent, k, limit=8, control=False):
        from .molecular_zb_growth import snap_parent

        occupations = parent.get("occupations", [])
        if not occupations:
            occ, _ = snap_parent(
                parent["symbols"],
                np.asarray(parent["positions"]),
                self.spec,
                self.model,
                parent_id=parent["minimum_id"],
                k=parent["k"],
                p=parent["p"],
            )
            if occ:
                occupations = [occupation_to_record(occ)]
        jobs = []
        pending = {}
        base_parent = parent
        for record in occupations[:2]:
            route_parent = base_parent
            record_id = corrected_occupation(record)["occupation_id"]
            for origin in base_parent.get("occupation_origins", []):
                origin_id = corrected_occupation(origin["occupation"])["occupation_id"]
                if origin_id == record_id:
                    route_parent = dict(
                        base_parent,
                        **{
                            n: origin[n]
                            for n in ["symbols", "positions", "edges", "energy_eV"]
                        },
                    )
                    break
            occ = occupation_from_record(corrected_occupation(record))
            occ.parent_id = route_parent["minimum_id"]
            # Symbols can differ after geometric basin consolidation. Do not transfer an
            # atom mapping to another route; use the origin's graph for geometric proposals.
            units = ztype_shed_units(
                occ,
                self.spec,
                parent_symbols=route_parent["symbols"],
                parent_coordinates=np.asarray(route_parent["positions"]),
                parent_edges=route_parent["edges"],
                ligand_bond_length=2.9,
            )
            for shed in range(min(2, route_parent["p"]) + 1):
                for pm in self.growth.window_for(route_parent["k"]).monomer_p_values:
                    if not 1 <= route_parent["p"] - shed + pm <= self.config.p_max[k]:
                        continue
                    children = grow_zb_children(
                        occ,
                        s=shed,
                        p_m=pm,
                        spec=self.spec,
                        model=self.model,
                        cap=2,
                        ztype_units=units[:16],
                    )
                    for child in children:
                        pending.setdefault(child.occupation_id, child)
        by_p = defaultdict(deque)
        for child in pending.values():
            by_p[child.p].append(child)
        chosen = []
        while any(by_p.values()) and len(chosen) < limit:
            for pp in sorted(by_p):
                if by_p[pp] and len(chosen) < limit:
                    chosen.append(by_p[pp].popleft())
        for child in chosen:
            if self.expired():
                return jobs
            rng = self.rng("lattice", base_parent["minimum_id"], child.occupation_id)
            if control:
                from .molecular_growth import (
                    _enum_init,
                    _enum_one,
                    _embed_init,
                    _embed_one,
                )

                _enum_init(self.pack, self.spec, 3)
                shells, _, _ = _enum_one(
                    (
                        k,
                        child.p,
                        list(child.core_edges),
                        tuple(child.symbols),
                        child.coordinates.tolist(),
                    )
                )
                proposals = []
                _embed_init(
                    self.pack,
                    self.spec,
                    dict(
                        starts=1,
                        keep=1,
                        max_nfev=40,
                        overlap_min_A=0.75,
                        start_max_bond_error_A=1.0,
                    ),
                )
                for _, isomer in shells[:1]:
                    state = graph_state(list(isomer.symbols), list(isomer.graph.edges))
                    starts = _embed_one((state, dict(enumerate(child.coordinates))))
                    for _, coords in starts[:1]:
                        proposals.append(
                            Proposal(
                                k,
                                child.p,
                                list(isomer.symbols),
                                np.asarray(coords).tolist(),
                                sorted([list(sorted(e)) for e in isomer.graph.edges]),
                                [base_parent["minimum_id"]],
                                "lattice",
                                occupation_to_record(child),
                            )
                        )
                stats = dict(control_shells=len(shells))
            else:
                proposals, stats = embed_core(
                    k,
                    child.p,
                    list(child.symbols),
                    child.core_edges,
                    child.coordinates,
                    rng,
                    self.spec,
                    self.pack,
                    [base_parent["minimum_id"]],
                    "fresh",
                    occupation_to_record(child),
                    prior=self.config.bridge_prior,
                    limit=1,
                )
            self.event(
                dict(
                    event="enumeration",
                    arm="control" if control else "experimental",
                    stats=stats,
                )
            )
            jobs.extend(proposals)
        return jobs

    def proposals(self, arm, k, round_number):
        jobs = []
        if arm == "control":
            if round_number:
                return []
            for parent in self.parents(arm, k - 1):
                if self.expired():
                    break
                jobs.extend(self.lattice_proposals(parent, k, control=True))
        else:
            for parent in (
                self.parents(arm, k - 1) if round_number == 0 else self.parents(arm, k)
            ):
                if self.expired():
                    break
                local = []
                rng = self.rng("propose", arm, k, round_number, parent["minimum_id"])
                channels = (
                    ["growth"] if round_number == 0 else ["exchange", "reconstruction"]
                )
                for channel in channels:
                    local.extend(
                        local_proposals(
                            parent,
                            rng,
                            self.spec,
                            self.pack,
                            channel=channel,
                            limit=self.config.proposals_per_move,
                        )
                    )
                if round_number == 0 and k > 3:
                    local.extend(self.lattice_proposals(parent, k))
                if parent.get("role") == "audit" or parent.get("audit_derived", False):
                    for proposal in local:
                        proposal.audit_derived = True
                jobs.extend(local[: self.config.proposals_per_parent])
            if round_number == 0 and k <= 3:
                for p in range(1, self.config.p_max[k] + 1):
                    rng = self.rng("fresh", k, p)
                    for sy, edges in fresh_cores(k, p, rng, limit=4):
                        if self.expired():
                            break
                        new, stats = embed_core(
                            k,
                            p,
                            sy,
                            edges,
                            None,
                            rng,
                            self.spec,
                            self.pack,
                            [],
                            "fresh",
                            prior=self.config.bridge_prior,
                            limit=2,
                        )
                        self.event(dict(event="enumeration", arm=arm, stats=stats))
                        jobs.extend(new)
        out = {}
        for p in jobs:
            if p.k != k or not 1 <= p.p <= self.config.p_max[k]:
                continue
            if p.id in out:
                out[p.id].parents = sorted(set(out[p.id].parents + p.parents))
            else:
                out[p.id] = p
        pools = defaultdict(list)
        for p in out.values():
            pools[p.channel].append(p)
        pools = {
            c: deque(round_robin(sorted(ps, key=lambda p: p.id)))
            for c, ps in pools.items()
        }
        ordered = []
        counts = Counter()
        while any(pools.values()):
            available = [c for c in pools if pools[c]]
            c = min(
                available,
                key=lambda c: (counts[c] / max(self.config.shares.get(c, 1), 1e-9), c),
            )
            ordered.append(pools[c].popleft())
            counts[c] += 1
        return ordered

    def recover(self):
        """Recover charged interrupted calls and recorded max-cycle retries."""

        # Recover launched jobs once, as explicit charged retries.
        for ev in list(self.events):
            if (
                ev["event"] == "launch"
                and (ev["arm"], ev["id"], ev["attempt"]) not in self.completed
                and ev["attempt"] == 0
            ):
                proposal = Proposal(**ev["proposal"])
                proposal.attempt = 1
                self.evaluate([(ev["arm"], proposal)], ev["stage"])
        # A crash after recording a max-cycle endpoint must not lose its retry.
        launches = {
            (e["arm"], e["id"]): e
            for e in self.events
            if e["event"] == "launch" and e["attempt"] == 0
        }
        for ev in list(self.events):
            if (
                ev["event"] == "result"
                and ev["attempt"] == 0
                and not ev["converged"]
                and ev.get("positions")
                and (ev["arm"], ev["id"], 1) not in self.reserved
            ):
                launch = launches[ev["arm"], ev["id"]]
                proposal = Proposal(**launch["proposal"])
                self.evaluate(
                    [
                        (
                            ev["arm"],
                            replace(
                                proposal,
                                positions=ev["positions"],
                                attempt=1,
                                origin_id=ev["id"],
                            ),
                        )
                    ],
                    ev["stage"],
                )

    def run(self):
        self.setup()
        self.recover()
        if ("shared", 1, 0) not in self.stage_done:
            self.evaluate(
                [("shared", p) for p in self.prepare_queue("shared", 1, 0, self.seeds)],
                1,
            )
            if not self.expired():
                self.mark_stage("shared", 1, 0)
        for k in range(2, self.config.k_max + 1):
            for round_number in range(self.config.fixed_rounds + 1):
                if self.expired():
                    break
                queues = {
                    arm: deque(
                        self.prepare_queue(
                            arm,
                            k,
                            round_number,
                            lambda arm=arm: self.proposals(arm, k, round_number),
                        )
                    )
                    if (arm, k, round_number) not in self.stage_done
                    else deque()
                    for arm in ["control", "experimental"]
                }
                # Reserve 35% of experimental calls for fixed-k rounds.
                if round_number == 0:
                    limit = int(self.config.stage_limits()[k] * 0.65)
                    queues["experimental"] = deque(list(queues["experimental"])[:limit])
                while any(queues.values()) and not self.expired():
                    batch = []
                    for _ in range(self.config.workers):
                        available = [
                            a for a, q in queues.items() if q and self.allowed(a, k)
                        ]
                        if not available:
                            break
                        a = min(
                            available,
                            key=lambda a: (
                                self.worker_seconds[a] + sum(x == a for x, _ in batch),
                                a,
                            ),
                        )
                        batch.append((a, queues[a].popleft()))
                    if not batch:
                        break
                    self.evaluate(batch, k)
                for arm in queues:
                    if not self.expired():
                        self.mark_stage(arm, k, round_number)
            if self.expired():
                break
        self.checkpoint()
        self.report()

    def mark_stage(self, arm, k, round_number):
        self.event(dict(event="stage_done", arm=arm, stage=k, round=round_number))
        self.stage_done.add((arm, k, round_number))

    def report(self):
        bins = defaultdict(dict)
        for arm, rows in self.rows.items():
            grouped = defaultdict(list)
            for row in rows.values():
                grouped[row["k"], row["p"]].append(row)
            for (k, p), group in sorted(grouped.items()):
                primary = [r for r in group if r["role"] == "primary"]
                bins[f"k{k}_p{p}"][arm] = dict(
                    primary_minima=len(primary),
                    audit_minima=len(group) - len(primary),
                    best_energy_eV=min((r["energy_eV"] for r in primary), default=None),
                    minima=[r["minimum_id"] for r in group],
                )
        self.atomic_json(self.output / "composition_comparison.json", bins)
        counts, seconds, seen = Counter(), Counter(), defaultdict(set)
        curves, outcomes = [], Counter()
        for event in self.events:
            arm = event.get("arm")
            if event["event"] == "launch":
                counts[arm] += 1
            elif event["event"] == "result":
                seconds[arm] += event["seconds"]
                row = event.get("row")
                outcomes[arm, row["role"] if row else "rejected_or_failed"] += 1
                if row and row["role"] == "primary":
                    seen[arm].add(row.get("minimum_id", row["structure_id"]))
                curves.append(
                    dict(
                        arm=arm,
                        calls=counts[arm],
                        worker_seconds=seconds[arm],
                        primary_minima=len(seen[arm]),
                    )
                )
        self.atomic_json(self.output / "discovery_curves.json", curves)
        self.atomic_json(
            self.output / "relaxation_outcomes.json",
            [
                dict(arm=arm, outcome=outcome, count=n)
                for (arm, outcome), n in sorted(outcomes.items())
            ],
        )
        lines = [
            "# Bounded nucleation pilot",
            "",
            "These are proposed transformations, not verified kinetic pathways.",
            "",
            "| Arm | Calls | Worker hours | Primary minima | Audit minima |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm, rows in self.rows.items():
            lines.append(
                f"| {arm} | {sum(n for (a,k),n in self.calls.items() if a==arm)} | {self.worker_seconds[arm]/3600:.2f} | "
                f"{sum(r['role']=='primary' for r in rows.values())} | {sum(r['role']=='audit' for r in rows.values())} |"
            )
        lines += [
            "",
            "Shared seed calls are charged separately. Compare curves at matched calls and worker time.",
            "Budget exhaustion, missing parents and unfinished reservations are reported in status.json.",
            "All proposals and relaxation mappings are in events.jsonl; minima.json retains origin routes.",
            "composition_comparison.json compares energies only within identical (k,p) bins.",
            "discovery_curves.json uses journal order (batched launches), not physical elapsed kinetics; shared seeds are a separate baseline.",
            "Channel shares are proposal scheduling targets, not guaranteed realized call fractions: retries, infeasible moves and empty bins change them.",
            "Neither electronic energies across compositions nor proposal frequencies establish nucleation free energies or rates.",
        ]
        (self.output / "assessment.md").write_text("\n".join(lines) + "\n")
