# Bounded nucleation pilot

This is an opt-in search experiment, not a kinetic nucleation simulation. It compares corrected lattice/agnostic growth with a mixed graph, growth, ligand-exchange and reconstruction search. Existing large clusters are benchmarks, not seeds. Only k=1–2 seed structures are imported.

## Analyze existing runs

From the repository root, using an environment with the project's dependencies:

```bash
PYTHONPATH=src python tools/analyze_growth_corpus.py \
  /Users/ivaninfante/Documents/University/Programs/Nucleation/graphs/growth_zb \
  --output runs/nucleation_corpus_audit
```

The source corpus is read-only. Geometry consolidation can be slow for large runs. `--graphs-only` is a cheaper preliminary audit, but does not establish distinct relaxed minima. The outputs include per-atom-type coordination summaries, exact μ2/μ3 counts, bridge changes on relaxation, within-composition/family associations, held-out-run validation, bond-cutoff sensitivity and legacy identity aliases. Unknown legacy protocols remain segregated: absence of recorded settings is not evidence of equivalent calculations. Missing convergence is not treated as success.

The empirical μ2≈2p relationship is a search prior, not a hard chemical rule. The current pilot samples near the largest bridge counts found under its bounded enumeration and keeps exploratory shells. Exhaustion of a bounded enumeration does not prove a tier chemically infeasible. Terminal and μ3 alternatives remain important counterexamples. The analysis does not automatically train or modify the pilot configuration.

## Preview and run

```bash
PYTHONPATH=src python tools/run_nucleation_pilot.py \
  --seeds /Users/ivaninfante/Documents/University/Programs/Nucleation/graphs/runs/gxtb_cdse_target_k1k2_p1p5 \
  --output runs/nucleation_pilot_v1 --dry-run
```

Remove `--dry-run` to launch calculations. Configure the executable and parameter environment as for the existing geometry pack. The dry run prints budgets; it does not test the executable or generate proposals.

On the current HPC, the ready-to-submit launcher is `scripts/nucleation_pilot_24c.slurm`:

```bash
sbatch scripts/nucleation_pilot_24c.slurm
```

Its defaults follow the existing `/scratch/iinfante` layout. Override a path without editing the script with `sbatch --export=ALL,QD_ROOT=/path/to/QD_Builder,QD_SEEDS=/path/to/seeds,QD_OUT=/path/to/output scripts/nucleation_pilot_24c.slurm`. Resubmission with the same code, inputs and output resumes; a per-output `flock` prevents two jobs from writing concurrently. SLURM writes `nucleation_pilot_JOBID.out` and `.err` in the directory from which `sbatch` is called. Output is intentionally limited to launcher/preflight information and the existing backend diagnostics; detailed state is recorded in `status.json` and `events.jsonl`.

Defaults in `geometry_packs/cdse_cdcl2_zb/pilot.yaml`:

- At most 4,000 backend calls, including failures and one permitted retry: 128 shared seed calls and 1,936 calls per arm.
- Per-arm limits for k=2…6: 194, 290, 387, 484 and 581.
- At most 24 concurrent one-thread calculations, 150 optimization cycles and 1,800 seconds per call. Stop launching after 46 active hours. Use an external 48-hour scheduler limit for a strict wall-clock allocation, since proposal generation and subprocess cleanup also take time.
- Bounded proposals and shell enumeration; 12 primary and two audit parent slots per bin. Audit descendants cannot exceed 10% of calls at each arm/stage. This conservative running quota can discard early audit proposals.
- Growth/exchange/reconstruction/fresh proposal targets 50/20/15/15%; two fixed-k rounds. Realized fractions depend on feasibility, parents, retries and budgets.

For an HPC scheduler, request one task with 24 CPUs, set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1`, activate the tested project environment, then run the command above. Do not run multiple pilot processes against the same output directory.

## Resume and interpretation

Repeat exactly the same command and output path to resume. Protocol fingerprints include configuration, nucleation source files, executable, seed XYZ contents and seed index. Changed fingerprints require a new output directory. Keep the journal: reservations are charged before launch, including interrupted jobs; retry calls are charged again. Proposal queues are frozen before evaluation. A damaged/truncated journal fails closed and requires manual recovery, not automatic deletion. Paused time is excluded from the active-time allowance; an external allocation deadline is separate.

`status.json` exposes actual counts and missing stages. `events.jsonl` retains proposals, parents, endpoint audits, failures and retries. `minima.json` preserves multiple construction-to-minimum mappings and origin geometries. `composition_comparison.json` compares primary minima only at fixed (k,p); audit minima remain separate. `discovery_curves.json` gives journal-ordered discovery counts against calls and worker seconds, with shared seeds reported separately. Batched result ordering is deterministic, not a physical trajectory.

Compare arms at matched calls and worker cost, including rejection rates and family coverage, rather than interpreting one low energy as decisive. Electronic energies across different compositions require reservoir chemical potentials and thermal/solvation corrections before thermodynamic interpretation. Transition frequencies do not establish rates or barriers. Repeat a promising pilot with independent random seeds and validate representative minima at a higher electronic-structure level before expanding the search.

The ancestry checker now distinguishes evaluated occupations, clean converged connectivity-preserving endpoints and selected parent catalogs. Compatible Se sub-backbones are a necessary condition for monotone lattice growth to the reference, not proof of a connected kinetic pathway or recoverable full Cd/Cl structure. Legacy IDs are retained as aliases in analysis; original result folders are not rewritten.

## Automated lineage protocol: A → B → C

The comparison pilot above remains frozen as an experiment. The follow-up runner is a separate adaptive protocol:

```bash
PYTHONPATH=src python tools/run_adaptive_nucleation.py \
  --source /path/to/nucleation_pilot_v2 \
  --output /path/to/nucleation_adaptive_abc_v1 \
  --dry-run
```

Remove `--dry-run` to execute, or submit `scripts/nucleation_adaptive_abc_24c.slurm`. Phase A repeatedly applies fixed-composition topology changes, ligand exchange, coordinate reconstruction and k→k+1 backfill over k=4–6. Coarse coordination/ring/geometry families are the bounded survival unit; exact core graphs remain subfamilies, with multiple geometries retained inside them. The initial cohort is frozen and a bounded admission reserve accepts genuinely new families without evicting earlier lineages. Phase B grows and refines k=7; Phase C does the same at k=8. B and C remain locked until every requested k independently shows two consecutive plateaus, the usable-endpoint fraction is adequate, and every retained family has either launched or exhausted its transformations in the required number of cycles. Exhausting an operation budget or the cycle limit is a halt, not false convergence.

The default caps are in `geometry_packs/cdse_cdcl2_zb/adaptive_abc.yaml`. Growth and fixed-k calls have independent reservations so one cannot consume the other's coverage budget. Exchange receives the largest fixed-k proposal allocation, topology remains exploratory, and reconstruction runs every third cycle. The run imports `minima.json` without charging old calculations, fingerprints that source, removes self-lineage loops, and journals every cohort admission, family plan, frozen proposal queue, backend reservation, result, cycle decision and phase gate. Resume uses the identical command and output. Summary state is written to `adaptive_assessment.json`; detailed state remains in `events.jsonl`, `minima.json` and `status.json`. Minima snapshots are written periodically and at cycle boundaries rather than after every 24-call batch; the append-only journal remains the durable recovery source.

### Targeted Phase-A continuation after `adaptive_abc_v1`

The v1 result exhausted its k=5 and k=6 budgets before demonstrating separate plateaus and exposed per-cycle cohort churn. Continue from that immutable archive with the revised controller:

```bash
PYTHONPATH=src python tools/run_adaptive_nucleation.py \
  --config geometry_packs/cdse_cdcl2_zb/adaptive_phase_a_continuation.yaml \
  --source /path/to/nucleation_adaptive_abc_v1 \
  --output /path/to/nucleation_phase_a_continuation_v1 \
  --dry-run
```

On the current HPC, submit `scripts/nucleation_phase_a_continuation_24c.slurm`. This run performs only fixed-k k=5 work plus separately budgeted k=5→6 growth and fixed-k k=6 work, with a 6,200-call hard ceiling. It intentionally stops after Phase A for assessment; it never launches k=7 or k=8. Use a new output directory because the source archive and protocol fingerprint differ from the original A→B→C run.

For a new system, the phase controller itself does not change. Set `chemistry_adapter` to a built-in name or `module:Class`. An adapter supplies bootstrap proposals, composition validation, endpoint classification, family identity/archive behavior, fixed-k graph moves and k-growth proposals; geometry and electronic-structure settings remain in the system pack. A completed minima archive can seed the protocol directly. A raw seed directory with `index.csv` is also supported when the adaptive config gives `seed_calls > 0`, a k=1 call cap, and a Phase A range beginning low enough to connect the seed sizes to the desired frontier. This separation automates the workflow, but it does not pretend that Cd/Se/Cl valences or reaction moves are transferable to unrelated chemistry: a chemically new model requires one tested adapter, after which no edits to the A/B/C scheduler are needed.

### Final Phase-A completion

After `nucleation_phase_a_continuation_v1`, run `scripts/nucleation_phase_a_completion_24c.slurm`. Its completion configuration imports the exact frozen k=5/k=6 cohorts from the source `events.jsonl` and executes representative cycles 5--7. Only cohort membership is inherited: calculation budgets, queues, convergence counters and phase status are new. This prevents cycles 0--4 from being repeated and prevents newly discovered families from changing the bounded completion cohort.

### Phase B: bounded k=6 to k=7 extension

Submit `scripts/nucleation_phase_b_k7_24c.slurm` with `nucleation_phase_a_completion_v1` as its source. Phase B uses a stable discrete CN/ring family identity; continuous radius and tetrahedral-order descriptors remain geometry diagnostics and no longer rename lineages at bin boundaries. Its 150-family k=6 parent beam reserves up to 20% for stable families first discovered in the source run, while the k=7 beam retains a 15% admission reserve. Plateau decisions use newly discovered families within 1 eV of the best same-composition minimum; raw family novelty is still reported. The run has independent 2,800-call growth and 3,000-call fixed-k caps and cannot proceed to k=8.
