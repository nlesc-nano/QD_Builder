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

Its defaults follow the existing `/scratch/iinfante` layout. Override a path without editing the script with `sbatch --export=ALL,QD_ROOT=/path/to/QD_Builder,QD_SEEDS=/path/to/seeds,QD_OUT=/path/to/output scripts/nucleation_pilot_24c.slurm`. Resubmission with the same code, inputs and output resumes; a per-output `flock` prevents two jobs from writing concurrently.

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
