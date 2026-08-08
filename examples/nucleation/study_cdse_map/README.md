# CdSe/CdCl2 nucleation-map study

These inputs preserve the DFT-derived Cd/Se/Cl coordination, geometry, bridge,
and ring rules. Every generated `(k,p)` composition is neutral:

`Cd_(k+p) Se_k Cl_(2p) = k CdSe + p CdCl2`.

Run from the repository root, in this order:

```bash
nc-builder examples/nucleation/study_cdse_map/01_exact_k2_reference.yaml \
  -o runs/cdse_map/exact_k2

nc-builder examples/nucleation/study_cdse_map/02_exact_k3_calibration.yaml \
  -o runs/cdse_map/exact_k3

nc-builder examples/nucleation/study_cdse_map/03_guided_k5_equilibrium_map.yaml \
  -o runs/cdse_map/guided_k5
```

If a run is interrupted, repeat the same command with `--restart`. Do not use
`--force-restart` after changing a recipe; the checkpoint fingerprint is there
to prevent mixing different search definitions.

## Purpose of each run

- `01_exact_k2_reference.yaml`: cheap correctness/reference set; label this
  first and use it to assess whether the graph ranking recalls DFT-low states.
- `02_exact_k3_calibration.yaml`: expensive small-nucleus calibration. It keeps
  open motifs through k=3 and is the most important check before trusting the
  guided extension.
- `03_guided_k5_equilibrium_map.yaml`: production map. Its neutral p ladder has
  no coverage cutoff and stops only when growth closes or reaches the hard
  coordination capacity. It starts compact-ring growth at k=5,
  and bounds cost by narrowing after the exact k=2 row, a per-bin skeleton
  beam of 16 from k=3, and up to three guided passivation shells per core. Shell
  subset spaces above 1000 assignments use the DFT-derived greedy ordering.
  Use the exact k=3 run to quantify what this narrowing misses.

No `p_surf` or precursor-concentration window is used in these equilibrium
inputs. Coverage is selected later with chemical potentials. A separate
continuous-decoration run should be interpreted as a kinetic/path-memory
model, not as the equilibrium map.

## Broad k=6 calibration and DFT selection

The two broad calibration inputs are:

```bash
nc-builder \
  examples/nucleation/study_cdse_map/04_broad_k6_equilibrium_calibration.yaml \
  -o runs/cdse_map/broad_k6_equilibrium

nc-builder \
  examples/nucleation/study_cdse_map/05_pathway_k6_redecorated_calibration.yaml \
  -o runs/cdse_map/pathway_k6_redecorated
```

After both runs have completed through k=6, merge and select a 240-structure
DFT calibration set with:

```bash
python examples/nucleation/study_cdse_map/cp2k/select_dft_candidates.py \
  --source runs/cdse_map/broad_k6_equilibrium \
  --source runs/cdse_map/pathway_k6_redecorated \
  --exclude-manifest runs/cdse_map/dft_all/manifest.tsv \
  --output runs/cdse_map/dft_k6_calibration \
  --budget 240
```

If the pathway calculation is still running, its `k001`--`k005` checkpoint rows
can be selected without touching the active k=6 row:

```bash
python examples/nucleation/study_cdse_map/cp2k/select_dft_candidates.py \
  --source runs/cdse_map/broad_k6_equilibrium \
  --source runs/cdse_map/pathway_k6_redecorated \
  --allow-incomplete-source \
  --require-kmax 0 \
  --k-max 5 \
  --exclude-manifest runs/cdse_map/dft_all/manifest.tsv \
  --output runs/cdse_map/dft_k5_partial \
  --budget 220
```

`--allow-incomplete-source` reads only checkpoint directories containing
`DONE`; it never reads the in-progress k=6 row. Use a separate output directory
for this partial selection. After k=6 finishes, run the full command above with
a new output directory and `--require-kmax 6`.

Omit `--exclude-manifest` if the earlier DFT tree is unavailable or should not
be excluded. The default selection spans k=3-6 and targets 80% retained
structures and 20% soft-rejected controls. A rejected control must be neutral,
minimum-CN compliant, and rejected only by a lower coordination or Cl-ring
rank. The selector reconstructs an XYZ from `registry.json` when the generator
recorded a discarded structure but did not write its XYZ file.

The selector is deterministic and writes:

- `manifest.tsv`: the seven-column manifest consumed by `cp2k.slurm` and
  `submit_jobs.sh`;
- `selection.tsv`: score layer, structural family, ligand-shell hash, rings,
  bridges, source channel, lineage, rejection reason, and deduplication data;
- `selection_summary.json`: settings and counts before/after exclusion,
  merging, and sampling;
- `box_sizes.tsv`: the common cubic box used for every selected isomer in a
  `(k,p)` bin;
- one `start.xyz` and rendered `cp2k_job.in` inside every calculation folder.

Element-resolved pair-distance fingerprints remove duplicate DFT starting
geometries across the two bundles and the optional earlier manifest. Selection
is stratified across `(k,p)` bins, coordination-score layers, skeleton families,
ligand shells, bridge/ring counts, and source channels. Half of each rejected
bin quota stays near the retained score boundary; the other half is chosen for
descriptor diversity.

Submit one structure per array task from the directory containing the local
`cp2k.slurm` and `submit_jobs.sh` files:

```bash
bash submit_jobs.sh runs/cdse_map/dft_k6_calibration/manifest.tsv 1
```

## Structures to label

Start DFT/MLFF relaxations from retained `*_surface.xyz` files. Label every
retained isomer in the exact k=2 set. For k=3 and the guided run, label all
retained structures when a bin has at most eight; otherwise select structures
spanning distinct skeleton-family IDs, ring counts, bridge counts, and the top
three coordination-score layers. Include a small sample of discarded exact
structures to measure false negatives from the rule-based ranking.

Keep the original structure ID in the calculation directory and record the
relaxed connectivity. If several starting structures relax to the same final
minimum, treat them as one basin rather than independent isomers.

## CP2K preparation and submission

The files under `cp2k/` are:

- `template.in`: the common CP2K input; preparation replaces only the XYZ name
  and cubic box side.
- `prepare_cp2k_runs.py`: collects retained `*_surface.xyz` structures, removes
  exact coordinate duplicates across bundles, assigns one common box per
  `(k,p)`, and creates one isolated calculation directory per isomer.
- `cp2k.slurm`: executes a user-sized batch inside each SLURM array task.
- `submit_cp2k_batches.sh`: calculates the required array range and submits it.

After the nucleation bundles have been generated, prepare all unique retained
structures from the repository root:

```bash
python examples/nucleation/study_cdse_map/cp2k/prepare_cp2k_runs.py \
  --source runs/cdse_map/exact_k2 \
  --source runs/cdse_map/exact_k3 \
  --source runs/cdse_map/guided_k5 \
  --output runs/cdse_map/dft_all \
  --padding 12 \
  --min-box 20
```

For every bin, the script finds the largest of the x/y/z coordinate spans over
all its isomers, adds 12 Å, rounds upward to a whole Å, and enforces a minimum
20 Å box. The result is recorded in `box_sizes.tsv`. Each calculation is placed
under:

```text
runs/cdse_map/dft_all/kXXX/pXXX/STRUCTURE_ID/
  start.xyz
  cp2k_job.in
```

Submit one isomer per array task. The complete array enters the queue and the
HPC scheduler applies its configured running-job limit:

```bash
bash examples/nucleation/study_cdse_map/cp2k/submit_cp2k_batches.sh \
  runs/cdse_map/dft_all/manifest.tsv 1
```

The second argument is `BATCH_SIZE`. The SLURM template uses a 24-hour wall
time. A batch size of one gives the cleanest fault isolation; larger batches run
isomers sequentially within the same 24-hour allocation. For example, four
isomers per allocation, with concurrency left to the HPC scheduler:

```bash
bash examples/nucleation/study_cdse_map/cp2k/submit_cp2k_batches.sh \
  runs/cdse_map/dft_all/manifest.tsv 4
```

An optional third argument applies a local array throttle if desired; for
example, `4 6` limits execution to six array tasks concurrently. Omit it to put
the entire array into the queue and let the HPC enforce its own limits.

The equivalent direct submission is:

```bash
MANIFEST=runs/cdse_map/dft_all/manifest.tsv
N=$(( $(wc -l < "$MANIFEST") - 1 ))
sbatch --array=0-$((N-1)) \
  --export=ALL,MANIFEST=runs/cdse_map/dft_all/manifest.tsv,BATCH_SIZE=1 \
  examples/nucleation/study_cdse_map/cp2k/cp2k.slurm
```

Use the wrapper unless the array range is known: it derives the correct range
from the manifest. Completed calculations containing CP2K's `PROGRAM ENDED AT`
marker are skipped on resubmission. `.done`, `.failed`, and
`.scf_not_converged` files provide quick status checks. Because the input keeps
`IGNORE_CONVERGENCE_FAILURE`, do not use a `.scf_not_converged` calculation as
an energy label without inspecting and rerunning it.
