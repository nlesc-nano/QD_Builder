# Molecular CdSe/CdCl₂ DFT preparation

Turn a lattice-free molecular map into a CP2K job tree.

## Input

Output of `write_molecular_map` / `tools/run_molecular_map.py`:

```text
molecular_map/
  k001/p001/k001_p001_mol0001.xyz
  k002/p004/k002_p004_mol0007.xyz
  annotations.csv
  index.csv
```

## Prepare

```bash
python tools/molecular_dft/prepare_molecular_dft.py \
  --source runs/molecular_cdse_k2 \
  --output runs/molecular_cdse_k2_dft \
  --max-k 2
```

## Output

```text
dft_out/
  manifest.tsv
  box_sizes.tsv
  annotations.csv
  cp2k_one.slurm      # one structure = one job
  submit_jobs.sh
  k002/p004/k002_p004_mol0007/
    start.xyz
    cp2k_job.in
```

All isomers in a `(k,p)` bin share one cubic box (max span in the bin + padding 12 Å).

## On the HPC — independent jobs (recommended)

From the DFT root (200 structures → **200 separate `sbatch` jobs**):

```bash
cd /path/to/dft_tree
chmod +x submit_jobs.sh
# ensure cp2k_one.slurm is present (copied by prepare, or copy from tools/molecular_dft/)
./submit_jobs.sh manifest.tsv
```

- Manifest order: **k001/p001 → … → end** (jobs are submitted in that order).
- Each job `chdir`s into its isomer folder and runs only that `cp2k_job.in`.
- The scheduler can pack free cores with as many of these jobs as the queue allows.

Dry run (print only):

```bash
DRY_RUN=1 ./submit_jobs.sh manifest.tsv
```

Skip already finished optimizations (default):

```bash
SKIP_DONE=1 ./submit_jobs.sh manifest.tsv   # default
SKIP_DONE=0 ./submit_jobs.sh manifest.tsv   # resubmit everything
```

## After DFT — light transfer tree (start + relaxed only)

Heavy DFT folders contain trajectories, restarts, inputs, logs — hard to
transfer. Build a **new** tree with the same `k###/p###/id` layout but **only**
`start.xyz` + `final.xyz` per isomer:

```bash
# on the HPC, next to the heavy tree
python /path/to/QD_builder/tools/molecular_dft/extract_start_final.py \
  --root /path/to/heavy_dft \
  --out  /path/to/dft_light

# only finished jobs
python .../extract_start_final.py --root . --out ../dft_light --only-done
```

Light tree:

```text
dft_light/
  inventory.csv
  k001/p001/<structure_id>/
    start.xyz    # construction input
    final.xyz    # last complete frame of CP2K *-pos-*.xyz
  k002/p003/<structure_id>/
    start.xyz
    final.xyz
```

Nothing else is copied (no `cp2k_job.*`, no trajectories, no restarts).

Then transfer only the light folder:

```bash
rsync -av dft_light/  laptop:/path/to/dft_light/
# or:  tar czf dft_light.tgz dft_light && scp dft_light.tgz ...
```

Optional: also write `final.xyz` inside the heavy tree (`--write-in-place`).

Then graph comparison (when ready):

```bash
python tools/compare_molecular_start_final.py \
  --annotations annotations.csv \
  --start-dir dft_light \
  --final-root dft_light \
  --output start_final_report.csv
```
