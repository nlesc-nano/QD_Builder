# Installation

## Quick start (micromamba / mamba / conda)

```bash
micromamba create -f environment.yml -y
micromamba activate qd-builder
pip install -e .
```

`mamba` or `conda` work identically — substitute the command name. Verify:

```bash
python -c "import builder.nucleation.molecular as m; print('ok')"
python -m pytest tests -q -k "molecular or nucleation"
```

## What is in the environment

| group | packages | needed for |
|---|---|---|
| core | `numpy`, `scipy`, `networkx`, `pyyaml` | graph enumeration, motif reconstruction, packs |
| crystal | `pymatgen`, `rdkit` | Wulff construction and surface passivation paths |
| relaxation | `xtb`, `xtb-python`, `ase` | GFN1/GFN2 relaxation of embedded structures |
| analysis | `pandas`, `matplotlib-base` | `tools/` and `scripts/` post-processing |
| test | `pytest` | test suite |

The core group alone is enough to generate graphs with `--no-embed`.

## xTB in a separate environment

`xtb-python` and ASE can live in their own environment; `xtb_relax.py` then
shells out to it once per batch. Point the run pack at that interpreter:

```yaml
relaxation:
  enabled: true
  method: GFN1-xTB
  python: /path/to/envs/xtb/bin/python
```

Omit `python:` when xtb and ase are in the active environment, and the
relaxation runs in-process.

## HPC notes

**Per-(k,p) parallelism.** Each bin is independent — no shared state, no
cross-bin ordering. One bin per node:

```bash
python tools/run_molecular_map.py \
  --yaml  geometry_packs/cdse_cdcl2_motif_v2_xtb.yaml \
  --kmin 3 --kmax 3 --pmin "$P" --pmax "$P" \
  --output "runs/k3p${P}"
```

Outputs land in separate directories and can be merged afterwards. Note the
top-level `index.csv` in a multi-bin run is rewritten per bin, so when
merging, read the per-bin `<k>/<p>/motif_trials.csv` files rather than the
top-level CSV.

**Graph-only first.** `--no-embed` runs the enumerator without any 3D and
takes seconds per bin. Use it to size a bin before committing nodes to it.

**Threads.** The enumerator is single-threaded and the cost is xTB, so
allocate cores to the relaxation rather than to the map. Set
`OMP_NUM_THREADS` / `MKL_NUM_THREADS` to match your `--cpus-per-task`.

**Scratch.** A k=3 bin writes ~500 files (initial + relaxed XYZ per trial).
Write to node-local scratch and copy back, rather than straight to a shared
filesystem.

## Reproducibility

`python` is pinned to 3.11 in `environment.yml`; everything else is a lower
bound. For a fixed cluster deployment, snapshot the solve after creating the
environment:

```bash
micromamba env export --explicit > environment.lock.txt
```

## Known issues

- `tests/test_molecular_baseline.py` has two failing cases
  (`test_accepted_isomers_match_baseline`, `test_embedded_coordinates_match_baseline`).
  These compare against `tests/molecular_map_baseline.json`, which predates
  recent geometry-pack changes; the failures are pre-existing and not caused
  by the enumeration path.
- `xtb-python` from PyPI builds from source and needs a Fortran toolchain.
  Install it from conda-forge instead.
