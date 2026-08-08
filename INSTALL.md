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

## Production run pack (graph_rules)

Every knob below is read from `graph_rules` in the run pack. A key absent from
`nucleation_graph_rules_mapping()` in `geometry_pack.py` silently falls back to
its dataclass default, so check a new key round-trips before trusting it:

```python
from builder.nucleation.spec import load_nucleation_spec
print(load_nucleation_spec("pack.yaml").graph_rules)
```

```yaml
graph_rules:
  # --- combinatorics ------------------------------------------------------
  bridge_first_max_automorphisms: 64   # cap on core |Aut| before the beam
                                       # keys states by identity.  |Aut| grows
                                       # factorially in equivalent precursor
                                       # cations (mean 8388 at k4p9); 64 takes
                                       # that bin 1541 s -> 4.7 s.  Costs ~7%
                                       # of graphs at k4p7; 2048 is lossless
                                       # there but only 12x on k4p9.
  bridge_first_prefer_bridges_per_cd: 2  # soft bridge load.  THIS bounds load,
                                       # not the hard cap below.  3 cuts mean
                                       # terminal Cl 2.99 -> 1.82 at k3p4.
  bridge_first_hard_max_bridges_per_cd: 2  # inert: never reached in practice

  # --- selection ----------------------------------------------------------
  selection_order: compactness         # or bond_bands (historical default)
  selection_max_wiener_excess: 0.10    # PREFERRED: relative to the bin's own
                                       # most compact graph
  # selection_top_fraction: 0.25       # fixed rank cut -- see warning below

  # --- optional gates -----------------------------------------------------
  required_rings:                      # rings a graph MUST contain
    - {size: 8, min_count: 6, from_k: 4}
  bridge_first_maximize_bridged_pairs: false   # leave off; see note below
```

### Selection

`selection_order: compactness` ranks each bin's pool by Wiener index (sum of
shortest-path lengths, computed on the graph before any coordinates) and keeps
the most compact `selection_top_fraction`. Compactness tracks relative energy
at rho +0.32 to +0.78 across measured bins; keeping the top 70% retains ~89%
of each bin's best decile, top 50% retains ~80%.

### Use `selection_max_wiener_excess`, not `selection_top_fraction`

A fixed rank cut prunes the same fraction everywhere, which is wrong: the
Wiener spread within a bin ranges from 10% (k=2, where the descriptor carries
almost no information) to 39% (k3p3, where it discriminates). Worse, the bin's
lowest-*energy* structure is often not compact -- it sits at the 95th
compactness percentile in one bin and the 69th in another.

Measured over nine bins with known energies:

| cut | bins losing the bin energy minimum |
|---|---|
| top 20% by rank | **4 / 9** |
| top 30% | 3 / 9 |
| top 50% | 3 / 9 |
| top 70% | 2 / 9 |
| excess <= 0.05 | 3 / 9 |
| **excess <= 0.10** | **1 / 9** |
| excess <= 0.20 | 0 / 9 |

The relative cut adapts: at 0.10 it keeps 91-94% of the k=2 bins (nothing to
gain there) and 41-53% of the k=3 bins (where it pays). Verified end to end:
k3p3 186 -> 76 and k3p4 205 -> 108 at 0.10; 186 -> 17 at 0.05; 186 -> 163 at
0.20.

The one bin that still loses its minimum at 0.10 has its energy minimum at
excess 0.154 -- genuinely among the least compact graphs in that bin. No
compactness cut can retain it, so treat ~1-in-9 as the accuracy budget.

### Gates that did not work

- `bridge_first_maximize_bridged_pairs` — correlates with compactness post hoc
  (rho +0.58) but is a poor objective: a mu3 whose hosts already share an anion
  scores zero, so enabling it strips the mu3 family holding the k3p3 energy
  minimum. Default off.
- `required_rings` with `size: 8` — 8-rings correlate with energy (rho -0.19 to
  -0.66 over nine bins) but are 0.56-0.92 collinear with the Wiener index, and
  in a stratified check the residual signal flips sign in about half the
  strata. Prefer `selection_order: compactness`; ring counting is also the more
  expensive of the two. Kept as a hard chemical floor if you want one, but note
  `min_count: 1` prunes almost nothing (95-100% of k=3 graphs already have an
  8-ring) and a fitted `min_count(k, p)` does not transfer (best normalisation
  still has cv 0.61).

### Measured bin sizes (exhaustive, cap 64, graph-only)

| k | graphs | note |
|---|--------|------|
| 1 | 15     | p <= 3 |
| 2 | 139    | p <= 5 |
| 3 | 828    | p <= 7 |
| 4 | 4097   | p <= 9, largest bin 904 |
| 5 | 19799+ | p1-p5 only; projected ~45k over p <= 10 |

At `selection_top_fraction: 0.2` that is ~1000 xTB runs for k <= 4 (about 6
core-hours at the measured ~21 s/structure) and ~10000 for k <= 5.
