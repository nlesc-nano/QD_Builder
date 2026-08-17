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

### Optional: Cython beam-search kernel

Decoration at high skeleton symmetry spends most of its wall time in the
bridge-first `state_key` orbit.  An optional native extension (~5× on that
path) is built when a C compiler is available:

```bash
# regenerates C from .pyx if Cython is installed, then compiles
pip install -e ".[speed]"
# or, with Cython already present / with a pre-generated _beam_key.c:
python setup.py build_ext --inplace
```

Confirm the backend after install:

```bash
python -c "from builder.nucleation.molecular_bridge_first import _BEAM_KEY_BACKEND as b; print(b)"
# cython  → native extension in use
# python  → pure-Python fallback (still correct, slower on high-|Aut| bins)
```

Without the extension the package imports and runs unchanged.

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

## Pack layout

A run pack is composed from one file per pipeline stage, listed by the driver:

```
geometry_packs/cdse_cdcl2/
  run_gxtb.yaml      driver: composition, k/p range, relaxation, include list
  graph_rules.yaml   graph stage  -- which (k,p) graphs exist (no coordinates)
  motifs.yaml        vocabulary   -- which local environments are allowed
  embed.yaml         3D stage     -- bonds/angles/dihedrals, reconstruction
```

Run it with `--yaml geometry_packs/cdse_cdcl2/run_gxtb.yaml`; the driver is
both the run spec and the pack, so no wrapper file is needed.

Two rules make the split trustworthy, and both are enforced at load:

- **A setting lives in exactly one file.** Defining the same leaf twice raises,
  naming both files. There is no precedence to remember, and include order is
  irrelevant.
- **An unknown `graph_rules` key raises**, with a spelling suggestion. This
  used to fail silently: `required_rings` and
  `bridge_first_maximize_bridged_pairs` were both "enabled" in a pack for days
  while the enumeration ran on the default, because nothing checked the name.
  The vocabulary is `NUCLEATION_GRAPH_RULE_KEYS` in `geometry_pack.py`, and
  `tests/test_geometry_pack_include.py` asserts every name in it is actually
  consumed.

To see what was parsed:

```python
from builder.nucleation.spec import load_nucleation_spec
print(load_nucleation_spec("geometry_packs/cdse_cdcl2/run_gxtb.yaml").graph_rules)
```

### Where a setting goes

| you want to change | file |
|---|---|
| allowed coordination, forbidden pairs, ring floor | `graph_rules.yaml` |
| how decorations are enumerated, `|Aut|` cap | `graph_rules.yaml` |
| which graphs reach 3D (compactness cut) | `graph_rules.yaml` |
| which local environments exist at all | `motifs.yaml` |
| a bond length, angle, improper, clash distance | `embed.yaml` |
| k/p range, CIF, charges | driver |
| xTB vs g-xTB, binary, timeout | driver |

`motifs.yaml` is read for `center` + `linker_count` only. A motif may carry
its own geometry, but **only** in a pack with no `bonds:` table — with
`embed.yaml` present those numbers are silently dead, which is how the old
single-file pack ended up carrying a full set of motif bond lengths that
contradicted the tables actually in use. Note also that adding a `Cd` motif
would pin Cd to exactly the listed coordination numbers, so census entries
like `Cd4:Se1Cl3` belong in the `embed.yaml` angle table, not there.

The legacy `geometry_reference:` mechanism still loads, but it overrides whole
top-level sections rather than merging leaves — so a driver defining
`graph_rules` silently discarded *all* of the reference's graph rules and
relaxation settings. Prefer `include:`.

### Legacy single-file packs

`geometry_packs/cdse_cdcl2_motif_GXTB.yaml` and friends still work and are
kept for reproducing earlier runs. `cdse_cdcl2/` is verified to produce
identical graph rules, geometry tables and bin results
(`test_composed_pack_matches_legacy_single_file`).

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

## g-xTB relaxation

g-xTB is more accurate than GFN1 but has **no Python API**, so it is driven as
a command-line binary (`gxtb in.xyz --gxtb --opt`) rather than through
xtb-python/ASE. Select it with `method: g-xTB` and the backend switches
automatically:

```yaml
relaxation:
  enabled: true
  method: g-xTB
  binary: gxtb          # name/path of the executable
  xtb_path: /scratch/.../xtb-gxtb-linux-x86_64/share/xtb   # or export XTBPATH
  charge: 0             # k CdSe + p CdCl2 is neutral by construction
  timeout_s: 1800
```

On a cluster where PATH and XTBPATH are already exported, `binary: gxtb` alone
is enough. Use `geometry_packs/cdse_cdcl2/run_gxtb.yaml`. For a local build:

```bash
export GXTB_PATH=/path/to/xtb-gxtb-<platform>
export PATH="$GXTB_PATH/bin:$PATH"
export XTBPATH="$GXTB_PATH/share/xtb"
```

**Startup banner.** Before the first structure the run prints the backend it
resolved — binary, absolute path, `--version`, `XTBPATH`, `PATH`,
`OMP_NUM_THREADS` and the exact command line — and warns if the binary is not
on `PATH`. A run whose energies look wrong is nearly always a wrong binary or
a stale `XTBPATH`, and neither is visible from the results:

```
[relax] binary    : gxtb
[relax] resolved  : /path/to/xtb-bleed-macos-x86_64/bin/gxtb
[relax] version   : xtb version 6.7.1 (30c6303) compiled by ... on 2026-05-14
[relax] XTBPATH   : /path/to/xtb-bleed-macos-x86_64/share/xtb
[relax] command   : gxtb in.xyz --gxtb --opt
[relax] timeout_s : 1800    charge: 0
```

**Job size before 3D.** Once decoration finishes, each bin reports how many
graphs cleared the graph rules — the real size of the job, known before a
single embedding is attempted:

```
    GRAPHS: 10 graphs passed the graph rules from 1 skeletons
            (10.0 graphs/skeleton); 10 decorations were streamed
```

Energy, geometry and Wiberg orders are read from `xtbopt.xyz`, `wbo` and the
log; energies are converted Hartree -> eV so they are directly comparable with
the GFN1 path.

**Timing** (measured): 5 atoms 1 s, 21 atoms 15 s, 29 atoms 47 s — roughly
2-3x GFN1. The GFN1 default `timeout_s: 90` kills most k=4 structures; use
1800.

**Energies are not comparable between methods** — g-xTB reports a different
reference (a 5-atom cluster is -99522 eV under g-xTB, -371.8 eV under GFN1).
Only compare within one method and one composition.

## Audit thresholds for relaxed structures

These are *audit* thresholds for judging a relaxed geometry. They are not the
`pair_rules` `min_distance` values, which are construction epsilons for the
embedding (79% of *embedded* structures sit below the Cd-Cd construction floor,
so applying it post-relaxation would delete most of a run).

Set them in the pack under ``relaxation.artifact_min_distance`` (g-xTB packs
already do).  Code defaults match the table if the key is omitted.

```yaml
relaxation:
  artifact_min_distance:
    Cd-Cd: 2.80
    Se-Se: 2.80
    Cl-Se: 2.80
    Cl-Cl: 2.80
```

| pair | audit threshold | basis |
|---|---|---|
| Cd-Cd | **2.80** | see below |
| Se-Se | 2.80 | real Se-Se bonds form at 2.34 |
| Cl-Se | 2.80 | real Cl-Se bonds form at 2.19 |
| Cl-Cl | 2.80 | Cl₂-like contacts ~2.0; nonbond ≥~3.3 |

The Cd-Cd value is set from 42829 relaxed Cd-Cd distances. Nothing exists below
2.64 A, and the population splits by bridging context:

    bridging atoms   n       min    p1
    0 (bare)         7279    2.64   2.72
    1                25547   2.82   3.20
    2                9907    2.90   3.04

Doubly-bridged Cd pairs bottom out at exactly 2.90, so a 2.90 cutoff clips
legitimate Cd(mu-X)2Cd rhombi. At 2.80, 219 genuinely bare Cd...Cd contacts are
flagged and **zero** bridged pairs are misflagged; at 2.90 ten are lost and at
3.00, fifty-eight.
