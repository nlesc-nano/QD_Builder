# Command line

```text
nc-builder CRYSTAL.cif RECIPE.yaml [options]     # single material
nc-builder RECIPE.yaml [options]                 # stack (core/shell)
```

`python -m builder` is equivalent to `nc-builder`.

## Output

| Option | Default | Meaning |
|---|---|---|
| `-o`, `--out` | `nanocrystal.xyz` | output XYZ; the manifest is written next to it |
| `--center` | off | place the centroid at the origin |
| `--write-all` | off | write intermediate structures |
| `--verbose` | off | detailed log of every step |

## Size

| Option | Default | Meaning |
|---|---|---|
| `-r`, `--radius` | none | construction radius in Å |
| `--size-unit-cells` | none | size in unit cells (overrides the recipe) |
| `--aspect AX AY AZ` | none | aspect ratio when no size is given |

## Passivation

| Option | Default | Meaning |
|---|---|---|
| `--positive-q-mode` | `remove` | `remove` cations or `add` ligands when the particle is positive |
| `--prepass-mode` | from recipe | `standard` or `role-aware` |
| `--prepass-min-cn-terrace`, `-edge`, `-vertex` | from recipe | role thresholds |
| `--prune-min-cn` | 2 | pre-analysis prune threshold |
| `--prune-passes` | 10 | maximum prune passes |
| `--no-prune-mono` | – | disable the prune |

## Symmetry

| Option | Default | Meaning |
|---|---|---|
| `--proper-rotations-only` / `--no-proper-rotations-only` | from recipe | single-material mode |

## Core/shell

| Option | Default | Meaning |
|---|---|---|
| `--no-core-lattice-fit` | – | keep the core on the shell lattice |
| `--core-strain-width` | 2.0 Å | width of the strain transition |
| `--core-center` | `com` | `com` or `origin` for the lattice mapping |

## Diagnostics

| Option | Default | Meaning |
|---|---|---|
| `--scan-facets` | off | polarity scan of the CIF before building |
| `--scan-max-index` | 2 | maximum Miller index |
| `--scan-slab-size`, `--scan-vacuum-size` | 18, 20 Å | slab geometry |
| `--scan-shifts` | 8 | terminations sampled per orientation |

Setting the environment variable `QD_BUILDER_UNBUFFERED` flushes the log
line by line, which is useful when the builder runs behind a web service.

## Scripts

| Module | Purpose |
|---|---|
| `builder.scripts.generate_library` | library size series ({ref}`guide-library`) |
| `builder.scripts.build_janus_heterostructures` | Janus particles ({ref}`guide-janus`) |
| `builder.scripts.scan_heterointerfaces` | interface candidates of two crystals |
| `builder.scripts.analyze_cif_facets` | facet families and terminations of one crystal |
| `builder.scripts.scan_size_cells` | composition and size over a range of sizes |

Run each with `python -m <module> --help` for its options.
