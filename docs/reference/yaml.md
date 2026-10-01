(ref-yaml)=
# Recipe schema

This page lists every key of a single-material or stack recipe, with its
type, default and accepted values. The recipe is parsed by
`builder.config.parse_yaml_config`. A recipe with a `materials` key is a
stack (core/shell) recipe.

## Top level

| Key | Type | Default | Notes |
|---|---|---|---|
| `charges` | map element → int | required | formal charges |
| `passivation` | map | required | must name a `ligand` (or `anion_ligand`) |
| `facets` (alias `seeds`) | list or map | required unless `shape.mode: sphere` | single-material recipes |
| `size_unit_cells` | number, 3-list or string | none | e.g. `2`, `[2, 2, 1]`, `"2 2 1"` |
| `shape` | map | Wulff, aspect (1,1,1) | see below |
| `construction_origin` | map | behaves like `all` | see below |
| `symmetry.proper_rotations_only` | bool | `true` | |
| `facet_options.pair_opposites` | bool | `true` | add the opposite of non-terminated family seeds |
| `twins` | map or list | none | see below |
| `post_treatment` (alias `post-treatment`) | map | all disabled | see below |
| `experimental.exhausted_positive_q_fallback` | bool | `false` | last-resort cation removal |
| `stack` | map | see below | stack recipes |
| `materials` | list | – | selects stack mode |

## Facets

Each facet is a mapping `{hkl, gamma, scope, termination}`. The shorter
forms `{"100": 1.0, "111": 1.2}` and `[["100", 1.0]]` are also accepted.

| Field | Default | Values |
|---|---|---|
| `hkl` | required | `"111"`, `"-1-1-1"`, `"1 1 1"`, `"(1 1 1)"`, `[1, 1, 1]` |
| `gamma` | required | relative surface energy |
| `scope` | `family` | `family` or `facet` |
| `termination` | none | `cation_rich`, `anion_rich`, `stoichiometric` (= none) |

## Shape

| Key | Default | Values |
|---|---|---|
| `mode` | `wulff` | `wulff` or `sphere` |
| `sphere_planes` | 192 | at least 12 |
| `aspect` | (1, 1, 1) | 3-list, `{a, b, c}` or `"ax ay az"` |

## Construction origin

| Key | Meaning |
|---|---|
| `center_on_species` (alias `center_on`) | a species, a list of species, `"Cs, Pb"`, or `all` |
| `cartesian_shift` | shift in Å |
| `fractional_shift` | shift in fractional coordinates |

## Passivation

| Key | Default | Meaning |
|---|---|---|
| `ligand` (alias `anion_ligand`) | required | placeholder X-type ligand |
| `cation_ligand` | none | cationic placeholder symbol |
| `surf_tol` | 1.0 Å | surface-shell depth |
| `prepass_mode` | `standard` | `standard` or `role-aware` |
| `prepass_min_cn_terrace` | 3 | |
| `prepass_min_cn_edge` | 3 | |
| `prepass_min_cn_vertex` | 3 (1 if role-aware) | |
| `include_sublayer` | `false` | also passivate sublayer cations |
| `neutral_ligands` | disabled | legacy location of the L-type block |

## Post-treatments

Every block has `enabled` (default `false`) and `seed` (default 1337). All
blocks except the reconstruction take a list of `passes`. Every pass
accepts `ratio` (0–1, default 1.0), `target_count` (alias `count`, default
0, which takes precedence when positive), and `distribution` (`random`,
`uniform` or `segmented`, default `random`).

| Block | Pass keys | Chapter |
|---|---|---|
| `surface_reconstruction` | (no passes) `ligand`, `cation_removal` (`auto`, `mirror`, `max`) | {ref}`theory-reconstruction` |
| `alloying` (alias `alloy`) | `replace`, `with`, `with_charge`, `region` (`surface`, `core`, `both`) | {ref}`post-alloying` |
| `z_type_displacement` | `cation`, `anion`, `anion_count` | {ref}`post-z-type` |
| `neutral_exchange` | `cation`, `anion`, `anion_count`, `exchange_type` (`mxn`, `zwitterion`, `l_type`), `smiles` | {ref}`post-neutral-exchange` |
| `ligand_exchange` | `replace`, `smiles`, `charge`, `replace_charge`; block keys `ff` | {ref}`post-x-type` |
| `neutral_ligands` | `target` (`cation`, `anion`, `both`), `target_symbol`, `smiles`; block key `ff` | {ref}`post-l-type` |

## Twins

| Key | Default | Meaning |
|---|---|---|
| `hkl` | required | twin plane |
| `origin` | `center` | `center` or `[x, y, z]` |
| `intervals_angstrom` / `intervals_layers` | one required | slabs to reflect, in Å or layer spacings |
| `mirror_at` | `midplane` | `midplane` or `entry` |
| `snap_to_layers` | `false` | snap the mirror to a lattice plane |
| `operation` | `mirror` | `mirror` or `mirror+shift` |
| `shift_angstrom` / `shift_layers` | half a layer | normal shift for `mirror+shift` |
| `parallel_shift_angstrom` | none | in-plane glide |
| `swap_sublattice` | `false` | exchange cation and anion in the slab |
| `stitch_beyond` | `auto` (single material) | undo the glide beyond the slab |
| `refill_missing` | `true` | refill voids from a twinned template |

## Stack recipes

`stack` block:

| Key | Default | Values |
|---|---|---|
| `interface` | `abrupt` | `abrupt` or `mixed` |
| `mixing_width` | 3.0 Å | width of the mixed band |
| `geometry_reference` | `core` | parsed; the cut currently uses the outermost material's lattice |

Each entry of `materials`:

| Key | Default | Meaning |
|---|---|---|
| `name` | `material` | label |
| `cif` | required | bulk structure |
| `facets` | required unless sphere | as above |
| `shape` | as above | |
| `size_unit_cells` | none | thickness increment |
| `interface` | none | on the second material: `type`, `mixing_width`, `mixing_ratio` (0.5) |
