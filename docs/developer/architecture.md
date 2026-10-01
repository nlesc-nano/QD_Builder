# Architecture

## The pipeline

A single-material build runs through the following stages, all orchestrated
by `builder.main.main`:

```text
CIF + recipe
   │  config.parse_yaml_config
   ▼
facet seeds ── main._resolve_facet_terminations ── facets.expand_facets
   │
   ▼  for every centre variant
geometry.build_nanocrystal  (Wulff cut, auto-shift of coincident planes)
   │  cleanup.prune_low_coord_sites
   ▼
facets.detect_facets_from_nc  ──  termination check (flip and rebuild once)
   │
   ▼  main._run_passivation_and_write_outputs
passivation_iterative.charge_balance_iterative
   │   prepass (passivation.prepass_surface_cleanup)
   │   priority loop; ligand addition via ligand_sites.select_ligand_sites
   ▼
post-treatments
   surface_reconstruction → [stack relabel] → alloying (+ rebalance)
   → z_type_displacement → neutral_exchange → ligand_exchange (+ compensation)
   → neutral_ligands
   ▼
XYZ + JSON manifest
```

Stack recipes add a layer step: one cut on the outermost lattice, regions
per material, relabelling and rebalance, and a final core lattice fit
(`stack`, `geometry.apply_core_lattice_fit`).

## Modules

| Module | Responsibility |
|---|---|
| `config`, `nc_types` | recipe parsing and data classes |
| `facets`, `geometry` | normals, symmetry expansion, Wulff and sphere cuts |
| `analysis` | pair cut-offs, coordination, bulk references, virtual sites, reports |
| `cleanup` | pruning of low-coordination atoms |
| `passivation`, `passivation_iterative` | prepass, charge balance, ligand migration |
| `ligand_sites` | the shared ligand-site selector |
| `facet_reconstruction` | polar {111} reconstruction |
| `*_posttreat` | alloying, Z-type, neutral exchange, X-type exchange, L-type ligands |
| `stack`, `heterointerface`, `twinbound`, `twin_workflow` | heterostructures and twins |
| `library_record`, `scripts.generate_library` | structure records and library series |
| `io_utils` | XYZ and manifest writers |

## Design principles

- Every geometric threshold that matters physically is derived from the
  bulk lattice of the material, not from fixed ångström values: bond
  lengths, neighbour distances, shell separators and ligand spacings.
- Rules are general. A behaviour that should differ between material
  families is expressed through the charges and the bulk coordination, not
  through special cases for named compounds.
- Results are reproducible. Ties are broken deterministically, and every
  random choice is seeded from the recipe.
- Charge is never forced. When neutrality cannot be reached, the builder
  reports the residual charge instead of making an arbitrary edit.
