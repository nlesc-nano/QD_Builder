# API

The builder is used mostly from the command line, but its modules can be
imported directly. The QDSpace web application does this. The main entry
points are listed below.

## Construction

```{eval-rst}
.. automodule:: builder.facets
   :members: unit_normal, expand_facets, detect_facets_from_nc

.. automodule:: builder.geometry
   :members: build_nanocrystal, build_spherical_nanocrystal, apply_core_lattice_fit
```

## Bonding and analysis

```{eval-rst}
.. automodule:: builder.analysis
   :members: derive_pair_cuts_from_cif, coord_numbers_bipartite, compute_cif_virtual_sites
```

## Passivation and ligand sites

```{eval-rst}
.. automodule:: builder.passivation_iterative
   :members: charge_balance_iterative

.. automodule:: builder.ligand_sites
   :members: select_ligand_sites
```

## Reconstruction

```{eval-rst}
.. automodule:: builder.facet_reconstruction
   :members: reconstruct_polar_facets
```

## Library records

```{eval-rst}
.. automodule:: builder.library_record
   :members:
```

## Recipes

```{eval-rst}
.. automodule:: builder.config
   :members: parse_yaml_config
```
