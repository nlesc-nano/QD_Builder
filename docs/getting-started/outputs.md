# Output files

## The structure

The main output is an XYZ file. Its first line is the number of atoms and
its second the file name, followed by one line per atom with the element
and Cartesian coordinates in ångström. Atoms are ordered as cations, anions
and ligands; in core/shell particles they are ordered by layer. With
`--center`, the centroid of all atoms is placed at the origin.

## The manifest

Next to every XYZ file the builder writes a JSON manifest with the same
name.

| Key | Meaning |
|---|---|
| `counts` | number of atoms of each element |
| `total_charge` | net formal charge (corrected for exchanged molecular ligands) |
| `material` | material label |
| `size_unit_cells` | requested size |
| `construction_radius_ang`, `construction_diameter_ang` | radius and diameter of the Wulff construction |
| `actual_radius_ang`, `actual_size_unit_cells` | size of the particle actually obtained |
| `size_metrics` | box-span estimate of the diameter |
| `stack_size_metrics` | core/shell particles: size per layer |

Treatments that change the surface add their own ledgers, which record
exactly what was done:

| Key | Written by |
|---|---|
| `surface_reconstruction_ledger` | {ref}`theory-reconstruction` |
| `alloying_ledger` | {ref}`post-alloying` |
| `z_type_displacement_ledger` | {ref}`post-z-type` |
| `ligand_exchange_charge_ledger` | {ref}`post-x-type` |

## Several centres, several files

When `construction_origin` lists several species, or when it is omitted
(which is equivalent to `all`), the builder produces one particle per
centre. It appends the centre label to the file name, as in
`cdse_Cd.xyz` and `cdse_Se.xyz`, and copies the first variant to the name
given with `-o`. When the size is given on the command line with
`--size-unit-cells`, the files are named
`<Material>_c<centre>_rep<size>.xyz`, which is convenient for size scans.

## Intermediate structures

With `--write-all` the builder also writes the structure at each stage:
- `<name>_cut.xyz`, the pruned cut before passivation;
- `<name>_01_before_prepass.xyz` and `<name>_02_after_prepass.xyz`, around
  the removal of under-coordinated atoms;
- `<name>_03_stabilized_pre_Q.xyz`, once the structure is stable but not
  yet neutral;
- for core/shell particles, the core and every shell separately.

These files make it possible to follow each step of the construction
visually.
