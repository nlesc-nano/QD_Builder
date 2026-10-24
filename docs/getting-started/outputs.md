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
| `facet_geometry` | facet areas, edges and vertices of the Wulff polyhedron (below) |
| `site_roles` | facet, edge or vertex of every atom, aligned with the XYZ (below) |
| `composition` | the dot as $[MA]_m(MX_q)_k$ (below) |
| `target_search` | target mode only: every candidate and the best one ({ref}`target-mode`) |

Treatments that change the surface add their own ledgers, which record
exactly what was done:

| Key | Written by |
|---|---|
| `surface_reconstruction_ledger` | {ref}`theory-reconstruction` |
| `alloying_ledger` | {ref}`post-alloying` |
| `z_type_displacement_ledger` | {ref}`post-z-type` |
| `ligand_exchange_charge_ledger` | {ref}`post-x-type` |
| `z_type_ligand_ledger` | {ref}`post-z-type-ligands` |
| `z_type_target_ledger` | {ref}`z-type-target` |

(facet-geometry)=
## Facet geometry and site roles

Single-material Wulff builds (not spheres or stacks) report two polyhedra
with the facet normals of the construction, both clipped with a half-space
intersection, so a plane that only touches a corner has no face:

- `construction`: the final Wulff planes (terminations resolved, symmetry
  expanded, shifted off atom layers);
- `atom_fitted`: every plane moved to the outermost core atom along its
  normal, `d = max(n·r)` over the core atoms.

Each facet has a family label: the Miller family of its point-group orbit
(`100`, `110`, `210`, ...), with `A` or `B` for a polar family whose `+n`
and `-n` faces are not symmetry-equivalent. Orbits and polarity come from
the point group of the CIF, so any lattice and Miller index is labelled.
`A` exposes cations, `B` anions: in zinc blende `111A`/`111B`,
`211A`/`211B` and `311A`/`311B`, in wurtzite `001A`/`001B` and
`101A`/`101B`, while {100}, {110} and {210} carry no letter. The exposed
species is the recipe's resolved termination when it is `cation_rich` or
`anion_rich`, otherwise the bond rule used to resolve terminations (the
species that loses the fewest bonds across the plane). The two orbits of
a polar pair always get different letters. Hexagonal labels use the
three-index (hkl) of the recipe.

| Key in `facet_geometry.<polyhedron>` | Meaning |
|---|---|
| `areas_A2` | area per family label, Å² |
| `total_area_A2` | sum of the areas, Å² |
| `edges` | per facet-pair type (`100\|111A`, labels sorted): `n` edges and total `length_A`, Å |
| `n_faces`, `n_edges`, `n_vertices` | counts of the clipped polyhedron |
| `vertex_types` | vertices per type, the labels of the faces meeting there joined by `+` (`100+100+111A`) |
| `volume_A3` | volume, Å³ |
| `planes` | per oriented facet: `hkl`, `label`, `normal`, `d_A` (Å from the construction origin) and `area_A2` (0 when clipped away) |

`facet_geometry.origin_A` is the construction origin in the XYZ frame
(zero unless `--center` moved it). A polyhedron that cannot be formed (a
flat or empty core) has an `error` entry instead.

`site_roles` holds lists aligned with the atoms of the XYZ file:

| Key | Meaning |
|---|---|
| `roles` | `facet:<label>`, `edge:<label>\|<label>`, `vertex:<labels>` or `interior` |
| `is_core` | `true` for the inorganic core |
| `host` | for a ligand atom, the index of the core atom it binds to; `-1` for core atoms |
| `counts` | atoms per role and element, `core` and `ligand` |
| `tolerance_A` | distance tolerance used |

A core atom within `tolerance_A` of exactly one clipped face of the
atom-fitted polyhedron lies on that facet, of two adjacent faces on their
edge, and of three or more (or of two that meet only at a corner) at the
nearest vertex. Core atoms are the non-ligand atoms of the passivated cut;
complexes added by later post-treatments (Z-type MXn) are not. Ligand
atoms are grouped into molecules by covalent radii and take the role of
their host: the nearest core cation to the molecule's binding atom, or the
nearest core anion when that atom is a cation (Z-type metal).

The tolerance is set in the recipe:

```yaml
facet_geometry:
  enabled: true          # default
  role_tolerance: 1.0    # Å; zinc-blende Se lies 0.88 Å under a {111}A Cd layer
```

## Composition

`composition` writes the dot as $[MA]_m(MX_q)_k$:

| Key | Meaning |
|---|---|
| `formula_unit` | the bulk formula unit of the CIF (`CdSe`) |
| `m` | formula units in the core (the limiting species) |
| `core_excess` | core atoms beyond `m` formula units (`{"Cd": 13}`) |
| `x_type` | ligand symbol and number of X-type ligand atoms from the charge balance |
| `z_type` | Z-type units placed, per complex formula |
| `mx_unit` | the MX_q unit of the one excess cation (`CdCl2`) |
| `k_X` | excess cations in MX_q units ($N_\text{Cd} - N_\text{Se}$ for CdSe) |
| `k_Z`, `k` | Z-type MX_q units, and $k = k_X + k_Z$ |
| `formula` | e.g. `[CdSe]_55(CdCl2)_13` |
| `x_type_balanced` | the X-type ligands are exactly $q\,k_X$ |

The last six keys are present when the excess is a single cation species.

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
