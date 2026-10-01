(guide-recipes)=
# Writing a recipe

A recipe is a YAML file that describes one nanocrystal. It is combined with
a CIF file of the bulk crystal on the command line:

```bash
nc-builder crystal.cif recipe.yaml -o particle.xyz
```

This page walks through the sections of a single-material recipe in the
order in which the builder uses them. The complete list of keys, with
types and defaults, is in {ref}`ref-yaml`. Core/shell particles use a
different layout, described in {ref}`guide-core-shell`.

## Charges

```yaml
charges: {Cd: 2, Se: -2, Cl: -1}
```

Every element that can appear in the particle needs a formal charge: the
native elements, the placeholder ligand, and any element introduced by a
post-treatment. Charges drive the bonding model, the charge balance and
the choice of facet terminations. If the ligand is missing from the list,
it is given a charge of −1.

## Size

```yaml
size_unit_cells: [2.0, 2.0, 2.0]
```

The size is given in unit cells along the three lattice vectors. Unequal
values produce elongated particles. Alternatively, the radius can be given
in ångström on the command line with `-r`, and `--size-unit-cells` on the
command line overrides the recipe. Fractional values are allowed and are
useful for fine size series, since the particle is cut from a discrete
lattice and many sizes map onto the same structure.

## Centre

```yaml
construction_origin: {center_on_species: Se}
```

The particle is built around one atom of the given species. A list, such as
`[Cd, Se]`, or `all` produces one particle per species. Without this block
the builder behaves as with `all` and writes one particle for every native
species. Explicit shifts are available as `cartesian_shift` (Å) and
`fractional_shift`.

## Facets

```yaml
facets:
  - {hkl: "100",    scope: family, termination: cation_rich, gamma: 1.0}
  - {hkl: "111",    scope: family, termination: cation_rich, gamma: 1.0}
  - {hkl: "-1-1-1", scope: family, termination: anion_rich,  gamma: 1.0}
```

Each entry names a facet by its Miller indices and gives its relative
surface energy `gamma`. With `scope: family` (the default), the facet
stands for all symmetry-equivalent facets. With `scope: facet`, it is a
single oriented facet, and every equivalent facet must then be listed.

For polar facets, `termination` selects the layer that the facet exposes:
`cation_rich` or `anion_rich`. Non-polar facets take `stoichiometric` or
no termination. The theory behind these choices is given in
{ref}`theory-wulff` and {ref}`theory-polarity`.

The Miller indices can be written as `"111"`, `"-1-1-1"`, `"1 1 1"` or
`[1, 1, 1]`. Quote them in YAML, so that `111` is not read as a number.

## Shape

```yaml
shape: {mode: wulff}            # or: {mode: sphere, sphere_planes: 192}
```

The default Wulff polyhedron can be replaced by a sphere. `aspect` stretches
the particle along the three axes when no size in unit cells is given.

## Passivation

```yaml
passivation:
  ligand: Cl
  surf_tol: 2.0
  prepass_mode: role-aware
  prepass_min_cn_terrace: 3
  prepass_min_cn_edge: 2
  prepass_min_cn_vertex: 1
```

`ligand` is the placeholder X-type ligand. It must not be an element of
the bulk crystal: for CsPbCl₃, use Br, for example. `surf_tol` is the depth
in ångström within which atoms count as surface atoms. The prepass
settings decide how many bonds a terrace, edge or vertex cation must keep
to stay in the particle. The `role-aware` values above allow edges and
corners to remain less coordinated than terraces, and they are the ones
used for the QDSpace library. The model is described in
{ref}`theory-charge-passivation`.

Whether excess metal is removed or compensated with extra ligands is
chosen on the command line with `--positive-q-mode remove|add`.

## Symmetry

```yaml
symmetry: {proper_rotations_only: true}
facet_options: {pair_opposites: true}
```

Proper rotations keep the {111} and {$\bar1\bar1\bar1$} families of zinc
blende separate, and `pair_opposites` adds the opposite of every
non-terminated seed. Both are on by default.

## Post-treatments

```yaml
post_treatment:
  surface_reconstruction: {enabled: true, ligand: Cl}
  ligand_exchange:
    enabled: true
    passes:
      - {replace: Cl, smiles: "CCCCC(=O)O", ratio: 0.5}
```

Everything that modifies the passivated particle lives here: the {111}
reconstruction, alloying, Z-type displacement, neutral exchange, X-type
ligand exchange and L-type ligands. The blocks are applied in a fixed
order, whatever their order in the file. See {ref}`post-overview`.

## Twins

```yaml
twins:
  - hkl: "111"
    intervals_angstrom: [[-2.0, 2.0]]
    operation: mirror
```

Twin boundaries are introduced by reflecting slabs of the particle, as
described in {ref}`theory-heterostructures`.
