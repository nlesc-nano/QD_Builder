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

(facet-distances)=
## Facet distances instead of a size

```yaml
facet_distances: {"100": 12.4, "111A": 9.8, "111B": 14.1, "110": 13.0}   # Å
construction_origin: {center_on_species: Se}
```

`facet_distances` replaces `facets` and `size_unit_cells`: one plane
distance $h_f$ per facet family, in ångström from the construction origin.
A key is an hkl as in `facets`; a trailing `A` or `B` names the cation- or
anion-terminated orbit of a polar family (termination `cation_rich` or
`anion_rich`), so the polar zinc-blende {111} counts as two families. A
key without a letter for a polar family (`"111"`) puts both orbits at the
same distance. Internally every seed gets $\gamma_f = h_f$ and the radius
is $R = \min_g h_g$ with aspect (1, 1, 1), which places every plane at

$$
d_i = R\,\frac{\gamma_{f(i)}}{\gamma_\text{min}} = h_{f(i)} .
$$

The cut is therefore the same as the equivalent $\gamma$/size recipe. As in
every Wulff build, a plane within 0.05 Å of an atom layer is moved out by
0.25 Å, so a realised construction distance is $h_f$ or $h_f + 0.25$ Å
(the `planes` of `facet_geometry.construction` give the value used).
Pruning, the prepass and charge balance then act on the outermost layers:
the atom-fitted distances in `facet_geometry.atom_fitted` lie about one
layer inside $h_f$.

(target-mode)=
## Target mode: the build closest to a size and shape

```yaml
facet_distances: {"100": 10.0, "111A": 9.5, "111B": 11.5, "110": 12.0}
target: {m: 60, w: 1.0}        # optional: centres, roundings, max_builds
```

The cut is discrete, so a set of distances is a target rather than a
recipe. With `target`, the builder tries every construction centre and
every layer rounding of the distances, $h_f + r\,\delta_f$ with
$r \in$ `roundings` (default $-1, 0, 1$), and keeps the neutral build that
minimises

$$
J = \left(\frac{m_b - m}{m}\right)^2
  + w \sum_f \left(\frac{A_{f,b}}{A_b} - \frac{A_f(\vec h)}{A(\vec h)}\right)^2 .
$$

- $m_b$ counts formula units of the core of the build (`composition.m`), and
  $m$ is `target.m`. Without it, $m = V(\vec h)/v_\text{fu}$, the volume of the
  target polyhedron per formula unit of the CIF.
- $A_{f,b}$ are the atom-fitted areas of the build and $A_f(\vec h)$ those of
  the polyhedron with its planes at $h_f$.
- $\delta_f$ is the spacing of the cation layers along the family's normal,
  taken from the CIF: the plane spacing of the primitive lattice divided by
  the number of distinct layers of one sublattice per period. For zinc blende
  this gives $a/\sqrt3$ for {111}, $a/2$ for {100} and $a/(2\sqrt2)$ for
  {110}; for wurtzite (001) it gives $c/2$.
- `centres: auto` (the default) tries every species and the empty points of
  a 1/4 grid of the cell whose site symmetry is as high as that of the
  occupied sites, one per orbit. For zinc blende these are the tetrahedral
  and octahedral holes. A list of species or `construction_origin` mappings
  replaces them.

Distinct cuts are built once. Identical cuts from different roundings are
marked `same cut as candidate N`. Candidates skip the ligand-only
post-treatments (`z_type_ligands`, `neutral_ligands`), and the best one is
rebuilt with the full recipe. The recipe of that build is written next to
the output as `<name>_target.yaml`. The manifest's `target_search` holds
the target, $\delta_f$, the best candidate (centre, roundings, distances
used, $m_b$, $J$ and its two terms, areas, atom-fitted distances per family)
and every candidate tried. `max_builds` (400) caps the number of builds.

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
