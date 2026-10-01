(theory-wulff)=
# The Wulff construction

## Equilibrium shape

The equilibrium shape of a crystal minimises its total surface free energy
at fixed volume. Wulff showed that this shape is obtained by drawing, for
every orientation $\hat n$, a plane at a distance proportional to the
surface energy $\gamma(\hat n)$ from a common centre, and taking the inner
envelope of all planes {cite:p}`wulff1901`. For a crystal with a finite
set of facets, the envelope is a convex polyhedron: the intersection of the
half-spaces

$$
\hat n_i\cdot(\mathbf r - \mathbf r_0) \le d_i ,\qquad d_i \propto \gamma_i .
$$

Facets with low surface energy lie close to the centre and dominate the
shape. Facets with high energy lie far away and appear only as small
truncations, or not at all.

Colloidal nanocrystals are not equilibrium objects, since their shapes
depend on kinetics and on ligand binding. The Wulff construction
nonetheless provides a compact and reproducible way to express a target
shape. In QD_Builder the values $\gamma_i$ are *relative* energies chosen by
the user to obtain a desired morphology: a cube, an octahedron, a
cuboctahedron, or a tetrahedral zinc-blende particle with inequivalent
{111} and {$\bar1\bar1\bar1$} facets. They are not computed surface
energies.

## From seeds to oriented facets

The recipe lists *seed* facets, each with Miller indices and a relative
energy. By default a seed stands for its whole symmetry family
(`scope: family`). The builder expands it by applying the point-group
operations of the crystal, obtained from pymatgen's space-group analysis
{cite:p}`ong2013`. Each operation is applied to the reciprocal-lattice
vector $\mathbf G_{hkl} = h\mathbf b_1 + k\mathbf b_2 + l\mathbf b_3$, not
to its unit normal, so that the rotated vector can be converted back to
integer Miller indices for any lattice constant. The indices keep their
sign, so $(hkl)$ and $(\bar h\bar k\bar l)$ remain distinct facets.

By default only proper rotations are used (`symmetry.proper_rotations_only`).
In a non-centrosymmetric group such as F$\bar4$3m (zinc blende) this keeps
the four {111} facets and the four {$\bar1\bar1\bar1$} facets in separate
orbits, so the two can carry different surface energies and terminations.
When `facet_options.pair_opposites` is on (the default), a seed without a
termination is automatically paired with its opposite at the same energy.
A seed with `scope: facet` is taken literally, as one oriented facet with
its own energy. The builder then requires every symmetry-equivalent facet
to be listed explicitly.

## Size and aspect ratio

Sizes are specified in unit cells (`size_unit_cells`) or as a radius in
ångström (`-r`). A size $(s_a, s_b, s_c)$ is converted to physical lengths
$s_i\,\lvert\mathbf a_i\rvert$. Their minimum defines the construction
radius $R$, and their ratios define an aspect $(\alpha_x, \alpha_y,
\alpha_z)$. The plane distances are

$$
d_i = \frac{R}{\min_j \gamma_j}\;\gamma_i\;
\sqrt{(\alpha_x n_{i,x})^2 + (\alpha_y n_{i,y})^2 + (\alpha_z n_{i,z})^2}.
$$

The facets of lowest energy therefore sit exactly at $R$ for an isotropic
particle, and the others at $R\,\gamma_i/\gamma_\text{min}$. Aspect ratios
stretch the polyhedron along the Cartesian axes, which coincide with the
lattice axes for the cubic and orthorhombic cells the builder is designed
for. A size of 2 unit cells of CdSe ($a = 6.08$ Å) gives $R \approx 12.2$ Å.
Because the particle is cut from a discrete lattice, its actual size is
reported separately after construction.

## The centre of the particle

The lattice is generated around a construction origin $\mathbf r_0$, and
the choice of this origin changes the particle. A zinc-blende particle
centred on a cation and one centred on an anion of the same nominal size
differ in stoichiometry, in the terminations of their facets and in their
point symmetry. The recipe controls the origin with `construction_origin`.
`center_on_species: Se` places the nearest Se site of the unit cell at the
origin. A list of species, or `all`, produces one particle per species.
Explicit Cartesian or fractional shifts are also accepted. The option
`--center` merely translates the written coordinates so that their centroid
is at the origin.

## Planes that cut through atoms

A lattice plane that coincides with a Wulff plane poses a numerical
problem. Whether the atoms of that layer are included depends on a
floating-point tolerance, and symmetry-equivalent facets can end up
treated differently. The builder detects this case. If any atom lies within
0.05 Å of a plane, the plane is moved outward by 0.25 Å so that the whole
layer is included, and a `[wulff:shift]` message is printed. The particle
thus always ends on complete atomic layers.

## After the cut

The raw cut can contain atoms bonded by only one bond. These are pruned
iteratively before any analysis, with the threshold set by
`--prune-min-cn` (2 by default, or the role-aware vertex threshold if that
is lower). The builder then detects the facets that the particle actually
exposes. For every seed orientation it places a plane on the outermost
atom along $\pm\hat n$ and keeps the facet if the plane touches the
surface within the surface tolerance. These *detected* planes, which fit
the atoms tightly, rather than the construction planes, are the ones used
by passivation and in all reports.

Instead of a polyhedron, the builder can also cut a sphere
(`shape.mode: sphere`). The sphere is approximated by a large number of
planes (192 by default) with Fibonacci-distributed normals, all at the
same distance.
