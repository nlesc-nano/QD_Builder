(theory-bonding)=
# Bonds, coordination and surface roles

Every decision the builder takes after the Wulff cut depends on a single
question: which atoms are bonded to which. The answer determines the
coordination number (CN) of each atom, which atoms belong to the surface,
and how many bonds a surface atom is missing. This chapter describes the
bonding model on which passivation, reconstruction and the post-treatments
all rest.

## An ionic, bipartite picture

The builder assigns each species an integer formal charge, taken from the
`charges` block of the recipe (for CdSe, Cd +2 and Se −2). Bonds are
counted only between atoms of opposite charge. The coordination number of
atom $i$ is

$$
\text{CN}_i = \bigl\lvert\{\, j : q_i q_j < 0,\ d_{ij} \le r_c(s_i, s_j)\,\}\bigr\rvert ,
$$

where $r_c$ is a cut-off specific to the pair of species. In this
*bipartite* picture a cation is coordinated by anions and by anionic
ligands, and an anion by cations. The picture matches the chemistry of the
ionic and polar-covalent semiconductors the builder targets, and it gives
every surface atom an unambiguous number of missing bonds. An atom of zero
charge counts all neighbours within the cut-off.

## Calibrating the bond cut-offs from the bulk

Fixed covalent radii are a poor guide to bonding in compounds whose bond
lengths vary by several tenths of an ångström across a family. The builder
therefore calibrates every cut-off on the bulk crystal it is building from.
It expands the CIF to a 3×3×3 supercell. For every opposite-charge pair of
species it records, for each site, the distance to the nearest and to the
second-nearest partner. If the 1st percentile of the second-nearest
distances, $r_2^\text{min}$, exceeds the 99th percentile of the nearest
distances, $r_1^\text{max}$, the cut-off is placed halfway between them,

$$
r_c = \tfrac12\bigl(r_1^\text{max} + r_2^\text{min}\bigr).
$$

Otherwise $r_c = 1.05\,r_1^\text{max}$. Because the comparison is between
the nearest and the second-nearest *atom*, the second branch applies to any
structure in which an atom has several equivalent nearest neighbours, such
as the tetrahedral zinc-blende and the octahedral rock-salt lattices. For
these the cut-off is 5 % beyond the bulk bond length. With too few samples
the builder falls back to $1.25\,(r_a + r_b)$ from covalent radii.

Ligands need a cut-off as well. For an anionic ligand $L$ that is not part
of the bulk crystal, the cation–ligand cut-off is

$$
r_c(M, L) = \max\Bigl(1.25\,(r_M + r_L),\ \max_{X} r_c(M, X)\Bigr),
$$

with the maximum taken over the native anions $X$. The second term
guarantees that a ligand placed exactly on a vacant anion site of the
lattice is always recognised as bonded to its host. Without it, a small
placeholder such as Cl on a large anion site (for example Se in PbSe) could
fall outside the covalent cut-off, be classified as an orphan and be
removed again.

In core/shell structures, bond detection must not depend on the label an
atom carries. The cut-offs of all materials are therefore merged, and every
cation–anion pair in the same pair of charge classes receives the largest
cut-off of that class (see {ref}`theory-heterostructures`).

## Bulk coordination and the deficit

The number of bonds an atom *should* have is its bulk coordination,
$\text{CN}^\text{bulk}$. It is either counted directly from the CIF or
estimated as the most frequent coordination of interior atoms. It is
harmonised so that all species of the same charge share the largest value.
The *deficit* of a surface atom,

$$
\Delta_i = \max\bigl(0,\ \text{CN}^\text{bulk}(s_i) - \text{CN}_i\bigr),
$$

is the number of dangling bonds it exposes. Passivation reduces the
deficits of cations by adding ligands, while the prepass removes atoms
whose deficit is too large for them to be retained.

## Surface shells, depth and roles

Surface atoms are identified with respect to the facet planes of the
particle. A plane $f$ is described by its unit normal $\hat n_f$ and offset
$d_f$. The depth of atom $i$ below it is
$t_{if} = d_f - \hat n_f\cdot\mathbf r_i$. An atom belongs to the surface
shell of facet $f$ when $t_{if}$ is smaller than the surface tolerance
`surf_tol`. It belongs to the *outer* layer when $t < 0.35\,$`surf_tol`,
and to the *sublayer* when $0.35 \le t/\text{surf\_tol} < 1.2$.

Atoms at the junction of facets are more exposed than atoms on a terrace,
and the builder distinguishes three *roles*. By membership, an atom in one
surface shell is a terrace (*unique*) atom, in two an *edge* atom, and in
three or more a *vertex* atom. When the facet planes are available, the
role is determined geometrically instead. An atom within
$\max(0.75\,\text{surf\_tol}, 0.75)$ Å of a point where three planes
intersect is a vertex, and one within $\max(0.25\,\text{surf\_tol}, 0.35)$ Å
of a line where two planes intersect is an edge. Roles drive the
*role-aware* coordination thresholds of the prepass (see
{ref}`theory-charge-passivation`), which allow edges and vertices to retain
fewer bonds than terraces.

## Virtual lattice sites

Many operations need to know *where* a missing bond points. For every
surface atom the builder maps the particle back onto the bulk lattice. It
finds the fractional shift that brings the first native atoms onto
same-species CIF sites within 0.05 in fractional units. It then compares
the actual bonds of the atom with the ideal first-shell directions of the
corresponding bulk site. An ideal direction that is not matched by an
actual bond (cosine below 0.70) and points outward (positive projection on
the local surface normal) marks a missing neighbour. The position of that
neighbour is exactly the bulk lattice site
$\mathbf x = \mathbf x_\text{host} + \mathbf v^\text{bulk}$.

Missing sites of neighbouring hosts often coincide. Sites closer than
0.2 Å are merged into one *virtual site*, whose multiplicity $\mu$ is the
number of hosts that share it. A $\mu = 1$ site is an on-top position, a
$\mu = 2$ site bridges two cations, and a $\mu = 3$ site caps three. These
virtual sites are the natural positions for X-type ligands, since a ligand
placed there continues the anion sublattice of the crystal (see
{ref}`theory-ligand-sites`).
