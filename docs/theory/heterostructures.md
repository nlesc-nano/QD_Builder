(theory-heterostructures)=
# Heterostructures

Many of the most useful colloidal nanocrystals are not single materials. A
CdSe core overgrown with a ZnSe or ZnS shell confines the exciton away from
the surface; a PbS or CdSe domain fused to a perovskite forms a Janus
particle with two chemically distinct faces. Building such particles
atomistically raises two questions that a single-material construction
never faces. The first is geometric: the two lattices have different
lattice constants, and the model must decide where and how the mismatch is
accommodated. The second is chemical: the buried interface joins two
terminating layers, and the combination must be electrostatically sensible.

QD_Builder answers these questions in two different ways, depending on the
topology of the heterostructure. Concentric particles (core/shell and
core/crown) are built by *stack mode*, which cuts one particle on a single
shared lattice and then assigns chemistry by region. Particles in which two
materials meet across a single planar interface are built by the *Janus*
workflow, which searches for charge-compatible, lattice-matched interface
pairs and joins two half-particles.

## Concentric particles: one cut, many materials

### The shared lattice

Stack mode requires every material to share the same space group, for
example F$\bar4$3m for a family of zinc-blende compounds. Under this
condition the materials differ only by lattice constant and by the
identity of the cation and anion, and a single lattice can host all of
them. The builder cuts the whole particle from the lattice of the outermost
material that has a non-zero thickness. The outer surface, which will carry
the ligands, is therefore built with the correct bulk geometry of the
shell.

Each material contributes a *thickness increment* in unit cells, and the
layer boundaries follow from the running sum. A CdSe core of 1.5 cells
covered by a ZnSe shell of 1 cell produces boundaries at 1.5 and 2.5 cells.
A shell with the increment $[2, 2, 0]$ grows only in the plane and
produces a core/crown platelet. For every layer the builder constructs a
Wulff polyhedron (see {ref}`theory-wulff`) with that material's own facets
and surface energies, all on the shared lattice. The region of material $k$
is the set of points inside polyhedron $k$ but outside polyhedron $k-1$.

### Chemistry by relabelling

The particle is first passivated as if it were made of the reference
material alone. The builder then relabels every cation and anion in region
$k$ with the species of material $k$. Because relabelling preserves
formal charges only between isovalent species, the model is designed for
isovalent series such as Cd²⁺/Zn²⁺ with a common or isovalent anion. Each
material is assumed to have one cation and one anion.

Bond detection must not depend on the label an atom happens to carry. The
builder therefore merges the bond cut-offs of all materials. For every
element pair it keeps the largest cut-off found in any of the CIFs, and it
then gives every cation–anion pair in the same pair of charge classes the
largest cut-off of that class. A Zn–Se and a Cd–Se bond are thus recognised
with the same criterion, and coordination numbers are label-independent.

After relabelling, a second charge balance is run against the outermost
material with the coordination prepass disabled. This corrects any charge
change caused by the relabelling.

### Abrupt and mixed interfaces

By default the core/shell boundary is abrupt. With
`interface: {type: mixed}` the builder instead creates a graded interface.
It collects the core atoms within `mixing_width` (3 Å by default) of the
shell, and it converts a fraction `mixing_ratio` (0.5 by default) of these
cations and anions to the shell species. The converted atoms are chosen by
deterministic farthest-point sampling, so the alloyed band is spread
evenly around the core.

### Accommodating the lattice mismatch

A particle cut entirely on the shell lattice would compress or expand the
core to the shell's lattice constant everywhere, which is the fully
pseudomorphic limit. The builder relaxes this with a smooth affine
correction applied to the core region. Let $B_\text{core}$ and
$B_\text{shell}$ be the two lattice matrices. The map
$X = B_\text{core} B_\text{shell}^{-1}$ carries a point of the shell lattice
to the corresponding point of the core lattice. It is applied about the
core centre of mass $\mathbf c$ and weighted by the depth of each atom
inside the core polyhedron,

$$
\delta(\mathbf r) = \max\!\bigl(0,\ \min_j (d_j - \hat n_j\cdot\mathbf r)\bigr),
\qquad
w(\mathbf r) = \tfrac12 - \tfrac12\cos\!\Bigl(\pi\,\min\!\bigl(1, \delta/W\bigr)\Bigr),
$$

$$
\mathbf r' = \bigl(1 - w\bigr)\,\mathbf r + w\,\bigl[X(\mathbf r - \mathbf c) + \mathbf c\bigr].
$$

Atoms deeper than the width $W$ (2 Å by default, `--core-strain-width`)
adopt the bulk lattice of the core. Atoms at the core/shell boundary stay
on the shell lattice. In between, the cosine ramp interpolates smoothly, so
the interface is coherent while the interior of the core is unstrained.
Shell atoms and ligands are not moved. The correction is applied by
default in stack mode and can be switched off with `--no-core-lattice-fit`.

## Janus particles: joining two half-particles

### Terminations and their charges

A planar interface between two different crystals joins two terminating
layers, one from each side. The Janus workflow begins by enumerating the
possible terminations of every symmetry-distinct facet family of both
materials, up to a chosen Miller index. For a direction $\mathbf G_{hkl}$
each site is assigned the phase $(\mathbf f\cdot hkl) \bmod 1$ of its
fractional coordinate $\mathbf f$. Sites whose phases lie within
$\ell / d_{hkl}$ of each other are grouped into one atomic layer, with
$d_{hkl} = 1/\lvert\mathbf G_{hkl}\rvert$. Each layer receives its
composition and its formal charge $Q = \sum_i q_i$.

A family is *polar* when it contains charged layers and its $+hkl$ and
$-hkl$ directions are not symmetry-equivalent, so that the two sides of a
slab genuinely differ. This is the case Tasker identified as intrinsically
unstable {cite:p}`tasker1979`. A family with charged layers but equivalent
opposite directions is *termination-sensitive*. A family whose layers are
all neutral and stoichiometric is *non-polar*.

### Pairing by charge

Two terminations make an electrostatically reasonable interface when their
charges compensate. Pairs of opposite charge are preferred, then pairs of
two neutral layers; charged–neutral pairs are considered only on request.
Within a class, pairs are ranked by the residual charge
$\lvert Q_\text{core} + Q_\text{shell}\rvert$ and then by Miller-index
simplicity. No interface energy is computed. Charge compensation, lattice
matching and index simplicity act as physically motivated proxies for it.

### Lattice matching

Commensurability is tested with the Zur–McGill superlattice search as
implemented in pymatgen. For each candidate pair the in-plane lattices of
the two terminations are searched for supercells whose vector lengths,
area and angle agree within tolerances. The defaults are a length mismatch
of 3 %, an area-ratio tolerance of 9 %, an angle tolerance of 0.01° and a
maximum supercell area of 400 Å². The smallest match is kept, and the
largest relative length difference
$\max_i \lvert L_i^\text{shell} - L_i^\text{core}\rvert / L_i^\text{core}$
is reported as the lattice mismatch. Candidates without a match are
discarded, and the survivors are re-ranked by charge class, residual
charge, mismatch, angle and supercell area.

### Building the particle

In the full Janus construction (`wulff_janus`) both materials are first
built as complete Wulff particles on their own lattices, with their own
facets. Both are rotated so that the candidate interface normal points
along $+z$. The in-plane strain of the lattice match is applied entirely to
the shell side. The transformation $T$ that maps the shell supercell onto
the core supercell multiplies the shell's in-plane coordinates, while the
coordinate along the normal is left unchanged. Each particle is then cut at
a layer of the required termination: the one closest to the middle of the
particle whose charge and composition match the candidate. The core half is
placed below $z = 0$ and the shell half above a chosen interface distance
(2.8 Å by default).

The shell half can be clipped to the footprint of the core, either as a
bounding box or a convex hull, or as a *mushroom* cap that widens smoothly
with height as $h(z) = h_\text{core} + o\,\sin(\pi t/2)$, where
$t$ is the normalised height and $o$ the overhang. The outer surface is
described by planes rebuilt from the actual atoms along all facet
directions of both materials. Only these outer planes are passivated, so
the buried interface is never capped, and the positive-charge strategy can
be chosen separately for the core and shell sides.

Two simpler geometries are also available. The `radius` mode cuts a sphere
around the interface and is useful for inspecting the junction without
passivation. The `interface_cell` mode cuts a rectangular slab with a given
number of layers on each side.

## Twin boundaries

Stacking faults and twins are common in real zinc-blende and wurtzite
particles, and the builder can introduce them by reflection. A twin
directive names a lattice plane $(hkl)$ with unit normal
$\hat n \propto A^{-\mathsf T}\,hkl$ and spacing
$d_{hkl} = 1/\lVert A^{-\mathsf T}\,hkl\rVert$, where $A$ holds the lattice
vectors as columns. It also gives one or more intervals of the signed
coordinate $t = (\mathbf r - \mathbf o)\cdot\hat n$. Atoms inside an interval
are reflected through the plane $\hat n\cdot\mathbf x = c$,

$$
\mathbf r \mapsto \mathbf r - 2\,(\hat n\cdot\mathbf r - c)\,\hat n,
$$

with the mirror plane at the midplane or the entry of the interval,
optionally snapped to a lattice plane. A `mirror+shift` operation adds a
translation along the normal (half a layer spacing by default) and an
optional in-plane glide. The sublattices can be swapped to preserve the
correct polarity across the boundary. The material beyond the twinned slab
can be *stitched*, i.e. the glide undone, so that the far side stays on the
original lattice. After the operation the particle is re-cut with its
outer planes, and in single-material mode any voids left by the reflection
are refilled from a twinned template.
