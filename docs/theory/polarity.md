(theory-polarity)=
# Surface polarity and terminations

## Stacks of charged layers

A crystal viewed along a direction $\hat n$ is a stack of atomic layers. In
rock salt along ⟨100⟩, every layer contains cations and anions in equal
number and is neutral; such surfaces are *non-polar*. In zinc blende along
⟨111⟩, each layer contains a single species, and cation and anion layers
alternate. The stack therefore carries a dipole in every repeat unit.
Tasker classified ionic surfaces on this basis: stacks of neutral layers
(type 1), charged layers whose repeat unit has no dipole (type 2), and
charged layers whose repeat unit carries a dipole (type 3)
{cite:p}`tasker1979`. Type-3 surfaces are electrostatically unstable unless
their charge is compensated, by reconstruction, by adsorbates or, in
nanocrystals, by ligands.

A polar facet can end on either kind of layer. The recipe expresses this
with the `termination` of a facet seed: `cation_rich` asks for a facet whose
outermost layer consists of cations, `anion_rich` for one that ends on
anions, and `stoichiometric` (or no termination) leaves the choice to the
geometry of the cut. Non-polar facets such as rock-salt or perovskite {100}
are always stoichiometric.

## Which sign exposes which layer?

For a family such as {111}, the two signed directions $(111)$ and
$(\bar1\bar1\bar1)$ expose different layers when the crystal is cut. The
builder must decide which of the two to use as the construction normal for
a given termination, and the decision must not depend on how the CIF file
happens to be written. The same zinc-blende compound can be described with
the cation at the origin and the anion at $(\tfrac14, \tfrac14, \tfrac14)$,
with the anion at $(\tfrac14, \tfrac14, \tfrac34)$, or with the two species
exchanged. These settings are equivalent, but they invert the meaning of
the Miller-index sign.

The builder therefore derives the exposed layer from the bonding topology.
Cutting a crystal perpendicular to $\hat n$ breaks the bonds that point
outward, and the energetically preferred cut breaks as few bonds per
surface atom as possible. For every site of the unit cell the builder
counts the bonds whose projection on $\hat n$ is positive, i.e. the bonds
that would be severed. The species with the smallest positive count is the
one exposed by a facet with outward normal $\hat n$. On zinc-blende {111}
one species has one bond pointing out and the other three. The first is the
terminating layer, with a single dangling bond per atom, exactly as for a
real (111) surface. The charge of the exposed species then tells whether
the orientation is cation- or anion-terminated. The builder selects the
sign that realises the requested termination.

When both species lose the same number of bonds, as on rock-salt {111},
where every atom has three bonds on each side, the bond count cannot
decide. The builder then falls back to the layer that lies outermost in
the unit cell along the opposite normal. This choice depends on the
setting of the CIF, but it only matters for symmetric cases in which both
choices are chemically equivalent.

## Verifying the cut

The layer a facet exposes in a finite particle also depends on the size and
centre of the cut. After cutting and pruning, the builder therefore checks
every facet of a family that was requested with both terminations. It sums
the formal charge of the atoms within 0.25 Å of the facet plane. A positive
sum identifies a cation-terminated facet, a negative sum an
anion-terminated one. If a facet does not show the requested termination,
all terminated seeds are flipped and the particle is rebuilt once. If the
second attempt also fails, a particle with several centre variants skips
that variant, and a single build proceeds with a warning.

## Consequences for passivation

Polarity determines which atoms will need ligands. A cation-terminated
facet exposes metal atoms with dangling bonds, which are the hosts for
X-type ligands. An anion-terminated facet exposes chalcogen or pnictogen
atoms, which may be converted into ligand sites or, on zinc-blende {111},
reorganised by the reconstruction described in
{ref}`theory-reconstruction`. The requested terminations thus fix the
starting point for the charge balance of {ref}`theory-charge-passivation`.
