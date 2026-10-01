(theory-overview)=
# From a crystal to a nanocrystal model

A colloidal semiconductor nanocrystal is, to a first approximation, a
fragment of a bulk crystal. Its interior retains the lattice of the parent
compound, while its surface is reshaped by the growth conditions and
capped by ligands that bind to under-coordinated atoms and balance the
charge. Atomistic simulations of such particles, whether they target the
electronic structure, the surface chemistry or the dynamics, need models
that respect both facts. The inorganic core must be crystalline and
faceted, and the surface must be chemically plausible and electrically
neutral {cite:p}`boles2016,owen2015`.

QD_Builder constructs such models by following a sequence of physically
motivated steps. Each step addresses one aspect of the problem, and each is
the subject of a chapter in this part of the documentation.

**Shape.** The particle is carved from the bulk lattice as the intersection
of half-spaces whose distances from the centre scale with the relative
surface energies of the facets. This is the Wulff construction
({ref}`theory-wulff`). The choice of the atom at the centre and the
treatment of planes that pass through atomic layers determine which of the
many nearly identical cuts is obtained.

**Polarity.** Many facets of compound semiconductors are polar: a given
facet can end on a layer of cations or on a layer of anions, and the two
are chemically different. The builder decides which layer each facet
exposes from the bonding topology of the crystal ({ref}`theory-polarity`).

**Bonding.** Everything that follows depends on knowing which atoms are
bonded. The builder calibrates bond cut-offs on the bulk crystal, counts
coordination in an ionic, bipartite picture, classifies surface atoms as
terrace, edge or vertex atoms, and locates the lattice sites of missing
neighbours ({ref}`theory-bonding`).

**Neutrality.** A cut particle is almost never neutral. The builder removes
atoms the real particle could not retain and caps the surface with X-type
ligand placeholders until the formal charge vanishes
({ref}`theory-charge-passivation`). It places added ligands according to
explicit geometric rules derived from the bulk lattice
({ref}`theory-ligand-sites`).

**Reconstruction.** On zinc-blende particles, a neutral structure can still
be strongly polarised between cation- and anion-terminated {111} facets. An
optional reconstruction redistributes vacancies and ligands so that each
polar facet is compensated locally ({ref}`theory-reconstruction`).

**Composition.** The same machinery builds core/shell and Janus
heterostructures and twinned particles ({ref}`theory-heterostructures`).

**Identity.** Finally, size, centre and a fingerprint characterise each
structure so that libraries of models can be generated, deduplicated and
compared with relaxed geometries ({ref}`theory-identity`).

The result of these steps is a neutral, faceted particle with placeholder
ligands. The post-processing chapters (starting from {ref}`post-overview`)
describe how the placeholders can be replaced by real molecules and how the
surface can be modified further.

## Conventions

Lengths are in ångström. Formal charges are integers given in the recipe:
for CdSe, $q_\text{Cd} = +2$, $q_\text{Se} = -2$ and, for a chloride
placeholder, $q_\text{Cl} = -1$. The symbols $M$, $X$ and $L$ denote a
generic cation, native anion and anionic ligand. $a$ is the cubic lattice
constant; in zinc blende the cation–anion bond length is $b = a\sqrt3/4$ and
the nearest-neighbour distance within one sublattice is $d_{nn} = a/\sqrt2$.
Miller indices are written $(hkl)$ for a single oriented facet and $\{hkl\}$
for a family of symmetry-equivalent facets.
