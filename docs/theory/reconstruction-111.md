(theory-reconstruction)=
# The polar {111} reconstruction of zinc-blende particles

## Why polar facets must reconstruct

In a zinc-blende crystal the atomic planes perpendicular to a ⟨111⟩ direction
contain only one species, and cation and anion planes alternate. A particle
cut along these planes therefore exposes two chemically distinct kinds of
{111} facet. On one side the outermost layer consists entirely of metal
atoms, each with a single dangling bond pointing into the solvent; this is
the *cation-rich* (111) facet. On the opposite side the outermost layer is
made of chalcogen or pnictogen atoms with one dangling bond each; this is
the *anion-rich* ($\bar1\bar1\bar1$) facet. Along every ⟨111⟩ axis the crystal
behaves as a stack of charged sheets, and a slab terminated this way belongs
to Tasker's type-3 class of polar surfaces, whose electrostatic energy grows
with thickness unless the surface compensates the dipole
{cite:p}`tasker1979`.

The builder expresses the same problem in terms of formal charge. The
ordinary passivation (see {ref}`theory-charge-passivation`) makes the
particle neutral by attaching X-type ligands, but it does so site by site:
metal-rich facets receive ligands until their cations reach bulk
coordination, while the anion-rich facets are left untouched. The result is
a neutral particle whose charge is distributed very unevenly. The cation
facets carry a dense ligand layer, and the anion facets expose rows of
two-fold-coordinated chalcogen atoms whose lone pairs are exactly the
species that produce midgap states in II–VI nanocrystals
{cite:p}`houtepen2017`. The reconstruction described here removes that
imbalance. It rebuilds both polar facets so that each is locally charge
compensated, without changing the total charge of the particle.

Because it rearranges an already passivated particle, the reconstruction is
implemented as a post-treatment (`post_treatment.surface_reconstruction`),
and it runs immediately after the main charge balance.

## When the reconstruction applies

The model is formulated for binary zinc-blende compounds, and the builder
checks this before touching the structure. The bulk crystal must contain
exactly one cationic and one anionic species (the reconstruction ligand
excluded). The lattice must be cubic to within $10^{-3}$ Å and $10^{-2}$°.
Both species must be four-fold coordinated in the bulk, where the
coordination is counted from distinct opposite-charge neighbours within 1.2
times the shortest interatomic distance. The recipe must also ask for both
polar terminations: at least one {111} seed with `termination: cation_rich`
and at least one with `termination: anion_rich`. Finally, the cut particle
must actually expose at least one facet of each kind. When any of these
conditions fails the step is skipped, the structure is returned unchanged,
and the reason is written to the ledger. The reconstruction ligand must
carry a negative charge; if it is absent from the `charges` block it is
registered as a monovalent anion.

### Wurtzite

The same step runs on binary wurtzite cells (hexagonal, $a = b$,
$\gamma = 120°$, both species four-fold coordinated). Perpendicular to $c$
the wurtzite stacking has the same single-species, close-packed layers as
zinc blende perpendicular to ⟨111⟩, with one bond per atom along the normal,
so the (001) and ($00\bar1$) facets are the counterparts of (111) and
($\bar1\bar1\bar1$). The polar directions are then $(0\,0\,\pm1)$ instead of the
eight ⟨111⟩, the recipe must carry an (001) seed with `termination:
cation_rich` and one with `termination: anion_rich`, the in-plane cation
lattice is spanned by $\mathbf a$ and $\mathbf a + \mathbf b$ (cation-cation
distance $a$), the bond length is the shortest cation-anion distance of the
cell, and the cluster rotations are the proper rotations of the hexagonal
lattice. Everything below (vacancy pattern, ligand conversion, cation
removal and ligand addition) is unchanged. The {100} prism facets are
non-polar and are not touched.

## Reading polarity from the structure

Polarity is determined from the atoms the particle actually exposes, never
from the sign of a Miller index. For each of the eight directions
$\{\pm1,\pm1,\pm1\}$ the builder forms the Cartesian normal
$\hat n = (hkl)\,G^*/\lVert (hkl)\,G^*\rVert$ from the reciprocal lattice
$G^*$. It projects every native atom (cations and anions, not ligands) onto
$\hat n$ and collects the atoms whose projection lies within
$\ell = 0.4$ Å of the maximum. The value of $\ell$ is half the thickness of
one atomic layer. If this outer layer contains a single species, the
direction is a polar facet whose kind is that species. A facet may consist
of a single atom, since a single anion capping three cations is still an
anion-terminated ($\bar1\bar1\bar1$) apex. Directions whose outer slab mixes
both species are not polar facets and play no part in what follows.

All geometric thresholds are derived from the bulk lattice constant $a$.
The cation–anion bond length is $b = a\sqrt3/4$. Nearest neighbours of the
same species (both sublattices are fcc) are separated by
$d_{nn} = a/\sqrt2$. Two atoms are treated as nearest neighbours of the
same species when they are closer than $1.15\,d_{nn}$. Bonds between
opposite charges are taken from the calibrated pair cut-offs of the bulk
crystal (see {ref}`theory-bonding`). Only pairs of opposite formal charge
are bonded, so a ligand bonds to cations but never to anions.

## Stage 1: vacancies beneath the anion facets

On an anion-terminated facet every outer anion has three bonds into the
particle and one dangling bond. The builder creates cation vacancies in the
layer directly below. Removing a cation lowers the coordination of the
three outer anions it was bonded to from three to two. These now
two-coordinated anions are then converted into the monovalent
reconstruction ligand, for example $\text{Se}^{2-} \rightarrow \text{Cl}^-$.
For a single vacancy the formal charge changes by

$$
\Delta Q_\text{vac} = -q_\text{cat} + n_\text{conv}\,(q_\text{lig} - q_\text{an}),
$$

with $n_\text{conv} = 3$ converted anions. In a II–VI compound
($q_\text{cat} = 2$, $q_\text{an} = -2$, $q_\text{lig} = -1$) each vacancy
therefore adds $+1$. In a III–V compound it adds $-3 + 3\cdot2 = +3$.
The anion facet thus trades negative charge for positive charge, which is
later removed on the cation side.

Not every sub-surface cation may be removed. A candidate must be fully
coordinated (CN 4), bonded only to native anions, and must not belong to,
or neighbour, the outer layer of any other polar facet. This keeps
vacancies away from facet edges. In addition, every anion around the
candidate must retain at least two bonds after the removal. The builder
counts coordination explicitly rather than assuming three conversions per
vacancy. The count of three emerges from the geometry: two cations that
share an anion are exactly $d_{nn}$ apart, so vacancies that are never
nearest neighbours of one another never share an anion.

### Choosing a uniform pattern

The vacancies must not be adjacent, and within that constraint they should
be as numerous and as evenly spread as possible. The candidates of one
facet lie on a two-dimensional triangular lattice. The builder finds two
in-plane fcc vectors $\mathbf a_1, \mathbf a_2$ at 60°, assigns every
candidate integer coordinates $(i, j)$ by a least-squares fit of
$\mathbf r - \mathbf r_0 = i\,\mathbf a_1 + j\,\mathbf a_2$, and colours it
with $(i - j) \bmod 3$. Each colour class is a perfect
$\sqrt3\times\sqrt3$ sublattice at one-third coverage in which no two sites
are nearest neighbours, which is the densest uniform non-adjacent pattern
the lattice admits. The largest colour class is the default choice. An
exact maximum independent set is also computed, by branch and bound for up
to 64 candidates and greedily beyond that. It replaces the colour class
only when it is strictly larger, which happens on irregular patches where
the three-colouring is not optimal.

The number of vacancies is the same on every anion facet of the particle.
It is capped at the smallest facet's maximum, $n_\text{vac,max}$, so that
equivalent facets are reconstructed equivalently.

### Using the particle's symmetry

A particle cut from a Wulff construction usually retains a large part of
the cubic point group, and the reconstruction tries to preserve it. The
builder determines which of the 24 proper cubic rotations map the ionic
framework of the particle onto itself within 0.3 Å. Cations form one class,
and native anions together with ligands that bridge at least two cations
form the other. Terminal ligands are ignored, so the test does not depend
on where the charge balance happened to place them. When several anion
facets exist, the pattern is chosen once on a reference facet and copied
to every other facet through these rotations. It is kept only if every
facet admits the mapped sites, and among the possible reference facets the
one that yields the most sites wins. If no consistent mapping exists, each
facet is treated independently.

## Stage 2: breaking runs of under-coordinated anions

After the conversions, anions that survive on the facet can still form
continuous rows of three-fold-coordinated atoms, on the facet itself or
across its edges with neighbouring facets. The builder breaks these rows
with the smallest possible number of additional anion-to-ligand swaps.

The problem is posed as a minimum vertex cover. The nodes are the surviving
outer anions of the facet (the *anchors*). They are joined by any
under-coordinated (CN < 4) anion within $1.15\,d_{nn}$ of an anchor, which
in practice means the edge rows of the neighbouring facets. An edge joins
two nodes that are nearest neighbours, provided at least one of them is an
anchor; contacts between two edge atoms are not the facet's concern.
Converting a node to ligand removes all edges incident to it, so a vertex
cover is exactly a set of swaps after which no two under-coordinated anions
of the facet touch.

For connected components of up to 20 nodes the cover is found exactly, by
enumerating subsets of increasing size. Among covers of minimal size the
builder prefers those whose converted atoms are themselves not adjacent,
which yields the alternation Se–Se–Se → Se–Cl–Se rather than two
neighbouring swaps. It then prefers converting atoms that belong to the
facet over atoms on the neighbouring edge. Larger components fall back to a
greedy highest-degree cover. As with the vacancies, a symmetric solution is
attempted first and accepted only if it leaves every facet clean.

Very small facets follow from the same rule. A facet of three mutually
adjacent anions is a triangle whose minimum cover has two vertices, so two
of the three anions become ligands. A facet of a single anion, an apex
bonded to three cations, is converted to the ligand directly. After both
stages the charge accumulated on the anion side is

$$
\Delta Q_\text{an} = -q_\text{cat}\,N_\text{vac}
 + (q_\text{lig} - q_\text{an})\,(N_\text{conv} + N_\text{break}).
$$

## Stage 3: the cation facets

The cation-terminated facets are first returned to their bare state. Every
ligand whose host cations all lie in the outer layer of a cation facet, and
which sits at least $\ell$ above that layer, is a candidate for removal.
This covers on-top, bridging and hollow ligands alike. Ligands that occupy
lattice sites in the layer below are kept. The removal is guarded so that
no host falls below terrace coordination, $\text{CN}_\text{bulk} - 1 = 3$.
In practice this means that terrace cations lose their ligand, while edge
and vertex cations that miss two bonds keep one. Each stripped ligand
raises the charge by $|q_\text{lig}|$.

The positive charge now carried by the particle is compensated by removing
outer cations. At most

$$
n_\text{remove} = \left\lfloor Q / q_\text{cat} \right\rfloor
$$

cations can be removed. The policy `cation_removal` decides how many
actually are. Under `mirror`, the default (`auto` resolves to it for all
families), the number is capped at the number of vacancies created on the
anion side, $\min(n_\text{remove}, N_\text{vac})$. Each vacancy below an
anion facet is thus mirrored by a missing cation on a cation facet, and the
remaining charge is left to ligands. Under `max`, as many cations are
removed as the charge allows.

Removed cations are chosen by the same principles as the vacancies. They
must lie on a cation facet, away from facet edges, and their removal must
not take any neighbouring anion below three bonds, so that no new
anion-to-ligand conversion is triggered on this side. Cations without
ligand neighbours are preferred. The per-facet quota
$\lfloor n_\text{remove}/N_\text{facets}\rfloor$ is drawn from the densest
colour class by farthest-point sampling, mapped by symmetry when possible,
and any remainder is added one cation at a time to the facet with the
fewest removals so far.

## Stage 4: compensating the remainder with ligands

Whatever positive charge remains, $n_\text{add} = \lfloor Q/|q_\text{lig}|\rfloor$,
is compensated by adding ligands on the cation facets. The hosts are the
outer cations of those facets that are below bulk coordination, each able
to accept $4 - \text{CN}$ ligands. The sites are chosen by the same
selector that the builder uses for ordinary passivation, described in
{ref}`theory-ligand-sites`. In brief, II–VI particles prefer three-fold
hollow sites, then two-fold bridges, then on-top sites, while III–V
particles use on-top sites only. No cation is taken above four-fold
coordination. Every ligand keeps bulk-derived distances from anions, from
other ligands and from non-host cations. The ligands are balanced across
facets and spread within each one, and they are kept away from the
positions of the cations just removed.

The charge of one reconstruction attempt can be followed through the four
stages:

$$
\begin{aligned}
Q_1 &= Q_0 + \Delta Q_\text{an},\\
Q_2 &= Q_1 - q_\text{lig}\,N_\text{strip},\\
Q_3 &= Q_2 - q_\text{cat}\,N_\text{rm},\\
Q_\text{final} &= Q_3 + q_\text{lig}\,N_\text{add}.
\end{aligned}
$$

## Closing the charge balance

The four stages do not always reach $Q_\text{final} = 0$. The floor
divisions leave remainders, the mirror policy caps the number of removed
cations, and small facets may offer fewer ligand sites than needed. The
builder therefore treats the number of vacancies as the free parameter. It
repeats the whole reconstruction for
$N_\text{vac} = n_\text{vac,max}, n_\text{vac,max} - 1, \dots, 0$. It keeps
the attempt with the smallest $|Q_\text{final}|$, preferring more vacancies
on ties, and stops at the first attempt that is exactly neutral.

If even the best attempt leaves a residual charge, it is handed to the
general charge-balance routine with ligand addition enabled and the
coordination prepass disabled. This routine may also use the {100} facets
and the edges. A charge that cannot be removed even then is reported as a
warning and kept, rather than forced away by an arbitrary change to the
structure.

## Determinism

Every choice in the reconstruction is deterministic. Ties are broken
lexicographically by rounded distances and atom indices, the optimisation
problems are solved exactly where their size allows, and all loops iterate
in a fixed order. The same particle and recipe always produce the same
reconstructed structure. The `seed` key is accepted for compatibility but
does not influence the result.

## The reconstruction ledger

Every run writes a `surface_reconstruction_ledger` into the output
manifest, which records what was done and why.

| Key | Meaning |
|---|---|
| `status` | `applied`, or `skipped` together with a `reason` |
| `ligand` | reconstruction ligand |
| `cation`, `anion` | the binary pair |
| `cation_removal` | resolved policy (`mirror` or `max`) |
| `anion_facets[]` | per facet: `hkl`, `vacancies`, `anions_to_ligand`, `chain_breaks` |
| `cation_facets[]` | per facet: `hkl`, `ligands_stripped`, `cations_removed`, `ligands_added` |
| `anion_side_charge_delta` | $\Delta Q_\text{an}$ |
| `ligands_stripped`, `cations_removed`, `ligands_added` | totals over the cation facets |
| `ligands_added_by_mu` | added ligands by site type (`mu3`, `mu2`, `mu1`) |
| `symmetric` | whether the vacancy and cation-removal patterns were mapped by symmetry |
| `total_charge_before`, `total_charge_after` | $Q_0$ and the final charge |
| `charge_balance_fallback` | present only if the fallback ran: residual charge and what it changed |

## Parameters

The reconstruction is controlled by the `post_treatment.surface_reconstruction`
block of the recipe.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | run the reconstruction |
| `ligand` | passivation ligand | reconstruction ligand (must be negatively charged) |
| `cation_removal` | `auto` | `mirror` (= `auto`) or `max` |
| `seed` | `1337` | accepted, currently unused |

The older keys `facets`, `target_reduction`, `min_separation` and
`distribution` are still parsed but no longer have any effect. A note is
printed when they are set to non-default values. The geometric constants
($\ell = 0.4$ Å, the 1.15 neighbour factor, the 0.3 Å symmetry tolerance and
the ligand spacing factor 0.95) are fixed in
`builder.facet_reconstruction`.

```yaml
facets:
  - {hkl: "100",    scope: family, termination: cation_rich, gamma: 1.0}
  - {hkl: "111",    scope: family, termination: cation_rich, gamma: 1.0}
  - {hkl: "-1-1-1", scope: family, termination: anion_rich,  gamma: 1.0}
post_treatment:
  surface_reconstruction:
    enabled: true
    ligand: Cl
```
