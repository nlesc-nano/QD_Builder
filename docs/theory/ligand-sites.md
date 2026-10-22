(theory-ligand-sites)=
# Where ligands go

When a metal-rich particle is made neutral by adding ligands, the builder
must decide where each ligand goes. The choice is not cosmetic. A ligand
placed too close to a second cation creates a hidden bond that pushes that
cation above its bulk coordination. Two ligands placed too close together
form an unphysical contact. Ligands that cluster on one facet leave the
others bare. And a greedy choice of bridging sites can use up the available
hosts before the charge is balanced. QD_Builder addresses these problems
with a single site selector, `select_ligand_sites`. It is used both by the
ordinary passivation and by the {111} reconstruction (see
{ref}`theory-reconstruction`), so the same rules hold wherever ligands are
added.

## Candidate sites

The natural positions for an anionic ligand are the vacant anion sites of
the crystal lattice, the virtual sites introduced in
{ref}`theory-bonding`. A site shared by $\mu$ under-coordinated cations
is a $\mu$-bridging position: on-top ($\mu 1$), bridging ($\mu 2$) or
capping ($\mu 3$). A site is a candidate only if all of its hosts still
have a coordination deficit.

On the cation-terminated {111} facets of zinc-blende II–VI particles the
lattice offers no bridging positions, because every surface cation has a
single dangling bond that points straight out. Real II–VI surfaces
nevertheless bind halides and carboxylates in bridging modes, and the
builder generates two further kinds of site on these facets. The hosts are
outer cations with exactly one missing bond. Two such cations are adjacent
when they are closer than $1.15\,d_{nn}$, with $d_{nn} = a/\sqrt2$, and
their facet normals agree.

A **μ3 hollow** is placed above every triangle of mutually adjacent hosts,
at the point that is one bond length $b = a\sqrt3/4$ from all three,

$$
\mathbf x = \mathbf c + \sqrt{b^2 - \lvert\mathbf x_a - \mathbf c\rvert^2}\;\hat n ,
$$

where $\mathbf c$ is the centroid of the triangle. For an ideal triangle of
side $d_{nn}$ the height above the cation plane is $b/3$.

A **μ2 bridge** is placed above the midpoint $\mathbf m$ of an adjacent
pair, at bond length from both hosts, and tilted within the plane spanned
by the normal and the direction perpendicular to the pair,

$$
\mathbf x(\theta) = \mathbf m + h\,(\cos\theta\,\hat n + \sin\theta\,\hat t),
\qquad \theta \in [-60°, 60°].
$$

The tilt is scanned in 5° steps and the one with the largest clearance from
other atoms is kept. On flat {111} terraces no tilt clears both the anion
beneath and a third cation, so terraces receive μ3 and μ1 ligands, while
μ2 bridges appear at steps and edges.

## Geometric validity

In zinc-blende particles every candidate must satisfy three distance
conditions against atoms that are not its hosts. Each is derived from the
bulk lattice.

The ligand must stay at least $1.1\,b$ from any anion, which excludes
positions that would crowd the anion sublattice.

It must stay at least

$$
d_\text{lig} = 0.95\,\frac{a}{\sqrt2}
$$

from any other ligand, i.e. 95 % of the nearest-neighbour distance of the
fcc anion sublattice. Ligands on adjacent lattice sites are thus allowed,
while closer contacts are not.

It must stay at least the *shell separator* from any cation it is not
bonded to. Relative to a cation in zinc blende, the anions lie at
$\tfrac a4(n_1, n_2, n_3)$ with odd $n_i$. The first shell,
$n^2 = 3$, holds four anions at $b = a\sqrt3/4$. The second,
$n^2 = 11$, holds twelve at $a\sqrt{11}/4 = b\sqrt{11/3}$. The separator is
placed halfway between them,

$$
d_\text{cat} = \tfrac12\,b\,\Bigl(1 + \sqrt{11/3}\Bigr) \approx 1.457\,b ,
$$

which is 3.88 Å for CdSe. A ligand closer than $d_\text{cat}$ to a cation
would be, in effect, bonded to it. The condition therefore guarantees that
no added ligand silently raises a neighbouring cation above four-fold
coordination. A genuine lattice site sits $1.915\,b$ from non-host cations
and always passes.

For other structures only the ligand–ligand distance is enforced, with a
minimum of 3.0 Å (1.8 Å when sublayer hosts are included).

## Choosing among valid sites

The selector picks one ligand at a time until the required number,
$n = \lfloor Q/\lvert q_L\rvert\rfloor$, has been placed. Four rules decide
each pick, in order.

**Capacity.** A host accepts at most as many ligands as it has missing
bonds.

**Reachability.** A bridging site consumes one bond on each of its hosts.
Taken greedily, bridges can exhaust the hosts before all $n$ ligands are
placed. A bridge is therefore eligible only if the capacity that remains on
hosts with on-top sites still suffices for every ligand still needed:

$$
\sum_{h\in H_{\mu1}} \max\bigl(0,\ \text{left}_h - [h \in \text{hosts}]\bigr)
\;\ge\; n - n_\text{placed} - 1 .
$$

This budget keeps the target charge reachable while still preferring
bridges.

**Priority.** Candidates are ranked by the largest remaining deficit among
their hosts, so the lowest-coordinated cations are served first. Next come
candidates that do not over-coordinate any cation, and then the site type.
II–VI particles prefer μ3 over μ2 over μ1, which reflects the bridging
modes observed for halides and carboxylates on these surfaces. III–V
particles prefer on-top μ1 sites, and other materials are ranked by the net
number of bonds repaired.

**Uniformity.** Ligands are balanced across facets. Each candidate is
assigned to the facet plane it sits furthest outside of; this is not
necessarily the facet of its host, which may lie on an edge shared by two
facets. Only candidates on the facet with the fewest added ligands so far
are considered. Within that facet the builder takes the site farthest from
the ligands already added there, then the one with the least crowding
$\sum_j d_j^{-3}$, then the one farthest from all ligands (and from
positions to be avoided, such as cations just removed by the
reconstruction), and finally the lowest host index. The result is
deterministic and spreads the ligands evenly over and within the facets.

## A fallback for unusual structures

If the reference lattice cannot be registered, or the selector finds no
valid site, the builder falls back to an older tiered procedure. It groups
hosts by deficit, scores candidate positions by the coordination they
repair, and spreads them by farthest-point sampling with a 3 Å exclusion
radius. This path does not apply the zinc-blende distance conditions and
is not reached for the materials in the QDSpace library.
