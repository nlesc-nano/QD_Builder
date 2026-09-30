(theory-charge-passivation)=
# Charge balance and passivation

## Neutrality as the organising principle

A nanocrystal cut from a bulk lattice is almost never stoichiometric. The
surface exposes a different number of cations and anions depending on shape,
size and centre, so the naked particle carries a net formal charge
$Q = \sum_i q_i$ that can amount to tens of elementary charges. Real
colloidal particles are neutral. Their excess of surface metal is
compensated by anionic ligands such as carboxylates, halides or thiolates,
which in the covalent bond classification are one-electron X-type ligands
{cite:p}`green1995,owen2015`. The charge-orbital balance picture makes the
same point in electronic terms: a particle whose formal charge is balanced
by X-type ligands has, to first approximation, a clean gap
{cite:p}`voznyy2012`.

QD_Builder adopts this principle directly. The surface is capped with a
*placeholder* ligand, a single pseudo-atom of charge −1 (Cl by default),
and the particle is edited until $Q = 0$. The placeholder stands for any
X-type ligand. Once the structure is neutral, it can be exchanged for real
molecules (see {ref}`post-overview`), and the charge bookkeeping carries
over unchanged.

Four elementary moves change the charge, each by a known amount:

| Move | $\Delta Q$ | CdSe | InAs |
|---|---|---|---|
| swap a native anion for the ligand | $q_L - q_X$ | +1 | +2 |
| remove a cation | $-q_M$ | −2 | −3 |
| remove a native anion | $-q_X$ | +2 | +3 |
| add a ligand | $q_L$ | −1 | −1 |

The passivation algorithm is a sequence of such moves, chosen so that the
structure stays physically sensible at every step.

## The prepass: removing what cannot stay

Before the charge is addressed at all, the builder removes atoms that a
real particle could not retain. The Wulff cut can leave cations bonded to
only one or two anions, and anions hanging on a single bond. These are
removed or converted in a *prepass*.

Surface cations whose coordination falls below a threshold are removed. In
the `standard` mode the threshold is 3 everywhere. In the `role-aware` mode
it depends on the role of the atom (see {ref}`theory-bonding`), with
separate thresholds for terraces, edges and vertices
(`prepass_min_cn_terrace`, `_edge` and `_vertex`). The QDSpace presets use
3, 2 and 1. A vertex cation may thus keep a single bond, an edge cation two
and a terrace cation three. An edge that joins two *different* facet
families (for example {100} and {111}) is treated as terrace, because such
junctions are well-defined crystal edges rather than ragged corners.
Candidates are removed in batches, and the coordination is recomputed
after each batch.

The native anions of the surface are then examined once. An anion left
with a single bond is removed. An anion with two bonds is converted into
the ligand: it keeps its lattice position but becomes a monovalent
placeholder, which reflects that a two-coordinated chalcogen at a real
surface behaves like a terminal X-type site. After the prepass the facet
planes are tightened onto the remaining atoms.

## The main loop

The charge balance proceeds as a loop. In every iteration the builder
recomputes coordinations and surface memberships and then performs exactly
one action, taken from the first applicable priority.

**Priority 1: under-coordinated anions.** A surface anion with two or fewer
bonds is swapped for the ligand, provided that none of its cation
neighbours is left with three or fewer bonds.

**Priority 2: under-coordinated cations.** A surface cation below its
(role-dependent) threshold is removed. Among the candidates, vertices go
before edges and edges before terraces. Ties are broken by the distance
from sites edited earlier on the same facet, so that edits spread out,
then by depth and index. After every removal, ligands left without a host
are pruned, and surface anions that dropped below three bonds are swapped
for the ligand.

**Electrical step.** Once the structure is stable, the builder turns to the
charge.
- If $Q = 0$, the ligands are redistributed if this improves the
  coordination of the cations, and the loop ends.
- If $Q < 0$, the particle has too many anions. Surface anions with a
  coordination deficit are swapped for the ligand, one at a time, which
  raises the charge by $q_L - q_X$ per swap. The candidates are taken from
  the lowest coordination tier first and from vertices, edges and terraces
  in that order.
- If $Q > 0$, the particle has too much metal. The builder offers two
  strategies, selected with `--positive-q-mode`:
  - **`remove`** (the default) removes outer cations that are not bonded
    to a ligand, starting from the lowest coordination and the largest
    deficit. This is the stoichiometric route. If a removal overshoots to
    a negative charge, it is reverted and ligands are added instead.
  - **`add`** keeps every cation and adds ligands on the missing anion
    sites, one per unit of excess charge. This is the metal-rich route
    that corresponds to the X-type-passivated, cation-rich particles
    obtained in most syntheses {cite:p}`anderson2013`, and it is the mode
    used for the QDSpace II–VI, III–V and IV–VI series.

The placement of added ligands is itself a small optimisation problem and
is the subject of {ref}`theory-ligand-sites`.

## Redistribution at neutrality

A neutral structure can still have a poor ligand distribution: one cation
may carry two ligands while a neighbour is left with a dangling bond. At
neutrality the builder therefore migrates ligands. The fitness of the
structure is

$$
F = \sum_{\text{cations}} f_i,\qquad
f_i = \begin{cases}
0 & \text{CN}_i \ge T\\
-1 & \text{CN}_i = T - 1\\
-10 & \text{otherwise}
\end{cases}
$$

with $T$ the bulk coordination. A ligand whose hosts all remain at or above
$T - 1$ after its removal may move to a vacant bulk direction, or bridge
position, of a cation with $\text{CN} \le T - 2$. The move with the largest
gain is applied, and this is repeated while $F$ increases. Migration never
changes the charge.

## When neutrality cannot be reached

Not every particle can be made neutral with these moves. A tiny cluster
may have no removable cations left, or the surface may offer no free
sites. The loop never forces the charge. It stops with a `[halt]` message,
and the final charge is recorded as `total_charge` in the output manifest.
A cycle guard stops the loop if the same combination of charge, atom
count and ligand count recurs more than five times. The usual cause of
such a cycle is a ligand that is added and then pruned as an orphan, which
points to an inconsistent set of bond cut-offs.

An optional last resort is enabled with
`experimental: {exhausted_positive_q_fallback: true}`. When a small positive
charge remains, it tries removing a three- or four-coordinated surface
cation together with the associated anion swaps, and accepts the move if it
reduces $\lvert Q\rvert$ without crossing zero.

## Family-specific behaviour

The same algorithm serves all material families, but its consequences
differ.
- In zinc-blende **II–VI** compounds every structural move changes the
  charge in units of one or two, and exact neutrality is almost always
  reached. Added ligands may bridge several cations (see
  {ref}`theory-ligand-sites`).
- In zinc-blende **III–V** compounds an anion swap changes the charge by
  two, and ligands are placed on-top.
- In **rock-salt** compounds such as PbSe the cation coordination is six.
  The prepass thresholds, however, are absolute (a cation below three
  bonds is removed, an anion with two is converted), so a rock-salt corner
  may keep a lower coordination than its bulk value would suggest.
- In **perovskites** such as CsPbBr₃ both cations can host ligands. The
  {100}-terminated cubes used in the library are neutral after cation
  removal alone and need no added ligand {cite:p}`bodnarchuk2019`.

## Parameters

| Parameter | Default | Effect |
|---|---|---|
| `--positive-q-mode` | `remove` | strategy for $Q > 0$: `remove` or `add` |
| `passivation.prepass_mode` | `standard` | `standard` (threshold 3) or `role-aware` |
| `passivation.prepass_min_cn_terrace/_edge/_vertex` | 3 / 3 / 3 (vertex 1 when role-aware) | role thresholds |
| `passivation.surf_tol` | 1.0 Å | thickness of the surface shell |
| `passivation.include_sublayer` | `false` | also passivate sublayer cations |
| `passivation.ligand` | — | placeholder symbol, charge −1 unless given |
| `--prune-min-cn`, `--prune-passes` | 2, 10 | prune before facet detection |
| `experimental.exhausted_positive_q_fallback` | `false` | last-resort removal |

With `--write-all` the builder writes snapshots of the structure before and
after the prepass, and at the moment the structure is stable but not yet
neutral.
