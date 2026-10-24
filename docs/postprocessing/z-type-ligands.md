(post-z-type-ligands)=
# Z-type ligands

## Metal complexes on free anions

A Z-type ligand is a neutral two-electron acceptor: a metal complex such as
CdCl₂ or Cd(O₂CR)₂ that binds through its metal to a surface anion acting
as a Lewis base. On metal-rich particles these complexes cap the surface
chalcogen atoms, and they can be removed and re-adsorbed reversibly
{cite:p}`anderson2013`. The `z_type_ligands` treatment adds such complexes
to the anions that are left uncapped. It is the reverse of
{ref}`post-z-type`.

## Where a complex binds

A complex is bound to a *free core anion*, which must meet three
conditions:

1. It is a native surface anion whose coordination is below the bulk
   value. These are the anion sites of the L-type treatment
   ({ref}`post-l-type`), found with the same deeper surface shell.
2. No ligand atom lies within 3.5 Å of it. This is the test the X-type
   exchange applies to native anions ({ref}`post-x-type`).
3. The empty cation site on its missing bond is clear: no atom lies closer
   to it than $0.8\,d_{MX}$, where $d_{MX}$ is the bulk cation–anion bond
   length.

Each anion takes at most one complex. The metal is placed on that empty
cation site, i.e. on the CIF virtual site that an L-type donor would use.

## Building the complex on the lattice

The anions of the complex also go on lattice positions. The metal site is
mapped onto the CIF cell, and its first-shell anion sites are generated
from the bulk structure. A site already occupied by an atom counts as a
bond and is not filled again. Otherwise it is a candidate if no atom lies
closer to it than $0.8\,d_{MX}$. Candidates generated from neighbouring
metal sites coincide; two complexes that would use the same site compete
for it, and a site is given to one complex only. Each complex needs $n$
sites of its own. Among equally good choices, sites where X bridges to a
core cation (within $1.15\,d_{MX}$) or to another complex come first, then
the most outward ones.

## How many complexes fit

Full coverage is usually impossible. On an anion-terminated zinc-blende
{-1-1-1} facet the metal sites above the surface anions form a triangular
layer. The anion sites around them lie in the next layer, about one per
metal site, and each is shared by three metal sites. With $N$ free anions
on the facet, MX₂ units that each need two sites of their own satisfy

$$
2\,n_\text{units} \le N_\text{anion sites} \approx N
\quad\Rightarrow\quad n_\text{units} \le N/2 .
$$

Edge sites do better, because some of their anion sites hang over the
edge. On the Se-centred CdSe particle of two unit cells, 28 of the 42 free
Se atoms can receive a CdCl₂ unit. The bridging sites over core cations at
the edges are already held by the X-type ligands there, so all the Cl of
these complexes sit in the layer above the new metals.

## Distribution

Which sites receive a complex is set by `distribution`:

- `uniform` (default) *alternates* the sites first. In farthest-point
  order, a site is taken only if it shares no candidate anion site with a
  site already taken. On a {-1-1-1} facet this selects one of the three
  sublattices of the $\sqrt3\times\sqrt3$ pattern: every third surface
  anion, with the complexes 7.5 Å apart on CdSe and every one of them
  holding all its anion sites. Each takes the $n$ sites that the fewest
  other candidates need. The remaining sites are then *filled in*: each
  time the complex is added that leaves the most other candidates
  placeable, then the one whose anions bridge most, then the one farthest
  from the complexes already placed. The two stages together approach the
  bound above: 28 of 42 on the CdSe particle of two unit cells, and 97 of
  174 on the particle of four unit cells, where the exact maximum is 100.
  A partial ratio stops in the first stage while it can, so a third of the
  sites gives the alternating pattern exactly.
- `random` and `segmented` visit the sites in random order or as a compact
  patch, each taking its preferred free anion sites.

## Terminal anions

An anion bonded only to its own metal is terminal. On an anion-terminated
{111} facet some terminal anions cannot be avoided: each metal has three
anion sites in the layer above, so $U$ complexes form at most $3U$
metal–anion bonds, while making all $2U$ anions bridging would need $4U$.
When the choice costs no complex, sites where the anion bridges to a core
cation or to another complex are taken first; on the CdSe particle of two
unit cells about half the anions bridge. At an edge, a terminal anion sits
on a lattice anion site of the neighbouring facet. There it obeys the
same rules as that facet's ligands: a lattice position, clear of every
atom, at the anion-sublattice spacing from other ligands. Forbidding
terminal anions at edges altogether leaves only a few complexes on small
particles (4 instead of 28 on CdSe), so it is not done.

Molecular anions can still clash with each other when the plan is built
(the plan only reserves the anion sites); such complexes are skipped and
counted as crowded.

## Per facet family

```yaml
post_treatment:
  z_type_ligands:
    enabled: true
    passes:
      - complex: CdCl2
        targets: {"111B": 6, "111A": 2, "edge:100|111B": 1}
```

`targets` asks for a number of units per site family and replaces `ratio`
and `target_count`. A free anion belongs to a family through its site role
in the facet geometry ({ref}`facet-geometry`):

- a facet label (`"111B"`, or `"facet:111B"`);
- an exact edge or vertex role (`"edge:100|111B"`, `"vertex:100+100+111B"`);
- `"edge"` or `"vertex"` for all of them.

The free sites $s_f$ of a family are its free anions. Each family is planned
once over all of its free sites with the pass's `distribution`, and the
first $k_f$ units of that plan are placed, one at a time. A build with
$k_f + 1$ units therefore contains the one with $k_f$. A request above
$s_f$ is capped:

$$
k_f^{Z} = \min\left(s_f,\ k_f^\text{requested}\right) .
$$

Families are filled in the order given. A unit whose anion sites another
family has already taken uses the next free ones, or is skipped.

The ledger entry of the pass has `families`, with, per family:

| Key | Meaning |
|---|---|
| `free_sites` | $s_f$ |
| `crowded_sites` | anions of the family dropped by the 3.5 Å ligand rule |
| `blocked_sites` | anions whose cation site is not clear |
| `requested`, `capped`, `placed`, `skipped` | units asked for, after the cap, placed, and `requested - placed` |
| `skipped_crowded`, `skipped_no_anion_site`, `skipped_taken` | why units of the plan were not placed |

On the zinc-blende CdSe dots of the library preset, the 3.5 Å rule removes
no site. The free Se sit on the anion-terminated {111}B facets and under
Cd vacancies of {111}A, whose Se layer is 0.88 Å below the Cd and so within
the 1 Å role tolerance. {111}A facets with a complete Cd layer, and {100}
facets, have none: their Se is four-coordinated, the {100} surface being Cd
with bridging Cl after the prepass.

## The complex

The complex is written as an MXₙ formula (`CdCl2`, `ZnBr2`, `InCl3`) or as
an ionic SMILES with one atomic cation, e.g.
`[Cd+2].CC(=O)[O-].CC(=O)[O-]` for cadmium acetate. The fragment charges
must sum to zero,

$$
q_M + \sum_{k=1}^{n} q_{X_k} = 0,
$$

and other complexes are refused. Charges of atoms in a formula are taken
from `charges`, falling back to their common oxidation states.

A molecular anion is attached through the binding atom found for X-type
ligands. That atom sits on the anion site, with the body pointing along
the direction from the metal to the site. The molecule is then rotated
about this axis in 20° steps, and the pose with the largest distance to
its surroundings is kept.

## Charge

Every unit is neutral, so the charge of the particle does not change.
Molecular anions contain atoms (C, O, H) with no charge in the recipe. The
manifest ledger therefore records their molecular charge, and the final
charge counts each anion with its charge as a molecule.

## Order

The treatment runs after the X-type exchange and before `neutral_ligands`.
The metal of a complex is a native species with open coordination, so an
L-type pass that follows can bind a donor to it, giving for example
CdCl₂·NH₂R on the surface.

## Parameters

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | run the treatment |
| `seed` | 1337 | random seed |
| `ff` | `uff` | force field for molecular anions (`uff` or `mmff`) |
| `passes[].complex` | required | MXₙ formula or ionic SMILES (aliases `formula`, `smiles`) |
| `passes[].target_symbol` | any | native anion to bind to |
| `passes[].ratio` / `target_count` | 1.0 / 0 | fraction or number of free anions |
| `passes[].distribution` | `uniform` | `uniform`, `random` or `segmented` |
| `passes[].targets` | none | units per site family, see above |

The manifest contains a `z_type_ligand_ledger` with, for every pass, the
formula of the unit, the number of free anion sites, the units requested
and placed, how many anions bridge and how many are terminal, and how
many units could not be placed.

## Example

CdCl₂ on every free Se, spread uniformly, followed by methylamine on the
open cation sites:

```yaml
post_treatment:
  z_type_ligands:
    enabled: true
    passes:
      - complex: CdCl2
        distribution: uniform
        ratio: 1.0
  neutral_ligands:
    enabled: true
    passes:
      - {target: cation, smiles: CN, distribution: uniform, ratio: 1.0}
```
