(post-x-type)=
# X-type ligand exchange

## The exchange as a substitution

X-type exchange is the most common surface reaction of colloidal
nanocrystals. An anionic ligand bound to a surface cation is replaced by
another, for example chloride by a carboxylate or a thiolate. In the
builder's model the particle is already neutral, and its X-type ligands
are single placeholder atoms on anion lattice sites. The exchange replaces
each selected placeholder by a real molecule of the same charge, so that
neutrality is preserved and only the chemistry of the shell changes.

The same mechanism extends to two less obvious substitutions. A *native*
surface anion can be exchanged, for instance a two-coordinated Se²⁻ by a
thiolate. A native surface cation can be exchanged by a cationic molecule,
for instance Cd²⁺ by an alkylammonium ion. In these cases the charge of the
replaced atom differs from that of the molecule, and the builder restores
neutrality afterwards.

## Preparing the ligand

Every pass names the atom to replace (`replace`) and one or more ligands as
SMILES. The charge of the ligand is either given explicitly (`charge: -1`
or `+1`) or detected from its functional groups. The exchange is
formulated for monovalent ligands.

An **anionic ligand** is built from its neutral acid. The builder searches,
in order, for a carboxylic acid, a phosphonic acid, a sulfonic acid, a
thiol and an alcohol. The first match defines the binding group. The acidic
proton is removed, the donor atom receives a negative formal charge, and
the anion is re-embedded and re-optimised. The binding group defines the
donor atom $d_1$ that will occupy the vacated site and, for oxyanions, a
second oxygen $d_2$ that can bind a neighbouring cation. A carboxylate can
thus bridge two metal atoms.

A **cationic ligand** is built from a donor. A pre-charged ammonium or
phosphonium group is used as it is. Otherwise the first amine, phosphine or
thioether is protonated, or, if it carries no hydrogen, methylated, which
turns a tertiary amine into a quaternary ammonium.

The direction of the ligand's tail, $\hat v_\text{tail}$, is defined as the
unit vector from the binding group to the centroid of the remaining heavy
atoms. It is used to orient the molecule away from the surface.

## Choosing the sites

Candidate sites are the atoms of the requested symbol and charge. For
placeholders every such atom is a candidate. For native atoms, only surface
atoms that are under-coordinated, and not already next to a ligand, are
eligible, which excludes the interior of the particle. Each candidate must
be bonded to at least one host of opposite charge. The selection follows
the `ratio`/`target_count` and `distribution` conventions described in
{ref}`post-overview`. When several SMILES are given, they are assigned to
the selected sites in turn, which produces mixed ligand shells.

## Placing the molecule

The selected atoms are removed, and each molecule is placed with its donor
atom $d_1$ exactly on the vacated position. The tail is aligned with the
local outward normal $\hat n_\text{surf}$, obtained from the facets the
site belongs to. The molecule is then rotated about this axis in 10° steps,
and each of the 36 orientations is scored. The score starts from the
smallest clearance between ligand atoms and their environment, where two
atoms overlap when they are closer than the sum of their van der Waals
radii minus 0.4 Å. Overlaps are penalised quadratically, a thousand times
more strongly against the particle than against other ligands. Contacts
between the donor atoms and their metal partners are exempt. A
bidentate-binding term rewards orientations in which $d_2$ lies near
2.4 Å from a second cation,

$$
S_\text{bi} = 2\exp\!\Bigl[-\frac{(d - 2.4)^2}{2\cdot 0.6^2}\Bigr],
$$

and a repulsion term discourages tails that point toward neighbouring
exchange sites. The best orientation is kept.

To keep the procedure efficient without losing steric realism, the sites
are processed in batches of mutually distant sites (at least 10 Å apart).
Ligands of one batch are placed against the particle and the ligands of all
previous batches.

## Restoring neutrality

Ligand atoms are written as ordinary elements, and the `charges` block of a
recipe may assign charges to elements such as O or S for other purposes.
The builder therefore tracks the charge of the exchanged ligands
explicitly. The effective charge is

$$
Q = \sum_i q(s_i) \;-\; \sum_\text{ligands} \sum_{a\in\text{ligand}} q(s_a) \;+\; \sum_\text{ligands} q_\text{ligand},
$$

where the element charges of ligand atoms are replaced by the formal
charge of each molecule. If $Q \ne 0$, placeholder ligands are adjusted
while the exchanged molecules stay in place. For $Q < 0$,
$\lceil -Q/\lvert q_L\rvert\rceil$ placeholders are removed. For $Q > 0$,
placeholders are added on free lattice sites with the site selector of
{ref}`theory-ligand-sites`. This compensation deliberately does not rerun
the structural passivation.

The three kinds of exchange differ in their charge balance:

| Exchange | $\Delta Q$ per exchange | Compensation |
|---|---|---|
| Cl⁻ → carboxylate⁻ | 0 | none |
| Se²⁻ → thiolate⁻ | +1 | one placeholder added |
| Cd²⁺ → alkylammonium⁺ | −1 | one placeholder removed |

## Parameters

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | run the exchange |
| `seed` | 1337 | random seed |
| `ff` | `uff` | force field for conformers (`uff` or `mmff`) |
| `passes[].replace` | required | symbol of the atom to replace |
| `passes[].smiles` | required | one SMILES or a list |
| `passes[].charge` | detected | molecular charge, ±1 |
| `passes[].replace_charge` | from `charges` | charge of the replaced atom |
| `passes[].ratio` / `target_count` | 1.0 / 0 | fraction or number of sites |
| `passes[].distribution` | `random` | `random`, `uniform` or `segmented` |

The keys `sterics_mode` and `refinement_passes` are accepted but do not
currently affect the placement.

The manifest records every exchanged ligand in
`ligand_exchange_charge_ledger`, together with the element charge, the
correction and the resulting `total_charge`.

## Examples

Replacing 30 % of the chloride placeholders on an InP particle by
pentanoate:

```yaml
post_treatment:
  ligand_exchange:
    enabled: true
    passes:
      - replace: Cl
        charge: -1
        smiles: "CCCCC(=O)O"     # deprotonated to the carboxylate
        distribution: random
        ratio: 0.3
```

Exchanging native surface Se of CdSe for propanethiolate, spread evenly:

```yaml
      - replace: Se
        charge: -1
        smiles: "CCCS"
        distribution: uniform
        ratio: 0.4
```

Exchanging surface Cd for propylammonium:

```yaml
      - replace: Cd
        charge: 1
        smiles: "CCCN"
        distribution: uniform
        ratio: 0.29
```

On CsPbBr₃ the native bromide can be exchanged directly, with the charge of
the ligand detected from the acid group:

```yaml
      - replace: Br
        smiles: "CCC(=O)O"
        ratio: 0.6
```
