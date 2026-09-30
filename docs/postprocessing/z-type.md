(post-z-type)=
# Z-type displacement

## Metal complexes on the surface

Metal-rich nanocrystals can be described as a stoichiometric core covered by
neutral metal complexes: CdCl₂ or Cd(O₂CR)₂ units bound through their
metal to surface anions, which act as Lewis bases. In the covalent bond
classification these complexes are Z-type ligands, i.e. two-electron
acceptors. Anderson and co-workers showed that L-type donors such as amines
and phosphines displace them reversibly from CdSe, so that the
stoichiometry of a particle is not a fixed property but depends on its
environment {cite:p}`anderson2013`. Displacement exposes the surface
anions that the complexes were capping. The two-coordinated chalcogen
atoms left behind are the origin of midgap trap states
{cite:p}`houtepen2017`.

The `z_type_displacement` treatment models this process by removing neutral
MXₙ units from the surface. It creates nothing in their place.

## Neutral units

A unit consists of one surface cation $M$ and $n$ anions $X$ such that the
unit is neutral,

$$
n = \frac{q_M}{\lvert q_X\rvert}.
$$

The builder derives $n$ from the charges unless it is given. It refuses
combinations that do not form an integer neutral unit. For example, Cd²⁺
with Cl⁻ gives CdCl₂ ($n = 2$), Pb²⁺ with Br⁻ gives PbBr₂, and Cs⁺ with
Br⁻ gives CsBr. The anion can be a placeholder ligand, which is the usual
case (CdCl₂ from a Cl-passivated particle), or a native anion, in which
case a stoichiometric MX unit is removed from the surface.

## Selecting the units

The cations must lie at the surface. Native anions must lie at the surface
too, while placeholder ligands are eligible wherever they are. When both
species are native, atoms already next to a ligand are excluded, so that an
intact surface ion pair is removed. The number of units that can be formed
is limited by the smaller of the number of cations and the number of anion
groups. The requested fraction or count is taken from it.

Cations are ordered according to the chosen distribution. For each cation
in turn the builder collects its nearest available anions, preferring those
within

$$
r_\text{search} = \max\bigl(6\ \text{Å},\ 2.5\,r_c(M, X)\bigr).
$$

It takes the $n$ closest, so that each unit is a compact MXₙ group. Anions
are consumed as units are formed, so no anion is counted twice.

## Cleaning up

Removing a cation can leave placeholder ligands that were bonded only to it
without a host. Such orphans are moved to the nearest free lattice site
that has a host, at least 1.5 Å from other ligands, preferring bridging
sites. Orphans that cannot be relocated are removed.

## Charge

Each unit is neutral by construction, so the displacement preserves the
charge of the particle, and no global rebalance is performed. If a
non-neutral `anion_count` is set explicitly, or orphan ligands have to be
deleted, the charge changes. The manifest ledger records these cases so
they can be recognised.

## Parameters

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | run the treatment |
| `seed` | 1337 | random seed |
| `passes[].cation` | required | cation of the unit |
| `passes[].anion` | required | anion or placeholder ligand of the unit |
| `passes[].anion_count` | $q_M/\lvert q_X\rvert$ | anions per unit |
| `passes[].ratio` / `target_count` | 1.0 / 0 | fraction or number of units |
| `passes[].distribution` | `random` | `random`, `uniform` or `segmented` |

The manifest contains a `z_type_displacement_ledger` with, for every pass,
the formula of the unit, the number of units and atoms removed, and the
fate of orphan ligands.

## Example

Removing 30 % of the CdCl₂ units, spread uniformly, from a Cl-passivated
CdSe particle. This recipe is illustrative:

```yaml
post_treatment:
  z_type_displacement:
    enabled: true
    passes:
      - cation: Cd
        anion: Cl          # CdCl2, anion_count derived as 2
        distribution: uniform
        ratio: 0.3
```
