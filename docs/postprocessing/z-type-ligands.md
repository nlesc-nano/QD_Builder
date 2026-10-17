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
closer to it than $0.8\,d_{MX}$. The candidates are ranked by the number
of cations within $1.15\,d_{MX}$, so that sites on the missing bonds of
surface cations, where X bridges the new metal and the core, come first.
Ties are broken by the most outward direction. The first $n$ candidates
receive the anions. If fewer than $n$ sites are free, the unit is skipped.

Complexes are placed one at a time in the order set by `distribution`, and
each sees the ones placed before it. Full coverage is therefore often
impossible. On an anion-terminated zinc-blende {-1-1-1} facet the metal
sites above the surface anions form a complete cation layer. The anion
sites around them lie in the next layer, and each of them is shared by
three metal sites. A complete layer can hold one X per metal, while MX₂
needs two. On the Se-centred CdSe particle of two unit cells, about 25 of
the 42 free Se atoms receive a CdCl₂ unit. The anion sites of these facets
lie above the new metal layer, so X bridges to core cations only at the
edges with the cation-terminated {100} and {111} facets.

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
| `passes[].distribution` | `random` | `random`, `uniform` or `segmented` |

The manifest contains a `z_type_ligand_ledger` with, for every pass, the
formula of the unit, the number of free anion sites, the units requested
and placed, how many anions bridge to surface cations, and how many sites
were skipped.

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
