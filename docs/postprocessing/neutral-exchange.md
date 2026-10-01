(post-neutral-exchange)=
# Neutral exchange

## Replacing a surface complex

Z-type displacement removes surface metal complexes. In many syntheses and
post-synthetic treatments, however, a complex is not simply removed but
replaced: CdCl₂ by ZnCl₂ or InBr₃, a lead halide unit by another salt, or a
metal complex by a zwitterionic or neutral molecule that binds the same
site. The `neutral_exchange` treatment combines the two steps. It removes
a neutral MXₙ unit, selected exactly as in {ref}`post-z-type`, and places a
replacement in the space it leaves. The treatment is formulated so that the
particle is neutral before and after.

Three kinds of replacement are supported, selected with `exchange_type`.

## Exchanging for another metal complex (`mxn`)

The replacement is given either as a formula, such as `InBr3` or `ZnBr2`,
or as an ionic SMILES that combines one metal ion with anionic fragments,
such as `[Zn+2].CC(=O)[O-].CC(=O)[O-]` for zinc acetate. The new metal atom
takes the position of the removed cation, and each anion fragment takes the
position of one removed anion. An atomic anion is placed as a single atom.
A molecular anion, such as acetate, is built with RDKit and placed with its
donor atom on the anion site and its tail pointing away from the metal.

The replacement need not have the same charge balance as the unit it
replaces. Exchanging CdCl₂ for InBr₃ on two anion sites leaves one bromide
short. The builder computes the residual charge of the pass,

$$
\Delta Q = Q_\text{after} - Q_\text{before} - \delta_\text{fragments},
$$

where $\delta_\text{fragments}$ corrects for the element charges of
molecular fragments. It then compensates. A negative residual is removed by
deleting anions, preferring the replacement's own anions and then
placeholder ligands. A positive residual is compensated with the
replacement's anion: first through the ligand-placement machinery, and if
necessary directly on free lattice sites at least 0.8 Å from any atom. In
the InBr₃ example one additional Br is added, and the particle contains
In, three Br and two Cl fewer, with its charge unchanged. When the
replacement contains only molecular anions and the residual is positive,
the treatment warns rather than guesses a compensation.

## Exchanging for a zwitterion (`zwitterion`)

A zwitterion carries a cationic and an anionic group in one neutral
molecule, for example an ammonium–carboxylate. It can replace a metal
complex by binding with its anionic end to the surface cation site while
its cationic end sits where the metal was. The builder identifies the
cationic anchor (a charged N or P) and the anionic anchor (a charged O or
S). It aligns the vector between them with the direction from the removed
cation to the removed anion, and places the cationic anchor on the former
cation site. It then rotates the molecule about this axis in 10° steps,
choosing the orientation with no hard clashes whose tail points most
nearly along the surface. The molecule is neutral, so no charge
compensation is needed.

## Exchanging for a neutral molecule (`l_type`)

A neutral donor can also take the place of the complex. The builder
identifies its donor atom, which is the hydroxyl oxygen of an acid, a thiol
sulfur, an alcohol oxygen, or an amine, phosphine or thioether donor. It
places the donor on the first vacated anion site, with the molecule
oriented along the local surface normal. When the donor carries a
hydrogen, the molecule is rotated so that this hydrogen points toward the
nearest remaining anion, forming a hydrogen bond with the surface. The
charge is unchanged.

## Failure and reproducibility

A pass whose placement fails leaves the structure unchanged and says so in
the log. The selection of units and all random choices are seeded from the
block `seed` with fixed offsets per pass.

## Parameters

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | run the treatment |
| `seed` | 1337 | random seed |
| `passes[].cation`, `anion`, `anion_count` | as in Z-type | the unit to replace |
| `passes[].exchange_type` | `mxn` | `mxn` (alias `salt`), `zwitterion` or `l_type` |
| `passes[].smiles` | required | formula or SMILES of the replacement |
| `passes[].ratio` / `target_count` | 1.0 / 0 | fraction or number of units |
| `passes[].distribution` | `random` | `random`, `uniform` or `segmented` |

## Example

Three passes on a CsPbBr₃ particle, each acting on 30 % of the surface CsBr
units. The first exchanges them for InBr₃, the second for a zwitterion and
the third for a neutral carboxylic acid:

```yaml
post_treatment:
  neutral_exchange:
    enabled: true
    passes:
      - {cation: Cs, anion: Br, anion_count: 1, ratio: 0.3,
         exchange_type: mxn,        smiles: "InBr3"}
      - {cation: Cs, anion: Br, anion_count: 1, ratio: 0.3,
         exchange_type: zwitterion, smiles: "CCCCC[NH2+]CCC(=O)[O-]"}
      - {cation: Cs, anion: Br, anion_count: 1, ratio: 0.3,
         exchange_type: l_type,     smiles: "CCCCC(=O)O"}
```
