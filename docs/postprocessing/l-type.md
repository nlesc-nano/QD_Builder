(post-l-type)=
# L-type neutral ligands

## Filling open coordination sites

Charge balance and coordination are separate requirements. A particle can
be exactly neutral and still expose metal atoms with empty coordination
sites. This happens on edges and vertices, and on facets whose cations
received fewer X-type ligands than they have dangling bonds. On real
particles such sites are occupied by neutral two-electron donors, typically
amines or phosphines present in the synthesis. These L-type ligands do not
change the formal charge, and their binding is often dynamic
{cite:p}`anderson2013`. The `neutral_ligands` treatment attaches such
molecules to the remaining open sites.

## Where a neutral ligand binds

The treatment looks for surface atoms of the chosen kind (`target: cation`,
`anion` or `both`, optionally restricted to one element with
`target_symbol`) whose coordination is still below the bulk value. The
surface shell is taken slightly deeper than for passivation, namely the
surface tolerance plus one bond length. The reason is that on
cation-terminated facets the under-coordinated cations can sit one bond
below the plane of the outermost ligands.

For every eligible atom, the missing bonds are located on the bulk lattice,
exactly as for X-type ligands (see {ref}`theory-bonding`). The empty site
of each missing neighbour becomes a candidate anchoring position, and sites
shared by several hosts are merged. Sites already occupied by another
ligand (within 0.85 Å) are discarded. At most one neutral ligand is placed
per host atom, taking sites of higher multiplicity first.

## Attaching the molecule

The binding atom of the molecule is found from its functional groups. For
amines, phosphines and thioethers it is the donor N, P or S atom, and the
molecule is oriented along the vector from that atom to the centroid of
its heavy atoms. If no donor group is recognised, the most electronegative
heavy atom is used. For acids the anchor is the acidic hydrogen, since the
molecule is attached in its neutral, protonated form.

The anchor is placed on the vacant lattice site, and the molecule is
oriented along the direction from the host to that site. The builder then
explores 18 rotations about this axis (20° steps) and five small outward
displacements of the anchor (0 to 0.6 Å). For every pose it evaluates the
smallest clearance between the ligand and its environment, with the host
excluded, using van der Waals radii in the default `vdw` mode. A term that
penalises tails pointing toward neighbouring ligand sites is subtracted,
and the best of the 90 poses is kept. Sites are processed in batches of
mutually distant positions, as in the X-type exchange.

## Charge

The molecule is neutral and is never ionised, so the formal charge of the
particle does not change and no compensation is needed.

## Parameters

The block can be placed under `post_treatment.neutral_ligands`. The older
location `passivation.neutral_ligands` is still read when the former is
absent.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | run the treatment |
| `seed` | 1337 | seed for conformers and random ordering |
| `ff` | `uff` | force field for conformers |
| `passes[].target` | required | `cation`, `anion` or `both` |
| `passes[].target_symbol` | none | restrict to one element |
| `passes[].smiles` | required | the neutral molecule |
| `passes[].ratio` / `target_count` | 1.0 / 0 | fraction or number of open sites |
| `passes[].distribution` | `random` | `random`, `uniform` or `segmented` |

In this treatment the number of sites is the rounded fraction of the
available hosts, which can be zero for very small ratios. The keys
`sterics_mode`, `refinement_passes` and `offset_out` are accepted, but only
the choice of force field and the pose scan above affect the placement.

## Example

Binding propylamine to 60 % of the open cation sites of a CdSe particle,
spread uniformly:

```yaml
post_treatment:
  neutral_ligands:
    enabled: true
    passes:
      - target: cation
        smiles: "CCCN"
        distribution: uniform
        ratio: 0.6
```
