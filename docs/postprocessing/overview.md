(post-overview)=
# Post-processing: modifying the surface

## From placeholders to chemistry

The construction described in the theory chapters ends with a neutral
particle whose surface is capped by placeholder ligands: single pseudo-atoms
such as Cl that carry the charge of an X-type ligand and occupy the lattice
sites where anionic ligands bind. This model is complete from the point of
view of charge and coordination, and it is often exactly what is needed
for electronic-structure calculations on halide-capped particles. Real
samples, however, are capped by carboxylates, thiolates, amines and
phosphonates, contain surface metal complexes that can be removed or
exchanged, and may be alloyed. The post-treatments of QD_Builder bring the
model closer to this chemistry. Each operates on the neutral, passivated
particle, and each preserves or restores neutrality.

The treatments are organised along the covalent bond classification of
ligands {cite:p}`green1995`, which has become the standard language of
nanocrystal surface chemistry {cite:p}`owen2015,anderson2013`:

- **X-type ligands** are one-electron ligands, anionic when counted as ions,
  such as carboxylates, halides and thiolates. They balance the charge of
  surface cations. *Ligand exchange* ({ref}`post-x-type`) replaces
  placeholders, or native surface ions, by real charged molecules.
- **L-type ligands** are neutral two-electron donors such as amines and
  phosphines. They bind to under-coordinated metal atoms without changing
  the charge. *Neutral ligands* ({ref}`post-l-type`) attaches them to open
  coordination sites.
- **Z-type ligands** are neutral two-electron acceptors: metal complexes
  such as CdCl₂ or Cd(O₂CR)₂ bound to surface anions. *Z-type displacement*
  ({ref}`post-z-type`) removes such units from the surface, mimicking their
  displacement by L-type donors. *Neutral exchange*
  ({ref}`post-neutral-exchange`) replaces them by another metal complex, a
  zwitterion or a neutral molecule.
- *Alloying* ({ref}`post-alloying`) substitutes a fraction of the native
  cations or anions by another element.

The polar {111} *reconstruction* ({ref}`theory-reconstruction`) is also
implemented as a post-treatment. It is described with the theory because
it completes the construction of zinc-blende particles.

## Order of execution

All post-treatments live in the `post_treatment` block of the recipe, each
with an `enabled` switch. They are applied in a fixed order, independent of
their order in the file:

1. charge balance and passivation (always);
2. `surface_reconstruction`;
3. core/shell relabelling (stack mode only);
4. `alloying`, followed by a full charge rebalance;
5. `z_type_displacement`;
6. `neutral_exchange`, with charge compensation per pass;
7. `ligand_exchange`, followed by a ligand-only charge compensation;
8. `neutral_ligands`.

The order follows the chemistry. The composition of the inorganic lattice
is fixed first. Surface metal complexes are then removed or exchanged, the
X-type shell is converted to molecules, and neutral donors are finally
bound to the sites that remain open.

## Common conventions

Every treatment is organised in *passes*. A pass selects a set of sites and
applies one operation to a fraction of them. How many sites are treated is
given either by `ratio`, a fraction between 0 and 1 of the eligible sites,
or by `target_count`, an absolute number that takes precedence when
positive. A pass with neither is ignored.

Which sites are treated is controlled by `distribution`:

- `random` samples the sites with a seeded random generator.
- `uniform` spreads the selection as evenly as possible. With the proximity
  matrix $P_{ij} = e^{-\lvert\mathbf r_i - \mathbf r_j\rvert}$, the builder
  starts from the most isolated site (smallest $\sum_j P_{ij}$). It then
  repeatedly adds the site with the smallest summed proximity to those
  already chosen.
- `segmented` makes the opposite, greedy choice, always adding the site
  closest to the selection, and so produces a compact patch. Patches of
  this kind are useful for modelling partially exchanged surfaces with
  domains.

The `uniform` and `segmented` distributions are deterministic. Randomness,
in the `random` distribution and in conformer generation, is controlled by
the `seed` of each block (1337 by default), with fixed offsets per pass, so
every result is reproducible.

A block with `enabled: false` is not parsed further. Typing mistakes inside
a disabled block therefore go unnoticed until the block is enabled.

## Real molecules

Treatments that introduce molecules take them as SMILES strings and build
all-atom three-dimensional structures with RDKit. The molecule is
protonated or deprotonated as needed, embedded with the ETKDG method,
optimised with UFF (or MMFF when requested and parametrised), and then
placed on the surface. Placement uses a scan over rotations about the
surface normal, scored by steric clearance from the particle and from
previously placed ligands. The resulting coordinates contain every atom,
hydrogens included. RDKit is therefore required for these treatments; the
builder raises an informative error if it is missing. Treatments that only
remove or relabel atoms (Z-type displacement, alloying, and MXn exchanges
with atomic ions) do not need it.
