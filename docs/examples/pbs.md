# PbS: a rock-salt particle and the role of the centre

Lead chalcogenide particles crystallise in the rock-salt structure and are
bounded by non-polar {100} facets and polar {111} facets. The recipe
`docs/examples/recipes/pbs.yaml` gives both families the same relative
energy and asks for {111} facets that end on lead. The {100} seed carries
no termination, since every {100} layer of rock salt contains equal
numbers of Pb and S.

```bash
nc-builder examples/cifs/PbS.cif docs/examples/recipes/pbs.yaml \
    -o pbs.xyz --center --positive-q-mode add --verbose
```

## An anion-rich particle

Centred on S and two unit cells across, the cut particle turns out to be
sulfur-rich on every facet. The verbose report lists the shell charge of
each facet: −10 on each {100} face and −2 on each {111} face. At this size
and centre, the outermost layers along ⟨111⟩ contain both species, so the
requested Pb termination cannot be realised. Because only one termination
of the {111} family was requested, the builder does not rebuild the
particle; the flip-and-rebuild check applies only to families requested
with both terminations (see {ref}`theory-polarity`).

The particle is therefore neutralised from the negative side. In the
prepass, sulfur atoms left with two bonds are converted into Cl. The main
loop then converts three-coordinated surface sulfur into Cl, one atom at a
time, preferring vertices, then edges, then terraces, until the charge
vanishes. The final structure, Pb₁₄₀S₇₉Cl₁₂₂ (341 atoms, 2.3 nm), contains
122 chlorides, all of them on former sulfur sites.

## The same size, centred on lead

Centring the same recipe on Pb (`center_on_species: Pb`) gives a very
different particle, Pb₁₇₇S₁₄₀Cl₇₄ (391 atoms, 2.5 nm). It exposes lead on
its {111} facets and needs about half as many chlorides. For small
particles the choice of centre is as important as the facet energies, and
the QDSpace library therefore contains both centres at every size.

## Rock-salt specifics

On rock-salt {111} every atom has three bonds on each side, so the
bond-counting rule of {ref}`theory-polarity` cannot decide which layer is
exposed, and the builder falls back to the unit-cell rule. The
coordination thresholds of the prepass are absolute, so edge and corner
lead atoms may keep fewer than their six bulk bonds. This mirrors the
strongly under-coordinated corners of small PbS particles.

For the QDSpace library, the relative energy of {100} is set to 1.0. With
the lower value of 0.8, larger particles become cube-like with small,
irregular {111} corners.
