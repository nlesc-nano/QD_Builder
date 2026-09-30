# CsPbBr₃: a perovskite cube

Cesium lead halide nanocrystals are cubes bounded by {100} facets, and
experiments and calculations agree that their surfaces are terminated by
CsX-like layers {cite:p}`bodnarchuk2019`. The recipe
`docs/examples/recipes/cspbbr3.yaml` builds such a cube, three unit cells
across and centred on Cs.

```bash
nc-builder examples/cifs/CsPbBr3.cif docs/examples/recipes/cspbbr3.yaml \
    -o cspbbr3.xyz --center --positive-q-mode remove
```

Here the positive charge left by the cut is removed by eliminating surface
cations (`remove`), not compensated with added ligands. For a Cs-centred
cube with an integer number of unit cells, this yields a stoichiometrically
balanced particle, Cs₃₂₄Pb₂₁₆Br₇₅₆, with 1296 atoms and zero charge. It
contains $6^3 = 216$ PbBr₆ octahedra and a complete CsBr shell, and no
placeholder ligand is needed. The equivalent diameter is 4.9 nm.

Cubes of half-integer size instead end on PbBr₂ layers and require added
ligands, so only integer sizes are used in the QDSpace library. The
surface bromide can then be exchanged for carboxylates or other anions
with the X-type exchange of {ref}`post-x-type`, or CsBr units can be
replaced with the neutral exchange of {ref}`post-neutral-exchange`.

The placeholder ligand must not be a native element of the crystal. For
CsPbCl₃ a different symbol, such as Br, must therefore be used as the
placeholder.
