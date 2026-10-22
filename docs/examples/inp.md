# InP: a tetrahedral III–V particle

III–V particles such as InP grow as tetrahedra bounded mainly by
cation-terminated {111} facets. The recipe
`docs/examples/recipes/inp.yaml` expresses this with only the two polar
{111} families. The cation-terminated family has a lower relative energy
(0.8) than the anion-terminated one (1.6), so the In-terminated facets
dominate and the P-terminated facets appear as truncations of the corners.

```bash
nc-builder examples/cifs/InP_zb.cif docs/examples/recipes/inp.yaml \
    -o inp.xyz --center --positive-q-mode add
```

Because the plane distances scale with $\gamma_i/\gamma_\text{min}$, the
lowest-energy facets sit at the construction radius (11.8 Å for two unit
cells). The truncated corners reach much further out, and the particle is
larger than the nominal size suggests. The result is In₄₁₅P₃₄₈Cl₂₀₁, a
neutral particle of 964 atoms with an equivalent diameter of 3.7 nm.

Passivation proceeds as for CdSe, with two differences that follow from
the valence of the III–V lattice. Every In–P bond carries more charge, so
converting a surface P³⁻ into a Cl⁻ raises the charge by two instead of
one. Added chlorides are placed on-top (μ1) rather than in bridging
positions, in line with the terminal halide binding on III–V surfaces.

The {111} reconstruction can be enabled in the same way as for CdSe. On
III–V particles each vacancy under an anion facet changes the charge by
$-3 + 3\cdot 2 = +3$. On large P-terminated facets the cation facets may
then not offer enough removable In to compensate exactly. In that case
the builder keeps the residual charge and reports it rather than forcing
neutrality (see {ref}`theory-reconstruction`).
