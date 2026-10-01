(example-cdse)=
# CdSe: a reconstructed zinc-blende particle

This example follows a zinc-blende CdSe particle through the full
construction. The particle is two unit cells across and centred on Se.
Recipes: `docs/examples/recipes/cdse.yaml` and `cdse_recon.yaml`.

```bash
nc-builder examples/cifs/CdSe_zb.cif docs/examples/recipes/cdse.yaml \
    -o cdse.xyz --center --positive-q-mode add --verbose
```

## Shape

The recipe gives the {100}, {111} and {$\bar1\bar1\bar1$} families the same
relative energy, which produces a cuboctahedral shape truncated
differently on the two polar {111} orientations. Two unit cells of CdSe
correspond to a construction radius of 12.28 Å. The {100} planes coincide
exactly with atomic layers at this size, so the builder moves them outward
by 0.25 Å to keep complete layers, and reports this with `[wulff:shift]`.
Because the particle is centred on Se, its {100} and {111} facets can end
on Cd, as requested, while the four {$\bar1\bar1\bar1$} facets end on Se.

## Passivation

After pruning, the surface still contains selenium atoms held by only two
bonds, at the corners and along the Se-terminated facets. The prepass and
the first priority of the charge balance convert these into Cl, and each
conversion raises the charge by one. Most of the chlorides in the final
structure originate this way, on former Se sites. What remains is a small
positive charge of +4. With `--positive-q-mode add` the builder keeps the
cations and places four chlorides with the site selector of
{ref}`theory-ligand-sites`. On this particle all four are μ3 hollows on
the Cd-terminated {111} facets. The result is
Cd₁₇₆Se₁₄₇Cl₅₈, a neutral particle of 381 atoms with an equivalent-sphere
diameter of 2.6 nm.

## Reconstruction

The neutral particle is not compensated locally. Its Se-terminated facets
expose extended rows of three-coordinated Se, while the Cd-terminated
{111} facets are balanced only by the four hollow chlorides. Enabling the
{111} reconstruction changes this, and the ledger in `cdse_recon.json`
records each step:

```text
"anion_facets": [{"hkl": [1, 1, 1], "vacancies": 3, "anions_to_ligand": 9, "chain_breaks": 0}, ...],
"cation_facets": [{"hkl": [1, 1, -1], "ligands_stripped": 1, "cations_removed": 2, "ligands_added": 0}, ...],
"anion_side_charge_delta": 12,
"cations_removed": 8,
"total_charge_before": 0, "total_charge_after": 0
```

On each of the four anion-terminated facets, three non-adjacent Cd atoms
below the surface are removed. Each vacancy leaves three Se atoms with two
bonds, which become Cl, so each facet gains $3\,(-2 + 3) = +3$ and the
anion side as a whole $+12$. On the cation-terminated facets, four
chlorides are stripped, raising the charge to $+16$. Eight Cd atoms (two
per facet) are then removed, which returns the charge to exactly zero, so
no further ligand is needed. The reconstructed particle, Cd₁₅₆Se₁₁₁Cl₉₀,
has lost the rows of under-coordinated Se, and its polar facets are each
compensated locally.

The per-facet numbers show where the reconstruction acted, and
`total_charge_after` confirms that it preserved neutrality.
