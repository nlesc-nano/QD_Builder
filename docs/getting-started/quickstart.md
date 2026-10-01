# Quickstart

This page builds a first nanocrystal: a Cl-passivated zinc-blende CdSe
particle two unit cells across, centred on a selenium atom. Two inputs are
needed: the bulk crystal structure as a CIF file, and a *recipe* that
describes the particle.

## The recipe

```yaml
# docs/examples/recipes/cdse.yaml
charges: {Cd: 2, Se: -2, Cl: -1}
size_unit_cells: [2.0, 2.0, 2.0]
construction_origin: {center_on_species: Se}
passivation:
  ligand: Cl
  surf_tol: 2.0
  prepass_mode: role-aware
  prepass_min_cn_terrace: 3
  prepass_min_cn_edge: 2
  prepass_min_cn_vertex: 1
facets:
  - {hkl: "100",    scope: family, termination: cation_rich, gamma: 1.0}
  - {hkl: "111",    scope: family, termination: cation_rich, gamma: 1.0}
  - {hkl: "-1-1-1", scope: family, termination: anion_rich,  gamma: 1.0}
symmetry: {proper_rotations_only: true}
```

The recipe reads almost as a description of the particle:
- **`charges`** gives the formal charges, including that of the placeholder
  ligand.
- **`size_unit_cells`** and **`construction_origin`** fix the size and the
  central atom.
- **`passivation`** chooses the ligand and how strictly under-coordinated
  surface atoms are removed.
- **`facets`** lists the crystal faces and their relative surface energies.
  Here the {100} and {111} faces end on cadmium, and the
  {$\bar1\bar1\bar1$} faces end on selenium.

## Building

```bash
nc-builder examples/cifs/CdSe_zb.cif docs/examples/recipes/cdse.yaml \
    -o cdse.xyz --center --positive-q-mode add
```

The option `--positive-q-mode add` tells the builder to neutralise excess
metal by adding ligands rather than by removing cations, which gives the
metal-rich particles typical of colloidal syntheses. `--center` places the
centroid at the origin.

The builder writes `cdse.xyz`, a particle of 381 atoms with the composition
Cd₁₇₆Se₁₄₇Cl₅₈ and zero net charge, together with `cdse.json`, a manifest
that records the composition, the charge and the size. Adding `--verbose`
prints every step: the facets found, the surface atoms and their
coordination, and every atom removed, swapped or added during the charge
balance.

## Reconstructing the polar facets

The same particle with its {111} facets reconstructed needs two more lines
in the recipe:

```yaml
post_treatment:
  surface_reconstruction:
    enabled: true
    ligand: Cl
```

Built from `docs/examples/recipes/cdse_recon.yaml`, the particle becomes
Cd₁₅₆Se₁₁₁Cl₉₀. Selenium vacancies and Se→Cl conversions now compensate the
anion-terminated facets, and cadmium has been removed from the
cation-terminated ones. {ref}`example-cdse` walks through what happened.

## Where to go next

- The {ref}`guide-recipes` explains every section of a recipe.
- The theory chapters, starting from {ref}`theory-overview`, describe what
  the builder does and why.
- The post-processing chapters, starting from {ref}`post-overview`, show how
  to replace the Cl placeholders by real ligands.
