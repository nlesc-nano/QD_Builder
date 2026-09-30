(guide-core-shell)=
# Core/shell and core/crown particles

Concentric heterostructures are built in *stack mode*. The recipe contains
a `materials` list, with the core first and the shells after it, and is
given to the builder without a CIF, since each material names its own:

```bash
nc-builder examples/core-shell/cdse_znse_core_shell.yaml \
    -o cdse_znse.xyz --positive-q-mode add --verbose
```

## The recipe

```yaml
charges: {Cd: 2, Zn: 2, Se: -2, Cl: -1}
construction_origin: {center_on_species: Se}
passivation: {ligand: Cl, surf_tol: 2.0}
materials:
  - name: core
    cif: examples/cifs/CdSe_zb.cif
    size_unit_cells: [1.5, 1.5, 1.5]
    facets:
      - {hkl: "100",    scope: family, termination: cation_rich, gamma: 1.0}
      - {hkl: "111",    scope: family, termination: cation_rich, gamma: 1.0}
      - {hkl: "-1-1-1", scope: family, termination: anion_rich,  gamma: 1.0}
  - name: shell
    cif: examples/cifs/ZnSe_zb.cif
    size_unit_cells: [1.0, 1.0, 1.0]
    facets:
      - {hkl: "100",    scope: family, termination: cation_rich, gamma: 1.0}
      - {hkl: "111",    scope: family, termination: cation_rich, gamma: 1.0}
      - {hkl: "-1-1-1", scope: family, termination: anion_rich,  gamma: 1.0}
```

Charges, centre, passivation and post-treatments are global. Each material
has its own CIF, facets and size. The CIF paths are resolved relative to
the directory from which the builder is run.

## Thickness

The `size_unit_cells` of each material is a thickness *increment*. In the
example the core boundary lies at 1.5 cells and the outer surface at
1.5 + 1 = 2.5 cells, which gives a shell about one unit cell thick. Either
every material gives a size or none does.

A shell whose increment is zero along one axis grows only in the other two.
With `[2, 2, 0]`, a core becomes a core/crown platelet, as in
`examples/core-shell/cdse_znse_core_crown.yaml`.

## Requirements

All materials must share the same space group. The model assumes one
cation and one anion per material, and relabelling is meant for isovalent
series such as CdSe/ZnSe/ZnS. The placeholder ligand must not be native to
any of the materials.

## Interfaces

```yaml
stack:
  interface: mixed        # or: abrupt (default)
  mixing_width: 3.0       # Å
```

A mixed interface alloys a band around the core/shell boundary. The
fraction of converted atoms is set per shell with
`interface: {mixing_ratio: 0.5}` on the second material. The atoms are
chosen by deterministic farthest-point sampling.

## Strain

By default the core is mapped back from the shell lattice onto its own
lattice constant. The mapping is complete in the interior and fades to zero
at the interface over `--core-strain-width` (2 Å). Use
`--no-core-lattice-fit` to keep the whole particle on the shell lattice.
The model is described in {ref}`theory-heterostructures`.

## Outputs

Besides the full particle, the builder always writes the core alone
(`core.xyz` and `<name>_core.xyz`). With `--write-all` it also writes every
layer before passivation. The manifest reports the size of the core, of
each shell and of the whole particle.
