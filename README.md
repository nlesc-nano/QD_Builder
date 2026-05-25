# QD_Builder

QD_Builder builds and passivates quantum-dot models from CIF files. The main
builder supports faceted Wulff cuts, spherical cuts, core-shell particles, and
coordination-aware ligand passivation. The experimental Janus workflow builds
heterointerfaces by scanning facet terminations and matching 2D interface
lattices.

## Install

```bash
conda env create -f environment.yml
conda activate nc-builder
pip install -e .
```

or, in an existing Python environment:

```bash
pip install -e .
```

## Main Builder

Core-only, size defined in YAML:

```bash
python -m builder examples/cifs/InAs.cif examples/core-only/inas_wulff_size_cells.yaml \
  -o examples/out/inas.xyz --verbose --positive-q-mode add
```

Core-only spherical cut:

```bash
python -m builder examples/cifs/Pb4S3Br2_DFT.cif examples/core-only/pb4s3br2_sphere.yaml \
  -o examples/out/pb4s3br2_sphere.xyz --verbose --positive-q-mode add
```

Core-shell:

```bash
python -m builder examples/core-shell/cdse_znse_core_shell.yaml \
  -o examples/out/cdse_znse.xyz --verbose --positive-q-mode add
```

For core-shell, only the YAML is required (material CIFs come from `materials[].cif`):

```bash
python -m builder examples/core-shell/cdse_znse_core_shell.yaml -o examples/out/cdse_znse.xyz --positive-q-mode add
```

Single-material still uses CIF + YAML:

```bash
python -m builder examples/cifs/CdSe_zb.cif examples/core-only/inas_wulff_size_cells.yaml \
  -o examples/out/inas.xyz --positive-q-mode add
```

For core-shell mode, the positional CIF is ignored; material CIFs come from
`materials[].cif` in the YAML.

## Janus Heterostructures

The Janus workflow is experimental and lives in a separate script:

```bash
python scripts/build_janus_heterostructures.py examples/janus/cdse_pbs_wulff.yaml
```

Outputs are written under `examples/out/janus/` (see each example YAML `output.out_dir`).

Example with a faceted CsPbBr3 side and spherical Pb4S3Br2 cap:

```bash
python scripts/build_janus_heterostructures.py examples/janus/cspbbr3_pb4s3br2_mushroom.yaml
```

## YAML Notes

Single-material `size_unit_cells` is top-level:

```yaml
size_unit_cells: [2, 2, 2]
facets:
  - hkl: "100"
    scope: family
    termination: cation_rich
    gamma: 1.0
```

`scope: family` is the default and expands the given `hkl` to all
symmetry-equivalent oriented facets. Use `scope: facet` only when assigning
orientation-specific energies; in that case all symmetry-equivalent oriented
facets must be listed explicitly.

See `examples/core-only/inas_oriented_facet_scope.yaml` for a full
orientation-specific facet-energy example.

For core-shell, define size per layer:

```yaml
materials:
  - name: core
    cif: examples/cifs/CdSe_zb.cif
    size_unit_cells: [1.5, 1.5, 1.5]
    facets: [...]
  - name: shell
    cif: examples/cifs/ZnSe_zb.cif
    size_unit_cells: [1, 1, 1]
    facets: [...]
```

Stack-mode notes:

- Stack mode takes the YAML recipe only; each material's CIF path comes from
  the YAML `materials:` block (no CLI CIF argument).
- The shared Wulff replica lattice defaults to the **core** material CIF
  (`stack.geometry_reference: core`). Use `shortest` for legacy behavior.
- All materials must share the same CIF space group (e.g. F-43m for zinc-blende).
- `size_unit_cells` define replica topology; the shared geometry reference uses
  the shortest lattice parameter among the materials so swapping core/shell
  identity preserves the discrete Wulff shape.
- Core lattice fit is enabled by default; passivation runs on the shared reference
  geometry first, then core atoms are warped to the core CIF metric. Pass
  `--no-core-lattice-fit` to disable.
- Isovalent cations (e.g. Zn2+ and Cd2+) share bond cutoffs and bulk-CN targets
  in stack passivation so swapped core/shell identities yield identical
  stoichiometry.

Spherical cuts do not need `facets`:

```yaml
shape:
  mode: sphere
  sphere_planes: 192
```

## More Examples

See [examples/README.md](examples/README.md) for maintained example inputs and
commands.
