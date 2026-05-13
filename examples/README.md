# QD Builder Examples

Run commands from the repository root. The examples assume the `nc-builder`
environment is active; prefix commands with `mamba run -n nc-builder` if needed.

## Core-Only Wulff Cut

Build all construction-origin variants for an InAs Wulff particle using
`size_unit_cells` from the YAML:

```bash
python -m builder examples/cifs/InAs.cif examples/core-only/inas_wulff_size_cells.yaml \
  -o examples/out/inas.xyz --verbose --positive-q-mode add
```

Build an InAs Wulff particle where one symmetry-equivalent `{111}` cation-rich
set is assigned explicit oriented-facet energies:

```bash
python -m builder examples/cifs/InAs.cif examples/core-only/inas_oriented_facet_scope.yaml \
  -o examples/out/inas_oriented.xyz --verbose --positive-q-mode add
```

## Core-Only Spherical Cut

Build Pb4S3Br2 with an isotropic spherical outer shape instead of explicit
Miller facets:

```bash
python -m builder examples/cifs/Pb4S3Br2_DFT.cif examples/core-only/pb4s3br2_sphere.yaml \
  -o examples/out/pb4s3br2_sphere.xyz --verbose --positive-q-mode add
```

## Core-Shell

Build CdSe/ZnSe with per-layer `size_unit_cells`:

```bash
python -m builder examples/cifs/CdSe_zb.cif examples/core-shell/cdse_znse_core_shell.yaml \
  -o examples/out/cdse_znse.xyz --verbose --positive-q-mode add
```

Build a core-crown platelet-like shell where the shell grows only in x/y:

```bash
python -m builder examples/cifs/CdSe_zb.cif examples/core-shell/cdse_znse_core_crown.yaml \
  -o examples/out/cdse_znse_crown.xyz --verbose --positive-q-mode add
```

## Janus Heterostructures

Build a CdSe/PbS Janus heterostructure using ZSL lattice matching and Wulff
outer shapes:

```bash
python scripts/build_janus_heterostructures.py examples/janus/cdse_pbs_wulff.yaml
```

Build a CsPbBr3/Pb4S3Br2 Janus heterostructure with a faceted perovskite side,
a spherical Pb4S3Br2 shell cap, and a mushroom footprint that permits a
controlled shell overhang:

```bash
python scripts/build_janus_heterostructures.py examples/janus/cspbbr3_pb4s3br2_mushroom.yaml
```

The taller variant uses the same interface selection but a thicker core:

```bash
python scripts/build_janus_heterostructures.py examples/janus/cspbbr3_pb4s3br2_mushroom_tall.yaml
```

## Utility Scripts

Scan size-cell values for a standard builder YAML:

```bash
python scripts/scan_size_cells.py examples/cifs/InAs.cif examples/core-only/inas_wulff_size_cells.yaml \
  --sizes 1 1.5 2 2.5 3 --out-dir examples/out/inas_size_scan \
  --summary examples/out/inas_size_scan/summary.md \
  --csv examples/out/inas_size_scan/summary.csv --positive-q-mode add
```

Analyze low-index CIF facets and terminations:

```bash
python scripts/analyze_cif_facets.py examples/cifs/CdSe_zb.cif \
  --charges Cd=+2 Se=-2 --max-index 2
```
