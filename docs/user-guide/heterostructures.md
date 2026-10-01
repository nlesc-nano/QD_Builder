(guide-janus)=
# Janus heterostructures and interface scans

Particles in which two materials meet across a planar interface are built
with `builder.scripts.build_janus_heterostructures`. It combines an
interface search with the construction of the two halves. The theory is
given in {ref}`theory-heterostructures`.

## Exploring possible interfaces

Before building, it is useful to see which interfaces two materials can
form:

```bash
python -m builder.scripts.scan_heterointerfaces core.cif shell.cif \
    --charges Cd=2 Se=-2 Pb=2 S=-2 --zsl --out scan.md
```

The scan lists the charged terminations of every facet family up to
`--max-index` for both materials. It pairs them by charge compensation
and, with `--zsl`, keeps only lattice-matched pairs, reporting their
mismatch and supercell area. To inspect the facets of a single crystal:

```bash
python -m builder.scripts.analyze_cif_facets crystal.cif --charges Cd=2 Se=-2
```

## Building Janus particles

The Janus builder reads its own YAML format:

```yaml
materials:
  core:  {cif: ../cifs/CdSe_zb.cif, name: CdSe}
  shell: {cif: ../cifs/PbS.cif,     name: PbS}
charges: {Cd: 2, Se: -2, Pb: 2, S: -2, Cl: -1}
scan: {max_index: 1, layer_tol: 0.08}
matching: {method: zsl, max_length_tol: 0.08, max_area: 600}
candidates: {top: 1}
build:
  mode: wulff_janus
  core:  {size_unit_cells: 2, facets: [...]}
  shell: {size_unit_cells: 2, facets: [...]}
  interface_distance: 2.8
passivation: {enabled: true, ligand: Cl, positive_q_mode: add}
output: {out_dir: out/janus, prefix: cdse_pbs}
```

```bash
python -m builder.scripts.build_janus_heterostructures examples/janus/cdse_pbs_wulff.yaml
```

For each of the `top` candidates the script writes an XYZ file and a JSON
file, which records the terminations, their charges, the lattice match and
the layers actually found at the interface. A summary CSV covers all
candidates.

| Section | Keys |
|---|---|
| `scan` | `max_index` (1), `layer_tol` (0.08 Å), `allow_charged_neutral`, `signed`, `all_rotations` |
| `matching` | `method` (`zsl` or `none`), `max_area` (400 Å²), `max_length_tol` (0.03), `max_angle_tol` (0.01°), `max_area_ratio_tol` (0.09) |
| `candidates` | `top` (5), `match`, `core_family`, `shell_family`, `core_hkl`, `shell_hkl` |
| `build` | `mode` (`radius`, `interface_cell`, `wulff_janus`), `radius` (18 Å), `lateral_cells`, `core_layers`/`shell_layers` (6), `interface_distance` (2.8 Å), `min_separation` (1.2 Å), `match_core_footprint` (true), `footprint_shape` (`bbox`, `convex`, `mushroom`), `footprint_margin` (1 Å), `mushroom_overhang` |
| `passivation` | `enabled`, `ligand`, `surf_tol` (2 Å), `positive_q_mode`, `positive_q_mode_core`, `positive_q_mode_shell` |
| `output` | `out_dir`, `prefix` |

Only the outer surface is passivated; the buried interface is left
untouched. The positive-charge strategy can differ between the two halves,
for example `remove` on a perovskite core and `add` on a chalcogenide
shell. The examples in `examples/janus/` include a CsPbBr₃/Pb₄S₃Br₂
particle with a mushroom-shaped shell cap.

## Size scans

`builder.scripts.scan_size_cells` runs the builder over a range of sizes
and tabulates composition, charge and diameter:

```bash
python -m builder.scripts.scan_size_cells crystal.cif recipe.yaml \
    --start 1 --stop 3 --step 0.5 --positive-q-mode add
```
