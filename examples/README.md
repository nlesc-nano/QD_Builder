# QD Builder Examples

Run commands from the repository root. The examples assume the `nc-builder`
environment is active; prefix commands with `mamba run -n nc-builder` if needed.

## Nucleation Map

System packs (zincblende CdSe/InP, wurtzite CdSe, rock-salt PbS/PbSe), policy
knobs, and competitive ZB/WZ usage are documented in
[`nucleation/README.md`](nucleation/README.md).

Generate the inspectable CdSe/CdCl2 reference map at `k=1`:

```bash
nc-builder examples/nucleation/cdse_cdcl2.yaml
```

This creates `cdse_cdcl2_nucleation/` in the current directory with
`registry.json`, `nucleation.log`, and shell-safe XYZ folders such as
`structures/k001/p001/retained/` and `structures/k001/p001/discarded/`. Use
`-o path/to/bundle` to select another output directory. The equivalent Python
API is:

```bash
python - <<'PY'
from builder.nucleation import (
    generate_nucleation_result,
    load_nucleation_spec,
    write_nucleation_bundle,
)

spec = load_nucleation_spec("examples/nucleation/cdse_cdcl2.yaml")
result = generate_nucleation_result(spec)
write_nucleation_bundle(result, "examples/out/cdse_cdcl2_nucleation")
PY
```

The CLI prints live k/p progress by default. For long enumeration loops it
shows the theoretical ligand count, symmetry-orbit representatives actually
processed, valid candidates, bridge-search pruning, elapsed time, and DAG
frontier closure. Add `--verbose` for per-skeleton details.

The physical `p` bins count all CdCl2 units currently attached. All valid
inorganic skeletons seed later DAG bins; equivalent core-growth and
passivation routes are merged before Cl enumeration. Valid inferior ligand
isomers remain available as discarded XYZ files through `k=2`. The log records
per-element coordination values and retained/discarded decisions in
human-readable ASCII tables.

The YAML graph rules specify feasible-first minimum CN, hard maximum CN, and
unordered allowed element pairs. Ligands are redistributed over all compatible
outward sites for each fixed `(k,p)` stoichiometry. All construction geometry
comes automatically from the core CIF: allowed bonds use its nearest-neighbour
length and atoms occupy its rigid coordination sites. A bridge is possible only
at an exact shared lattice site unless a latent bridge rule is configured. The
CdSe/CdCl2 example enables Cl bridges between Cd atoms sharing Se and counts
those edges before ranking. Retained isomers additionally receive a
final-CN `*_surface.xyz` coordinate view using the optional CN-dependent
geometry rules. Complete bridge sets are accepted only when all final Cd
degrees remain within the configured maximum. Surface projection preserves the
ranked graph, places bridges first, and then rebuilds remaining terminal Cl
atoms; CN4 terminal Cl occupies an unassigned CIF tetrahedral direction. A
saturated Cd also rejects any arrangement in which a non-neighbor becomes
closer than one of its four graph neighbors. This is a relative slot check, not
a minimum-distance input. Rejected projected coordinates remain inspectable in
the discarded tree through `k=2`.
The original `*_construction_native.xyz` is preserved and remains the only
representation used for subsequent growth. Relaxation and energetic ranking
are intentionally left to later calculations.

### Reaching larger k

`cdse_cdcl2.yaml` is exact and intentionally stops at `kmax: 1`; exact enumeration
is affordable to about `k=3`. What limits it is the *skeleton* count, not the
ligand shell — distinct skeletons measured 4, 14, 243 at `k` = 1, 2, 3, since the
count is essentially the number of distinct lattice animals on the cation
sublattice and most of them are open or branched rather than compact.

`cdse_cdcl2_guided.yaml` trades completeness for reach in two controlled places:

```bash
nc-builder examples/nucleation/cdse_cdcl2_guided.yaml -o /tmp/nuc_guided
```

- `exact_through_k: 3` — grow `k=1..3` from every unique skeleton, then carry
  forward only the cores of retained structures. Rows up to and including that `k`
  are still enumerated in full; only what *leaves* a row is narrowed.
- `mode: guided` — one ligand shell per skeleton, bridging sites first then the
  most undercoordinated cation, with no ligand-isomer enumeration.

Both announce themselves. `registry.json` carries a `completeness` block naming
every active approximation and the `k` through which enumeration is still
complete, and the progress stream prints a `WARNING` line. An exact run states its
guarantees positively rather than staying silent, and a narrowing rule that
happens to drop nothing is reported as no loss — so the warning stays meaningful.

Treat guided output as a *sample* of each bin, not a survey: distinct ligand
arrangements on the same skeleton are never generated. The hard rules still hold —
max-CN caps, allowed bonds and the surface gate are enforced on the guided path
exactly as on the exact one.

To compare the two on the same system, run `mode: exact` at `kmax: 2` and check
`greedy_incumbent_matches_selection` per bin in `registry.json`: it records
whether one guided construction already attained the exact winner's score.

To build the separate smallest faceted Wulff reference, Cd16Se13Cl6, run:

```bash
python -m builder examples/cifs/CdSe_zb.cif \
  examples/core-only/cdse_minimum_wulff.yaml \
  -o examples/out/cdse_minimum.xyz --positive-q-mode add
```

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
python -m builder examples/core-shell/cdse_znse_core_shell.yaml \
  -o examples/out/cdse_znse.xyz --verbose --positive-q-mode add
```

Build a core-crown platelet-like shell where the shell grows only in x/y:

```bash
python -m builder examples/core-shell/cdse_znse_core_crown.yaml \
  -o examples/out/cdse_znse_crown.xyz --verbose --positive-q-mode add
```

## Janus Heterostructures

Build a CdSe/PbS Janus heterostructure using ZSL lattice matching and Wulff
outer shapes:

```bash
python scripts/build_janus_heterostructures.py examples/janus/cdse_pbs_wulff.yaml
```

Outputs land in `examples/out/janus/cdse_pbs_wulff/`.

Build a CsPbBr3/Pb4S3Br2 Janus heterostructure with a faceted perovskite side,
a spherical Pb4S3Br2 shell cap, and a mushroom footprint that permits a
controlled shell overhang:

```bash
python scripts/build_janus_heterostructures.py examples/janus/cspbbr3_pb4s3br2_mushroom.yaml
```

Outputs land in `examples/out/janus/cspbbr3_pb4s3br2_mushroom/`.

The taller variant uses the same interface selection but a thicker core:

```bash
python scripts/build_janus_heterostructures.py examples/janus/cspbbr3_pb4s3br2_mushroom_tall.yaml
```

Outputs land in `examples/out/janus/cspbbr3_pb4s3br2_mushroom_tall/`.

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
