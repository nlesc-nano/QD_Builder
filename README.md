# QD_Builder

QD_Builder builds and passivates quantum-dot models from CIF files. The main
builder supports faceted Wulff cuts, spherical cuts, core-shell particles, and
coordination-aware ligand passivation. The experimental Janus workflow builds
heterointerfaces by scanning facet terminations and matching 2D interface
lattices.

The Python nucleation API additionally enumerates symmetry-distinct, unrelaxed
core-growth and precursor-passivation maps in discrete `(k, p)` coordinates.

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

## Nucleation API

Nucleation YAML files are detected automatically:

```bash
nc-builder examples/nucleation/cdse_cdcl2.yaml
```

The CIF path is resolved relative to the YAML file. By default the command
creates `<recipe-name>_nucleation/`; use `-o path/to/bundle` to override it.
The bundle contains `registry.json`, `nucleation.log`, and separate shell-safe
XYZ trees such as `structures/k001/p001/retained/` and
`structures/k001/p001/discarded/`. Retained isomers contain both
`*_construction_native.xyz` and `*_surface.xyz`; discarded isomers contain
only construction-native coordinates and are written only through `k=2`.
XYZ names and title lines use shell-safe characters. During generation the CLI
reports theoretical combinations, symmetry-orbit representatives, actual
embeddings, merged skeleton routes, bridge-search pruning, and closed
frontiers. Long loops emit a heartbeat approximately every five seconds;
`--verbose` adds per-skeleton detail.

The same operation is available programmatically:

```python
from builder.nucleation import (
    generate_nucleation_result,
    load_nucleation_spec,
    write_nucleation_bundle,
)

spec = load_nucleation_spec("examples/nucleation/cdse_cdcl2.yaml")
result = generate_nucleation_result(spec)
write_nucleation_bundle(result, "examples/out/cdse_cdcl2_nucleation")
```

An opt-in performance audit reports theoretical combinations versus the orbit
representatives actually evaluated, without writing an output bundle:

```bash
python scripts/benchmark_nucleation.py \
  examples/nucleation/cdse_cdcl2.yaml --kmax 3 --progress
```

The retained registry is indexed as
`registry[k][p] -> list[ClusterRecord]`; valid non-isomorphic structures that
lose the coordination comparison are stored in `discarded_registry[k][p]`.
Routes reaching the same skeleton are merged before ligand enumeration.
Downward stripping validates registered parentage and does not regenerate
structures. Discarded records above `k=2` are reduced to counts, and locally
dominated bridge layers are pruned there.

### Reach and completeness

Distinct inorganic skeletons grow exponentially in `k` — measured 4, 14, 243 for
`k` = 1, 2, 3 — while the retained set stays flat, because the count is
essentially the number of distinct lattice animals on the cation sublattice and
most of them are open or branched rather than compact. Keeping those is right for
a small nucleus and wrong once the interior is bulk-like, so two knobs bound the
work:

```yaml
nucleation:
  exact_through_k: 3     # last k grown from EVERY unique skeleton (default 3)
  mode: exact            # exact | guided   (default exact)
```

`exact_through_k` narrows the `k -> k+1` growth step at and above that `k` to the
cores of *retained* structures. Rows up to and including it are still enumerated
in full — only what *leaves* the row is narrowed. This is the existing selection
score applied one level earlier (for fixed `(k,p)`, maximising bond count is
maximising compactness), not a new chemical parameter.

`mode: guided` places one ligand shell per skeleton in the passivation order —
bridging sites first, since one ligand there forms two bonds, then the most
undercoordinated cation — and does **not** enumerate ligand isomers at all. Cost
then scales with the number of skeletons rather than with the number of distinct
ligand arrangements.

Both are approximations, and each names itself in `registry.json`:

```json
"completeness": {
  "mode": "guided",
  "enumeration_complete_through_k": 0,
  "approximations": [
    {"stage": "ligand_placement", "method": "guided_passivation_order",
     "completeness": "not_guaranteed", "effect": "..."}
  ]
}
```

An exact run positively asserts its guarantees rather than staying silent, and an
approximate run also warns on the progress stream. Switching a narrowing rule on
that happens to drop nothing is reported as *no* loss, so the warning stays
meaningful. Note `discarded_counts` is exact for `k<=2` and a **lower bound**
above it, since bases provably unable to win their bin never become records.

The nucleation YAML declares only chemical graph rules:

```yaml
graph_rules:
  min_cn: {Cd: 2, Se: 2, Cl: 1}
  max_cn: {Cd: 4, Se: 4, Cl: 2}
  allowed_bonds: [[Cd, Se], [Cd, Cl]]
```

Optional retained-only surface templates may also be declared:

```yaml
geometry_rules:
  Cd: {cn2: linear, cn3: trigonal_planar, cn4: tetrahedral}
  Se: {all: tetrahedral}
  Cl: {all: tetrahedral}
```

DFT-informed latent ligand bridges may be enabled during graph screening:

```yaml
graph_rules:
  bridging:
    Cl:
      host: Cd
      shared_neighbor: Se
      surface_angle_deg: 90.0
      min_bridged_host_cn: 3     # default; 1 disables the rule
```

### Chemical options

Two rules change *which* structure a bin retains, and `registry.json` records
what was used.

`bridging.<ligand>.min_bridged_host_cn` sets the lowest final coordination both
cations of a bridge may have. With `3` — **the default** — a bridge that would
leave either cation at CN 2 is not formed. On CdSe/CdCl2 this reproduces four DFT
relaxations: at `k=1 p=1` the only available bridge would leave one Cd at CN 2 and
the relaxed structure instead keeps both Cd at CN 2 with terminal Cl and a linear
Se–Cd–Cl; at `k=1 p=2` three bridges between CN-3 and CN-4 Cd all hold; at
`k=2 p=0` a CN-2 Cd relaxes to a pseudo-linear Se–Cd–Se; and at `k=2 p=3` the four
structures the rule retains include the three that were relaxed, which land within
0.87 kcal/mol of each other.

That last bin is why the rule is on by default. Without it, `k=2 p=3` retains a
single 18-edge `Cd[2,4,4,4,4] Se[2,4]` structure that is not any of the relaxed
ones — one extra bond bought by stranding a Cd and an Se at CN 2, because `edges`
sits ahead of the coordination-evenness term in the score tuple. Enabling it costs
search time (`bridge_search_states` at k≤2 rises 1447 → 2390) because the rule is
non-monotone in the bridge set — adding a bridge raises its acceptor's CN — so
maximum-cardinality enumeration no longer suffices on its own.

Bond count is **not** a stability proxy here. Relaxing all four `k=2 p=3` members
gave final bond counts of 15, 16, 16 and 18 spanning only 1.8 kcal/mol — and the
*lowest* energy is the 15-bond structure, the highest the 18-bond one. The
strongest single datum is `k=2 p=1`: the rule's 6-bond `Cd[2,2,2]` sits
**5.4 kcal/mol below** the 8-bond bridged `Cd[2,3,3]` the old default kept, with
both connectivities holding through relaxation. Two fewer bonds, decisively lower.

The Pauling electrostatic valence `V = Σ |q(anion)|/CN(anion)` is reported per
cation in `registry.json` and `nucleation.log` and **ranks nothing**. It was
promising — the cation that shed a ligand looked like the most oversaturated one —
but it did not survive scrutiny on two counts. It does not order isomers: at
`k=2 p=3` the member with the *highest* `max V` is the lowest in energy, and at
`k=3 p=2` a `max V` 2.67 → 2.17 pair differs by only 0.57 kcal/mol, inside DFT
error. And the per-atom "which cation sheds" correlation is not robust: it depends
on which `max_cn`-valid reading of the relaxed coordinates you pick, and the
~3.07 Å bridge distance below makes several readings equally admissible. Treat the
number as a diagnostic to accumulate, not a signal to score on.

The rule is stated on the **finished structure**, not on the construction step,
and that distinction is load-bearing. Phrasing it as a minimum coordination for
the bridge *donor* has no effect at all: the identical structure is reachable from
a ligand arrangement whose donor is three-coordinate, and the route-merging DAG
rebuilds it there. Any rule expressed on how a structure was built leaks the same
way, because isomorphic results from different routes are merged by design.

`bond_count_scope: all | skeleton` chooses what the score's bond-count component
counts. Under `skeleton` only cation–anion bonds count, so ligand bonds and
bridges stop buying rank. It reaches the same `k=2 p=3` family as
`min_bridged_host_cn: 3` by an independent route, but it contradicts the
`k=1 p=1` relaxation — it keeps the bridged `Cd[2,3]` there — so `all` remains the
default and the bridging rule carries the chemistry instead.

`examples/nucleation/cdse_cdcl2_dft_rules.yaml` is the same system with the
bridging rule **off**, kept runnable as the counterfactual. Raise `kmax` to 2 in
both and diff `k=2 p=3` to see the bin that decides it.

**Reading a bundle: use the graph, not a distance cutoff.** A rhombic bridge puts
its ligand about 3.07 Å from each host cation, which is outside the
`|d − bond_length| ≤ site_tolerance` shell the module itself uses to infer bonds.
So a declared bridge is not a bond by distance, and a third cation at that same
distance is geometrically indistinguishable from a real bridge partner — a naive
cutoff read of the XYZ can show a μ3 ligand that `max_cn` forbids in the graph.
`registry.json` carries the authoritative connectivity.

Geometry templates do not participate in screening. All nucleation candidates occupy
rigid CIF-derived virtual sites, and every allowed bond uses the core
monomer's nearest-neighbour distance. The construction-native retained
coordinates remain the source for growth to the next k. A configured bridge
adds a latent graph edge when the complete bridge set leaves every Cd at or
below its configured maximum CN, while its construction-native Cl coordinate
remains on a tetrahedral site. Maximum-cardinality bridge patterns and an unbridged baseline
are symmetry filtered before CN/bond ranking. Thus bridge edges are already
included in the graph coordination used to rank construction-native records,
even though an XYZ viewer cannot infer a long latent bond from their coordinates.
After the registry is complete, the surface coordinate block places bridges in
the Cd-Se-Cd plane, recomputes final Cd coordination, and rebuilds every
remaining terminal Cl from that final environment. A terminal Cl on CN4 Cd is
assigned to an unoccupied tetrahedral direction extracted from the CIF rather
than to an arbitrary continuous best-fit direction. For saturated Cd, its four
graph neighbors must remain its four nearest chemically compatible neighbors;
this relative slot-consistency rule has no fixed distance cutoff. Failed
surface arrangements do not seed later sweeps and are written as
`*_surface_rejected.xyz` in the discarded tree through `k=2`. Graph edges never
change.
Full structural relaxation and energetic ranking remain downstream operations. Ligands are indistinguishable
within a fixed `(k,p)` bin and are distributed over every compatible outward
site rather than remaining assigned to their historical precursor center.
Minimum CN is feasible-first: compliant isomers are preferred whenever they
exist, while unavoidable seed deficits remain available and are logged.
Structures are then ranked by bond count and coordination-deficit balance. The
log reports every retained and discarded isomer's per-element CN values, CN
histograms, minimum-CN violations, total CN, bond count, source operation, and
selection reason. It is organized into terminal-safe ASCII tables for run
configuration, sweeps, bins, isomers, surface geometry, rejection reasons, and
final totals.
The raw ranking tuple remains available in `registry.json`.

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
  the selected reference lattice so shared-cation and shared-anion stacks with
  matching replica counts preserve the same discrete Wulff topology.
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
