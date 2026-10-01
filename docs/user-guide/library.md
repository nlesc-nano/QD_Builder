(guide-library)=
# Generating library series

The QDSpace library contains, for every material, a series of builder
structures that span sizes and centres. The series are produced by
`builder.scripts.generate_library` from a small configuration file. For each
centre and each size, the generator builds the recipe, optionally adds a
reconstructed variant, checks the result, removes duplicates and writes a
record per unique structure.

```bash
python -m builder.scripts.generate_library examples/library/cdse_zb.yaml --out out/
```

## The configuration

```yaml
material: CdSe
family: II-VI
phase: zinc-blende
cif: ../cifs/CdSe_zb.cif          # relative to this file
native_elements: [Cd, Se]         # formula order; everything else is a ligand
centres: [Cd, Se]
sizes: {start: 0.5, stop: 6.0, step: 0.25}
positive_q_mode: add
reconstruction: auto              # "never" disables the reconstructed variant
min_core_atoms: 20
min_interatomic_distance: 1.8     # Å
min_anion_fraction_kept: 0.5
preset:                           # a complete builder recipe
  charges: {Cd: 2, Se: -2, Cl: -1}
  passivation: {ligand: Cl, surf_tol: 2.0, prepass_mode: role-aware,
                prepass_min_cn_terrace: 3, prepass_min_cn_edge: 2,
                prepass_min_cn_vertex: 1}
  facets:
    - {hkl: "100",    scope: family, termination: cation_rich, gamma: 1.0}
    - {hkl: "111",    scope: family, termination: cation_rich, gamma: 1.0}
    - {hkl: "-1-1-1", scope: family, termination: anion_rich,  gamma: 1.0}
  symmetry: {proper_rotations_only: true}
```

| Key | Default | Meaning |
|---|---|---|
| `material`, `family`, `phase` | required | copied into every record |
| `cif` | required | bulk structure, relative to the configuration |
| `native_elements` | required | elements of the inorganic core, in formula order |
| `centres` | required | species to centre on (`construction_origin.center_on_species`) |
| `sizes` | required | list, or `{start, stop, step}` (inclusive), in unit cells |
| `isotropic_cells` | `false` | scale the cell counts so a size N spans N × min(a, b, c) along every axis (non-cubic cells such as wurtzite) |
| `preset` | required | the builder recipe; `size_unit_cells` and the centre are filled in |
| `preset_overrides` | none | size-dependent changes, see below |
| `positive_q_mode` | `add` | passed to the builder |
| `reconstruction` | `auto` | `never` skips the {111}-reconstructed variant |
| `min_core_atoms` | 20 | reject smaller cores |
| `min_interatomic_distance` | 1.8 Å | reject clashes |
| `min_anion_fraction_kept` | 0.5 | reject reconstructions that lose most anions |
| `out_dir` | `out` | output root; the material name is appended |

A recipe that works well at one size may need adjusting at another.
`preset_overrides` holds a list of entries, each with a threshold
`min_unit_cells` and preset keys that replace the preset's for all sizes at
or above the threshold. The entries are applied in order:

```yaml
preset_overrides:
  - min_unit_cells: 2.75
    facets:
      - {hkl: "100", scope: family, gamma: 1.0}
      - {hkl: "111", scope: family, termination: cation_rich, gamma: 1.0}
```

## What the generator does

Each build runs the builder in-process with `--center` and the configured
positive-charge mode. Unless `reconstruction: never` is set, a second build
enables the {111} reconstruction (see {ref}`theory-reconstruction`). The
reconstructed variant is recorded only if the reconstruction was actually
applied, and it points to its clean parent.

Every structure is then checked. It is rejected if:
- it carries a net charge;
- its core has fewer than `min_core_atoms` atoms;
- two atoms are closer than `min_interatomic_distance`;
- it is centred on a different species than requested (the centre is the atom
  at the construction origin, so polar sites such as wurtzite's, whose core
  centroid lies up to ~1 Å off that atom along c, are attributed correctly);
- it is a reconstruction that kept less than `min_anion_fraction_kept` of
  its parent's anions.

Accepted structures are compared by fingerprint (see
{ref}`theory-identity`). Sizes are processed in increasing order, so a
structure that reappears at a larger size is recorded as a duplicate of the
first, and the larger size is appended to its `unit_cells_all`.

## Output

```
out/CdSe/
├── review.csv                 one row per build: kept, duplicate, rejected, skipped
├── review.md                  the same as a readable table
└── CdSe-Se-Cd68Se55Cl26-clean/
    ├── start.xyz
    ├── recipe.yaml            the exact recipe that produced it
    └── record.json
```

The record (schema version 1) contains the identifier, material, family,
phase, surface (`clean` or `reconstructed`), formula and composition split
into core and ligands, total charge, centre and its offset, the size
descriptors, the fingerprint, and the provenance. Provenance consists of the
generator, configuration, QD_Builder commit and whether the source tree was
modified, the CIF, the recipe and, for reconstructions, the reconstruction
ledger.

The repository ships configurations for all QDSpace templates in
`examples/library/`:
- zinc-blende II–VI (Cd, Zn, Hg chalcogenides; the Hg compounds add a
  stoichiometric {110});
- zinc-blende III–V (Ga and In pnictides);
- rock-salt IV–VI (PbS, PbSe, PbTe);
- cubic CsPbX₃ perovskites, centred on Cs;
- wurtzite CdSe (`cdse_wz.yaml`): six {100} prism facets plus the polar
  (001) cation-rich / (00-1) anion-rich pair, which take the same
  reconstruction as zinc-blende {111} / {-1-1-1}.

Variant recipes of a series (e.g. `inp_zb_100.yaml`, the III–V default plus
a cation-rich {100} at γ = 1.2) write to their own output directory and are
served from `builder_<tag>/` next to the default `builder/`; the webapp
ingest keeps one copy of a structure that two recipes produce.
