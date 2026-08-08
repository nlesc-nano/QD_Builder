# Nucleation system packs

The nucleation engine is lattice- and chemistry-agnostic. **YAML packs** set
roles, graph rules, and lattice policy. Construction sites always come from the
CIF first shell.

## Pack matrix

| Pack | CIF | Lattice | `kmax` (smoke) | Ring length | Geometry defaults | Terminal motifs |
|---|---|---|---|---|---|---|
| `cdse_cdcl2.yaml` | CdSe_zb | zincblende II–VI | 1 | 6 | zb_tetrahedral | zb_mx2 (via default) |
| `cdse_cdcl2_guided.yaml` | CdSe_zb | zincblende II–VI | larger maps | 6 | zb_tetrahedral | default |
| `cdse_wz_cdcl2_k1.yaml` | CdSe_wz | **wurtzite** II–VI | 1 | 6 | zb_tetrahedral | **none** |
| `inp_incl3_k1.yaml` | InP_zb | zincblende III–V | 1 | 6 | zb_tetrahedral | **none** |
| `pbs_pbcl2_k1.yaml` | PbS | **rock-salt** IV–VI | 1 | **4** | **none** | **none** |
| `pbse_pbcl2_k1.yaml` | PbSe | **rock-salt** IV–VI | 1 | **4** | **none** | **none** |

DFT-calibrated bridge rules (`min_bridged_host_cn: 3`) are only locked for
CdSe/CdCl2 zincblende. Other packs use provisional graph rules.

## Molecular rules

`cdse_molecular_rules.yaml` contains only run/composition settings and points to
`geometry_packs/cdse_cdcl2_molecular.yaml`. The geometry pack is the sole source
of molecular CN limits, pair permissions and distance floors, bond lengths,
angles, allowed Cl host signatures, and reviewed hard impropers. At present the
only hard dihedral is Cd(CN=3) planarity (0°); ordinary proper torsions are DFT
evidence only.

### Lattice-free molecular map (graph enum + embed)

```bash
python tools/run_molecular_map.py \
  --yaml examples/nucleation/cdse_molecular_rules.yaml \
  --kmax 1 --pmax 3 \
  --output runs/molecular_cdse_k1
```

This enumerates unique Cd–Se–Cl graphs under H1/H4–H7, embeds with the geometry
pack (including tabulated bond lengths and explicit Cd CN3 planarity), and
writes XYZ + `index.csv`.
No CIF virtual sites. Python API: `generate_molecular_map`, `write_molecular_map`.

**Cl decoration mode** (`geometry_packs/cdse_cdcl2_molecular.yaml` →
`graph_rules.decoration_mode`):

| Mode | Role |
|------|------|
| `graph_multiset` (**default**) | Degree-first μ₁/μ₂/μ₃ host multisets + geometry modes. |
| `skeleton_bridge_first` | Embed Cd–Se skeleton → Cd–Cd distance window for μ₂ candidates → bridges first (max 2 bridge bonds/Cd) → terminals (allowed on bridged Cd). High-p passivation: k≥2,p≥4 ⇒ min Cd CN 3. |
| `tet_sites` / `pack_sites` | Experimental site enumerators. |

Enable bridge-first in the geometry pack:

```yaml
graph_rules:
  decoration_mode: skeleton_bridge_first
```

## Policy knobs (engine vs chemistry)

**Search / completeness** (cost and path dependence):

- `mode`, `exact_through_k`, beams, retain caps, occupation, decoration packages

**Chemistry / lattice** (which structures win):

- `graph_rules` (min/max CN, allowed bonds, bridging)
- `geometry_rules` + `geometry_defaults`
- `inorganic_ring_length` (compact growth and ring beams)
- `passivation_ring_policy` / `ring_lengths`
- `terminal_motifs` (`zb_mx2` or `none`)
- molecular filters above when enabled

Flat keys and nested blocks both work:

```yaml
lattice_policy:
  inorganic_ring_length: 4
  geometry_defaults: none
surface_geometry:
  terminal_motifs: none
```

## Competitive zincblende vs wurtzite

Run the **same chemistry** on two CIFs and compare bundles — no dual-lattice
engine:

```bash
nc-builder examples/nucleation/cdse_cdcl2.yaml \
  -o runs/cdse_zb_k1

nc-builder examples/nucleation/cdse_wz_cdcl2_k1.yaml \
  -o runs/cdse_wz_k1
```

Compare:

- which `(k, p)` bins retain structures
- ring / coordination metadata in `registry.json`
- construction-native XYZ under `structures/`

Align graph rules and charges between the two YAMLs before interpreting
differences as lattice preference. The wurtzite CIF (`examples/cifs/CdSe_wz.cif`)
is a standard P6₃mc model for smoke tests; replace it with a refined structure
when doing production maps.

## Rock-salt (PbS / PbSe)

```bash
nc-builder examples/nucleation/pbs_pbcl2_k1.yaml -o runs/pbs_k1
nc-builder examples/nucleation/pbse_pbcl2_k1.yaml -o runs/pbse_k1
```

Notes:

- Bulk CN is **6** from the CIF; set `max_cn` accordingly.
- Prefer **4-rings** (`inorganic_ring_length: 4`) for compact growth at higher k.
- Keep `terminal_motifs: none` until a rock-salt surface table is calibrated.
- Exact mode at k=1 is slower than ZB (more free sites / higher p capacity).
  Use `mode: guided` for larger maps.

## CdSe zincblende study map

Longer calibration runs live under `study_cdse_map/` (see that directory’s
README). Those recipes preserve DFT-backed Cd/Se/Cl rules.

## Python API

```python
from builder.nucleation import (
    generate_nucleation_result,
    load_nucleation_spec,
    write_nucleation_bundle,
)

spec = load_nucleation_spec("examples/nucleation/pbs_pbcl2_k1.yaml")
result = generate_nucleation_result(spec)
write_nucleation_bundle(result, "runs/pbs_k1")
```
