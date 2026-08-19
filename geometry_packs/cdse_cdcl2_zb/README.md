# CdSe / CdCl2 — zinc-blende occupation growth (k = 1 → 4)

Dedicated pack for move Z.  Do not mix with `cdse_cdcl2/` (combinatorial A + kinetic B).

| file | role |
|---|---|
| `run_gxtb.yaml` | driver: CIF + g-xTB + includes |
| `graph_rules.yaml` | CN, allowed pairs, 2p Cl chemistry only |
| `motifs.yaml` | Se/Cl environments |
| `embed.yaml` | bond tables for placing Cl on the fixed zb core |
| `growth.yaml` | parent k = 1…3, child k = 2…4; A/B off; soft rules off |

Cores from zb sites.  Cl + 3D from the pack (`motif_bridge_*` then
`embed.yaml`).  Soft rules off.  No diamond scores, no Move B.

```bash
python tools/run_molecular_growth.py \
  --pack-dir geometry_packs/cdse_cdcl2_zb \
  --growth growth.yaml \
  --parents /path/to/gxtb_cdse_target_k1k2_p1p5 \
  --k-from 1 --k-to 4 --p-parents all \
  --output /path/to/growth_zb_k1_to_k4
```
