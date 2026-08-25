# CdSe / CdCl2 — zinc-blende occupation growth (k = 1 → 13)

Dedicated pack for move Z.  Do not mix with `cdse_cdcl2/` (combinatorial A + kinetic B).

| file | role |
|---|---|
| `run_gxtb.yaml` | driver: CIF + g-xTB + includes |
| `graph_rules.yaml` | CN, allowed pairs, 2p Cl chemistry only |
| `motifs.yaml` | Se/Cl environments |
| `embed.yaml` | bond tables for placing Cl on the fixed zb core |
| `growth.yaml` | parent k = 1…12, child k = 2…13; A/B off; soft rules off |
| `k13_wulff_core.yaml` | report-only Cd16Se13 endpoint reference |

The propagated state has two representations.  `zb_occupations.jsonl` stores
the exact ZB site occupation and its complete parent routes; relaxed XYZ files
store the g-xTB minima.  The next k is always generated from the former.  The
relaxed parent contributes only soft frontier ordering (optional WBO, local
coordination, and displacement), never new lattice coordinates.

For each child occupation the graph rules generate valid 2p Cl decorations.
μ2 host pairs are restricted to lattice Cd–Cd within 4.75 Å that do not pass
through an occupied cation.  Cl is seeded at the outward sphere intersection
(or an unused tet direction for terminals) on the CIF core, then a short
clash-aware least-squares fit morphs **core + Cl** onto `embed.yaml` (core is
not snapped back to CIF).  An optional frozen-core g-xTB refine can follow;
unconstrained g-xTB is the minimiser.  A converged endpoint enters
`index.csv` only when the Cd-Se topology is unchanged and all post-relaxation
graph/artifact audits pass.  Other relaxed endpoints are retained as
`*_offpath.xyz` but cannot propagate.

At k=13, `k013_endpoint_diagnostics.jsonl` compares each occupation with the
known Cd16Se13 Wulff-like core.  Site overlap, assignment RMSD, coordination,
radius of gyration, and axis extents are diagnostics only; g-xTB energy and
structural diversity still control parent retention.

Before the next k, propagation-eligible endpoints are consolidated into
relaxed basins.  Equality requires the same final element-labelled graph,
compatible optional WBO values, similar internal pair distances, and small
core/full RMSD after graph-constrained atom permutation and optimal proper
rotation.  Reflection is not allowed.  The lowest-energy endpoint represents
the basin for ranking, while every distinct stored ZB occupation reaching it
remains available as a lattice-growth route.  Thus rotations, translations,
atom renumberings, and duplicate starts do not consume multiple parent-minimum
slots, but distinct ligand conformers or lattice routes are not silently lost.
Each bin writes `minimum_clusters.json` with the exact thresholds, basin
representatives, raw members, invariant comparison metrics, and occupation
routes used for that decision.

```bash
python tools/run_molecular_growth.py \
  --pack-dir geometry_packs/cdse_cdcl2_zb \
  --growth growth.yaml \
  --parents /path/to/gxtb_cdse_target_k1k2_p1p5 \
  --k-from 1 --k-to 13 --p-parents all \
  --output /path/to/growth_zb_k1_to_k13
```
