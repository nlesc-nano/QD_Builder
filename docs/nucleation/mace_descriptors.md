# Descriptors for MACE-MH-1 / PBE geometries

Module: `src/builder/nucleation/mace_descriptors.py`. Tests:
`tests/test_mace_descriptors.py`. Rescoring tool:
`tools/rescore_mace_descriptors.py`. Calibrated 2026-09-15 on
`runs/mace_import/reopt_k7_k10` (309 g-xTB→MACE pairs, k=7–10) and
`runs/mace_import/dft_compare` (141 CP2K/PBE→MACE pairs, k=1–6).

## Why the legacy descriptors misbehave on MACE/PBE geometries

`search_analysis.descriptors` builds the bond graph with one cutoff per pair
(Cd–Se 3.25 Å, Cd–Cl 2.90 Å from `graph_rules.yaml`). Pair-distance
profiles show where that cutoff sits on each potential:

| Cd–Se | peak | first RDF minimum | p95 of bonded | share of bonded contacts in 2.8–3.25 Å |
|---|---|---|---|---|
| g-xTB (309 endpoints) | 2.57 | **3.17** | 2.77 | 1.6 % |
| MACE (same 309, re-relaxed) | 2.62 | **3.32** | 2.93 | 10 % |
| PBE (141 DFT endpoints) | 2.62 | **3.32** | 2.91 | 11 % |

| Cd–Cl | peak | first minimum | p95 of bonded |
|---|---|---|---|
| g-xTB | 2.47 | 3.22 | 2.64 |
| MACE | 2.52 | 3.47 | 2.70 |
| PBE | 2.52 | 3.27 | 2.76 |

g-xTB has a clean gap at 2.9–3.2 Å; MACE and PBE fill it with a
2.9–3.3 Å tail that the 3.25 Å cutoff cuts through. Consequences on the
k7–10 batch: the legacy `core_hash` "changed" in 180/309 relaxations, and
Se looked 5/6-coordinate in 288 cases — but at a 2.95 Å threshold the Se CN
histogram is unchanged by MACE ({3:1538, 4:955, 5:42} → {3:1669, 4:821, 5:42}).
The extra coordination is 4 strong bonds + 1–2 long contacts, on both MACE
and PBE.

## Two-tier bond model

| tier | Cd–Se | Cd–Cl | used for |
|---|---|---|---|
| strong | ≤ 2.95 Å | ≤ 2.80 Å | scaffold certificate (`core_hash_strong`), Se/Cd CN, tetrahedrality, rings |
| all | ≤ 3.35 Å | ≤ 3.15 Å | connectivity, bridging (`mu2`/`mu3`, bridge load), `core_hash_all` |
| long | difference | difference | `n_cdse_long`, `n_cdcl_long`, `se_with_long_contact` |

Thresholds: strong = round(p95 of the bonded population on MACE/PBE);
all = first RDF minimum, which MACE and PBE agree on.

## Descriptor list

Coordination (strong and all): `se_cn_*`, `cd_cn_se_*`,
`cd_environments_strong` ("nSe,nCl"), `cl_cn_*`, `se_cn4_fraction_strong`,
`se_cn34_fraction_strong`, `cd_detached_*` (Cd with no Se neighbour = a
shed CdCl₂ unit).

Bridging (all-bond graph): `terminal`, `mu2`, `mu3`, `isolated_cl`,
`bridge_load_{mean,max,var}`.

Tetrahedrality / ZB (strong graph unless noted):

- `q4_se_strong`, `q4_cd_strong`: standard q4 for strong-CN4 atoms.
- `pyr_se_strong`, `pyr_cd_strong`: pyramidality for strong-CN3 atoms —
  `1 − |1 − |Σ unit vectors||`, 1 = three tetrahedral directions (ZB
  surface site), 0 = trigonal planar.
- `q4_nn4_se`, `q4_nn4_cd`: **cutoff-free** q4 from the four nearest
  counter-ions of every atom.
- `cdsecd_tet_fraction`, `secdse_tet_fraction`: share of bond angles within
  ±15° of 109.47°.
- `rings4_strong`, `rings6_strong`, `ring4_fraction`: ZB cores have only
  6-rings; 4-rings mark rocksalt/square packing.
- `zb_score`: mean of the available `q4_*_strong`, `pyr_*_strong`, angle
  fractions and `1 − ring4_fraction`, in [0,1].

Certificates: `core_hash_strong`, `core_hash_all`, `graph_hash_strong`
(Weisfeiler–Lehman, element-labelled), `se_scaffold_components_all`
(number of separate Se-containing cores), `n_components_all`.

Geometry: `radius_A`, `min_distances` per element pair.

## `scaffold_change(before, after)`

| label | rule |
|---|---|
| `split` | more Se-containing core components after than before |
| `retained` | strong and all core hashes identical |
| `retained_long_contacts_changed` | strong hash identical, only long contacts differ |
| `minor` | strong hash differs by ≤ 1 Cd–Se strong edge and no Se lost CN3/CN4 |
| `changed` | anything else |

Also returns `d_n_cdse_strong`, `d_n_cdse_long`, `d_se_cn34`,
`d_cd_detached`, `d_zb_score`.

## Rescoring results

k7–10 batch (309 g-xTB → MACE):

| label | n |
|---|---:|
| retained | 94 |
| retained_long_contacts_changed | 62 |
| minor | 98 |
| changed | 55 |
| split | 0 |

Legacy single-cutoff "core retained" was 129/309; the strong scaffold is
identical in 156/309 and unchanged-or-minor in 254/309 (82 %). The 10
apparent "core splits" are CdCl₂ units shedding from the core (3 re-attach),
now reported as `d_cd_detached`, not as scaffold loss.

Cutoff-free view of what MACE actually changes: `q4_nn4_se` 0.34 → 0.28,
`pyr_se_strong` 0.42 → 0.33, long Cd–Se contacts 140 → 456. MACE/PBE
genuinely flattens Se environments and adds long contacts; that is the PBE
packing the DFT comparison confirmed, and it is now measured directly rather
than showing up as a CN artefact.

DFT comparison (141 PBE → MACE): 125 retained, 6 minor, 6 changed, 0 split;
strong Se CN pooled {3:302, 4:89, 5:27} → {3:298, 4:88, 5:29}; `zb_score`
0.460 → 0.466. Same descriptor set, same verdict as the RMSD analysis.

## Wiring in the run path

`relaxation.descriptor_cutoffs` in the pack YAML selects the model
(`legacy` | `two_tier` | `auto`, default `auto` = `two_tier` for the MACE
backend, `legacy` otherwise). `run_mace_mh1.yaml` sets `two_tier`
explicitly. Resolution lives in `xtb_relax.descriptor_model()` /
`descriptor_bond_cutoffs()`.

Under `two_tier` the `Pilot`:

- builds every post-relax bond graph (`describe_minimum`, chemistry-rule
  violations, connectivity drift, legacy `final` descriptors and the WL
  `core_hash` used for lineage and family keys) from the **strong** tier
  (Cd–Se 2.95 / Cd–Cl 2.80 Å) instead of the pack's 3.25 / 2.90 Å;
- records `final["descriptor_model"]`, `final["bond_cutoffs"]` and the full
  two-tier block in `final["two_tier"]`, including
  `scaffold_change` computed against the proposal's pre-relaxation
  coordinates and `construction_core_hash_strong`;
- puts `descriptor_model` and `bond_cutoffs` into the protocol's
  `relaxation_backend` block, so a MACE archive built with two-tier graphs
  cannot be resumed or merged with one built on legacy graphs.

g-xTB packs are untouched (`auto` → `legacy`), so existing archives and
their fingerprints are unaffected.

Guidance for the controller (`adaptive.py` still keys families on
`final["se_cn"]` / `cd_environments` / `core_hash`, which are now strong-tier
values under MACE):

- ZB-family protection: use `final["two_tier"]["zb_score"]` and
  `q4_nn4_se` measured on MACE endpoints; do not port g-xTB-era
  `Se CN4 fraction` targets.
- Artefact/reconstruction screening: `se_scaffold_components_all > 1`,
  `cd_detached_all` increments, `min_distances` against the run's artefact
  floors (skip the Se–Se floor for p=0).
- Rule mining (handoff step 4): condition on `n_cdse_long`,
  `se_with_long_contact` and `bridge_load_var` as well as `mu2/p`.
