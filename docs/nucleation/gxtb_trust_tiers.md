# g-xTB archive trust tiers for MACE import

Operational definitions for the tier table in
`CLAUDE_HANDOFF_NUCLEATION_MACE.md`. Implemented by
`tools/gxtb_tier_classify.py`; the numbers below are the ones that script
uses, and the observed counts come from running it on the k9 and k10
canonical archives on 2026-09-15.

## Two orthogonal labels, not one ladder

The handoff lists A–D as one ladder, but A answers a different question from
B/C/D, and a structure can be both. Every minimum therefore gets two labels:

| label | question | values |
|---|---|---|
| **record tier** | Is the construction/lineage record internally valid, so the structure can seed family tracking regardless of how its endpoint looks? | `A` / `-` |
| **endpoint tier** | How far can the relaxed g-xTB geometry itself be trusted as a MACE starting point? | `B` / `C` / `D` |

Combined tier strings are `AB`, `AC`, `AD`, `-C`, `-D`. `-B` cannot occur
(an invalid record cannot have a clean endpoint by construction, see below).

All labels are computed from fields already present in `minima.json`
(`construction`, `final`, `violations`, `converged`, `positions`) plus
pairwise distances. No energies are compared across protocols; the g-xTB
energy is carried along as `gxtb_energy_eV` for within-bin historical use
only.

## Record tier A — construction/lineage record is valid

All of the following must hold:

| check | field | rule |
|---|---|---|
| composition | `symbols`, `k`, `p` | `n_Cd = k+p`, `n_Se = k`, `n_Cl = 2p` |
| graph present | `edges` | non-empty |
| certificates | `construction.core_hash`, `construction.graph_hash` | both present |
| connected construction | `construction.n_components`, `construction.core_components` | both `== 1` |
| ancestry | `routes` or `source_minimum_ids` or `occupations` | at least one non-empty |

Any failure → record tier `-`. Observed: 171/12058 (k9) fail, all on
`construction_disconnected` — these are audit-derived structures whose Cd/Se
core was already split at construction time under a relaxed policy. They are
retained for audit but must not seed Se-family tracking.

**Caveat.** `occupations` is empty for all but 2 records in the merged
canonical archives, so the ZB occupation ID the handoff asks to preserve is
*not* recoverable from `minima.json`. Tier A relies on `core_hash` /
`graph_hash` / `routes` instead. If lattice-derived occupation IDs are
needed, they must be pulled from the per-run `events.jsonl` proposal records
during import.

## Endpoint tiers B / C / D

Assignment starts at `B` and is demoted by the first failing rule in the
column shown. Rules are evaluated in order; every triggered rule is appended
to `reasons` so the demotion is auditable.

### Demote to D — artifact, reconstruction, or unusable parent

| rule | field / threshold |
|---|---|
| missing geometry or energy | `positions` empty or `energy_eV` absent |
| optimizer failure | `converged == False` and status not `maxcycle` |
| disconnected cluster | `final.n_components != 1` |
| inorganic core split | `final.core_components != 1` (equivalently violation `inorganic_disconnected`) |
| hard violation | any of `inorganic_disconnected`, `min_cn` |
| artifact contact | min Cd–Cd, Se–Se, Cl–Se, or Cl–Cl distance `< 2.80 Å` (same floors as `run_mace_mh1.yaml: artifact_min_distance`) |
| collapsed bond | min Cd–Se `< 2.25 Å` or Cd–Cl `< 2.15 Å` |
| scaffold lost | `construction.core_hash != final.core_hash` **and** (core split, or ≥ 25 % of Se atoms with CN ≥ 3 lost CN ≥ 3) |

Bond floors are set ~0.1 Å below the 1st percentile of relaxed g-xTB
distances (Cd–Se 2.37 Å, Cd–Cl 2.29 Å over 12 k structures), so they only
trip on genuinely collapsed contacts. The artifact floors are inherited
unchanged from the MACE run configuration so that D under g-xTB and
"artifact-pruned" under MACE mean the same geometric thing.

Permitted use: audit only. Never a primary parent, never re-optimized as
part of a beam quota. May be re-optimized in a dedicated *diagnostic* batch
to see whether MACE repairs the reconstruction — but such results go into a
diagnostic namespace, not the population.

### Demote to C — usable geometry, ranking not trusted

| rule | field / threshold |
|---|---|
| maxcycle endpoint | status `maxcycle` (accepted under `accept_maxcycle`) |
| scaffold changed | `construction.core_hash != final.core_hash` but not lost per the D rule |
| soft violation | any of `mu3_host_bridge_overlap`, `max_cn`, `unsupported_motif`, `bridges_per_cd_pair`, `min_bridged_host_cn`, `cl_on_cn4_cd`, `mono_se_dual_terminal` |

Permitted use: the g-xTB energy ranking within one `(k,p)` bin is
historical/provisional only. The geometry *is* a legitimate diverse MACE
seed — especially `scaffold_changed` structures, which are exactly the ones
where MACE may disagree with g-xTB about basin identity. Tier C structures
enter the MACE population only after re-ranking with MACE.

Unknown violation tags are treated as soft but flagged
(`endpoint:unknown_violation:*`) so the list can be extended deliberately
rather than silently.

### B — clean endpoint

Everything above passes: converged, single connected cluster with connected
Cd/Se core, no violations, no short contacts, and the exact pre-relaxation
Se-scaffold certificate (`core_hash`) is preserved through relaxation.

Permitted use: diverse MACE starting geometry and default primary parent.
B corresponds to `role == primary` **and** scaffold retained; in the k9
archive 2662 of the 5917 `primary` minima are B, the other 3227 are C because
their core hash changed (the archive's `primary` role only tracks
violations, not scaffold retention).

## Observed distribution

Endpoint tier by `k` (k10 canonical archive, 15 525 minima; k≤9 rows are
identical in the k9 archive because k10 was merged on top of it):

| k | B | C | D |
|---|---:|---:|---:|
| 1 | 9 | 0 | 4 |
| 2 | 75 | 61 | 44 |
| 3 | 40 | 117 | 19 |
| 4 | 181 | 327 | 53 |
| 5 | 478 | 1113 | 73 |
| 6 | 672 | 1935 | 96 |
| 7 | 258 | 1251 | 35 |
| 8 | 372 | 1640 | 79 |
| 9 | 577 | 2430 | 119 |
| 10 | 565 | 2786 | 116 |
| **all** | **3227** | **11624** | **674** |

Two things stand out:

1. **Only ~21 % of endpoints are B.** The dominant reason for C is
   `scaffold_changed` (core hash differs after relaxation), not chemistry
   violations. That is the headline result for the MACE transition: g-xTB
   moved the Se scaffold in most relaxations, so the "which structures are
   ZB-related" question genuinely has to be re-asked with MACE.
2. **D is small (4 %)** because the canonical archives were already
   artifact-pruned upstream. The D rules will matter more when importing raw
   per-run archives (`nucleation_phase_c_*_v1/minima.json` before merge) or
   the older `growth_*` runs.

Full reason counts are in the `--summary` output of the classifier.

## How the tiers feed the stratified re-optimization subset (handoff step 2)

Per `(k,p)` bin with `k ∈ {7,8,9,10}`:

- all `AB` primaries (bounded: top-N by g-xTB energy plus one per distinct
  `construction.core_hash`);
- an equal number of `AC` structures chosen to maximise distinct
  `final.core_hash` — these test whether MACE agrees with g-xTB's scaffold
  rearrangements;
- a small fixed quota (e.g. 3) of `AD` with reason `scaffold_lost` as
  diagnostics, in a separate namespace;
- nothing from `-C` / `-D` in the first pass.

Every imported structure keeps `tier`, `reasons`, `construction_core_hash`,
`final_core_hash`, `source_protocol`, and `gxtb_energy_eV`, and gets
`mace_status: pending` until the MACE endpoint and MACE protocol fingerprint
are attached.

## Running it

```bash
python tools/gxtb_tier_classify.py \
  /Users/ivaninfante/Documents/University/Programs/Nucleation/graphs/growth_zb/nucleation_phase_c_k10_canonical_v1/minima.json \
  --out runs/mace_import/k10_tiers.jsonl --summary runs/mace_import/k10_tiers.md
```

Multiple `minima.json` paths can be given; each row records its `archive`.
