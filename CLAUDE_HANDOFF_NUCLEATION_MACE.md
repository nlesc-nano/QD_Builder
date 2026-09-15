# Nucleation project handoff: MACE-MH-1 production transition

## Executive decision

MACE-MH-1 is intended to replace g-xTB as the production geometry optimizer for
the CdSe/CdCl2 nucleation and growth search. This is a methodological decision,
not a request to discard the existing g-xTB work.

The existing g-xTB archive is retained as:

- a large source of starting geometries and alternative lineages;
- a record of which graph constructions, decorations, and growth transitions
  were attempted;
- a source of motif and coordination statistics;
- a historical comparison against the previous potential-energy surface.

It is **not** treated as the final authority for absolute energies, ranking, or
which structures are chemically plausible. Structures that enter a new MACE
population must be re-optimized and re-ranked consistently with MACE.

## Scientific objective

The project studies neutral, charge-balanced, closed-shell clusters of the form

```text
Cd_(k+p) Se_k Cl_(2p)
```

using graph construction, embedding into 3-D, and geometry optimization. `k`
primarily controls the Se skeleton; `p` controls the additional Cd and Cl
decoration.

The target is not simply the lowest-energy arbitrary cluster. The search should
find chemically plausible growth families that retain zinc-blende-related
patterns as `k` increases: tetrahedral/adamantane-like units, T1 and fused cage
motifs, coherent Se frameworks, sensible Cd coordination, and realistic
chloride bridging. The search must remain broad enough to avoid losing lineages
but bounded enough to avoid combinatorial explosion.

The number of `mu2` chloride bridges is an important empirical descriptor. The
existing work found a strong relation close to `n_mu2 ~ 2p` on relaxed
structures. This is a useful prior, not a hard universal rule.

## What has already been done

The nucleation code has completed several adaptive Phase A/B/C searches,
including broad and targeted continuation through approximately `k=9` and
`k=10`. These runs generated thousands of structures in the `growth_zb`
folders and related nucleation archives.

The previous controller added:

- bounded proposal budgets;
- Se-lineage and cross-`p` family tracking;
- target-compatible continuation;
- low-`p` lineage repair;
- high-`p` transition handling;
- canonicalization and duplicate consolidation;
- resumable, checkpointed searches.

Those data and ancestry records should be reused. Do not restart blindly from
`k=1`, and do not regenerate equivalent populations only because the optimizer
has changed.

## MACE backend status

Commit:

```text
7d58b09 feat(nucleation): add persistent MACE relaxation backend
```

Relevant files:

- `src/builder/nucleation/xtb_relax.py`
- `src/builder/nucleation/pilot.py`
- `tools/mace_worker.py`
- `tools/compare_mace_existing.py`
- `geometry_packs/cdse_cdcl2_zb/run_mace_mh1.yaml`

The worker follows the established test recipe:

1. load `/Users/ivaninfante/.cache/mace/macemh1model` on CPU;
2. convert the checkpoint to float32 before constructing the calculator;
3. use head `omat_pbe`;
4. use `device=mps` and `default_dtype=float32`;
5. isolate the MPS-safe `Tensor.double()` workaround inside the worker;
6. keep one model resident and process structures through a persistent queue.

MACE and g-xTB have separate protocol identities and energy namespaces. Their
raw total energies must never be mixed in one ranking.

The current managed environment reports MPS unavailable despite arm64 and
`mps_built=True`; a direct MPS tensor allocation fails. This appears to be the
known PyTorch/macOS 26 compatibility problem, not a MACE integration choice.
The production run must use MPS once the environment preflight succeeds; it
must not silently fall back to CPU.

## How to reuse the g-xTB archive without trusting it blindly

Every old structure should be assigned both its original g-xTB provenance and
its future MACE status. Preserve:

- exact construction graph and occupation ID;
- exact Se scaffold ID and cross-`p` family ID;
- Cd realization and Cl-decoration ID;
- input geometry and g-xTB endpoint;
- g-xTB energy, convergence status, Wiberg data, and artifact flags;
- MACE input/endpoint and MACE protocol fingerprint;
- DFT endpoint/reference information where available.

Use the following trust tiers for old g-xTB results:

| tier | interpretation | permitted use |
|---|---|---|
| A | construction/lineage record is internally valid | always reuse as a seed or ancestry record |
| B | g-xTB endpoint has plausible distances, connectivity, and motifs | use as a diverse MACE starting geometry |
| C | g-xTB energy ranking within one fixed `(k,p)` bin | historical/provisional only; re-rank with MACE |
| D | artifact, severe reconstruction, disconnected or failed endpoint | retain for audit, do not use as a primary parent |

The old archive should therefore seed MACE in two complementary ways:

1. re-optimize selected g-xTB endpoints to test how MACE changes their basins;
2. re-embed/redecorate selected construction graphs and optimize directly with
   MACE, so the comparison is not biased by g-xTB geometry distortion.

Do not only carry forward the lowest g-xTB structures. Preserve a quota of
lineage-diverse, motif-diverse, and cross-`p` representatives.

## Mining additional decoration rules

The existing archive is large enough to mine decoration priors before doing a
new broad search. The goal is to discover conditional rules, not to replace the
chemical search with a rigid score.

For every `(k,p)` and Se scaffold family, calculate at least:

- `n_mu2`, `n_mu3`, terminal Cl count, and isolated Cl count;
- bridge-load vector per Cd, its mean, maximum, and variance;
- number of Cd atoms with one bridge, multiple bridges, or mixed terminal/bridge
  coordination;
- Cd–Se, Cd–Cl, and Cl–Cd–Cl distance/angle distributions;
- Cd and Se coordination-number histograms;
- Cd/Se coordination deficits relative to the intended local target;
- Se CN4 fraction and Se tetrahedrality;
- Cd CN3/CN4 fractions and Cd tetrahedrality;
- ZB/Se-scaffold retention and motif identities before and after relaxation;
- scaffold survival across `p` for the same Se skeleton;
- duplicate-basin collapse rates;
- artifact/short-contact/disconnection rates;
- relaxed endpoint quality and convergence status.

For each proposed rule, report conditional survival/yield rather than a global
correlation. Examples:

- probability of retaining the Se scaffold given a bridge-load variance range;
- probability of preserving Cd CN3/CN4 motifs given `n_mu2/p`;
- whether a terminal chloride is beneficial when local bridge load is already
  saturated;
- whether distributed bridges outperform clustering at each `k`;
- whether the same Se scaffold prefers different decorations at different `p`;
- which decoration patterns repeatedly collapse to the same relaxed basin.

The `n_mu2 ~ 2p` relation should become a soft prior with an exploration tail:
sample near the relation heavily, but always retain some below/above-relation
decorations so that new regimes are not excluded.

Rules must be mined on a training subset of structures and checked on held-out
families or later `k` values. Otherwise the controller will merely rediscover
the bias already imposed by the old proposal generator.

## Preventing combinatorial explosion

The primary unit of search should be a Se scaffold family, not an independent
full Cd/Cl structure at every `(k,p)`.

Recommended hierarchy:

```text
Se scaffold family
  -> a small set of Cd realizations at each p
      -> a compact, deliberately diverse Cl decoration panel
          -> MACE relaxation and basin consolidation
```

Use bounded beam quotas per Se family and per `p`:

- retain the best MACE candidates;
- retain at least one representative for every surviving scaffold family;
- retain motif-diverse and decoration-diverse audit candidates;
- retain a fixed fraction of g-xTB-diverse seeds even when their old energy is
  mediocre;
- stop expanding a cell when new structures repeatedly collapse into existing
  basins;
- expand a cell when a new Se family survives or a new decoration motif is
  observed;
- allocate a reserved exploration fraction at every `k`.

Energy ranking is only within the same composition and MACE protocol. Family
coverage and scaffold survival must be explicit selection objectives alongside
energy.

## Preserving zinc-blende growth patterns

At every growth step, retain the exact pre-relaxation Se scaffold certificate.
Do not infer lineage from the relaxed endpoint alone.

For each child, record:

- parent Se scaffold and added Se site/unit;
- ZB occupation ID when lattice-derived;
- geometric embedding and its orientation;
- Cd realization and chloride decoration ancestry;
- relaxed Se-family classification;
- whether the endpoint retained, distorted, or lost the scaffold.

Selection should protect ZB-related families even when they are not the absolute
lowest-energy MACE minima. A family that is slightly higher in energy but
continues coherently from `k` to `k+1` is scientifically more valuable than an
isolated collapsed minimum.

## Different decoration policies at low and high p

The decoration search should not use one fixed rule for all `p`.

At low `p`, the search should emphasize:

- bridge-first and near-`2p` decorations;
- sparse, well-separated bridge placement;
- completion of the most undercoordinated Cd/Se environments;
- preservation of the ZB-derived core;
- a small number of terminal-Cl alternatives.

At high `p`, blindly adding more of the same bridge pattern can create strained
starting geometries and cause optimizer crashes or artificial reconstructions.
High-`p` decoration should therefore include:

- distributed bridge load rather than bridge clustering;
- alternating bridge orientations and lower load variance;
- mixed bridge/terminal arrangements that relieve local overcoordination;
- decorations that cap exposed surface sites without forcing extra Cd–Se bonds;
- multiple spatially distinct placements with the same global `n_mu2`;
- a deliberately broader diversity quota than at low `p`;
- pre-relax distance/coordination screening and smaller MACE multi-start batches.

The high-`p` policy is not permission to abandon the growth family. It is a way
to reach the same Se lineage with chemically safer ligand arrangements. A
high-`p` cell should be judged by scaffold retention and stable optimization as
well as energy.

## Recommended MACE production sequence

1. Fix and verify the MPS environment; record the exact Torch/MACE/ASE versions.
2. Re-optimize a stratified, already existing subset from `growth_zb` and the
   CP2K archive with MACE.
3. Compare MACE endpoints to DFT endpoints using geometry, coordination,
   scaffold retention, and motif descriptors—not raw cross-method energies.
4. Mine decoration rules and define soft priors plus explicit exploration
   quotas.
5. Import old g-xTB structures into a new MACE archive with source IDs and
   trust tiers.
6. Run targeted continuation from the best existing Se families at the first
   difficult growth transitions (`k=8 -> 9`, then `9 -> 10`).
7. Analyze each epoch automatically and checkpoint a next-stage manifest.
8. Extend to broader `k` only after the MACE run demonstrates family continuity,
   not merely after reaching an energy plateau.

The decision to use MACE as the production optimizer is already made. DFT and
g-xTB are reference/comparison sources used to understand confidence and to
preserve the accumulated search effort; they are not competing production
optimizers unless a specific diagnostic requires one.

## Automation and restart requirements

The local controller should be durable without an active Claude/Codex session:

```text
load/import old structures
-> generate bounded proposals
-> MACE-relax through one persistent MPS worker
-> atomically write each result
-> classify scaffold/motif/decoration outcomes
-> consolidate basins
-> update rule statistics
-> allocate next bounded epoch
-> checkpoint and continue/stop at a predefined gate
```

Each result must be journaled with proposal ID, ancestry, backend fingerprint,
attempt number, status, coordinates checksum, and analysis descriptors. A killed
process resumes pending/interrupted work and never relies on a human reading a
log to determine completion.

The optimization process itself consumes no language-model tokens. Claude can be
invoked sparsely to inspect completed epochs and review reports; the local
controller must not depend on continuous agent polling.

