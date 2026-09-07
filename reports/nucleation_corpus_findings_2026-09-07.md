# Growth-corpus findings, 2026-09-07

This report summarizes the geometry-aware audit in `runs/nucleation_corpus_audit_2026-09-07`. It describes the structures found by historical searches; it is not a thermodynamic ensemble and cannot provide nucleation rates or free energies.

## Main result

The empirical μ2≈2p relationship is reproducible enough to use as a proposal prior, but the data do not support enforcing it as a constraint. In `growth_agnostic_k5_k1_to_k6`, the 34 fixed-(k,p) energy winners have μ2 = 0.955(2p) − 0.749 (r=0.986). Their deficits `2p−μ2` are 0 for 8 bins, 1 for 14 and 2 for 12. The older `growth_lattice_motifs_k1_to_k8` run gives a slope of 0.914 and r=0.989 across 122 bins, but 26 winners have deficits larger than two (maximum six).

This is not merely a pooled-composition effect: among geometry-deduplicated groups with fixed run, k, p and inorganic-core graph, 97/113 agnostic fits favor more μ2 chloride and 16/113 favor less; the median slope is about −0.300 eV per added μ2. The corresponding older-motif result is 120/138 versus 18/138, with a median near −0.294 eV/μ2. These are electronic-energy associations among sampled structures, not bridge formation energies. Multiple testing, restricted generation and residual geometric differences within a graph family remain confounders.

Leave-one-run-out checks predict held-out winner μ2 counts with mean absolute errors of roughly 0.42–0.87 bridges. Near-maximum coverage is 0.78–1.00 depending on the held-out run. The split excludes exact species-pair-distance signatures at 0.001 Å but is not truly independent: related generators, ancestry and near-duplicate geometries can cross folds. A prospective randomized pilot remains necessary.

## Coordination patterns

Among 2,677 accepted agnostic endpoints, the pooled atom counts are:

- Cl: 4,835 terminal (26.6%), 13,217 μ2 (72.7%) and 120 μ3 (0.7%). Thus μ2 dominates, while terminal chloride remains a substantial minority and μ3 is rare under these generators and filters.
- Se: 2,862 CN2 (22.9%), 6,371 CN3 (50.9%) and 3,276 CN4 (26.2%). Most Se remains surface-like at these small sizes.
- Cd, described as `(CN_Se,CN_Cl)`: the largest classes are (1,2)=5,726, (2,1)=5,707, (1,3)=2,854 and (2,2)=2,502. Only 535/21,595 Cd atoms (2.5%) are (4,0). A bulk-like tetrahedral Cd population is therefore not the dominant motif at k≤6.

The older motif corpus shows the same qualitative picture: 52,892 μ2, 18,403 terminal and 569 μ3 chlorides among accepted endpoints. These pooled counts weight large structures and heavily sampled bins more strongly; they must not be read as equilibrium fractions.

Accepted agnostic structures have bridge deficits 0–2 in 2,035/2,677 cases (76.0%), whereas the older motif run has 6,019/9,016 (66.8%). The agnostic generator is therefore more concentrated around the bridge-rich region even before restricting to winners.

## Relaxation and construction bias

Bridge topology is not stable during relaxation. Across all 10,939 agnostic attempts, μ2 is unchanged in 2,353, increases in 2,151 and decreases in 6,435. Large negative changes occur, so construction-time bridge count should be treated as an initial-condition descriptor, never as the relaxed answer. These transition counts include rejected and failed structures; they diagnose the proposal-to-relaxation map rather than stable chemistry.

Starting shells containing terminal chloride dominate the archive by construction (10,382/10,939 agnostic attempts). Their accepted fraction is 24.2%, versus 29.8% without terminal chloride; their clean/converged connectivity-preserving fraction is 6.0%, versus 11.5%. The comparison is confounded by composition, core and proposal route, but it justifies stratified sampling and reporting rather than silently dropping terminal-shell candidates.

The most frequent recorded rejection is `bridges_per_cd_pair:2>1` (45,288 occurrences), followed by disconnected inorganic cores. This shows that a large fraction of compute was spent constructing arrangements already outside the heuristic rule set. The new bounded enumerator enforces graph capacity and symmetry deduplication before embedding, but rule-violating, connected, converged endpoints are retained in a small audit population because relaxation can invalidate construction-time labels and because the rules themselves are hypotheses.

## Wulff ancestry

With translation-corrected lattice identity, `growth_agnostic_k5_k1_to_k6` contains compatible clean, converged Se sub-backbones through k=6: 1 at k=2, 4 at k=3, 10 at k=4, 12 at k=5 and 6 at k=6. The saved selected-parent counts are 1, 4, 8, 7 and 0 respectively. The k=6 zero may reflect overwritten/missing selection metadata, and compatible backbones at each size do not prove a continuous selected-parent path. Compatibility is only a necessary condition for monotone Se-sublattice growth toward Cd16Se13Cl6; it says nothing by itself about Cd/Cl recovery, barriers or kinetics.

## Consequence for the pilot

Use 80% of chloride-shell proposals near the feasible maximum and next two μ2 tiers, with 20% exploratory proposals. Keep terminal and μ3 alternatives, and compare the corrected lattice arm against the mixed graph/search arm at matched backend calls and worker time. Select parents separately within (k,p), cap repeated core geometries, retain energy/diversity/exploration slots, and keep heuristic-rule violators in a separately budgeted audit lineage. Do not adapt this prior from the same pilot used to evaluate it.

The first decision point should be after the bounded k≤6 comparison, not after a single low-energy structure. Expand only if the experimental arm improves independent-family discovery, fixed-composition best energies and rejection efficiency without collapsing to one core/shell family. Revalidate representative minima with solvent/thermal corrections and a higher-level electronic-structure method before using them to infer chemical potentials or a nucleation free-energy landscape.
