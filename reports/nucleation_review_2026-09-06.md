# Critical review of Move Z nucleation and the two growth runs

Reviewed 6 September 2026, repository HEAD `fe63463`. This is an assessment; no implementation changes or new quantum-chemical calculations were made.

## Assessment

Move Z is a useful chemically informed generator of zinc-blende-derived Cd–Se–Cl clusters. It is not yet a predictive nucleation model. Its strongest features are neutral composition bookkeeping, explicit ligand-shell generation, unconstrained relaxation, and preservation of alternative lattice construction routes. Its chief limitations are restricted structural exploration, heuristic pruning, a conflation of connectivity with crystallinity, and insufficient separation between construction ancestry and physical reaction pathways.

The most consequential finding is an error in the lattice identity function. The existing ancestry diagnostic reports premature loss of the Wulff lineage. Independent integer-lattice matching gives a substantially different result.

## Scope and method

Inputs are the canonical `zb_occupations.jsonl` manifests in:

- `/Users/ivaninfante/Documents/University/Programs/Nucleation/graphs/growth_zb/growth_lattice_motifs_k1_to_k8`
- `/Users/ivaninfante/Documents/University/Programs/Nucleation/graphs/growth_zb/growth_agnostic_k5_k1_to_k6`

The older directory contains 44,361 distinct structure records and extends through k=13 despite its name. Its log ends during an optimization batch, so it is a partial snapshot of that continuation. The newer directory contains 10,939 records and its local log reports completion through k=6. Neither manifest contains duplicate structure IDs. Files with suffixes such as ` 2` were not counted as additional calculations. Some older canonical parent catalogs were overwritten by resume with empty lists, which prevents a complete historical parent-selection reconstruction from those files.

Coordination statistics below use stored final bond graphs. For the fixed-composition comparisons, only converged endpoints passing the code's chemical checks were included. Counts describe a biased search sample, not equilibrium populations or independent minima. Multiple starts/decorations can converge into the same basin. Energies are compared within identical (k,p) bins. Minima XYZ files were additionally checked with perturbed distance cutoffs.

The growth YAML is available, but `map_used.yaml` contains only a source-path comment. The historical graph/embedding settings and executable versions are therefore not fully snapshotted. Implementation findings refer to current HEAD; identical execution of every historical commit cannot be established from these files alone.

## 1. The ancestry diagnostic gives false negatives

`molecular_zb_growth.py::_occupation_shape_certificate` rounds absolute Cartesian coordinates on a 0.2 Å grid before removing translation. Translation changes rounding errors and thus the identity of identical fragments. Reproduction:

```python
points = [[0, 0, 0], [3.0702712, 3.0702712, 0]]
# certificate(points, tolerance=0.2):
# 7871c31a24ac984b72702cc5
# certificate(points + 0.05, tolerance=0.2):
# 7f60bac94c60f2ac173e862f
```

The same issue affects occupation identities and the anion-backbone keys used by the latest retention rules. It can split identical occupations into different IDs, consume diversity slots, duplicate work, and corrupt comparisons with the reference. The hash is not an invariant distance graph despite the function's introductory wording.

I independently removed translation, converted Se positions to exact integer lattice coordinates using a/4 = 1.5351356 Å, and canonicalized under the 48 signed axis permutations. The largest departure from integer coordinates across the run occupations was about 1.2e-15 lattice units, so this was not a loose geometric matching tolerance.

Number of distinct target-compatible Se backbones among topology-preserved, converged, chemically accepted endpoints:

| k | Older run | New run |
|---|---:|---:|
| 4 | 10 | 10 |
| 5 | 12 | 12 |
| 6 | 6 | 6 |
| 7 | 4 | — |
| 8 | 4 | — |
| 9 | 1 | — |
| 10 | 1 | — |
| 11–13 | 0 | — |

Thus the new run has not lost geometric compatibility by k=6. In the older run, compatible anion subsets occur through k=10; none occurs among the recorded k=11–13 jobs. This supersedes the existing script's claimed loss at k=4 or k=5. It does not prove an uninterrupted selected-parent path: per-size compatible subsets, physical intermediates, and actual parent-child reachability are different tests. Channel B can also introduce independent roots.

Composition makes the constraint stronger. With p_m >= 1 and s <= 2, p can decrease by at most one per growth step. Reaching (13,3) from (k,p) therefore requires p <= 16-k. At k=9 the older run's only preserved compatible endpoint has p=11, already incompatible with this bound. Clean lattice routes with reachable p remain at k=9. At k=10 all recorded compatible candidates have p=10–13: none can reach p=3 by k=13 under these moves. This is a real limitation of the move set, independent of the erroneous hash.

Fix identities before further tuning ancestry heuristics. Use exact lattice-site indices and symmetry operations, not coarse absolute-coordinate rounding. Add translated/permuted/rotated-fragment invariance checks. A proper ancestry audit should distinguish generated candidates, evaluated candidates, clean routes, converged minima, selected parents, and descendants, with (k,p) reachability attached to every edge.

## 2. What Move Z actually computes

The composition is Cd_(k+p)Se_kCl_(2p), which is formally neutral under Cd(II), Se(-II), and Cl(-I). Each step removes s CdCl2 equivalents, attaches one CdSe pair on vacant lattice sites, adds p_m Cd sites, reconstructs the entire 2p chloride shell, and optimizes. The formula p_child = p_parent - s + p_m is sound.

This is a construction operation, not a demonstrated elementary reaction. Borrowed-Cl removal can select a second chloride from anywhere on the parent. Full redecoration loses shell continuity. No barrier, ligand-transfer pathway, solvent reorganization, or kinetic competition is computed. A low Cd–Se Wiberg sum is a useful proposal heuristic but is not a desorption free energy or activation barrier. Bridging-Cl departure also disrupts bonds to retained Cd atoms.

The literature supports reversible metal-complex/Z-type exchange but shows strong dependence on ligand chemistry and solution conditions; it does not establish this particular combinatorial move as a mechanism. See [dynamic metal-carboxylate binding and displacement](https://pmc.ncbi.nlm.nih.gov/articles/PMC4102385/) and [anion-dependent Z-type displacement](https://pubmed.ncbi.nlm.nih.gov/39579139/).

Preserving the ideal occupation after its relaxed endpoint changes topology is valuable for exploration. However, the collapsed endpoint's energy belongs to that molecular geometry, not to the ideal crystalline intermediate propagated next. Keep construction nodes and relaxed minima separate and connect them by explicit relaxation mappings. Store reaction proposals separately from verified physical transformations.

## 3. Coordination patterns and search yield

Fraction of manifest endpoints meeting all current propagation conditions:

| k | Older run | New run |
|---|---:|---:|
| 2 | 98/400 = 24.5% | 104/384 = 27.1% |
| 3 | 238/1906 = 12.5% | 246/1905 = 12.9% |
| 4 | 156/2398 = 6.5% | 161/2475 = 6.5% |
| 5 | 156/2901 = 5.4% | 105/2865 = 3.7% |
| 6 | 76/3302 = 2.3% | 66/3310 = 2.0% |

The new branch improves some bin minima but not all: new-minus-old minimum energies are -0.708 eV at (4,6), -0.404 eV at (6,5), and +0.430 eV at (6,6). It is not established as generally superior, and the configuration changes are bundled rather than a controlled single-factor experiment.

Atom-weighted coordination at fixed composition, across converged chemically accepted endpoints:

| Quantity | Older (6,3), 171 endpoints | New (6,3), 169 endpoints | Older (13,3), 159 endpoints |
|---|---:|---:|---:|
| Mean Cd–Se CN around Cd | 2.018 | 2.032 | 2.550 |
| Cd with four Se neighbours | 2.0% | 2.8% | 9.7% |
| Mean Se–Cd CN | 3.027 | 3.047 | 3.139 |
| Se with four Cd neighbours | 22.6% | 21.5% | 28.6% |
| Cl bridging two Cd | 71.9% | 69.2% | 72.3% |

The dominant Cd environment at (6,3) is (nSe,nCl)=(2,1), followed by environments such as (1,2), (3,0), and (2,2). This makes total Cd CN a poor stand-alone proxy for crystallinity: CdSeCl3 and CdSe4 both have CN=4 but represent very different roles. Chloride bridging supplies much of the coordination. At fixed p=3 the inorganic network densifies with k, while Se remains mostly three-coordinate and many Cd remain surface-like.

Coordination bookkeeping itself imposes a useful relation: (k+p) times mean Cd–Se CN equals k times mean Se–Cd CN. At positive p, Cd must have lower average inorganic CN than Se. This asymmetry is not necessarily a defect.

The reference Cd16Se13 core has twelve Cd of Se-CN=2 and four of Se-CN=4; twelve Se of Cd-CN=3 and one of Cd-CN=4. Its mean Se CN is only 40/13 = 3.077. More four-coordinate Se does not automatically mean closer to this target. At (13,3), the older run's lowest converged accepted endpoint instead has Cd–Se CN counts {1:2, 2:6, 3:7, 4:1} and Se–Cd CN counts {2:1, 3:11, 4:1}. It is not the intended Wulff core.

Compare energy and structural features within (k,p), and preferably after basin deduplication. High-p clusters carry more passivating Cd and Cl; pooled correlations can mistake this compositional effect for an independent structural rule.

## 4. Connectivity is not crystallinity, nor is CN4 the nucleation event

The current `topology_status='preserved'` checks equality of Cd–Se edge lists. It does not require the relaxed geometry to fit ZB. Median preserved core RMSDs at k=2–4 are around 1.1–1.2 Å, and maxima exceed 2 Å. A flexible ring can preserve every bond while losing crystalline geometry.

Report separate descriptors: connectivity retention; optimally fitted lattice residual; local tetrahedral angular order; ring conformation; Cd- and Se-specific coordination; strain and shape. Test local order without demanding that a small surface-rich cluster be an undistorted bulk fragment.

The code comments calling a 3→4 coordination change “the nucleation event” overinterpret a local feature. A critical nucleus requires thermodynamic or dynamical evidence, such as a free-energy bottleneck and competition between growth and dissolution. The present irreversible size-growth search cannot identify that from a first CN4 atom.

At (6,3), the best preserved endpoint is 1.063 eV above the best accepted converged endpoint in the older run and 1.804 eV above it in the new run. Molecular reconstruction is competitive within the chosen Hamiltonian. This could reflect real small-cluster chemistry, sampling limitations, or method error; it should not simply be filtered away.

## 5. Chemical acceptance rules are part of the answer being imposed

The rules cap Cd and Se CN at four and enforce particular bridge arrangements. These are useful priors for tetrahedral CdSe-derived structures, but should not all be treated as universal chemical impossibilities after unconstrained optimization.

The new manifest reports mu3/host-bridge overlap for 4,681 jobs, shared-Cd-pair bridge violations for 3,479, inorganic disconnection for 2,615, and g-xTB abnormal termination for 794. Categories overlap. These outcomes should not be merged into one chemical-failure rate. In particular, 246 converged new-run endpoints, and 603 older-run endpoints, were rejected with only the shared-pair bridge-rule violation recorded. Those structures form a useful validation set; their acceptability is not established merely by convergence.

Recomputing bonds in the converged bin-winning XYZ files shows cutoff sensitivity. Changing Cd–Se 3.25 Å to 3.10 Å changes the bond count in 20/122 older-run winners and 5/34 new-run winners; increasing it to 3.40 Å changes 27/122 and 6/34. These are sensitivity tests, not alternative validated cutoffs. Use bond lengths together with WBO and continuous coordination, and distinguish weak contacts from robust bonds.

Geometry convergence is also not a vibrational minimum test. More immediately, 969 older-run and 34 new-run endpoints are marked chemically clean but unconverged. The code can propagate their occupations, and `_opt_zb_occupations` can insert their energies into `child_minima` and rankings. Preserve them as unfinished search routes if useful, but exclude their energies from reported minima until converged. The comparisons in this report excluded them.

## 6. Specific implementation improvements

1. **Repair lattice identities.** This affects both scientific interpretation and computational efficiency. Rebuild derived ancestry analyses afterward; do not assume old IDs correspond one-to-one to physical shapes.
2. **Correct ligand/core coordinate-frame handling.** `_growth_site_priority` counts relaxed Cl neighbours around ideal `parent.coordinates`. Core displacement is aligned separately, but Cl coordinates are not transformed with that alignment. For collapsed routes, ideal core positions and collapsed-shell positions are mixed. Use the actual relaxed graph for occupancy or consistently align the entire decorated parent. Remove shed ligands from post-shed occupancy feedback.
3. **Count the intended rings.** `_compactness_from_core` counts six-membered members of `networkx.cycle_basis`, not all six-rings. The result can depend on the chosen basis/node order in fused cages. Reuse an invariant short-ring census and cache it.
4. **Apply caps to distinct candidates.** `attach_cdse` truncates before symmetry deduplication; `_add_precursor_cd` also prunes before final symmetry reduction. Equivalent orientations can consume the cap. Use exact integer lattice keys, deduplicate first, then select diverse representatives.
5. **Retain multiple reactive shells per occupation.** `_select_zb_occupation_parents` first chooses one representative per occupation. Other decorations can expose different leaving groups or attachment sites even when their core is identical. Preserve a small diversity of shell/reactivity classes.
6. **Make cross-p rescue genuinely per-bin and size-aware.** `cross_p_retention` uses a hard-coded p=5 division. Presence in any lean bin suppresses rescue in all lean bins, despite the per-bin intent; rescue also scans original parents and can bypass the normal min-p and cap filtering. Use explicit coverage targets, coordination deficits, and reservoir conditions. Below the compactness threshold, an early return means the reserved-seat code is not used.
7. **Audit successive ranking bottlenecks together.** Compact attach/pad selection, parent selection, and shell selection optimize different proxies. Shell ranking penalizes saturated anions while growth promotes coordination completion. A soft score followed by a hard cap can still exclude a whole class. Track family survival after every stage and allocate explicit exploration slots.
8. **Make resume reproducible.** The motif completion check matches (k,p) without a configuration hash. Save expanded graph/embedding settings, source commit, executable version, seeds, budgets, and parent IDs; distinguish completed bins produced under different protocols.

## 7. Physical extensions and efficient next experiments

Keep Move Z as a crystalline-proposal channel, but couple it to relaxed-coordinate growth/reconstruction and, if general CdSe nucleation is the goal, competing WZ/stacking-fault and noncrystalline channels. Permit fixed-k ligand exchange and CdCl2 desorption/readsorption; tying all ligand loss to CdSe addition is what makes some passivated branches unable to reach lean endpoints. Reversible moves and selected oligomer attachment are natural subsequent extensions.

The first benchmarks should be small and diagnostic: corrected ancestry at k=4–6 and k=8–10; translation-invariance checks; dedup-before-cap versus current caps; a representative set of bridge-rule rejections; and constrained versus unconstrained relaxation of the reference cluster. Measure distinct clean basins, structural diversity, lineage coverage, and discovery per CPU-hour, not only preserved-job percentage.

Use a modest DFT reference set spanning the Wulff candidate, competing molecular minima, undercoordinated surface motifs, selected higher-CN/rejected structures, and CdCl2 removal energies. g-xTB is a broad general-purpose method, but its general benchmarks do not establish the relevant Cd–Se–Cl ordering: [g-xTB method description](https://www.cambridge.org/engage/chemrxiv/article-details/685434533ba0887c335fc974). Calibrate bond definitions and rank uncertainty against this set before learning stronger pruning rules.

Finite-temperature solution thermodynamics requires more than the present electronic-energy grand-potential tables. Include appropriate precursor chemical potentials, solvation, entropy, and speciation for the intended synthesis. For a package-addition proposal, the relevant balance is G_child + s*mu_CdCl2 - G_parent - mu_package. Absolute energies across different p cannot be compared directly. Chloride-only models also omit the sterics and dynamics of organic shells. The experimentally demonstrated influence of ligands on CdSe polymorph selection supports keeping this uncertainty explicit: [surface versus interior control](https://pubs.acs.org/doi/10.1021/ja5020025).

For efficiency, first eliminate false duplicate occupations, cache lattice/shape/decoration work, retain all provenance when merging routes, and adapt shell budgets to distinct-basin discovery with a minimum exploration allowance. Distinguish numerical failures, fragmented endpoints, uncertain motif-rule rejections, and usable molecular minima. More workers help throughput, but a scientifically defensible reduction of duplicate and uninformative jobs is more valuable than simply increasing the current fixed 400-shell budget.

The observed link between low Cd coordination and mobile/reconstructing ligand environments is consistent with [Cosseddu et al., Ligand dynamics on the surface of CdSe nanocrystals](https://pubs.rsc.org/en/content/articlehtml/2023/nr/d2nr06681e). That study concerns larger carboxylate-capped particles, so it supports the qualitative interpretation rather than validating these chloride-cluster pathways directly.

Recommended order: repair identity and reporting; re-audit existing data; validate chemical rules and energetics on a small reference set; then add reversible shell/reconstruction moves and controlled budget ablations. Continuing the present run farther may collect useful structures, but it cannot establish a nucleation mechanism or resolve biases already introduced at earlier stages.
