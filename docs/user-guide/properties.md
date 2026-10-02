(guide-properties)=
# Computing properties of library structures

`qdprops` is a separate package in the repository that takes library records
(a folder with `record.json` and `start.xyz`, see {ref}`guide-library`),
relaxes them and computes ground-state properties. It uses two engines:

- **MACE-MH-1**, a universal machine-learned potential, for the potential
  energy surface: the relaxation, the forces and the Hessian. The model has
  several heads trained on different reference data; a run uses one head for
  everything (default `omat_pbe`) and records it.
- **GFN2-xTB** (`xtb --gfn 2`) for electronic properties at the MACE
  geometry. g-xTB from the bleeding-edge build can be selected instead
  (`Settings.xtb_method = "gxtb"`).

```bash
PYTHONPATH=src python -m qdprops run path/to/CdSe-Se-Cd16Se13Cl6-clean
PYTHONPATH=src python -m qdprops batch path/to/library --max-atoms 500
```

## Environment

Everything runs in the `qd_builder` conda environment, which needs `torch`
and `mace-torch` (0.3.16) next to the builder's dependencies. On macOS the
pip torch wheel bundles its own OpenMP runtime, which clashes with the conda
`llvm-openmp` already loaded by numpy and scipy (`OMP: Error #15`). The fix is
to make torch use the environment's runtime:

```bash
cd $CONDA_PREFIX/lib/python3.11/site-packages/torch/lib
mv libomp.dylib libomp.dylib.torch-bundled
ln -s $CONDA_PREFIX/lib/libomp.dylib libomp.dylib
```

The MACE model is read from `QDPROPS_MACE_MODEL`, by default
`~/.cache/mace/macemh1model`; xtb from `QDPROPS_XTB` (default: the `xtb`
conda env). The g-xTB binary is read from `QDPROPS_GXTB`, and the extra
dynamic-library directories it needs from `QDPROPS_XTB_LIBS`
(`:`-separated). Reference species are cached in `QDPROPS_REFS`
(default `~/.cache/qdprops/references`). MACE runs in float64 on the CPU by default; the Apple GPU
(MPS) has no float64, which the Hessian needs.

## Steps

| Step | Engine | What it computes |
|---|---|---|
| `relax` | MACE-MH-1 | BFGS (then FIRE if needed) to `fmax` = 0.01 eV/Å; relaxation energy, RMSD and largest displacement from the start geometry |
| `structure` | — | bond graph of start and relaxed geometry, ligand detachment or migration, CN histograms, core/surface atoms, ligand binding modes (μ1/μ2/μ3), bond-length distributions for core and shell, core strain against the bulk bond |
| `hessian` | MACE-MH-1 | harmonic frequencies, imaginary modes, ZPE, U_vib, S_vib, Cv and F_vib from 50 to 800 K, vibrational density of states (total, per element, per core/surface/ligand) |
| `vibspec` | MACE-MH-1 + g-xTB | IR intensities and non-resonant Raman activities, depolarisation ratios, point group and irreps of the modes, mode character (core/surface/ligand, breathing), bulk Γ-point optical frequency |
| `electronic` | GFN2-xTB | total energy, HOMO, LUMO and orbital gap, partial charges, dipole moment, the xtb force at the MACE minimum, vertical IP and EA (`xtb --vipea`) and the fundamental gap IP − EA |
| `stability` | MACE-MH-1 | decomposition into bulk MA and MX_q monomers, binding against MA and MX_q monomers: ΔE, ΔE + ΔZPE, ΔG(T) |
| `detachment` | MACE-MH-1 | stepwise Z-type MX_q removal by a beam search: ΔE and ΔG(T) per step, every symmetry-distinct first site with its facet/edge/vertex location, the equilibrium shell ⟨k⟩(T, Δμ_MXq) |
| `solvation` | GFN2-xTB | Generalized Born solvation of the dot, every stripped state and the monomers, as a function of ε; ddCOSMO and ALPB checks |
| `report` | — | `ground_state.html` (interactive, the webapp's Properties tab) and `ground_state.png` |

Bonds are opposite-charge pairs closer than 1.2 times the bulk cation–anion
bond for native pairs, and 1.25 times the covalent-radius sum for ligand
pairs. The builder's own cutoffs are tighter (they are tuned to ideal cut
geometries); relaxation stretches some bonds by several per cent.

The Hessian is the analytic MACE one up to 300 atoms and a central finite
difference of forces (δ = 0.01 Å) above. It is mass-weighted, and the
translations and rotations are removed exactly by diagonalising it in the
orthogonal complement of their vectors. The thermochemistry is that of
harmonic oscillators over the real modes. The VDOS is broadened with a
5 cm⁻¹ Gaussian and projected with the squared mass-weighted amplitudes.

The IP and EA come from `xtb --vipea`: the IPEA-xTB ΔSCC with its empirical
shift. GFN2 absolute orbital levels are shifted, so a plain ΔSCF with GFN2
overestimates both (13.0 and 7.5 eV for Cd16Se13Cl6 against 6.3 and 1.3 eV).
Their difference, the fundamental gap, is more meaningful than the orbital
gap.

## Vibrational spectra

Peak positions are the MACE-MH-1 harmonic frequencies; g-xTB supplies only the
response derivatives. For each normal mode k the dot is displaced to
x₀ ± h e_k/√m (h = 0.1 amu^½ Å) and g-xTB is run in a static field ±F
(F = 0.02 V/Å) along x, y and z, six runs per geometry. The dipole μ is the
mean of each ± pair, the polarisability α_ij = [μ_i(+F e_j) − μ_i(−F e_j)]/2F,
and central differences give ∂μ/∂Q_k and ∂α/∂Q_k:

- IR intensity A_k = (N_A π/3c) |∂μ/∂Q_k|², in km/mol;
- Raman activity S_k = 45a′² + 7γ′², with a′ = tr(α′)/3 and
  γ′² = ½[(α′xx − α′yy)² + (α′yy − α′zz)² + (α′zz − α′xx)² + 6(α′xy² + α′yz² + α′xz²)];
- depolarisation ratio ρ_k = 3γ′²/(45a′² + 4γ′²), below ¾ only for totally
  symmetric modes;
- Stokes intensity (ν₀ − ν_k)⁴/ν_k · S_k/(1 − e^(−hcν_k/k_BT)) at 532 nm and 300 K.

Points to know about g-xTB here:

- `--efield` is in V/Å, not atomic units as `--help` says: with V/Å the
  induced dipole and the energy change −½αF² give the same α (1038 bohr³,
  154 Å³, for Cd16Se13Cl6).
- The flag needs a space (`--efield 0,0,0.02`); `--efield=…` is silently ignored.
- `--alpha` does nothing with `--gxtb`, and GFN2 rejects fields in xtb, so the
  finite field with g-xTB is the only route.
- Each run starts from the field-free wavefunction of the undisplaced dot. A
  restart file from a field run gave a slightly asymmetric α.
- g-xTB is not at its own minimum at the MACE geometry (gradient norm
  ≈ 0.1 Eh/bohr for Cd16Se13Cl6). This is the usual hybrid compromise for
  intensities; the value is recorded.

The derivatives are computed along the MACE modes as the hessian step gives
them, and checkpointed per mode. Symmetry adaptation is applied afterwards as
a rotation, so it never invalidates the checkpoint.

- **Grouping.** Modes within 0.5 cm⁻¹ of the lowest mode of a set form one
  (near-)degenerate set; sets do not chain.
- **Projection.** Within each set, projection operators of the point group
  (pymatgen, tolerance 0.1 Å) give one subspace per irreducible
  representation.
- **Alignment.** The basis of each subspace is chosen closest to the original
  modes, so accidentally near-degenerate modes of the same irrep are not mixed.
- **Labels.** For C1, Cs, C2, C2v, C3, C3v, D2d and Td the modes get irrep
  labels (C2v in Mulliken's convention, with the plane holding more atoms as
  σv′(yz)). For other groups only the totally symmetric modes are flagged. On Cd16Se13Cl6 (Td) the selection rules hold to
numerical precision: only T₂ modes carry IR intensity (other modes
< 10⁻⁵ km/mol), A₂ and T₁ modes have no Raman activity, and ρ = 0 for A₁.
Halving h changes intensities by < 0.6 %, and halving F changes Raman
activities by up to 3 %.

Each mode is also described by:

- the shares of its mass-weighted amplitude on core, surface and ligand atoms
  and on each element;
- its radial share;
- its overlap with a uniform breathing of the dot.

The class is descriptive:

- *breathing*: overlap ≥ 0.3;
- *M–X ligand*: ligand share > 0.5;
- in the optical range, ν ≥ 0.75 ν_TO: *core optical* (core share ≥ 0.4) or
  *surface optical*;
- otherwise *core acoustic-like* or *surface / torsional*.

ν_TO is the bulk Γ-point optical frequency from MACE force constants of the
relaxed primitive cell. It is the TO frequency only: LO–TO splitting needs
Born charges and the long-range dipole term, which MACE does not have.

The spectra are non-resonant. Measured QD Raman spectra are usually resonant
and dominated by the LO mode and its overtones (Fröhlich coupling), so
compare peak positions and symmetries, not relative intensities. The cost is
6(2(3N − 6) + 1) + 1 g-xTB single points: 46 s for Cd16Se13Cl6 and about
1.5 h for Cd68Se55Cl26 on 12 concurrent single-threaded runs. Per-mode
results are checkpointed in `vibspec_cache.json`.

## Energetics

A charge-balanced binary dot M_a A_b X_c (cation charge q, anion −q,
monovalent ligand X) is $[\mathrm{MA}]_n(\mathrm{MX}_q)_m$ with $n = b$ and
$m = c/q$, e.g. Cd16Se13Cl6 = (CdSe)13(CdCl2)3. The references are computed
with the same MACE head and cached per material:

- **Bulk MA**, the sink of the inorganic core: dots ripen and grow towards
  the bulk, so the free energy against bulk MA is the driving force of
  growth and dissolution. The record's CIF is cell-relaxed, and its phonons
  (finite displacements in a supercell of at least 12 Å) give F_vib(T).
- **MA monomer**, the diatomic molecule. It is not a species in solution,
  but it is the usual reference of cluster binding energies.
- **MX_q monomer** (linear CdCl2, trigonal InCl3, …): detached Z-type
  ligands stay in solution as molecular complexes, so the monomer, not its
  crystal, is the relevant sink.

Molecules and dots are ideal-gas solutes at a 1 M standard state, with
harmonic vibrations and rigid-rotor rotation (the dot's rotational symmetry
number from its point group). Solvation and the binding of L-type donors
(amines, phosphines) to the detached MX_q are not included, which makes
detachment free energies upper bounds.

- decomposition: $[\mathrm{MA}]_n(\mathrm{MX}_q)_m \to n\,\mathrm{MA(bulk)} + m\,\mathrm{MX}_q$,
  the excess (surface) free energy of the dot; positive for any finite dot
  and decreasing per MA unit with size;
- binding: the same into monomers, per unit;
- stepwise detachment (beam search, see below);
- equilibrium shell: with MX_q in solution at $\mu = \mu^\circ(T) + \Delta\mu$,
  $\Delta\mu = kT\ln(c/1\,\mathrm{M})$, the mean number of removed units
  ⟨k⟩ is the Boltzmann average over every configuration the search evaluated.

## Z-type ligand desorption

A unit is a surface cation M with q ligands. MX_q leaves preferably as one
molecule: a cation with at least q bonded ligands gives units of its own
ligands only; a cation with fewer is completed by the closest other ligand in
space, whatever the distance (such units are flagged non-molecular).

Removing unit u from the relaxed structure $R_k$ costs
$\Delta E_k(u) = E[\mathrm{relax}(R_k - u)] + E(\mathrm{MX}_q) - E(R_k)$.
The search re-enumerates the units on each relaxed structure and keeps the
two best structures per level (beam width 2),
$\mathcal S_{k+1} = \mathrm{best}_2\{\mathrm{relax}(R-u): R\in\mathcal S_k, u\in U(R)\}$,
so the units follow the reconstruction of the shell. A removal only perturbs
its surroundings: units farther than $r_c = \max(7\,\text{Å}, 0.6\times$ the
dot's span$)$ keep their previous ΔE as an estimate, and an estimate is relaxed
before it can be chosen and whenever it lies within 1 eV of the best relaxed
candidate (a removal can make a distant unit much cheaper). Every committed
step is a MACE relaxation, and the harmonic Hessians along the best path give
$\Delta G_k(T)$. On Cd16Se13Cl6 the search reproduces the exhaustive search
over all removal sequences.

A fixed-lattice cluster expansion, $E(\sigma) = E_0 + \sum_u J_u\sigma_u +
\sum_{u<v} J_{uv}\sigma_u\sigma_v$ with sites defined on the intact dot, was
tried first: it reproduces single and pair removals, but fails once the shell
reconstructs (new units appear that the lattice does not contain).

## In solution

`solvation` adds an implicit solvent to every species at its MACE-MH-1
geometry: Generalized Born electrostatics of the GFN2-xTB charges,
$\Delta G_\mathrm{solv}(\varepsilon) = -\tfrac12(1-1/\varepsilon)\sum_{ij}q_iq_j/f_\mathrm{GB}(r_{ij})$,
$f_\mathrm{GB} = \sqrt{r^2 + R_iR_j e^{-r^2/4R_iR_j}}$, with
Hawkins–Cramer–Truhlar Born radii from Bondi radii scaled by 1.15. It is
exactly continuous in ε. The self-consistent GFN2-xTB ddCOSMO, which also lets
the electrons polarise, proved numerically unstable for the ligand-stripped
dots (diverging, or jumping between SCC solutions from one ε to the next);
with the 1.15 scale GB reproduces ddCOSMO within about 0.1 eV for the
monomers and Cd16Se13Cl6, and gives less for larger, more polarisable dots.
Cavity and dispersion terms are not included. ddCOSMO (ε = 2.4, 80) and ALPB
(named solvents) are stored as checks where they converge.

With $G_i^\mathrm{sol}(T,\varepsilon) = G_i^\circ(T) + \Delta G_{\mathrm{solv},i}(\varepsilon)$
(ideal solutes at 1 M) and $\mu_i = G_i^\mathrm{sol} + kT\ln(c_i/1\,\mathrm M)$:

- $\Delta G_\mathrm{dec} = [G^\mathrm{sol}(\mathrm{dot}) - n\,G(\mathrm{MA, bulk}) - m\,\mu(\mathrm{MX}_q)]/n$;
- $\Delta G_\mathrm{bind} = [G^\mathrm{sol}(\mathrm{dot}) - n\,\mu(\mathrm{MA}) - m\,\mu(\mathrm{MX}_q)]/(n+m)$,
  and the dissolution temperature $T_\mathrm{diss}$ where it reaches zero;
- the stepwise ladder $\Delta G_k = G^\mathrm{sol}(\mathrm{dot}_k) - G^\mathrm{sol}(\mathrm{dot}_{k-1}) + \mu(\mathrm{MX}_q)$.

The "Stability & ligands in solution" section of `ground_state.html`
evaluates these live. Each panel has its own controls, only for the
quantities that enter it:

| Panel | Controls |
|---|---|
| ΔG_dec(T) | solvent model / ε, [MX_q] (MA is the bulk solid) |
| ΔG_bind(T), T_diss | solvent / ε, [MA] monomer, [MX_q], precursor stabilisation |
| Stepwise ladder | solvent / ε, [MX_q], T |
| Desorption isotherm ⟨k⟩([MX_q]) | solvent / ε, T |
| Equilibrium ligand shell ⟨k⟩([MX_q], T) | solvent / ε |

The solvent model is Generalized Born with ε from a slider, a named ALPB
solvent, or the gas phase.

The precursor stabilisation Δμ_prec is a constant offset in the monomer
chemical potential, μ(MA) = G°(molecule) + ΔG_solv + Δμ_prec + k_BT ln c. It
models a monomer that is really bound in a molecular precursor. The
concentration term is the ideal-dilution entropy and scales with T; the
offset does not. The free energies are tabulated from 50 to 1500 K.

The ligand-shell map belongs to this section, not the vacuum one, because
ligand exchange is an equilibrium with the solution. The vacuum section shows
ΔG_dec, ΔG_bind and the ΔE/ΔG desorption ladder at a 1 M standard state.

The bond-length plot compares the dot with the MACE-MH-1 bulk bond (the CIF
cell-relaxed with the same model, `qdprops.bulk`) and with the experimental
bulk bond. For CdSe the model gives 2.686 Å and experiment 2.620 Å (zinc
blende, a = 6.05 Å; wurtzite a = 4.299 Å, c = 7.010 Å; Landolt–Börnstein).
The core strain is quoted against the MACE value.

## Synthesis dashboard

```bash
PYTHONPATH=src python -m qdprops synthesis rec_Cd16 rec_Cd68 -o synthesis.html
```

puts several dots, with all their ligand-stripped states, in equilibrium
with the monomers. For total concentrations $C_\mathrm{MA}$, $C_\mathrm{MX}$,
$c_i = e^{-\Delta G_i^\circ/kT} c_\mathrm{MA}^{n_i} c_\mathrm{MX}^{m_i}$ with
$\Delta G_i^\circ = G_i^\mathrm{sol} - n_iG_\mathrm{MA}^\mathrm{sol} - m_iG_\mathrm{MX}^\mathrm{sol}$
and the mass balances $C_\mathrm{MA} = c_\mathrm{MA} + \sum n_ic_i$,
$C_\mathrm{MX} = c_\mathrm{MX} + \sum m_ic_i$ are solved for the free monomers
(nested bisection in log space, continued from high T). Optionally bulk MA precipitates
once $c_\mathrm{MA}$ exceeds its solubility
$c_\mathrm{sat} = e^{-(G_\mathrm{MA}^\mathrm{sol} - G_\mathrm{bulk})/kT}$.
The page shows the yield of each dot family against T (with the dissolution
onset), the populations against the $[\mathrm{MX}_q]/[\mathrm{MA}]$ ratio,
the free monomers and the supersaturation $S = c_\mathrm{MA}/c_\mathrm{sat}$,
and the ligand coverage. Each panel has its own controls: solvent, total
[MA], [MX_q]/[MA] ratio, precursor stabilisation and bulk precipitation. The
ratio scan takes T in place of the ratio.

The monomers are the bare MA and MX_q molecules; real precursors (metal
carboxylates, phosphine chalcogenides) are more stable, which the precursor
shift mimics, and specific coordination of the dissolved MX_q by L-type
ligands is not included.

## Output

Results go to `<record>/props/`:

```
props/
├── relaxed.xyz        MACE-MH-1 minimum
├── relax.json         one file per step: summary, details, provenance,
├── structure.json     and the hash of its inputs
├── hessian.json
├── vibspec.json       IR/Raman per mode, point group, irreps, mode character
├── vibspec_modes.npz  symmetry-adapted modes, ∂μ/∂Q and ∂α/∂Q
├── vibspec_cache.json checkpoint of the per-mode g-xTB derivatives
├── electronic.json
├── stability.json
├── detachment.json
├── detach_<k>.xyz     dot after k removals, best path
├── detach_cache.json  checkpoint of the removal relaxations and Hessians
├── solvation.json     GB, ddCOSMO and ALPB solvation of every species
├── solution.json      species free energies for the dashboards
├── modes.npz          frequencies and mass-weighted normal modes
├── spectra.json       VDOS curves and thermochemistry against T
├── ground_state.html  interactive summary (Plotly), shown in the webapp
├── ground_state.png   static summary
└── properties.json    step summaries and provenance, read by the webapp ingest
```

Each step's result is reused when its inputs are unchanged: the step
settings, the start geometry, the results it depends on and the source of
the step itself. Deleting a step file, or `--force`, recomputes it.
