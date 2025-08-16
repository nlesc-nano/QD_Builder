# QD_Builder

**Quantum Dot Builder** — Build and passivate zinc-blende nanocrystal models from CIF files using Wulff construction, facet-specific energies, and a coordination-aware ligand passivation workflow.

Originally inspired by *NanoCrystal: A Web‑Based Crystallographic Tool for the Construction of Nanoparticles Based on Their Crystal Habit* ([pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/30351055/)), QD_Builder expands upon these concepts by integrating:

- **Signed Miller indices**: separate energies for both polarities (e.g. `111` vs `‑1-1-1`).
- **Coordination-aware ligand passivation**: swaps under-coordinated surface anions with ligands (e.g., Cl) in a facet-aware, balanced, and iterative fashion.
- **Charge balance enforcement** via stepwise swaps, selective removals/additions, and parity-aware logic.
- **Reporting tools**: per-atom/facet CN breakdowns, role classification (unique, edge, vertex), and outer vs sublayer distinction.

---

##  Installation

### Using Conda environment (recommended)

```bash
git clone git@github.com:nlesc-nano/QD_Builder.git
cd QD_Builder
conda env create -f environment.yml
conda activate nc-builder
```

This sets up all dependencies, then installs QD_Builder in editable mode.

---

### Using pip

If you're in any Python environment:

```bash
git clone git@github.com:nlesc-nano/QD_Builder.git
cd QD_Builder
pip install -e .
```

This installs the `nc-builder` command in your PATH, pointing to your local development copy.

---

Alternatively, install directly from GitHub without cloning:

```bash
pip install git+ssh://git@github.com/nlesc-nano/QD_Builder.git
```

---

##  Quick Usage Example

Once installed, run:

```bash
nc-builder structure.cif config.yaml -r 24 -o final.xyz --verbose --center --write-all
```

Key flags:

| Flag           | Purpose                                          |
|----------------|--------------------------------------------------|
| `-r, --radius` | Wulff construction radius in Å                   |
| `--center`     | Center output coordinates (origin at geometric center) |
| `--write-all`  | Save intermediate files like `_cut.xyz`          |
| `--verbose`    | Enable detailed logging of each step             |
| `--parity {remove,add}` | Strategy for odd-charge resolution (`add` by default) |
| `--no-prune-mono` | Skip the CN<2 cleanup step if desired         |

---

##  Example YAML Config

```yaml
facets:
  - hkl: 100
    gamma: 1.0
  - hkl: 111
    gamma: 0.8
  - hkl: "-1-1-1"
    gamma: 1.0

passivation:
  ligand: Cl
  surf_tol: 2.0

charges:
  Cd:  +3
  Se:  -3
  Cl:  -1
```

You can define facet energies separately for polar facets. Signed hkl values are supported in various formats: `111`, `-100`, `1-10`, `(1 1 1)`, `[-1,0,0]`, etc.

---

##  How It Works

1. **NanoCrystal-style Wulff cut**: Builds the nanocrystal shape using input CIF and facet weights.
2. **Pruning**: Removes dangling monocoordinated atoms (CN < 2) to clean artifacts.
3. **Facet detection & reporting**: Identifies actual exposed facets and classifies surface atoms by CN, layer (outer vs sublayer), and role.
4. **Layer-aware anion passivation**:
   - Swaps under-coordinated anions for ligands (e.g., Se → Cl).
   - Conducted facet-wise with distance spacing (farthest-point sampling), role prioritization, and charge balance logic.
5. **Charge neutrality enforcement**: Uses a parity strategy (`add` or `remove`) to correct odd mismatches.
6. **Fallbacks**: If balancing fails via swaps, the tool can remove previously placed ligands or add new ones at cation sites.
7. **Output**: Final XYZ → `final.xyz`, plus optional logs, cut models, and manifests.

---

##  Citation / Inspiration

- Based on the **NanoCrystal** web tool (ACS JCIM 2018) ([pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/30351055/)), which used Wulff constructions to generate equilibrium-shaped nanocrystals.
- Extends this with coordination-aware chemistry, ligand passivation, and in-depth surface analysis.

If you use QD_Builder in your research, please cite this repository and the NanoCrystal paper.

---

##  License

Released under the **MIT License**. See the `LICENSE` file for full terms.
