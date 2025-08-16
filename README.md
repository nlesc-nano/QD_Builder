# nanocrystal-builder

Wulff-construction nanocrystal builder with **coordination-aware** surface analysis and passivation.

- Builds a Wulff-cut NC from a bulk CIF.
- Detects exposed facets and classifies atoms (unique / edge / vertex; outer / sublayer).
- Passivates under-coordinated anions with a ligand (e.g., Cl⁻), **stepwise** until charge neutrality.
- Supports **signed Miller indices** so polar facet pairs (e.g. `111` vs `-1-1-1`) can have **different γ**.
- Facet-aware, **farthest-point** sampling to spread swaps/removals/additions uniformly across each face.
- Optional pruning of **monocoordinated** artifacts prior to analysis.

---

## Install

### Via conda (recommended)
```bash
git clone https://github.com/your-user/nanocrystal-builder.git
cd nanocrystal-builder
conda env create -f environment.yml
conda activate nc-builder

