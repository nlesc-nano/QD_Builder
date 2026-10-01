# Installation

QD_Builder is distributed as the Python package `nanocrystal-builder`, which
installs the importable module `builder`. It requires Python 3.9 or later.
The core dependencies are NumPy, SciPy (≥ 1.11), PyYAML and pymatgen
(≥ 2023.7) {cite:p}`ong2013`. pymatgen provides the crystal structures,
symmetry analysis and lattice matching on which the construction rests.

## With conda (recommended)

The repository ships an `environment.yml` that creates an environment called
`nc-builder` from conda-forge and installs the package in editable mode:

```bash
conda env create -f environment.yml
conda activate nc-builder
```

The environment also contains RDKit. RDKit is needed only by the ligand
post-treatments that build real molecules from SMILES strings (see
{ref}`post-overview`). It is imported lazily, so the construction and
passivation of a particle work without it.

## With pip

Inside an existing environment, install the package from the repository
root:

```bash
pip install -e .
```

The pip dependencies do not include RDKit. Install it separately
(`conda install -c conda-forge rdkit` or `pip install rdkit`) if you intend
to exchange placeholder ligands for molecules.

## Running the builder

Installation registers the console script `nc-builder`. The same entry point
is available as a module:

```bash
nc-builder --help
python -m builder --help
```

The auxiliary tools live in `builder.scripts` and are run as modules. The
tools are the library series generator, the heterointerface scanner, the
Janus builder, the size scanner and the CIF facet analyser. For example:

```bash
python -m builder.scripts.generate_library examples/library/cdse_zb.yaml
python -m builder.scripts.analyze_cif_facets examples/cifs/CdSe.cif
```

## Building this documentation

The documentation is written in MyST Markdown and built with Sphinx:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```
