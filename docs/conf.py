"""Sphinx configuration for the QD_Builder documentation."""
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "QD_Builder"
author = "Ivan Infante"
copyright = "2026, Ivan Infante"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
]

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist", "attrs_inline"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = ["rdkit", "ase", "scm", "nanoCAT", "CAT", "FOX", "plams"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "QD_Builder"
html_static_path = ["_static"]
math_number_all = False
