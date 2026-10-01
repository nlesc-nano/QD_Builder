# QD_Builder

QD_Builder turns a bulk crystal structure into an atomistic model of a
colloidal nanocrystal that can be handed directly to an electronic-structure
code. Starting from a CIF file and a short YAML recipe, it carves a faceted
particle out of the bulk lattice with a Wulff construction, decides which
atomic layer each facet exposes, removes atoms that the real particle could
not retain, and caps the surface with charge-compensating ligands until the
particle is neutral. A family of post-treatments then brings the model
closer to the chemistry of real samples: polar facets can be reconstructed,
placeholder ligands can be exchanged for real molecules, neutral donors can
be bound to under-coordinated metals, metal–ligand complexes can be
displaced, and the lattice can be alloyed.

The same engine builds core–shell particles and Janus heterostructures, and
generates the size series that populate the
[QDSpace library](https://quantumdotspace.org).

This documentation is organised along the path a structure takes through
the builder. *Getting started* shows how to install the package and build a
first particle. The *User guide* explains how to express a structure in a
recipe. The *Theory* chapters develop the physical model behind each
construction step, from the Wulff shape to the {111} reconstruction of
zinc-blende particles, and *Post-processing* describes the ligand
treatments in the same way. The *Reference* collects the recipe schema, the
command-line options and the API.

```{toctree}
:caption: Getting started
:maxdepth: 2

getting-started/installation
getting-started/quickstart
getting-started/outputs
```

```{toctree}
:caption: User guide
:maxdepth: 2

user-guide/recipes
user-guide/core-shell
user-guide/heterostructures
user-guide/library
```

```{toctree}
:caption: Theory
:maxdepth: 2

theory/overview
theory/wulff
theory/polarity
theory/bonding
theory/charge-passivation
theory/ligand-sites
theory/reconstruction-111
theory/heterostructures
theory/structure-identity
```

```{toctree}
:caption: Post-processing
:maxdepth: 2

postprocessing/overview
postprocessing/x-type
postprocessing/l-type
postprocessing/neutral-exchange
postprocessing/z-type
postprocessing/alloying
```

```{toctree}
:caption: Worked examples
:maxdepth: 1

examples/cdse
examples/inp
examples/pbs
examples/cspbbr3
```

```{toctree}
:caption: Reference
:maxdepth: 2

reference/yaml
reference/cli
reference/api
reference/glossary
reference/bibliography
```

```{toctree}
:caption: Developer notes
:maxdepth: 1

developer/architecture
developer/testing
```
