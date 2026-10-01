# Testing

The regression tests live in `tests/`. Each builds a small particle from a
recipe in the same directory and checks composition, charge or invariance
properties. They run the builder as a module:

```bash
PYTHONPATH=src python -m pytest tests/test_cdse_x_type_native_exchange.py
PYTHONPATH=src python tests/test_stack_swap_invariance.py
```

pytest is not a runtime dependency; install it separately
(`pip install pytest`). The ligand tests also need RDKit.

| Test | What it checks |
|---|---|
| `test_core_only_regression.py` | single-material builds stay unchanged |
| `test_stack_swap_invariance.py` | core/shell relabelling does not depend on labels |
| `test_ligand_exchange_regression.py` | X-type exchange of placeholders |
| `test_cdse_x_type_native_exchange.py` | exchange of native Se and Cd, and its charge compensation |
| `test_cspbbr3_anion_exchange.py` | native bromide exchange on CsPbBr₃ |
| `test_neutral_ligand_uniform.py` | uniform distribution of L-type ligands |
| `test_mxn_exchange_regression.py` | MXn, zwitterion and L-type neutral exchange |
| `test_z_type_exchange.py` | neutral exchange on CsPbBr₃ |

The library generator serves as an additional end-to-end check. Every
series in `examples/library/` must regenerate without charged, clashing or
off-centre structures, and its `review.md` records every build.
