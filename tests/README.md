# Tests

Regression tests write generated structures to `tests/out/`. That directory is
gitignored; re-create artifacts by running:

```bash
PYTHONPATH=src python tests/test_stack_swap_invariance.py
PYTHONPATH=src python -c "from tests.test_core_only_regression import *; test_inp_anion_rich_termination_at_cut(); test_inp_anion_rich_charge_balance_add_mode()"
```

Tracked inputs live alongside the test modules (`tests/*.yaml`, `examples/cifs/`).
