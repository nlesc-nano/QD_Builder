"""Coordination-optimized nanocrystal nucleation maps with dual coordinates.

Public API is re-exported here. Implementation lives in submodules
(types, spec, lattice, graph_ops, scoring, surface, checkpoint, bundle, engine).

Private helpers remain importable as ``builder.nucleation._name`` for tests and
internal call sites. Star-imports skip underscore names, so submodules are
copied into this package namespace explicitly.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from . import (
    bundle,
    checkpoint,
    engine,
    geometry_pack,
    graph_ops,
    lattice,
    molecular,
    molecular_rules,
    scoring,
    spec,
    surface,
    types,
)

_SUBMODULES: tuple[ModuleType, ...] = (
    types,
    spec,
    graph_ops,
    lattice,
    scoring,
    surface,
    molecular_rules,
    geometry_pack,
    molecular,
    checkpoint,
    bundle,
    engine,
)


def _export_module(module: ModuleType) -> None:
    for name, value in vars(module).items():
        if name.startswith("__") and name not in {"__all__"}:
            continue
        globals()[name] = value


for _module in _SUBMODULES:
    _export_module(_module)

# Stable public surface (also used by setuptools / docs).
__all__ = [
    "AtomRecord",
    "ClusterRecord",
    "NucleationRegistry",
    "NucleationResult",
    "ProgressCallback",
    "SweepAudit",
    "generate_molecular_map",
    "generate_nucleation_map",
    "generate_nucleation_result",
    "is_nucleation_yaml",
    "load_geometry_pack",
    "load_nucleation_spec",
    "nucleation_result_to_dict",
    "registry_to_dict",
    "write_molecular_map",
    "write_nucleation_bundle",
    "write_nucleation_json",
]

# Touch Any so typing import is not unused under some linters.
_: Any = None
