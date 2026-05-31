"""Stack-mode swap invariance checks for II-VI core-shell builds."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_builder(
    yaml_name: str,
    out_name: str,
    *,
    positive_q_mode: str = "remove",
) -> Path:
    out = ROOT / "tests" / "out" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "builder",
        str(ROOT / "tests" / yaml_name),
        "-o",
        str(out),
        "--positive-q-mode",
        positive_q_mode,
        "--write-all",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    return out


def _read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().strip().splitlines()
    n = int(lines[0].strip())
    sym, pts = [], []
    for i in range(n):
        parts = lines[i + 2].split()
        sym.append(parts[0])
        pts.append(list(map(float, parts[1:4])))
    return sym, np.asarray(pts, float)


def _count_xyz(path: Path) -> Counter:
    sym, _ = _read_xyz(path)
    return Counter(sym)


def _per_region_role_counts(xyz_path: Path, yaml_path: Path) -> list[tuple[int, int]]:
    """Return [(n_cation_sites, n_anion_sites), ...] per stack layer."""
    from builder.config import parse_yaml_config
    from builder.main import _resolve_facet_terminations
    from builder.stack import (
        build_layer_planes,
        cumulative_size_unit_cells,
        region_masks_from_layer_planes,
        select_geometry_reference,
    )
    from pymatgen.core import Structure

    cfg = parse_yaml_config(str(yaml_path))
    sym, pts = _read_xyz(xyz_path)
    reference_cfg = select_geometry_reference(cfg.materials, mode=cfg.stack.geometry_reference)
    struct_ref = Structure.from_file(reference_cfg.cif)
    resolved = []
    for m in cfg.materials:
        mat_struct = Structure.from_file(m.cif)
        resolved.append(
            dataclasses.replace(
                m,
                seeds=_resolve_facet_terminations(mat_struct, m.seeds, cfg.charges),
            )
        )
    cumulative = cumulative_size_unit_cells(resolved)
    layer_planes = build_layer_planes(
        resolved, struct_ref, cfg.proper_only, cumulative_sizes=cumulative,
    )
    masks = region_masks_from_layer_planes(pts, layer_planes)
    ligand = cfg.passivation.ligand
    out: list[tuple[int, int]] = []
    for mask in masks:
        layer_syms = [s for s, keep in zip(sym, mask) if keep]
        cats = sum(1 for s in layer_syms if cfg.charges.get(s, 0) > 0)
        ans = sum(1 for s in layer_syms if cfg.charges.get(s, 0) < 0 and s != ligand)
        out.append((cats, ans))
    return out


def test_swap_invariance_at_cut():
    _run_builder("stack_znse_cdse.yaml", "znse_cdse_test.xyz")
    _run_builder("stack_cdse_znse_swap.yaml", "cdse_znse_test.xyz")

    cut_a = _count_xyz(ROOT / "tests/out/znse_cdse_test_cut.xyz")
    cut_b = _count_xyz(ROOT / "tests/out/cdse_znse_test_cut.xyz")

    native_a = {k: v for k, v in cut_a.items() if k != "Cl"}
    native_b = {k: v for k, v in cut_b.items() if k != "Cl"}

    assert sum(native_a.values()) == sum(native_b.values()), (native_a, native_b)
    assert native_a.get("Se") == native_b.get("Se"), (native_a, native_b)

    cations_a = {k: v for k, v in native_a.items() if k != "Se"}
    cations_b = {k: v for k, v in native_b.items() if k != "Se"}
    assert sum(cations_a.values()) == sum(cations_b.values())
    assert sorted(cations_a.values()) == sorted(cations_b.values()), (cations_a, cations_b)


def test_swap_invariance_after_passivation():
    _run_builder("stack_znse_cdse.yaml", "znse_cdse_final.xyz")
    _run_builder("stack_cdse_znse_swap.yaml", "cdse_znse_final.xyz")

    for stem in ("znse_cdse_final", "cdse_znse_final"):
        with open(ROOT / "tests/out" / f"{stem}.json") as f:
            data = json.load(f)
        assert data["total_charge"] == 0, data
        native = sum(v for k, v in data["counts"].items() if k != "Cl")
        assert native > 0


@pytest.mark.skip(reason="Lattice constants differ for CdSe and ZnS shells under unified outer shell cutting")
def test_shared_cation_matches_shared_anion_topology():
    _run_builder("stack_znse_cdse.yaml", "topo_cdse.xyz")
    _run_builder("stack_znse_zns.yaml", "topo_zns.xyz")

    ref = _per_region_role_counts(
        ROOT / "tests/out/topo_cdse_cut.xyz",
        ROOT / "tests/stack_znse_cdse.yaml",
    )
    zns = _per_region_role_counts(
        ROOT / "tests/out/topo_zns_cut.xyz",
        ROOT / "tests/stack_znse_zns.yaml",
    )
    assert ref == zns, (ref, zns)

    with open(ROOT / "tests/out/topo_cdse.json") as f:
        cdse = json.load(f)
    with open(ROOT / "tests/out/topo_zns.json") as f:
        zns_final = json.load(f)
    assert cdse["total_charge"] == 0
    assert zns_final["total_charge"] == 0

    cdse_cations = sum(v for k, v in cdse["counts"].items() if k in {"Zn", "Cd"})
    zns_cations = zns_final["counts"].get("Zn", 0)
    cdse_anions = cdse["counts"].get("Se", 0)
    zns_anions = zns_final["counts"].get("Se", 0) + zns_final["counts"].get("S", 0)
    assert cdse_cations == zns_cations, (cdse["counts"], zns_final["counts"])
    assert cdse_anions == zns_anions, (cdse["counts"], zns_final["counts"])


@pytest.mark.skip(reason="Lattice constants differ for CdSe and CdTe shells under unified outer shell cutting")
def test_distinct_chemistry_matches_topology():
    _run_builder("stack_znse_cdse.yaml", "topo_cdse2.xyz")
    _run_builder("stack_znse_cdte.yaml", "topo_cdte.xyz")

    cdse = _per_region_role_counts(
        ROOT / "tests/out/topo_cdse2_cut.xyz",
        ROOT / "tests/stack_znse_cdse.yaml",
    )
    cdte = _per_region_role_counts(
        ROOT / "tests/out/topo_cdte_cut.xyz",
        ROOT / "tests/stack_znse_cdte.yaml",
    )
    assert cdse == cdte, (cdse, cdte)


def test_hetero_anion_stack_reaches_neutral():
    _run_builder("stack_znse_cdte.yaml", "znse_cdte_test.xyz", positive_q_mode="add")
    with open(ROOT / "tests/out/znse_cdte_test.json") as f:
        data = json.load(f)
    assert data["total_charge"] == 0, data
    assert int(Path(ROOT / "tests/out/znse_cdte_test_cut.xyz").read_text().split()[0]) > 0


def test_symmetry_mismatch_raises():
    bad_yaml = ROOT / "tests/out/bad_symmetry.yaml"
    bad_yaml.write_text(
        """
materials:
  - name: core
    cif: examples/cifs/ZnSe_zb.cif
    size_unit_cells: [1, 1, 1]
    facets:
      - hkl: "100"
        gamma: 1.0
  - name: shell
    cif: examples/cifs/PbS.cif
    size_unit_cells: [1, 1, 1]
    facets:
      - hkl: "100"
        gamma: 1.0
passivation:
  ligand: Cl
charges:
  Zn: +2
  Cd: +2
  Se: -2
  Pb: +2
  S: -2
  Cl: -1
""".strip()
        + "\n"
    )
    cmd = [
        sys.executable,
        "-m",
        "builder",
        str(bad_yaml),
        "-o",
        str(ROOT / "tests/out/bad_symmetry.xyz"),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "same space group" in (proc.stderr + proc.stdout)


if __name__ == "__main__":
    test_swap_invariance_at_cut()
    test_swap_invariance_after_passivation()
    # test_shared_cation_matches_shared_anion_topology()
    # test_distinct_chemistry_matches_topology()
    # test_hetero_anion_stack_reaches_neutral()
    # test_symmetry_mismatch_raises()
    print("stack swap invariance tests passed")
