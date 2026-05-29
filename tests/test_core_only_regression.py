"""Core-only regression checks: termination sign and charge balance."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]


def _run_builder(cif: Path, yaml_path: Path, out: Path, *, positive_q_mode: str = "add") -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "builder",
        str(cif),
        str(yaml_path),
        "-o",
        str(out),
        "--positive-q-mode",
        positive_q_mode,
        "--write-all",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)


def _read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().strip().splitlines()
    n = int(lines[0].strip())
    syms, pts = [], []
    for i in range(n):
        parts = lines[i + 2].split()
        syms.append(parts[0])
        pts.append(list(map(float, parts[1:4])))
    return syms, np.asarray(pts, float)


def test_inp_anion_rich_termination_at_cut() -> None:
    """Anion-rich family {111} must expose P on the outer {111} layer at cut stage."""
    from builder.config import parse_yaml_config
    from builder.facets import expand_facets, unit_normal
    from builder.geometry import build_nanocrystal, dedupe_points
    from builder.main import _prune_before_facet_detection, _resolve_facet_terminations
    from builder.stack import size_unit_cells_to_radius_aspect

    cif = ROOT / "examples/cifs/InP_zb.cif"
    yaml_path = ROOT / "tests/test_inp_anion_rich.yaml"
    struct = Structure.from_file(cif)
    cfg = parse_yaml_config(str(yaml_path))
    charges = cfg.charges

    seeds = _resolve_facet_terminations(struct, cfg.seeds, charges)
    assert [(f.h, f.k, f.l) for f in seeds] == [(-1, -1, -1)]

    radius, aspect = size_unit_cells_to_radius_aspect(struct, (2, 2, 2))
    wulff = expand_facets(struct, seeds, proper_only=cfg.proper_only)
    syms, pts, _ = build_nanocrystal(struct, wulff, radius, aspect=aspect)
    syms, pts = dedupe_points(syms, pts, tol=1e-3)

    class Args:
        prune_mono = True
        prune_min_cn = 2
        prune_passes = 10
        verbose = False

    syms, pts = _prune_before_facet_detection(syms, pts, args=Args())
    n = unit_normal(struct, (1, 1, 1))
    proj = pts @ n
    d_plane = float(np.max(proj))
    surf_tol = cfg.passivation.surf_tol
    shell = [i for i, p in enumerate(proj) if d_plane - float(p) < surf_tol]
    surf = Counter(syms[i] for i in shell)
    q_cut = sum(charges.get(s, 0) * Counter(syms)[s] for s in Counter(syms))

    assert surf.get("P", 0) > surf.get("In", 0), f"expected P-rich {111} shell, got {surf}"
    assert q_cut < 0, f"anion-rich cut should have negative bulk Q, got {q_cut:+d}"


def test_inp_anion_rich_charge_balance_add_mode() -> None:
    """Core-only InP build with add-mode passivation must reach Q=0."""
    cif = ROOT / "examples/cifs/InP_zb.cif"
    yaml_path = ROOT / "tests/test_inp_anion_rich.yaml"
    out = ROOT / "tests/out/inp_anion_rich_regression.xyz"
    _run_builder(cif, yaml_path, out, positive_q_mode="add")

    manifest = json.loads(out.with_suffix(".json").read_text())
    assert manifest["total_charge"] == 0, manifest

    syms, _ = _read_xyz(out)
    counts = Counter(syms)
    assert counts["In"] > counts["P"] or counts["Cl"] > 0


def test_mixed_family_facet_scope_keeps_full_oriented_set() -> None:
    """Terminated family facets must not be swallowed by explicit facet entries."""
    from builder.config import parse_yaml_config
    from builder.facets import expand_facets
    from builder.main import _resolve_facet_terminations

    cif = ROOT / "examples/cifs/InAs.cif"
    yaml_path = ROOT / "examples/core-only/inas_oriented_facet_scope.yaml"
    struct = Structure.from_file(cif)
    cfg = parse_yaml_config(str(yaml_path))

    seeds = _resolve_facet_terminations(struct, cfg.seeds, cfg.charges)
    expanded = expand_facets(struct, seeds, proper_only=cfg.proper_only)
    hkl111 = {
        (f.h, f.k, f.l)
        for f in expanded
        if sorted(abs(x) for x in (f.h, f.k, f.l)) == [1, 1, 1]
    }

    assert len(hkl111) == 8, sorted(hkl111)

    out = ROOT / "tests/out/inas_oriented_scope_regression.xyz"
    _run_builder(cif, yaml_path, out, positive_q_mode="add")
    cut_atoms = int(out.with_name(out.stem + "_cut.xyz").read_text().split()[0])
    manifest = json.loads(out.with_suffix(".json").read_text())

    assert cut_atoms == 421
    assert manifest["counts"] == {"In": 153, "Cl": 110, "As": 106}
    assert manifest["total_charge"] == 31


if __name__ == "__main__":
    test_inp_anion_rich_termination_at_cut()
    test_inp_anion_rich_charge_balance_add_mode()
    test_mixed_family_facet_scope_keeps_full_oriented_set()
    print("core-only regression tests passed")
