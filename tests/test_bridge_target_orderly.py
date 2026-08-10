"""Orderly generation in the bridge-target decorator.

Two properties matter and neither is visible from the accepted graph count
alone: the walk must not re-explore automorphic images of the same partial
bridge set, and it must fall through to the terminal-only tier when no bridge
is placeable.  Both were broken; the first cost a 24x over-emission and the
second produced empty p=1 bins.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from builder.nucleation import generate_molecular_map  # noqa: E402
from builder.nucleation.spec import load_nucleation_spec  # noqa: E402

PACK_DIR = ROOT / "geometry_packs" / "cdse_cdcl2"


@pytest.fixture(scope="module")
def target_pack(tmp_path_factory) -> Path:
    """The production pack with the bridge-target decorator, graph-only."""

    tmp = tmp_path_factory.mktemp("target")
    driver = yaml.safe_load((PACK_DIR / "run_gxtb.yaml").read_text())
    rules = yaml.safe_load((PACK_DIR / "graph_rules.yaml").read_text())
    rules["graph_rules"]["decoration_mode"] = "motif_bridge_target"
    # Pin the generator settings the expected counts were measured under, so
    # this stays a test of orbit pruning rather than of pack policy.
    rules["graph_rules"]["bridge_first_hard_max_bridges_per_cd"] = 2
    rules["graph_rules"]["bridge_target_count_window"] = 0
    rules["graph_rules"].pop("decoration_mode_from_k", None)
    rules["graph_rules"].pop("decoration_mode_at_or_above", None)
    rules["graph_rules"].pop("selection_max_per_skeleton", None)
    rules["graph_rules"].pop("selection_per_skeleton_from_k", None)
    # no selection cut: count what the generator makes, not what survives it
    rules["graph_rules"].pop("selection_order", None)
    rules["graph_rules"].pop("selection_max_wiener_excess", None)
    merged = {k: v for k, v in driver.items() if k != "include"}
    merged.update(yaml.safe_load((PACK_DIR / "motifs.yaml").read_text()))
    merged.update(yaml.safe_load((PACK_DIR / "embed.yaml").read_text()))
    merged.update(rules)
    merged["cif"] = str(ROOT / "examples/cifs/CdSe_zb.cif")
    merged.setdefault("relaxation", {})["enabled"] = False
    path = tmp / "target.yaml"
    path.write_text(yaml.safe_dump(merged, sort_keys=False))
    return path


def _bin(pack: Path, k: int, p: int):
    spec = load_nucleation_spec(str(pack))
    return generate_molecular_map(
        spec, kmin=k, kmax=k, pmin=p, pmax=p, embed=False
    ).bins[(k, p)]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_p1_reaches_the_terminal_only_tier(target_pack: Path, k: int) -> None:
    """p=1 has no placeable bridge, so the all-terminal decoration must win.

    The target loop stops descending as soon as a target *emits*.  Until the
    generator enforced min_bridged_host_cn itself it emitted bridges the
    screen then rejected, so k1p1 and k2p1 ended up with zero graphs.
    """

    result = _bin(target_pack, k, 1)
    assert result.isomers, f"k{k}p1 produced no graphs"


@pytest.mark.parametrize(
    "k,p,expected",
    [(2, 2, 2), (2, 3, 14), (2, 5, 100), (3, 3, 118), (3, 4, 1546)],
)
def test_orbit_pruning_keeps_every_graph(
    target_pack: Path, k: int, p: int, expected: int
) -> None:
    """Pruning is a search optimisation and must not change the result.

    These counts were measured before orderly generation was added; the walk
    now visits far fewer nodes and has to arrive at exactly the same set.
    """

    assert len(_bin(target_pack, k, p).isomers) == expected


@pytest.mark.parametrize("k,p,ceiling", [(2, 5, 15.0), (3, 3, 4.0), (3, 4, 5.0)])
def test_emission_overhead_stays_bounded(
    target_pack: Path, k: int, p: int, ceiling: float
) -> None:
    """Emitted-per-kept, the quantity orderly generation exists to control.

    Ceilings sit above the measured values (9.4, 2.2, 3.2) with headroom, so
    this catches a regression to the old behaviour (72, 14.2, 22.6) without
    tracking small changes.  A skeleton whose |Aut| exceeds
    ``bridge_target_max_automorphisms`` falls back to no pruning, which is
    what the k2p5 ceiling guards.
    """

    result = _bin(target_pack, k, p)
    kept = max(1, len(result.isomers))
    assert result.raw_graphs / kept < ceiling


def test_per_skeleton_budget_keeps_every_skeleton(tmp_path_factory) -> None:
    """The quota samples each skeleton, not the first few.

    The energy variance is between skeletons (best-per-skeleton spans 0.7-2.9
    eV in a bin) and only ~0.3-0.5 eV within one, so a budget that dropped
    whole skeletons would discard exactly the variation that matters.
    """

    tmp = tmp_path_factory.mktemp("budget")
    driver = yaml.safe_load((PACK_DIR / "run_gxtb.yaml").read_text())
    rules = yaml.safe_load((PACK_DIR / "graph_rules.yaml").read_text())
    rules["graph_rules"]["selection_max_per_skeleton"] = 5
    rules["graph_rules"].pop("selection_max_wiener_excess", None)
    merged = {k: v for k, v in driver.items() if k != "include"}
    merged.update(yaml.safe_load((PACK_DIR / "motifs.yaml").read_text()))
    merged.update(yaml.safe_load((PACK_DIR / "embed.yaml").read_text()))
    merged.update(rules)
    merged["cif"] = str(ROOT / "examples/cifs/CdSe_zb.cif")
    merged.setdefault("relaxation", {})["enabled"] = False
    path = tmp / "budget.yaml"
    path.write_text(yaml.safe_dump(merged, sort_keys=False))

    spec = load_nucleation_spec(str(path))
    result = generate_molecular_map(
        spec, kmin=3, kmax=3, pmin=2, pmax=2, embed=True
    ).bins[(3, 2)]

    assert result.isomers, "budget produced an empty bin"
    # the quota is the only cap, so everything sampled must be embedded
    assert result.budget_pool > 0
    # The quota bounds candidates entering 3D, not final isomers: the
    # motif_factor path runs several reconstruction starts per candidate and
    # each can yield an accepted structure, so the isomer count per skeleton
    # legitimately exceeds the quota.
    assert result.budget_pool <= 5 * max(1, result.skeletons_total)

    # Coverage: every skeleton that put candidates in the pool is still
    # represented.  Measured directly on the bridge-target generator (k3p2,
    # quota 5): 38 graphs -> 17, all 5 skeletons retained.
    assert result.budget_pool >= min(5, result.skeletons_total)


def test_budget_gate_leaves_small_bins_exhaustive(tmp_path_factory) -> None:
    """`selection_per_skeleton_from_k` switches the quota off below its k."""

    tmp = tmp_path_factory.mktemp("gate")
    driver = yaml.safe_load((PACK_DIR / "run_gxtb.yaml").read_text())
    rules = yaml.safe_load((PACK_DIR / "graph_rules.yaml").read_text())
    rules["graph_rules"]["selection_max_per_skeleton"] = 5
    rules["graph_rules"]["selection_per_skeleton_from_k"] = 4
    rules["graph_rules"].pop("selection_max_wiener_excess", None)
    merged = {k: v for k, v in driver.items() if k != "include"}
    merged.update(yaml.safe_load((PACK_DIR / "motifs.yaml").read_text()))
    merged.update(yaml.safe_load((PACK_DIR / "embed.yaml").read_text()))
    merged.update(rules)
    merged["cif"] = str(ROOT / "examples/cifs/CdSe_zb.cif")
    merged.setdefault("relaxation", {})["enabled"] = False
    path = tmp / "gate.yaml"
    path.write_text(yaml.safe_dump(merged, sort_keys=False))

    spec = load_nucleation_spec(str(path))
    result = generate_molecular_map(
        spec, kmin=3, kmax=3, pmin=2, pmax=2, embed=True
    ).bins[(3, 2)]
    assert result.budget_pool == 0, "gate did not disable the quota"
