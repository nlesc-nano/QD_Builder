"""The composed-pack loader: includes, collision detection, key validation.

These guard the property that makes a split pack readable -- a setting lives
in exactly one file, and a name the loader does not act on is an error rather
than a silent fallback to a default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from builder.nucleation.geometry_pack import (  # noqa: E402
    NUCLEATION_GRAPH_RULE_KEYS,
    load_geometry_pack,
)

PACK = ROOT / "geometry_packs" / "cdse_cdcl2" / "run_gxtb.yaml"
LEGACY = ROOT / "geometry_packs" / "cdse_cdcl2_motif_GXTB.yaml"


def test_composed_pack_loads() -> None:
    pack = load_geometry_pack(PACK)
    assert pack.bonds, "geometry tables did not arrive from embed.yaml"
    rules = pack.nucleation_graph_rules_mapping()
    assert rules["decoration_mode"] == "motif_bridge_first"
    assert rules["selection_order"] == "compactness"


def test_composed_pack_matches_legacy_single_file() -> None:
    """Splitting the pack changed no rule and no geometry number."""

    new = load_geometry_pack(PACK)
    old = load_geometry_pack(LEGACY)
    assert new.nucleation_graph_rules_mapping() == (
        old.nucleation_graph_rules_mapping()
    )
    assert new.bonds == old.bonds
    for section in (
        "angles", "angle_sum_cn3", "dihedrals", "rings",
        "nonbonded", "nonbonded_1_4", "junctions", "reconstruction",
    ):
        assert new.raw.get(section) == old.raw.get(section), section
    # the motif block keeps its vocabulary; only the dead geometry was dropped
    assert {
        name: (m["center"], m["linker_count"])
        for name, m in new.raw["motifs"].items()
    } == {
        name: (m["center"], m["linker_count"])
        for name, m in old.raw["motifs"].items()
    }


def _write_pack(tmp_path: Path, **overrides: object) -> Path:
    """Copy the composed pack into ``tmp_path`` so files can be perturbed."""

    src = PACK.parent
    for name in ("run_gxtb.yaml", "graph_rules.yaml", "motifs.yaml", "embed.yaml"):
        (tmp_path / name).write_text((src / name).read_text())
    driver = yaml.safe_load((tmp_path / "run_gxtb.yaml").read_text())
    driver["cif"] = str(ROOT / "examples/cifs/CdSe_zb.cif")
    driver.update(overrides)
    (tmp_path / "run_gxtb.yaml").write_text(yaml.safe_dump(driver, sort_keys=False))
    return tmp_path / "run_gxtb.yaml"


def test_same_key_in_two_files_is_an_error(tmp_path: Path) -> None:
    path = _write_pack(tmp_path)
    (tmp_path / "motifs.yaml").write_text(
        (tmp_path / "motifs.yaml").read_text()
        + "\ngraph_rules:\n  decoration_mode: motif_graph\n"
    )
    with pytest.raises(ValueError) as excinfo:
        load_geometry_pack(path)
    message = str(excinfo.value)
    assert "graph_rules.decoration_mode" in message
    # both filenames, so the duplicate can be found without bisecting
    assert "graph_rules.yaml" in message and "motifs.yaml" in message


def test_unknown_graph_rule_key_is_rejected(tmp_path: Path) -> None:
    path = _write_pack(tmp_path)
    text = (tmp_path / "graph_rules.yaml").read_text()
    (tmp_path / "graph_rules.yaml").write_text(
        text.replace("selection_order:", "selection_ordr:")
    )
    with pytest.raises(ValueError) as excinfo:
        load_geometry_pack(path)
    assert "selection_ordr" in str(excinfo.value)
    assert "selection_order" in str(excinfo.value)  # the suggestion


def test_missing_include_is_reported(tmp_path: Path) -> None:
    path = _write_pack(tmp_path, include=["graph_rules.yaml", "nope.yaml"])
    with pytest.raises(FileNotFoundError):
        load_geometry_pack(path)


def test_nested_include_is_rejected(tmp_path: Path) -> None:
    path = _write_pack(tmp_path)
    (tmp_path / "motifs.yaml").write_text(
        "include: [embed.yaml]\n" + (tmp_path / "motifs.yaml").read_text()
    )
    with pytest.raises(ValueError, match="nested 'include'"):
        load_geometry_pack(path)


def test_every_declared_rule_key_is_actually_consumed() -> None:
    """A key in the vocabulary must reach the spec, or it is a lie.

    Each name is set to a sentinel and the pack reloaded; the key must show up
    either in the graph-rules mapping or on a pack accessor.  Without this the
    vocabulary could drift into advertising keys the loader ignores -- the
    exact failure the strict check exists to prevent.
    """

    pack = load_geometry_pack(PACK)
    mapping = pack.nucleation_graph_rules_mapping()
    accessors = {
        "require_inorganic_connected",
        "enforce_min_cn",
        "coordination",
    }
    missing = sorted(
        key
        for key in NUCLEATION_GRAPH_RULE_KEYS
        if key not in mapping and key not in accessors
    )
    assert not missing, f"declared but never consumed: {missing}"


def test_spec_loads_composed_pack() -> None:
    """The run driver is usable as a nucleation spec, not just as a pack."""

    from builder.nucleation.spec import load_nucleation_spec

    spec = load_nucleation_spec(str(PACK))
    assert spec.graph_rules.decoration_mode == "motif_bridge_first"
