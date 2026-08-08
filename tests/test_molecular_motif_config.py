from __future__ import annotations

from pathlib import Path

from builder.nucleation import load_geometry_pack, load_nucleation_spec


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "geometry_packs/cdse_cdcl2_molecular_v2.yaml"
MOTIF = ROOT / "geometry_packs/cdse_cdcl2_motif_v1.yaml"


def test_builder_yamls_are_self_contained_and_directly_runnable() -> None:
    for path, mode in (
        (CURRENT, "skeleton_bridge_first"),
        (MOTIF, "motif_bridge_first"),
    ):
        spec = load_nucleation_spec(path)
        pack = load_geometry_pack(path)
        assert Path(spec.geometry_pack).resolve() == path.resolve()
        assert spec.graph_rules.decoration_mode == mode
        assert pack.raw["relaxation"]["method"] == "GFN1-xTB"
        assert Path(spec.cif).is_file()


def test_motif_yaml_owns_motifs_and_reconstruction() -> None:
    pack = load_geometry_pack(MOTIF)
    assert set(pack.raw["motifs"]) == {
        "Se-Cd2", "Se-Cd3", "Se-Cd4", "Cl-Cd1", "Cl-Cd2", "Cl-Cd3"
    }
    assert pack.raw["junctions"]["coplanar_shared_pair"][0]["when"] == {
        "anion": "Se", "anion_cn": 2, "ligand": "Cl", "ligand_cn": 2
    }
    assert "nonbonded" not in pack.raw
    assert "skeleton_motifs" not in pack.raw
    assert pack.motifs == {}
    assert pack.raw["reconstruction"]["method"] == "motif_factor"
    assert pack.raw["reconstruction"]["audit"] == "clashes_only"
    assert pack.raw["geometry_reference"] == "cdse_cdcl2_bridge_first.yaml"
    assert pack.raw["graph_rules"]["bridge_first_hard_max_bridges_per_cd"] == 2
    assert pack.raw["graph_rules"]["forbid_mu3_host_bridge_overlap"] is True
    # The reference pack supplies the detailed executable tables; the motif
    # file only owns the graph vocabulary and run policy.
    assert len(pack.bonds) > 15
    # Runtime YAML stays concise: motif-local geometry is compiled in memory
    # instead of duplicated as current-builder bond/angle/dihedral tables.
    text = MOTIF.read_text()
    assert "bond_A_by_linker_cn: {2: 2.5, 3: 2.6, 4: 2.65}" in text
    assert "\nbonds:" not in text
    assert "\nangles:" not in text
    assert len(text.splitlines()) < 160
