"""ZB motif census, Channel B injection, and job-status labels."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from builder.nucleation.molecular_growth import GrowthLog, _zb_motif_bin_done
from builder.nucleation.molecular_zb_growth import (
    lattice_k1_occupation,
    lattice_model,
)
from builder.nucleation.spec import load_nucleation_spec
from builder.nucleation.zb_motifs import (
    build_motif_occupation,
    census_occupations,
    classify_occupation_motifs,
    expected_motifs,
    inject_missing_motifs,
    job_status_label,
    short_fail_cause,
)

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "geometry_packs" / "cdse_cdcl2_zb"


@pytest.fixture(scope="module")
def zb_spec(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("zb_motif_spec")
    driver = yaml.safe_load((PACK / "run_gxtb.yaml").read_text())
    rules = yaml.safe_load((PACK / "graph_rules.yaml").read_text())
    merged = {k: v for k, v in driver.items() if k != "include"}
    merged.update(yaml.safe_load((PACK / "motifs.yaml").read_text()))
    merged.update(yaml.safe_load((PACK / "embed.yaml").read_text()))
    merged.update(rules)
    merged["cif"] = str(ROOT / "examples/cifs/CdSe_zb.cif")
    merged.setdefault("relaxation", {})["enabled"] = False
    path = tmp / "map.yaml"
    path.write_text(yaml.safe_dump(merged, sort_keys=False))
    return load_nucleation_spec(str(path))


def test_expected_motifs_by_stoich() -> None:
    assert "T1" in expected_motifs(1, 3)
    assert "chair" not in expected_motifs(2, 2)
    assert "chair" in expected_motifs(3, 2)
    assert "adamantane" in expected_motifs(4, 2)
    assert "T3" in expected_motifs(4, 6)
    assert "two_cage" in expected_motifs(5, 4)
    assert "T3" not in expected_motifs(4, 2)


def test_t1_occupation_is_secd4(zb_spec) -> None:
    model = lattice_model(zb_spec)
    occ = lattice_k1_occupation(zb_spec, model, p=3)
    assert occ is not None
    counts = classify_occupation_motifs(occ)
    assert counts["T1"] >= 1


def test_build_t3_contains_adamantane(zb_spec) -> None:
    model = lattice_model(zb_spec)
    occ = build_motif_occupation("T3", zb_spec, model, k=4, p=6)
    assert occ is not None
    assert occ.k == 4 and occ.p == 6
    counts = classify_occupation_motifs(occ)
    assert counts["adamantane"] >= 1
    assert counts["T3"] >= 1


def test_build_and_detect_adamantane(zb_spec) -> None:
    model = lattice_model(zb_spec)
    occ = build_motif_occupation("adamantane", zb_spec, model, k=4, p=2)
    assert occ is not None, "Channel B could not build Cd6Se4 adamantane"
    assert occ.k == 4 and occ.p == 2
    assert occ.parent_id == "channel_b"
    counts = classify_occupation_motifs(occ)
    assert counts["adamantane"] >= 1
    assert counts["chair"] == 1
    assert counts["n6"] >= 4


def test_inject_missing_t1_into_empty_bin(zb_spec) -> None:
    model = lattice_model(zb_spec)
    filled, injected = inject_missing_motifs([], 1, 3, zb_spec, model)
    assert "T1" in injected
    assert filled
    assert census_occupations(filled).get("T1", 0) >= 1


def test_inject_skips_when_motif_already_present(zb_spec) -> None:
    model = lattice_model(zb_spec)
    occ = lattice_k1_occupation(zb_spec, model, p=3)
    assert occ is not None
    filled, injected = inject_missing_motifs([occ], 1, 3, zb_spec, model)
    assert "T1" not in injected
    assert len(filled) == 1


def test_job_status_labels() -> None:
    assert job_status_label(
        chemically_ok=True,
        propagation_eligible=True,
        topology_status="preserved",
    ) == ("in-path", "")
    assert job_status_label(
        chemically_ok=True,
        propagation_eligible=False,
        topology_status="changed",
    ) == ("off-path", "topology_changed")
    assert job_status_label(
        chemically_ok=True,
        propagation_eligible=False,
        topology_status="preserved",
    ) == ("off-path", "unconverged")
    kind, cause = job_status_label(
        chemically_ok=False,
        propagation_eligible=False,
        topology_status="changed",
        violations=["artifact:Cl-Se:0-6:2.20<2.80"],
    )
    assert kind == "failed"
    assert cause == "artifact:Cl-Se"
    assert short_fail_cause([], "abnormal termination of xtb") == "gxtb_abort"


def test_zb_motif_bin_done_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "zb_motifs.jsonl"
    path.write_text(
        '{"bin": "k007_p009", "k": 7, "p": 9}\n',
        encoding="utf-8",
    )
    assert _zb_motif_bin_done(tmp_path, 7, 9)
    assert not _zb_motif_bin_done(tmp_path, 7, 10)
    assert not _zb_motif_bin_done(tmp_path, 6, 9)


def test_growth_log_status_column(tmp_path: Path, capsys) -> None:
    log = GrowthLog(log_path=tmp_path / "growth_run.log")
    log.begin_block(3, label="Z k=5 p=8")
    log(
        "[growth-job] k=5 p=8 move=Z id=k005_p008_Zabc "
        "E_eV=-586456.7 t_s=10.0 recon_s=0.0 relax=in-path"
    )
    log(
        "[growth-job] k=5 p=8 move=Z id=k005_p008_Zdef "
        "E_eV=-586456.8 t_s=10.0 recon_s=0.0 "
        "relax=off-path err=topology_changed"
    )
    log(
        "[growth-job] k=5 p=8 move=Z id=k005_p008_Zghi "
        "E_eV=n/a t_s=10.0 recon_s=0.0 relax=failed err=artifact:Cl-Se"
    )
    log.close()
    text = (tmp_path / "growth_run.log").read_text()
    assert "in-path" in text
    assert "off-path:topology_changed" in text
    assert "failed:artifact:Cl-Se" in text
