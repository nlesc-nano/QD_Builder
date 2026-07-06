from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _require_rdkit() -> None:
    if importlib.util.find_spec("rdkit") is None:
        raise unittest.SkipTest("RDKit is required for charged ligand exchange placement")


def _read_symbols(path: Path) -> list[str]:
    lines = path.read_text().strip().splitlines()
    return [line.split()[0] for line in lines[2:]]


def _run_builder(yaml_name: str, out_name: str) -> Counter:
    _require_rdkit()
    out = ROOT / "tests/out" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "builder",
        str(ROOT / "examples/cifs/CdSe_zb.cif"),
        str(ROOT / "tests" / yaml_name),
        "-o",
        str(out),
        "--positive-q-mode",
        "add",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    return Counter(_read_symbols(out))


def test_cdse_native_anion_x_type_exchange_preserves_cd_and_balances_charge() -> None:
    counts = _run_builder(
        "test_cdse_x_type_native_anion_exchange.yaml",
        "cdse_x_type_native_anion_exchange.xyz",
    )
    exchanges = counts["S"]

    assert exchanges == 31
    assert counts["Cd"] == 477
    assert counts["Se"] == 414 - exchanges
    assert counts["Cl"] == 126 + exchanges
    assert 2 * counts["Cd"] - 2 * counts["Se"] - counts["Cl"] - exchanges == 0


def test_cdse_native_cation_x_type_exchange_preserves_se_and_balances_charge() -> None:
    counts = _run_builder(
        "test_cdse_x_type_native_cation_exchange.yaml",
        "cdse_x_type_native_cation_exchange.xyz",
    )
    exchanges = counts["N"]

    assert exchanges == 7
    assert counts["Cd"] == 477 - exchanges
    assert counts["Se"] == 414
    assert counts["Cl"] == 126 - exchanges
    assert 2 * counts["Cd"] - 2 * counts["Se"] - counts["Cl"] + exchanges == 0
