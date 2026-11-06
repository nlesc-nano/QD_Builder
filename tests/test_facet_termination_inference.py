"""
A facet's metadata `termination` must be inferable from its real native
atoms when the caller never configured one explicitly -- correct for every
caller (CLI YAML, --from-xyz, or a webapp that never heard of
"stoichiometric" at all), not just one that opts in.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_builder(cif: Path, yaml_path: Path, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "builder", str(cif), str(yaml_path), "-o", str(out), "--center"]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    metadata_path = out.parent / f"{out.stem}_metadata.json"
    return json.loads(metadata_path.read_text())


def test_polar_family_infers_cation_and_anion_rich(tmp_path):
    """{111}/{-1-1-1} on zinc-blende InP: no termination configured at all,
    yet every orientation must still resolve to a pure, correctly-signed
    cation_rich/anion_rich -- never "stoichiometric" (would mean the
    outer/single-membership filtering failed to exclude the opposite-
    species sublayer and shared edge/vertex atoms)."""
    cif = ROOT / "examples/cifs/InP_zb.cif"
    yaml_path = ROOT / "tests/test_facet_termination_inference.yaml"
    out = tmp_path / "polar.xyz"
    doc = _run_builder(cif, yaml_path, out)

    by_hkl = {tuple(f["hkl"]): f["termination"] for f in doc["facets"] if f["family_label"] == "{111}"}
    assert by_hkl, "expected {111}-family facets in the metadata"
    assert set(by_hkl.values()) == {"cation_rich", "anion_rich"}, by_hkl
    # Opposite faces must carry opposite termination.
    for hkl, term in by_hkl.items():
        opposite = tuple(-x for x in hkl)
        if opposite in by_hkl:
            assert by_hkl[opposite] != term, f"{hkl} and its opposite face {opposite} both {term}"


def test_nonpolar_family_infers_stoichiometric(tmp_path):
    """{110} on zinc-blende InP is non-polar by construction (equal
    cation/anion in every exposed plane) -- must infer "stoichiometric"
    with no explicit termination configured."""
    cif = ROOT / "examples/cifs/InP_zb.cif"
    yaml_path = ROOT / "tests/test_facet_termination_inference_110.yaml"
    out = tmp_path / "nonpolar.xyz"
    doc = _run_builder(cif, yaml_path, out)

    terms = {f["termination"] for f in doc["facets"] if f["family_label"] == "{110}"}
    assert terms == {"stoichiometric"}, terms


def test_explicit_termination_overrides_inference(tmp_path):
    """An explicit caller-provided termination always wins over inference,
    even if the real atom composition would've suggested otherwise."""
    cif = ROOT / "examples/cifs/InP_zb.cif"
    yaml_path = ROOT / "tests/test_inp_anion_rich.yaml"
    out = tmp_path / "explicit.xyz"
    doc = _run_builder(cif, yaml_path, out)

    configured = [f for f in doc["facets"] if f["termination"] is not None]
    assert configured, "expected at least one facet with an explicitly configured termination"
