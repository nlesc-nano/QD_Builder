"""Facet geometry of the Wulff polyhedra, per-atom site roles, and charge balance of pure {100} zinc-blende cubes."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from builder.polyhedron import clip, core_roles, facet_labels, fitted_planes, summary

ROOT = Path(__file__).resolve().parents[1]
CDSE = ROOT / "examples/cifs/CdSe_zb.cif"
CUBE = [np.array(v, float) for v in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]
TETRA = [np.array(v, float) / np.sqrt(3) for v in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]


def _cube(d=1.0):
    return [(n, d) for n in CUBE], ["100"] * 6


def test_cube_geometry():
    g = summary(clip(*_cube()))
    assert np.isclose(g["areas_A2"]["100"], 24.0)
    assert g["edges"]["100|100"]["n"] == 12 and np.isclose(g["edges"]["100|100"]["length_A"], 24.0)
    assert g["n_vertices"] == 8 and g["vertex_types"] == {"100+100+100": 8}
    assert np.isclose(g["volume_A3"], 8.0)


def test_truncated_cube_mixes_facet_types():
    planes, labels = _cube()
    planes += [(n, 2.5 / np.sqrt(3)) for n in TETRA]
    labels += ["111A"] * 4
    g = summary(clip(planes, labels))
    assert set(g["areas_A2"]) == {"100", "111A"}
    assert g["edges"]["100|111A"]["n"] == 12                    # four corner triangles, three edges each
    assert g["vertex_types"] == {"100+100+100": 4, "100+100+111A": 12}
    assert np.isclose(g["volume_A3"], 8.0 - 4 * (0.5 ** 3) / 6, atol=1e-3)   # four corners of legs 0.5 cut off


def test_plane_touching_a_corner_has_no_face():
    planes, labels = _cube()
    planes.append((np.array([1, 1, 1]) / np.sqrt(3), 3 / np.sqrt(3)))   # through the (1,1,1) corner only
    labels.append("111A")
    g = summary(clip(planes, labels))
    assert "111A" not in g["areas_A2"] and g["n_vertices"] == 8 and g["planes"][-1]["area_A2"] == 0.0


def test_fitted_planes_follow_outermost_atoms():
    planes, labels = _cube(5.0)
    pts = np.array([[1.0, 0, 0], [-2.0, 0, 0], [0, 1.5, 0], [0, -1.5, 0], [0, 0, 3.0], [0, 0, -1.0]])
    d = [dd for _n, dd in fitted_planes(planes, pts)]
    assert np.allclose(d, [1.0, 2.0, 1.5, 1.5, 3.0, 1.0])


def test_roles_from_clipped_cube():
    planes, labels = _cube(5.0)
    poly = clip(planes, labels)
    pts = np.array([[0, 0, 0], [5.0, 0, 0], [4.6, 1.0, -1.0], [5.0, 4.7, 0.0], [4.8, 4.8, 4.8], [3.5, 0, 0]])
    assert core_roles(poly, pts, tol=1.0) == [
        "interior", "facet:100", "facet:100", "edge:100|100", "vertex:100+100+100", "interior"]


def test_roles_ignore_planes_without_area():
    # A {111} plane that only grazes the corner must not make the corner region a 111 facet.
    planes, labels = _cube(5.0)
    planes.append((np.array([1, 1, 1]) / np.sqrt(3), 15 / np.sqrt(3) - 0.2))   # tiny 111 corner triangle
    labels.append("111A")
    poly = clip(planes, labels)
    roles = core_roles(poly, np.array([[5.0, 2.0, 2.0], [5.0, 4.2, 0.0]]), tol=1.0)
    assert roles == ["facet:100", "edge:100|100"]


def test_zincblende_labels_split_polar_111():
    from pymatgen.core import Structure
    from builder.facets import expand_facets
    from builder.main import _resolve_facet_terminations
    from builder.nc_types import Facet

    struct = Structure.from_file(CDSE)
    charges = {"Cd": 2, "Se": -2}
    seeds = _resolve_facet_terminations(struct, [
        Facet(1, 0, 0, 1.0, termination="cation_rich"),
        Facet(1, 1, 1, 1.0, termination="cation_rich"),
        Facet(-1, -1, -1, 1.0, termination="anion_rich"),
        Facet(1, 1, 0, 1.0),
    ], charges)
    facets = expand_facets(struct, seeds, proper_only=True)
    labels = facet_labels(struct, facets, charges)
    assert Counter(labels) == {"100": 6, "111A": 4, "111B": 4, "110": 12}



def test_labels_generic_high_index_and_hexagonal():
    from pymatgen.core import Structure
    from builder.facets import expand_facets
    from builder.nc_types import Facet

    charges = {"Cd": 2, "Se": -2}
    zb = Structure.from_file(CDSE)
    facets = expand_facets(zb, [Facet(2, 1, 0, 1.0), Facet(2, 1, 1, 1.0), Facet(-2, -1, -1, 1.0)], proper_only=True)
    assert Counter(facet_labels(zb, facets, charges)) == {"210": 12, "211A": 12, "211B": 12}   # {210} is not polar
    wz = Structure.from_file(ROOT / "examples/library/cifs/CdSe_wz.cif")
    facets = expand_facets(wz, [Facet(1, 0, 0, 1.0), Facet(0, 0, 1, 1.0), Facet(0, 0, -1, 1.0),
                                Facet(1, 0, 1, 1.0), Facet(1, 0, -1, 1.0)], proper_only=True)
    assert Counter(facet_labels(wz, facets, charges)) == {"100": 6, "001A": 1, "001B": 1, "101A": 6, "101B": 6}


# ---------------- builds ----------------
PRESET = yaml.safe_load((ROOT / "examples/library/cdse_zb.yaml").read_text())["preset"]


def _build(facets, size, centre, mode="add", tmp_path=None):
    from builder.scripts.generate_library import _build as build
    recipe = copy.deepcopy(PRESET)
    recipe["facets"] = facets
    recipe["size_unit_cells"] = [size] * 3
    recipe["construction_origin"] = {"center_on_species": centre}
    return build(str(CDSE), recipe, mode, Path(tmp_path) / f"{centre}_{size}_{mode}")


def _charge(symbols):
    c = Counter(symbols)
    return 2 * c["Cd"] - 2 * c["Se"] - c["Cl"]


def test_small_dot_geometry_and_roles(tmp_path):
    res = _build(PRESET["facets"], 1.5, "Cd", tmp_path=tmp_path)
    assert "error" not in res and _charge(res["symbols"]) == 0
    geo, roles = res["manifest"]["facet_geometry"], res["manifest"]["site_roles"]
    for key in ("construction", "atom_fitted"):
        g = geo[key]
        assert set(g["areas_A2"]) <= {"100", "111A", "111B"} and g["total_area_A2"] > 0
        assert g["n_vertices"] - g["n_edges"] + g["n_faces"] == 2          # Euler
        assert np.isclose(sum(g["areas_A2"].values()), g["total_area_A2"], atol=0.01)
    assert geo["atom_fitted"]["volume_A3"] < geo["construction"]["volume_A3"]
    syms = res["symbols"]
    assert len(roles["roles"]) == len(roles["host"]) == len(roles["is_core"]) == len(syms)
    for i, s in enumerate(syms):
        assert roles["is_core"][i] == (s != "Cl")
        if s == "Cl":
            h = roles["host"][i]
            assert syms[h] == "Cd" and roles["roles"][i] == roles["roles"][h]
            assert np.linalg.norm(res["pts"][i] - res["pts"][h]) < 3.2
        else:
            assert roles["host"][i] == -1
    kinds = Counter(r.split(":")[0] for r, c in zip(roles["roles"], roles["is_core"]) if c)
    assert kinds["interior"] > 0 and kinds["facet"] > 0 and kinds["edge"] > 0


@pytest.mark.parametrize("centre,size,mode", [("Cd", 2.0, "add"), ("Se", 1.5, "add"), ("Cd", 1.5, "remove")])
def test_pure_100_cube_is_neutral(centre, size, mode, tmp_path):
    for term in ("cation_rich", None):
        facet = {"hkl": "100", "scope": "family", "gamma": 1.0, **({"termination": term} if term else {})}
        res = _build([facet], size, centre, mode, tmp_path=tmp_path)
        assert "error" not in res
        assert _charge(res["symbols"]) == 0
        geo = res["manifest"]["facet_geometry"]["atom_fitted"]
        assert set(geo["areas_A2"]) == {"100"} and geo["edges"]["100|100"]["n"] == 12


def test_cut_consumed_by_prepass_is_a_clean_rejection(tmp_path):
    facets = [{"hkl": "-1-1-1", "scope": "family", "termination": "anion_rich", "gamma": 1.0},
              {"hkl": "110", "scope": "family", "gamma": 1.0}]
    res = _build(facets, 0.5, "Se", tmp_path=tmp_path)
    assert res["error"].startswith("ValueError: surface prepass removed every native atom")


def test_large_111a_tetrahedron_builds(tmp_path):
    # Used to crash in the ligand migration at neutrality (an acceptor cation on no facet).
    facets = [{"hkl": "111", "scope": "family", "termination": "cation_rich", "gamma": 1.0}]
    res = _build(facets, 3.5, "Cd", tmp_path=tmp_path)
    assert "error" not in res and _charge(res["symbols"]) == 0
    assert set(res["manifest"]["facet_geometry"]["atom_fitted"]["areas_A2"]) == {"111A"}
