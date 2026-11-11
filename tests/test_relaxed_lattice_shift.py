"""The fractional shift between a nanocrystal and its CIF cell survives relaxation.

A relaxed dot's surface atoms move by more than the strict test allows (0.05 fractional, ~0.3 A for CdSe); the shift is
then the one most native atoms agree on, so the L-type sites of relaxed structures (--from-xyz) are found.
"""
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

from builder import analysis
from builder.analysis import nc_bulk_frac_shift

CIF = Path(__file__).resolve().parents[1] / "examples" / "cifs" / "CdSe_zb.cif"


def _dot(radius=12.0):
    bs = Structure.from_file(CIF)
    sc = bs * (6, 6, 6)
    c = sc.lattice.get_cartesian_coords([0.5, 0.5, 0.5])
    keep = [s for s in sc.sites if np.linalg.norm(s.coords - c) < radius]
    return bs, [s.specie.symbol for s in keep], np.array([s.coords for s in keep]) - c


def _on_lattice(bs, syms, pts, shift, tol=0.06):
    f = (bs.lattice.get_fractional_coords(pts) + shift + 0.5) % 1 - 0.5
    ok = []
    for e, g in zip(syms, f):
        sites = np.array([s.frac_coords for s in bs.sites if s.specie.symbol == e])
        ok.append(np.linalg.norm((sites - g + 0.5) % 1 - 0.5, axis=1).min() <= tol)
    return np.array(ok)


def test_strict_shift_on_the_built_dot():
    bs, syms, pts = _dot()
    sh = nc_bulk_frac_shift(syms, pts + np.array([0.37, -1.21, 2.05]), bs)
    assert sh is not None and _on_lattice(bs, syms, pts + np.array([0.37, -1.21, 2.05]), sh).all()


def test_shift_of_a_relaxed_dot(monkeypatch):
    bs, syms, pts = _dot()
    rng = np.random.default_rng(3)
    r = np.linalg.norm(pts, axis=1)
    moved = pts + np.where((r > 6.0)[:, None], rng.normal(0.0, 0.15, pts.shape), 0.0)   # the surface relaxes (~0.25 A)
    moved += np.array([0.37, -1.21, 2.05])
    sh = nc_bulk_frac_shift(syms, moved, bs)
    assert sh is not None
    core = r <= 6.0
    assert _on_lattice(bs, np.array(syms)[core], moved[core], sh).all()   # the core sits on its own sites
    monkeypatch.setattr(analysis, "_frac_shift_by_vote", lambda *a, **k: None)
    assert nc_bulk_frac_shift(syms, moved, bs) is None                    # the strict test alone fails here
