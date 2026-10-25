"""A quick check of the compiled kernels against brute-force NumPy counts."""
import numpy as np


def self_check(n: int = 400, seed: int = 0) -> bool:
    """
    Compare pair_counts, pair_counts_local and cross_counts with all-pairs
    NumPy counts on random points. Raises AssertionError on a mismatch;
    returns True otherwise. Used as the wheel test in CI.
    """
    from ._core import cross_counts, pair_counts, pair_counts_local

    rng = np.random.default_rng(seed)
    pts = rng.uniform(-8.0, 8.0, (n, 3))
    types = rng.integers(0, 3, n).astype(np.int32)
    charge = np.array([2.0, -2.0, 0.0])[types]
    cutoff = np.array([[2.4, 2.9, 2.6], [2.9, 2.7, 2.5], [2.6, 2.5, 2.2]])

    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    near = (d <= cutoff[types[:, None], types[None, :]]) & ~np.eye(n, dtype=bool)
    opposite = (charge[:, None] * charge[None, :] < 0) | (charge[:, None] == 0)

    for bipartite, expected in ((False, near.sum(1)), (True, (near & opposite).sum(1))):
        counts, undecided = pair_counts(pts, types, charge, cutoff, bipartite, 1e-9)
        assert len(undecided) == 0, "random points should not sit on a cutoff"
        assert np.array_equal(counts, expected), f"pair_counts mismatch (bipartite={bipartite})"

        seeds = pts[:7]
        radius = float(cutoff.max()) + 1e-6
        atoms, local, undecided = pair_counts_local(pts, types, charge, cutoff, bipartite, 1e-9, seeds, radius)
        assert len(undecided) == 0
        assert np.array_equal(local, expected[atoms]), "pair_counts_local mismatch"
        must = np.flatnonzero((np.sqrt(((pts[:, None, :] - seeds[None, :, :]) ** 2).sum(-1)) <= radius).any(1))
        assert set(must.tolist()) <= set(atoms.tolist()), "pair_counts_local missed atoms near a seed"

    q, r = np.arange(0, n, 3), np.setdiff1d(np.arange(n), np.arange(0, n, 3))
    counts, undecided = cross_counts(pts[q], types[q], pts[r], types[r], cutoff, 1e-9)
    assert len(undecided) == 0
    assert np.array_equal(counts, (d[np.ix_(q, r)] <= cutoff[types[q][:, None], types[r][None, :]]).sum(1)), \
        "cross_counts mismatch"
    return True
