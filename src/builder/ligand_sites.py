# src/builder/ligand_sites.py
"""
One ligand-site selector for every place that adds compensating ligands
(build passivation with --positive-q-mode add, and the {111} reconstruction).

Rules, applied the same way everywhere:

1. Capacity.  A cation takes at most its missing bonds (``host_deficit``).
   For zinc blende, no ligand may come closer to a cation it is not bonded
   to than the bulk bonded/non-bonded separator (midway between the first-
   and second-shell cation-anion distances), so no hidden bond pushes a
   cation above bulk CN.
2. Site types.  Bulk-lattice sites of the missing anions (mu1 on-top, and
   natural mu2/mu3 lattice bridges) for every material.  II-VI zinc blende
   also gets mu3 hollows and mu2 bridges over same-CN cations of
   cation-terminated {111} facets.  Preference: II-VI mu3 > mu2 > mu1;
   III-V mu1 first; other materials by net repair, then multiplicity.
3. Geometry (zinc blende).  Every candidate keeps 1.1 x bond from anions,
   LIGAND_SEP x the anion-anion distance from other ligands, and the shell
   separator from non-host cations.
4. Priority.  Lowest CN first (largest remaining deficit among a site's
   hosts, re-evaluated after every pick).  A bridge is only taken if the
   free cations left after it can still host every ligand still needed, at
   worst as mu1, so the charge stays reachable.
5. Uniformity.  Ligands are balanced across facets by the facet plane they
   sit outside of (not by their host, which may sit on an edge shared by two
   facets).  Within a facet: farthest from the ligands already added on it,
   then least crowding (sum 1/d^3), then farthest from all ligands, then
   lowest host index (deterministic).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .analysis import PairCuts, _pair_cut_calibrated, compute_cif_virtual_sites
from .nc_types import Plane

ABOVE_TOL = 0.4   # Å above a facet's outer native layer: an added (not lattice) ligand


def _native_planes(
    symbols: Sequence[str],
    pts: NDArray[np.float64],
    planes: Sequence[Plane],
    ligand: str,
) -> List[Tuple[NDArray[np.float64], float]]:
    """Unit normals with the offset of the outermost native (non-ligand) atom."""
    native = np.array([s != ligand for s in symbols])
    out = []
    for n, _d in planes:
        n = np.asarray(n, float)
        n = n / np.linalg.norm(n)
        out.append((n, float(np.max(pts[native] @ n))))
    return out


def _facet_of(pos: NDArray[np.float64], nplanes) -> int:
    """Facet plane the position sits outside of (largest height above it)."""
    return int(np.argmax([float(pos @ n) - d for n, d in nplanes]))


def select_ligand_sites(
    symbols: List[str],
    pts: NDArray[np.float64],
    *,
    host_deficit: Dict[int, int],
    n_needed: int,
    struct,
    charges: Dict[str, int],
    ligand: str,
    pair_cuts: Optional[PairCuts],
    planes: Sequence[Plane],
    surf_tol: float,
    zb_pair: Optional[Tuple[str, str]],
    bulk_map: Optional[Dict[str, int]] = None,
    avoid: Iterable[NDArray[np.float64]] = (),
    position_ok: Optional[Callable[[NDArray[np.float64]], bool]] = None,
    min_ligand_dist: float = 3.0,
) -> List[dict]:
    """
    Choose up to ``n_needed`` ligand sites on the given hosts.  Returns
    ``[{"pos", "hosts", "mu", "facet"}]`` in pick order.
    """
    from .facet_reconstruction import (
        _bridging_sites,
        _bulk_bond_length,
        _cation_shell_separator,
        _ligand_separation,
        _nn_cation_distance,
        _polar_111_facets,
    )
    from .analysis import coord_numbers_bipartite
    from .passivation_iterative import _trial_ligand_addition_score

    pts = np.asarray(pts, float)
    hosts_all = {h: int(d) for h, d in host_deficit.items() if int(d) > 0}
    if n_needed <= 0 or not hosts_all:
        return []
    is_ii_vi = zb_pair is not None and int(charges.get(zb_pair[0], 0)) == 2
    is_iii_v = zb_pair is not None and not is_ii_vi

    # --- candidates: lattice sites, plus II-VI {111} bridges ------------------
    mask = np.zeros(len(symbols), dtype=bool)
    mask[list(hosts_all)] = True
    lattice = compute_cif_virtual_sites(symbols, pts, charges, pair_cuts, struct, mask, list(planes), surf_tol)
    cands: List[dict] = []
    for site in lattice:
        hs = tuple(int(h) for h in site["hosts"])
        if all(h in hosts_all for h in hs):
            cands.append({"pos": np.asarray(site["pos"], float), "hosts": hs, "mu": len(hs)})

    geometry = None
    if zb_pair is not None:
        bond = _bulk_bond_length(struct)
        d_nn = _nn_cation_distance(struct)
        ligand_idx = {j for j, s in enumerate(symbols) if s == ligand}
        cation_idx = {j for j, s in enumerate(symbols) if int(charges.get(s, 0)) > 0}
        anion_idx = {j for j, s in enumerate(symbols) if int(charges.get(s, 0)) < 0 and s != ligand}
        geometry = (bond, _ligand_separation(bond), _cation_shell_separator(bond))
        if is_ii_vi:
            cat_facets = [f for f in _polar_111_facets(symbols, pts, struct, *zb_pair) if f.kind == "cation"]
            host_normal = {
                i: f.normal for f in cat_facets for i in f.outer if hosts_all.get(i) == 1
            }
            if host_normal:
                for mu in (3, 2):
                    for pos, hs in _bridging_sites(host_normal, pts, mu, d_nn, bond, ligand_idx, cation_idx):
                        cands.append({"pos": np.asarray(pos, float), "hosts": tuple(hs), "mu": mu})

    tree = cKDTree(pts)
    lig_pts_all = np.asarray([pts[j] for j, s in enumerate(symbols) if s == ligand], float)
    if not len(lig_pts_all):
        lig_pts_all = None

    def clear(c: dict) -> bool:
        """Rule 3: same distance limits for every candidate type (zinc blende)."""
        if position_ok is not None and not position_ok(c["pos"]):
            return False
        if geometry is None:
            return lig_pts_all is None or float(np.min(np.linalg.norm(lig_pts_all - c["pos"], axis=1))) >= min_ligand_dist
        bond, lig_sep, cat_sep = geometry
        for j in tree.query_ball_point(c["pos"], cat_sep + 0.5):
            if j in c["hosts"]:
                continue
            d = float(np.linalg.norm(pts[j] - c["pos"]))
            if j in cation_idx and d < cat_sep:
                return False
            if j in ligand_idx and d < lig_sep:
                return False
            if j in anion_idx and d < 1.1 * bond:
                return False
        return True

    cands = [c for c in cands if clear(c)]
    if not cands:
        return []

    # --- per-candidate scores ------------------------------------------------
    cn_bi = coord_numbers_bipartite(symbols, pts, charges, pair_cuts=pair_cuts)
    nplanes = _native_planes(symbols, pts, planes, ligand)
    for c in cands:
        _s, _g, touched, over = _trial_ligand_addition_score(
            symbols, pts, charges, ligand, pair_cuts, c["pos"],
            host_idx=c["hosts"][0], cn_before=cn_bi, pt_tree=tree, bulk_map=bulk_map,
            max_search_cut=max(4.5, abs(_pair_cut_calibrated(symbols[c["hosts"][0]], ligand, pair_cuts)) + 1.0),
            ref_struct=struct,
        )
        c["over"] = int(over)
        c["net"] = int(touched) - 2 * int(over)
        c["type_rank"] = -c["mu"] if is_iii_v else c["mu"]
        c["facet"] = _facet_of(c["pos"], nplanes)

    # --- existing ligands: spacing, balance and spread references -----------
    lig_pts = [pts[j] for j, s in enumerate(symbols) if s == ligand]
    counts: Dict[int, int] = defaultdict(int)
    added_on: Dict[int, List[NDArray[np.float64]]] = defaultdict(list)
    for q in lig_pts:
        k = _facet_of(q, nplanes)
        n, d = nplanes[k]
        if float(q @ n) - d > ABOVE_TOL:      # added earlier (not on a lattice anion site)
            counts[k] += 1
            added_on[k].append(q)
    taken = list(lig_pts) + [np.asarray(a, float) for a in avoid]
    spacing = geometry[1] if geometry is not None else min_ligand_dist
    avoid_pts = [np.asarray(a, float) for a in avoid]

    use: Dict[int, int] = defaultdict(int)
    mu1_hosts = {c["hosts"][0] for c in cands if c["mu"] == 1}

    def left(h: int) -> int:
        return hosts_all[h] - use[h]

    # sum(max(0, left(h))) over mu1 hosts, kept up to date as hosts are used
    free_total = sum(max(0, left(h)) for h in mu1_hosts)

    def free_capacity(extra: Tuple[int, ...] = ()) -> int:
        return free_total - sum(1 for h in set(extra) if h in mu1_hosts and left(h) >= 1)

    taken_arr = np.asarray(taken, float) if taken else None
    avoid_arr = np.asarray(avoid_pts) if avoid_pts else None

    def same_facet_stats(c):
        """Distance to the nearest ligand added on c's facet, and their crowding."""
        same = added_on[c["facet"]]
        if same:
            d = np.linalg.norm(np.asarray(same) - c["pos"], axis=1)
            return float(d.min()), float(np.sum(1.0 / np.maximum(d, 1e-6) ** 3))
        return np.inf, 0.0

    def crowd_rounded(c) -> float:
        """round(crowding, 6) of same_facet_stats.  The running sum decides it
        unless it lies within 1e-9 of a rounding boundary; then the full sum
        over the facet's ligands is taken, as same_facet_stats computes it."""
        scaled = c["crowd_sum"] * 1e6
        if abs(scaled - math.floor(scaled) - 0.5) < 1e-3:
            return round(same_facet_stats(c)[1], 6)
        return round(c["crowd_sum"], 6)

    # Spread-key ingredients per candidate, kept current as ligands are
    # picked: d_new (nearest pick), d_taken (nearest existing ligand or
    # `avoid` point), d_vac (nearest `avoid` point), d_same/crowd_sum
    # (ligands added on the candidate's facet).
    for c in cands:
        c["d_new"] = np.inf
        c["d_taken"] = float(np.min(np.linalg.norm(taken_arr - c["pos"], axis=1))) if taken_arr is not None else np.inf
        c["d_vac"] = float(np.min(np.linalg.norm(avoid_arr - c["pos"], axis=1))) if avoid_arr is not None else np.inf
        c["d_same"], c["crowd_sum"] = same_facet_stats(c)

    # Host capacity per candidate (smallest and largest left() over its
    # hosts) and its priority level; a pick changes them only for the
    # candidates that share one of its hosts.
    def refresh_level(c) -> None:
        lefts = [left(h) for h in c["hosts"]]
        c["min_left"] = min(lefts)
        need = max(lefts)
        c["level"] = (need, c["over"] == 0, c["type_rank"]) if zb_pair is not None else (need, c["net"], c["type_rank"])

    by_host: Dict[int, List[dict]] = defaultdict(list)
    for c in cands:
        refresh_level(c)
        for h in set(c["hosts"]):
            by_host[h].append(c)

    picks: List[dict] = []
    new_pts: List[NDArray[np.float64]] = []
    while len(picks) < n_needed:
        remaining_after = n_needed - len(picks) - 1
        elig = []
        alive = []
        for c in cands:
            # A full host or a pick closer than `spacing` excludes c for good
            # (hosts only fill, picks only add), so c is dropped from cands.
            if c["min_left"] < 1:
                continue
            if new_pts:
                d_new = c["d_new"]
                if abs(d_new - spacing) < 1e-6:
                    d_new = float(np.min(np.linalg.norm(np.asarray(new_pts) - c["pos"], axis=1)))
                if d_new < spacing:
                    continue
            alive.append(c)
            if c["mu"] >= 2 and free_capacity(c["hosts"]) < remaining_after:
                continue
            elig.append(c)
        cands = alive
        if not elig:
            break

        best_level = max(c["level"] for c in elig)
        elig = [c for c in elig if c["level"] == best_level]
        least = min(counts[c["facet"]] for c in elig)
        elig = [c for c in elig if counts[c["facet"]] == least]

        def spread_key(c):
            d_all = min(c["d_taken"], c["d_new"])      # nearest of existing ligands + picks
            return (round(c["d_same"], 2), -crowd_rounded(c), round(min(d_all, c["d_vac"]), 6),
                    tuple(-h for h in c["hosts"]))

        # Only candidates with the largest leading key can win, so the rest
        # of the key is evaluated for those alone (same order, same winner).
        top = max(round(c["d_same"], 2) for c in elig)
        best = max((c for c in elig if round(c["d_same"], 2) == top), key=spread_key)
        cands = [c for c in cands if c is not best]
        picks.append({"pos": best["pos"], "hosts": best["hosts"], "mu": best["mu"], "facet": best["facet"]})
        new_pts.append(best["pos"])
        added_on[best["facet"]].append(best["pos"])
        counts[best["facet"]] += 1
        for h in best["hosts"]:
            before = max(0, left(h)) if h in mu1_hosts else 0
            use[h] += 1
            if h in mu1_hosts:
                free_total += max(0, left(h)) - before
        for h in set(best["hosts"]):
            for c in by_host[h]:
                refresh_level(c)
        if cands:
            d = np.linalg.norm(np.asarray([c["pos"] for c in cands]) - best["pos"], axis=1)
            for c, dc in zip(cands, d.tolist()):
                if dc < c["d_new"]:
                    c["d_new"] = dc
                if c["facet"] == best["facet"]:
                    if dc < c["d_same"]:
                        c["d_same"] = dc                # |c - best| is the new facet distance
                    c["crowd_sum"] += 1.0 / max(dc, 1e-6) ** 3
    return picks
