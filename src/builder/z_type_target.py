# src/builder/z_type_target.py
"""
Bring a charge-neutral dot to a target number of MX_q units.

After charge balance the core holds k_X units (its excess cations, e.g.
k_X = N_Cd - N_Se for CdSe/Cl). For a target k > k_X the missing k - k_X
units are added as Z-type complexes on free surface anions (the Z-type
ligand machinery); for k < k_X the extra ones are displaced, each a core
cation with q ligands, its own first. Every step moves one neutral unit, and
the steps form one deterministic sequence, so the build for k contains the
one for k + 1 (displacement) or k - 1 (addition).

Units are split over `per_family` by its weights with the divisor (D'Hondt)
rule: the next unit goes to the family with the largest w_f / (n_f + 1)
among those that still have a site. Without `per_family` every site counts.
A family is a facet label ("111B"), an edge or vertex role
("edge:100|111B"), or "edge" / "vertex"; for addition it is the role of the
host anion, for displacement that of the removed cation.

A displacement is refused when it would leave a ligand with no cation or
split the structure into pieces.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .analysis import _pair_cut_calibrated, derive_pair_cuts_from_cif, nc_bulk_frac_shift
from .composition import core_composition


def _matches(family: str, role: Optional[str]) -> bool:
    from .z_type_ligand_posttreat import site_family_matches
    if family == "all":
        return role is not None
    return role is not None and site_family_matches(family, role)


def _next_family(families: Sequence[Tuple[str, float]], counts: Dict[str, int], exhausted: set) -> Optional[str]:
    best = None
    for fam, wt in families:
        if wt <= 0 or fam in exhausted:
            continue
        score = wt / (counts[fam] + 1)
        if best is None or score > best[0] + 1e-12:
            best = (score, fam)
    return None if best is None else best[1]


def _shares(families: Sequence[Tuple[str, float]], n: int) -> Dict[str, int]:
    """Divisor-rule split of n units over the weights, ignoring site limits."""
    counts = {f: 0 for f, _w in families}
    for _ in range(n):
        fam = _next_family(families, counts, set())
        if fam is None:
            break
        counts[fam] += 1
    return counts


# ---------------- addition ----------------
def _add_units(syms, pts, cfg, struct, planes, roles, families, n_add, complex_, distribution, seed):
    from .neutral_ligand_posttreat import _native_core_species
    from .z_type_ligand_posttreat import (
        _anion_site_candidates, _family_plans, _free_sites, _parse_complex, _place_unit,
        _prepare_anion_fragment, _site_users, z_type_ligand_host_sites,
    )
    parsed = _parse_complex(complex_, cfg.charges)
    metal, frags = parsed["cation_symbol"], parsed["anion_fragments"]
    cfg.charges.setdefault(metal, int(parsed["cation_charge"]))
    preps = []
    for k, frag in enumerate(frags):
        if frag["is_atomic"]:
            cfg.charges.setdefault(frag["symbol"], int(frag["charge"]))
            preps.append(None)
        else:
            preps.append(_prepare_anion_fragment(frag["mol"], seed + 101 + k, cfg.post_treatment.z_type_ligands.ff))
    work_syms = list(syms)
    work_pts = [np.asarray(p, float) for p in np.asarray(pts, float)]
    f_shift = nc_bulk_frac_shift(work_syms, np.asarray(work_pts), struct)
    if f_shift is None:
        raise ValueError("cannot map the particle onto the CIF lattice")
    native = _native_core_species(cfg, struct)
    dropped: Dict[str, list] = {}
    cur = np.asarray(work_pts, float)
    sites = z_type_ligand_host_sites(work_syms, cur, cfg, struct, planes, native, dropped=dropped)
    cands = _anion_site_candidates(sites, work_syms, cur, cfg, struct, planes, f_shift)
    n = len(frags)

    def fam_of(site_list):
        return {f: [i for i, s in enumerate(site_list) if _matches(f, roles[s["hosts"][0]])] for f, _w in families}

    fam_sites = fam_of(sites)
    crowded = fam_of(dropped.get("crowded", []))
    plans = _family_plans(sites, cands, n, fam_sites, distribution, seed)
    pointer = {f: 0 for f, _w in families}
    users = _site_users(cands)
    used_ids: set = set()
    taken: set = set()
    counts = {f: 0 for f, _w in families}
    exhausted: set = set()
    order: List[str] = []
    while sum(counts.values()) < n_add:
        fam = _next_family(families, counts, exhausted)
        if fam is None:
            break
        placed = False
        while pointer[fam] < len(plans[fam]):
            idx, take = plans[fam][pointer[fam]]
            pointer[fam] += 1
            if idx in taken:
                continue
            if any(c["id"] in used_ids for c in take):
                take = _free_sites(idx, cands, users, used_ids, taken)[:n]
                if len(take) < n:
                    continue
            if _place_unit(sites[idx], take, metal, frags, preps, work_syms, work_pts) is None:
                continue
            used_ids.update(c["id"] for c in take)
            taken.add(idx)
            placed = True
            break
        if not placed:
            exhausted.add(fam)
            continue
        counts[fam] += 1
        order.append(fam)
    per_family = {
        f: {"weight": w, "sites": len(fam_sites[f]), "crowded_sites": len(crowded[f]), "added": counts[f]}
        for f, w in families
    }
    entry = {"formula": complex_, "placed": sum(counts.values()),
             "anion_symbols": [f["symbol"] if f["is_atomic"] else None for f in frags]}
    return work_syms, np.asarray(work_pts, float), per_family, order, entry


# ---------------- displacement ----------------
def _bonds(syms, pts, charges, cuts):
    """Cation-anion bonds (pairs i<j) by the calibrated pair cutoffs."""
    P = np.asarray(pts, float)
    tree = cKDTree(P)
    out = []
    for i, j in tree.query_pairs(4.0):
        qi, qj = charges.get(syms[i], 0), charges.get(syms[j], 0)
        if qi * qj >= 0:
            continue
        if np.linalg.norm(P[i] - P[j]) <= _pair_cut_calibrated(syms[i], syms[j], cuts) + 1e-6:
            out.append((i, j))
    return out


def _n_components(n_atoms: int, bonds, removed: set) -> int:
    keep = np.array([i not in removed for i in range(n_atoms)])
    idx = np.cumsum(keep) - 1
    e = [(idx[i], idx[j]) for i, j in bonds if keep[i] and keep[j]]
    m = int(keep.sum())
    if m == 0:
        return 0
    if not e:
        return m
    a, b = np.array(e).T
    g = coo_matrix((np.ones(len(e)), (a, b)), shape=(m, m))
    return int(connected_components(g, directed=False)[0])


def _removal_unit(c: int, syms, pts, nbr, metal, ligand, q, core) -> Optional[List[int]]:
    """Cation c and q ligands, its own first; None if a ligand would be left without a cation."""
    P = np.asarray(pts, float)
    own = [x for x in nbr[c] if syms[x] == ligand]
    sole = [x for x in own if sum(1 for h in nbr[x] if h != c) == 0]
    if len(sole) > q:
        return None
    rest = sorted((x for x in own if x not in sole),
                  key=lambda x: (len(nbr[x]), float(np.linalg.norm(P[x] - P[c]))))
    chosen = sole + rest[: q - len(sole)]
    if len(chosen) < q:
        others = [i for i, s in enumerate(syms) if s == ligand and i not in chosen and not core[i]]
        others.sort(key=lambda x: float(np.linalg.norm(P[x] - P[c])))
        chosen += others[: q - len(chosen)]
    return chosen if len(chosen) == q else None


def _remove_units(syms, pts, cfg, roles, core, families, n_remove, metal, ligand, q, cif_path):
    cuts = derive_pair_cuts_from_cif(cif_path, cfg.charges, safety=1.00)
    work_syms, work_pts = list(syms), np.asarray(pts, float).copy()
    work_roles, work_core = list(roles), np.asarray(core, bool).copy()
    native_anions = {s for s, c in zip(syms, core) if c and cfg.charges.get(s, 0) < 0}
    counts = {f: 0 for f, _w in families}
    initial = {}
    exhausted: set = set()
    order: List[str] = []
    while sum(counts.values()) < n_remove:
        bonds = _bonds(work_syms, work_pts, cfg.charges, cuts)
        nbr = defaultdict(list)
        for i, j in bonds:
            nbr[i].append(j)
            nbr[j].append(i)
        n_comp = _n_components(len(work_syms), bonds, set())
        centre = work_pts[work_core].mean(axis=0)

        def candidates(fam):
            out = []
            for c, s in enumerate(work_syms):
                if s != metal or not work_core[c] or not _matches(fam, work_roles[c]):
                    continue
                n_own = sum(1 for x in nbr[c] if work_syms[x] == ligand)
                if n_own == 0:
                    continue
                cn_core = sum(1 for x in nbr[c] if work_syms[x] in native_anions)
                out.append(((cn_core, -n_own, -float(np.linalg.norm(work_pts[c] - centre)),
                             tuple(np.round(work_pts[c], 3))), c))
            return [c for _key, c in sorted(out)]

        if not initial:
            initial = {f: len(candidates(f)) for f, _w in families}
        fam = _next_family(families, counts, exhausted)
        if fam is None:
            break
        done = False
        for c in candidates(fam):
            unit = _removal_unit(c, work_syms, work_pts, nbr, metal, ligand, q, work_core)
            if unit is None:
                continue
            removed = {c, *unit}
            if _n_components(len(work_syms), bonds, removed) > n_comp:
                continue
            keep = [i for i in range(len(work_syms)) if i not in removed]
            work_syms = [work_syms[i] for i in keep]
            work_pts = work_pts[keep]
            work_roles = [work_roles[i] for i in keep]
            work_core = work_core[keep]
            done = True
            break
        if not done:
            exhausted.add(fam)
            continue
        counts[fam] += 1
        order.append(fam)
    per_family = {f: {"weight": w, "candidates": initial.get(f, 0), "removed": counts[f]} for f, w in families}
    return work_syms, work_pts, per_family, order


def run_z_type_target(
    syms: List[str],
    pts: NDArray[np.float64],
    cfg,
    struct,
    planes,
    cif_path: str,
    *,
    roles: Optional[List[str]],
    core: np.ndarray,
    ligand: str,
) -> Tuple[List[str], NDArray[np.float64], dict, Optional[dict]]:
    """
    Returns the structure, the `z_type_target_ledger` and, for additions, a
    Z-type ledger entry (formula, placed) for the composition.
    """
    spec = cfg.post_treatment.z_type_target
    print("\n[post-treatment] ── Z-type target ───────────────────────────────────────")
    comp = core_composition(struct, syms, core, cfg.charges, ligand)
    k_x, mx = comp.get("k_X"), comp.get("mx_unit")
    ledger = {"target_total": int(spec.target_total), "k_X": k_x, "complex": spec.complex or mx}
    if k_x is None or mx is None:
        print("  [warning] the core has no single excess cation to count in MX_q units; skipping.")
        return syms, pts, {**ledger, "direction": "none", "error": "k_X undefined"}, None
    families = list(spec.per_family) or [("all", 1.0)]
    if any(f != "all" for f, _w in families) and roles is None:
        print("  [warning] per-family targets need the facet geometry (site roles); skipping.")
        return syms, pts, {**ledger, "direction": "none", "error": "no site roles"}, None
    roles = list(roles) if roles is not None else ["facet:any" if c else None for c in core]
    delta = int(spec.target_total) - int(k_x)
    ledger["delta"] = delta
    shares = _shares(families, abs(delta))
    entry = None
    if delta > 0:
        syms, pts, per_family, order, entry = _add_units(
            syms, pts, cfg, struct, planes, roles, families, delta, ledger["complex"], spec.distribution,
            cfg.post_treatment.z_type_ligands.seed,
        )
        done = sum(v["added"] for v in per_family.values())
        ledger.update(direction="add", units=done, achieved_total=int(k_x) + done)
    elif delta < 0:
        metal = next(iter(comp["core_excess"]))
        q = int(round(float(cfg.charges.get(metal, 0)) / abs(float(cfg.charges.get(ligand, -1)))))
        syms, pts, per_family, order = _remove_units(
            syms, pts, cfg, roles, core, families, -delta, metal, ligand, q, cif_path,
        )
        done = sum(v["removed"] for v in per_family.values())
        ledger.update(direction="displace", units=done, achieved_total=int(k_x) - done)
    else:
        per_family, order = {f: {"weight": w} for f, w in families}, []
        ledger.update(direction="none", units=0, achieved_total=int(k_x))
    for f in per_family:
        per_family[f]["requested"] = shares.get(f, 0)
    ledger.update(shortfall=abs(delta) - ledger["units"], per_family=per_family, order=order)
    print(f"  → k_X = {k_x}, target {spec.target_total}: {ledger['direction']} {ledger['units']} "
          f"{ledger['complex']} unit(s); " + ", ".join(
              f"{f}: {v.get('added', v.get('removed', 0))}" for f, v in per_family.items()))
    if ledger["shortfall"]:
        print(f"  [warning] {ledger['shortfall']} unit(s) short of the target: no valid site left")
    return syms, pts, ledger, entry
