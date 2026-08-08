"""Pack-derived virtual sites for molecular Cl decoration.

Unlike lattice nucleation (CIF tetrahedral holes), sites here come from the
geometry pack tables on a fixed Cd–Se frame.  Bridge placement bumps host
coordination numbers and **rebuilds** terminal virtual sites for those hosts
so a later terminal on a CN=3 acceptor uses the CN3 table, not a leftover CN2
direction.

Speed model (v2)
----------------
* Precompute all μ₂/μ₃ sites once per frame (Cd positions fixed).
* Enumerate **bridges first** (non-decreasing site index), then fill terminals.
* At most two azimuths per Cd–Cd pair; max_shared=1 → one Cl per pair.
* Hard **search-node** budget (not only emission cap).
* One clean frame per CN vector by default (frame_options).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import networkx as nx
import numpy as np

from ..nc_types import NucleationSpec
from .geometry_pack import GeometryPack
from .molecular import (
    EXACT_BOND_TOLERANCE,
    ExactEmbeddingError,
    _DecorationStatus,
    _DegreeSlice,
    _cation_degree_vectors,
    _clean_frames,
    _cross3,
    _index_blocks,
    _orthonormal_basis,
    _surplus_combinatorially_feasible,
    _three_sphere_intersections,
    _unit,
)
from .types import _State

FloatArray = np.ndarray

_POS_QUANT = 0.05
# Default cap on recursive nodes per (skeleton, CN vector, frame).
_DEFAULT_NODE_BUDGET = 50_000


@dataclass(frozen=True)
class VirtualSite:
    """One geometrically proposed Cl position and its Cd hosts."""

    hosts: Tuple[int, ...]  # sorted Cd atom ids
    position: Tuple[float, float, float]
    kind: str  # terminal | mu2 | mu3
    site_id: int = 0  # stable index for non-decreasing enumeration

    @property
    def key(self) -> Tuple[object, ...]:
        q = tuple(int(round(c / _POS_QUANT)) for c in self.position)
        return (self.kind, self.hosts, q)


def _quantize_pos(pos: FloatArray) -> Tuple[float, float, float]:
    return (
        float(round(float(pos[0]) / _POS_QUANT) * _POS_QUANT),
        float(round(float(pos[1]) / _POS_QUANT) * _POS_QUANT),
        float(round(float(pos[2]) / _POS_QUANT) * _POS_QUANT),
    )


def _contact_floor(spec: NucleationSpec, a: str, b: str) -> float:
    from .molecular_rules import pair_key

    key = pair_key(a, b)
    rule = spec.graph_rules.pair_rules.get(key)
    if rule is None or rule.bond_allowed:
        return 0.0
    return float(rule.min_distance or 0.0)


def _clashes(
    pos: FloatArray,
    coords: FloatArray,
    placed_mask: Sequence[bool],
    atoms: Sequence,
    spec: NucleationSpec,
    *,
    ignore: Sequence[int] = (),
) -> bool:
    ignore_set = set(ignore)
    cl_sym = spec.precursor.ligand
    for idx, is_placed in enumerate(placed_mask):
        if not is_placed or idx in ignore_set:
            continue
        floor = _contact_floor(spec, cl_sym, atoms[idx].symbol)
        if floor <= 0.0:
            continue
        if float(np.linalg.norm(pos - coords[idx])) < floor - 1.0e-6:
            return True
    return False


def terminal_directions_for_new_cn(
    fixed_dirs: Sequence[FloatArray],
    new_cn: int,
    pack: GeometryPack,
    host_symbol: str = "Cd",
) -> List[FloatArray]:
    """Candidate unit directions for a ligand that raises host CN to ``new_cn``.

    Directions use pack angle defaults for ``new_cn`` (not the pre-bond CN).
    After a bridge, a CN3 acceptor's terminal slots come from the CN3 table.
    """

    fixed = [
        d
        for d in (_unit(np.asarray(v, dtype=float)) for v in fixed_dirs)
        if d is not None
    ]
    if not fixed:
        return [np.array([1.0, 0.0, 0.0])]

    if new_cn <= 2:
        mean = np.zeros(3)
        for d in fixed:
            mean = mean + d
        opp = _unit(-mean)
        return [opp] if opp is not None else [np.array([1.0, 0.0, 0.0])]

    if new_cn == 3:
        angle = pack.center_angle_deg(host_symbol, 3, default=120.0) or 120.0
        if len(fixed) == 1:
            axis = fixed[0]
            _, u, _v = _orthonormal_basis(axis)
            half = np.radians(angle)
            return [
                np.cos(half) * axis + np.sin(half) * u,
                np.cos(half) * axis - np.sin(half) * u,
            ]
        a, b = fixed[0], fixed[1]
        normal = _unit(_cross3(a, b))
        if normal is None:
            _, normal, _ = _orthonormal_basis(a)
        bis = _unit(a + b)
        candidates: List[FloatArray] = []
        if bis is not None:
            out = _unit(-bis)
            if out is not None:
                candidates.append(out)
            perp = _unit(_cross3(normal, bis)) if normal is not None else None
            if perp is not None:
                theta = np.radians(angle)
                for sign in (-1.0, 1.0):
                    cand = _unit(np.cos(theta) * a + sign * np.sin(theta) * perp)
                    if cand is not None:
                        candidates.append(cand)
        # Keep at most 2 distinct directions (quantized).
        return _unique_dirs(candidates, limit=2) or [np.array([1.0, 0.0, 0.0])]

    # new_cn >= 4
    tet = pack.center_angle_deg(host_symbol, 4, default=109.471) or 109.471
    if len(fixed) == 1:
        axis = fixed[0]
        _, u, v = _orthonormal_basis(axis)
        half = np.radians(tet)
        return _unique_dirs(
            [
                np.cos(half) * axis + np.sin(half) * u,
                np.cos(half) * axis - np.sin(half) * u,
                np.cos(half) * axis + np.sin(half) * v,
            ],
            limit=2,
        )
    if len(fixed) == 2:
        a, b = fixed[0], fixed[1]
        normal = _unit(_cross3(a, b))
        if normal is None:
            _, normal, _ = _orthonormal_basis(a)
        bis = _unit(a + b)
        if bis is None:
            return [normal] if normal is not None else []
        elev = np.radians(max(tet - 90.0, 15.0))
        out: List[FloatArray] = []
        for sign in (-1.0, 1.0):
            cand = _unit(np.cos(elev) * (-bis) + sign * np.sin(elev) * normal)
            if cand is not None:
                out.append(cand)
        return _unique_dirs(out, limit=2)
    mean = np.zeros(3)
    for d in fixed:
        mean = mean + d
    seed = _unit(-mean)
    return [seed] if seed is not None else [np.array([1.0, 0.0, 0.0])]


def _unique_dirs(
    dirs: Sequence[FloatArray], *, limit: int = 4, tol: float = 0.15
) -> List[FloatArray]:
    kept: List[FloatArray] = []
    for d in dirs:
        u = _unit(np.asarray(d, dtype=float))
        if u is None:
            continue
        if any(float(np.dot(u, k)) > 1.0 - tol for k in kept):
            continue
        kept.append(u)
        if len(kept) >= limit:
            break
    return kept


def precompute_bridge_sites(
    *,
    cd_list: Sequence[int],
    frame: FloatArray,
    target_cn: Sequence[int],
    need0: Sequence[int],
    pack: GeometryPack,
    spec: NucleationSpec,
    se_ids: Sequence[int] = (),
    inorganic_edges: Sequence[Tuple[int, int]] = (),
    allow_mu3: bool = True,
) -> List[VirtualSite]:
    """All μ₂/μ₃ sites for one frame (Cd fixed). At most 2 points per pair.

    For μ₂, prefer the azimuth **opposite a shared Se** (embedder convention) so
    Cl–Se contacts clear the pair-rule floor.  A second antipodal candidate is
    kept only when no shared Se exists.
    """

    n_cd = len(cd_list)
    min_bridge = int(spec.graph_rules.min_bridged_host_cn)
    max_cl_hosts = int(spec.graph_rules.max_cn[spec.precursor.ligand])
    allowed = set(
        spec.graph_rules.allowed_neighbor_signatures.get(
            spec.precursor.ligand, ()
        )
    )
    cation = spec.core.cation
    se_set = set(se_ids)
    # Se neighbors of each Cd from the skeleton.
    se_of: Dict[int, Set[int]] = {h: set() for h in cd_list}
    for left, right in inorganic_edges:
        if left in se_of and right in se_set:
            se_of[left].add(right)
        elif right in se_of and left in se_set:
            se_of[right].add(left)

    def sig_ok(n_hosts: int) -> bool:
        return not allowed or f"{cation}{n_hosts}" in allowed

    sites: List[VirtualSite] = []
    seen: Set[Tuple[object, ...]] = set()

    def add(hosts: Tuple[int, ...], pos: FloatArray, kind: str) -> None:
        # Exact coordinates for placement; quantize only the identity key.
        exact = (float(pos[0]), float(pos[1]), float(pos[2]))
        key = (kind, hosts, _quantize_pos(pos))
        if key in seen:
            return
        seen.add(key)
        sites.append(
            VirtualSite(
                hosts=hosts,
                position=exact,
                kind=kind,
                site_id=len(sites),
            )
        )

    if max_cl_hosts >= 2 and sig_ok(2):
        for s1, s2 in combinations(range(n_cd), 2):
            if need0[s1] <= 0 or need0[s2] <= 0:
                continue
            if target_cn[s1] < min_bridge or target_cn[s2] < min_bridge:
                continue
            h1, h2 = cd_list[s1], cd_list[s2]
            r1 = pack.bond_length(
                "CdCl_bridge", int(target_cn[s1]), 2, default=2.40
            )
            r2 = pack.bond_length(
                "CdCl_bridge", int(target_cn[s2]), 2, default=2.40
            )
            sep = float(np.linalg.norm(frame[h2] - frame[h1]))
            if sep < 1.0e-12:
                continue
            axial = (r1 * r1 - r2 * r2 + sep * sep) / (2.0 * sep)
            height_sq = r1 * r1 - axial * axial
            if height_sq < -EXACT_BOND_TOLERANCE:
                continue
            height = sqrt(max(0.0, height_sq))
            ab_u, u_perp, v_perp = _orthonormal_basis(frame[h2] - frame[h1])
            base = frame[h1] + axial * ab_u
            hosts = (min(h1, h2), max(h1, h2))
            if height < 1.0e-8:
                add(hosts, base, "mu2")
                continue
            # Shared Se between the two hosts: push Cl opposite that Se.
            shared = sorted(se_of[h1].intersection(se_of[h2]))
            candidates: List[FloatArray] = []
            if shared:
                se = shared[0]
                toward = frame[se] - base
                toward = toward - float(np.dot(toward, ab_u)) * ab_u
                ref = _unit(-toward)  # opposite Se (embedder convention)
                if ref is None:
                    ref = u_perp
                candidates.append(base + height * ref)
                candidates.append(base - height * ref)
            else:
                candidates.append(base + height * u_perp)
                candidates.append(base - height * u_perp)

            def se_clearance(p: FloatArray) -> float:
                anions = shared if shared else list(se_set)
                if not anions:
                    return 0.0
                return min(
                    float(np.linalg.norm(p - frame[s]))
                    for s in anions
                    if s < frame.shape[0]
                )

            # Prefer better Se clearance; keep both if both clear later clash filter.
            candidates.sort(key=se_clearance, reverse=True)
            for pos in candidates[:2]:
                add(hosts, pos, "mu2")

    if allow_mu3 and max_cl_hosts >= 3 and sig_ok(3):
        for s1, s2, s3 in combinations(range(n_cd), 3):
            if need0[s1] <= 0 or need0[s2] <= 0 or need0[s3] <= 0:
                continue
            if (
                target_cn[s1] < min_bridge
                or target_cn[s2] < min_bridge
                or target_cn[s3] < min_bridge
            ):
                continue
            hosts_t = (cd_list[s1], cd_list[s2], cd_list[s3])
            radii = [
                pack.bond_length(
                    "CdCl_bridge", int(target_cn[s]), 3, default=2.55
                )
                for s in (s1, s2, s3)
            ]
            try:
                pts = _three_sphere_intersections(
                    [frame[h] for h in hosts_t], radii
                )
            except ExactEmbeddingError:
                continue
            ordered = tuple(sorted(hosts_t))
            # Prefer μ3 farther from nearby Se (same contact idea).
            def score(p: FloatArray) -> float:
                if not se_set:
                    return float(np.linalg.norm(p))
                return min(
                    float(np.linalg.norm(p - frame[s]))
                    for s in se_set
                    if s < frame.shape[0]
                )

            best = max(pts, key=score)
            add(ordered, best, "mu3")

    sites.sort(key=lambda s: (0 if s.kind == "mu3" else 1, s.hosts, s.position))
    return [
        VirtualSite(hosts=s.hosts, position=s.position, kind=s.kind, site_id=i)
        for i, s in enumerate(sites)
    ]


def _terminal_sites_for_state(
    *,
    cd_list: Sequence[int],
    coords: FloatArray,
    placed: Sequence[bool],
    current_cn: Sequence[int],
    target_cn: Sequence[int],
    need: Sequence[int],
    fixed_dirs: Sequence[List[FloatArray]],
    pack: GeometryPack,
    spec: NucleationSpec,
    atoms: Sequence,
) -> Dict[int, List[VirtualSite]]:
    """Terminal virtual sites per Cd slot, using post-placement CN."""

    cation = spec.core.cation
    allowed = set(
        spec.graph_rules.allowed_neighbor_signatures.get(
            spec.precursor.ligand, ()
        )
    )
    if allowed and f"{cation}1" not in allowed:
        return {}

    by_slot: Dict[int, List[VirtualSite]] = {}
    for slot, host in enumerate(cd_list):
        if need[slot] <= 0:
            continue
        new_cn = int(current_cn[slot]) + 1
        r = pack.bond_length(
            "CdCl_terminal", int(target_cn[slot]), 1, default=2.33
        )
        dirs = terminal_directions_for_new_cn(
            fixed_dirs[slot], new_cn, pack, host_symbol=cation
        )
        local: List[VirtualSite] = []
        for direction in dirs:
            d = _unit(direction)
            if d is None:
                continue
            pos = coords[host] + r * d
            if _clashes(pos, coords, placed, atoms, spec, ignore=(host,)):
                continue
            local.append(
                VirtualSite(
                    hosts=(host,),
                    position=_quantize_pos(pos),
                    kind="terminal",
                    site_id=0,
                )
            )
        # At most two terminal directions per host (CN-aware).
        if local:
            by_slot[slot] = local[:2]
    return by_slot


def iter_pack_site_decorations(
    k: int,
    p: int,
    inorganic_edges: Sequence[Tuple[int, int]],
    spec: NucleationSpec,
    pack: GeometryPack,
    *,
    frame: FloatArray,
    target_degree: Sequence[int],
    max_assignments: int = 0,
    status: Optional[_DecorationStatus] = None,
    node_budget: int = _DEFAULT_NODE_BUDGET,
) -> Iterable[Tuple[Tuple[int, int], ...]]:
    """Stream Cl–Cd edge sets: bridges first, then terminals, on one frame."""

    if status is None:
        status = _DecorationStatus()
    if p == 0:
        yield ()
        return

    se_ids, cd_ids, cl_ids = _index_blocks(k, p)
    cd_list = list(cd_ids)
    cl_list = list(cl_ids)
    n_cd = len(cd_list)
    n_cl = len(cl_list)
    position = {host: index for index, host in enumerate(cd_list)}

    base_degrees = [0] * n_cd
    for left, right in inorganic_edges:
        if left in position:
            base_degrees[position[left]] += 1
        if right in position:
            base_degrees[position[right]] += 1

    target = [int(v) for v in target_degree]
    if len(target) != n_cd:
        return
    surplus0 = [target[i] - base_degrees[i] for i in range(n_cd)]
    if any(s < 0 for s in surplus0) or sum(surplus0) < n_cl:
        return
    if not _surplus_combinatorially_feasible(
        surplus0, base_degrees, target, n_cl, spec
    ):
        status.infeasible += 1
        return

    total_s = sum(surplus0)
    # extra = sum(size-1) over ligands = total_s - n_cl must be realised by bridges.
    extra_needed = total_s - n_cl
    if extra_needed < 0:
        return

    se_set = set(se_ids)
    skeleton_dirs: List[List[FloatArray]] = [[] for _ in range(n_cd)]
    for left, right in inorganic_edges:
        if left in position and right in se_set:
            d = _unit(frame[right] - frame[left])
            if d is not None:
                skeleton_dirs[position[left]].append(d)
        elif right in position and left in se_set:
            d = _unit(frame[left] - frame[right])
            if d is not None:
                skeleton_dirs[position[right]].append(d)

    inorganic = nx.Graph()
    inorganic.add_nodes_from(
        (node, {"element": spec.core.anion}) for node in se_ids
    )
    inorganic.add_nodes_from(
        (node, {"element": spec.core.cation}) for node in cd_list
    )
    inorganic.add_edges_from(inorganic_edges)
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        inorganic,
        inorganic,
        node_match=nx.algorithms.isomorphism.categorical_node_match(
            "element", ""
        ),
    )
    host_maps = [
        {host: mapping[host] for host in cd_list}
        for mapping in matcher.isomorphisms_iter()
    ]
    status.automorphisms = max(status.automorphisms, len(host_maps))

    n_atoms = max(cl_list) + 1 if cl_list else max(cd_list) + 1

    class _A:
        __slots__ = ("symbol",)

        def __init__(self, symbol: str):
            self.symbol = symbol

    atoms_sym = [_A("X") for _ in range(n_atoms)]
    for s in se_ids:
        atoms_sym[s] = _A(spec.core.anion)
    for c in cd_list:
        atoms_sym[c] = _A(spec.core.cation)
    for c in cl_list:
        atoms_sym[c] = _A(spec.precursor.ligand)

    coords = np.zeros((n_atoms, 3), dtype=float)
    n_frame = min(frame.shape[0], n_atoms)
    coords[:n_frame] = np.asarray(frame[:n_frame], dtype=float)
    placed = [False] * n_atoms
    for s in se_ids:
        placed[s] = True
    for c in cd_list:
        placed[c] = True

    bridge_catalog = precompute_bridge_sites(
        cd_list=cd_list,
        frame=coords,
        target_cn=target,
        need0=surplus0,
        pack=pack,
        spec=spec,
        se_ids=list(se_ids),
        inorganic_edges=inorganic_edges,
        allow_mu3=True,
    )
    # Filter catalog clashes with inorganic only (pair-rule floors).
    bridge_catalog = [
        s
        for s in bridge_catalog
        if not _clashes(
            np.asarray(s.position, dtype=float),
            coords,
            placed,
            atoms_sym,
            spec,
            ignore=s.hosts,
        )
    ]
    for i, s in enumerate(bridge_catalog):
        bridge_catalog[i] = VirtualSite(
            hosts=s.hosts, position=s.position, kind=s.kind, site_id=i
        )

    max_shared = int(
        spec.graph_rules.max_shared_ligands_per_host_pair
        or getattr(spec, "bridges_per_cd_pair", 0)
        or 1
    )
    if max_shared <= 0:
        max_shared = 1
    forbid_dual = bool(spec.graph_rules.forbid_mono_se_dual_terminal)
    mono_se = {i for i, sk in enumerate(base_degrees) if sk == 1}

    current = list(base_degrees)
    need = list(surplus0)
    fixed_dirs = [list(d) for d in skeleton_dirs]
    pair_bridges: Dict[Tuple[int, int], int] = {}
    host_bridge = [0] * n_cd
    host_term = [0] * n_cd

    chosen_bridge_ids: List[int] = []
    chosen_edges: List[Tuple[int, int]] = []
    cl_index = 0
    seen_emit: Set[Tuple[object, ...]] = set()
    emitted = 0
    nodes = 0

    def multiset_canonical_hosts(
        host_sets: Sequence[Tuple[int, ...]],
    ) -> Tuple[Tuple[int, ...], ...]:
        best = tuple(sorted(host_sets))
        for host_map in host_maps:
            image = tuple(
                sorted(tuple(sorted(host_map[h] for h in hs)) for hs in host_sets)
            )
            if image < best:
                best = image
        return best

    def emit_decoration(
        host_sets: Sequence[Tuple[int, ...]],
    ) -> Iterable[Tuple[Tuple[int, int], ...]]:
        nonlocal emitted
        if forbid_dual and any(
            host_term[s] == 2 and host_bridge[s] == 0 for s in mono_se
        ):
            status.infeasible += 1
            return
        key = multiset_canonical_hosts(host_sets)
        if key in seen_emit:
            status.symmetry_pruned += 1
            return
        seen_emit.add(key)
        emitted += 1
        if max_assignments > 0 and emitted > max_assignments:
            status.truncated = True
            return
        yield tuple(sorted(chosen_edges))

    def fill_terminals(
        bridge_host_sets: List[Tuple[int, ...]],
    ) -> Iterable[Tuple[Tuple[int, int], ...]]:
        """Place remaining Cl as terminals only; CN-aware dirs after bridges."""

        nonlocal cl_index, nodes
        remaining = n_cl - cl_index
        if remaining != sum(need):
            status.infeasible += 1
            return
        if remaining == 0:
            yield from emit_decoration(bridge_host_sets)
            return

        # Greedy host order: fill slots with need, one preferred dir each time.
        # Branch only when a host has multiple dirs (capped at 1 in practice).
        term_sets: List[Tuple[int, ...]] = []

        def rec_term() -> Iterable[Tuple[Tuple[int, int], ...]]:
            nonlocal cl_index, nodes
            nodes += 1
            if node_budget > 0 and nodes > node_budget:
                return
            if status.truncated:
                return
            if cl_index == n_cl:
                if any(n != 0 for n in need):
                    status.infeasible += 1
                    return
                yield from emit_decoration(bridge_host_sets + term_sets)
                return

            # pick lowest slot with need (canonical)
            try:
                slot = next(s for s in range(n_cd) if need[s] > 0)
            except StopIteration:
                status.infeasible += 1
                return

            by_slot = _terminal_sites_for_state(
                cd_list=cd_list,
                coords=coords,
                placed=placed,
                current_cn=current,
                target_cn=target,
                need=need,
                fixed_dirs=fixed_dirs,
                pack=pack,
                spec=spec,
                atoms=atoms_sym,
            )
            options = by_slot.get(slot, [])
            if not options:
                status.geometry_pruned += 1
                return

            host = cd_list[slot]
            cl = cl_list[cl_index]
            for site in options:
                pos = np.asarray(site.position, dtype=float)
                coords[cl] = pos
                placed[cl] = True
                chosen_edges.append((cl, host))
                term_sets.append((host,))
                need[slot] -= 1
                current[slot] += 1
                host_term[slot] += 1
                d = _unit(pos - coords[host])
                if d is not None:
                    fixed_dirs[slot].append(d)
                cl_index += 1

                yield from rec_term()

                cl_index -= 1
                if fixed_dirs[slot]:
                    fixed_dirs[slot].pop()
                host_term[slot] -= 1
                current[slot] -= 1
                need[slot] += 1
                chosen_edges.pop()
                term_sets.pop()
                placed[cl] = False
                if status.truncated:
                    return

        yield from rec_term()

    def rec_bridges(start: int, extra_left: int) -> Iterable[Tuple[Tuple[int, int], ...]]:
        """Place bridges; ``extra_left`` is remaining (sum size-1) to cover."""

        nonlocal cl_index, nodes
        nodes += 1
        if node_budget > 0 and nodes > node_budget:
            return
        if status.truncated:
            return

        remaining_cl = n_cl - cl_index
        remaining_need = sum(need)
        # Feasibility: remaining_need >= remaining_cl (each Cl ≥1) and
        # remaining_need - remaining_cl == extra_left (exactly).
        if remaining_need < remaining_cl or remaining_need - remaining_cl != extra_left:
            status.infeasible += 1
            return

        # Option: stop bridging and finish with terminals when extra is done.
        if extra_left == 0:
            if remaining_need == remaining_cl:
                yield from fill_terminals(
                    [bridge_catalog[i].hosts for i in chosen_bridge_ids]
                )
            return

        if remaining_cl == 0:
            status.infeasible += 1
            return

        for idx in range(start, len(bridge_catalog)):
            site = bridge_catalog[idx]
            size = len(site.hosts)
            site_extra = size - 1
            if site_extra > extra_left:
                continue
            slots = [position[h] for h in site.hosts]
            if any(need[s] <= 0 for s in slots):
                continue
            if any(current[s] + 1 > target[s] for s in slots):
                continue
            if size == 2:
                pair = site.hosts
                if pair_bridges.get(pair, 0) >= max_shared:
                    continue
            # Clash with already placed Cl.
            pos = np.asarray(site.position, dtype=float)
            if _clashes(pos, coords, placed, atoms_sym, spec, ignore=site.hosts):
                continue

            cl = cl_list[cl_index]
            coords[cl] = pos
            placed[cl] = True
            edges = [(cl, h) for h in site.hosts]
            chosen_edges.extend(edges)
            chosen_bridge_ids.append(idx)
            for s, h in zip(slots, site.hosts):
                need[s] -= 1
                current[s] += 1
                host_bridge[s] += 1
                d = _unit(pos - coords[h])
                if d is not None:
                    fixed_dirs[s].append(d)
            if size == 2:
                pair_bridges[site.hosts] = pair_bridges.get(site.hosts, 0) + 1
            cl_index += 1

            # Next bridge index: i+1 always (each catalog entry used at most once;
            # antipodal pair sites are distinct entries).
            yield from rec_bridges(idx + 1, extra_left - site_extra)

            cl_index -= 1
            if size == 2:
                pair_bridges[site.hosts] -= 1
                if pair_bridges[site.hosts] == 0:
                    del pair_bridges[site.hosts]
            for s in slots:
                if fixed_dirs[s]:
                    fixed_dirs[s].pop()
                host_bridge[s] -= 1
                current[s] -= 1
                need[s] += 1
            for _ in edges:
                chosen_edges.pop()
            chosen_bridge_ids.pop()
            placed[cl] = False
            if status.truncated:
                return

    status.degree_slices += 1
    status.degree_vectors_used += 1
    status.modes_kept = max(status.modes_kept, len(bridge_catalog))
    yield from rec_bridges(0, extra_needed)


def iter_cl_attachments_pack_sites(
    k: int,
    p: int,
    inorganic_edges: Sequence[Tuple[int, int]],
    spec: NucleationSpec,
    pack: GeometryPack,
    *,
    max_assignments: int = 0,
    status: Optional[_DecorationStatus] = None,
    degree_vectors: Optional[Sequence[Tuple[int, ...]]] = None,
    slice_builder: Optional[
        Callable[[Tuple[int, ...]], Optional[_DegreeSlice]]
    ] = None,
    frame_options: int = 0,
    state: Optional[_State] = None,
    cation_ids: Optional[Sequence[int]] = None,
    node_budget: int = _DEFAULT_NODE_BUDGET,
) -> Iterable[Tuple[Tuple[int, int], ...]]:
    """Degree-first pack-site decoration over orbit-min CN vectors."""

    if status is None:
        status = _DecorationStatus()
    se_ids, cd_ids, _ = _index_blocks(k, p)
    cd_list = list(cd_ids)
    n_cd = len(cd_list)
    position = {host: i for i, host in enumerate(cd_list)}
    base_degrees = [0] * n_cd
    for left, right in inorganic_edges:
        if left in position:
            base_degrees[position[left]] += 1
        if right in position:
            base_degrees[position[right]] += 1

    inorganic = nx.Graph()
    inorganic.add_nodes_from(
        (node, {"element": spec.core.anion}) for node in se_ids
    )
    inorganic.add_nodes_from(
        (node, {"element": spec.core.cation}) for node in cd_list
    )
    inorganic.add_edges_from(inorganic_edges)
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        inorganic,
        inorganic,
        node_match=nx.algorithms.isomorphism.categorical_node_match(
            "element", ""
        ),
    )
    host_maps = [
        {host: mapping[host] for host in cd_list}
        for mapping in matcher.isomorphisms_iter()
    ]

    def orbit_min_degree(degree: Sequence[int]) -> Tuple[int, ...]:
        best = tuple(int(v) for v in degree)
        for host_map in host_maps:
            image = [0] * n_cd
            for index in range(n_cd):
                image[position[host_map[cd_list[index]]]] = int(degree[index])
            candidate = tuple(image)
            if candidate < best:
                best = candidate
        return best

    if degree_vectors is None:
        vectors = _cation_degree_vectors(
            base_degrees, 2 * p, spec, limit=20000
        )
        if not vectors:
            return
        degree_vectors = [tuple(int(v) for v in d) for d in vectors]

    status.degree_vectors_total += len(degree_vectors)
    feasible = {
        tuple(int(v) for v in d) for d in degree_vectors if len(d) == n_cd
    }
    orbit_reps = sorted(
        {orbit_min_degree(d) for d in feasible}.intersection(feasible)
    )

    # Prefer lower total surplus first (fewer bridges → smaller trees).
    def surplus_sum(degree: Tuple[int, ...]) -> int:
        return sum(int(degree[i]) - base_degrees[i] for i in range(n_cd))

    orbit_reps.sort(key=lambda d: (surplus_sum(d), d))

    cations = list(cation_ids) if cation_ids is not None else cd_list
    # Slot-order diversity matters for which Cd–Cd pairs are sphere-feasible.
    # Default (frame_options=0) keeps a small budget, not every order.
    frame_limit = 8 if frame_options <= 0 else max(1, frame_options)

    for degree in orbit_reps:
        surplus = [int(degree[i]) - base_degrees[i] for i in range(n_cd)]
        if any(s < 0 for s in surplus) or sum(surplus) < 2 * p:
            continue
        if not _surplus_combinatorially_feasible(
            surplus, base_degrees, degree, 2 * p, spec
        ):
            status.infeasible += 1
            continue

        if state is None:
            continue

        degrees_full = [state.graph.degree[i] for i in range(len(state.atoms))]
        for cat, d in zip(cations, degree):
            degrees_full[cat] = int(d)
        built_frames, _ = _clean_frames(
            state, pack, spec, degrees_full, limit=frame_limit
        )
        if not built_frames:
            status.geometry_pruned += 1
            continue

        # slice_builder only for side-effect frame cache if provided
        if slice_builder is not None:
            slice_builder(degree)

        for fr, _pl in built_frames:
            yield from iter_pack_site_decorations(
                k,
                p,
                inorganic_edges,
                spec,
                pack,
                frame=np.asarray(fr, dtype=float),
                target_degree=degree,
                max_assignments=max_assignments,
                status=status,
                node_budget=node_budget,
            )
            if status.truncated:
                return
