"""Tetrahedral slot scaffold for molecular Cl decoration (Layer A).

After the Cd–Se frame is fixed:

1. Each Cd gets vacant **tetrahedral** directions relative to already-placed
   Se neighbours (ideal tet template; not a crystal CIF).
2. Cd–Cd pairs that can host a μ₂ get an explicit **bridge slot** at the
   sphere-intersection preferred opposite a shared Se (when any).
3. Nearby points merge into multi-host sites (``site_tolerance``).
4. Decoration = discrete occupations of those sites (topology only).

Final coordinates (Layer B) are produced by the existing pack embedder
(``embed_molecular_state``): linear / trigonal / bridge tables may **move** Cl
off the tet points.  The tet scaffold only decides connectivity.
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
    _unit,
)
from .types import _State

FloatArray = np.ndarray

# Ideal tetrahedron (vertex directions from centre).
_TET_DIRS: Tuple[FloatArray, ...] = tuple(
    v / np.linalg.norm(v)
    for v in (
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, -1.0, -1.0]),
        np.array([-1.0, 1.0, -1.0]),
        np.array([-1.0, -1.0, 1.0]),
    )
)

_DEFAULT_NODE_BUDGET = 100_000


@dataclass(frozen=True)
class TetSite:
    """One discrete Cl slot (terminal or multi-host after merge)."""

    hosts: Tuple[int, ...]  # sorted Cd ids
    position: Tuple[float, float, float]
    site_id: int

    @property
    def size(self) -> int:
        return len(self.hosts)


def _ideal_tet_rotation(occupied: Sequence[FloatArray]) -> List[FloatArray]:
    """Return 4 unit tet directions, best-aligned to ``occupied`` directions.

    Greedy assignment of occupied vectors to tet slots, then apply a rotation
    that maps the first matched ideal axis onto the first occupied direction
    and aligns a second axis in the common plane when possible.
    """

    occ = [d for d in (_unit(np.asarray(v, dtype=float)) for v in occupied) if d is not None]
    if not occ:
        return [np.array(v, dtype=float) for v in _TET_DIRS]

    # Build an orthonormal frame from the first occupied direction.
    z = occ[0]
    if len(occ) >= 2:
        x = _unit(occ[1] - float(np.dot(occ[1], z)) * z)
        if x is None:
            _, x, _ = _orthonormal_basis(z)
    else:
        _, x, _ = _orthonormal_basis(z)
    y = _unit(_cross3(z, x))
    assert y is not None and x is not None

    # Ideal tet: bond angle arccos(-1/3) ≈ 109.47°.  With Se along +z, the
    # three free ligand dirs have free·z = -1/3 (not +1/3).
    cos_tet = -1.0 / 3.0
    sin_tet = sqrt(1.0 - cos_tet * cos_tet)
    free: List[FloatArray] = []
    for k in range(3):
        phi = 2.0 * np.pi * k / 3.0
        free.append(
            cos_tet * z + sin_tet * (np.cos(phi) * x + np.sin(phi) * y)
        )
    if len(occ) == 1:
        return free

    # 2+ occupied: match tet template by assigning and return unmatched ideals
    # rotated so first ideal maps to first occupied.
    # Use lattice-style score on fixed _TET_DIRS after rotating first slot → occ[0].
    # Build rotation taking e0=(1,1,1)/√3 to occ[0].
    e0 = _TET_DIRS[0]
    # Rodrigues-ish: use frames
    ez = e0
    _, ex, _ = _orthonormal_basis(ez)
    ey = _unit(_cross3(ez, ex))
    assert ey is not None and ex is not None
    # Map ez→z, ex→x, ey→y
    R = np.column_stack((x, y, z)) @ np.column_stack((ex, ey, ez)).T
    rotated = [R @ d for d in _TET_DIRS]
    # Which ideals are used by occupied?
    used: Set[int] = set()
    for o in occ:
        best_i, best_dot = -1, -2.0
        for i, d in enumerate(rotated):
            if i in used:
                continue
            dot = float(np.dot(o, d))
            if dot > best_dot:
                best_dot, best_i = dot, i
        if best_i >= 0:
            used.add(best_i)
    return [rotated[i] for i in range(4) if i not in used]


def _tet_free_directions(occupied: Sequence[FloatArray]) -> List[FloatArray]:
    """Vacant unit directions for a centre with the given occupied bond dirs."""

    occ = [d for d in (_unit(np.asarray(v, dtype=float)) for v in occupied) if d is not None]
    n_occ = len(occ)
    if n_occ >= 4:
        return []
    if n_occ == 0:
        return [np.array(v, dtype=float) for v in _TET_DIRS]
    if n_occ == 1:
        return _ideal_tet_rotation(occ)
    # 2 or 3 occupied: return unmatched tet slots after alignment
    free = _ideal_tet_rotation(occ)
    # For n_occ==1, _ideal_tet_rotation already returns 3 free dirs.
    # For n_occ>=2 it returns unmatched; keep them.
    return free


def build_tet_sites(
    *,
    cd_list: Sequence[int],
    se_ids: Sequence[int],
    frame: FloatArray,
    inorganic_edges: Sequence[Tuple[int, int]],
    target_cn: Sequence[int],
    pack: GeometryPack,
    spec: NucleationSpec,
) -> List[TetSite]:
    """Build merged tet + bridge slots for one frame and target CN vector."""

    n_cd = len(cd_list)
    se_set = set(se_ids)
    position = {h: i for i, h in enumerate(cd_list)}
    min_bridge = int(spec.graph_rules.min_bridged_host_cn)
    tol = float(getattr(spec, "site_tolerance", 0.20) or 0.20)
    r_term_default = 2.33

    # Se neighbours per Cd
    se_of: Dict[int, List[int]] = {h: [] for h in cd_list}
    for left, right in inorganic_edges:
        if left in position and right in se_set:
            se_of[left].append(right)
        elif right in position and left in se_set:
            se_of[right].append(left)

    # Collect raw (host_set, position) before merge
    raw: List[Tuple[Tuple[int, ...], FloatArray]] = []

    # --- Terminal tet slots per Cd ---
    for slot, host in enumerate(cd_list):
        need = int(target_cn[slot]) - len(se_of[host])
        if need <= 0:
            continue
        occupied = []
        for se in se_of[host]:
            d = _unit(frame[se] - frame[host])
            if d is not None:
                occupied.append(d)
        free = _tet_free_directions(occupied)
        r = pack.bond_length(
            "CdCl_terminal", int(target_cn[slot]), 1, default=r_term_default
        )
        # At most `need` free dirs matter for capacity; keep all free dirs as
        # alternate slots (≤3 for mono-Se).
        for direction in free:
            d = _unit(direction)
            if d is None:
                continue
            pos = frame[host] + r * d
            raw.append(((host,), pos))

    # --- Explicit μ₂ bridge slots (sphere intersection) ---
    for s1, s2 in combinations(range(n_cd), 2):
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
        ab_u, u_perp, _ = _orthonormal_basis(frame[h2] - frame[h1])
        base = frame[h1] + axial * ab_u
        hosts = (min(h1, h2), max(h1, h2))
        shared = sorted(set(se_of[h1]).intersection(se_of[h2]))
        if height < 1.0e-8:
            raw.append((hosts, base))
            continue
        if shared:
            se = shared[0]
            toward = frame[se] - base
            toward = toward - float(np.dot(toward, ab_u)) * ab_u
            ref = _unit(-toward)
            if ref is None:
                ref = u_perp
            raw.append((hosts, base + height * ref))
        else:
            raw.append((hosts, base + height * u_perp))
            raw.append((hosts, base - height * u_perp))

    # --- Merge by spatial proximity ---
    # Each entry: [host_set, position_array, count]
    merged: List[list] = []
    for hosts, pos in raw:
        pos = np.asarray(pos, dtype=float)
        found = None
        for item in merged:
            if float(np.linalg.norm(item[1] - pos)) < tol:
                found = item
                break
        if found is None:
            merged.append([set(hosts), pos.copy(), 1])
        else:
            found[0].update(hosts)
            n = found[2] + 1
            found[1][:] = (found[1] * found[2] + pos) / n
            found[2] = n

    max_hosts = int(spec.graph_rules.max_cn[spec.precursor.ligand])
    allowed = set(
        spec.graph_rules.allowed_neighbor_signatures.get(
            spec.precursor.ligand, ()
        )
    )
    cation = spec.core.cation

    sites: List[TetSite] = []
    for hosts_set, pos, _n in merged:
        hosts = tuple(sorted(hosts_set))
        if len(hosts) > max_hosts:
            continue
        if allowed and f"{cation}{len(hosts)}" not in allowed:
            continue
        # multi-host sites need all hosts bridge-capable
        if len(hosts) >= 2:
            if any(
                target_cn[position[h]] < min_bridge for h in hosts
            ):
                continue
        sites.append(
            TetSite(
                hosts=hosts,
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                site_id=len(sites),
            )
        )

    # Prefer multi-host first (bridges), then terminals; stable order.
    sites.sort(key=lambda s: (-s.size, s.hosts, s.position))
    return [
        TetSite(hosts=s.hosts, position=s.position, site_id=i)
        for i, s in enumerate(sites)
    ]


def iter_tet_site_decorations(
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
    """Stream Cl–Cd edge sets by occupying tet/bridge slots on one frame."""

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
    position = {h: i for i, h in enumerate(cd_list)}

    base = [0] * n_cd
    for left, right in inorganic_edges:
        if left in position:
            base[position[left]] += 1
        if right in position:
            base[position[right]] += 1

    target = [int(v) for v in target_degree]
    if len(target) != n_cd:
        return
    need0 = [target[i] - base[i] for i in range(n_cd)]
    if any(s < 0 for s in need0) or sum(need0) < n_cl:
        return
    if not _surplus_combinatorially_feasible(
        need0, base, target, n_cl, spec
    ):
        status.infeasible += 1
        return

    total_s = sum(need0)
    extra_needed = total_s - n_cl  # sum (size-1) over chosen sites
    if extra_needed < 0:
        return

    sites = build_tet_sites(
        cd_list=cd_list,
        se_ids=list(se_ids),
        frame=np.asarray(frame, dtype=float),
        inorganic_edges=inorganic_edges,
        target_cn=target,
        pack=pack,
        spec=spec,
    )
    if not sites:
        status.geometry_pruned += 1
        return

    # Automorphisms for emit prune
    inorganic = nx.Graph()
    inorganic.add_nodes_from(
        (n, {"element": spec.core.anion}) for n in se_ids
    )
    inorganic.add_nodes_from(
        (n, {"element": spec.core.cation}) for n in cd_list
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
        {h: m[h] for h in cd_list} for m in matcher.isomorphisms_iter()
    ]
    status.automorphisms = max(status.automorphisms, len(host_maps))
    status.modes_kept = max(status.modes_kept, len(sites))

    max_shared = int(
        spec.graph_rules.max_shared_ligands_per_host_pair
        or getattr(spec, "bridges_per_cd_pair", 0)
        or 1
    )
    if max_shared <= 0:
        max_shared = 1
    forbid_dual = bool(spec.graph_rules.forbid_mono_se_dual_terminal)
    mono_se = {i for i, b in enumerate(base) if b == 1}

    need = list(need0)
    host_bridge = [0] * n_cd
    host_term = [0] * n_cd
    pair_bridges: Dict[Tuple[int, int], int] = {}
    chosen_ids: List[int] = []
    chosen_edges: List[Tuple[int, int]] = []
    cl_index = 0
    seen_emit: Set[Tuple[Tuple[int, ...], ...]] = set()
    emitted = 0
    nodes = 0

    def multiset_canonical(
        host_sets: Sequence[Tuple[int, ...]],
    ) -> Tuple[Tuple[int, ...], ...]:
        best = tuple(sorted(host_sets))
        for host_map in host_maps:
            image = tuple(
                sorted(
                    tuple(sorted(host_map[h] for h in hs)) for hs in host_sets
                )
            )
            if image < best:
                best = image
        return best

    def emit() -> Iterable[Tuple[Tuple[int, int], ...]]:
        nonlocal emitted
        if any(n != 0 for n in need):
            status.infeasible += 1
            return
        if forbid_dual and any(
            host_term[s] == 2 and host_bridge[s] == 0 for s in mono_se
        ):
            status.infeasible += 1
            return
        host_sets = [sites[i].hosts for i in chosen_ids]
        key = multiset_canonical(host_sets)
        if key in seen_emit:
            status.symmetry_pruned += 1
            return
        seen_emit.add(key)
        emitted += 1
        if max_assignments > 0 and emitted > max_assignments:
            status.truncated = True
            return
        yield tuple(sorted(chosen_edges))

    def rec(start: int, extra_left: int) -> Iterable[Tuple[Tuple[int, int], ...]]:
        nonlocal cl_index, nodes
        nodes += 1
        # Soft stop: do not set status.truncated (that flag is for max_assignments
        # and aborts the whole bin).  Just end this frame's search.
        if node_budget > 0 and nodes > node_budget:
            return
        if status.truncated:
            return

        remaining_cl = n_cl - cl_index
        remaining_need = sum(need)
        if remaining_need < remaining_cl:
            status.infeasible += 1
            return
        if remaining_need - remaining_cl != extra_left:
            status.infeasible += 1
            return

        if remaining_cl == 0:
            yield from emit()
            return

        # Must place remaining_cl more sites; if extra_left==0 all terminals.
        for idx in range(start, len(sites)):
            site = sites[idx]
            size = site.size
            site_extra = size - 1
            if site_extra > extra_left:
                continue
            slots = [position[h] for h in site.hosts]
            if any(need[s] <= 0 for s in slots):
                continue
            if any(base[s] + (target[s] - base[s] - need[s]) + 1 > target[s] for s in slots):
                # equivalent: current_cn would exceed target — track via need
                pass
            if size == 2:
                pair = site.hosts
                if pair_bridges.get(pair, 0) >= max_shared:
                    continue
            # After taking this site, each host need decreases by 1.
            # Capacity: need[s] >= 1 already checked.

            cl = cl_list[cl_index]
            edges = [(cl, h) for h in site.hosts]
            chosen_edges.extend(edges)
            chosen_ids.append(idx)
            for s in slots:
                need[s] -= 1
                if size >= 2:
                    host_bridge[s] += 1
                else:
                    host_term[s] += 1
            if size == 2:
                pair_bridges[site.hosts] = pair_bridges.get(site.hosts, 0) + 1
            cl_index += 1

            yield from rec(idx + 1, extra_left - site_extra)

            cl_index -= 1
            if size == 2:
                pair_bridges[site.hosts] -= 1
                if pair_bridges[site.hosts] == 0:
                    del pair_bridges[site.hosts]
            for s in slots:
                if size >= 2:
                    host_bridge[s] -= 1
                else:
                    host_term[s] -= 1
                need[s] += 1
            for _ in edges:
                chosen_edges.pop()
            chosen_ids.pop()
            if status.truncated:
                return

    status.degree_slices += 1
    status.degree_vectors_used += 1
    yield from rec(0, extra_needed)


def iter_cl_attachments_tet_sites(
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
    """Degree-first tet-slot decoration over orbit-min CN vectors."""

    if status is None:
        status = _DecorationStatus()
    se_ids, cd_ids, _ = _index_blocks(k, p)
    cd_list = list(cd_ids)
    n_cd = len(cd_list)
    position = {h: i for i, h in enumerate(cd_list)}
    base = [0] * n_cd
    for left, right in inorganic_edges:
        if left in position:
            base[position[left]] += 1
        if right in position:
            base[position[right]] += 1

    inorganic = nx.Graph()
    inorganic.add_nodes_from(
        (n, {"element": spec.core.anion}) for n in se_ids
    )
    inorganic.add_nodes_from(
        (n, {"element": spec.core.cation}) for n in cd_list
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
        {h: m[h] for h in cd_list} for m in matcher.isomorphisms_iter()
    ]

    def orbit_min_degree(degree: Sequence[int]) -> Tuple[int, ...]:
        best = tuple(int(v) for v in degree)
        for host_map in host_maps:
            image = [0] * n_cd
            for i in range(n_cd):
                image[position[host_map[cd_list[i]]]] = int(degree[i])
            cand = tuple(image)
            if cand < best:
                best = cand
        return best

    if degree_vectors is None:
        vectors = _cation_degree_vectors(base, 2 * p, spec, limit=20000)
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
    orbit_reps.sort(
        key=lambda d: (
            sum(int(d[i]) - base[i] for i in range(n_cd)),
            d,
        )
    )

    cations = list(cation_ids) if cation_ids is not None else cd_list
    frame_limit = 8 if frame_options <= 0 else max(1, frame_options)

    if state is None:
        return

    for degree in orbit_reps:
        surplus = [int(degree[i]) - base[i] for i in range(n_cd)]
        if any(s < 0 for s in surplus) or sum(surplus) < 2 * p:
            continue
        if not _surplus_combinatorially_feasible(
            surplus, base, degree, 2 * p, spec
        ):
            status.infeasible += 1
            continue

        degrees_full = [state.graph.degree[i] for i in range(len(state.atoms))]
        for cat, d in zip(cations, degree):
            degrees_full[cat] = int(d)
        built, _ = _clean_frames(
            state, pack, spec, degrees_full, limit=frame_limit
        )
        if not built:
            status.geometry_pruned += 1
            continue
        if slice_builder is not None:
            slice_builder(degree)

        for fr, _pl in built:
            yield from iter_tet_site_decorations(
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
