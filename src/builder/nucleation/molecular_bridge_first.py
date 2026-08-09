"""Sequential chemically ordered Cl decoration on a 3D skeleton.

Decoration mode ``skeleton_bridge_first``:

Passivation order (chemical, greedy tiers + beam)
-------------------------------------------------
1. Terminal on **stranded** CN=1 (no legal bridge partner in the distance window).
2. While any Cd has CN < 3: place either
   - a μ2 bridge on a distance-legal pair (prefer load ≤ 2 / Cd; hard max 2),
     only if both hosts can still finish ≥ min_bridged_host_cn, and
     max_shared bridges on that pair is respected; or
   - a terminal on the lowest-CN hosts (CN1, then CN2).
   Both kinds enter the beam so we do **not** exhaust every bridge before
   terminals (DFT/multiset: k1p2 is typically 1 μ2 + 3 terminals, not 2 μ2).
3. Terminals CN3 → 4 once all Cd ≥ 3.
4. Once all Cd have CN ≥ 3, continue with any remaining legal bridges before
   terminal fill, subject to the same hard two-bridges-per-Cd cap.
5. Drain remaining Cl as terminals.

For ``bridge_first_p1_terminal_policy: unrestricted`` the p=1 path keeps
bridge candidates first but enumerates terminal hosts without lowest-CN
priority.  The p>=2 tiers below remain unchanged.

The bridge-first mode makes the two-bridge cap a hard chemical rule. Distances
only gate which pairs may bridge; search is discrete beam, not a free CN-vector
product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
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
    _DecorationStatus,
    _DegreeSlice,
    _cation_degree_vectors,
    _index_blocks,
    iter_cl_attachments,
)
from .types import _State

FloatArray = np.ndarray

DEFAULT_CD_CD_BRIDGE_MIN = 3.05
DEFAULT_CD_CD_BRIDGE_MAX = 4.75
DEFAULT_PREFER_BRIDGE_PER_CD = 2
DEFAULT_HARD_MAX_BRIDGE_PER_CD = 2
DEFAULT_BEAM = 32
DEFAULT_PASSIVATE_P_GE = 100  # disabled hard floor
DEFAULT_PASSIVATE_K_GE = 2


def _skel_degrees(
    cd_list: Sequence[int],
    inorganic_edges: Sequence[Tuple[int, int]],
) -> List[int]:
    pos = {h: i for i, h in enumerate(cd_list)}
    deg = [0] * len(cd_list)
    for a, b in inorganic_edges:
        if a in pos:
            deg[pos[a]] += 1
        if b in pos:
            deg[pos[b]] += 1
    return deg


def _sphere_intersect_mu2(
    frame: FloatArray, h1: int, h2: int, r1: float, r2: float
) -> bool:
    sep = float(np.linalg.norm(frame[h2] - frame[h1]))
    if sep < 1.0e-12:
        return False
    if sep > r1 + r2 + EXACT_BOND_TOLERANCE:
        return False
    if sep + EXACT_BOND_TOLERANCE < abs(r1 - r2):
        return False
    axial = (r1 * r1 - r2 * r2 + sep * sep) / (2.0 * sep)
    return r1 * r1 - axial * axial >= -EXACT_BOND_TOLERANCE


def bridge_pair_candidates(
    cd_list: Sequence[int],
    frame: FloatArray,
    pack: GeometryPack,
    *,
    d_min: float,
    d_max: float,
    cn_for_radius: int = 3,
) -> List[Tuple[int, int]]:
    """Cd pairs in distance window with sphere intersection at given CN radii."""

    pairs: List[Tuple[int, int]] = []
    r = pack.bond_length("CdCl_bridge", cn_for_radius, 2, default=2.40)
    for i, j in combinations(range(len(cd_list)), 2):
        h1, h2 = cd_list[i], cd_list[j]
        d = float(np.linalg.norm(frame[h1] - frame[h2]))
        if d < d_min or d > d_max:
            continue
        if not _sphere_intersect_mu2(frame, h1, h2, r, r):
            continue
        pairs.append((min(h1, h2), max(h1, h2)))
    pairs.sort(
        key=lambda pr: float(np.linalg.norm(frame[pr[0]] - frame[pr[1]]))
    )
    return pairs


def _min_cd_cn_policy(k: int, p: int, spec: NucleationSpec) -> int:
    p_ge = int(
        getattr(
            spec.graph_rules, "passivate_min_cd_cn_p_ge", DEFAULT_PASSIVATE_P_GE
        )
        or DEFAULT_PASSIVATE_P_GE
    )
    k_ge = int(
        getattr(
            spec.graph_rules, "passivate_min_cd_cn_k_ge", DEFAULT_PASSIVATE_K_GE
        )
        or DEFAULT_PASSIVATE_K_GE
    )
    high_min = int(getattr(spec.graph_rules, "passivate_min_cd_cn", 3) or 3)
    base = int(spec.graph_rules.min_cn.get(spec.core.cation, 2) or 2)
    if not spec.enforce_min_cn:
        base = 0
    if k >= k_ge and p >= p_ge:
        return max(base, high_min)
    return max(base, 2) if spec.enforce_min_cn else base


@dataclass
class _PlaceState:
    """Partial decoration state for beam search."""

    cn: List[int]
    n_bridge: List[int]
    n_term: List[int]
    remaining_cl: int
    # edges as (cl_index_in_order, host_id) — cl assigned at emit
    bridges: List[Tuple[int, int]]  # host pairs
    mu3: List[Tuple[int, int, int]]
    terminals_on: List[int]  # host slot index for each terminal in order
    log: List[str] = field(default_factory=list)

    def clone(self) -> "_PlaceState":
        return _PlaceState(
            cn=list(self.cn),
            n_bridge=list(self.n_bridge),
            n_term=list(self.n_term),
            remaining_cl=self.remaining_cl,
            bridges=list(self.bridges),
            mu3=list(self.mu3),
            terminals_on=list(self.terminals_on),
            log=list(self.log),
        )

    def key(self) -> Tuple:
        return (
            tuple(self.cn),
            tuple(self.n_bridge),
            tuple(self.n_term),
            tuple(sorted(self.bridges)),
            tuple(sorted(self.mu3)),
            tuple(sorted(self.terminals_on)),
        )


def _can_finish_ge(
    cn_after: int,
    room_left: int,
    min_final: int,
    remaining_cl_after: int,
    need_other: int = 0,
) -> bool:
    """Whether host can still reach min_final with terminals from remaining Cl."""

    need = max(0, min_final - cn_after)
    if need > room_left:
        return False
    return need + need_other <= remaining_cl_after


def sequential_passivate(
    *,
    k: int,
    p: int,
    cd_list: Sequence[int],
    skel: Sequence[int],
    pairs: Sequence[Tuple[int, int]],
    triples: Sequence[Tuple[int, int, int]] = (),
    required_profiles: Sequence[Tuple[int, ...]] = (),
    position: Dict[int, int],
    spec: NucleationSpec,
    n_cl: int,
    beam_width: int = DEFAULT_BEAM,
    prefer_bridge_per_cd: int = DEFAULT_PREFER_BRIDGE_PER_CD,
    hard_max_bridge_per_cd: int = DEFAULT_HARD_MAX_BRIDGE_PER_CD,
    status: Optional[_DecorationStatus] = None,
    host_maps: Sequence[Dict[int, int]] = (),
    strict_bridge_first: bool = False,
    se_shared_pairs: Sequence[Tuple[int, int]] = (),
) -> List[_PlaceState]:
    """Run priority passivation; return finished beam states (complete Cl use)."""

    if status is None:
        status = _DecorationStatus()
    n_cd = len(cd_list)
    max_cd = int(spec.graph_rules.max_cn[spec.core.cation])
    min_bridge = int(spec.graph_rules.min_bridged_host_cn)
    forbid_dual = bool(spec.graph_rules.forbid_mono_se_dual_terminal)
    min_cd_final = _min_cd_cn_policy(k, p, spec)
    p1_unrestricted = (
        p == 1
        and str(
            getattr(
                spec.graph_rules,
                "bridge_first_p1_terminal_policy",
                "unrestricted",
            )
        ).strip().lower()
        == "unrestricted"
    )
    mono_se = {i for i, s in enumerate(skel) if s == 1}
    max_shared = int(
        getattr(spec.graph_rules, "max_shared_ligands_per_host_pair", 1) or 1
    )
    forbid_mu3_overlap = bool(
        getattr(spec.graph_rules, "forbid_mu3_host_bridge_overlap", False)
    )
    # Compactness steering, OFF by default.  Maximising newly-bridged Cd pairs
    # correlates with compactness post hoc (rho +0.58) but is a poor objective:
    # a mu3 whose hosts already share a Se contributes zero, so switching this
    # on strips the mu3 family that holds the k3p3 energy minimum (66 -> 25
    # graphs).  Left available for experiments; do not enable without checking
    # energies on the reduced set.
    maximize_bridged_pairs = bool(
        getattr(
            spec.graph_rules, "bridge_first_maximize_bridged_pairs", False
        )
    )
    se_shared_pairs = frozenset(
        (min(int(a), int(b)), max(int(a), int(b))) for a, b in se_shared_pairs
    )
    if max_shared <= 0:
        max_shared = 1
    # The soft load preference is what really bounds bridge load: ``over_pref``
    # is ranked above the bridge-count term, so a Cd taking one more bridge
    # than this loses the beam long before ``hard_max_bridge_per_cd`` applies.
    # Total-bridge target.  prefer_bridge_per_cd caps the load on ONE Cd; this
    # steers the whole decoration toward the 2p total the stable structures
    # actually have.  Measured: n_bridges(best) = 0.96*(2p) - 0.21, r=0.994.
    target_fraction = float(
        getattr(spec.graph_rules, "bridge_first_target_bridge_fraction", 0.0)
        or 0.0
    )
    bridge_target = int(round(target_fraction * n_cl)) if target_fraction > 0 else 0

    prefer_bridge_per_cd = int(
        getattr(
            spec.graph_rules,
            "bridge_first_prefer_bridges_per_cd",
            prefer_bridge_per_cd,
        )
        or prefer_bridge_per_cd
    )

    pair_list = [(min(a, b), max(a, b)) for a, b in pairs]
    pair_set = set(pair_list)

    # The skeleton automorphisms preserve Cd/Se labels.  Store their action in
    # local Cd-slot coordinates so equivalent partial ligand histories share a
    # single beam state without changing the chemical alphabet.
    slot_maps: List[Tuple[int, ...]] = []
    for host_map in host_maps:
        try:
            slot_maps.append(
                tuple(position[host_map[host]] for host in cd_list)
            )
        except KeyError:
            continue
    if not slot_maps:
        slot_maps = [tuple(range(n_cd))]

    # Inverses as tuples (list indexing, not dict hashing) and the composed
    # host relabelling cd_list[mapping[position[host]]] precomputed per
    # automorphism, so the hot loop does one dict lookup instead of three.
    slot_inverses: List[Tuple[int, ...]] = []
    host_relabels: List[Dict[int, int]] = []
    for mapping in slot_maps:
        inverse = [0] * n_cd
        for original, mapped in enumerate(mapping):
            inverse[mapped] = original
        slot_inverses.append(tuple(inverse))
        host_relabels.append(
            {host: cd_list[mapping[position[host]]] for host in cd_list}
        )

    # An asymmetric skeleton has only the identity automorphism, so there is
    # nothing to canonicalize -- the single candidate *is* the key.  This is
    # the common case once k grows, and it skips the whole relabelling.
    identity_only = len(slot_maps) == 1 and slot_maps[0] == tuple(range(n_cd))

    # Canonicalizing a state costs O(|Aut| . n log n), and |Aut| grows
    # factorially in the number of symmetry-equivalent precursor cations: it
    # reaches a mean of 8388 (max 31104) at k=4 p=9, where a single bin spent
    # 1541 s in decoration with 92% of that inside state_key.  Above the cap we
    # key by identity instead.  That admits symmetry-duplicate *beam states* --
    # it never admits a duplicate graph, because emitted graphs are
    # deduplicated by isomorphism certificate downstream regardless.  The cost
    # is beam slots spent on equivalent states, so the cap trades a little
    # diversity for a large constant factor exactly where the group explodes.
    aut_cap = int(
        getattr(spec.graph_rules, "bridge_first_max_automorphisms", 0) or 0
    )
    if aut_cap > 0 and len(slot_maps) > aut_cap:
        if status is not None:
            status.automorphism_cap_hits = (
                getattr(status, "automorphism_cap_hits", 0) + 1
            )
        identity_only = True

    def _identity_key(st: _PlaceState) -> Tuple:
        """Key for an asymmetric skeleton: no relabelling to apply.

        ``st.bridges`` entries are already ``(min, max)`` (they come from
        ``pair_list``) and ``st.mu3`` triples are stored pre-sorted by
        ``apply_mu3``, so only the outer ordering is needed here.
        """

        return (
            tuple(st.cn),
            tuple(st.n_bridge),
            tuple(st.n_term),
            tuple(sorted(st.bridges)),
            tuple(sorted(st.mu3)),
            tuple(sorted(st.terminals_on)),
            st.remaining_cl,
        )

    def state_key(st: _PlaceState) -> Tuple:
        """Canonicalize a partial state under skeleton host automorphisms."""

        if identity_only:
            return _identity_key(st)
        best: Optional[Tuple] = None
        for mapping, inverse, relabel in zip(
            slot_maps, slot_inverses, host_relabels
        ):
            bridge_list = []
            for a, b in st.bridges:
                ra = relabel[a]
                rb = relabel[b]
                bridge_list.append((ra, rb) if ra <= rb else (rb, ra))
            bridge_list.sort()
            bridges = tuple(bridge_list)
            mu3_list = []
            for x, y, z in st.mu3:
                rx = relabel[x]
                ry = relabel[y]
                rz = relabel[z]
                # three-element sorting network: cheaper than sorted() here
                if rx > ry:
                    rx, ry = ry, rx
                if ry > rz:
                    ry, rz = rz, ry
                if rx > ry:
                    rx, ry = ry, rx
                mu3_list.append((rx, ry, rz))
            mu3_list.sort()
            mu3 = tuple(mu3_list)
            terminals = tuple(sorted(mapping[slot] for slot in st.terminals_on))
            candidate = (
                tuple(map(st.cn.__getitem__, inverse)),
                tuple(map(st.n_bridge.__getitem__, inverse)),
                tuple(map(st.n_term.__getitem__, inverse)),
                bridges,
                mu3,
                terminals,
                st.remaining_cl,
            )
            if best is None or candidate < best:
                best = candidate
        return best

    def free_cap(st: _PlaceState, i: int) -> int:
        return max_cd - st.cn[i]

    def pair_bridge_count(st: _PlaceState, pr: Tuple[int, int]) -> int:
        pr = (min(pr[0], pr[1]), max(pr[0], pr[1]))
        return sum(1 for b in st.bridges if b == pr) + sum(
            1 for tri in st.mu3 if pr[0] in tri and pr[1] in tri
        )

    def apply_mu3(st: _PlaceState, tri: Tuple[int, int, int], tag: str) -> Optional[_PlaceState]:
        slots = [position[h] for h in tri]
        if st.remaining_cl < 1 or any(free_cap(st, i) < 1 for i in slots):
            return None
        # A μ3 cap owns all three Cd hosts for chloride bridging.  Do not
        # place it on a Cd that already carries another bridge (and do not
        # let a later bridge reuse one of these hosts).
        if forbid_mu3_overlap and any(st.n_bridge[i] > 0 for i in slots):
            return None
        if any(st.n_bridge[i] >= hard_max_bridge_per_cd for i in slots):
            return None
        if any(pair_bridge_count(st, pair) >= max_shared for pair in combinations(tri, 2)):
            return None
        nxt = st.clone()
        for i in slots:
            nxt.cn[i] += 1
            nxt.n_bridge[i] += 1
        nxt.remaining_cl -= 1
        nxt.mu3.append(tuple(sorted(tri)))
        nxt.log.append(f"mu3 {tri} [{tag}]")
        return nxt

    def legal_bridge_pair(
        st: _PlaceState,
        pr: Tuple[int, int],
        *,
        allow_over_prefer: bool = False,
    ) -> bool:
        pr = (min(pr[0], pr[1]), max(pr[0], pr[1]))
        if pr not in pair_set:
            return False
        s1, s2 = position[pr[0]], position[pr[1]]
        if forbid_mu3_overlap and any(
            cd_list[s1] in tri or cd_list[s2] in tri for tri in st.mu3
        ):
            return False
        if st.n_bridge[s1] >= hard_max_bridge_per_cd:
            return False
        if st.n_bridge[s2] >= hard_max_bridge_per_cd:
            return False
        # Soft prefer: block 3rd+ bridge/Cd unless late allow_over_prefer
        if not allow_over_prefer:
            if st.n_bridge[s1] >= prefer_bridge_per_cd:
                return False
            if st.n_bridge[s2] >= prefer_bridge_per_cd:
                return False
        if pair_bridge_count(st, pr) >= max_shared:
            return False
        if free_cap(st, s1) < 1 or free_cap(st, s2) < 1:
            return False
        if st.remaining_cl < 1:
            return False
        # both must be able to finish ≥ min_bridge after this bridge
        rem_after = st.remaining_cl - 1
        cn1, cn2 = st.cn[s1] + 1, st.cn[s2] + 1
        room1, room2 = max_cd - cn1, max_cd - cn2
        need1 = max(0, min_bridge - cn1)
        need2 = max(0, min_bridge - cn2)
        if need1 > room1 or need2 > room2:
            return False
        if need1 + need2 > rem_after:
            return False
        return True

    def can_bridge_host(st: _PlaceState, i: int) -> bool:
        if free_cap(st, i) < 1 or st.n_bridge[i] >= hard_max_bridge_per_cd:
            return False
        hi = cd_list[i]
        for pr in pair_list:
            if hi not in pr:
                continue
            if legal_bridge_pair(st, pr, allow_over_prefer=True):
                return True
        return False

    def apply_terminal(st: _PlaceState, i: int, tag: str) -> Optional[_PlaceState]:
        if st.remaining_cl < 1 or free_cap(st, i) < 1:
            return None
        # dual-terminal mono-se ban if no bridge
        if (
            forbid_dual
            and i in mono_se
            and st.n_bridge[i] == 0
            and st.n_term[i] >= 1
        ):
            return None
        nxt = st.clone()
        nxt.cn[i] += 1
        nxt.n_term[i] += 1
        nxt.remaining_cl -= 1
        nxt.terminals_on.append(i)
        nxt.log.append(f"term@{i} cn→{nxt.cn[i]} [{tag}]")
        return nxt

    def apply_bridge(
        st: _PlaceState,
        pr: Tuple[int, int],
        tag: str,
        *,
        allow_over_prefer: bool = False,
    ) -> Optional[_PlaceState]:
        pr = (min(pr[0], pr[1]), max(pr[0], pr[1]))
        if not legal_bridge_pair(st, pr, allow_over_prefer=allow_over_prefer):
            return None
        s1, s2 = position[pr[0]], position[pr[1]]
        nxt = st.clone()
        nxt.cn[s1] += 1
        nxt.cn[s2] += 1
        nxt.n_bridge[s1] += 1
        nxt.n_bridge[s2] += 1
        nxt.remaining_cl -= 1
        nxt.bridges.append(pr)
        nxt.log.append(
            f"bridge {pr} cn→({nxt.cn[s1]},{nxt.cn[s2]}) "
            f"nbr→({nxt.n_bridge[s1]},{nxt.n_bridge[s2]}) [{tag}]"
        )
        return nxt

    def new_bridged_pairs(st: _PlaceState) -> int:
        """Cd-Cd pairs this decoration links that the core did not already.

        Compactness of the finished graph tracks the number of Cd pairs
        sharing *any* common neighbour.  Pairs already sharing a Se are fixed
        by the core, so only the pairs a chloride newly creates can be steered
        here: a terminal creates none, a mu2 one, a mu3 three.  Minimising
        terminals therefore falls out of maximising this, with no threshold on
        the terminal count itself.
        """

        created: Set[Tuple[int, int]] = set()
        for left, right in st.bridges:
            pair = (left, right) if left < right else (right, left)
            if pair not in se_shared_pairs:
                created.add(pair)
        for tri in st.mu3:
            for left, right in combinations(tri, 2):
                pair = (left, right) if left < right else (right, left)
                if pair not in se_shared_pairs:
                    created.add(pair)
        return len(created)

    def score_state(st: _PlaceState) -> Tuple:
        """Higher is better for beam ranking.

        Order:
        1. Kill CN1 (must reach min CN 2).
        2. Meet min_cd_final on every host.
        3. Soft-penalize bridge load above prefer (2).
        4. Prefer decorations that link more distinct Cd pairs (compactness);
           see ``new_bridged_pairs``.  Only active in strict motif mode.
        5. Strict motif mode prefers larger bridge graphs; the historical
           path prefers **fewer** bridges for the ordinary p>=2 path.  The p=1
           unrestricted policy reverses the historical tie-break.
        6. Then raise remaining low CN / spend Cl.
        """

        n_cn1 = sum(1 for c in st.cn if c < 2)
        n_below_min = sum(
            1 for c in st.cn if min_cd_final > 0 and c < min_cd_final
        )
        n_low = sum(1 for c in st.cn if c < 3)
        n_br = sum(st.n_bridge)
        over_pref = sum(
            max(0, b - prefer_bridge_per_cd) for b in st.n_bridge
        )
        ring_deficit = min(
            (
                sum(max(0, profile[i] - st.cn[i]) for i in range(n_cd))
                for profile in required_profiles
            ),
            default=0,
        )
        n_bridge_target = n_br if strict_bridge_first else (
            -n_br if not p1_unrestricted else n_br
        )
        # Distance from the target total, negated so closer ranks higher.  A
        # plain "more bridges is better" term saturates against the per-Cd cap
        # and cannot express "stop here"; this can.
        to_target = -abs(n_br - bridge_target) if bridge_target else 0
        # 0 leaves the historical ordering bit-for-bit unchanged.
        pair_term = new_bridged_pairs(st) if maximize_bridged_pairs else 0
        return (
            -n_cn1,
            -ring_deficit,
            -n_below_min,
            -over_pref,
            to_target,
            n_bridge_target,
            pair_term,
            -n_low,
            -st.remaining_cl,
        )

    def expand(st: _PlaceState) -> List[_PlaceState]:
        """Chemically ordered expansion with beam diversity.

        Bridges are mainly a **CN1 rescue** tool (and late high-p extras).
        After every Cd is ≥ 2, terminals come first (CN2→3, then 3→4);
        extra bridges are only reconsidered when Cl remains and terminals
        cannot place, or in the late third-bridge tier.
        """

        if st.remaining_cl <= 0:
            return []
        children: List[_PlaceState] = []

        if p1_unrestricted:
            # p=1 has only two Cl ligands.  Keep bridge candidates first, but
            # do not force terminals onto the currently lowest-CN Cd.  The
            # ordinary final CN and graph legality checks still apply.
            for tri in triples:
                ch = apply_mu3(st, tuple(sorted(tri)), "p1_mu3_first")
                if ch is not None:
                    children.append(ch)
            for pr in pair_list:
                ch = apply_bridge(
                    st, pr, "p1_mu2_first", allow_over_prefer=False
                )
                if ch is not None:
                    children.append(ch)
            for i in range(n_cd):
                ch = apply_terminal(st, i, "p1_terminal_unordered")
                if ch is not None:
                    children.append(ch)
            if children:
                return children

        if strict_bridge_first:
            # Motif mode explores bridge extensions first, then appends legal
            # terminal extensions as lower-priority siblings.  This preserves
            # lower-bridge alternatives (including the two-bridge Cd4Se
            # structure) while making bridge-rich states win the beam ranking.
            for tri in triples:
                ch = apply_mu3(st, tuple(sorted(tri)), "strict_mu3_first")
                if ch is not None:
                    children.append(ch)
            for pr in pair_list:
                ch = apply_bridge(st, pr, "strict_mu2_first")
                if ch is not None:
                    children.append(ch)
            for i in range(n_cd):
                ch = apply_terminal(st, i, "terminal_after_bridge_options")
                if ch is not None:
                    children.append(ch)
            if children:
                return children

        # --- Priority 1: stranded CN=1 pure terminals ---
        for i in range(n_cd):
            if st.cn[i] == 1 and not can_bridge_host(st, i):
                ch = apply_terminal(st, i, "stranded_cn1")
                if ch is not None:
                    children.append(ch)
        if children:
            return children

        has_cn1 = any(c == 1 for c in st.cn)
        alive_profiles = [
            profile
            for profile in required_profiles
            if all(max(0, profile[i] - st.cn[i]) <= st.remaining_cl for i in range(n_cd))
        ]
        needed_slots = {
            i
            for profile in alive_profiles
            for i in range(n_cd)
            if st.cn[i] < profile[i]
        }

        if has_cn1:
            # --- Priority 2: CN1 rescue — bridge *or* terminal (beam) ---
            for pr in pair_list:
                s1, s2 = position[pr[0]], position[pr[1]]
                if st.cn[s1] != 1 and st.cn[s2] != 1:
                    continue
                if st.cn[s1] == 1 and st.cn[s2] == 1:
                    tag = "bridge_cn1_cn1"
                else:
                    tag = "bridge_help_cn1"
                ch = apply_bridge(st, pr, tag, allow_over_prefer=False)
                if ch is not None:
                    children.append(ch)
            for tri in triples:
                if not any(st.cn[position[h]] == 1 for h in tri):
                    continue
                ch = apply_mu3(st, tuple(sorted(tri)), "mu3_help_cn1")
                if ch is not None:
                    children.append(ch)
            for i in range(n_cd):
                if st.cn[i] == 1:
                    ch = apply_terminal(st, i, "term_cn1")
                    if ch is not None:
                        children.append(ch)
            if children:
                return children

        # --- Priority 3: all Cd ≥ 2 → terminals on CN=2 first ---
        # Do **not** add more bridges while CN2 can take a terminal; this is
        # what keeps k2p2 at 0–1 μ2 like multiset/DFT instead of 3–4 μ2.
        cn2_terms: List[_PlaceState] = []
        if needed_slots:
            for tri in triples:
                if not any(position[h] in needed_slots for h in tri):
                    continue
                ch = apply_mu3(st, tuple(sorted(tri)), "ring_demand_mu3")
                if ch is not None:
                    cn2_terms.append(ch)
            for pr in pair_list:
                if not any(position[h] in needed_slots for h in pr):
                    continue
                ch = apply_bridge(st, pr, "ring_demand_mu2", allow_over_prefer=False)
                if ch is not None:
                    cn2_terms.append(ch)
        for i in sorted(
            range(n_cd),
            key=lambda j: (0 if st.n_bridge[j] > 0 else 1, j),
        ):
            if st.cn[i] == 2 and (not needed_slots or i in needed_slots):
                ch = apply_terminal(st, i, "raise_cn2_to_3")
                if ch is not None:
                    cn2_terms.append(ch)
        if cn2_terms:
            return cn2_terms

        # --- Priority 4: optional bridge among CN≥2 if no CN2 terminal left ---
        # (e.g. all free hosts already CN3+ or dual-term ban blocked terms)
        still_under3 = any(c < 3 for c in st.cn)
        if still_under3:
            for pr in pair_list:
                s1, s2 = position[pr[0]], position[pr[1]]
                if st.cn[s1] < 2 or st.cn[s2] < 2:
                    continue
                ch = apply_bridge(
                    st, pr, "bridge_cn2plus", allow_over_prefer=False
                )
                if ch is not None:
                    children.append(ch)
            if children:
                return children

        # --- Priority 5: terminals CN3 → 4 ---
        for i in range(n_cd):
            if st.cn[i] == 3 and free_cap(st, i) > 0:
                ch = apply_terminal(st, i, "raise_cn3_to_4")
                if ch is not None:
                    children.append(ch)
        if children:
            return children

        # --- Priority 6: late third bridge / Cd (high-p mono-Se CN4) ---
        if all(c >= 3 for c in st.cn):
            for pr in pair_list:
                ch = apply_bridge(
                    st, pr, "bridge_third_late", allow_over_prefer=True
                )
                if ch is not None:
                    children.append(ch)
            if children:
                return children

        # --- Priority 7: drain remaining Cl as terminals ---
        for i in sorted(range(n_cd), key=lambda j: (st.cn[j], j)):
            if free_cap(st, i) > 0:
                ch = apply_terminal(st, i, "drain")
                if ch is not None:
                    children.append(ch)
        return children

    # Beam search
    initial = _PlaceState(
        cn=list(skel),
        n_bridge=[0] * n_cd,
        n_term=[0] * n_cd,
        remaining_cl=n_cl,
        bridges=[],
        mu3=[],
        terminals_on=[],
        log=["start skel=" + str(list(skel))],
    )
    beam: List[_PlaceState] = [initial]
    finished: List[_PlaceState] = []
    seen_keys: Set[Tuple] = set()

    # Safety: max steps
    max_steps = n_cl + 2
    effective_beam_width = beam_width
    if p1_unrestricted:
        # Keep every one-ligand motif for the small p=1 alphabet, while still
        # bounding pathological larger-k runs.
        effective_beam_width = max(
            beam_width,
            min(128, len(triples) + len(pair_list) + n_cd),
        )
    for _step in range(max_steps):
        if not beam:
            break
        nxt_beam: List[_PlaceState] = []
        for st in beam:
            if st.remaining_cl == 0:
                finished.append(st)
                continue
            kids = expand(st)
            if not kids:
                # stuck with Cl left — still keep if min_cn ok enough
                status.infeasible += 1
                continue
            for ch in kids:
                raw_key = ch.key()
                key = state_key(ch)
                if key in seen_keys:
                    status.revisited += 1
                    if len(slot_maps) > 1 and raw_key != key[:-1]:
                        status.symmetry_pruned += 1
                    continue
                seen_keys.add(key)
                nxt_beam.append(ch)
        # rank and trim
        nxt_beam.sort(key=score_state, reverse=True)
        beam = nxt_beam[: max(1, effective_beam_width)]
        if all(st.remaining_cl == 0 for st in beam) and beam:
            finished.extend(beam)
            break

    # complete any remaining in beam that finished
    for st in beam:
        if st.remaining_cl == 0:
            finished.append(st)

    # filter finished by global rules
    good: List[_PlaceState] = []
    for st in finished:
        if st.remaining_cl != 0:
            continue
        ok = True
        for i in range(n_cd):
            if min_cd_final > 0 and st.cn[i] < min_cd_final:
                ok = False
                break
            if st.n_bridge[i] > 0 and st.cn[i] < min_bridge:
                ok = False
                break
            if (
                forbid_dual
                and i in mono_se
                and st.n_term[i] >= 2
                and st.n_bridge[i] == 0
            ):
                ok = False
                break
        if ok:
            good.append(st)
        else:
            status.infeasible += 1

    # dedup by key
    uniq: Dict[Tuple, _PlaceState] = {}
    for st in good:
        uniq[st.key()] = st
    return list(uniq.values())


def _emit_edges(
    st: _PlaceState,
    cd_list: Sequence[int],
    cl_list: Sequence[int],
) -> Tuple[Tuple[int, int], ...]:
    edges: List[Tuple[int, int]] = []
    cl_i = 0
    for tri in st.mu3:
        cl = cl_list[cl_i]
        cl_i += 1
        edges.extend((cl, host) for host in tri)
    for pr in st.bridges:
        cl = cl_list[cl_i]
        cl_i += 1
        edges.append((cl, pr[0]))
        edges.append((cl, pr[1]))
    for slot in st.terminals_on:
        cl = cl_list[cl_i]
        cl_i += 1
        edges.append((cl, cd_list[slot]))
    return tuple(sorted(edges))


def iter_cl_attachments_bridge_first(
    k: int,
    p: int,
    inorganic_edges: Sequence[Tuple[int, int]],
    spec: NucleationSpec,
    pack: GeometryPack,
    *,
    max_assignments: int = 0,
    status: Optional[_DecorationStatus] = None,
    frame_options: int = 0,
    state: Optional[_State] = None,
    cation_ids: Optional[Sequence[int]] = None,
    degree_vectors: Optional[Sequence[Tuple[int, ...]]] = None,
    slice_builder: Optional[
        Callable[[Tuple[int, ...]], Optional[_DegreeSlice]]
    ] = None,
    required_degree_profiles: Optional[Sequence[Tuple[int, ...]]] = None,
    beam_width: int = DEFAULT_BEAM,
    allowed_bridge_pairs: Optional[Set[Tuple[int, int]]] = None,
    hard_max_bridge_per_cd: int = DEFAULT_HARD_MAX_BRIDGE_PER_CD,
    strict_bridge_first: bool = False,
) -> Iterable[Tuple[Tuple[int, int], ...]]:
    """Graph-only sequential μ3/μ2-first passivation.

    The chemical beam remains graph-based. When the caller supplies an
    advisory allowed_bridge_pairs set from a conservative skeleton frame,
    grossly overlong Cd host pairs are removed before expansion; final CN
    radii and exact sphere feasibility are still checked during embedding.
    """

    if status is None:
        status = _DecorationStatus()
    if state is None:
        return
    _se_ids, cd_ids, cl_ids = _index_blocks(k, p)
    cd_list = list(cd_ids)
    cl_list = list(cl_ids)
    base = _skel_degrees(cd_list, inorganic_edges)
    profiles = [
        tuple(int(x) for x in profile)
        for profile in (required_degree_profiles or ())
        if len(profile) == len(cd_list)
    ]
    position = {host: i for i, host in enumerate(cd_list)}
    pairs = list(combinations(cd_list, 2))
    if allowed_bridge_pairs is not None:
        pair_set = {
            (min(int(left), int(right)), max(int(left), int(right)))
            for left, right in allowed_bridge_pairs
        }
        pairs = [pair for pair in pairs if pair in pair_set]
    triples = [
        triple for triple in combinations(cd_list, 3)
        if allowed_bridge_pairs is None
        or all(
            (min(int(left), int(right)), max(int(left), int(right))) in pair_set
            for left, right in combinations(triple, 2)
        )
    ]
    inorganic = nx.Graph()
    inorganic.add_nodes_from(
        (node, {"element": spec.core.anion}) for node in _se_ids
    )
    inorganic.add_nodes_from(
        (
            node,
            {
                # Ring-demand signatures are part of the colored skeleton for
                # decoration symmetry.  A graph automorphism may not exchange
                # a forced-ring Cd with an unconstrained precursor Cd.
                "element": (
                    spec.core.cation,
                    tuple(
                        sorted(profile[position[node]] for profile in profiles)
                    ),
                ),
            },
        )
        for node in cd_list
    )
    inorganic.add_edges_from(inorganic_edges)
    host_maps = []
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        inorganic,
        inorganic,
        node_match=nx.algorithms.isomorphism.categorical_node_match(
            "element", ""
        ),
    )
    host_maps.extend(matcher.isomorphisms_iter())
    status.automorphisms = max(status.automorphisms, len(host_maps))
    # Cd pairs already sharing an anion in the core.  A chloride bridging such
    # a pair adds no new Cd-Cd link, so the compactness term must not reward it.
    anion_hosts: Dict[int, List[int]] = {}
    for left, right in inorganic_edges:
        if left in position and right not in position:
            anion_hosts.setdefault(int(right), []).append(int(left))
        elif right in position and left not in position:
            anion_hosts.setdefault(int(left), []).append(int(right))
    core_shared_pairs = {
        (min(a, b), max(a, b))
        for hosts in anion_hosts.values()
        for a, b in combinations(sorted(hosts), 2)
    }

    states = sequential_passivate(
        k=k,
        p=p,
        cd_list=cd_list,
        skel=base,
        pairs=pairs,
        triples=triples,
        required_profiles=profiles,
        position=position,
        spec=spec,
        n_cl=len(cl_list),
        beam_width=beam_width,
        status=status,
        host_maps=host_maps,
        hard_max_bridge_per_cd=int(hard_max_bridge_per_cd),
        strict_bridge_first=strict_bridge_first,
        se_shared_pairs=tuple(sorted(core_shared_pairs)),
    )
    emitted = 0
    for completed in states:
        if profiles and not any(
            all(completed.cn[i] >= profile[i] for i in range(len(cd_list)))
            for profile in profiles
        ):
            status.infeasible += 1
            continue
        if max_assignments > 0 and emitted >= max_assignments:
            status.truncated = True
            return
        emitted += 1
        yield _emit_edges(completed, cd_list, cl_list)
