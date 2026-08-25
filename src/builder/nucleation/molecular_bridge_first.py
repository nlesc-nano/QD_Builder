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
    Mapping,
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


# Optional Cython beam-key kernel (falls back to pure Python).
try:
    from . import _beam_key as _beam_key  # type: ignore

    _BEAM_KEY_BACKEND = "cython" if getattr(_beam_key, "is_cython", lambda: False)() else "ext"
except Exception:  # noqa: BLE001
    from . import _beam_key_fallback as _beam_key  # type: ignore

    _BEAM_KEY_BACKEND = "python"


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
        return _beam_key.identity_state_key(
            self.cn,
            self.n_bridge,
            self.n_term,
            self.bridges,
            self.mu3,
            self.terminals_on,
            self.remaining_cl,
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

    # Inverses as tuples (list indexing, not dict hashing) and dense host
    # relabel arrays (index = host_id, value = relabeled host) so the Cython
    # hot loop does one integer index instead of a dict hash.
    slot_inverses: List[Tuple[int, ...]] = []
    max_host_id = max(cd_list) if cd_list else 0
    host_relabels: List[Tuple[int, ...]] = []
    for mapping in slot_maps:
        inverse = [0] * n_cd
        for original, mapped in enumerate(mapping):
            inverse[mapped] = original
        slot_inverses.append(tuple(inverse))
        dense = [-1] * (max_host_id + 1)
        for host in cd_list:
            dense[host] = cd_list[mapping[position[host]]]
        host_relabels.append(tuple(dense))

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
        """Key for an asymmetric skeleton: no relabelling to apply."""

        return _beam_key.identity_state_key(
            st.cn,
            st.n_bridge,
            st.n_term,
            st.bridges,
            st.mu3,
            st.terminals_on,
            st.remaining_cl,
        )

    def state_key(st: _PlaceState) -> Tuple:
        """Canonicalize a partial state under skeleton host automorphisms."""

        if identity_only:
            return _identity_key(st)
        return _beam_key.canonical_state_key(
            st.cn,
            st.n_bridge,
            st.n_term,
            st.bridges,
            st.mu3,
            st.terminals_on,
            st.remaining_cl,
            slot_maps,
            slot_inverses,
            host_relabels,
        )

    def free_cap(st: _PlaceState, i: int) -> int:
        return max_cd - st.cn[i]

    def pair_bridge_count(st: _PlaceState, pr: Tuple[int, int]) -> int:
        pr = (min(pr[0], pr[1]), max(pr[0], pr[1]))
        return int(
            _beam_key.pair_bridge_count(st.bridges, st.mu3, pr[0], pr[1])
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


def _saturated_host_bridges(
    bridges: Sequence[Tuple[int, int]],
    cn: Sequence[int],
    position: Mapping[int, int],
    max_cation_cn: int,
) -> int:
    """Count mu2 bridges whose *two* hosts both finish at the maximum CN.

    Measured over the k1-k4 zb run: a shell carrying one of these yields 3.0%
    propagation-eligible endpoints against 13.5% without, because a Cd already
    saturated by the core has no room to relax around the bridge.  The count is
    over final CN (skeleton + ligand load), not skeleton CN -- which is what
    ``bridge_target_min_host_cn_cap`` filters and why that knob does not
    subsume this one.
    """

    total = 0
    for left, right in bridges:
        a_slot = position.get(int(left))
        b_slot = position.get(int(right))
        if a_slot is None or b_slot is None:
            continue
        if cn[a_slot] >= max_cation_cn and cn[b_slot] >= max_cation_cn:
            total += 1
    return total


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
    max_saturated_host_bridges = int(
        getattr(spec.graph_rules, "max_saturated_host_bridges", -1)
    )
    emitted = 0
    for completed in states:
        if profiles and not any(
            all(completed.cn[i] >= profile[i] for i in range(len(cd_list)))
            for profile in profiles
        ):
            status.infeasible += 1
            continue
        if max_saturated_host_bridges >= 0 and _saturated_host_bridges(
            completed.bridges,
            completed.cn,
            position,
            int(spec.graph_rules.max_cn[spec.core.cation]),
        ) > max_saturated_host_bridges:
            status.infeasible += 1
            continue
        if max_assignments > 0 and emitted >= max_assignments:
            status.truncated = True
            return
        emitted += 1
        yield _emit_edges(completed, cd_list, cl_list)


def iter_cl_attachments_bridge_target(
    k: int,
    p: int,
    inorganic_edges: Sequence[Tuple[int, int]],
    spec: NucleationSpec,
    pack: Optional[GeometryPack] = None,
    *,
    max_assignments: int = 0,
    status: Optional[_DecorationStatus] = None,
    state: Optional[_State] = None,
    cation_ids: Optional[Sequence[int]] = None,
    hard_max_bridge_per_cd: int = DEFAULT_HARD_MAX_BRIDGE_PER_CD,
    ring_closing_only: bool = True,
    min_host_cn_cap: int = 2,
    max_shared_per_pair: int = 2,
    avoid_triangles: bool = True,
    count_window: int = 0,
    max_emissions: int = 0,
    max_walk_nodes: int = 0,
) -> Iterable[Tuple[Tuple[int, int], ...]]:
    """Enumerate decorations that bridge as much chloride as the core allows.

    The measured rule is that the most stable structure of a bin uses almost
    every chloride as a bridge -- ``n_bridges(best) = 0.96*(2p) - 0.21`` with
    r=0.994 over 24 bins, 11 of 24 winners carrying no terminal at all.  The
    greedy tier search cannot reach that: it interleaves terminals to satisfy
    ``min_bridged_host_cn`` and strands itself, topping out at 5 bridges where
    the best k=4 p=4 structure has 8.

    So rather than decorate and hope, this picks the bridge set *first*: choose
    the largest feasible number of mu2 bridges over distinct Cd pairs, subject
    to the per-Cd load cap and each host's free valence, then spend whatever
    chloride is left on terminals.  Because it never enumerates the
    terminal-heavy bulk, the search is smaller than the tier version, not
    larger.

    Pairs are chosen in strictly increasing index order, so each bridge *set*
    is visited once rather than once per ordering.

    ``max_emissions`` / ``max_walk_nodes`` (or the pack knobs
    ``bridge_target_max_emissions_per_skeleton`` /
    ``bridge_target_max_nodes_per_skeleton``) hard-stop the walk so high-p
    bins cannot stream millions of graphs before the reservoir.  Both are
    counted *per productive bridge count*, so the total for one skeleton is
    bounded by ``max_emissions * (count_window + 1)``.  0 = unlimited.
    """

    if status is None:
        status = _DecorationStatus()
    se_ids, cd_ids, cl_ids = _index_blocks(k, p)
    cd_list = list(cation_ids) if cation_ids is not None else list(cd_ids)
    cl_list = list(cl_ids)
    n_cd = len(cd_list)
    n_cl = len(cl_list)
    if n_cd == 0 or n_cl == 0:
        return
    position = {host: i for i, host in enumerate(cd_list)}
    max_cd = int(spec.graph_rules.max_cn[spec.core.cation])
    min_cd_final = _min_cd_cn_policy(k, p, spec)
    max_shared = int(
        getattr(spec.graph_rules, "max_shared_ligands_per_host_pair", 1) or 1
    )
    if max_shared_per_pair > 0:
        max_shared = min(max_shared, max_shared_per_pair)
    min_bridge_cn = int(getattr(spec.graph_rules, "min_bridged_host_cn", 0) or 0)
    forbid_dual = bool(
        getattr(spec.graph_rules, "forbid_mono_se_dual_terminal", False)
    )
    skel = _skel_degrees(cd_list, inorganic_edges)
    mono_se = {i for i, s in enumerate(skel) if s == 1}
    # Per-skeleton early stops: pack knobs, overridable by explicit args.
    # These are *deliberate* sample limits (not incomplete enumeration): they
    # must not set status.truncated, or the bin aborts as if max_assignments
    # were hit.
    if max_emissions <= 0:
        max_emissions = int(
            getattr(
                spec.graph_rules,
                "bridge_target_max_emissions_per_skeleton",
                0,
            )
            or 0
        )
    if max_walk_nodes <= 0:
        max_walk_nodes = int(
            getattr(
                spec.graph_rules,
                "bridge_target_max_nodes_per_skeleton",
                0,
            )
            or 0
        )
    max_saturated_host_bridges = int(
        getattr(spec.graph_rules, "max_saturated_host_bridges", -1)
    )
    # Global assignment cap (bin-level safety guard) is separate.
    assignment_cap = int(max_assignments) if max_assignments > 0 else 0

    # Skeleton automorphisms, in Cd-slot coordinates, for orderly generation.
    # Without them the walk explores one branch per automorphic image of the
    # same bridge set: measured 44845 emissions for 1867 distinct graphs over
    # k1p2..k3p4 (24x), and 56x at k3p5.  Deduplicating the finished graphs
    # cannot recover that -- the cost is in the tree, so the pruning has to
    # happen at every node (see ``walk_bridges``).
    inorganic = nx.Graph()
    inorganic.add_nodes_from(
        (node, {"element": spec.core.anion}) for node in se_ids
    )
    inorganic.add_nodes_from(
        (node, {"element": spec.core.cation}) for node in cd_list
    )
    inorganic.add_edges_from(inorganic_edges)
    aut_cap = int(
        getattr(spec.graph_rules, "bridge_target_max_automorphisms", 4096) or 0
    )
    slot_maps: List[Tuple[int, ...]] = []
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        inorganic,
        inorganic,
        node_match=nx.algorithms.isomorphism.categorical_node_match(
            "element", ""
        ),
    )
    for host_map in matcher.isomorphisms_iter():
        try:
            slot_maps.append(tuple(position[host_map[host]] for host in cd_list))
        except KeyError:
            continue
        # |Aut| grows factorially in equivalent precursor cations; past the cap
        # the orbit computation costs more than the branches it removes.
        if aut_cap > 0 and len(slot_maps) > aut_cap:
            slot_maps = [tuple(range(n_cd))]
            break
    if not slot_maps:
        slot_maps = [tuple(range(n_cd))]
    identity = tuple(range(n_cd))
    trivial_group = len(slot_maps) == 1 and slot_maps[0] == identity

    def _slot_pairs(chosen: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
        return tuple(sorted(
            (min(position[a], position[b]), max(position[a], position[b]))
            for a, b in chosen
        ))

    def _apply(mapping: Tuple[int, ...],
               pairs: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
        return tuple(sorted(
            (min(mapping[x], mapping[y]), max(mapping[x], mapping[y]))
            for x, y in pairs
        ))

    def _stabiliser(group: Sequence[Tuple[int, ...]],
                    chosen: Sequence[Tuple[int, int]]) -> List[Tuple[int, ...]]:
        """The subgroup fixing the bridge set chosen so far.

        Shrinks fast as bridges are committed, so the orbit test at deeper
        nodes is nearly free.
        """

        target = _slot_pairs(chosen)
        return [m for m in group if _apply(m, target) == target]

    # Room for chloride on each host, and the cap on how much of it may be
    # bridging.  A hard cap of 0 means "no cap".
    cap = hard_max_bridge_per_cd if hard_max_bridge_per_cd > 0 else n_cl
    room = [max(0, max_cd - skel[i]) for i in range(n_cd)]
    bridge_room = [min(cap, room[i]) for i in range(n_cd)]

    pair_list = [
        (min(a, b), max(a, b)) for a, b in combinations(cd_list, 2)
    ]
    # Bridging is driven by free valence, not by symmetry among Cd pairs.
    # Measured over 98088 bridged host pairs: a Cd with skeleton CN 1 hosts 2.20
    # bridges on average and is bare only 1.3% of the time, while CN 3 hosts
    # 0.98 and CN 4 only 0.49 (bare 59%).  79.6% of all bridges have at least
    # one host at skeleton CN <= 1, and pairs whose *both* hosts sit at CN >= 3
    # account for 0.1%.  Dropping those removes combinations that essentially
    # never occur, which is where the blow-up lived.
    if min_host_cn_cap > 0:
        pair_list = [
            (a, b) for a, b in pair_list
            if min(skel[position[a]], skel[position[b]]) <= min_host_cn_cap
        ]
    # Hungriest hosts first: a CN1 Cd almost always takes two bridges, so
    # committing those early prunes far more than exploring them symmetrically.
    pair_list.sort(
        key=lambda pr: (
            min(skel[position[pr[0]]], skel[position[pr[1]]]),
            skel[position[pr[0]]] + skel[position[pr[1]]],
            pr,
        )
    )
    if ring_closing_only:
        # A chloride bridge is not placed between arbitrary Cd: measured over
        # 98472 bridged host pairs, 54.2% close a 4-ring and 44.6% a 6-ring,
        # and nothing else reaches 1.5%.  In graph terms the two hosts sit at
        # distance 2 (shared anion) or 4 (two-anion path) in the cation-anion
        # core, so pairs further apart cannot host a bridge and need not be
        # enumerated.
        core = nx.Graph()
        core.add_nodes_from(list(se_ids) + cd_list)
        core.add_edges_from(inorganic_edges)
        dist = dict(nx.all_pairs_shortest_path_length(core))
        pair_list = [
            (a, b) for a, b in pair_list
            if dist.get(a, {}).get(b, 99) in (2, 4)
        ]
    # On a metric skeleton (Move Z) hop 4 can be 8.68 Å through an occupied
    # cation.  Drop those pairs when lattice coordinates are present.
    if state is not None and spec.graph_rules.bridge_cd_cd_max_distance is not None:
        host_xyz = np.asarray(
            [atom.coordinates for atom in state.atoms], dtype=float
        )
        n_need = max(cd_list) + 1 if cd_list else 0
        if (
            host_xyz.shape[0] >= n_need
            and n_need > 0
            and np.all(np.isfinite(host_xyz[:n_need]))
            and not np.allclose(host_xyz[:n_need], 0.0)
        ):
            from .molecular_zb_growth import zb_metric_bridge_pairs

            pair_list = zb_metric_bridge_pairs(
                pair_list, host_xyz, cd_list, spec
            )
    # Ceiling on bridges: every bridge consumes two host slots, and there are
    # only ``sum(bridge_room)`` of them; it also cannot exceed the Cl budget or
    # the number of distinct pairs available at max_shared per pair.
    slot_ceiling = sum(bridge_room) // 2
    pair_ceiling = len(pair_list) * max(1, max_shared)
    ceiling = min(n_cl, slot_ceiling, pair_ceiling)

    # Cheap pre-walk feasibility: total free Cd valence must hold every Cl
    # (each Cl needs at least one host slot; a μ2 uses two).
    total_room = sum(room)
    if total_room < n_cl:
        return
    # No bridge pairs and no way to place pure terminals under min_cn is rare;
    # ceiling 0 with n_cl > 0 still allows the target=0 terminal-only tier.
    if ceiling == 0 and n_cl > 0 and total_room < n_cl:
        return

    # ``emitted`` is the whole-skeleton total the bin-level assignment guard
    # counts.  The per-skeleton emission and node caps are *per productive
    # bridge count*: they used to be global, so with ``count_window`` open the
    # top tier spent the entire budget and the walk returned before it could
    # step down -- and a tier whose every emission is filtered out (by the
    # saturated-host cap, say) burned the node budget producing nothing.
    emitted = 0
    window_emitted = 0
    walk_nodes = 0
    seen_keys: Set[Tuple] = set()

    def _hit_skel_emission_cap() -> bool:
        return max_emissions > 0 and window_emitted >= max_emissions

    def _hit_assignment_cap() -> bool:
        return assignment_cap > 0 and emitted >= assignment_cap

    def _hit_any_emission_cap() -> bool:
        return _hit_skel_emission_cap() or _hit_assignment_cap()

    def _hit_node_cap() -> bool:
        return max_walk_nodes > 0 and walk_nodes >= max_walk_nodes

    def _stop_walk(*, as_incomplete: bool = False) -> None:
        """End the walk; only assignment-cap hits mark the bin incomplete."""

        if as_incomplete or _hit_assignment_cap():
            status.truncated = True

    def terminal_fills(load: List[int], budget: int) -> Iterable[Tuple[int, ...]]:
        """Multisets of terminal hosts using exactly ``budget`` chloride."""

        if budget == 0:
            yield ()
            return
        free = [room[i] - load[i] for i in range(n_cd)]

        def walk(start: int, left: int, acc: List[int]):
            if left == 0:
                yield tuple(acc)
                return
            for i in range(start, n_cd):
                if free[i] <= 0:
                    continue
                free[i] -= 1
                acc.append(i)
                yield from walk(i, left - 1, acc)
                acc.pop()
                free[i] += 1

        yield from walk(0, budget, [])

    def finish(chosen: List[Tuple[int, int]], load: List[int],
               group: Sequence[Tuple[int, ...]] = ()):
        nonlocal emitted, window_emitted
        budget = n_cl - len(chosen)
        for terms in terminal_fills(load, budget):
            if _hit_any_emission_cap():
                return
            cn = [skel[i] + load[i] for i in range(n_cd)]
            for slot in terms:
                cn[slot] += 1
            if min_cd_final > 0 and any(c < min_cd_final for c in cn):
                continue
            # A bridged host must finish at min_bridged_host_cn.  Without this
            # the generator emits decorations the downstream screen rejects,
            # and because the target loop stops descending as soon as a target
            # *emits*, the terminal-only tier below was never reached -- which
            # is why k1p1 and k2p1 produced nothing at all.
            if min_bridge_cn > 0 and any(
                load[i] > 0 and cn[i] < min_bridge_cn for i in range(n_cd)
            ):
                continue
            # Cap bridges between two hosts that finish saturated.  Applied
            # here rather than in walk_bridges because the terminal fill is
            # what settles the final CN of each host.
            if max_saturated_host_bridges >= 0 and _saturated_host_bridges(
                chosen, cn, position, max_cd
            ) > max_saturated_host_bridges:
                continue
            # Same rule as mono_se_dual_terminal_violations: a mono-Se Cd with
            # two terminal Cl and no bridge is illegal.  Enforce here so the
            # productive-target loop never stops on a shell the screen dumps.
            if forbid_dual and terms:
                term_on = [0] * n_cd
                for slot in terms:
                    term_on[slot] += 1
                if any(
                    load[i] == 0 and term_on[i] >= 2 for i in mono_se
                ):
                    continue
            st = _PlaceState(
                cn=cn,
                n_bridge=list(load),
                n_term=[0] * n_cd,
                remaining_cl=0,
                bridges=list(chosen),
                mu3=[],
                terminals_on=list(terms),
            )
            # Terminals are canonicalised under whatever symmetry survives the
            # bridge set, so equivalent terminal placements collapse too.
            if len(group) > 1:
                bridge_slots = _slot_pairs(chosen)
                key = min(
                    (
                        _apply(mapping, bridge_slots),
                        tuple(sorted(mapping[slot] for slot in terms)),
                    )
                    for mapping in group
                )
            else:
                key = (tuple(sorted(chosen)), tuple(sorted(terms)))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            yield _emit_edges(st, cd_list, cl_list)
            emitted += 1
            window_emitted += 1

    def _closes_triangle(chosen: Sequence[Tuple[int, int]], a: int, b: int) -> bool:
        """Whether bridging (a, b) makes three Cd mutually bridged.

        Triangles in the bridge network correlate with *higher* energy in all
        21 measured bins (rho +0.19), so they are skipped on the first pass.
        """

        na = {y for x, y in chosen if x == a} | {x for x, y in chosen if y == a}
        nb = {y for x, y in chosen if x == b} | {x for x, y in chosen if y == b}
        return bool(na & nb)

    def walk_bridges(start: int, load: List[int],
                     chosen: List[Tuple[int, int]], target: int,
                     used: Dict[Tuple[int, int], int], share_cap: int,
                     no_triangles: bool,
                     group: Sequence[Tuple[int, ...]]):
        nonlocal walk_nodes
        if _hit_any_emission_cap() or _hit_node_cap():
            _stop_walk()
            return
        walk_nodes += 1
        if _hit_node_cap():
            _stop_walk()
            return
        if len(chosen) == target:
            yield from finish(chosen, load, group)
            return
        # Orderly generation: two candidate pairs in the same orbit under the
        # stabiliser of ``chosen`` produce isomorphic children, because the
        # automorphism carrying one to the other fixes everything already
        # placed.  Exploring one representative per orbit prunes the tree
        # rather than the leaves, which is where the duplication lives.
        orbit_seen: Set[Tuple[Tuple[int, int], ...]] = set()
        for idx in range(start, len(pair_list)):
            if _hit_any_emission_cap() or _hit_node_cap():
                _stop_walk()
                return
            a, b = pair_list[idx]
            s1, s2 = position[a], position[b]
            if used.get((a, b), 0) >= share_cap:
                continue
            if load[s1] >= bridge_room[s1] or load[s2] >= bridge_room[s2]:
                continue
            if no_triangles and _closes_triangle(chosen, a, b):
                continue
            if len(group) > 1:
                extended = _slot_pairs(chosen) + (
                    (min(s1, s2), max(s1, s2)),
                )
                canon = min(
                    _apply(mapping, extended) for mapping in group
                )
                if canon in orbit_seen:
                    continue
                orbit_seen.add(canon)
                child_group = _stabiliser(group, chosen + [(a, b)])
            else:
                child_group = group
            load[s1] += 1
            load[s2] += 1
            used[(a, b)] = used.get((a, b), 0) + 1
            chosen.append((a, b))
            # Lower bound on the saturated-host bridge count: a host already
            # at max CN from core + bridges can only gain from terminals, so
            # this pair is committed to finishing saturated on both sides.
            # Pruning in the tree rather than at finish() is what stops a
            # fully rejected tier from consuming the node budget.
            if max_saturated_host_bridges >= 0 and sum(
                1
                for left, right in chosen
                if skel[position[left]] + load[position[left]] >= max_cd
                and skel[position[right]] + load[position[right]] >= max_cd
            ) > max_saturated_host_bridges:
                chosen.pop()
                used[(a, b)] -= 1
                load[s1] -= 1
                load[s2] -= 1
                continue
            # A pair may be revisited only when doubles are open, so the walk
            # restarts at idx rather than idx+1 in that case.
            nxt = idx if share_cap > 1 else idx + 1
            yield from walk_bridges(nxt, load, chosen, target, used,
                                    share_cap, no_triangles, child_group)
            chosen.pop()
            used[(a, b)] -= 1
            load[s1] -= 1
            load[s2] -= 1

    # Aim at full saturation and step down only if a target yields nothing:
    # the stable structures sit at the top of this range.
    # Tiers, in the order the chemistry suggests: one chloride per pair and no
    # triangles first; then allow triangles; and only when distinct pairs are
    # exhausted, a second chloride on a pair.  Measured: 92.0% of bridged pairs
    # carry exactly one Cl and doubles only appear as p rises and pairs run out
    # (0.3% of pairs at k5p1, 8.9% at k5p11); 3+ never occurs.
    tiers = [(1, True), (1, False)] if avoid_triangles else [(1, False)]
    # Clamp to the chemical rule.  Without this the doubles tier is built from
    # bridge_target_max_shared_per_pair alone and can place two Cl on a pair
    # that max_shared_ligands_per_host_pair forbids: the screen would then
    # reject the emission while the tier loop counted it as productive and
    # stopped descending -- the same failure that emptied the p=1 bins.
    doubles_cap = min(max_shared_per_pair, max_shared)
    if doubles_cap > 1:
        tiers.append((doubles_cap, False))
    # ``count_window`` is how many *productive* bridge counts to emit, not how
    # many to try: 0 keeps the historical behaviour of stopping at the first
    # one that yields anything.  Widening it covers the observed spread --
    # only 5.7% of relaxed structures sit at exactly 2p bridges but 51% are
    # within 2 of it -- which is what turns a single shell into a map.
    # Productive means finish() actually yielded: every decoration rule that
    # used to live only in the screen (min_bridged, min_cn, mono-Se dual
    # terminal) is enforced there, so a shell of pure rejects cannot stop
    # the descent.
    windows = 0
    for target in range(ceiling, -1, -1):
        if _hit_assignment_cap():
            _stop_walk()
            return
        window_emitted = 0
        walk_nodes = 0
        produced = False
        for share_cap, no_tri in tiers:
            for edges in walk_bridges(0, [0] * n_cd, [], target, {},
                                      share_cap, no_tri, slot_maps):
                produced = True
                yield edges
                if _hit_any_emission_cap() or _hit_node_cap():
                    _stop_walk()
                    return
            if produced:
                break
        if produced:
            windows += 1
            if windows > count_window:
                return
