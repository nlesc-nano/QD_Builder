"""Pure-Python fallback for ``_beam_key`` when the Cython extension is absent.

Build the extension with::

    pip install -e ".[speed]"
    # or
    python setup.py build_ext --inplace

The API matches ``_beam_key.pyx``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

HostRelabel = Union[Dict[int, int], Sequence[int]]


def identity_state_key(
    cn: Sequence[int],
    n_bridge: Sequence[int],
    n_term: Sequence[int],
    bridges: Sequence[Tuple[int, int]],
    mu3: Sequence[Tuple[int, ...]],
    terminals_on: Sequence[int],
    remaining_cl: int,
) -> Tuple:
    return (
        tuple(cn),
        tuple(n_bridge),
        tuple(n_term),
        tuple(sorted(bridges)),
        tuple(sorted(mu3)),
        tuple(sorted(terminals_on)),
        int(remaining_cl),
    )


def pair_bridge_count(
    bridges: Sequence[Tuple[int, int]],
    mu3: Sequence[Tuple[int, ...]],
    a: int,
    b: int,
) -> int:
    lo, hi = (a, b) if a <= b else (b, a)
    count = sum(1 for pr in bridges if pr == (lo, hi))
    count += sum(1 for tri in mu3 if lo in tri and hi in tri)
    return count


def _relabel_get(relabel: HostRelabel, host: int) -> int:
    return int(relabel[host])


def canonical_state_key(
    cn: Sequence[int],
    n_bridge: Sequence[int],
    n_term: Sequence[int],
    bridges: Sequence[Tuple[int, int]],
    mu3: Sequence[Tuple[int, ...]],
    terminals_on: Sequence[int],
    remaining_cl: int,
    slot_maps: Sequence[Sequence[int]],
    slot_inverses: Sequence[Sequence[int]],
    host_relabels: Sequence[HostRelabel],
) -> Tuple:
    if not slot_maps:
        return identity_state_key(
            cn, n_bridge, n_term, bridges, mu3, terminals_on, remaining_cl
        )
    best: Optional[Tuple] = None
    n_cd = len(cn)
    for mapping, inverse, relabel in zip(
        slot_maps, slot_inverses, host_relabels
    ):
        bridge_list = []
        for a, b in bridges:
            ra = _relabel_get(relabel, a)
            rb = _relabel_get(relabel, b)
            bridge_list.append((ra, rb) if ra <= rb else (rb, ra))
        bridge_list.sort()
        mu3_list = []
        for tri in mu3:
            rx = _relabel_get(relabel, tri[0])
            ry = _relabel_get(relabel, tri[1])
            rz = _relabel_get(relabel, tri[2])
            if rx > ry:
                rx, ry = ry, rx
            if ry > rz:
                ry, rz = rz, ry
            if rx > ry:
                rx, ry = ry, rx
            mu3_list.append((rx, ry, rz))
        mu3_list.sort()
        terminals = tuple(sorted(int(mapping[slot]) for slot in terminals_on))
        candidate = (
            tuple(int(cn[inverse[i]]) for i in range(n_cd)),
            tuple(int(n_bridge[inverse[i]]) for i in range(n_cd)),
            tuple(int(n_term[inverse[i]]) for i in range(n_cd)),
            tuple(bridge_list),
            tuple(mu3_list),
            terminals,
            int(remaining_cl),
        )
        if best is None or candidate < best:
            best = candidate
    return best  # type: ignore[return-value]


def is_cython() -> bool:
    return False
