"""Graph connectivity: full vs inorganic (Cd–Se) subgraph."""

from __future__ import annotations

from typing import List, Sequence, Tuple


def _components(n: int, edges: Sequence[Tuple[int, int]]) -> int:
    if n == 0:
        return 0
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    return len({find(i) for i in range(n)})


def full_components(
    n_atoms: int, neighbors: Sequence[Sequence[int]]
) -> int:
    edges: List[Tuple[int, int]] = []
    for i, neigh in enumerate(neighbors):
        for j in neigh:
            if i < j:
                edges.append((i, j))
    return _components(n_atoms, edges)


def inorganic_components(
    symbols: Sequence[str],
    neighbors: Sequence[Sequence[int]],
) -> Tuple[int, int]:
    """Return (n_components, n_inorganic_atoms) on Cd–Se subgraph.

    Only edges between Cd and Se count. Isolated Cd/Se atoms each form a
    component; Cl is ignored entirely.
    """

    inorganic_ids = [
        i for i, sym in enumerate(symbols) if sym in {"Cd", "Se"}
    ]
    if not inorganic_ids:
        return 0, 0
    index = {old: new for new, old in enumerate(inorganic_ids)}
    edges: List[Tuple[int, int]] = []
    for i in inorganic_ids:
        for j in neighbors[i]:
            if symbols[j] not in {"Cd", "Se"}:
                continue
            if symbols[i] == symbols[j]:
                continue  # should not appear in chemical graph
            a, b = index[i], index[j]
            if a < b:
                edges.append((a, b))
    return _components(len(inorganic_ids), edges), len(inorganic_ids)
