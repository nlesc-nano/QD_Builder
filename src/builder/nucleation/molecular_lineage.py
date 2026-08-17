"""Grow (k, p) cores from accepted (k-1, p') cores instead of re-enumerating.

Exhaustive skeleton enumeration grows about five-fold per unit k (measured
8 -> 52 -> 314 accepted cores at p=4 for k=3,4,5), so by k=6 it stops being
affordable.  The lattice engine already solves this by shedding precursor
packages from accepted parents and regrowing; this module does the same thing
for the lattice-free molecular map, where it is considerably simpler because a
molecular skeleton carries no ligands to begin with.

One lineage step is::

    parent core at (k, p)          k Se, k+p Cd
      -> shed s precursor Cd       k Se, k+p-s Cd
      -> add one CdSe monomer      k+1 Se, k+p-s+1 Cd
      == child core at (k+1, p_out) with p_out = p - s

The child edge lists are returned in the canonical ``_index_blocks(k+1, p_out)``
layout so they drop straight into ``enumerate_molecular_bin(...,
precomputed_skeletons=...)``.

This is a *greedy* strategy: a child reachable only from a parent that was
filtered out is never generated.  Validate it against an exhaustive bin before
relying on it -- ``lineage_vs_exhaustive`` in the tests does exactly that.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

from ..nc_types import NucleationSpec
from .types import _State

Edge = Tuple[int, int]
EdgeList = Tuple[Edge, ...]


def _blocks(k: int, p: int) -> Tuple[range, range]:
    """(se_ids, cd_ids) for a bare core, mirroring ``_index_blocks``."""

    return range(0, k), range(k, k + k + p)


def _canonical_relabel(
    graph: nx.Graph,
    se_nodes: Sequence[int],
    cd_nodes: Sequence[int],
    k_out: int,
    p_out: int,
) -> Optional[EdgeList]:
    """Relabel to the canonical (k_out, p_out) index layout."""

    se_out, cd_out = _blocks(k_out, p_out)
    if len(se_nodes) != len(se_out) or len(cd_nodes) != len(cd_out):
        return None
    mapping: Dict[int, int] = {}
    for old, new in zip(sorted(se_nodes), se_out):
        mapping[old] = new
    for old, new in zip(sorted(cd_nodes), cd_out):
        mapping[old] = new
    edges = sorted(
        (min(mapping[a], mapping[b]), max(mapping[a], mapping[b]))
        for a, b in graph.edges
    )
    return tuple(edges)


def _core_state(
    edges: EdgeList, k: int, p: int, spec: NucleationSpec
) -> _State:
    """Build a bare-core ``_State`` for the rule checks."""

    from .molecular import _atoms_for_composition, _roles_for_composition
    from .molecular import _symbols_for_composition

    symbols = _symbols_for_composition(spec, k, 0)
    roles = _roles_for_composition(spec, k, 0)
    # ``p`` extra cations sit in the cation block; the bare-core helpers key on
    # (k, 0), so extend explicitly rather than reusing a (k, p) composition
    # that would also allocate ligand slots we do not have.
    cation = spec.core.cation
    symbols = list(symbols)
    roles = list(roles)
    for _ in range(p):
        symbols.append(cation)
        roles.append("precursor_center")
    atoms = _atoms_for_composition(tuple(symbols), tuple(roles))
    graph = nx.Graph()
    graph.add_nodes_from(range(len(atoms)))
    graph.add_edges_from(edges)
    return _State(atoms=atoms, graph=graph)


def _core_is_legal(
    edges: EdgeList, k: int, p: int, spec: NucleationSpec
) -> bool:
    """Whether a bare core passes the skeleton-level graph rules."""

    from .molecular import _skeleton_graph_violations

    state = _core_state(edges, k, p, spec)
    if not nx.is_connected(state.graph):
        return False
    return not _skeleton_graph_violations(state, spec)


def core_certificate(
    edges: EdgeList, k: int, p: int, spec: NucleationSpec
) -> Tuple[object, ...]:
    """Isomorphism-invariant key for a bare core.

    The canonical index layout fixes which *block* a node sits in, but not
    which node within the block, so two identical cores routinely come out
    with different edge tuples.  Deduplicating on the raw tuple therefore
    keeps isomorphic copies and makes a lineage generation look far larger
    than it is.
    """

    from .molecular import _graph_certificate

    return _graph_certificate(_core_state(edges, k, p, spec))


def shed_and_grow(
    parent_edges: EdgeList,
    *,
    k: int,
    p: int,
    p_out: int,
    spec: NucleationSpec,
    max_children: int = 20000,
    attach: str = "enumerate",
) -> List[EdgeList]:
    """Children at ``(k + 1, p_out)`` grown from one parent core at ``(k, p)``.

    ``p - p_out`` precursor cations are shed, then one core monomer (one
    cation + one anion) is attached.

    ``attach``:
      * ``enumerate`` — all legal subset attaches (survey / recall).
      * ``local`` — one surface event: new Se–Cd always bonded, new Se to
        exactly one existing Cd, new Cd to 0–2 existing Se.
    """

    shed = p - p_out
    if shed < 0:
        return []
    mode = str(attach or "enumerate").lower()
    if mode not in {"local", "enumerate"}:
        raise ValueError(f"attach must be 'local' or 'enumerate', got {attach!r}")
    se_ids, cd_ids = _blocks(k, p)
    max_se = int(spec.graph_rules.max_cn.get(spec.core.anion, 4))
    max_cd = int(spec.graph_rules.max_cn.get(spec.core.cation, 4))

    base = nx.Graph()
    base.add_nodes_from(list(se_ids) + list(cd_ids))
    base.add_edges_from(parent_edges)

    # Precursor cations are the tail of the cation block.  Shedding a *core*
    # cation would change the core formula, not the precursor count.
    precursor_cd = list(cd_ids)[k:]
    children: Dict[EdgeList, None] = {}
    seen_certs: Set[Tuple[object, ...]] = set()

    shed_sets: Iterable[Tuple[int, ...]]
    shed_sets = combinations(precursor_cd, shed) if shed else [()]
    for drop in shed_sets:
        stripped = base.copy()
        stripped.remove_nodes_from(drop)
        kept_se = [n for n in stripped if n in se_ids]
        kept_cd = [n for n in stripped if n in cd_ids]
        if not kept_se or not kept_cd:
            continue

        new_se = max(list(se_ids) + list(cd_ids)) + 1
        new_cd = new_se + 1
        # Room left on each existing site once the monomer attaches.
        open_cd = [n for n in kept_cd if stripped.degree(n) < max_cd]
        open_se = [n for n in kept_se if stripped.degree(n) < max_se]

        if mode == "local":
            attach_iter = _local_attach_patterns(
                open_cd, open_se, max_cd=max_cd, max_se=max_se
            )
        else:
            attach_iter = _enumerate_attach_patterns(
                open_cd, open_se, max_cd=max_cd, max_se=max_se
            )

        for bond_monomer, se_partners, cd_partners in attach_iter:
            child = stripped.copy()
            child.add_node(new_se)
            child.add_node(new_cd)
            if bond_monomer:
                child.add_edge(new_se, new_cd)
            for host in se_partners:
                child.add_edge(new_se, host)
            for anion in cd_partners:
                child.add_edge(new_cd, anion)
            if not nx.is_connected(child):
                continue
            relabelled = _canonical_relabel(
                child,
                kept_se + [new_se],
                kept_cd + [new_cd],
                k + 1,
                p_out,
            )
            if relabelled is None:
                continue
            if relabelled in children:
                continue
            if not _core_is_legal(
                relabelled, k + 1, p_out, spec
            ):
                continue
            cert = core_certificate(
                relabelled, k + 1, p_out, spec
            )
            if cert in seen_certs:
                continue
            seen_certs.add(cert)
            children[relabelled] = None
            if len(children) >= max_children:
                return list(children)
    return list(children)


def _local_attach_patterns(
    open_cd: Sequence[int],
    open_se: Sequence[int],
    *,
    max_cd: int,
    max_se: int,
) -> Iterable[Tuple[bool, Tuple[int, ...], Tuple[int, ...]]]:
    """One CdSe addition at a surface site (always bond the new pair)."""

    del max_se  # new Se uses one existing host; CN check is on the host
    if not open_cd:
        return
    # new Cd already bonded to new Se, so at most max_cd-1 extra Se links
    cd_link_max = min(2, max(0, int(max_cd) - 1), len(open_se))
    for host in open_cd:
        for n_cd_links in range(0, cd_link_max + 1):
            for cd_partners in combinations(open_se, n_cd_links):
                yield True, (int(host),), tuple(cd_partners)


def _enumerate_attach_patterns(
    open_cd: Sequence[int],
    open_se: Sequence[int],
    *,
    max_cd: int,
    max_se: int,
) -> Iterable[Tuple[bool, Tuple[int, ...], Tuple[int, ...]]]:
    """All legal subset attaches (historical / survey path)."""

    for bond_monomer in (True, False):
        se_room = max_se - (1 if bond_monomer else 0)
        cd_room = max_cd - (1 if bond_monomer else 0)
        for n_se_links in range(0, min(se_room, len(open_cd)) + 1):
            for se_partners in combinations(open_cd, n_se_links):
                for n_cd_links in range(0, min(cd_room, len(open_se)) + 1):
                    for cd_partners in combinations(open_se, n_cd_links):
                        if not bond_monomer and not (
                            se_partners and cd_partners
                        ):
                            continue
                        if not se_partners and not cd_partners:
                            continue
                        yield bond_monomer, tuple(se_partners), tuple(cd_partners)


def grow_generation(
    parents: Sequence[EdgeList],
    *,
    k: int,
    p: int,
    p_out: int,
    spec: NucleationSpec,
    max_children: int = 20000,
    attach: str = "enumerate",
) -> List[EdgeList]:
    """Deduplicated children at ``(k + 1, p_out)`` from many parents.

    Dedup is on the canonical edge tuple, which is exact for cores because the
    index layout is itself canonical -- two parents that lead to the same child
    produce the same tuple.
    """

    seen: Dict[EdgeList, None] = {}
    certs: Set[Tuple[object, ...]] = set()
    for parent in parents:
        for child in shed_and_grow(
            parent,
            k=k,
            p=p,
            p_out=p_out,
            spec=spec,
            max_children=max_children,
            attach=attach,
        ):
            cert = core_certificate(child, k + 1, p_out, spec)
            if cert in certs:
                continue
            certs.add(cert)
            seen[child] = None
            if len(seen) >= max_children:
                return list(seen)
    return list(seen)


def cores_from_run(run_dir: str, k: int, p: int) -> List[EdgeList]:
    """Read the distinct cation-anion cores out of a finished bin.

    Uses ``<run>/k###/p###/motif_trials.csv``, whose ``source_edges`` column is
    the graph as enumerated.  The per-bin file is the reliable one: the
    top-level ``index.csv`` is rewritten per bin during a multi-bin run.
    """

    import csv
    from pathlib import Path

    se_ids, cd_ids = _blocks(k, p)
    inorganic = set(se_ids) | set(cd_ids)
    path = Path(run_dir) / f"k{k:03d}" / f"p{p:03d}" / "motif_trials.csv"
    if not path.exists():
        return []
    out: Dict[EdgeList, None] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("source_edges") or "").strip()
            if not raw:
                continue
            edges = []
            for token in raw.split("|"):
                left, right = (int(x) for x in token.split("-"))
                if left in inorganic and right in inorganic:
                    edges.append((min(left, right), max(left, right)))
            if edges:
                out[tuple(sorted(set(edges)))] = None
    return list(out)


__all__ = [
    "core_certificate",
    "shed_and_grow",
    "grow_generation",
    "cores_from_run",
]
