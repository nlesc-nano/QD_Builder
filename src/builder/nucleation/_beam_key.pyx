# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython hot path for bridge-first beam state keys.

Profile note (see molecular_bridge_first): at large |Aut| the pure-Python
``state_key`` can dominate decoration wall time.  This module speeds:

  * identity keys (asymmetric skeletons)
  * canonical keys under host automorphisms
  * pair_bridge_count

Build with::

    pip install -e ".[speed]"
    # or
    python setup.py build_ext --inplace

If the extension is not built, ``molecular_bridge_first`` keeps the pure-Python
fallback in ``_beam_key_fallback``.
"""

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple identity_state_key(
    list cn,
    list n_bridge,
    list n_term,
    list bridges,
    list mu3,
    list terminals_on,
    int remaining_cl,
):
    """Fast identity key (no automorphism orbit)."""

    return (
        tuple(cn),
        tuple(n_bridge),
        tuple(n_term),
        tuple(sorted(bridges)),
        tuple(sorted(mu3)),
        tuple(sorted(terminals_on)),
        remaining_cl,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int pair_bridge_count(
    list bridges,
    list mu3,
    int a,
    int b,
):
    """How many mu2/mu3 entries cover host pair (a,b)."""

    cdef int lo, hi, count, i, n, t0, t1, t2
    cdef object pr, tri

    if a <= b:
        lo = a
        hi = b
    else:
        lo = b
        hi = a

    count = 0
    n = len(bridges)
    for i in range(n):
        pr = bridges[i]
        if <int>pr[0] == lo and <int>pr[1] == hi:
            count += 1

    n = len(mu3)
    for i in range(n):
        tri = mu3[i]
        t0 = <int>tri[0]
        t1 = <int>tri[1]
        t2 = <int>tri[2]
        if (t0 == lo or t1 == lo or t2 == lo) and (
            t0 == hi or t1 == hi or t2 == hi
        ):
            count += 1
    return count


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _sort3_int(int *x, int *y, int *z) noexcept nogil:
    cdef int t
    if x[0] > y[0]:
        t = x[0]
        x[0] = y[0]
        y[0] = t
    if y[0] > z[0]:
        t = y[0]
        y[0] = z[0]
        z[0] = t
    if x[0] > y[0]:
        t = x[0]
        x[0] = y[0]
        y[0] = t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple canonical_state_key(
    list cn,
    list n_bridge,
    list n_term,
    list bridges,
    list mu3,
    list terminals_on,
    int remaining_cl,
    list slot_maps,
    list slot_inverses,
    list host_relabels,
):
    """Canonicalize a partial state under skeleton host automorphisms.

    Arguments mirror the pure-Python loop in ``molecular_bridge_first.state_key``:

    * ``slot_maps``: list of sequences length n_cd (slot -> slot)
    * ``slot_inverses``: list of sequences (inverse maps)
    * ``host_relabels``: either
        - list of dict host_id -> relabeled host_id, or
        - list of dense sequences where index=host_id and value=relabeled
          (preferred; denser, no hash lookups)
    """

    cdef int n_aut = len(slot_maps)
    cdef int n_cd = len(cn)
    cdef int a, i, n_b, n_m, n_t, j
    cdef int ra, rb, rx, ry, rz
    cdef int h0, h1, h2, slot
    cdef list bridge_list, mu3_list, term_list
    cdef list cn_t, nb_t, nt_t
    cdef object mapping, inverse, relabel
    cdef object pr, tri, best
    cdef object bridges_t, mu3_t, terminals_t, candidate
    cdef bint relabel_is_dict

    if n_aut <= 0:
        return identity_state_key(
            cn, n_bridge, n_term, bridges, mu3, terminals_on, remaining_cl
        )

    best = None
    n_b = len(bridges)
    n_m = len(mu3)
    n_t = len(terminals_on)

    # Detect dense array vs dict once (all entries share the same shape).
    relabel_is_dict = isinstance(host_relabels[0], dict)

    for a in range(n_aut):
        mapping = slot_maps[a]
        inverse = slot_inverses[a]
        relabel = host_relabels[a]

        bridge_list = []
        for i in range(n_b):
            pr = bridges[i]
            h0 = <int>pr[0]
            h1 = <int>pr[1]
            if relabel_is_dict:
                ra = <int>relabel[h0]
                rb = <int>relabel[h1]
            else:
                ra = <int>relabel[h0]
                rb = <int>relabel[h1]
            if ra <= rb:
                bridge_list.append((ra, rb))
            else:
                bridge_list.append((rb, ra))
        bridge_list.sort()
        bridges_t = tuple(bridge_list)

        mu3_list = []
        for i in range(n_m):
            tri = mu3[i]
            h0 = <int>tri[0]
            h1 = <int>tri[1]
            h2 = <int>tri[2]
            if relabel_is_dict:
                rx = <int>relabel[h0]
                ry = <int>relabel[h1]
                rz = <int>relabel[h2]
            else:
                rx = <int>relabel[h0]
                ry = <int>relabel[h1]
                rz = <int>relabel[h2]
            _sort3_int(&rx, &ry, &rz)
            mu3_list.append((rx, ry, rz))
        mu3_list.sort()
        mu3_t = tuple(mu3_list)

        # terminals: explicit loop (no genexp — Cython forbids closures in cpdef)
        term_list = []
        for j in range(n_t):
            slot = <int>terminals_on[j]
            term_list.append(<int>mapping[slot])
        term_list.sort()
        terminals_t = tuple(term_list)

        cn_t = []
        nb_t = []
        nt_t = []
        for i in range(n_cd):
            j = <int>inverse[i]
            cn_t.append(<int>cn[j])
            nb_t.append(<int>n_bridge[j])
            nt_t.append(<int>n_term[j])

        candidate = (
            tuple(cn_t),
            tuple(nb_t),
            tuple(nt_t),
            bridges_t,
            mu3_t,
            terminals_t,
            remaining_cl,
        )
        if best is None or candidate < best:
            best = candidate

    return best


cpdef bint is_cython():
    return True
