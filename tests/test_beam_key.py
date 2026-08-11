"""Equivalence of Cython and pure-Python beam-key kernels."""

from __future__ import annotations

from builder.nucleation import _beam_key_fallback as py


def _sample_args():
    cn = [2, 1, 0, 2]
    n_bridge = [1, 0, 0, 1]
    n_term = [0, 1, 0, 0]
    bridges = [(10, 11), (12, 13)]
    mu3 = [(10, 11, 12)]
    terminals_on = [1, 2]
    remaining_cl = 3
    slot_maps = [
        (0, 1, 2, 3),
        (1, 0, 2, 3),
        (0, 1, 3, 2),
    ]
    slot_inverses = []
    for mapping in slot_maps:
        inv = [0] * 4
        for original, mapped in enumerate(mapping):
            inv[mapped] = original
        slot_inverses.append(tuple(inv))
    # dense host relabel for host ids 10..13
    host_relabels = []
    cd_list = [10, 11, 12, 13]
    for mapping in slot_maps:
        dense = [-1] * 14
        for i, host in enumerate(cd_list):
            dense[host] = cd_list[mapping[i]]
        host_relabels.append(tuple(dense))
    return (
        cn,
        n_bridge,
        n_term,
        bridges,
        mu3,
        terminals_on,
        remaining_cl,
        slot_maps,
        slot_inverses,
        host_relabels,
    )


def test_fallback_identity_and_pair():
    cn, nb, nt, br, mu, term, rem, *_ = _sample_args()
    key = py.identity_state_key(cn, nb, nt, br, mu, term, rem)
    assert key[-1] == rem
    assert key[0] == tuple(cn)
    assert py.pair_bridge_count(br, mu, 10, 11) == 2  # one mu2 + one mu3 edge
    assert py.is_cython() is False


def test_fallback_canonical_is_min_orbit():
    args = _sample_args()
    key = py.canonical_state_key(*args)
    # identity orbit member must be >= the canonical key under lex order
    id_key = py.identity_state_key(*args[:7])
    assert key <= id_key


def test_cython_matches_fallback_when_built():
    try:
        from builder.nucleation import _beam_key as cy
    except ImportError:
        return
    if not getattr(cy, "is_cython", lambda: False)():
        return
    args = _sample_args()
    assert cy.identity_state_key(*args[:7]) == py.identity_state_key(*args[:7])
    assert cy.pair_bridge_count(args[3], args[4], 10, 11) == py.pair_bridge_count(
        args[3], args[4], 10, 11
    )
    assert cy.canonical_state_key(*args) == py.canonical_state_key(*args)
    assert cy.is_cython() is True
