"""
config.py::_parse_facets -- an explicit `termination: stoichiometric` choice
must survive parsing as a real, distinguishable value, not collapse into the
same `None` as "termination was never mentioned at all". See
surface_metadata.py's `facet_terminations`/`facets[].termination`, which
both depend on this distinction being real.
"""
import textwrap


def _write_yaml(tmp_path, body: str):
    path = tmp_path / "config.yml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_explicit_stoichiometric_is_distinct_from_unconfigured(tmp_path):
    from builder.config import parse_yaml_config

    yaml_path = _write_yaml(tmp_path, """\
        facets:
          - hkl: "100"
            scope: facet
            termination: stoichiometric
            gamma: 1.0
          - hkl: "111"
            scope: facet
            gamma: 1.0
        passivation:
          ligand: Cl
        charges:
          In: 3
          P: -3
          Cl: -1
    """)
    cfg = parse_yaml_config(str(yaml_path))
    by_hkl = {(f.h, f.k, f.l): f for f in cfg.seeds}

    # Explicitly declared -- a real choice, not a guess.
    assert by_hkl[(1, 0, 0)].termination == "stoichiometric"
    # Never mentioned at all -- genuinely unknown, distinct from the above.
    assert by_hkl[(1, 1, 1)].termination is None


def test_stoichiometric_aliases_also_map_to_the_real_value(tmp_path):
    from builder.config import parse_yaml_config

    yaml_path = _write_yaml(tmp_path, """\
        facets:
          - hkl: "100"
            scope: facet
            termination: "Stoichiometric"
            gamma: 1.0
        passivation:
          ligand: Cl
        charges:
          In: 3
          P: -3
          Cl: -1
    """)
    cfg = parse_yaml_config(str(yaml_path))
    assert cfg.seeds[0].termination == "stoichiometric"


def test_empty_or_null_termination_stays_genuinely_unspecified(tmp_path):
    # An empty/none/null string is the key being present but blank -- not
    # the same deliberate statement as the word "stoichiometric" -- stays
    # None, same as never mentioning termination at all.
    from builder.config import parse_yaml_config

    yaml_path = _write_yaml(tmp_path, """\
        facets:
          - hkl: "100"
            scope: facet
            termination: none
            gamma: 1.0
        passivation:
          ligand: Cl
        charges:
          In: 3
          P: -3
          Cl: -1
    """)
    cfg = parse_yaml_config(str(yaml_path))
    assert cfg.seeds[0].termination is None


def test_pair_opposites_still_auto_adds_stoichiometric_facets_opposite(tmp_path):
    # pair_opposites must keep auto-expanding a facet's opposite direction
    # for a "stoichiometric"-tagged facet exactly as it already does for an
    # unconfigured one -- "stoichiometric" is not a directional preference
    # the way cation_rich/anion_rich are, so it must not disable auto-pairing.
    from builder.config import parse_yaml_config

    yaml_path = _write_yaml(tmp_path, """\
        facets:
          - hkl: "100"
            scope: family
            termination: stoichiometric
            gamma: 1.0
        passivation:
          ligand: Cl
        charges:
          In: 3
          P: -3
          Cl: -1
    """)
    cfg = parse_yaml_config(str(yaml_path))
    by_hkl = {(f.h, f.k, f.l): f for f in cfg.seeds}
    assert (1, 0, 0) in by_hkl
    assert (-1, 0, 0) in by_hkl, "opposite direction must still be auto-added"
    # The auto-added opposite inherits the same real termination value.
    assert by_hkl[(-1, 0, 0)].termination == "stoichiometric"


def test_pair_opposites_skips_auto_add_for_a_real_directional_preference(tmp_path):
    # Unchanged existing behavior: cation_rich/anion_rich IS a directional
    # preference, so auto-pairing is correctly skipped for it (the caller
    # must list both signed directions explicitly in that case).
    from builder.config import parse_yaml_config

    yaml_path = _write_yaml(tmp_path, """\
        facets:
          - hkl: "100"
            scope: family
            termination: cation_rich
            gamma: 1.0
        passivation:
          ligand: Cl
        charges:
          In: 3
          P: -3
          Cl: -1
    """)
    cfg = parse_yaml_config(str(yaml_path))
    by_hkl = {(f.h, f.k, f.l): f for f in cfg.seeds}
    assert (1, 0, 0) in by_hkl
    assert (-1, 0, 0) not in by_hkl
