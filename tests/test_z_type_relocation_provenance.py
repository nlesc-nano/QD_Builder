import numpy as np

from builder.z_type_displacement_posttreat import _lookup_original_provenance


def test_lookup_original_provenance_matches_position_and_symbol():
    entries = [
        {"position": [0.0, 0.0, 0.0], "symbol": "Cl", "mechanism": "charge_balance"},
        {"position": [5.0, 0.0, 0.0], "symbol": "Cl", "mechanism": "polar_surface_reconstruction"},
    ]
    # Returns the whole matched entry (not just its mechanism string), so
    # the caller can also carry replaced_element/neutrality_redistributed
    # forward.
    found = _lookup_original_provenance(np.array([0.0, 0.0, 0.0]), "Cl", entries)
    assert found == entries[0]


def test_lookup_original_provenance_returns_none_when_no_match():
    entries = [
        {"position": [0.0, 0.0, 0.0], "symbol": "Cl", "mechanism": "charge_balance"},
    ]
    # Wrong symbol at the same position -- not a match.
    assert _lookup_original_provenance(np.array([0.0, 0.0, 0.0]), "Br", entries) is None
    # Right symbol, far-away position -- not a match.
    assert _lookup_original_provenance(np.array([50.0, 50.0, 50.0]), "Cl", entries) is None
    # No entries at all (plain, non-lattice-substitution ligand).
    assert _lookup_original_provenance(np.array([0.0, 0.0, 0.0]), "Cl", None) is None
