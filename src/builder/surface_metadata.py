# nanocrystal_builder/surface_metadata.py
"""
Reusable Python API for the surface metadata JSON export.

This assembles the document: a description of every atom in the QD (`qd_atoms`,
surface- or core-classified) via QD_Builder's existing facet/Wulff analysis,
the exact facet definitions used to build the QD, and the current ligand
passivation state.

Describes what exists geometrically/chemically -- every atom's
crystallographic environment, coordination, surface topology, ligand
fragments, and a real (never guessed) ligand -> host attachment mapping,
each tagged with the QD_Builder mechanism that produced it (see README
"Surface Metadata (JSON)" for the full field-by-field reference).

Callers supply already-computed (symbols, pts, planes, facets, charges,
surf_tol) -- exactly what facet_atom_report/surface_report already take --
so this module has no opinion on how those were obtained (a live build in
main.py, or a facet re-detection pass against a finished XYZ, as
QDSpaceWebApp's backend does for its sibling z-type/post-treatment option
endpoints).

Nothing here is wired into the CLI's default output path: this is purely an
on-demand API for the web app's metadata-download button and for other
external scripts.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .analysis import PairCuts, _facet_family_label, surface_atom_records
from .nc_types import Facet, Plane


def _find_atom_at_position(
    pts: NDArray[np.float64],
    symbols: List[str],
    position: Optional[List[float]],
    expected_symbol: Optional[str],
    tol: float,
) -> Optional[int]:
    if position is None:
        return None
    d = np.linalg.norm(pts - np.asarray(position, float), axis=1)
    idx = int(np.argmin(d))
    if float(d[idx]) > tol:
        return None
    if expected_symbol is not None and symbols[idx] != expected_symbol:
        return None
    return idx


def _detect_global_offset(
    pts: NDArray[np.float64],
    symbols: List[str],
    entries: List[Dict[str, Any]],
    *,
    round_decimals: int = 3,
) -> Optional[np.ndarray]:
    """
    Detect a uniform global translation applied to the structure since
    `entries`' positions were captured -- e.g. `center_coords` (recenter on
    the centroid), applied once near the end of the build pipeline, well
    after polar facet reconstruction runs. A rigid translation shifts every
    atom by the same vector, so (final position of the true match) -
    (captured position) is identical across every entry; any other,
    incorrect pairing gives an essentially arbitrary vector. The most common
    delta (by a wide margin, for more than a couple of entries) is the true
    offset. Returns the vector to ADD to a captured position to match the
    current structure, or None if it can't be determined (no entries, or no
    candidate atom of the expected symbol at all).
    """
    votes: Dict[tuple, int] = {}
    for entry in entries:
        pos = entry.get("position")
        sym = entry.get("symbol")
        if pos is None or sym is None:
            continue
        pos = np.asarray(pos, float)
        for ci in (i for i, s in enumerate(symbols) if s == sym):
            key = tuple(np.round(pts[ci] - pos, round_decimals))
            votes[key] = votes.get(key, 0) + 1
    if not votes:
        return None
    best_key = max(votes, key=votes.get)
    return np.asarray(best_key, float)


def _resolve_positional_entries(
    symbols: List[str],
    pts: NDArray[np.float64],
    entries: Optional[List[Dict[str, Any]]],
    *,
    tol: float = 1e-3,
) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Resolve a list of position-captured entries (each carrying "position"/
    "symbol" keys) against the current atom array, by position rather than
    index -- shared by every caller that needs this (lattice-substitution
    anchors, alloying's qd_atom provenance tag). A single global offset is
    detected once from the whole entry list (see _detect_global_offset) to
    correct for any uniform recentering translation applied since capture,
    then each entry's own position is exact-matched. An entry whose position
    no longer matches an atom of the expected symbol (removed, or changed,
    by something downstream) is dropped entirely, never guessed.

    Returns (resolved_index, entry) pairs, one per entry that still matches.
    """
    if not entries:
        return []
    pts = np.asarray(pts, float)
    offset = _detect_global_offset(pts, symbols, entries)
    if offset is None:
        return []
    out: List[Tuple[int, Dict[str, Any]]] = []
    for entry in entries:
        pos = entry.get("position")
        anchor_pos = (np.asarray(pos, float) + offset).tolist() if pos is not None else None
        idx = _find_atom_at_position(pts, symbols, anchor_pos, entry.get("symbol"), tol)
        if idx is not None:
            out.append((idx, entry))
    return out


def _resolve_atom_provenance_by_position(
    symbols: List[str],
    pts: NDArray[np.float64],
    entries: Optional[List[Dict[str, Any]]],
    *,
    tol: float = 1e-3,
) -> Dict[int, str]:
    """
    Resolve position-captured `atom_provenance` entries (each {"position":
    [...], "symbol": str, "provenance": str}, e.g. one per alloying-
    replaced cation) into the index->provenance map `surface_atom_records`
    expects. Position-based for the same reason as everywhere else in this
    module: immune to any atom-count-changing step (z-type, further charge
    balancing, ...) that runs between capture and this call.
    """
    return {
        idx: entry.get("provenance")
        for idx, entry in _resolve_positional_entries(symbols, pts, entries, tol=tol)
    }


def _resolve_lattice_substitutions(
    symbols: List[str],
    pts: NDArray[np.float64],
    entries: Optional[List[Dict[str, Any]]],
    *,
    tol: float = 1e-3,
) -> List[dict]:
    """
    Resolve position-captured lattice-substitution ledger entries (see
    builder.facet_reconstruction.reconstruct_polar_facets -- a native anion
    converted in place to the ligand species) against the current atom
    array. Positions, not indices, because several pipeline stages can run
    between capture and this call (compaction, reordering, and a uniform
    recentering translation) with no index remap for this purpose; a global
    offset is detected and corrected for (see _detect_global_offset), then
    positions are exact-matched per atom.

    Returns {"anchor_atom_index_0based": int, "host_atom_indices_0based":
    [int, ...], "replaced_symbol": Optional[str], "mechanism": Optional[str]}
    per entry. `replaced_symbol` is the native anion species this lattice
    site originally held (see builder.facet_reconstruction), passed through
    unresolved -- it's a plain element symbol, not a position, so there's
    nothing to match against the current structure. `mechanism` identifies
    which real build-time process produced this entry (see
    _build_lattice_substitution_records), also passed through unresolved.
    An entry whose anchor position no longer matches an atom of the
    expected symbol (removed, or changed, by something downstream) is
    dropped entirely; a host position that doesn't resolve is dropped from
    that entry's host list only -- checked against *that host's own*
    expected symbol (`host_symbols`, per-host), not one symbol shared
    across every host of the anchor: a mixed host set (e.g. one native
    cation + one alloy-replaced cation) must let each host resolve
    independently, or a surviving host gets wrongly dropped just because a
    sibling host's species differs from it.
    """
    if not entries:
        return []
    pts = np.asarray(pts, float)
    offset = _detect_global_offset(pts, symbols, entries)
    if offset is None:
        return []
    resolved = []
    for idx, entry in _resolve_positional_entries(symbols, pts, entries, tol=tol):
        host_symbols = entry.get("host_symbols") or []
        host_indices = [
            hi for hp, hs in zip(entry.get("host_positions") or [], host_symbols)
            if (hi := _find_atom_at_position(
                pts, symbols, (np.asarray(hp, float) + offset).tolist(), hs, tol,
            )) is not None
        ]
        resolved.append({
            "anchor_atom_index_0based": idx,
            "host_atom_indices_0based": host_indices,
            "replaced_symbol": entry.get("replaced_symbol"),
            "mechanism": entry.get("mechanism"),
        })
    return resolved


def resolve_prior_ligand_provenance(
    symbols: List[str],
    pts: NDArray[np.float64],
    prior_symbols: List[str],
    prior_pts: NDArray[np.float64],
    prior_ligands: Optional[List[Dict[str, Any]]],
    *,
    tol: float = 1e-3,
    alloying_provenance: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Re-verify a prior surface-metadata document's `ligands` against the
    *current* atom array, by position (same mechanism as
    `_resolve_lattice_substitutions`/`_detect_global_offset`) rather than by
    index -- this is for QDSpaceWebApp's repassivation flow, which can
    receive a structure that already carries ligands a different call (the
    original build, or an earlier repassivation pass) placed, with no ledger
    of its own for them. Chaining the prior document this way, re-verified by
    position, is safe precisely because nothing is trusted blindly: an atom
    that moved beyond `tol`, or was removed since, simply fails to resolve
    and is dropped -- same graceful-degradation convention as everywhere else
    in this module.

    `prior_symbols`/`prior_pts`: the exact array the prior document's
    `ligands` entries index into (i.e. the XYZ it was built to accompany) --
    NOT this call's own `symbols`/`pts`. A ligand's host is recorded
    directly as a raw `host_atom_indices_0based` index into that array, so
    resolving its prior position is a plain lookup, no `qd_atoms`
    cross-reference needed.

    Returns a list in the same shape `build_surface_metadata_document`'s own
    `ligand_provenance` parameter expects (so the caller can simply
    concatenate it in), with each entry's original `provenance` value carried
    through unchanged -- a prior `"x_type"` ligand that's still genuinely
    there is still `"x_type"`, not re-labeled. An entry is only emitted when
    *every* one of its atoms (and every host that was resolved in the prior
    document) still resolves; a partially-vanished ligand is dropped
    entirely rather than reported with some atoms missing.

    `alloying_provenance`: optional position-captured entries (see
    build_surface_metadata_document's `atom_provenance` -- same list, reused
    as-is), one per cation this call's own alloying pass replaced in place.
    A host whose recorded prior species no longer matches at its position is
    real information most of the time (the atom moved or was removed) --
    but if alloying is exactly why it changed, the bond is still real, just
    under a new symbol: that host is kept (resolved against its *new* real
    species) rather than dropped.
    """
    if not prior_ligands:
        return []
    prior_pts = np.asarray(prior_pts, float)
    pts = np.asarray(pts, float)

    offset_entries = [
        {"position": prior_pts[i].tolist(), "symbol": prior_symbols[i]}
        for lig in prior_ligands
        for i in (lig.get("atom_indices_0based") or [])
    ]
    offset = _detect_global_offset(pts, symbols, offset_entries)
    if offset is None:
        return []

    def _resolve(old_idx: int) -> Optional[int]:
        pos = (prior_pts[old_idx] + offset).tolist()
        return _find_atom_at_position(pts, symbols, pos, prior_symbols[old_idx], tol)

    resolved: List[Dict[str, Any]] = []
    for lig in prior_ligands:
        old_atom_indices = lig.get("atom_indices_0based") or []
        if not old_atom_indices:
            continue
        new_by_old = {i: _resolve(i) for i in old_atom_indices}
        if any(v is None for v in new_by_old.values()):
            continue  # at least one atom of this ligand no longer matches -> drop entirely

        old_anchor = lig.get("anchor_atom_index_0based")
        if old_anchor is None or old_anchor not in new_by_old:
            # The prior document itself never resolved this ligand's real
            # provenance (e.g. it was already an unresolved/malformed
            # placeholder there) -- nothing real to chain forward. Emitting
            # an entry with a null anchor here would make
            # _assign_ligands_from_provenance flag it as *malformed*
            # (a genuine-bug signal), not "still unresolved" -- so this
            # atom is simply left uncovered, same as if no prior entry
            # existed for it at all.
            continue
        new_anchor = new_by_old[old_anchor]

        old_host_indices = lig.get("host_atom_indices_0based") or []
        host_elements = lig.get("host_elements") or []
        new_hosts = []
        for old_h, el in zip(old_host_indices, host_elements):
            host_pos = (prior_pts[old_h] + offset).tolist()
            nh = _find_atom_at_position(pts, symbols, host_pos, el, tol)
            if nh is None and alloying_provenance:
                new_sym = next(
                    (
                        entry.get("symbol") for entry in alloying_provenance
                        if entry.get("position") is not None
                        and float(np.linalg.norm(np.asarray(entry["position"], float) - np.asarray(host_pos))) <= tol
                    ),
                    None,
                )
                if new_sym is not None:
                    nh = _find_atom_at_position(pts, symbols, host_pos, new_sym, tol)
            if nh is not None:
                new_hosts.append(nh)

        resolved.append({
            "type": lig.get("type"),
            "anchor_atom_index_0based": new_anchor,
            "atom_indices_0based": sorted(new_by_old.values()),
            "bound_host_atom_indices_0based": new_hosts,
            "provenance": lig.get("provenance"),
        })
    return resolved


def _build_lattice_substitution_records(
    symbols: List[str],
    pts: NDArray[np.float64],
    resolved: List[dict],
    qd_atoms_by_index: Dict[int, dict],
) -> List[dict]:
    """
    Build `ligands` entries for resolved lattice-substitution atoms. Real,
    build-time provenance (the true bonded cations, known from the
    reconstruction's own bond graph) -- same uniform field set as any other
    ligand record (see _assign_ligands_from_provenance). Never emitted as a
    parallel `qd_atoms` entry, since it's still a ligand, not a native atom.
    A host that doesn't resolve to a known qd_atom is dropped from the
    host-list fields, same convention as real `ligand_provenance` hosts --
    `qd_atoms_by_index` covers every atom (native and ligand, surface and
    core, see surface_atom_records), so a former lattice bond to a
    now-interior cation is kept, not silently dropped just because that
    cation itself isn't facet-exposed; the same dict also supplies the
    anchor's own geometric record for its facet label (see
    _facet_label_fields_from_own_position).

    Three real mechanisms feed `resolved`'s entries, told apart by
    `mechanism`: `reconstruct_polar_facets`'s *optional* polar-{111}
    reconstruction post-treatment (`"polar_surface_reconstruction"`),
    `prepass_surface_cleanup`'s *always-on* baseline CN==2 anion-swap step
    (`"prepass_cleanup"`) that runs as part of ordinary passivation on
    every build regardless of that post-treatment toggle, and z-type
    displacement's orphan-ligand relocation (`"z_type_relocated"`, or
    `"{original_provenance}_z_type_relocated"` when the ligand's provenance
    before being relocated is itself known -- see
    z_type_displacement_posttreat._relocate_orphan_ligands). `mechanism` is
    passed straight through as `provenance`, not coerced into a fixed set,
    so new mechanism strings need no change here. Only the first two convert
    a native anion to the ligand species *in place*, so only those get
    `"swap": True` -- a relocated ligand has moved to a freshly chosen
    exterior site and no longer occupies a former native lattice position,
    even if it originally did.
    """
    records = []
    for entry in resolved:
        anchor_idx = entry["anchor_atom_index_0based"]
        host_recs = [
            qd_atoms_by_index[h]
            for h in entry["host_atom_indices_0based"]
            if h in qd_atoms_by_index
        ]
        host_recs.sort(key=lambda r: float(np.linalg.norm(pts[anchor_idx] - r["position"])))
        mechanism = entry.get("mechanism")
        record = {
            "type": symbols[anchor_idx],
            "anchor_element": symbols[anchor_idx],
            "anchor_atom_index_0based": anchor_idx,
            "atom_indices_0based": [anchor_idx],
            "host_atom_indices_0based": [r["atom_index_0based"] for r in host_recs],
            "host_elements": [r["element"] for r in host_recs],
            "anchor_host_distances": [
                float(np.linalg.norm(pts[anchor_idx] - r["position"])) for r in host_recs
            ],
            "provenance": mechanism,
            "binding_site": _binding_site_from_host_count(len(host_recs)),
        }
        if mechanism in ("prepass_cleanup", "polar_surface_reconstruction"):
            record["swap"] = True
        record.update(_facet_label_fields_from_own_position(qd_atoms_by_index.get(anchor_idx)))
        records.append(record)
    return records


def _facet_label_fields_from_own_position(own_record: Optional[dict]) -> Dict[str, list]:
    """
    A ligand's `facet_ids`/`facet_hkls`/`facet_families`, recomputed by the
    same standard geometric procedure as every `qd_atoms` entry
    (`surface_atom_records`/`_facet_memberships`) against the ligand's own
    current position -- not inherited from its host(s). A host's own
    `qd_atoms` record can itself go stale (removed, replaced, or the ligand
    later re-hosted elsewhere by a further repassivation step) even when
    the ligand's own position hasn't moved at all; the ligand's position is
    where the QD's growth front genuinely sits, so that position's own,
    always-current membership is the accurate classification.

    No `facet_terminations`: termination (`cation_rich`/`anion_rich`/
    `stoichiometric`) is a property of which native species occupies a
    lattice site under a given Wulff seed configuration -- it doesn't apply
    to a ligand sitting in open space beyond the lattice, regardless of
    which facet it's geometrically near.

    `own_record` is `None` (or has no `facet_ids` key) for a ligand that
    isn't within `surf_tol` of any plane (geometrically "core") -- reports
    all-empty, same convention as a core `qd_atoms` record.
    """
    if not own_record:
        return {"facet_ids": [], "facet_hkls": [], "facet_families": []}
    return {
        "facet_ids": list(own_record.get("facet_ids") or []),
        "facet_hkls": [list(h) for h in (own_record.get("facet_hkls") or [])],
        "facet_families": list(own_record.get("facet_families") or []),
    }


def _binding_site_from_host_count(n_hosts: int) -> Optional[str]:
    """
    Standard coordination-chemistry μ-notation for how many real host
    cations a ligand is bonded to -- a pure function of host count, same
    convention regardless of which mechanism placed the ligand.
    """
    if n_hosts == 1:
        return "μ1-terminal"
    if n_hosts == 2:
        return "μ2-bridging"
    if n_hosts == 3:
        return "μ3-face-capping"
    if n_hosts >= 4:
        return "μ4-tetrahedro-capping"
    return None


def _assign_ligands_from_provenance(
    symbols: List[str],
    pts: NDArray[np.float64],
    ligand_indices: List[int],
    qd_atoms_by_index: Dict[int, dict],
    *,
    ligand_provenance: Optional[List[Dict[str, Any]]],
) -> tuple[List[dict], List[int], set]:
    """
    Build the `ligands` list from real, build-time-captured provenance only
    -- no geometric fallback. Every QD_Builder placement mechanism already
    knows its real host(s) at placement time (see
    builder.main._run_passivation_and_write_outputs and
    backend.api_builder._run_repassivation_posttreatment for where each
    mechanism's ledger is captured and tagged with its own `provenance`
    mechanism name). See build_surface_metadata_document's `ligand_provenance`
    param.

    A lattice-substitution ligand (`"polar_surface_reconstruction"` or
    `"prepass_cleanup"` provenance) chained forward from a prior document
    (see resolve_prior_ligand_provenance) arrives here via this generic
    path, not _build_lattice_substitution_records, but gets the exact same
    uniform field set as every other ligand -- see
    _facet_label_fields_from_own_position and the `swap` tag below.

    `qd_atoms_by_index` covers every atom (native and ligand, see
    surface_atom_records) -- host lookups below only ever resolve to native
    indices (hosts always come from a real bond graph or virtual-site
    lookup over native cations), but the same dict also supplies the
    ligand's own geometric record for its facet label.

    Returns `(ligands, unresolved_indices, malformed_indices)`.
    `unresolved_indices` are ligand atoms with no matching
    `ligand_provenance` entry at all -- only expected for atoms already
    present in a structure before a QDSpaceWebApp repassivation call, which
    this call never itself placed and so has no ledger for (see
    build_surface_metadata_document's `warnings`). `malformed_indices` is
    the subset of those naming an entry that *does* reference them but
    fails internal validation (missing/inconsistent anchor) -- a genuine
    bug signal, distinct from "no entry at all".
    """
    ligand_index_set = set(ligand_indices)

    records: List[dict] = []
    covered: set = set()
    malformed: set = set()
    for entry in ligand_provenance or []:
        atom_indices = sorted(
            int(i) for i in (entry.get("atom_indices_0based") or []) if int(i) in ligand_index_set
        )
        anchor_idx = entry.get("anchor_atom_index_0based")
        if not atom_indices:
            continue  # nothing attributable -- not even a malformed-entry signal
        if anchor_idx is None or int(anchor_idx) not in atom_indices:
            malformed.update(atom_indices)
            continue  # malformed entry -> its atoms end up in unresolved_indices below
        anchor_idx = int(anchor_idx)

        # A ligand can be bonded to more than one surface atom at once
        # (bridging/multidentate binding) -- keep every host that resolves
        # to a known qd_atom, closest first, rather than collapsing to a
        # single "primary" one. A host index that doesn't resolve (see
        # build_surface_metadata_document's `ligand_provenance` docstring)
        # is simply dropped from the list, not treated as fatal.
        raw_host_indices = entry.get("bound_host_atom_indices_0based")
        if raw_host_indices is None:
            legacy = entry.get("bound_host_atom_index_0based")
            raw_host_indices = [legacy] if legacy is not None else []
        host_recs = [
            qd_atoms_by_index[int(h)]
            for h in raw_host_indices
            if h is not None and int(h) in qd_atoms_by_index
        ]
        host_recs.sort(key=lambda r: float(np.linalg.norm(pts[anchor_idx] - r["position"])))

        record = {
            "type": entry.get("type") or symbols[anchor_idx],
            "anchor_element": symbols[anchor_idx],
            "anchor_atom_index_0based": anchor_idx,
            "atom_indices_0based": atom_indices,
            "host_atom_indices_0based": [r["atom_index_0based"] for r in host_recs],
            "host_elements": [r["element"] for r in host_recs],
            "anchor_host_distances": [
                float(np.linalg.norm(pts[anchor_idx] - r["position"])) for r in host_recs
            ],
            "provenance": entry.get("provenance"),
            "binding_site": _binding_site_from_host_count(len(host_recs)),
        }
        record.update(_facet_label_fields_from_own_position(qd_atoms_by_index.get(anchor_idx)))
        if entry.get("provenance") in ("polar_surface_reconstruction", "prepass_cleanup"):
            record["swap"] = True
        records.append(record)
        covered.update(atom_indices)

    unresolved = sorted(i for i in ligand_indices if i not in covered)
    malformed -= covered
    return records, unresolved, malformed


def build_surface_metadata_document(
    symbols: List[str],
    pts: NDArray[np.float64],
    planes: List[Plane],
    facets: List[Facet],
    charges: Dict[str, int],
    surf_tol: float,
    *,
    native_species: Optional[Iterable[str]] = None,
    material: Optional[str] = None,
    xyz_filename: Optional[str] = None,
    builder_params: Optional[Dict[str, Any]] = None,
    true_bulk_cn: Optional[Dict[str, int]] = None,
    pair_cuts: Optional[PairCuts] = None,
    facet_seed_terminations: Optional[Dict[tuple, str]] = None,
    ligand_provenance: Optional[List[Dict[str, Any]]] = None,
    lattice_substitution_provenance: Optional[List[Dict[str, Any]]] = None,
    atom_provenance: Optional[List[Dict[str, Any]]] = None,
    prepass_mode: str = "standard",
    prior_metadata_supplied: bool = False,
) -> Dict[str, Any]:
    """
    Assemble the full surface metadata document.

    `symbols`/`pts` must be in the exact same order as the XYZ file this
    metadata is meant to accompany. `atom_index_0based` values in the output
    are positions in this array.

    `native_species`: the set of element symbols that belong to the crystal
    lattice (e.g. both the cation and anion species of whatever material was
    built) -- typically {s.specie.symbol for s in struct.sites} from the
    build's CIF Structure, optionally unioned with any alloying-replacement
    elements. Every atom whose element is in this set is a `qd_atoms` entry
    (surface- or core-classified, see surface_atom_records); every other
    atom is a ligand atom, resolved via `ligand_provenance`. Defaults to
    "every distinct symbol in `symbols`" when omitted, i.e. no ligand/native
    split -- only correct when the caller has already excluded ligand atoms.

    `facet_seed_terminations`: optional {(h, k, l): termination_label} lookup
    (typically built from the Wulff-facet seeds used to construct the QD) to
    label each detected facet's requested termination, e.g. "cation_rich".
    Detected facets not present in this mapping get termination=None.

    `ligand_provenance`: list of *real*, build-time-captured ligand->host
    assignments (already in this array's index space), one dict per placed
    ligand fragment:
        {"type": str, "anchor_atom_index_0based": int,
         "atom_indices_0based": [int, ...],
         "bound_host_atom_indices_0based": [int, ...],
         "provenance": str}
    `provenance` is the real QD_Builder mechanism name that placed this
    ligand (e.g. `"x_type"`, `"l_type"`, `"z_type_neutral_exchange"`,
    `"charge_balance"` for the original build's baseline halide
    placeholder) -- see main.py's `_run_passivation_and_write_outputs` and
    backend.api_builder._run_repassivation_posttreatment for where each
    mechanism's ledger is captured and tagged. `bound_host_atom_indices_0based`
    may list more than one host (bridging/multidentate binding) -- every one
    that resolves to a known qd_atom is kept, closest-to-anchor first; one
    that doesn't resolve is simply dropped from the list rather than treated
    as fatal. A single legacy `bound_host_atom_index_0based: int` is also
    accepted for convenience. There is no geometric fallback: every QD_Builder
    placement mechanism already knows its real host(s) at placement time, so
    a ligand atom with no matching entry here is only expected when a
    structure already carried it into a QDSpaceWebApp repassivation call this
    function has no ledger for -- such atoms are still emitted (`provenance:
    null`) but also listed in the document's `warnings`.

    `lattice_substitution_provenance`: optional list of position-captured
    entries (see builder.facet_reconstruction.reconstruct_polar_facets and
    builder.passivation.prepass_surface_cleanup) for a native anion
    converted in place into the ligand species -- still a ligand (kept in
    `ligands`, not moved to `qd_atoms`, since it's still a ligand, not a
    native atom), but one that occupies a real former lattice position
    rather than sitting as a freely-placed exterior ligand. Resolved by
    position (not index) against this call's `pts`, so it's robust
    regardless of how many compaction/reorder steps happened between
    capture and this call. Each resolved entry is reported with real host
    provenance from the producing mechanism's own bond graph, the same
    uniform `binding_site`/`facet_ids`/`facet_hkls`/`facet_families` fields
    as any other ligand (see _build_lattice_substitution_records), and
    `"swap": True`. Takes precedence over `ligand_provenance` for whichever
    ligand atoms it covers.

    `atom_provenance`: optional list of position-captured entries, one per
    non-native `qd_atoms` site (e.g. `{"position": [...], "symbol": "Zn",
    "provenance": "alloying"}` per alloying-replaced cation). Position-based
    like `lattice_substitution_provenance`, for the same reason: robust to
    any atom-count-changing step (z-type, further charge balancing, ...)
    that runs between capture and this call. Resolved here (see
    _resolve_atom_provenance_by_position) into the index->mechanism-name map
    `surface_atom_records` expects; defaults to `"native"` for any index not
    present (including one whose entry no longer resolves).

    `prepass_mode`: the build's own `cfg.passivation.prepass_mode` --
    `qd_atoms`' `role` is computed with the real build's own geometric
    classifier when this is `"role-aware"` (see surface_atom_records).

    `prior_metadata_supplied`: whether the caller had a prior surface-
    metadata document available to chain ligand provenance from at all
    (see resolve_prior_ligand_provenance) -- only affects the `reason`
    text of a `warnings` entry for an atom that still isn't covered by
    anything: `"no_prior_metadata_supplied"` when this is `False` (nothing
    to chain from, even in principle -- always the case for an original
    build, which has no "prior"), `"prior_position_match_failed"` when
    `True` (chaining was attempted but this atom still didn't resolve).

    Returns a plain dict, JSON-serializable as-is.
    """
    symbols = list(symbols)
    pts = np.asarray(pts, float)
    n = len(symbols)

    if native_species is None:
        native_indices = list(range(n))
        ligand_indices: List[int] = []
    else:
        native_set = set(native_species)
        native_indices = [i for i in range(n) if symbols[i] in native_set]
        ligand_indices = [i for i in range(n) if symbols[i] not in native_set]

    lattice_subs_resolved = _resolve_lattice_substitutions(symbols, pts, lattice_substitution_provenance)
    atom_provenance_resolved = _resolve_atom_provenance_by_position(symbols, pts, atom_provenance)

    # Facet membership / coordination are computed against the *full* atom
    # array (native + ligand together) so a passivated surface atom's true
    # (ligand-inclusive) coordination is reflected; only native atoms are
    # emitted as qd_atoms records (see surface_atom_records docstring).
    # Every atom is included here (not just native ones) so every ligand
    # also gets a real, standard geometric record -- its own facet_ids/
    # facet_hkls/facet_families, used below instead of inheriting from a
    # host. Lattice-substitution and generic ligand records are still
    # merged into ligands[], never emitted as a parallel qd_atoms entry.
    qd_atoms_all = surface_atom_records(
        symbols, pts, planes, facets, surf_tol, charges,
        true_bulk_cn=true_bulk_cn,
        pair_cuts=pair_cuts,
        atom_provenance=atom_provenance_resolved,
        prepass_mode=prepass_mode,
    )
    native_index_set = set(native_indices)
    qd_atoms = [r for r in qd_atoms_all if r["atom_index_0based"] in native_index_set]
    qd_atoms_by_index = {r["atom_index_0based"]: r for r in qd_atoms_all}

    lattice_substitution_records = _build_lattice_substitution_records(
        symbols, pts, lattice_subs_resolved, qd_atoms_by_index,
    )
    lattice_sub_covered = {rec["atom_indices_0based"][0] for rec in lattice_substitution_records}
    ligand_indices_remaining = [i for i in ligand_indices if i not in lattice_sub_covered]

    ligands, unresolved_ligand_indices, malformed_ligand_indices = _assign_ligands_from_provenance(
        symbols, pts, ligand_indices_remaining, qd_atoms_by_index,
        ligand_provenance=ligand_provenance,
    )
    warnings: List[Dict[str, Any]] = []
    for i in unresolved_ligand_indices:
        ligands.append({
            "type": symbols[i],
            "anchor_element": None,
            "anchor_atom_index_0based": None,
            "atom_indices_0based": [i],
            "host_atom_indices_0based": [],
            "host_elements": [],
            "anchor_host_distances": [],
            "provenance": None,
            "binding_site": None,
            **_facet_label_fields_from_own_position(qd_atoms_by_index.get(i)),
        })
        if i in malformed_ligand_indices:
            reason = "malformed_ledger_entry"
        elif prior_metadata_supplied:
            reason = "prior_position_match_failed"
        else:
            reason = "no_prior_metadata_supplied"
        warnings.append({"atom_index_0based": i, "reason": reason})
    ligands = lattice_substitution_records + ligands
    ligands.sort(key=lambda r: min(r["atom_indices_0based"]))
    for ligand_id, rec in enumerate(ligands):
        rec["ligand_id"] = ligand_id

    facet_seed_terminations = facet_seed_terminations or {}
    facets_out = []
    for facet_id, (f, (normal, d)) in enumerate(zip(facets, planes)):
        hkl = (int(f.h), int(f.k), int(f.l))
        family = sorted((abs(hkl[0]), abs(hkl[1]), abs(hkl[2])), reverse=True)
        facets_out.append({
            "facet_id": facet_id,
            "hkl": list(hkl),
            "family": family,
            "family_label": _facet_family_label(*hkl),
            "termination": facet_seed_terminations.get(hkl, getattr(f, "termination", None)),
            "normal": [float(x) for x in normal],
            "plane_offset": float(d),
        })

    doc = {
        "structure": {
            "material": material,
            "atom_count": n,
            "coordinate_units": "angstrom",
            "xyz_filename": xyz_filename,
        },
        "builder": builder_params or {},
        "facets": facets_out,
        "qd_atoms": qd_atoms,
        "ligands": ligands,
        "warnings": warnings,
    }

    validate_metadata_matches_xyz(doc, symbols)
    return doc


def validate_metadata_matches_xyz(doc: Dict[str, Any], xyz_symbols: List[str]) -> None:
    """
    Raise ValueError if `doc` is not self-consistent with `xyz_symbols`
    (the exact symbol list, in order, of the XYZ file this metadata is meant
    to accompany). Checked internally by build_surface_metadata_document
    before it returns; also exported so tests/callers can re-check a document
    loaded back from disk against a downloaded XYZ.
    """
    n = len(xyz_symbols)
    atom_count = doc.get("structure", {}).get("atom_count")
    if atom_count != n:
        raise ValueError(
            f"surface metadata atom_count={atom_count} does not match "
            f"XYZ atom count={n}"
        )

    def _check_index(label: str, idx, element: Optional[str] = None) -> None:
        if not (0 <= idx < n):
            raise ValueError(f"{label}: index={idx} out of range for {n} XYZ atoms")
        if element is not None and xyz_symbols[idx] != element:
            raise ValueError(
                f"{label}: element={element!r} does not match XYZ atom {idx} ({xyz_symbols[idx]!r})"
            )

    for rec in doc.get("qd_atoms", []):
        idx0 = rec["atom_index_0based"]
        _check_index(f"qd_atom_id={rec.get('qd_atom_id')}", idx0, rec["element"])

    for lig in doc.get("ligands", []):
        label = f"ligand_id={lig.get('ligand_id')}"
        # anchor_atom_index_0based/anchor_element are None for a ligand with
        # no recoverable provenance -- only expected for an atom a
        # QDSpaceWebApp repassivation call didn't itself place (see
        # build_surface_metadata_document's `warnings`) -- skip the check
        # rather than treat a deliberately-unclaimed anchor as an
        # inconsistency.
        if lig.get("anchor_atom_index_0based") is not None:
            _check_index(f"{label} anchor_atom_index_0based", lig["anchor_atom_index_0based"], lig["anchor_element"])
        for i in lig.get("atom_indices_0based", []):
            _check_index(f"{label} atom_indices_0based", i)

        host_idx = lig.get("host_atom_indices_0based", [])
        host_el = lig.get("host_elements", [])
        host_dist = lig.get("anchor_host_distances", [])
        if not (len(host_idx) == len(host_el) == len(host_dist)):
            raise ValueError(f"{label}: host_atom_indices_0based/host_elements/anchor_host_distances length mismatch")
        for idx, el in zip(host_idx, host_el):
            _check_index(f"{label} host_atom_indices_0based", idx, el)


def write_surface_metadata_json(path: str, doc: Dict[str, Any]) -> None:
    """Write `doc` (as returned by build_surface_metadata_document) to `path`."""
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
