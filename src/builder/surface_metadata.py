# nanocrystal_builder/surface_metadata.py
"""
Assembles the surface-metadata JSON document: `qd_atoms`, `facets`, `ligands`.
See README "Surface Metadata (JSON)" for the field reference.
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
    """Most common (captured -> current) delta over `entries`, to add back to a
    captured position; None if undetermined."""
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
    """Resolve {"position", "symbol"} entries against the current atom array
    (see _detect_global_offset). Returns (resolved_index, entry) pairs."""
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
    """{"position", "symbol", "provenance"} entries -> index->provenance map."""
    return {
        idx: entry.get("provenance")
        for idx, entry in _resolve_positional_entries(symbols, pts, entries, tol=tol)
    }


def _resolve_material_by_position(
    symbols: List[str],
    pts: NDArray[np.float64],
    entries: Optional[List[Dict[str, Any]]],
    *,
    tol: float = 1e-3,
) -> Dict[int, str]:
    """Same as _resolve_atom_provenance_by_position, for the core/shell
    `material` tag instead."""
    return {
        idx: entry.get("material")
        for idx, entry in _resolve_positional_entries(symbols, pts, entries, tol=tol)
    }


def _resolve_lattice_substitutions(
    symbols: List[str],
    pts: NDArray[np.float64],
    entries: Optional[List[Dict[str, Any]]],
    *,
    tol: float = 1e-3,
    charges: Optional[Dict[str, int]] = None,
) -> List[dict]:
    """Resolve lattice-substitution ledger entries (see
    facet_reconstruction.reconstruct_polar_facets) against the current atom
    array; see _build_lattice_substitution_records. `charges`, if given,
    lets a host whose species changed since capture still resolve as a cation."""
    if not entries:
        return []
    pts = np.asarray(pts, float)
    offset = _detect_global_offset(pts, symbols, entries)
    if offset is None:
        return []
    resolved = []
    for idx, entry in _resolve_positional_entries(symbols, pts, entries, tol=tol):
        host_symbols = entry.get("host_symbols") or []
        host_indices = []
        for hp, hs in zip(entry.get("host_positions") or [], host_symbols):
            host_pos = (np.asarray(hp, float) + offset).tolist()
            hi = _find_atom_at_position(pts, symbols, host_pos, hs, tol)
            if hi is None and charges:
                # Fallback: species may have changed since capture (e.g.
                # alloying) -- accept a real cation at this exact position.
                candidate = _find_atom_at_position(pts, symbols, host_pos, None, tol)
                if candidate is not None and charges.get(symbols[candidate], 0) > 0:
                    hi = candidate
            if hi is not None:
                host_indices.append(hi)
        resolved.append({
            "anchor_atom_index_0based": idx,
            "host_atom_indices_0based": host_indices,
            "replaced_element": entry.get("replaced_element"),
            "mechanism": entry.get("mechanism"),
            "neutrality_redistributed": entry.get("neutrality_redistributed"),
            "z_type_relocated": entry.get("z_type_relocated"),
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
    """Re-verify a prior metadata document's `ligands` by position against the
    current atom array (`prior_symbols`/`prior_pts` index into the prior
    document, not `symbols`/`pts`). Returns a list shaped like
    build_surface_metadata_document's `ligand_provenance`; drops a ligand
    whose atoms/hosts no longer resolve."""
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
            # Prior document never resolved this ligand -- nothing to chain.
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

        out_entry = {
            "type": lig.get("type"),
            "anchor_atom_index_0based": new_anchor,
            "atom_indices_0based": sorted(new_by_old.values()),
            "bound_host_atom_indices_0based": new_hosts,
            "provenance": lig.get("provenance"),
        }
        if lig.get("swap"):
            out_entry["swap"] = True
            if lig.get("replaced_element") is not None:
                out_entry["replaced_element"] = lig["replaced_element"]
        if lig.get("z_type_relocated"):
            out_entry["z_type_relocated"] = True
        if lig.get("neutrality_redistributed"):
            out_entry["neutrality_redistributed"] = True
        resolved.append(out_entry)
    return resolved


def _build_lattice_substitution_records(
    symbols: List[str],
    pts: NDArray[np.float64],
    resolved: List[dict],
    qd_atoms_by_index: Dict[int, dict],
) -> List[dict]:
    """`ligands` entries for resolved lattice-substitution atoms, same field
    set as _assign_ligands_from_provenance."""
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
        if entry.get("replaced_element") is not None:
            record["swap"] = True
            record["replaced_element"] = entry["replaced_element"]
        if entry.get("neutrality_redistributed"):
            record["neutrality_redistributed"] = True
        if entry.get("z_type_relocated"):
            record["z_type_relocated"] = True
        record.update(_facet_label_fields_from_own_position(qd_atoms_by_index.get(anchor_idx)))
        records.append(record)
    return records


def _facet_label_fields_from_own_position(own_record: Optional[dict]) -> Dict[str, list]:
    """A ligand's own facet_ids/facet_hkls/facet_families, not inherited from
    its host. `own_record` is None for a ligand beyond `surf_tol`."""
    if not own_record:
        return {"facet_ids": [], "facet_hkls": [], "facet_families": []}
    return {
        "facet_ids": list(own_record.get("facet_ids") or []),
        "facet_hkls": [list(h) for h in (own_record.get("facet_hkls") or [])],
        "facet_families": list(own_record.get("facet_families") or []),
    }


def _binding_site_from_host_count(n_hosts: int) -> Optional[str]:
    """Standard coordination-chemistry μ-notation for host count."""
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
    """Build `ligands` from real, build-time-captured provenance only -- no
    geometric fallback. Returns (ligands, unresolved_indices, malformed_indices)."""
    ligand_index_set = set(ligand_indices)

    records: List[dict] = []
    covered: set = set()
    malformed: set = set()
    for entry in ligand_provenance or []:
        atom_indices = sorted(
            int(i) for i in (entry.get("atom_indices_0based") or []) if int(i) in ligand_index_set
        )
        anchor_idx_raw = entry.get("anchor_atom_index_0based")
        if not atom_indices:
            continue  # nothing attributable -- not even a malformed-entry signal
        extra_host_indices: List[int] = []
        if anchor_idx_raw is None:
            malformed.update(atom_indices)
            continue  # malformed entry -> its atoms end up in unresolved_indices below
        anchor_idx_raw = int(anchor_idx_raw)
        if anchor_idx_raw not in atom_indices:
            # Anchor got swept into qd_atoms (element collision, e.g. an
            # MXn cation sharing a symbol with an alloying replacement) --
            # treat it as this fragment's host instead of failing it.
            if anchor_idx_raw in qd_atoms_by_index:
                extra_host_indices = [anchor_idx_raw]
                anchor_idx = min(atom_indices)
            else:
                malformed.update(atom_indices)
                continue  # malformed entry -> its atoms end up in unresolved_indices below
        else:
            anchor_idx = anchor_idx_raw

        # Bridging/multidentate binding: keep every host that resolves to a
        # known qd_atom, closest first; one that doesn't resolve is dropped.
        raw_host_indices = entry.get("bound_host_atom_indices_0based")
        if raw_host_indices is None:
            legacy = entry.get("bound_host_atom_index_0based")
            raw_host_indices = [legacy] if legacy is not None else []
        raw_host_indices = list(raw_host_indices) + extra_host_indices
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
        if entry.get("swap"):
            record["swap"] = True
            if entry.get("replaced_element") is not None:
                record["replaced_element"] = entry["replaced_element"]
        if entry.get("z_type_relocated"):
            record["z_type_relocated"] = True
        if entry.get("neutrality_redistributed"):
            record["neutrality_redistributed"] = True
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
    material_provenance: Optional[List[Dict[str, Any]]] = None,
    prepass_mode: str = "standard",
    prior_metadata_supplied: bool = False,
) -> Dict[str, Any]:
    """
    Assemble the full surface metadata document. `symbols`/`pts` must match
    the XYZ this metadata accompanies.

    - native_species: lattice element symbols, splits qd_atoms vs ligands; defaults to every symbol present.
    - facet_seed_terminations: optional {(h,k,l): termination}; unconfigured facets get it inferred (see _inferred_facet_termination).
    - ligand_provenance: real build-time ligand->host assignments (see _assign_ligands_from_provenance); unmatched atoms go in `warnings`.
    - lattice_substitution_provenance: position-resolved native-anion->ligand swaps (see _resolve_lattice_substitutions), takes precedence over ligand_provenance.
    - atom_provenance / material_provenance: position-resolved qd_atoms `provenance`/`material` tags.
    - prepass_mode: "role-aware" computes qd_atoms `role` with the build's own geometric classifier.
    - prior_metadata_supplied: whether a prior document was available to chain from, for `warnings` reason text.

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

    lattice_subs_resolved = _resolve_lattice_substitutions(
        symbols, pts, lattice_substitution_provenance, charges=charges,
    )
    atom_provenance_resolved = _resolve_atom_provenance_by_position(symbols, pts, atom_provenance)
    material_resolved = _resolve_material_by_position(symbols, pts, material_provenance)

    # Full atom array (native + ligand) so coordination includes ligand
    # bonds; only native atoms are kept as qd_atoms below, ligands get their
    # own geometric record here too (used by _facet_label_fields_from_own_position).
    qd_atoms_all = surface_atom_records(
        symbols, pts, planes, facets, surf_tol, charges,
        true_bulk_cn=true_bulk_cn,
        pair_cuts=pair_cuts,
        atom_provenance=atom_provenance_resolved,
        material=material_resolved,
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

    # Per-facet native-sublattice composition, for _inferred_facet_termination
    # below. Restricted to outer-layer, single-facet-membership atoms only --
    # otherwise the opposite-species sublayer and shared edge/vertex atoms
    # make every polar facet look mixed (verified empirically).
    facet_native_sublattices: Dict[int, set] = {}
    for rec in qd_atoms:
        if rec.get("layer") != "outer":
            continue
        fids = rec.get("facet_ids") or []
        if len(fids) != 1:
            continue
        sub = rec.get("sublattice")
        if sub is None:
            continue
        facet_native_sublattices.setdefault(fids[0], set()).add(sub)

    def _inferred_facet_termination(facet_id: int) -> Optional[str]:
        """Termination inferred from the facet's real native atoms (cf.
        heterointerface._richness); None if nothing to infer from."""
        signs = facet_native_sublattices.get(facet_id)
        if not signs:
            return None
        if signs == {"cation"}:
            return "cation_rich"
        if signs == {"anion"}:
            return "anion_rich"
        return "stoichiometric"

    facet_seed_terminations = facet_seed_terminations or {}
    facets_out = []
    for facet_id, (f, (normal, d)) in enumerate(zip(facets, planes)):
        hkl = (int(f.h), int(f.k), int(f.l))
        family = sorted((abs(hkl[0]), abs(hkl[1]), abs(hkl[2])), reverse=True)
        termination = facet_seed_terminations.get(hkl, getattr(f, "termination", None))
        if termination is None:
            termination = _inferred_facet_termination(facet_id)
        facets_out.append({
            "facet_id": facet_id,
            "hkl": list(hkl),
            "family": family,
            "family_label": _facet_family_label(*hkl),
            "termination": termination,
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

    validate_metadata_matches_xyz(doc, symbols, pts)
    return doc


def validate_metadata_matches_xyz(
    doc: Dict[str, Any],
    xyz_symbols: List[str],
    xyz_pts: Optional[NDArray[np.float64]] = None,
    *,
    pos_tol: float = 1e-4,
) -> None:
    """Raise ValueError if `doc` isn't self-consistent with `xyz_symbols` (the
    XYZ this metadata accompanies). `xyz_pts`, if given, also checks positions/
    distances against it; skipped when omitted."""
    n = len(xyz_symbols)
    atom_count = doc.get("structure", {}).get("atom_count")
    if atom_count != n:
        raise ValueError(
            f"surface metadata atom_count={atom_count} does not match "
            f"XYZ atom count={n}"
        )
    xyz_pts = np.asarray(xyz_pts, float) if xyz_pts is not None else None

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
        if xyz_pts is not None:
            dist = float(np.linalg.norm(np.asarray(rec["position"], float) - xyz_pts[idx0]))
            if dist > pos_tol:
                raise ValueError(
                    f"qd_atom_id={rec.get('qd_atom_id')}: position is {dist:.6f} Å "
                    f"from XYZ atom {idx0}'s real position (tol={pos_tol})"
                )

    for lig in doc.get("ligands", []):
        label = f"ligand_id={lig.get('ligand_id')}"
        # anchor is None for a ligand with no recoverable provenance -- skip.
        anchor_idx = lig.get("anchor_atom_index_0based")
        if anchor_idx is not None:
            _check_index(f"{label} anchor_atom_index_0based", anchor_idx, lig["anchor_element"])
        for i in lig.get("atom_indices_0based", []):
            _check_index(f"{label} atom_indices_0based", i)

        host_idx = lig.get("host_atom_indices_0based", [])
        host_el = lig.get("host_elements", [])
        host_dist = lig.get("anchor_host_distances", [])
        if not (len(host_idx) == len(host_el) == len(host_dist)):
            raise ValueError(f"{label}: host_atom_indices_0based/host_elements/anchor_host_distances length mismatch")
        for idx, el, dist in zip(host_idx, host_el, host_dist):
            _check_index(f"{label} host_atom_indices_0based", idx, el)
            if xyz_pts is not None and anchor_idx is not None:
                real_dist = float(np.linalg.norm(xyz_pts[anchor_idx] - xyz_pts[idx]))
                if abs(real_dist - dist) > pos_tol:
                    raise ValueError(
                        f"{label}: anchor_host_distances claims {dist:.6f} Å to host "
                        f"{idx}, but XYZ positions give {real_dist:.6f} Å (tol={pos_tol})"
                    )


def write_surface_metadata_json(path: str, doc: Dict[str, Any]) -> None:
    """Write `doc` (as returned by build_surface_metadata_document) to `path`."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
