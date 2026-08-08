"""Per-job analysis orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .angles import (
    BondSample,
    AngleSample,
    collect_angles,
    collect_bonds,
    k_window,
    motif_flags,
)
from .bonds import BondCutoffs, analyze_frame, formula_matches_kp
from .connectivity import full_components, inorganic_components
from .discover import JobRecord
from .dihedrals import (
    ImproperDihedralSample,
    ProperDihedralSample,
    collect_improper_dihedrals,
    collect_proper_dihedrals,
)
from .xyz_io import Frame, first_and_last_frames, load_start_xyz


@dataclass
class StructureResult:
    job: JobRecord
    n_frames: int
    has_last: bool
    clean: bool
    quarantine_reasons: List[str] = field(default_factory=list)
    structure_row: Dict[str, Any] = field(default_factory=dict)
    bond_samples: List[BondSample] = field(default_factory=list)
    angle_samples: List[AngleSample] = field(default_factory=list)
    improper_dihedral_samples: List[ImproperDihedralSample] = field(
        default_factory=list
    )
    proper_dihedral_samples: List[ProperDihedralSample] = field(
        default_factory=list
    )
    same_species_samples: List[Tuple[str, float]] = field(default_factory=list)
    start_row: Optional[Dict[str, Any]] = None


def _frame_core(
    frame: Frame,
    *,
    job: JobRecord,
    cutoffs: BondCutoffs,
    linear_threshold_deg: float,
    label: str,
) -> Tuple[
    Dict[str, Any],
    List[str],
    List[BondSample],
    List[AngleSample],
    List[ImproperDihedralSample],
    List[ProperDihedralSample],
    List[Tuple[str, float]],
]:
    graph = analyze_frame(frame, cutoffs)
    reasons: List[str] = []
    counts = graph.formula_counts()
    formula_ok, formula_msg = formula_matches_kp(counts, job.k, job.p)
    if not formula_ok:
        reasons.append(f"formula:{formula_msg}")
    if graph.n_cd_cd:
        reasons.append(f"Cd-Cd:{graph.n_cd_cd}")
    if graph.n_se_se:
        reasons.append(f"Se-Se:{graph.n_se_se}")
    if graph.n_cl_cl:
        reasons.append(f"Cl-Cl:{graph.n_cl_cl}")

    full_cc = full_components(len(graph.symbols), graph.neighbors)
    inorg_cc, n_inorg = inorganic_components(graph.symbols, graph.neighbors)
    motifs = motif_flags(
        frame, graph, linear_threshold_deg=linear_threshold_deg
    )
    bonds = collect_bonds(graph, k=job.k)
    angles = collect_angles(frame, graph)
    improper_dihedrals = collect_improper_dihedrals(frame, graph)
    proper_dihedrals = collect_proper_dihedrals(frame, graph)

    row: Dict[str, Any] = {
        "label": label,
        "root": job.root_label,
        "structure_id": job.structure_id,
        "job_dir": str(job.job_dir),
        "k": job.k,
        "p": job.p,
        "k_window": k_window(job.k),
        "n_atoms": len(graph.symbols),
        "n_Cd": counts.get("Cd", 0),
        "n_Se": counts.get("Se", 0),
        "n_Cl": counts.get("Cl", 0),
        "formula_ok": int(formula_ok),
        "formula_msg": formula_msg,
        "n_cd_cd": graph.n_cd_cd,
        "n_se_se": graph.n_se_se,
        "n_cl_cl": graph.n_cl_cl,
        "has_homonuclear": int(graph.has_homonuclear_contact),
        "full_components": full_cc,
        "inorganic_components": inorg_cc,
        "n_inorganic_atoms": n_inorg,
        "inorganic_connected": int(inorg_cc <= 1 and n_inorg > 0),
        "full_connected": int(full_cc <= 1 and len(graph.symbols) > 0),
        **motifs,
    }
    return (
        row,
        reasons,
        bonds,
        angles,
        improper_dihedrals,
        proper_dihedrals,
        graph.same_species_samples,
    )


def analyze_job(
    job: JobRecord,
    *,
    cutoffs: BondCutoffs,
    linear_threshold_deg: float = 160.0,
    compare_start: bool = False,
) -> StructureResult:
    first, last, n_frames = first_and_last_frames(job.traj_path)
    if last is None:
        return StructureResult(
            job=job,
            n_frames=0,
            has_last=False,
            clean=False,
            quarantine_reasons=["missing_or_empty_trajectory"],
            structure_row={
                "label": "final",
                "root": job.root_label,
                "structure_id": job.structure_id,
                "job_dir": str(job.job_dir),
                "k": job.k,
                "p": job.p,
                "n_frames": 0,
                "has_last": 0,
            },
        )

    row, reasons, bonds, angles, improper, proper, same_sp = _frame_core(
        last,
        job=job,
        cutoffs=cutoffs,
        linear_threshold_deg=linear_threshold_deg,
        label="final",
    )
    row["n_frames"] = n_frames
    row["has_last"] = 1
    row["traj_path"] = str(job.traj_path)
    clean = len(reasons) == 0
    row["clean"] = int(clean)
    row["quarantine_reasons"] = ";".join(reasons)

    start_row = None
    if compare_start:
        start_frame = None
        if job.start_path is not None:
            start_frame = load_start_xyz(job.start_path)
        if start_frame is None:
            start_frame = first
        if start_frame is not None:
            start_row, _, _, _, _, _, _ = _frame_core(
                start_frame,
                job=job,
                cutoffs=cutoffs,
                linear_threshold_deg=linear_threshold_deg,
                label="start",
            )
            start_row["n_frames"] = n_frames
            start_row["source"] = (
                "start.xyz" if job.start_path is not None else "traj_first"
            )

    return StructureResult(
        job=job,
        n_frames=n_frames,
        has_last=True,
        clean=clean,
        quarantine_reasons=reasons,
        structure_row=row,
        bond_samples=bonds if clean else [],
        angle_samples=angles if clean else [],
        improper_dihedral_samples=improper if clean else [],
        proper_dihedral_samples=proper if clean else [],
        same_species_samples=same_sp,
        start_row=start_row,
    )
