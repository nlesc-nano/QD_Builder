"""Discover CP2K job directories under DFT roots."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


K_DIR_RE = re.compile(r"^k(?P<k>\d+)$", re.IGNORECASE)
P_DIR_RE = re.compile(r"^p(?P<p>\d+)$", re.IGNORECASE)
KP_IN_NAME_RE = re.compile(
    r"k(?P<k>\d+)_p(?P<p>\d+)", re.IGNORECASE
)


@dataclass(frozen=True)
class JobRecord:
    """One DFT job with trajectory path and parsed labels."""

    root_label: str
    root_path: Path
    job_dir: Path
    traj_path: Path
    start_path: Optional[Path]
    structure_id: str
    k: Optional[int]
    p: Optional[int]


def _parse_kp_from_path(job_dir: Path) -> tuple[Optional[int], Optional[int]]:
    k_val: Optional[int] = None
    p_val: Optional[int] = None
    for part in job_dir.parts:
        mk = K_DIR_RE.match(part)
        if mk:
            k_val = int(mk.group("k"))
        mp = P_DIR_RE.match(part)
        if mp:
            p_val = int(mp.group("p"))
    if k_val is None or p_val is None:
        match = KP_IN_NAME_RE.search(job_dir.name)
        if match:
            if k_val is None:
                k_val = int(match.group("k"))
            if p_val is None:
                p_val = int(match.group("p"))
    return k_val, p_val


def discover_jobs(
    roots: Iterable[Path | str],
    *,
    traj_name: str = "CdSe-pos-1.xyz",
    start_name: str = "start.xyz",
) -> List[JobRecord]:
    """Find all job dirs that contain a trajectory XYZ."""

    jobs: List[JobRecord] = []
    seen: set[Path] = set()
    for root_raw in roots:
        root = Path(root_raw).expanduser().resolve()
        if not root.is_dir():
            continue
        label = root.name
        for traj in sorted(root.rglob(traj_name)):
            job_dir = traj.parent.resolve()
            if job_dir in seen:
                continue
            seen.add(job_dir)
            start = job_dir / start_name
            k_val, p_val = _parse_kp_from_path(job_dir)
            jobs.append(
                JobRecord(
                    root_label=label,
                    root_path=root,
                    job_dir=job_dir,
                    traj_path=traj,
                    start_path=start if start.is_file() else None,
                    structure_id=job_dir.name,
                    k=k_val,
                    p=p_val,
                )
            )
    jobs.sort(key=lambda j: (j.k if j.k is not None else -1, j.p if j.p is not None else -1, j.structure_id))
    return jobs
