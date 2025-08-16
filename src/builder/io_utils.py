# nanocrystal_builder/io_utils.py
from __future__ import annotations
import json, os
from typing import List
import numpy as np
from numpy.typing import NDArray
from collections import Counter

def write_xyz(path: str, symbols: List[str], pts: NDArray[np.float64]) -> None:
    with open(path, "w") as fh:
        fh.write(f"{len(symbols)}\n{os.path.basename(path)}\n")
        for s, (x, y, z) in zip(symbols, pts):
            fh.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")

def center_coords(pts: NDArray[np.float64]) -> NDArray[np.float64]:
    com = np.mean(pts, axis=0)
    return pts - com

def write_manifest(prefix: str, symbols: List[str], charges: dict):
    out = {
        "counts": Counter(symbols),
        "total_charge": int(sum(charges[s] for s in symbols if s in charges)),
    }
    with open(f"{prefix}.json", "w") as fh:
        json.dump(out, fh, indent=2, default=int)

