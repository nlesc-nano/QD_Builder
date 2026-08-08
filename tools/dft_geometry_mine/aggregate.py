"""Aggregate samples into summary statistics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    w = pos - lo
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w


def summarize_values(values: Sequence[float]) -> Dict[str, object]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
            "recommended": None,
        }
    arr = sorted(float(v) for v in values)
    n = len(arr)
    mean = sum(arr) / n
    var = sum((x - mean) ** 2 for x in arr) / n
    std = math.sqrt(var)
    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "median": round(_percentile(arr, 0.5), 4),
        "p10": round(_percentile(arr, 0.1), 4),
        "p90": round(_percentile(arr, 0.9), 4),
        "min": round(arr[0], 4),
        "max": round(arr[-1], 4),
        # Construction default: median of clean set.
        "recommended": round(_percentile(arr, 0.5), 3),
    }


def aggregate_groups(
    samples: Iterable[Tuple[str, float]],
) -> Dict[str, Dict[str, object]]:
    """Group key -> value samples."""

    buckets: Dict[str, List[float]] = defaultdict(list)
    for key, value in samples:
        buckets[key].append(float(value))
    return {key: summarize_values(vals) for key, vals in sorted(buckets.items())}


def rate(numer: int, denom: int) -> Optional[float]:
    if denom <= 0:
        return None
    return round(numer / denom, 4)
