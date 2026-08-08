"""Multi-frame XYZ reading (complete frames only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Frame:
    """One complete XYZ frame."""

    symbols: Tuple[str, ...]
    coordinates: Tuple[Tuple[float, float, float], ...]
    comment: str = ""
    index: int = 0  # 0-based frame index in file

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)


def _normalize_symbol(raw: str) -> str:
    symbol = raw.strip()
    if not symbol:
        return symbol
    # Strip oxidation-state suffixes like Cd2+ if present.
    out = []
    for char in symbol:
        if char.isalpha():
            out.append(char)
        else:
            break
    text = "".join(out)
    if not text:
        return symbol
    return text[0].upper() + text[1:].lower()


def read_xyz_frames(path: Path | str) -> List[Frame]:
    """Return all complete frames from a multi-frame XYZ file."""

    path = Path(path)
    frames: List[Frame] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    i = 0
    frame_index = 0
    n_lines = len(lines)
    while i < n_lines:
        while i < n_lines and not lines[i].strip():
            i += 1
        if i >= n_lines:
            break
        try:
            count = int(lines[i].strip().split()[0])
        except (ValueError, IndexError):
            # Skip garbage line and continue scanning.
            i += 1
            continue
        if i + 1 + count > n_lines:
            # Truncated trailing frame.
            break
        comment = lines[i + 1].strip() if i + 1 < n_lines else ""
        symbols: List[str] = []
        coords: List[Tuple[float, float, float]] = []
        ok = True
        for row in lines[i + 2 : i + 2 + count]:
            parts = row.split()
            if len(parts) < 4:
                ok = False
                break
            try:
                symbols.append(_normalize_symbol(parts[0]))
                coords.append(
                    (float(parts[1]), float(parts[2]), float(parts[3]))
                )
            except ValueError:
                ok = False
                break
        if ok and len(symbols) == count:
            frames.append(
                Frame(
                    symbols=tuple(symbols),
                    coordinates=tuple(coords),
                    comment=comment,
                    index=frame_index,
                )
            )
            frame_index += 1
        i += 2 + count
    return frames


def first_and_last_frames(
    path: Path | str,
) -> Tuple[Optional[Frame], Optional[Frame], int]:
    """Return (first, last, n_complete_frames)."""

    frames = read_xyz_frames(path)
    if not frames:
        return None, None, 0
    return frames[0], frames[-1], len(frames)


def load_start_xyz(path: Path | str) -> Optional[Frame]:
    """Load a single-frame or multi-frame start geometry (use first frame)."""

    frames = read_xyz_frames(path)
    return frames[0] if frames else None


def count_symbols(symbols: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts
