"""mtime-aware CSV cache shared by the dashboard routers.

The routers used to hold their frames in module-level globals that were only
ever populated once. Anything that rewrote the CSVs afterwards — a PDF upload,
or re-running scripts/score.py against a live server — was invisible until the
process restarted. Keying the cache on the file's mtime makes a rewrite the
signal to re-read, so no caller has to remember to invalidate anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

_cache: dict[Path, tuple[float, pd.DataFrame]] = {}


def load_csv_cached(path: Path, read: Callable[[Path], pd.DataFrame]) -> pd.DataFrame:
    """Return ``read(path)``, re-reading only when the file changed on disk."""
    mtime = path.stat().st_mtime
    cached = _cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    frame = read(path)
    _cache[path] = (mtime, frame)
    return frame
