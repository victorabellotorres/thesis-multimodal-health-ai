from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .errors import Nutrition5kError


@dataclass(frozen=True)
class ClearSummary:
    removed: int
    absent: int


def clear_preparation(data_root: Path) -> ClearSummary:
    """Remove the local Nutrition5k data-preparation state.

    Source code, the Python environment, baseline outputs, and all paths
    outside the explicitly supplied data root are preserved.
    """
    removed = 0
    absent = 0
    if not data_root.exists():
        absent += 1
    elif not data_root.is_dir():
        raise Nutrition5kError(f"expected a directory to clear: {data_root}")
    else:
        try:
            shutil.rmtree(data_root)
        except OSError as exc:
            raise Nutrition5kError(f"failed to clear {data_root}: {exc}") from exc
        removed += 1
    return ClearSummary(removed=removed, absent=absent)
