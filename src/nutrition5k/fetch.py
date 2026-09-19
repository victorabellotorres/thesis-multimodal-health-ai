from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .dataset import METADATA_FILES, SPLIT_FILES
from .errors import Nutrition5kError

GCS_BASE_URL = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset"
REQUIRED_FILES = (*METADATA_FILES, *SPLIT_FILES.values())


@dataclass(frozen=True)
class FetchSummary:
    downloaded: int
    skipped: int


def fetch_dataset_index(
    root: Path, progress: Callable[[str], None] | None = None
) -> FetchSummary:
    """Fetch the metadata and RGB split files required by this pipeline.

    Existing non-empty files are preserved so that interrupted fetches can be
    resumed. Each new file is first written beside its destination and then
    atomically moved into place.
    """
    downloaded = 0
    skipped = 0
    for relative in REQUIRED_FILES:
        destination = root / relative
        if destination.is_file() and destination.stat().st_size > 0:
            skipped += 1
            if progress:
                progress(f"skip {relative}")
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        url = f"{GCS_BASE_URL}/{relative}"
        if progress:
            progress(f"fetch {relative}")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            if temporary.stat().st_size == 0:
                temporary.unlink(missing_ok=True)
                raise Nutrition5kError(f"downloaded an empty file from {url}")
            temporary.replace(destination)
        except urllib.error.HTTPError as exc:
            temporary.unlink(missing_ok=True)
            raise Nutrition5kError(
                f"failed to fetch required Nutrition5k file {relative} from {url}: {exc}"
            ) from exc
        except (OSError, urllib.error.URLError) as exc:
            temporary.unlink(missing_ok=True)
            raise Nutrition5kError(
                f"failed to fetch required Nutrition5k file {relative} from {url}: {exc}"
            ) from exc
        downloaded += 1

    return FetchSummary(downloaded=downloaded, skipped=skipped)
