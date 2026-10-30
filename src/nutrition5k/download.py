from __future__ import annotations

import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .dataset import OVERHEAD_DIR, OVERHEAD_FILES, DishRecord
from .fetch import GCS_BASE_URL

DOWNLOAD_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5
# Parallel downloads: each file is small, so the time goes to waiting on the network.
DOWNLOAD_THREADS = 8


@dataclass(frozen=True)
class DownloadSummary:
    downloaded: int
    skipped: int
    missing: int
    failed: tuple[str, ...] = ()  # dish ids to retry on the next run


def download_overhead(
    records: tuple[DishRecord, ...],
    root: Path,
    progress: Callable[[str], None] | None = None,
) -> DownloadSummary:
    """Download the overhead RGB and raw depth images of every dish.

    A file that is absent (404) or empty upstream counts as missing. Any other
    error marks the whole dish as failed and the download moves on.
    """

    def dish(record: DishRecord) -> tuple[int, int, int, bool]:
        downloaded = skipped = missing = 0
        for name in OVERHEAD_FILES:
            relative = f"{OVERHEAD_DIR}/{record.dish_id}/{name}"
            destination = root / relative
            if destination.is_file() and destination.stat().st_size > 0:
                skipped += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            try:
                _download(f"{GCS_BASE_URL}/{relative}", temporary)
            except Exception as exc:  # anything unexpected fails this dish, never the whole run
                temporary.unlink(missing_ok=True)
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                    missing += 1
                    if progress:
                        progress(f"missing upstream {relative}")
                    continue
                if progress:
                    progress(f"failed {relative}: {exc}")
                return downloaded, skipped, missing, True
            if temporary.stat().st_size == 0:
                temporary.unlink()
                missing += 1
                if progress:
                    progress(f"empty upstream {relative}")
                continue
            temporary.replace(destination)
            downloaded += 1
        return downloaded, skipped, missing, False

    with ThreadPoolExecutor(DOWNLOAD_THREADS) as pool:
        results = list(pool.map(dish, records))
    return DownloadSummary(
        downloaded=sum(result[0] for result in results),
        skipped=sum(result[1] for result in results),
        missing=sum(result[2] for result in results),
        failed=tuple(record.dish_id for record, result in zip(records, results) if result[3]),
    )


def _download(url: str, destination: Path) -> None:
    """Retry transient network errors (timeouts, resets); HTTP errors are final."""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                with destination.open("wb") as output:
                    shutil.copyfileobj(response, output)
            return
        except urllib.error.HTTPError:
            raise
        except OSError:
            if attempt == DOWNLOAD_ATTEMPTS:
                raise
            time.sleep(RETRY_DELAY_SECONDS * attempt)
