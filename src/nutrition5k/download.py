from __future__ import annotations

import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .dataset import DishRecord

GCS_BASE_URL = (
    "https://storage.googleapis.com/nutrition5k_dataset/"
    "nutrition5k_dataset/imagery/side_angles"
)
DOWNLOAD_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5


@dataclass(frozen=True)
class DownloadSummary:
    downloaded: int
    skipped: int
    missing: int
    failed: tuple[str, ...] = ()  # dish ids to retry on the next run


def download_side_angle_videos(
    records: tuple[DishRecord, ...],
    root: Path,
    progress: Callable[[str], None] | None = None,
    one_video: bool = False,
) -> DownloadSummary:
    """Download all cameras, or only the first available camera per dish.

    A camera that is absent (404) or empty upstream counts as missing. Any other
    error marks the whole dish as failed and the download moves on.
    """
    downloaded = 0
    skipped = 0
    missing = 0
    failed = []
    for record in records:
        dish_dir = root / "imagery" / "side_angles" / record.dish_id
        dish_dir.mkdir(parents=True, exist_ok=True)
        for camera in "ABCD":
            destination = dish_dir / f"camera_{camera}.h264"
            if destination.is_file() and destination.stat().st_size > 0:
                skipped += 1
                if progress:
                    progress(f"skip {record.dish_id} camera {camera}")
                if one_video:
                    break
                continue

            url = f"{GCS_BASE_URL}/{record.dish_id}/camera_{camera}.h264"
            temporary = destination.with_suffix(destination.suffix + ".part")
            if progress:
                progress(f"download {record.dish_id} camera {camera}")
            try:
                _download(url, temporary)
            except Exception as exc:  # anything unexpected fails this dish, never the whole run
                temporary.unlink(missing_ok=True)
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                    missing += 1
                    if progress:
                        progress(f"missing upstream {record.dish_id} camera {camera}")
                    continue
                failed.append(record.dish_id)
                if progress:
                    progress(f"failed {record.dish_id} camera {camera} from {url}: {exc}")
                break
            if temporary.stat().st_size == 0:
                temporary.unlink()
                missing += 1
                if progress:
                    progress(f"empty upstream {record.dish_id} camera {camera}")
                continue
            temporary.replace(destination)
            downloaded += 1
            if one_video:
                break

    return DownloadSummary(downloaded, skipped, missing, tuple(failed))


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
