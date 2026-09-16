from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .dataset import DishRecord
from .errors import Nutrition5kError

GCS_BASE_URL = (
    "https://storage.googleapis.com/nutrition5k_dataset/"
    "nutrition5k_dataset/imagery/side_angles"
)


@dataclass(frozen=True)
class DownloadSummary:
    downloaded: int
    skipped: int
    missing: int


def download_side_angle_videos(
    records: tuple[DishRecord, ...],
    root: Path,
    progress: Callable[[str], None] | None = None,
) -> DownloadSummary:
    """Download cameras A-D for selected dishes using atomic local writes."""
    downloaded = 0
    skipped = 0
    missing = 0
    for record in records:
        dish_dir = root / "imagery" / "side_angles" / record.dish_id
        dish_dir.mkdir(parents=True, exist_ok=True)
        available_for_dish = 0
        for camera in "ABCD":
            destination = dish_dir / f"camera_{camera}.h264"
            if destination.is_file() and destination.stat().st_size > 0:
                skipped += 1
                available_for_dish += 1
                if progress:
                    progress(f"skip {record.dish_id} camera {camera}")
                continue

            url = f"{GCS_BASE_URL}/{record.dish_id}/camera_{camera}.h264"
            temporary = destination.with_suffix(destination.suffix + ".part")
            if progress:
                progress(f"download {record.dish_id} camera {camera}")
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    with temporary.open("wb") as output:
                        shutil.copyfileobj(response, output)
                if temporary.stat().st_size == 0:
                    temporary.unlink(missing_ok=True)
                    raise Nutrition5kError(f"downloaded an empty video from {url}")
                temporary.replace(destination)
            except urllib.error.HTTPError as exc:
                temporary.unlink(missing_ok=True)
                if exc.code == 404:
                    missing += 1
                    if progress:
                        progress(f"missing upstream {record.dish_id} camera {camera}")
                    continue
                raise Nutrition5kError(
                    f"failed to download {record.dish_id} camera {camera} from {url}: {exc}"
                ) from exc
            except (OSError, urllib.error.URLError) as exc:
                temporary.unlink(missing_ok=True)
                raise Nutrition5kError(
                    f"failed to download {record.dish_id} camera {camera} from {url}: {exc}"
                ) from exc
            downloaded += 1
            available_for_dish += 1

        if available_for_dish == 0:
            raise Nutrition5kError(
                f"no side-angle videos are available for selected dish {record.dish_id}"
            )

    return DownloadSummary(downloaded=downloaded, skipped=skipped, missing=missing)
