from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .dataset import DishRecord
from .errors import Nutrition5kError


def extract_sampled_frames(
    records: tuple[DishRecord, ...], root: Path, stride: int, overwrite: bool = False
) -> int:
    """Extract every ``stride``-th RGB frame from all four side-angle videos."""
    if shutil.which("ffmpeg") is None:
        raise Nutrition5kError("ffmpeg is required to extract side-angle frames")
    extracted_dishes = 0
    for record in records:
        dish_dir = root / "imagery" / "side_angles" / record.dish_id
        output_dir = dish_dir / f"frames_sampled{stride}"
        videos = [dish_dir / f"camera_{camera}.h264" for camera in "ABCD"]
        missing = [video for video in videos if not video.is_file()]
        if missing:
            raise Nutrition5kError(
                f"missing side-angle video for {record.dish_id}: {missing[0]}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        for camera, video in zip("ABCD", videos):
            output_pattern = output_dir / f"camera_{camera}_frame_%03d.jpeg"
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y" if overwrite else "-n",
                "-i",
                str(video),
                "-vf",
                f"select=not(mod(n\\,{stride}))",
                "-vsync",
                "vfr",
                str(output_pattern),
            ]
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as exc:
                raise Nutrition5kError(
                    f"ffmpeg failed for {video} with exit code {exc.returncode}"
                ) from exc
        extracted_dishes += 1
    return extracted_dishes

