from __future__ import annotations

import subprocess
from pathlib import Path

from .dataset import DishRecord
from .errors import Nutrition5kError


def _ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise Nutrition5kError(
            "imageio-ffmpeg is required to extract side-angle frames; "
            "install the project with 'python3 -m pip install -e .'"
        ) from exc


def extract_sampled_frames(
    records: tuple[DishRecord, ...], root: Path, stride: int, overwrite: bool = False
) -> int:
    """Extract every ``stride``-th RGB frame from all four side-angle videos."""
    ffmpeg_executable = _ffmpeg_executable()
    extracted_dishes = 0
    for record in records:
        dish_dir = root / "imagery" / "side_angles" / record.dish_id
        output_dir = dish_dir / f"frames_sampled{stride}"
        videos = [
            (camera, dish_dir / f"camera_{camera}.h264") for camera in "ABCD"
        ]
        available_videos = [
            (camera, video) for camera, video in videos if video.is_file()
        ]
        if not available_videos:
            raise Nutrition5kError(
                f"no side-angle videos found for {record.dish_id} in {dish_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        for camera, video in available_videos:
            output_pattern = output_dir / f"camera_{camera}_frame_%03d.jpeg"
            existing_frames = tuple(
                output_dir.glob(f"camera_{camera}_frame_*.jpeg")
            )
            if existing_frames and not overwrite:
                continue
            command = [
                ffmpeg_executable,
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
