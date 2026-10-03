from __future__ import annotations

import subprocess
from pathlib import Path

from .dataset import FRAME_DIRS, FRAME_SHORT_SIDE, FRAME_STRIDE, DishRecord, FrameMode
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
    records: tuple[DishRecord, ...], root: Path, mode: FrameMode = "minimal"
) -> None:
    """Extract frames for one of the two supported frame-extraction modes.

    ``complete`` samples every ``FRAME_STRIDE``-th frame from every available
    camera; ``minimal`` takes only the first frame of the first available camera.
    """
    ffmpeg_executable = _ffmpeg_executable()
    for record in records:
        dish_dir = root / "imagery" / "side_angles" / record.dish_id
        output_dir = dish_dir / FRAME_DIRS[mode]
        videos = [
            (camera, dish_dir / f"camera_{camera}.h264") for camera in "ABCD"
        ]
        available_videos = [
            (camera, video) for camera, video in videos if video.is_file()
        ]
        if not available_videos:
            continue  # unavailable upstream; training skips dishes without frames
        output_dir.mkdir(parents=True, exist_ok=True)
        videos_to_extract = available_videos[:1] if mode == "minimal" else available_videos
        # Scale the shorter side to FRAME_SHORT_SIDE, keeping the aspect ratio.
        scale = (
            f"scale='if(gt(iw,ih),-2,{FRAME_SHORT_SIDE})':"
            f"'if(gt(iw,ih),{FRAME_SHORT_SIDE},-2)'"
        )
        frame_filter = (
            ["-frames:v", "1", "-vf", scale]
            if mode == "minimal"
            else ["-vf", f"select=not(mod(n\\,{FRAME_STRIDE})),{scale}", "-vsync", "vfr"]
        )
        for camera, video in videos_to_extract:
            output_pattern = output_dir / f"camera_{camera}_frame_%03d.jpeg"
            existing_frames = tuple(
                output_dir.glob(f"camera_{camera}_frame_*.jpeg")
            )
            if existing_frames:
                continue
            command = [
                ffmpeg_executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-i",
                str(video),
                *frame_filter,
                "-q:v",
                "3",
                str(output_pattern),
            ]
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as exc:
                raise Nutrition5kError(
                    f"ffmpeg failed for {video} with exit code {exc.returncode}"
                ) from exc
