from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .clear import clear_preparation
from .dataset import FRAME_MODES, FrameMode, load_dataset
from .download import download_side_angle_videos
from .errors import Nutrition5kError
from .fetch import fetch_dataset_index
from .frames import extract_sampled_frames
from .training import run_average_baseline

DATA_ROOT = Path("data/nutrition5k")
OUTPUT_ROOT = Path("outputs/average-baseline")
MOBILENET_ROOT = Path("outputs/mobilenet")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nutrition5k thesis experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("clear", help="remove local Nutrition5k data preparation")
    commands.add_parser("fetch", help="download required metadata and RGB splits")
    commands.add_parser("check", help="check the local dataset")
    commands.add_parser("prepare", help="download and extract images")
    commands.add_parser("baseline", help="run the average baseline")
    train = commands.add_parser("train", help="train a MobileNet regressor")
    train.add_argument("--backbone", choices=("v2", "v3", "v4s", "v4m"), default="v4s")
    train.add_argument(
        "--unfreeze",
        default="all",
        help="backbone stages to train from the end: 0 (frozen), N, or all",
    )
    train.add_argument(
        "--heads", choices=("single", "grouped", "per-target"), default="single"
    )
    train.add_argument(
        "--hidden",
        type=int,
        action="append",
        help="shared FC layer size, repeatable; 0 for none (default: 512)",
    )
    train.add_argument("--image-size", type=int, default=224)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--seed", type=int, default=0)
    for name in ("check", "prepare", "train"):
        commands.choices[name].add_argument(
            "--mode",
            choices=FRAME_MODES,
            default="minimal",
            help=(
                "complete: every 5th frame of all cameras; "
                "minimal: first frame of the first available camera"
            ),
        )
    for name in ("check", "prepare", "baseline", "train"):
        commands.choices[name].add_argument(
            "--full", action="store_true", help="use the full split instead of the pilot"
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "clear":
            return _clear()
        if args.command == "fetch":
            return _fetch()
        if args.command == "prepare":
            return _prepare(args.full, args.mode)

        if args.command == "check":
            dataset = load_dataset(DATA_ROOT, full=args.full, frame_mode=args.mode)
            frame_count = sum(
                len(record.frame_paths) for record in dataset.train + dataset.test
            )
            print(f"{dataset.mode}: {len(dataset.train)} train, {len(dataset.test)} test")
            print(f"{args.mode} frames: {frame_count}")
            return 0

        if args.command == "train":
            return _train(args)

        dataset = load_dataset(DATA_ROOT, full=args.full)
        output = run_average_baseline(dataset, OUTPUT_ROOT)
        print(f"results written to {output}")
        return 0
    except Nutrition5kError as exc:
        print(f"nutrition5k: {exc}", file=sys.stderr)
        return 2


def _clear() -> int:
    summary = clear_preparation(DATA_ROOT)
    print(
        f"Nutrition5k preparation cleared: {summary.removed} directories removed, "
        f"{summary.absent} already absent"
    )
    return 0


def _fetch() -> int:
    summary = fetch_dataset_index(
        DATA_ROOT, progress=lambda message: print(message, file=sys.stderr)
    )
    print(
        f"dataset index ready: {summary.downloaded} files downloaded, "
        f"{summary.skipped} already present"
    )
    return 0


def _prepare(full: bool, mode: FrameMode) -> int:
    dataset = load_dataset(DATA_ROOT, full=full)
    records = dataset.train + dataset.test
    summary = download_side_angle_videos(
        records,
        dataset.root,
        one_video=mode == "minimal",
        progress=lambda message: print(message, file=sys.stderr),
    )
    extract_sampled_frames(records, dataset.root, mode)
    print(
        f"{dataset.mode} ({mode}) ready: {summary.downloaded} videos downloaded, "
        f"{summary.skipped} already present, {summary.missing} unavailable"
    )
    return 0


def _train(args: argparse.Namespace) -> int:
    # Imported here so the data commands work without the optional training extra.
    from .training.mobilenet import TrainConfig, run_mobilenet

    hidden = tuple(size for size in args.hidden or [512] if size)
    config = TrainConfig(
        backbone=args.backbone,
        unfreeze=None if args.unfreeze == "all" else int(args.unfreeze),
        heads=args.heads,
        hidden=hidden,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    dataset = load_dataset(DATA_ROOT, full=args.full, frame_mode=args.mode)
    output = run_mobilenet(dataset, args.mode, MOBILENET_ROOT, config)
    print(f"results written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
