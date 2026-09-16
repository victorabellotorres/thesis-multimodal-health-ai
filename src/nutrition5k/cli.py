from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .baseline import AverageBaseline
from .dataset import TARGET_NAMES, Dataset, load_dataset
from .download import download_side_angle_videos
from .errors import Nutrition5kError
from .evaluation import TARGET_UNITS, evaluate_predictions, write_json, write_predictions
from .frames import extract_sampled_frames

DATA_ROOT = Path("data/nutrition5k")
OUTPUT_ROOT = Path("outputs/average-baseline")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nutrition5k thesis experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="check the local dataset")
    commands.add_parser("prepare", help="download and extract images")
    commands.add_parser("baseline", help="run the average baseline")
    for name in ("check", "prepare", "baseline"):
        commands.choices[name].add_argument(
            "--full", action="store_true", help="use the full split instead of the pilot"
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            return _prepare(args.full)

        dataset = load_dataset(DATA_ROOT, full=args.full)
        if args.command == "check":
            frame_count = sum(
                len(record.frame_paths) for record in dataset.train + dataset.test
            )
            print(f"{dataset.mode}: {len(dataset.train)} train, {len(dataset.test)} test")
            print(f"sampled frames: {frame_count}")
            return 0

        return _run_baseline(dataset)
    except Nutrition5kError as exc:
        print(f"nutrition5k: {exc}", file=sys.stderr)
        return 2


def _prepare(full: bool) -> int:
    dataset = load_dataset(DATA_ROOT, full=full)
    records = dataset.train + dataset.test
    summary = download_side_angle_videos(
        records, dataset.root, progress=lambda message: print(message, file=sys.stderr)
    )
    extract_sampled_frames(records, dataset.root)
    print(
        f"{dataset.mode} ready: {summary.downloaded} videos downloaded, "
        f"{summary.skipped} already present, {summary.missing} unavailable"
    )
    return 0


def _run_baseline(dataset: Dataset) -> int:
    model = AverageBaseline.fit(dataset.train)
    predictions = model.predict(record.dish_id for record in dataset.test)
    output = OUTPUT_ROOT / dataset.mode

    write_json(
        output / "model.json",
        {
            "training_count": model.training_count,
            "means": dict(zip(TARGET_NAMES, model.means, strict=True)),
            "units": TARGET_UNITS,
        },
    )
    write_predictions(output / "predictions.csv", predictions)
    write_json(output / "evaluation.json", evaluate_predictions(dataset.test, predictions))
    print(f"results written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
