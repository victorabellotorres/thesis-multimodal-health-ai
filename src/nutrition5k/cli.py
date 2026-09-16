from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .dataset import DatasetConfig, DatasetMode, Nutrition5kDataset
from .download import download_side_angle_videos
from .errors import Nutrition5kError
from .baseline import AverageBaseline
from .evaluation import evaluate_predictions, write_evaluation, write_predictions
from .frames import extract_sampled_frames


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and prepare Nutrition5k")
    parser.add_argument(
        "command",
        choices=(
            "smoke",
            "download-videos",
            "extract-frames",
            "prepare-imagery",
            "average-baseline",
        ),
        help="operation to run",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/nutrition5k"),
        help="extracted official dataset root (default: data/nutrition5k)",
    )
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument(
        "--require-frames",
        action="store_true",
        help="make smoke fail if a selected dish has no sampled frames",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing extracted frames"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for baseline artifacts (required for average-baseline)",
    )
    return parser


def _representative(record: object) -> dict[str, object]:
    values = asdict(record)  # type: ignore[arg-type]
    values["frame_paths"] = [str(path) for path in values["frame_paths"]]
    return values


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = DatasetConfig(
        root=args.root,
        mode=DatasetMode(args.mode),
        pilot_seed=args.seed,
        frame_stride=args.frame_stride,
        require_frames=args.require_frames if args.command == "smoke" else False,
    )
    try:
        dataset = Nutrition5kDataset.load(config)
        selected = dataset.train + dataset.test
        if args.command in ("download-videos", "prepare-imagery"):
            download = download_side_angle_videos(
                selected,
                config.root,
                args.overwrite,
                progress=lambda message: print(message, file=sys.stderr),
            )
            print(
                f"videos downloaded: {download.downloaded}; "
                f"already present: {download.skipped}; "
                f"missing upstream: {download.missing}"
            )
            if args.command == "download-videos":
                return 0

        if args.command in ("extract-frames", "prepare-imagery"):
            count = extract_sampled_frames(
                selected, config.root, config.frame_stride, args.overwrite
            )
            print(f"extracted sampled frames for {count} dishes")
            return 0

        if args.command == "average-baseline":
            if args.output_dir is None:
                raise Nutrition5kError("--output-dir is required for average-baseline")
            output_dir = args.output_dir.expanduser().resolve()
            parameters_path = output_dir / "parameters.json"
            predictions_path = output_dir / "predictions.csv"
            evaluation_path = output_dir / "evaluation.json"
            run_path = output_dir / "run.json"
            existing = [path for path in (parameters_path, predictions_path, evaluation_path, run_path) if path.exists()]
            if existing and not args.overwrite:
                raise Nutrition5kError(
                    f"baseline output already exists at {output_dir}; use --overwrite to replace it"
                )
            # Only this call receives training records. The test split is used
            # after fitting exclusively for IDs, prediction, and evaluation.
            fitted = AverageBaseline.fit(dataset.train)
            fitted.save(parameters_path)
            loaded = AverageBaseline.load(parameters_path)
            predictions = loaded.predict(record.dish_id for record in dataset.test)
            write_predictions(predictions_path, predictions)
            evaluation = evaluate_predictions(dataset.test, predictions)
            write_evaluation(evaluation_path, evaluation)
            artifact = fitted.artifact()
            run = {
                "schema_version": 1,
                "model": "average_baseline",
                "dataset_mode": config.mode.value,
                "dataset_root": str(config.root),
                "split_counts": {"train": len(dataset.train), "test": len(dataset.test)},
                "target_names": artifact["target_names"],
                "target_units": artifact["target_units"],
                "configuration": {
                    "pilot_seed": config.pilot_seed if config.mode is DatasetMode.PILOT else None,
                    "pilot_train_size": config.pilot_train_size if config.mode is DatasetMode.PILOT else None,
                    "pilot_test_size": config.pilot_test_size if config.mode is DatasetMode.PILOT else None,
                },
                "artifacts": {
                    "parameters": str(parameters_path),
                    "predictions": str(predictions_path),
                    "evaluation": str(evaluation_path),
                },
            }
            run_path.parent.mkdir(parents=True, exist_ok=True)
            run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(run, indent=2, sort_keys=True))
            return 0

        summary = {
            "mode": config.mode.value,
            "root": str(config.root),
            "pilot_seed": config.pilot_seed if config.mode is DatasetMode.PILOT else None,
            "frame_stride": config.frame_stride,
            "counts": {"train": len(dataset.train), "test": len(dataset.test)},
            "selected_ids": {
                "train": [record.dish_id for record in dataset.train],
                "test": [record.dish_id for record in dataset.test],
            },
            "representative_records": {
                "train": _representative(dataset.train[0]),
                "test": _representative(dataset.test[0]),
            },
        }
        print(json.dumps(summary, indent=2))
        return 0
    except Nutrition5kError as exc:
        print(f"nutrition5k: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
