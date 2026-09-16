from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .dataset import DatasetConfig, DatasetMode, Nutrition5kDataset
from .errors import Nutrition5kError
from .frames import extract_sampled_frames


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and prepare Nutrition5k")
    parser.add_argument(
        "command", choices=("smoke", "extract-frames"), help="operation to run"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/nutrition5k"),
        help="extracted official dataset root (default: data/nutrition5k)",
    )
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument(
        "--require-frames",
        action="store_true",
        help="make smoke fail if a selected dish has no sampled frames",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing extracted frames"
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
        require_frames=args.require_frames,
    )
    try:
        dataset = Nutrition5kDataset.load(config)
        if args.command == "extract-frames":
            selected = dataset.train + dataset.test
            count = extract_sampled_frames(
                selected, config.root, config.frame_stride, args.overwrite
            )
            print(f"extracted sampled frames for {count} dishes")
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

