from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from .clear import clear_preparation
from .dataset import IMAGE_SIZE, OVERHEAD_DIR, OVERHEAD_FILES, TARGET_NAMES, load_dataset
from .download import download_overhead
from .errors import Nutrition5kError
from .evaluation import TARGET_UNITS
from .fetch import fetch_dataset_index
from .models.families import BACKBONES, FAMILY, FIXED_SIZE
from .preprocess import write_preprocess
from .training import run_average_baseline

DATA_ROOT = Path("data/nutrition5k")
OUTPUT_ROOT = Path("outputs/average-baseline")
# Trained runs go to outputs/<family>/<split>/<backbone>/<run>/.
RUNS_ROOT = Path("outputs")
# Dishes per download batch; `prepare` reports its progress after each one.
PREPARE_BATCH = 100


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nutrition5k thesis experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("clear", help="remove local Nutrition5k data preparation")
    commands.add_parser("fetch", help="download required metadata and overhead splits")
    check = commands.add_parser("check", help="check the local dataset")
    check.add_argument("--all", action="store_true", help="list every affected dish")
    commands.add_parser("prepare", help="download the overhead images, then preprocess")
    commands.add_parser(
        "preprocess",
        help="list dishes with impossible labels (left out of train and validation) and scan groups",
    )
    commands.add_parser("baseline", help="run the average baseline")
    train = commands.add_parser("train", help="train a regressor on a backbone of any family")
    train.add_argument("--backbone", choices=BACKBONES, default="v4s")
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
    train.add_argument("--backbone-lr", type=float, default=1e-4)
    train.add_argument("--head-width", type=int, default=256, help="hidden layer of each head; 0: linear")
    train.add_argument("--pooling", choices=("cls", "mean", "cls+mean"), help="plain ViTs only")
    train.add_argument("--augment", choices=("flip", "d4"), default="flip")
    train.add_argument(
        "--all-seeds",
        action="store_true",
        help="run every fixed seed (0, 1, 2) and write mean ± SD across them",
    )
    heads = commands.add_parser(
        "heads",
        help="stage 1: train regression heads on a frozen backbone, from cached features",
    )
    heads.add_argument("--backbone", choices=BACKBONES, required=True)
    heads.add_argument(
        "--grid", action="store_true", help="the 9 heads of the reduced grid (ignores the head options)"
    )
    heads.add_argument("--heads", choices=("single", "grouped", "per-target"), default="single")
    heads.add_argument(
        "--hidden", type=int, action="append", help="shared FC layer size, repeatable; 0 for none (default: 512)"
    )
    heads.add_argument("--head-width", type=int, default=256, help="hidden layer of each head; 0: linear")
    heads.add_argument("--pooling", choices=("cls", "mean", "cls+mean"), help="plain ViTs only (default: cls, as timm)")
    heads.add_argument("--augment", choices=("flip", "d4"), default="flip")
    heads.add_argument("--epochs", type=int, default=80)
    heads.add_argument("--seed", type=int, default=0)
    heads.add_argument("--num-workers", type=int, default=4, help="for the one-off feature extraction")
    predict = commands.add_parser(
        "predict", help="estimate one dish from its overhead image with a trained run"
    )
    predict.add_argument(
        "--run", type=Path, required=True, help="run directory, e.g. outputs/mobilenet/full/v3/<run>"
    )
    predict.add_argument("image", type=Path, help="overhead rgb.png of one dish (640x480)")
    for name in ("check", "prepare", "baseline", "train", "heads"):
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
            return _prepare(args.full)
        if args.command == "preprocess":
            return _preprocess(DATA_ROOT)

        if args.command == "check":
            return _check(args.full, args.all)

        if args.command == "train":
            return _train(args)
        if args.command == "heads":
            return _heads(args)
        if args.command == "predict":
            return _predict(args.run, args.image)

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


def _check(full: bool, show_all: bool = False) -> int:
    """Report overhead image coverage; return 1 if any dish has no RGB image or an unexpected size."""
    dataset = load_dataset(DATA_ROOT, full=full)
    print(f"{dataset.mode}: {len(dataset.train)} train, {len(dataset.test)} test")
    without_image: list[str] = []
    without_depth: list[str] = []
    wrong_size: list[str] = []
    for split, records in (("train", dataset.train), ("test", dataset.test)):
        with_image = [record for record in records if record.image]
        print(f"  {split}: {len(with_image)}/{len(records)} dishes with an overhead image")
        without_image += [record.dish_id for record in records if not record.image]
        for record in with_image:
            if not (record.image.parent / OVERHEAD_FILES[1]).is_file():
                without_depth.append(record.dish_id)
            with Image.open(record.image) as image:
                if image.size != IMAGE_SIZE:
                    wrong_size.append(f"{record.dish_id} ({image.size[0]}x{image.size[1]})")
    size = "x".join(map(str, IMAGE_SIZE))
    if without_depth:
        print(f"{len(without_depth)} dishes have no depth image (unused by the models): " + _dishes(without_depth, show_all))
    if wrong_size:
        print(f"{len(wrong_size)} images are not {size}, as CROP_BOX assumes: " + _dishes(wrong_size, show_all))
    if without_image:
        print(f"{len(without_image)} dishes have no overhead image: " + _dishes(without_image, show_all))
    if not (without_image or wrong_size):
        print(f"all {len(dataset.train) + len(dataset.test)} dishes have a {size} overhead image")
        return 0
    return 1


def _dishes(dish_ids: list[str], show_all: bool, shown: int = 5) -> str:
    if show_all or len(dish_ids) <= shown:
        return ", ".join(dish_ids)
    return ", ".join(dish_ids[:shown]) + f", ... ({len(dish_ids) - shown} more, see --all)"


def _preprocess(root: Path) -> int:
    excluded, groups = write_preprocess(root)
    print(
        f"preprocess: {excluded} excluded dishes (left out of train and validation), "
        f"{groups} scan groups (validation holds out whole groups), written to {root}"
    )
    return 0


def _prepare(full: bool) -> int:
    """Download the overhead images of the dishes that lack them, then preprocess."""
    dataset = load_dataset(DATA_ROOT, full=full)
    pending = [
        record
        for record in dataset.train + dataset.test
        if not all(
            (dataset.root / OVERHEAD_DIR / record.dish_id / name).is_file() for name in OVERHEAD_FILES
        )
    ]
    downloaded = skipped = missing = 0
    failed: list[str] = []
    for start in range(0, len(pending), PREPARE_BATCH):
        batch = tuple(pending[start : start + PREPARE_BATCH])
        summary = download_overhead(
            batch, dataset.root, progress=lambda message: print(message, file=sys.stderr)
        )
        downloaded += summary.downloaded
        skipped += summary.skipped
        missing += summary.missing
        failed += summary.failed
        print(f"prepared {start + len(batch)}/{len(pending)} dishes", file=sys.stderr)
    print(
        f"{dataset.mode} ready: {len(pending) - len(failed)} dishes prepared, "
        f"{downloaded} files downloaded, {skipped} already present, {missing} unavailable"
    )
    _preprocess(dataset.root)
    if failed:
        print(
            f"{len(failed)} dishes failed and will be retried by the next prepare: "
            + ", ".join(failed),
            file=sys.stderr,
        )
        return 1
    return 0


def _train(args: argparse.Namespace) -> int:
    # Imported here so the data commands work without the optional training extra.
    from .training.regression import TrainConfig, run_seeds, run_training

    family = FAMILY[args.backbone]
    fixed = FIXED_SIZE.get(family)
    if fixed and args.image_size != fixed:
        raise Nutrition5kError(f"{family} backbones run at {fixed} px, not {args.image_size}")
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
        backbone_lr=args.backbone_lr,
        head_width=args.head_width,
        pooling=args.pooling,
        augment=args.augment,
    )
    dataset = load_dataset(DATA_ROOT, full=args.full)
    if not dataset.excluded:
        raise Nutrition5kError(
            "run `nutrition5k preprocess` first: train leaves out the excluded dishes "
            "and validates on whole scan groups"
        )
    run = run_seeds if args.all_seeds else run_training
    output = run(dataset, RUNS_ROOT / family, config)
    print(f"results written to {output}")
    return 0


def _heads(args: argparse.Namespace) -> int:
    from .training.heads import GRID, HeadConfig, extract_features, train_head

    dataset = load_dataset(DATA_ROOT, full=args.full)
    if not dataset.excluded:
        raise Nutrition5kError("run `nutrition5k preprocess` first")
    folder = RUNS_ROOT / FAMILY[args.backbone] / dataset.mode / "heads" / args.backbone
    extract_features(dataset, args.backbone, folder, num_workers=args.num_workers)
    common = dict(backbone=args.backbone, epochs=args.epochs, seed=args.seed)
    if args.grid:
        configs = [HeadConfig(**common, heads=h, hidden=hidden, head_width=w) for hidden, h, w in GRID]
    else:
        hidden = tuple(size for size in args.hidden or [512] if size)
        configs = [
            HeadConfig(
                **common,
                heads=args.heads,
                hidden=hidden,
                head_width=args.head_width,
                pooling=args.pooling,
                augment=args.augment,
            )
        ]
    for config in configs:
        train_head(dataset, folder, config)
    print(f"results written to {folder}")
    return 0


def _predict(run: Path, image: Path) -> int:
    for path in (run / "config.json", run / "best.pt", image):
        if not path.is_file():
            raise Nutrition5kError(f"missing file: {path}")
    with Image.open(image) as opened:
        if opened.size != IMAGE_SIZE:
            raise Nutrition5kError(
                f"{image} is {opened.size[0]}x{opened.size[1]}; the fixed crop needs a "
                f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} overhead image"
            )
    from .training.regression import predict_image

    estimate = predict_image(run, image)
    print(f"{run.name}, {image}:")
    for name, value in zip(TARGET_NAMES, estimate):
        print(f"  {name}: {value:.1f} {TARGET_UNITS[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
