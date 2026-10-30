from __future__ import annotations

import copy
import csv
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ..dataset import CROP_BOX, TARGET_NAMES, Dataset, DishRecord, TargetValues, split_validation
from ..evaluation import evaluate_predictions, summarize_seeds, write_json, write_predictions
from ..models import AverageBaseline
from ..models.nutrition_net import BACKBONES, HEAD_WIDTH, NutritionNet


# Fixed seeds for repeated runs; their spread gives the run-to-run SD.
SEEDS = (0, 1, 2)
DEFAULT_BACKBONE_LR = 1e-4
HISTORY_COLUMNS = (
    "epoch",
    "train_loss",
    "val_loss",
    "val_mean_pmae",
    *(f"val_{name}_pmae" for name in TARGET_NAMES),
    "seconds",
    "gpu_memory_gb",
)


@dataclass(frozen=True)
class TrainConfig:
    backbone: str = "v4s"
    unfreeze: int | None = None  # None trains the whole backbone
    heads: str = "single"
    hidden: tuple[int, ...] = (512,)
    image_size: int = 224
    epochs: int = 30
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 0
    head_lr: float = 1e-3
    backbone_lr: float = DEFAULT_BACKBONE_LR
    validation_fraction: float = 0.1
    head_width: int = HEAD_WIDTH  # hidden layer of each head; 0 makes the heads linear
    pooling: str | None = None  # plain ViTs only; None keeps timm's CLS token
    augment: str = "flip"  # "flip": random horizontal flip; "d4": random rotation by 90° and flip

    def base_name(self, dataset_mode: str) -> str:
        """Run name without the seed, shared by every seed of one configuration."""
        unfreeze = "all" if self.unfreeze is None else str(self.unfreeze)
        hidden = "x".join(map(str, self.hidden)) or "0"
        # Non-default values add a suffix, so the default names stay short and unchanged.
        extra = "".join(
            (
                "" if self.head_width == HEAD_WIDTH else f"-hw{self.head_width}",
                f"-{self.pooling}" if self.pooling else "",
                "" if self.augment == "flip" else f"-{self.augment}",
                "" if self.backbone_lr == DEFAULT_BACKBONE_LR else f"-blr{self.backbone_lr:g}",
            )
        )
        return f"{self.backbone}-ft_{unfreeze}-{self.heads}-h{hidden}-{self.image_size}{extra}-{dataset_mode}"

    def run_name(self, dataset_mode: str) -> str:
        return f"{self.base_name(dataset_mode)}-s{self.seed}"


class DishImages(torch.utils.data.Dataset):
    """The overhead image of each dish, with its targets divided by the training means."""

    def __init__(self, records, transform, scale: TargetValues) -> None:
        self.records = records
        self.transform = transform
        self.scale = torch.tensor(scale)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.image).convert("RGB")
        return self.transform(image), torch.tensor(record.targets) / self.scale


def run_training(dataset: Dataset, output_root: Path, config: TrainConfig) -> Path:
    """Train, select by validation, evaluate on test, and persist every artifact."""
    name = config.run_name(dataset.mode)
    output = output_root / dataset.mode / config.backbone / name
    if (output / "evaluation.json").is_file():
        print(f"already done: {output}")
        return output
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _device()

    train, validation, test, dropped = splits(dataset, config.validation_fraction)
    scale = AverageBaseline.fit(train).means

    model = NutritionNet(
        config.backbone, config.heads, config.hidden, head_width=config.head_width, pooling=config.pooling
    )
    model.freeze_backbone(config.unfreeze)
    model.to(device)
    eval_transform = _transform(model, config.image_size)
    train_transform = _transform(model, config.image_size, augment=config.augment)

    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    head_parameters = list(model.head.parameters())
    groups = [{"params": head_parameters, "lr": config.head_lr}]
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": config.backbone_lr})
    optimizer = torch.optim.AdamW(groups)

    run_config = {
        **asdict(config),
        "backbone_id": BACKBONES[config.backbone],
        "dataset_mode": dataset.mode,
        "crop_box": list(CROP_BOX),  # (x0, y0, side) of the overhead image, resized to image_size
        "batchnorm": "frozen",  # backbone BatchNorm statistics stay at their ImageNet values
        "drop_rate": 0.0,  # dropout and stochastic depth are off for every backbone
        "drop_path_rate": 0.0,
        "weight_decay": optimizer.defaults["weight_decay"],  # AdamW default, on every parameter
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "dropped_without_image": dropped,
        "excluded_dishes": sorted(dataset.excluded),  # label errors, out of train and validation
        "validation_split": "scan groups" if dataset.groups else "dishes",
        "target_scale": scale,
        "environment": {
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "torch": torch.__version__,
            "timm": timm.__version__,
            "git_commit": _git_commit(),
            "started_at": datetime.now().isoformat(timespec="seconds"),  # latest (re)start
        },
    }
    write_json(output / "config.json", run_config)
    tracker = _wandb_run(output.name, run_config)

    loader = DataLoader(
        DishImages(train, train_transform, scale),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train) > config.batch_size,
    )
    start, best = 0, float("inf")
    # The global volume is eventually consistent: reading best.pt back right after writing it
    # can return an older version, so the best weights are also kept in memory.
    best_state = None
    if (output / "last.pt").is_file():
        state = torch.load(output / "last.pt", map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["epoch"] + 1, state["best"]
        print(f"resuming {output.name} at epoch {start}")

    for epoch in range(start, config.epochs):
        began = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        total, count = 0.0, 0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            loss = torch.nn.functional.l1_loss(model(images), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
            count += len(images)

        predictions = _predict(model, validation, eval_transform, scale, config, device)
        evaluation = evaluate_predictions(validation, predictions)
        val_pmae = evaluation["mean_pmae"]
        target_pmaes = [evaluation["metrics"][name]["pmae"] for name in TARGET_NAMES]
        seconds = time.perf_counter() - began
        memory = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None
        row = (
            epoch,
            total / count,
            _scaled_l1(validation, predictions, scale),
            val_pmae,
            *target_pmaes,
            seconds,
            memory,
        )
        _append_history(output / "history.csv", row)
        if tracker:
            tracker.log({k: v for k, v in zip(HISTORY_COLUMNS, row) if v is not None}, step=epoch)
        print(f"epoch {epoch}: train loss {total / count:.4f}, val mean PMAE {val_pmae:.2f}%")
        if val_pmae < best:
            best = val_pmae
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output / "best.pt")
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best": best},
            output / "last.pt",
        )

    # best.pt is read only when no epoch improved after a resume.
    if best_state is None:
        best_state = torch.load(output / "best.pt", map_location=device)
    model.load_state_dict(best_state)
    for name, records in (("_val", validation), ("", test)):
        predictions = _predict(model, records, eval_transform, scale, config, device)
        write_predictions(output / f"predictions{name}.csv", predictions)
        evaluation = evaluate_predictions(records, predictions)
        write_json(output / f"evaluation{name}.json", evaluation)
        if tracker:
            split = "val" if name else "test"
            tracker.summary[f"{split}_mean_pmae"] = evaluation["mean_pmae"]
            for target in TARGET_NAMES:
                tracker.summary[f"{split}_{target}_pmae"] = evaluation["metrics"][target]["pmae"]
    if dataset.excluded:
        # The official test again, without the excluded dishes; reported, never used to choose.
        clean = tuple(record for record in test if record.dish_id not in dataset.excluded)
        write_json(
            output / "evaluation_clean.json",
            evaluate_predictions(clean, {record.dish_id: predictions[record.dish_id] for record in clean}),
        )
    if tracker:
        tracker.finish()
    return output


@torch.no_grad()
def predict_image(run: Path, image: Path) -> TargetValues:
    """Predict one dish from its overhead image with a run's best checkpoint."""
    config = json.loads((run / "config.json").read_text())
    model = NutritionNet(
        config["backbone"],
        config["heads"],
        tuple(config["hidden"]),
        pretrained=False,
        head_width=config.get("head_width", HEAD_WIDTH),
        pooling=config.get("pooling"),
    )
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    model.eval()
    transform = _transform(model, config["image_size"])
    output = model(transform(Image.open(image).convert("RGB")).unsqueeze(0))[0]
    return tuple((output * torch.tensor(config["target_scale"])).tolist())


def run_seeds(dataset: Dataset, output_root: Path, config: TrainConfig) -> Path:
    """Run one configuration with every fixed seed and write mean ± SD across them."""
    runs = [run_training(dataset, output_root, replace(config, seed=seed)) for seed in SEEDS]
    name = config.base_name(dataset.mode)
    output = output_root / dataset.mode / "seeds" / f"{name}.json"
    write_json(
        output,
        {
            "seeds": list(SEEDS),
            "runs": [run.name for run in runs],
            **{
                split: summarize_seeds(
                    [json.loads((run / f"evaluation{suffix}.json").read_text()) for run in runs]
                )
                for split, suffix in (("validation", "_val"), ("test", ""), ("test_clean", "_clean"))
                if (runs[0] / f"evaluation{suffix}.json").is_file()
            },
        },
    )
    return output


@torch.no_grad()
def _predict(
    model, records: tuple[DishRecord, ...], transform, scale, config: TrainConfig, device
) -> dict[str, TargetValues]:
    """One prediction per dish, from its overhead image, in target units."""
    model.eval()
    loader = DataLoader(
        DishImages(records, transform, scale),
        batch_size=config.batch_size * 2,
        num_workers=config.num_workers,
    )
    outputs = torch.cat([model(images.to(device)).cpu() for images, _ in loader])
    outputs = outputs * torch.tensor(scale)
    return {record.dish_id: tuple(row.tolist()) for record, row in zip(records, outputs)}


def splits(dataset: Dataset, validation_fraction: float):
    """Train, validation and test records with an image, and how many each split lost without one."""
    train, validation = split_validation(dataset.train, validation_fraction, dataset.groups)
    records = {"train": train, "validation": validation, "test": dataset.test}
    dropped = {split: sum(not record.image for record in rows) for split, rows in records.items()}
    kept = (tuple(record for record in rows if record.image) for rows in records.values())
    return (*kept, dropped)


def dihedral(image: Image.Image, variant: int) -> Image.Image:
    """One of the 8 symmetries of the square crop: ``variant % 4`` quarter turns, flipped if ``>= 4``.

    A top-down plate stays a valid image under all of them, and the portion size does not change.
    """
    if variant >= 4:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    turns = (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)
    return image.transpose(turns[variant % 4]) if variant % 4 else image


# Variants each augmentation draws from: identity and horizontal flip, or all 8 symmetries.
AUGMENT_VARIANTS = {"flip": (0, 4), "d4": tuple(range(8))}


class RandomDihedral:
    """Apply one of ``variants`` at random (a class, not a lambda, so DataLoader workers can pickle it)."""

    def __init__(self, variants: tuple[int, ...]) -> None:
        self.variants = variants

    def __call__(self, image: Image.Image) -> Image.Image:
        return dihedral(image, random.choice(self.variants))


def crop_plate(image: Image.Image) -> Image.Image:
    """The fixed square of the overhead image that holds the plate (``CROP_BOX``)."""
    x0, y0, side = CROP_BOX
    return image.crop((x0, y0, x0 + side, y0 + side))


def _transform(model: NutritionNet, image_size: int, augment: str | None = None):
    """Crop the plate, then the backbone's own resize and normalisation at ``image_size``.

    Training (``augment`` set) applies a random variant from ``AUGMENT_VARIANTS`` after the crop,
    so the plate stays inside it.
    """
    steps = [crop_plate]
    if augment:
        steps.append(RandomDihedral(AUGMENT_VARIANTS[augment]))
    return Compose([*steps, _backbone_transform(model, image_size)])


def _backbone_transform(model: NutritionNet, image_size: int):
    """The backbone's own resize and normalisation of the square plate crop at ``image_size``."""
    data_config = resolve_model_data_config(model.backbone)
    data_config["input_size"] = (3, image_size, image_size)
    data_config["crop_pct"] = 1.0  # resize the whole square, without a second centre crop
    return create_transform(**data_config)


def _scaled_l1(records, predictions, scale) -> float:
    """The training loss on dish-level predictions: mean |error| / training mean."""
    errors = [
        abs(estimate - truth) / mean
        for record in records
        for estimate, truth, mean in zip(predictions[record.dish_id], record.targets, scale)
    ]
    return sum(errors) / len(errors)


def _append_history(path: Path, row: tuple) -> None:
    new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(HISTORY_COLUMNS)
        writer.writerow(row)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _wandb_run(name: str, config: dict):
    """Mirror the run to Weights & Biases only when WANDB_API_KEY is set.

    The run name doubles as the W&B id, so a resumed run continues the same
    curves. Local files remain the record of the results.
    """
    if not os.environ.get("WANDB_API_KEY"):
        return None
    import wandb

    return wandb.init(project="nutrition5k", name=name, id=name, resume="allow", config=config)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
