from __future__ import annotations

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
from torchvision.transforms import Compose, RandomHorizontalFlip

from ..dataset import TARGET_NAMES, Dataset, DishRecord, TargetValues, split_validation
from ..evaluation import evaluate_predictions, summarize_seeds, write_json, write_predictions
from ..models import AverageBaseline
from ..models.mobilenet import BACKBONES, NutritionNet


# Fixed seeds for repeated runs; their spread gives the run-to-run SD.
SEEDS = (0, 1, 2)
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
    backbone_lr: float = 1e-4
    validation_fraction: float = 0.1

    def base_name(self, dataset_mode: str, frame_mode: str) -> str:
        """Run name without the seed, shared by every seed of one configuration."""
        unfreeze = "all" if self.unfreeze is None else str(self.unfreeze)
        hidden = "x".join(map(str, self.hidden)) or "0"
        return (
            f"{self.backbone}-ft_{unfreeze}-{self.heads}-h{hidden}-{self.image_size}"
            f"-{frame_mode}-{dataset_mode}"
        )

    def run_name(self, dataset_mode: str, frame_mode: str) -> str:
        return f"{self.base_name(dataset_mode, frame_mode)}-s{self.seed}"


class TrainFrames(torch.utils.data.Dataset):
    """One random frame per dish per epoch, with targets divided by the training means."""

    def __init__(self, records, transform, scale: TargetValues) -> None:
        self.records = records
        self.transform = transform
        self.scale = torch.tensor(scale)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(random.choice(record.frame_paths)).convert("RGB")
        return self.transform(image), torch.tensor(record.targets) / self.scale


class EvalFrames(torch.utils.data.Dataset):
    """Every frame of every dish, tagged with the dish position."""

    def __init__(self, records, transform) -> None:
        self.items = [
            (index, path) for index, record in enumerate(records) for path in record.frame_paths
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        dish, path = self.items[index]
        return self.transform(Image.open(path).convert("RGB")), dish


def run_mobilenet(
    dataset: Dataset, frame_mode: str, output_root: Path, config: TrainConfig
) -> Path:
    """Train, select by validation, evaluate on test, and persist every artifact."""
    output = output_root / config.run_name(dataset.mode, frame_mode)
    if (output / "evaluation.json").is_file():
        print(f"already done: {output}")
        return output
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _device()

    train, validation = split_validation(dataset.train, config.validation_fraction)
    test = dataset.test
    dropped = {
        "train": sum(not record.frame_paths for record in train),
        "validation": sum(not record.frame_paths for record in validation),
        "test": sum(not record.frame_paths for record in test),
    }
    train, validation, test = (
        tuple(record for record in records if record.frame_paths)
        for records in (train, validation, test)
    )
    scale = AverageBaseline.fit(train).means

    model = NutritionNet(config.backbone, config.heads, config.hidden)
    model.freeze_backbone(config.unfreeze)
    model.to(device)
    eval_transform = _eval_transform(model, config.image_size)
    train_transform = Compose([RandomHorizontalFlip(), eval_transform])

    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    head_parameters = [*model.trunk.parameters(), *model.heads.parameters()]
    groups = [{"params": head_parameters, "lr": config.head_lr}]
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": config.backbone_lr})
    optimizer = torch.optim.AdamW(groups)

    run_config = {
        **asdict(config),
        "backbone_id": BACKBONES[config.backbone],
        "dataset_mode": dataset.mode,
        "frame_mode": frame_mode,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "dropped_without_frames": dropped,
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
        TrainFrames(train, train_transform, scale),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train) > config.batch_size,
    )
    start, best = 0, float("inf")
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
            torch.save(model.state_dict(), output / "best.pt")
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best": best},
            output / "last.pt",
        )

    model.load_state_dict(torch.load(output / "best.pt", map_location=device))
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
    if tracker:
        tracker.finish()
    return output


@torch.no_grad()
def predict_images(run: Path, images: list[Path]) -> TargetValues:
    """Predict one dish from one or more of its images with a run's best checkpoint."""
    config = json.loads((run / "config.json").read_text())
    model = NutritionNet(
        config["backbone"], config["heads"], tuple(config["hidden"]), pretrained=False
    )
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    model.eval()
    transform = _eval_transform(model, config["image_size"])
    batch = torch.stack([transform(Image.open(path).convert("RGB")) for path in images])
    # Same as evaluation: average the scaled outputs over the images, then undo the scaling.
    mean = model(batch).mean(dim=0) * torch.tensor(config["target_scale"])
    return tuple(mean.tolist())


def run_seeds(
    dataset: Dataset, frame_mode: str, output_root: Path, config: TrainConfig
) -> Path:
    """Run one configuration with every fixed seed and write mean ± SD across them."""
    runs = [
        run_mobilenet(dataset, frame_mode, output_root, replace(config, seed=seed))
        for seed in SEEDS
    ]
    output = output_root / "seeds" / f"{config.base_name(dataset.mode, frame_mode)}.json"
    write_json(
        output,
        {
            "seeds": list(SEEDS),
            "runs": [run.name for run in runs],
            **{
                split: summarize_seeds(
                    [json.loads((run / f"evaluation{suffix}.json").read_text()) for run in runs]
                )
                for split, suffix in (("validation", "_val"), ("test", ""))
            },
        },
    )
    return output


@torch.no_grad()
def _predict(
    model, records: tuple[DishRecord, ...], transform, scale, config: TrainConfig, device
) -> dict[str, TargetValues]:
    """Average the predictions over all frames of each dish."""
    model.eval()
    loader = DataLoader(
        EvalFrames(records, transform),
        batch_size=config.batch_size * 2,
        num_workers=config.num_workers,
    )
    sums = torch.zeros(len(records), 5)
    counts = torch.zeros(len(records), 1)
    for images, dishes in loader:
        outputs = model(images.to(device)).cpu()
        sums.index_add_(0, dishes, outputs)
        counts.index_add_(0, dishes, torch.ones(len(dishes), 1))
    means = sums / counts * torch.tensor(scale)
    return {record.dish_id: tuple(row.tolist()) for record, row in zip(records, means)}


def _eval_transform(model: NutritionNet, image_size: int):
    """The backbone's own preprocessing (resize, crop, normalise) at ``image_size``."""
    data_config = resolve_model_data_config(model.backbone)
    data_config["input_size"] = (3, image_size, image_size)
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
