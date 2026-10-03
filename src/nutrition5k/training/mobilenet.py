from __future__ import annotations

import csv
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip

from ..dataset import Dataset, DishRecord, TargetValues, split_validation
from ..evaluation import evaluate_predictions, write_json, write_predictions
from ..models import AverageBaseline
from ..models.mobilenet import BACKBONES, NutritionNet


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

    def run_name(self, dataset_mode: str, frame_mode: str) -> str:
        unfreeze = "all" if self.unfreeze is None else str(self.unfreeze)
        hidden = "x".join(map(str, self.hidden)) or "0"
        return (
            f"{self.backbone}-ft_{unfreeze}-{self.heads}-h{hidden}-{self.image_size}"
            f"-{frame_mode}-{dataset_mode}-s{self.seed}"
        )


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
    data_config = resolve_model_data_config(model.backbone)
    data_config["input_size"] = (3, config.image_size, config.image_size)
    eval_transform = create_transform(**data_config)
    train_transform = Compose([RandomHorizontalFlip(), eval_transform])

    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    head_parameters = [*model.trunk.parameters(), *model.heads.parameters()]
    groups = [{"params": head_parameters, "lr": config.head_lr}]
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": config.backbone_lr})
    optimizer = torch.optim.AdamW(groups)

    write_json(
        output / "config.json",
        {
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
        },
    )

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
        val_pmae = evaluate_predictions(validation, predictions)["mean_pmae"]
        seconds = time.perf_counter() - began
        _append_history(output / "history.csv", (epoch, total / count, val_pmae, seconds))
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
        write_json(output / f"evaluation{name}.json", evaluate_predictions(records, predictions))
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


def _append_history(path: Path, row: tuple) -> None:
    new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(("epoch", "train_loss", "val_mean_pmae", "seconds"))
        writer.writerow(row)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
