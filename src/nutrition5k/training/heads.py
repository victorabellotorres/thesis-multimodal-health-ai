"""Stage 1: regression heads on frozen backbones, trained on cached features.

The backbone runs once per image and symmetry (``dihedral``); every head then trains in seconds on
the saved features, so many head configurations can be compared on every backbone. A head trained
here is the ``--unfreeze 0`` model of ``regression.run_training``, without re-running the backbone.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ..dataset import CROP_BOX, TARGET_NAMES, Dataset
from ..evaluation import evaluate_predictions, write_json, write_predictions
from ..models import AverageBaseline
from ..models.nutrition_net import BACKBONES, HEAD_WIDTH, NutritionNet, RegressionHead, pool_tokens
from .regression import (
    AUGMENT_VARIANTS,
    _append_history,
    _backbone_transform,
    _device,
    _git_commit,
    _scaled_l1,
    crop_plate,
    dihedral,
    splits,
)

IMAGE_SIZE = 224
VARIANTS = 8
# The 9 heads of the reduced grid: trunk × (linear single, single with a hidden layer, grouped).
# A linear grouped head is identical to a linear single one, so it is not repeated.
GRID = tuple(
    (hidden, heads, width)
    for hidden in ((), (512,), (1024, 1024))
    for heads, width in (("single", 0), ("single", HEAD_WIDTH), ("grouped", HEAD_WIDTH))
)


@dataclass(frozen=True)
class HeadConfig:
    backbone: str
    heads: str = "single"
    hidden: tuple[int, ...] = (512,)
    head_width: int = HEAD_WIDTH
    pooling: str | None = None  # plain ViTs: "cls" (timm's default), "mean" or "cls+mean"
    augment: str = "flip"
    epochs: int = 80
    batch_size: int = 32
    seed: int = 0
    lr: float = 1e-3
    validation_fraction: float = 0.1

    def run_name(self, dataset_mode: str) -> str:
        hidden = "x".join(map(str, self.hidden)) or "0"
        pooling = f"-{self.pooling}" if self.pooling else ""
        return (
            f"{self.backbone}-frozen-{self.heads}-h{hidden}-hw{self.head_width}{pooling}"
            f"-{self.augment}-{dataset_mode}-s{self.seed}"
        )


class _Variant(torch.utils.data.Dataset):
    """Every image with one fixed symmetry, preprocessed for the backbone."""

    def __init__(self, records, variant: int, transform) -> None:
        self.records, self.variant, self.transform = records, variant, transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        image = crop_plate(Image.open(self.records[index].image).convert("RGB"))
        return self.transform(dihedral(image, self.variant))


@torch.no_grad()
def extract_features(dataset: Dataset, backbone: str, folder: Path, batch_size=64, num_workers=4) -> Path:
    """Save the frozen backbone's features of every image under its 8 symmetries (``features.pt``).

    Plain ViTs keep both the CLS token and the mean of the patch tokens, so pooling can be compared.
    """
    path = folder / "features.pt"
    if path.is_file():
        return path
    device = _device()
    model = NutritionNet(backbone).to(device).eval()
    records = tuple(record for record in dataset.train + dataset.test if record.image)
    vit = hasattr(model.backbone, "num_prefix_tokens")
    names = ("cls", "mean") if vit else ("pooled",)
    features = {name: [] for name in names}
    began = time.perf_counter()
    for variant in range(VARIANTS):
        loader = DataLoader(
            _Variant(records, variant, _backbone_transform(model, IMAGE_SIZE)),
            batch_size=batch_size,
            num_workers=num_workers,
        )
        outputs = {name: [] for name in names}
        for images in loader:
            images = images.to(device)
            if vit:
                tokens = model.backbone.forward_features(images)
                prefix = model.backbone.num_prefix_tokens
                outputs["cls"].append(pool_tokens(tokens, prefix, "cls").cpu())
                outputs["mean"].append(pool_tokens(tokens, prefix, "mean").cpu())
            else:
                outputs["pooled"].append(model.backbone(images).cpu())
        for name in names:
            features[name].append(torch.cat(outputs[name]))
        print(f"{backbone}: variant {variant + 1}/{VARIANTS} extracted")
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dish_ids": [record.dish_id for record in records],
            # (dishes, variants, features), float16 to keep the file small
            "features": {name: torch.stack(rows, dim=1).half() for name, rows in features.items()},
            "backbone_id": BACKBONES[backbone],
            "image_size": IMAGE_SIZE,
            "crop_box": list(CROP_BOX),
            "seconds": time.perf_counter() - began,
            "device": str(device),
        },
        path,
    )
    return path


def train_head(dataset: Dataset, folder: Path, config: HeadConfig) -> Path:
    """Train one head on the cached features of ``folder``, select by validation, evaluate on test."""
    output = folder / config.run_name(dataset.mode)
    if (output / "evaluation.json").is_file():
        print(f"already done: {output}")
        return output
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _device()

    cache = torch.load(folder / "features.pt")
    stored = cache["features"]
    if config.pooling and "cls" not in stored:
        raise ValueError(f"{config.backbone} has no CLS token; pooling applies to plain ViTs only")
    if "pooled" in stored:
        features = stored["pooled"]
    elif config.pooling == "cls+mean":
        features = torch.cat((stored["cls"], stored["mean"]), dim=2)
    else:  # timm's default for ViTs is the CLS token
        features = stored["mean" if config.pooling == "mean" else "cls"]
    features = features.float().to(device)
    row = {dish_id: index for index, dish_id in enumerate(cache["dish_ids"])}

    train, validation, test, dropped = splits(dataset, config.validation_fraction)
    scale = AverageBaseline.fit(train).means
    scale_tensor = torch.tensor(scale, device=device)
    train_rows = torch.tensor([row[record.dish_id] for record in train], device=device)
    targets = torch.tensor([record.targets for record in train], device=device) / scale_tensor
    variants = torch.tensor(AUGMENT_VARIANTS[config.augment], device=device)

    head = RegressionHead(features.shape[2], config.heads, config.hidden, config.head_width).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr)
    predict = partial(_predict, head, features, row, scale_tensor)

    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "config.json",
        {
            **asdict(config),
            "backbone_id": BACKBONES[config.backbone],
            "dataset_mode": dataset.mode,
            "unfreeze": 0,  # frozen backbone: the head is the only trained part
            "image_size": cache["image_size"],
            "crop_box": cache["crop_box"],
            "features": features.shape[2],
            "head_parameters": sum(p.numel() for p in head.parameters()),
            "weight_decay": optimizer.defaults["weight_decay"],
            "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
            "dropped_without_image": dropped,
            "excluded_dishes": sorted(dataset.excluded),
            "validation_split": "scan groups" if dataset.groups else "dishes",
            "target_scale": scale,
            "environment": {
                "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                "torch": torch.__version__,
                "timm": timm.__version__,
                "git_commit": _git_commit(),
                "started_at": datetime.now().isoformat(timespec="seconds"),
            },
        },
    )
    if (output / "history.csv").is_file():  # an interrupted head restarts; it only takes seconds
        (output / "history.csv").unlink()

    best, best_state = float("inf"), None
    drop_last = len(train) > config.batch_size
    for epoch in range(config.epochs):
        began = time.perf_counter()
        head.train()
        order = torch.randperm(len(train), device=device)
        total, count = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            if drop_last and len(batch) < config.batch_size:
                break
            # One random symmetry per image, as the image augmentation does in full training.
            variant = variants[torch.randint(len(variants), (len(batch),), device=device)]
            loss = torch.nn.functional.l1_loss(head(features[train_rows[batch], variant]), targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
            count += len(batch)

        predictions = predict(validation)
        evaluation = evaluate_predictions(validation, predictions)
        val_pmae = evaluation["mean_pmae"]
        _append_history(
            output / "history.csv",
            (
                epoch,
                total / count,
                _scaled_l1(validation, predictions, scale),
                val_pmae,
                *(evaluation["metrics"][name]["pmae"] for name in TARGET_NAMES),
                time.perf_counter() - began,
                None,  # gpu_memory_gb, not tracked for heads
            ),
        )
        if val_pmae < best:
            best, best_state = val_pmae, copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    torch.save(best_state, output / "best.pt")
    for name, records in (("_val", validation), ("", test)):
        predictions = predict(records)
        write_predictions(output / f"predictions{name}.csv", predictions)
        write_json(output / f"evaluation{name}.json", evaluate_predictions(records, predictions))
    if dataset.excluded:
        clean = tuple(record for record in test if record.dish_id not in dataset.excluded)
        write_json(
            output / "evaluation_clean.json",
            evaluate_predictions(clean, {record.dish_id: predictions[record.dish_id] for record in clean}),
        )
    print(f"{output.name}: best val mean PMAE {best:.2f}%")
    return output


@torch.no_grad()
def _predict(head, features, row, scale, records):
    """One prediction per dish from its original image (variant 0), in target units."""
    head.eval()
    rows = torch.tensor([row[record.dish_id] for record in records], device=features.device)
    outputs = (head(features[rows, 0]) * scale).cpu()
    return {record.dish_id: tuple(values.tolist()) for record, values in zip(records, outputs)}

