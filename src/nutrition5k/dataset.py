from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from .errors import Nutrition5kError

Split = Literal["train", "test"]
TargetValues = tuple[float, float, float, float, float]

PILOT_SEED = 20260920
PILOT_SIZES = {"train": 32, "test": 8}
FRAME_STRIDE = 5
TARGET_NAMES = (
    "total_mass",
    "total_calories",
    "total_fat",
    "total_carb",
    "total_protein",
)
METADATA_FILES = (
    "metadata/dish_metadata_cafe1.csv",
    "metadata/dish_metadata_cafe2.csv",
)
SPLIT_FILES: dict[Split, str] = {
    "train": "dish_ids/splits/rgb_train_ids.txt",
    "test": "dish_ids/splits/rgb_test_ids.txt",
}


@dataclass(frozen=True)
class DishRecord:
    dish_id: str
    total_mass: float
    total_calories: float
    total_fat: float
    total_carb: float
    total_protein: float
    frame_paths: tuple[Path, ...] = ()

    @property
    def targets(self) -> TargetValues:
        return (
            self.total_mass,
            self.total_calories,
            self.total_fat,
            self.total_carb,
            self.total_protein,
        )


@dataclass(frozen=True)
class Dataset:
    root: Path
    mode: str
    train: tuple[DishRecord, ...]
    test: tuple[DishRecord, ...]


def load_dataset(root: Path, full: bool = False) -> Dataset:
    """Load the official RGB split, or the project's fixed 32/8 pilot."""
    root = root.expanduser().resolve()
    metadata = _load_metadata(root)
    split_ids = {split: _load_split(root, split) for split in SPLIT_FILES}

    overlap = set(split_ids["train"]) & set(split_ids["test"])
    if overlap:
        raise Nutrition5kError(
            f"official train/test splits overlap (for example {min(overlap)})"
        )

    if not full:
        split_ids = {
            split: _select_pilot(ids, PILOT_SIZES[split], split)
            for split, ids in split_ids.items()
        }

    records: dict[Split, tuple[DishRecord, ...]] = {}
    for split, ids in split_ids.items():
        missing = [dish_id for dish_id in ids if dish_id not in metadata]
        if missing:
            raise Nutrition5kError(
                f"{len(missing)} {split} dish IDs have no metadata; first: {missing[0]}"
            )
        records[split] = tuple(
            replace(metadata[dish_id], frame_paths=_frame_paths(root, dish_id))
            for dish_id in ids
        )

    return Dataset(
        root=root,
        mode="full" if full else "pilot",
        train=records["train"],
        test=records["test"],
    )


def _required_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise Nutrition5kError(f"required Nutrition5k file is missing: {path}")
    return path


def _load_metadata(root: Path) -> dict[str, DishRecord]:
    records: dict[str, DishRecord] = {}
    for relative in METADATA_FILES:
        path = _required_file(root, relative)
        with path.open(newline="", encoding="utf-8") as handle:
            for line_number, row in enumerate(csv.reader(handle), start=1):
                if not row or all(not value.strip() for value in row):
                    continue
                if line_number == 1 and row[0].strip().lower() == "dish_id":
                    continue
                record = _parse_metadata_row(path, line_number, row)
                if record.dish_id in records:
                    raise Nutrition5kError(
                        f"duplicate dish ID {record.dish_id} in {path}:{line_number}"
                    )
                records[record.dish_id] = record
    if not records:
        raise Nutrition5kError("the Nutrition5k metadata files contain no dishes")
    return records


def _parse_metadata_row(path: Path, line_number: int, row: list[str]) -> DishRecord:
    location = f"{path}:{line_number}"
    if len(row) < 6:
        raise Nutrition5kError(f"malformed metadata row at {location}")

    dish_id = row[0].strip()
    if not dish_id.startswith("dish_"):
        raise Nutrition5kError(f"invalid dish ID at {location}: {dish_id!r}")
    try:
        calories, mass, fat, carb, protein = map(float, row[1:6])
    except ValueError as exc:
        raise Nutrition5kError(f"malformed metadata row at {location}") from exc

    # Released CSVs omit the documented ingredient-count column. Validate both
    # layouts, but keep ingredient details out of the image-model interface.
    ingredient_start = 6
    if len(row) > 6 and not row[6].strip().startswith("ingr_"):
        try:
            ingredient_count = int(row[6])
        except ValueError as exc:
            raise Nutrition5kError(f"malformed metadata row at {location}") from exc
        ingredient_start = 7
    else:
        ingredient_count, remainder = divmod(len(row) - ingredient_start, 7)
        if remainder:
            raise Nutrition5kError(f"malformed metadata row at {location}")
    if ingredient_count < 0 or len(row) != ingredient_start + 7 * ingredient_count:
        raise Nutrition5kError(f"malformed metadata row at {location}")

    return DishRecord(dish_id, mass, calories, fat, carb, protein)


def _load_split(root: Path, split: Split) -> list[str]:
    path = _required_file(root, SPLIT_FILES[split])
    dish_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    dish_ids = [dish_id for dish_id in dish_ids if dish_id]
    if not dish_ids:
        raise Nutrition5kError(f"split file is empty: {path}")
    if any(not dish_id.startswith("dish_") for dish_id in dish_ids):
        raise Nutrition5kError(f"invalid dish ID in split file: {path}")
    if len(dish_ids) != len(set(dish_ids)):
        raise Nutrition5kError(f"duplicate dish ID in split file: {path}")
    return dish_ids


def _select_pilot(dish_ids: list[str], size: int, split: Split) -> list[str]:
    if len(dish_ids) < size:
        raise Nutrition5kError(
            f"pilot requires {size} {split} dishes, but only {len(dish_ids)} exist"
        )
    ranked = sorted(
        dish_ids,
        key=lambda dish_id: hashlib.sha256(
            f"{PILOT_SEED}:{split}:{dish_id}".encode()
        ).digest(),
    )
    return sorted(ranked[:size])


def _frame_paths(root: Path, dish_id: str) -> tuple[Path, ...]:
    directory = (
        root / "imagery" / "side_angles" / dish_id / f"frames_sampled{FRAME_STRIDE}"
    )
    return tuple(sorted((*directory.glob("*.jpeg"), *directory.glob("*.jpg"))))
