from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Iterable, Literal

from .errors import Nutrition5kError

Split = Literal["train", "test"]

METADATA_FILES = (
    "metadata/dish_metadata_cafe1.csv",
    "metadata/dish_metadata_cafe2.csv",
)
SPLIT_FILES: dict[Split, str] = {
    "train": "dish_ids/splits/rgb_train_ids.txt",
    "test": "dish_ids/splits/rgb_test_ids.txt",
}
TARGET_NAMES = (
    "total_mass",
    "total_calories",
    "total_fat",
    "total_carb",
    "total_protein",
)


class DatasetMode(str, Enum):
    PILOT = "pilot"
    FULL = "full"


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    mode: DatasetMode = DatasetMode.PILOT
    pilot_seed: int = 20260916
    pilot_train_size: int = 32
    pilot_test_size: int = 8
    frame_stride: int = 5
    require_frames: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())
        if not isinstance(self.mode, DatasetMode):
            try:
                object.__setattr__(self, "mode", DatasetMode(self.mode))
            except ValueError as exc:
                raise Nutrition5kError(
                    f"invalid dataset mode {self.mode!r}; expected 'pilot' or 'full'"
                ) from exc
        if self.pilot_train_size <= 0 or self.pilot_test_size <= 0:
            raise Nutrition5kError("pilot train and test sizes must both be positive")
        if self.frame_stride <= 0:
            raise Nutrition5kError("frame stride must be positive")


@dataclass(frozen=True)
class DishRecord:
    dish_id: str
    split: Split
    total_mass: float
    total_calories: float
    total_fat: float
    total_carb: float
    total_protein: float
    num_ingredients: int
    frame_paths: tuple[Path, ...] = ()

    @property
    def targets(self) -> tuple[float, float, float, float, float]:
        """Return targets in the stable order given by ``TARGET_NAMES``."""
        return (
            self.total_mass,
            self.total_calories,
            self.total_fat,
            self.total_carb,
            self.total_protein,
        )


class Nutrition5kDataset:
    """Dish-level Nutrition5k view independent of pilot/full selection.

    Nutrition values are labels, not model inputs. Each record optionally lists
    every pre-extracted side-angle RGB frame for the configured sampling stride.
    """

    def __init__(
        self,
        config: DatasetConfig,
        train: Iterable[DishRecord],
        test: Iterable[DishRecord],
    ) -> None:
        self.config = config
        self._records: dict[Split, tuple[DishRecord, ...]] = {
            "train": tuple(train),
            "test": tuple(test),
        }

    @classmethod
    def load(cls, config: DatasetConfig) -> "Nutrition5kDataset":
        metadata = _load_metadata(config.root)
        split_ids = {split: _load_split(config.root, split) for split in SPLIT_FILES}
        overlap = set(split_ids["train"]) & set(split_ids["test"])
        if overlap:
            example = sorted(overlap)[0]
            raise Nutrition5kError(
                f"official train/test splits overlap (for example {example})"
            )

        if config.mode is DatasetMode.PILOT:
            split_ids = {
                "train": _select_pilot_ids(
                    split_ids["train"], config.pilot_train_size, config.pilot_seed, "train"
                ),
                "test": _select_pilot_ids(
                    split_ids["test"], config.pilot_test_size, config.pilot_seed, "test"
                ),
            }

        records: dict[Split, list[DishRecord]] = {"train": [], "test": []}
        for split in ("train", "test"):
            missing = [dish_id for dish_id in split_ids[split] if dish_id not in metadata]
            if missing:
                raise Nutrition5kError(
                    f"{len(missing)} {split} dish IDs have no metadata; first: {missing[0]}"
                )
            for dish_id in split_ids[split]:
                record = replace(metadata[dish_id], split=split)
                frames = _frame_paths(config, dish_id)
                if config.require_frames and not frames:
                    expected = _frame_directory(config, dish_id)
                    raise Nutrition5kError(
                        f"no sampled frames found for {dish_id}; expected JPEGs in {expected}"
                    )
                records[split].append(replace(record, frame_paths=frames))
        return cls(config, records["train"], records["test"])

    def records(self, split: Split) -> tuple[DishRecord, ...]:
        try:
            return self._records[split]
        except KeyError as exc:
            raise Nutrition5kError(
                f"invalid split {split!r}; expected 'train' or 'test'"
            ) from exc

    @property
    def train(self) -> tuple[DishRecord, ...]:
        return self._records["train"]

    @property
    def test(self) -> tuple[DishRecord, ...]:
        return self._records["test"]


def _required_path(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise Nutrition5kError(
            f"required Nutrition5k file is missing: {path}; check --root and dataset setup"
        )
    return path


def _load_metadata(root: Path) -> dict[str, DishRecord]:
    records: dict[str, DishRecord] = {}
    for relative in METADATA_FILES:
        path = _required_path(root, relative)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for line_number, row in enumerate(reader, start=1):
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
        raise Nutrition5kError("the Nutrition5k metadata files contain no dish records")
    return records


def _parse_metadata_row(path: Path, line_number: int, row: list[str]) -> DishRecord:
    location = f"{path}:{line_number}"
    if len(row) < 6:
        raise Nutrition5kError(f"malformed metadata row at {location}: expected at least 6 fields")
    dish_id = row[0].strip()
    if not dish_id.startswith("dish_"):
        raise Nutrition5kError(f"invalid dish ID at {location}: {dish_id!r}")
    try:
        total_calories, total_mass, total_fat, total_carb, total_protein = (
            float(value) for value in row[1:6]
        )
    except ValueError as exc:
        raise Nutrition5kError(
            f"non-numeric nutrition value at {location}: {exc}"
        ) from exc

    # The official README documents a num_ingrs field at index 6, while the
    # released CSV files omit it. Support both layouts and infer the count for
    # the released files.
    ingredient_start = 6
    if len(row) > 6 and not row[6].strip().startswith("ingr_"):
        try:
            num_ingredients = int(row[6])
        except ValueError as exc:
            raise Nutrition5kError(
                f"invalid ingredient count at {location}: {row[6]!r}"
            ) from exc
        ingredient_start = 7
    else:
        ingredient_fields = len(row) - ingredient_start
        if ingredient_fields % 7:
            raise Nutrition5kError(
                f"malformed ingredient data at {location}: {ingredient_fields} "
                "fields cannot form 7-field ingredient groups"
            )
        num_ingredients = ingredient_fields // 7
    if num_ingredients < 0:
        raise Nutrition5kError(f"negative ingredient count at {location}")
    expected_fields = ingredient_start + (7 * num_ingredients)
    if len(row) != expected_fields:
        raise Nutrition5kError(
            f"malformed ingredient data at {location}: expected {expected_fields} fields "
            f"for {num_ingredients} ingredients, found {len(row)}"
        )
    return DishRecord(
        dish_id=dish_id,
        split="train",
        total_mass=total_mass,
        total_calories=total_calories,
        total_fat=total_fat,
        total_carb=total_carb,
        total_protein=total_protein,
        num_ingredients=num_ingredients,
    )


def _load_split(root: Path, split: Split) -> list[str]:
    path = _required_path(root, SPLIT_FILES[split])
    dish_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    dish_ids = [dish_id for dish_id in dish_ids if dish_id]
    if not dish_ids:
        raise Nutrition5kError(f"split file is empty: {path}")
    invalid = next((dish_id for dish_id in dish_ids if not dish_id.startswith("dish_")), None)
    if invalid:
        raise Nutrition5kError(f"invalid dish ID {invalid!r} in split file {path}")
    if len(set(dish_ids)) != len(dish_ids):
        raise Nutrition5kError(f"duplicate dish ID in split file {path}")
    return dish_ids


def _select_pilot_ids(
    dish_ids: list[str], size: int, seed: int, split: Split
) -> list[str]:
    if len(dish_ids) < size:
        raise Nutrition5kError(
            f"pilot requires {size} {split} dishes, but its split contains {len(dish_ids)}"
        )
    ranked = sorted(
        dish_ids,
        key=lambda dish_id: hashlib.sha256(
            f"{seed}:{split}:{dish_id}".encode("utf-8")
        ).digest(),
    )
    return sorted(ranked[:size])


def _frame_directory(config: DatasetConfig, dish_id: str) -> Path:
    return (
        config.root
        / "imagery"
        / "side_angles"
        / dish_id
        / f"frames_sampled{config.frame_stride}"
    )


def _frame_paths(config: DatasetConfig, dish_id: str) -> tuple[Path, ...]:
    directory = _frame_directory(config, dish_id)
    return tuple(sorted((*directory.glob("*.jpeg"), *directory.glob("*.jpg"))))
