from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from .errors import Nutrition5kError

Split = Literal["train", "test"]
TargetValues = tuple[float, float, float, float, float]

PILOT_SEED = 20260920
PILOT_SIZES = {"train": 32, "test": 8}
# One overhead capture per dish (Intel RealSense D435): RGB and raw 16-bit depth (x 1e-4 = m).
OVERHEAD_DIR = "imagery/realsense_overhead"
OVERHEAD_FILES = ("rgb.png", "depth_raw.png")
IMAGE_SIZE = (640, 480)
# Fixed square (x0, y0, side) of the RGB image that holds the plate, chosen in views_and_crop.ipynb;
# models resize it to their input size (448 -> 224 is exactly half).
CROP_BOX = (146, 0, 448)
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
# Written together by `preprocess`: dishes with impossible labels, left out of train (and so of
# validation), and the scan group of every dish, so validation holds out whole plates.
EXCLUDED_FILE = "excluded_dishes.csv"
GROUPS_FILE = "dish_groups.csv"
# Official split of the dishes with an overhead capture.
SPLIT_FILES: dict[Split, str] = {
    "train": "dish_ids/splits/depth_train_ids.txt",
    "test": "dish_ids/splits/depth_test_ids.txt",
}


@dataclass(frozen=True)
class DishRecord:
    dish_id: str
    total_mass: float
    total_calories: float
    total_fat: float
    total_carb: float
    total_protein: float
    image: Path | None = None  # overhead rgb.png; None when it is not downloaded

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
    # Every dish listed in excluded_dishes.csv; test keeps them, so it can be reported with and without.
    excluded: frozenset[str] = frozenset()
    # Dish id → scan group (see preprocess.scan_groups); empty before `preprocess`.
    groups: Mapping[str, str] = field(default_factory=dict)


def load_dataset(root: Path, full: bool = False) -> Dataset:
    """Load the official overhead split, or the project's fixed 32/8 pilot, without excluded train dishes."""
    root = root.expanduser().resolve()
    metadata = _load_metadata(root)
    excluded = _load_exclusions(root)
    groups = _load_groups(root)
    if bool(excluded) != bool(groups):
        raise Nutrition5kError(
            f"only one of {EXCLUDED_FILE} and {GROUPS_FILE} exists; run `nutrition5k preprocess`"
        )
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
            replace(metadata[dish_id], image=_image_path(root, dish_id))
            for dish_id in ids
        )

    return Dataset(
        root=root,
        mode="full" if full else "pilot",
        train=tuple(record for record in records["train"] if record.dish_id not in excluded),
        test=records["test"],
        excluded=excluded,
        groups=groups,
    )


def _load_exclusions(root: Path) -> frozenset[str]:
    path = root / EXCLUDED_FILE
    if not path.is_file():
        return frozenset()
    with path.open(newline="", encoding="utf-8") as handle:
        return frozenset(row["dish_id"] for row in csv.DictReader(handle))


def _load_groups(root: Path) -> dict[str, str]:
    path = root / GROUPS_FILE
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["dish_id"]: row["group"] for row in csv.DictReader(handle)}


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


def split_validation(
    records: tuple[DishRecord, ...], fraction: float = 0.1, groups: Mapping[str, str] | None = None
) -> tuple[tuple[DishRecord, ...], tuple[DishRecord, ...]]:
    """Hold out a deterministic fraction of training dishes for model selection.

    Whole scan groups are held out, in sha256 order of the group id, until the fraction is
    reached, so no plate has scans on both sides. Without groups each dish is its own group.
    """
    groups = groups or {}
    members: dict[str, list[str]] = {}
    for record in records:
        members.setdefault(groups.get(record.dish_id, record.dish_id), []).append(record.dish_id)
    ranked = sorted(
        members,
        key=lambda group: hashlib.sha256(f"{PILOT_SEED}:validation:{group}".encode()).digest(),
    )
    target = max(1, round(len(records) * fraction))
    held_out: set[str] = set()
    for group in ranked:
        if len(held_out) >= target:
            break
        held_out.update(members[group])
    train = tuple(record for record in records if record.dish_id not in held_out)
    validation = tuple(record for record in records if record.dish_id in held_out)
    return train, validation


def _image_path(root: Path, dish_id: str) -> Path | None:
    path = root / OVERHEAD_DIR / dish_id / OVERHEAD_FILES[0]
    return path if path.is_file() else None
