"""Metadata-only mean-prediction reference baseline."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .dataset import DishRecord, TARGET_NAMES
from .errors import Nutrition5kError
from .evaluation import TARGET_UNITS


@dataclass(frozen=True)
class AverageBaseline:
    """Predict the per-target mean fitted from a training split only."""

    means: tuple[float, float, float, float, float]
    training_count: int

    @classmethod
    def fit(cls, training_records: Iterable[DishRecord]) -> "AverageBaseline":
        records = tuple(training_records)
        if not records:
            raise Nutrition5kError("cannot fit average baseline with an empty training split")
        values = [record.targets for record in records]
        if any(not all(math.isfinite(value) for value in vector) for vector in values):
            raise Nutrition5kError("training targets must be finite numeric values")
        means = tuple(sum(vector[index] for vector in values) / len(values) for index in range(len(TARGET_NAMES)))
        return cls(means=means, training_count=len(records))  # type: ignore[arg-type]

    def predict(self, dish_ids: Iterable[str]) -> dict[str, tuple[float, ...]]:
        return {dish_id: self.means for dish_id in dish_ids}

    def artifact(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model": "average_baseline",
            "target_names": list(TARGET_NAMES),
            "target_units": TARGET_UNITS,
            "training_count": self.training_count,
            "means": dict(zip(TARGET_NAMES, self.means, strict=True)),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.artifact(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "AverageBaseline":
        if not path.is_file():
            raise Nutrition5kError(f"baseline parameter file is missing: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["schema_version"] != 1 or data["model"] != "average_baseline":
                raise KeyError("schema_version/model")
            if tuple(data["target_names"]) != TARGET_NAMES or data["target_units"] != TARGET_UNITS:
                raise KeyError("target schema")
            means = tuple(float(data["means"][name]) for name in TARGET_NAMES)
            training_count = int(data["training_count"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise Nutrition5kError(f"invalid average baseline parameter file: {path}") from exc
        if training_count <= 0 or not all(math.isfinite(value) for value in means):
            raise Nutrition5kError(f"invalid average baseline parameter file: {path}")
        return cls(means=means, training_count=training_count)  # type: ignore[arg-type]
