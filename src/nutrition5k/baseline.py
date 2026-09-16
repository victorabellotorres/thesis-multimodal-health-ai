from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from .dataset import DishRecord, TARGET_NAMES, TargetValues
from .errors import Nutrition5kError


@dataclass(frozen=True)
class AverageBaseline:
    """Predict the training-set mean for every dish."""

    means: TargetValues
    training_count: int

    @classmethod
    def fit(cls, records: Iterable[DishRecord]) -> "AverageBaseline":
        targets = [record.targets for record in records]
        if not targets:
            raise Nutrition5kError("cannot fit the baseline without training dishes")
        if not all(math.isfinite(value) for row in targets for value in row):
            raise Nutrition5kError("training targets must be finite numbers")

        means = cast(
            TargetValues,
            tuple(
                sum(row[index] for row in targets) / len(targets)
                for index in range(len(TARGET_NAMES))
            ),
        )
        return cls(means, len(targets))

    def predict(self, dish_ids: Iterable[str]) -> dict[str, TargetValues]:
        return {dish_id: self.means for dish_id in dish_ids}
