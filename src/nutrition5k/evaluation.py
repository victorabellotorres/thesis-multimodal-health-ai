from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from .dataset import DishRecord, TARGET_NAMES, TargetValues
from .errors import Nutrition5kError

TARGET_UNITS = {
    "total_mass": "g",
    "total_calories": "kcal",
    "total_fat": "g",
    "total_carb": "g",
    "total_protein": "g",
}


def evaluate_predictions(
    records: Iterable[DishRecord], predictions: Mapping[str, TargetValues]
) -> dict[str, object]:
    """Calculate MAE, percentage MAE, RMSE, R² and ground-truth SD per target."""
    records = tuple(records)
    expected_ids = {record.dish_id for record in records}
    if not records:
        raise Nutrition5kError("cannot evaluate an empty test split")
    if set(predictions) != expected_ids:
        raise Nutrition5kError("prediction dish IDs do not match the test split")

    count = len(records)
    metrics: dict[str, dict[str, float | str | None]] = {}
    for index, name in enumerate(TARGET_NAMES):
        truth = [record.targets[index] for record in records]
        estimates = [predictions[record.dish_id][index] for record in records]
        if not all(math.isfinite(value) for value in truth + estimates):
            raise Nutrition5kError("targets and predictions must be finite numbers")

        mean_truth = sum(truth) / count
        mae = sum(abs(t - e) for t, e in zip(truth, estimates)) / count
        # Residual and total sums of squares, shared by RMSE, R² and SD.
        ss_res = sum((t - e) ** 2 for t, e in zip(truth, estimates))
        ss_tot = sum((t - mean_truth) ** 2 for t in truth)

        metrics[name] = {
            "unit": TARGET_UNITS[name],
            "mae": mae,
            "pmae": None if mean_truth == 0 else mae / mean_truth * 100,
            "rmse": math.sqrt(ss_res / count),
            "rsquared": None if ss_tot == 0 else 1 - ss_res / ss_tot,
            # Population SD of the ground truth (divides by n, not n - 1).
            "SD": math.sqrt(ss_tot / count),
        }

    pmaes = [metric["pmae"] for metric in metrics.values()]
    mean_pmae = None if None in pmaes else sum(pmaes) / len(pmaes)

    return {"test_count": count, "mean_pmae": mean_pmae, "metrics": metrics}

def write_predictions(path: Path, predictions: Mapping[str, TargetValues]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dish_id", *TARGET_NAMES))
        for dish_id in sorted(predictions):
            writer.writerow((dish_id, *predictions[dish_id]))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
