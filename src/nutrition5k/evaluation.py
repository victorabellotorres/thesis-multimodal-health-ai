"""Model-independent Nutrition5k regression evaluation.

The primary metrics follow the Nutrition5k convention: MAE in the native
target unit and percentage MAE, calculated as ``100 * MAE / mean(target)`` for
each target over the selected test dishes. It is undefined when that target's
mean ground truth is zero.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .dataset import DishRecord, TARGET_NAMES
from .errors import Nutrition5kError

TARGET_UNITS = {
    "total_mass": "g",
    "total_calories": "kcal",
    "total_fat": "g",
    "total_carb": "g",
    "total_protein": "g",
}
PREDICTION_COLUMNS = ("dish_id", *TARGET_NAMES)


def prediction_rows(predictions: Mapping[str, tuple[float, ...]]) -> list[dict[str, object]]:
    """Return deterministic, schema-checked rows for the prediction CSV."""
    rows: list[dict[str, object]] = []
    for dish_id in sorted(predictions):
        values = predictions[dish_id]
        _validate_vector(values, f"prediction for {dish_id}")
        rows.append({"dish_id": dish_id, **dict(zip(TARGET_NAMES, values, strict=True))})
    return rows


def write_predictions(path: Path, predictions: Mapping[str, tuple[float, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(prediction_rows(predictions))


def load_predictions(path: Path) -> dict[str, tuple[float, ...]]:
    if not path.is_file():
        raise Nutrition5kError(f"prediction file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(PREDICTION_COLUMNS):
            raise Nutrition5kError(
                f"invalid prediction columns in {path}; expected {list(PREDICTION_COLUMNS)}"
            )
        predictions: dict[str, tuple[float, ...]] = {}
        for line_number, row in enumerate(reader, start=2):
            dish_id = (row.get("dish_id") or "").strip()
            if not dish_id or dish_id in predictions:
                raise Nutrition5kError(f"invalid or duplicate dish ID at {path}:{line_number}")
            try:
                values = tuple(float(row[name]) for name in TARGET_NAMES)
            except (TypeError, ValueError) as exc:
                raise Nutrition5kError(f"non-numeric prediction at {path}:{line_number}") from exc
            _validate_vector(values, f"prediction at {path}:{line_number}")
            predictions[dish_id] = values
    return predictions


def evaluate_predictions(
    test_records: Iterable[DishRecord], predictions: Mapping[str, tuple[float, ...]]
) -> dict[str, Any]:
    """Evaluate exactly one finite prediction vector for each test dish."""
    records = tuple(test_records)
    expected_ids = {record.dish_id for record in records}
    actual_ids = set(predictions)
    if actual_ids != expected_ids:
        missing, extra = sorted(expected_ids - actual_ids), sorted(actual_ids - expected_ids)
        raise Nutrition5kError(
            "prediction dish IDs do not match test split"
            + (f"; missing: {missing[:3]}" if missing else "")
            + (f"; unexpected: {extra[:3]}" if extra else "")
        )
    if not records:
        raise Nutrition5kError("cannot evaluate an empty test split")

    per_dish: list[dict[str, Any]] = []
    target_errors: dict[str, list[float]] = {name: [] for name in TARGET_NAMES}
    truths: dict[str, list[float]] = {name: [] for name in TARGET_NAMES}
    predicted: dict[str, list[float]] = {name: [] for name in TARGET_NAMES}
    for record in sorted(records, key=lambda item: item.dish_id):
        truth, estimate = record.targets, predictions[record.dish_id]
        _validate_vector(truth, f"ground truth for {record.dish_id}")
        _validate_vector(estimate, f"prediction for {record.dish_id}")
        absolute = tuple(abs(value - actual) for value, actual in zip(estimate, truth, strict=True))
        percentage = tuple(
            None if actual == 0 else 100.0 * error / abs(actual)
            for error, actual in zip(absolute, truth, strict=True)
        )
        per_dish.append(
            {
                "dish_id": record.dish_id,
                "absolute_error": dict(zip(TARGET_NAMES, absolute, strict=True)),
                "percentage_error": dict(zip(TARGET_NAMES, percentage, strict=True)),
            }
        )
        for name, actual, estimate_value, error, percent in zip(
            TARGET_NAMES, truth, estimate, absolute, percentage, strict=True
        ):
            truths[name].append(actual)
            predicted[name].append(estimate_value)
            target_errors[name].append(error)
    by_target = {
        name: {
            "unit": TARGET_UNITS[name],
            "mae": _mean(target_errors[name]),
            "percentage_mae": _percentage_mae(target_errors[name], truths[name]),
            "percentage_denominator_mean_ground_truth": _mean(truths[name]),
            "zero_ground_truth_count": sum(value == 0 for value in truths[name]),
            "r2": _r2(truths[name], predicted[name]),
        }
        for name in TARGET_NAMES
    }
    percentage_maes = [
        by_target[name]["percentage_mae"]
        for name in TARGET_NAMES
        if by_target[name]["percentage_mae"] is not None
    ]
    return {
        "schema_version": 1,
        "target_names": list(TARGET_NAMES),
        "target_units": TARGET_UNITS,
        "test_count": len(records),
        "percentage_mae_policy": "100 * MAE / mean ground truth; null if the mean ground truth is zero",
        "metrics_by_target": by_target,
        "aggregate": {
            "macro_percentage_mae": _mean(percentage_maes) if percentage_maes else None,
        },
        "per_dish": per_dish,
    }


def write_evaluation(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_vector(values: tuple[float, ...], context: str) -> None:
    if len(values) != len(TARGET_NAMES) or not all(math.isfinite(value) for value in values):
        raise Nutrition5kError(f"{context} must contain {len(TARGET_NAMES)} finite numeric targets")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _r2(truth: list[float], estimate: list[float]) -> float | None:
    if len(truth) < 2:
        return None
    mean_truth = _mean(truth)
    total = sum((value - mean_truth) ** 2 for value in truth)
    if total == 0:
        return None
    residual = sum((actual - predicted) ** 2 for actual, predicted in zip(truth, estimate, strict=True))
    return 1.0 - residual / total


def _percentage_mae(errors: list[float], truth: list[float]) -> float | None:
    denominator = _mean(truth)
    if denominator == 0:
        return None
    return 100.0 * _mean(errors) / denominator
