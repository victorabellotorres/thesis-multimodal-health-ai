from __future__ import annotations

from pathlib import Path

from ..dataset import TARGET_NAMES, Dataset
from ..evaluation import TARGET_UNITS, evaluate_predictions, write_json, write_predictions
from ..models import AverageBaseline


def run_average_baseline(dataset: Dataset, output_root: Path) -> Path:
    """Fit, evaluate, and persist the average baseline for one dataset mode."""
    model = AverageBaseline.fit(dataset.train)
    predictions = model.predict(record.dish_id for record in dataset.test)
    output = output_root / dataset.mode

    write_json(
        output / "model.json",
        {
            "training_count": model.training_count,
            "means": dict(zip(TARGET_NAMES, model.means, strict=True)),
            "units": TARGET_UNITS,
        },
    )
    write_predictions(output / "predictions.csv", predictions)
    write_json(output / "evaluation.json", evaluate_predictions(dataset.test, predictions))
    return output
