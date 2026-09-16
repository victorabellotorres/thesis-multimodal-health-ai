from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nutrition5k.baseline import AverageBaseline
from nutrition5k.dataset import DatasetConfig, DatasetMode, DishRecord, Nutrition5kDataset
from nutrition5k.errors import Nutrition5kError
from nutrition5k.evaluation import evaluate_predictions
from nutrition5k.frames import extract_sampled_frames


def _metadata_row(dish_id: str, value: int) -> str:
    return ",".join(
        (
            dish_id,
            str(100 + value),
            str(200 + value),
            "10.0",
            "20.0",
            "30.0",
            "1",
            "ingr_0000000001",
            "test ingredient",
            "200.0",
            "100.0",
            "10.0",
            "20.0",
            "30.0",
        )
    )


def _make_dataset(root: Path, train_count: int = 40, test_count: int = 12) -> None:
    metadata_dir = root / "metadata"
    split_dir = root / "dish_ids" / "splits"
    metadata_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    train_ids = [f"dish_{index:010d}" for index in range(train_count)]
    test_ids = [f"dish_{1000 + index:010d}" for index in range(test_count)]
    all_ids = train_ids + test_ids
    midpoint = len(all_ids) // 2
    (metadata_dir / "dish_metadata_cafe1.csv").write_text(
        "\n".join(_metadata_row(dish_id, index) for index, dish_id in enumerate(all_ids[:midpoint]))
        + "\n",
        encoding="utf-8",
    )
    (metadata_dir / "dish_metadata_cafe2.csv").write_text(
        "\n".join(
            _metadata_row(dish_id, midpoint + index)
            for index, dish_id in enumerate(all_ids[midpoint:])
        )
        + "\n",
        encoding="utf-8",
    )
    (split_dir / "rgb_train_ids.txt").write_text("\n".join(reversed(train_ids)) + "\n", encoding="utf-8")
    (split_dir / "rgb_test_ids.txt").write_text("\n".join(reversed(test_ids)) + "\n", encoding="utf-8")


class Nutrition5kDatasetTest(unittest.TestCase):
    def test_pilot_is_deterministic_and_has_required_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            first = Nutrition5kDataset.load(DatasetConfig(root=root))
            second = Nutrition5kDataset.load(DatasetConfig(root=root))

            self.assertEqual((len(first.train), len(first.test)), (32, 8))
            self.assertEqual([record.dish_id for record in first.train], [record.dish_id for record in second.train])

    def test_full_mode_uses_every_official_split_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            dataset = Nutrition5kDataset.load(DatasetConfig(root=root, mode=DatasetMode.FULL))

            self.assertEqual((len(dataset.train), len(dataset.test)), (40, 12))

    def test_malformed_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            (root / "metadata" / "dish_metadata_cafe1.csv").write_text("dish_bad,not-a-number\n", encoding="utf-8")

            with self.assertRaisesRegex(Nutrition5kError, "malformed metadata row"):
                Nutrition5kDataset.load(DatasetConfig(root=root))

    def test_overlapping_splits_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            test_path = root / "dish_ids" / "splits" / "rgb_test_ids.txt"
            test_path.write_text("dish_0000000000\n" + test_path.read_text(encoding="utf-8"), encoding="utf-8")

            with self.assertRaisesRegex(Nutrition5kError, "splits overlap"):
                Nutrition5kDataset.load(DatasetConfig(root=root))

    def test_missing_required_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(Nutrition5kError, "required Nutrition5k file is missing"):
                Nutrition5kDataset.load(DatasetConfig(root=Path(temporary)))

    def test_frame_extraction_samples_all_four_cameras(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            record = Nutrition5kDataset.load(DatasetConfig(root=root)).train[0]
            dish_dir = root / "imagery" / "side_angles" / record.dish_id
            dish_dir.mkdir(parents=True)
            for camera in "ABCD":
                (dish_dir / f"camera_{camera}.h264").touch()

            with patch("nutrition5k.frames._ffmpeg_executable", return_value="/usr/bin/ffmpeg"), patch("nutrition5k.frames.subprocess.run") as run:
                extract_sampled_frames((record,), root, stride=5)

            self.assertEqual(run.call_count, 4)


class AverageBaselineTest(unittest.TestCase):
    def test_fits_training_means_and_predicts_a_constant_vector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            dataset = Nutrition5kDataset.load(DatasetConfig(root=root))
            model = AverageBaseline.fit(dataset.train)
            expected = tuple(
                sum(record.targets[index] for record in dataset.train) / len(dataset.train)
                for index in range(5)
            )

            self.assertEqual(model.means, expected)
            self.assertTrue(all(vector == expected for vector in model.predict(record.dish_id for record in dataset.test).values()))

    def test_evaluator_uses_official_percentage_mae_aggregation(self) -> None:
        records = (
            DishRecord("dish_a", "test", 2.0, 2.0, 2.0, 2.0, 2.0, 0),
            DishRecord("dish_b", "test", 8.0, 8.0, 8.0, 8.0, 8.0, 0),
        )
        result = evaluate_predictions(
            records,
            {"dish_a": (4.0, 4.0, 4.0, 4.0, 4.0), "dish_b": (4.0, 4.0, 4.0, 4.0, 4.0)},
        )

        self.assertEqual(result["metrics_by_target"]["total_mass"]["percentage_mae"], 60.0)


if __name__ == "__main__":
    unittest.main()
