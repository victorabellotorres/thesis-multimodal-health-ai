from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nutrition5k.baseline import AverageBaseline
from nutrition5k.cli import main
from nutrition5k.dataset import DishRecord, load_dataset
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
    cafe1_rows = (
        _metadata_row(dish_id, index)
        for index, dish_id in enumerate(all_ids[:midpoint])
    )
    (metadata_dir / "dish_metadata_cafe1.csv").write_text(
        "\n".join(cafe1_rows) + "\n",
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
    (split_dir / "rgb_train_ids.txt").write_text(
        "\n".join(reversed(train_ids)) + "\n", encoding="utf-8"
    )
    (split_dir / "rgb_test_ids.txt").write_text(
        "\n".join(reversed(test_ids)) + "\n", encoding="utf-8"
    )


class Nutrition5kDatasetTest(unittest.TestCase):
    def test_prepare_full_selects_the_full_dataset(self) -> None:
        dataset = SimpleNamespace(root=Path("/data"), mode="full", train=(), test=())
        summary = SimpleNamespace(downloaded=0, skipped=0, missing=0)
        with (
            patch("nutrition5k.cli.load_dataset", return_value=dataset) as load,
            patch(
                "nutrition5k.cli.download_side_angle_videos", return_value=summary
            ) as download,
            patch("nutrition5k.cli.extract_sampled_frames") as extract,
        ):
            self.assertEqual(main(["prepare", "--full"]), 0)

        load.assert_called_once_with(Path("data/nutrition5k"), full=True)
        download.assert_called_once()
        extract.assert_called_once_with((), Path("/data"))

    def test_pilot_is_deterministic_and_has_required_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            first = load_dataset(root)
            second = load_dataset(root)

            self.assertEqual((len(first.train), len(first.test)), (32, 8))
            first_ids = [record.dish_id for record in first.train]
            second_ids = [record.dish_id for record in second.train]
            self.assertEqual(first_ids, second_ids)

    def test_full_mode_uses_every_official_split_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            dataset = load_dataset(root, full=True)

            self.assertEqual((len(dataset.train), len(dataset.test)), (40, 12))

    def test_malformed_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            bad_metadata = root / "metadata" / "dish_metadata_cafe1.csv"
            bad_metadata.write_text("dish_bad,not-a-number\n", encoding="utf-8")

            with self.assertRaisesRegex(Nutrition5kError, "malformed metadata row"):
                load_dataset(root)

    def test_overlapping_splits_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            test_path = root / "dish_ids" / "splits" / "rgb_test_ids.txt"
            existing = test_path.read_text(encoding="utf-8")
            test_path.write_text("dish_0000000000\n" + existing, encoding="utf-8")

            with self.assertRaisesRegex(Nutrition5kError, "splits overlap"):
                load_dataset(root)

    def test_missing_required_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(Nutrition5kError, "required Nutrition5k file is missing"):
                load_dataset(Path(temporary))

    def test_frame_extraction_samples_all_four_cameras(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            record = load_dataset(root).train[0]
            dish_dir = root / "imagery" / "side_angles" / record.dish_id
            dish_dir.mkdir(parents=True)
            for camera in "ABCD":
                (dish_dir / f"camera_{camera}.h264").touch()

            with (
                patch(
                    "nutrition5k.frames._ffmpeg_executable",
                    return_value="/usr/bin/ffmpeg",
                ),
                patch("nutrition5k.frames.subprocess.run") as run,
            ):
                extract_sampled_frames((record,), root)

            self.assertEqual(run.call_count, 4)


class AverageBaselineTest(unittest.TestCase):
    def test_fits_training_means_and_predicts_a_constant_vector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            dataset = load_dataset(root)
            model = AverageBaseline.fit(dataset.train)
            expected = tuple(
                sum(record.targets[index] for record in dataset.train) / len(dataset.train)
                for index in range(5)
            )

            self.assertEqual(model.means, expected)
            dish_ids = (record.dish_id for record in dataset.test)
            self.assertTrue(
                all(vector == expected for vector in model.predict(dish_ids).values())
            )

    def test_evaluator_uses_official_percentage_mae_aggregation(self) -> None:
        records = (
            DishRecord("dish_a", 2.0, 2.0, 2.0, 2.0, 2.0),
            DishRecord("dish_b", 8.0, 8.0, 8.0, 8.0, 8.0),
        )
        result = evaluate_predictions(
            records,
            {"dish_a": (4.0, 4.0, 4.0, 4.0, 4.0), "dish_b": (4.0, 4.0, 4.0, 4.0, 4.0)},
        )

        self.assertEqual(result["metrics"]["total_mass"]["percentage_mae"], 60.0)


if __name__ == "__main__":
    unittest.main()
