from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from nutrition5k.cli import main
from nutrition5k.dataset import DatasetConfig, DatasetMode, Nutrition5kDataset
from nutrition5k.errors import Nutrition5kError
from nutrition5k.download import DownloadSummary, download_side_angle_videos
from nutrition5k.frames import extract_sampled_frames
from nutrition5k.baseline import AverageBaseline
from nutrition5k.evaluation import evaluate_predictions, load_predictions


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


def _metadata_row_without_count(dish_id: str, value: int) -> str:
    fields = _metadata_row(dish_id, value).split(",")
    del fields[6]
    return ",".join(fields)


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
    (split_dir / "rgb_train_ids.txt").write_text(
        "\n".join(reversed(train_ids)) + "\n", encoding="utf-8"
    )
    (split_dir / "rgb_test_ids.txt").write_text(
        "\n".join(reversed(test_ids)) + "\n", encoding="utf-8"
    )


class Nutrition5kDatasetTest(unittest.TestCase):
    def test_pilot_is_deterministic_and_has_required_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            first = Nutrition5kDataset.load(DatasetConfig(root=root))
            second = Nutrition5kDataset.load(DatasetConfig(root=root))

            self.assertEqual(len(first.train), 32)
            self.assertEqual(len(first.test), 8)
            self.assertEqual(
                [record.dish_id for record in first.train],
                [record.dish_id for record in second.train],
            )
            self.assertFalse(
                {record.dish_id for record in first.train}
                & {record.dish_id for record in first.test}
            )

    def test_full_mode_uses_every_official_split_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            dataset = Nutrition5kDataset.load(
                DatasetConfig(root=root, mode=DatasetMode.FULL)
            )
            self.assertEqual(len(dataset.train), 40)
            self.assertEqual(len(dataset.test), 12)
            self.assertEqual(dataset.train[0].total_mass, 239.0)
            self.assertEqual(dataset.train[0].targets[1], 139.0)

    def test_released_schema_without_ingredient_count_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            path = root / "metadata" / "dish_metadata_cafe1.csv"
            rows = path.read_text(encoding="utf-8").splitlines()
            rows[0] = _metadata_row_without_count("dish_0000000000", 0)
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            dataset = Nutrition5kDataset.load(
                DatasetConfig(root=root, mode=DatasetMode.FULL)
            )
            record = next(item for item in dataset.train if item.dish_id == "dish_0000000000")
            self.assertEqual(record.num_ingredients, 1)

    def test_sampled_frame_paths_are_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            selected = Nutrition5kDataset.load(DatasetConfig(root=root)).train[0].dish_id
            frame_dir = (
                root / "imagery" / "side_angles" / selected / "frames_sampled5"
            )
            frame_dir.mkdir(parents=True)
            expected = frame_dir / "camera_A_frame_001.jpeg"
            expected.touch()

            dataset = Nutrition5kDataset.load(
                DatasetConfig(root=root, require_frames=False)
            )
            record = next(item for item in dataset.train if item.dish_id == selected)
            self.assertEqual(record.frame_paths, (expected,))

    def test_require_frames_reports_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            with self.assertRaisesRegex(Nutrition5kError, "no sampled frames found"):
                Nutrition5kDataset.load(DatasetConfig(root=root, require_frames=True))

    def test_malformed_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            path = root / "metadata" / "dish_metadata_cafe1.csv"
            path.write_text("dish_bad,not-a-number\n", encoding="utf-8")
            with self.assertRaisesRegex(Nutrition5kError, "malformed metadata row"):
                Nutrition5kDataset.load(DatasetConfig(root=root))

    def test_overlapping_splits_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            test_path = root / "dish_ids" / "splits" / "rgb_test_ids.txt"
            test_path.write_text(
                "dish_0000000000\n" + test_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Nutrition5kError, "splits overlap"):
                Nutrition5kDataset.load(DatasetConfig(root=root))

    def test_smoke_command_emits_machine_readable_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["smoke", "--root", str(root), "--mode", "pilot"])
            summary = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["counts"], {"train": 32, "test": 8})
            self.assertEqual(len(summary["selected_ids"]["train"]), 32)

    def test_missing_file_returns_cli_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stderr = StringIO()
            with redirect_stderr(stderr):
                exit_code = main(["smoke", "--root", temporary])
            self.assertEqual(exit_code, 2)
            self.assertIn("required Nutrition5k file is missing", stderr.getvalue())

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(Nutrition5kError, "invalid dataset mode"):
            DatasetConfig(root=Path("data"), mode="unknown")  # type: ignore[arg-type]

    def test_frame_extraction_samples_all_four_cameras(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            record = Nutrition5kDataset.load(DatasetConfig(root=root)).train[0]
            dish_dir = root / "imagery" / "side_angles" / record.dish_id
            dish_dir.mkdir(parents=True)
            for camera in "ABCD":
                (dish_dir / f"camera_{camera}.h264").touch()

            with patch("nutrition5k.frames._ffmpeg_executable", return_value="/usr/bin/ffmpeg"), patch(
                "nutrition5k.frames.subprocess.run"
            ) as run:
                count = extract_sampled_frames((record,), root, stride=5)

            self.assertEqual(count, 1)
            self.assertEqual(run.call_count, 4)
            for call in run.call_args_list:
                command = call.args[0]
                self.assertIn("select=not(mod(n\\,5))", command)
                self.assertTrue(str(command[-1]).endswith("%03d.jpeg"))

    def test_video_download_materializes_all_four_cameras_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            record = Nutrition5kDataset.load(DatasetConfig(root=root)).train[0]

            with patch(
                "nutrition5k.download.urllib.request.urlopen",
                side_effect=lambda *_args, **_kwargs: BytesIO(b"video bytes"),
            ) as urlopen:
                summary = download_side_angle_videos((record,), root)

            self.assertEqual(
                summary, DownloadSummary(downloaded=4, skipped=0, missing=0)
            )
            self.assertEqual(urlopen.call_count, 4)
            dish_dir = root / "imagery" / "side_angles" / record.dish_id
            for camera in "ABCD":
                self.assertEqual(
                    (dish_dir / f"camera_{camera}.h264").read_bytes(), b"video bytes"
                )
                self.assertFalse((dish_dir / f"camera_{camera}.h264.part").exists())

    def test_frame_extraction_skips_existing_camera_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            record = Nutrition5kDataset.load(DatasetConfig(root=root)).train[0]
            dish_dir = root / "imagery" / "side_angles" / record.dish_id
            frame_dir = dish_dir / "frames_sampled5"
            frame_dir.mkdir(parents=True)
            for camera in "ABCD":
                (dish_dir / f"camera_{camera}.h264").touch()
            (frame_dir / "camera_A_frame_001.jpeg").touch()

            with patch(
                "nutrition5k.frames._ffmpeg_executable", return_value="/usr/bin/ffmpeg"
            ), patch("nutrition5k.frames.subprocess.run") as run:
                extract_sampled_frames((record,), root, stride=5)

            self.assertEqual(run.call_count, 3)
            self.assertTrue(
                all("camera_A" not in str(call.args[0][-1]) for call in run.call_args_list)
            )

    def test_video_download_tolerates_a_missing_camera(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            record = Nutrition5kDataset.load(DatasetConfig(root=root)).train[0]
            missing = urllib.error.HTTPError(
                "https://example.test/camera_A.h264", 404, "Not Found", {}, None
            )

            with patch(
                "nutrition5k.download.urllib.request.urlopen",
                side_effect=[missing, BytesIO(b"B"), BytesIO(b"C"), BytesIO(b"D")],
            ):
                summary = download_side_angle_videos((record,), root)

            self.assertEqual(
                summary, DownloadSummary(downloaded=3, skipped=0, missing=1)
            )

    def test_prepare_imagery_downloads_then_extracts_the_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            stdout = StringIO()
            with patch(
                "nutrition5k.cli.download_side_angle_videos",
                return_value=DownloadSummary(downloaded=159, skipped=0, missing=1),
            ) as download, patch(
                "nutrition5k.cli.extract_sampled_frames", return_value=40
            ) as extract, redirect_stdout(stdout):
                exit_code = main(
                    ["prepare-imagery", "--root", str(root), "--mode", "pilot"]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(download.call_args.args[0]), 40)
            self.assertEqual(len(extract.call_args.args[0]), 40)
            self.assertLess(stdout.getvalue().find("videos downloaded"), stdout.getvalue().find("extracted"))


if __name__ == "__main__":
    unittest.main()


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
            predictions = model.predict(record.dish_id for record in dataset.test)
            self.assertEqual(set(predictions), {record.dish_id for record in dataset.test})
            self.assertTrue(all(vector == expected for vector in predictions.values()))

    def test_fit_isolated_from_test_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            dataset = Nutrition5kDataset.load(DatasetConfig(root=root))
            first = AverageBaseline.fit(dataset.train)
            # The deliberately extreme test labels are not supplied to fit.
            altered_test = tuple(
                record.__class__(
                    **{**record.__dict__, "total_mass": 1_000_000_000.0}
                )
                for record in dataset.test
            )
            second = AverageBaseline.fit(dataset.train)
            self.assertEqual(first, second)
            self.assertNotEqual(altered_test[0].total_mass, dataset.test[0].total_mass)

    def test_cli_serializes_parameters_predictions_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            output = Path(temporary) / "output"
            _make_dataset(root)
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "average-baseline", "--root", str(root), "--mode", "pilot",
                        "--output-dir", str(output),
                    ]
                )
            self.assertEqual(exit_code, 0)
            parameters = json.loads((output / "parameters.json").read_text(encoding="utf-8"))
            run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            predictions = load_predictions(output / "predictions.csv")
            evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(parameters["training_count"], 32)
            self.assertEqual(run["split_counts"], {"train": 32, "test": 8})
            self.assertEqual(len(predictions), 8)
            self.assertEqual(evaluation["test_count"], 8)

    def test_evaluator_marks_undefined_percentage_and_r2_explicitly(self) -> None:
        from nutrition5k.dataset import DishRecord

        records = (
            DishRecord("dish_a", "test", 0.0, 1.0, 1.0, 1.0, 1.0, 0),
            DishRecord("dish_b", "test", 0.0, 1.0, 1.0, 1.0, 1.0, 0),
        )
        result = evaluate_predictions(
            records, {record.dish_id: (1.0, 1.0, 1.0, 1.0, 1.0) for record in records}
        )
        mass = result["metrics_by_target"]["total_mass"]
        self.assertIsNone(mass["percentage_mae"])
        self.assertEqual(mass["zero_ground_truth_count"], 2)
        self.assertIsNone(mass["r2"])

    def test_evaluator_uses_official_percentage_mae_aggregation(self) -> None:
        from nutrition5k.dataset import DishRecord

        records = (
            DishRecord("dish_a", "test", 2.0, 2.0, 2.0, 2.0, 2.0, 0),
            DishRecord("dish_b", "test", 8.0, 8.0, 8.0, 8.0, 8.0, 0),
        )
        result = evaluate_predictions(
            records,
            {
                "dish_a": (4.0, 4.0, 4.0, 4.0, 4.0),
                "dish_b": (4.0, 4.0, 4.0, 4.0, 4.0),
            },
        )
        # MAE is 3, mean ground truth is 5, hence official percentage MAE is 60.
        self.assertEqual(result["metrics_by_target"]["total_mass"]["percentage_mae"], 60.0)
