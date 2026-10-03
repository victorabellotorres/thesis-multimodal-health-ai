from __future__ import annotations

import importlib.util
import tempfile
import urllib.error
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nutrition5k.clear import clear_preparation
from nutrition5k.cli import main
from nutrition5k.dataset import DishRecord, load_dataset
from nutrition5k.download import download_side_angle_videos
from nutrition5k.errors import Nutrition5kError
from nutrition5k.evaluation import evaluate_predictions
from nutrition5k.fetch import fetch_dataset_index
from nutrition5k.frames import extract_sampled_frames
from nutrition5k.models import AverageBaseline


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
    def test_clear_removes_prepared_data_and_preserves_baseline_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data" / "nutrition5k"
            output_root = root / "outputs" / "average-baseline"
            (data_root / "imagery").mkdir(parents=True)
            (data_root / "imagery" / "video.h264").touch()
            output_root.mkdir(parents=True)
            (output_root / "evaluation.json").touch()

            summary = clear_preparation(data_root)

            self.assertEqual((summary.removed, summary.absent), (1, 0))
            self.assertFalse(data_root.exists())
            self.assertTrue(output_root.exists())

    def test_clear_command_uses_only_the_project_data_root(self) -> None:
        summary = SimpleNamespace(removed=1, absent=0)
        with patch("nutrition5k.cli.clear_preparation", return_value=summary) as clear:
            self.assertEqual(main(["clear"]), 0)

        clear.assert_called_once_with(Path("data/nutrition5k"))

    def test_fetch_downloads_required_files_and_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def response() -> MagicMock:
                result = MagicMock()
                result.__enter__.return_value = result
                result.read.side_effect = [b"contents", b""]
                return result

            with patch(
                "nutrition5k.fetch.urllib.request.urlopen",
                side_effect=lambda *args, **kwargs: response(),
            ) as urlopen:
                first = fetch_dataset_index(root)

            self.assertEqual((first.downloaded, first.skipped), (4, 0))
            self.assertEqual(urlopen.call_count, 4)
            self.assertTrue((root / "metadata/dish_metadata_cafe1.csv").is_file())
            self.assertTrue((root / "dish_ids/splits/rgb_test_ids.txt").is_file())

            with patch("nutrition5k.fetch.urllib.request.urlopen") as urlopen:
                second = fetch_dataset_index(root)

            self.assertEqual((second.downloaded, second.skipped), (0, 4))
            urlopen.assert_not_called()

    def test_fetch_command_uses_the_default_data_root(self) -> None:
        summary = SimpleNamespace(downloaded=4, skipped=0)
        with patch("nutrition5k.cli.fetch_dataset_index", return_value=summary) as fetch:
            self.assertEqual(main(["fetch"]), 0)

        fetch.assert_called_once_with(
            Path("data/nutrition5k"), progress=fetch.call_args.kwargs["progress"]
        )

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
            self.assertEqual(main(["prepare", "--full", "--mode", "complete"]), 0)

        load.assert_called_once_with(Path("data/nutrition5k"), full=True)
        download.assert_called_once_with(
            (), Path("/data"), one_video=False, progress=download.call_args.kwargs["progress"]
        )
        extract.assert_called_once_with((), Path("/data"), "complete")

    def test_prepare_defaults_to_minimal_mode(self) -> None:
        dataset = SimpleNamespace(root=Path("/data"), mode="pilot", train=(), test=())
        summary = SimpleNamespace(downloaded=0, skipped=0, missing=0)
        with (
            patch("nutrition5k.cli.load_dataset", return_value=dataset),
            patch("nutrition5k.cli.download_side_angle_videos", return_value=summary) as download,
            patch("nutrition5k.cli.extract_sampled_frames") as extract,
        ):
            self.assertEqual(main(["prepare"]), 0)

        download.assert_called_once_with(
            (), Path("/data"), one_video=True, progress=download.call_args.kwargs["progress"]
        )
        extract.assert_called_once_with((), Path("/data"), "minimal")

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

    def test_complete_mode_samples_every_fifth_frame_of_each_camera(self) -> None:
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
                extract_sampled_frames((record,), root, "complete")

            self.assertEqual(run.call_count, 4)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertTrue(any("camera_A.h264" in part for command in commands for part in command))
            self.assertIn("select=not(mod(n\\,5))", run.call_args.args[0])
            self.assertNotIn("-frames:v", run.call_args.args[0])
            self.assertIn("frames_sampled5", run.call_args.args[0][-1])

    def test_minimal_mode_extracts_first_frame_of_first_available_camera(self) -> None:
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
                extract_sampled_frames((record,), root, "minimal")

            self.assertEqual(run.call_count, 1)
            self.assertTrue(any("camera_A.h264" in part for part in run.call_args.args[0]))
            self.assertIn("-frames:v", run.call_args.args[0])
            self.assertIn("frames_first", run.call_args.args[0][-1])

    def test_video_download_uses_first_available_camera_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            record = load_dataset(root).train[0]
            response = MagicMock()
            response.__enter__.return_value = response
            response.read.side_effect = [b"video", b""]

            with patch(
                "nutrition5k.download.urllib.request.urlopen",
                side_effect=[
                    urllib.error.HTTPError("url", 404, "missing", {}, None),
                    response,
                ],
            ):
                summary = download_side_angle_videos((record,), root, one_video=True)

            self.assertEqual(summary.downloaded, 1)
            self.assertEqual(summary.missing, 1)
            self.assertTrue(
                (root / "imagery" / "side_angles" / record.dish_id / "camera_B.h264").is_file()
            )


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

        self.assertEqual(result["metrics"]["total_mass"]["pmae"], 60.0)


@unittest.skipUnless(importlib.util.find_spec("timm"), "training extra not installed")
class MobileNetTest(unittest.TestCase):
    def test_every_head_layout_predicts_five_targets_in_order(self) -> None:
        import torch

        from nutrition5k.models.mobilenet import NutritionNet

        images = torch.zeros(2, 3, 64, 64)
        for heads in ("single", "grouped", "per-target"):
            model = NutritionNet("v4s", heads, hidden=(), pretrained=False).eval()
            model.freeze_backbone(1)
            self.assertEqual(model(images).shape, (2, 5))
        self.assertFalse(model.backbone.conv_stem.weight.requires_grad)
        self.assertTrue(model.backbone.conv_head.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
