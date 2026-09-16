from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from nutrition5k.cli import main
from nutrition5k.dataset import DatasetConfig, DatasetMode, Nutrition5kDataset
from nutrition5k.errors import Nutrition5kError
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

            with patch("nutrition5k.frames.shutil.which", return_value="/usr/bin/ffmpeg"), patch(
                "nutrition5k.frames.subprocess.run"
            ) as run:
                count = extract_sampled_frames((record,), root, stride=5)

            self.assertEqual(count, 1)
            self.assertEqual(run.call_count, 4)
            for call in run.call_args_list:
                command = call.args[0]
                self.assertIn("select=not(mod(n\\,5))", command)
                self.assertTrue(str(command[-1]).endswith("%03d.jpeg"))


if __name__ == "__main__":
    unittest.main()
