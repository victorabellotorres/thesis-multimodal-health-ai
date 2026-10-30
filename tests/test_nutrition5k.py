from __future__ import annotations

import importlib.util
import io
import tempfile
import urllib.error
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nutrition5k.clear import clear_preparation
from nutrition5k.cli import main
from PIL import Image

from nutrition5k.dataset import DishRecord, load_dataset, split_validation
from nutrition5k.download import download_overhead
from nutrition5k.errors import Nutrition5kError
from nutrition5k.evaluation import evaluate_predictions, summarize_seeds
from nutrition5k.fetch import fetch_dataset_index
from nutrition5k.models import AverageBaseline
from nutrition5k.preprocess import MANUAL_EXCLUSIONS, label_errors, scan_groups, write_preprocess


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
    (split_dir / "depth_train_ids.txt").write_text(
        "\n".join(reversed(train_ids)) + "\n", encoding="utf-8"
    )
    (split_dir / "depth_test_ids.txt").write_text(
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
            self.assertTrue((root / "dish_ids/splits/depth_test_ids.txt").is_file())

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

    def test_prepare_downloads_only_dishes_without_both_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = (SimpleNamespace(dish_id="dish_1"), SimpleNamespace(dish_id="dish_2"))
            done = root / "imagery" / "realsense_overhead" / "dish_1"
            done.mkdir(parents=True)
            (done / "rgb.png").touch()
            (done / "depth_raw.png").touch()
            dataset = SimpleNamespace(root=root, mode="full", train=records, test=())
            summary = SimpleNamespace(downloaded=2, skipped=0, missing=0, failed=())
            with (
                patch("nutrition5k.cli.load_dataset", return_value=dataset) as load,
                patch("nutrition5k.cli.write_preprocess", return_value=(0, 0)) as preprocess,
                patch("nutrition5k.cli.download_overhead", return_value=summary) as download,
            ):
                self.assertEqual(main(["prepare", "--full"]), 0)

        load.assert_called_once_with(Path("data/nutrition5k"), full=True)
        download.assert_called_once_with(
            records[1:], root, progress=download.call_args.kwargs["progress"]
        )
        preprocess.assert_called_once_with(root)

    def test_prepare_reports_failed_dishes(self) -> None:
        records = (SimpleNamespace(dish_id="dish_1"),)
        dataset = SimpleNamespace(root=Path("/missing"), mode="pilot", train=records, test=())
        summary = SimpleNamespace(downloaded=0, skipped=0, missing=0, failed=("dish_1",))
        with (
            patch("nutrition5k.cli.load_dataset", return_value=dataset),
            patch("nutrition5k.cli.write_preprocess", return_value=(0, 0)),
            patch("nutrition5k.cli.download_overhead", return_value=summary),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            self.assertEqual(main(["prepare"]), 1)

        self.assertIn("1 dishes failed and will be retried by the next prepare: dish_1", stderr.getvalue())

    def test_check_reports_missing_images_and_unexpected_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dish_id, size in (("dish_1", (640, 480)), ("dish_3", (320, 240))):
                folder = root / dish_id
                folder.mkdir()
                Image.new("RGB", size).save(folder / "rgb.png")
            (root / "dish_1" / "depth_raw.png").touch()
            records = (
                SimpleNamespace(dish_id="dish_1", image=root / "dish_1" / "rgb.png"),
                SimpleNamespace(dish_id="dish_2", image=None),
                SimpleNamespace(dish_id="dish_3", image=root / "dish_3" / "rgb.png"),
            )
            dataset = SimpleNamespace(root=root, mode="full", train=records[:2], test=records[2:])
            with (
                patch("nutrition5k.cli.load_dataset", return_value=dataset),
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
            ):
                self.assertEqual(main(["check", "--full"]), 1)

        output = stdout.getvalue()
        self.assertIn("train: 1/2 dishes with an overhead image", output)
        self.assertIn("1 dishes have no depth image (unused by the models): dish_3", output)
        self.assertIn("1 images are not 640x480, as CROP_BOX assumes: dish_3 (320x240)", output)
        self.assertIn("1 dishes have no overhead image: dish_2", output)

    def test_pilot_is_deterministic_and_finds_downloaded_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            first = load_dataset(root)
            image = root / "imagery" / "realsense_overhead" / first.train[0].dish_id / "rgb.png"
            image.parent.mkdir(parents=True)
            image.touch()
            second = load_dataset(root)

            self.assertEqual((len(first.train), len(first.test)), (32, 8))
            first_ids = [record.dish_id for record in first.train]
            second_ids = [record.dish_id for record in second.train]
            self.assertEqual(first_ids, second_ids)
            self.assertIsNone(first.train[0].image)
            self.assertEqual(second.train[0].image, image.resolve())
            self.assertIsNone(second.train[1].image)

    def test_preprocess_finds_impossible_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata").mkdir()
            ingredient = "ingr_0000000001,olives,{grams},10.0,1.0,1.0,1.0"
            rows = (
                "dish_1,100.0,200.0,5.0,10.0,5.0," + ingredient.format(grams=200.0),  # consistent
                "dish_2,9000.0,7974.0,800.0,500.0,60.0," + ingredient.format(grams=7974.0),  # huge ingredient
                "dish_3,2000.0,100.0,200.0,10.0,5.0," + ingredient.format(grams=100.0),  # 20 kcal/g
                "dish_4,100.0,150.0,5.0,10.0,5.0," + ingredient.format(grams=100.0),  # grams != mass
            )
            (root / "metadata" / "dish_metadata_cafe1.csv").write_text("\n".join(rows) + "\n")
            (root / "metadata" / "dish_metadata_cafe2.csv").write_text("")

            errors = label_errors(root)

        self.assertEqual(set(errors), {"dish_2", "dish_3", "dish_4"})
        self.assertIn("1500", errors["dish_2"])
        self.assertIn("kcal/g", errors["dish_3"])
        self.assertIn("total mass", errors["dish_4"])

    def test_preprocess_writes_exclusions_and_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata").mkdir()
            ingredient = "ingr_0000000001,olives,{grams},10.0,1.0,1.0,1.0"
            (root / "metadata" / "dish_metadata_cafe1.csv").write_text(
                "dish_2,9000.0,7974.0,800.0,500.0,60.0," + ingredient.format(grams=7974.0) + "\n"
            )
            (root / "metadata" / "dish_metadata_cafe2.csv").write_text("")

            count, group_count = write_preprocess(root)
            excluded = {line.split(",")[0] for line in (root / "excluded_dishes.csv").read_text().splitlines()[1:]}
            groups = (root / "dish_groups.csv").read_text().splitlines()

        self.assertEqual(excluded, {"dish_2", *MANUAL_EXCLUSIONS})
        self.assertEqual(count, len(excluded))
        self.assertEqual((groups[0], groups[1], group_count), ("dish_id,group", "dish_2,dish_2", 1))

    def test_scan_groups_join_consecutive_scans_of_one_plate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata").mkdir()

            def row(dish_id: str, *ingredients: tuple[str, float]) -> str:
                items = ",".join(f"{key},name,{grams},1.0,1.0,1.0,1.0" for key, grams in ingredients)
                return f"{dish_id},100.0,{sum(grams for _, grams in ingredients)},1.0,1.0,1.0,{items}"

            rows = (
                row("dish_1000", ("ingr_1", 50.0)),
                row("dish_1030", ("ingr_1", 50.0), ("ingr_2", 30.0)),  # adds an item, 30 s later
                row("dish_1060", ("ingr_2", 30.0), ("ingr_1", 49.5)),  # same plate scanned again
                row("dish_1090", ("ingr_3", 80.0)),  # another plate
                row("dish_2000", ("ingr_3", 80.0), ("ingr_4", 5.0)),  # contains it, but 910 s later
            )
            (root / "metadata" / "dish_metadata_cafe1.csv").write_text("\n".join(rows) + "\n")
            (root / "metadata" / "dish_metadata_cafe2.csv").write_text("")

            groups = scan_groups(root)

        self.assertEqual(
            groups,
            {"dish_1000": "dish_1000", "dish_1030": "dish_1000", "dish_1060": "dish_1000",
             "dish_1090": "dish_1090", "dish_2000": "dish_2000"},
        )

    def test_validation_holds_out_whole_groups(self) -> None:
        records = tuple(
            DishRecord(f"dish_{index:04d}", 1.0, 1.0, 1.0, 1.0, 1.0) for index in range(100)
        )
        groups = {record.dish_id: f"plate_{index // 4}" for index, record in enumerate(records)}

        train, validation = split_validation(records, 0.1, groups)
        ungrouped = split_validation(records, 0.1)

        held_out = {groups[record.dish_id] for record in validation}
        self.assertTrue(held_out.isdisjoint(groups[record.dish_id] for record in train))
        self.assertEqual(len(validation), 12)  # three whole groups of four reach the 10 dishes
        self.assertEqual(len(ungrouped[1]), 10)  # without groups, exactly 10% of the dishes

    def test_excluded_dishes_leave_train_but_stay_in_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root, train_count=4, test_count=2)
            (root / "excluded_dishes.csv").write_text(
                "dish_id,reasons\ndish_0000000001,x\ndish_0000001000,x\n"
            )
            (root / "dish_groups.csv").write_text("dish_id,group\ndish_0000000000,dish_0000000000\n")

            dataset = load_dataset(root, full=True)

        self.assertNotIn("dish_0000000001", {record.dish_id for record in dataset.train})
        self.assertEqual(len(dataset.train), 3)
        self.assertIn("dish_0000001000", {record.dish_id for record in dataset.test})
        self.assertEqual(dataset.excluded, {"dish_0000000001", "dish_0000001000"})

    def test_exclusions_without_groups_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root, train_count=4, test_count=2)
            (root / "excluded_dishes.csv").write_text("dish_id,reasons\ndish_0000000001,x\n")

            with self.assertRaisesRegex(Nutrition5kError, "run `nutrition5k preprocess`"):
                load_dataset(root, full=True)

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
            test_path = root / "dish_ids" / "splits" / "depth_test_ids.txt"
            existing = test_path.read_text(encoding="utf-8")
            test_path.write_text("dish_0000000000\n" + existing, encoding="utf-8")

            with self.assertRaisesRegex(Nutrition5kError, "splits overlap"):
                load_dataset(root)

    def test_missing_required_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(Nutrition5kError, "required Nutrition5k file is missing"):
                load_dataset(Path(temporary))

    def test_overhead_download_counts_files_missing_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            record = load_dataset(root).train[0]
            response = MagicMock()
            response.__enter__.return_value = response
            response.read.side_effect = [b"image", b""]

            with patch(
                "nutrition5k.download.urllib.request.urlopen",
                side_effect=[response, urllib.error.HTTPError("url", 404, "missing", {}, None)],
            ):
                summary = download_overhead((record,), root)

            folder = root / "imagery" / "realsense_overhead" / record.dish_id
            self.assertEqual((summary.downloaded, summary.missing), (1, 1))
            self.assertTrue((folder / "rgb.png").is_file())
            self.assertFalse((folder / "depth_raw.png").exists())

    def test_overhead_download_retries_timeouts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            record = load_dataset(root).train[0]
            responses = [MagicMock(), MagicMock()]
            for response in responses:
                response.__enter__.return_value = response
                response.read.side_effect = [b"image", b""]

            with patch(
                "nutrition5k.download.urllib.request.urlopen",
                side_effect=[TimeoutError("The read operation timed out"), *responses],
            ), patch("nutrition5k.download.time.sleep"):
                summary = download_overhead((record,), root)

            self.assertEqual(summary.downloaded, 2)

    def test_overhead_download_skips_empty_files_and_failed_dishes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            first, second = load_dataset(root).train[:2]

            def urlopen(url: str, timeout: int) -> MagicMock:
                if first.dish_id in url:
                    raise TimeoutError("timed out")
                response = MagicMock()
                response.__enter__.return_value = response
                response.read.side_effect = [b"image", b""] if url.endswith("rgb.png") else [b""]
                return response

            with patch("nutrition5k.download.urllib.request.urlopen", side_effect=urlopen), patch(
                "nutrition5k.download.time.sleep"
            ):
                summary = download_overhead((first, second), root)

            self.assertEqual(summary.failed, (first.dish_id,))
            self.assertEqual((summary.downloaded, summary.missing), (1, 1))
            folder = root / "imagery" / "realsense_overhead" / second.dish_id
            self.assertTrue((folder / "rgb.png").is_file())
            self.assertFalse((folder / "depth_raw.png").exists())


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

    def test_seed_summary_reports_mean_and_sample_sd(self) -> None:
        records = (DishRecord("dish_a", 10.0, 10.0, 10.0, 10.0, 10.0),)
        evaluations = [
            evaluate_predictions(records, {"dish_a": (value,) * 5})
            for value in (8.0, 10.0, 13.0)
        ]
        summary = summarize_seeds(evaluations)

        self.assertAlmostEqual(summary["mean_pmae"]["mean"], 50 / 3)
        self.assertAlmostEqual(summary["mean_pmae"]["sd"], (700 / 3) ** 0.5)
        self.assertEqual(summary["metrics"]["total_mass"]["rsquared"], {"mean": None, "sd": None})


@unittest.skipUnless(importlib.util.find_spec("timm"), "training extra not installed")
class MobileNetTest(unittest.TestCase):
    def test_every_head_layout_predicts_five_targets_in_order(self) -> None:
        import torch

        from nutrition5k.models.nutrition_net import NutritionNet

        images = torch.zeros(2, 3, 64, 64)
        for heads in ("single", "grouped", "per-target"):
            model = NutritionNet("v4s", heads, hidden=(), pretrained=False).eval()
            model.freeze_backbone(1)
            self.assertEqual(model(images).shape, (2, 5))
        self.assertFalse(model.backbone.conv_stem.weight.requires_grad)
        self.assertTrue(model.backbone.conv_head.weight.requires_grad)

    def test_backbone_batchnorm_keeps_its_statistics_but_still_trains(self) -> None:
        import torch

        from nutrition5k.models.nutrition_net import NutritionNet

        model = NutritionNet("v4s", "single", hidden=(), pretrained=False).train()
        norms = [m for m in model.backbone.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        self.assertTrue(norms)
        self.assertFalse(any(norm.training for norm in norms))
        self.assertTrue(all(norm.weight.requires_grad for norm in norms))
        self.assertTrue(model.backbone.conv_stem.training)

        before = norms[0].running_mean.clone()
        model(torch.rand(2, 3, 64, 64))
        self.assertTrue(torch.equal(norms[0].running_mean, before))

    def test_predict_crops_the_plate_and_undoes_the_target_scale(self) -> None:
        import json

        import torch

        from nutrition5k.models.nutrition_net import NutritionNet
        from nutrition5k.training.regression import _transform, crop_plate, predict_image

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model = NutritionNet("v4s", "grouped", hidden=(8,), pretrained=False).eval()
            torch.save(model.state_dict(), run / "best.pt")
            scale = [100.0, 200.0, 10.0, 20.0, 30.0]
            config = {"backbone": "v4s", "heads": "grouped", "hidden": [8], "image_size": 64}
            (run / "config.json").write_text(json.dumps({**config, "target_scale": scale}))
            image = run / "rgb.png"
            Image.effect_noise((640, 480), 64).convert("RGB").save(image)
            Image.new("RGB", (320, 240)).save(run / "small.png")

            self.assertEqual(crop_plate(Image.open(image)).size, (448, 448))
            batch = _transform(model, 64)(Image.open(image).convert("RGB")).unsqueeze(0)
            self.assertEqual(batch.shape, (1, 3, 64, 64))
            with torch.no_grad():
                expected = model(batch)[0] * torch.tensor(scale)
            estimate = predict_image(run, image)
            for value, reference in zip(estimate, expected.tolist(), strict=True):
                self.assertAlmostEqual(value, reference, places=4)

            self.assertEqual(main(["predict", "--run", directory, "missing.png"]), 2)
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                self.assertEqual(main(["predict", "--run", directory, str(run / "small.png")]), 2)
            self.assertIn("needs a 640x480 overhead image", stderr.getvalue())


@unittest.skipUnless(importlib.util.find_spec("timm"), "training extra not installed")
class FamilyTest(unittest.TestCase):
    def test_every_backbone_belongs_to_exactly_one_family(self) -> None:
        from nutrition5k.models.families import BACKBONES, FAMILIES, FAMILY

        self.assertEqual(sum(len(family) for family in FAMILIES.values()), len(BACKBONES))
        self.assertEqual(set(FAMILY), set(BACKBONES))

    def test_every_non_mobilenet_backbone_freezes_all_but_its_last_stage(self) -> None:
        import torch

        from nutrition5k.models.families import FAMILIES
        from nutrition5k.models.nutrition_net import NutritionNet

        images = torch.zeros(1, 3, 224, 224)
        for family, backbones in FAMILIES.items():
            if family == "mobilenet":  # covered by the MobileNet tests
                continue
            for backbone in backbones:
                with self.subTest(backbone=backbone):
                    model = NutritionNet(backbone, "grouped", hidden=(), pretrained=False).eval()
                    model.freeze_backbone(1)
                    self.assertEqual(model(images).shape, (1, 5))
                    stages = [getattr(model.backbone, name, None) for name in ("blocks", "stages", "layers")]
                    stages = next(stage for stage in stages if stage is not None)
                    self.assertFalse(any(p.requires_grad for p in stages[0].parameters()))
                    self.assertTrue(all(p.requires_grad for p in stages[-1].parameters()))
                    stem = next(
                        getattr(model.backbone, name)
                        for name in ("patch_embed", "stem", "conv_stem")
                        if hasattr(model.backbone, name)
                    )
                    self.assertFalse(any(p.requires_grad for p in stem.parameters()))

    def test_backbone_lr_is_in_the_run_name_only_when_changed(self) -> None:
        from nutrition5k.training.regression import TrainConfig

        default = TrainConfig(backbone="deit_t").run_name("full")
        changed = TrainConfig(backbone="deit_t", backbone_lr=3e-5).run_name("full")
        self.assertEqual(default, "deit_t-ft_all-single-h512-224-full-s0")
        self.assertEqual(changed, "deit_t-ft_all-single-h512-224-blr3e-05-full-s0")

    def test_head_options_add_a_suffix_only_when_changed(self) -> None:
        from nutrition5k.training.heads import HeadConfig
        from nutrition5k.training.regression import TrainConfig

        changed = TrainConfig(backbone="deit_t", head_width=0, pooling="mean", augment="d4")
        self.assertEqual(changed.run_name("full"), "deit_t-ft_all-single-h512-224-hw0-mean-d4-full-s0")
        head = HeadConfig(backbone="v3", heads="grouped", hidden=(1024, 1024))
        self.assertEqual(head.run_name("full"), "v3-frozen-grouped-h1024x1024-hw256-flip-full-s0")

    def test_every_backbone_has_no_dropout_or_drop_path(self) -> None:
        import torch

        from nutrition5k.models.families import BACKBONES
        from nutrition5k.models.nutrition_net import NutritionNet

        for backbone in BACKBONES:
            with self.subTest(backbone=backbone):
                model = NutritionNet(backbone, pretrained=False)
                for module in model.modules():
                    self.assertFalse(getattr(module, "drop_prob", 0))  # timm DropPath
                    if isinstance(module, torch.nn.Dropout):
                        self.assertEqual(module.p, 0)

    def test_linear_heads_and_vit_pooling(self) -> None:
        import torch

        from nutrition5k.models.nutrition_net import NutritionNet, RegressionHead

        single, grouped = RegressionHead(8, "single", (), 0), RegressionHead(8, "grouped", (), 0)
        count = lambda head: sum(p.numel() for p in head.parameters())
        self.assertEqual(count(single), count(grouped))  # a linear head is the same however it is split
        self.assertEqual(grouped(torch.zeros(2, 8)).shape, (2, 5))

        images = torch.rand(2, 3, 224, 224)
        default = NutritionNet("deit_t", pretrained=False).eval()
        cls = NutritionNet("deit_t", pretrained=False, pooling="cls").eval()
        cls.load_state_dict(default.state_dict())
        both = NutritionNet("deit_t", pretrained=False, pooling="cls+mean").eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(default.features(images), cls.features(images), atol=1e-6))
            self.assertEqual(both.features(images).shape, (2, 384))
            self.assertEqual(both(images).shape, (2, 5))
        with self.assertRaises(ValueError):
            NutritionNet("v3", pretrained=False, pooling="mean")

    def test_dihedral_gives_eight_distinct_symmetries(self) -> None:
        from nutrition5k.training.regression import AUGMENT_VARIANTS, dihedral

        image = Image.effect_noise((16, 16), 64)
        variants = {dihedral(image, variant).tobytes() for variant in range(8)}
        self.assertEqual(len(variants), 8)
        self.assertEqual(dihedral(image, 0).tobytes(), image.tobytes())
        flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        self.assertEqual(dihedral(image, AUGMENT_VARIANTS["flip"][1]).tobytes(), flipped.tobytes())

    def test_train_requires_preprocess(self) -> None:
        dataset = SimpleNamespace(excluded=frozenset())
        with (
            patch("nutrition5k.cli.load_dataset", return_value=dataset),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            self.assertEqual(main(["train"]), 2)
        self.assertIn("run `nutrition5k preprocess` first", stderr.getvalue())

    def test_fixed_size_families_reject_other_image_sizes(self) -> None:
        for backbone, family in (("swin_t", "hierarchical"), ("deit_t", "plain_vit")):
            with self.subTest(backbone=backbone):
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    self.assertEqual(main(["train", "--backbone", backbone, "--image-size", "256"]), 2)
                self.assertIn(f"{family} backbones run at 224 px", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
