# Multimodal nutrition thesis code

This repository currently provides a small Nutrition5k data interface. The same
API returns dish records in `pilot` and `full` modes, so models and evaluators
do not need mode-specific logic.

## Nutrition5k setup

Nutrition5k was published by Thames et al. with 5,006 cafeteria dish scans.
The official project and data are licensed under CC BY 4.0:

- <https://github.com/google-research-datasets/Nutrition5k>
- `gs://nutrition5k_dataset/nutrition5k_dataset/`

The complete compressed dataset is approximately 181.4 GB. Raw data belongs in
the ignored `data/nutrition5k/` directory and must never be committed. To fetch
the metadata and official splits needed by both metadata and visual workflows:

```bash
mkdir -p data/nutrition5k
gsutil -m cp -r \
  gs://nutrition5k_dataset/nutrition5k_dataset/metadata \
  gs://nutrition5k_dataset/nutrition5k_dataset/dish_ids \
  data/nutrition5k/
```

Expected input layout:

```text
data/nutrition5k/
├── metadata/
│   ├── dish_metadata_cafe1.csv
│   └── dish_metadata_cafe2.csv
├── dish_ids/splits/
│   ├── rgb_train_ids.txt
│   └── rgb_test_ids.txt
└── imagery/side_angles/dish_.../
    ├── camera_A.h264 ... camera_D.h264
    └── frames_sampled5/              # generated when requested
```

The RGB split is the official split used for the side-angle visual model. The
dataset interface treats mass, calories, fat, carbohydrate, and protein as
prediction targets; it does not expose them as image-model inputs.

## Usage

Python 3.10 or later is required. An editable installation is convenient but
not required:

```bash
python3 -m pip install -e .
python3 -m nutrition5k smoke --root data/nutrition5k --mode pilot
```

Pilot mode deterministically samples 32 train and 8 test dishes from the
official split using seed `20260920` and SHA-256 ranking. This is the first seed
from `20260916` onward whose selected dishes all have side-angle video coverage,
and it makes the sample independent of input order and Python version. The
smoke command prints the exact IDs, counts, and one representative record as
JSON; it is read-only and does not download videos or extract frames. The
checked pilot manifest is recorded in
[`docs/nutrition5k-pilot.md`](docs/nutrition5k-pilot.md).

Materialize the complete visual pilot with one command:

```bash
python3 -m nutrition5k prepare-imagery \
  --root data/nutrition5k --mode pilot --frame-stride 5
```

This selects the same 32/8 dishes, downloads the available cameras A-D for each
dish directly from the official bucket, and extracts every fifth frame. Some
official dishes omit an individual camera file; these are reported and the
remaining camera views are retained, while a dish with no available video is
rejected. Existing non-empty videos are skipped, making interrupted preparation
resumable. `--overwrite` forces downloads and frame extraction to be regenerated.

Full metadata mode changes only one option:

```bash
python3 -m nutrition5k smoke --root data/nutrition5k --mode full
```

## Average baseline and shared evaluation

The metadata-only average baseline learns one raw-unit mean for each of mass
(`g`), calories (`kcal`), fat (`g`), carbohydrate (`g`), and protein (`g`)
from the selected training split. It then reloads those parameters and emits
the same five-value vector for each selected test dish. It does not read test
targets while fitting and needs neither extracted images nor a GPU.

```bash
python3 -m nutrition5k average-baseline \
  --root data/nutrition5k --mode pilot \
  --output-dir outputs/average-baseline/pilot
```

Use `--mode full` with the same command once the complete metadata and official
splits are available. The output directory contains `parameters.json`,
`predictions.csv`, `evaluation.json`, and `run.json`. Existing artifacts are
protected unless `--overwrite` is supplied.

`evaluation.json` is the shared model-independent evaluator contract. It
validates the dish IDs and finite five-target prediction schema, reports MAE in
native units and percentage MAE per target, emits per-dish errors, and provides
macro summaries. Its percentage-MAE calculation matches the official
Nutrition5k script: `100 × MAE ÷ mean(ground truth)` per target; it is `null`
if that mean is zero. R² is secondary and is `null` when the test target has
fewer than two observations or has zero variance.

The two preparation stages can also be run separately:

```bash
python3 -m nutrition5k download-videos \
  --root data/nutrition5k --mode pilot
python3 -m nutrition5k extract-frames \
  --root data/nutrition5k --mode pilot --frame-stride 5
python3 -m nutrition5k smoke \
  --root data/nutrition5k --mode pilot --frame-stride 5 --require-frames
```

Installing the project with `python3 -m pip install -e .` installs
`imageio-ffmpeg`, which provides the FFmpeg executable used for extraction.
Run `extract-frames` with `--overwrite` only when existing outputs should be
regenerated. Frame names and the `frames_sampled5/` layout match the official
extraction script.

## Python interface

```python
from pathlib import Path
from nutrition5k import DatasetConfig, DatasetMode, Nutrition5kDataset

dataset = Nutrition5kDataset.load(
    DatasetConfig(root=Path("data/nutrition5k"), mode=DatasetMode.PILOT)
)
for dish in dataset.train:
    print(dish.dish_id, dish.frame_paths, dish.targets)
```

Run the focused tests without third-party packages:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
