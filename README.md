# Multimodal nutrition thesis code

This repository currently provides a small, dependency-free Nutrition5k data
interface. The same API returns dish records in `pilot` and `full` modes, so
models and evaluators do not need mode-specific logic.

## Nutrition5k setup

Nutrition5k was published by Thames et al. with 5,006 cafeteria dish scans.
The official project and data are licensed under CC BY 4.0:

- <https://github.com/google-research-datasets/Nutrition5k>
- `gs://nutrition5k_dataset/nutrition5k_dataset/`

The complete compressed dataset is approximately 181.4 GB. Raw data belongs in
the ignored `data/nutrition5k/` directory and must never be committed. To fetch
only the metadata, official splits, and side-angle videos needed here:

```bash
mkdir -p data/nutrition5k/imagery
gsutil -m cp -r \
  gs://nutrition5k_dataset/nutrition5k_dataset/metadata \
  gs://nutrition5k_dataset/nutrition5k_dataset/dish_ids \
  data/nutrition5k/
gsutil -m cp -r \
  gs://nutrition5k_dataset/nutrition5k_dataset/imagery/side_angles \
  data/nutrition5k/imagery/
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
official split using seed `20260916` and SHA-256 ranking. This makes the sample
independent of input order and Python version. The smoke command prints the
exact IDs, counts, and one representative record as JSON. The checked pilot
manifest is recorded in [`docs/nutrition5k-pilot.md`](docs/nutrition5k-pilot.md).
Full mode changes only one option:

```bash
python3 -m nutrition5k smoke --root data/nutrition5k --mode full
```

Extract every fifth frame from cameras A-D for the selected pilot dishes:

```bash
python3 -m nutrition5k extract-frames \
  --root data/nutrition5k --mode pilot --frame-stride 5
python3 -m nutrition5k smoke \
  --root data/nutrition5k --mode pilot --frame-stride 5 --require-frames
```

`ffmpeg` must be available for extraction. Run `extract-frames` with
`--overwrite` only when existing outputs should be regenerated. Frame names
and the `frames_sampled5/` layout match the official extraction script.

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
