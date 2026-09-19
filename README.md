# Nutrition5k thesis code

This repository contains the small data pipeline and average baseline used by
the thesis. Run every command from this directory.

## Structure

```text
notebooks/             thin Colab entry points
src/nutrition5k/
  models/              model definitions
  training/            training and experiment runners
  dataset.py           dataset records and deterministic splits
  evaluation.py        shared metrics and artifact writers
tests/                 focused automated tests
```

Notebooks call the package instead of duplicating model or training code. Raw
data, outputs, runs, checkpoints, and the local virtual environment are ignored
by Git.

## Setup

Python 3.10 or newer is required:

```bash
python3 -m pip install -e .
```

For Colab, open
[`notebooks/01_colab_smoke_test.ipynb`](https://colab.research.google.com/github/victorabellotorres/thesis-multimodal-health-ai/blob/main/notebooks/01_colab_smoke_test.ipynb).
It clones this repository, installs it in editable mode, fetches only the small
official dataset index, and runs the pilot baseline and test suite. The notebook
deliberately leaves visual data preparation as an optional step.

Fetch the official metadata and RGB split files once:

```bash
python3 -m nutrition5k fetch
```

Raw data stays in the ignored `data/nutrition5k/` directory. The source is the
official [Nutrition5k dataset](https://github.com/google-research-datasets/Nutrition5k),
licensed under CC BY 4.0.

## Commands

There are five commands:

```bash
# Remove all local Nutrition5k data preparation, preserving baseline outputs.
python3 -m nutrition5k clear

# Download the required official metadata and RGB splits.
python3 -m nutrition5k fetch

# Validate the local 32/8 pilot.
python3 -m nutrition5k check

# Download all available side-angle videos and extract their first frames.
python3 -m nutrition5k prepare

# Download only the first available video and extract only its first frame.
python3 -m nutrition5k prepare --one-video --one-frame

# Fit and evaluate the metadata-only average baseline.
python3 -m nutrition5k baseline
```

Add `--full` to any command to use the complete official RGB split:

```bash
python3 -m nutrition5k prepare --full
python3 -m nutrition5k baseline --full
```

By default, preparation tries cameras A through D and downloads every available
video for each dish. `--one-video` stops after the first available camera.
Both options work in pilot and full mode. Extraction writes the first frame of
each downloaded video by default. `--one-frame` extracts only the first frame
of the first available video for each dish. Full preparation still downloads
thousands of videos. Its exact download size is **TBD**; the complete
Nutrition5k dataset is approximately 181.4 GB.
Check available disk space first. Existing videos and frames are skipped, so an
interrupted download can be resumed with the same command.

The baseline writes three readable files to
`outputs/average-baseline/pilot/` or `outputs/average-baseline/full/`:

- `model.json`: training means
- `predictions.csv`: one prediction per test dish
- `evaluation.json`: MAE and percentage MAE for the five nutrition targets

The pilot selection is fixed, while video and frame reduction are explicit
`prepare` options. Existing videos and frames are reused automatically.

`fetch` is also resumable: it preserves existing non-empty metadata and split
files, and downloads new files through temporary `.part` files. To begin the
entire data pipeline from scratch, run `clear`, then `fetch` followed by
`prepare` (add `--full` for the complete RGB split). `clear` removes only
`data/nutrition5k/`; it preserves `outputs/average-baseline/`, source code,
the Git repository, and `.venv/`.

## Python use

```python
from pathlib import Path
from nutrition5k import load_dataset

dataset = load_dataset(Path("data/nutrition5k"))
for dish in dataset.train:
    print(dish.dish_id, dish.frame_paths, dish.targets)
```

Run the focused tests with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
