# Nutrition5k thesis code

This repository contains the small data pipeline and average baseline used by
the thesis. Run every command from this directory.

## Setup

Python 3.10 or newer is required:

```bash
python3 -m pip install -e .
```

Download the official metadata and RGB split files once:

```bash
mkdir -p data/nutrition5k
gsutil -m cp -r \
  gs://nutrition5k_dataset/nutrition5k_dataset/metadata \
  gs://nutrition5k_dataset/nutrition5k_dataset/dish_ids \
  data/nutrition5k/
```

Raw data stays in the ignored `data/nutrition5k/` directory. The source is the
official [Nutrition5k dataset](https://github.com/google-research-datasets/Nutrition5k),
licensed under CC BY 4.0.

## Commands

There are only three commands:

```bash
# Validate the local 32/8 pilot.
python3 -m nutrition5k check

# Download its side-angle videos and extract every fifth frame.
python3 -m nutrition5k prepare

# Fit and evaluate the metadata-only average baseline.
python3 -m nutrition5k baseline
```

Add `--full` to any command to use the complete official RGB split:

```bash
python3 -m nutrition5k prepare --full
python3 -m nutrition5k baseline --full
```

Full preparation downloads thousands of side-angle videos. Its exact download
size is **TBD**; the complete Nutrition5k dataset is approximately 181.4 GB.
Check available disk space first. Existing videos and frames are skipped, so an
interrupted download can be resumed with the same command.

The baseline writes three readable files to
`outputs/average-baseline/pilot/` or `outputs/average-baseline/full/`:

- `model.json`: training means
- `predictions.csv`: one prediction per test dish
- `evaluation.json`: MAE and percentage MAE for the five nutrition targets

The pilot selection and frame sampling are fixed research decisions, so they
are not CLI settings. Existing videos and frames are reused automatically.

## Python use

```python
from pathlib import Path
from nutrition5k import load_dataset

dataset = load_dataset(Path("data/nutrition5k"))
for dish in dataset.train:
    print(dish.dish_id, dish.frame_paths, dish.targets)
```

Run the nine focused tests with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
