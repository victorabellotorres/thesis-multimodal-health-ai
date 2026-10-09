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

There are seven commands (`train` and `predict` are described below):

```bash
# Remove all local Nutrition5k data preparation, preserving baseline outputs.
python3 -m nutrition5k clear

# Download the required official metadata and RGB splits.
python3 -m nutrition5k fetch

# Validate the local 32/8 pilot.
python3 -m nutrition5k check

# Minimal mode (default): download only the first available camera and
# extract only its first frame.
python3 -m nutrition5k prepare

# Complete mode: download every available side-angle camera and extract every
# fifth frame of each video.
python3 -m nutrition5k prepare --mode complete

# Fit and evaluate the metadata-only average baseline.
python3 -m nutrition5k baseline
```

Add `--full` to any command to use the complete official RGB split:

```bash
python3 -m nutrition5k prepare --full
python3 -m nutrition5k baseline --full
```

`prepare` supports exactly two frame-extraction modes, in pilot and full mode:

- `minimal` (default): downloads only the first available camera and writes its first
  frame to `imagery/side_angles/<dish_id>/frames_first_320/`, for quick inspection
  and smoke tests. It is the default so that a mistaken command never starts
  the long complete download.
- `complete`: tries cameras A through D, downloads every available
  video, and writes every fifth frame of each video to
  `imagery/side_angles/<dish_id>/frames_sampled5_320/`. This matches the official
  Nutrition5k side-angle procedure and is the input for learned visual models.

`check --mode complete` checks the complete frames instead of the minimal ones. It
reports, per split, how many dishes have frames and the frames per dish, the frames
per camera, and lists the dishes with no frames, with leftover videos (unfinished
`prepare`) and, in complete mode, with some cameras missing. It exits with 1 when
any dish has no frames or leftover videos. Each list shows its first 5 dishes;
add `--all` to print every one.
Frames are stored with their shorter side scaled to 320 px (the models use
224-256 px inputs). `prepare` works in batches of 100 dishes: it downloads the
videos of a batch, extracts their frames, and deletes the videos, so only one
batch of videos is on disk at a time. Dishes that already have frames and no
leftover videos are skipped, so an interrupted run resumes with the same
command.

The baseline writes three readable files to
`outputs/average-baseline/pilot/` or `outputs/average-baseline/full/`:

- `model.json`: training means
- `predictions.csv`: one prediction per test dish
- `evaluation.json`: MAE and percentage MAE for the five nutrition targets

The pilot selection is fixed, while the frame-extraction mode is an explicit
`prepare` option. Existing videos and frames are reused automatically.

`fetch` is also resumable: it preserves existing non-empty metadata and split
files, and downloads new files through temporary `.part` files. To begin the
entire data pipeline from scratch, run `clear`, then `fetch` followed by
`prepare` (add `--full` for the complete RGB split). `clear` removes only
`data/nutrition5k/`; it preserves `outputs/average-baseline/`, source code,
the Git repository, and `.venv/`.

## MobileNet training

Training needs the optional extra (PyTorch, timm, plotting, W&B):

```bash
python3 -m pip install -e .[train]
```

`train` fine-tunes an ImageNet-1k MobileNet from timm (`v2`, `v3`, `v4s`,
`v4m`) to regress the five targets. It holds out a fixed 10% of the training
dishes for validation, picks the best epoch by validation mean PMAE, and only
then evaluates on the official test split:

```bash
# Default: V4-Conv-S, full fine-tuning, one shared head, 512-unit trunk, 224 px.
python3 -m nutrition5k train --full --mode complete

# Ablations: frozen backbone (--unfreeze 0) or last N stages, head grouping,
# trunk size (--hidden 0 for none, repeat for several layers), resolution.
python3 -m nutrition5k train --backbone v4s --unfreeze 2 --heads grouped \
    --hidden 1024 --hidden 1024 --image-size 256 --full --mode complete
```

Each configuration writes to `outputs/mobilenet/<run-name>/`:

- `config.json`: every hyperparameter, parameter counts, split sizes, target
  scale, and the environment (GPU, torch and timm versions, git commit, start
  time)
- `history.csv`: one row per epoch with `train_loss`, `val_loss` (the same
  scaled L1, on dish-level validation predictions), validation mean and
  per-target PMAE, `seconds`, and peak `gpu_memory_gb`
- `best.pt` (best validation epoch), `last.pt` (for resuming)
- `predictions[_val].csv` plus `evaluation[_val].json` (MAE, PMAE, RMSE, R²
  per target)

A finished run is skipped when repeated, and an interrupted run resumes from
`last.pt`.

If the `WANDB_API_KEY` environment variable is set, each run is also mirrored
to [Weights & Biases](https://wandb.ai) (project `nutrition5k`, run id = run
name, so a resumed run continues the same curves): the `history.csv` rows,
the config, and the final validation and test PMAE. Without the variable
nothing is sent. The local files remain the record of the results.

`predict` estimates one dish from one or more of its images with a run's
`best.pt` (the images are averaged, as in evaluation). It runs on CPU:

```bash
python3 -m nutrition5k predict --run outputs/mobilenet/<run-name> photo.jpg
```

`--all-seeds` runs the configuration with each fixed seed (0, 1, 2, `SEEDS`
in `training/mobilenet.py`) and writes the mean and sample SD of every
validation and test metric across them to
`outputs/mobilenet/seeds/<run-name-without-seed>.json`:

```bash
python3 -m nutrition5k train --full --mode complete --all-seeds
```

Notebooks:

- `notebooks/02_experiments.ipynb` runs the baseline and the staged
  experiments locally, on Colab (results on Google Drive) or on a RunPod pod
  (results on the network volume).
- `notebooks/03_results.ipynb` trains nothing; it reads `outputs/` and shows
  comparison tables and charts for choosing configurations by validation,
  test metrics of the final models, and a gallery of test dishes with real
  and predicted values.

## RunPod

Only `train` needs a GPU. Data preparation runs on the laptop, and the frames
are uploaded once:

1. Locally: `python3 -m nutrition5k prepare --full --mode complete`, then
   `tar -cf data-full-complete.tar --exclude='*.h264' data`.
2. On RunPod, create a network volume and a Secure Cloud GPU pod with the
   official PyTorch template in the same data center, with at least 8 vCPUs
   (they decode the JPEGs). To use W&B, store your key as a RunPod secret and
   set the pod environment variable `WANDB_API_KEY` to
   `{{ RUNPOD_SECRET_<secret name> }}`.
3. Upload the data: `runpodctl send data-full-complete.tar` on the laptop, and
   `runpodctl receive <code>` inside `/workspace/nutrition5k/` on the pod.
4. Open `notebooks/02_experiments.ipynb` on the pod (VS Code Remote-SSH or the
   pod's JupyterLab). It clones this repository to `/root/`, extracts the
   `.tar` to the pod's local disk and links `outputs/` to
   `/workspace/nutrition5k/outputs`. Long runs can also be launched with the
   same `python -m nutrition5k train ...` command inside `tmux`.
5. After each stage, copy the results to the laptop and stop the pod:
   `tar -cf outputs.tar --exclude=last.pt outputs/` and `runpodctl send
   outputs.tar` on the pod, `runpodctl receive <code>` and `tar -xf
   outputs.tar` in this directory locally.

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
