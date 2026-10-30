# Nutrition5k thesis code

This repository contains the small data pipeline and average baseline used by
the thesis. Run every command from this directory.

## Structure

```text
notebooks/             thin entry points (local or RunPod)
  smoke_test           fetch the index, pilot baseline and tests
  data_preprocess/
    nutrition5k/       dataset exploration, incremental scans, views and crop
  <family>/            experiments and results of one backbone family
  results_general      one finalist per family, side by side
src/nutrition5k/
  models/              model definitions: nutrition_net.py (shared regressor),
                       families.py (registry) and one module per family:
                       mobilenet.py, hybrid_mobile.py, plain_vit.py,
                       hierarchical.py
  training/            training and experiment runners (regression.py: shared
                       by every family)
  results.py           tables and charts read from outputs/ (results notebooks)
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

`notebooks/smoke_test.ipynb` fetches only the small official dataset index
and runs the pilot baseline and test suite. The notebook
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

`preprocess` (also run at the end of `prepare`; it only needs the metadata)
writes `excluded_dishes.csv` to `data/nutrition5k/`: the dishes whose labels
are impossible or inconsistent (ingredient grams more than 10% away from the
total mass, more than 9 kcal/g, or one ingredient over 1.5 kg, such as "olives
7974 g"), plus the dishes listed by hand in `MANUAL_EXCLUSIONS`
(`src/nutrition5k/preprocess.py`; for now one whose images show a phone, not
food). Those dishes are left out of train, and so of validation. Test keeps
them: runs and the baseline write `evaluation.json` on the official test and
`evaluation_clean.json` without them.

It also writes `dish_groups.csv`: the scan group of every dish. Nutrition5k
scans a plate after each added item (and once more after mixing it); a dish
joins the previous dish's group when it was scanned at most 300 s later and
contains all its ingredients with at least the same grams. Validation holds out
whole groups, so no plate has scans in both train and validation (the official
split does the same for train and test). Runs trained on preprocessed data end
in `-clean`; `train` refuses to run with only one of the two files.

Runs made before `preprocess` existed are archived in
`outputs/previous_runs_to_data_preprocess/` (dish-by-dish validation, label
errors in train), not comparable with the new ones.

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

The baseline writes these readable files to
`outputs/average-baseline/pilot/` or `outputs/average-baseline/full/`:

- `model.json`: training means
- `predictions.csv`: one prediction per test dish
- `evaluation.json`: MAE and percentage MAE for the five nutrition targets
- `predictions_val.csv` and `evaluation_val.json`: the same on the validation
  split used by `train`, with the means fitted on the remaining training dishes

The pilot selection is fixed, while the frame-extraction mode is an explicit
`prepare` option. Existing videos and frames are reused automatically.

`fetch` is also resumable: it preserves existing non-empty metadata and split
files, and downloads new files through temporary `.part` files. To begin the
entire data pipeline from scratch, run `clear`, then `fetch` followed by
`prepare` (add `--full` for the complete RGB split). `clear` removes only
`data/nutrition5k/`; it preserves `outputs/average-baseline/`, source code,
the Git repository, and `.venv/`.

## Training

Training needs the optional extra (PyTorch, timm, plotting, W&B):

```bash
python3 -m pip install -e .[train]
```

`train` fine-tunes an ImageNet-1k MobileNet from timm (`v2`, `v3`, `v4s`,
`v4m`, all pretrained at 224 px, and `v4m-r256`, the same V4-Conv-M pretrained
at 256 px) to regress the five targets. It holds out a fixed 10% of the training
dishes for validation (whole scan groups, after `preprocess`), picks the best epoch by validation mean PMAE, and only
then evaluates on the official test split:

```bash
# Default: V4-Conv-S, full fine-tuning, one shared head, 512-unit trunk, 224 px.
python3 -m nutrition5k train --full --mode complete

# Ablations: frozen backbone (--unfreeze 0) or last N stages, head grouping,
# trunk size (--hidden 0 for none, repeat for several layers), resolution.
python3 -m nutrition5k train --backbone v4s --unfreeze 2 --heads grouped \
    --hidden 1024 --hidden 1024 --image-size 256 --full --mode complete
```

The same command trains every backbone with the same recipe. Backbones are
grouped by design family (`models/families.py`):

| Family | Backbones |
|---|---|
| `mobilenet` | `v2`, `v3`, `v4s`, `v4m`, `v4m-r256` |
| `hybrid_mobile` | `fastvit_s12`, `fastvit_sa12`, `v4hm` (MobileNetV4-Hybrid-M) |
| `plain_vit` | `deit_t`, `deit_s`, `dinov2_s` |
| `hierarchical` | `swin_t`, `convnext_t` |

`plain_vit` and `hierarchical` run at 224 px only. `--unfreeze N` trains the
last N blocks (MobileNets, DeiT, DINOv2) or stages (FastViT, Swin, ConvNeXt).
`--backbone-lr` changes the backbone learning rate (default 1e-4; another
value adds `-blr<lr>` to the run name):

```bash
python3 -m nutrition5k train --backbone dinov2_s --unfreeze 3 --backbone-lr 3e-5 \
    --full --mode complete
```

Each configuration writes to `outputs/<family>/<full|pilot>/<backbone>/<run-name>/`:

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
python3 -m nutrition5k predict --run outputs/mobilenet/full/v3/<run-name> photo.jpg
```

`--all-seeds` runs the configuration with each fixed seed (0, 1, 2, `SEEDS`
in `training/regression.py`) and writes the mean and sample SD of every
validation and test metric across them to
`outputs/<family>/<full|pilot>/seeds/<run-name-without-seed>.json`:

```bash
python3 -m nutrition5k train --full --mode complete --all-seeds
```

Notebooks:

- `notebooks/<family>/experiments.ipynb` runs the baseline and the staged
  experiments of one family locally or on a RunPod pod (results on the volume).
- `notebooks/<family>/results.ipynb` trains nothing; it reads `outputs/<family>/` and shows
  comparison tables and charts for choosing configurations by validation,
  cost (MACs, CPU latency), test metrics of the final models, and a gallery of
  test dishes with real and predicted values.
- `notebooks/results_general.ipynb` shows one finalist per family (averaged
  over its seeds): per-target error and error against cost.

Runs from before the split by family lived in `outputs/transformer/`. On a
volume that still has that layout, move them once from the repository root
(`outputs/` links to the volume):

```bash
mkdir -p outputs/plain_vit/full outputs/hybrid_mobile/full outputs/hierarchical/full
mv outputs/transformer/full/deit_t outputs/transformer/full/dinov2_s outputs/plain_vit/full/
mv outputs/transformer/full/fastvit_sa12 outputs/hybrid_mobile/full/
mv outputs/transformer/full/swin_t outputs/hierarchical/full/
```

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
4. Open `notebooks/<family>/experiments.ipynb` on the pod (VS Code Remote-SSH or the
   pod's JupyterLab). It clones this repository to `/root/`, extracts the
   `.tar` to the pod's local disk and links `outputs/` to
   `/workspace/nutrition5k/outputs`. Long runs can also be launched with the
   same `python -m nutrition5k train ...` command inside `tmux`.
5. After each stage, copy the results to the laptop and stop the pod:
   `tar -cf outputs.tar --exclude=last.pt outputs/` and `runpodctl send
   outputs.tar` on the pod, `runpodctl receive <code>` and `tar -xf
   outputs.tar` in this directory locally.

### Starting a pod from the web terminal

Paste these commands in the pod's web terminal (Connect > Web Terminal) once
the pod is running. They do the same as the setup cells of
`notebooks/mobilenet/experiments.ipynb`, so the notebook is optional:

```bash
# Repository on the pod's local disk, with the training extra.
git clone https://github.com/victorabellotorres/thesis-multimodal-health-ai.git /root/thesis-multimodal-health-ai
cd /root/thesis-multimodal-health-ai
pip install -e .[train]

# Results on the volume; frames extracted to the local disk.
mkdir -p /workspace/nutrition5k/outputs
ln -s /workspace/nutrition5k/outputs outputs
tar -xf /workspace/nutrition5k/data-full-complete.tar
python -m nutrition5k check --mode complete --full

# Long run that survives a closed terminal.
tmux new -s train
python -m nutrition5k train --full --mode complete
```

On a pod that already has the repository, update it instead of cloning:
`git -C /root/thesis-multimodal-health-ai pull --ff-only`. A new pod loses
`/root/`, so run the block again (the `outputs` link and the volume data
persist). Reattach to a running session with `tmux attach -t train`.

PyTorch and CUDA come from the RunPod template, not from this repository. The
versions used for the stage 1 and stage 2 runs are TBD. To record them, run on
the pod `python -c "import torch; print(torch.__version__, torch.version.cuda)"`
and `pip freeze > requirements-lock.txt`, then commit the lock file.

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
