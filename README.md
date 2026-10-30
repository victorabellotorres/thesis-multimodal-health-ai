# Nutrition5k thesis code

This repository contains the small data pipeline, average baseline and model
training used by the thesis. Every model sees the **overhead** photo of each
dish (Nutrition5k `realsense_overhead`). Run every command from this directory.

## Structure

```text
notebooks/             thin entry points (local or RunPod)
  smoke_test           fetch the index, pilot baseline and tests
  data_preprocess/
    nutrition5k/       dataset exploration, incremental scans, overhead crop
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

`antiguo/` (also ignored) keeps the data and runs of the side-angle pipeline
used until 2026-10-30: `antiguo/data/`, `antiguo/outputs/` (with the RunPod
runs and their `last.pt` checkpoints; `last.runpod.pt` is the volume's copy
where it differed from the local one) and `antiguo/data-{full,pilot}-complete.tar`.
They are not comparable with the overhead runs.

## Setup

Python 3.10 or newer is required:

```bash
python3 -m pip install -e .
```

`notebooks/smoke_test.ipynb` fetches only the small official dataset index
and runs the pilot baseline and test suite. The notebook
deliberately leaves visual data preparation as an optional step.

Fetch the official metadata and overhead split files once:

```bash
python3 -m nutrition5k fetch
```

Raw data stays in the ignored `data/nutrition5k/` directory. The source is the
official [Nutrition5k dataset](https://github.com/google-research-datasets/Nutrition5k),
licensed under CC BY 4.0.

## Commands

There are eight commands (`train` and `predict` are described below):

```bash
# Remove all local Nutrition5k data preparation, preserving baseline outputs.
python3 -m nutrition5k clear

# Download the required official metadata and overhead splits.
python3 -m nutrition5k fetch

# Download the overhead images of the 32/8 pilot, then preprocess.
python3 -m nutrition5k prepare

# Validate the local pilot.
python3 -m nutrition5k check

# Only the metadata steps of prepare: excluded dishes and scan groups.
python3 -m nutrition5k preprocess

# Fit and evaluate the metadata-only average baseline.
python3 -m nutrition5k baseline
```

Add `--full` to `prepare`, `check`, `baseline` and `train` to use the complete
official overhead split (2,758 train and 507 test dishes, the `depth_*_ids.txt`
files) instead of the pilot:

```bash
python3 -m nutrition5k prepare --full
python3 -m nutrition5k baseline --full
```

`prepare` downloads, for every dish of the split, the overhead capture of the
Intel RealSense D435 to `data/nutrition5k/imagery/realsense_overhead/<dish_id>/`:
`rgb.png` (640x480) and `depth_raw.png` (16-bit depth, x 1e-4 = metres; the
models do not use it yet). Dishes that already have both files are skipped, so
an interrupted run resumes with the same command, and files go through
temporary `.part` files. A file absent upstream counts as unavailable; a dish
with any other error is reported and retried by the next `prepare`.

The models do not see the whole photo: they take the fixed square
`CROP_BOX = (146, 0, 448)` (x0, y0, side; `src/nutrition5k/dataset.py`), which
holds the plate, and resize it to their input size (448 to 224 is exactly
half). The crop was chosen on a sample of train dishes in
`notebooks/data_preprocess/nutrition5k/views_and_crop.ipynb`.

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
split does the same for train and test). `train` refuses to run before
`preprocess`.

`check` reports, per split, how many dishes have an overhead image, and lists
the dishes with no image, with no depth image, or with an image that is not
640x480 (the crop assumes that size). It exits with 1 when any dish has no RGB
image or an unexpected size. Three train dishes of the official split have no
overhead image upstream (`dish_1557862384`, `dish_1558109511`,
`dish_1558109714`), so `check --full` always lists them and exits with 1;
`train` leaves them out. Each list shows its first 5 dishes; add `--all` to
print every one.

The baseline writes these readable files to
`outputs/average-baseline/pilot/` or `outputs/average-baseline/full/`:

- `model.json`: training means
- `predictions.csv`: one prediction per test dish
- `evaluation.json`: MAE and percentage MAE for the five nutrition targets
- `predictions_val.csv` and `evaluation_val.json`: the same on the validation
  split used by `train`, with the means fitted on the remaining training dishes

The pilot selection is fixed. `fetch` is also resumable: it preserves existing
non-empty metadata and split files, and downloads new files through temporary
`.part` files. To begin the
entire data pipeline from scratch, run `clear`, then `fetch` followed by
`prepare` (add `--full` for the complete overhead split). `clear` removes only
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
python3 -m nutrition5k train --full

# Ablations: frozen backbone (--unfreeze 0) or last N stages, head grouping,
# trunk size (--hidden 0 for none, repeat for several layers), resolution.
python3 -m nutrition5k train --backbone v4s --unfreeze 2 --heads grouped \
    --hidden 1024 --hidden 1024 --image-size 256 --full
```

The same command trains every backbone with the same recipe. Backbones are
grouped by design family (`models/families.py`):

| Family | Backbones |
|---|---|
| `mobilenet` | `v2`, `v3`, `v4s`, `v4m`, `v4m-r256` |
| `hybrid_mobile` | `fastvit_s12`, `fastvit_sa12`, `v4hm` (MobileNetV4-Hybrid-M) |
| `plain_vit` | `deit_t`, `deit_s`, `dinov2_s` |
| `hierarchical` | `swin_t`, `convnext_t` |

Every image goes through the same steps: the fixed plate crop, a random
horizontal flip (training only), and the backbone's own resize and
normalisation. `plain_vit` and `hierarchical` run at 224 px only. `--unfreeze N` trains the
last N blocks (MobileNets, DeiT, DINOv2) or stages (FastViT, Swin, ConvNeXt).
`--backbone-lr` changes the backbone learning rate (default 1e-4; another
value adds `-blr<lr>` to the run name):

```bash
python3 -m nutrition5k train --backbone dinov2_s --unfreeze 3 --backbone-lr 3e-5 --full
```

Each configuration writes to `outputs/<family>/<full|pilot>/<backbone>/<run-name>/`,
with run names such as `v4s-ft_all-single-h512-224-full-s0`
(backbone, fine-tuned stages, heads, trunk, input size, split, seed):

- `config.json`: every hyperparameter, the crop box, parameter counts, split
  sizes, target scale, and the environment (GPU, torch and timm versions, git
  commit, start time)
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

`predict` estimates one dish from its overhead `rgb.png` (640x480, so the
fixed crop applies) with a run's `best.pt`. It runs on CPU:

```bash
python3 -m nutrition5k predict --run outputs/mobilenet/full/v3/<run-name> \
    data/nutrition5k/imagery/realsense_overhead/<dish_id>/rgb.png
```

`--all-seeds` runs the configuration with each fixed seed (0, 1, 2, `SEEDS`
in `training/regression.py`) and writes the mean and sample SD of every
validation and test metric across them to
`outputs/<family>/<full|pilot>/seeds/<run-name-without-seed>.json`:

```bash
python3 -m nutrition5k train --full --all-seeds
```

Every backbone is created with dropout and stochastic depth at 0 (timm's
defaults differ: `swin_t` has drop path 0.1), and `config.json` records both and
the AdamW weight decay. `--head-width` sets the hidden layer of each head (0:
linear), `--pooling` (plain ViTs only: `cls`, `mean`, `cls+mean`) the pooling of
the tokens, and `--augment d4` replaces the horizontal flip by a random rotation
by 90° and flip of the square crop.

`heads` (stage 1) trains regression heads on a frozen backbone. The first call
runs the backbone once over every image under its 8 symmetries and caches the
features in `outputs/<family>/<full|pilot>/heads/<backbone>/features.pt`; each
head then trains on them in seconds (80 epochs, head lr 1e-3, same loss and
validation as `train`) and writes the same files as a run to
`heads/<backbone>/<backbone>-frozen-<heads>-h<trunk>-hw<head width>[-<pooling>]-<augment>-<split>-s<seed>/`.
`--grid` runs the 9 heads of the reduced grid (trunk 0, 512, 1024×1024 ×
linear single, single and grouped with a 256 hidden layer):

```bash
python3 -m nutrition5k heads --backbone v4m --grid --full
python3 -m nutrition5k heads --backbone deit_t --pooling mean --full
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

## RunPod

Only `train` and `heads` need a GPU. Data preparation runs on the laptop, and the images
are uploaded once:

1. Locally: `python3 -m nutrition5k prepare --full`, then
   `tar -cf data-full-overhead.tar data`.
2. On RunPod, create a network volume and a Secure Cloud GPU pod with the
   official PyTorch template in the same data center, with at least 8 vCPUs
   (they decode the images). To use W&B, store your key as a RunPod secret and
   set the pod environment variable `WANDB_API_KEY` to
   `{{ RUNPOD_SECRET_<secret name> }}`.
3. Upload the data: `runpodctl send data-full-overhead.tar` on the laptop, and
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
pip install --break-system-packages -e .[train]  # the image's Python is "externally managed" (PEP 668)

# Results on the volume; images extracted to the local disk.
mkdir -p /workspace/nutrition5k/outputs
ln -s /workspace/nutrition5k/outputs outputs
tar -xf /workspace/nutrition5k/data-full-overhead.tar
python -m nutrition5k check --full

# Long run that survives a closed terminal.
tmux new -s train
python -m nutrition5k train --full
```

The side-angle runs and `.tar` files were copied from the volume to the
laptop's `antiguo/` and deleted from the volume on 2026-10-30, when
`data-full-overhead.tar` was uploaded (its SHA-256 is in `hash-tar.txt`).

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
    print(dish.dish_id, dish.image, dish.targets)
```

Run the focused tests with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
