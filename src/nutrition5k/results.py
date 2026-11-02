"""Read saved runs from outputs/ and summarise them; shared by the results notebooks.

Nothing here trains. Decisions are taken on validation; test is only reported.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from .dataset import load_dataset
from .models.families import FAMILIES, FAMILY

OUTPUTS = Path("outputs")
TARGETS = ("total_mass", "total_calories", "total_fat", "total_carb", "total_protein")
METRICS = {"pmae": "PMAE (%)", "mae": "MAE", "rmse": "RMSE", "rsquared": "R²"}  # R² is the only one where higher is better
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, MUTED = "#0b0b0b", "#52514e"
SUMMARY = ["run", "status", "epochs_done", "best_epoch", "val_mean_pmae", "test_mean_pmae",
           "test_clean_mean_pmae", "params_M", "trainable_M", "sec_per_epoch"]
CONFIG_COLUMNS = ["family", "backbone", "unfreeze", "heads", "hidden", "image_size", "backbone_lr"]


def read_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def load_runs(families=tuple(FAMILIES), split="full", root=OUTPUTS) -> pd.DataFrame:
    """One row per run in <root>/<family>/<split>/<backbone>/<run>/."""
    rows = []
    for family in families:
        for config_path in sorted((root / family / split).glob("*/*/config.json")):
            run = config_path.parent
            config = read_json(config_path)
            history = pd.read_csv(run / "history.csv") if (run / "history.csv").is_file() else None
            val, test = read_json(run / "evaluation_val.json"), read_json(run / "evaluation.json")
            row = {
                "run": run.name,
                "path": run,
                "family": FAMILY.get(config["backbone"], family),
                "backbone": config["backbone"],
                "unfreeze": "all" if config["unfreeze"] is None else str(config["unfreeze"]),
                "heads": config["heads"],
                "hidden": "x".join(map(str, config["hidden"])) or "0",
                "image_size": config["image_size"],
                "backbone_lr": config.get("backbone_lr", 1e-4),
                "pooling": config.get("pooling"),
                "augment": config.get("augment", "flip"),
                "schedule": config.get("schedule", "constant"),
                "head_width": config.get("head_width", 256),
                "weight_decay": config.get("weight_decay", 0.01),
                "drop_path_rate": config.get("drop_path_rate", 0.0),
                "seed": config["seed"],
                "params_M": config["parameters"] / 1e6,
                "trainable_M": config["trainable_parameters"] / 1e6,
                # Share of the backbone that trains (written since stage 2).
                "trainable_pct": 100 * config["trainable_backbone_parameters"] / config["backbone_parameters"]
                if "backbone_parameters" in config else None,
                "epochs_done": 0 if history is None else len(history),
                "best_epoch": None if history is None else int(history.loc[history["val_mean_pmae"].idxmin(), "epoch"]),
                "sec_per_epoch": None if history is None else history["seconds"].mean(),
                "gpu": config["environment"]["gpu"],
                "gpu_memory_gb": None if history is None else history["gpu_memory_gb"].max(),
                "status": "done" if test else "running",
                "val_mean_pmae": val["mean_pmae"] if val else None,
                "test_mean_pmae": test["mean_pmae"] if test else None,
                "test_clean_mean_pmae": _clean_mean_pmae(run),
            }
            rows.append(row | _overfitting(history) | _metric_columns(val, test))
    return pd.DataFrame(rows)


def head_label(trunk: str, heads: str, head_width: int) -> str:
    """Short name of a stage-1 head, e.g. "512 · single 256" or "0 · linear"."""
    return f"{trunk} · {'linear' if not head_width else f'{heads} {head_width}'}"


# The 9 heads of the stage-1 grid, simplest first (same order as training.heads.GRID).
HEAD_GRID = [head_label(trunk, heads, width) for trunk in ("0", "512", "1024x1024")
             for heads, width in (("single", 0), ("single", 256), ("grouped", 256))]
DEFAULT_HEAD = head_label("512", "single", 256)


def load_heads(families=tuple(FAMILIES), split="full", root=OUTPUTS) -> pd.DataFrame:
    """One row per finished stage-1 head in <root>/<family>/<split>/heads/<backbone>/<run>/."""
    rows = []
    for family in families:
        for config_path in sorted((root / family / split / "heads").glob("*/*/config.json")):
            run = config_path.parent
            config = read_json(config_path)
            val, test = read_json(run / "evaluation_val.json"), read_json(run / "evaluation.json")
            if not test:
                continue
            history = pd.read_csv(run / "history.csv") if (run / "history.csv").is_file() else None
            trunk = "x".join(map(str, config["hidden"])) or "0"
            rows.append({
                "run": run.name,
                "path": run,
                "family": family,
                "backbone": config["backbone"],
                "head": head_label(trunk, config["heads"], config["head_width"]),
                "pooling": config["pooling"] or "default",
                "augment": config["augment"],
                "schedule": config.get("schedule", "constant"),
                "seed": config["seed"],
                "head_params_k": config["head_parameters"] / 1e3,
                "val_mean_pmae": val["mean_pmae"],
                "test_mean_pmae": test["mean_pmae"],
                "best_epoch": None if history is None else int(history.loc[history["val_mean_pmae"].idxmin(), "epoch"]),
                **{f"val_{target.removeprefix('total_')}": val["metrics"][target]["pmae"] for target in TARGETS},
                **_overfitting(history),
            })
    return pd.DataFrame(rows)


def head_grid(heads: pd.DataFrame) -> pd.DataFrame:
    """The 9 grid heads (flip, default pooling, constant lr, seed 0) of every backbone."""
    grid = heads[(heads["augment"] == "flip") & (heads["pooling"] == "default") & (heads["seed"] == 0)
                 & (heads["schedule"] == "constant")]
    return grid[grid["head"].isin(HEAD_GRID)]


def head_table(heads: pd.DataFrame):
    """Validation mean PMAE: one row per grid head, one column per backbone plus the family mean.

    The best value of each column is highlighted.
    """
    table = head_grid(heads).pivot(index="head", columns="backbone", values="val_mean_pmae").reindex(HEAD_GRID)
    table["mean"] = table.mean(axis=1)
    best = "font-weight: bold; background-color: #d7f0e3"
    return table.style.format("{:.2f}", na_rep="—").highlight_min(props=best)


def best_heads(heads: pd.DataFrame) -> pd.DataFrame:
    """Per backbone: the best grid head against the default head and the seed spread of the default head."""
    grid = head_grid(heads)
    default = heads[(heads["head"] == DEFAULT_HEAD) & (heads["augment"] == "flip") & (heads["pooling"] == "default")
                    & (heads["schedule"] == "constant")]
    rows = []
    for backbone, group in grid.groupby("backbone"):
        best = group.loc[group["val_mean_pmae"].idxmin()]
        chosen = group[group["head"] == DEFAULT_HEAD].iloc[0]
        seeds = default[default["backbone"] == backbone]["val_mean_pmae"]
        rows.append({
            "backbone": backbone,
            "best head": best["head"],
            "best val": best["val_mean_pmae"],
            "best test": best["test_mean_pmae"],
            "default val": chosen["val_mean_pmae"],
            "default test": chosen["test_mean_pmae"],
            "default - best": chosen["val_mean_pmae"] - best["val_mean_pmae"],
            "seeds": len(seeds),
            "default val, mean ± SD of seeds": f"{seeds.mean():.2f} ± {seeds.std():.2f}",
        })
    return pd.DataFrame(rows).set_index("backbone").round(2)


def extras_table(heads: pd.DataFrame) -> pd.DataFrame:
    """Validation mean PMAE of the default head (seed 0): flip vs D4 and, for plain ViTs, the pooling.

    ``flip`` and ``d4`` use the default pooling (CLS in plain ViTs); ``mean`` and ``cls+mean`` use flip.
    """
    default = heads[(heads["head"] == DEFAULT_HEAD) & (heads["seed"] == 0) & (heads["schedule"] == "constant")]
    label = default["augment"].where(default["pooling"] == "default", default["pooling"])
    table = default.assign(variant=label).pivot(index="backbone", columns="variant", values="val_mean_pmae")
    table["d4 - flip"] = table["d4"] - table["flip"]
    return table.round(2)


def target_errors(heads: pd.DataFrame) -> pd.DataFrame:
    """Validation PMAE per target of the default head and of the best grid head of every backbone."""
    grid = head_grid(heads)
    best = grid.loc[grid.groupby("backbone")["val_mean_pmae"].idxmin()].assign(which="best")
    default = grid[grid["head"] == DEFAULT_HEAD].assign(which="default")
    columns = [f"val_{target.removeprefix('total_')}" for target in TARGETS] + ["val_mean_pmae"]
    table = pd.concat([default, best]).set_index(["backbone", "which", "head"])[columns]
    return table.sort_index().round(2)


def head_steps(heads: pd.DataFrame, steps: dict) -> pd.DataFrame:
    """Mean change of validation mean PMAE per family (and over every backbone) from one grid head to another.

    ``steps`` maps a name to (from head, to head); negative means the second head is better. ``seed SD`` is
    the mean SD across seeds 0-2 of the default head: the noise to compare the changes with.
    """
    grid = head_grid(heads).pivot(index=["family", "backbone"], columns="head", values="val_mean_pmae")
    table = pd.DataFrame({name: grid[after] - grid[before] for name, (before, after) in steps.items()})
    default = heads[(heads["head"] == DEFAULT_HEAD) & (heads["augment"] == "flip") & (heads["pooling"] == "default")
                    & (heads["schedule"] == "constant")]
    table["seed SD"] = default.groupby(["family", "backbone"])["val_mean_pmae"].std()
    by_family = table.groupby("family").mean().reindex([family for family in FAMILIES if family in table.index])
    by_family.loc["all backbones"] = table.mean()
    return by_family.round(2)


def head_sizes(heads: pd.DataFrame, labels: list) -> pd.DataFrame:
    """Per family and head, averaged over its backbones: head parameters (thousands), the head's share (%) of the
    whole model and validation mean PMAE, with the size ratio and PMAE change relative to the first head."""
    from .models.nutrition_net import NutritionNet

    grid = head_grid(heads)
    grid = grid[grid["head"].isin(labels)]
    backbone_k = grid["backbone"].map({
        backbone: sum(p.numel() for p in NutritionNet(backbone, pretrained=False).backbone.parameters()) / 1e3
        for backbone in grid["backbone"].unique()
    })
    grid = grid.assign(head_pct=100 * grid["head_params_k"] / (grid["head_params_k"] + backbone_k))
    table = grid.groupby(["family", "head"])[["head_params_k", "head_pct", "val_mean_pmae"]].mean()
    first = table.xs(labels[0], level="head").reindex(table.index.get_level_values("family"))
    table["params ×"] = table["head_params_k"] / first["head_params_k"].values
    table["Δ val PMAE"] = table["val_mean_pmae"] - first["val_mean_pmae"].values
    order = pd.MultiIndex.from_product([[family for family in FAMILIES if family in grid["family"].values], labels])
    return table.reindex(order).round(2)


# Stage 2: the fixed head (512 · single 256), D4, cosine lr, cls+mean pooling in plain ViTs, seed 0.
# The common recipe of stage 2, which the finalists keep; stage 3 changes one value at a time.
STAGE2 = dict(heads="single", hidden="512", head_width=256, image_size=224, backbone_lr=1e-4, augment="d4",
              schedule="cosine", weight_decay=0.01, drop_path_rate=0.0, seed=0)
# Depth levels by trainable share of the backbone: plain ViTs use 6 / 10 blocks (49% / 81%) where the
# other backbones use 1 / 2 stages.
DEPTH_LEVELS = {"0": "frozen", "1": "low", "6": "low", "2": "high", "10": "high", "all": "all"}


def stage2_runs(runs: pd.DataFrame, heads: pd.DataFrame) -> pd.DataFrame:
    """Every stage-2 run, plus the cached frozen head of each backbone as ``unfreeze`` 0."""
    tuned = select(runs[runs["status"] == "done"], **STAGE2)
    tuned = tuned[(tuned["family"] == "plain_vit") == (tuned["pooling"] == "cls+mean")]
    frozen = heads[(heads["head"] == DEFAULT_HEAD) & (heads["augment"] == "d4") & (heads["schedule"] == "cosine")
                   & (heads["seed"] == 0)]
    frozen = frozen[(frozen["family"] == "plain_vit") == (frozen["pooling"] == "cls+mean")]
    frozen = frozen[["run", "path", "family", "backbone", "val_mean_pmae", "test_mean_pmae", "best_epoch",
                     "train_loss_best", "val_loss_best", "gap", "val_last5", "late_rise"]].assign(
        unfreeze="0", trainable_pct=0.0)
    stage = pd.concat([frozen, tuned], ignore_index=True)
    return stage.assign(level=stage["unfreeze"].map(DEPTH_LEVELS))


def depth_table(stage: pd.DataFrame):
    """Validation mean PMAE per backbone (rows) and depth level (columns); the best of each row is highlighted.

    The column names are the levels; ``unfreeze`` is 1 / 2 (6 / 10 in plain ViTs).
    """
    table = stage.pivot(index="backbone", columns="level", values="val_mean_pmae")
    table = table.reindex(columns=[level for level in dict.fromkeys(DEPTH_LEVELS.values()) if level in table])
    best = "font-weight: bold; background-color: #d7f0e3"
    return table.style.format("{:.2f}", na_rep="—").highlight_min(axis=1, props=best)


def overfitting_table(stage: pd.DataFrame):
    """Per run: trainable share, best epoch, validation at the best epoch and in the last 5 epochs, the losses
    at the best epoch and test (reported only).

    Overfitting shows as a large ``gap`` (validation minus train loss, same scaled L1) and a positive
    ``late_rise`` (validation of the last 5 epochs above the best one). The frozen heads train on stored features,
    so they have no comparable time or memory per epoch ("—").
    """
    columns = ["backbone", "unfreeze", "trainable_pct", "best_epoch", "val_mean_pmae", "val_last5", "late_rise",
               "train_loss_best", "val_loss_best", "gap", "test_mean_pmae", "sec_per_epoch", "gpu_memory_gb"]
    table = stage.reindex(columns=columns).sort_values(["backbone", "trainable_pct"]).set_index(["backbone", "unfreeze"])
    return table.style.format(precision=2, na_rep="—").format("{:.0f}", subset=["best_epoch"])


def depth_steps(stage: pd.DataFrame) -> pd.DataFrame:
    """Mean validation PMAE and trainable share per family and depth level, and the change against ``all``
    (computed per backbone, then averaged; negative means the level is better than training everything)."""
    levels = list(dict.fromkeys(DEPTH_LEVELS.values()))
    reference = stage[stage["level"] == "all"].set_index("backbone")["val_mean_pmae"]
    stage = stage.assign(vs_all=stage["val_mean_pmae"] - stage["backbone"].map(reference))
    table = stage.groupby(["family", "level"])[["trainable_pct", "val_mean_pmae", "vs_all"]].mean()
    families = [family for family in FAMILIES if family in stage["family"].values]
    return table.reindex(pd.MultiIndex.from_product([families, levels])).round(2)


def backbone_costs(stage: pd.DataFrame, level="all", repeats=30) -> pd.DataFrame:
    """Per backbone without the head: parameters, weights in MB (fp32), MACs and median CPU latency per image
    (batch 1, 224 px), and the time and GPU memory per epoch of fine-tuning at ``level`` (None: each row's own
    level, for one row per backbone such as the finalists).

    The last row is the stage-2 head of the largest backbone: its input, and so its size, grows with the features.
    """
    import torch

    from .models.nutrition_net import NutritionNet

    size = STAGE2["image_size"]
    image = torch.randn(1, 3, size, size)
    trained = (stage if level is None else stage[stage["level"] == level]).set_index("backbone")
    rows, heads = {}, {}
    for backbone in stage["backbone"].unique():
        pooling = "cls+mean" if FAMILY[backbone] == "plain_vit" else None
        model = NutritionNet(backbone, STAGE2["heads"], (512,), pretrained=False, pooling=pooling).eval()
        with torch.no_grad():
            features = model.features(image)
        rows[backbone] = _size(model.backbone) | _measure(model.features, image, repeats) | {
            "train_s_per_epoch": trained["sec_per_epoch"].get(backbone),
            "train_gpu_gb": trained["gpu_memory_gb"].get(backbone),
        }
        heads[backbone] = _size(model.head) | _measure(model.head, features, repeats)
    largest = max(heads, key=lambda backbone: heads[backbone]["params_M"])
    rows[f"head {DEFAULT_HEAD} (on {largest})"] = heads[largest]
    return pd.DataFrame.from_dict(rows, orient="index")


def pareto(stage: pd.DataFrame, costs: pd.DataFrame, cost: str, title: str) -> pd.DataFrame:
    """Best stage-2 validation PMAE of each backbone (any depth level) against one cost column of
    ``backbone_costs``. ``front`` marks the Pareto front: no cheaper backbone is more accurate."""
    best = stage.loc[stage.groupby("backbone")["val_mean_pmae"].idxmin(), ["backbone", "level", "val_mean_pmae"]]
    table = best.set_index("backbone").join(costs[cost]).sort_values(cost)
    table["front"] = table["val_mean_pmae"] < table["val_mean_pmae"].cummin().shift(fill_value=float("inf"))
    fig, ax = plt.subplots(figsize=(7, 4))
    front = table[table["front"]]
    ax.plot(front[cost], front["val_mean_pmae"], color=MUTED, linewidth=1, drawstyle="steps-post", zorder=2)
    ax.scatter(table[cost], table["val_mean_pmae"], s=64, zorder=3, edgecolor="white", linewidth=1.5,
               color=[SERIES[0] if on_front else MUTED for on_front in table["front"]])
    for name, row in table.iterrows():
        ax.annotate(f"{name} ({row['level']})", (row[cost], row["val_mean_pmae"]), xytext=(7, 4),
                    textcoords="offset points", color=INK, fontsize=9)
    if table[cost].max() > 10 * table[cost].min():
        ax.set_xscale("log")
    ax.set_xlabel(cost, color=MUTED)
    ax.set_ylabel("Best validation mean PMAE (%)", color=MUTED)
    _style(ax, title)
    plt.show()
    return table.round(2)


def loss_curves(chosen: pd.DataFrame, label: str, title: str) -> None:
    """Train (dashed) and validation (solid) loss per epoch, one colour per run, on the same scaled L1.

    Overfitting: the train loss keeps falling while the validation loss flattens or rises.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    chosen = chosen.dropna(subset=["path"])
    for color, (_, row) in zip(SERIES, chosen.sort_values(label, key=lambda values: values.astype(str)).iterrows()):
        history = pd.read_csv(row["path"] / "history.csv")
        ax.plot(history["epoch"], history["val_loss"], color=color, linewidth=2, label=str(row[label]))
        ax.plot(history["epoch"], history["train_loss"], color=color, linewidth=1.5, linestyle="--")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("Epoch", color=MUTED)
    ax.set_ylabel("L1 on scaled targets (dashed: train)", color=MUTED)
    ax.legend(frameon=False, title=label, loc="center left", bbox_to_anchor=(1, 0.5))
    _style(ax, title)
    plt.show()


def head_chart(heads: pd.DataFrame, title: str) -> None:
    """Validation mean PMAE of every grid head (rows), one colour per backbone (lower is better)."""
    grid = head_grid(heads)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    rows = {head: y for y, head in enumerate(reversed(HEAD_GRID))}
    for color, (backbone, group) in zip(SERIES, grid.groupby("backbone")):
        ax.scatter(group["val_mean_pmae"], group["head"].map(rows), s=64, color=color, edgecolor="white",
                   linewidth=1.5, zorder=3, label=backbone)
    mean = grid.groupby("head")["val_mean_pmae"].mean()
    ax.scatter(mean, mean.index.map(rows), marker="|", s=260, color=INK, linewidth=2, zorder=2, label="mean")
    ax.set_yticks(list(rows.values()), list(rows))
    ax.set_xlabel("Validation mean PMAE (%)", color=MUTED)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    _style(ax, title, grid="x")
    plt.show()


def baseline_row(split="full") -> pd.DataFrame:
    """The average baseline as one more row (no history), to compare with the models."""
    root = OUTPUTS / "average-baseline" / split
    val, test = read_json(root / "evaluation_val.json"), read_json(root / "evaluation.json")
    if not val:
        print("no evaluation_val.json for the baseline: run `python -m nutrition5k baseline` again")
    row = {
        "run": "average baseline", "backbone": "baseline", "status": "done", "params_M": 0, "trainable_M": 0,
        "val_mean_pmae": val["mean_pmae"] if val else None,
        "test_mean_pmae": test["mean_pmae"] if test else None,
        "test_clean_mean_pmae": _clean_mean_pmae(root),
    }
    return pd.DataFrame([row | _metric_columns(val, test)])


def select(runs: pd.DataFrame, **fixed) -> pd.DataFrame:
    """Runs matching every given field, e.g. select(runs, backbone="v4s", heads="single")."""
    chosen = runs
    for field, value in fixed.items():
        chosen = chosen[chosen[field].astype(str) == str(value)]
    return chosen.sort_values("val_mean_pmae")


def compare(chosen: pd.DataFrame, label: str, title: str, baseline=None) -> None:
    """Everything needed to pick a winner at one stage: summary, per-target validation metrics and charts."""
    if baseline is not None:
        chosen = pd.concat([chosen, baseline.assign(**{label: "baseline"})])
    display(chosen[SUMMARY + [label]])
    if chosen["val_mean_pmae"].notna().any():
        display(target_table(chosen, label))
        bar_pmae(chosen, label, title)
        bar_targets(chosen, label, f"{title}: validation PMAE per target")
        curves(chosen, label, f"{title}: validation curves")


def steps(runs: pd.DataFrame, backbone: str, plan: dict) -> None:
    """One greedy stage-2 path: {title: (label, fixed fields)}; each step compares ``label`` within the backbone."""
    for title, (label, fixed) in plan.items():
        chosen = select(runs, backbone=backbone, **fixed)
        print(f"{backbone} - {title}")
        if chosen[label].nunique() < 2:
            print("  nothing to compare yet")
            continue
        compare(chosen, label, f"{backbone} - {title}")


def bar_pmae(chosen: pd.DataFrame, label: str, title: str) -> None:
    """Validation mean PMAE per run (lower is better); the baseline row, if any, is grey."""
    done = chosen.dropna(subset=["val_mean_pmae"]).sort_values("val_mean_pmae", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(done) + 1.2))
    colors = [MUTED if name == "average baseline" else SERIES[0] for name in done["run"]]
    bars = ax.barh(done[label].astype(str), done["val_mean_pmae"], height=0.6, color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, color=INK, fontsize=9)
    ax.set_xlabel("Validation mean PMAE (%)", color=MUTED)
    _style(ax, title, grid="x")
    plt.show()


def curves(chosen: pd.DataFrame, label: str, title: str) -> None:
    """Validation mean PMAE per epoch; at most eight runs per chart; the big circle marks the best epoch."""
    fig, ax = plt.subplots(figsize=(7, 4))
    # Colour follows the label value (sorted), not the ranking, so it is stable across views.
    colors = dict(zip(sorted(chosen[label].astype(str).unique()), SERIES))
    for _, row in chosen.sort_values(label, key=lambda values: values.astype(str)).iterrows():
        if str(row[label]) not in colors or pd.isna(row["path"]):  # the baseline has no history
            continue
        history = pd.read_csv(row["path"] / "history.csv")
        color = colors[str(row[label])]
        ax.plot(history["epoch"], history["val_mean_pmae"], color=color, linewidth=2, marker="o", markersize=5,
                label=str(row[label]))
        best = history.loc[history["val_mean_pmae"].idxmin()]
        ax.plot(best["epoch"], best["val_mean_pmae"], marker="o", markersize=11, fillstyle="none", color=color,
                markeredgewidth=2)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("Epoch", color=MUTED)
    ax.set_ylabel("Validation mean PMAE (%)", color=MUTED)
    ax.legend(frameon=False, title=label)
    _style(ax, title)
    plt.show()


def target_table(chosen: pd.DataFrame, label: str, split="val"):
    """Every metric per target (columns) for each run (rows); the best value of each column is highlighted."""
    chosen = chosen.set_index(chosen[label].astype(str))
    columns = {}
    for metric, name in METRICS.items():
        for target in TARGETS:
            columns[(name, target.removeprefix("total_"))] = chosen[f"{split}_{target}_{metric}"]
        if metric == "pmae":
            columns[(name, "mean")] = chosen[f"{split}_mean_pmae"]
    table = pd.DataFrame(columns)
    higher = [column for column in table.columns if column[0] == METRICS["rsquared"]]
    lower = [column for column in table.columns if column not in higher]
    best = "font-weight: bold; background-color: #d7f0e3"
    return (table.style.format("{:.2f}", na_rep="—")
            .highlight_min(subset=lower, props=best)
            .highlight_max(subset=higher, props=best))


def bar_targets(chosen: pd.DataFrame, label: str, title: str, split="val") -> None:
    """PMAE per target (and the mean) for each run, side by side."""
    done = chosen.dropna(subset=[f"{split}_mean_pmae"]).sort_values(label, key=lambda values: values.astype(str))
    names = [target.removeprefix("total_") for target in TARGETS] + ["mean"]
    width = 0.8 / max(len(done), 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (color, (_, row)) in enumerate(zip(SERIES, done.iterrows())):
        values = [row[f"{split}_{target}_pmae"] for target in TARGETS] + [row[f"{split}_mean_pmae"]]
        positions = [x + (i - (len(done) - 1) / 2) * width for x in range(len(names))]
        ax.bar(positions, values, width=width * 0.9, color=color, label=str(row[label]))
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel(f"{'Validation' if split == 'val' else 'Test'} PMAE (%)", color=MUTED)
    ax.legend(frameon=False, title=label)
    _style(ax, title, grid="y")
    plt.show()


def target_curves(path: Path) -> None:
    """Validation PMAE per target and epoch of one run; the vertical line marks the chosen epoch."""
    history = pd.read_csv(path / "history.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for color, target in zip(SERIES, TARGETS):
        ax.plot(history["epoch"], history[f"val_{target}_pmae"], color=color, linewidth=2, marker="o",
                markersize=5, label=target.removeprefix("total_"))
    ax.plot(history["epoch"], history["val_mean_pmae"], color=INK, linewidth=2, label="mean")
    ax.axvline(history["val_mean_pmae"].idxmin(), color=MUTED, linestyle="--", linewidth=1)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("Epoch", color=MUTED)
    ax.set_ylabel("Validation PMAE (%)", color=MUTED)
    ax.legend(frameon=False)
    _style(ax, path.name)
    ax.title.set_fontsize(10)
    plt.show()


def seed_groups(runs: pd.DataFrame) -> pd.DataFrame:
    """Finished runs of every configuration that has more than one seed."""
    done = runs[runs["status"] == "done"]
    counts = done.groupby(CONFIG_COLUMNS)["seed"].transform("nunique")
    return done[counts > 1]


def seed_table(seed_runs: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SD across seeds of each configuration, on validation, test and clean test."""
    return seed_runs.groupby(CONFIG_COLUMNS).agg(
        seeds=("seed", "nunique"),
        val_mean=("val_mean_pmae", "mean"),
        val_sd=("val_mean_pmae", "std"),
        test_mean=("test_mean_pmae", "mean"),
        test_sd=("test_mean_pmae", "std"),
        test_clean_mean=("test_clean_mean_pmae", "mean"),
        test_clean_sd=("test_clean_mean_pmae", "std"),
    ).sort_values("val_mean")


def seed_spread(seed_runs: pd.DataFrame, title="Mean ± SD across seeds (dots: each seed)") -> None:
    """Mean PMAE of every seed (dots) and mean ± SD (bar) per configuration, on validation and test."""
    labels = seed_runs.apply(_seed_label, axis=1)
    order = seed_runs.groupby(labels)["val_mean_pmae"].mean().sort_values(ascending=False).index
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.6 * len(order) + 1.6), sharey=True)
    for ax, split, name in zip(axes, ("val", "test"), ("Validation", "Test")):
        values = seed_runs[f"{split}_mean_pmae"].groupby(labels)
        for y, label in enumerate(order):
            points = values.get_group(label)
            ax.errorbar(points.mean(), y, xerr=points.std(), fmt="o", color=SERIES[0], markersize=8, capsize=4,
                        linewidth=2, zorder=3)
            ax.scatter(points, [y] * len(points), s=22, color=SERIES[0], alpha=0.35, zorder=2)
            ax.annotate(f"{points.mean():.1f} ± {points.std():.1f}", (points.mean(), y), xytext=(0, 9),
                        textcoords="offset points", ha="center", color=INK, fontsize=8)
        ax.set_yticks(range(len(order)), order)
        ax.set_ylim(-0.6, len(order) - 0.2)
        ax.set_xlabel(f"{name} mean PMAE (%)", color=MUTED)
        _style(ax, None, grid="x")
    fig.suptitle(title, x=0.01, ha="left", color=INK)
    plt.tight_layout()
    plt.show()


def seed_targets(seed_runs: pd.DataFrame, split="test") -> None:
    """PMAE per target (and the mean), mean ± SD across seeds, one colour per configuration."""
    labels = seed_runs.apply(_seed_label, axis=1)
    columns = [f"{split}_{target}_pmae" for target in TARGETS] + [f"{split}_mean_pmae"]
    grouped = seed_runs[columns].groupby(labels)
    mean, sd = grouped.mean(), grouped.std()
    names = [target.removeprefix("total_") for target in TARGETS] + ["mean"]
    width = 0.8 / len(mean)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (color, label) in enumerate(zip(SERIES, mean.index)):
        positions = [x + (i - (len(mean) - 1) / 2) * width for x in range(len(names))]
        ax.bar(positions, mean.loc[label], yerr=sd.loc[label], width=width * 0.9, color=color, label=label,
               error_kw={"ecolor": INK, "capsize": 3, "linewidth": 1})
    ax.set_xticks(range(len(names)), names)
    name = "validation" if split == "val" else "test"
    ax.set_ylabel(f"{name.capitalize()} PMAE (%)", color=MUTED)
    ax.legend(frameon=False)
    _style(ax, f"{name.capitalize()} PMAE per target, mean ± SD across seeds", grid="y")
    plt.show()


def seed_files(family: str, split="full") -> pd.DataFrame:
    """The mean ± SD summaries that `train --all-seeds` writes to outputs/<family>/<split>/seeds/."""
    rows = []
    for path in sorted((OUTPUTS / family / split / "seeds").glob("*.json")):
        summary = read_json(path)
        row = {"configuration": path.stem}
        for name, prefix in (("validation", "val"), ("test", "test"), ("test_clean", "test_clean")):
            if name in summary:
                mean_pmae = summary[name]["mean_pmae"]
                row[f"{prefix}_mean_pmae"] = f"{mean_pmae['mean']:.2f} ± {mean_pmae['sd']:.2f}"
        for target in TARGETS:
            pmae = summary["test"]["metrics"][target]["pmae"]
            row[f"test_{target.removeprefix('total_')}_pmae"] = f"{pmae['mean']:.2f} ± {pmae['sd']:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def test_metrics(chosen: pd.DataFrame, split="full") -> pd.DataFrame:
    """MAE, PMAE, RMSE and R² per target on test for each run, with the baseline as reference."""
    sources = {row["run"]: row["path"] / "evaluation.json" for _, row in chosen.iterrows()}
    sources["average baseline"] = OUTPUTS / "average-baseline" / split / "evaluation.json"
    rows = []
    for model, path in sources.items():
        evaluation = read_json(path)
        if not evaluation:
            continue
        for target in TARGETS:
            metric = evaluation["metrics"][target]
            rows.append({
                "model": model, "target": target.removeprefix("total_"), "unit": metric["unit"],
                "MAE": metric["mae"], "PMAE (%)": metric["pmae"], "RMSE": metric["rmse"], "R²": metric["rsquared"],
            })
        rows.append({"model": model, "target": "mean", "PMAE (%)": evaluation["mean_pmae"]})
    return pd.DataFrame(rows).set_index(["model", "target"])


def gallery(path: Path, split="full", count=4, seed=0) -> None:
    """Lowest-error, highest-error and random test dishes of one run, with real and predicted values."""
    try:
        from .dataset import load_dataset

        dataset = load_dataset(Path("data/nutrition5k"), full=split == "full")
    except Exception as exc:  # no data/
        print(f"the gallery needs the images in data/: {exc}")
        return
    records = {record.dish_id: record for record in dataset.test if record.image}
    predicted = pd.read_csv(path / "predictions.csv").set_index("dish_id")
    predicted = predicted.loc[predicted.index.intersection(list(records))]
    truth = pd.DataFrame({dish: records[dish].targets for dish in predicted.index}, index=TARGETS).T
    # Error of a dish: the mean over the five targets of |error| / mean real value on test (as PMAE).
    error = ((predicted[list(TARGETS)] - truth).abs() / truth.mean()).mean(axis=1).sort_values()
    rest = error.iloc[count:-count]
    groups = {
        "lowest error": error.index[:count],
        "highest error": error.index[-count:][::-1],
        "random": rest.sample(min(count, len(rest)), random_state=seed).index,
    }
    fig, axes = plt.subplots(len(groups), count, figsize=(3.6 * count, 3.4 * len(groups)), squeeze=False)
    for row, (title, dishes) in zip(axes, groups.items()):
        for ax, dish in zip(row, dishes):
            ax.imshow(plt.imread(records[dish].image))
            lines = [
                f"{target.removeprefix('total_')}: {truth.loc[dish, target]:.0f} → {predicted.loc[dish, target]:.0f}"
                for target in TARGETS
            ]
            ax.set_title(f"{title}\n{dish}", loc="left", color=INK, fontsize=9)
            ax.set_xlabel("\n".join(lines) + f"\nerror {error[dish]:.0%}", color=MUTED, fontsize=8, loc="left")
            ax.set_xticks([]), ax.set_yticks([])
        for ax in row[len(dishes):]:
            ax.axis("off")
    fig.suptitle(f"{path.name}: real → predicted (g, kcal)", x=0.01, ha="left", color=INK)
    plt.tight_layout()
    plt.show()


def finalists(runs: pd.DataFrame, configurations: list) -> pd.DataFrame:
    """One row per chosen configuration (a run name without ``-s<seed>``), its numeric columns averaged over its
    seeds. ``val_sd`` is the SD across seeds; ``path`` is the first seed's."""
    done = runs[runs["status"] == "done"]
    done = done.assign(configuration=[re.sub(r"-s\d+$", "", run) for run in done["run"]])
    rows = []
    for configuration in configurations:
        group = done[done["configuration"] == configuration].sort_values("seed")
        if group.empty:
            print(f"{configuration}: no finished runs")
            continue
        row = group.iloc[0].copy()
        numeric = group.select_dtypes("number").mean()
        row[numeric.index] = numeric
        row["run"], row["seeds"], row["val_sd"] = configuration, len(group), group["val_mean_pmae"].std()
        rows.append(row)
    final = pd.DataFrame(rows)
    return final.assign(level=final["unfreeze"].map(DEPTH_LEVELS))



def calories_from_macros(final: pd.DataFrame, split="val", data=Path("data/nutrition5k")) -> pd.DataFrame:
    """Calorie PMAE of the calorie head (``head``) and of 9·fat + 4·carb + 4·protein from the predicted macros
    (``from_macros``), mean over each finalist's seeds; ``delta`` > 0: the head is better. Also prints how far the
    same formula is from the labelled calories on the true macros of the training dishes."""
    dataset = load_dataset(data, full=True)
    calories = {record.dish_id: record.total_calories for record in dataset.train + dataset.test}
    train = pd.DataFrame([record.targets for record in dataset.train], columns=TARGETS)
    floor = _pmae(_atwater(train), train["total_calories"])
    print(f"9·fat + 4·carb + 4·protein vs the labelled calories ({len(train)} train dishes): {floor:.2f}% PMAE")
    file = "predictions_val.csv" if split == "val" else "predictions.csv"
    rows = {}
    for _, finalist in final.iterrows():
        scores = []
        for run in sorted(Path(finalist["path"]).parent.glob(f"{finalist['run']}-s*")):
            predicted = pd.read_csv(run / file)
            true = predicted["dish_id"].map(calories)
            scores.append({"head": _pmae(predicted["total_calories"], true),
                           "from_macros": _pmae(_atwater(predicted), true)})
        rows[finalist["backbone"]] = pd.DataFrame(scores).mean()
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table.assign(delta=table["from_macros"] - table["head"]).round(2)


def _atwater(targets: pd.DataFrame) -> pd.Series:
    """Calories from the macros with the general Atwater factors (kcal per gram)."""
    return 9 * targets["total_fat"] + 4 * targets["total_carb"] + 4 * targets["total_protein"]


def _pmae(predicted: pd.Series, true: pd.Series) -> float:
    return (predicted - true).abs().mean() / true.mean() * 100


def recipe_changes(runs: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """Seed-0 runs of each finalist (same backbone, level and pooling) that change the common recipe (``STAGE2``),
    with ``delta``: their validation mean PMAE minus the finalist's mean over seeds (negative: better).
    A ``-r256`` backbone (pretrained at 256 px) counts as its 224-px twin."""
    recipe = pd.Series({key: value for key, value in STAGE2.items() if key != "seed"}).astype(str)
    keys = ["backbone", "unfreeze", "pooling"]
    done = runs[(runs["status"] == "done") & (runs["seed"] == 0)]
    done = done.assign(weights=done["backbone"], backbone=done["backbone"].str.removesuffix("-r256"),
                       pooling=done["pooling"].fillna("-"))
    base = final[keys + ["val_mean_pmae"]].assign(pooling=final["pooling"].fillna("-"))
    merged = done.merge(base, on=keys, suffixes=("", "_recipe"))
    differs = merged[recipe.index].astype(str).ne(recipe)
    merged = merged[differs.any(axis=1)]
    labels = [" · ".join((["r256 weights"] if row["weights"] != row["backbone"] else [])
                         + [f"{key} {row[key]}" for key in recipe.index[differs.loc[index]]])
              for index, row in merged.iterrows()]
    merged = merged.assign(change=labels, delta=merged["val_mean_pmae"] - merged["val_mean_pmae_recipe"])
    columns = ["backbone", "change", "val_mean_pmae", "val_mean_pmae_recipe", "delta", "test_mean_pmae",
               "best_epoch", "gap", "late_rise", "path"]
    return merged[columns].sort_values(["backbone", "change"]).reset_index(drop=True)


def _seed_label(row) -> str:
    return f"{row['backbone']} · ft {row['unfreeze']} · h{row['hidden']} · {row['image_size']} px"


def _overfitting(history) -> dict:
    """Train and validation loss at the best epoch, their gap, and the validation PMAE of the last 5 epochs."""
    if history is None or history.empty:
        return {}
    best = history.loc[history["val_mean_pmae"].idxmin()]
    last5 = history["val_mean_pmae"].tail(5).mean()
    return {
        "train_loss_best": best["train_loss"],
        "val_loss_best": best["val_loss"],
        "gap": best["val_loss"] - best["train_loss"],
        "val_last5": last5,
        "late_rise": last5 - best["val_mean_pmae"],
    }


def _size(module) -> dict:
    """Parameters (millions) and weights in MB (fp32, buffers included)."""
    tensors = [*module.parameters(), *module.buffers()]
    return {"params_M": sum(p.numel() for p in module.parameters()) / 1e6,
            "size_MB": sum(t.numel() * t.element_size() for t in tensors) / 1e6}


def _measure(forward, inputs, repeats=30) -> dict:
    """MACs (millions) and median CPU latency in ms of one call of ``forward``."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    with torch.no_grad():
        with FlopCounterMode(display=False) as counter:
            forward(inputs)
        for _ in range(5):  # warm-up
            forward(inputs)
        times = []
        for _ in range(repeats):
            began = time.perf_counter()
            forward(inputs)
            times.append(time.perf_counter() - began)
    # The counter reports FLOPs; one multiply-accumulate is two of them.
    return {"MMACs": counter.get_total_flops() / 2 / 1e6, "cpu_ms": 1000 * sorted(times)[repeats // 2]}


def _clean_mean_pmae(folder: Path):
    """Test mean PMAE without the excluded dishes (evaluation_clean.json), if the run wrote it."""
    clean = read_json(folder / "evaluation_clean.json")
    return clean["mean_pmae"] if clean else None


def _metric_columns(val, test) -> dict:
    """Every metric per target, as val_<target>_<metric> and test_<target>_<metric>."""
    return {
        f"{prefix}_{target}_{metric}": evaluation["metrics"][target][metric] if evaluation else None
        for prefix, evaluation in (("val", val), ("test", test))
        for target in TARGETS
        for metric in METRICS
    }


def _style(ax, title, grid="both") -> None:
    if title:
        ax.set_title(title, loc="left", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color="#e6e6e3", linewidth=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout()
