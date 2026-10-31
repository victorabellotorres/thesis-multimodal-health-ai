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
            trunk = "x".join(map(str, config["hidden"])) or "0"
            rows.append({
                "run": run.name,
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
                **{f"val_{target.removeprefix('total_')}": val["metrics"][target]["pmae"] for target in TARGETS},
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
STAGE2 = dict(heads="single", hidden="512", image_size=224, backbone_lr=1e-4, augment="d4", schedule="cosine", seed=0)
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
    frozen = frozen[["run", "family", "backbone", "val_mean_pmae", "test_mean_pmae"]].assign(
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


def overfitting_table(stage: pd.DataFrame) -> pd.DataFrame:
    """Per run: trainable share, best epoch, validation at the best epoch and in the last 5 epochs, the losses
    at the best epoch and test (reported only).

    Overfitting shows as a large ``gap`` (validation minus train loss, same scaled L1) and a positive
    ``late_rise`` (validation of the last 5 epochs above the best one).
    """
    columns = ["backbone", "unfreeze", "trainable_pct", "best_epoch", "val_mean_pmae", "val_last5", "late_rise",
               "train_loss_best", "val_loss_best", "gap", "test_mean_pmae", "sec_per_epoch", "gpu_memory_gb"]
    return stage.reindex(columns=columns).sort_values(["backbone", "trainable_pct"]).set_index(
        ["backbone", "unfreeze"]).round(2)


def depth_steps(stage: pd.DataFrame) -> pd.DataFrame:
    """Mean validation PMAE and trainable share per family and depth level, and the change against ``all``
    (computed per backbone, then averaged; negative means the level is better than training everything)."""
    levels = list(dict.fromkeys(DEPTH_LEVELS.values()))
    reference = stage[stage["level"] == "all"].set_index("backbone")["val_mean_pmae"]
    stage = stage.assign(vs_all=stage["val_mean_pmae"] - stage["backbone"].map(reference))
    table = stage.groupby(["family", "level"])[["trainable_pct", "val_mean_pmae", "vs_all"]].mean()
    families = [family for family in FAMILIES if family in stage["family"].values]
    return table.reindex(pd.MultiIndex.from_product([families, levels])).round(2)


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


def inference_cost(row, repeats=30) -> dict:
    """MACs and median CPU latency (batch 1) of a run's architecture; the cost does not depend on the weights."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from .models.nutrition_net import NutritionNet

    hidden = () if row["hidden"] == "0" else tuple(map(int, row["hidden"].split("x")))
    model = NutritionNet(row["backbone"], row["heads"], hidden, pretrained=False).eval()
    size = int(row["image_size"])
    image = torch.randn(1, 3, size, size)
    with torch.no_grad():
        with FlopCounterMode(display=False) as counter:
            model(image)
        for _ in range(5):  # warm-up
            model(image)
        times = []
        for _ in range(repeats):
            began = time.perf_counter()
            model(image)
            times.append(time.perf_counter() - began)
    # The counter reports FLOPs; one multiply-accumulate is two of them.
    return {"MMACs": counter.get_total_flops() / 2 / 1e6, "cpu_ms": 1000 * sorted(times)[repeats // 2]}


def cost(chosen: pd.DataFrame, label="backbone", title="Accuracy vs compute (lower left is better)") -> pd.DataFrame:
    """Size and inference cost of each run, plus validation error against MACs (one point per run)."""
    chosen = chosen[chosen["status"] == "done"]
    table = chosen[["family", label, "val_mean_pmae", "params_M"]].assign(
        size_MB=[(path / "best.pt").stat().st_size / 1e6 for path in chosen["path"]]
    )
    table = table.join(pd.DataFrame([inference_cost(row) for _, row in chosen.iterrows()], index=chosen.index))
    table = table.join(chosen[["gpu_memory_gb", "sec_per_epoch", "gpu"]]).set_index(label)
    fig, ax = plt.subplots(figsize=(7, 4))
    families = list(dict.fromkeys(table["family"]))
    for color, family in zip(SERIES, families):
        group = table[table["family"] == family]
        ax.scatter(group["MMACs"], group["val_mean_pmae"], s=64, color=color, zorder=3, label=family)
        for name, row in group.iterrows():
            ax.annotate(name, (row["MMACs"], row["val_mean_pmae"]), xytext=(7, 4), textcoords="offset points",
                        color=INK, fontsize=9)
    if table["MMACs"].max() > 10 * table["MMACs"].min():
        ax.set_xscale("log")
    ax.set_xlabel("MACs per image (millions)", color=MUTED)
    ax.set_ylabel("Validation mean PMAE (%)", color=MUTED)
    if len(families) > 1:
        ax.legend(frameon=False)
    _style(ax, title)
    plt.show()
    return table


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


def finalists(runs: pd.DataFrame, chosen: dict) -> pd.DataFrame:
    """One row per family: the chosen configuration, its numeric columns averaged over its seeds.

    ``chosen`` maps a family to a run name with or without the ``-s<seed>`` suffix. ``None`` takes the
    configuration with the most seeds (the stage-3 finals) and, among those, the lowest mean validation
    PMAE. ``val_sd`` is the SD across seeds; ``path`` is the first seed's.
    """
    done = select(runs[runs["status"] == "done"])
    done = done.assign(configuration=[re.sub(r"-s\d+$", "", run) for run in done["run"]])
    rows = []
    for family, name in chosen.items():
        family_runs = done[done["family"] == family]
        if family_runs.empty:
            print(f"{family}: no finished runs")
            continue
        if name is None:
            ranking = family_runs.groupby("configuration")["val_mean_pmae"].agg(["count", "mean"])
            configuration = ranking.sort_values(["count", "mean"], ascending=[False, True]).index[0]
        else:
            configuration = re.sub(r"-s\d+$", "", name)
        group = family_runs[family_runs["configuration"] == configuration]
        row = group.iloc[0].copy()
        numeric = group.select_dtypes("number").mean()
        row[numeric.index] = numeric
        row["run"], row["seeds"], row["val_sd"] = configuration, len(group), group["val_mean_pmae"].std()
        rows.append(row)
    return pd.DataFrame(rows)


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
