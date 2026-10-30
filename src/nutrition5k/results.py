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
                "seed": config["seed"],
                "params_M": config["parameters"] / 1e6,
                "trainable_M": config["trainable_parameters"] / 1e6,
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
            rows.append(row | _metric_columns(val, test))
    return pd.DataFrame(rows)


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
