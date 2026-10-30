from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from .dataset import EXCLUDED_FILE, GROUPS_FILE, METADATA_FILES

# A dish's ingredient grams may differ from its total mass by up to 10%.
MASS_TOLERANCE = 0.10
# Pure fat has about 9 kcal/g, so a denser dish is impossible.
MAX_KCAL_PER_GRAM = 9.0
# Typical ingredients weigh under 200 g; the broken labels show values such as 7974 g.
MAX_INGREDIENT_GRAMS = 1500.0
# Dishes the rules cannot catch, found by inspection.
MANUAL_EXCLUSIONS = {
    # Official repository issue #10; the overhead photo (and every side-angle frame) shows a phone, no food.
    "dish_1559172356": "images show a phone, not food (GitHub issue #10)",
}
# Two scans of one plate are at most this many seconds apart (dish ids are Unix timestamps).
MAX_SCAN_GAP = 300
# The scale may read one gram differently between two scans of the same plate.
GRAMS_TOLERANCE = 1.0


def _metadata(root: Path) -> Iterator[tuple[str, float, float, dict[str, float]]]:
    """Dish id, calories, mass and {ingredient id: grams} of every metadata dish."""
    for relative in METADATA_FILES:
        with (root / relative).open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if not row or row[0].strip().lower() == "dish_id":
                    continue
                # Same two layouts as the dataset loader: with or without an ingredient-count column.
                start = 7 if len(row) > 6 and not row[6].strip().startswith("ingr_") else 6
                ingredients = {row[index]: float(row[index + 2]) for index in range(start, len(row), 7)}
                yield row[0].strip(), float(row[1]), float(row[2]), ingredients


def label_errors(root: Path) -> dict[str, str]:
    """Dish id → reasons, for every metadata dish whose labels are impossible or inconsistent."""
    errors: dict[str, str] = {}
    for dish_id, calories, mass, ingredients in _metadata(root):
        grams = list(ingredients.values())
        reasons = []
        if grams and mass > 0 and abs(sum(grams) / mass - 1) > MASS_TOLERANCE:
            reasons.append("ingredient grams differ from total mass")
        if mass > 0 and calories / mass > MAX_KCAL_PER_GRAM:
            reasons.append(f"more than {MAX_KCAL_PER_GRAM:g} kcal/g")
        if grams and max(grams) > MAX_INGREDIENT_GRAMS:
            reasons.append(f"an ingredient over {MAX_INGREDIENT_GRAMS:g} g")
        if reasons:
            errors[dish_id] = "; ".join(reasons)
    return errors


def scan_groups(root: Path) -> dict[str, str]:
    """Dish id → group id (the group's first dish): every scan of one physical plate.

    Nutrition5k scans a plate after each added item, and seems to scan the finished plate once
    more after mixing it. A dish joins the previous dish's group when it was scanned at most
    MAX_SCAN_GAP seconds later and contains all of its ingredients with at least the same grams.
    The official train/test split keeps such incremental scans together (Thames et al., §3.2).
    """
    dishes = sorted(_metadata(root), key=lambda dish: int(dish[0].removeprefix("dish_")))
    groups: dict[str, str] = {}
    previous: tuple[str, int, dict[str, float]] | None = None
    for dish_id, _, _, ingredients in dishes:
        time = int(dish_id.removeprefix("dish_"))
        groups[dish_id] = dish_id
        if previous:
            previous_id, previous_time, previous_ingredients = previous
            if (
                time - previous_time <= MAX_SCAN_GAP
                and previous_ingredients
                and all(
                    ingredients.get(key, -1.0) >= grams - GRAMS_TOLERANCE
                    for key, grams in previous_ingredients.items()
                )
            ):
                groups[dish_id] = groups[previous_id]
        previous = (dish_id, time, ingredients)
    return groups


def write_preprocess(root: Path) -> tuple[int, int]:
    """Write ``excluded_dishes.csv`` and ``dish_groups.csv``; return the excluded and group counts."""
    excluded = label_errors(root) | MANUAL_EXCLUSIONS
    groups = scan_groups(root)
    for name, header, rows in (
        (EXCLUDED_FILE, ("dish_id", "reasons"), sorted(excluded.items())),
        (GROUPS_FILE, ("dish_id", "group"), sorted(groups.items())),
    ):
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
    return len(excluded), len(set(groups.values()))
