from __future__ import annotations

import csv
from pathlib import Path

from .dataset import EXCLUDED_FILE, METADATA_FILES

# A dish's ingredient grams may differ from its total mass by up to 10%.
MASS_TOLERANCE = 0.10
# Pure fat has about 9 kcal/g, so a denser dish is impossible.
MAX_KCAL_PER_GRAM = 9.0
# Typical ingredients weigh under 200 g; the broken labels show values such as 7974 g.
MAX_INGREDIENT_GRAMS = 1500.0
# Dishes the rules cannot catch, found by inspection.
MANUAL_EXCLUSIONS = {
    # Official repository issue #10; every side-angle frame shows a phone, no food.
    "dish_1559172356": "images show a phone, not food (GitHub issue #10)",
}


def label_errors(root: Path) -> dict[str, str]:
    """Dish id → reasons, for every metadata dish whose labels are impossible or inconsistent."""
    errors: dict[str, str] = {}
    for relative in METADATA_FILES:
        with (root / relative).open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if not row or row[0].strip().lower() == "dish_id":
                    continue
                calories, mass = float(row[1]), float(row[2])
                # Same two layouts as the dataset loader: with or without an ingredient-count column.
                start = 7 if len(row) > 6 and not row[6].strip().startswith("ingr_") else 6
                grams = [float(row[index + 2]) for index in range(start, len(row), 7)]
                reasons = []
                if grams and mass > 0 and abs(sum(grams) / mass - 1) > MASS_TOLERANCE:
                    reasons.append("ingredient grams differ from total mass")
                if mass > 0 and calories / mass > MAX_KCAL_PER_GRAM:
                    reasons.append(f"more than {MAX_KCAL_PER_GRAM:g} kcal/g")
                if grams and max(grams) > MAX_INGREDIENT_GRAMS:
                    reasons.append(f"an ingredient over {MAX_INGREDIENT_GRAMS:g} g")
                if reasons:
                    errors[row[0].strip()] = "; ".join(reasons)
    return errors


def write_exclusions(root: Path) -> tuple[Path, int]:
    """Write the dishes with label errors, plus the manual exclusions, to ``excluded_dishes.csv``."""
    errors = label_errors(root) | MANUAL_EXCLUSIONS
    path = root / EXCLUDED_FILE
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dish_id", "reasons"))
        writer.writerows(sorted(errors.items()))
    return path, len(errors)
