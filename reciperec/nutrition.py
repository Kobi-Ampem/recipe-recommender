"""Nutrition filters for recommendations. Recipe suggestions only, not dietary advice.

RAW_recipes.csv stores nutrition as a list string per recipe. Per the
dataset's Kaggle description the order is: calories, then total fat, sugar,
sodium, protein, saturated fat and carbohydrates, each as a percent of daily
value (PDV). The values are as entered on Food.com and are not checked;
some recipes list a whole batch, so very large numbers exist.

Filtering happens after ranking: the model scores every recipe for its
taste signal, then recipes outside the limits are removed before taking
the top k. Mixing nutrition into the score would make the model's ranking
and the user's hard limits impossible to tell apart.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from reciperec.data import DATA_DIR
from reciperec.metrics import top_k

NUTRITION = [
    "calories",
    "fat_pdv",
    "sugar_pdv",
    "sodium_pdv",
    "protein_pdv",
    "saturated_fat_pdv",
    "carbs_pdv",
]


def parse_nutrition(column: pd.Series) -> pd.DataFrame:
    """'[51.5, 0.0, ...]' strings to one float column per nutrient."""
    parts = column.str.strip("[]").str.split(",", expand=True).astype(float)
    parts.columns = NUTRITION
    return parts


def load_recipes(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Recipe name, minutes and nutrition, indexed by the original recipe id."""
    path = data_dir / "foodcom" / "RAW_recipes.csv"
    df = pd.read_csv(path, usecols=["id", "name", "minutes", "nutrition"])
    df = pd.concat([df.drop(columns="nutrition"), parse_nutrition(df["nutrition"])], axis=1)
    return df.set_index("id")


@dataclass
class Limits:
    """Hard limits; None means no limit. PDV is percent of daily value."""

    max_calories: float | None = None
    max_sugar_pdv: float | None = None
    max_sodium_pdv: float | None = None
    max_saturated_fat_pdv: float | None = None
    min_protein_pdv: float | None = None
    max_minutes: float | None = None

    def mask(self, recipes: pd.DataFrame) -> np.ndarray:
        """True for each recipe (row) inside every set limit."""
        keep = np.ones(len(recipes), dtype=bool)
        for f in fields(self):
            limit = getattr(self, f.name)
            if limit is None:
                continue
            bound, column = f.name.split("_", 1)
            values = recipes[column].to_numpy()
            keep &= values <= limit if bound == "max" else values >= limit
        return keep


def allowed_items(limits: Limits, recipes: pd.DataFrame, item_ids: pd.Index) -> np.ndarray:
    """Limits mask in the model's item order. Recipes with no nutrition data are excluded."""
    aligned = recipes.reindex(item_ids)
    return limits.mask(aligned) & aligned[NUTRITION].notna().all(axis=1).to_numpy()


def filtered_top_k(scores: np.ndarray, allowed: np.ndarray, k: int = 10) -> np.ndarray:
    """Top k per row among allowed items only, best first."""
    scores = np.where(allowed[None, :], scores, -np.inf)
    return top_k(scores, k)
