"""Recommend recipes for a new visitor, with nutrition limits.

uv run python -m reciperec.recommend --search "banana bread"
uv run python -m reciperec.recommend --like 40893 54272 --max-calories 500 --max-minutes 45

Cold start: a new visitor has no history, so there is nothing to train on.
If they name a few recipes they like, item-kNN scores every recipe by
similarity to those picks. This needs no retraining, which BPR would (a new
user has no embedding). With no usable picks, they get the most popular
recipes within their limits. Popularity is also a tiny tie-break, so when
similarity runs out the list fills with well-reviewed recipes, not noise.

The model is fitted on all Food.com 5-core interactions (no hold-out: this
is for use, not evaluation), with item-kNN settings chosen on validation.
Recipe suggestions only, not dietary or medical advice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from reciperec.data import load_foodcom
from reciperec.models.itemknn import ItemKNN
from reciperec.nutrition import Limits, allowed_items, filtered_top_k, load_recipes
from reciperec.split import to_csr

TUNING = Path(__file__).resolve().parent.parent / "runs" / "tuning-foodcom-5core-itemknn.json"


def tidy(name: str) -> str:
    """Food.com names have stray spaces and are lower case."""
    return " ".join(str(name).split()).capitalize()


def cold_start_scores(sim: sp.csr_matrix, popularity: np.ndarray, liked: np.ndarray) -> np.ndarray:
    """Similarity to the liked items, popularity as a tie-break; liked items excluded."""
    pop = popularity / max(popularity.max(), 1)
    scores = pop * 1e-6
    if len(liked):
        picks = sp.csr_matrix(
            (np.ones(len(liked), dtype=np.float32), (np.zeros(len(liked), dtype=int), liked)),
            shape=(1, sim.shape[0]),
        )
        scores = scores + (picks @ sim).toarray().ravel()
        scores[liked] = -np.inf
    return scores


def explain_picks(
    sim: sp.csr_matrix, liked: np.ndarray, top: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Why each recommended item scored: its main pick, that pick's share of the score, and
    how many other picks contributed.

    A recommendation's kNN score is the sum of sim[pick, item] over the picks, so the largest
    term is the main reason and its share says how dominant it is. On Food.com similarities
    are small and close, so a share near 50% is common and is shown rather than hidden.
    Returns main -1 where no pick contributed (a popularity fill).
    """
    n = len(top)
    if not len(liked) or not n:
        return np.full(n, -1), np.zeros(n), np.zeros(n, dtype=int)
    parts = sim[liked][:, top].toarray()  # picks x recommendations
    total = parts.sum(axis=0)
    main = np.where(total > 0, liked[parts.argmax(axis=0)], -1)
    share = np.divide(parts.max(axis=0), total, out=np.zeros(n), where=total > 0)
    others = np.maximum((parts > 0).sum(axis=0) - 1, 0)
    return main, share, others


class RecipeRecommender:
    def __init__(self, params: dict | None = None):
        if params is None:
            params = json.loads(TUNING.read_text())["best"]["params"] if TUNING.exists() else {}
        data = load_foodcom()
        matrix = to_csr(data.df, data.n_users, data.n_items)
        self.knn = ItemKNN(**params).fit(matrix)
        self.popularity = np.asarray(matrix.sum(axis=0)).ravel()
        self.item_ids = data.item_ids
        self.index = pd.Series(np.arange(len(data.item_ids)), index=data.item_ids)
        self.recipes = load_recipes().reindex(data.item_ids)
        self.recipes["reviews"] = self.popularity.astype(int)

    def search(self, text: str, n: int = 10) -> pd.DataFrame:
        """Recipes in the model whose name contains text, most reviewed first."""
        hits = self.recipes[self.recipes["name"].str.contains(text, case=False, na=False)]
        return hits.sort_values("reviews", ascending=False).head(n)

    def recommend(self, liked: list[int], limits: Limits | None = None, k: int = 10):
        """Top k recipes as a DataFrame; liked are original Food.com recipe ids."""
        known = [r for r in liked if r in self.index.index]
        liked_idx = self.index.loc[known].to_numpy() if known else np.array([], dtype=int)
        scores = cold_start_scores(self.knn.sim, self.popularity, liked_idx)
        allowed = allowed_items(limits or Limits(), self.recipes, self.item_ids)
        top = filtered_top_k(scores[None, :], allowed, k)[0]
        top = top[np.isfinite(scores[top]) & allowed[top]]  # fewer than k may pass the limits
        out = self.recipes.iloc[top].copy()
        out["score"] = scores[top]
        main, share, others = explain_picks(self.knn.sim, liked_idx, top)
        out["why"] = [
            "popular"
            if m < 0
            else f"because you liked {tidy(self.recipes['name'].iloc[m])}"
            + (f" ({sh:.0%}, +{o} more)" if o else "")
            for m, sh, o in zip(main, share, others)
        ]
        return out, [r for r in liked if r not in known]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--search", help="find recipe ids by name")
    parser.add_argument("--like", type=int, nargs="*", default=[], help="Food.com recipe ids")
    parser.add_argument("--k", type=int, default=10)
    for f in (
        "max_calories",
        "max_sugar_pdv",
        "max_sodium_pdv",
        "max_saturated_fat_pdv",
        "min_protein_pdv",
        "max_minutes",
    ):
        parser.add_argument("--" + f.replace("_", "-"), type=float, dest=f)
    args = parser.parse_args()

    rec = RecipeRecommender()
    cols = ["name", "minutes", "calories", "sugar_pdv", "protein_pdv", "reviews"]
    if args.search:
        print(rec.search(args.search)[cols].to_string())
        return
    limits = Limits(**{f: getattr(args, f) for f in Limits.__dataclass_fields__})
    out, unknown = rec.recommend(args.like, limits, args.k)
    if unknown:
        print(f"Not in the model (too few reviews): {unknown}. Using the rest.")
    print(out[[*cols, "why"]].to_string())
    print("\nRecipe suggestions only, not dietary or medical advice.")


if __name__ == "__main__":
    main()
