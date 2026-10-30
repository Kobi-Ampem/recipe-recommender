"""Local demo: pick recipes you like, set nutrition limits, get suggestions.

uv run streamlit run app.py

Runs on your machine only: the Food.com data cannot be redistributed, so
there is no hosted version. Settings can be preset in the URL, for example
?like=25885,2886&max_calories=400 (used for the README screenshots).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from reciperec.nutrition import Limits
from reciperec.recommend import RecipeRecommender

RUNS = Path(__file__).resolve().parent / "runs"
LIMITS = {
    "max_calories": ("Max calories", 50.0, 2000.0, 500.0, 50.0),
    "max_sugar_pdv": ("Max sugar (% daily value)", 0.0, 200.0, 20.0, 5.0),
    "max_sodium_pdv": ("Max sodium (% daily value)", 0.0, 200.0, 20.0, 5.0),
    "max_saturated_fat_pdv": ("Max saturated fat (% daily value)", 0.0, 200.0, 30.0, 5.0),
    "min_protein_pdv": ("Min protein (% daily value)", 0.0, 200.0, 30.0, 5.0),
    "max_minutes": ("Max cooking time (minutes)", 5.0, 240.0, 30.0, 5.0),
}
COLUMNS = {
    "name": "Recipe",
    "minutes": "Minutes",
    "calories": "Calories",
    "sugar_pdv": "Sugar %DV",
    "sodium_pdv": "Sodium %DV",
    "protein_pdv": "Protein %DV",
    "reviews": "Reviews",
    "why": "Why",
}

st.set_page_config(page_title="Recipe recommender", page_icon="🍲", layout="wide")


@st.cache_resource(show_spinner="Loading Food.com and building the recipe model (about 30 s)...")
def load_model() -> RecipeRecommender:
    return RecipeRecommender()


def tidy(name: str) -> str:
    return " ".join(str(name).split()).capitalize()


def query_floats(key: str) -> float | None:
    try:
        return float(st.query_params[key]) if key in st.query_params else None
    except ValueError:
        return None


rec = load_model()
recipes = rec.recipes

st.title("Recipe recommender")
st.caption(
    "Suggestions from what people reviewed on Food.com, with nutrition limits. "
    "Recipe suggestions only, not dietary or medical advice."
)

# Sidebar: hard limits, applied after ranking.
st.sidebar.header("Limits")
limit_values = {}
for key, (label, lo, hi, default, step) in LIMITS.items():
    preset = query_floats(key)
    on = st.sidebar.checkbox(label, value=preset is not None, key=f"on_{key}")
    if on:
        limit_values[key] = st.sidebar.slider(
            label,
            lo,
            hi,
            preset if preset is not None else default,
            step,
            format="%.0f",
            key=key,
            label_visibility="collapsed",
        )
k = st.sidebar.slider("How many suggestions", 5, 30, 10)

# Picks: the 5,000 most-reviewed recipes, searchable by typing.
choices = recipes.sort_values("reviews", ascending=False).head(5000)
labels = {rid: f"{tidy(row['name'])} ({row['reviews']} reviews)" for rid, row in choices.iterrows()}
preset_likes = [int(x) for x in st.query_params.get("like", "").split(",") if x.strip().isdigit()]
picked = st.multiselect(
    "Recipes you like (type to search; leave empty for popular picks)",
    options=list(labels),
    default=[r for r in preset_likes if r in labels],
    format_func=labels.get,
    max_selections=10,
)

out, _ = rec.recommend(picked, Limits(**limit_values), k)
if out.empty:
    st.warning("No recipes meet all of these limits. Try loosening one.")
else:
    shown = out[list(COLUMNS)].rename(columns=COLUMNS)
    shown["Recipe"] = shown["Recipe"].map(tidy)
    shown["Why"] = shown["Why"].map({"similar to your picks": "your picks"}).fillna(shown["Why"])
    shown.index = pd.RangeIndex(1, len(shown) + 1)
    st.dataframe(
        shown,
        use_container_width=True,
        column_config={
            "Recipe": st.column_config.TextColumn(width=340),
            "Why": st.column_config.TextColumn(width=90),
            **{
                c: st.column_config.NumberColumn(format="%.0f")
                for c in ("Calories", "Sugar %DV", "Sodium %DV", "Protein %DV")
            },
        },
    )
    if limit_values:
        st.caption(
            f"{len(out)} shown. Limits remove recipes after ranking; they never change the order."
        )

with st.expander("How it works"):
    st.markdown(
        """
- **With picks:** item-kNN. Two recipes are similar when the same people reviewed both; every
  recipe is scored by its similarity to your picks. A new visitor needs no retraining.
- **Without picks:** the most-reviewed recipes that meet your limits.
- **Limits** are hard filters applied after ranking. Nutrition values are as entered on Food.com
  and are not checked.
- Model fitted on the Food.com 5-core: users and recipes with at least 5 reviews.
"""
    )
    run = RUNS / "foodcom-5core.json"
    if run.exists():
        results = json.loads(run.read_text())["results"]
        names = {
            "popularity": "Popularity",
            "itemknn": "Item-kNN",
            "puresvd": "PureSVD",
            "bpr": "BPR-MF",
        }
        rows = []
        for m, r in results.items():
            v = r.get("vs_popularity")
            rows.append(
                {
                    "Model": names.get(m, m),
                    "Test NDCG@10": round(r["ndcg@10"], 4),
                    "vs popularity": "" if not v else f"{v['gain']:+.0%}",
                    "Coverage": f"{r['coverage@10']:.1%}",
                }
            )
        st.markdown("Held-out test results (newest 20% of each user's reviews):")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
