import numpy as np
import pandas as pd

from reciperec.nutrition import Limits, allowed_items, filtered_top_k, parse_nutrition

RECIPES = pd.concat(
    [
        pd.DataFrame({"name": ["cake", "salad", "steak"], "minutes": [60, 10, 25]}),
        parse_nutrition(
            pd.Series(
                [
                    "[450.0, 30.0, 120.0, 10.0, 8.0, 40.0, 20.0]",
                    "[120.0, 5.0, 4.0, 15.0, 6.0, 2.0, 3.0]",
                    "[600.0, 50.0, 0.0, 30.0, 110.0, 70.0, 0.0]",
                ]
            )
        ),
    ],
    axis=1,
).set_index(pd.Index([10, 20, 30], name="id"))


def test_parse_nutrition_columns():
    assert RECIPES.loc[10, "sugar_pdv"] == 120.0
    assert RECIPES.loc[30, "protein_pdv"] == 110.0


def test_limits_combine():
    assert Limits().mask(RECIPES).tolist() == [True, True, True]
    assert Limits(max_sugar_pdv=50).mask(RECIPES).tolist() == [False, True, True]
    assert Limits(max_sugar_pdv=50, min_protein_pdv=50).mask(RECIPES).tolist() == [
        False,
        False,
        True,
    ]
    assert Limits(max_minutes=30, max_calories=500).mask(RECIPES).tolist() == [False, True, False]


def test_allowed_items_follow_model_order_and_drop_unknown():
    # Model item order differs from the file, and recipe 99 has no nutrition row.
    allowed = allowed_items(Limits(max_calories=500), RECIPES, pd.Index([30, 99, 20, 10]))
    assert allowed.tolist() == [False, False, True, True]


def test_filter_applies_after_ranking():
    scores = np.array([[0.9, 0.8, 0.1, 0.5]])
    allowed = np.array([False, True, True, True])
    assert filtered_top_k(scores, allowed, k=2).tolist() == [[1, 3]]
