"""Load interaction logs and turn them into implicit feedback.

Every dataset comes out in the same shape: a DataFrame with integer
`user`, `item` and `ts` columns, where user and item ids run from 0 with
no gaps. Ratings are dropped on purpose: on Food.com nearly every review
is 5 stars, so "this user reviewed this recipe" is the signal, not the score.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
FOODCOM_KAGGLE = "shuyangli94/food-com-recipes-and-user-interactions"


@dataclass
class Interactions:
    df: pd.DataFrame  # columns: user, item, ts
    n_users: int
    n_items: int
    item_ids: pd.Index  # original item id for each encoded item
    name: str


def kcore(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep only users and items with at least k interactions.

    Dropping a sparse user can push an item under k and the other way round,
    so repeat until nothing changes.
    """
    while True:
        users = df["user"].value_counts()
        items = df["item"].value_counts()
        keep = df["user"].map(users).ge(k) & df["item"].map(items).ge(k)
        if keep.all():
            return df
        df = df[keep]


def encode(df: pd.DataFrame, name: str) -> Interactions:
    """Map raw ids to 0..n-1 and wrap the result."""
    df = df.drop_duplicates(["user", "item"], keep="first")
    user_codes, _ = pd.factorize(df["user"])
    item_codes, item_ids = pd.factorize(df["item"])
    out = pd.DataFrame({"user": user_codes, "item": item_codes, "ts": df["ts"].to_numpy()})
    return Interactions(out, int(user_codes.max()) + 1, int(item_codes.max()) + 1, item_ids, name)


def load_movielens_1m(data_dir: Path = DATA_DIR) -> Interactions:
    """MovieLens 1M, every rating treated as an interaction.

    Licence: research use only, no redistribution, cite Harper and Konstan (2015).
    """
    path = data_dir / "ml-1m" / "ratings.dat"
    if not path.exists():
        print(f"Downloading MovieLens 1M from {ML1M_URL}")
        with urllib.request.urlopen(ML1M_URL) as resp:
            zipfile.ZipFile(io.BytesIO(resp.read())).extractall(data_dir)
    df = pd.read_csv(path, sep="::", engine="python", names=["user", "item", "rating", "ts"])
    df = df.sort_values(["user", "ts", "item"], kind="stable")
    return encode(df[["user", "item", "ts"]], "ml-1m")


def load_foodcom(data_dir: Path = DATA_DIR, k: int = 5) -> Interactions:
    """Food.com reviews (Majumder et al., 2019), filtered to a k-core.

    Needs RAW_interactions.csv in data/foodcom/. Download it with
    `kaggle datasets download -d shuyangli94/food-com-recipes-and-user-interactions`
    (Kaggle API token required). Licence on Kaggle: "Data files (c) Original Authors".
    """
    path = data_dir / "foodcom" / "RAW_interactions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download {FOODCOM_KAGGLE} from Kaggle and unzip it into "
            f"{path.parent}/ (see README)."
        )
    df = pd.read_csv(path, usecols=["user_id", "recipe_id", "date"])
    df = df.rename(columns={"user_id": "user", "recipe_id": "item"})
    df["ts"] = pd.to_datetime(df["date"]).astype("int64") // 10**9
    df = df.sort_values(["user", "ts", "item"], kind="stable")
    df = kcore(df[["user", "item", "ts"]], k)
    return encode(df, f"foodcom-{k}core")


LOADERS = {"ml-1m": load_movielens_1m, "foodcom": load_foodcom}
