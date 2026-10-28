"""Split each user's history in time: the past trains, the future tests.

A random split leaks the future into training (the model sees a review
written after the ones it is asked to predict), which inflates every
score. Holding out each user's most recent interactions is the honest
version of "what will this person do next?".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def temporal_user_split(
    df: pd.DataFrame, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last `test_frac` of each user's interactions (at least one) go to test.

    Users with a single interaction stay entirely in train: there is
    nothing to learn from to predict their one item.
    Expects df sorted by user then ts (the loaders do this).
    """
    sizes = df.groupby("user", sort=False)["item"].transform("size").to_numpy()
    rank = df.groupby("user", sort=False).cumcount().to_numpy()
    n_test = np.maximum(1, np.floor(sizes * test_frac)).astype(int)
    is_test = (rank >= sizes - n_test) & (sizes > 1)
    return df[~is_test], df[is_test]


def to_csr(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    """Binary user x item matrix."""
    data = np.ones(len(df), dtype=np.float32)
    m = sp.csr_matrix((data, (df["user"], df["item"])), shape=(n_users, n_items))
    m.data[:] = 1.0  # duplicates would otherwise sum
    return m
