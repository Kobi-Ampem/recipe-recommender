"""Recommend what most people interacted with in train.

The baseline every other model has to beat. On many datasets it is
surprisingly hard to beat by much, which is exactly why it is reported.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class Popularity:
    def fit(self, train: sp.csr_matrix) -> Popularity:
        self.counts = np.asarray(train.sum(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.counts, (len(users), len(self.counts))).copy()
