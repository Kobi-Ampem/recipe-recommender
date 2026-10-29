"""PureSVD (Cremonesi, Koren and Turrin, 2010): matrix factorisation by SVD.

Truncated SVD finds the `factors` directions that explain most of the
user x recipe matrix: hidden tastes such as "baking" or "quick weeknight",
learned rather than named. Each recipe becomes a short vector (a row of V).

A user's scores are their row projected onto those tastes and back:
x_u V V^T. Recipes close to what they reviewed, in taste space, score high.

Every recipe a user has not reviewed counts as a 0 here, which is crude
(unseen is not the same as disliked) but makes the model one fast SVD
with a single setting.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


class PureSVD:
    def __init__(self, factors: int = 64, seed: int = 0):
        self.factors = factors
        self.seed = seed

    def fit(self, train: sp.csr_matrix) -> PureSVD:
        svd = TruncatedSVD(n_components=self.factors, random_state=self.seed)
        svd.fit(train)
        self.v = svd.components_.T.astype(np.float32)  # items x factors
        self.train = train
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.asarray(self.train[users] @ self.v) @ self.v.T
