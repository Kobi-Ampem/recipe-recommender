"""Item-based nearest neighbours: "people who reviewed this also reviewed that".

Each recipe is represented by the users who reviewed it (a column of the
user x item matrix). Two recipes are similar when they share reviewers,
measured by cosine similarity, which divides by each recipe's popularity
so a hugely popular recipe is not similar to everything.

A user's score for a recipe is the sum of its similarity to every recipe
they have. Only the `neighbors` most similar recipes are kept per recipe:
the full table would not fit in memory (41k x 41k floats is ~7 GB) and the
weak similarities are mostly noise.

`shrink` damps similarities built on little evidence. Two recipes with two
reviewers each that share one score 0.5 on plain cosine, the same as two
recipes with 200 reviewers that share 100. Multiplying by
overlap / (overlap + shrink) keeps the second and pulls the first towards 0.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class ItemKNN:
    def __init__(self, neighbors: int = 100, shrink: float = 0.0, chunk: int = 1000):
        self.neighbors = neighbors
        self.shrink = shrink
        self.chunk = chunk

    def fit(self, train: sp.csr_matrix) -> ItemKNN:
        items = train.T.tocsr()  # one row per item
        n = items.shape[0]
        k = min(self.neighbors, n - 1)
        # Binary data: cosine = shared reviewers / sqrt(reviewers of a * reviewers of b)
        items_t = items.T.tocsc()
        norms = np.sqrt(np.diff(items.indptr)).astype(np.float32)
        rows, cols, vals = [], [], []
        for start in range(0, n, self.chunk):
            stop = min(start + self.chunk, n)
            overlap = (items[start:stop] @ items_t).toarray()  # shared reviewers
            sims = overlap / np.maximum(norms[start:stop, None] * norms[None, :], 1e-12)
            if self.shrink:
                sims *= overlap / (overlap + self.shrink)
            sims[np.arange(stop - start), np.arange(start, stop)] = 0.0  # not your own neighbour
            top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
            top_vals = np.take_along_axis(sims, top, axis=1)
            keep = top_vals > 0
            rows.append(np.repeat(np.arange(start, stop), k)[keep.ravel()])
            cols.append(top[keep])
            vals.append(top_vals[keep])
        # sim[i, j]: how much having item i counts towards item j
        self.sim = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
            dtype=np.float32,
        )
        self.train = train
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.train[users] @ self.sim).toarray()
