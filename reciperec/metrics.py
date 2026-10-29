"""Top-k ranking metrics, computed over the full catalogue.

Each test user gets a score for every item. Items they already have in
train are removed, the top k are kept, and those are checked against the
held-out items. No sampled negatives: sampling 100 random items to rank
against is common but makes weak models look strong.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import scipy.sparse as sp


class Recommender(Protocol):
    def fit(self, train: sp.csr_matrix) -> Recommender: ...

    def score(self, users: np.ndarray) -> np.ndarray:
        """Dense (len(users), n_items) array; higher means recommend sooner."""
        ...


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first."""
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def evaluate(
    model: Recommender,
    train: sp.csr_matrix,
    test: sp.csr_matrix,
    k: int = 10,
    batch_size: int = 512,
) -> dict[str, float]:
    """Mean metrics over test users; see evaluate_detailed."""
    return evaluate_detailed(model, train, test, k, batch_size)[0]


def evaluate_detailed(
    model: Recommender,
    train: sp.csr_matrix,
    test: sp.csr_matrix,
    k: int = 10,
    batch_size: int = 512,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Mean precision@k, recall@k, NDCG@k over users with test items, plus coverage.

    Coverage is the share of the catalogue that appears in anyone's top k:
    a model that recommends the same ten items to everyone scores near zero.
    Scoring runs in batches so a large catalogue fits in memory.

    Also returns, for slice analysis: the test users, each one's NDCG@k
    and each one's top-k items.
    """
    users = np.flatnonzero(np.diff(test.indptr))
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    precision, recall, ndcg, tops = [], [], [], []
    recommended = np.zeros(train.shape[1], dtype=bool)

    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        scores = np.asarray(model.score(batch), dtype=np.float32)
        seen = train[batch]
        scores[seen.nonzero()] = -np.inf
        top = top_k(scores, k)
        recommended[top.ravel()] = True
        tops.append(top)

        truth = test[batch]
        hits = np.asarray(truth[np.arange(len(batch))[:, None], top].todense()) > 0
        n_true = np.diff(truth.indptr)
        n_hits = hits.sum(axis=1)

        precision.append(n_hits / k)
        recall.append(n_hits / n_true)
        dcg = (hits * discounts).sum(axis=1)
        idcg = np.cumsum(discounts)[np.minimum(n_true, k) - 1]
        ndcg.append(dcg / idcg)

    per_user = np.concatenate(ndcg)
    summary = {
        f"precision@{k}": float(np.concatenate(precision).mean()),
        f"recall@{k}": float(np.concatenate(recall).mean()),
        f"ndcg@{k}": float(per_user.mean()),
        f"coverage@{k}": float(recommended.mean()),
        "users": len(users),
    }
    return summary, users, per_user, np.concatenate(tops)
