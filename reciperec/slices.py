"""Where a model wins or loses: by user activity, and how much it leans on hits.

An average NDCG can hide that a model helps heavy users and hurts light
ones. Popularity bias is the share of recommendations that come from the
most popular 1% of recipes in training: high means the model mostly
repeats the bestsellers.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# Training interactions per user; upper bounds are exclusive.
ACTIVITY_BINS = [(1, 5), (5, 10), (10, 20), (20, 50), (50, None)]


def bin_label(lo: int, hi: int | None) -> str:
    return f"{lo}+" if hi is None else f"{lo}-{hi - 1}"


def slice_metrics(
    train: sp.csr_matrix, users: np.ndarray, per_user_ndcg: np.ndarray, tops: np.ndarray
) -> dict:
    activity = np.diff(train.indptr)[users]
    by_activity = {}
    for lo, hi in ACTIVITY_BINS:
        in_bin = (activity >= lo) & (activity < (hi if hi is not None else np.inf))
        if in_bin.any():
            by_activity[bin_label(lo, hi)] = {
                "users": int(in_bin.sum()),
                "ndcg@10": float(per_user_ndcg[in_bin].mean()),
            }

    counts = np.asarray(train.sum(axis=0)).ravel()
    n_top = max(1, int(np.ceil(len(counts) * 0.01)))
    is_top = np.zeros(len(counts), dtype=bool)
    is_top[np.argsort(-counts, kind="stable")[:n_top]] = True
    return {"by_activity": by_activity, "top1pct_share": float(is_top[tops].mean())}


def paired_bootstrap(
    model: np.ndarray, baseline: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Relative NDCG gain over the baseline with a 95% interval, resampling users.

    Both arrays are per-user scores on the same users in the same order.
    Resampling users together keeps the pairing: each draw asks "on this
    set of users, how much better is the model?". If the whole interval is
    above 0, the gain is unlikely to be luck.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(model), size=(n_boot, len(model)))
    gains = model[idx].mean(axis=1) / baseline[idx].mean(axis=1) - 1
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return {
        "gain": float(model.mean() / baseline.mean() - 1),
        "ci95": [float(lo), float(hi)],
        "significant": bool(lo > 0 or hi < 0),
    }
