import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from reciperec.data import kcore
from reciperec.metrics import evaluate, top_k
from reciperec.split import temporal_user_split, to_csr


class Fixed:
    """A model that returns the same hand-written scores for everyone."""

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)

    def fit(self, train):
        return self

    def score(self, users):
        return np.tile(self.scores, (len(users), 1))


def csr(rows, n_items):
    return sp.csr_matrix(np.array(rows, dtype=np.float32).reshape(-1, n_items))


def test_top_k_is_ordered_best_first():
    scores = np.array([[0.1, 0.9, 0.5, 0.7]])
    assert top_k(scores, 3).tolist() == [[1, 3, 2]]


def test_perfect_ranking_scores_one():
    train = csr([[1, 0, 0, 0]], 4)
    test = csr([[0, 1, 1, 0]], 4)
    m = evaluate(Fixed([0, 9, 8, 1]), train, test, k=2)
    assert m["precision@2"] == 1.0
    assert m["recall@2"] == 1.0
    assert m["ndcg@2"] == pytest.approx(1.0)


def test_train_items_are_never_recommended():
    # Item 0 has the top score but the user already has it, so it must be skipped.
    train = csr([[1, 0, 0]], 3)
    test = csr([[0, 0, 1]], 3)
    m = evaluate(Fixed([9, 1, 5]), train, test, k=1)
    assert m["precision@1"] == 1.0


def test_ndcg_rewards_hits_near_the_top():
    train = csr([[0, 0, 0]], 3)
    test = csr([[0, 0, 1]], 3)
    early = evaluate(Fixed([0, 1, 9]), train, test, k=2)["ndcg@2"]
    late = evaluate(Fixed([9, 0, 1]), train, test, k=2)["ndcg@2"]
    assert early == pytest.approx(1.0)
    assert late == pytest.approx(1 / np.log2(3))


def test_users_without_test_items_are_skipped():
    train = csr([[1, 0], [1, 0]], 2)
    test = csr([[0, 1], [0, 0]], 2)
    assert evaluate(Fixed([0, 1]), train, test, k=1)["users"] == 1


def test_split_holds_out_each_users_latest():
    df = pd.DataFrame(
        {"user": [0, 0, 0, 0, 0, 1], "item": [0, 1, 2, 3, 4, 0], "ts": [1, 2, 3, 4, 5, 1]}
    )
    train, test = temporal_user_split(df, test_frac=0.2)
    assert test["item"].tolist() == [4]  # user 0's most recent
    assert 1 in train["user"].tolist()  # single-interaction user kept in train
    assert len(train) + len(test) == len(df)


def test_kcore_repeats_until_stable():
    # Item 2 has only one user, so after dropping it user 1 falls under k too.
    df = pd.DataFrame({"user": [0, 0, 1, 1, 2, 2], "item": [0, 1, 0, 2, 0, 1]})
    out = kcore(df, k=2)
    assert set(out["user"]) == {0, 2}
    assert set(out["item"]) == {0, 1}


def test_to_csr_is_binary():
    df = pd.DataFrame({"user": [0, 0], "item": [1, 1]})
    assert to_csr(df, 1, 2).toarray().tolist() == [[0, 1]]
