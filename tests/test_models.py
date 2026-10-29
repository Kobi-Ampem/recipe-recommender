"""Every model must learn two obvious taste groups from a tiny matrix."""

import numpy as np
import pytest
import scipy.sparse as sp

from reciperec.models import MODELS

# Users 0-3 like items 0-2 (sweet), users 4-7 like items 3-5 (savoury).
# User 8 has two sweet items and should be offered the third, item 2.
ROWS = [[1, 1, 1, 0, 0, 0]] * 4 + [[0, 0, 0, 1, 1, 1]] * 4 + [[1, 1, 0, 0, 0, 0]]
TRAIN = sp.csr_matrix(np.array(ROWS, dtype=np.float32))
SMALL = {
    "itemknn": {"neighbors": 3},
    "puresvd": {"factors": 2},
    "bpr": {"factors": 4, "lr": 0.05, "reg": 0.0, "epochs": 150, "batch_size": 8},
}


@pytest.mark.parametrize("name", [m for m in MODELS if m != "popularity"])
def test_model_recommends_within_the_users_taste_group(name):
    model = MODELS[name](**SMALL.get(name, {})).fit(TRAIN)
    scores = model.score(np.array([8]))
    assert scores.shape == (1, 6)
    unseen = [2, 3, 4, 5]
    assert unseen[int(np.argmax(scores[0, unseen]))] == 2


def test_itemknn_keeps_only_top_neighbours():
    model = MODELS["itemknn"](neighbors=1).fit(TRAIN)
    assert (np.diff(model.sim.indptr) <= 1).all()
    assert model.sim.diagonal().sum() == 0


def test_bpr_early_stopping_records_best_epoch():
    # The metrics rank a top 10, so pad the catalogue to 12 recipes.
    train = sp.hstack([TRAIN, sp.csr_matrix((9, 6), dtype=np.float32)]).tocsr()
    val = sp.csr_matrix(np.eye(9, 12, dtype=np.float32))
    model = MODELS["bpr"](factors=4, epochs=6, eval_every=2, batch_size=8).fit(train, val=val)
    assert model.fit_info["best_epoch"] in {2, 4, 6}
    assert model.fit_info["history"][0]["epoch"] == 2
