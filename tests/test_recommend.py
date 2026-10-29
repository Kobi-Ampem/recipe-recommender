import numpy as np
import scipy.sparse as sp

from reciperec.recommend import cold_start_scores

# Item 0 is similar to item 1 only; item 3 is the most popular.
SIM = sp.csr_matrix(
    np.array([[0, 0.9, 0, 0], [0.9, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
)
POP = np.array([5, 1, 2, 50], dtype=np.float32)


def test_picks_drive_the_ranking_and_are_excluded():
    scores = cold_start_scores(SIM, POP, np.array([0]))
    assert np.argmax(scores) == 1
    assert scores[0] == -np.inf
    # Unrelated items fall back to popularity order.
    assert scores[3] > scores[2]


def test_no_picks_means_popularity():
    scores = cold_start_scores(SIM, POP, np.array([], dtype=int))
    assert np.argsort(-scores).tolist() == [3, 0, 2, 1]
