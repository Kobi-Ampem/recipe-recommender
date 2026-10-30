import numpy as np
import pytest
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


def test_explain_picks_names_the_main_contributor():
    from reciperec.recommend import explain_picks

    # Picks 0 and 2. Item 1 owes most to pick 0; item 3 only to pick 2; item 4 to nobody.
    sim = sp.csr_matrix(
        np.array(
            [
                [0, 0.9, 0, 0.0, 0],
                [0, 0, 0, 0, 0],
                [0, 0.2, 0, 0.5, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
    )
    main, share, others = explain_picks(sim, np.array([0, 2]), np.array([1, 3, 4]))
    assert main.tolist() == [0, 2, -1]
    assert share[0] == pytest.approx(0.9 / 1.1)
    assert share[1] == pytest.approx(1.0)
    assert others.tolist() == [1, 0, 0]


def test_explain_picks_without_picks():
    from reciperec.recommend import explain_picks

    main, _, others = explain_picks(SIM, np.array([], dtype=int), np.array([1, 2]))
    assert main.tolist() == [-1, -1]
    assert others.tolist() == [0, 0]
