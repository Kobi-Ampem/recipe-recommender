import numpy as np
import scipy.sparse as sp

from reciperec.slices import slice_metrics


def test_slices_bin_users_and_measure_popularity_share():
    # 200 items; item 0 and 1 are the most popular (top 1% = 2 items).
    rows = np.zeros((3, 200), dtype=np.float32)
    rows[0, :3] = 1  # 3 training items -> bin 1-4
    rows[1, :12] = 1  # 12 -> bin 10-19
    rows[2, [0, 1]] = 1  # 2 -> bin 1-4
    train = sp.csr_matrix(rows)
    users = np.array([0, 1, 2])
    ndcg = np.array([0.0, 1.0, 0.5])
    tops = np.array([[0, 1, 50], [1, 60, 70], [80, 90, 100]])
    out = slice_metrics(train, users, ndcg, tops)
    assert out["by_activity"]["1-4"] == {"users": 2, "ndcg@10": 0.25}
    assert out["by_activity"]["10-19"]["ndcg@10"] == 1.0
    assert out["top1pct_share"] == 3 / 9


def test_bootstrap_separates_real_gain_from_noise():
    from reciperec.slices import paired_bootstrap

    rng = np.random.default_rng(1)
    base = rng.random(2000)
    assert paired_bootstrap(base * 1.5, base)["significant"]
    noisy = paired_bootstrap(base + rng.normal(0, 0.3, 2000), base)
    assert not noisy["significant"]
    assert noisy["ci95"][0] < 0 < noisy["ci95"][1]
