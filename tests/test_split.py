import pandas as pd

from reciperec.split import train_val_test_split


def test_validation_sits_between_train_and_test():
    # One user, ten interactions in time order.
    df = pd.DataFrame({"user": [0] * 10, "item": range(10), "ts": range(10)})
    train, val, test = train_val_test_split(df, test_frac=0.2, val_frac=0.25)
    assert test["item"].tolist() == [8, 9]
    assert val["item"].tolist() == [6, 7]
    assert train["item"].tolist() == list(range(6))
    assert train["ts"].max() < val["ts"].min() <= val["ts"].max() < test["ts"].min()


def test_slices_cover_everything_once():
    df = pd.DataFrame(
        {"user": [0, 0, 0, 1, 1, 2], "item": [0, 1, 2, 0, 1, 0], "ts": [1, 2, 3, 1, 2, 1]}
    )
    train, val, test = train_val_test_split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert pd.concat([train, val, test]).sort_index().equals(df)
