"""CP3.1 checks: removing a user's training data leaves everything else intact."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from data.split import DataSplit, remove_user_from_split


def toy_split():
    train = pd.DataFrame({"user_idx": [0, 0, 1, 2, 2], "movie_idx": [0, 1, 1, 2, 3]})
    val = pd.DataFrame({"user_idx": [0, 1], "movie_idx": [4, 4]})
    test = pd.DataFrame({"user_idx": [0, 1, 2], "movie_idx": [5, 5, 5]})
    return DataSplit(train, val, test, num_users=3, num_movies=6, seed=42)


def test_only_target_user_train_rows_removed():
    split = remove_user_from_split(toy_split(), 0)
    assert 0 not in set(split.train["user_idx"])
    assert len(split.train) == 3
    assert len(split.val) == 2 and len(split.test) == 3


def test_invalid_user_rejected():
    with pytest.raises(ValueError):
        remove_user_from_split(toy_split(), 9)
