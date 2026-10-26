"""Unit tests for targeted retraining unlearning strategy (CP15/CP16)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch

from data.split import DataSplit
from models.lightgcn import LightGCN
from unlearning.strategies.targeted_retraining import targeted_retraining_unlearning


def make_toy_split(
    train_rows: list[tuple[int, int]],
    val_rows: list[tuple[int, int]] | None = None,
    test_rows: list[tuple[int, int]] | None = None,
    num_users: int = 4,
    num_movies: int = 6,
    seed: int = 42,
) -> DataSplit:
    cols = ["user_idx", "movie_idx"]
    return DataSplit(
        train=pd.DataFrame(train_rows, columns=cols),
        val=pd.DataFrame(val_rows or [], columns=cols),
        test=pd.DataFrame(test_rows or [], columns=cols),
        num_users=num_users,
        num_movies=num_movies,
        seed=seed,
    )


@pytest.fixture
def toy_setup():
    torch.manual_seed(42)
    train_rows = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 3), (2, 4), (3, 4), (3, 5)]
    val_rows = [(0, 2), (1, 3)]
    test_rows = [(0, 3), (1, 4)]
    split = make_toy_split(train_rows, val_rows, test_rows, num_users=4, num_movies=6)
    model = LightGCN(num_users=4, num_movies=6, embedding_dim=16, num_layers=2)
    return model, split


def test_targeted_retraining_creates_fresh_model(toy_setup):
    model, split = toy_setup
    weight_before = model.embedding.weight.clone().detach()

    retrained_model, metadata = targeted_retraining_unlearning(
        model, split, target_user=0, epochs=2, eval_every=1
    )

    # Original model must be unmodified
    assert torch.equal(model.embedding.weight, weight_before)
    # Retrained model is a fresh model instance with different weights
    assert isinstance(retrained_model, LightGCN)
    assert not torch.equal(retrained_model.embedding.weight, model.embedding.weight)


def test_targeted_retraining_excludes_target_user_interactions(toy_setup):
    model, split = toy_setup
    retrained_model, metadata = targeted_retraining_unlearning(
        model, split, target_user=0, epochs=1, eval_every=1
    )

    assert metadata["strategy"] == "targeted_retraining"
    assert metadata["target_user"] == 0
    # Original train had 8 interactions, target user 0 had 2, so remaining must be 6
    assert metadata["train_interactions_remaining"] == 6
    assert metadata["forgotten_interactions_count"] == 2
    assert "training_result" in metadata


def test_targeted_retraining_preserves_val_and_test(toy_setup):
    model, split = toy_setup
    val_before = split.val.copy()
    test_before = split.test.copy()

    targeted_retraining_unlearning(
        model, split, target_user=1, epochs=1, eval_every=1
    )

    pd.testing.assert_frame_equal(split.val, val_before)
    pd.testing.assert_frame_equal(split.test, test_before)


def test_targeted_retraining_invalid_target_user_raises_error(toy_setup):
    model, split = toy_setup

    with pytest.raises(ValueError, match="target_user -1 is outside"):
        targeted_retraining_unlearning(model, split, target_user=-1, epochs=1)

    with pytest.raises(ValueError, match="target_user 10 is outside"):
        targeted_retraining_unlearning(model, split, target_user=10, epochs=1)


def test_targeted_retraining_user_without_interactions_raises_error():
    train_rows = [(0, 0), (1, 1)]
    split = make_toy_split(train_rows, num_users=3, num_movies=3)
    model = LightGCN(num_users=3, num_movies=3, embedding_dim=8, num_layers=1)

    with pytest.raises(ValueError, match="no training interactions to unlearn"):
        targeted_retraining_unlearning(model, split, target_user=2, epochs=1)
