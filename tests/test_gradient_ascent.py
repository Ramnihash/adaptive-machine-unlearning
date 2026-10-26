"""Unit tests for bounded gradient ascent unlearning strategy (CP15/CP16)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch

from data.split import DataSplit
from models.lightgcn import LightGCN
from unlearning.strategies.gradient_ascent import gradient_ascent_unlearning


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


def test_gradient_ascent_does_not_mutate_original_model(toy_setup):
    model, split = toy_setup
    weight_before = model.embedding.weight.clone().detach()

    unlearned, metadata = gradient_ascent_unlearning(
        model, split, target_user=0, epochs=2, learning_rate=0.01
    )

    assert torch.equal(model.embedding.weight, weight_before)
    assert not torch.equal(unlearned.embedding.weight, model.embedding.weight)


def test_gradient_ascent_updates_and_no_nan_or_inf(toy_setup):
    model, split = toy_setup
    unlearned, metadata = gradient_ascent_unlearning(
        model, split, target_user=0, epochs=3, learning_rate=0.01, max_grad_norm=1.0
    )

    weights = unlearned.embedding.weight
    assert not torch.isnan(weights).any()
    assert not torch.isinf(weights).any()
    assert metadata["strategy"] == "gradient_ascent"
    assert metadata["target_user"] == 0
    assert metadata["forgotten_interactions_count"] == 2
    assert metadata["unlearning_seconds"] > 0


def test_gradient_ascent_preserves_val_and_test(toy_setup):
    model, split = toy_setup
    val_before = split.val.copy()
    test_before = split.test.copy()

    gradient_ascent_unlearning(model, split, target_user=1, epochs=1)

    pd.testing.assert_frame_equal(split.val, val_before)
    pd.testing.assert_frame_equal(split.test, test_before)


def test_gradient_ascent_reset_target_embedding_flag(toy_setup):
    model, split = toy_setup

    # Without reset: target user base embedding is modified only by gradient ascent
    unlearned_no_reset, meta_no_reset = gradient_ascent_unlearning(
        model, split, target_user=0, epochs=2, reset_target_embedding=False
    )
    assert meta_no_reset["reset_target_embedding"] is False

    # With reset: target user base embedding is re-initialized
    unlearned_reset, meta_reset = gradient_ascent_unlearning(
        model, split, target_user=0, epochs=2, reset_target_embedding=True
    )
    assert meta_reset["reset_target_embedding"] is True


def test_gradient_ascent_invalid_target_user_raises_error(toy_setup):
    model, split = toy_setup

    with pytest.raises(ValueError, match="target_user -1 is outside"):
        gradient_ascent_unlearning(model, split, target_user=-1)

    with pytest.raises(ValueError, match="target_user 10 is outside"):
        gradient_ascent_unlearning(model, split, target_user=10)


def test_gradient_ascent_user_without_interactions_raises_error():
    # User 3 has no interactions
    train_rows = [(0, 0), (1, 1), (2, 2)]
    split = make_toy_split(train_rows, num_users=4, num_movies=4)
    model = LightGCN(num_users=4, num_movies=4, embedding_dim=8, num_layers=1)

    with pytest.raises(ValueError, match="no training interactions to unlearn"):
        gradient_ascent_unlearning(model, split, target_user=3)
