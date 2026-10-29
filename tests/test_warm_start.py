"""Tests for warm-start retraining and the naive fine-tuning baseline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
import torch

from data.split import DataSplit
from models.lightgcn import LightGCN
from unlearning.strategies.warm_start import warm_start_retraining


@pytest.fixture
def toy_setup():
    torch.manual_seed(42)
    cols = ["user_idx", "movie_idx"]
    rows = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 3), (2, 4), (3, 4), (3, 5)]
    split = DataSplit(
        train=pd.DataFrame(rows, columns=cols),
        val=pd.DataFrame([(0, 2), (1, 3)], columns=cols),
        test=pd.DataFrame([(0, 3), (1, 4)], columns=cols),
        num_users=4,
        num_movies=6,
        seed=42,
    )
    return LightGCN(num_users=4, num_movies=6, embedding_dim=16, num_layers=2), split


def test_warm_start_resets_target_and_keeps_original(toy_setup):
    model, split = toy_setup
    before = model.embedding.weight.clone().detach()

    unlearned, meta = warm_start_retraining(model, split, 0, epochs=2, batch_size=4)

    assert torch.equal(model.embedding.weight, before)
    assert not torch.equal(unlearned.embedding.weight[0], before[0])
    assert meta["strategy"] == "warm_start_retraining"
    assert meta["train_interactions_remaining"] == 6
    assert meta["forgotten_interactions_count"] == 2
    assert len(meta["losses"]) == 2


def test_naive_finetune_keeps_target_embedding(toy_setup):
    model, split = toy_setup
    before = model.embedding.weight[0].clone().detach()

    # Embedding reg only touches users that appear in batches, and the target has none.
    unlearned, meta = warm_start_retraining(
        model, split, 0, epochs=2, batch_size=4, reset_target_embedding=False
    )

    assert meta["strategy"] == "naive_finetune"
    assert torch.allclose(unlearned.embedding.weight[0], before, atol=1e-6)


def test_warm_start_rejects_bad_input(toy_setup):
    model, split = toy_setup
    with pytest.raises(ValueError):
        warm_start_retraining(model, split, 99)
    with pytest.raises(ValueError):
        warm_start_retraining(model, split, 0, epochs=0)
