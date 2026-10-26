"""Unit tests for approximate localized gradient adaptation unlearning strategy (CP15/CP16)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch

from data.split import DataSplit, remove_user_from_split
from models.lightgcn import LightGCN
from unlearning.strategies.localized_update import (
    get_affected_neighborhood,
    localized_update_unlearning,
)


def make_toy_split(
    train_rows: list[tuple[int, int]],
    val_rows: list[tuple[int, int]] | None = None,
    test_rows: list[tuple[int, int]] | None = None,
    num_users: int = 5,
    num_movies: int = 8,
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
    # Target user 0 interacts with movies 0 and 1.
    # User 1 also interacts with movie 1 (2-hop neighbor of user 0).
    # User 2 interacts with movie 2 and 3.
    # User 3 interacts with movie 4 and 5 (distant, unaffected).
    # User 4 interacts with movie 6 and 7 (distant, unaffected).
    train_rows = [
        (0, 0), (0, 1),
        (1, 1), (1, 2),
        (2, 2), (2, 3),
        (3, 4), (3, 5),
        (4, 6), (4, 7),
    ]
    val_rows = [(0, 2), (1, 3)]
    test_rows = [(0, 3), (1, 4)]
    split = make_toy_split(train_rows, val_rows, test_rows, num_users=5, num_movies=8)
    model = LightGCN(num_users=5, num_movies=8, embedding_dim=16, num_layers=2)
    return model, split


def test_get_affected_neighborhood_structure(toy_setup):
    _, split = toy_setup
    hop1_items, neighbor_users, affected_nodes = get_affected_neighborhood(split, target_user=0)

    # 1-hop items for user 0 are {0, 1}
    assert set(hop1_items.tolist()) == {0, 1}
    # 2-hop neighbor is user 1 (also interacted with movie 1)
    assert set(neighbor_users.tolist()) == {1}
    # Unaffected users are {2, 3, 4}
    assert 2 not in neighbor_users and 3 not in neighbor_users and 4 not in neighbor_users


def test_localized_update_preserves_caller_model(toy_setup):
    model, split = toy_setup
    weight_before = model.embedding.weight.clone().detach()

    unlearned, metadata = localized_update_unlearning(
        model, split, target_user=0, epochs=2, learning_rate=0.01
    )

    assert torch.equal(model.embedding.weight, weight_before)
    assert not torch.equal(unlearned.embedding.weight, model.embedding.weight)


def test_unaffected_nodes_strictly_frozen_and_unchanged(toy_setup):
    model, split = toy_setup
    _, _, affected_nodes = get_affected_neighborhood(split, target_user=0)

    total_nodes = split.num_users + split.num_movies
    unaffected_nodes = np.setdiff1d(np.arange(total_nodes), affected_nodes)

    unlearned, metadata = localized_update_unlearning(
        model, split, target_user=0, epochs=3, learning_rate=0.01
    )

    # Every unaffected node MUST remain bitwise identical to the original model
    for node in unaffected_nodes:
        assert torch.equal(
            unlearned.embedding.weight[node], model.embedding.weight[node]
        ), f"Node {node} was modified but is outside the affected neighborhood!"

    # The target user base embedding MUST differ
    assert not torch.equal(
        unlearned.embedding.weight[0], model.embedding.weight[0]
    )


def test_localized_update_metadata_and_approximation_field(toy_setup):
    model, split = toy_setup
    unlearned, metadata = localized_update_unlearning(
        model, split, target_user=0, epochs=1
    )

    assert metadata["strategy"] == "localized_update"
    assert metadata["target_user"] == 0
    assert metadata["approximation"] == "localized_gradient_adaptation"
    assert metadata["hop1_items_count"] == 2
    assert metadata["neighbor_users_count"] == 1
    assert metadata["unlearning_seconds"] > 0


def test_localized_update_invalid_target_user_raises_error(toy_setup):
    model, split = toy_setup

    with pytest.raises(ValueError, match="target_user -1 is outside"):
        localized_update_unlearning(model, split, target_user=-1)

    with pytest.raises(ValueError, match="target_user 20 is outside"):
        localized_update_unlearning(model, split, target_user=20)


def test_localized_update_user_without_interactions_raises_error():
    train_rows = [(0, 0), (1, 1)]
    split = make_toy_split(train_rows, num_users=3, num_movies=3)
    model = LightGCN(num_users=3, num_movies=3, embedding_dim=8, num_layers=1)

    with pytest.raises(ValueError, match="no training interactions to unlearn"):
        localized_update_unlearning(model, split, target_user=2)


from unlearning.strategies.localized_update import (
    get_affected_neighborhood,
    is_member_sorted,
    localized_update_unlearning,
)


def test_positive_keys_built_from_all_split_star_train(toy_setup, monkeypatch):
    model, split = toy_setup
    split_star = remove_user_from_split(split, 0)

    captured_keys = []
    import unlearning.strategies.localized_update as loc_module
    original_member = loc_module.is_member_sorted

    def spy_member(query, sorted_keys):
        captured_keys.append(sorted_keys.clone().detach())
        return original_member(query, sorted_keys)

    monkeypatch.setattr(loc_module, "is_member_sorted", spy_member)

    unlearned, metadata = localized_update_unlearning(
        model, split, target_user=0, epochs=1, batch_size=4
    )

    assert len(captured_keys) > 0
    actual_keys = captured_keys[0].cpu().numpy()

    # Reconstruct expected keys from ALL of split_star.train
    expected_keys = np.sort(
        split_star.train["user_idx"].to_numpy() * split.num_movies
        + split_star.train["movie_idx"].to_numpy()
    )

    # 1. Must match ALL split_star.train keys exactly
    np.testing.assert_array_equal(actual_keys, expected_keys)

    # 2. Interacted items of unaffected users (not in affected_df) must be in positive_keys
    u3_pos_key = 3 * split.num_movies + 4  # user 3, movie 4
    assert u3_pos_key in actual_keys

    # 3. Target user's removed interactions must NOT be in positive_keys
    target_pos_key = 0 * split.num_movies + 0
    assert target_pos_key not in actual_keys


def test_negative_sampling_rejects_retained_neighbor_positive(toy_setup, monkeypatch):
    model, split = toy_setup
    call_count = 0

    def mock_randint(high, size, device=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Propose movie 2 (which is a retained positive for user 1)
            return torch.tensor([2], device=device)
        # Resample with movie 7 (valid negative)
        return torch.tensor([7], device=device)

    monkeypatch.setattr(torch, "randint", mock_randint)

    unlearned, metadata = localized_update_unlearning(
        model, split, target_user=0, epochs=1, batch_size=1
    )

    assert call_count > 1


def test_is_member_sorted_matches_torch_isin():
    """Verify is_member_sorted exactly matches torch.isin across all edge cases."""
    # 1. Empty keys
    q = torch.tensor([1, 2, 3], dtype=torch.long)
    k_empty = torch.tensor([], dtype=torch.long)
    assert torch.equal(is_member_sorted(q, k_empty), torch.isin(q, k_empty))

    # 2. Empty query
    k = torch.tensor([1, 2, 3], dtype=torch.long)
    q_empty = torch.tensor([], dtype=torch.long)
    assert torch.equal(is_member_sorted(q_empty, k), torch.isin(q_empty, k))

    # 3. Both empty
    assert torch.equal(is_member_sorted(q_empty, k_empty), torch.isin(q_empty, k_empty))

    # 4. Boundaries, duplicates, present, and absent keys
    # Keys with duplicates: [10, 20, 20, 30, 50]
    keys = torch.tensor([10, 20, 20, 30, 50], dtype=torch.long)
    # Queries: smaller than min (5), equal min (10), in between absent (15),
    # duplicate present (20, 20), absent (25), present (30), absent (45),
    # equal max (50), greater than max (55)
    queries = torch.tensor([5, 10, 15, 20, 20, 25, 30, 45, 50, 55], dtype=torch.long)
    expected = torch.isin(queries, keys)
    actual = is_member_sorted(queries, keys)
    assert torch.equal(actual, expected)

    # 5. Single element keys
    single_key = torch.tensor([42], dtype=torch.long)
    test_q = torch.tensor([41, 42, 43], dtype=torch.long)
    assert torch.equal(is_member_sorted(test_q, single_key), torch.isin(test_q, single_key))

    # 6. Random tests across multiple seeds and sizes
    for seed in (0, 42, 123):
        torch.manual_seed(seed)
        rand_keys = torch.sort(torch.randint(0, 10000, (1000,))).values
        rand_queries = torch.randint(0, 10000, (500,))
        assert torch.equal(
            is_member_sorted(rand_queries, rand_keys),
            torch.isin(rand_queries, rand_keys),
        )
