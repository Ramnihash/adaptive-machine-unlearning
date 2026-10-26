"""Unit tests for user influence estimation and tier classification (Person 2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from data.split import DataSplit
from influence import (
    DEFAULT_THRESHOLDS,
    UserInfluence,
    classify_tier,
    classify_user_tiers,
    compute_influence_scores,
    compute_user_degrees,
    estimate_influence,
)


def make_toy_split(
    train_rows: list[tuple[int, int]],
    val_rows: list[tuple[int, int]] | None = None,
    test_rows: list[tuple[int, int]] | None = None,
    num_users: int = 4,
    num_movies: int = 10,
    seed: int = 42,
) -> DataSplit:
    """Helper to construct small synthetic DataSplit instances."""
    cols = ["user_idx", "movie_idx"]
    train_df = pd.DataFrame(train_rows, columns=cols)
    val_df = pd.DataFrame(val_rows or [], columns=cols)
    test_df = pd.DataFrame(test_rows or [], columns=cols)
    return DataSplit(
        train=train_df,
        val=val_df,
        test=test_df,
        num_users=num_users,
        num_movies=num_movies,
        seed=seed,
    )


def test_compute_user_degrees_correct_counts():
    train_rows = [(0, 1), (0, 2), (1, 3), (3, 1), (3, 2), (3, 4)]
    train_df = pd.DataFrame(train_rows, columns=["user_idx", "movie_idx"])
    degrees = compute_user_degrees(train_df, num_users=4)

    assert degrees.tolist() == [2, 1, 0, 3]


def test_zero_interaction_user_has_zero_degree_and_lowest_score():
    # User 2 has 0 interactions
    train_rows = [(0, 1), (0, 2), (1, 3), (3, 1), (3, 2), (3, 4)]
    split = make_toy_split(train_rows, num_users=4)

    inf = estimate_influence(split)
    assert inf[2].degree == 0
    # user 2 has the lowest score among all users
    for u in (0, 1, 3):
        assert inf[2].score < inf[u].score


def test_tied_degrees_produce_equal_scores_independent_of_order():
    # Users 0, 1, 2 each have degree 2; user 3 has degree 5
    train_rows = [
        (0, 10), (0, 11),
        (1, 12), (1, 13),
        (2, 14), (2, 15),
        (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
    ]
    degrees = compute_user_degrees(pd.DataFrame(train_rows, columns=["user_idx", "movie_idx"]), num_users=4)
    scores = compute_influence_scores(degrees, method="percentile")

    # Tied degrees must produce strictly identical scores
    assert scores[0] == scores[1] == scores[2]
    assert scores[3] > scores[0]


def test_deterministic_repeatability_under_row_shuffling():
    # Shuffling training rows must not alter degrees or scores
    rng = np.random.default_rng(123)
    train_rows = [
        (0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6), (3, 7), (3, 8)
    ]
    split_a = make_toy_split(train_rows, num_users=4)

    shuffled_rows = [train_rows[i] for i in rng.permutation(len(train_rows))]
    split_b = make_toy_split(shuffled_rows, num_users=4)

    inf_a = estimate_influence(split_a)
    inf_b = estimate_influence(split_b)

    for u in range(4):
        assert inf_a[u].degree == inf_b[u].degree
        assert inf_a[u].score == inf_b[u].score
        assert inf_a[u].tier == inf_b[u].tier


def test_single_user_produces_score_half():
    split = make_toy_split([(0, 1), (0, 2)], num_users=1)
    res = estimate_influence(split, target_user=0)

    assert isinstance(res, UserInfluence)
    assert res.user_idx == 0
    assert res.degree == 2
    assert res.score == 0.5
    assert res.tier == "medium"


def test_all_users_equal_degrees_produces_score_half():
    # All users have degree 3
    train_rows = [
        (0, 1), (0, 2), (0, 3),
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3),
    ]
    split = make_toy_split(train_rows, num_users=3)
    inf = estimate_influence(split)

    for u in range(3):
        assert inf[u].degree == 3
        assert inf[u].score == 0.5
        assert inf[u].tier == "medium"


def test_split_isolation_ignores_val_and_test():
    # Changes in val and test data must NEVER change influence metrics
    train_rows = [(0, 1), (1, 2), (1, 3)]
    split_1 = make_toy_split(train_rows, val_rows=[], test_rows=[], num_users=2)
    split_2 = make_toy_split(
        train_rows,
        val_rows=[(0, 5), (0, 6), (0, 7), (0, 8)],
        test_rows=[(0, 9), (0, 10)],
        num_users=2,
    )

    inf_1 = estimate_influence(split_1)
    inf_2 = estimate_influence(split_2)

    for u in range(2):
        assert inf_1[u] == inf_2[u]


def test_tier_boundary_classification():
    # low <= 0.33, medium <= 0.66, high > 0.66
    assert classify_tier(0.00) == "low"
    assert classify_tier(0.33) == "low"
    assert classify_tier(0.330001) == "medium"
    assert classify_tier(0.66) == "medium"
    assert classify_tier(0.660001) == "high"
    assert classify_tier(1.00) == "high"


def test_custom_thresholds_override():
    custom = {"low": 0.20, "high": 0.80}
    assert classify_tier(0.20, thresholds=custom) == "low"
    assert classify_tier(0.25, thresholds=custom) == "medium"
    assert classify_tier(0.80, thresholds=custom) == "medium"
    assert classify_tier(0.81, thresholds=custom) == "high"


def test_classify_user_tiers_vectorized():
    scores = np.array([0.10, 0.33, 0.50, 0.66, 0.90])
    tiers = classify_user_tiers(scores)
    assert tiers == ["low", "low", "medium", "medium", "high"]


def test_optional_minmax_scoring_method():
    degrees = np.array([10, 20, 30])
    scores = compute_influence_scores(degrees, method="minmax")
    assert scores.tolist() == [0.0, 0.5, 1.0]

    # Tied minmax also returns 0.5 when min == max
    tied = compute_influence_scores(np.array([5, 5, 5]), method="minmax")
    assert tied.tolist() == [0.5, 0.5, 0.5]


def test_target_user_parameter_and_validation():
    train_rows = [(0, 1), (1, 2)]
    split = make_toy_split(train_rows, num_users=2)

    target_res = estimate_influence(split, target_user=1)
    assert isinstance(target_res, UserInfluence)
    assert target_res.user_idx == 1

    with pytest.raises(ValueError, match="target_user 5 is outside"):
        estimate_influence(split, target_user=5)

    with pytest.raises(ValueError, match="target_user -1 is outside"):
        estimate_influence(split, target_user=-1)


def test_invalid_arguments_raise_value_errors():
    with pytest.raises(ValueError, match="num_users must be positive"):
        compute_user_degrees(pd.DataFrame(), num_users=0)

    with pytest.raises(ValueError, match="user_idx values must be within"):
        compute_user_degrees(pd.DataFrame({"user_idx": [5]}), num_users=3)

    with pytest.raises(ValueError, match="train DataFrame must contain a 'user_idx' column"):
        compute_user_degrees(pd.DataFrame({"movie_idx": [1]}), num_users=2)

    with pytest.raises(ValueError, match="degrees array cannot be empty"):
        compute_influence_scores(np.array([]))

    with pytest.raises(ValueError, match="degrees must be non-negative"):
        compute_influence_scores(np.array([-1, 2]))

    with pytest.raises(ValueError, match="Unknown scoring method"):
        compute_influence_scores(np.array([1, 2]), method="unknown")

    with pytest.raises(ValueError, match="score must be between 0.0 and 1.0"):
        classify_tier(1.5)

    with pytest.raises(ValueError, match="Invalid thresholds"):
        classify_tier(0.5, thresholds={"low": 0.8, "high": 0.2})
