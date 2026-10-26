"""Unit tests for the adaptive unlearning router (CP13/CP14)."""

from dataclasses import FrozenInstanceError
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from data.split import DataSplit
from influence import UserInfluence
from unlearning import (
    DEFAULT_STRATEGY_MAP,
    RoutingDecision,
    route_unlearning,
    select_strategy,
)


def make_toy_split(
    train_rows: list[tuple[int, int]],
    val_rows: list[tuple[int, int]] | None = None,
    test_rows: list[tuple[int, int]] | None = None,
    num_users: int = 6,
    num_movies: int = 30,
    seed: int = 42,
) -> DataSplit:
    """Helper to construct synthetic DataSplit instances."""
    cols = ["user_idx", "movie_idx"]
    return DataSplit(
        train=pd.DataFrame(train_rows, columns=cols),
        val=pd.DataFrame(val_rows or [], columns=cols),
        test=pd.DataFrame(test_rows or [], columns=cols),
        num_users=num_users,
        num_movies=num_movies,
        seed=seed,
    )


def test_routing_decision_immutability():
    inf = UserInfluence(user_idx=1, degree=5, score=0.5, tier="medium")
    decision = RoutingDecision(target_user=1, influence=inf, strategy="localized_update")

    assert decision.target_user == 1
    assert decision.influence == inf
    assert decision.strategy == "localized_update"

    with pytest.raises(FrozenInstanceError):
        decision.strategy = "other"  # type: ignore[misc]


def test_select_strategy_default_mapping():
    assert select_strategy("low") == "gradient_ascent"
    assert select_strategy("medium") == "localized_update"
    assert select_strategy("high") == "targeted_retraining"


def test_select_strategy_custom_mapping():
    custom = {
        "low": "my_low_algo",
        "medium": "my_med_algo",
        "high": "my_high_algo",
    }
    assert select_strategy("low", strategy_map=custom) == "my_low_algo"
    assert select_strategy("medium", strategy_map=custom) == "my_med_algo"
    assert select_strategy("high", strategy_map=custom) == "my_high_algo"


def test_select_strategy_unknown_tier_raises_error():
    with pytest.raises(ValueError, match="Unknown tier 'ultra_low'"):
        select_strategy("ultra_low")

    with pytest.raises(ValueError, match="Unknown tier ''"):
        select_strategy("")


def test_select_strategy_missing_tier_entry_raises_error():
    incomplete = {"low": "gradient_ascent"}
    with pytest.raises(ValueError, match="Strategy mapping has no entry for tier 'medium'"):
        select_strategy("medium", strategy_map=incomplete)


def test_select_strategy_empty_or_whitespace_raises_error():
    with pytest.raises(ValueError, match="must be a non-empty string"):
        select_strategy("low", strategy_map={"low": "  ", "medium": "m", "high": "h"})

    with pytest.raises(ValueError, match="must be a non-empty string"):
        select_strategy("low", strategy_map={"low": "", "medium": "m", "high": "h"})


def test_select_strategy_empty_mapping_raises_error():
    with pytest.raises(ValueError, match="must be a non-empty dictionary"):
        select_strategy("low", strategy_map={})


def test_route_unlearning_three_tiers():
    # Construct a 6-user split with distinct degrees:
    # Users 0, 1: 1 interaction each  (bottom 2 / 6 -> score ~ 0.166 <= 0.33 -> low)
    # Users 2, 3: 5 interactions each  (middle 2 / 6 -> score ~ 0.500 <= 0.66 -> medium)
    # Users 4, 5: 15 interactions each (top 2 / 6   -> score ~ 0.833 > 0.66  -> high)
    rows = []
    for m in [1]:
        rows.append((0, m))
        rows.append((1, m))
    for m in range(2, 7):
        rows.append((2, m))
        rows.append((3, m))
    for m in range(7, 22):
        rows.append((4, m))
        rows.append((5, m))

    split = make_toy_split(rows, num_users=6)

    # Route low-tier user
    dec_low = route_unlearning(split, target_user=0)
    assert dec_low.target_user == 0
    assert dec_low.influence.tier == "low"
    assert dec_low.strategy == "gradient_ascent"

    # Route medium-tier user
    dec_med = route_unlearning(split, target_user=2)
    assert dec_med.target_user == 2
    assert dec_med.influence.tier == "medium"
    assert dec_med.strategy == "localized_update"

    # Route high-tier user
    dec_high = route_unlearning(split, target_user=4)
    assert dec_high.target_user == 4
    assert dec_high.influence.tier == "high"
    assert dec_high.strategy == "targeted_retraining"


def test_route_unlearning_custom_strategy_map():
    rows = [(0, 1), (1, 2), (1, 3)]
    split = make_toy_split(rows, num_users=2)
    custom_map = {
        "low": "custom_ga",
        "medium": "custom_lu",
        "high": "custom_tr",
    }
    # All users tied (N=2, degrees 1 and 2, but min < max):
    dec = route_unlearning(split, target_user=0, strategy_map=custom_map)
    assert dec.strategy in custom_map.values()


def test_route_unlearning_custom_thresholds_passed_through():
    # 2 users with degree 1 and degree 3.
    # Scores: user 0 -> 0.25, user 1 -> 0.75
    rows = [(0, 1), (1, 2), (1, 3), (1, 4)]
    split = make_toy_split(rows, num_users=2)

    # Default thresholds: user 1 (score 0.75) is > 0.66 -> high
    dec_default = route_unlearning(split, target_user=1)
    assert dec_default.influence.tier == "high"
    assert dec_default.strategy == "targeted_retraining"

    # With custom thresholds where high is 0.80: user 1 (score 0.75) is medium
    custom_thresh = {"low": 0.33, "high": 0.80}
    dec_custom = route_unlearning(split, target_user=1, thresholds=custom_thresh)
    assert dec_custom.influence.tier == "medium"
    assert dec_custom.strategy == "localized_update"


def test_route_unlearning_invalid_target_users_raise_error():
    split = make_toy_split([(0, 1), (1, 2)], num_users=2)

    with pytest.raises(ValueError, match="target_user -1 is outside"):
        route_unlearning(split, target_user=-1)

    with pytest.raises(ValueError, match="target_user 5 is outside"):
        route_unlearning(split, target_user=5)


def test_route_unlearning_deterministic_repeatability():
    rng = np.random.default_rng(999)
    train_rows = [(0, 1), (0, 2), (1, 3), (2, 4), (2, 5), (2, 6)]
    split_a = make_toy_split(train_rows, num_users=3)

    shuffled_rows = [train_rows[i] for i in rng.permutation(len(train_rows))]
    split_b = make_toy_split(shuffled_rows, num_users=3)

    for u in range(3):
        dec_a = route_unlearning(split_a, target_user=u)
        dec_b = route_unlearning(split_b, target_user=u)
        assert dec_a.target_user == dec_b.target_user
        assert dec_a.influence == dec_b.influence
        assert dec_a.strategy == dec_b.strategy


def test_router_does_not_mutate_data_split():
    train_rows = [(0, 1), (0, 2), (1, 3)]
    val_rows = [(0, 4)]
    test_rows = [(1, 5)]
    split = make_toy_split(train_rows, val_rows, test_rows, num_users=2)

    train_before = split.train.copy()
    val_before = split.val.copy()
    test_before = split.test.copy()

    route_unlearning(split, target_user=0)

    pd.testing.assert_frame_equal(split.train, train_before)
    pd.testing.assert_frame_equal(split.val, val_before)
    pd.testing.assert_frame_equal(split.test, test_before)
