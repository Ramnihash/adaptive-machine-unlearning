"""Unit tests for forgetting, consistency and membership-inference helpers.

These use tiny synthetic score matrices to check the formulas; they are not
experimental results.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from data.split import DataSplit
from evaluation.consistency_metrics import consistency_metrics, jaccard
from evaluation.forgetting_metrics import forgotten_item_metrics
from privacy.mia_dataset import build_mia_pairs, verify_mia_pairs
from privacy.mia_user_level import build_user_attack_data, fit_user_attack, verify_user_attack_data


def test_forgotten_items_at_top_give_full_hit_rate():
    scores = np.zeros((1, 20))
    scores[0, [3, 4, 5]] = [3.0, 2.0, 1.0]
    metrics = forgotten_item_metrics(scores, 0, np.array([3, 4, 5]), k=3)
    assert metrics["forgotten_hit_rate@3"] == 1.0
    assert metrics["forgotten_mean_percentile"] > 0.9


def test_jaccard_identical_and_disjoint():
    a = np.array([[1, 2, 3]])
    assert jaccard(a, a)[0] == 1.0
    assert jaccard(a, np.array([[4, 5, 6]]))[0] == 0.0


def test_consistency_of_identical_models_is_one():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(6, 30))
    seen = pd.DataFrame({"user_idx": [0, 1], "movie_idx": [0, 1]})
    metrics = consistency_metrics(scores, scores, seen, target_user=2, k=5)
    assert metrics["jaccard@5_retained"] == 1.0
    assert metrics["score_correlation_retained"] == pytest.approx(1.0)


def toy_split(num_users=40, num_movies=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = {"train": [], "val": [], "test": []}
    for user in range(num_users):
        items = rng.permutation(num_movies)[:30]
        for name, chunk in zip(rows, (items[:20], items[20:25], items[25:30])):
            rows[name] += [(user, m) for m in chunk]
    frames = [pd.DataFrame(rows[n], columns=["user_idx", "movie_idx"]) for n in ("train", "val", "test")]
    return DataSplit(*frames, num_users=num_users, num_movies=num_movies, seed=seed)


def test_interaction_mia_pairs_are_balanced_and_user_disjoint():
    split = toy_split()
    pairs = build_mia_pairs(split, num_users=30, exclude_users=(0,))
    report = verify_mia_pairs(pairs, exclude_users=(0,))
    assert not report["user_overlap_between_splits"]


def test_user_level_attack_learns_planted_signal():
    # Mock: members' history items score high, non-members' do not.
    split = toy_split()
    removed = np.arange(25, 40)
    scores = np.random.default_rng(1).normal(size=(40, 60))
    for user in range(40):
        items = split.train.loc[split.train["user_idx"] == user, "movie_idx"].to_numpy()
        if user not in removed:
            scores[user, items] += 3
    data = build_user_attack_data(scores, split.train, removed, exclude_users=(0,))
    verify_user_attack_data(data, exclude_users=(0,))
    assert fit_user_attack(data)["metrics"]["test"]["auc"] > 0.8
