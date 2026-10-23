"""Run the attack on a model and on the forgotten user (CP5.3, CP5.4)."""

import numpy as np
import pandas as pd

from data.split import DataSplit
from privacy.mia_attack import ATTACK_CONFIG, attack_metrics, membership_probability, train_attack
from privacy.mia_dataset import build_mia_pairs, pair_features, verify_mia_pairs


def fit_attack_on_original(
    original_scores: np.ndarray,
    split: DataSplit,
    target_user: int,
    num_users: int = 2000,
    seed: int = 42,
) -> dict:
    """Train the attack on θ₀'s outputs for other users and measure it on held-out users.

    Returns the classifier, its metrics on the validation and test users, the
    dataset verification report and its configuration.
    """
    pairs = build_mia_pairs(split, num_users=num_users, exclude_users=(target_user,), seed=seed)
    verification = verify_mia_pairs(pairs, exclude_users=(target_user,))

    features = pair_features(original_scores, pairs["user_idx"].to_numpy(), pairs["movie_idx"].to_numpy())
    labels = pairs["label"].to_numpy()
    part = pairs["attack_split"].to_numpy()

    classifier = train_attack(features[part == "train"], labels[part == "train"])
    return {
        "classifier": classifier,
        "val": attack_metrics(classifier, features[part == "val"], labels[part == "val"]),
        "test": attack_metrics(classifier, features[part == "test"], labels[part == "test"]),
        "verification": verification,
        "config": {**ATTACK_CONFIG, "num_users": num_users, "seed": seed},
    }


def bootstrap_auc_interval(labels: np.ndarray, probability: np.ndarray, seed: int = 42, rounds: int = 1000) -> tuple[float, float]:
    """95% bootstrap interval for AUC; wide when the forgotten user has few interactions."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(rounds):
        pick = rng.integers(0, len(labels), len(labels))
        if len(set(labels[pick])) < 2:
            continue
        values.append(roc_auc_score(labels[pick], probability[pick]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def attack_forgotten_user(
    classifier,
    scores: np.ndarray,
    split: DataSplit,
    target_user: int,
) -> dict[str, float]:
    """Apply the θ₀-trained attack to a model's outputs for the forgotten user.

    Members are the user's original training interactions (what should be
    forgotten); non-members are the user's validation and test interactions.
    A model that has forgotten the user should rate both groups alike
    (AUC near 0.5, advantage near 0).
    """
    from sklearn.metrics import roc_auc_score

    forgotten = split.train.loc[split.train["user_idx"] == target_user, "movie_idx"].to_numpy()
    held_out = pd.concat([split.val, split.test])
    held_out = held_out.loc[held_out["user_idx"] == target_user, "movie_idx"].to_numpy()

    movies = np.concatenate([forgotten, held_out])
    users = np.full(len(movies), target_user)
    labels = np.concatenate([np.ones(len(forgotten)), np.zeros(len(held_out))])

    probability = membership_probability(classifier, pair_features(scores, users, movies))
    low, high = bootstrap_auc_interval(labels, probability)

    return {
        "auc": float(roc_auc_score(labels, probability)),
        "auc_ci95_low": low,
        "auc_ci95_high": high,
        "mean_member_probability_forgotten": float(probability[labels == 1].mean()),
        "mean_member_probability_held_out": float(probability[labels == 0].mean()),
        "advantage": float(probability[labels == 1].mean() - probability[labels == 0].mean()),
        "n_forgotten": int(len(forgotten)),
        "n_held_out": int(len(held_out)),
    }
