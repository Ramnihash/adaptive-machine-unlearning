"""Membership-inference dataset and features (CP5.2).

Protocol
--------
Question: given a model's output for a (user, movie) pair, was that interaction
in the model's training data?

Members:      a user's training interactions.
Non-members:  the same user's held-out validation/test interactions. They are
              interactions the user really made, so a model that merely
              predicts taste scores them high too; only memorisation separates
              members from them.
Balance:      per user, equal numbers of members and non-members.
Features:     computed from the model's scores only and scale-free (z-score and
              rank percentile of the item within the user's own score list), so
              they do not depend on embedding scale.
No leakage:   the attack classifier is trained, validated and tested on
              disjoint users, features contain no ids, and the forgotten target
              user is never part of attack training.
"""

import numpy as np
import pandas as pd

from data.split import DataSplit

FEATURE_NAMES = ["score_zscore", "score_percentile"]


def pair_features(scores: np.ndarray, users: np.ndarray, movies: np.ndarray) -> np.ndarray:
    """Scale-free features for (user, movie) pairs from a model's score matrix."""
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True) + 1e-12
    zscore = (scores[users, movies] - mean[users, 0]) / std[users, 0]

    # Percentile of each pair's item within its user's list, computed per user.
    percentile = np.empty(len(users))
    for user in np.unique(users):
        rows = users == user
        order = scores[user].argsort().argsort()
        percentile[rows] = order[movies[rows]] / (scores.shape[1] - 1)

    return np.column_stack([zscore, percentile])


def build_mia_pairs(
    split: DataSplit,
    num_users: int = 2000,
    exclude_users: tuple[int, ...] = (),
    seed: int = 42,
) -> pd.DataFrame:
    """Balanced member / non-member pairs with a user-disjoint attack split.

    Returns columns user_idx, movie_idx, label (1 member, 0 non-member) and
    attack_split (train / val / test, 60/20/20 by user).
    """
    rng = np.random.default_rng(seed)
    held_out = pd.concat([split.val, split.test], ignore_index=True)

    candidates = np.setdiff1d(np.arange(split.num_users), np.array(exclude_users, dtype=int))
    users = rng.choice(candidates, size=min(num_users, len(candidates)), replace=False)

    train_by_user = dict(tuple(split.train.groupby("user_idx")["movie_idx"]))
    held_by_user = dict(tuple(held_out.groupby("user_idx")["movie_idx"]))

    rows = []
    for position, user in enumerate(users):
        members = train_by_user[user].to_numpy()
        others = held_by_user[user].to_numpy()
        n = min(len(members), len(others))
        members = rng.choice(members, size=n, replace=False)
        others = rng.choice(others, size=n, replace=False)

        fraction = position / len(users)
        attack_split = "train" if fraction < 0.6 else "val" if fraction < 0.8 else "test"

        rows += [(user, m, 1, attack_split) for m in members]
        rows += [(user, o, 0, attack_split) for o in others]

    return pd.DataFrame(rows, columns=["user_idx", "movie_idx", "label", "attack_split"])


def verify_mia_pairs(pairs: pd.DataFrame, exclude_users: tuple[int, ...] = ()) -> dict[str, object]:
    """Check class balance and that no user appears in two attack splits."""
    users_per_split = {name: set(group["user_idx"]) for name, group in pairs.groupby("attack_split")}
    names = list(users_per_split)
    overlap = any(
        users_per_split[a] & users_per_split[b]
        for i, a in enumerate(names) for b in names[i + 1:]
    )
    balance = pairs.groupby("attack_split")["label"].mean().round(3).to_dict()
    duplicated = bool(pairs.duplicated(["user_idx", "movie_idx"]).any())
    excluded_present = bool(set(pairs["user_idx"]) & set(exclude_users))

    report = {
        "pairs": len(pairs),
        "members_share_per_split": balance,
        "user_overlap_between_splits": overlap,
        "duplicate_pairs": duplicated,
        "excluded_user_present": excluded_present,
    }
    if overlap or duplicated or excluded_present or any(abs(v - 0.5) > 1e-9 for v in balance.values()):
        raise ValueError(f"MIA dataset failed verification: {report}")
    return report
