"""User-level membership inference with a shadow model (CP5.2, CP5.3, CP5.4).

Protocol
--------
Question: was this user's data used to train the model? This is exactly what
unlearning must erase.

Interaction-level attacks (member vs held-out items of the same user) turned out
to be weak on LightGCN, so this is the primary privacy measurement.

Shadow model: a LightGCN trained like θ₀ on D_total minus a random set of users
(the stage train_shadow). For the shadow model we know the ground truth:
  members      = users that were in its training data
  non-members  = users that were removed from it
The attacker knows each user's claimed interaction history and looks at how the
model scores those items. Features (scale-free, from model scores only plus the
history size) summarise where the history items rank in the user's list.

The attack is trained on the shadow model, tested on disjoint shadow users, then
applied to θ₀, θ*, the baselines and θᵤ for the forgotten target user. A model
that has forgotten the user should look like a non-member.

No leakage: attack train / validation / test users are disjoint, features hold
no ids, the target user is excluded from attack training, and the attack never
sees θ₀, θ* or θᵤ while being fitted.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

USER_FEATURES = [
    "mean_zscore", "mean_percentile", "median_percentile", "share_in_top5pct", "log_history_size",
]

USER_ATTACK_CONFIG = {
    "classifier": "GradientBoostingClassifier",
    "n_estimators": 100,
    "max_depth": 3,
    "random_state": 42,
}


def history_features(scores: np.ndarray, history: pd.DataFrame, users: np.ndarray) -> np.ndarray:
    """One feature row per user, from where their history items rank in the model's output."""
    num_items = scores.shape[1]
    mean = scores.mean(axis=1)
    std = scores.std(axis=1) + 1e-12
    grouped = dict(tuple(history.groupby("user_idx")["movie_idx"]))

    rows = []
    for user in users:
        items = grouped[int(user)].to_numpy()
        row = scores[user]
        percentile = row.argsort().argsort()[items] / (num_items - 1)
        zscore = (row[items] - mean[user]) / std[user]
        rows.append([
            zscore.mean(),
            percentile.mean(),
            np.median(percentile),
            (percentile >= 0.95).mean(),
            np.log1p(len(items)),
        ])
    return np.array(rows)


def build_user_attack_data(
    shadow_scores: np.ndarray,
    history: pd.DataFrame,
    removed_users: np.ndarray,
    exclude_users: tuple[int, ...] = (),
    seed: int = 42,
) -> dict:
    """Balanced members / non-members from the shadow model, split by user 60/20/20."""
    rng = np.random.default_rng(seed)
    non_members = np.setdiff1d(removed_users, np.array(exclude_users, dtype=int))
    all_users = np.arange(shadow_scores.shape[0])
    candidates = np.setdiff1d(all_users, np.concatenate([removed_users, np.array(exclude_users, dtype=int)]))
    if len(candidates) < len(non_members):
        raise ValueError("Need at least as many member users as removed users for a balanced attack set.")
    members = rng.choice(candidates, size=len(non_members), replace=False)

    users = np.concatenate([members, non_members])
    labels = np.concatenate([np.ones(len(members)), np.zeros(len(non_members))]).astype(int)
    order = rng.permutation(len(users))
    users, labels = users[order], labels[order]

    cut_train, cut_val = int(0.6 * len(users)), int(0.8 * len(users))
    part = np.array(["train"] * cut_train + ["val"] * (cut_val - cut_train) + ["test"] * (len(users) - cut_val))

    features = history_features(shadow_scores, history, users)
    return {"users": users, "labels": labels, "part": part, "features": features}


def verify_user_attack_data(data: dict, exclude_users: tuple[int, ...] = ()) -> dict:
    """Class balance per split, no user in two splits, target user absent."""
    users, labels, part = data["users"], data["labels"], data["part"]
    shares = {name: round(float(labels[part == name].mean()), 3) for name in ("train", "val", "test")}
    overlap = len(set(users)) != len(users)
    target_present = bool(set(users.tolist()) & set(exclude_users))
    report = {
        "users": int(len(users)),
        "members_share_per_split": shares,
        "users_in_two_splits": overlap,
        "target_user_present": target_present,
    }
    if overlap or target_present or any(abs(v - 0.5) > 0.06 for v in shares.values()):
        raise ValueError(f"User-level MIA dataset failed verification: {report}")
    return report


def fit_user_attack(data: dict) -> dict:
    """Train on shadow-train users; report AUC / accuracy on shadow-val and shadow-test users."""
    params = {k: v for k, v in USER_ATTACK_CONFIG.items() if k != "classifier"}
    train = data["part"] == "train"
    classifier = GradientBoostingClassifier(**params).fit(data["features"][train], data["labels"][train])

    metrics = {}
    for name in ("val", "test"):
        rows = data["part"] == name
        probability = classifier.predict_proba(data["features"][rows])[:, 1]
        metrics[name] = {
            "auc": float(roc_auc_score(data["labels"][rows], probability)),
            "accuracy": float(accuracy_score(data["labels"][rows], probability >= 0.5)),
            "n": int(rows.sum()),
        }
    return {"classifier": classifier, "metrics": metrics, "config": USER_ATTACK_CONFIG}


def user_membership_probability(classifier, scores: np.ndarray, history: pd.DataFrame, users: np.ndarray) -> np.ndarray:
    """Probability that each user was in the training data of the model that produced scores."""
    return classifier.predict_proba(history_features(scores, history, users))[:, 1]
