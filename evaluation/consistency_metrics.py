"""Consistency between a model and the gold standard θ* (CP5.7)."""

import numpy as np
import pandas as pd


def mask_seen(scores: np.ndarray, seen: pd.DataFrame) -> np.ndarray:
    """Copy of scores with already-seen (user, movie) pairs set to -inf."""
    masked = scores.copy()
    masked[seen["user_idx"].to_numpy(), seen["movie_idx"].to_numpy()] = -np.inf
    return masked


def top_k_lists(scores: np.ndarray, k: int) -> np.ndarray:
    """Top-k movie indices per user (unordered inside the list)."""
    return np.argpartition(-scores, k, axis=1)[:, :k]


def jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Jaccard similarity of two (users, k) index arrays."""
    result = np.empty(len(a))
    for row, (x, y) in enumerate(zip(a, b)):
        overlap = len(np.intersect1d(x, y))
        result[row] = overlap / (len(x) + len(y) - overlap)
    return result


def consistency_metrics(
    scores: np.ndarray,
    reference_scores: np.ndarray,
    seen: pd.DataFrame,
    target_user: int,
    k: int = 10,
) -> dict[str, float]:
    """Compare a model's recommendations with θ*'s under identical masking.

    seen must be the same for both models so the comparison is fair. Reported
    separately for retained users and for the forgotten user, plus the Pearson
    correlation of scores on retained users.
    """
    a = top_k_lists(mask_seen(scores, seen), k)
    b = top_k_lists(mask_seen(reference_scores, seen), k)
    similarity = jaccard(a, b)

    retained = np.arange(len(scores)) != target_user
    pearson = np.corrcoef(scores[retained].ravel(), reference_scores[retained].ravel())[0, 1]

    return {
        f"jaccard@{k}_retained": float(similarity[retained].mean()),
        f"jaccard@{k}_target": float(similarity[target_user]),
        "score_correlation_retained": float(pearson),
    }
