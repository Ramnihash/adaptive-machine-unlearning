"""Recommendation-level forgetting: do the removed interactions still surface? (CP5.1, CP5.5)"""

import numpy as np


def forgotten_item_metrics(
    scores: np.ndarray,
    user: int,
    forgotten_items: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    """How strongly a model still ranks the target user's forgotten items.

    Nothing is masked, so a model that memorised the user ranks their old items
    at the top. After successful unlearning the values drop to chance level.

    hit_rate@k:  share of the user's top-k that are forgotten items.
    mean_percentile: average rank position of forgotten items, 1.0 = top of the
                 list and about 0.5 = random.
    """
    row = scores[user]
    top_k = np.argpartition(-row, k)[:k]
    hit_rate = float(np.isin(top_k, forgotten_items).sum() / k)

    order = row.argsort().argsort()
    percentile = order[forgotten_items] / (len(row) - 1)

    return {
        f"forgotten_hit_rate@{k}": hit_rate,
        "forgotten_mean_percentile": float(percentile.mean()),
    }
