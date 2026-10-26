"""Recommendation-level forgetting: do the removed interactions still surface? (CP5.1, CP5.5)"""

import numpy as np


def forgotten_item_metrics(
    scores: np.ndarray,
    user: int,
    forgotten_items: np.ndarray | list[int] | set[int],
    k: int = 10,
) -> dict[str, float]:
    """How strongly a model still ranks the target user's forgotten items.

    Nothing is masked, so a model that memorised the user ranks their old items
    at the top. After successful unlearning the values drop to chance level.

    hit_rate@k:  share of the user's top-k that are forgotten items.
    mean_percentile: average rank position of forgotten items, 1.0 = top of the
                 list and about 0.5 = random.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    row = np.asarray(scores[user], dtype=np.float64)
    catalog_size = len(row)
    if catalog_size <= 0:
        raise ValueError("Catalog size must be positive, got 0.")

    forgotten_items = np.asarray(forgotten_items, dtype=np.int64)
    if len(forgotten_items) == 0:
        return {
            f"forgotten_hit_rate@{k}": 0.0,
            "forgotten_mean_percentile": float("nan"),
        }

    effective_k = min(k, catalog_size)
    if effective_k < catalog_size:
        top_k = np.argpartition(-row, effective_k - 1)[:effective_k]
    else:
        top_k = np.arange(catalog_size)

    hit_rate = float(np.isin(top_k, forgotten_items).sum() / effective_k)

    if catalog_size > 1:
        order = row.argsort().argsort()
        percentile = order[forgotten_items] / (catalog_size - 1)
        mean_percentile = float(percentile.mean())
    else:
        mean_percentile = 1.0

    return {
        f"forgotten_hit_rate@{k}": hit_rate,
        "forgotten_mean_percentile": mean_percentile,
    }
