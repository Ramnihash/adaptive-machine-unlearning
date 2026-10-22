"""Top-K recommendation and held-out ranking metrics (CP2.3, CP2.4)."""

import math

import numpy as np
import pandas as pd
import torch

from models.lightgcn import LightGCN


def group_items(ratings: pd.DataFrame) -> dict[int, np.ndarray]:
    """Map each user index to the array of movie indices they interacted with."""
    return {
        int(user): group["movie_idx"].to_numpy(dtype=np.int64)
        for user, group in ratings.groupby("user_idx")
    }


@torch.no_grad()
def recommend_top_k(
    model: LightGCN,
    adjacency: torch.Tensor,
    user_indices: np.ndarray,
    exclude: list[pd.DataFrame],
    k: int = 10,
    batch_size: int = 1024,
) -> np.ndarray:
    """Return the top-k movie indices for each user, skipping already-seen items."""
    model.eval()
    user_embeddings, movie_embeddings = model.get_all_embeddings(adjacency)

    seen = pd.concat(exclude, ignore_index=True) if exclude else None
    seen_items = group_items(seen) if seen is not None else {}

    recommendations = []

    for start in range(0, len(user_indices), batch_size):
        batch_users = user_indices[start:start + batch_size]
        scores = user_embeddings[torch.as_tensor(batch_users, device=user_embeddings.device)]
        scores = scores @ movie_embeddings.T

        rows, cols = [], []
        for row, user in enumerate(batch_users):
            items = seen_items.get(int(user))
            if items is not None:
                rows.append(np.full(len(items), row))
                cols.append(items)

        if rows:
            scores[np.concatenate(rows), np.concatenate(cols)] = -torch.inf

        recommendations.append(torch.topk(scores, k, dim=1).indices.cpu().numpy())

    return np.concatenate(recommendations)


def ranking_metrics(
    recommendations: np.ndarray,
    user_indices: np.ndarray,
    relevant: dict[int, np.ndarray],
    k: int = 10,
) -> dict[str, float]:
    """Mean Recall@k and NDCG@k over users that have relevant items."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    recalls, ndcgs = [], []

    for user, ranked in zip(user_indices, recommendations[:, :k]):
        targets = relevant.get(int(user))
        if targets is None or len(targets) == 0:
            continue

        hits = np.isin(ranked, targets)
        ideal = discounts[:min(len(targets), k)].sum()

        recalls.append(hits.sum() / len(targets))
        ndcgs.append((hits * discounts).sum() / ideal)

    if not recalls:
        return {f"recall@{k}": math.nan, f"ndcg@{k}": math.nan, "num_users": 0}

    return {
        f"recall@{k}": float(np.mean(recalls)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "num_users": len(recalls),
    }


def evaluate(
    model: LightGCN,
    adjacency: torch.Tensor,
    target: pd.DataFrame,
    exclude: list[pd.DataFrame],
    k: int = 10,
) -> dict[str, float]:
    """Evaluate on held-out interactions in `target`, masking items in `exclude`.

    Validation: target=val, exclude=[train].
    Test: target=test, exclude=[train, val].
    """
    relevant = group_items(target)
    user_indices = np.array(sorted(relevant), dtype=np.int64)
    recommendations = recommend_top_k(model, adjacency, user_indices, exclude, k=k)
    return ranking_metrics(recommendations, user_indices, relevant, k=k)
