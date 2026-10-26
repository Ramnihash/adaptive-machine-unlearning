"""Approximate localized gradient adaptation unlearning strategy (CP15/CP16)."""

import copy
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from data.split import DataSplit, remove_user_from_split
from models.lightgcn import LightGCN
from models.train import build_adjacency


def is_member_sorted(query: torch.Tensor, sorted_keys: torch.Tensor) -> torch.Tensor:
    """Check membership of query elements in sorted_keys via binary search.

    Equivalent to torch.isin(query, sorted_keys) when sorted_keys is sorted in
    ascending order, but significantly faster on CPU.
    """
    if sorted_keys.numel() == 0 or query.numel() == 0:
        return torch.zeros_like(query, dtype=torch.bool)
    idx = torch.searchsorted(sorted_keys, query)
    return (idx < sorted_keys.numel()) & (
        sorted_keys[idx.clamp(max=sorted_keys.numel() - 1)] == query
    )


def get_affected_neighborhood(
    split: DataSplit,
    target_user: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify the 1-hop items and 2-hop neighbor users connected to target_user.

    Args:
        split: DataSplit containing training interactions.
        target_user: Target user index.

    Returns:
        tuple of (1-hop item indices, 2-hop user indices, full affected node indices).
    """
    train = split.train
    # 1-hop items: movies the target user interacted with
    hop1_items = np.unique(
        train.loc[train["user_idx"] == target_user, "movie_idx"].to_numpy()
    )

    # 2-hop users: users sharing at least one interaction with hop1_items (excluding target)
    if len(hop1_items) > 0:
        neighbor_users = np.unique(
            train.loc[
                train["movie_idx"].isin(hop1_items) & (train["user_idx"] != target_user),
                "user_idx",
            ].to_numpy()
        )
    else:
        neighbor_users = np.empty(0, dtype=np.int64)

    # Combine into node indices for LightGCN embedding table [0..num_users + num_movies - 1]
    affected_nodes = np.unique(
        np.concatenate([
            np.array([target_user], dtype=np.int64),
            neighbor_users,
            hop1_items + split.num_users,
        ])
    )

    return hop1_items, neighbor_users, affected_nodes


def localized_update_unlearning(
    model: LightGCN,
    split: DataSplit,
    target_user: int,
    epochs: int = 5,
    learning_rate: float = 0.001,
    batch_size: int = 1024,
    regularization: float = 1e-4,
    reset_target_embedding: bool = True,
    seed: int = 42,
) -> tuple[LightGCN, dict[str, Any]]:
    """Unlearn a medium-influence user via approximate localized gradient adaptation.

    NOTE: This is an approximate localized gradient adaptation on the affected
    subgraph neighborhood, NOT exact full retraining.

    Workflow:
    1. Identify 1-hop items and 2-hop neighbor users in the interaction graph.
    2. Build the post-removal graph without target user interactions.
    3. Re-initialize the target user's base embedding.
    4. Fine-tune parameters on interactions in the affected neighborhood with gradients
       outside the affected neighborhood strictly zeroed out.

    Args:
        model: Pre-trained LightGCN model (remains unmutated).
        split: DataSplit containing original training interactions.
        target_user: Zero-based user index to unlearn.
        epochs: Number of localized adaptation epochs.
        learning_rate: Optimizer learning rate.
        batch_size: Mini-batch size.
        regularization: L2 regularization coefficient.
        reset_target_embedding: If True, re-initializes the target user's base embedding.
        seed: Random seed for reproducibility.

    Returns:
        tuple[LightGCN, dict]: Unlearned model copy and execution metadata.

    Raises:
        ValueError: If target_user is out of bounds or has no training interactions.
    """
    if not (0 <= target_user < split.num_users):
        raise ValueError(
            f"target_user {target_user} is outside 0..{split.num_users - 1}."
        )

    if model.num_users != split.num_users or model.num_movies != split.num_movies:
        raise ValueError(
            f"Model dimensions (num_users={model.num_users}, num_movies={model.num_movies}) "
            f"do not match DataSplit dimensions (num_users={split.num_users}, num_movies={split.num_movies})."
        )

    user_rows = split.train[split.train["user_idx"] == target_user]
    if user_rows.empty:
        raise ValueError(f"User {target_user} has no training interactions to unlearn.")

    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    start_time = time.perf_counter()
    device = next(model.parameters()).device
    torch.manual_seed(seed)

    # 1. Identify affected neighborhood
    hop1_items, neighbor_users, affected_nodes = get_affected_neighborhood(
        split, target_user
    )

    # 2. Build post-removal graph and adjacency
    split_star = remove_user_from_split(split, target_user)
    unlearned_model = copy.deepcopy(model).to(device)
    unlearned_model.train()
    adjacency_star = build_adjacency(unlearned_model, split_star, device)

    # 3. Re-initialize target user's base embedding
    if reset_target_embedding:
        with torch.no_grad():
            nn.init.normal_(unlearned_model.embedding.weight[target_user], std=0.1)

    # 4. Prepare training interactions involving affected neighborhood
    affected_df = split_star.train[
        split_star.train["user_idx"].isin(neighbor_users)
        | split_star.train["movie_idx"].isin(hop1_items)
    ]
    # Fallback if no neighbor interactions exist
    if affected_df.empty:
        affected_df = split_star.train

    num_users = split.num_users
    num_movies = split.num_movies
    total_nodes = num_users + num_movies

    user_indices = torch.tensor(
        affected_df["user_idx"].to_numpy(), dtype=torch.long, device=device
    )
    movie_indices = torch.tensor(
        affected_df["movie_idx"].to_numpy(), dtype=torch.long, device=device
    )
    num_interactions = len(user_indices)

    # Pre-compute positive keys from ALL retained training interactions for negative rejection
    all_train_users = torch.tensor(
        split_star.train["user_idx"].to_numpy(), dtype=torch.long, device=device
    )
    all_train_movies = torch.tensor(
        split_star.train["movie_idx"].to_numpy(), dtype=torch.long, device=device
    )
    positive_keys = torch.sort(all_train_users * num_movies + all_train_movies).values

    # Gradient mask: True for affected nodes, False for frozen/unaffected nodes
    is_affected = torch.zeros(total_nodes, dtype=torch.bool, device=device)
    is_affected[affected_nodes] = True

    optimizer = Adam(unlearned_model.parameters(), lr=learning_rate)

    if num_interactions > 0:
        for _ in range(epochs):
            permutation = torch.randperm(num_interactions, device=device)
            for start in range(0, num_interactions, batch_size):
                batch = permutation[start:start + batch_size]
                users = user_indices[batch]
                pos = movie_indices[batch]
                neg = torch.randint(num_movies, (len(batch),), device=device)

                for _ in range(10):
                    invalid = is_member_sorted(users * num_movies + neg, positive_keys)
                    if not invalid.any():
                        break
                    neg[invalid] = torch.randint(
                        num_movies, (int(invalid.sum().item()),), device=device
                    )

                # Recompute propagated embeddings against post-removal adjacency
                embeddings = unlearned_model.propagate(adjacency_star)
                u_emb = embeddings[users]
                pos_emb = embeddings[pos + num_users]
                neg_emb = embeddings[neg + num_users]

                pos_scores = (u_emb * pos_emb).sum(dim=1)
                neg_scores = (u_emb * neg_emb).sum(dim=1)

                ranking_loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
                reg_loss = (
                    unlearned_model.embedding(users).square().sum(dim=1)
                    + unlearned_model.embedding(pos + num_users).square().sum(dim=1)
                    + unlearned_model.embedding(neg + num_users).square().sum(dim=1)
                ).mean()

                loss = ranking_loss + regularization * reg_loss

                optimizer.zero_grad()
                loss.backward()

                # Strictly freeze all unaffected embeddings by zeroing their gradients
                with torch.no_grad():
                    if unlearned_model.embedding.weight.grad is not None:
                        unlearned_model.embedding.weight.grad[~is_affected] = 0.0

                optimizer.step()

    unlearned_model.eval()
    elapsed = time.perf_counter() - start_time

    metadata = {
        "strategy": "localized_update",
        "target_user": target_user,
        "unlearning_seconds": elapsed,
        "approximation": "localized_gradient_adaptation",
        "epochs": epochs,
        "learning_rate": learning_rate,
        "reset_target_embedding": reset_target_embedding,
        "hop1_items_count": int(len(hop1_items)),
        "neighbor_users_count": int(len(neighbor_users)),
        "affected_nodes_count": int(len(affected_nodes)),
    }

    return unlearned_model, metadata
