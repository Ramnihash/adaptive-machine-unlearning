"""Bounded gradient-ascent unlearning strategy (CP15/CP16)."""

import copy
import time
from typing import Any

import torch
from torch import nn
from torch.optim import Adam

from data.split import DataSplit, remove_user_from_split
from models.lightgcn import LightGCN
from models.train import build_adjacency


def gradient_ascent_unlearning(
    model: LightGCN,
    split: DataSplit,
    target_user: int,
    epochs: int = 5,
    learning_rate: float = 0.001,
    max_grad_norm: float = 1.0,
    regularization: float = 1e-4,
    reset_target_embedding: bool = True,
    batch_size: int = 1024,
    seed: int = 42,
) -> tuple[LightGCN, dict[str, Any]]:
    """Unlearn a low-influence target user using bounded gradient ascent on forgotten interactions.

    Gradient ascent operates in the reverse direction of the BPR ranking loss on
    the target user's training interactions, diminishing the predicted affinity for
    those items. The target user's interactions are removed from the interaction graph
    prior to update steps. Gradient clipping and an embedding regularization penalty
    prevent parameter divergence.

    Note on Target Reset and Forgetting Guarantees:
        When `reset_target_embedding=True`, the target user's base embedding is
        reinitialized after the gradient ascent updates complete, overwriting any
        gradient updates to the target user's base embedding. Importantly, resetting
        the target user's embedding does not by itself prove complete forgetting, and
        the gradient ascent updates to item embeddings (and sampled negative item
        embeddings) can affect recommendations for other retained users under LightGCN
        message passing.

    Args:
        model: Pre-trained LightGCN model (remains unmutated).
        split: DataSplit containing original training interactions.
        target_user: Zero-based user index to unlearn.
        epochs: Number of gradient ascent epochs.
        learning_rate: Gradient ascent learning rate.
        max_grad_norm: Maximum gradient norm threshold for clipping (<=0 disables).
        regularization: L2 regularization coefficient to prevent embedding explosion.
        reset_target_embedding: If True, re-initializes the target user's base embedding
            after the ascent updates complete. Does not by itself guarantee complete
            forgetting, and item-embedding changes can affect other users' recommendations.
        batch_size: Mini-batch size for forgotten interactions.
        seed: Random seed for repeatability.

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

    # Work on an isolated deep copy to preserve the original model
    unlearned_model = copy.deepcopy(model).to(device)
    unlearned_model.train()

    # Build post-removal graph and adjacency
    split_star = remove_user_from_split(split, target_user)
    adjacency_star = build_adjacency(unlearned_model, split_star, device)

    user_positives = torch.tensor(
        user_rows["movie_idx"].to_numpy(), dtype=torch.long, device=device
    )
    num_interactions = len(user_positives)
    num_movies = split.num_movies
    num_users = split.num_users

    optimizer = Adam(unlearned_model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        permutation = torch.randperm(num_interactions, device=device)
        for start in range(0, num_interactions, batch_size):
            batch = permutation[start:start + batch_size]
            pos = user_positives[batch]
            users = torch.full((len(batch),), target_user, dtype=torch.long, device=device)
            neg = torch.randint(num_movies, (len(batch),), device=device)

            model_embeddings = unlearned_model.propagate(adjacency_star)
            u_emb = model_embeddings[users]
            pos_emb = model_embeddings[pos + num_users]
            neg_emb = model_embeddings[neg + num_users]

            pos_scores = (u_emb * pos_emb).sum(dim=1)
            neg_scores = (u_emb * neg_emb).sum(dim=1)

            # Minimizing logsigmoid(pos - neg) is equivalent to ascending the standard BPR loss
            ascent_loss = torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()

            reg_loss = (
                unlearned_model.embedding(users).square().sum(dim=1)
                + unlearned_model.embedding(pos + num_users).square().sum(dim=1)
                + unlearned_model.embedding(neg + num_users).square().sum(dim=1)
            ).mean()

            loss = ascent_loss + regularization * reg_loss

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(unlearned_model.parameters(), max_grad_norm)
            optimizer.step()

    # Configurable target user base embedding re-initialization
    if reset_target_embedding:
        with torch.no_grad():
            nn.init.normal_(unlearned_model.embedding.weight[target_user], std=0.1)

    unlearned_model.eval()
    elapsed = time.perf_counter() - start_time

    metadata = {
        "strategy": "gradient_ascent",
        "target_user": target_user,
        "unlearning_seconds": elapsed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "reset_target_embedding": reset_target_embedding,
        "forgotten_interactions_count": num_interactions,
    }

    return unlearned_model, metadata
