"""Warm-start retraining and the naive fine-tuning baseline (CP6.1)."""

import copy
import time
from typing import Any

import torch
from torch import nn
from torch.optim import Adam

from data.split import DataSplit, remove_user_from_split
from models.lightgcn import LightGCN
from models.train import build_adjacency
from unlearning.strategies.localized_update import is_member_sorted


def warm_start_retraining(
    model: LightGCN,
    split: DataSplit,
    target_user: int,
    epochs: int = 10,
    learning_rate: float = 0.001,
    batch_size: int = 4096,
    regularization: float = 1e-4,
    reset_target_embedding: bool = True,
    seed: int = 42,
) -> tuple[LightGCN, dict[str, Any]]:
    """Continue training a copy of the model on every retained interaction.

    The target user's interactions are removed from the graph and the training set,
    and, when reset_target_embedding is True, the target user's embedding is
    re-initialised. Training then starts from the original weights, so it needs far
    fewer epochs than the from-scratch retrain used for θ*.

    With reset_target_embedding=False this is the naive fine-tuning baseline: the
    target's embedding is never touched by the retained data, so it is not forgotten.
    """
    if not (0 <= target_user < split.num_users):
        raise ValueError(f"target_user {target_user} is outside 0..{split.num_users - 1}.")
    if model.num_users != split.num_users or model.num_movies != split.num_movies:
        raise ValueError("Model dimensions do not match DataSplit dimensions.")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")

    user_rows = split.train[split.train["user_idx"] == target_user]
    if user_rows.empty:
        raise ValueError(f"User {target_user} has no training interactions to unlearn.")

    start_time = time.perf_counter()
    device = next(model.parameters()).device
    torch.manual_seed(seed)

    split_star = remove_user_from_split(split, target_user)
    unlearned_model = copy.deepcopy(model).to(device)
    unlearned_model.train()
    adjacency = build_adjacency(unlearned_model, split_star, device)

    if reset_target_embedding:
        with torch.no_grad():
            nn.init.normal_(unlearned_model.embedding.weight[target_user], std=0.1)

    num_users = split.num_users
    num_movies = split.num_movies
    train = split_star.train
    user_indices = torch.tensor(train["user_idx"].to_numpy(), dtype=torch.long, device=device)
    movie_indices = torch.tensor(train["movie_idx"].to_numpy(), dtype=torch.long, device=device)
    positive_keys = torch.sort(user_indices * num_movies + movie_indices).values
    num_interactions = len(train)

    optimizer = Adam(unlearned_model.parameters(), lr=learning_rate)
    losses: list[float] = []

    for _ in range(epochs):
        permutation = torch.randperm(num_interactions, device=device)
        epoch_loss = 0.0
        for start in range(0, num_interactions, batch_size):
            batch = permutation[start:start + batch_size]
            users = user_indices[batch]
            positive = movie_indices[batch]
            negative = torch.randint(num_movies, (len(batch),), device=device)

            for _ in range(10):
                invalid = is_member_sorted(users * num_movies + negative, positive_keys)
                if not invalid.any():
                    break
                negative[invalid] = torch.randint(
                    num_movies, (int(invalid.sum().item()),), device=device
                )

            embeddings = unlearned_model.propagate(adjacency)
            positive_scores = (embeddings[users] * embeddings[positive + num_users]).sum(dim=1)
            negative_scores = (embeddings[users] * embeddings[negative + num_users]).sum(dim=1)

            ranking_loss = -torch.nn.functional.logsigmoid(positive_scores - negative_scores).mean()
            reg_loss = (
                unlearned_model.embedding(users).square().sum(dim=1)
                + unlearned_model.embedding(positive + num_users).square().sum(dim=1)
                + unlearned_model.embedding(negative + num_users).square().sum(dim=1)
            ).mean()
            loss = ranking_loss + regularization * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        losses.append(epoch_loss / num_interactions)

    unlearned_model.eval()
    elapsed = time.perf_counter() - start_time

    metadata = {
        "strategy": "warm_start_retraining" if reset_target_embedding else "naive_finetune",
        "target_user": target_user,
        "unlearning_seconds": elapsed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "reset_target_embedding": reset_target_embedding,
        "train_interactions_remaining": num_interactions,
        "forgotten_interactions_count": len(user_rows),
        "losses": losses,
    }
    return unlearned_model, metadata
