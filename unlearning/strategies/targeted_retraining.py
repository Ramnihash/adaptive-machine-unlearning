"""Targeted retraining unlearning strategy (CP15/CP16)."""

from dataclasses import asdict
import time
from typing import Any

from data.split import DataSplit, remove_user_from_split
from models.lightgcn import LightGCN
from models.train import train_lightgcn


def targeted_retraining_unlearning(
    model: LightGCN,
    split: DataSplit,
    target_user: int,
    epochs: int | None = None,
    embedding_dim: int | None = None,
    num_layers: int | None = None,
    learning_rate: float = 0.001,
    batch_size: int = 4096,
    regularization: float = 1e-4,
    eval_every: int = 5,
    patience: int = 5,
    k: int = 10,
    seed: int = 42,
) -> tuple[LightGCN, dict[str, Any]]:
    """Unlearn a high-influence user by retraining a fresh LightGCN from remaining interactions.

    This implements full retraining on D_total \\ D_forget, identical to the gold-standard
    retraining protocol (theta*). The original model's learned weights are never reused.
    Validation and test splits are strictly preserved for evaluation.

    Args:
        model: Reference LightGCN model used to inherit embedding dimension and layers if unspecified.
        split: DataSplit containing original interactions.
        target_user: Zero-based user index to unlearn.
        epochs: Number of training epochs (defaults to 100 if None).
        embedding_dim: Model embedding dimension (defaults to model.embedding_dim if None).
        num_layers: Number of LightGCN propagation layers (defaults to model.num_layers if None).
        learning_rate: Adam optimizer learning rate.
        batch_size: Mini-batch size.
        regularization: L2 regularization coefficient.
        eval_every: Evaluation interval in epochs.
        patience: Evaluations without improvement before early stopping (if active).
        k: Top-k ranking cutoff for evaluation.
        seed: Random seed for reproducible initialization and sampling.

    Returns:
        tuple[LightGCN, dict]: Freshly retrained model and retraining metadata.

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

    start_time = time.perf_counter()

    # 1. Remove target user interactions from training split
    split_star = remove_user_from_split(split, target_user)

    dim = embedding_dim if embedding_dim is not None else model.embedding_dim
    layers = num_layers if num_layers is not None else model.num_layers
    n_epochs = epochs if epochs is not None else 100
    k_eval = max(1, min(k, split_star.num_movies))

    # 2. Retrain a fresh model from scratch on remaining interactions
    retrained_model, training_result = train_lightgcn(
        split_star,
        epochs=n_epochs,
        embedding_dim=dim,
        num_layers=layers,
        learning_rate=learning_rate,
        batch_size=batch_size,
        regularization=regularization,
        seed=seed,
        eval_every=eval_every,
        patience=patience,
        k=k_eval,
        select_best=False,
    )

    # train_lightgcn trains on the GPU when one exists; hand back a model on the caller's device
    # so downstream evaluation (built for the input model's device) does not mix devices.
    retrained_model = retrained_model.to(next(model.parameters()).device)

    elapsed = time.perf_counter() - start_time

    metadata = {
        "strategy": "targeted_retraining",
        "target_user": target_user,
        "unlearning_seconds": elapsed,
        "epochs": n_epochs,
        "embedding_dim": dim,
        "num_layers": layers,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "regularization": regularization,
        "seed": seed,
        "train_interactions_remaining": len(split_star.train),
        "forgotten_interactions_count": len(user_rows),
        "training_result": asdict(training_result),
    }

    return retrained_model, metadata
