"""Load saved models for evaluation and compute full user x movie score matrices."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.split import DataSplit, remove_users_from_split
from models.lightgcn import LightGCN
from models.train import build_adjacency


@dataclass
class ModelBundle:
    """A trained model plus everything needed to evaluate it consistently.

    train is the interactions the model was trained on (D_total for θ₀,
    D_total - D_forget for θ*), which decides what is masked when recommending.
    """

    name: str
    model: LightGCN
    adjacency: torch.Tensor
    train: pd.DataFrame
    runtime_seconds: float | None = None


def load_bundle(
    name: str,
    checkpoint: Path | str,
    split: DataSplit,
    forgotten_users: list[int] | None = None,
    num_layers: int = 3,
    runtime_seconds: float | None = None,
) -> ModelBundle:
    """Load a checkpoint saved by the experiment runner.

    forgotten_users lists users the model was trained without (θ*, θᵤ, shadow), so
    the graph is rebuilt from the same reduced training data.
    """
    state = torch.load(checkpoint, map_location="cpu")
    expected_nodes = split.num_users + split.num_movies
    checkpoint_nodes = state["embedding.weight"].shape[0]
    if checkpoint_nodes != expected_nodes:
        raise ValueError(
            f"Checkpoint node count ({checkpoint_nodes}) does not match DataSplit total nodes "
            f"({expected_nodes} = {split.num_users} users + {split.num_movies} movies) for {name!r}."
        )
    embedding_dim = state["embedding.weight"].shape[1]

    model = LightGCN(
        num_users=split.num_users,
        num_movies=split.num_movies,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
    )
    model.load_state_dict(state)
    model.eval()

    train_split = split if not forgotten_users else remove_users_from_split(split, forgotten_users)
    adjacency = build_adjacency(model, train_split, torch.device("cpu"))

    return ModelBundle(name, model, adjacency, train_split.train, runtime_seconds)


@torch.no_grad()
def score_matrix(bundle: ModelBundle) -> np.ndarray:
    """Return the (num_users, num_movies) matrix of predicted scores."""
    users, movies = bundle.model.get_all_embeddings(bundle.adjacency)
    return (users @ movies.T).cpu().numpy()
