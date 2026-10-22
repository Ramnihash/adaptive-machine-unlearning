from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from typing import cast
from data.preprocessing import preprocess_movielens
from graph.graph_builder import build_interaction_graph
from models.lightgcn import create_lightgcn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "checkpoints"


def train_lightgcn(
    epochs: int = 10,
    embedding_dim: int = 64,
    num_layers: int = 3,
    learning_rate: float = 0.001,
    batch_size: int = 4096,
    regularization: float = 1e-4,
    seed: int = 42,
) -> tuple[nn.Module, list[float]]:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("Epochs and batch size must be positive.")

    torch.manual_seed(seed)

    data = preprocess_movielens()
    graph = build_interaction_graph(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_lightgcn(
        graph,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
    ).to(device)

    edge_index = graph.edge_index.to(device)
    adjacency = model.normalize_graph(edge_index)

    ratings = data.ratings
    user_indices = torch.tensor(
        ratings["user_idx"].to_numpy(),
        dtype=torch.long,
        device=device,
    )
    movie_indices = torch.tensor(
        ratings["movie_idx"].to_numpy(),
        dtype=torch.long,
        device=device,
    )

    positive_items: dict[int, set[int]] = {}

    for user, group in ratings.groupby("user_idx"):
        user_id = cast(int, user)
        positive_items[user_id] = set(group["movie_idx"].astype(int))

    optimizer = Adam(model.parameters(), lr=learning_rate)
    losses = []
    num_interactions = len(ratings)
    num_users = graph.num_users
    num_movies = graph.num_movies

    for epoch in range(epochs):
        model.train()

        permutation = torch.randperm(num_interactions, device=device)
        epoch_loss = 0.0
        processed = 0

        for start in range(0, num_interactions, batch_size):
            batch = permutation[start:start + batch_size]

            users = user_indices[batch]
            positive = movie_indices[batch]

            negative_values = torch.randint(
                num_movies,
                (len(batch),),
                device=device,
            )

            for attempt in range(10):
                invalid = torch.tensor(
                    [
                        int(item) in positive_items[int(user)]
                        for user, item in zip(
                            users.cpu().tolist(),
                            negative_values.cpu().tolist(),
                        )
                    ],
                    dtype=torch.bool,
                    device=device,
                )

                if not invalid.any():
                    break

                replacements = torch.randint(
                    num_movies,
                    (int(invalid.sum().item()),),
                    device=device,
                )
                negative_values[invalid] = replacements

            model_embeddings = model.propagate(adjacency)

            user_embeddings = model_embeddings[users]
            positive_embeddings = model_embeddings[
                positive + num_users
            ]
            negative_embeddings = model_embeddings[
                negative_values + num_users
            ]

            positive_scores = (
                user_embeddings * positive_embeddings
            ).sum(dim=1)

            negative_scores = (
                user_embeddings * negative_embeddings
            ).sum(dim=1)

            ranking_loss = -torch.nn.functional.logsigmoid(
                positive_scores - negative_scores
            ).mean()

            raw_user_embeddings = model.embedding(users)
            raw_positive_embeddings = model.embedding(
                positive + num_users
            )
            raw_negative_embeddings = model.embedding(
                negative_values + num_users
            )

            reg_loss = (
                raw_user_embeddings.square().sum(dim=1)
                + raw_positive_embeddings.square().sum(dim=1)
                + raw_negative_embeddings.square().sum(dim=1)
            ).mean()

            loss = ranking_loss + regularization * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch)
            processed += len(batch)

        average_loss = epoch_loss / processed
        losses.append(average_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Loss: {average_loss:.6f}"
        )

    return model, losses


def save_model(model: nn.Module, filename: str = "lightgcn.pt") -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / filename
    torch.save(model.state_dict(), path)
    return path


if __name__ == "__main__":
    trained_model, training_losses = train_lightgcn()
    saved_path = save_model(trained_model)
    print(f"Model saved to: {saved_path}")
