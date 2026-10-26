"""Train LightGCN on the training split and evaluate on held-out data (CP2.2-CP2.4)."""

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from data.split import DataSplit, load_split, save_split, split_interactions, SPLIT_DIR
from data.preprocessing import preprocess_movielens
from graph.graph_builder import build_graph_from_ratings
from models.evaluate import evaluate
from models.lightgcn import LightGCN


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "checkpoints"


@dataclass
class TrainingResult:
    """Losses, validation history and final held-out metrics of one training run."""

    losses: list[float] = field(default_factory=list)
    val_history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    best_val: dict = field(default_factory=dict)
    test: dict = field(default_factory=dict)
    train_seconds: float = 0.0


def get_split(
    seed: int = 42,
    processed_dir: Path | str | None = None,
    dataset: str = "movielens-1m",
    raw_dir: Path | str | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> DataSplit:
    """Load the saved split, or create and save it if it does not exist yet.

    A cached split is only reused if its seed, dataset identity, and split
    ratios match the requested configuration. If the cached split cannot be
    validated, it is rebuilt from the raw dataset.
    """
    dir_path = Path(processed_dir) if processed_dir else SPLIT_DIR
    summary_path = dir_path / "split_summary.json"

    if summary_path.is_file():
        try:
            with open(summary_path) as file:
                summary = json.load(file)
        except Exception:
            summary = {}

        cached_seed = summary.get("seed")
        cached_dataset = summary.get("dataset")
        cached_val_ratio = summary.get("val_ratio")
        cached_test_ratio = summary.get("test_ratio")

        seed_matches = (cached_seed == seed)
        ratio_matches = (
            (cached_val_ratio is None or abs(cached_val_ratio - val_ratio) < 1e-6)
            and (cached_test_ratio is None or abs(cached_test_ratio - test_ratio) < 1e-6)
        )

        dataset_valid = False
        if cached_dataset is not None:
            dataset_valid = (cached_dataset == dataset)
        else:
            # Legacy summary without explicit dataset identity: validate against raw data if available
            raw_path = Path(raw_dir) if raw_dir else (PROJECT_ROOT / "data" / "ml-1m")
            ratings_file = raw_path / "ratings.dat"
            if dataset == "movielens-1m" and ratings_file.is_file():
                total_cached = summary.get("train", 0) + summary.get("val", 0) + summary.get("test", 0)
                try:
                    with open(ratings_file, "rb") as f:
                        line_count = sum(1 for _ in f)
                    if line_count == total_cached and line_count > 0:
                        dataset_valid = True
                except Exception:
                    dataset_valid = False

        if seed_matches and ratio_matches and dataset_valid:
            return load_split(dir_path)

    raw_path = Path(raw_dir) if raw_dir else (PROJECT_ROOT / "data" / "ml-1m")
    if dataset == "movielens-1m":
        processed = preprocess_movielens(data_dir=raw_path)
    else:
        raise ValueError(
            f"Unsupported dataset for automatic build: {dataset!r}. "
            "Only 'movielens-1m' can be automatically loaded from raw data."
        )

    split = split_interactions(
        processed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        dataset=dataset,
    )
    save_split(split, dir_path, dataset=dataset, val_ratio=val_ratio, test_ratio=test_ratio)
    return split


def build_adjacency(model: LightGCN, split: DataSplit, device: torch.device) -> torch.Tensor:
    """Normalised adjacency built from training interactions only."""
    graph = build_graph_from_ratings(split.train, split.num_users, split.num_movies)
    return model.normalize_graph(graph.edge_index.to(device))


def train_lightgcn(
    split: DataSplit | None = None,
    epochs: int = 100,
    embedding_dim: int = 64,
    num_layers: int = 3,
    learning_rate: float = 0.001,
    batch_size: int = 4096,
    regularization: float = 1e-4,
    seed: int = 42,
    eval_every: int = 5,
    patience: int = 5,
    k: int = 10,
    select_best: bool = True,
) -> tuple[LightGCN, TrainingResult]:
    """Train with BPR loss on split.train, keep the best epoch by validation NDCG@k.

    patience counts evaluations (not epochs) without improvement before stopping.
    select_best=False trains for the full epoch budget and keeps the final weights
    (no early stopping), which is what θ* needs to match a fully trained θ₀.
    """
    if epochs <= 0 or batch_size <= 0 or eval_every <= 0:
        raise ValueError("Epochs, batch size and eval_every must be positive.")

    torch.manual_seed(seed)

    if split is None:
        split = get_split(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_users = split.num_users
    num_movies = split.num_movies

    model = LightGCN(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
    ).to(device)

    adjacency = build_adjacency(model, split, device)

    train = split.train
    user_indices = torch.tensor(train["user_idx"].to_numpy(), dtype=torch.long, device=device)
    movie_indices = torch.tensor(train["movie_idx"].to_numpy(), dtype=torch.long, device=device)

    # Every training (user, movie) pair as one integer key, for fast negative checks.
    positive_keys = torch.sort(user_indices * num_movies + movie_indices).values

    optimizer = Adam(model.parameters(), lr=learning_rate)
    result = TrainingResult()
    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    evaluations_without_improvement = 0
    num_interactions = len(train)
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()

        permutation = torch.randperm(num_interactions, device=device)
        epoch_loss = 0.0

        for start in range(0, num_interactions, batch_size):
            batch = permutation[start:start + batch_size]

            users = user_indices[batch]
            positive = movie_indices[batch]
            negative = torch.randint(num_movies, (len(batch),), device=device)

            # Resample negatives that are actually training positives.
            for _ in range(10):
                invalid = torch.isin(users * num_movies + negative, positive_keys)
                if not invalid.any():
                    break
                negative[invalid] = torch.randint(
                    num_movies, (int(invalid.sum().item()),), device=device
                )

            model_embeddings = model.propagate(adjacency)

            user_embeddings = model_embeddings[users]
            positive_embeddings = model_embeddings[positive + num_users]
            negative_embeddings = model_embeddings[negative + num_users]

            positive_scores = (user_embeddings * positive_embeddings).sum(dim=1)
            negative_scores = (user_embeddings * negative_embeddings).sum(dim=1)

            ranking_loss = -torch.nn.functional.logsigmoid(
                positive_scores - negative_scores
            ).mean()

            reg_loss = (
                model.embedding(users).square().sum(dim=1)
                + model.embedding(positive + num_users).square().sum(dim=1)
                + model.embedding(negative + num_users).square().sum(dim=1)
            ).mean()

            loss = ranking_loss + regularization * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch)

        average_loss = epoch_loss / num_interactions
        result.losses.append(average_loss)
        message = f"Epoch {epoch}/{epochs} - Loss: {average_loss:.6f}"

        if epoch % eval_every == 0 or epoch == epochs:
            val = evaluate(model, adjacency, split.val, exclude=[split.train], k=k)
            result.val_history.append({"epoch": epoch, **val})
            message += f" - Val Recall@{k}: {val[f'recall@{k}']:.4f} NDCG@{k}: {val[f'ndcg@{k}']:.4f}"

            if val[f"ndcg@{k}"] > best_score:
                best_score = val[f"ndcg@{k}"]
                best_state = copy.deepcopy(model.state_dict())
                result.best_epoch = epoch
                result.best_val = val
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1

        print(message, flush=True)

        if select_best and evaluations_without_improvement >= patience:
            print(f"Early stopping: no validation improvement in {patience} evaluations.")
            break

    result.train_seconds = time.perf_counter() - start_time

    if select_best:
        model.load_state_dict(best_state)
    else:
        result.best_epoch = epochs
    result.test = evaluate(model, adjacency, split.test, exclude=[split.train, split.val], k=k)
    print(
        f"Best epoch {result.best_epoch} - Test Recall@{k}: {result.test[f'recall@{k}']:.4f} "
        f"NDCG@{k}: {result.test[f'ndcg@{k}']:.4f}"
    )

    return model, result


def save_model(model: nn.Module, filename: str = "lightgcn.pt") -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / filename
    torch.save(model.state_dict(), path)
    return path


def save_result(result: TrainingResult, filename: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / filename
    with open(path, "w") as file:
        json.dump(asdict(result), file, indent=2)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LightGCN θ₀ on the training split.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="theta_0")
    args = parser.parse_args()

    trained_model, training_result = train_lightgcn(
        epochs=args.epochs, eval_every=args.eval_every, seed=args.seed
    )
    print(f"Model saved to: {save_model(trained_model, f'{args.name}.pt')}")
    print(f"Metrics saved to: {save_result(training_result, f'{args.name}_metrics.json')}")
