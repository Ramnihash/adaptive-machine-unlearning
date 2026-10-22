"""Create reproducible per-user train/validation/test splits (CP1.3)."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.preprocessing import ProcessedData, preprocess_movielens


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass
class DataSplit:
    """Store train/validation/test interactions with model indices."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    num_users: int
    num_movies: int
    seed: int


def split_interactions(
    data: ProcessedData,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> DataSplit:
    """Split each user's interactions randomly into train/val/test.

    Splitting per user keeps every user in the training set, which the
    unlearning experiments need (each target user must have training
    interactions to forget).
    """
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("Split ratios must be between 0 and 1.")

    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be below 1.")

    rng = np.random.default_rng(seed)

    # Sort first so the split never depends on file row order.
    ratings = data.ratings.sort_values(["user_idx", "movie_idx"])
    labels = np.empty(len(ratings), dtype=object)
    position = 0

    for _, group in ratings.groupby("user_idx", sort=True):
        count = len(group)
        num_test = max(1, int(round(count * test_ratio)))
        num_val = max(1, int(round(count * val_ratio)))

        if count - num_test - num_val < 1:
            raise ValueError("A user has too few ratings to split.")

        order = rng.permutation(count)
        user_labels = np.full(count, "train", dtype=object)
        user_labels[order[:num_test]] = "test"
        user_labels[order[num_test:num_test + num_val]] = "val"

        labels[position:position + count] = user_labels
        position += count

    ratings = ratings.assign(split=labels)
    columns = ["user_id", "movie_id", "rating", "timestamp", "user_idx", "movie_idx"]

    def part(name: str) -> pd.DataFrame:
        return ratings.loc[ratings["split"] == name, columns].reset_index(drop=True)

    return DataSplit(
        train=part("train"),
        val=part("val"),
        test=part("test"),
        num_users=data.num_users,
        num_movies=data.num_movies,
        seed=seed,
    )


def remove_user_from_split(split: DataSplit, user_idx: int) -> DataSplit:
    """Return a copy of the split without the target user's training interactions.

    This is D_total - D_forget for the gold-standard retrain θ*. Validation and
    test interactions are left untouched so metrics stay comparable with θ₀.
    """
    if not 0 <= user_idx < split.num_users:
        raise ValueError(f"user_idx {user_idx} is outside 0..{split.num_users - 1}.")

    kept = split.train[split.train["user_idx"] != user_idx].reset_index(drop=True)

    if len(kept) == len(split.train):
        raise ValueError(f"User {user_idx} has no training interactions to remove.")

    return DataSplit(
        train=kept,
        val=split.val,
        test=split.test,
        num_users=split.num_users,
        num_movies=split.num_movies,
        seed=split.seed,
    )


def split_fingerprint(split: DataSplit) -> str:
    """Return a hash that is identical whenever the split is identical."""
    digest = hashlib.sha256()

    for frame in (split.train, split.val, split.test):
        pairs = frame[["user_idx", "movie_idx"]].to_numpy(dtype=np.int64)
        digest.update(pairs.tobytes())

    return digest.hexdigest()[:16]


def save_split(split: DataSplit, output_dir: Path = SPLIT_DIR) -> Path:
    """Save the split as CSV files plus a JSON summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    split.train.to_csv(output_dir / "train.csv", index=False)
    split.val.to_csv(output_dir / "val.csv", index=False)
    split.test.to_csv(output_dir / "test.csv", index=False)

    summary = {
        "seed": split.seed,
        "num_users": split.num_users,
        "num_movies": split.num_movies,
        "train": len(split.train),
        "val": len(split.val),
        "test": len(split.test),
        "fingerprint": split_fingerprint(split),
    }

    with open(output_dir / "split_summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    return output_dir


def load_split(output_dir: Path = SPLIT_DIR) -> DataSplit:
    """Load a split saved by save_split."""
    with open(output_dir / "split_summary.json") as file:
        summary = json.load(file)

    return DataSplit(
        train=pd.read_csv(output_dir / "train.csv"),
        val=pd.read_csv(output_dir / "val.csv"),
        test=pd.read_csv(output_dir / "test.csv"),
        num_users=summary["num_users"],
        num_movies=summary["num_movies"],
        seed=summary["seed"],
    )


if __name__ == "__main__":
    processed = preprocess_movielens()
    first = split_interactions(processed, seed=42)
    second = split_interactions(processed, seed=42)

    print(f"Train: {len(first.train)}  Val: {len(first.val)}  Test: {len(first.test)}")
    print(f"Fingerprint: {split_fingerprint(first)}")
    print(f"Same seed gives same split: {split_fingerprint(first) == split_fingerprint(second)}")

    train_pairs = set(zip(first.train["user_idx"], first.train["movie_idx"]))
    test_pairs = set(zip(first.test["user_idx"], first.test["movie_idx"]))
    val_pairs = set(zip(first.val["user_idx"], first.val["movie_idx"]))
    print(f"No overlap between splits: {not (train_pairs & test_pairs or train_pairs & val_pairs or val_pairs & test_pairs)}")
    print(f"Every user in train: {first.train['user_idx'].nunique() == first.num_users}")

    saved = save_split(first)
    print(f"Saved to: {saved}")
