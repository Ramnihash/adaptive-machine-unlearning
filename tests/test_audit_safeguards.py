"""Regression tests for audit safeguards: dataset reuse, forgetting top-k bounds,

reference-model requirements, and model/split compatibility.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from data.split import DataSplit, load_split, save_split
from evaluation.forgetting_metrics import forgotten_item_metrics
from evaluation.harness import evaluate_models
from evaluation.model_io import ModelBundle, load_bundle
from experiments.run_experiment import validate_experiment_config, unlearn_stage
from models.lightgcn import LightGCN
from models.train import build_adjacency, get_split
from unlearning.pipeline import run_adaptive_unlearning


def _make_dummy_split(num_users=4, num_movies=6, seed=42, dataset="synthetic"):
    rows = []
    for u in range(num_users):
        for m in range(min(num_movies, 3)):
            rows.append({"user_idx": u, "movie_idx": m, "rating": 5, "timestamp": 1000, "user_id": u, "movie_id": m})
    df = pd.DataFrame(rows)
    return DataSplit(
        train=df.copy(),
        val=df.copy(),
        test=df.copy(),
        num_users=num_users,
        num_movies=num_movies,
        seed=seed,
        dataset=dataset,
    )


# =========================================================================
# 1. Dataset compatibility and cached split validation
# =========================================================================

def test_cached_synthetic_split_not_silently_reused_for_movielens(tmp_path):
    """Ensure a cached synthetic split is rejected for a MovieLens-1M run."""
    synthetic_split = _make_dummy_split(num_users=10, num_movies=15, seed=42, dataset="synthetic")
    save_split(synthetic_split, output_dir=tmp_path, dataset="synthetic")

    # Calling get_split requesting movielens-1m on this dir must NOT return the synthetic split
    # Since data/ml-1m/ratings.dat does not exist in tmp_path, it must attempt to build and raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="ratings.dat"):
        get_split(
            seed=42,
            processed_dir=tmp_path,
            dataset="movielens-1m",
            raw_dir=tmp_path / "nonexistent_raw",
        )


def test_cached_legacy_split_without_dataset_tag_rejected_if_raw_missing(tmp_path):
    """A legacy split summary without dataset tag cannot be verified if raw data is missing."""
    summary = {
        "seed": 42,
        "num_users": 10,
        "num_movies": 15,
        "train": 30,
        "val": 30,
        "test": 30,
        "fingerprint": "abcd1234efgh5678",
    }
    with open(tmp_path / "split_summary.json", "w") as f:
        json.dump(summary, f)

    # Missing raw dataset: cannot validate legacy split, so it attempts rebuild and fails explicitly
    with pytest.raises(FileNotFoundError, match="ratings.dat"):
        get_split(
            seed=42,
            processed_dir=tmp_path,
            dataset="movielens-1m",
            raw_dir=tmp_path / "missing_raw",
        )


def test_matching_dataset_and_seed_reuses_cached_split(tmp_path):
    """A cached split with matching dataset, seed, and ratios is safely reused."""
    split = _make_dummy_split(num_users=5, num_movies=8, seed=42, dataset="custom_bench")
    save_split(split, output_dir=tmp_path, dataset="custom_bench", val_ratio=0.1, test_ratio=0.1)

    loaded = get_split(
        seed=42,
        processed_dir=tmp_path,
        dataset="custom_bench",
        val_ratio=0.1,
        test_ratio=0.1,
    )
    assert loaded.num_users == 5
    assert loaded.num_movies == 8
    assert loaded.dataset == "custom_bench"


def test_split_ratio_mismatch_forces_rebuild(tmp_path):
    """If cached split was generated with different ratios, it must not be reused."""
    split = _make_dummy_split(num_users=5, num_movies=8, seed=42, dataset="custom_bench")
    save_split(split, output_dir=tmp_path, dataset="custom_bench", val_ratio=0.1, test_ratio=0.1)

    with pytest.raises(ValueError, match="Unsupported dataset for automatic build"):
        get_split(
            seed=42,
            processed_dir=tmp_path,
            dataset="custom_bench",
            val_ratio=0.2,  # Different ratio!
            test_ratio=0.1,
        )


# =========================================================================
# 2. Forgetting metric top-k boundary and small catalogs
# =========================================================================

def test_forgetting_metrics_catalog_smaller_than_k():
    """k >= catalog size should evaluate over available items without crashing."""
    scores = np.array([[10.0, 5.0, 2.0, 1.0, 0.0]])  # 1 user, 5 movies
    # Request k=10 on a 5-item catalog
    metrics = forgotten_item_metrics(scores, user=0, forgotten_items=[0, 1], k=10)
    assert "forgotten_hit_rate@10" in metrics
    # Items 0 and 1 are the top 2 items out of 5; both are in the catalog's top-5
    assert metrics["forgotten_hit_rate@10"] == 2 / 5
    assert metrics["forgotten_mean_percentile"] > 0.7


def test_forgetting_metrics_catalog_size_one():
    """Catalog with single item handles ranking without ZeroDivisionError."""
    scores = np.array([[5.0]])  # 1 user, 1 movie
    metrics = forgotten_item_metrics(scores, user=0, forgotten_items=[0], k=10)
    assert metrics["forgotten_hit_rate@10"] == 1.0
    assert metrics["forgotten_mean_percentile"] == 1.0


def test_forgetting_metrics_empty_forgotten_items():
    """Empty forgotten items returns 0 hit rate and NaN mean percentile safely."""
    scores = np.array([[1.0, 2.0, 3.0, 4.0]])
    metrics = forgotten_item_metrics(scores, user=0, forgotten_items=[], k=2)
    assert metrics["forgotten_hit_rate@2"] == 0.0
    assert np.isnan(metrics["forgotten_mean_percentile"])


def test_forgetting_metrics_invalid_catalog_or_k():
    """Invalid catalog size or non-positive k raises ValueError."""
    scores = np.zeros((1, 0))
    with pytest.raises(ValueError, match="Catalog size"):
        forgotten_item_metrics(scores, user=0, forgotten_items=[0], k=5)

    scores = np.zeros((1, 5))
    with pytest.raises(ValueError, match="k must be positive"):
        forgotten_item_metrics(scores, user=0, forgotten_items=[0], k=0)


def test_forgetting_metrics_normal_catalog():
    """Standard catalog (20 movies, k=5) computes correct hit rate and percentile."""
    scores = np.zeros((1, 20))
    scores[0, [2, 3]] = [10.0, 8.0]
    metrics = forgotten_item_metrics(scores, user=0, forgotten_items=[2, 3], k=5)
    assert metrics["forgotten_hit_rate@5"] == 2 / 5
    assert metrics["forgotten_mean_percentile"] > 0.9


# =========================================================================
# 3. Reference-model explicit requirements
# =========================================================================

def test_validate_experiment_config_missing_theta_star():
    """Configuring comparative evaluate stage without theta_star raises descriptive ValueError."""
    cfg = {
        "experiment": {"stages": ["train_theta0", "evaluate"]},
        "evaluation": {
            "mode": "comparative",
            "checkpoints": {"theta_0": {"path": "checkpoints/theta_0.pt"}},
        },
    }
    with pytest.raises(ValueError, match="evaluation stage requires 'theta_star'"):
        validate_experiment_config(cfg)


def test_validate_experiment_config_theta0_vs_thetau_allows_missing_theta_star():
    """Explicit theta0_vs_thetau evaluation mode does not require theta_star."""
    cfg = {
        "experiment": {"stages": ["train_theta0", "evaluate"]},
        "evaluation": {
            "mode": "theta0_vs_thetau",
            "checkpoints": {"theta_0": {"path": "checkpoints/theta_0.pt"}},
        },
    }
    # Should not raise
    validate_experiment_config(cfg)


def test_validate_experiment_config_missing_theta0():
    """Configuring evaluate stage without theta_0 checkpoint or stage raises descriptive error."""
    cfg = {
        "experiment": {"stages": ["evaluate"]},
        "evaluation": {
            "mode": "theta0_vs_thetau",
            "checkpoints": {},
        },
    }
    with pytest.raises(ValueError, match="evaluation stage requires 'theta_0'"):
        validate_experiment_config(cfg)


def test_evaluate_models_without_theta_star_sets_consistency_none():
    """evaluate_models with reference=None produces consistency_vs_theta_star=None without crashing."""
    rng = np.random.default_rng(0)
    num_users, num_movies = 40, 60
    rows = {"train": [], "val": [], "test": []}
    for user in range(num_users):
        items = rng.permutation(num_movies)[:30]
        for name, chunk in zip(rows, (items[:20], items[20:25], items[25:30])):
            rows[name] += [(user, m) for m in chunk]
    frames = [pd.DataFrame(rows[n], columns=["user_idx", "movie_idx"]) for n in ("train", "val", "test")]
    split = DataSplit(*frames, num_users=num_users, num_movies=num_movies, seed=42)

    model0 = LightGCN(num_users=num_users, num_movies=num_movies, embedding_dim=8, num_layers=1)
    adj0 = build_adjacency(model0, split, torch.device("cpu"))
    bundle0 = ModelBundle("theta_0", model0, adj0, split.train)

    model_u = LightGCN(num_users=num_users, num_movies=num_movies, embedding_dim=8, num_layers=1)
    adj_u = build_adjacency(model_u, split, torch.device("cpu"))
    bundle_u = ModelBundle("unlearned", model_u, adj_u, split.train)

    bundles = {"theta_0": bundle0, "unlearned": bundle_u}

    results = evaluate_models(
        bundles=bundles,
        split=split,
        target_user=0,
        reference=None,  # Explicitly None (no theta_star)
        original="theta_0",
        k=5,
        mia_users=10,
        seed=42,
    )
    assert results["models"]["theta_0"]["consistency_vs_theta_star"] is None
    assert results["models"]["unlearned"]["consistency_vs_theta_star"] is None


# =========================================================================
# 4. Target lists and model/split compatibility
# =========================================================================

def test_unlearn_stage_rejects_multiple_target_users():
    """unlearn_stage must reject multiple target users until batch unlearning is implemented."""
    cfg = {
        "experiment": {"seed": 42, "stages": ["unlearn"]},
        "unlearning": {"target_users": [39, 45, 100]},
    }
    with pytest.raises(ValueError, match="Multiple target users are not yet supported"):
        validate_experiment_config(cfg)


def test_run_adaptive_unlearning_rejects_dimension_mismatch():
    """run_adaptive_unlearning must raise descriptive ValueError if model/split dimensions mismatch."""
    split = _make_dummy_split(num_users=4, num_movies=6)
    # Model created with 5 users instead of 4
    mismatched_model = LightGCN(num_users=5, num_movies=6, embedding_dim=8, num_layers=1)

    config = {
        "experiment": {"seed": 42},
        "unlearning": {"thresholds": {"low": 0.33, "high": 0.66}},
    }
    with pytest.raises(ValueError, match="Model dimensions .* do not match DataSplit dimensions"):
        run_adaptive_unlearning(
            model=mismatched_model,
            split=split,
            target_user=0,
            config=config,
        )


def test_load_bundle_rejects_checkpoint_node_mismatch(tmp_path):
    """load_bundle must raise descriptive ValueError if checkpoint nodes don't match DataSplit."""
    split = _make_dummy_split(num_users=4, num_movies=6)  # 10 total nodes
    # Checkpoint saved with 15 total nodes
    wrong_model = LightGCN(num_users=5, num_movies=10, embedding_dim=8, num_layers=1)
    ckpt_path = tmp_path / "wrong_theta0.pt"
    torch.save(wrong_model.state_dict(), ckpt_path)

    with pytest.raises(ValueError, match="Checkpoint node count .* does not match DataSplit total nodes"):
        load_bundle(
            name="theta_0",
            checkpoint=ckpt_path,
            split=split,
        )
