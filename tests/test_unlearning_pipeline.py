"""Integration tests for adaptive unlearning pipeline and experiment runner (CP19)."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch

from data.split import DataSplit
from experiments.config import load_config
from experiments.run_manager import RunManager
from experiments.run_experiment import unlearn_stage, STAGE_REGISTRY
from models.lightgcn import LightGCN
from unlearning.pipeline import run_adaptive_unlearning
from unlearning.strategies.localized_update import get_affected_neighborhood


@pytest.fixture
def synthetic_experiment():
    torch.manual_seed(42)
    cols = ["user_idx", "movie_idx"]

    # 6 users, 20 movies.
    # User 0: degree 1 -> low tier (percentile ~0.08 <= 0.33)
    # User 3: degree 3 -> low tier (percentile ~0.25 <= 0.33)
    # User 1: degree 5 -> medium tier (percentile ~0.42)
    # User 4: degree 7 -> medium tier (percentile ~0.58)
    # User 5: degree 10 -> high tier (percentile ~0.75)
    # User 2: degree 12 -> high tier (percentile ~0.92 > 0.66)
    train_rows = (
        [(0, 0)]
        + [(3, 5), (3, 6), (3, 7)]
        + [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
        + [(4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14)]
        + [(5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18)]
        + [(2, m) for m in range(12)]
    )

    val_rows = [
        (0, 1), (1, 5), (2, 12), (3, 8), (4, 15), (5, 19)
    ]
    test_rows = [
        (0, 2), (1, 6), (2, 13), (3, 9), (4, 16), (5, 0)
    ]

    split = DataSplit(
        train=pd.DataFrame(train_rows, columns=cols),
        val=pd.DataFrame(val_rows, columns=cols),
        test=pd.DataFrame(test_rows, columns=cols),
        num_users=6,
        num_movies=20,
        seed=42,
    )

    model = LightGCN(
        num_users=6,
        num_movies=20,
        embedding_dim=16,
        num_layers=2,
    )

    config = {
        "experiment": {
            "name": "integration_test",
            "seed": 42,
            "stages": ["unlearn"],
            "results_dir": "experiments/results",
        },
        "model": {
            "name": "lightgcn",
            "embedding_dim": 16,
            "num_layers": 2,
            "learning_rate": 0.01,
            "regularization": 1e-4,
            "batch_size": 16,
            "epochs": 2,
            "eval_every": 1,
            "patience": 2,
        },
        "evaluation": {
            "k": 5,
            "target_user": 0,
            "mia_users": 10,
            "checkpoints": {},
        },
        "unlearning": {
            "target_users": [0],
            "thresholds": {"low": 0.33, "high": 0.66},
            "strategies": {
                "low": "gradient_ascent",
                "medium": "localized_update",
                "high": "targeted_retraining",
            },
            "gradient_ascent": {
                "epochs": 2,
                "learning_rate": 0.01,
                "max_grad_norm": 1.0,
                "batch_size": 8,
            },
            "localized_update": {
                "epochs": 2,
                "learning_rate": 0.01,
                "batch_size": 8,
            },
            "targeted_retraining": {
                "epochs": 2,
                "learning_rate": 0.01,
                "batch_size": 16,
            },
        },
    }

    return model, split, config


def test_pipeline_low_tier_dispatches_gradient_ascent(synthetic_experiment):
    model, split, config = synthetic_experiment
    orig_weight = model.embedding.weight.clone().detach()

    unlearned_model, result = run_adaptive_unlearning(
        model=model,
        split=split,
        target_user=0,
        config=config,
    )

    # 1. Routing verified
    assert result["routing"]["target_user"] == 0
    assert result["routing"]["influence_tier"] == "low"
    assert result["routing"]["selected_strategy"] == "gradient_ascent"

    # 2. Strategy metadata normalized
    assert result["strategy"]["name"] == "gradient_ascent"
    assert result["strategy"]["runtime_seconds"] > 0
    assert result["strategy"]["unlearning_seconds"] == result["strategy"]["runtime_seconds"]
    assert result["strategy"]["parameters"]["epochs"] == 2

    # 3. Caller model preserved (not mutated)
    assert torch.equal(model.embedding.weight, orig_weight)
    assert not torch.equal(unlearned_model.embedding.weight, orig_weight)

    # 4. Evaluation recorded before and after
    assert "utility_retained" in result["evaluation"]["before"]
    assert "utility_retained" in result["evaluation"]["after"]
    assert "forgetting" in result["evaluation"]["before"]
    assert "forgetting" in result["evaluation"]["after"]
    assert result["evaluation"]["k"] == 5


def test_pipeline_medium_tier_dispatches_localized_update(synthetic_experiment):
    model, split, config = synthetic_experiment
    orig_weight = model.embedding.weight.clone().detach()

    unlearned_model, result = run_adaptive_unlearning(
        model=model,
        split=split,
        target_user=1,
        config=config,
    )

    # 1. Routing verified
    assert result["routing"]["target_user"] == 1
    assert result["routing"]["influence_tier"] == "medium"
    assert result["routing"]["selected_strategy"] == "localized_update"

    # 2. Strategy metadata normalized
    assert result["strategy"]["name"] == "localized_update"
    assert result["strategy"]["runtime_seconds"] > 0
    assert result["strategy"]["details"]["approximation"] == "localized_gradient_adaptation"

    # 3. Unaffected nodes are bitwise unchanged
    _, _, affected_nodes = get_affected_neighborhood(split, target_user=1)
    total_nodes = split.num_users + split.num_movies
    unaffected_nodes = np.setdiff1d(np.arange(total_nodes), affected_nodes)
    for node in unaffected_nodes:
        assert torch.equal(unlearned_model.embedding.weight[node], orig_weight[node])

    # 4. Caller model preserved
    assert torch.equal(model.embedding.weight, orig_weight)


def test_pipeline_high_tier_dispatches_targeted_retraining(synthetic_experiment):
    model, split, config = synthetic_experiment
    orig_weight = model.embedding.weight.clone().detach()

    unlearned_model, result = run_adaptive_unlearning(
        model=model,
        split=split,
        target_user=2,
        config=config,
    )

    # 1. Routing verified
    assert result["routing"]["target_user"] == 2
    assert result["routing"]["influence_tier"] == "high"
    assert result["routing"]["selected_strategy"] == "targeted_retraining"

    # 2. Strategy metadata normalized
    assert result["strategy"]["name"] == "targeted_retraining"
    assert result["strategy"]["runtime_seconds"] > 0
    assert "train_interactions_remaining" in result["strategy"]["details"]

    # 3. Fresh model created and original model preserved
    assert unlearned_model is not model
    assert torch.equal(model.embedding.weight, orig_weight)


def test_pipeline_invalid_targets_raise_errors(synthetic_experiment):
    model, split, config = synthetic_experiment

    # Out of bounds
    with pytest.raises(ValueError, match="outside 0..5"):
        run_adaptive_unlearning(model, split, target_user=-1, config=config)

    with pytest.raises(ValueError, match="outside 0..5"):
        run_adaptive_unlearning(model, split, target_user=10, config=config)

    # User with zero interactions
    empty_user_split = DataSplit(
        train=split.train[split.train["user_idx"] != 0],
        val=split.val,
        test=split.test,
        num_users=6,
        num_movies=20,
        seed=42,
    )
    with pytest.raises(ValueError, match="no training interactions to unlearn"):
        run_adaptive_unlearning(model, empty_user_split, target_user=0, config=config)


def test_pipeline_strictly_preserves_val_and_test_data(synthetic_experiment):
    model, split, config = synthetic_experiment

    val_before = split.val.copy(deep=True)
    test_before = split.test.copy(deep=True)

    unlearned_model, result = run_adaptive_unlearning(
        model=model,
        split=split,
        target_user=1,
        config=config,
    )

    pd.testing.assert_frame_equal(split.val, val_before)
    pd.testing.assert_frame_equal(split.test, test_before)


def test_pipeline_custom_thresholds_and_strategy_map(synthetic_experiment):
    model, split, config = synthetic_experiment

    # Override thresholds so user 1 (percentile ~0.42) becomes "low"
    custom_config = dict(config)
    custom_config["unlearning"] = dict(config["unlearning"])
    custom_config["unlearning"]["thresholds"] = {"low": 0.5, "high": 0.8}
    custom_config["unlearning"]["strategies"] = {
        "low": "gradient_ascent",
        "medium": "localized_update",
        "high": "targeted_retraining",
    }

    _, result = run_adaptive_unlearning(
        model=model,
        split=split,
        target_user=1,
        config=custom_config,
    )

    assert result["routing"]["influence_tier"] == "low"
    assert result["routing"]["selected_strategy"] == "gradient_ascent"


def test_pipeline_with_run_manager_saves_checkpoint(synthetic_experiment, tmp_path):
    model, split, config = synthetic_experiment

    with RunManager(config, results_dir=tmp_path) as run:
        unlearned_model, result = run_adaptive_unlearning(
            model=model,
            split=split,
            target_user=0,
            config=config,
            run=run,
        )

        assert result["run_id"] == run.run_id
        checkpoint_file = Path(result["checkpoint"])
        assert checkpoint_file.is_file()
        assert checkpoint_file.name == "theta_u.pt"

        # Verify saved state is loadable
        loaded = torch.load(checkpoint_file, map_location="cpu")
        assert "embedding.weight" in loaded


def test_unlearn_stage_in_run_experiment(synthetic_experiment, tmp_path, monkeypatch):
    model, split, config = synthetic_experiment

    # Monkeypatch get_split in models.train where unlearn_stage imports it from
    monkeypatch.setattr("models.train.get_split", lambda seed, *args, **kwargs: split)

    assert "unlearn" in STAGE_REGISTRY

    with RunManager(config, results_dir=tmp_path) as run:
        # Pre-save theta_0.pt as if train_theta0 had just completed
        theta_0_path = run.checkpoint_path("theta_0.pt")
        torch.save(model.state_dict(), theta_0_path)

        # Run unlearn_stage
        stage_result = unlearn_stage(config, run)

        assert stage_result["routing"]["selected_strategy"] == "gradient_ascent"
        assert stage_result["run_id"] == run.run_id
        assert Path(stage_result["checkpoint"]).is_file()


def test_privacy_status_graceful_handling(synthetic_experiment):
    model, split, config = synthetic_experiment

    # Create a split where target user 0 has no held-out validation or test interactions
    split_no_heldout = DataSplit(
        train=split.train,
        val=split.val[split.val["user_idx"] != 0],
        test=split.test[split.test["user_idx"] != 0],
        num_users=split.num_users,
        num_movies=split.num_movies,
        seed=split.seed,
    )

    unlearned_model, result = run_adaptive_unlearning(
        model=model,
        split=split_no_heldout,
        target_user=0,
        config=config,
    )

    privacy = result["evaluation"]["privacy"]
    assert privacy["status"] == "unsupported_or_insufficient_data"
    assert "no held-out" in privacy["note"].lower()
    assert privacy["mia_interaction"] is None
