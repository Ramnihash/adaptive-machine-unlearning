"""Common evaluation harness for θ₀, θ*, baselines and unlearned models θᵤ (CP5.1).

Every model is evaluated under identical conditions: the same split, the same
target user, the same masking and the same attack classifier trained on θ₀.
Results are returned as plain dictionaries following RESULT_SCHEMA.
"""

import json
from pathlib import Path

import pandas as pd

from data.split import DataSplit
from evaluation.consistency_metrics import consistency_metrics
from evaluation.forgetting_metrics import forgotten_item_metrics
from evaluation.model_io import ModelBundle, score_matrix
from models.evaluate import evaluate
from privacy.mia_evaluation import attack_forgotten_user, fit_attack_on_original
from privacy.mia_user_level import (
    build_user_attack_data, fit_user_attack, user_membership_probability, verify_user_attack_data,
)
import numpy as np

RESULT_SCHEMA = {
    "model": "name of the evaluated model, e.g. theta_0, theta_star, bpr, unlearned",
    "run_id": "experiment run that produced the checkpoint, or null",
    "runtime_seconds": "training / retraining / unlearning time in seconds, or null",
    "utility_retained": "recall@k, ndcg@k on held-out test items of all users except the target",
    "utility_target": "recall@k, ndcg@k of the forgotten user on their own test items",
    "forgetting": "how strongly the forgotten user's removed items are still recommended",
    "consistency_vs_theta_star": "top-k Jaccard and score correlation against θ*, or null for θ* itself",
    "privacy_mia_user": "user-level shadow-model attack: probability that the forgotten user was in the training data",
    "privacy_mia_interaction": "interaction-level attack on the forgotten user's removed items vs held-out items (weak on LightGCN, kept for completeness)",
}


def read_run_info(checkpoint: Path | str) -> tuple[str | None, float | None]:
    """Run ID and measured training time stored next to a checkpoint by the experiment runner."""
    run_dir = Path(checkpoint).parent.parent
    run_id, seconds = None, None
    try:
        run_id = json.loads((run_dir / "metadata.json").read_text())["run_id"]
        stages = json.loads((run_dir / "results.json").read_text())["stages"]
        stage = next(iter(stages.values()))
        seconds = stage.get("retrain_seconds", stage.get("train_seconds"))
    except (OSError, KeyError, StopIteration, json.JSONDecodeError):
        pass
    return run_id, seconds


def evaluate_models(
    bundles: dict[str, ModelBundle],
    split: DataSplit,
    target_user: int,
    reference: str = "theta_star",
    original: str = "theta_0",
    k: int = 10,
    mia_users: int = 2000,
    seed: int = 42,
    shadow: ModelBundle | None = None,
    shadow_removed_users: list[int] | None = None,
) -> dict:
    """Evaluate every bundle.

    The interaction-level attack is fitted once on θ₀. The user-level attack is
    fitted on the shadow model (if given) and applied to every model.
    """
    scores = {name: score_matrix(bundle) for name, bundle in bundles.items()}
    forgotten_items = split.train.loc[split.train["user_idx"] == target_user, "movie_idx"].to_numpy()

    attack = fit_attack_on_original(scores[original], split, target_user, num_users=mia_users, seed=seed)

    user_attack = None
    if shadow is not None:
        shadow_scores = score_matrix(shadow)
        data = build_user_attack_data(
            shadow_scores, split.train, np.array(shadow_removed_users), exclude_users=(target_user,), seed=seed
        )
        user_attack = {
            "verification": verify_user_attack_data(data, exclude_users=(target_user,)),
            **fit_user_attack(data),
        }
        rng = np.random.default_rng(seed)
        reference_users = rng.choice(np.setdiff1d(np.arange(split.num_users), [target_user]), 500, replace=False)

    # θ*'s training data plus validation data: the same masking for every model.
    seen = pd.concat([bundles[reference].train, split.val], ignore_index=True)

    results = {}
    for name, bundle in bundles.items():
        exclude = [bundle.train, split.val]
        results[name] = {
            "model": name,
            "runtime_seconds": bundle.runtime_seconds,
            "utility_retained": evaluate(
                bundle.model, bundle.adjacency, split.test, exclude, k=k, skip_users=(target_user,)
            ),
            "utility_target": evaluate(
                bundle.model, bundle.adjacency, split.test, exclude, k=k, only_users=(target_user,)
            ),
            "forgetting": forgotten_item_metrics(scores[name], target_user, forgotten_items, k=k),
            "consistency_vs_theta_star": (
                None if name == reference
                else consistency_metrics(scores[name], scores[reference], seen, target_user, k=k)
            ),
            "privacy_mia_interaction": attack_forgotten_user(attack["classifier"], scores[name], split, target_user),
        }
        if user_attack is not None:
            clf = user_attack["classifier"]
            target_p = user_membership_probability(clf, scores[name], split.train, np.array([target_user]))[0]
            others_p = user_membership_probability(clf, scores[name], split.train, reference_users)
            results[name]["privacy_mia_user"] = {
                "target_membership_probability": float(target_p),
                "retained_users_mean_probability": float(others_p.mean()),
                "target_percentile_among_retained": float((others_p < target_p).mean()),
            }

    return {
        "target_user": target_user,
        "k": k,
        "attack": {
            "validation": attack["val"],
            "test": attack["test"],
            "verification": attack["verification"],
            "config": attack["config"],
        },
        "user_attack": None if user_attack is None else {
            "validation": user_attack["metrics"]["val"],
            "test": user_attack["metrics"]["test"],
            "verification": user_attack["verification"],
            "config": user_attack["config"],
        },
        "models": results,
        "schema": RESULT_SCHEMA,
    }
