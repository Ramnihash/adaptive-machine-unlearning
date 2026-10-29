"""Adaptive machine unlearning integration pipeline (CP19).

Coordinates influence routing, strategy dispatch, model unlearning,
and pre/post evaluation across all unlearning strategies.
"""

from dataclasses import asdict
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from data.split import DataSplit, remove_user_from_split
from evaluation.forgetting_metrics import forgotten_item_metrics
from models.evaluate import evaluate
from models.lightgcn import LightGCN
from models.train import build_adjacency
from privacy.mia_evaluation import attack_forgotten_user, fit_attack_on_original
from unlearning.router import RoutingDecision, route_unlearning
from unlearning.strategies.gradient_ascent import gradient_ascent_unlearning
from unlearning.strategies.localized_update import localized_update_unlearning
from unlearning.strategies.targeted_retraining import targeted_retraining_unlearning
from unlearning.strategies.warm_start import warm_start_retraining


def run_adaptive_unlearning(
    model: LightGCN,
    split: DataSplit,
    target_user: int,
    config: dict[str, Any],
    run: Any | None = None,
) -> tuple[LightGCN, dict[str, Any]]:
    """Execute the end-to-end adaptive unlearning pipeline for a target user.

    1. Validates input model, split, and target user.
    2. Calculates target user influence and routes to the appropriate strategy
       using configured thresholds and strategy mappings.
    3. Dispatches to the chosen strategy without mutating the original model.
    4. Evaluates recommendation performance and forgetting before and after unlearning.
    5. Measures interaction-level membership privacy where supported by available data.
    6. Normalizes metadata and saves unlearned model checkpoint if a RunManager is provided.

    Args:
        model: Trained original LightGCN model (remains unmutated).
        split: DataSplit containing training, validation, and test interactions.
        target_user: Zero-based user index to unlearn.
        config: Experiment configuration dictionary.
        run: Optional RunManager instance for run ID and checkpoint saving.

    Returns:
        tuple[LightGCN, dict[str, Any]]: The unlearned model and normalized pipeline metadata.

    Raises:
        TypeError: If model or split types are invalid.
        ValueError: If target_user is invalid, has no training interactions, or strategy is unknown.
    """
    if not isinstance(model, LightGCN):
        raise TypeError(f"model must be an instance of LightGCN, got {type(model).__name__}.")
    if not isinstance(split, DataSplit):
        raise TypeError(f"split must be an instance of DataSplit, got {type(split).__name__}.")
    if not (0 <= target_user < split.num_users):
        raise ValueError(
            f"target_user {target_user} is outside 0..{split.num_users - 1}."
        )

    if model.num_users != split.num_users or model.num_movies != split.num_movies:
        raise ValueError(
            f"Model dimensions (num_users={model.num_users}, num_movies={model.num_movies}) "
            f"do not match DataSplit dimensions (num_users={split.num_users}, num_movies={split.num_movies})."
        )
    expected_nodes = split.num_users + split.num_movies
    if model.embedding.weight.shape[0] != expected_nodes:
        raise ValueError(
            f"Model embedding weight rows ({model.embedding.weight.shape[0]}) does not match "
            f"expected total nodes ({expected_nodes} = {split.num_users} users + {split.num_movies} movies)."
        )

    user_rows = split.train[split.train["user_idx"] == target_user]
    if user_rows.empty:
        raise ValueError(f"User {target_user} has no training interactions to unlearn.")

    # Preserve snapshot of original weights to ensure caller's model is not mutated
    orig_weight_snapshot = model.embedding.weight.clone().detach()

    device = next(model.parameters()).device
    model_cfg = config.get("model", {})
    exp_cfg = config.get("experiment", {})
    unlearning_cfg = config.get("unlearning", {})
    eval_cfg = config.get("evaluation", {})

    seed = int(exp_cfg.get("seed", 42))
    k = int(eval_cfg.get("k", 10))
    k_eval = max(1, min(k, split.num_movies))

    # 1. Route to unlearning strategy using configured thresholds and mappings
    thresholds = unlearning_cfg.get("thresholds")
    strategy_map = unlearning_cfg.get("strategies")
    scoring_method = unlearning_cfg.get("scoring_method", "percentile")

    decision: RoutingDecision = route_unlearning(
        split=split,
        target_user=target_user,
        strategy_map=strategy_map,
        thresholds=thresholds,
        scoring_method=scoring_method,
    )

    selected_strategy = decision.strategy
    strategy_params: dict[str, Any] = {}

    # 2. Dispatch to the selected strategy
    if selected_strategy == "gradient_ascent":
        ga_cfg = unlearning_cfg.get("gradient_ascent", {})
        epochs = int(ga_cfg.get("epochs", 5))
        lr = float(ga_cfg.get("learning_rate", 0.001))
        max_grad_norm = float(ga_cfg.get("max_grad_norm", 1.0))
        reg = float(ga_cfg.get("regularization", model_cfg.get("regularization", 1e-4)))
        reset_emb = bool(ga_cfg.get("reset_target_embedding", True))
        batch_size = int(ga_cfg.get("batch_size", 1024))

        strategy_params = {
            "epochs": epochs,
            "learning_rate": lr,
            "max_grad_norm": max_grad_norm,
            "regularization": reg,
            "reset_target_embedding": reset_emb,
            "batch_size": batch_size,
            "seed": seed,
        }
        unlearned_model, strategy_meta = gradient_ascent_unlearning(
            model=model,
            split=split,
            target_user=target_user,
            epochs=epochs,
            learning_rate=lr,
            max_grad_norm=max_grad_norm,
            regularization=reg,
            reset_target_embedding=reset_emb,
            batch_size=batch_size,
            seed=seed,
        )

    elif selected_strategy == "localized_update":
        loc_cfg = unlearning_cfg.get("localized_update", {})
        epochs = int(loc_cfg.get("epochs", 5))
        lr = float(loc_cfg.get("learning_rate", 0.001))
        reg = float(loc_cfg.get("regularization", model_cfg.get("regularization", 1e-4)))
        reset_emb = bool(loc_cfg.get("reset_target_embedding", True))
        batch_size = int(loc_cfg.get("batch_size", 1024))

        strategy_params = {
            "epochs": epochs,
            "learning_rate": lr,
            "regularization": reg,
            "reset_target_embedding": reset_emb,
            "batch_size": batch_size,
            "seed": seed,
        }
        unlearned_model, strategy_meta = localized_update_unlearning(
            model=model,
            split=split,
            target_user=target_user,
            epochs=epochs,
            learning_rate=lr,
            regularization=reg,
            reset_target_embedding=reset_emb,
            batch_size=batch_size,
            seed=seed,
        )

    elif selected_strategy == "targeted_retraining":
        tr_cfg = unlearning_cfg.get("targeted_retraining", {})
        epochs = int(tr_cfg.get("epochs", model_cfg.get("epochs", 100)))
        embedding_dim = int(model_cfg.get("embedding_dim", model.embedding_dim))
        num_layers = int(model_cfg.get("num_layers", model.num_layers))
        lr = float(tr_cfg.get("learning_rate", model_cfg.get("learning_rate", 0.001)))
        batch_size = int(tr_cfg.get("batch_size", model_cfg.get("batch_size", 4096)))
        reg = float(tr_cfg.get("regularization", model_cfg.get("regularization", 1e-4)))
        eval_every = int(model_cfg.get("eval_every", 5))
        patience = int(model_cfg.get("patience", 5))

        strategy_params = {
            "epochs": epochs,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "learning_rate": lr,
            "batch_size": batch_size,
            "regularization": reg,
            "eval_every": eval_every,
            "patience": patience,
            "k": k_eval,
            "seed": seed,
        }
        unlearned_model, strategy_meta = targeted_retraining_unlearning(
            model=model,
            split=split,
            target_user=target_user,
            epochs=epochs,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            learning_rate=lr,
            batch_size=batch_size,
            regularization=reg,
            eval_every=eval_every,
            patience=patience,
            k=k_eval,
            seed=seed,
        )

    elif selected_strategy in ("warm_start_retraining", "naive_finetune"):
        ws_cfg = unlearning_cfg.get(selected_strategy, {})
        epochs = int(ws_cfg.get("epochs", 10))
        lr = float(ws_cfg.get("learning_rate", model_cfg.get("learning_rate", 0.001)))
        batch_size = int(ws_cfg.get("batch_size", model_cfg.get("batch_size", 4096)))
        reg = float(ws_cfg.get("regularization", model_cfg.get("regularization", 1e-4)))
        reset_emb = bool(ws_cfg.get("reset_target_embedding", selected_strategy == "warm_start_retraining"))

        strategy_params = {
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "regularization": reg,
            "reset_target_embedding": reset_emb,
            "seed": seed,
        }
        unlearned_model, strategy_meta = warm_start_retraining(
            model=model,
            split=split,
            target_user=target_user,
            epochs=epochs,
            learning_rate=lr,
            batch_size=batch_size,
            regularization=reg,
            reset_target_embedding=reset_emb,
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown or unsupported unlearning strategy: {selected_strategy!r}")

    # Verify original model was not mutated
    if not torch.equal(model.embedding.weight, orig_weight_snapshot):
        raise RuntimeError("Original model parameters were modified during unlearning.")

    # 3. Post-removal graph and evaluation
    split_star = remove_user_from_split(split, target_user)
    adj_orig = build_adjacency(model, split, device)
    adj_unlearned = build_adjacency(unlearned_model, split_star, device)

    exclude_orig = [split.train, split.val]
    exclude_unlearned = [split_star.train, split.val]

    target_has_test = bool((split.test["user_idx"] == target_user).any())

    utility_retained_before = evaluate(
        model, adj_orig, split.test, exclude_orig, k=k_eval, skip_users=(target_user,)
    )
    utility_target_before = (
        evaluate(model, adj_orig, split.test, exclude_orig, k=k_eval, only_users=(target_user,))
        if target_has_test
        else {f"recall@{k_eval}": float("nan"), f"ndcg@{k_eval}": float("nan"), "num_users": 0}
    )

    utility_retained_after = evaluate(
        unlearned_model, adj_unlearned, split.test, exclude_unlearned, k=k_eval, skip_users=(target_user,)
    )
    utility_target_after = (
        evaluate(unlearned_model, adj_unlearned, split.test, exclude_unlearned, k=k_eval, only_users=(target_user,))
        if target_has_test
        else {f"recall@{k_eval}": float("nan"), f"ndcg@{k_eval}": float("nan"), "num_users": 0}
    )

    # Full score matrices for forgetting and MIA evaluation
    with torch.no_grad():
        u_emb_orig, m_emb_orig = model.get_all_embeddings(adj_orig)
        scores_orig = (u_emb_orig @ m_emb_orig.T).cpu().numpy()

        u_emb_unl, m_emb_unl = unlearned_model.get_all_embeddings(adj_unlearned)
        scores_unlearned = (u_emb_unl @ m_emb_unl.T).cpu().numpy()

    # Recommendation-level forgetting metrics
    forgotten_items = split.train.loc[
        split.train["user_idx"] == target_user, "movie_idx"
    ].to_numpy()

    forgetting_before = (
        forgotten_item_metrics(scores_orig, target_user, forgotten_items, k=k_eval)
        if split.num_movies > 0
        else None
    )
    forgetting_after = (
        forgotten_item_metrics(scores_unlearned, target_user, forgotten_items, k=k_eval)
        if split.num_movies > 0
        else None
    )

    # Membership inference privacy evaluation
    privacy_metrics: dict[str, Any] = {
        "mia_interaction": None,
        "status": "unsupported_or_insufficient_data",
        "note": None,
    }

    target_held_out = pd.concat([split.val, split.test], ignore_index=True)
    target_held_out_count = int((target_held_out["user_idx"] == target_user).sum())
    mia_users = int(eval_cfg.get("mia_users", 2000))

    if target_held_out_count > 0 and len(forgotten_items) > 0:
        try:
            attack = fit_attack_on_original(
                scores_orig,
                split,
                target_user=target_user,
                num_users=mia_users,
                seed=seed,
            )
            mia_before = attack_forgotten_user(attack["classifier"], scores_orig, split, target_user)
            mia_after = attack_forgotten_user(attack["classifier"], scores_unlearned, split, target_user)
            privacy_metrics = {
                "mia_interaction": {
                    "before": mia_before,
                    "after": mia_after,
                },
                "status": "computed",
                "note": "Interaction-level MIA evaluated against held-out validation/test interactions.",
            }
        except Exception as exc:
            privacy_metrics = {
                "mia_interaction": None,
                "status": "unsupported_or_insufficient_data",
                "note": f"MIA classifier could not be fitted on available candidate split: {exc}",
            }
    else:
        privacy_metrics = {
            "mia_interaction": None,
            "status": "unsupported_or_insufficient_data",
            "note": "Target user has no held-out validation/test interactions for interaction-level MIA.",
        }

    # 4. Save checkpoint if run manager provided
    checkpoint_saved: str | None = None
    if run is not None and hasattr(run, "checkpoint_path"):
        ckpt_path = run.checkpoint_path("theta_u.pt")
        torch.save(unlearned_model.state_dict(), ckpt_path)
        checkpoint_saved = str(ckpt_path)

    elapsed_time = strategy_meta.get("unlearning_seconds")

    # 5. Build normalized result dictionary
    result: dict[str, Any] = {
        "run_id": getattr(run, "run_id", None),
        "target_user": target_user,
        "checkpoint": checkpoint_saved,
        "routing": {
            "target_user": target_user,
            "influence_score": float(decision.influence.score),
            "influence_tier": decision.influence.tier,
            "training_degree": int(decision.influence.degree),
            "selected_strategy": decision.strategy,
            "thresholds": thresholds,
            "scoring_method": scoring_method,
        },
        "strategy": {
            "name": decision.strategy,
            "runtime_seconds": elapsed_time,
            "unlearning_seconds": elapsed_time,
            "parameters": strategy_params,
            "details": strategy_meta,
        },
        "evaluation": {
            "k": k_eval,
            "before": {
                "utility_retained": utility_retained_before,
                "utility_target": utility_target_before,
                "forgetting": forgetting_before,
            },
            "after": {
                "utility_retained": utility_retained_after,
                "utility_target": utility_target_after,
                "forgetting": forgetting_after,
            },
            "privacy": privacy_metrics,
        },
        "reproducibility": {
            "seed": seed,
            "device": str(device),
        },
    }

    return unlearned_model, result


# Canonical aliases
adaptive_unlearning_stage = run_adaptive_unlearning
unlearn = run_adaptive_unlearning
