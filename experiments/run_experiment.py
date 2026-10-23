"""Single command entry point for experiments (CP7.1).

    python -m experiments.run_experiment --config experiments/configs/default.yaml model.learning_rate=0.01

Stages listed in experiment.stages run in order (train_theta0, train_theta_star, train_bpr, train_shadow, evaluate). --dry-run only saves config and metadata.
"""
import argparse

from experiments.config import load_config
from experiments.run_manager import RunManager



def train_theta0(config, run):
    """Train the original LightGCN model θ₀ on the training split and save it in the run folder."""
    from models.train import get_split, train_lightgcn
    import torch
    from dataclasses import asdict

    model_cfg = config["model"]
    if model_cfg["name"] != "lightgcn":
        raise ValueError(f"Model {model_cfg['name']!r} is not implemented yet.")

    split = get_split(config["experiment"]["seed"])
    model, result = train_lightgcn(
        split,
        epochs=model_cfg["epochs"],
        embedding_dim=model_cfg["embedding_dim"],
        num_layers=model_cfg["num_layers"],
        learning_rate=model_cfg["learning_rate"],
        batch_size=model_cfg["batch_size"],
        regularization=model_cfg["regularization"],
        seed=config["experiment"]["seed"],
        eval_every=model_cfg["eval_every"],
        patience=model_cfg["patience"],
        k=config["evaluation"]["k"],
    )
    torch.save(model.state_dict(), run.checkpoint_path("theta_0.pt"))
    return asdict(result)


def train_bpr(config, run):
    """BPR baseline: matrix factorisation trained with the BPR loss on the same split.

    This is LightGCN with zero propagation layers (final embedding = raw embedding),
    so BPR and LightGCN share the loss, sampler, evaluation and hyperparameters and
    differ only in graph propagation.
    """
    from dataclasses import asdict
    import torch
    from models.train import get_split, train_lightgcn

    model_cfg = config["model"]
    seed = config["experiment"]["seed"]
    split = get_split(seed)
    model, result = train_lightgcn(
        split,
        epochs=model_cfg["epochs"],
        embedding_dim=model_cfg["embedding_dim"],
        num_layers=0,
        learning_rate=model_cfg["learning_rate"],
        batch_size=model_cfg["batch_size"],
        regularization=model_cfg["regularization"],
        seed=seed,
        eval_every=model_cfg["eval_every"],
        patience=model_cfg["patience"],
        k=config["evaluation"]["k"],
    )
    torch.save(model.state_dict(), run.checkpoint_path("bpr.pt"))
    return {"model": "bpr (LightGCN with num_layers=0)", **asdict(result)}


def _forgotten_users(name, entry, target):
    """Users a checkpoint was trained without: the target for θ*/θᵤ, the removed set for the shadow."""
    if name == "shadow":
        return _shadow_removed_users(entry["path"])
    return [target] if entry.get("forgotten", False) else None


def _shadow_removed_users(checkpoint):
    import json
    from pathlib import Path
    results = json.loads((Path(checkpoint).parent.parent / "results.json").read_text())
    return results["stages"]["train_shadow"]["removed_users"]


def train_shadow(config, run):
    """Shadow model for the user-level membership attack: θ₀'s recipe on D_total minus random users."""
    import numpy as np
    import torch
    from data.split import remove_users_from_split
    from models.train import get_split, train_lightgcn

    model_cfg = config["model"]
    shadow_cfg = config["evaluation"]["shadow"]
    seed = config["experiment"]["seed"]
    target = int(config["evaluation"]["target_user"])

    split = get_split(seed)
    rng = np.random.default_rng(shadow_cfg["seed"])
    candidates = np.setdiff1d(np.arange(split.num_users), [target])
    removed = np.sort(rng.choice(candidates, size=int(shadow_cfg["fraction"] * split.num_users), replace=False))

    model, result = train_lightgcn(
        remove_users_from_split(split, removed),
        epochs=model_cfg["epochs"],
        embedding_dim=model_cfg["embedding_dim"],
        num_layers=model_cfg["num_layers"],
        learning_rate=model_cfg["learning_rate"],
        batch_size=model_cfg["batch_size"],
        regularization=model_cfg["regularization"],
        seed=seed + 1,
        eval_every=model_cfg["eval_every"],
        patience=model_cfg["patience"],
        k=config["evaluation"]["k"],
        select_best=False,
    )
    torch.save(model.state_dict(), run.checkpoint_path("shadow.pt"))
    return {"removed_users": [int(u) for u in removed], "train_seconds": result.train_seconds,
            "test": result.test, "val_history": result.val_history, "losses": result.losses}


def evaluate_models_stage(config, run):
    """Evaluate saved models under identical conditions and save the comparison."""
    from evaluation.harness import evaluate_models, read_run_info
    from evaluation.model_io import load_bundle
    from models.train import get_split

    eval_cfg = config["evaluation"]
    target = int(eval_cfg["target_user"])
    split = get_split(config["experiment"]["seed"])

    bundles, run_ids = {}, {}
    for name, entry in eval_cfg["checkpoints"].items():
        if not entry.get("path"):
            continue
        run_id, seconds = read_run_info(entry["path"])
        run_ids[name] = run_id
        bundles[name] = load_bundle(
            name,
            entry["path"],
            split,
            forgotten_users=_forgotten_users(name, entry, target),
            num_layers=entry["num_layers"],
            runtime_seconds=seconds,
        )
    for needed in ("theta_0", "theta_star"):
        if needed not in bundles:
            raise ValueError(f"evaluation.checkpoints.{needed}.path must be set.")

    shadow, removed = bundles.pop("shadow", None), None
    if shadow is not None:
        removed = _shadow_removed_users(eval_cfg["checkpoints"]["shadow"]["path"])

    results = evaluate_models(
        bundles, split, target, k=eval_cfg["k"], mia_users=eval_cfg["mia_users"],
        seed=config["experiment"]["seed"], shadow=shadow, shadow_removed_users=removed,
    )
    for name, run_id in run_ids.items():
        if name in results["models"]:
            results["models"][name]["run_id"] = run_id
    results["shadow_run_id"] = run_ids.get("shadow")
    return results


def train_theta_star(config, run):
    """Gold standard θ*: retrain from scratch after removing each target user's training data."""
    import time
    import torch
    from data.split import remove_user_from_split
    from models.evaluate import evaluate
    from models.train import build_adjacency, get_split, train_lightgcn

    model_cfg = config["model"]
    targets = config["unlearning"]["target_users"]
    if not targets:
        raise ValueError("Set unlearning.target_users, e.g. 'unlearning.target_users=[39]'.")
    if len(targets) != 1:
        raise ValueError("One target user per θ* run for now.")
    target = int(targets[0])
    k = config["evaluation"]["k"]
    seed = config["experiment"]["seed"]

    split = get_split(seed)
    forgotten = int((split.train["user_idx"] == target).sum())
    split_star = remove_user_from_split(split, target)

    model, result = train_lightgcn(
        split_star,
        epochs=model_cfg["epochs"],
        embedding_dim=model_cfg["embedding_dim"],
        num_layers=model_cfg["num_layers"],
        learning_rate=model_cfg["learning_rate"],
        batch_size=model_cfg["batch_size"],
        regularization=model_cfg["regularization"],
        seed=seed,
        eval_every=model_cfg["eval_every"],
        patience=model_cfg["patience"],
        k=k,
        select_best=False,
    )

    device = next(model.parameters()).device
    adjacency = build_adjacency(model, split_star, device)
    exclude = [split_star.train, split_star.val]
    others = evaluate(model, adjacency, split.test, exclude, k=k, skip_users=(target,))
    forgotten_user = evaluate(model, adjacency, split.test, exclude, k=k, only_users=(target,))

    torch.save(model.state_dict(), run.checkpoint_path("theta_star.pt"))
    return {
        "target_user": target,
        "forgotten_train_interactions": forgotten,
        "train_interactions_remaining": len(split_star.train),
        "retrain_seconds": result.train_seconds,
        "test_all_users": result.test,
        "test_other_users": others,
        "test_target_user": forgotten_user,
        "val_history": result.val_history,
        "losses": result.losses,
    }


# Registry of pipeline stages; experiment.stages in the config picks which run, in order.
# Hemanth's unlearning and Rahul's evaluation stages are added here during integration.
STAGE_REGISTRY = {
    "train_theta0": train_theta0,
    "train_theta_star": train_theta_star,
    "train_bpr": train_bpr,
    "train_shadow": train_shadow,
    "evaluate": evaluate_models_stage,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Experiment YAML merged over configs/default.yaml")
    parser.add_argument("--results-dir", help="Override experiment.results_dir")
    parser.add_argument("--dry-run", action="store_true", help="Only save config and metadata")
    parser.add_argument("overrides", nargs="*", help="section.key=value overrides")
    args = parser.parse_args(argv)

    config = load_config(args.config, args.overrides)
    with RunManager(config, results_dir=args.results_dir) as run:
        results = {"stages": {}}
        for name in [] if args.dry_run else config["experiment"]["stages"]:
            if name not in STAGE_REGISTRY:
                raise ValueError(f"Unknown stage {name!r}. Available: {sorted(STAGE_REGISTRY)}")
            results["stages"][name] = STAGE_REGISTRY[name](config, run)
        if args.dry_run:
            results["note"] = "dry run: pipeline stages skipped"
        run.save_results(results)
    print(f"Run {run.run_id} {run.metadata['status']}: {run.run_dir}")
    return run


if __name__ == "__main__":
    main()
