"""Single command entry point for experiments (CP7.1).

    python -m experiments.run_experiment --config experiments/configs/default.yaml model.learning_rate=0.01

Stages listed in experiment.stages run in order (train_theta0, train_theta_star, train_bpr). --dry-run only saves config and metadata.
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
