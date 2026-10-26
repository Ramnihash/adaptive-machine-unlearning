"""Single command entry point for experiments (CP7.1).

    python -m experiments.run_experiment --config experiments/configs/default.yaml model.learning_rate=0.01

Stages listed in experiment.stages run in order (train_theta0, train_theta_star, train_bpr, train_shadow, evaluate). --dry-run only saves config and metadata.
"""
import argparse

from experiments.config import load_config
from experiments.run_manager import RunManager



def _resolve_split(config):
    """Load split respecting config['experiment']['seed'] and data configuration."""
    from models.train import get_split
    seed = config["experiment"]["seed"]
    data_cfg = config.get("data", {})
    return get_split(
        seed=seed,
        processed_dir=data_cfg.get("processed_dir"),
        dataset=data_cfg.get("dataset", "movielens-1m"),
        raw_dir=data_cfg.get("raw_dir"),
        val_ratio=data_cfg.get("val_ratio", 0.1),
        test_ratio=data_cfg.get("test_ratio", 0.1),
    )


def train_theta0(config, run):
    """Train the original LightGCN model θ₀ on the training split and save it in the run folder."""
    from models.train import train_lightgcn
    import torch
    from dataclasses import asdict

    model_cfg = config["model"]
    if model_cfg["name"] != "lightgcn":
        raise ValueError(f"Model {model_cfg['name']!r} is not implemented yet.")

    split = _resolve_split(config)
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
    split = _resolve_split(config)
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

    split = _resolve_split(config)
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
    split = _resolve_split(config)
    eval_mode = eval_cfg.get("mode", "comparative")
    require_theta_star = eval_cfg.get("require_theta_star", eval_mode != "theta0_vs_thetau")

    checkpoints_cfg = eval_cfg.setdefault("checkpoints", {})

    # Auto-resolve theta_0 from current run if not explicitly configured
    if not checkpoints_cfg.get("theta_0", {}).get("path"):
        run_theta_0 = run.checkpoint_path("theta_0.pt")
        if run_theta_0.is_file():
            checkpoints_cfg.setdefault("theta_0", {})["path"] = str(run_theta_0)
            checkpoints_cfg["theta_0"].setdefault("num_layers", config.get("model", {}).get("num_layers", 3))

    # Auto-resolve theta_star from current run if not explicitly configured
    if not checkpoints_cfg.get("theta_star", {}).get("path"):
        run_theta_star = run.checkpoint_path("theta_star.pt")
        if run_theta_star.is_file():
            checkpoints_cfg.setdefault("theta_star", {})["path"] = str(run_theta_star)
            checkpoints_cfg["theta_star"].setdefault("num_layers", config.get("model", {}).get("num_layers", 3))
            checkpoints_cfg["theta_star"].setdefault("forgotten", True)

    # Auto-resolve unlearned/theta_u from current run if not configured
    if "unlearned" not in checkpoints_cfg:
        run_theta_u = run.checkpoint_path("theta_u.pt")
        if run_theta_u.is_file():
            checkpoints_cfg["unlearned"] = {
                "path": str(run_theta_u),
                "num_layers": config.get("model", {}).get("num_layers", 3),
                "forgotten": True,
            }

    bundles, run_ids = {}, {}
    for name, entry in checkpoints_cfg.items():
        if not entry.get("path"):
            continue
        run_id, seconds = read_run_info(entry["path"])
        run_ids[name] = run_id
        bundles[name] = load_bundle(
            name,
            entry["path"],
            split,
            forgotten_users=_forgotten_users(name, entry, target),
            num_layers=entry.get("num_layers", 3),
            runtime_seconds=seconds,
        )

    if "theta_0" not in bundles:
        raise ValueError(
            "evaluation stage requires 'theta_0', but evaluation.checkpoints.theta_0.path is not set. "
            "Ensure 'train_theta0' precedes 'evaluate' in experiment.stages or set evaluation.checkpoints.theta_0.path."
        )

    if require_theta_star and "theta_star" not in bundles:
        raise ValueError(
            "evaluation stage requires 'theta_star', but evaluation.checkpoints.theta_star.path is not set. "
            "Ensure 'train_theta_star' precedes 'evaluate' in experiment.stages, set evaluation.checkpoints.theta_star.path, "
            "or configure evaluation.mode='theta0_vs_thetau' / evaluation.require_theta_star=false for two-model evaluation."
        )

    shadow, removed = bundles.pop("shadow", None), None
    if shadow is not None:
        removed = _shadow_removed_users(checkpoints_cfg["shadow"]["path"])

    reference = "theta_star" if "theta_star" in bundles else None
    results = evaluate_models(
        bundles, split, target, reference=reference, k=eval_cfg["k"],
        mia_users=eval_cfg["mia_users"], seed=config["experiment"]["seed"],
        shadow=shadow, shadow_removed_users=removed,
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

    split = _resolve_split(config)
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


def unlearn_stage(config, run):
    """Run adaptive unlearning for the target user, save theta_u.pt, and return normalized results."""
    from pathlib import Path
    import torch
    from models.lightgcn import LightGCN
    from models.train import get_split, MODEL_DIR
    from unlearning.pipeline import run_adaptive_unlearning

    unlearning_cfg = config.get("unlearning", {})
    targets = unlearning_cfg.get("target_users", [])
    if isinstance(targets, (list, tuple)):
        if len(targets) > 1:
            raise ValueError(
                f"unlearning.target_users has {len(targets)} users: {targets}. "
                "Multiple target users are not yet supported in a single unlearning run; "
                "specify exactly one target user (e.g. unlearning.target_users=[39])."
            )
        elif len(targets) == 1:
            target_user = int(targets[0])
        elif "target_user" in config.get("evaluation", {}):
            target_user = int(config["evaluation"]["target_user"])
        else:
            raise ValueError(
                "Specify target user in unlearning.target_users (e.g. unlearning.target_users=[39]) "
                "or evaluation.target_user."
            )
    elif targets is not None:
        target_user = int(targets)
    elif "target_user" in config.get("evaluation", {}):
        target_user = int(config["evaluation"]["target_user"])
    else:
        raise ValueError(
            "Specify target user in unlearning.target_users (e.g. unlearning.target_users=[39]) "
            "or evaluation.target_user."
        )

    seed = config["experiment"]["seed"]
    split = _resolve_split(config)

    # Resolve theta_0 checkpoint from current run or evaluation checkpoints
    theta_0_path = run.checkpoint_path("theta_0.pt")
    state = None
    if theta_0_path.is_file():
        state = torch.load(theta_0_path, map_location="cpu")
    else:
        configured_path = config.get("evaluation", {}).get("checkpoints", {}).get("theta_0", {}).get("path")
        if configured_path and Path(configured_path).is_file():
            state = torch.load(configured_path, map_location="cpu")
        elif (MODEL_DIR / "theta_0.pt").is_file():
            state = torch.load(MODEL_DIR / "theta_0.pt", map_location="cpu")

    if state is None:
        raise FileNotFoundError(
            f"Original model checkpoint (theta_0.pt) not found in run {run.run_id} or evaluation.checkpoints. "
            "Ensure 'train_theta0' precedes 'unlearn' in experiment.stages or specify evaluation.checkpoints.theta_0.path."
        )

    expected_nodes = split.num_users + split.num_movies
    checkpoint_nodes = state["embedding.weight"].shape[0]
    if checkpoint_nodes != expected_nodes:
        raise ValueError(
            f"Checkpoint node count ({checkpoint_nodes}) does not match DataSplit total nodes "
            f"({expected_nodes} = {split.num_users} users + {split.num_movies} movies). "
            "Ensure the checkpoint was trained on the same dataset split."
        )

    model_cfg = config.get("model", {})
    embedding_dim = state["embedding.weight"].shape[1]
    num_layers = int(model_cfg.get("num_layers", 3))

    model = LightGCN(
        num_users=split.num_users,
        num_movies=split.num_movies,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
    )
    model.load_state_dict(state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    unlearned_model, result = run_adaptive_unlearning(
        model=model,
        split=split,
        target_user=target_user,
        config=config,
        run=run,
    )

    return result


def validate_experiment_config(config):
    """Validate pipeline stages, targets, and checkpoint requirements before execution."""
    stages = config.get("experiment", {}).get("stages", [])
    for name in stages:
        if name not in STAGE_REGISTRY:
            raise ValueError(f"Unknown stage {name!r}. Available: {sorted(STAGE_REGISTRY)}")

    unlearning_cfg = config.get("unlearning", {})
    targets = unlearning_cfg.get("target_users", [])
    if isinstance(targets, (list, tuple)) and len(targets) > 1:
        raise ValueError(
            f"unlearning.target_users has {len(targets)} users: {targets}. "
            "Multiple target users are not yet supported in a single unlearning run; "
            "specify exactly one target user (e.g. unlearning.target_users=[39])."
        )

    if "evaluate" in stages:
        eval_cfg = config.get("evaluation", {})
        eval_mode = eval_cfg.get("mode", "comparative")
        require_theta_star = eval_cfg.get("require_theta_star", eval_mode != "theta0_vs_thetau")

        has_theta0 = (
            bool(eval_cfg.get("checkpoints", {}).get("theta_0", {}).get("path"))
            or ("train_theta0" in stages)
        )
        if not has_theta0:
            raise ValueError(
                "evaluation stage requires 'theta_0', but evaluation.checkpoints.theta_0.path is not set "
                "and 'train_theta0' is not in experiment.stages."
            )

        if require_theta_star:
            has_theta_star = (
                bool(eval_cfg.get("checkpoints", {}).get("theta_star", {}).get("path"))
                or ("train_theta_star" in stages)
            )
            if not has_theta_star:
                raise ValueError(
                    "evaluation stage requires 'theta_star', but evaluation.checkpoints.theta_star.path is not set "
                    "and 'train_theta_star' is not in experiment.stages. Set evaluation.checkpoints.theta_star.path, "
                    "add 'train_theta_star' to stages, or set evaluation.mode='theta0_vs_thetau'."
                )


# Registry of pipeline stages; experiment.stages in the config picks which run, in order.
STAGE_REGISTRY = {
    "train_theta0": train_theta0,
    "train_theta_star": train_theta_star,
    "train_bpr": train_bpr,
    "train_shadow": train_shadow,
    "unlearn": unlearn_stage,
    "adaptive_unlearn": unlearn_stage,
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
    validate_experiment_config(config)

    with RunManager(config, results_dir=args.results_dir) as run:
        results = {"stages": {}}
        for name in [] if args.dry_run else config["experiment"]["stages"]:
            results["stages"][name] = STAGE_REGISTRY[name](config, run)
        if args.dry_run:
            results["note"] = "dry run: pipeline stages skipped"
        run.save_results(results)
    print(f"Run {run.run_id} {run.metadata['status']}: {run.run_dir}")
    return run


if __name__ == "__main__":
    main()
