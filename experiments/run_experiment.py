"""Single command entry point for experiments (CP7.1).

    python -m experiments.run_experiment --config experiments/configs/default.yaml model.learning_rate=0.01

Stages in STAGES run in order (currently: train θ₀). Later stages add
unlearning θᵤ, θ* and evaluation. --dry-run only saves config and metadata.
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


# Each stage is a function (config, run) -> dict of results, run in order.
# Hemanth's unlearning and Rahul's evaluation stages are added here during integration.
STAGES = [train_theta0]


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
        for stage in [] if args.dry_run else STAGES:
            results["stages"][stage.__name__] = stage(config, run)
        if args.dry_run:
            results["note"] = "dry run: pipeline stages skipped"
        run.save_results(results)
    print(f"Run {run.run_id} {run.metadata['status']}: {run.run_dir}")
    return run


if __name__ == "__main__":
    main()
