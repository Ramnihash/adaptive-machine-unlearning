"""Run IDs and metadata saving for reproducible experiments (CP7.1).

Each run gets its own folder under results_dir:

    <results_dir>/<run_id>/
        config.yaml      resolved config used for the run
        metadata.json    run ID, timestamps, seed, git commit, environment, status
        results.json     metrics written by the experiment
        checkpoints/     model files saved through RunManager.checkpoint_path()
"""
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from experiments.config import REPO_ROOT


def config_hash(config):
    """Short stable hash of the config, so identical settings are easy to spot."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def make_run_id(config, now=None):
    """Run ID = UTC timestamp + experiment name + config hash, e.g. 20261022-130501-default-1a2b3c4d."""
    now = now or datetime.now(timezone.utc)
    name = str(config["experiment"]["name"]).replace(" ", "_")
    return f"{now:%Y%m%d-%H%M%S}-{name}-{config_hash(config)}"


def git_info():
    def run(*args):
        try:
            out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def environment_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module in ("numpy", "pandas", "torch", "sklearn"):
        try:
            info[module] = __import__(module).__version__
        except ImportError:
            info[module] = None
    return info


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class RunManager:
    """Creates a run folder and records config, metadata, results and checkpoint paths."""

    def __init__(self, config, results_dir=None):
        self.config = config
        self.run_id = make_run_id(config)
        base = Path(results_dir or config["experiment"]["results_dir"])
        if not base.is_absolute():
            base = REPO_ROOT / base
        self.run_dir = base / self.run_id
        if self.run_dir.exists():
            raise FileExistsError(f"Run folder already exists: {self.run_dir}")
        self.metadata = {
            "run_id": self.run_id,
            "experiment": config["experiment"]["name"],
            "seed": config["experiment"]["seed"],
            "config_hash": config_hash(config),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "status": "running",
            "git": git_info(),
            "environment": environment_info(),
            "checkpoints": {},
        }

    def __enter__(self):
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "checkpoints").mkdir()
        set_seed(self.config["experiment"]["seed"])
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)
        self._write_metadata()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.metadata["status"] = "failed" if exc_type else "completed"
        if exc_type:
            self.metadata["error"] = f"{exc_type.__name__}: {exc}"
        self._write_metadata()
        return False

    def checkpoint_path(self, name):
        """Path for a model file (e.g. 'theta_0.pt'); it is recorded in metadata."""
        path = self.run_dir / "checkpoints" / name
        self.metadata["checkpoints"][name] = str(path.relative_to(REPO_ROOT))
        self._write_metadata()
        return path

    def save_results(self, results):
        with open(self.run_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    def _write_metadata(self):
        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
