"""CP7.1 checks for the experiment manager. Uses a temp results folder only."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from experiments.config import load_config
from experiments.run_manager import RunManager, make_run_id
from experiments.run_experiment import main


def test_overrides_are_applied():
    config = load_config(overrides=["model.learning_rate=0.01", "experiment.name=smoke"])
    assert config["model"]["learning_rate"] == 0.01
    assert config["experiment"]["name"] == "smoke"
    assert config["model"]["embedding_dim"] == 64


def test_run_id_changes_with_config():
    a = load_config()
    b = load_config(overrides=["model.learning_rate=0.01"])
    assert make_run_id(a) != make_run_id(b)


def test_dry_run_saves_config_metadata_results(tmp_path):
    run = main(["--dry-run", "--results-dir", str(tmp_path), "experiment.name=smoke"])
    run_dir = Path(run.run_dir)
    saved_config = yaml.safe_load((run_dir / "config.yaml").read_text())
    metadata = json.loads((run_dir / "metadata.json").read_text())
    results = json.loads((run_dir / "results.json").read_text())
    assert saved_config["experiment"]["name"] == "smoke"
    assert metadata["run_id"] == run.run_id
    assert metadata["status"] == "completed"
    assert metadata["git"]["commit"]
    assert "note" in results


def test_failed_run_is_recorded(tmp_path):
    config = load_config(overrides=["experiment.name=fail"])
    try:
        with RunManager(config, results_dir=tmp_path) as run:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    metadata = json.loads((run.run_dir / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert "boom" in metadata["error"]
