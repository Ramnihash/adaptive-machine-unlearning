"""Summary table and run-time lookup, using small hand-written result files (not experimental data)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.harness import read_run_info
from evaluation.summary import summary_rows, to_markdown


def write_run(tmp_path, name, stage_name, stage):
    run = tmp_path / name
    (run / "checkpoints").mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({"run_id": name}))
    (run / "results.json").write_text(json.dumps({"stages": {stage_name: stage}}))
    return run / "checkpoints" / "x.pt"


def test_unlearn_run_time_is_read_from_nested_strategy(tmp_path):
    checkpoint = write_run(tmp_path, "run1", "unlearn", {"strategy": {"unlearning_seconds": 12.5}})
    assert read_run_info(checkpoint) == ("run1", 12.5)


def test_summary_marks_unlearned_row_with_routing(tmp_path):
    unlearned = write_run(tmp_path, "u", "unlearn", {"routing": {"influence_tier": "low", "training_degree": 21, "selected_strategy": "gradient_ascent"}})
    model = lambda: {
        "runtime_seconds": 1.0,
        "utility_retained": {"ndcg@10": 0.2, "recall@10": 0.1},
        "forgetting": {"forgotten_hit_rate@10": 0.0},
        "privacy_mia_user": {"target_membership_probability": 0.01},
        "consistency_vs_theta_star": {"jaccard@10_retained": 0.9},
    }
    results = tmp_path / "eval.json"
    results.write_text(json.dumps({"stages": {"evaluate": {
        "target_user": 13, "k": 10, "checkpoint_paths": {"unlearned": str(unlearned)},
        "models": {"theta_0": model(), "unlearned": model()},
    }}}))
    rows = summary_rows([results])
    assert [r["model"] for r in rows] == ["theta_0", "unlearned"]
    assert rows[1]["strategy"] == "gradient_ascent" and rows[1]["tier"] == "low"
    assert "gradient_ascent" in to_markdown(rows)
