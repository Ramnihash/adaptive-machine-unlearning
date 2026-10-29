"""Summary table across target users: one row per (user, model) from evaluate-stage results.

    python -m evaluation.summary results_u13.json results_u39.json results_u25.json

Reads the evaluate stage's results.json files. For the unlearned model it also
reads the routing decision (tier, strategy) from the unlearn run saved next to
its checkpoint.
"""

import json
import sys
from pathlib import Path


def _stage(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    return data.get("stages", {}).get("evaluate", data)


def routing_for(evaluation: dict, result_path: Path | None = None) -> dict:
    """Routing decision of the unlearn run behind the 'unlearned' checkpoint, or {}.

    If the original checkpoint path no longer exists (for example after a restart), a saved
    run folder named unlearn_u<user> next to the evaluation folder is used instead.
    """
    candidates = []
    checkpoint = evaluation.get("checkpoint_paths", {}).get("unlearned")
    if checkpoint:
        candidates.append(Path(checkpoint).parent.parent / "results.json")
    if result_path is not None:
        candidates.append(Path(result_path).parent.parent / f"unlearn_u{evaluation['target_user']}" / "results.json")
    for candidate in candidates:
        try:
            return json.loads(candidate.read_text())["stages"]["unlearn"]["routing"]
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return {}


def summary_rows(result_paths: list[Path]) -> list[dict]:
    rows = []
    for path in result_paths:
        evaluation = _stage(path)
        k = evaluation["k"]
        routing = routing_for(evaluation, Path(path))
        for name, model in evaluation["models"].items():
            consistency = model.get("consistency_vs_theta_star") or {}
            membership = (model.get("privacy_mia_user") or {}).get("target_membership_probability")
            rows.append({
                "user": evaluation["target_user"],
                "tier": routing.get("influence_tier", ""),
                "degree": routing.get("training_degree", ""),
                "strategy": routing.get("selected_strategy", "") if name == "unlearned" else "",
                "model": name,
                "seconds": model.get("runtime_seconds"),
                f"ndcg@{k}_retained": model["utility_retained"][f"ndcg@{k}"],
                f"recall@{k}_retained": model["utility_retained"][f"recall@{k}"],
                f"forgotten_hit@{k}": model["forgetting"][f"forgotten_hit_rate@{k}"],
                "membership_prob": membership,
                f"jaccard@{k}_vs_theta_star": consistency.get(f"jaccard@{k}_retained"),
            })
    return rows


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return ""
    columns = list(rows[0])

    def cell(value):
        if value is None or value == "":
            return "-"
        return f"{value:.4f}" if isinstance(value, float) and abs(value) < 100 else (
            f"{value:.0f}" if isinstance(value, float) else str(value))

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(row[c]) for c in columns) + " |" for row in rows]
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m evaluation.summary results.json [results.json ...]")
    print(to_markdown(summary_rows([Path(p) for p in sys.argv[1:]])))
