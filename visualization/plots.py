"""Comparison charts from the evaluation stage's results.json (CP8.2).

    python -m visualization.plots path/to/results.json output_dir

One PNG per question. Each model keeps the same colour in every chart, and every
bar carries its value, so nothing depends on colour alone.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

# Fixed colour per model (categorical slots 1-4), so a model never changes colour between charts.
MODELS = {
    "theta_0": ("θ₀ original", "#2a78d6"),
    "theta_star": ("θ* retrained", "#eb6834"),
    "bpr": ("BPR baseline", "#1baf7a"),
    "unlearned": ("θᵤ unlearned", "#eda100"),
    "naive_finetune": ("Naive fine-tune", "#e87ba4"),
}


def _bars(ax, names, values, title, ylabel, fmt="{:.3f}", ylim=None):
    labels = [MODELS.get(n, (n, "#888888"))[0] for n in names]
    colours = [MODELS.get(n, (n, "#888888"))[1] for n in names]
    bars = ax.bar(labels, values, color=colours, width=0.6)
    for bar, value in zip(bars, values):
        ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9, color=INK, xytext=(0, 3), textcoords="offset points")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_facecolor(SURFACE)


def _figure(width=6.5, height=4):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    return fig, ax


def plot_all(results_path: Path, out_dir: Path) -> list[Path]:
    data = json.loads(Path(results_path).read_text())
    data = data.get("stages", {}).get("evaluate", data)
    models = data["models"]
    names = [n for n in MODELS if n in models]
    k = data["k"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig, name):
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        written.append(path)

    # 1. Utility on retained users
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), facecolor=SURFACE)
    for ax, metric in zip(axes, (f"recall@{k}", f"ndcg@{k}")):
        values = [models[n]["utility_retained"][metric] for n in names]
        _bars(ax, names, values, f"{"NDCG" if "ndcg" in metric else "Recall"}@{k} on retained users", metric, ylim=(0, max(values) * 1.2))
    save(fig, "utility.png")

    # 2. Runtime
    runtimes = [(n, models[n]["runtime_seconds"]) for n in names if models[n].get("runtime_seconds")]
    if runtimes:
        fig, ax = _figure()
        _bars(ax, [n for n, _ in runtimes], [v for _, v in runtimes], "Training / retraining time", "seconds", "{:.0f}")
        save(fig, "runtime.png")

    # 3. Forgetting: do the removed items still surface?
    fig, ax = _figure()
    values = [models[n]["forgetting"][f"forgotten_hit_rate@{k}"] for n in names]
    _bars(ax, names, values, f"Removed items still in the target user's top-{k}", "share of top-k", "{:.2f}", ylim=(0, 1.15))
    save(fig, "forgetting.png")

    # 4. Consistency with θ*
    compared = [n for n in names if models[n]["consistency_vs_theta_star"]]
    if compared:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), facecolor=SURFACE)
        for ax, key, title in (
            (axes[0], f"jaccard@{k}_retained", "Top-k overlap with θ*, retained users"),
            (axes[1], f"jaccard@{k}_target", "Top-k overlap with θ*, forgotten user"),
        ):
            _bars(ax, compared, [models[n]["consistency_vs_theta_star"][key] for n in compared], title, "Jaccard", "{:.2f}", ylim=(0, 1.15))
        save(fig, "consistency.png")

    # 5. User-level membership attack on the forgotten user
    if all("privacy_mia_user" in models[n] for n in names):
        fig, ax = _figure()
        values = [models[n]["privacy_mia_user"]["target_membership_probability"] for n in names]
        _bars(ax, names, values, "Attacker's belief the forgotten user was in training", "membership probability", "{:.2f}", ylim=(0, 1.15))
        save(fig, "privacy.png")

    return written


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m visualization.plots results.json output_dir")
    for path in plot_all(Path(sys.argv[1]), Path(sys.argv[2])):
        print(path)
