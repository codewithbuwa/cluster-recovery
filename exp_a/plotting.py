import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from scr.training import TrainResult


def stack_quality(results: list[TrainResult]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = results[0].eval_steps
    values = np.vstack([r.expected_quality for r in results])
    return steps, values.mean(axis=0), values.std(axis=0)


def moving_average(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_experiment_a(payload: dict[str, object], figure_path: Path) -> None:


    results = payload["results"]
    diagnostics = payload["diagnostics"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    fig.suptitle("Experiment A - CPO vs KTO under base-rate heterogeneity", fontsize=13, y=1.01)
    ax = axes[0]

    styles = {
        "kto": ("KTO", "steelblue", "-"),
        "cpo": ("CPO", "darkorange", "-"),
        "oracle_bob_only": ("Oracle-Bob-only", "black", "--"),
    }
    for key, (label, color, linestyle) in styles.items():
        steps, mean, std = stack_quality(results[key])
        ax.plot(
            steps,
            mean,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    ax.set_title("(a) True-quality recovery")
    ax.set_xlabel("Training step")
    ax.set_ylabel("E[q]")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    for panel_idx, (key, title) in enumerate(
        [
            ("kto_grad_weight_by_cluster", "(b-KTO) Gradient calibration"),
            ("cpo_grad_weight_by_cluster", "(b-CPO) Gradient calibration"),
        ],
        start=1,
    ):
        ax = axes[panel_idx]
        weights = diagnostics[key][0]
        step_axis = np.arange(len(weights))
        smooth_steps = step_axis[19:]
        ax.plot(
            smooth_steps,
            moving_average(weights[:, 0]),
            color="forestgreen",
            label="Alice (c=0)",
            linewidth=2,
        )
        ax.plot(
            smooth_steps,
            moving_average(weights[:, 1]),
            color="firebrick",
            label="Bob (c=1)",
            linewidth=2,
        )
        ax.axhline(0.25, color="black", linestyle=":", linewidth=1, label="max sigma'")
        ax.axhline(0.02, color="grey", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylabel("sigma'(beta m) mean per cluster")
        ax.set_ylim(-0.01, 0.27)
        ax.legend()
        ax.grid(True, alpha=0.3)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
