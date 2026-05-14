import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from exp_c.summary import method_values


def _mean_std(results: dict[str, list]) -> tuple[np.ndarray, np.ndarray]:
    values = method_values(results)
    means = np.asarray([values["kto"].mean(), values["cpo"].mean()])
    stds = np.asarray([values["kto"].std(), values["cpo"].std()])
    return means, stds


def _moving_average(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _label_bars(ax, bars) -> None:
    baseline = ax.get_ylim()[0]
    for bar in bars:
        value = bar.get_height()
        label_y = baseline + (value - baseline) / 2
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{value:.4f}",
            ha="center",
            va="center",
            rotation=90,
            color="black",
            fontsize=8,
        )


def plot_experiment_c(payload: dict[str, object], figure_path: Path) -> None:

    results = payload["results"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    fig.suptitle("Experiment C - Ablations", fontsize=13)
    colors = {"kto": "steelblue", "cpo": "darkorange"}

    variants = list(results["c1"].keys())
    x = np.arange(len(variants))
    width = 0.36
    kto_means = []
    kto_stds = []
    cpo_means = []
    cpo_stds = []
    for variant in variants:
        values = method_values(results["c1"][variant])
        kto_means.append(values["kto"].mean())
        kto_stds.append(values["kto"].std())
        cpo_means.append(values["cpo"].mean())
        cpo_stds.append(values["cpo"].std())
    kto_bars = axes[0].bar(x - width / 2, kto_means, width, yerr=kto_stds, capsize=4, color=colors["kto"], alpha=0.85, label="KTO")
    cpo_bars = axes[0].bar(x + width / 2, cpo_means, width, yerr=cpo_stds, capsize=4, color=colors["cpo"], alpha=0.85, label="CPO")
    axes[0].set_title("C1: Reference variant")
    axes[0].set_xticks(x, variants)
    axes[0].set_ylabel("Final E[q]")
    axes[0].set_ylim(0.4, 1.0)
    _label_bars(axes[0], kto_bars)
    _label_bars(axes[0], cpo_bars)
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    means, stds = _mean_std(results["c2"])
    c2_bars = axes[1].bar(["KTO", "CPO"], means, yerr=stds, capsize=5, color=[colors["kto"], colors["cpo"]], alpha=0.85, width=0.5)
    axes[1].set_title("C2: Noise-only heterogeneity")
    axes[1].set_ylabel("Final E[q]")
    axes[1].set_ylim(0.4, 1.0)
    _label_bars(axes[1], c2_bars)
    axes[1].grid(True, axis="y", alpha=0.3)

    beta_values = np.asarray(payload["beta_values"], dtype=float)
    kto_means = []
    kto_stds = []
    cpo_means = []
    cpo_stds = []
    for beta in beta_values:
        values = method_values(results["c3"][beta])
        kto_means.append(values["kto"].mean())
        kto_stds.append(values["kto"].std())
        cpo_means.append(values["cpo"].mean())
        cpo_stds.append(values["cpo"].std())
    axes[2].errorbar(beta_values, kto_means, yerr=kto_stds, fmt="o-", color=colors["kto"], capsize=4, linewidth=2, label="KTO")
    axes[2].errorbar(beta_values, cpo_means, yerr=cpo_stds, fmt="s-", color=colors["cpo"], capsize=4, linewidth=2, label="CPO")
    axes[2].set_xscale("log")
    axes[2].set_title("C3: beta sweep")
    axes[2].set_xlabel("beta")
    axes[2].set_ylabel("Final E[q]")
    axes[2].set_ylim(0.4, 1.0)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    means, stds = _mean_std(results["c4"])
    c4_bars = axes[3].bar(["KTO", "CPO (random c)"], means, yerr=stds, capsize=5, color=[colors["kto"], "grey"], alpha=0.85, width=0.5)
    axes[3].set_title("C4: Misspecified clusters")
    axes[3].set_ylabel("Final E[q]")
    axes[3].set_ylim(0.4, 1.0)
    _label_bars(axes[3], c4_bars)
    axes[3].grid(True, axis="y", alpha=0.3)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

