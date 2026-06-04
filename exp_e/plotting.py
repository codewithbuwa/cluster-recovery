from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "kto": "KTO",
    "cpo": "CPO",
    "mixed_cpo": "Mixed CPO",
    "dpo": "DPO",
    "global_alpha0": "Global z\nalpha=0",
    "cluster_alpha0": "Cluster z_k\nalpha=0",
    "global_alpha05": "Global z\nalpha=0.5",
    "cluster_alpha05": "Cluster z_k\nalpha=0.5",
}

COLORS = {
    "kto": "steelblue",
    "cpo": "darkorange",
    "mixed_cpo": "seagreen",
    "dpo": "mediumpurple",
    "global_alpha0": "steelblue",
    "cluster_alpha0": "darkorange",
    "global_alpha05": "mediumseagreen",
    "cluster_alpha05": "seagreen",
}


def _mean_std(runs: list | None) -> tuple[float, float]:
    if runs is None:
        return np.nan, np.nan
    values = np.asarray([run.expected_quality[-1] for run in runs])
    return float(values.mean()), float(values.std())


def _values(runs: list | None) -> np.ndarray | None:
    if runs is None:
        return None
    return np.asarray([run.expected_quality[-1] for run in runs], dtype=np.float64)


def plot_budget_sweep(payload: dict[str, object], output_path: Path) -> None:
    f_values = list(payload["pair_fraction_values"])
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    fig.suptitle("Budget sweep: when does mixing pay off?", fontsize=13)
    for method in ("kto", "cpo", "mixed_cpo", "dpo"):
        means, stds = zip(*[_mean_std(payload["results"][f][method]) for f in f_values])
        ax.errorbar(f_values, means, yerr=stds, marker="o", capsize=4, linewidth=2, label=LABELS[method], color=COLORS[method])
    ax.set_title("total effort = 256 labels/step, pi_A = 0.9")
    ax.set_xlabel("Pairwise effort fraction f")
    ax.set_ylabel("Final E[q]")
    ax.set_ylim(0.55, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_budget_sweep_deltas(payload: dict[str, object], output_path: Path) -> None:
    f_values = list(payload["pair_fraction_values"])
    comparisons = (
        ("mixed_cpo", "dpo", "Mixed CPO - DPO", "seagreen"),
        ("mixed_cpo", "cpo", "Mixed CPO - CPO", "darkorange"),
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for left_key, right_key, label, color in comparisons:
        xs = []
        means = []
        stds = []
        for f_value in f_values:
            left = _values(payload["results"][f_value][left_key])
            right = _values(payload["results"][f_value][right_key])
            if left is None or right is None:
                continue
            delta = left - right
            xs.append(f_value)
            means.append(float(delta.mean()))
            stds.append(float(delta.std()))

        if xs:
            ax.errorbar(xs, means, yerr=stds, marker="o", capsize=4, linewidth=2, label=label, color=color)

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Budget sweep deltas")
    ax.set_xlabel("Pairwise effort fraction f")
    ax.set_ylabel("Final E[q] difference")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_alpha_sweep(payload: dict[str, object], output_path: Path) -> None:
    alpha_values = list(payload["alpha_values"])
    means, stds = zip(*[_mean_std(payload["results"][alpha]) for alpha in alpha_values])
    means_array = np.asarray(means)
    best_alpha = alpha_values[int(np.nanargmax(means_array))]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.errorbar(alpha_values, means, yerr=stds, marker="o", capsize=4, linewidth=2, color=COLORS["mixed_cpo"])
    ax.axvline(best_alpha, color="black", linestyle="--", linewidth=1.2, label=f"argmax alpha = {best_alpha:g}")
    ax.annotate(
        "CPO\n(unary only)",
        xy=(0.0, means[0]),
        xytext=(0.08, means[0] + 0.035),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        fontsize=9,
    )
    ax.annotate(
        "DPO\n(pairs only)",
        xy=(1.0, means[-1]),
        xytext=(0.78, means[-1] - 0.06),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        fontsize=9,
    )
    ax.set_title("Alpha sweep at fixed mixed budget")
    ax.set_xlabel("Loss mixing alpha")
    ax.set_ylabel("Final E[q]")
    ax.set_ylim(0.55, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_pia_sweep(payload: dict[str, object], output_path: Path) -> None:
    pi_values = list(payload["pi_a_values"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharex=True)
    kto_means = np.asarray([_mean_std(payload["results"][pi]["kto"])[0] for pi in pi_values])
    for method in ("kto", "cpo", "mixed_cpo", "dpo"):
        means, stds = zip(*[_mean_std(payload["results"][pi][method]) for pi in pi_values])
        axes[0].errorbar(pi_values, means, yerr=stds, marker="o", capsize=4, linewidth=2, label=LABELS[method], color=COLORS[method])
        if method != "kto":
            axes[1].plot(pi_values, np.asarray(means) - kto_means, marker="o", linewidth=2, label=LABELS[method], color=COLORS[method])
    axes[0].set_title("Final quality")
    axes[0].set_ylabel("Final E[q]")
    axes[1].set_title("Difference vs. KTO")
    axes[1].set_ylabel("Delta = E[q] - E[q] KTO")
    for ax in axes:
        ax.set_xlabel("pi_A")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].legend(frameon=False)
    fig.suptitle("pi_A sweep at fixed alpha = 0.5", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ref_ablation(payload: dict[str, object], output_path: Path) -> None:
    groups = ("alpha=0", "alpha=0.5")
    global_keys = ("global_alpha0", "global_alpha05")
    cluster_keys = ("cluster_alpha0", "cluster_alpha05")
    global_means, global_stds = zip(*[_mean_std(payload["results"][key]) for key in global_keys])
    cluster_means, cluster_stds = zip(*[_mean_std(payload["results"][key]) for key in cluster_keys])

    x = np.arange(len(groups))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    global_bars = ax.bar(
        x - width / 2,
        global_means,
        width,
        yerr=global_stds,
        capsize=5,
        color=COLORS["global_alpha0"],
        alpha=0.85,
        label="Global z",
    )
    cluster_bars = ax.bar(
        x + width / 2,
        cluster_means,
        width,
        yerr=cluster_stds,
        capsize=5,
        color=COLORS["cluster_alpha0"],
        alpha=0.85,
        label="Per-cluster z_k",
    )
    ax.bar_label(global_bars, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(cluster_bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x, groups)
    ax.set_title("Per-cluster z_k vs. mixing")
    ax.set_ylabel("Final E[q]")
    ax.set_ylim(0.55, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot(payload: dict[str, object], output_path: Path) -> None:
    name = payload["name"]
    if name == "budget_sweep":
        plot_budget_sweep(payload, output_path)
    elif name == "alpha_sweep":
        plot_alpha_sweep(payload, output_path)
    elif name == "pia_sweep":
        plot_pia_sweep(payload, output_path)
    elif name == "ref_ablation":
        plot_ref_ablation(payload, output_path)
    else:
        raise ValueError(f"unknown Experiment E payload: {name}")
