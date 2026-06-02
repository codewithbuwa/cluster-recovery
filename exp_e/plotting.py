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


def plot_budget_sweep(payload: dict[str, object], output_path: Path) -> None:
    f_values = list(payload["pair_fraction_values"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method in ("kto", "cpo", "mixed_cpo", "dpo"):
        means, stds = zip(*[_mean_std(payload["results"][f][method]) for f in f_values])
        ax.errorbar(f_values, means, yerr=stds, marker="o", capsize=4, linewidth=2, label=LABELS[method], color=COLORS[method])
    ax.set_title("Experiment E: budget sweep")
    ax.set_xlabel("Pairwise effort fraction f")
    ax.set_ylabel("Final E[q]")
    ax.set_ylim(0.55, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_alpha_sweep(payload: dict[str, object], output_path: Path) -> None:
    alpha_values = list(payload["alpha_values"])
    means, stds = zip(*[_mean_std(payload["results"][alpha]) for alpha in alpha_values])
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.errorbar(alpha_values, means, yerr=stds, marker="o", capsize=4, linewidth=2, color=COLORS["mixed_cpo"])
    ax.set_title("Experiment E: alpha sweep")
    ax.set_xlabel("Loss mixing alpha")
    ax.set_ylabel("Final E[q]")
    ax.set_ylim(0.55, 1.0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_pia_sweep(payload: dict[str, object], output_path: Path) -> None:
    pi_values = list(payload["pi_a_values"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True)
    kto_means = np.asarray([_mean_std(payload["results"][pi]["kto"])[0] for pi in pi_values])
    for method in ("kto", "cpo", "mixed_cpo", "dpo"):
        means, stds = zip(*[_mean_std(payload["results"][pi][method]) for pi in pi_values])
        axes[0].errorbar(pi_values, means, yerr=stds, marker="o", capsize=4, linewidth=2, label=LABELS[method], color=COLORS[method])
        if method != "kto":
            axes[1].plot(pi_values, np.asarray(means) - kto_means, marker="o", linewidth=2, label=LABELS[method], color=COLORS[method])
    axes[0].set_title("Final quality")
    axes[0].set_ylabel("Final E[q]")
    axes[1].set_title("Advantage over KTO")
    axes[1].set_ylabel("Delta E[q]")
    for ax in axes:
        ax.set_xlabel("pi_A")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].legend(frameon=False)
    fig.suptitle("Experiment E: pi_A sweep", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ref_ablation(payload: dict[str, object], output_path: Path) -> None:
    keys = ("global_alpha0", "cluster_alpha0", "global_alpha05", "cluster_alpha05")
    means, stds = zip(*[_mean_std(payload["results"][key]) for key in keys])
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(range(len(keys)), means, yerr=stds, capsize=5, color=[COLORS[key] for key in keys], alpha=0.85)
    ax.set_xticks(range(len(keys)), [LABELS[key] for key in keys])
    ax.set_title("Experiment E: reference/mixing ablation")
    ax.set_ylabel("Final E[q]")
    ax.set_ylim(0.55, 1.0)
    ax.grid(axis="y", alpha=0.25)
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
