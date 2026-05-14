import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from exp_d.summary import final_values


def _mean_std(runs: list) -> tuple[float, float]:
    values = np.asarray([run.expected_quality[-1] for run in runs])
    return float(values.mean()), float(values.std())


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


def plot_experiment_d(payload: dict[str, object], figure_path: Path) -> None:

    results = payload["results"]
    clustering = payload["clustering"]
    ksweep = payload["ksweep"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    keys = ["kto", "cpo_random", "cpo_learned", "cpo_oracle"]
    labels = ["KTO", "CPO\nrandom", "CPO\nlearned", "CPO\noracle"]
    colors = ["#1f77b4", "grey", "mediumpurple", "#ff7f0e"]
    values = final_values(results)
    means = [values[key].mean() for key in keys]
    stds = [values[key].std() for key in keys]
    bars = axes[0].bar(labels, means, yerr=stds, color=colors)
    _label_bars(axes[0], bars)
    axes[0].set_title("(a) K=2 comparison")
    axes[0].set_ylabel("Final expected true quality")
    axes[0].grid(axis="y", alpha=0.2)

    groups = ["random", "learned", "oracle"]
    x = np.arange(len(groups))
    width = 0.32
    nmi = [clustering[group]["nmi"].mean() for group in groups]
    nmi_std = [clustering[group]["nmi"].std() for group in groups]
    pur = [clustering[group]["purity"].mean() for group in groups]
    pur_std = [clustering[group]["purity"].std() for group in groups]
    nmi_bars = axes[1].bar(x - width / 2, nmi, width, yerr=nmi_std, color="indigo", label="NMI")
    purity_bars = axes[1].bar(x + width / 2, pur, width, yerr=pur_std, color="plum", label="Purity")
    _label_bars(axes[1], nmi_bars)
    _label_bars(axes[1], purity_bars)
    axes[1].set_xticks(x, ["Random", "Learned", "Oracle"])
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("(b) Cluster recovery")
    axes[1].set_ylabel("Score")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    k_values = sorted(ksweep.keys())
    learned_mean, learned_std, random_mean, random_std = [], [], [], []
    for k in k_values:
        mean, std = _mean_std(ksweep[k]["learned"])
        learned_mean.append(mean)
        learned_std.append(std)
        mean, std = _mean_std(ksweep[k]["random"])
        random_mean.append(mean)
        random_std.append(std)
    axes[2].errorbar(k_values, learned_mean, yerr=learned_std, marker="s", color="mediumpurple", label="CPO learned")
    axes[2].errorbar(k_values, random_mean, yerr=random_std, marker="o", color="grey", label="CPO random")
    axes[2].axhline(values["cpo_oracle"].mean(), color="#ff7f0e", linestyle="--", label="Oracle K=2")
    axes[2].axhline(values["kto"].mean(), color="#1f77b4", linestyle=":", label="KTO K=1")
    axes[2].set_xscale("log")
    axes[2].set_title("(c) K-sweep")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Final expected true quality")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(alpha=0.2)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
