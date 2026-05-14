from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from exp_b.summary import final_values


def plot_experiment_b(payload: dict[str, object], figure_path: Path) -> None:

    values = final_values(payload)
    pi_a_values = np.asarray(payload["pi_a_values"], dtype=float)

    kto_mean = np.asarray([values[pi_a]["kto"].mean() for pi_a in pi_a_values])
    kto_std = np.asarray([values[pi_a]["kto"].std() for pi_a in pi_a_values])
    cpo_mean = np.asarray([values[pi_a]["cpo"].mean() for pi_a in pi_a_values])
    cpo_std = np.asarray([values[pi_a]["cpo"].std() for pi_a in pi_a_values])
    delta = cpo_mean - kto_mean
    delta_se = np.sqrt(kto_std**2 + cpo_std**2) / np.sqrt(len(payload["seeds"]))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    fig.suptitle("Experiment B - pi_A sweep", fontsize=13)

    axes[0].errorbar(pi_a_values, kto_mean, yerr=kto_std, fmt="o-", color="steelblue", capsize=4, linewidth=2, label="KTO")
    axes[0].errorbar(pi_a_values, cpo_mean, yerr=cpo_std, fmt="s-", color="darkorange", capsize=4, linewidth=2, label="CPO")
    axes[0].set_title("(left) Absolute quality")
    axes[0].set_xlabel("pi_A")
    axes[0].set_ylabel("Final E[q]")
    axes[0].set_ylim(0.4, 1.0)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(pi_a_values, delta, yerr=delta_se, fmt="o-", color="purple", capsize=4, linewidth=2)
    axes[1].axhline(0.0, color="black", linewidth=1.2)
    axes[1].set_title("(right) CPO advantage")
    axes[1].set_xlabel("pi_A")
    axes[1].set_ylabel("Delta = E[q]CPO - E[q]KTO")
    axes[1].grid(True, alpha=0.3)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
