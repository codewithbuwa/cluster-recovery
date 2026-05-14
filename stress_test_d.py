"""
Experiment D stress test: how does CPO with learned clusters behave when
clustering is genuinely difficult?

Re-runs Experiment D under four threshold regimes with 10 seeds each, always using
40 labels per annotator. It measures the learned-vs-oracle quality gap and saves
a three-panel diagnostic figure.

Run from the project root:

    .venv/bin/python3 stress_test_d.py

Outputs:

    outputs/exp_d/stress_results.pkl
    outputs/exp_d/stress_comparison.png
"""

import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_d.clustering import (
    compute_signatures,
    kmeans_1d,
    normalized_mutual_information,
    purity,
    random_partition,
)
from exp_d.experiment import _run_offline
from scr.config import TrainConfig, WorldConfig
from scr.samplers import OfflineDataset
from scr.world import SyntheticWorld


@dataclass(frozen=True)
class _RegimeCfg:
    """Minimal config object that exp_d._run_offline knows how to consume."""

    world: WorldConfig
    train: TrainConfig
    seeds: tuple[int, ...]


REGIMES = ["baseline", "moderate", "closer", "very_close"]
REGIME_LABELS = {
    "baseline": "Baseline\nT=(0.25, 0.75)",
    "moderate": "Moderate T\nT=(0.35, 0.65)",
    "closer": "Closer T\nT=(0.40, 0.60)",
    "very_close": "Very close T\nT=(0.45, 0.55)",
}
METHODS = ["kto", "cpo_random", "cpo_learned", "cpo_oracle"]
METHOD_LABELS = {
    "kto": "KTO",
    "cpo_random": "CPO random",
    "cpo_learned": "CPO learned",
    "cpo_oracle": "CPO oracle",
}
COLORS = {
    "kto": "#1f77b4",
    "cpo_random": "grey",
    "cpo_learned": "mediumpurple",
    "cpo_oracle": "#ff7f0e",
}


def run_regime(name: str, world_cfg: WorldConfig, train_cfg: TrainConfig, seeds: tuple[int, ...]) -> dict[str, object]:
    """Run KTO plus three CPO variants for one stress-test world."""

    cfg = _RegimeCfg(world=world_cfg, train=train_cfg, seeds=seeds)
    t0 = time.time()
    print(f"\n{'=' * 78}")
    print(
        f"REGIME: {name}  "
        f"(tau_A={world_cfg.tau_a}, tau_B={world_cfg.tau_b}, "
        f"samples_per_annotator=40)"
    )
    print(f"{'=' * 78}")

    results = {
        "kto": _run_offline(
            cfg,
            1,
            lambda dataset, seed, k: None,
            method="kto",
            log_prefix=None,
        ),
        "cpo_random": _run_offline(
            cfg,
            2,
            lambda dataset, seed, k: random_partition(world_cfg.n_annotators, k, seed),
            log_prefix=None,
        ),
        "cpo_learned": _run_offline(
            cfg,
            2,
            lambda dataset, seed, k: kmeans_1d(compute_signatures(dataset), k, seed),
            log_prefix=None,
        ),
        "cpo_oracle": _run_offline(
            cfg,
            2,
            lambda dataset, seed, k: dataset.annotator_cluster.copy(),
            log_prefix=None,
        ),
    }

    clustering = {"learned": {"nmi": [], "purity": [], "n_wrong": []}}
    for seed in seeds:
        world = SyntheticWorld(world_cfg, seed)
        dataset = OfflineDataset(world, world_cfg, seed)
        truth = dataset.annotator_cluster
        learned = kmeans_1d(compute_signatures(dataset), 2, seed)

        clustering["learned"]["nmi"].append(normalized_mutual_information(learned, truth))
        clustering["learned"]["purity"].append(purity(learned, truth))

        agree_same = int((learned == truth).sum())
        agree_flip = int((learned == (1 - truth)).sum())
        clustering["learned"]["n_wrong"].append(len(truth) - max(agree_same, agree_flip))

    print(f"  ran in {time.time() - t0:.1f}s")
    return {
        "results": results,
        "clustering": clustering,
        "world": world_cfg,
        "train": train_cfg,
        "seeds": seeds,
    }


def plot_stress(out: dict[str, dict[str, object]], figure_path: Path) -> None:
    """Save the three-panel stress comparison figure."""

    data = {}
    for name in REGIMES:
        regime = out[name]
        finals = {
            method: np.asarray([run.expected_quality[-1] for run in regime["results"][method]])
            for method in METHODS
        }
        data[name] = {
            "finals": finals,
            "nmi": np.asarray(regime["clustering"]["learned"]["nmi"]),
            "nwrong": np.asarray(regime["clustering"]["learned"]["n_wrong"]),
        }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(REGIMES))
    width = 0.20
    for idx, method in enumerate(METHODS):
        means = [data[regime]["finals"][method].mean() for regime in REGIMES]
        stds = [data[regime]["finals"][method].std() for regime in REGIMES]
        ax.bar(
            x + (idx - 1.5) * width,
            means,
            width,
            yerr=stds,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            capsize=2,
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[regime] for regime in REGIMES], fontsize=9)
    ax.set_ylabel("Final E[q]")
    ax.set_title("(a) Methods across regimes (10 seeds)")
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower center", fontsize=8, frameon=False, ncol=4, bbox_to_anchor=(0.5, -0.30))
    ax.grid(axis="y", alpha=0.25)

    for idx, regime in enumerate(REGIMES):
        learned_mean = data[regime]["finals"]["cpo_learned"].mean()
        oracle_mean = data[regime]["finals"]["cpo_oracle"].mean()
        gap = learned_mean - oracle_mean
        ax.annotate(
            f"L-O = {gap:+.3f}",
            xy=(idx, 0.97),
            ha="center",
            fontsize=8,
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="grey", linewidth=0.4),
        )

    ax = axes[1]
    nmi_means = [data[regime]["nmi"].mean() for regime in REGIMES]
    nmi_stds = [data[regime]["nmi"].std() for regime in REGIMES]
    nwrong_means = [data[regime]["nwrong"].mean() for regime in REGIMES]
    nwrong_stds = [data[regime]["nwrong"].std() for regime in REGIMES]

    ax2 = ax.twinx()
    positions = np.arange(len(REGIMES))
    ax.bar(
        positions - 0.2,
        nmi_means,
        0.4,
        yerr=nmi_stds,
        color="indigo",
        capsize=3,
        edgecolor="black",
        linewidth=0.4,
        label="NMI",
    )
    ax2.bar(
        positions + 0.2,
        nwrong_means,
        0.4,
        yerr=nwrong_stds,
        color="firebrick",
        capsize=3,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
        label="# misclassified",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([REGIME_LABELS[regime] for regime in REGIMES], fontsize=9)
    ax.set_ylabel("NMI (learned vs truth)", color="indigo")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelcolor="indigo")
    ax2.set_ylabel("# misclassified (of 120)", color="firebrick")
    ax2.set_ylim(0, 60)
    ax2.tick_params(axis="y", labelcolor="firebrick")
    ax.set_title("(b) Cluster recovery degrades sharply")
    ax.grid(axis="y", alpha=0.25)
    for idx, (nmi_mean, wrong_mean) in enumerate(zip(nmi_means, nwrong_means)):
        ax.text(idx - 0.2, nmi_mean + 0.04, f"{nmi_mean:.2f}", ha="center", fontsize=8, color="indigo")
        ax2.text(idx + 0.2, wrong_mean + 2.5, f"{wrong_mean:.0f}", ha="center", fontsize=8, color="firebrick")

    ax = axes[2]
    markers = {"baseline": "o", "moderate": "^", "closer": "s", "very_close": "D"}
    rng = np.random.default_rng(42)
    for regime in REGIMES:
        learned = data[regime]["finals"]["cpo_learned"]
        oracle = data[regime]["finals"]["cpo_oracle"]
        nmi = data[regime]["nmi"]
        gap = learned - oracle
        jitter = rng.uniform(-0.012, 0.012, size=len(nmi))
        ax.scatter(
            nmi + jitter,
            gap,
            s=55,
            marker=markers[regime],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            label=regime,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(-0.02, color="grey", linewidth=0.6, linestyle="--", label="+/- 0.02 criterion")
    ax.axhline(+0.02, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Clustering NMI (learned vs truth)")
    ax.set_ylabel("Learned E[q] - Oracle E[q]")
    ax.set_title("(c) Learning cost rises as clustering breaks")
    ax.set_xlim(-0.05, 1.08)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=0.25)
    ax.annotate(
        "baseline: all 10\nseeds at gap ~= 0\n(NMI ~= 1.0)",
        xy=(0.98, 0.001),
        xytext=(0.78, 0.030),
        fontsize=7.5,
        ha="left",
        arrowprops=dict(arrowstyle="->", color="grey", lw=0.6),
    )

    fig.suptitle("Experiment D stress test - threshold separation sweep, fixed 40 labels per annotator", fontsize=12, y=1.04)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {figure_path}")


def summarize(payload: dict[str, object], name: str) -> None:
    results = payload["results"]
    clustering = payload["clustering"]
    finals = {
        method: np.asarray([run.expected_quality[-1] for run in runs])
        for method, runs in results.items()
    }
    learned = finals["cpo_learned"]
    oracle = finals["cpo_oracle"]
    kto = finals["kto"]
    random_runs = finals["cpo_random"]
    nmi = np.asarray(clustering["learned"]["nmi"])
    pur = np.asarray(clustering["learned"]["purity"])
    nwrong = np.asarray(clustering["learned"]["n_wrong"])
    n_total = payload["world"].n_annotators

    print(f"\n{name} summary (10 seeds, 10 decimals):")
    print(f"  KTO          mean={kto.mean():.10f}  std={kto.std():.10f}")
    print(f"  CPO random   mean={random_runs.mean():.10f}  std={random_runs.std():.10f}")
    print(f"  CPO learned  mean={learned.mean():.10f}  std={learned.std():.10f}")
    print(f"  CPO oracle   mean={oracle.mean():.10f}  std={oracle.std():.10f}")
    print(f"  Learned - oracle (mean diff) : {learned.mean() - oracle.mean():+.4e}")
    print(f"  Clustering: mean NMI = {nmi.mean():.4f}, mean purity = {pur.mean():.4f}")
    print(f"  Misclassified annotators per seed (of {n_total}): {nwrong.tolist()}")


def main() -> None:
    train_cfg = TrainConfig(steps=250, batch_size=128, learning_rate=0.2)
    seeds = tuple(range(10))

    regimes = {
        "baseline": WorldConfig(
            pi_a=0.85,
            tau_a=0.25,
            tau_b=0.75,
            n_annotators=120,
            samples_per_annotator=40,
        ),
        "moderate": WorldConfig(
            pi_a=0.85,
            tau_a=0.35,
            tau_b=0.65,
            n_annotators=120,
            samples_per_annotator=40,
        ),
        "closer": WorldConfig(
            pi_a=0.85,
            tau_a=0.40,
            tau_b=0.60,
            n_annotators=120,
            samples_per_annotator=40,
        ),
        "very_close": WorldConfig(
            pi_a=0.85,
            tau_a=0.45,
            tau_b=0.55,
            n_annotators=120,
            samples_per_annotator=40,
        ),
    }

    out = {
        name: run_regime(name, world, train_cfg, seeds)
        for name, world in regimes.items()
    }

    for name in regimes:
        summarize(out[name], name)

    out_dir = Path("outputs/exp_d")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "stress_results.pkl"
    with results_path.open("wb") as handle:
        pickle.dump(out, handle)
    print(f"\nSaved results: {results_path}")

    plot_stress(out, out_dir / "stress_comparison.png")


if __name__ == "__main__":
    main()
