import argparse
import pickle
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp_d.summary import final_values


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_relative(path: Path) -> Path:
    return path if path.is_absolute() else _repo_root() / path


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


def _label_bars_vertical_5dp(ax, bars) -> None:
    y0, y1 = ax.get_ylim()
    axis_span = y1 - y0
    min_inside_fraction = 0.12
    for bar in bars:
        height = bar.get_height()
        bar_extent = height - y0
        if bar_extent >= min_inside_fraction * axis_span:
            label_y = y0 + bar_extent / 2.0
            va = "center"
        else:
            label_y = height + 0.02 * axis_span
            va = "bottom"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            label_y,
            f"{height:.5f}",
            ha="center",
            va=va,
            rotation=90,
            color="black",
            fontsize=8.5,
        )


def plot_panel_b(payload: dict[str, object], figure_path: Path) -> None:
    refit_logs = payload["refit_logs"]
    clustering = payload["clustering"]
    config = payload["config"]

    def _stats(logs_per_seed):
        steps = [entry["step"] for entry in logs_per_seed[0] if "nmi" in entry]
        arr = np.asarray(
            [
                [entry["nmi"] for entry in seed_log if "nmi" in entry]
                for seed_log in logs_per_seed
            ]
        )
        return steps, arr.mean(axis=0), arr.std(axis=0)

    hard_steps, hard_mean, hard_std = _stats([seed_log["hard"] for seed_log in refit_logs])
    _soft_steps, soft_mean, soft_std = _stats([seed_log["soft"] for seed_log in refit_logs])

    fig, ax = plt.subplots(1, 1, figsize=(11.0, 4.8))
    ax.set_ylim(0, 1.05)

    n_steps = len(hard_steps)
    positions = np.arange(n_steps, dtype=np.float64)
    width = 0.38

    hard_bars = ax.bar(
        positions - width / 2,
        hard_mean,
        width=width,
        yerr=hard_std,
        color="#6a51a3",
        capsize=3,
        label="Hard streaming (K-means)",
    )
    soft_bars = ax.bar(
        positions + width / 2,
        soft_mean,
        width=width,
        yerr=soft_std,
        color="#fdae6b",
        capsize=3,
        label="Soft streaming (GMM)",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"step {step}" for step in hard_steps])
    _label_bars_vertical_5dp(ax, hard_bars)
    _label_bars_vertical_5dp(ax, soft_bars)

    offline_nmi = float(np.mean([entry["offline_nmi"] for entry in clustering]))
    ax.axhline(
        offline_nmi,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        label=f"Offline K-means ({offline_nmi:.5f})",
    )

    ax.set_xlabel("Refit step")
    ax.set_ylabel("NMI vs ground truth")
    ax.set_title(
        f"Hard vs soft streaming, label-rate signature  "
        f"(refit every {config.refit_every}, warmup {config.warmup})"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def plot_online_variant_panel(payload: dict[str, object], figure_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), constrained_layout=True)
    _draw_online_variant_quality_panel(axes[0], payload)
    _draw_online_variant_recovery_panel(axes[1], payload)
    _draw_online_variant_ksweep_panel(axes[2], payload)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def plot_online_variant_quality(payload: dict[str, object], figure_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.6), constrained_layout=True)
    _draw_online_variant_quality_panel(ax, payload)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def plot_online_variant_recovery(payload: dict[str, object], figure_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.6), constrained_layout=True)
    _draw_online_variant_recovery_panel(ax, payload)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def plot_online_variant_ksweep(payload: dict[str, object], figure_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.6), constrained_layout=True)
    _draw_online_variant_ksweep_panel(ax, payload)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def _online_variant_style() -> tuple[list[str], list[str], list[str]]:
    variants = ["hard_online", "hard_offline", "soft_online"]
    labels = ["CPO-hard\nonline", "CPO-hard\noffline", "CPO-soft\nonline"]
    colors = ["#6a51a3", "#2ca02c", "#fdae6b"]
    return variants, labels, colors


def _draw_online_variant_quality_panel(ax, payload: dict[str, object]) -> None:
    results = payload["results"]
    variants, labels, colors = _online_variant_style()
    values = final_values(results)
    means = [values[variant].mean() for variant in variants]
    stds = [values[variant].std() for variant in variants]
    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=4)
    _label_bars(ax, bars)
    ax.set_title("(a) Final quality")
    ax.set_ylabel("Final expected true quality")
    ax.grid(axis="y", alpha=0.2)


def _draw_online_variant_recovery_panel(ax, payload: dict[str, object]) -> None:
    recovery = payload["recovery"]
    variants, labels, _colors = _online_variant_style()
    x = np.arange(len(variants))
    width = 0.32
    nmi = [float(recovery[variant]["nmi"].mean()) for variant in variants]
    nmi_std = [float(recovery[variant]["nmi"].std()) for variant in variants]
    pur = [float(recovery[variant]["purity"].mean()) for variant in variants]
    pur_std = [float(recovery[variant]["purity"].std()) for variant in variants]
    nmi_bars = ax.bar(x - width / 2, nmi, width, yerr=nmi_std, color="indigo", capsize=3, label="NMI")
    purity_bars = ax.bar(x + width / 2, pur, width, yerr=pur_std, color="plum", capsize=3, label="Purity")
    _label_bars(ax, nmi_bars)
    _label_bars(ax, purity_bars)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.1)
    ax.set_title("(b) Cluster recovery")
    ax.set_ylabel("Score")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)


def _draw_online_variant_ksweep_panel(ax, payload: dict[str, object]) -> None:
    ksweep = payload["ksweep"]
    variants, _labels, colors = _online_variant_style()
    k_values = sorted(ksweep.keys())
    markers = {"hard_online": "s", "hard_offline": "o", "soft_online": "^"}
    line_labels = {
        "hard_online": "CPO-hard-online",
        "hard_offline": "CPO-hard-offline",
        "soft_online": "CPO-soft-online",
    }
    for variant, color in zip(variants, colors):
        mean, std = [], []
        for k in k_values:
            k_mean, k_std = _mean_std(ksweep[k][variant])
            mean.append(k_mean)
            std.append(k_std)
        ax.errorbar(
            k_values,
            mean,
            yerr=std,
            marker=markers[variant],
            color=color,
            capsize=3,
            label=line_labels[variant],
        )
    ax.set_xscale("log")
    ax.set_title("(c) K-sweep")
    ax.set_xlabel("K")
    ax.set_ylabel("Final expected true quality")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)


def plot_learned_replacement_panel(
    base_payload: dict[str, object],
    variant_payload: dict[str, object],
    variant: str,
    figure_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    _draw_replacement_quality_panel(axes[0], base_payload, variant_payload, variant)
    _draw_replacement_recovery_panel(axes[1], base_payload, variant_payload, variant)
    _draw_replacement_ksweep_panel(axes[2], base_payload, variant_payload, variant)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def _replacement_variant_names() -> dict[str, tuple[str, str, str]]:
    return {
        "hard_online": ("CPO-hard\nonline", "Hard online", "CPO-hard-online"),
        "hard_offline": ("CPO-hard\noffline", "Hard offline", "CPO-hard-offline"),
        "soft_online": ("CPO-soft\nonline", "Soft online", "CPO-soft-online"),
    }


def _draw_replacement_quality_panel(
    ax,
    base_payload: dict[str, object],
    variant_payload: dict[str, object],
    variant: str,
) -> None:
    short_label, title_label, _line_label = _replacement_variant_names()[variant]
    base_values = final_values(base_payload["results"])
    variant_values = final_values(variant_payload["results"])
    labels = ["KTO", "CPO\nrandom", short_label, "CPO\noracle"]
    colors = ["#1f77b4", "grey", "#6a51a3" if "online" in variant else "#2ca02c", "#ff7f0e"]
    series = [
        base_values["kto"],
        base_values["cpo_random"],
        variant_values[variant],
        base_values["cpo_oracle"],
    ]
    means = [values.mean() for values in series]
    stds = [values.std() for values in series]
    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=4)
    _label_bars(ax, bars)
    ax.set_title(f"(a) K=2 comparison: {title_label}")
    ax.set_ylabel("Final expected true quality")
    ax.grid(axis="y", alpha=0.2)


def _draw_replacement_recovery_panel(
    ax,
    base_payload: dict[str, object],
    variant_payload: dict[str, object],
    variant: str,
) -> None:
    short_label, _title_label, _line_label = _replacement_variant_names()[variant]
    base_clustering = base_payload["clustering"]
    variant_recovery = variant_payload["recovery"][variant]
    groups = ["Random", short_label.replace("\n", " "), "Oracle"]
    x = np.arange(len(groups))
    width = 0.32
    nmi_values = [
        base_clustering["random"]["nmi"],
        variant_recovery["nmi"],
        base_clustering["oracle"]["nmi"],
    ]
    purity_values = [
        base_clustering["random"]["purity"],
        variant_recovery["purity"],
        base_clustering["oracle"]["purity"],
    ]
    nmi = [values.mean() for values in nmi_values]
    nmi_std = [values.std() for values in nmi_values]
    pur = [values.mean() for values in purity_values]
    pur_std = [values.std() for values in purity_values]
    nmi_bars = ax.bar(x - width / 2, nmi, width, yerr=nmi_std, color="indigo", capsize=3, label="NMI")
    purity_bars = ax.bar(x + width / 2, pur, width, yerr=pur_std, color="plum", capsize=3, label="Purity")
    _label_bars(ax, nmi_bars)
    _label_bars(ax, purity_bars)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1.1)
    ax.set_title("(b) Cluster recovery")
    ax.set_ylabel("Score")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)


def _draw_replacement_ksweep_panel(
    ax,
    base_payload: dict[str, object],
    variant_payload: dict[str, object],
    variant: str,
) -> None:
    _short_label, _title_label, line_label = _replacement_variant_names()[variant]
    base_values = final_values(base_payload["results"])
    base_ksweep = base_payload["ksweep"]
    variant_ksweep = variant_payload["ksweep"]
    k_values = sorted(base_ksweep.keys())
    variant_mean, variant_std, random_mean, random_std = [], [], [], []
    for k in k_values:
        mean, std = _mean_std(variant_ksweep[k][variant])
        variant_mean.append(mean)
        variant_std.append(std)
        mean, std = _mean_std(base_ksweep[k]["random"])
        random_mean.append(mean)
        random_std.append(std)
    ax.errorbar(k_values, variant_mean, yerr=variant_std, marker="s", color="#6a51a3", capsize=3, label=line_label)
    ax.errorbar(k_values, random_mean, yerr=random_std, marker="o", color="grey", capsize=3, label="CPO random")
    ax.axhline(base_values["cpo_oracle"].mean(), color="#ff7f0e", linestyle="--", label="Oracle K=2")
    ax.axhline(base_values["kto"].mean(), color="#1f77b4", linestyle=":", label="KTO K=1")
    ax.set_xscale("log")
    ax.set_title("(c) K-sweep")
    ax.set_xlabel("K")
    ax.set_ylabel("Final expected true quality")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)


def plot_all_algorithm_ksweep(
    base_payload: dict[str, object],
    variant_payload: dict[str, object],
    figure_path: Path,
) -> None:
    width_cm, height_cm = 18.0, 15.0
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(width_cm / 2.54, height_cm / 2.54),
        constrained_layout=True,
    )
    base_values = final_values(base_payload["results"])
    base_ksweep = base_payload["ksweep"]
    variant_ksweep = variant_payload["ksweep"]
    k_values = sorted(base_ksweep.keys())

    curves = [
        ("CPO random", "random", base_ksweep, "o", "grey"),
        ("CPO learned", "learned", base_ksweep, "s", "mediumpurple"),
        ("CPO-hard-online", "hard_online", variant_ksweep, "^", "#6a51a3"),
        ("CPO-hard-offline", "hard_offline", variant_ksweep, "D", "#2ca02c"),
        ("CPO-soft-online", "soft_online", variant_ksweep, "v", "#fdae6b"),
    ]
    for label, key, sweep, marker, color in curves:
        mean = [_mean_std(sweep[k][key])[0] for k in k_values]
        ax.plot(k_values, mean, marker=marker, color=color, linewidth=1.8, label=label)

    ax.axhline(base_values["cpo_oracle"].mean(), color="#ff7f0e", linestyle="--", linewidth=1.5, label="Oracle K=2")
    ax.axhline(base_values["kto"].mean(), color="#1f77b4", linestyle=":", linewidth=1.8, label="KTO K=1")
    ax.set_xscale("log")
    ax.set_title("K-sweep across clustering variants")
    ax.set_xlabel("K")
    ax.set_ylabel("Final expected true quality")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def parse_panel_b_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run online clustering flows and write panel_b.png.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/exp_d/"),
        help="Directory for panel_b.png and the pickle cache.",
    )
    parser.add_argument(
        "--from-cache",
        type=Path,
        default=None,
        help="Plot from an existing online_flows_results.pkl instead of rerunning.",
    )
    return parser.parse_args()


def main_panel_b() -> None:
    from exp_d.online_flows import OnlineFlowConfig, run_online_flows

    args = parse_panel_b_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "online_flows_results.pkl"
    figure_path = output_dir / "panel_b.png"

    if args.from_cache is None:
        payload = run_online_flows(OnlineFlowConfig(output_dir=output_dir))
        with cache_path.open("wb") as f:
            pickle.dump(payload, f)
    else:
        with args.from_cache.open("rb") as f:
            payload = pickle.load(f)

    plot_panel_b(payload, figure_path)
    print(f"Saved figure: {figure_path}")
    if args.from_cache is None:
        print(f"Saved cache:  {cache_path}")


def parse_online_variant_panel_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace Experiment D learned clustering with online/offline variants.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/exp_d/"),
        help="Directory for the PNG figures and pickle caches.",
    )
    parser.add_argument(
        "--from-cache",
        type=Path,
        default=None,
        help="Plot from an existing online_variant_panel_results.pkl instead of rerunning.",
    )
    parser.add_argument(
        "--base-cache",
        type=Path,
        default=None,
        help="Original learned_clusters_results.pkl with KTO/random/oracle baselines.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Re-run the Experiment D and online-variant flows instead of requiring existing caches.",
    )
    return parser.parse_args()


def main_online_variant_panel() -> None:
    from exp_d.experiment import run_experiment_d
    from exp_d.online_flows import OnlineFlowConfig, run_online_variant_panel

    args = parse_online_variant_panel_args()
    output_dir = _repo_relative(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_cache_path = _repo_relative(args.base_cache) if args.base_cache is not None else output_dir / "learned_clusters_results.pkl"
    cache_path = output_dir / "online_variant_panel_results.pkl"
    figure_paths = {
        "hard_online": output_dir / "panel_b_hard_online.png",
        "hard_offline": output_dir / "panel_b_hard_offline.png",
        "soft_online": output_dir / "panel_b_soft_online.png",
        "all_ksweep": output_dir / "panel_b_all_k_sweep.png",
    }

    if base_cache_path.exists():
        with base_cache_path.open("rb") as f:
            base_payload = pickle.load(f)
    elif args.recompute:
        config = OnlineFlowConfig(output_dir=output_dir)
        base_payload = run_experiment_d(config)
        with base_cache_path.open("wb") as f:
            pickle.dump(base_payload, f)
    else:
        raise SystemExit(
            f"Missing base cache: {base_cache_path}. "
            f"Run exp_d/run.py once, or pass --recompute to regenerate it."
        )

    from_cache = _repo_relative(args.from_cache) if args.from_cache is not None else cache_path
    used_existing_variant_cache = from_cache.exists()
    if used_existing_variant_cache:
        with from_cache.open("rb") as f:
            payload = pickle.load(f)
        print(f"Used variant cache: {from_cache}")
    elif args.from_cache is not None:
        raise SystemExit(f"Missing variant cache: {from_cache}")
    else:
        payload = run_online_variant_panel(OnlineFlowConfig(output_dir=output_dir))
        with cache_path.open("wb") as f:
            pickle.dump(payload, f)
        print(f"Saved cache:  {cache_path}")

    for variant, figure_path in figure_paths.items():
        if variant == "all_ksweep":
            continue
        plot_learned_replacement_panel(base_payload, payload, variant, figure_path)
        print(f"Saved figure: {figure_path}")
    plot_all_algorithm_ksweep(base_payload, payload, figure_paths["all_ksweep"])
    print(f"Saved figure: {figure_paths['all_ksweep']}")
    print(f"Used base cache: {base_cache_path}")
