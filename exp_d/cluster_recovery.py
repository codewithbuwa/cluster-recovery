import argparse
import pickle
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_d.plotting import (
    plot_all_algorithm_ksweep,
    plot_experiment_d,
    plot_learned_replacement_panel,
    plot_panel_b,
)
from scr.artifacts import copy_to_latex_and_beamer_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Experiment D cluster-recovery figures from one bundled pickle.")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("outputs/exp_d/cluster_recovery_results.pkl"),
        help="Path to the bundled Exp D cluster-recovery pickle payload.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/exp_d"),
        help="Directory where Exp D PNG figures should be written.",
    )
    return parser.parse_args()


def _load_pickle(path: Path) -> object:
    with path.open("rb") as f:
        return pickle.load(f)


def load_bundle(path: Path) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"Missing bundled pickle: {path}")
    payload = _load_pickle(path)
    print(f"Used bundle: {path}")
    return payload


def render_all(payload: dict[str, object], output_dir: Path) -> list[Path]:
    learned_payload = payload["learned_clusters"]
    online_flows_payload = payload["online_flows"]
    online_variant_payload = payload["online_variant_panel"]

    figure_paths = {
        "learned_clusters": output_dir / "learned_clusters.png",
        "panel_b": output_dir / "panel_b.png",
        "hard_online": output_dir / "panel_b_hard_online.png",
        "hard_offline": output_dir / "panel_b_hard_offline.png",
        "soft_online": output_dir / "panel_b_soft_online.png",
        "all_ksweep": output_dir / "panel_b_all_k_sweep.png",
    }

    plot_experiment_d(learned_payload, figure_paths["learned_clusters"])
    print(f"Saved figure: {figure_paths['learned_clusters']}")

    plot_panel_b(online_flows_payload, figure_paths["panel_b"])
    print(f"Saved figure: {figure_paths['panel_b']}")

    for variant in ("hard_online", "hard_offline", "soft_online"):
        plot_learned_replacement_panel(
            learned_payload,
            online_variant_payload,
            variant,
            figure_paths[variant],
        )
        print(f"Saved figure: {figure_paths[variant]}")

    plot_all_algorithm_ksweep(
        learned_payload,
        online_variant_payload,
        figure_paths["all_ksweep"],
    )
    print(f"Saved figure: {figure_paths['all_ksweep']}")
    return list(figure_paths.values())


def main() -> None:
    args = parse_args()
    payload = load_bundle(args.bundle)
    for figure_path in render_all(payload, args.output_dir):
        for presentation_path in copy_to_latex_and_beamer_images(figure_path):
            print(f"Saved presentation figure: {presentation_path}")


if __name__ == "__main__":
    main()
