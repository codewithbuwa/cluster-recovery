import argparse
import pickle
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_d.config import ExperimentDConfig
from scr.plot_env import configure_matplotlib_cache

configure_matplotlib_cache()

from exp_d.cluster_recovery import render_all
from exp_d.experiment import run_experiment_d
from exp_d.online_flows import OnlineFlowConfig, run_online_flows, run_online_variant_panel
from scr.artifacts import copy_to_latex_and_beamer_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPO Experiment D.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ExperimentDConfig.output_dir,
        help="Directory for the PNG and pickle cache.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Run training and summary only; do not write PNG figures.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse an existing bundled pickle instead of recomputing the experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentDConfig(output_dir=args.output_dir)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "cluster_recovery_results.pkl"

    if args.use_cache and cache_path.exists():
        with cache_path.open("rb") as f:
            bundle = pickle.load(f)
        print(f"Used cache: {cache_path}")
    elif args.use_cache:
        raise FileNotFoundError(
            f"cache does not exist: {cache_path}. Run without --use-cache first to compute it."
        )
    else:
        online_config = OnlineFlowConfig(output_dir=output_dir)
        bundle = {
            "learned_clusters": run_experiment_d(config),
            "online_flows": run_online_flows(online_config),
            "online_variant_panel": run_online_variant_panel(online_config),
        }
        with cache_path.open("wb") as f:
            pickle.dump(bundle, f)
        print(f"Saved cache: {cache_path}")

    if not args.skip_plot:
        figure_paths = render_all(bundle, output_dir)
        presentation_figure_paths = [
            copied_path
            for figure_path in figure_paths
            for copied_path in copy_to_latex_and_beamer_images(figure_path)
        ]

    if not args.skip_plot:
        for presentation_figure_path in presentation_figure_paths:
            print(f"Saved presentation figure: {presentation_figure_path}")


if __name__ == "__main__":
    main()
