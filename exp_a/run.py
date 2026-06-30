import argparse
import pickle
import sys
from dataclasses import replace
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_a.config import ExperimentAConfig
from exp_a.experiment import run_experiment_a
from scr.plot_env import configure_matplotlib_cache

configure_matplotlib_cache()

from exp_a.plotting import plot_experiment_a
from exp_a.summary import summarize
from scr.artifacts import copy_to_latex_and_beamer_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPO Experiment A.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the PNG, pickle cache, and text summary.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="?",
        const=0.5,
        default=None,
        help="Enable the mixed unary/pairwise extension with this loss weight; bare --alpha uses 0.5.",
    )
    parser.add_argument(
        "--pair-fraction",
        type=float,
        default=None,
        help="Fraction of label-equivalent effort spent on pairwise labels when --alpha is set.",
    )
    parser.add_argument(
        "--total-effort",
        type=int,
        default=None,
        help="Total label-equivalent samples per step when --alpha is set.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Run training and summary only; do not write the PNG figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = ExperimentAConfig()
    train_config = base_config.train
    output_dir = args.output_dir
    if args.alpha is not None:
        if not 0.0 <= args.alpha <= 1.0:
            raise ValueError(f"--alpha must be in [0, 1], got {args.alpha}")
        pair_fraction = 0.5 if args.pair_fraction is None else args.pair_fraction
        total_effort = args.total_effort or 2 * train_config.batch_size
        train_config = replace(
            train_config,
            alpha=args.alpha,
            pair_fraction=pair_fraction,
            total_effort=total_effort,
        )
        if output_dir is None:
            alpha_label = f"{args.alpha:g}".replace(".", "p")
            output_dir = Path(f"outputs/exp_a_alpha_{alpha_label}/")

    if output_dir is None:
        output_dir = ExperimentAConfig.output_dir

    config = ExperimentAConfig(train=train_config, output_dir=output_dir)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "grad_weight_results.pkl"
    figure_path = output_dir / "grad_weight_diagnostic.png"
    payload = run_experiment_a(config)
    with cache_path.open("wb") as f:
        pickle.dump(payload, f)

    if not args.skip_plot:
        plot_experiment_a(payload, figure_path)
        presentation_figure_paths = copy_to_latex_and_beamer_images(figure_path)

    summary = summarize(payload)
    print(summary)
    print(f"Saved cache: {cache_path}")
    if not args.skip_plot:
        print(f"Saved figure: {figure_path}")
        for presentation_figure_path in presentation_figure_paths:
            print(f"Saved presentation figure: {presentation_figure_path}")


if __name__ == "__main__":
    main()
