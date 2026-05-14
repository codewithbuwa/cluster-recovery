import argparse
import pickle
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_a.config import ExperimentAConfig
from exp_a.experiment import run_experiment_a
from exp_a.plotting import plot_experiment_a
from exp_a.summary import summarize
from scr.artifacts import copy_to_latex_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPO Experiment A.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ExperimentAConfig.output_dir,
        help="Directory for the PNG, pickle cache, and text summary.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Run training and summary only; do not write the PNG figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentAConfig(output_dir=args.output_dir)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "grad_weight_results.pkl"
    figure_path = output_dir / "grad_weight_diagnostic.png"
    payload = run_experiment_a(config)
    with cache_path.open("wb") as f:
        pickle.dump(payload, f)

    if not args.skip_plot:
        plot_experiment_a(payload, figure_path)
        latex_figure_path = copy_to_latex_images(figure_path)

    summary = summarize(payload)
    print(summary)
    print(f"Saved cache: {cache_path}")
    if not args.skip_plot:
        print(f"Saved figure: {figure_path}")
        print(f"Saved LaTeX figure: {latex_figure_path}")


if __name__ == "__main__":
    main()
