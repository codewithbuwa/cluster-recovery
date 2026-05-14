import argparse
import pickle
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_b.config import ExperimentBConfig
from exp_b.experiment import run_experiment_b
from exp_b.plotting import plot_experiment_b
from exp_b.summary import summarize
from scr.artifacts import copy_to_latex_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPO Experiment B.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ExperimentBConfig.output_dir,
        help="Directory for the PNG and pickle cache.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Run training and summary only; do not write the PNG figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentBConfig(output_dir=args.output_dir)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "pi_sweep_results.pkl"
    figure_path = output_dir / "pi_sweep.png"

    payload = run_experiment_b(config)
    with cache_path.open("wb") as f:
        pickle.dump(payload, f)

    if not args.skip_plot:
        plot_experiment_b(payload, figure_path)
        latex_figure_path = copy_to_latex_images(figure_path)

    print(summarize(payload))
    print(f"Saved cache: {cache_path}")
    if not args.skip_plot:
        print(f"Saved figure: {figure_path}")
        print(f"Saved LaTeX figure: {latex_figure_path}")


if __name__ == "__main__":
    main()
