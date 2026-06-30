import argparse
import pickle
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_c.config import ExperimentCConfig
from exp_c.experiment import run_experiment_c
from scr.plot_env import configure_matplotlib_cache

configure_matplotlib_cache()

from exp_c.plotting import plot_experiment_c
from exp_c.summary import summarize
from scr.artifacts import copy_to_latex_and_beamer_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPO Experiment C.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ExperimentCConfig.output_dir,
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
    config = ExperimentCConfig(output_dir=args.output_dir)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "ablations_results.pkl"
    figure_path = output_dir / "ablations.png"
    payload = run_experiment_c(config)
    with cache_path.open("wb") as f:
        pickle.dump(payload, f)

    if not args.skip_plot:
        plot_experiment_c(payload, figure_path)
        presentation_figure_paths = copy_to_latex_and_beamer_images(figure_path)

    print(summarize(payload))
    print(f"Saved cache: {cache_path}")
    if not args.skip_plot:
        print(f"Saved figure: {figure_path}")
        for presentation_figure_path in presentation_figure_paths:
            print(f"Saved presentation figure: {presentation_figure_path}")


if __name__ == "__main__":
    main()
