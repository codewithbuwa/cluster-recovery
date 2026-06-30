import argparse
import pickle
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scr.plot_env import configure_matplotlib_cache

configure_matplotlib_cache()

from exp_b.plotting import plot_experiment_b
from scr.artifacts import copy_to_latex_and_beamer_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Experiment B pi_A sweep from pickle.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/exp_b/pi_sweep_results.pkl"),
        help="Path to the saved Experiment B pickle payload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/exp_b/pi_sweep.png"),
        help="Path where the pi sweep PNG should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("rb") as f:
        payload = pickle.load(f)
    plot_experiment_b(payload, args.output)
    presentation_paths = copy_to_latex_and_beamer_images(args.output)
    print(f"Saved figure: {args.output}")
    for presentation_path in presentation_paths:
        print(f"Saved presentation figure: {presentation_path}")


if __name__ == "__main__":
    main()
