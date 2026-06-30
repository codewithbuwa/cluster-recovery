import argparse
import pickle
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_a.plotting import plot_experiment_a
from scr.artifacts import copy_to_latex_and_beamer_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Experiment A grad-weight diagnostic from pickle.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/exp_a/grad_weight_results.pkl"),
        help="Path to the saved Experiment A pickle payload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/exp_a/grad_weight_diagnostic.png"),
        help="Path where the grad-weight diagnostic PNG should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("rb") as f:
        payload = pickle.load(f)
    plot_experiment_a(payload, args.output)
    presentation_paths = copy_to_latex_and_beamer_images(args.output)
    print(f"Saved figure: {args.output}")
    for presentation_path in presentation_paths:
        print(f"Saved presentation figure: {presentation_path}")


if __name__ == "__main__":
    main()
