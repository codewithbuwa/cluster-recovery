import argparse
import pickle
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_c.plotting import plot_experiment_c


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Experiment C ablations from pickle.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/exp_c/ablations_results.pkl"),
        help="Path to the saved Experiment C pickle payload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/exp_c/ablations.png"),
        help="Path where the ablations PNG should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("rb") as f:
        payload = pickle.load(f)
    plot_experiment_c(payload, args.output)
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
