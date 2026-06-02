import argparse
import pickle
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_e.config import ExperimentEConfig
from exp_e.experiment import run_alpha_sweep, run_budget_sweep, run_pia_sweep, run_ref_ablation
from exp_e.summary import summarize


RUNNERS = {
    "budget_sweep": run_budget_sweep,
    "alpha_sweep": run_alpha_sweep,
    "pia_sweep": run_pia_sweep,
    "ref_ablation": run_ref_ablation,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPO follow-up Experiment E.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ExperimentEConfig.output_dir,
        help="Directory for Experiment E caches, summaries, and figures.",
    )
    parser.add_argument("--budget-sweep", action="store_true", help="Run the pairwise-effort budget sweep.")
    parser.add_argument("--alpha-sweep", action="store_true", help="Run the fixed-budget alpha sweep.")
    parser.add_argument("--pia-sweep", action="store_true", help="Run the pi_A heterogeneity sweep.")
    parser.add_argument("--ref-ablation", action="store_true", help="Run the global-vs-cluster reference ablation.")
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Read selected experiment caches and summaries only; do not train or write PNG figures.",
    )
    return parser.parse_args()


def _selected_names(args: argparse.Namespace) -> list[str]:
    names = []
    if args.budget_sweep:
        names.append("budget_sweep")
    if args.alpha_sweep:
        names.append("alpha_sweep")
    if args.pia_sweep:
        names.append("pia_sweep")
    if args.ref_ablation:
        names.append("ref_ablation")
    return names or list(RUNNERS)


def main() -> None:
    args = parse_args()
    config = ExperimentEConfig(output_dir=args.output_dir)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in _selected_names(args):
        cache_path = output_dir / f"{name}_results.pkl"
        summary_path = output_dir / f"{name}_summary.txt"
        figure_path = output_dir / f"{name}.png"

        if args.skip_plot:
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"cache does not exist for {name}: {cache_path}. "
                    "Run without --skip-plot first to compute it."
                )
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            print(f"Used cache: {cache_path}")
        else:
            payload = RUNNERS[name](config)
            with cache_path.open("wb") as f:
                pickle.dump(payload, f)
            print(f"Saved cache: {cache_path}")

        summary = summarize(payload)
        summary_path.write_text(summary)
        print(summary)
        print(f"Saved summary: {summary_path}")

        if not args.skip_plot:
            from exp_e.plotting import plot
            from scr.artifacts import copy_to_latex_images

            plot(payload, figure_path)
            latex_figure_path = copy_to_latex_images(figure_path)
            print(f"Saved figure: {figure_path}")
            print(f"Saved LaTeX figure: {latex_figure_path}")


if __name__ == "__main__":
    main()
