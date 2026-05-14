import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_c.config import ExperimentCConfig
from exp_c.experiment import _run_pair
from scr.policy import expected_quality, softmax
from scr.world import SyntheticWorld


CLUSTER_NAMES = ("Alice", "Bob")


def cluster_policy_stats(theta: np.ndarray, world: SyntheticWorld, cluster_id: int) -> dict[str, float]:
    cfg = world.config
    policy = softmax(theta)
    tau = cfg.tau_a if cluster_id == 0 else cfg.tau_b
    eps = cfg.eps_a if cluster_id == 0 else cfg.eps_b
    clean_desirable = world.q > tau
    clean_rate = float(np.mean(np.sum(policy * clean_desirable, axis=1)))
    noisy_rate = eps + (1.0 - 2.0 * eps) * clean_rate
    return {
        "e_q": expected_quality(theta, world.q),
        "clean_desirable_rate": clean_rate,
        "noisy_desirable_rate": float(noisy_rate),
    }


def summarize_runs(label: str, results: dict[str, list], world_config) -> list[str]:
    lines = [label, "-" * len(label)]
    for method, runs in results.items():
        rows = []
        for run in runs:
            if run.final_theta is None:
                raise ValueError("TrainResult is missing final_theta; rerun training with the current code.")
            world_seed = run.seed - 10_000
            world = SyntheticWorld(world_config, seed=world_seed)
            rows.append([cluster_policy_stats(run.final_theta, world, k) for k in range(2)])

        lines.append(f"{method.upper()}")
        shared_eq = np.asarray([row[0]["e_q"] for row in rows])
        lines.append(f"  shared E[q]: {shared_eq.mean():.4f} +/- {shared_eq.std():.4f}")
        for cluster_id, name in enumerate(CLUSTER_NAMES):
            clean = np.asarray([row[cluster_id]["clean_desirable_rate"] for row in rows])
            noisy = np.asarray([row[cluster_id]["noisy_desirable_rate"] for row in rows])
            lines.append(
                f"  {name}: clean desirable={clean.mean():.4f} +/- {clean.std():.4f}, "
                f"noisy label desirable={noisy.mean():.4f} +/- {noisy.std():.4f}"
            )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Alice/Bob-facing policy metrics for Experiment C2 and C4."
    )
    parser.add_argument("--seeds", type=int, default=None, help="Use seeds range(N) instead of config.seeds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentCConfig()
    seeds = tuple(range(args.seeds)) if args.seeds is not None else config.seeds

    noise_only_world = replace(
        config.world,
        pi_a=0.9,
        tau_a=0.5,
        tau_b=0.5,
        eps_a=0.3,
        eps_b=0.02,
    )
    base_world = replace(
        config.world,
        pi_a=0.9,
        tau_a=0.25,
        tau_b=0.75,
        eps_a=0.05,
        eps_b=0.05,
    )

    c2 = _run_pair(noise_only_world, config.train, seeds, log_prefix=None)
    c4 = _run_pair(base_world, config.train, seeds, log_prefix=None, cluster_mode="random")

    lines = [
        "Note: raw E[q] is shared because q is shared and the policy is not cluster-conditioned.",
        "Cluster-specific rows report how often each cluster would call the policy's samples desirable.",
        "",
    ]
    lines.extend(summarize_runs("C2 noise-only heterogeneity", c2, noise_only_world))
    lines.append("")
    lines.extend(summarize_runs("C4 random clusters", c4, base_world))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
