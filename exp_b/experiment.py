from dataclasses import replace

from exp_b.config import ExperimentBConfig
from scr.training import TrainResult, train_method
from scr.world import SyntheticWorld


def run_experiment_b(config: ExperimentBConfig) -> dict[str, object]:
    results: dict[float, dict[str, list[TrainResult]]] = {}

    for pi_a in config.pi_a_values:
        world_config = replace(
            config.world,
            pi_a=pi_a,
            tau_a=0.25,
            tau_b=0.75,
            eps_a=0.05,
            eps_b=0.05,
        )
        results[pi_a] = {"kto": [], "cpo": []}

        for seed in config.seeds:
            world = SyntheticWorld(world_config, seed=seed)
            train_seed = 10_000 + seed

            results[pi_a]["kto"].append(
                train_method(
                    world,
                    method="kto",
                    seed=train_seed,
                    train_config=config.train,
                    log_prefix=f"ExpB pi_A={pi_a:g}",
                )
            )
            results[pi_a]["cpo"].append(
                train_method(
                    world,
                    method="cpo",
                    seed=train_seed,
                    train_config=config.train,
                    log_prefix=f"ExpB pi_A={pi_a:g}",
                )
            )

    return {
        "base_world_config": config.world,
        "train_config": config.train,
        "seeds": config.seeds,
        "pi_a_values": config.pi_a_values,
        "results": results,
    }
