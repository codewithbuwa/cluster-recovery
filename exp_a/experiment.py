import numpy as np

from exp_a.config import ExperimentAConfig
from scr.training import TrainResult, train_method
from scr.world import SyntheticWorld


def run_experiment_a(config: ExperimentAConfig) -> dict[str, object]:
    results: dict[str, list[TrainResult]] = {"kto": [], "cpo": [], "oracle_bob_only": []}
    diagnostics = {}

    for seed in config.seeds:
        world = SyntheticWorld(config.world, seed=seed)
        train_seed = 10_000 + seed

        kto = train_method(
            world,
            method="kto",
            seed=train_seed,
            train_config=config.train,
            record_grad_weights=True,
            log_prefix="ExpA",
        )
        cpo = train_method(
            world,
            method="cpo",
            seed=train_seed,
            train_config=config.train,
            record_grad_weights=True,
            log_prefix="ExpA",
        )
        oracle = train_method(
            world,
            method="kto",
            seed=50_000 + seed,
            train_config=config.train,
            force_cluster=1,
            log_prefix="ExpA oracle_bob_only",
        )

        results["kto"].append(kto)
        results["cpo"].append(cpo)
        results["oracle_bob_only"].append(oracle)

    diagnostics["kto_grad_weight_by_cluster"] = np.stack(
        [result.grad_weight_by_cluster for result in results["kto"]]
    )
    diagnostics["cpo_grad_weight_by_cluster"] = np.stack(
        [result.grad_weight_by_cluster for result in results["cpo"]]
    )

    return {
        "world_config": config.world,
        "train_config": config.train,
        "seeds": config.seeds,
        "results": results,
        "diagnostics": diagnostics,
    }
