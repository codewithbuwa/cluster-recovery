from dataclasses import replace

from exp_c.config import ExperimentCConfig
from scr.training import TrainResult, train_method
from scr.world import SyntheticWorld


REFERENCE_VARIANTS = {
    "r|U": "undesirable",
    "r": "all",
    "KL": "kl",
}


def _run_pair(
    world_config,
    train_config,
    seeds: tuple[int, ...],
    log_prefix: str | None = None,
    **train_kwargs,
) -> dict[str, list[TrainResult]]:
    results = {"kto": [], "cpo": []}
    for seed in seeds:
        world = SyntheticWorld(world_config, seed=seed)
        train_seed = 10_000 + seed
        results["kto"].append(
            train_method(
                world,
                method="kto",
                seed=train_seed,
                train_config=train_config,
                log_prefix=log_prefix,
                **train_kwargs,
            )
        )
        results["cpo"].append(
            train_method(
                world,
                method="cpo",
                seed=train_seed,
                train_config=train_config,
                log_prefix=log_prefix,
                **train_kwargs,
            )
        )
    return results


def run_experiment_c(config: ExperimentCConfig) -> dict[str, object]:
    base_world = replace(
        config.world,
        pi_a=0.9,
        tau_a=0.25,
        tau_b=0.75,
        eps_a=0.05,
        eps_b=0.05,
    )

    c1 = {}
    for label, reference_variant in REFERENCE_VARIANTS.items():
        c1[label] = _run_pair(
            base_world,
            config.train,
            config.seeds,
            log_prefix=f"ExpC C1 reference={label}",
            reference_variant=reference_variant,
        )

    noise_only_world = replace(
        config.world,
        pi_a=0.9,
        tau_a=0.5,
        tau_b=0.5,
        eps_a=0.3,
        eps_b=0.02,
    )
    c2 = _run_pair(
        noise_only_world,
        config.train,
        config.seeds,
        log_prefix="ExpC C2 noise_only",
        record_grad_weights=True,
        record_references=True,
    )

    c3 = {}
    for beta in config.beta_values:
        beta_train = replace(config.train, beta=beta)
        c3[beta] = _run_pair(base_world, beta_train, config.c3_seeds, log_prefix=f"ExpC C3 beta={beta:g}")

    c4 = _run_pair(base_world, config.train, config.seeds, log_prefix="ExpC C4 random_cluster", cluster_mode="random")

    return {
        "base_world_config": base_world,
        "noise_only_world_config": noise_only_world,
        "train_config": config.train,
        "seeds": config.seeds,
        "c3_seeds": config.c3_seeds,
        "beta_values": config.beta_values,
        "results": {
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "c4": c4,
        },
    }
