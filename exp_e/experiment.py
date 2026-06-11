from dataclasses import replace

from exp_e.config import ExperimentEConfig
from scr.config import TrainConfig, WorldConfig
from scr.training import TrainResult, train_method, valid_budget_sweep_cell
from scr.world import SyntheticWorld


METHOD_SPECS = {
    "kto": ("kto", 0.0),
    "cpo": ("cpo", 0.0),
    "mixed_cpo": ("cpo", 0.5),
    "dpo": ("dpo", 1.0),
}


def _with_budget(train: TrainConfig, alpha: float, n_unary: int, n_pair: int) -> TrainConfig:
    total_effort = n_unary + 2 * n_pair
    pair_fraction = 0.0 if total_effort == 0 else (2 * n_pair) / total_effort
    return replace(
        train,
        alpha=alpha,
        pair_fraction=pair_fraction,
        total_effort=total_effort,
    )


def _budget_counts(effort: int, pair_fraction: float) -> tuple[int, int]:
    n_unary = int((1.0 - pair_fraction) * effort)
    n_pair = int(pair_fraction * effort / 2.0)
    return n_unary, n_pair


def _run_method(
    world_config: WorldConfig,
    train_config: TrainConfig,
    method: str,
    seeds: tuple[int, ...],
    log_prefix: str | None = None,
) -> list[TrainResult]:
    runs = []
    for seed in seeds:
        world = SyntheticWorld(world_config, seed=seed)
        runs.append(
            train_method(
                world,
                method=method,
                seed=10_000 + seed,
                train_config=train_config,
                record_references=True,
                log_prefix=log_prefix,
            )
        )
    return runs


def run_budget_sweep(config: ExperimentEConfig) -> dict[str, object]:
    results: dict[float, dict[str, list[TrainResult] | None]] = {}
    for pair_fraction in config.pair_fraction_values:
        n_unary, n_pair = _budget_counts(config.effort, pair_fraction)
        results[pair_fraction] = {}
        for label, (method, alpha) in METHOD_SPECS.items():
            train = _with_budget(config.train, alpha=alpha, n_unary=n_unary, n_pair=n_pair)
            if not valid_budget_sweep_cell(method, train):
                results[pair_fraction][label] = None
                continue

            results[pair_fraction][label] = _run_method(
                config.world,
                train,
                method,
                config.budget_seeds,
                log_prefix=f"BUDGET SWEEP f={pair_fraction:g}",
            )

    return {
        "name": "budget_sweep",
        "world_config": config.world,
        "train_config": config.train,
        "seeds": config.budget_seeds,
        "effort": config.effort,
        "pair_fraction_values": config.pair_fraction_values,
        "results": results,
    }


def run_alpha_sweep(config: ExperimentEConfig) -> dict[str, object]:
    results = {}
    for alpha in config.alpha_values:
        n_unary = 0 if alpha == 1.0 else config.fixed_n_unary
        n_pair = 0 if alpha == 0.0 else config.fixed_n_pair
        train = _with_budget(config.train, alpha=alpha, n_unary=n_unary, n_pair=n_pair)
        method = "dpo" if alpha == 1.0 else "cpo"
        results[alpha] = _run_method(
            config.world,
            train,
            method,
            config.alpha_seeds,
            log_prefix=f"ALPHA SWEEP alpha={alpha:g}",
        )

    return {
        "name": "alpha_sweep",
        "world_config": config.world,
        "train_config": config.train,
        "seeds": config.alpha_seeds,
        "alpha_values": config.alpha_values,
        "fixed_n_unary": config.fixed_n_unary,
        "fixed_n_pair": config.fixed_n_pair,
        "results": results,
    }


def run_alpha_pair_sweep(config: ExperimentEConfig) -> dict[str, object]:
    results: dict[int, dict[float, list[TrainResult] | None]] = {}
    for n_pair in config.pair_budget_values:
        results[n_pair] = {}
        for alpha in config.alpha_pair_values:
            train = _with_budget(
                config.train,
                alpha=alpha,
                n_unary=config.fixed_n_unary,
                n_pair=n_pair,
            )
            method = "dpo" if alpha == 1.0 else "cpo"
            results[n_pair][alpha] = _run_method(
                config.world,
                train,
                method,
                config.alpha_pair_seeds,
                log_prefix=f"ALPHA-PAIR SWEEP N_pair={n_pair} alpha={alpha:g}",
            )

    return {
        "name": "alpha_pair_sweep",
        "world_config": config.world,
        "train_config": config.train,
        "seeds": config.alpha_pair_seeds,
        "alpha_values": config.alpha_pair_values,
        "pair_budget_values": config.pair_budget_values,
        "fixed_n_unary": config.fixed_n_unary,
        "results": results,
    }


def run_pia_sweep(config: ExperimentEConfig) -> dict[str, object]:
    results = {}
    for pi_a in config.pi_a_values:
        world_config = replace(config.world, pi_a=pi_a, tau_a=0.25, tau_b=0.75, eps_a=0.05, eps_b=0.05)
        results[pi_a] = {}
        method_configs = {
            "kto": ("kto", _with_budget(config.train, alpha=0.0, n_unary=config.effort, n_pair=0)),
            "cpo": ("cpo", _with_budget(config.train, alpha=0.0, n_unary=config.effort, n_pair=0)),
            "mixed_cpo": (
                "cpo",
                _with_budget(config.train, alpha=0.5, n_unary=config.fixed_n_unary, n_pair=config.fixed_n_pair),
            ),
            "dpo": ("dpo", _with_budget(config.train, alpha=1.0, n_unary=0, n_pair=config.effort // 2)),
        }
        for label, (method, train) in method_configs.items():
            results[pi_a][label] = _run_method(
                world_config,
                train,
                method,
                config.pia_seeds,
                log_prefix=f"PI_A SWEEP pi_A={pi_a:g}",
            )

    return {
        "name": "pia_sweep",
        "base_world_config": config.world,
        "train_config": config.train,
        "seeds": config.pia_seeds,
        "pi_a_values": config.pi_a_values,
        "effort": config.effort,
        "fixed_n_unary": config.fixed_n_unary,
        "fixed_n_pair": config.fixed_n_pair,
        "results": results,
    }


def run_ref_ablation(config: ExperimentEConfig) -> dict[str, object]:
    cells = {
        "global_alpha0": (
            "kto",
            _with_budget(
                config.train,
                alpha=0.0,
                n_unary=config.fixed_n_unary,
                n_pair=config.fixed_n_pair,
            ),
        ),
        "cluster_alpha0": (
            "cpo",
            _with_budget(
                config.train,
                alpha=0.0,
                n_unary=config.fixed_n_unary,
                n_pair=config.fixed_n_pair,
            ),
        ),
        "global_alpha05": (
            "kto",
            _with_budget(config.train, alpha=0.5, n_unary=config.fixed_n_unary, n_pair=config.fixed_n_pair),
        ),
        "cluster_alpha05": (
            "cpo",
            _with_budget(config.train, alpha=0.5, n_unary=config.fixed_n_unary, n_pair=config.fixed_n_pair),
        ),
    }
    secondary_n_pair = 8
    secondary_cells = {
        "global_alpha05": (
            "kto",
            _with_budget(
                config.train,
                alpha=0.5,
                n_unary=config.fixed_n_unary,
                n_pair=secondary_n_pair,
            ),
        ),
        "cluster_alpha05": (
            "cpo",
            _with_budget(
                config.train,
                alpha=0.5,
                n_unary=config.fixed_n_unary,
                n_pair=secondary_n_pair,
            ),
        ),
    }
    results = {
        label: _run_method(config.world, train, method, config.reference_seeds, log_prefix=f"REF ABLATION")
        for label, (method, train) in cells.items()
    }
    secondary_results = {
        label: _run_method(
            config.world,
            train,
            method,
            config.reference_seeds,
            log_prefix="REF ABLATION N_pair=8",
        )
        for label, (method, train) in secondary_cells.items()
    }
    return {
        "name": "ref_ablation",
        "world_config": config.world,
        "train_config": config.train,
        "seeds": config.reference_seeds,
        "fixed_n_unary": config.fixed_n_unary,
        "fixed_n_pair": config.fixed_n_pair,
        "nominal_primary_counts": {
            "n_unary": config.fixed_n_unary,
            "n_pair": config.fixed_n_pair,
        },
        "secondary_n_pair": secondary_n_pair,
        "results": results,
        "secondary_results": secondary_results,
    }
