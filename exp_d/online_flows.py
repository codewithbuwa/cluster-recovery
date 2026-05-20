from dataclasses import dataclass
from pathlib import Path

import numpy as np

from exp_d.clustering import compute_signatures, kmeans_1d, normalized_mutual_information, purity
from exp_d.config import ExperimentDConfig
from scr.samplers import OfflineDataset
from scr.training import TrainResult, train_offline_method
from scr.training_online import OnlineTrainResult, train_online_hard, train_online_soft
from scr.world import SyntheticWorld


@dataclass(frozen=True)
class OnlineFlowConfig(ExperimentDConfig):
    warmup: int = 25
    refit_every: int = 50
    output_dir: Path = Path("outputs/exp_d/")


def run_online_flows(config: OnlineFlowConfig) -> dict[str, object]:
    hard: list[OnlineTrainResult] = []
    soft: list[OnlineTrainResult] = []
    refit_logs: list[dict[str, list[dict[str, float]]]] = []
    clustering: list[dict[str, float]] = []
    for seed in config.seeds:
        world = SyntheticWorld(config.world, seed)
        hard_dataset = OfflineDataset(world, config.world, seed)
        soft_dataset = OfflineDataset(world, config.world, seed)
        train_seed = 10_000 + seed
        offline_labels = kmeans_1d(compute_signatures(hard_dataset), config.world.n_clusters, seed)
        clustering.append(
            {
                "offline_nmi": normalized_mutual_information(offline_labels, hard_dataset.annotator_cluster),
                "offline_purity": purity(offline_labels, hard_dataset.annotator_cluster),
            }
        )
        hard_result = train_online_hard(
            world,
            hard_dataset,
            seed=train_seed,
            train_config=config.train,
            n_clusters=config.world.n_clusters,
            warmup=config.warmup,
            refit_every=config.refit_every,
            log_prefix="ExpD-online",
        )
        soft_result = train_online_soft(
            world,
            soft_dataset,
            seed=train_seed,
            train_config=config.train,
            n_clusters=config.world.n_clusters,
            warmup=config.warmup,
            refit_every=config.refit_every,
            log_prefix="ExpD-online",
        )
        hard.append(hard_result)
        soft.append(soft_result)
        refit_logs.append(
            {
                "hard": _refit_entries(hard_result),
                "soft": _refit_entries(soft_result),
            }
        )

    return {
        "config": config,
        "world_config": config.world,
        "train_config": config.train,
        "seeds": config.seeds,
        "warmup": config.warmup,
        "refit_every": config.refit_every,
        "refit_logs": refit_logs,
        "clustering": clustering,
        "results": {
            "hard_online": hard,
            "soft_online": soft,
        },
    }


def run_online_variant_panel(config: OnlineFlowConfig) -> dict[str, object]:
    results: dict[str, list[TrainResult]] = {
        "hard_online": [],
        "hard_offline": [],
        "soft_online": [],
    }
    recovery: dict[str, dict[str, list[float]]] = {
        "hard_online": {"nmi": [], "purity": []},
        "hard_offline": {"nmi": [], "purity": []},
        "soft_online": {"nmi": [], "purity": []},
    }

    for seed in config.seeds:
        seed_payload = _run_seed_variants(config, seed, config.world.n_clusters)
        for key in results:
            results[key].append(seed_payload[key].train_result if isinstance(seed_payload[key], OnlineTrainResult) else seed_payload[key])
        recovery["hard_online"]["nmi"].append(float(seed_payload["hard_online"].nmi[-1]))
        recovery["hard_online"]["purity"].append(float(seed_payload["hard_online"].purity[-1]))
        recovery["soft_online"]["nmi"].append(float(seed_payload["soft_online"].nmi[-1]))
        recovery["soft_online"]["purity"].append(float(seed_payload["soft_online"].purity[-1]))
        recovery["hard_offline"]["nmi"].append(float(seed_payload["hard_offline_nmi"]))
        recovery["hard_offline"]["purity"].append(float(seed_payload["hard_offline_purity"]))

    recovery_arrays = {
        variant: {metric: np.asarray(values) for metric, values in metrics.items()}
        for variant, metrics in recovery.items()
    }

    ksweep: dict[int, dict[str, list[TrainResult]]] = {}
    for n_clusters in config.k_values:
        ksweep[n_clusters] = {"hard_online": [], "hard_offline": [], "soft_online": []}
        for seed in config.seeds:
            seed_payload = _run_seed_variants(config, seed, n_clusters)
            ksweep[n_clusters]["hard_online"].append(seed_payload["hard_online"].train_result)
            ksweep[n_clusters]["hard_offline"].append(seed_payload["hard_offline"])
            ksweep[n_clusters]["soft_online"].append(seed_payload["soft_online"].train_result)

    return {
        "config": config,
        "world_config": config.world,
        "train_config": config.train,
        "seeds": config.seeds,
        "k_values": config.k_values,
        "results": results,
        "recovery": recovery_arrays,
        "ksweep": ksweep,
    }


def _run_seed_variants(config: OnlineFlowConfig, seed: int, n_clusters: int) -> dict[str, object]:
    world = SyntheticWorld(config.world, seed)
    hard_online_dataset = OfflineDataset(world, config.world, seed)
    soft_online_dataset = OfflineDataset(world, config.world, seed)
    hard_offline_dataset = OfflineDataset(world, config.world, seed)
    train_seed = 10_000 + seed

    hard_online = train_online_hard(
        world,
        hard_online_dataset,
        seed=train_seed,
        train_config=config.train,
        n_clusters=n_clusters,
        warmup=config.warmup,
        refit_every=config.refit_every,
        log_prefix=f"ExpD-panel K={n_clusters}",
    )
    soft_online = train_online_soft(
        world,
        soft_online_dataset,
        seed=train_seed,
        train_config=config.train,
        n_clusters=n_clusters,
        warmup=config.warmup,
        refit_every=config.refit_every,
        log_prefix=f"ExpD-panel K={n_clusters}",
    )
    labels = kmeans_1d(compute_signatures(hard_offline_dataset), n_clusters, seed)
    hard_offline = train_offline_method(
        world,
        hard_offline_dataset,
        method="cpo",
        seed=train_seed,
        train_config=config.train,
        n_clusters=n_clusters,
        annotator_clusters=labels,
        log_prefix=f"ExpD-panel K={n_clusters} hard_offline",
    )
    return {
        "hard_online": hard_online,
        "hard_offline": hard_offline,
        "soft_online": soft_online,
        "hard_offline_nmi": normalized_mutual_information(labels, hard_offline_dataset.annotator_cluster),
        "hard_offline_purity": purity(labels, hard_offline_dataset.annotator_cluster),
    }


def _refit_entries(result: OnlineTrainResult) -> list[dict[str, float]]:
    return [
        {
            "step": int(step),
            "nmi": float(nmi),
            "purity": float(purity_value),
            "coverage": float(coverage),
        }
        for step, nmi, purity_value, coverage in zip(
            result.refit_steps,
            result.nmi,
            result.purity,
            result.coverage,
        )
    ]
