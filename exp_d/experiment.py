import numpy as np

from exp_d.clustering import compute_signatures, kmeans_1d, normalized_mutual_information, purity, random_partition
from exp_d.config import ExperimentDConfig
from scr.samplers import OfflineDataset
from scr.training import TrainResult, train_offline_method
from scr.world import SyntheticWorld


def _run_offline(
    config: ExperimentDConfig,
    n_clusters: int,
    cluster_builder,
    method: str = "cpo",
    log_prefix: str | None = None,
) -> list[TrainResult]:
    runs = []
    for seed in config.seeds:
        world = SyntheticWorld(config.world, seed)
        dataset = OfflineDataset(world, config.world, seed)
        annotator_clusters = None if method == "kto" else cluster_builder(dataset, seed, n_clusters)
        runs.append(
            train_offline_method(
                world,
                dataset,
                method=method,
                seed=10_000 + seed,
                train_config=config.train,
                n_clusters=n_clusters,
                annotator_clusters=annotator_clusters,
                log_prefix=log_prefix,
            )
        )
    return runs


def run_experiment_d(config: ExperimentDConfig) -> dict[str, object]:
    results = {
        "kto": _run_offline(config, 1, lambda dataset, seed, k: None, method="kto", log_prefix="ExpD kto"),
        "cpo_random": _run_offline(
            config,
            2,
            lambda dataset, seed, k: random_partition(config.world.n_annotators, k, seed),
            log_prefix="ExpD cpo_random",
        ),
        "cpo_learned": _run_offline(
            config,
            2,
            lambda dataset, seed, k: kmeans_1d(compute_signatures(dataset), k, seed),
            log_prefix="ExpD cpo_learned",
        ),
        "cpo_oracle": _run_offline(
            config,
            2,
            lambda dataset, seed, k: dataset.annotator_cluster.copy(),
            log_prefix="ExpD cpo_oracle",
        ),
    }

    clustering = {"random": {"nmi": [], "purity": []}, "learned": {"nmi": [], "purity": []}, "oracle": {"nmi": [], "purity": []}}
    for seed in config.seeds:
        world = SyntheticWorld(config.world, seed)
        dataset = OfflineDataset(world, config.world, seed)
        truth = dataset.annotator_cluster
        predicted = {
            "random": random_partition(config.world.n_annotators, 2, seed),
            "learned": kmeans_1d(compute_signatures(dataset), 2, seed),
            "oracle": truth.copy(),
        }
        for key, labels in predicted.items():
            clustering[key]["nmi"].append(normalized_mutual_information(labels, truth))
            clustering[key]["purity"].append(purity(labels, truth))
    for key in clustering:
        clustering[key]["nmi"] = np.asarray(clustering[key]["nmi"])
        clustering[key]["purity"] = np.asarray(clustering[key]["purity"])

    ksweep = {}
    for n_clusters in config.k_values:
        if n_clusters == 1:
            kto_runs = _run_offline(
                config,
                1,
                lambda dataset, seed, k: None,
                method="kto",
                log_prefix=f"ExpD ksweep K={n_clusters} kto",
            )
            ksweep[n_clusters] = {"learned": kto_runs, "random": kto_runs}
        else:
            ksweep[n_clusters] = {
                "learned": _run_offline(
                    config,
                    n_clusters,
                    lambda dataset, seed, k: kmeans_1d(compute_signatures(dataset), k, seed),
                    log_prefix=f"ExpD ksweep K={n_clusters} learned",
                ),
                "random": _run_offline(
                    config,
                    n_clusters,
                    lambda dataset, seed, k: random_partition(config.world.n_annotators, k, seed),
                    log_prefix=f"ExpD ksweep K={n_clusters} random",
                ),
            }

    return {
        "world_config": config.world,
        "train_config": config.train,
        "seeds": config.seeds,
        "k_values": config.k_values,
        "results": results,
        "clustering": clustering,
        "ksweep": ksweep,
    }
