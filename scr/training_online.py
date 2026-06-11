from dataclasses import dataclass

import numpy as np

from exp_d.clustering import normalized_mutual_information, purity
from exp_d.clustering_online import OnlineHardClusterer, OnlineSoftClusterer
from exp_d.clustering_soft import soft_normalized_mutual_information, soft_purity
from scr.config import TrainConfig
from scr.policy import expected_quality, softmax
from scr.samplers import OfflineDataset
from scr.training import (
    TrainResult,
    _adam_update,
    _cluster_count_array,
    _expected_desirability_by_cluster,
    _expected_quality_per_prompt,
    _loss_grad,
)
from scr.world import SyntheticWorld


@dataclass
class OnlineTrainResult:
    train_result: TrainResult
    refit_steps: np.ndarray
    nmi: np.ndarray
    purity: np.ndarray
    coverage: np.ndarray


def train_online_hard(
    world: SyntheticWorld,
    dataset: OfflineDataset,
    seed: int,
    train_config: TrainConfig,
    n_clusters: int = 2,
    warmup: int = 25,
    refit_every: int = 50,
    log_prefix: str | None = None,
) -> OnlineTrainResult:
    cfg = world.config
    clusterer = OnlineHardClusterer(cfg.n_annotators, n_clusters, seed)
    theta = np.zeros((cfg.n_prompts, cfg.n_responses), dtype=np.float64)
    adam_m = np.zeros_like(theta)
    adam_v = np.zeros_like(theta)
    ref = _HardReference(n_clusters)

    eval_steps: list[int] = []
    eval_quality: list[float] = []
    eval_quality_per_prompt: list[np.ndarray] = []
    eval_desirability_by_cluster: list[np.ndarray] = []
    theta_snapshots: list[np.ndarray] = []
    unary_seen = np.zeros(cfg.n_clusters, dtype=np.int64)
    unary_seen_logs: list[np.ndarray] = []
    pair_seen_logs: list[int] = []
    refit_steps: list[int] = []
    nmi: list[float] = []
    pur: list[float] = []
    coverage: list[float] = []

    for step in range(train_config.steps):
        x, y, desirable, true_cluster, annotator = dataset.sample(train_config.batch_size)
        clusterer.observe(annotator, desirable)
        if step >= warmup and (step - warmup) % refit_every == 0:
            info = clusterer.refit(step)
            labels = clusterer.labels
            refit_steps.append(step)
            nmi.append(normalized_mutual_information(labels, dataset.annotator_cluster))
            pur.append(purity(labels, dataset.annotator_cluster))
            coverage.append(float(info["coverage"]))

        clusters = clusterer.for_annotators(annotator)
        policy = softmax(theta)
        rewards = np.log(policy[x, y]) + np.log(cfg.n_responses)
        ref.update(rewards, desirable, clusters, train_config.ema_rate)
        z_i = ref.get(clusters)
        grad, _ = _loss_grad(theta, x, y, desirable, clusters, z_i, train_config, cfg.n_responses)
        adam_m, adam_v = _adam_update(theta, grad, adam_m, adam_v, step + 1, train_config.learning_rate)
        unary_seen += np.bincount(true_cluster, minlength=cfg.n_clusters)
        unary_seen_logs.append(unary_seen.copy())
        pair_seen_logs.append(0)

        if step % train_config.eval_every == 0:
            eval_steps.append(step)
            quality = expected_quality(theta, world.q)
            eval_quality.append(quality)
            eval_quality_per_prompt.append(_expected_quality_per_prompt(theta, world.q))
            eval_desirability_by_cluster.append(
                _expected_desirability_by_cluster(theta, world.q, world)
            )
            theta_snapshots.append(theta.copy())
            if log_prefix is not None:
                print(f"{log_prefix} method=hard_online seed={seed} step={step} E[q]={quality:.4f}", flush=True)

    _append_final_eval(
        theta,
        world,
        train_config,
        eval_steps,
        eval_quality,
        eval_quality_per_prompt,
        eval_desirability_by_cluster,
        theta_snapshots,
        log_prefix,
        "hard_online",
        seed,
    )
    return OnlineTrainResult(
        train_result=TrainResult(
            method="hard_online",
            seed=seed,
            eval_steps=np.asarray(eval_steps),
            expected_quality=np.asarray(eval_quality),
            grad_weight_by_cluster=None,
            final_theta=theta.copy(),
            world_rewards=world.q,
            expected_quality_per_prompt=np.asarray(eval_quality_per_prompt),
            expected_desirability_by_cluster=np.asarray(eval_desirability_by_cluster),
            theta_snapshots=np.asarray(theta_snapshots),
            n_unary_per_cluster_seen=_cluster_count_array(unary_seen_logs, cfg.n_clusters),
            n_pair_seen=np.asarray(pair_seen_logs, dtype=np.int64),
        ),
        refit_steps=np.asarray(refit_steps),
        nmi=np.asarray(nmi),
        purity=np.asarray(pur),
        coverage=np.asarray(coverage),
    )


def train_online_soft(
    world: SyntheticWorld,
    dataset: OfflineDataset,
    seed: int,
    train_config: TrainConfig,
    n_clusters: int = 2,
    warmup: int = 25,
    refit_every: int = 50,
    log_prefix: str | None = None,
) -> OnlineTrainResult:
    cfg = world.config
    clusterer = OnlineSoftClusterer(cfg.n_annotators, n_clusters, seed)
    theta = np.zeros((cfg.n_prompts, cfg.n_responses), dtype=np.float64)
    adam_m = np.zeros_like(theta)
    adam_v = np.zeros_like(theta)
    ref = _SoftReference(n_clusters)

    eval_steps: list[int] = []
    eval_quality: list[float] = []
    eval_quality_per_prompt: list[np.ndarray] = []
    eval_desirability_by_cluster: list[np.ndarray] = []
    theta_snapshots: list[np.ndarray] = []
    unary_seen = np.zeros(cfg.n_clusters, dtype=np.int64)
    unary_seen_logs: list[np.ndarray] = []
    pair_seen_logs: list[int] = []
    refit_steps: list[int] = []
    nmi: list[float] = []
    pur: list[float] = []
    coverage: list[float] = []

    for step in range(train_config.steps):
        x, y, desirable, true_cluster, annotator = dataset.sample(train_config.batch_size)
        clusterer.observe(annotator, desirable)
        if step >= warmup and (step - warmup) % refit_every == 0:
            info = clusterer.refit(step)
            responsibilities = clusterer.responsibilities
            refit_steps.append(step)
            nmi.append(soft_normalized_mutual_information(responsibilities, dataset.annotator_cluster))
            pur.append(soft_purity(responsibilities, dataset.annotator_cluster))
            coverage.append(float(info["coverage"]))

        sample_responsibilities = clusterer.for_annotators(annotator)
        policy = softmax(theta)
        rewards = np.log(policy[x, y]) + np.log(cfg.n_responses)
        ref.update(rewards, desirable, sample_responsibilities, train_config.ema_rate)
        z_i = ref.per_sample_reference(sample_responsibilities)
        placeholder_cluster = sample_responsibilities.argmax(axis=1).astype(np.int64)
        grad, _ = _loss_grad(theta, x, y, desirable, placeholder_cluster, z_i, train_config, cfg.n_responses)
        adam_m, adam_v = _adam_update(theta, grad, adam_m, adam_v, step + 1, train_config.learning_rate)
        unary_seen += np.bincount(true_cluster, minlength=cfg.n_clusters)
        unary_seen_logs.append(unary_seen.copy())
        pair_seen_logs.append(0)

        if step % train_config.eval_every == 0:
            eval_steps.append(step)
            quality = expected_quality(theta, world.q)
            eval_quality.append(quality)
            eval_quality_per_prompt.append(_expected_quality_per_prompt(theta, world.q))
            eval_desirability_by_cluster.append(
                _expected_desirability_by_cluster(theta, world.q, world)
            )
            theta_snapshots.append(theta.copy())
            if log_prefix is not None:
                print(f"{log_prefix} method=soft_online seed={seed} step={step} E[q]={quality:.4f}", flush=True)

    _append_final_eval(
        theta,
        world,
        train_config,
        eval_steps,
        eval_quality,
        eval_quality_per_prompt,
        eval_desirability_by_cluster,
        theta_snapshots,
        log_prefix,
        "soft_online",
        seed,
    )
    return OnlineTrainResult(
        train_result=TrainResult(
            method="soft_online",
            seed=seed,
            eval_steps=np.asarray(eval_steps),
            expected_quality=np.asarray(eval_quality),
            grad_weight_by_cluster=None,
            final_theta=theta.copy(),
            world_rewards=world.q,
            expected_quality_per_prompt=np.asarray(eval_quality_per_prompt),
            expected_desirability_by_cluster=np.asarray(eval_desirability_by_cluster),
            theta_snapshots=np.asarray(theta_snapshots),
            n_unary_per_cluster_seen=_cluster_count_array(unary_seen_logs, cfg.n_clusters),
            n_pair_seen=np.asarray(pair_seen_logs, dtype=np.int64),
        ),
        refit_steps=np.asarray(refit_steps),
        nmi=np.asarray(nmi),
        purity=np.asarray(pur),
        coverage=np.asarray(coverage),
    )


class _HardReference:
    def __init__(self, n_clusters: int):
        self.z = np.zeros(n_clusters, dtype=np.float64)

    def update(self, rewards: np.ndarray, desirable: np.ndarray, cluster: np.ndarray, rho: float) -> None:
        for cluster_id in range(len(self.z)):
            mask = (cluster == cluster_id) & (desirable == 0.0)
            if np.any(mask):
                value = float(rewards[mask].mean())
                self.z[cluster_id] = (1.0 - rho) * self.z[cluster_id] + rho * value

    def get(self, cluster: np.ndarray) -> np.ndarray:
        return self.z[cluster]


class _SoftReference:
    def __init__(self, n_clusters: int):
        self.z = np.zeros(n_clusters, dtype=np.float64)

    def update(
        self,
        rewards: np.ndarray,
        desirable: np.ndarray,
        sample_responsibilities: np.ndarray,
        rho: float,
    ) -> None:
        undesirable = desirable == 0.0
        if not np.any(undesirable):
            return
        u_rewards = rewards[undesirable]
        u_weights = sample_responsibilities[undesirable]
        weight_per_cluster = u_weights.sum(axis=0)
        for cluster_id in range(len(self.z)):
            denom = weight_per_cluster[cluster_id]
            if denom <= 1e-12:
                continue
            value = float((u_weights[:, cluster_id] * u_rewards).sum() / denom)
            self.z[cluster_id] = (1.0 - rho) * self.z[cluster_id] + rho * value

    def per_sample_reference(self, sample_responsibilities: np.ndarray) -> np.ndarray:
        return sample_responsibilities @ self.z


def _append_final_eval(
    theta: np.ndarray,
    world: SyntheticWorld,
    train_config: TrainConfig,
    eval_steps: list[int],
    eval_quality: list[float],
    eval_quality_per_prompt: list[np.ndarray],
    eval_desirability_by_cluster: list[np.ndarray],
    theta_snapshots: list[np.ndarray],
    log_prefix: str | None,
    method: str,
    seed: int,
) -> None:
    if not eval_steps or eval_steps[-1] != train_config.steps:
        eval_steps.append(train_config.steps)
        quality = expected_quality(theta, world.q)
        eval_quality.append(quality)
        eval_quality_per_prompt.append(_expected_quality_per_prompt(theta, world.q))
        eval_desirability_by_cluster.append(
            _expected_desirability_by_cluster(theta, world.q, world)
        )
        theta_snapshots.append(theta.copy())
        if log_prefix is not None:
            print(f"{log_prefix} method={method} seed={seed} step={train_config.steps} E[q]={quality:.4f}", flush=True)
