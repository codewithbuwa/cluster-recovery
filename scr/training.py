from dataclasses import dataclass

import numpy as np

from scr.config import TrainConfig
from scr.policy import expected_quality, softmax
from scr.reference import ReferencePoint
from scr.samplers import OfflineDataset
from scr.world import SyntheticWorld


@dataclass
class TrainResult:
    method: str
    seed: int
    eval_steps: np.ndarray
    expected_quality: np.ndarray
    grad_weight_by_cluster: np.ndarray | None
    reference_values: np.ndarray | None = None
    final_theta: np.ndarray | None = None


def _adam_update(
    theta: np.ndarray,
    grad: np.ndarray,
    adam_m: np.ndarray,
    adam_v: np.ndarray,
    step: int,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    adam_m = beta1 * adam_m + (1.0 - beta1) * grad
    adam_v = beta2 * adam_v + (1.0 - beta2) * (grad * grad)
    m_hat = adam_m / (1.0 - beta1**step)
    v_hat = adam_v / (1.0 - beta2**step)
    theta -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
    return adam_m, adam_v


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.where(
        values >= 0,
        1.0 / (1.0 + np.exp(-values)),
        np.exp(values) / (1.0 + np.exp(values)),
    )


def _reward_grad_to_theta(
    policy: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dloss_dr: np.ndarray,
    normalizer: int,
) -> np.ndarray:
    grad = np.zeros_like(policy)
    np.add.at(grad, (x, y), dloss_dr)
    for idx in range(len(x)):
        grad[x[idx], :] -= dloss_dr[idx] * policy[x[idx], :]
    return grad / normalizer


def _pair_loss_grad(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    desirable: np.ndarray,
    cluster: np.ndarray,
    train_config: TrainConfig,
    n_responses: int,
) -> np.ndarray:
    policy = softmax(theta)
    rewards = np.log(policy[x, y]) + np.log(n_responses)
    dloss_dr = np.zeros(len(x), dtype=np.float64)
    pair_count = 0

    for prompt_id in np.unique(x):
        prompt_mask = x == prompt_id
        for cluster_id in np.unique(cluster[prompt_mask]):
            group_mask = prompt_mask & (cluster == cluster_id)
            desirable_idx = np.flatnonzero(group_mask & (desirable == 1.0))
            undesirable_idx = np.flatnonzero(group_mask & (desirable == 0.0))
            for good_idx in desirable_idx:
                deltas = rewards[good_idx] - rewards[undesirable_idx]
                coeffs = -train_config.beta * (1.0 - _sigmoid(train_config.beta * deltas))
                dloss_dr[good_idx] += coeffs.sum()
                np.add.at(dloss_dr, undesirable_idx, -coeffs)
                pair_count += len(undesirable_idx)

    if pair_count == 0:
        return np.zeros_like(theta)
    return _reward_grad_to_theta(policy, x, y, dloss_dr, pair_count)


def _loss_grad(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    desirable: np.ndarray,
    cluster: np.ndarray,
    z_i: np.ndarray,
    train_config: TrainConfig,
    n_responses: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= train_config.alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {train_config.alpha}")

    policy = softmax(theta)
    rewards = np.log(policy[x, y]) + np.log(n_responses)
    sign = np.where(desirable == 1.0, 1.0, -1.0)
    margin = np.where(desirable == 1.0, rewards - z_i, z_i - rewards)
    sigmoid = _sigmoid(train_config.beta * margin)
    grad_weight = sigmoid * (1.0 - sigmoid)
    lambdas = np.where(desirable == 1.0, train_config.lambda_desirable, train_config.lambda_undesirable)
    dloss_dr = lambdas * (-train_config.beta * grad_weight) * sign

    unary_grad = _reward_grad_to_theta(policy, x, y, dloss_dr, len(x))
    if train_config.alpha == 0.0:
        return unary_grad, grad_weight

    pair_grad = _pair_loss_grad(theta, x, y, desirable, cluster, train_config, n_responses)
    grad = (1.0 - train_config.alpha) * unary_grad + train_config.alpha * pair_grad
    return grad, grad_weight


def train_method(
    world: SyntheticWorld,
    method: str,
    seed: int,
    train_config: TrainConfig,
    record_grad_weights: bool = False,
    record_references: bool = False,
    force_cluster: int | None = None,
    reference_variant: str = "undesirable",
    cluster_mode: str = "true",
    log_prefix: str | None = None,
) -> TrainResult:
    if method not in {"kto", "cpo"}:
        raise ValueError(f"unknown method: {method}")
    if reference_variant not in {"undesirable", "all", "kl"}:
        raise ValueError(f"unknown reference variant: {reference_variant}")
    if cluster_mode not in {"true", "random"}:
        raise ValueError(f"unknown cluster mode: {cluster_mode}")

    cfg = world.config
    rng = np.random.default_rng(seed)
    theta = np.zeros((cfg.n_prompts, cfg.n_responses), dtype=np.float64)
    adam_m = np.zeros_like(theta)
    adam_v = np.zeros_like(theta)
    n_clusters = world.config.n_clusters if method == "cpo" else 1
    ref = ReferencePoint(
        world.config,
        train_config,
        n_clusters=n_clusters,
        pooled=method == "kto",
        variant=reference_variant,
    )

    eval_steps = []
    eval_quality = []
    grad_logs: list[list[float]] = []
    reference_logs: list[np.ndarray] = []
    random_cluster_rng = np.random.default_rng(seed + 89_999) if cluster_mode == "random" else None

    for step in range(train_config.steps):
        x, y, desirable, true_cluster = world.sample_batch(
            rng, train_config.batch_size, force_cluster=force_cluster
        )
        if method == "cpo" and cluster_mode == "random":
            cpo_cluster = random_cluster_rng.integers(0, 2, size=train_config.batch_size)
        else:
            cpo_cluster = true_cluster
        ref_cluster = cpo_cluster if method == "cpo" else np.zeros_like(true_cluster)
        policy = softmax(theta)
        rewards = np.log(policy[x, y]) + np.log(cfg.n_responses)
        z_i = ref.get(ref_cluster)
        if record_references:
            reference_logs.append(ref.z.copy())
        grad, grad_weight = _loss_grad(theta, x, y, desirable, ref_cluster, z_i, train_config, cfg.n_responses)

        adam_m, adam_v = _adam_update(
            theta, grad, adam_m, adam_v, step + 1, train_config.learning_rate
        )
        ref.update(rewards, x, ref_cluster, desirable, policy)

        if record_grad_weights:
            means = []
            for k in range(2):
                mask = true_cluster == k
                means.append(float(grad_weight[mask].mean()) if np.any(mask) else np.nan)
            grad_logs.append(means)

        if step % train_config.eval_every == 0:
            eval_steps.append(step)
            quality = expected_quality(theta, world.q)
            eval_quality.append(quality)
            if log_prefix is not None:
                print(
                    f"{log_prefix} method={method} seed={seed} step={step} "
                    f"E[q]={quality:.4f}",
                    flush=True,
                )

    if not eval_steps or eval_steps[-1] != train_config.steps:
        eval_steps.append(train_config.steps)
        quality = expected_quality(theta, world.q)
        eval_quality.append(quality)
        if log_prefix is not None:
            print(
                f"{log_prefix} method={method} seed={seed} step={train_config.steps} "
                f"E[q]={quality:.4f}",
                flush=True,
            )

    return TrainResult(
        method=method,
        seed=seed,
        eval_steps=np.asarray(eval_steps),
        expected_quality=np.asarray(eval_quality),
        grad_weight_by_cluster=np.asarray(grad_logs) if record_grad_weights else None,
        reference_values=np.asarray(reference_logs) if record_references else None,
        final_theta=theta.copy(),
    )


def train_offline_method(
    world: SyntheticWorld,
    dataset: OfflineDataset,
    method: str,
    seed: int,
    train_config: TrainConfig,
    n_clusters: int,
    annotator_clusters: np.ndarray | None = None,
    reference_variant: str = "undesirable",
    log_prefix: str | None = None,
) -> TrainResult:
    if method not in {"kto", "cpo"}:
        raise ValueError(f"unknown method: {method}")

    cfg = world.config
    theta = np.zeros((cfg.n_prompts, cfg.n_responses), dtype=np.float64)
    adam_m = np.zeros_like(theta)
    adam_v = np.zeros_like(theta)
    ref = ReferencePoint(
        cfg,
        train_config,
        n_clusters=n_clusters,
        pooled=method == "kto",
        variant=reference_variant,
    )

    eval_steps = []
    eval_quality = []

    for step in range(train_config.steps):
        x, y, desirable, true_cluster, annotator = dataset.sample(train_config.batch_size)
        if method == "cpo" and annotator_clusters is not None:
            cluster = annotator_clusters[annotator].astype(np.int64)
        else:
            cluster = true_cluster.astype(np.int64)
        cluster = np.clip(cluster, 0, n_clusters - 1)
        ref_cluster = cluster if method == "cpo" else np.zeros_like(cluster)

        policy = softmax(theta)
        rewards = np.log(policy[x, y]) + np.log(cfg.n_responses)
        z_i = ref.get(ref_cluster)
        grad, _ = _loss_grad(theta, x, y, desirable, ref_cluster, z_i, train_config, cfg.n_responses)
        adam_m, adam_v = _adam_update(
            theta, grad, adam_m, adam_v, step + 1, train_config.learning_rate
        )
        ref.update(rewards, x, ref_cluster, desirable, policy)

        if step % train_config.eval_every == 0:
            eval_steps.append(step)
            quality = expected_quality(theta, world.q)
            eval_quality.append(quality)
            if log_prefix is not None:
                print(
                    f"{log_prefix} method={method} seed={seed} step={step} "
                    f"E[q]={quality:.4f}",
                    flush=True,
                )

    if not eval_steps or eval_steps[-1] != train_config.steps:
        eval_steps.append(train_config.steps)
        quality = expected_quality(theta, world.q)
        eval_quality.append(quality)
        if log_prefix is not None:
            print(
                f"{log_prefix} method={method} seed={seed} step={train_config.steps} "
                f"E[q]={quality:.4f}",
                flush=True,
            )

    return TrainResult(
        method=method,
        seed=seed,
        eval_steps=np.asarray(eval_steps),
        expected_quality=np.asarray(eval_quality),
        grad_weight_by_cluster=None,
        reference_values=None,
        final_theta=theta.copy(),
    )
