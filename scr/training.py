from dataclasses import dataclass

import numpy as np

from scr.config import TrainConfig
from scr.policy import expected_quality, softmax
from scr.reference import ReferencePoint
from scr.samplers import OfflineDataset
from scr.world import SyntheticWorld


_PAIR_RNG_OFFSET = 1_000_000


@dataclass
class TrainResult:
    method: str
    seed: int
    eval_steps: np.ndarray
    expected_quality: np.ndarray
    grad_weight_by_cluster: np.ndarray | None
    reference_values: np.ndarray | None = None
    final_theta: np.ndarray | None = None
    world_rewards: np.ndarray | None = None
    expected_quality_per_prompt: np.ndarray | None = None
    expected_desirability_by_cluster: np.ndarray | None = None
    theta_snapshots: np.ndarray | None = None
    n_unary_per_cluster_seen: np.ndarray | None = None
    n_pair_seen: np.ndarray | None = None


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


def _kl_grad_to_theta(policy: np.ndarray) -> np.ndarray:
    log_ratio = np.log(policy) + np.log(policy.shape[1])
    kl_per_prompt = np.sum(policy * log_ratio, axis=1, keepdims=True)
    return policy * (log_ratio - kl_per_prompt) / policy.shape[0]


def _expected_quality_per_prompt(theta: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.sum(softmax(theta) * q, axis=1)


def _expected_desirability_by_cluster(
    theta: np.ndarray,
    q: np.ndarray,
    world: SyntheticWorld,
) -> np.ndarray:
    policy = softmax(theta)
    cfg = world.config
    if cfg.n_clusters == 1:
        thresholds = np.asarray([cfg.tau_a])
        noise_rates = np.asarray([cfg.eps_a])
    elif cfg.n_clusters == 2:
        thresholds = np.asarray([cfg.tau_a, cfg.tau_b])
        noise_rates = np.asarray([cfg.eps_a, cfg.eps_b])
    else:
        raise ValueError("expected desirability supports one or two annotator clusters")

    values = []
    for threshold, noise_rate in zip(thresholds, noise_rates):
        clean_desirable = q > threshold
        desirable_probability = np.where(clean_desirable, 1.0 - noise_rate, noise_rate)
        values.append(float(np.mean(np.sum(policy * desirable_probability, axis=1))))
    return np.asarray(values)


def _cluster_count_array(logs: list[np.ndarray], n_clusters: int) -> np.ndarray:
    return np.asarray(logs, dtype=np.int64).reshape(len(logs), n_clusters)


def _pair_loss_grad(
    theta: np.ndarray,
    x: np.ndarray,
    y_winner: np.ndarray,
    y_loser: np.ndarray,
    train_config: TrainConfig,
    n_responses: int,
) -> np.ndarray:
    policy = softmax(theta)
    reward_winner = np.log(policy[x, y_winner]) + np.log(n_responses)
    reward_loser = np.log(policy[x, y_loser]) + np.log(n_responses)
    margin = reward_winner - reward_loser
    coeff = -train_config.beta * (1.0 - _sigmoid(train_config.beta * margin))

    pair_x = np.concatenate([x, x])
    pair_y = np.concatenate([y_winner, y_loser])
    dloss_dr = np.concatenate([coeff, -coeff])
    return _reward_grad_to_theta(policy, pair_x, pair_y, dloss_dr, len(x))


def _batch_sizes(train_config: TrainConfig) -> tuple[int, int]:
    if not 0.0 <= train_config.pair_fraction <= 1.0:
        raise ValueError(f"pair_fraction must be in [0, 1], got {train_config.pair_fraction}")

    if train_config.total_effort is None:
        if train_config.pair_fraction == 0.0:
            return train_config.batch_size, 0
        total_effort = train_config.batch_size
    else:
        total_effort = train_config.total_effort

    if total_effort < 0:
        raise ValueError(f"total_effort must be non-negative, got {total_effort}")

    def stable_floor(value: float) -> int:
        nearest_integer = round(value)
        if np.isclose(value, nearest_integer, rtol=0.0, atol=1e-10):
            return int(nearest_integer)
        return int(value)

    n_unary = stable_floor((1.0 - train_config.pair_fraction) * total_effort)
    n_pair = stable_floor(train_config.pair_fraction * total_effort / 2.0)
    return n_unary, n_pair


def valid_budget_sweep_cell(method: str, train_config: TrainConfig) -> bool:
    """Protocol filter for CPO_part2 Experiment 1 budget-sweep cells."""
    n_unary, n_pair = _batch_sizes(train_config)
    alpha = train_config.alpha

    if alpha == 0.0:
        return n_unary > 0
    if alpha == 1.0:
        return n_pair > 0
    if 0.0 < alpha < 1.0:
        return n_unary > 0 and n_pair > 0
    return False


def _loss_grad(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    desirable: np.ndarray,
    cluster: np.ndarray,
    z_i: np.ndarray,
    train_config: TrainConfig,
    n_responses: int,
    pair_x: np.ndarray | None = None,
    pair_winner: np.ndarray | None = None,
    pair_loser: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= train_config.alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {train_config.alpha}")

    policy = softmax(theta)
    unary_grad = np.zeros_like(theta)
    grad_weight = np.empty(0, dtype=np.float64)

    if len(x) > 0 and train_config.alpha < 1.0:
        rewards = np.log(policy[x, y]) + np.log(n_responses)
        sign = np.where(desirable == 1.0, 1.0, -1.0)
        margin = np.where(desirable == 1.0, rewards - z_i, z_i - rewards)
        sigmoid = _sigmoid(train_config.beta * margin)
        grad_weight = sigmoid * (1.0 - sigmoid)
        lambdas = np.where(desirable == 1.0, train_config.lambda_desirable, train_config.lambda_undesirable)
        dloss_dr = lambdas * (-train_config.beta * grad_weight) * sign
        unary_grad = _reward_grad_to_theta(policy, x, y, dloss_dr, len(x))

    pair_grad = np.zeros_like(theta)
    if (
        pair_x is not None
        and pair_winner is not None
        and pair_loser is not None
        and len(pair_x) > 0
        and train_config.alpha > 0.0
    ):
        pair_grad = _pair_loss_grad(
            theta,
            pair_x,
            pair_winner,
            pair_loser,
            train_config,
            n_responses,
        )

    grad = (1.0 - train_config.alpha) * unary_grad + train_config.alpha * pair_grad
    if train_config.eta != 0.0:
        grad += train_config.eta * _kl_grad_to_theta(policy)
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
    if method not in {"kto", "cpo", "dpo"}:
        raise ValueError(f"unknown method: {method}")
    if reference_variant not in {"desirable", "undesirable", "all", "kl"}:
        raise ValueError(f"unknown reference variant: {reference_variant}")
    if cluster_mode not in {"true", "random"}:
        raise ValueError(f"unknown cluster mode: {cluster_mode}")
    if method == "dpo" and train_config.alpha != 1.0:
        raise ValueError("dpo requires TrainConfig(alpha=1.0)")

    cfg = world.config
    unary_rng = np.random.default_rng(seed)
    pair_rng = np.random.default_rng(seed + _PAIR_RNG_OFFSET)
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
    eval_quality_per_prompt: list[np.ndarray] = []
    eval_desirability_by_cluster: list[np.ndarray] = []
    theta_snapshots: list[np.ndarray] = []
    grad_logs: list[list[float]] = []
    reference_logs: list[np.ndarray] = []
    unary_seen = np.zeros(cfg.n_clusters, dtype=np.int64)
    pair_seen = 0
    unary_seen_logs: list[np.ndarray] = []
    pair_seen_logs: list[int] = []
    random_cluster_rng = np.random.default_rng(seed + 89_999) if cluster_mode == "random" else None

    for step in range(train_config.steps):
        n_unary, n_pair = _batch_sizes(train_config)
        if train_config.alpha == 0.0:
            n_pair = 0
        if train_config.alpha == 1.0:
            n_unary = 0

        if n_unary > 0:
            x, y, desirable, true_cluster = world.sample_batch(
                unary_rng, n_unary, force_cluster=force_cluster
            )
        else:
            x = np.empty(0, dtype=np.int64)
            y = np.empty(0, dtype=np.int64)
            desirable = np.empty(0, dtype=np.float64)
            true_cluster = np.empty(0, dtype=np.int64)

        if method == "cpo" and cluster_mode == "random":
            cpo_cluster = random_cluster_rng.integers(0, 2, size=n_unary)
        else:
            cpo_cluster = true_cluster
        ref_cluster = cpo_cluster if method == "cpo" else np.zeros_like(true_cluster)

        if n_pair > 0:
            pair_x, pair_winner, pair_loser = world.sample_pair_batch(
                pair_rng,
                n_pair,
                pair_noise=train_config.pair_noise,
            )
        else:
            pair_x = None
            pair_winner = None
            pair_loser = None

        unary_seen += np.bincount(true_cluster, minlength=cfg.n_clusters)
        if pair_x is not None:
            pair_seen += len(pair_x)
        unary_seen_logs.append(unary_seen.copy())
        pair_seen_logs.append(pair_seen)

        policy = softmax(theta)
        rewards = np.log(policy[x, y]) + np.log(cfg.n_responses) if n_unary > 0 else np.empty(0)
        z_i = ref.get(ref_cluster)
        if record_references:
            reference_logs.append(ref.z.copy())
        grad, grad_weight = _loss_grad(
            theta,
            x,
            y,
            desirable,
            ref_cluster,
            z_i,
            train_config,
            cfg.n_responses,
            pair_x=pair_x,
            pair_winner=pair_winner,
            pair_loser=pair_loser,
        )

        adam_m, adam_v = _adam_update(
            theta, grad, adam_m, adam_v, step + 1, train_config.learning_rate
        )
        if n_unary > 0:
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
            eval_quality_per_prompt.append(_expected_quality_per_prompt(theta, world.q))
            eval_desirability_by_cluster.append(
                _expected_desirability_by_cluster(theta, world.q, world)
            )
            theta_snapshots.append(theta.copy())
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
        eval_quality_per_prompt.append(_expected_quality_per_prompt(theta, world.q))
        eval_desirability_by_cluster.append(
            _expected_desirability_by_cluster(theta, world.q, world)
        )
        theta_snapshots.append(theta.copy())
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
        world_rewards=world.q,
        expected_quality_per_prompt=np.asarray(eval_quality_per_prompt),
        expected_desirability_by_cluster=np.asarray(eval_desirability_by_cluster),
        theta_snapshots=np.asarray(theta_snapshots),
        n_unary_per_cluster_seen=_cluster_count_array(unary_seen_logs, cfg.n_clusters),
        n_pair_seen=np.asarray(pair_seen_logs, dtype=np.int64),
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
    eval_quality_per_prompt: list[np.ndarray] = []
    eval_desirability_by_cluster: list[np.ndarray] = []
    theta_snapshots: list[np.ndarray] = []
    unary_seen = np.zeros(cfg.n_clusters, dtype=np.int64)
    unary_seen_logs: list[np.ndarray] = []
    pair_seen_logs: list[int] = []

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
                print(
                    f"{log_prefix} method={method} seed={seed} step={step} "
                    f"E[q]={quality:.4f}",
                    flush=True,
                )

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
        world_rewards=world.q,
        expected_quality_per_prompt=np.asarray(eval_quality_per_prompt),
        expected_desirability_by_cluster=np.asarray(eval_desirability_by_cluster),
        theta_snapshots=np.asarray(theta_snapshots),
        n_unary_per_cluster_seen=_cluster_count_array(unary_seen_logs, cfg.n_clusters),
        n_pair_seen=np.asarray(pair_seen_logs, dtype=np.int64),
    )
