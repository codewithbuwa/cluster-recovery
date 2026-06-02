import numpy as np

from scr.config import WorldConfig


class SyntheticWorld:
    def __init__(self, config: WorldConfig, seed: int):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.q = self.rng.uniform(0.0, 1.0, size=(config.n_prompts, config.n_responses))

    def label(
        self,
        x: np.ndarray,
        y: np.ndarray,
        cluster: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        cfg = self.config
        tau = np.where(cluster == 0, cfg.tau_a, cfg.tau_b)
        eps = np.where(cluster == 0, cfg.eps_a, cfg.eps_b)
        clean = self.q[x, y] > tau
        flips = rng.binomial(1, eps).astype(np.int8)
        return np.logical_xor(clean.astype(np.int8), flips).astype(np.int8)

    def sample_batch(
        self,
        rng: np.random.Generator,
        batch_size: int,
        force_cluster: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        x = rng.integers(0, cfg.n_prompts, size=batch_size)
        y = rng.integers(0, cfg.n_responses, size=batch_size)

        if force_cluster is None:
            if cfg.n_clusters == 1:
                c = np.zeros(batch_size, dtype=np.int64)
            elif cfg.n_clusters == 2:
                c = rng.binomial(1, 1.0 - cfg.pi_a, size=batch_size).astype(np.int64)
            else:
                raise ValueError("SyntheticWorld online sampling supports one or two clusters")
        else:
            if force_cluster < 0 or force_cluster >= cfg.n_clusters:
                raise ValueError(f"force_cluster={force_cluster} is outside [0, {cfg.n_clusters})")
            c = np.full(batch_size, force_cluster, dtype=np.int64)

        desirable = self.label(x, y, c, rng)
        return x, y, desirable, c

    def sample_pair_batch(
        self,
        rng: np.random.Generator,
        batch_size: int,
        pair_noise: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        x = rng.integers(0, cfg.n_prompts, size=batch_size)
        y_a = rng.integers(0, cfg.n_responses, size=batch_size)
        y_b = rng.integers(0, cfg.n_responses - 1, size=batch_size)
        y_b = np.where(y_b >= y_a, y_b + 1, y_b)

        a_wins = self.q[x, y_a] >= self.q[x, y_b]
        y_winner = np.where(a_wins, y_a, y_b)
        y_loser = np.where(a_wins, y_b, y_a)

        flips = rng.binomial(1, pair_noise, size=batch_size).astype(bool)
        noisy_winner = np.where(flips, y_loser, y_winner)
        noisy_loser = np.where(flips, y_winner, y_loser)
        return x, noisy_winner, noisy_loser
