import numpy as np

from scr.config import WorldConfig
from scr.world import SyntheticWorld


class OnlineSampler:
    def __init__(self, world: SyntheticWorld, seed: int):
        self.world = world
        self.config = world.config
        self.rng = np.random.default_rng(seed + 10_000)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        x = self.rng.integers(0, cfg.n_prompts, size=batch_size)
        y = self.rng.integers(0, cfg.n_responses, size=batch_size)
        cluster = self.rng.binomial(1, 1.0 - cfg.pi_a, size=batch_size).astype(np.int64)
        desirable = self.world.label(x, y, cluster, self.rng)
        return x, y, desirable, cluster


class OracleBobSampler:
    def __init__(self, world: SyntheticWorld, seed: int):
        self.world = world
        self.config = world.config
        self.rng = np.random.default_rng(seed + 50_000)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        x = self.rng.integers(0, cfg.n_prompts, size=batch_size)
        y = self.rng.integers(0, cfg.n_responses, size=batch_size)
        cluster = np.ones(batch_size, dtype=np.int64)
        desirable = self.world.label(x, y, cluster, self.rng)
        return x, y, desirable, cluster


class OfflineDataset:
    def __init__(self, world: SyntheticWorld, config: WorldConfig, seed: int):
        self.config = config
        rng = np.random.default_rng(seed + 20_000)
        n_annotators = config.n_annotators
        samples_per_annotator = config.samples_per_annotator
        total = n_annotators * samples_per_annotator
        self.annotator_cluster = rng.binomial(1, 1.0 - config.pi_a, size=n_annotators).astype(np.int64)
        self.x = rng.integers(0, config.n_prompts, size=total)
        self.y = rng.integers(0, config.n_responses, size=total)
        self.annotator = np.repeat(np.arange(n_annotators), samples_per_annotator).astype(np.int64)
        self.cluster = self.annotator_cluster[self.annotator]
        self.desirable = world.label(self.x, self.y, self.cluster, rng)
        self.rng = np.random.default_rng(seed + 30_000)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, len(self.x), size=batch_size)
        return self.x[idx], self.y[idx], self.desirable[idx], self.cluster[idx], self.annotator[idx]
