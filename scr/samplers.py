import numpy as np

from scr.config import WorldConfig
from scr.world import SyntheticWorld


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
