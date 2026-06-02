import numpy as np

from scr.config import TrainConfig, WorldConfig


def kl_per_prompt(policy: np.ndarray, world_config: WorldConfig) -> np.ndarray:
    return np.sum(policy * (np.log(policy) + np.log(world_config.n_responses)), axis=1)


class ReferencePoint:
    def __init__(
        self,
        world_config: WorldConfig,
        train_config: TrainConfig,
        n_clusters: int = 2,
        pooled: bool = False,
        variant: str = "undesirable",
    ):
        if variant not in {"desirable","undesirable", "all", "kl"}:
            raise ValueError(f"unknown reference variant: {variant}")
        self.world_config = world_config
        self.train_config = train_config
        self.n_clusters = n_clusters
        self.pooled = pooled
        self.variant = variant
        self.z = np.zeros(1 if pooled else n_clusters, dtype=np.float64)

    def update(
        self,
        rewards: np.ndarray,
        x: np.ndarray,
        cluster: np.ndarray,
        desirable: np.ndarray,
        policy: np.ndarray,
    ) -> None:
        n_refs = 1 if self.pooled else self.n_clusters
        for ref_idx in range(n_refs):
            cluster_mask = np.ones(len(cluster), dtype=bool) if self.pooled else (cluster == ref_idx)
            if self.variant == "kl":
                if np.any(cluster_mask):
                    kl_values = kl_per_prompt(policy, self.world_config)
                    value = float(kl_values[x[cluster_mask]].mean())
                    self._update_one(ref_idx, value)
            elif self.variant == "all":
                if np.any(cluster_mask):
                    self._update_one(ref_idx, float(rewards[cluster_mask].mean()))
            elif self.variant == "desirable":
                mask = cluster_mask & (desirable == 1.0)
                if np.any(mask):
                    self._update_one(ref_idx, float(rewards[mask].mean()))
            else:
                mask = cluster_mask & (desirable == 0.0)
                if np.any(mask):
                    self._update_one(ref_idx, float(rewards[mask].mean()))

    def _update_one(self, ref_idx: int, value: float) -> None:
        rho = self.train_config.ema_rate
        self.z[ref_idx] = (1.0 - rho) * self.z[ref_idx] + rho * value

    def get(self, cluster: np.ndarray) -> np.ndarray:
        if self.pooled:
            return np.full(len(cluster), self.z[0])
        return self.z[np.clip(cluster, 0, self.n_clusters - 1)]
