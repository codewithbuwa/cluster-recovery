from dataclasses import dataclass

import numpy as np

from exp_d.clustering import kmeans_1d
from exp_d.clustering_soft import gmm_1d_soft


def _match_components(new_centers: np.ndarray, previous_centers: np.ndarray | None) -> np.ndarray:
    """Return old-index labels for each new component.

    The project uses K<=2 in the online comparison, but this handles small K
    generally by brute-force permutation search to avoid a SciPy dependency.
    """
    n_clusters = len(new_centers)
    if previous_centers is None:
        return np.argsort(new_centers)

    best_perm = None
    best_cost = float("inf")
    for perm in _permutations(np.arange(n_clusters)):
        cost = float(np.sum((new_centers - previous_centers[np.asarray(perm)]) ** 2))
        if cost < best_cost:
            best_cost = cost
            best_perm = np.asarray(perm, dtype=np.int64)
    return best_perm


def _permutations(values: np.ndarray):
    if len(values) == 1:
        yield values
        return
    for idx, value in enumerate(values):
        rest = np.delete(values, idx)
        for suffix in _permutations(rest):
            yield np.concatenate(([value], suffix))


@dataclass
class StreamingSignatures:
    n_annotators: int

    def __post_init__(self) -> None:
        self.sums = np.zeros(self.n_annotators, dtype=np.float64)
        self.counts = np.zeros(self.n_annotators, dtype=np.float64)

    def observe(self, annotator: np.ndarray, desirable: np.ndarray) -> None:
        np.add.at(self.sums, annotator, desirable)
        np.add.at(self.counts, annotator, 1.0)

    def values(self) -> np.ndarray:
        signatures = np.divide(
            self.sums,
            np.maximum(self.counts, 1.0),
            out=np.full(self.n_annotators, 0.5, dtype=np.float64),
            where=self.counts > 0,
        )
        signatures[self.counts == 0] = 0.5
        return signatures


class OnlineHardClusterer:
    def __init__(self, n_annotators: int, n_clusters: int, seed: int):
        self.n_clusters = n_clusters
        self.seed = seed
        self.signatures = StreamingSignatures(n_annotators)
        self.labels = np.zeros(n_annotators, dtype=np.int64)
        self.prev_centers: np.ndarray | None = None

    def observe(self, annotator: np.ndarray, desirable: np.ndarray) -> None:
        self.signatures.observe(annotator, desirable)

    def refit(self, step: int) -> dict[str, object]:
        sig = self.signatures.values()
        raw_labels = kmeans_1d(sig, self.n_clusters, self.seed + 100 * step)
        raw_centers = _centers_from_labels(sig, raw_labels, self.n_clusters)
        perm = _match_components(raw_centers, self.prev_centers)
        labels = perm[raw_labels]
        centers = np.zeros_like(raw_centers)
        centers[perm] = raw_centers
        self.labels = labels.astype(np.int64)
        self.prev_centers = centers
        return {"centers": centers.copy(), "coverage": float(self.signatures.counts.mean())}

    def for_annotators(self, annotator: np.ndarray) -> np.ndarray:
        return self.labels[annotator]


class OnlineSoftClusterer:
    def __init__(self, n_annotators: int, n_clusters: int, seed: int):
        self.n_clusters = n_clusters
        self.seed = seed
        self.signatures = StreamingSignatures(n_annotators)
        self.responsibilities = np.zeros((n_annotators, n_clusters), dtype=np.float64)
        self.responsibilities[:, 0] = 1.0
        self.prev_means: np.ndarray | None = None

    def observe(self, annotator: np.ndarray, desirable: np.ndarray) -> None:
        self.signatures.observe(annotator, desirable)

    def refit(self, step: int) -> dict[str, object]:
        raw_w, info = gmm_1d_soft(self.signatures.values(), self.n_clusters, self.seed + 100 * step)
        raw_means = info["means"]
        perm = _match_components(raw_means, self.prev_means)
        responsibilities = np.zeros_like(raw_w)
        responsibilities[:, perm] = raw_w
        means = np.zeros_like(raw_means)
        means[perm] = raw_means
        self.responsibilities = responsibilities
        self.prev_means = means
        return {"centers": means.copy(), "coverage": float(self.signatures.counts.mean())}

    def for_annotators(self, annotator: np.ndarray) -> np.ndarray:
        return self.responsibilities[annotator]


def _centers_from_labels(signatures: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    centers = np.zeros(n_clusters, dtype=np.float64)
    for cluster in range(n_clusters):
        mask = labels == cluster
        centers[cluster] = float(signatures[mask].mean()) if np.any(mask) else 0.5
    return centers
