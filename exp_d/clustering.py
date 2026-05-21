import os

import numpy as np

from scr.samplers import OfflineDataset


def compute_signatures(dataset: OfflineDataset) -> np.ndarray:
    signatures = np.zeros(dataset.config.n_annotators, dtype=np.float64)
    counts = np.zeros(dataset.config.n_annotators, dtype=np.int64)
    for annotator, desirable in zip(dataset.annotator, dataset.desirable):
        signatures[annotator] += desirable
        counts[annotator] += 1
    return signatures / np.maximum(counts, 1)


def kmeans_1d(
    signatures: np.ndarray,
    n_clusters: int,
    seed: int,
    init_centers: np.ndarray | None = None,
) -> np.ndarray:
    try:
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
        from sklearn.cluster import KMeans

        if init_centers is None:
            init = "k-means++"
            n_init = 10
        else:
            init = np.asarray(init_centers, dtype=np.float64).reshape(n_clusters, 1)
            n_init = 1
        return KMeans(n_clusters=n_clusters, random_state=seed, init=init, n_init=n_init).fit_predict(
            signatures.reshape(-1, 1)
        ).astype(np.int64)
    except ImportError:
        if init_centers is None:
            quantiles = np.linspace(0, 100, n_clusters + 2)[1:-1]
            centers = np.percentile(signatures, quantiles)
        else:
            centers = np.asarray(init_centers, dtype=np.float64).copy()
        for _ in range(100):
            distances = np.abs(signatures[:, None] - centers[None, :])
            labels = distances.argmin(axis=1).astype(np.int64)
            new_centers = centers.copy()
            for cluster in range(n_clusters):
                mask = labels == cluster
                if np.any(mask):
                    new_centers[cluster] = signatures[mask].mean()
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        return labels


def random_partition(n_annotators: int, n_clusters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.zeros(n_annotators, dtype=np.int64)
    for idx, annotator in enumerate(rng.permutation(n_annotators)):
        labels[annotator] = idx % n_clusters
    return labels


def normalized_mutual_information(predicted: np.ndarray, truth: np.ndarray) -> float:
    try:
        from sklearn.metrics import normalized_mutual_info_score

        return float(normalized_mutual_info_score(truth, predicted))
    except ImportError:
        contingency = _contingency(predicted, truth)
        total = contingency.sum()
        pred_mass = contingency.sum(axis=1)
        truth_mass = contingency.sum(axis=0)
        mutual_info = 0.0
        for i in range(contingency.shape[0]):
            for j in range(contingency.shape[1]):
                value = contingency[i, j]
                if value > 0:
                    mutual_info += (value / total) * np.log((value * total) / (pred_mass[i] * truth_mass[j]))
        pred_entropy = -np.sum((pred_mass / total) * np.log(np.maximum(pred_mass / total, 1e-12)))
        truth_entropy = -np.sum((truth_mass / total) * np.log(np.maximum(truth_mass / total, 1e-12)))
        denom = (pred_entropy + truth_entropy) / 2.0
        return float(mutual_info / denom) if denom > 0 else 1.0


def purity(predicted: np.ndarray, truth: np.ndarray) -> float:
    contingency = _contingency(predicted, truth)
    return float(contingency.max(axis=1).sum() / len(predicted))


def _contingency(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    n_pred = int(predicted.max()) + 1
    n_truth = int(truth.max()) + 1
    contingency = np.zeros((n_pred, n_truth), dtype=np.float64)
    for pred_value, truth_value in zip(predicted, truth):
        contingency[pred_value, truth_value] += 1
    return contingency
