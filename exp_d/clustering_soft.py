"""Soft clustering on 1-D annotator signatures.

This module is the soft-clustering counterpart to ``exp_d/clustering.py``.
It exposes:

- ``gmm_1d_soft``: 1-D Gaussian Mixture EM that returns the (n_annotators, K)
  responsibility matrix W[a, k] = P(cluster=k | sig(a)).
- ``soft_normalized_mutual_information`` and ``soft_purity``: clustering-quality
  metrics that consume responsibilities directly rather than argmax labels.
  These are the right diagnostics when the ground-truth boundary is fuzzy
  (the stress regime where hard NMI crashes but soft assignment can still
  recover the correct mixing weights).

Nothing in here is imported by the original Experiment D pipeline, so the
existing pre-registered D results stay reproducible bit-for-bit.

The GMM uses a tiny variance floor (``1e-4``) to avoid degenerate components
when a cluster has only a few annotators. K-means++ initialisation is borrowed
from ``exp_d/clustering.kmeans_1d`` to get a deterministic, well-separated
starting point.
"""

import os

import numpy as np

from exp_d.clustering import kmeans_1d


_VAR_FLOOR = 1e-4
_LOG_2PI = float(np.log(2.0 * np.pi))


def gmm_1d_soft(
    signatures: np.ndarray,
    n_clusters: int,
    seed: int,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """Fit a 1-D K-component Gaussian mixture by EM.

    Parameters
    ----------
    signatures:
        Shape ``(n_annotators,)``. Annotator-level summary statistic
        (e.g. empirical desirable-rate from ``compute_signatures``).
    n_clusters:
        Number of mixture components ``K``.
    seed:
        Seed used for K-means++ initialisation. The EM iterations themselves
        are deterministic given the initial means.
    max_iter:
        Hard cap on EM iterations.
    tol:
        Convergence tolerance on the change in log-likelihood per iteration.

    Returns
    -------
    responsibilities:
        Shape ``(n_annotators, n_clusters)``. Row-stochastic: each row sums
        to 1.
    info:
        Dictionary with ``means``, ``variances``, ``weights``,
        ``log_likelihood``, and ``n_iter`` for diagnostics.
    """
    n_annotators = signatures.shape[0]
    if n_clusters == 1:
        responsibilities = np.ones((n_annotators, 1), dtype=np.float64)
        info = {
            "means": np.array([signatures.mean()]),
            "variances": np.array([max(signatures.var(), _VAR_FLOOR)]),
            "weights": np.array([1.0]),
            "log_likelihood": float("nan"),
            "n_iter": 0,
        }
        return responsibilities, info

    # Initialise means with K-means hard labels (stable, deterministic).
    initial_labels = kmeans_1d(signatures, n_clusters, seed)
    means = np.zeros(n_clusters, dtype=np.float64)
    variances = np.zeros(n_clusters, dtype=np.float64)
    weights = np.zeros(n_clusters, dtype=np.float64)
    for k in range(n_clusters):
        mask = initial_labels == k
        if not np.any(mask):
            # Fall back to a quantile if K-means returned an empty cluster.
            means[k] = float(np.quantile(signatures, (k + 1) / (n_clusters + 1)))
            variances[k] = max(signatures.var(), _VAR_FLOOR)
            weights[k] = 1.0 / n_clusters
        else:
            means[k] = float(signatures[mask].mean())
            variances[k] = max(float(signatures[mask].var()), _VAR_FLOOR)
            weights[k] = float(mask.mean())

    prev_log_likelihood = -np.inf
    log_likelihood = -np.inf
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        # E-step: responsibilities from current parameters.
        log_probs = _log_gaussian(signatures, means, variances) + np.log(weights + 1e-12)
        log_sum = _logsumexp_rows(log_probs)
        log_responsibilities = log_probs - log_sum[:, None]
        responsibilities = np.exp(log_responsibilities)
        log_likelihood = float(log_sum.sum())

        # M-step: re-estimate parameters.
        n_eff = responsibilities.sum(axis=0)
        n_eff_safe = np.maximum(n_eff, 1e-12)
        new_means = (responsibilities * signatures[:, None]).sum(axis=0) / n_eff_safe
        diff = signatures[:, None] - new_means[None, :]
        new_variances = (responsibilities * diff * diff).sum(axis=0) / n_eff_safe
        new_variances = np.maximum(new_variances, _VAR_FLOOR)
        new_weights = n_eff / n_annotators

        means = new_means
        variances = new_variances
        weights = new_weights

        if log_likelihood - prev_log_likelihood < tol:
            break
        prev_log_likelihood = log_likelihood

    info = {
        "means": means,
        "variances": variances,
        "weights": weights,
        "log_likelihood": log_likelihood,
        "n_iter": n_iter,
    }
    return responsibilities, info


def hard_random_partition_responsibilities(
    n_annotators: int, n_clusters: int, seed: int
) -> np.ndarray:
    """Random partition expressed as one-hot responsibilities."""
    from exp_d.clustering import random_partition

    labels = random_partition(n_annotators, n_clusters, seed)
    responsibilities = np.zeros((n_annotators, n_clusters), dtype=np.float64)
    responsibilities[np.arange(n_annotators), labels] = 1.0
    return responsibilities


def hard_oracle_responsibilities(annotator_cluster: np.ndarray, n_clusters: int) -> np.ndarray:
    """Oracle ground-truth cluster expressed as one-hot responsibilities."""
    responsibilities = np.zeros((annotator_cluster.shape[0], n_clusters), dtype=np.float64)
    responsibilities[np.arange(annotator_cluster.shape[0]), annotator_cluster] = 1.0
    return responsibilities


def soft_normalized_mutual_information(
    responsibilities: np.ndarray, truth: np.ndarray
) -> float:
    """NMI between a soft assignment and a hard ground-truth labelling.

    Uses the joint distribution P(pred, truth) = (1/N) * sum_a W[a, pred] *
    1{truth[a] = c}, then plugs the resulting contingency into the standard
    arithmetic-mean-normalised mutual information formula.
    """
    n_annotators = responsibilities.shape[0]
    n_clusters = responsibilities.shape[1]
    n_truth = int(truth.max()) + 1

    joint = np.zeros((n_clusters, n_truth), dtype=np.float64)
    for c in range(n_truth):
        mask = truth == c
        if np.any(mask):
            joint[:, c] = responsibilities[mask].sum(axis=0)
    joint /= n_annotators

    pred_mass = joint.sum(axis=1)
    truth_mass = joint.sum(axis=0)

    mutual_info = 0.0
    for i in range(n_clusters):
        for j in range(n_truth):
            value = joint[i, j]
            if value > 0:
                mutual_info += value * np.log(value / (pred_mass[i] * truth_mass[j] + 1e-12))

    pred_entropy = -np.sum(pred_mass * np.log(np.maximum(pred_mass, 1e-12)))
    truth_entropy = -np.sum(truth_mass * np.log(np.maximum(truth_mass, 1e-12)))
    denom = 0.5 * (pred_entropy + truth_entropy)
    if denom <= 0.0:
        return 1.0
    return float(mutual_info / denom)


def soft_purity(responsibilities: np.ndarray, truth: np.ndarray) -> float:
    """Soft-purity: fraction of mass that lands on the best-matching truth class.

    For each predicted cluster k, take the largest column in
    sum_a W[a, k] * 1{truth[a] = c}, then sum across k and divide by N.
    Reduces to standard purity when responsibilities are one-hot.
    """
    n_annotators = responsibilities.shape[0]
    n_truth = int(truth.max()) + 1
    contingency = np.zeros((responsibilities.shape[1], n_truth), dtype=np.float64)
    for c in range(n_truth):
        mask = truth == c
        if np.any(mask):
            contingency[:, c] = responsibilities[mask].sum(axis=0)
    return float(contingency.max(axis=1).sum() / n_annotators)


def _log_gaussian(values: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    # Shape: (n_annotators, n_clusters)
    diff = values[:, None] - means[None, :]
    return -0.5 * (_LOG_2PI + np.log(variances)[None, :] + diff * diff / variances[None, :])


def _logsumexp_rows(matrix: np.ndarray) -> np.ndarray:
    row_max = matrix.max(axis=1)
    return row_max + np.log(np.exp(matrix - row_max[:, None]).sum(axis=1) + 1e-300)
