import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_quality(theta: np.ndarray, q: np.ndarray) -> float:
    policy = softmax(theta)
    return float(np.mean(np.sum(policy * q, axis=1)))

