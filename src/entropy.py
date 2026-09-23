"""
entropy.py

Implements Shannon entropy calculation over model output logits,
per Equations (1) and (2) of the paper.
"""

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Converts raw logits into a probability distribution.

    Args:
        logits: 1D array of raw logit scores for the vocabulary.

    Returns:
        1D array of probabilities (sums to 1).
    """
    z = logits - np.max(logits)  # numerical stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def shannon_entropy(logits: np.ndarray, base: int = 2) -> float:
    """
    Computes the Shannon entropy H(X) of the token distribution
    derived from raw logits.

    H(X) = - sum_i p(x_i) * log_base(p(x_i))

    Args:
        logits: 1D array of raw logit scores.
        base: Logarithm base (2 for bits, "e" for nats -- pass any
            value other than 2 to use natural log).

    Returns:
        Scalar entropy value.
    """
    probs = softmax(logits)
    probs = probs[probs > 0]  # avoid log(0)
    if base == 2:
        return float(-np.sum(probs * np.log2(probs)))
    return float(-np.sum(probs * np.log(probs)))


def delta_entropy(entropy_base: float, entropy_perturbed: float) -> float:
    """
    Computes the entropy delta between the base and perturbed prompt,
    used to detect the Critical Entropy Threshold.

    Args:
        entropy_base: Entropy of the unperturbed prompt.
        entropy_perturbed: Entropy of the perturbed prompt.

    Returns:
        Delta entropy (positive = increased uncertainty).
    """
    return entropy_perturbed - entropy_base
