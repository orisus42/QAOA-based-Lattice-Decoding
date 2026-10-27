"""Measurement metrics, conditional on fixed per-instance ideal probabilities."""
import numpy as np


def repeated_success(probabilities, shots):
    """Probability of at least one optimum among independent identical shots."""
    probabilities = np.asarray(probabilities, dtype=float)
    if shots < 0 or int(shots) != shots:
        raise ValueError("shots must be a nonnegative integer")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if shots == 0:
        return np.zeros_like(probabilities)
    with np.errstate(divide="ignore"):
        return -np.expm1(shots*np.log1p(-probabilities))


def solved_count_distribution(probabilities):
    """Poisson-binomial distribution of solved instances with independent shots.

This describes measurement variability on a fixed cohort, not uncertainty about
the wider population of lattice problems or hardware noise.
"""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError("expected a vector of per-instance probabilities")
    repeated_success(probabilities, 1)  # validate domain
    distribution = np.array([1.0])
    for p in probabilities:
        distribution = np.convolve(distribution, [1-p, p])
    return distribution / distribution.sum()


def solved_count_interval(probabilities):
    distribution = solved_count_distribution(probabilities)
    return np.searchsorted(np.cumsum(distribution), [0.025, 0.975]).tolist()
