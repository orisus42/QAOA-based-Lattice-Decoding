"""Public-coefficient phase sparsification; never changes the scored objective.

Inspired by Liu, Shaydulin, Safro (QCE 2022), arXiv:2205.00118.
Magnitude pruning is a simple heuristic, not their ground-state-preserving
construction, and carries no guarantee of retaining minimizers or success.
"""
import numpy as np


def prune_interactions(J, fraction, *, random_seed=None):
    """Drop floor(fraction * nonzero edges), with deterministic lexicographic ties."""
    J = np.asarray(J, dtype=float)
    if J.ndim != 2 or J.shape[0] != J.shape[1] or not 0 <= fraction <= 1:
        raise ValueError("square interaction matrix and fraction in [0,1] required")
    if not np.allclose(J, np.triu(J, 1), atol=0, rtol=0):
        raise ValueError("J must contain only strict upper-triangular couplings")
    edges = np.argwhere(np.abs(J) > 1e-12)
    if random_seed is None:
        order = sorted(range(len(edges)), key=lambda k: (abs(J[tuple(edges[k])]), *edges[k]))
    else:
        order = np.random.default_rng(random_seed).permutation(len(edges))
    removed = edges[np.asarray(order[:int(np.floor(fraction*len(edges)))], dtype=int)]
    sparse = J.copy()
    if len(removed):
        sparse[removed[:, 0], removed[:, 1]] = 0
    return sparse, removed


def diagonal_energy(h, J):
    """Enumerate phase eigenvalues only for these small verification experiments."""
    n = len(h)
    spins = 1-2*((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
    return spins@h+np.sum((spins@J)*spins, axis=1)


def expected_best_cost(costs, probabilities, shots, incumbent=None):
    """Exact expectation of the minimum cost in independent samples, with ties."""
    if shots < 1 or int(shots) != shots:
        raise ValueError("positive integer shot budget required")
    c = np.asarray(costs)
    p = np.asarray(probabilities, dtype=float)
    if c.shape != p.shape or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("cost/probability shape and normalization mismatch")
    if incumbent is not None:
        c = np.minimum(c, incumbent)
    values, inverse = np.unique(c, return_inverse=True)
    masses = np.bincount(inverse, weights=p, minlength=len(values))
    tail = np.clip(1-np.cumsum(masses)[:-1], 0, 1)
    return float(values[0]+np.dot(np.diff(values), tail**shots))
