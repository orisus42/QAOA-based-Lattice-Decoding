"""Exact finite-budget statistics, separate from solver inputs and decisions."""
from hashlib import sha256
import numpy as np
from .sampling import repeated_success


def matrix_group(A, q):
    """Canonical public lattice group for the fixed augmented embedding."""
    values = np.asarray(A, dtype='<i8') % q
    shape = np.asarray(values.shape, dtype='<i8')
    return sha256(shape.tobytes() + values.astype('<i8').tobytes()).hexdigest()


def distribution_metrics(costs, probabilities, incumbent, optimum, budgets):
    """Group equal costs once; evaluate exact minima, retaining Babai on ties."""
    costs = np.asarray(costs)
    p = np.asarray(probabilities, dtype=float)
    if costs.shape != p.shape or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError('invalid probability distribution')
    p = p / p.sum()
    values, inverse = np.unique(costs, return_inverse=True)
    mass = np.bincount(inverse, weights=p, minlength=len(values))
    # Reverse cumulative sum avoids subtracting nearly one near the upper tail.
    survival = np.r_[np.cumsum(mass[::-1])[::-1], 0.]
    survival[0] = 1.
    survival = np.clip(survival, 0, 1)
    hybrid_values = np.minimum(values, incumbent)
    p_opt = float(np.clip(p[costs == optimum].sum(), 0, 1))
    p_improve = float(np.clip(p[costs < incumbent].sum(), 0, 1))
    result = {}
    for budget in budgets:
        if budget < 1 or int(budget) != budget:
            raise ValueError('positive integer budgets required')
        pmf = survival[:-1]**budget - survival[1:]**budget
        assert pmf.min() >= -1e-12 and np.isclose(pmf.sum(), 1)
        result[str(budget)] = {
            'standalone_success': float(repeated_success(p_opt, budget)),
            'hybrid_success': 1. if incumbent == optimum else float(repeated_success(p_opt, budget)),
            'strict_improvement': float(repeated_success(p_improve, budget)),
            'standalone_expected_cost': float(pmf @ values),
            'standalone_expected_distance': float(pmf @ np.sqrt(values)),
            'hybrid_expected_cost': float(pmf @ hybrid_values),
            'hybrid_expected_distance': float(pmf @ np.sqrt(hybrid_values)),
        }
    return p_opt, p_improve, result


def paired_retention(babai, full, candidate, bootstrap_indices):
    """Ratio of mean extra improvements; bootstrap matched independent groups."""
    base = np.asarray(babai) - np.asarray(full)
    new = np.asarray(babai) - np.asarray(candidate)
    if base.sum() <= 0:
        return {'estimate': None, 'ci95': None, 'undefined_bootstraps': len(bootstrap_indices)}
    denominators = base[bootstrap_indices].sum(axis=1)
    numerators = new[bootstrap_indices].sum(axis=1)
    valid = denominators > 0
    ratios = numerators[valid] / denominators[valid]
    return {'estimate': float(new.sum()/base.sum()),
            'ci95': np.quantile(ratios, [.025, .975]).tolist() if valid.any() else None,
            'undefined_bootstraps': int((~valid).sum())}
