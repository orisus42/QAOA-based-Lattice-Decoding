"""Classical references for the explicitly defined small-secret CVP objective.

B = [[qI, A], [0, I]], y = [t, 0], z = [w, s].
||Bz-y||^2 = ||qw+As-t||^2 + ||s||^2, with integer w and s.

The norm objective favors short error AND short secret. It is a surrogate for
the centered-binomial prior, not its exact maximum-likelihood objective.
The implementation uses int64 arithmetic and is intended only for tiny inputs.
"""

from fractions import Fraction
from math import isqrt
import numpy as np
from sympy import Matrix


def embedding(A, target, q):
    A, target = np.asarray(A, dtype=np.int64), np.asarray(target, dtype=np.int64)
    d = len(target)
    if A.shape != (d, d) or target.shape != (d,) or q < 3 or q % 2 == 0:
        raise ValueError("expected square A, vector target, and odd q >= 3")
    I = np.eye(d, dtype=np.int64)
    B = np.block([[q*I, A], [np.zeros_like(I), I]])
    return B, np.concatenate([target, np.zeros(d, dtype=np.int64)])


def lll_reduce(B):
    """Return column basis R and exact unimodular transform U with R = B @ U."""
    rows, transform = Matrix(np.asarray(B).T.tolist()).lll_transform()
    return np.array(rows.T.tolist(), dtype=np.int64), np.array(transform.T.tolist(), dtype=np.int64)


def babai(B, target):
    """Nearest-plane coefficients using rational Gram-Schmidt; ties round up."""
    B, target = np.asarray(B, dtype=np.int64), np.asarray(target, dtype=np.int64)
    if B.shape != (len(target), len(target)):
        raise ValueError("expected a square full-rank column basis")
    columns = [[Fraction(int(x)) for x in B[:, j]] for j in range(B.shape[1])]
    star, norms = [], []
    for column in columns:
        v = column.copy()
        for previous, norm in zip(star, norms):
            mu = sum(x*y for x, y in zip(column, previous)) / norm
            v = [x-mu*y for x, y in zip(v, previous)]
        norm = sum(x*x for x in v)
        if not norm:
            raise ValueError("basis is rank deficient")
        star.append(v)
        norms.append(norm)
    residual = [Fraction(int(x)) for x in target]
    coefficients = np.zeros(len(target), dtype=np.int64)
    for j in range(len(target)-1, -1, -1):
        value = sum(x*y for x, y in zip(residual, star[j])) / norms[j]
        rounded = value + Fraction(1, 2)
        coefficients[j] = rounded.numerator // rounded.denominator
        residual = [x-int(coefficients[j])*y for x, y in zip(residual, columns[j])]
    return coefficients


def squared_cost(B, target, coefficients):
    residual = B @ coefficients - target
    return int(residual @ residual)


def integer_ball(dimension, squared_radius):
    """All integer vectors in a Euclidean ball, using integer bounds only."""
    if dimension < 0 or squared_radius < 0:
        raise ValueError("dimension and squared radius must be nonnegative")

    def visit(prefix, remaining, slots):
        if slots == 0:
            yield tuple(prefix)
            return
        radius = isqrt(remaining)
        for coordinate in range(-radius, radius+1):
            yield from visit(prefix+[coordinate], remaining-coordinate*coordinate, slots-1)

    yield from visit([], int(squared_radius), dimension)


def exact_cvp(A, target, q, incumbent=None, max_candidates=1_000_000):
    """Exhaustive full-lattice CVP, not just a finite arbitrary secret box.

Any solution no worse than a feasible incumbent of cost C has ||s||^2 <= C.
For each such integer s, nearest q-multiples give the optimal w independently.
Thus enumerating this ball certifies the global optimum, including every tie.
Odd q and integer target make the closest w unique for each s.
No planted secret or planted error is an input to this reference solver.
"""
    A, target = np.asarray(A, dtype=np.int64), np.asarray(target, dtype=np.int64)
    B, y = embedding(A, target, q)
    d = len(target)
    if incumbent is None:
        incumbent = np.concatenate([(target + q//2)//q, np.zeros(d, dtype=np.int64)])
    incumbent = np.asarray(incumbent, dtype=np.int64)
    if incumbent.shape != (2*d,):
        raise ValueError("incumbent must contain d wrap coefficients and d secret coefficients")
    bound = squared_cost(B, y, incumbent)
    best, solutions, examined = bound, [], 0
    for coordinates in integer_ball(d, bound):
        examined += 1
        if examined > max_candidates:
            raise RuntimeError("Exact enumeration budget exceeded; no approximate result returned")
        secret = np.array(coordinates, dtype=np.int64)
        wrap = (target - A @ secret + q//2)//q
        coefficients = np.concatenate([wrap, secret])
        cost = squared_cost(B, y, coefficients)
        if cost < best:
            best, solutions = cost, []
        if cost == best:
            solutions.append(coefficients.tolist())
    return {
        "cost": best,
        "minimizers": solutions,
        "initial_cost_bound": bound,
        "secret_vectors_examined": examined,
        "certificate": "All integer s with ||s||^2 <= incumbent cost; exact optimal q-lift for each s.",
    }
