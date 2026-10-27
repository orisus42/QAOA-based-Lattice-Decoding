"""Toy Module-LWE in coefficient representation.

Ring: Z_q[x]/(x**n + 1). Polynomial coefficients are stored constant-first.
A has shape (k, k, n); s, e, t have shape (k, n).
Flattening orders polynomials first, then their coefficients.
This preserves an algebraic pattern of ML-KEM, not its security parameters,
seed expansion, NTT implementation, serialization, or complete KEM protocol.
"""

from dataclasses import dataclass
import numpy as np
from sympy import isprime


@dataclass(frozen=True)
class Parameters:
    n: int = 2
    k: int = 2
    q: int = 17
    eta: int = 2

    def __post_init__(self):
        if self.n < 2 or self.n & (self.n - 1):
            raise ValueError("n must be a power of two, at least two")
        if self.k < 1 or self.eta < 1:
            raise ValueError("k and eta must be positive")
        if self.q < 3 or not isprime(self.q):
            raise ValueError("q must be an odd prime")


@dataclass
class Instance:
    parameters: Parameters
    seed: int
    A: np.ndarray
    secret: np.ndarray
    error: np.ndarray
    target: np.ndarray


def negacyclic_product(a, b):
    """Integer polynomial product reduced by x**n = -1, before mod q."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    if a.ndim != 1 or a.shape != b.shape or not a.size:
        raise ValueError("polynomials must be nonempty, equally sized vectors")
    n = len(a)
    ordinary = np.convolve(a, b)
    result = ordinary[:n].copy()
    result[:n-1] -= ordinary[n:]
    return result


def module_product(A, vector):
    """Multiply a square matrix of polynomials by a polynomial vector."""
    A, vector = np.asarray(A, dtype=np.int64), np.asarray(vector, dtype=np.int64)
    if A.ndim != 3 or A.shape[0] != A.shape[1] or vector.shape != A.shape[1:]:
        raise ValueError("expected A[k,k,n] and vector[k,n]")
    k, _, n = A.shape
    result = np.zeros((k, n), dtype=np.int64)
    for i in range(k):
        for j in range(k):
            result[i] += negacyclic_product(A[i, j], vector[j])
    return result


def expand_module_matrix(A):
    """The integer matrix for the same polynomial map, built by signed shifts."""
    A = np.asarray(A, dtype=np.int64)
    if A.ndim != 3 or A.shape[0] != A.shape[1]:
        raise ValueError("expected square polynomial matrix A[k,k,n]")
    k, _, n = A.shape
    expanded = np.zeros((k*n, k*n), dtype=np.int64)
    for i in range(k):
        for j in range(k):
            for column in range(n):
                shifted = np.roll(A[i, j], column).copy()
                shifted[:column] *= -1
                expanded[i*n:(i+1)*n, j*n+column] = shifted
    return expanded


def sample_cbd(rng, eta, shape):
    """Sum eta fair bits minus another eta fair bits, independently per entry."""
    if eta < 1:
        raise ValueError("eta must be positive")
    bits = rng.integers(0, 2, size=tuple(shape)+(2*eta,), dtype=np.int64)
    return bits[..., :eta].sum(axis=-1) - bits[..., eta:].sum(axis=-1)


def generate_instance(seed, parameters=None):
    p = parameters or Parameters()
    rng = np.random.default_rng(seed)
    A = rng.integers(0, p.q, size=(p.k, p.k, p.n), dtype=np.int64)
    secret = sample_cbd(rng, p.eta, (p.k, p.n))
    error = sample_cbd(rng, p.eta, (p.k, p.n))
    target = (module_product(A, secret) + error) % p.q
    return Instance(p, int(seed), A, secret, error, target)
