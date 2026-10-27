"""Public-data-only bounded lattice-to-Ising encoding for tiny experiments."""
from dataclasses import dataclass
import numpy as np
from .reference import babai


@dataclass
class BinaryBox:
    basis: np.ndarray
    target: np.ndarray
    center: np.ndarray
    offset: np.ndarray
    weights: np.ndarray
    costs: np.ndarray
    h: np.ndarray
    J: np.ndarray
    constant: float
    normalization: float
    energy: np.ndarray

    @property
    def nq(self):
        return len(self.h)

    def decode(self, bitstrings):
        values = np.asarray(bitstrings, dtype=np.int64).reshape(-1)
        if np.any((values < 0) | (values >= 2**self.nq)):
            raise ValueError("bitstring outside encoded register")
        bits = (values[:, None] >> np.arange(self.nq)) & 1
        return self.offset + bits @ self.weights.T


def encode_babai_box(basis, target):
    """Two bits per coefficient: z_i in {z0_i-2, z0_i-1, z0_i, z0_i+1}.

The center uses Babai on the supplied (normally LLL-reduced) column basis.
No optimum, planted secret, or error is an input. This is not a coverage proof.
All bitstring costs are retained only because this experiment is tiny.
"""
    B, y = np.asarray(basis, dtype=np.int64), np.asarray(target, dtype=np.int64)
    d = len(y)
    if B.shape != (d, d) or 2*d > 20:
        raise ValueError("expected a square basis and at most 20 encoded qubits")
    center = babai(B, y)
    nq = 2*d
    W = np.zeros((d, nq), dtype=np.int64)
    for i in range(d):
        W[i, 2*i:2*i+2] = [1, 2]
    offset = center-2
    D, r = B@W, B@offset-y
    midpoint = r+D.sum(axis=1)/2
    h = -(midpoint@D)
    J = np.triu(D.T@D/2, 1)
    constant = float(midpoint@midpoint+np.sum(D*D)/4)
    scale = float(np.sqrt(h@h+np.sum(J*J)))
    if not scale:
        raise ValueError("constant objective has no meaningful QAOA normalization")
    bits = (np.arange(2**nq)[:, None] >> np.arange(nq)) & 1
    residuals = bits@D.T+r
    costs = np.sum(residuals*residuals, axis=1)
    # Exhaustive independent equality check of the Ising expression.
    spins = 1-2*bits
    recovered = constant+spins@h+np.sum((spins@J)*spins, axis=1)
    np.testing.assert_allclose(recovered, costs, atol=1e-9, rtol=0)
    return BinaryBox(B, y, center, offset, W, costs, h/scale, J/scale,
                     constant, scale, (costs-constant)/scale)
