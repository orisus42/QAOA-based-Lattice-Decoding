"""p=1 QAOA with matching circuit, statevector, and analytic energy conventions.

H=sum h_i Z_i + sum_{i<j} J_ij Z_i Z_j; U=exp(-i beta sum X)exp(-i gamma H).
The analytic expectation is established prior work, not our contribution:
Ozaeta, van Dam, McMahon, https://arxiv.org/abs/2012.03421.
"""
import numpy as np
from qiskit import QuantumCircuit


def circuit(h, J, angles):
    gamma, beta = angles
    q = QuantumCircuit(len(h))
    q.h(range(len(h)))
    for i, value in enumerate(h):
        if abs(value) > 1e-12:
            q.rz(2*gamma*value, i)
    for i, j in zip(*np.triu_indices(len(h), 1)):
        if abs(J[i, j]) > 1e-12:
            q.rzz(2*gamma*J[i, j], i, j)
    q.rx(2*beta, range(len(h)))
    return q


def state_probabilities(energy, angles):
    energy = np.asarray(energy, dtype=float)
    nq = int(np.log2(energy.size))
    if energy.shape != (2**nq,):
        raise ValueError("energy must be a nonempty power-of-two vector")
    gamma, beta = angles
    state = np.exp(-1j*gamma*energy)/np.sqrt(energy.size)
    c, s = np.cos(beta), -1j*np.sin(beta)
    for k in range(nq):
        view = state.reshape(-1, 2, 2**k)
        a, b = view[:, 0, :].copy(), view[:, 1, :].copy()
        view[:, 0, :] = c*a+s*b
        view[:, 1, :] = s*a+c*b
    return np.abs(state)**2


class P1Energy:
    """Exact p=1 expectation from h,J, without any decoding reference labels."""
    def __init__(self, h, J):
        self.h = np.asarray(h, dtype=float)
        J = np.asarray(J, dtype=float)
        self.full = np.triu(J, 1)+np.triu(J, 1).T
        self.i, self.j = np.triu_indices(len(h), 1)
        self.edges = self.full[self.i, self.j]
        outside = np.array([[k for k in range(len(h)) if k not in [i, j]]
                            for i, j in zip(self.i, self.j)], dtype=int).reshape(len(self.i), max(0, len(h)-2))
        self.left = self.full[self.i[:, None], outside]
        self.right = self.full[self.j[:, None], outside]
        self.diff = self.left-self.right
        self.plus = self.left+self.right

    def __call__(self, angles):
        gamma, beta = angles
        h, g = self.h, 2*gamma
        single = np.sum(h*np.sin(g*h)*np.cos(g*self.full).prod(axis=1))
        mixed = np.sum(self.edges*np.sin(g*self.edges)*(
            np.cos(g*h[self.i])*np.cos(g*self.left).prod(axis=1)
            +np.cos(g*h[self.j])*np.cos(g*self.right).prod(axis=1)))
        pair = np.sum(self.edges*(
            np.cos(g*(h[self.i]-h[self.j]))*np.cos(g*self.diff).prod(axis=1)
            -np.cos(g*(h[self.i]+h[self.j]))*np.cos(g*self.plus).prod(axis=1)))
        return float(np.sin(2*beta)*single+0.5*np.sin(4*beta)*mixed+0.5*np.sin(2*beta)**2*pair)
