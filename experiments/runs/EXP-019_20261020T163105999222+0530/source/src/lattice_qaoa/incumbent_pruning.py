"""Budgeted signed-Ising adaptation of incumbent-informed MaxCut pruning.

Inspired by Liu, Shaydulin and Safro (2022), Algorithms 1/2. This is not an
exact reproduction: weighted, signed lattice Ising models include fields,
and a fixed interaction cap is imposed instead of removing all uncut edges.
"""
import numpy as np


def incumbent_mask(J, reference_spins, budget):
    J=np.asarray(J, dtype=float); b=np.asarray(reference_spins)
    if J.shape!=(len(b),len(b)) or np.any(np.tril(J)!=0):
        raise ValueError('Expected upper-triangular interactions')
    if not np.all(np.isin(b,[-1,1])):
        raise ValueError('Reference must contain spins -1 or +1')
    edges=[(int(i),int(j)) for i,j in zip(*np.nonzero(J))]
    if not 0<=budget<=len(edges):raise ValueError('Invalid budget')
    # Minimize H: negative signed contributions favour the incumbent.
    # Keep the most favourable contributions; deterministic index tie-break.
    ranked=sorted(edges,key=lambda edge:(J[edge]*b[edge[0]]*b[edge[1]],*edge))
    out=np.zeros_like(J)
    for edge in ranked[:budget]:out[edge]=J[edge]
    return out
