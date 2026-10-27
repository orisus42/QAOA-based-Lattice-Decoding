"""Experimental grouped phase surrogate, not an established novel algorithm.

Keep two-bit integer coordinates intact. Linearize deleted cross-coordinate
couplings about a public incumbent. This is related to existing QUBO splitting
and linearization methods; the caller always scores the original objective.
"""
import numpy as np


def coordinate_mask(J, gate_cap):
    J=np.asarray(J,dtype=float)
    n=len(J)
    if n%2 or J.shape!=(n,n) or np.any(np.tril(J)!=0):
        raise ValueError('expected upper-triangular two-bit-coordinate couplings')
    out=np.zeros_like(J)
    for i in range(0,n,2):out[i,i+1]=J[i,i+1]
    used=np.count_nonzero(out)
    if gate_cap<used:raise ValueError('cap cannot retain all coordinate interiors')
    groups=[]
    for a in range(n//2):
        for b in range(a+1,n//2):
            edges=[(i,j) for i in [2*a,2*a+1] for j in [2*b,2*b+1] if J[i,j]!=0]
            if edges:groups.append((sum(J[e]**2 for e in edges),a,b,edges))
    for _,a,b,edges in sorted(groups,key=lambda g:(-g[0],g[1],g[2])):
        if used+len(edges)<=gate_cap:
            for e in edges:out[e]=J[e]
            used+=len(edges)
    return out


def centered_fields(h,J,retained,reference_spins):
    """Return corrected fields and constant; exact on single-coordinate moves.

    E(s)-E_surrogate(s) = sum_dropped J_ij (s_i-b_i)(s_j-b_j).
    Only masks that retain each complete within-coordinate term provide the
    single-coordinate guarantee. The identity itself holds for any edge mask.
    """
    h=np.asarray(h,dtype=float);J=np.asarray(J,dtype=float)
    kept=np.asarray(retained,dtype=float);b=np.asarray(reference_spins)
    if b.shape!=h.shape or not np.all(np.isin(b,[-1,1])):
        raise ValueError('reference must be a public incumbent spin vector')
    if not np.all((kept==0)|(kept==J)):raise ValueError('retained must be an edge mask')
    dropped=J-kept
    return h+(dropped+dropped.T)@b, -float(b@dropped@b)
