"""Public-residual p=1 phase selection under a logical RZZ gate budget.

An experimental heuristic, not a novelty or optimal-sampling guarantee.
Selection uses polynomial-size analytic expectations, never the 2**n codebook.
"""
import time
import numpy as np
from .reference import babai
from .mechanisms import BatchP1Energy,masked
from .qaoa import P1Energy


def public_phase(basis,target,center=None,ignore_residual=False):
    B=np.asarray(basis,dtype=np.int64);y=np.asarray(target,dtype=np.int64)
    if B.shape!=(len(y),len(y)):raise ValueError('expected square column basis')
    z=babai(B,y) if center is None else np.asarray(center,dtype=np.int64)
    r=B@z-y
    if ignore_residual:r=np.zeros_like(r)
    W=np.zeros((len(y),2*len(y)),dtype=np.int64)
    for i in range(len(y)):W[i,2*i:2*i+2]=[1,2]
    D=B@W
    # The binary box midpoint is z0-1/2 in each integer coordinate.
    midpoint=r-.5*B.sum(axis=1)
    h=-(midpoint@D);J=np.triu(D.T@D/2,1)
    scale=float(np.sqrt(h@h+np.sum(J*J)))
    if not scale:raise ValueError('constant objective')
    return h/scale,J/scale,scale,r


def select_masks(h,J,budgets,angles):
    """Backward greedy deletion plus magnitude fallback at each requested K.

    At each step minimize expectation of the ORIGINAL h,J, at fixed angles.
    Ties use ascending (i,j). At a budget K, also score ordinary magnitude
    pruning and keep it if no worse. This guarantees no worse mean energy at
    the selection angles, not a better probability of sampling an optimum.
    Budgets count logical nonzero RZZ terms, not routed native gates or time.
    """
    started=time.perf_counter();h=np.asarray(h,dtype=float);J=np.asarray(J,dtype=float)
    if J.shape!=(len(h),len(h)) or np.any(np.tril(J)!=0):raise ValueError('upper triangular J required')
    edges=[(int(i),int(j)) for i,j in zip(*np.triu_indices(len(h),1)) if abs(J[i,j])>1e-12]
    M=len(edges);budgets=sorted(set(budgets),reverse=True)
    if not budgets or any(not isinstance(k,(int,np.integer)) or k<0 or k>M for k in budgets):raise ValueError('invalid gate budget')
    kept=J.copy();remaining=edges.copy();evaluations=0;trace=[];result={}
    def record(k,path_value=None):
        nonlocal evaluations
        ordinary=masked(J,M-k)[0]
        values=BatchP1Energy([h,h],[kept,ordinary],[J,J]).values(angles);evaluations+=2
        fallback=bool(values[1]<=values[0]+1e-14)
        chosen=ordinary if fallback else kept
        result[k]={'J':chosen.copy(),'greedy_J':kept.copy(),'selection_energy':float(min(values)),
            'greedy_energy':float(values[0]),'magnitude_energy':float(values[1]),'used_magnitude_fallback':fallback,
            'analytic_expectation_evaluations':evaluations,'selection_seconds':time.perf_counter()-started,
            'logical_rzz':k,'trace_prefix_length':len(trace)}
        assert np.count_nonzero(chosen)==k
    if M in budgets:record(M)
    while len(remaining)>min(budgets):
        candidates=[]
        for edge in remaining:
            trial=kept.copy();trial[edge]=0.;candidates.append(trial)
        values=BatchP1Energy([h]*len(candidates),candidates,[J]*len(candidates)).values(angles)
        evaluations+=len(candidates);winner=int(np.argmin(values));edge=remaining.pop(winner)
        kept[edge]=0.
        trace.append({'removed_edge':list(edge),'remaining':len(remaining),'original_energy':float(values[winner]),'cumulative_expectation_evaluations':evaluations})
        if len(remaining) in budgets:record(len(remaining))
    return result,trace


def mask_from_edges(J,edges):
    out=np.zeros_like(J)
    for i,j in edges:out[i,j]=J[i,j]
    return out


def edge_list(J):return [[int(i),int(j)] for i,j in zip(*np.nonzero(J))]
