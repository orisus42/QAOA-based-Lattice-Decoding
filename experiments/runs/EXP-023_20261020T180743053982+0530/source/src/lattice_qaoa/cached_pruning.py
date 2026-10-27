"""Cached fixed-angle p=1 evaluator for the existing greedy pruning rule.

This changes implementation cost, not the selected objective or search policy.
Every retained coefficient is its original value or zero. Therefore all
trigonometric factors can be tabulated once per instance and angle pair.
No statevector, codebook, reference optimum or planted secret is used.
"""
import time
import numpy as np
from .mechanisms import BatchP1Energy,masked


class FixedAngleMaskEnergy:
    """Exact analytic original-objective energy for batches of boolean masks."""
    def __init__(self,h,J,angles):
        self.h=np.asarray(h,dtype=float);self.J=np.asarray(J,dtype=float)
        n=len(self.h)
        if self.J.shape!=(n,n) or np.any(np.tril(self.J)!=0):
            raise ValueError('upper triangular J required')
        if not np.all(np.isfinite(self.J)) or not np.all(np.isfinite(self.h)):
            raise ValueError('finite coefficients required')
        if len(angles)!=2:raise ValueError('one gamma,beta pair required')
        gamma,beta=angles;g=2*gamma
        self.i,self.j=np.triu_indices(n,1)
        self.outside=np.array([[k for k in range(n) if k not in (i,j)]
                             for i,j in zip(self.i,self.j)],dtype=int).reshape(len(self.i),max(n-2,0))
        full=self.J+self.J.T
        left=full[self.i[:,None],self.outside];right=full[self.j[:,None],self.outside]
        self.cos_full=np.cos(g*full)
        self.sin_edges=np.sin(g*full[self.i,self.j])
        self.cos_left=np.cos(g*left);self.cos_right=np.cos(g*right)
        self.cos_diff=np.cos(g*(left-right));self.cos_plus=np.cos(g*(left+right))
        self.sin_h=np.sin(g*self.h);self.cos_h=np.cos(g*self.h)
        self.cos_hdiff=np.cos(g*(self.h[self.i]-self.h[self.j]))
        self.cos_hplus=np.cos(g*(self.h[self.i]+self.h[self.j]))
        self.score_edges=self.J[self.i,self.j]
        self.prefactors=(np.sin(2*beta),.5*np.sin(4*beta),.5*np.sin(2*beta)**2)

    def values(self,upper_masks):
        masks=np.asarray(upper_masks,dtype=bool)
        if masks.ndim!=3 or masks.shape[1:]!=self.J.shape:raise ValueError('batch of square masks required')
        full=masks|masks.transpose(0,2,1)
        left=full[:,self.i[:,None],self.outside]
        right=full[:,self.j[:,None],self.outside]
        lc=np.where(left,self.cos_left,1.)
        rc=np.where(right,self.cos_right,1.)
        both=left&right
        # cos(-x)=cos(x); a factor with neither edge present is exactly one.
        only=np.where(left,self.cos_left,np.where(right,self.cos_right,1.))
        diff=np.where(both,self.cos_diff,only)
        plus=np.where(both,self.cos_plus,only)
        single=np.sum(self.h*self.sin_h*np.where(full,self.cos_full,1.).prod(axis=2),axis=1)
        mixed=np.sum(self.score_edges*np.where(full[:,self.i,self.j],self.sin_edges,0.)*(
            self.cos_h[self.i]*lc.prod(axis=2)+self.cos_h[self.j]*rc.prod(axis=2)),axis=1)
        pair=np.sum(self.score_edges*(self.cos_hdiff*diff.prod(axis=2)-self.cos_hplus*plus.prod(axis=2)),axis=1)
        a,b,c=self.prefactors
        return a*single+b*mixed+c*pair


def select_masks_cached(h,J,budgets,angles):
    """Same backward greedy path and magnitude fallback as select_masks.

    Near-tied decisions are recomputed with the original evaluator. This
    prevents a cache/reduction-roundoff difference from changing the path.
    Fallback comparisons also use the original evaluator. Analytic-call counts
    include tie recomputation: this optimization reduces CPU work per score,
    not the number of greedy decisions or circuit interactions.
    """
    started=time.perf_counter()
    h=np.asarray(h,dtype=float);J=np.asarray(J,dtype=float)
    cache=FixedAngleMaskEnergy(h,J,angles)
    edges=[(int(i),int(j)) for i,j in zip(*np.triu_indices(len(h),1)) if abs(J[i,j])>1e-12]
    M=len(edges);budgets=sorted(set(budgets),reverse=True)
    if not budgets or any(not isinstance(k,(int,np.integer)) or k<0 or k>M for k in budgets):
        raise ValueError('invalid gate budget')
    kept=J.copy();remaining=edges.copy();trace=[];result={}
    evaluations=0;tie_recomputations=0
    def record(k):
        nonlocal evaluations
        ordinary=masked(J,M-k)[0]
        values=BatchP1Energy([h,h],[kept,ordinary],[J,J]).values(angles);evaluations+=2
        fallback=bool(values[1]<=values[0]+1e-14)
        chosen=ordinary if fallback else kept
        result[k]={'J':chosen.copy(),'greedy_J':kept.copy(),'selection_energy':float(min(values)),
                   'greedy_energy':float(values[0]),'magnitude_energy':float(values[1]),
                   'used_magnitude_fallback':fallback,'analytic_expectation_evaluations':evaluations,
                   'tie_recomputations':tie_recomputations,'selection_seconds':time.perf_counter()-started,
                   'logical_rzz':k,'trace_prefix_length':len(trace)}
        assert np.count_nonzero(chosen)==k
    if M in budgets:record(M)
    while len(remaining)>min(budgets):
        e=np.asarray(remaining,dtype=int)
        masks=np.broadcast_to(kept!=0,(len(remaining),*J.shape)).copy()
        masks[np.arange(len(e)),e[:,0],e[:,1]]=False
        values=cache.values(masks);evaluations+=len(e)
        close=np.flatnonzero(values<=values.min()+1e-12*(1+np.max(np.abs(values))))
        if len(close)>1:
            trials=[np.where(masks[a],J,0.) for a in close]
            original=BatchP1Energy([h]*len(close),trials,[J]*len(close)).values(angles)
            winner=int(close[int(np.argmin(original))])
            value=float(np.min(original));tie_recomputations+=len(close);evaluations+=len(close)
        else:
            winner=int(np.argmin(values));value=float(values[winner])
        edge=remaining.pop(winner);kept[edge]=0.
        trace.append({'removed_edge':list(edge),'remaining':len(remaining),'original_energy':value,
                      'cumulative_expectation_evaluations':evaluations})
        if len(remaining) in budgets:record(len(remaining))
    return result,trace
