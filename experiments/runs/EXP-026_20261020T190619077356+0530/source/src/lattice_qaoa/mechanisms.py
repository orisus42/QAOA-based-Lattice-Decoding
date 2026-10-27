"""Development ablations. No reference optimum or secret enters these rules."""
import numpy as np
from .qaoa import P1Energy


def ordered_edges(J, policy='weak', seed=0):
    edges = [(int(i),int(j)) for i,j in zip(*np.triu_indices(len(J),1)) if abs(J[i,j])>1e-12]
    rng = np.random.default_rng(seed)
    if policy=='random':
        rng.shuffle(edges)
        return edges
    if policy=='ties':
        # Randomize only equal-magnitude order, never the magnitude ranking.
        tie = {e:float(rng.random()) for e in edges}
        return sorted(edges,key=lambda e:(abs(J[e]),tie[e]))
    key = lambda e:(abs(J[e]),e[0],e[1])
    if policy=='weak': return sorted(edges,key=key)
    if policy=='strong': return sorted(edges,key=lambda e:(-abs(J[e]),e[0],e[1]))
    if policy=='protect_coordinate': return sorted(edges,key=lambda e:(e[0]//2==e[1]//2,*key(e)))
    if policy=='drop_coordinate': return sorted(edges,key=lambda e:(e[0]//2!=e[1]//2,*key(e)))
    raise ValueError(policy)


def masked(J, count, policy='weak', seed=0):
    edges=ordered_edges(J,policy,seed)
    if not 0<=count<=len(edges): raise ValueError('invalid edge count')
    out=np.array(J,copy=True)
    for edge in edges[:count]: out[edge]=0.
    return out,edges[:count]


def weight90(J):
    """Prior-art style: retain >=90% of pair squared weight with fewest edges."""
    edges=ordered_edges(J,'strong');out=np.zeros_like(J)
    target=.9*np.sum(J*J);acc=0.
    for edge in edges:
        if acc>=target:break
        out[edge]=J[edge];acc+=J[edge]**2
    return out


class BatchP1Energy:
    """Vectorized exact p=1 original-objective expectation; optional split phases."""
    def __init__(self, hs, phase_Js, score_Js, score_hs=None):
        score_hs=hs if score_hs is None else score_hs
        if not len(hs)==len(phase_Js)==len(score_Js)==len(score_hs):
            raise ValueError('mismatched training cohort lengths')
        models=[P1Energy(h,j,observable_h=sh,observable_J=s)
                for h,j,sh,s in zip(hs,phase_Js,score_hs,score_Js)]
        if not models: raise ValueError('empty training cohort')
        self.i,self.j=models[0].i,models[0].j
        for name in ['h','full','score_h','score_edges','edges','left','right','diff','plus']:
            setattr(self,name,np.stack([getattr(m,name) for m in models]))

    def values(self,angles):
        if len(angles)==2: gh=gj=2*angles[0];beta=angles[1]
        else: gh,gj,beta=2*angles[0],2*angles[1],angles[2]
        h=self.h
        single=np.sum(self.score_h*np.sin(gh*h)*np.cos(gj*self.full).prod(axis=2),axis=1)
        mixed=np.sum(self.score_edges*np.sin(gj*self.edges)*(
            np.cos(gh*h[:,self.i])*np.cos(gj*self.left).prod(axis=2)
            +np.cos(gh*h[:,self.j])*np.cos(gj*self.right).prod(axis=2)),axis=1)
        pair=np.sum(self.score_edges*(
            np.cos(gh*(h[:,self.i]-h[:,self.j]))*np.cos(gj*self.diff).prod(axis=2)
            -np.cos(gh*(h[:,self.i]+h[:,self.j]))*np.cos(gj*self.plus).prod(axis=2)),axis=1)
        return np.sin(2*beta)*single+.5*np.sin(4*beta)*mixed+.5*np.sin(2*beta)**2*pair

    def __call__(self,angles):return float(self.values(angles).mean())


class CachedMetrics:
    """Repeated minima statistics with a fixed cost codebook; optional secret audit."""
    def __init__(self,costs,incumbent,optimum,budgets,secret_mask=None,babai_secret=False):
        self.costs=np.asarray(costs);self.incumbent=incumbent;self.optimum=optimum;self.budgets=budgets
        self.values,self.inverse=np.unique(costs,return_inverse=True)
        self.secret_mask=secret_mask;self.babai_secret=babai_secret

    def __call__(self,p):
        p=np.asarray(p,dtype=float);assert abs(p.sum()-1)<1e-10 and p.min()>=0
        p=p/p.sum();mass=np.bincount(self.inverse,weights=p,minlength=len(self.values))
        survival=np.r_[np.cumsum(mass[::-1])[::-1],0.];survival[0]=1;survival=np.clip(survival,0,1)
        po=float(p[self.costs==self.optimum].sum());pi=float(p[self.costs<self.incumbent].sum())
        result={'p_optimum':po,'p_improve':pi,'by_budget':{}}
        if self.secret_mask is not None:
            sm=np.bincount(self.inverse,weights=p*self.secret_mask,minlength=len(mass))
            ratios=np.divide(sm,mass,out=np.zeros_like(sm),where=mass>0)
            recovered=np.where(self.values<self.incumbent,ratios,float(self.babai_secret))
            result['p_sample_secret']=float(p[self.secret_mask].sum())
        for shots in self.budgets:
            pmf=survival[:-1]**shots-survival[1:]**shots
            hit=lambda v:float(-np.expm1(shots*np.log1p(-min(v,1-1e-16)))) if v<1 else 1.
            entry={'hybrid_success':1. if self.incumbent==self.optimum else hit(po),
                'standalone_success':hit(po),'strict_improvement':hit(pi),
                'hybrid_expected_distance':float(pmf@np.sqrt(np.minimum(self.values,self.incumbent))),
                'hybrid_expected_cost':float(pmf@np.minimum(self.values,self.incumbent))}
            if self.secret_mask is not None:
                entry['any_sample_secret_probability']=hit(result['p_sample_secret'])
                entry['hybrid_returned_secret_probability']=float(pmf@recovered)
            result['by_budget'][str(shots)]=entry
        return result
