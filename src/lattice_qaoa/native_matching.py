"""Public-only exact native-CZ matching of two frozen phase-mask families.

This is an experimental comparison protocol, not a new decoding algorithm.
Compilation uses symbolic gamma/beta; subsequent angle binding cannot change
the chosen native gate count. No artificial identity gates are added.
"""
import time
import numpy as np
from qiskit import transpile
from qiskit.circuit import Parameter
from .qaoa import circuit
from .cached_pruning import select_masks_cached
from .incumbent_pruning import incumbent_mask
from .residual_pruning import edge_list


def match_counts(resources,caps):
    """Largest jointly attainable CZ count <= cap; ties use largest logical K."""
    output={}
    families=['cached','incumbent']
    for cap in caps:
        feasible={name:{r['native_cz'] for r in resources[name] if r['native_cz']<=cap} for name in families}
        common=feasible['cached']&feasible['incumbent']
        if not common:raise ValueError('no jointly attainable native count')
        count=max(common)
        chosen={name:min((r for r in resources[name] if r['native_cz']==count),
                         key=lambda r:(-r['logical_k'],r.get('duration_unitary_s',0),r.get('routing_seed',0)))
                for name in families}
        ks={name:chosen[name]['logical_k'] for name in families}
        output[int(cap)]={'native_cz':count,'logical_k':ks,'slack':int(cap)-count,
                         'routing_seed':{name:chosen[name].get('routing_seed',0) for name in families}}
    return output


def near_match_counts(resources,caps,tolerance=2):
    """Maximize common resource use under each cap with <=tolerance CZ gap.

    Tie order: smaller gap, more logical terms in total, more balanced K,
    shorter summed duration, then deterministic family K/seed order.
    Original exact intersections remain a separately reported diagnostic.
    """
    if tolerance<0:raise ValueError('nonnegative tolerance required')
    result={}
    for cap in caps:
        options=[]
        for a in resources['cached']:
            for b in resources['incumbent']:
                ca,cb=a['native_cz'],b['native_cz']
                if max(ca,cb)>cap or abs(ca-cb)>tolerance:continue
                key=(-min(ca,cb),abs(ca-cb),-(a['logical_k']+b['logical_k']),
                     -min(a['logical_k'],b['logical_k']),
                     a.get('duration_unitary_s',0)+b.get('duration_unitary_s',0),
                     -a['logical_k'],-b['logical_k'],a.get('routing_seed',0),b.get('routing_seed',0))
                options.append((key,a,b))
        if not options:raise ValueError('no matched pair under cap')
        _,a,b=min(options,key=lambda v:v[0])
        result[int(cap)]={'native_cz':{'cached':a['native_cz'],'incumbent':b['native_cz']},
            'logical_k':{'cached':a['logical_k'],'incumbent':b['logical_k']},
            'routing_seed':{'cached':a.get('routing_seed',0),'incumbent':b.get('routing_seed',0)},
            'absolute_cz_gap':abs(a['native_cz']-b['native_cz']),
            'slack':{'cached':cap-a['native_cz'],'incumbent':cap-b['native_cz']}}
    return result


def compile_families(h,J,target,logical_grid,selection_angles,routing_seeds):
    """Return two routed mask families; only h,J and backend resources enter."""
    grid=sorted(set(logical_grid))
    if grid[0]!=0 or grid[-1]>np.count_nonzero(J):raise ValueError('grid must include0 and fit original edge count')
    start=time.perf_counter()
    greedy,trace=select_masks_cached(h,J,grid,selection_angles)
    cached_seconds=time.perf_counter()-start
    start=time.perf_counter()
    cheap={k:incumbent_mask(J,np.tile([1,-1],len(h)//2),k) for k in grid}
    cheap_seconds=time.perf_counter()-start
    if isinstance(routing_seeds,int):routing_seeds=[routing_seeds]
    gamma=Parameter('gamma');beta=Parameter('beta')
    masks={'cached':{k:greedy[k]['J'] for k in grid},'incumbent':cheap}
    circuits={};resources={}
    for name in masks:
        circuits[name]={};resources[name]=[]
        for k in grid:
            qc=circuit(h,masks[name][k],[gamma,beta]);qc.barrier()
            for routing_seed in routing_seeds:
                start=time.perf_counter()
                routed=transpile(qc,target=target,initial_layout=list(range(len(h))),optimization_level=3,
                                 seed_transpiler=routing_seed,scheduling_method='alap')
                seconds=time.perf_counter()-start
                circuits[name][k,routing_seed]=routed
                resources[name].append({'logical_k':int(k),'routing_seed':routing_seed,'native_cz':int(routed.count_ops().get('cz',0)),
                    'duration_unitary_s':routed.estimate_duration(target),'depth':routed.depth(),
                    'native_sx':int(routed.count_ops().get('sx',0)),'compilation_seconds':seconds,
                    'edges':edge_list(masks[name][k])})
    diagnostics={'mask_selection_seconds':{'cached':cached_seconds,'incumbent':cheap_seconds},
                 'resources':resources,'cached_trace':trace,
                 'analytic_evaluations':greedy[grid[0]]['analytic_expectation_evaluations']}
    return masks,circuits,diagnostics


def bind_angles(template,angles):
    values={'gamma':float(angles[0]),'beta':float(angles[1])}
    bound=template.assign_parameters({p:values[p.name] for p in template.parameters},inplace=False)
    assert not bound.parameters and bound.count_ops().get('cz',0)==template.count_ops().get('cz',0)
    return bound
