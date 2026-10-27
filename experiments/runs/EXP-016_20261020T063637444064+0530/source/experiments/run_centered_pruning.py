"""EXP-016: test a concrete candidate, retaining unfavorable outcomes."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.optimize import differential_evolution
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.centered import coordinate_mask,centered_fields
from lattice_qaoa.mechanisms import BatchP1Energy,CachedMetrics,masked
from lattice_qaoa.sparsify import diagonal_energy
from lattice_qaoa.qaoa import state_probabilities,circuit
from lattice_qaoa.evaluation import matrix_group
from lattice_qaoa.instances import expand_module_matrix
from research_run import ResearchRun,save


def variants(h,J,percentages):
    out={'full':(h,J,0.)};M=np.count_nonzero(J)
    b=np.tile([1,-1],len(h)//2) # offset=center-2, binary weights [1,2]
    for pct in percentages:
        kept=coordinate_mask(J,M-int(pct*M/100));K=np.count_nonzero(kept)
        sh,c=centered_fields(h,J,kept,b)
        out[f'centered{pct}']=(sh,kept,c)
        out[f'grouped{pct}']=(h,kept,0.)
        out[f'matched_magnitude{pct}']=(h,masked(J,M-K)[0],0.)
    return out


def main():
    cfg=json.loads((ROOT/'experiments/configs/EXP-016.json').read_text())
    src=ROOT/cfg['development_run']/'instances.json';trp=ROOT/cfg['training_run']/'training.json'
    run=ResearchRun('EXP-016',Path(__file__),[src,trp])
    try:
        tr=json.loads(trp.read_text())['instances'];cases=json.loads(src.read_text())
        train_groups={matrix_group(expand_module_matrix(np.array(c['A_polynomials'])),17) for c in tr}
        dev_groups={matrix_group(c['A_expanded'],17) for c in cases}
        assert len(train_groups)==16 and len(dev_groups)==64 and not train_groups&dev_groups
        hs=[np.array(c['h']) for c in tr];js=[np.array(c['J']) for c in tr]
        v=[variants(h,j,cfg['percentages']) for h,j in zip(hs,js)]
        spec=cfg['optimizer'];bounds=np.array([spec['gamma_bounds'],spec['beta_bounds']]);fits={}
        for name in v[0]:
            objective=BatchP1Energy([x[name][0] for x in v],[x[name][1] for x in v],js,score_hs=hs)
            rng=np.random.default_rng(spec['seed']);init=rng.uniform(bounds[:,0],bounds[:,1],(spec['population'],2));init[0]=cfg['angles']
            fit=differential_evolution(objective,bounds,init=init,seed=spec['seed'],maxiter=spec['generations'],tol=0,atol=0,polish=False,workers=1,updating='immediate')
            assert fit.nfev==spec['evaluation_budget']
            fits[name]={'angles':fit.x.tolist(),'training_mean_original_energy':float(fit.fun),'evaluations':fit.nfev,'optimizer_success':bool(fit.success),'message':str(fit.message)}
            print('Fit',name,round(fit.fun,6),flush=True)
        save(run.path/'training.json',fits)
        (run.path/'training_frozen_at.txt').write_text(__import__('datetime').datetime.now().astimezone().isoformat())
        rows=[];checks={'statevectors':0,'independent_qiskit_checks':0,'max_qiskit_error':0.,'single_coordinate_states_checked':0,'max_preservation_error':0.,'train_dev_overlap':0}
        for ci,x in enumerate(cases):
            box=encode_babai_box(np.array(x['reduced_basis']),np.r_[x['public_target'],np.zeros(4,dtype=int)])
            metric=CachedMetrics(box.costs,x['lll_babai_cost'],x['reference']['cost'],cfg['budgets'])
            vs=variants(box.h,box.J,cfg['percentages'])
            # Independently identify all encoded states changing <=one INTEGER coordinate.
            decoded=box.decode(np.arange(len(box.costs)))
            one=np.sum(decoded!=box.center,axis=1)<=1
            for name,(h,j,c) in vs.items():
                energy=diagonal_energy(h,j)
                if name.startswith('centered'):
                    delta=float(np.max(abs(box.energy[one]-energy[one]-c)))
                    assert delta<1e-10
                    checks['single_coordinate_states_checked']+=int(one.sum())
                    checks['max_preservation_error']=max(checks['max_preservation_error'],delta)
                for policy,a in [('same',cfg['angles']),('trained',fits[name]['angles'])]:
                    p=state_probabilities(energy,a);checks['statevectors']+=1
                    rows.append({'id':x['id'],'index':ci,'method':name,'angles_policy':policy,'angles':a,
                        'logical_rzz':int(np.count_nonzero(j)),**metric(p)})
                    if ci==0:
                        independent=Statevector.from_instruction(circuit(h,j,a)).probabilities()
                        delta=float(np.max(abs(p-independent)));assert delta<1e-10
                        checks['independent_qiskit_checks']+=1;checks['max_qiskit_error']=max(checks['max_qiskit_error'],delta)
            if (ci+1)%16==0:print('Evaluated',ci+1,'/64',flush=True)
        save(run.path/'per_instance.json',rows);save(run.path/'validation.json',checks)
        boot=np.random.default_rng(cfg['bootstrap_seed']).integers(0,64,(cfg['bootstrap_replicates'],64))
        summary={'experiment_id':'EXP-016','count':64,'training_count':16,'checks':checks,'methods':{},'qpu_seconds':0,'limitations':cfg['limitations']}
        for name in vs:
            for policy in ['same','trained']:
                selected=[r for r in rows if r['method']==name and r['angles_policy']==policy]
                baseline=[r for r in rows if r['method']=='full' and r['angles_policy']==policy]
                result={'mean_logical_rzz':float(np.mean([r['logical_rzz'] for r in selected])),'by_budget':{}}
                for b in cfg['budgets']:
                    a=np.array([r['by_budget'][str(b)]['hybrid_success'] for r in selected]);f=np.array([r['by_budget'][str(b)]['hybrid_success'] for r in baseline])
                    result['by_budget'][str(b)]={'hybrid_success':float(a.mean()),'gain_over_full':float((a-f).mean()),'pointwise_ci95_gain':np.quantile((a-f)[boot].mean(axis=1),[.025,.975]).tolist()}
                summary['methods'][name+'_'+policy]=result
        run.finish(summary)
    except Exception as e:run.finish(error=e);raise


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
