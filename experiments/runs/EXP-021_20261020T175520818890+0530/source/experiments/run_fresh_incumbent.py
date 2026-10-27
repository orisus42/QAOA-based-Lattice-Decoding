"""EXP021: a fully frozen head-to-head on new public matrix groups."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
from itertools import product
from math import comb
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from qiskit import transpile,qpy
from qiskit.quantum_info import Statevector
from lattice_qaoa.instances import Parameters,generate_instance,expand_module_matrix
from lattice_qaoa.reference import embedding,lll_reduce,babai,exact_cvp,squared_cost
from lattice_qaoa.residual_pruning import public_phase,select_masks,edge_list
from lattice_qaoa.incumbent_pruning import incumbent_mask
from lattice_qaoa.mechanisms import CachedMetrics,masked
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import P1Energy,state_probabilities,circuit
from lattice_qaoa.sparsify import diagonal_energy
from lattice_qaoa.classical import local_refine,lift_secrets
from lattice_qaoa.evaluation import matrix_group,distribution_metrics
from lattice_qaoa.device import target_from_snapshot
from research_run import ResearchRun,save


def summarize(rows,cfg):
    n=len(rows)
    idx=np.random.default_rng(cfg['bootstrap_seed']).integers(0,n,(cfg['bootstrap_replicates'],n))
    def stat(a):
        a=np.asarray(a,dtype=float)
        return {'mean':float(a.mean()),'ci95':np.quantile(a[idx].mean(axis=1),[.025,.975]).tolist()}
    out={'experiment_id':cfg['experiment_id'],'count':n,'methods':{},'paired':{},
         'babai_optimum_count':sum(r['babai_cost']==r['optimum'] for r in rows),
         'box_coverage_count':sum(r['box_covers_optimum'] for r in rows),
         'unique_planted_optimum_count':sum(r['unique_planted_optimum'] for r in rows),
         'classical_refine2_optimum_count':sum(r['classical']['cost']==r['optimum'] for r in rows),
         'classical_refine2_seconds':stat([r['classical']['seconds'] for r in rows]),
         'qpu_seconds':0,'limitations':cfg['limitations']}
    keys=['hybrid_success','hybrid_expected_distance','hybrid_returned_secret_probability']
    for name in rows[0]['methods']:
        out['methods'][name]={'by_budget':{str(b):{key:stat([r['methods'][name]['by_budget'][str(b)][key] for r in rows]) for key in keys} for b in cfg['budgets']}}
        for key in ['native_cz','duration_unitary_s','selection_seconds','selection_expectation_calls']:
            if key in rows[0]['methods'][name]:
                out['methods'][name][key]=stat([r['methods'][name][key] for r in rows])
        out['methods'][name]['observed_success_count']={str(b):sum(r['methods'][name]['observed'][str(b)]['hybrid_cost']==r['optimum'] for r in rows) for b in cfg['budgets']}
    for a,b in [('residual','incumbent'),('residual_same','incumbent_same'),('residual','magnitude'),('residual','full')]:
        out['paired'][a+'_minus_'+b]={str(sh):{key:stat([r['methods'][a]['by_budget'][str(sh)][key]-r['methods'][b]['by_budget'][str(sh)][key] for r in rows]) for key in keys} for sh in cfg['budgets']}
    out['primary_improvement_supported']=out['paired']['residual_minus_incumbent']['1024']['hybrid_success']['ci95'][0]>0
    return out


def main():
    cfg=json.loads((ROOT/'experiments/configs/EXP-021.json').read_text())
    prior=ROOT/cfg['previous_run'];control=ROOT/cfg['incumbent_run']
    inputs=[prior/'cohort.json',prior/'training.json',control/'training.json',ROOT/cfg['calibration']]
    run=ResearchRun('EXP-021',Path(__file__),inputs)
    rows=[];instances=[]
    try:
        old=json.loads(inputs[0].read_text())
        excluded=set(old['excluded_groups'])|set(old['fresh_groups'])
        fits=json.loads(inputs[1].read_text());cheap=json.loads(inputs[2].read_text())
        angles={'full':fits['full']['angles'],'magnitude':fits['magnitude24']['angles'],
                'residual':fits['residual24']['angles'],'incumbent':cheap['angles'],
                'residual_same':cfg['selection_angles'],'incumbent_same':cfg['selection_angles']}
        save(run.path/'frozen_models.json',{'frozen_at':datetime.now().astimezone().isoformat(),
             'angles':angles,'input_hashes':run.manifest['input_sha256'],
             'config_sha256':sha256((run.path/'config.json').read_bytes()).hexdigest(),
             'fresh_outcomes_not_yet_generated':True})
        p=Parameters(**cfg['parameters'])
        fresh=[generate_instance(cfg['fresh_first_seed']+i,p) for i in range(cfg['fresh_count'])]
        groups=[matrix_group(expand_module_matrix(x.A),p.q) for x in fresh]
        assert len(set(groups))==len(groups) and not set(groups)&excluded
        save(run.path/'cohort.json',{'fresh_groups':groups,'fresh_seeds':[x.seed for x in fresh],
                                   'excluded_groups':sorted(excluded),'fresh_overlap_count':0})
        support=np.array(list(product(range(-p.eta,p.eta+1),repeat=4)))
        masses=np.array([np.prod([comb(2*p.eta,p.eta+int(v))/2**(2*p.eta) for v in s]) for s in support])
        target=target_from_snapshot(json.loads(inputs[3].read_text()))
        checks={'statevectors':0,'independent_qiskit_checks':0,'analytic_checks':0,
                'cost_codebooks':0,'order_statistic_checks':0,'group_overlap':0}
        for folder in ['selection','circuits','samples']:(run.path/folder).mkdir()
        for ci,x in enumerate(fresh):
            ident=f'FRESH-{ci:04d}'
            A=expand_module_matrix(x.A);t=x.target.ravel();secret=x.secret.ravel()
            B,y=embedding(A,t,p.q);R,U=lll_reduce(B)
            np.testing.assert_array_equal(R,B@U)
            z=babai(R,y);h,j,_,_=public_phase(R,y,z);k=cfg['logical_budget']
            start=time.perf_counter()
            chosen,trace=select_masks(h,j,[k],cfg['selection_angles'])
            residual_seconds=time.perf_counter()-start
            start=time.perf_counter();cheap_j=incumbent_mask(j,np.tile([1,-1],len(h)//2),k)
            cheap_seconds=time.perf_counter()-start
            start=time.perf_counter();mag=masked(j,np.count_nonzero(j)-k)[0]
            mag_seconds=time.perf_counter()-start
            masks={'full':j,'magnitude':mag,'residual':chosen[k]['J'],'incumbent':cheap_j,
                   'residual_same':chosen[k]['J'],'incumbent_same':cheap_j}
            selection={'edges':{name:edge_list(jj) for name,jj in masks.items()},
                       'residual_trace':trace,'residual_fallback':chosen[k]['used_magnitude_fallback'],
                       'seconds':{'residual':residual_seconds,'incumbent':cheap_seconds,'magnitude':mag_seconds},
                       'residual_expectation_calls':chosen[k]['analytic_expectation_evaluations']}
            save(run.path/'selection'/f'{ident}.json',selection)
            # Everything above uses public data only; labels and codebooks start here.
            box=encode_babai_box(R,y)
            np.testing.assert_allclose(box.h,h,atol=1e-13);np.testing.assert_allclose(box.J,j,atol=1e-13)
            inc=squared_cost(B,y,U@z);ref=exact_cvp(A,t,p.q,U@z,cfg['exact_reference_candidate_cap']);opt=ref['cost']
            coords=box.decode(np.arange(len(box.costs)))@U.T
            np.testing.assert_array_equal(np.sum((coords@B.T-y)**2,axis=1),box.costs);checks['cost_codebooks']+=1
            secret_mask=np.all(coords[:,4:]==secret,axis=1);babai_secret=bool(np.array_equal((U@z)[4:],secret))
            metric=CachedMetrics(box.costs,inc,opt,cfg['budgets'],secret_mask,babai_secret)
            start=time.perf_counter()
            classical=local_refine(R,y,z,2,cfg['classical_refinement_evaluation_cap'])
            classical['seconds']=time.perf_counter()-start
            row={'id':ident,'seed':x.seed,'group':groups[ci],'babai_cost':inc,'optimum':opt,
                 'box_covers_optimum':bool(np.any(box.costs==opt)),
                 'unique_planted_optimum':bool(len(ref['minimizers'])==1 and np.array_equal(np.array(ref['minimizers'][0])[4:],secret)),
                 'classical':classical,'methods':{}}
            instances.append({'id':ident,'seed':x.seed,'A_polynomials':x.A.tolist(),'public_target':t.tolist(),
                 'reduced_basis':R.tolist(),'basis_transform':U.tolist(),'planted_secret':secret.tolist(),
                 'planted_error':x.error.ravel().tolist(),'reference':ref})
            saved_circuits=[];samples={}
            for mi,(name,jj) in enumerate(masks.items()):
                prob=state_probabilities(diagonal_energy(h,jj),angles[name]);checks['statevectors']+=1
                np.testing.assert_allclose(prob@box.energy,P1Energy(h,jj,observable_J=j)(angles[name]),atol=1e-12)
                checks['analytic_checks']+=1
                if ci<2:
                    direct=Statevector.from_instruction(circuit(h,jj,angles[name])).probabilities()
                    np.testing.assert_allclose(prob,direct,atol=1e-12);checks['independent_qiskit_checks']+=1
                    independent=distribution_metrics(box.costs,prob,inc,opt,[1024])[2]['1024']
                    for key in ['hybrid_success','hybrid_expected_distance']:
                        np.testing.assert_allclose(metric(prob)['by_budget']['1024'][key],independent[key],atol=1e-10)
                    checks['order_statistic_checks']+=1
                data={'angles':angles[name],'logical_rzz':int(np.count_nonzero(jj)),**metric(prob)}
                if name in ['full','magnitude','residual','incumbent']:
                    qc=circuit(h,jj,angles[name]);qc.barrier()
                    routed=transpile(qc,target=target,initial_layout=list(range(16)),optimization_level=3,
                                     seed_transpiler=cfg['routing_seed'],scheduling_method='alap')
                    data.update(native_cz=int(routed.count_ops().get('cz',0)),duration_unitary_s=routed.estimate_duration(target))
                    saved_circuits.append(routed)
                if name in selection['seconds']:data['selection_seconds']=selection['seconds'][name]
                if name=='residual':data['selection_expectation_calls']=selection['residual_expectation_calls']
                rng=np.random.default_rng(np.random.SeedSequence([cfg['measurement_seed'],ci,mi]))
                sample=rng.choice(len(prob),max(cfg['budgets']),p=prob/prob.sum()).astype(np.uint16);samples[name]=sample
                data['observed']={str(b):{'hybrid_cost':int(min(inc,box.costs[sample[:b]].min()))} for b in cfg['budgets']}
                row['methods'][name]=data
            _,costs=lift_secrets(A,t,p.q,support)
            for mi,name in enumerate(['uniform_box','cbd_secret']):
                c=box.costs if mi==0 else costs
                prob=np.full(len(c),1/len(c)) if mi==0 else masses
                sm=secret_mask if mi==0 else np.all(support==secret,axis=1)
                data=CachedMetrics(c,inc,opt,cfg['budgets'],sm,babai_secret)(prob)
                rng=np.random.default_rng(np.random.SeedSequence([cfg['measurement_seed'],ci,100+mi]))
                sample=rng.choice(len(c),max(cfg['budgets']),p=prob).astype(np.uint16);samples[name]=sample
                data['observed']={str(b):{'hybrid_cost':int(min(inc,c[sample[:b]].min()))} for b in cfg['budgets']}
                row['methods'][name]=data
            with (run.path/'circuits'/f'{ident}.qpy').open('wb') as f:qpy.dump(saved_circuits,f)
            np.savez_compressed(run.path/'samples'/f'{ident}.npz',**samples)
            rows.append(row)
            if (ci+1)%8==0:
                save(run.path/'per_instance.json',rows);save(run.path/'instances.json',instances);save(run.path/'validation.json',checks)
                print('Fresh',ci+1,'/',len(fresh),'elapsed',round(time.perf_counter()-run.clock),'s',flush=True)
        result=summarize(rows,cfg);result['checks']=checks
        save(run.path/'per_instance.json',rows);save(run.path/'instances.json',instances);save(run.path/'validation.json',checks)
        run.finish(result)
        print(json.dumps({'primary':result['paired']['residual_minus_incumbent']['1024'],'methods':{n:v['by_budget']['1024']['hybrid_success']['mean'] for n,v in result['methods'].items()}},indent=2))
    except Exception as exc:
        save(run.path/'per_instance.json',rows);save(run.path/'instances.json',instances)
        run.finish(error=exc);raise


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
