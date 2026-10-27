"""EXP023: fixed-mask quality preservation and controlled selection timing."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from lattice_qaoa.instances import Parameters,generate_instance,expand_module_matrix
from lattice_qaoa.reference import embedding,lll_reduce,babai,exact_cvp,squared_cost
from lattice_qaoa.residual_pruning import public_phase,select_masks,edge_list
from lattice_qaoa.cached_pruning import select_masks_cached
from lattice_qaoa.incumbent_pruning import incumbent_mask
from lattice_qaoa.mechanisms import masked,CachedMetrics
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.sparsify import diagonal_energy
from lattice_qaoa.qaoa import state_probabilities
from lattice_qaoa.evaluation import matrix_group
from research_run import ResearchRun,save


def summarize(rows,cfg):
    n=len(rows);rng=np.random.default_rng(cfg['bootstrap_seed'])
    idx=rng.integers(0,n,(cfg['bootstrap_replicates'],n))
    def stat(a,indices=idx):
        a=np.array(a,dtype=float)
        return {'mean':float(a.mean()),'ci95':np.quantile(a[indices].mean(axis=1),[.025,.975]).tolist()}
    timing=rows[:cfg['timing_count']]
    bt=rng.integers(0,len(timing),(cfg['bootstrap_replicates'],len(timing)))
    med={m:np.array([np.median(r['timing_seconds'][m]) for r in timing]) for m in ['original','cached','incumbent','magnitude']}
    ratio=med['original']/med['cached']
    ratio_boot=med['original'][bt].mean(axis=1)/med['cached'][bt].mean(axis=1)
    out={'experiment_id':'EXP-023','count':n,'timing_count':len(timing),
         'identical_mask_count':sum(r['identical_mask'] for r in rows),
         'identical_path_count':sum(r['identical_path'] for r in rows),
         'max_probability_difference':max(r['max_probability_difference'] for r in rows),
         'quality_preservation_passed':all(r['identical_mask'] and r['identical_path'] and r['max_probability_difference']==0 for r in rows),
         'babai_optimum_count':sum(r['babai_cost']==r['optimum'] for r in rows),
         'box_coverage_count':sum(r['box_covers_optimum'] for r in rows),
         'hybrid_success':{str(b):stat([r['quality']['by_budget'][str(b)]['hybrid_success'] for r in rows]) for b in cfg['budgets']},
         'timing':{m:stat(a,bt) for m,a in med.items()},
         'speedup_ratio_of_mean_medians':{'ratio':float(med['original'].mean()/med['cached'].mean()),
             'ci95':np.quantile(ratio_boot,[.025,.975]).tolist()},
         'median_per_instance_speedup':float(np.median(ratio)),
         'selection_calls':{m:stat([r['calls'][m] for r in rows]) for m in ['original','cached']},
         'tie_recomputations':sum(r['tie_recomputations'] for r in rows),
         'limitations':cfg['limitations'],'qpu_seconds':0}
    return out


def main():
    cfg=json.loads((ROOT/'experiments/configs/EXP-023.json').read_text())
    prior=ROOT/cfg['comparison_run'];dev=ROOT/cfg['development_run']
    inputs=[prior/'cohort.json',prior/'frozen_models.json',prior/'manifest.json',
            dev/'summary.json',dev/'manifest.json',ROOT/cfg['training_run']/'training.json']
    run=ResearchRun('EXP-023',Path(__file__),inputs)
    rows=[];instances=[]
    try:
        assert json.loads(inputs[2].read_text())['status']=='completed'
        ds=json.loads(inputs[3].read_text())
        assert ds['identical_mask_count']==ds['identical_path_count']==64
        assert json.loads(inputs[4].read_text())['status']=='completed'
        angles=json.loads(inputs[1].read_text())['angles']['residual']
        old=json.loads(inputs[0].read_text());excluded=set(old['excluded_groups'])|set(old['fresh_groups'])
        save(run.path/'frozen_protocol.json',{'frozen_at':datetime.now().astimezone().isoformat(),
             'angles':angles,'config_sha256':sha256((run.path/'config.json').read_bytes()).hexdigest(),
             'development_masks_and_paths_all_match':True,'fresh_outcomes_not_yet_generated':True,
             'threadpools':threadpool_info()})
        k=cfg['logical_budget'];a=cfg['selection_angles']
        # One training-instance warmup, excluded from measurements.
        warm=json.loads(inputs[5].read_text())['instances'][0]
        wh,wj,_,_=public_phase(np.array(warm['reduced_basis']),np.r_[warm['public_target'],np.zeros(4,dtype=int)])
        select_masks(wh,wj,[k],a);select_masks_cached(wh,wj,[k],a)
        incumbent_mask(wj,np.tile([1,-1],8),k);masked(wj,np.count_nonzero(wj)-k)
        p=Parameters(**cfg['parameters'])
        fresh=[generate_instance(cfg['fresh_first_seed']+i,p) for i in range(cfg['fresh_count'])]
        groups=[matrix_group(expand_module_matrix(x.A),p.q) for x in fresh]
        assert len(set(groups))==len(groups) and not set(groups)&excluded
        save(run.path/'cohort.json',{'fresh_groups':groups,'fresh_seeds':[x.seed for x in fresh],
                                   'excluded_groups':sorted(excluded),'fresh_overlap_count':0})
        rng=np.random.default_rng(cfg['timing_order_seed'])
        (run.path/'selection').mkdir()
        for ci,x in enumerate(fresh):
            ident=f'CACHE-{ci:04d}';A=expand_module_matrix(x.A);t=x.target.ravel()
            B,y=embedding(A,t,p.q);R,U=lll_reduce(B);z=babai(R,y)
            h,j,_,_=public_phase(R,y,z)
            functions={'original':lambda:select_masks(h,j,[k],a),
                       'cached':lambda:select_masks_cached(h,j,[k],a),
                       'incumbent':lambda:incumbent_mask(j,np.tile([1,-1],len(h)//2),k),
                       'magnitude':lambda:masked(j,np.count_nonzero(j)-k)[0]}
            timings={name:[] for name in functions};orders=[];results={}
            reps=cfg['timing_repetitions'] if ci<cfg['timing_count'] else 1
            for rep in range(reps):
                order=rng.permutation(list(functions)).tolist();orders.append(order)
                for name in order:
                    inner=cfg['cheap_timing_inner_repetitions'] if name in ['incumbent','magnitude'] else 1
                    start=time.perf_counter()
                    for _ in range(inner):value=functions[name]()
                    duration=(time.perf_counter()-start)/inner
                    timings[name].append(duration)
                    if name in results and name in ['original','cached']:
                        np.testing.assert_array_equal(value[0][k]['J'],results[name][0][k]['J'])
                    results[name]=value
            old_result,old_trace=results['original'];new_result,new_trace=results['cached']
            np.testing.assert_array_equal(old_result[k]['J'],new_result[k]['J'])
            assert [v['removed_edge'] for v in old_trace]==[v['removed_edge'] for v in new_trace]
            np.testing.assert_allclose([v['original_energy'] for v in old_trace],[v['original_energy'] for v in new_trace],atol=1e-12)
            save(run.path/'selection'/f'{ident}.json',{'original_edges':edge_list(old_result[k]['J']),
                 'cached_edges':edge_list(new_result[k]['J']),'original_trace':old_trace,'cached_trace':new_trace,
                 'original_fallback':old_result[k]['used_magnitude_fallback'],
                 'cached_fallback':new_result[k]['used_magnitude_fallback']})
            # Evaluation-only codebooks and optima come after public-only selection.
            box=encode_babai_box(R,y);inc=squared_cost(R,y,z)
            ref=exact_cvp(A,t,p.q,U@z,cfg['exact_reference_candidate_cap'])
            original_p=state_probabilities(diagonal_energy(h,old_result[k]['J']),angles)
            cached_p=state_probabilities(diagonal_energy(h,new_result[k]['J']),angles)
            np.testing.assert_array_equal(original_p,cached_p)
            metric=CachedMetrics(box.costs,inc,ref['cost'],cfg['budgets'])
            old_quality=metric(original_p);new_quality=metric(cached_p)
            assert old_quality==new_quality
            row={'id':ident,'seed':x.seed,'group':groups[ci],'identical_mask':True,'identical_path':True,
                 'max_probability_difference':float(np.max(abs(original_p-cached_p))),
                 'timing_seconds':timings,'timing_orders':orders,'timing_repetitions':reps,
                 'calls':{name:results[name][0][k]['analytic_expectation_evaluations'] for name in ['original','cached']},
                 'tie_recomputations':new_result[k]['tie_recomputations'],'babai_cost':inc,'optimum':ref['cost'],
                 'box_covers_optimum':bool(np.any(box.costs==ref['cost'])),'quality':new_quality}
            rows.append(row)
            instances.append({'id':ident,'seed':x.seed,'A_polynomials':x.A.tolist(),'public_target':t.tolist(),
                 'reduced_basis':R.tolist(),'basis_transform':U.tolist(),'planted_secret':x.secret.ravel().tolist(),
                 'planted_error':x.error.ravel().tolist(),'reference':ref})
            if (ci+1)%8==0:
                save(run.path/'per_instance.json',rows);save(run.path/'instances.json',instances)
                print('Fresh cache',ci+1,'/128','elapsed',round(time.perf_counter()-run.clock),'s',flush=True)
        summary=summarize(rows,cfg)
        save(run.path/'per_instance.json',rows);save(run.path/'instances.json',instances)
        run.finish(summary);print(json.dumps(summary,indent=2))
    except Exception as exc:
        save(run.path/'per_instance.json',rows);save(run.path/'instances.json',instances)
        run.finish(error=exc);raise


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
