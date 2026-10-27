"""EXP-017: frozen residual-guided pruning, public-only selection, fresh groups."""
from pathlib import Path
from datetime import datetime
from itertools import product
from math import comb
from hashlib import sha256
import json
import re
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.optimize import differential_evolution
from threadpoolctl import threadpool_limits
from qiskit import transpile,qpy
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from lattice_qaoa.residual_pruning import public_phase,select_masks,edge_list,mask_from_edges
from lattice_qaoa.mechanisms import BatchP1Energy,CachedMetrics,masked,weight90
from lattice_qaoa.instances import Parameters,generate_instance,expand_module_matrix
from lattice_qaoa.reference import embedding,lll_reduce,babai,exact_cvp,squared_cost
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import P1Energy,state_probabilities,circuit
from lattice_qaoa.sparsify import diagonal_energy
from lattice_qaoa.classical import local_refine,lift_secrets
from lattice_qaoa.evaluation import matrix_group,distribution_metrics
from lattice_qaoa.device import target_from_snapshot,logical_to_physical_indices
from research_run import ResearchRun,save


def prepare(B,y,cfg):
    t=time.perf_counter();z=babai(B,y)
    h,J,scale,r=public_phase(B,y,z)
    masks={'full':J.copy(),'weight90':weight90(J),'product':np.zeros_like(J)}
    diagnostics={'residual':r.tolist(),'incumbent':z.tolist(),'normalization':scale,'methods':{}}
    selected,trace=select_masks(h,J,cfg['gate_budgets'],cfg['selection_angles'])
    gh,gj,_,_=public_phase(B,y,z,ignore_residual=True)
    geometry,gtrace=select_masks(gh,gj,cfg['gate_budgets'],cfg['selection_angles'])
    for k in cfg['gate_budgets']:
        masks[f'magnitude{k}']=masked(J,np.count_nonzero(J)-k)[0]
        masks[f'residual{k}']=selected[k]['J']
        masks[f'geometry{k}']=mask_from_edges(J,edge_list(geometry[k]['J']))
        for name,obj in [(f'residual{k}',selected[k]),(f'geometry{k}',geometry[k])]:
            diagnostics['methods'][name]={a:v for a,v in obj.items() if a not in ['J','greedy_J']}
            diagnostics['methods'][name]['edges']=edge_list(masks[name])
    diagnostics.update(total_prepare_seconds=time.perf_counter()-t,residual_trace=trace,geometry_trace=gtrace)
    return h,J,z,masks,diagnostics


def fit_global(hs,js,phases,cfg):
    spec=cfg['optimizer'];bounds=np.array([spec['gamma_bounds'],spec['beta_bounds']])
    objective=BatchP1Energy(hs,phases,js)
    init=np.random.default_rng(spec['seed']).uniform(bounds[:,0],bounds[:,1],(spec['population'],2));init[0]=cfg['selection_angles']
    t=time.perf_counter()
    result=differential_evolution(objective,bounds,init=init,seed=spec['seed'],maxiter=spec['generations'],tol=0,atol=0,polish=False,workers=1,updating='immediate')
    assert result.nfev==spec['evaluation_budget']
    return {'angles':result.x.tolist(),'training_mean_original_energy':float(result.fun),'objective_evaluations':result.nfev,
        'instance_expectation_evaluations':result.nfev*len(hs),'seconds':time.perf_counter()-t,'optimizer_success':bool(result.success),'message':str(result.message)}


def effort_fit(h,j,score_j,anchor,budget,cfg,seed):
    spec=cfg['optimizer'];bounds=np.array([spec['gamma_bounds'],spec['beta_bounds']]);pop=spec['population']
    objective=P1Energy(h,j,observable_J=score_j)
    init=np.random.default_rng(seed).uniform(bounds[:,0],bounds[:,1],(pop,2));init[0]=anchor;init[1]=cfg['selection_angles']
    start=time.perf_counter()
    fit=differential_evolution(objective,bounds,init=init,seed=seed,maxiter=budget//pop-1,tol=0,atol=0,polish=False,workers=1,updating='immediate')
    assert fit.nfev<=budget
    return {'angles':fit.x.tolist(),'energy':float(fit.fun),'expectation_calls':fit.nfev,'seconds':time.perf_counter()-start,'optimizer_success':bool(fit.success)}


def summary_rows(rows,cfg,seed):
    n=len(rows);boot=np.random.default_rng(seed).integers(0,n,(cfg['bootstrap_replicates'],n))
    result={'count':n,'babai_optimum_count':sum(r['babai_cost']==r['optimum'] for r in rows),'coverage_count':sum(r['box_covers_optimum'] for r in rows),
        'unique_planted_optimum_count':sum(r['unique_planted_optimum'] for r in rows),'methods':{},'classical':{},'paired':{}}
    def stat(a):
        a=np.array(a,dtype=float)
        return {'mean':float(a.mean()),'ci95':np.quantile(a[boot].mean(axis=1),[.025,.975]).tolist()}
    for name in rows[0]['methods']:
        rr=[r['methods'][name] for r in rows]
        data={'by_budget':{}}
        for b in cfg['budgets']:
            data['by_budget'][str(b)]={key:stat([r['by_budget'][str(b)][key] for r in rr])
                for key in ['hybrid_success','hybrid_expected_distance','hybrid_expected_cost','strict_improvement','hybrid_returned_secret_probability']}
            data['by_budget'][str(b)]['observed_hybrid_success_count']=sum(r['observed'][str(b)]['hybrid_cost']==x['optimum'] for r,x in zip(rr,rows))
        for key in ['logical_rzz','native_cz','duration_unitary_s','selection_seconds','selection_expectation_calls','expected_original_energy']:
            if key in rr[0]:data[key]=stat([r[key] for r in rr])
        result['methods'][name]=data
    for name in ['refine1','refine2']:
        result['classical'][name]={'optimum_count':sum(r['classical'][name]['cost']==r['optimum'] for r in rows),
            'secret_count':sum(r['classical'][name]['secret_recovered'] for r in rows),
            'mean_distance':float(np.mean([np.sqrt(r['classical'][name]['cost']) for r in rows])),
            'evaluations':stat([r['classical'][name]['evaluations'] for r in rows]),'seconds':stat([r['classical'][name]['seconds'] for r in rows])}
    pairs=[('residual24_trained','magnitude24_trained'),('residual24_same','magnitude24_same'),
        ('residual24_trained','geometry24_trained'),('residual24_trained','magnitude24_effort'),('residual24_trained','full_trained')]
    for a,b in pairs:
        result['paired'][a+'_minus_'+b]={str(s):{key:stat([r['methods'][a]['by_budget'][str(s)][key]-r['methods'][b]['by_budget'][str(s)][key] for r in rows])
            for key in ['hybrid_success','hybrid_expected_distance']} for s in cfg['budgets']}
    primary=result['paired']['residual24_trained_minus_magnitude24_trained']['1024']['hybrid_success']
    result['primary_improvement_supported']=primary['ci95'][0]>0
    result['selection']={f'{kind}{k}':{'fallback_count':sum(r['selection']['methods'][f'{kind}{k}']['used_magnitude_fallback'] for r in rows),
        'mean_calls':float(np.mean([r['selection']['methods'][f'{kind}{k}']['analytic_expectation_evaluations'] for r in rows])),
        'mean_seconds':float(np.mean([r['selection']['methods'][f'{kind}{k}']['selection_seconds'] for r in rows]))}
        for kind in ['residual','geometry'] for k in cfg['gate_budgets']}
    return result


def figures(run,summary,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='whitegrid',context='talk',font_scale=.8)
    fig,axes=plt.subplots(1,3,figsize=(16,5.6),layout='constrained')
    colors={'full':'#246785','magnitude':'#D97935','residual':'#8A3E72','geometry':'#6E9484','cbd_secret':'#4B48A4'}
    fresh=summary['cohorts']['fresh'];baseline=100*fresh['babai_optimum_count']/fresh['count']
    for family in ['magnitude','residual','geometry']:
        means=[];low=[];high=[]
        for k in [16,24,32]:
            v=fresh['methods'][f'{family}{k}_trained']['by_budget']['1024']['hybrid_success']
            means.append(v['mean']*100);low.append((v['mean']-v['ci95'][0])*100);high.append((v['ci95'][1]-v['mean'])*100)
        axes[0].errorbar([16,24,32],means,yerr=[low,high],color=colors[family],marker='o',lw=2,capsize=4,label=family)
    axes[0].axhline(baseline,color='#444444',ls=':',label='Babai');axes[0].set(xlabel='Logical RZZ budget',ylabel='Mean optimum success (%)',title='128 fresh groups · 1,024 shots')
    axes[0].legend(fontsize=9)
    names=['full_trained','magnitude24_trained','residual24_trained','geometry24_trained']
    for name in names:
        r=fresh['methods'][name];family=name.split('24')[0].split('_')[0]
        axes[1].scatter(r['native_cz']['mean'],100*r['by_budget']['1024']['hybrid_success']['mean'],s=95,color=colors[family])
        axes[1].annotate(family,(r['native_cz']['mean'],100*r['by_budget']['1024']['hybrid_success']['mean']),xytext=(5,7),textcoords='offset points',fontsize=9)
    axes[1].set(xlabel='Mean routed native CZ gates',ylabel='Mean optimum success (%)',title='Equal logical caps ≠ equal routing cost')
    for name in ['full_trained','magnitude24_trained','residual24_trained','cbd_secret','uniform_box']:
        v=fresh['methods'][name]['by_budget'];family=name.split('24')[0].split('_')[0]
        color=colors.get(family,'#7F8990') if name!='cbd_secret' else colors['cbd_secret']
        axes[2].plot(cfg_b:=[256,1024,4096,16384],[100*v[str(b)]['hybrid_success']['mean'] for b in cfg_b],marker='o',label=name.replace('_trained',''),color=color)
    axes[2].axhline(100*fresh['classical']['refine2']['optimum_count']/128,color='#333333',ls='--',label='classical 2-coordinate')
    axes[2].set(xscale='log',xlabel='Samples per instance',ylabel='Mean optimum success (%)',title='Strong classical controls included')
    axes[2].legend(fontsize=8,loc='lower right')
    for ax in axes:sns.despine(ax=ax)
    fig.savefig(run.path/'comparison.svg');fig.savefig(run.path/'comparison.png',dpi=220);plt.close(fig)


def main():
    cfg=json.loads((ROOT/'experiments/configs/EXP-017.json').read_text())
    paths=[ROOT/cfg['training_run']/'training.json']+[ROOT/cfg[k]/'instances.json' for k in ['development_run','previous_validation_run','previous_parameter_run']]+[ROOT/cfg['calibration']]
    run=ResearchRun('EXP-017',Path(__file__),paths)
    try:
        training=json.loads(paths[0].read_text())['instances'];old=json.loads(paths[1].read_text())
        excluded={matrix_group(expand_module_matrix(np.array(x['A_polynomials'])),17) for x in training}
        for path in paths[1:4]:
            for x in json.loads(path.read_text()):
                q=int(re.search(r'_q(\d+)_',x['id']).group(1)) if '_q' in x['id'] else 17
                A=x.get('A_expanded',expand_module_matrix(np.array(x['A_polynomials'])))
                excluded.add(matrix_group(A,q))
        prepared=[];train_records=[]
        for i,x in enumerate(training):
            B=np.array(x['reduced_basis']);y=np.r_[x['public_target'],np.zeros(4,dtype=int)]
            h,j,z,m,d=prepare(B,y,cfg)
            np.testing.assert_allclose(h,x['h'],atol=1e-13);np.testing.assert_allclose(j,x['J'],atol=1e-13)
            prepared.append((h,j,m));train_records.append({'seed':x['seed'],'selection':d,'masks':{name:edge_list(jj) for name,jj in m.items()}})
            if (i+1)%4==0:print('Training masks',i+1,'/16',flush=True)
        save(run.path/'training_masks.json',train_records)
        fits={name:fit_global([x[0] for x in prepared],[x[1] for x in prepared],[x[2][name] for x in prepared],cfg) for name in prepared[0][2]}
        save(run.path/'training.json',fits)
        save(run.path/'frozen_protocol.json',{'frozen_at':datetime.now().astimezone().isoformat(),'config_sha256':sha256((run.path/'config.json').read_bytes()).hexdigest(),
            'training_sha256':sha256((run.path/'training.json').read_bytes()).hexdigest(),'fresh_outcomes_not_yet_generated':True,'primary_comparison':cfg['primary_comparison']})
        p=Parameters(**cfg['parameters']);fresh=[generate_instance(cfg['fresh_first_seed']+i,p) for i in range(cfg['fresh_count'])]
        groups=[matrix_group(expand_module_matrix(x.A),p.q) for x in fresh]
        assert len(set(groups))==len(groups) and not set(groups)&excluded
        save(run.path/'cohort.json',{'fresh_groups':groups,'fresh_seeds':[x.seed for x in fresh],'excluded_groups':sorted(excluded),'fresh_overlap_count':0})
        snap=json.loads(paths[4].read_text());target=target_from_snapshot(snap);sim=AerSimulator(method='statevector',max_parallel_threads=1)
        support=np.array(list(product(range(-p.eta,p.eta+1),repeat=4)),dtype=np.int64)
        masses=np.array([np.prod([comb(2*p.eta,p.eta+int(v))/2**(2*p.eta) for v in s]) for s in support]);assert abs(masses.sum()-1)<1e-14
        checks={'statevectors':0,'independent_qiskit_checks':0,'routed_ideal_checks':0,'max_probability_error':0.,'max_analytic_error':0.,'order_statistic_checks':0,'cost_codebooks_checked':0,'fresh_overlap_count':0}
        rows=[];instances=[];(run.path/'selection').mkdir();(run.path/'circuits').mkdir();(run.path/'samples').mkdir()
        for cohort in ['development','fresh']:
            cases=old if cohort=='development' else fresh
            for ci,x in enumerate(cases):
                prep_clock=time.perf_counter()
                if cohort=='development':
                    ident=x['id'];A=np.array(x['A_expanded']);t=np.array(x['public_target']);B=np.array(x['embedding_basis']);R=np.array(x['reduced_basis']);U=np.array(x['basis_transform']);y=np.r_[t,np.zeros(4,dtype=int)]
                    secret=np.array(x['planted_secret']);error=np.array(x['planted_error']);seed=x['seed'];Ap=x['A_polynomials']
                else:
                    ident=f'NEW-{ci:04d}';A=expand_module_matrix(x.A);t=x.target.ravel();secret=x.secret.ravel();error=x.error.ravel();seed=x.seed;Ap=x.A.tolist()
                    B,y=embedding(A,t,p.q);R,U=lll_reduce(B)
                np.testing.assert_array_equal(R,B@U)
                classical_preprocess_seconds=time.perf_counter()-prep_clock
                # Public selection finishes BEFORE reference/secret evaluation below.
                h,j,z,masks,selection=prepare(R,y,cfg)
                effort=effort_fit(h,masks['magnitude24'],j,fits['magnitude24']['angles'],selection['methods']['residual24']['analytic_expectation_evaluations'],cfg,20261020173+ci+(1000 if cohort=='fresh' else 0))
                save(run.path/'selection'/f'{ident}.json',selection)
                box=encode_babai_box(R,y);np.testing.assert_allclose(box.h,h,atol=1e-13);np.testing.assert_allclose(box.J,j,atol=1e-13);np.testing.assert_array_equal(box.center,z)
                checks['cost_codebooks_checked']+=len(box.costs)
                inc=squared_cost(B,y,U@z);ref=exact_cvp(A,t,p.q,U@z,cfg['exact_reference_candidate_cap']);opt=ref['cost']
                if cohort=='development':assert ref['cost']==x['reference']['cost']
                np.testing.assert_array_equal((A@secret+error)%p.q,t)
                coords=box.decode(np.arange(len(box.costs)))@U.T
                secret_mask=np.all(coords[:,4:]==secret,axis=1);babai_secret=bool(np.array_equal((U@z)[4:],secret))
                metric=CachedMetrics(box.costs,inc,opt,cfg['budgets'],secret_mask,babai_secret)
                unique=len(ref['minimizers'])==1 and np.array_equal(np.array(ref['minimizers'][0])[4:],secret)
                row={'id':ident,'cohort':cohort,'seed':seed,'group':matrix_group(A,p.q),'babai_cost':inc,'optimum':opt,'babai_secret':babai_secret,
                    'unique_planted_optimum':bool(unique),'planted_is_optimum':bool(int(secret@secret+error@error)==opt),'box_covers_optimum':bool(np.any(box.costs==opt)),
                    'classical_preprocess_seconds':classical_preprocess_seconds,'selection':{'methods':selection['methods'],'total_prepare_seconds':selection['total_prepare_seconds']},'methods':{},'classical':{}}
                for changes in [1,2]:
                    timer=time.perf_counter();c=local_refine(R,y,z,changes,cfg['classical_refinement_evaluation_cap']);c['seconds']=time.perf_counter()-timer
                    c['secret_recovered']=bool(np.array_equal((U@np.array(c['coefficients']))[4:],secret));row['classical'][f'refine{changes}']=c
                instances.append({'id':ident,'cohort':cohort,'seed':seed,'A_polynomials':Ap,'public_target':t.tolist(),'planted_secret':secret.tolist(),'planted_error':error.tolist(),
                    'reduced_basis':R.tolist(),'basis_transform':U.tolist(),'reference':ref})
                policies=[(name+'_'+policy,jj,cfg['selection_angles'] if policy=='same' else fits[name]['angles']) for name,jj in masks.items() for policy in ['same','trained']]
                policies.append(('magnitude24_effort',masks['magnitude24'],effort['angles']))
                stored_samples={};routed_saved=[]
                for mi,(name,jj,angles) in enumerate(policies):
                    prob=state_probabilities(diagonal_energy(h,jj),angles);checks['statevectors']+=1
                    energy=float(prob@box.energy);delta=abs(energy-P1Energy(h,jj,observable_J=j)(angles));assert delta<1e-10
                    checks['max_analytic_error']=max(checks['max_analytic_error'],delta)
                    data={'logical_rzz':int(np.count_nonzero(jj)),'edges':edge_list(jj),'angles':angles,'expected_original_energy':energy,**metric(prob)}
                    base=name.rsplit('_',1)[0]
                    if base.startswith(('residual','geometry')):
                        detail=selection['methods'][base];data['selection_seconds']=detail['selection_seconds'];data['selection_expectation_calls']=detail['analytic_expectation_evaluations']
                    if name=='magnitude24_effort':data.update(selection_seconds=effort['seconds'],selection_expectation_calls=effort['expectation_calls'],effort_fit=effort)
                    if ci<2:
                        independent=Statevector.from_instruction(circuit(h,jj,angles)).probabilities()
                        delta=float(np.max(abs(prob-independent)));assert delta<1e-10
                        checks['independent_qiskit_checks']+=1;checks['max_probability_error']=max(checks['max_probability_error'],delta)
                        old_stats=distribution_metrics(box.costs,prob,inc,opt,[1024])[2]['1024']
                        for key in ['hybrid_success','hybrid_expected_distance','hybrid_expected_cost']:
                            np.testing.assert_allclose(data['by_budget']['1024'][key],old_stats[key],atol=1e-10)
                            checks['order_statistic_checks']+=1
                    if name in ['full_trained','magnitude24_trained','residual24_trained','geometry24_trained']:
                        qc=circuit(h,jj,angles);qc.barrier()
                        routed=transpile(qc,target=target,initial_layout=list(range(16)),optimization_level=3,seed_transpiler=cfg['routing_seed'],scheduling_method='alap')
                        data.update(native_cz=int(routed.count_ops().get('cz',0)),duration_unitary_s=routed.estimate_duration(target),permutation=routed.layout.final_index_layout())
                        routed_saved.append(routed)
                        if ci<2:
                            probe=routed.copy();probe.save_probabilities();out=sim.run(probe,shots=1).result();assert out.success
                            actual=np.asarray(out.data(0)['probabilities'])[logical_to_physical_indices(data['permutation'])]
                            delta=float(np.max(abs(prob-actual)));assert delta<1e-10
                            checks['routed_ideal_checks']+=1;checks['max_probability_error']=max(checks['max_probability_error'],delta)
                    rng=np.random.default_rng(np.random.SeedSequence([cfg['measurement_seed'],0 if cohort=='development' else 1,ci,mi]))
                    sample=rng.choice(len(prob),max(cfg['budgets']),p=prob/prob.sum()).astype(np.uint16);stored_samples[name]=sample
                    data['observed']={str(b):{'hybrid_cost':int(min(inc,box.costs[sample[:b]].min()))} for b in cfg['budgets']}
                    row['methods'][name]=data
                candidates,costs=lift_secrets(A,t,p.q,support)
                for mi,name in enumerate(['uniform_box','cbd_secret']):
                    c=box.costs if mi==0 else costs;prob=np.full(len(c),1/len(c)) if mi==0 else masses
                    sm=secret_mask if mi==0 else np.all(support==secret,axis=1)
                    data=CachedMetrics(c,inc,opt,cfg['budgets'],sm,babai_secret)(prob)
                    rng=np.random.default_rng(np.random.SeedSequence([cfg['measurement_seed'],0 if cohort=='development' else 1,ci,100+mi]))
                    sample=rng.choice(len(prob),max(cfg['budgets']),p=prob).astype(np.uint16);stored_samples[name]=sample
                    data['observed']={str(b):{'hybrid_cost':int(min(inc,c[sample[:b]].min()))} for b in cfg['budgets']};row['methods'][name]=data
                np.savez_compressed(run.path/'samples'/f'{ident}.npz',**stored_samples)
                with (run.path/'circuits'/f'{ident}.qpy').open('wb') as f:qpy.dump(routed_saved,f)
                rows.append(row)
                with (run.path/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'cohort':cohort,'completed':ci+1,'recorded_at':datetime.now().astimezone().isoformat()})+'\n')
                if (ci+1)%8==0:
                    save(run.path/'per_instance.json',rows);save(run.path/'instances.json',instances);save(run.path/'validation.json',checks)
                    print(cohort,ci+1,'/',len(cases),'elapsed',round(time.perf_counter()-run.clock),'s',flush=True)
        save(run.path/'per_instance.json',rows);save(run.path/'instances.json',instances);save(run.path/'validation.json',checks)
        summary={'experiment_id':'EXP-017','count':192,'training_count':16,'cohorts':{},'checks':checks,'qpu_seconds':0,'limitations':cfg['limitations']}
        for i,cohort in enumerate(['development','fresh']):summary['cohorts'][cohort]=summary_rows([r for r in rows if r['cohort']==cohort],cfg,cfg['bootstrap_seed']+i)
        figures(run,summary,rows);run.finish(summary)
    except Exception as e:run.finish(error=e);raise


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
