"""EXP-011: exhaustive development sweep and public-data-only ablations."""
from pathlib import Path
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import circuit,state_probabilities
from lattice_qaoa.sparsify import diagonal_energy,prune_interactions
from lattice_qaoa.mechanisms import BatchP1Energy,CachedMetrics,ordered_edges,masked,weight90
from lattice_qaoa.evaluation import matrix_group,distribution_metrics
from lattice_qaoa.instances import expand_module_matrix
from research_run import ResearchRun,save


def train(cfg,hs,js,fraction=None,split=False):
    phase=js if split else [prune_interactions(j,fraction)[0] for j in js]
    objective=BatchP1Energy(hs,phase,js)
    spec=cfg['optimizer'];bounds=[spec['gamma_bounds']]*(2 if split else 1)+[spec['beta_bounds']]
    rng=np.random.default_rng(spec['seed']);bounds=np.array(bounds)
    init=rng.uniform(bounds[:,0],bounds[:,1],size=(spec['population'],len(bounds)))
    a=cfg['angles'];init[0]=[a[0],a[0],a[1]] if split else a
    result=differential_evolution(objective,bounds,init=init,seed=spec['seed'],maxiter=spec['generations'],
        tol=0,atol=0,polish=False,workers=1,updating='immediate')
    assert result.nfev==spec['evaluation_budget']
    return {'angles':result.x.tolist(),'training_mean_original_energy':float(result.fun),'objective_evaluations':result.nfev,
        'optimizer_success':bool(result.success),'message':str(result.message),'split':split}


def main():
    cfg=json.loads((ROOT/'experiments/configs/EXP-011.json').read_text())
    source=ROOT/cfg['development_run']/'instances.json';training=ROOT/cfg['training_run']/'training.json'
    run=ResearchRun('EXP-011',Path(__file__),[source,training])
    try:
        cases=json.loads(source.read_text());tr=json.loads(training.read_text())
        old_groups={matrix_group(c['A_expanded'],17) for c in cases}
        train_groups={matrix_group(expand_module_matrix(np.array(c['A_polynomials'])),17) for c in tr['instances']}
        assert len(old_groups)==64 and len(train_groups)==16 and not old_groups&train_groups
        hs=[np.array(x['h']) for x in tr['instances']];js=[np.array(x['J']) for x in tr['instances']]
        fits={}
        for percent in cfg['trained_percentages']:
            fits[str(percent)]=train(cfg,hs,js,percent/100)
            save(run.path/'training.json',fits)
            print('Fit',percent,'energy',round(fits[str(percent)]['training_mean_original_energy'],6),flush=True)
        fits['split_full']=train(cfg,hs,js,split=True);save(run.path/'training.json',fits)
        # Freeze the entire fitted parameter table before loading evaluation outputs.
        (run.path/'training_frozen_at.txt').write_text(__import__('datetime').datetime.now().astimezone().isoformat())
        angles=cfg['angles'];budgets=cfg['budgets'];all_rows=[];grid=[];mechanisms=[];details=[]
        checks={'training_groups':16,'development_groups':64,'overlap':0,'statevectors':0,'independent_circuit_checks':0,'max_circuit_discrepancy':0.,'all_distinct_edge_masks':0}
        spins=(1-2*((np.arange(65536)[:,None]>>np.arange(16))&1)).astype(np.int8)
        for ci,x in enumerate(cases):
            box=encode_babai_box(np.array(x['reduced_basis']),np.r_[x['public_target'],np.zeros(4,dtype=int)])
            inc=x['lll_babai_cost'];opt=x['reference']['cost'];metric=CachedMetrics(box.costs,inc,opt,budgets)
            edges=ordered_edges(box.J);M=len(edges);phase=box.energy.copy();field=spins@box.h
            cost_bottom=set(np.argsort(box.costs,kind='stable')[:max(1,len(box.costs)//100)])
            results_by_count={};retained=box.J.copy();totalw=float(np.sum(box.J**2))
            for dropped in range(M+1):
                p=state_probabilities(phase,angles);stats=metric(p)
                record={'id':x['id'],'index':ci,'policy':'same_angles','removed_count':dropped,'nonzero_edges':M,
                    'removed_fraction':dropped/M,'logical_rzz':M-dropped,
                    'removed_pair_squared_weight':float(np.sum((box.J-retained)**2)/totalw),
                    'retained_within_coordinate':int(sum(abs(retained[i,i+1])>1e-12 for i in range(0,16,2))),
                    **stats}
                all_rows.append(record);results_by_count[dropped]=record;checks['statevectors']+=1;checks['all_distinct_edge_masks']+=1
                if dropped==int(.75*M):
                    difference=box.energy-phase
                    low=set(np.argsort(phase,kind='stable')[:len(cost_bottom)])
                    details.append({'id':x['id'],'removed_pair_squared_weight':record['removed_pair_squared_weight'],
                        'phase_rank_spearman':float(spearmanr(box.energy,phase).statistic),
                        'lowest_one_percent_overlap':len(low&cost_bottom)/len(cost_bottom),
                        'p1_same_angle_state_fidelity':float(abs(np.exp(-1j*angles[0]*difference).mean())**2),
                        'full_ground_state_mean_sparse_rank':float(np.mean(np.searchsorted(np.sort(phase),phase[box.costs==opt],side='left')/len(phase)))})
                if dropped<M:
                    i,j=edges[dropped];phase-=box.J[i,j]*spins[:,i]*spins[:,j];retained[i,j]=0.
            np.testing.assert_allclose(phase,field,atol=1e-12)
            with (run.path/'sweep_all_counts.jsonl').open('a',encoding='utf-8') as f:
                for r in all_rows[-(M+1):]:f.write(json.dumps(r)+'\n')
            for percent in cfg['trained_percentages']:
                r=results_by_count[int(percent*M/100)]
                grid.append({**r,'percent':percent})
                J,_=prune_interactions(box.J,percent/100)
                energy=diagonal_energy(box.h,J);trained=fits[str(percent)]['angles']
                p=state_probabilities(energy,trained)
                grid.append({'id':x['id'],'index':ci,'policy':'trained_angles','percent':percent,
                    'logical_rzz':int(np.count_nonzero(J)),**metric(p)})
                checks['statevectors']+=1
            control={'full':box.J,'weak75':masked(box.J,int(.75*M))[0],
                'strong75':masked(box.J,int(.75*M),'strong')[0],
                'protect_coordinate75':masked(box.J,int(.75*M),'protect_coordinate')[0],
                'drop_coordinate75':masked(box.J,int(.75*M),'drop_coordinate')[0],
                'weight90':weight90(box.J),'no_pairs100':np.zeros_like(box.J)}
            for seed in range(5):
                control[f'random75_{seed}']=masked(box.J,int(.75*M),'random',cfg['control_seed']+seed)[0]
                control[f'ties75_{seed}']=masked(box.J,int(.75*M),'ties',cfg['control_seed']+seed)[0]
            control['full_scaled']=box.J*np.sqrt(np.sum(control['weak75']**2)/totalw)
            control['split_full']=box.J
            for name,J in control.items():
                h=box.h; a=angles
                if name=='split_full':
                    gh,gj,beta=fits['split_full']['angles'];h=gh*box.h;J=gj*J;a=[1.,beta]
                p=state_probabilities(diagonal_energy(h,J),a)
                row={'id':x['id'],'index':ci,'policy':name,'logical_rzz':int(np.count_nonzero(J)),**metric(p)}
                mechanisms.append(row);checks['statevectors']+=1
                if ci==0:
                    independent=Statevector.from_instruction(circuit(h,J,a)).probabilities()
                    delta=float(np.max(abs(p-independent)));assert delta<1e-10
                    checks['independent_circuit_checks']+=1;checks['max_circuit_discrepancy']=max(checks['max_circuit_discrepancy'],delta)
                    old=distribution_metrics(box.costs,p,inc,opt,budgets)[2]
                    for b in budgets:
                        for key in ['hybrid_success','hybrid_expected_distance','hybrid_expected_cost']:
                            np.testing.assert_allclose(row['by_budget'][str(b)][key],old[str(b)][key],atol=1e-10)
            if (ci+1)%8==0:
                save(run.path/'grid.json',grid);save(run.path/'mechanisms.json',mechanisms);save(run.path/'diagnostics.json',details)
                print('Evaluated',ci+1,'/64; statevectors',checks['statevectors'],'elapsed',round(time.perf_counter()-run.clock),flush=True)
        save(run.path/'validation.json',checks)
        boot=np.random.default_rng(cfg['bootstrap_seed']).integers(0,len(cases),size=(cfg['bootstrap_replicates'],len(cases)))
        full=[r for r in mechanisms if r['policy']=='full'];full.sort(key=lambda r:r['index'])
        def aggregate(selected):
            selected=sorted(selected,key=lambda r:r['index']);assert len(selected)==64
            result={'mean_logical_rzz':float(np.mean([r['logical_rzz'] for r in selected])),'by_budget':{}}
            for b in budgets:
                by={}
                for key in ['hybrid_success','hybrid_expected_distance']:
                    v=np.array([r['by_budget'][str(b)][key] for r in selected]);f=np.array([r['by_budget'][str(b)][key] for r in full]);diff=v-f
                    by[key]={'mean':float(v.mean()),'difference_from_old_full':float(diff.mean()),
                        'paired_pointwise_ci95':np.quantile(diff[boot].mean(axis=1),[.025,.975]).tolist()}
                result['by_budget'][str(b)]=by
            return result
        summary={'experiment_id':'EXP-011','count':64,'training_count':16,'checks':checks,
            'babai_optimum_count':sum(x['lll_babai_cost']==x['reference']['cost'] for x in cases),
            'grid':{},'mechanisms':{},'qpu_seconds':0,'limits':cfg['limitations']}
        for policy in ['same_angles','trained_angles']:
            summary['grid'][policy]={str(p):aggregate([r for r in grid if r['policy']==policy and r['percent']==p]) for p in cfg['trained_percentages']}
        for name in control:summary['mechanisms'][name]=aggregate([r for r in mechanisms if r['policy']==name])
        # Average randomized replicates WITHIN each instance before cohort uncertainty.
        for prefix in ['random75','ties75']:
            averaged=[]
            for i in range(64):
                selected=[r for r in mechanisms if r['index']==i and r['policy'].startswith(prefix+'_')]
                averaged.append({'index':i,'logical_rzz':selected[0]['logical_rzz'],
                    'by_budget':{str(b):{key:float(np.mean([r['by_budget'][str(b)][key] for r in selected]))
                        for key in ['hybrid_success','hybrid_expected_distance']} for b in budgets}})
            summary['mechanisms'][prefix+'_replicate_mean']=aggregate(averaged)
        run.finish(summary)
    except Exception as error:run.finish(error=error);raise


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
