"""Paired IBM-noise estimates with separate Monte Carlo and group uncertainty."""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
from threadpoolctl import threadpool_limits
from lattice_qaoa.device import hybrid_metrics_from_low_masses
from analyze_hybrid_refinement import check_manifest


def stat(point,mc,pop):
    return {'estimate':float(point),'fixed_cohort_mc_ci95':np.quantile(mc,[.025,.975]).tolist(),
            'group_and_mc_ci95':np.quantile(pop,[.025,.975]).tolist()}


def main():
    run=Path(sys.argv[1]).resolve()
    assert not (run/'summary.json').exists(), 'Never overwrite completed analysis'
    audited=check_manifest(run)
    cfg=json.loads((run/'config.json').read_text());cohort=json.loads((run/'cohort.json').read_text())
    rows=json.loads((run/'routed_instances.json').read_text())
    prior=json.loads((run/'input'/cfg['input_run']/'summary.json').read_text())
    snap=json.loads((run/'input'/cfg['calibration']/'snapshot.json').read_text())
    N=len(rows);T=cfg['trajectories_per_circuit'];B=cfg['bootstrap_replicates']
    names=cfg['methods'];budgets=cfg['budgets'];fails=set(cohort['babai_failure_indices']);random_ids=set(cohort['random_subset_indices'])
    base=np.sqrt([r['babai_cost'] for r in rows]);good=np.array([r['babai_cost']==r['optimum_cost'] for r in rows])
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    group_counts=rng.multinomial(N,np.full(N,1/N),size=B)
    pop_base=group_counts@base/N;pop_success=group_counts@good.astype(float)/N
    mc={};pop={};points={};jackknife={};random_points={};random_mc={}
    for name in names:
        mc[name]={};pop[name]={};points[name]={};jackknife[name]={};random_points[name]={};random_mc[name]={}
        for shots in budgets:
            k=str(shots)
            points[name][k]={'distance':float(base.mean()),'success':float(good.mean()),'improved':0.}
            jackknife[name][k]={'distance':float(base.mean()),'success':float(good.mean())}
            mc[name][k]={'distance':np.full(B,base.mean()),'success':np.full(B,good.mean()),'improved':np.zeros(B)}
            pop[name][k]={'distance':pop_base.copy(),'success':pop_success.copy(),'improved':np.zeros(B)}
            random_points[name][k]=0.;random_mc[name][k]=np.zeros(B)
    reports=[]
    for i in cohort['noise_union_indices']:
        row=rows[i];report={'id':row['id'],'source_index':i,'methods':{}}
        for name in names:
            file=run/'noise'/f'{row["id"]}_{name}.npz'
            with np.load(file) as data:low=data['low_cost_masses'].copy()
            assert low.shape==(T,row['babai_cost']+1)
            weights=rng.multinomial(T,np.full(T,1/T),size=B)
            resampled=weights@low/T
            mean=low.mean(axis=0);loo=(low.sum(axis=0)-low)/(T-1)
            p=low[:,row['optimum_cost']]
            report['methods'][name]={'single_shot_success_estimate':float(p.mean()),'single_shot_mc_standard_error':float(p.std(ddof=1)/np.sqrt(T)),'by_budget':{}}
            for shots in budgets:
                k=str(shots)
                value=hybrid_metrics_from_low_masses(mean,row['babai_cost'],row['optimum_cost'],shots)
                boot=hybrid_metrics_from_low_masses(resampled,row['babai_cost'],row['optimum_cost'],shots)
                leave=hybrid_metrics_from_low_masses(loo,row['babai_cost'],row['optimum_cost'],shots)
                report['methods'][name]['by_budget'][k]={a:float(b) for a,b in value.items()}
                if i in fails:
                    for label,key in [('distance','hybrid_expected_distance'),('success','hybrid_success'),('improved','strict_improvement')]:
                        offset=base[i] if label=='distance' else 0.
                        points[name][k][label]+=(float(value[key])-offset)/N
                        mc[name][k][label]+=(boot[key]-offset)/N
                        pop[name][k][label]+=group_counts[:,i]*(boot[key]-offset)/N
                        if label!='improved':
                            correction=T*float(value[key])-(T-1)*float(np.mean(leave[key]))
                            jackknife[name][k][label]+=(correction-offset)/N
                if i in random_ids:
                    random_points[name][k]+=float(value['standalone_success'])/len(random_ids)
                    random_mc[name][k]+=boot['standalone_success']/len(random_ids)
        reports.append(report)
    methods={};comparisons={}
    for name in names:
        method={'resources':{},'by_budget':{},'random_subset_qaoa_only':{}}
        for key in ['native_cz','depth','two_qubit_depth','duration_unitary_s','duration_with_readout_s','idle_delay_count']:
            values=np.array([r['methods'][name][key] for r in rows])
            method['resources'][key]={'mean':float(values.mean()),'min':float(values.min()),'median':float(np.median(values)),'max':float(values.max())}
        for shots in budgets:
            k=str(shots);p=points[name][k]
            m={'hybrid_distance':stat(p['distance'],mc[name][k]['distance'],pop[name][k]['distance']),
               'hybrid_success':stat(p['success'],mc[name][k]['success'],pop[name][k]['success']),
               'strict_improved_fraction':stat(p['improved'],mc[name][k]['improved'],pop[name][k]['improved']),
               'extra_distance_improvement':stat(base.mean()-p['distance'],base.mean()-mc[name][k]['distance'],pop_base-pop[name][k]['distance']),
               'expected_recovered_count':N*p['success']-good.sum(),
               'jackknife_bias_check':{'distance_correction':jackknife[name][k]['distance']-p['distance'],
                                       'success_correction':jackknife[name][k]['success']-p['success']},
               'ideal_hybrid_success':prior['methods'][name]['by_budget'][k]['hybrid_success']['mean'],
               'ideal_hybrid_distance':prior['methods'][name]['by_budget'][k]['hybrid_expected_distance']['mean']}
            method['by_budget'][k]=m
            method['random_subset_qaoa_only'][k]={'count':len(random_ids),'estimated_success':random_points[name][k],
                'fixed_subset_mc_ci95':np.quantile(random_mc[name][k],[.025,.975]).tolist(),
                'note':'Not an all512 standalone estimate; this subset is outcome-independent.'}
        methods[name]=method
        if name!='full_fixed':
            comparison={}
            for shots in budgets:
                k=str(shots)
                pdelta=points[name][k]['success']-points['full_fixed'][k]['success']
                ddelta=points[name][k]['distance']-points['full_fixed'][k]['distance']
                numerator=base.mean()-points[name][k]['distance'];denominator=base.mean()-points['full_fixed'][k]['distance']
                ratio_mc=(base.mean()-mc[name][k]['distance'])/(base.mean()-mc['full_fixed'][k]['distance'])
                ratio_pop=(pop_base-pop[name][k]['distance'])/(pop_base-pop['full_fixed'][k]['distance'])
                comparison[k]={'paired_success_difference':stat(pdelta,mc[name][k]['success']-mc['full_fixed'][k]['success'],pop[name][k]['success']-pop['full_fixed'][k]['success']),
                    'paired_distance_difference':stat(ddelta,mc[name][k]['distance']-mc['full_fixed'][k]['distance'],pop[name][k]['distance']-pop['full_fixed'][k]['distance']),
                    'extra_distance_benefit_ratio':stat(numerator/denominator,ratio_mc,ratio_pop)}
            comparisons[name]=comparison
    reset=max(g['duration'] for g in snap['gates'] if g['name']=='reset')
    repeat=snap['default_rep_delay'];cases=cfg['planning']['candidate_cases'];planning=[]
    summed_time=sum(methods[n]['resources']['duration_with_readout_s']['mean']+reset+repeat for n in names)
    for shots in cfg['planning']['candidate_shots']:
        active=shots*cases*summed_time
        planning.append({'cases':cases,'methods':len(names),'shots_per_circuit':shots,'total_executions':cases*len(names)*shots,
            'single_job_2s_overhead_estimate_seconds':2+active,
            'one_job_per_circuit_2s_overhead_estimate_seconds':2*cases*len(names)+active,
            'note':'Two overhead scenarios, not bounds or a guarantee. Actual quota and calibration must be refreshed before execution.'})
    checks=json.loads((run/'validation.json').read_text());checks['manifest_hashes_before_analysis']=audited
    summary={'experiment_id':'EXP-008','run_id':run.name,'count':N,'backend':snap['backend'],
        'calibration_last_update':snap['calibration_last_update'],'calibration_fetched_at':snap['fetched_at'],
        'physical_qubits':snap['physical_qubits'],'qubits':16,'depth_p':1,'primary_budget':cfg['primary_budget'],
        'babai':prior['babai'],'classical_refinement':prior['classical_refinement'],
        'noise_union_count':len(cohort['noise_union_indices']),'babai_failure_count':len(fails),'random_subset_count':len(random_ids),
        'trajectories_per_circuit':T,'methods':methods,'comparisons_to_full':comparisons,
        'planning':{'reset_duration_s':reset,'default_rep_delay_s':repeat,'reserve_seconds':cfg['planning']['reserve_seconds'],'estimates':planning},
        'validation':checks,'qpu_jobs_submitted':0,'qpu_seconds':0,'limitations':cfg['limitations'],
        'timing':json.loads((run/'timing.json').read_text())}
    for filename,data in [('noisy_per_instance.json',reports),('summary.json',summary)]:
        (run/filename).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    manifest=json.loads((run/'manifest.json').read_text())
    for filename in ['noisy_per_instance.json','summary.json']:
        manifest['output_sha256'][filename]=sha256((run/filename).read_bytes()).hexdigest()
    manifest.update(status='completed',analysis_finished_at=datetime.now().astimezone().isoformat())
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    for name,m in methods.items():
        print(name,'CZ',m['resources']['native_cz']['mean'],'duration us',1e6*m['resources']['duration_unitary_s']['mean'],
              'primary',m['by_budget']['1024'],flush=True)
    print('COMPARISONS',json.dumps(comparisons),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
