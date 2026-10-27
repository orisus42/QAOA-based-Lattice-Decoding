"""EXP026: frozen local-only near-native-budget IBM noise comparison."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
from itertools import product
from math import comb
import json
import time
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from qiskit import qpy
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from threadpoolctl import threadpool_limits
from lattice_qaoa.native_matching import compile_families,match_counts,near_match_counts,bind_angles
from lattice_qaoa.native_noise_stats import analyze_noise
from lattice_qaoa.instances import Parameters,generate_instance,expand_module_matrix
from lattice_qaoa.reference import embedding,lll_reduce,babai,exact_cvp,squared_cost
from lattice_qaoa.residual_pruning import public_phase,edge_list
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import state_probabilities
from lattice_qaoa.sparsify import diagonal_energy
from lattice_qaoa.classical import local_refine,lift_secrets
from lattice_qaoa.mechanisms import CachedMetrics
from lattice_qaoa.evaluation import matrix_group
from lattice_qaoa.device import target_from_snapshot,gate_noise,idle_noise_pass,logical_to_physical_indices,readout_channel,hybrid_metrics_from_low_masses
from run_residual_pruning import fit_global
from research_run import ResearchRun,save


def prepare(h,j,target,cfg):
    masks,templates,details=compile_families(h,j,target,cfg['logical_grid'],cfg['selection_angles'],cfg['routing_seeds'])
    matching=near_match_counts(details['resources'],cfg['native_caps'],cfg['native_gap_tolerance'])
    selected={};circuits={};resources={}
    for cap,pair in matching.items():
        for family in ['cached','incumbent']:
            name=f'{family}_cap{cap}';k=pair['logical_k'][family];seed=pair['routing_seed'][family]
            selected[name]=masks[family][k];circuits[name]=templates[family][k,seed]
            resources[name]=next(r for r in details['resources'][family] if r['logical_k']==k and r['routing_seed']==seed)
    for family in ['cached','incumbent']:
        name=family+'24';seed=cfg['routing_seeds'][0]
        selected[name]=masks[family][24];circuits[name]=templates[family][24,seed]
        resources[name]=next(r for r in details['resources'][family] if r['logical_k']==24 and r['routing_seed']==seed)
    details.update(matching=matching,exact_matching_diagnostic=match_counts(details['resources'],cfg['native_caps']))
    return selected,circuits,resources,details


def main():
    cfg=json.loads((ROOT/'experiments/configs/EXP-026.json').read_text())
    inputs=[ROOT/cfg['training_run']/'training.json',ROOT/cfg['previous_fresh_run']/'cohort.json',
        ROOT/cfg['fixed_angles_run']/'frozen_models.json',ROOT/cfg['calibration'],
        ROOT/cfg['profile_run']/'summary.json',ROOT/'experiments/run_residual_pruning.py']
    run=ResearchRun('EXP-026',Path(__file__),inputs)
    rows=[];instances=[]
    try:
        snap=json.loads(inputs[3].read_text());target=target_from_snapshot(snap)
        original=json.loads(inputs[2].read_text())['angles']
        training=json.loads(inputs[0].read_text())['instances'];train=[];train_records=[]
        for ti,x in enumerate(training):
            R=np.array(x['reduced_basis']);y=np.r_[x['public_target'],np.zeros(4,dtype=int)]
            h,j,_,_=public_phase(R,y)
            masks,templates,resources,details=prepare(h,j,target,cfg)
            train.append((h,j,masks))
            train_records.append({'seed':x['seed'],'details':details,'selected':resources})
            if (ti+1)%4==0:print('Training resource profiles',ti+1,'/16',flush=True)
        fits={name:fit_global([v[0] for v in train],[v[1] for v in train],[v[2][name] for v in train],cfg)
              for name in cfg['methods'] if '_cap' in name}
        angles={name:fit['angles'] for name,fit in fits.items()}
        angles.update(cached24=original['residual'],incumbent24=original['incumbent'])
        save(run.path/'training_profiles.json',train_records);save(run.path/'training.json',fits)
        save(run.path/'frozen_protocol.json',{'frozen_at':datetime.now().astimezone().isoformat(),'angles':angles,
            'config_sha256':sha256((run.path/'config.json').read_bytes()).hexdigest(),
            'training_sha256':sha256((run.path/'training.json').read_bytes()).hexdigest(),
            'fresh_outcomes_not_yet_generated':True})
        previous=json.loads(inputs[1].read_text());excluded=set(previous['excluded_groups'])|set(previous['fresh_groups'])
        p=Parameters(**cfg['parameters'])
        fresh=[generate_instance(cfg['fresh_first_seed']+i,p) for i in range(cfg['fresh_count'])]
        groups=[matrix_group(expand_module_matrix(x.A),p.q) for x in fresh]
        assert len(set(groups))==len(groups) and not set(groups)&excluded
        sentinels=sorted(np.random.default_rng(cfg['sentinel_seed']).choice(len(fresh),cfg['sentinel_count'],replace=False).tolist())
        cohort={'fresh_groups':groups,'fresh_seeds':[x.seed for x in fresh],'excluded_groups':sorted(excluded),
                'fresh_overlap_count':0,'sentinel_indices':sentinels}
        save(run.path/'cohort.json',cohort)
        for folder in ['selection','circuits','noise']:(run.path/folder).mkdir()
        ideal=AerSimulator(method='statevector',max_parallel_threads=1)
        checks={'routed_ideal_circuits':0,'max_routed_probability_error':0.,'native_edges_checked':0,
                'near_match_pairs':0,'max_native_cz_gap':0,'noisy_circuits':0,'trajectories':0,
                'max_trajectory_norm_deviation':0.,'group_overlap':0}
        support=np.array(list(product(range(-p.eta,p.eta+1),repeat=4)))
        masses=np.array([np.prod([comb(2*p.eta,p.eta+int(v))/2**(2*p.eta) for v in s]) for s in support])
        for ci,x in enumerate(fresh):
            ident=f'NATIVE-{ci:04d}';A=expand_module_matrix(x.A);t=x.target.ravel()
            start=time.perf_counter();B,y=embedding(A,t,p.q);R,U=lll_reduce(B);z=babai(R,y);h,j,_,_=public_phase(R,y,z)
            preprocessing_seconds=time.perf_counter()-start
            start=time.perf_counter();masks,templates,resources,details=prepare(h,j,target,cfg)
            preparation_seconds=time.perf_counter()-start
            save(run.path/'selection'/f'{ident}.json',details)
            # Reference/secret information is used only after the public selection.
            box=encode_babai_box(R,y);inc=squared_cost(R,y,z)
            ref=exact_cvp(A,t,p.q,U@z,cfg['exact_reference_candidate_cap']);opt=ref['cost']
            np.testing.assert_allclose(box.h,h,atol=1e-13);np.testing.assert_allclose(box.J,j,atol=1e-13)
            start=time.perf_counter();classical=local_refine(R,y,z,2,cfg['classical_refinement_evaluation_cap'])
            classical['seconds']=time.perf_counter()-start
            _,prior_costs=lift_secrets(A,t,p.q,support)
            prior=CachedMetrics(prior_costs,inc,opt,cfg['budgets'])(masses)
            row={'id':ident,'index':ci,'seed':x.seed,'group':groups[ci],'babai_cost':inc,'optimum':opt,
                'box_covers_optimum':bool(np.any(box.costs==opt)),'classical':classical,'classical_prior':prior,
                'preprocessing_seconds':preprocessing_seconds,'all_masks_and_compilation_seconds':preparation_seconds,
                'matching':details['matching'],'methods':{}}
            instances.append({'id':ident,'seed':x.seed,'A_polynomials':x.A.tolist(),'public_target':t.tolist(),
                'reduced_basis':R.tolist(),'basis_transform':U.tolist(),'planted_secret':x.secret.ravel().tolist(),
                'planted_error':x.error.ravel().tolist(),'reference':ref})
            saved=[];ideal_mass={}
            low=lambda pp:np.stack([pp[...,box.costs==cost].sum(axis=-1) for cost in range(inc+1)],axis=-1)
            for name in cfg['methods']:
                routed=bind_angles(templates[name],angles[name]);saved.append(routed)
                native_cz=int(routed.count_ops().get('cz',0))
                assert native_cz==resources[name]['native_cz']
                for ins in routed.data:
                    if ins.operation.num_qubits==2:
                        pair=tuple(routed.find_bit(q).index for q in ins.qubits)
                        assert ins.operation.name=='cz' and pair in target['cz'];checks['native_edges_checked']+=1
                order=logical_to_physical_indices(routed.layout.final_index_layout())
                probe=routed.copy();probe.save_probabilities()
                result=ideal.run(probe,shots=1).result();assert result.success
                physical=np.asarray(result.data(0)['probabilities']);prob=physical[order]
                direct=state_probabilities(diagonal_energy(h,masks[name]),angles[name])
                delta=float(np.max(abs(prob-direct)));assert delta<1e-10
                checks['routed_ideal_circuits']+=1;checks['max_routed_probability_error']=max(checks['max_routed_probability_error'],delta)
                raw_ideal=low(prob);raw_read=low(readout_channel(physical,snap)[order])
                ideal_mass[name+'_ideal']=raw_ideal[None,:];ideal_mass[name+'_readout_only']=raw_read[None,:]
                row['methods'][name]={'logical_k':resources[name]['logical_k'],'native_cz':native_cz,
                    'duration_unitary_s':routed.estimate_duration(target),'depth':routed.depth(),
                    'angles':angles[name],'edges':edge_list(masks[name]),'permutation':routed.layout.final_index_layout(),
                    'ideal':{str(b):{k:float(v) for k,v in hybrid_metrics_from_low_masses(raw_ideal,inc,opt,b).items()} for b in cfg['budgets']}}
            for cap in cfg['native_caps']:
                left,right=row['methods'][f'cached_cap{cap}']['native_cz'],row['methods'][f'incumbent_cap{cap}']['native_cz']
                assert max(left,right)<=cap and abs(left-right)<=cfg['native_gap_tolerance']
                checks['near_match_pairs']+=1;checks['max_native_cz_gap']=max(checks['max_native_cz_gap'],abs(left-right))
            with (run.path/'circuits'/f'{ident}.qpy').open('wb') as f:qpy.dump(saved,f)
            np.savez_compressed(run.path/'noise'/f'{ident}_ideal.npz',**ideal_mass)
            rows.append(row)
            if (ci+1)%8==0:
                save(run.path/'per_instance.json',rows);save(run.path/'instances.json',instances);save(run.path/'validation.json',checks)
                print('Routed/verified',ci+1,'/128 elapsed',round(time.perf_counter()-run.clock),'s',flush=True)
        save(run.path/'per_instance.json',rows);save(run.path/'instances.json',instances)
        failures=[i for i,r in enumerate(rows) if r['babai_cost']>r['optimum']]
        noisy_ids=sorted(set(failures)|set(sentinels))
        cohort.update(babai_failure_indices=failures,noise_indices=noisy_ids,noise_selection_rule=cfg['noise_cohort'])
        save(run.path/'cohort.json',cohort)
        noise_model=gate_noise(target);save(run.path/'gate_noise_model.json',noise_model.to_dict(serializable=True))
        sim=AerSimulator(noise_model=noise_model,**cfg['simulator']);idle=PassManager(idle_noise_pass(snap))
        run.manifest.update(phase='noise_simulation',routing_finished_at=datetime.now().astimezone().isoformat());save(run.path/'manifest.json',run.manifest)
        for ni,index in enumerate(noisy_ids):
            r=rows[index];x=instances[index]
            box=encode_babai_box(np.array(x['reduced_basis']),np.r_[x['public_target'],np.zeros(4,dtype=int)])
            low=lambda pp:np.stack([pp[...,box.costs==cost].sum(axis=-1) for cost in range(r['babai_cost']+1)],axis=-1)
            with (run.path/'circuits'/f'{r["id"]}.qpy').open('rb') as f:circuits=qpy.load(f)
            with np.load(run.path/'noise'/f'{r["id"]}_ideal.npz') as f:ideal_raw={key:f[key].copy() for key in f.files}
            for mi,name in enumerate(cfg['methods']):
                started=time.perf_counter();q=idle.run(circuits[mi]);q.save_probabilities(pershot=True)
                seed=cfg['noise_seed']+100*index+mi
                result=sim.run(q,shots=cfg['trajectories_per_circuit'],seed_simulator=seed).result();assert result.success
                prob=np.asarray(result.data(0)['probabilities'])
                assert prob.shape==(cfg['trajectories_per_circuit'],65536)
                delta=float(np.max(abs(prob.sum(axis=1)-1)))
                np.testing.assert_allclose(prob.sum(axis=1),1,atol=cfg['normalization_tolerance'],rtol=0)
                order=logical_to_physical_indices(r['methods'][name]['permutation'])
                before=prob[:,order];read=readout_channel(prob,snap)[:,order]
                np.testing.assert_allclose(read.sum(axis=1),1,atol=cfg['normalization_tolerance'],rtol=0)
                raw={'ideal':ideal_raw[name+'_ideal'],'readout_only':ideal_raw[name+'_readout_only'],
                     'gate_idle':low(before),'gate_idle_readout':low(read)}
                assert all(np.all(a[:,:r['optimum']]<1e-12) for a in raw.values())
                np.savez_compressed(run.path/'noise'/f'{r["id"]}_{name}.npz',**raw,
                    mean_before_readout=before.mean(axis=0),mean_after_readout=read.mean(axis=0),seed=np.array(seed))
                checks['noisy_circuits']+=1;checks['trajectories']+=len(prob)
                checks['max_trajectory_norm_deviation']=max(checks['max_trajectory_norm_deviation'],delta)
                with (run.path/'noise_progress.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'recorded_at':datetime.now().astimezone().isoformat(),'id':r['id'],'method':name,
                        'seed':seed,'seconds':time.perf_counter()-started,'trajectories':len(prob),
                        'max_norm_deviation':delta})+'\n')
                save(run.path/'validation.json',checks)
            print('Noise complete',ni+1,'/',len(noisy_ids),'elapsed',round(time.perf_counter()-run.clock),'s',flush=True)
        def load_noise(ident,name):
            with np.load(run.path/'noise'/f'{ident}_{name}.npz') as f:
                return {key:f[key].copy() for key in ['ideal','readout_only','gate_idle','gate_idle_readout']}
        summary=analyze_noise(rows,load_noise,cfg)
        summary.update(experiment_id='EXP-026',count=len(rows),training_count=16,noise_count=len(noisy_ids),
            babai_optimum_count=len(rows)-len(failures),box_coverage_count=sum(r['box_covers_optimum'] for r in rows),
            classical_refine2_optimum_count=sum(r['classical']['cost']==r['optimum'] for r in rows),
            classical_prior_success1024=float(np.mean([r['classical_prior']['by_budget']['1024']['hybrid_success'] for r in rows])),
            backend=snap['backend'],calibration_fetched_at=snap['fetched_at'],checks=checks,
            physical_qubits=snap['physical_qubits'],trajectories_per_circuit=cfg['trajectories_per_circuit'],
            qpu_seconds=0,qpu_jobs_submitted=0,limitations=cfg['limitations'])
        summary['resource_matching']={str(cap):{'mean_native_gap_cached_minus_incumbent':float(np.mean([
            r['methods'][f'cached_cap{cap}']['native_cz']-r['methods'][f'incumbent_cap{cap}']['native_cz'] for r in rows])),
            'exact_equal_count':sum(r['methods'][f'cached_cap{cap}']['native_cz']==r['methods'][f'incumbent_cap{cap}']['native_cz'] for r in rows),
            'zero_gate_pair_count':sum(max(r['methods'][f'cached_cap{cap}']['native_cz'],r['methods'][f'incumbent_cap{cap}']['native_cz'])==0 for r in rows)}
            for cap in cfg['native_caps']}
        summary['classical_overhead']={'mean_preprocess_seconds':float(np.mean([r['preprocessing_seconds'] for r in rows])),
            'mean_both_families_masks_and_compile_seconds':float(np.mean([r['all_masks_and_compilation_seconds'] for r in rows])),
            'compiled_candidates_per_instance':2*len(cfg['logical_grid'])*len(cfg['routing_seeds'])}
        save(run.path/'validation.json',checks);run.finish(summary)
        print(json.dumps(summary['paired']['cap160']['gate_idle_readout']['1024'],indent=2))
    except Exception as error:
        save(run.path/'per_instance.json',rows);save(run.path/'instances.json',instances)
        run.finish(error=error);raise


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
