"""EXP-012: same routed circuits, isolated calibrated noise components; local only."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from qiskit import transpile,qpy
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from threadpoolctl import threadpool_limits
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import circuit,state_probabilities
from lattice_qaoa.sparsify import prune_interactions,diagonal_energy
from lattice_qaoa.device import target_from_snapshot,gate_noise,idle_noise_pass,logical_to_physical_indices,readout_channel,hybrid_metrics_from_low_masses
from research_run import ResearchRun,save


def main():
    cfg=json.loads((ROOT/'experiments/configs/EXP-012.json').read_text())
    source=ROOT/cfg['development_run']/'instances.json';cal=ROOT/cfg['calibration']
    run=ResearchRun('EXP-012',Path(__file__),[source]+list(cal.glob('*.json')))
    try:
        (run.path/'circuits').mkdir();(run.path/'noise').mkdir()
        snap=json.loads((cal/'snapshot.json').read_text());target=target_from_snapshot(snap)
        cases=json.loads(source.read_text());fail=[i for i,x in enumerate(cases) if x['lll_babai_cost']>x['reference']['cost']]
        save(run.path/'cohort.json',{'source_count':len(cases),'nonoptimal_indices':fail,'selection':cfg['cohort']})
        gate=gate_noise(target);idle=PassManager(idle_noise_pass(snap))
        save(run.path/'gate_noise_model.json',gate.to_dict(serializable=True))
        options=dict(method='statevector',max_parallel_threads=4,max_parallel_shots=4,max_parallel_experiments=1,
            zero_threshold=cfg['zero_threshold'])
        sims={'ideal':AerSimulator(method='statevector',max_parallel_threads=1),
            'gate_channels':AerSimulator(noise_model=gate,**options),
            'idle_only':AerSimulator(**options),'gate_and_idle':AerSimulator(noise_model=gate,**options)}
        checks={'ideal_circuits_verified':0,'max_routed_discrepancy':0.,'noisy_circuits':0,'trajectories':0}
        rows=[];noise_store={}
        for i in fail:
            x=cases[i];box=encode_babai_box(np.array(x['reduced_basis']),np.r_[x['public_target'],np.zeros(4,dtype=int)])
            inc=x['lll_babai_cost'];opt=x['reference']['cost'];saved=[]
            low=lambda p:np.stack([p[...,box.costs==c].sum(axis=-1) for c in range(inc+1)],axis=-1)
            for mi,(name,fraction) in enumerate(cfg['methods'].items()):
                J,_=prune_interactions(box.J,fraction);qc=circuit(box.h,J,cfg['angles']);qc.barrier()
                routed=transpile(qc,target=target,initial_layout=list(range(16)),optimization_level=cfg['optimization_level'],
                    seed_transpiler=cfg['routing_seed'],scheduling_method='alap')
                saved.append(routed);order=logical_to_physical_indices(routed.layout.final_index_layout())
                probe=routed.copy();probe.save_probabilities()
                result=sims['ideal'].run(probe,shots=1).result();assert result.success
                physical=np.asarray(result.data(0)['probabilities']);p=physical[order]
                delta=float(max(abs(p-state_probabilities(diagonal_energy(box.h,J),cfg['angles']))));assert delta<1e-10
                checks['ideal_circuits_verified']+=1;checks['max_routed_discrepancy']=max(checks['max_routed_discrepancy'],delta)
                raw={'ideal':low(p)[None,:],'readout_only':low(readout_channel(physical,snap)[order])[None,:]}
                for mode_index,mode in enumerate(['gate_channels','idle_only','gate_and_idle']):
                    q=idle.run(routed) if mode!='gate_channels' else routed.copy()
                    q.save_probabilities(pershot=True)
                    result=sims[mode].run(q,shots=cfg['trajectories_per_circuit'],seed_simulator=cfg['noise_seed']+100*i+10*mi+mode_index).result()
                    assert result.success
                    probability=np.asarray(result.data(0)['probabilities'])
                    assert probability.shape==(cfg['trajectories_per_circuit'],65536)
                    np.testing.assert_allclose(probability.sum(axis=1),1,atol=1e-10)
                    raw[mode]=low(probability[:,order]);checks['noisy_circuits']+=1;checks['trajectories']+=len(probability)
                    if mode=='gate_and_idle':raw['gate_idle_readout']=low(readout_channel(probability,snap)[:,order])
                np.savez_compressed(run.path/'noise'/f'{x["id"]}_{name}.npz',**raw)
                noise_store[i,name]=raw
                row={'index':i,'id':x['id'],'method':name,'babai_cost':inc,'optimum_cost':opt,
                    'native_cz':int(routed.count_ops().get('cz',0)),'duration_unitary_s':routed.estimate_duration(target),
                    'permutation':routed.layout.final_index_layout(),'channels':{}}
                for channel,masses in raw.items():
                    row['channels'][channel]={str(b):{k:float(v) for k,v in hybrid_metrics_from_low_masses(masses.mean(axis=0),inc,opt,b).items()} for b in cfg['budgets']}
                rows.append(row);save(run.path/'per_instance.json',rows)
                print('Noise complete',x['id'],name,'circuits',checks['noisy_circuits'],flush=True)
            with (run.path/'circuits'/f'{x["id"]}.qpy').open('wb') as f:qpy.dump(saved,f)
        save(run.path/'validation.json',checks)
        # Preserve pairing across noise channels: gate+idle and its readout reuse
        # the same sampled trajectories. Instance-bootstrap indices are shared.
        rng=np.random.default_rng(cfg['bootstrap_seed']);B=cfg['bootstrap_replicates'];groupboot=rng.integers(0,64,size=(B,64))
        bootvalues={};pointvalues={};summary={'experiment_id':'EXP-012','count':64,'simulated_babai_failures':len(fail),'checks':checks,'methods':{},'paired_gains':{},'qpu_seconds':0,
            'limits':['One saved IBM Fez calibration and one routing seed, approximate independent Markovian noise.','128 trajectories estimate probabilities; nonlinear finite-shot statistics are plug-in estimates, not hardware observations.','Only Babai failures simulated; remaining hybrid outputs are exactly optimal. No full-cohort standalone claim.','Pointwise intervals combine paired development-group resampling with trajectory uncertainty; exploratory, not final validation.','Channel effects need not add because noise changes the state and interacts with later channels.']}
        for mi,name in enumerate(cfg['methods']):
            summary['methods'][name]={}
            trajectory_boot={i:rng.integers(0,cfg['trajectories_per_circuit'],size=(B,cfg['trajectories_per_circuit'])) for i in fail}
            for channel in cfg['channels']:
                summary['methods'][name][channel]={}
                for b in cfg['budgets']:
                    points=np.ones(64);samples=np.ones((B,64));fixed=np.ones((B,64))
                    distances=np.sqrt([x['lll_babai_cost'] for x in cases]);dist_samples=np.tile(distances,(B,1))
                    jackknife=[]
                    for i in fail:
                        x=cases[i];raw=noise_store[i,name][channel];mean=raw.mean(axis=0)
                        means=raw[trajectory_boot[i]].mean(axis=1) if len(raw)>1 else np.repeat(mean[None,:],B,axis=0)
                        v=hybrid_metrics_from_low_masses(mean,x['lll_babai_cost'],x['reference']['cost'],b)
                        bv=hybrid_metrics_from_low_masses(means,x['lll_babai_cost'],x['reference']['cost'],b)
                        points[i]=v['hybrid_success'];samples[:,i]=bv['hybrid_success'];distances[i]=v['hybrid_expected_distance'];dist_samples[:,i]=bv['hybrid_expected_distance']
                        if len(raw)>1:
                            loo=(len(raw)*mean-raw)/(len(raw)-1)
                            lv=hybrid_metrics_from_low_masses(loo,x['lll_babai_cost'],x['reference']['cost'],b)['hybrid_success']
                            jackknife.append(float((len(raw)-1)*(lv.mean()-points[i])))
                    fixed=samples.mean(axis=1);resampled=samples[np.arange(B)[:,None],groupboot].mean(axis=1)
                    pointvalues[name,channel,b]=points;bootvalues[name,channel,b]=resampled
                    summary['methods'][name][channel][str(b)]={'hybrid_success':float(points.mean()),
                        'hybrid_mean_distance':float(distances.mean()),'group_plus_mc_ci95':np.quantile(resampled,[.025,.975]).tolist(),
                        'fixed_cohort_mc_ci95':np.quantile(fixed,[.025,.975]).tolist(),
                        'jackknife_estimated_success_bias':sum(jackknife)/64}
        for channel in cfg['channels']:
            summary['paired_gains'][channel]={}
            for b in cfg['budgets']:
                difference=pointvalues['weak75',channel,b].mean()-pointvalues['full',channel,b].mean()
                bootstrap=bootvalues['weak75',channel,b]-bootvalues['full',channel,b]
                summary['paired_gains'][channel][str(b)]={'gain_percentage_points':float(100*difference),'pointwise_ci95_pp':(100*np.quantile(bootstrap,[.025,.975])).tolist()}
        run.finish(summary)
    except Exception as error:run.finish(error=error);raise


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
