"""Frozen-policy replication and synthetic gate-noise pilot; zero cloud calls."""
import json
import time
import importlib.metadata
import joblib
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error
from feasibility import ROOT, OUT, SEED, make_instance, scores, probabilities, serial, summarize


def grouped_delta(a,b,instances,seed=SEED+71):
    keys=np.array([x['lattice_key'] for x in instances])
    groups=np.unique(keys)
    delta=np.asarray(a)-np.asarray(b)
    sums=np.array([delta[keys==g].sum() for g in groups])
    counts=np.array([(keys==g).sum() for g in groups])
    r=np.random.default_rng(seed).integers(len(groups),size=(4000,len(groups)))
    values=sums[r].sum(axis=1)/counts[r].sum(axis=1)
    return {'mean':float(delta.mean()),'ci95':np.quantile(values,[.025,.975]).tolist(),'bootstrap_unit':'lattice'}


def qaoa_circuit(x,angles):
    nq=x['nq']
    circuit=QuantumCircuit(nq)
    circuit.h(range(nq))
    for gamma,beta in np.array(angles).reshape(-1,2):
        for k,h in enumerate(x['h']):
            if abs(h)>1e-12: circuit.rz(2*gamma*h,k)
        for k in range(nq):
            for j in range(k+1,nq):
                if abs(x['J'][k,j])>1e-12: circuit.rzz(2*gamma*x['J'][k,j],k,j)
        circuit.rx(2*beta,range(nq))
    return circuit


def readout_channel(prob,nq,p01,p10):
    prob=prob.copy()
    for k in range(nq):
        v=prob.reshape(-1,2,2**k)
        a,b=v[:,0,:].copy(),v[:,1,:].copy()
        v[:,0,:]=(1-p01)*a+p10*b
        v[:,1,:]=p01*a+(1-p10)*b
    return prob


def noise_model(p1,p2,a1,a2):
    model=NoiseModel()
    one=depolarizing_error(p1,1).compose(amplitude_damping_error(a1))
    two=depolarizing_error(p2,2).compose(amplitude_damping_error(a2).tensor(amplitude_damping_error(a2)))
    model.add_all_qubit_quantum_error(one,['sx','x'])
    model.add_all_qubit_quantum_error(two,['cx'])
    return model


def main():
    started=time.time()
    original=json.loads((OUT/'instances.json').read_text())
    result=json.loads((OUT/'pilot_results.json').read_text())
    arrays=np.load(OUT/'pilot_arrays.npz')
    model=joblib.load(OUT/'selector.joblib')
    schedules=result['schedules']
    prior_keys={x['lattice_key'] for x in original}
    rng=np.random.default_rng(SEED+1000)
    fresh=[]
    attempts=0
    # Fresh, distinct lattices, with the original model and angles frozen.
    while len(fresh)<200:
        x=make_instance(rng,600+len(fresh))
        attempts+=1
        if x['lattice_key'] not in prior_keys:
            fresh.append(x)
            prior_keys.add(x['lattice_key'])
    P=np.column_stack([scores(fresh,a) for a in schedules])
    choices=model.predict(np.stack([x['features'] for x in fresh])).argmax(axis=1)
    fixedchoices=np.array([result['size_fixed_schedules'][str(x['nq'])] for x in fresh])
    chosen=P[np.arange(len(fresh)),choices]
    fixed=P[np.arange(len(fresh)),fixedchoices]
    summary={'scope':'LOCAL ONLY. Synthetic noise; no IBM calibration and no QPU use.',
        'replication':{'seed':SEED+1000,'generation_attempts':attempts,'audit':summarize(fresh),
            'learned':float(chosen.mean()),'fixed_by_size':float(fixed.mean()),
            'delta':grouped_delta(chosen,fixed,fresh),
            'caveat':'New lattices exclude all original lattice groups; only one fresh seed and tiny dimensions.'},
        'original_grouped_deltas':{},'noise':{},'validation':{},'runtime_seconds':None}
    byid={x['id']:x for x in original}
    test=[byid[i] for i in result['split_ids']['test']]
    pt=arrays['Ptest']; chosen0=pt[np.arange(len(test)),arrays['choices']]
    for name,col in [('fixed_by_size','sizechoices'),('cluster_rule','clusterchoices')]:
        summary['original_grouped_deltas'][name]=grouped_delta(chosen0,pt[np.arange(len(test)),arrays[col]],test)
    # The first 12 fresh instances of each size; selection never uses outcomes.
    ids=sum(([i for i,x in enumerate(fresh) if x['nq']==nq][:12] for nq in [6,8]),[])
    subset=[fresh[i] for i in ids]
    circuits=[]; maps=[]; meta=[]; max_error=0.; compiled_error=0.
    for x in subset:
        for k,a in enumerate(schedules):
            circuit=qaoa_circuit(x,a)
            expected=probabilities(x['energy_norm'],a)[0]
            max_error=max(max_error,float(abs(expected-Statevector.from_instruction(circuit).probabilities()).max()))
            coupling=sum(([[j,j+1],[j+1,j]] for j in range(x['nq']-1)),[])
            compiled=transpile(circuit,basis_gates=['rz','sx','x','cx'],coupling_map=coupling,
                initial_layout=list(range(x['nq'])),optimization_level=2,seed_transpiler=SEED)
            permutation=compiled.layout.final_index_layout()
            logical=np.arange(2**x['nq'])
            physical=sum((((logical>>j)&1)<<p for j,p in enumerate(permutation)))
            compiled_prob=Statevector.from_instruction(compiled).probabilities()[physical]
            compiled_error=max(compiled_error,float(abs(compiled_prob-expected).max()))
            meta.append({'id':x['id'],'schedule':k,'qubits':x['nq'],'cx':compiled.count_ops().get('cx',0),'depth':compiled.depth()})
            compiled.save_probabilities()
            circuits.append(compiled); maps.append(physical)
    assert max_error<1e-10 and compiled_error<1e-10
    summary['validation']={'direct_statevector_max_error':max_error,'routed_statevector_max_error':compiled_error,
        'circuits':len(circuits),'topology':'synthetic bidirectional line; rz,sx,x,cx basis',
        'cx_range':[min(x['cx'] for x in meta),max(x['cx'] for x in meta)],
        'depth_range':[min(x['depth'] for x in meta),max(x['depth'] for x in meta)]}
    print('Independent circuit validation:',json.dumps(summary['validation']),flush=True)
    print('Frozen replication:',json.dumps(summary['replication']),flush=True)
    profiles={'light':(.0001,.003,.0001,.001,.01,.02),'moderate':(.0005,.01,.0003,.002,.02,.04)}
    noisy_arrays={}
    for name,(p1,p2,a1,a2,r01,r10) in profiles.items():
        sim=AerSimulator(method='density_matrix',noise_model=noise_model(p1,p2,a1,a2),
            max_parallel_threads=2,max_parallel_experiments=1)
        job=sim.run(circuits,shots=1,seed_simulator=SEED)
        output=job.result()
        pn=np.zeros((len(subset),4))
        for j in range(len(circuits)):
            x=subset[j//4]
            physical_prob=np.asarray(output.data(j)['probabilities'])
            logical_prob=readout_channel(physical_prob,x['nq'],r01,r10)[maps[j]]
            pn[j//4,j%4]=logical_prob[x['good']].sum()
        nc=pn[np.arange(len(ids)),choices[ids]]
        nf=pn[np.arange(len(ids)),fixedchoices[ids]]
        summary['noise'][name]={'count':len(ids),'model':{'depolarizing_1q':p1,'depolarizing_2q':p2,'amplitude_damping_1q':a1,'amplitude_damping_2q':a2,'readout_0_to_1':r01,'readout_1_to_0':r10},
            'learned':float(nc.mean()),'fixed_by_size':float(nf.mean()),'delta':grouped_delta(nc,nf,subset),
            'ideal_learned_same_subset':float(chosen[ids].mean()),'ideal_fixed_same_subset':float(fixed[ids].mean()),
            'oracle':float(pn.max(axis=1).mean()),'rank_changed_count':int(np.sum(pn.argmax(axis=1)!=P[ids].argmax(axis=1))),
            'note':'Exact density-matrix probabilities with a synthetic gate/readout model; not sampled hardware counts or fitted IBM noise.'}
        noisy_arrays[name]=pn
        print('Synthetic noise',name,json.dumps(summary['noise'][name]),flush=True)
    summary['runtime_seconds']=time.time()-started
    summary['versions']={n:importlib.metadata.version(n) for n in ['numpy','scipy','sympy','scikit-learn','qiskit','qiskit-aer','joblib']}
    (OUT/'hardware_readiness.json').write_text(json.dumps(summary,indent=2,default=serial))
    (OUT/'replication_instances.json').write_text(json.dumps(fresh,default=serial))
    (OUT/'compiled_circuit_metrics.json').write_text(json.dumps(meta,indent=2))
    np.savez(OUT/'noise_arrays.npz',ids=ids,choices=choices,fixedchoices=fixedchoices,ideal=P,**noisy_arrays)
    print('Completed in',round(summary['runtime_seconds'],2),'seconds',flush=True)


if __name__=='__main__': main()
