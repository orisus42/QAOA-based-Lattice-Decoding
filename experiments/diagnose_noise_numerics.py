"""Numerical diagnostic independent of decoder performance; no QPU calls."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from qiskit import transpile,qpy
from qiskit_aer import AerSimulator
from threadpoolctl import threadpool_limits
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import circuit
from lattice_qaoa.device import target_from_snapshot,gate_noise
from research_run import ResearchRun,save

def main():
    source=ROOT/'experiments/runs/EXP-001_20261018T120321801881+0530/instances.json'
    cal=ROOT/'research/calibrations/ibm_fez_20261020T050035937321+0530/snapshot.json'
    run=ResearchRun('DIAG-013',Path(__file__),[source,cal]);cfg=run.config
    try:
        snap=json.loads(cal.read_text());target=target_from_snapshot(snap);x=json.loads(source.read_text())[cfg['source_index']]
        b=encode_babai_box(np.array(x['reduced_basis']),np.r_[x['public_target'],np.zeros(4,dtype=int)])
        q=circuit(b.h,b.J,[1.3393885543560986,-.3516024709404892]);q.barrier()
        q=transpile(q,target=target,initial_layout=list(range(16)),optimization_level=3,seed_transpiler=2026102030,scheduling_method='alap')
        with (run.path/'circuit.qpy').open('wb') as f:qpy.dump(q,f)
        q.save_probabilities(pershot=True);rows=[]
        for i,setting in enumerate(cfg['settings']):
            timer=time.perf_counter()
            sim=AerSimulator(method='statevector',noise_model=gate_noise(target),max_parallel_threads=4,max_parallel_shots=4,zero_threshold=0,**setting)
            r=sim.run(q,shots=cfg['shots'],seed_simulator=cfg['seed']).result();assert r.success
            p=np.array(r.data(0)['probabilities']);norm=p.sum(axis=1)
            np.savez_compressed(run.path/f'probabilities_{i}.npz',probabilities=p,norms=norm)
            rows.append({'settings':setting,'seconds':time.perf_counter()-timer,'max_norm_deviation':float(np.max(abs(norm-1))),'fusion':r.results[0].metadata.get('fusion')})
            save(run.path/'diagnostics.json',rows);print(rows[-1],flush=True)
        run.finish({'experiment_id':'DIAG-013','results':rows,'qpu_seconds':0,'limits':'One reproducible circuit/seed; identifies an association with fusion settings, not a complete upstream root-cause analysis.'})
    except Exception as e:run.finish(error=e);raise

if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
