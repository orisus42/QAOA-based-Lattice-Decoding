"""EXP025: public-resource feasibility before committing fresh evaluation."""
from pathlib import Path
import json
import time
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.transpiler import PassManager
from threadpoolctl import threadpool_limits
from lattice_qaoa.native_matching import compile_families,match_counts,near_match_counts,bind_angles
from lattice_qaoa.residual_pruning import public_phase
from lattice_qaoa.device import target_from_snapshot,gate_noise,idle_noise_pass,logical_to_physical_indices,readout_channel
from lattice_qaoa.qaoa import state_probabilities
from lattice_qaoa.sparsify import diagonal_energy
from research_run import ResearchRun,save

def main():
    cfg=json.loads((ROOT/'experiments/configs/EXP-025.json').read_text())
    inputs=[ROOT/cfg['development_run']/'instances.json',ROOT/cfg['calibration']]
    run=ResearchRun('EXP-025',Path(__file__),inputs)
    try:
        cases=json.loads(inputs[0].read_text());snap=json.loads(inputs[1].read_text());target=target_from_snapshot(snap)
        rows=[]
        for index in cfg['development_indices']:
            x=cases[index];h,j,_,_=public_phase(np.array(x['reduced_basis']),np.r_[x['public_target'],np.zeros(4,dtype=int)])
            start=time.perf_counter()
            masks,circuits,details=compile_families(h,j,target,cfg['logical_grid'],cfg['selection_angles'],cfg['routing_seeds'])
            matched=near_match_counts(details['resources'],cfg['native_caps'],cfg['native_gap_tolerance'])
            row={'id':x['id'],'match':matched,'exact_match':match_counts(details['resources'],cfg['native_caps']),
                 'seconds':time.perf_counter()-start,'details':details}
            if index==cfg['development_indices'][0]:
                name='cached';k=matched[160]['logical_k'][name]
                bound=bind_angles(circuits[name][k,matched[160]['routing_seed'][name]],cfg['selection_angles'])
                order=logical_to_physical_indices(bound.layout.final_index_layout())
                ideal=AerSimulator(method='statevector',max_parallel_threads=1)
                probe=bound.copy();probe.save_probabilities()
                result=ideal.run(probe,shots=1).result();assert result.success
                prob=np.asarray(result.data(0)['probabilities'])[order]
                np.testing.assert_allclose(prob,state_probabilities(diagonal_energy(h,masks[name][k]),cfg['selection_angles']),atol=1e-12)
                noisy=PassManager(idle_noise_pass(snap)).run(bound);noisy.save_probabilities(pershot=True)
                sim=AerSimulator(noise_model=gate_noise(target),method='statevector',max_parallel_threads=4,
                    max_parallel_shots=4,max_parallel_experiments=1,zero_threshold=0.,fusion_max_qubit=1)
                start=time.perf_counter();result=sim.run(noisy,shots=cfg['profile_trajectories'],seed_simulator=cfg['noise_seed']).result()
                assert result.success
                p=np.asarray(result.data(0)['probabilities'])
                np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-10,rtol=0)
                read=readout_channel(p,snap);np.testing.assert_allclose(read.sum(axis=1),1,atol=1e-10,rtol=0)
                row['noise_profile']={'seconds':time.perf_counter()-start,'trajectories':len(p),
                    'max_norm_deviation':float(np.max(abs(p.sum(axis=1)-1))),'ideal_verified':True}
            rows.append(row);save(run.path/'per_instance.json',rows)
            print(x['id'],'matched',matched,'seconds',row['seconds'],flush=True)
        run.finish({'experiment_id':'EXP-025','count':len(rows),'matching':[r['match'] for r in rows],
            'profile':rows[0]['noise_profile'],'qpu_seconds':0,'limitations':cfg['limitations']})
    except Exception as error:run.finish(error=error);raise

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
