"""Independent tiny density-matrix cross-check of the trajectory noise model."""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from qiskit import QuantumCircuit,transpile
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from lattice_qaoa.device import target_from_snapshot,gate_noise,idle_noise_pass,readout_channel
from lattice_qaoa.qaoa import circuit
from threadpoolctl import threadpool_limits


def main():
    run=Path(sys.argv[1]).resolve();cfg=json.loads((run/'config.json').read_text())
    snap=json.loads((ROOT/cfg['calibration']/'snapshot.json').read_text())
    edges=[g['qubits'] for g in snap['gates'] if g['name']=='cz']
    nodes=set(edges[0])
    for pair in edges:
        if len(nodes|set(pair))==3 and nodes&set(pair):nodes|=set(pair);break
    nodes=sorted(nodes);mapping={n:i for i,n in enumerate(nodes)}
    small=dict(snap);small['num_qubits']=3;small['qubits']=[snap['qubits'][q] for q in nodes]
    small['gates']=[dict(g,qubits=[mapping[q] for q in g['qubits']]) for g in snap['gates'] if set(g['qubits'])<=set(nodes)]
    target=target_from_snapshot(small)
    qc=QuantumCircuit(3)
    layer=circuit(np.array([.2,-.4,.3]),np.array([[0,.3,-.2],[0,0,.4],[0,0,0]]),[1.1,-.35])
    for _ in range(5):qc.compose(layer,inplace=True)
    qc.barrier()
    compiled=transpile(qc,target=target,initial_layout=[0,1,2],optimization_level=3,seed_transpiler=2026102050,scheduling_method='alap')
    noisy=PassManager(idle_noise_pass(small)).run(compiled)
    dm=noisy.copy();dm.save_probabilities()
    sv=noisy.copy();sv.save_probabilities(pershot=True)
    model=gate_noise(target)
    exact=np.asarray(AerSimulator(method='density_matrix',noise_model=model,max_parallel_threads=1).run(dm,shots=1).result().data(0)['probabilities'])
    trajectories=np.asarray(AerSimulator(method='statevector',noise_model=model,max_parallel_threads=2,max_parallel_shots=2).run(sv,shots=4096,seed_simulator=2026102051).result().data(0)['probabilities'])
    exact=readout_channel(exact,small);trajectories=readout_channel(trajectories,small)
    mean=trajectories.mean(axis=0);se=trajectories.std(axis=0,ddof=1)/np.sqrt(len(trajectories))
    z=np.abs(mean-exact)/np.maximum(se,1e-15)
    assert np.all(z<5),z
    ideal=compiled.copy();ideal.save_probabilities()
    idealp=np.asarray(AerSimulator(method='statevector',max_parallel_threads=1).run(ideal,shots=1).result().data(0)['probabilities'])
    result={'recorded_at':datetime.now().astimezone().isoformat(),'scope':'Independent 3-qubit circuit validation only, not a Module-LWE performance experiment',
        'trajectories':4096,'exact_density_matrix_probabilities':exact.tolist(),'trajectory_average_probabilities':mean.tolist(),
        'standard_errors':se.tolist(),'maximum_standardized_discrepancy':float(z.max()),
        'maximum_absolute_discrepancy':float(np.max(np.abs(mean-exact))),
        'total_variation_from_noiseless':float(.5*np.abs(exact-idealp).sum()),
        'physical_qubits':[snap['physical_qubits'][q] for q in nodes], 'seed':2026102051,'qpu_seconds':0}
    out=run/'model_validation';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'validation_source.py')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
