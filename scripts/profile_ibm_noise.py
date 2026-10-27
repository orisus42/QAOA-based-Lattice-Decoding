"""Local simulator runtime feasibility on an OLD development case; no QPU jobs."""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from qiskit import transpile
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from lattice_qaoa.device import target_from_snapshot, gate_noise, idle_noise_pass, logical_to_physical_indices
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import circuit, state_probabilities
from lattice_qaoa.sparsify import prune_interactions, diagonal_energy
from threadpoolctl import threadpool_limits


def main():
    calibration = Path(sys.argv[1])
    snap = json.loads((calibration/'snapshot.json').read_text())
    target = target_from_snapshot(snap)
    x = json.loads((ROOT/'experiments/runs/EXP-001_20261018T120321801881+0530/instances.json').read_text())[0]
    box = encode_babai_box(np.array(x['reduced_basis']), np.r_[x['public_target'], np.zeros(4, dtype=int)])
    fullangles = [1.3393885543560986, -.3516024709404892]
    out = ROOT/'experiments/runs'/('PRE-008_'+datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%f%z'))
    out.mkdir()
    shutil.copy2(__file__, out/'profile_source.py')
    simulator = AerSimulator(method='statevector', noise_model=gate_noise(target),
        max_parallel_threads=4, max_parallel_shots=4, max_parallel_experiments=1, max_memory_mb=2048)
    report = {'scope': 'Runtime profiling on old development MLWE-000; not a new quality evaluation',
        'started_at': datetime.now().astimezone().isoformat(), 'calibration': str(calibration), 'qpu_seconds': 0, 'cases': []}
    for fraction in [0., .75]:
        J, _ = prune_interactions(box.J, fraction)
        qc = circuit(box.h, J, fullangles); qc.barrier()
        routed = transpile(qc, target=target, initial_layout=list(range(16)),
            optimization_level=3, seed_transpiler=2026102030, scheduling_method='alap')
        permutation = routed.layout.final_index_layout()
        order = logical_to_physical_indices(permutation)
        ideal = routed.copy(); ideal.save_probabilities()
        expected = state_probabilities(diagonal_energy(box.h, J), fullangles)
        actual = AerSimulator(method='statevector',max_parallel_threads=1).run(ideal).result().data(0)['probabilities']
        delta = float(np.max(np.abs(np.asarray(actual)[order]-expected)))
        assert delta < 1e-10
        noisy = PassManager(idle_noise_pass(snap)).run(routed)
        noisy.save_probabilities(pershot=True)
        clock = time.perf_counter()
        result = simulator.run(noisy, shots=16, seed_simulator=2026102031).result()
        assert result.success
        elapsed = time.perf_counter()-clock
        probs = np.asarray(result.data(0)['probabilities'])
        assert probs.shape == (16, 65536)
        row = {'fraction': fraction, 'cz': int(routed.count_ops().get('cz', 0)),
            'depth': routed.depth(), 'duration_s': routed.estimate_duration(target),
            'idle_delays': int(routed.count_ops().get('delay', 0)), 'probability_error': delta,
            'trajectories': 16, 'seconds': elapsed, 'seconds_per_trajectory': elapsed/16,
            'simulation_metadata': result.results[0].metadata}
        report['cases'].append(row)
        print(json.dumps({k:v for k,v in row.items() if k != 'simulation_metadata'}), flush=True)
    report['finished_at'] = datetime.now().astimezone().isoformat()
    (out/'profile.json').write_text(json.dumps(report, indent=2, default=str)+'\n', encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({'files_sha256': {p.name:sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}}, indent=2)+'\n', encoding='utf-8')
    print('PROFILE:', out, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='blas'): main()
