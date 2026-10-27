"""EXP-008: local-only IBM-calibrated routed/noisy QAOA; never submits a job."""
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from qiskit import transpile, qpy
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from threadpoolctl import threadpool_limits
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import circuit, state_probabilities
from lattice_qaoa.sparsify import prune_interactions, diagonal_energy
from lattice_qaoa.device import (target_from_snapshot, gate_noise, idle_noise_pass,
    logical_to_physical_indices, readout_channel, hybrid_metrics_from_low_masses)
from analyze_hybrid_refinement import check_manifest


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str)+'\n', encoding='utf-8')


def main():
    started = datetime.now().astimezone(); timer = time.perf_counter()
    config_path = ROOT/'experiments/configs/EXP-008.json'
    cfg = json.loads(config_path.read_text())
    source = ROOT/cfg['input_run']; cal = ROOT/cfg['calibration']
    run = ROOT/'experiments/runs'/f"EXP-008_{started.strftime('%Y%m%dT%H%M%S%f%z')}"
    run.mkdir(exist_ok=False); (run/'circuits').mkdir(); (run/'noise').mkdir()
    save(run/'config.json', cfg)
    manifest = {'run_id': run.name, 'status': 'routing', 'started_at': started.isoformat(),
        'command': 'python experiments/run_ibm_noise.py', 'python': platform.python_version(),
        'packages': {k: version(k) for k in ['numpy','sympy','qiskit','qiskit-aer','qiskit-ibm-runtime','threadpoolctl','scipy','seaborn','matplotlib']},
        'source_sha256': {}, 'input_sha256': {}, 'output_sha256': {}, 'qpu_jobs_submitted': 0, 'qpu_seconds': 0}
    sources = list((ROOT/'src/lattice_qaoa').glob('*.py')) + list((ROOT/'tests').glob('test_*.py')) + [
        Path(__file__), config_path, ROOT/'experiments/analyze_hybrid_refinement.py',
        ROOT/'scripts/analyze_ibm_noise.py', ROOT/'requirements-research.txt']
    inputs = [source/n for n in ['instances.json','per_instance.json','summary.json','config.json','manifest.json']]
    inputs += list(cal.glob('*.json')) + [cal/'capture_source.py',ROOT/cfg['profile_run']/'profile.json']
    for kind, paths in [('source', sources), ('input', inputs)]:
        for path in paths:
            rel = path.relative_to(ROOT); dest = run/kind/rel
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
            manifest[kind+'_sha256'][rel.as_posix()] = sha256(path.read_bytes()).hexdigest()
    save(run/'manifest.json', manifest)
    print('RUN:', run, flush=True)
    try:
        checks = {'input_manifest_hashes': check_manifest(source), 'routed_ideal_circuits_verified': 0,
            'native_two_qubit_edges_verified': 0, 'max_routed_probability_discrepancy': 0.,
            'noisy_circuits_completed': 0, 'noise_trajectories': 0, 'readout_normalization_checks': 0,
            'low_cost_mass_agreement_checks': 0}
        snap = json.loads((cal/'snapshot.json').read_text())
        for name, digest in json.loads((cal/'manifest.json').read_text())['files_sha256'].items():
            assert sha256((cal/name).read_bytes()).hexdigest() == digest
        target = target_from_snapshot(snap)
        instances = json.loads((source/'instances.json').read_text())
        earlier = json.loads((source/'per_instance.json').read_text())
        previous_cfg = json.loads((source/'config.json').read_text())
        assert len(instances) == cfg['routing_count']
        random_ids = sorted(np.random.default_rng(cfg['random_subset_seed']).choice(len(instances), cfg['random_subset_count'], replace=False).tolist())
        failures = [i for i, r in enumerate(earlier) if r['babai_cost'] > r['optimum_cost']]
        noise_ids = sorted(set(random_ids) | set(failures))
        save(run/'cohort.json', {'all_count': len(instances), 'random_subset_indices': random_ids,
             'babai_failure_indices': failures, 'noise_union_indices': noise_ids,
             'rule': cfg['noise_cohort'], 'selected_before_noise_at': datetime.now().astimezone().isoformat()})
        ideal_sim = AerSimulator(method='statevector',max_parallel_threads=1)
        routed_rows = []
        gate_model = gate_noise(target)
        save(run/'native_gate_noise_model.json',gate_model.to_dict(serializable=True))
        for i, (x, old) in enumerate(zip(instances, earlier)):
            box = encode_babai_box(np.array(x['reduced_basis']), np.r_[x['public_target'], np.zeros(4,dtype=int)])
            row = {'id': x['id'], 'source_index': i, 'group': x['group'], 'babai_cost': old['babai_cost'],
                'optimum_cost': old['optimum_cost'], 'methods': {}}
            circuits, checks_circuits, expected = [], [], []
            for name in cfg['methods']:
                spec = previous_cfg['methods'][name]
                J, removed = prune_interactions(box.J, spec['fraction'])
                assert removed.tolist() == old['methods'][name]['removed_edges']
                logical = circuit(box.h, J, spec['angles']); logical.barrier()
                routed = transpile(logical, target=target, initial_layout=list(range(16)),
                    optimization_level=cfg['optimization_level'], seed_transpiler=cfg['routing_seed'], scheduling_method='alap')
                permutation = routed.layout.final_index_layout()
                order = logical_to_physical_indices(permutation)
                for instruction in routed.data:
                    op = instruction.operation
                    if op.num_qubits == 2:
                        pair = tuple(routed.find_bit(q).index for q in instruction.qubits)
                        assert op.name == 'cz' and pair in target['cz']
                        checks['native_two_qubit_edges_verified'] += 1
                duration = routed.estimate_duration(target)
                row['methods'][name] = {'fraction': spec['fraction'], 'angles': spec['angles'],
                    'final_logical_to_patch_permutation': permutation, 'native_cz': int(routed.count_ops().get('cz',0)),
                    'depth': routed.depth(), 'two_qubit_depth': routed.depth(lambda op: op.operation.num_qubits == 2),
                    'duration_unitary_s': duration, 'duration_with_readout_s': duration+max(g['duration'] for g in snap['gates'] if g['name']=='measure'),
                    'idle_delay_count': int(routed.count_ops().get('delay',0)), 'count_ops': dict(routed.count_ops()),
                    'ideal_by_budget': {str(b):old['methods'][name]['by_budget'][str(b)] for b in cfg['budgets']}}
                expected.append(state_probabilities(diagonal_energy(box.h,J),spec['angles']))
                circuits.append(routed)
                check = routed.copy(); check.save_probabilities()
                checks_circuits.append(check)
            result = ideal_sim.run(checks_circuits,shots=1).result()
            assert result.success
            for j,name in enumerate(cfg['methods']):
                order = logical_to_physical_indices(row['methods'][name]['final_logical_to_patch_permutation'])
                delta = float(np.max(np.abs(np.asarray(result.data(j)['probabilities'])[order]-expected[j])))
                assert delta < 1e-10
                checks['max_routed_probability_discrepancy'] = max(checks['max_routed_probability_discrepancy'],delta)
                checks['routed_ideal_circuits_verified'] += 1
            with (run/'circuits'/f'{x["id"]}.qpy').open('wb') as f:qpy.dump(circuits,f)
            routed_rows.append(row)
            if (i+1)%64==0:
                print(f'Routed and verified {i+1}/512 instances; elapsed {time.perf_counter()-timer:.1f}s',flush=True)
        save(run/'routed_instances.json',routed_rows)
        routing_seconds = time.perf_counter()-timer
        manifest.update(status='noise_simulation',routing_finished_at=datetime.now().astimezone().isoformat())
        save(run/'manifest.json',manifest)
        sim = AerSimulator(noise_model=gate_model,**cfg['simulator'])
        idle = PassManager(idle_noise_pass(snap))
        for ni,i in enumerate(noise_ids):
            x=instances[i];row=routed_rows[i]
            box=encode_babai_box(np.array(x['reduced_basis']),np.r_[x['public_target'],np.zeros(4,dtype=int)])
            with (run/'circuits'/f'{x["id"]}.qpy').open('rb') as f:circuits=qpy.load(f)
            for j,name in enumerate(cfg['methods']):
                clock=time.perf_counter()
                noisy=idle.run(circuits[j]);noisy.save_probabilities(pershot=True)
                seed=cfg['noise_seed']+13*i+j
                result=sim.run(noisy,shots=cfg['trajectories_per_circuit'],seed_simulator=seed).result()
                assert result.success
                probability=np.asarray(result.data(0)['probabilities'])
                assert probability.shape==(cfg['trajectories_per_circuit'],65536)
                np.testing.assert_allclose(probability.sum(axis=1),1,atol=1e-10)
                order=logical_to_physical_indices(row['methods'][name]['final_logical_to_patch_permutation'])
                pre_readout=probability.mean(axis=0)[order]
                probability=readout_channel(probability,snap)[:,order]
                np.testing.assert_allclose(probability.sum(axis=1),1,atol=1e-10)
                checks['readout_normalization_checks']+=len(probability)
                incumbent=row['babai_cost'];optimum=row['optimum_cost']
                low=np.stack([probability[:,box.costs==c].sum(axis=1) for c in range(incumbent+1)],axis=1)
                assert np.all(low[:,:optimum]<1e-12)
                mean=probability.mean(axis=0)
                np.testing.assert_allclose(low.mean(axis=0),[mean[box.costs==c].sum() for c in range(incumbent+1)],atol=1e-12)
                checks['low_cost_mass_agreement_checks']+=1
                file=run/'noise'/f'{x["id"]}_{name}.npz'
                np.savez_compressed(file,low_cost_masses=low,mean_logical_probability=mean,
                    mean_before_readout_logical_probability=pre_readout,simulation_seed=np.array(seed))
                checks['noisy_circuits_completed']+=1;checks['noise_trajectories']+=len(probability)
                checkpoint={'recorded_at':datetime.now().astimezone().isoformat(),'id':x['id'],'method':name,
                    'seed':seed,'trajectories':len(probability),'seconds':time.perf_counter()-clock,
                    'file':file.relative_to(run).as_posix(),'sha256':sha256(file.read_bytes()).hexdigest(),
                    'simulator_parallel_shots':result.results[0].metadata.get('parallel_shots'),
                    'simulator_parallel_state_update':result.results[0].metadata.get('parallel_state_update')}
                with (run/'noise_progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(checkpoint)+'\n')
            print(f'Noisy trajectories finished {ni+1}/{len(noise_ids)} instances; elapsed {time.perf_counter()-timer:.1f}s',flush=True)
        save(run/'validation.json',checks)
        save(run/'timing.json',{'routing_seconds':routing_seconds,'simulation_and_routing_seconds':time.perf_counter()-timer})
        manifest.update(status='simulation_completed',finished_at=datetime.now().astimezone().isoformat())
        for path in run.rglob('*'):
            if path.is_file() and path.name!='manifest.json' and path.relative_to(run).parts[0] not in ['source','input']:
                manifest['output_sha256'][path.relative_to(run).as_posix()]=sha256(path.read_bytes()).hexdigest()
        save(run/'manifest.json',manifest)
        print('SIMULATION COMPLETE:',run,flush=True)
    except Exception as error:
        manifest.update(status='failed',finished_at=datetime.now().astimezone().isoformat(),error=repr(error))
        save(run/'manifest.json',manifest)
        raise


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
