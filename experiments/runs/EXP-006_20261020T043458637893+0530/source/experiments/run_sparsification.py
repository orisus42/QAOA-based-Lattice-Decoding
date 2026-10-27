"""EXP-004: frozen weak-edge pruning with full-objective evaluation and controls."""
from datetime import datetime, timezone, timedelta
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
from scipy.optimize import differential_evolution
from qiskit import transpile
from qiskit_aer import AerSimulator
from threadpoolctl import threadpool_limits
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import circuit, state_probabilities, P1Energy
from lattice_qaoa.sparsify import prune_interactions, diagonal_energy, expected_best_cost
from lattice_qaoa.sampling import repeated_success, solved_count_interval


def now():
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def summarize_values(values):
    x = np.asarray(values, dtype=float)
    return {'mean': float(x.mean()), 'min': float(x.min()),
            'median': float(np.median(x)), 'max': float(x.max())}


def summarize(rows, key, config, bootstrap_indices):
    values = [r['methods'][key] for r in rows]
    base = [r['methods']['full_fixed'] for r in rows]
    p = np.array([v['optimum_probability'] for v in values])
    p0 = np.array([v['optimum_probability'] for v in base])
    ratios = p/p0
    q = repeated_success(p, config['shots'])
    q0 = repeated_success(p0, config['shots'])
    cx = np.array([v['compiled_cx'] for v in values])
    cx0 = np.array([v['compiled_cx'] for v in base])
    boot_p = p[bootstrap_indices].mean(axis=1)/p0[bootstrap_indices].mean(axis=1)
    boot_q = q[bootstrap_indices].mean(axis=1)/q0[bootstrap_indices].mean(axis=1)
    by_budget = []
    for shots in config['budgets']:
        expected = repeated_success(p, shots)
        by_budget.append({'shots': shots, 'expected_solved_fraction': float(expected.mean()),
            'observed_solved_count': sum(v['observed'][str(shots)]['solved'] for v in values),
            'measurement_only_95pct_solved_count_interval': solved_count_interval(expected)})
    bad = ~np.array([r['babai_optimal'] for r in rows])
    pair_order = np.argsort(ratios)
    return {
        'mean_single_shot_optimum_probability': float(p.mean()),
        'single_shot_probability_retention_ratio': float(p.mean()/p0.mean()),
        'single_shot_retention_bootstrap_95pct': np.quantile(boot_p,[.025,.975]).tolist(),
        'expected_solved_fraction_1024': float(q.mean()),
        'solved_fraction_retention_ratio_1024': float(q.mean()/q0.mean()),
        'solved_retention_bootstrap_95pct': np.quantile(boot_q,[.025,.975]).tolist(),
        'by_budget': by_budget,
        'mean_expected_cost': float(np.mean([v['expected_original_cost'] for v in values])),
        'mean_expected_best_cost_1024': float(np.mean([v['expected_best_cost'] for v in values])),
        'mean_expected_hybrid_best_cost_1024': float(np.mean([v['expected_hybrid_best_cost'] for v in values])),
        'expected_hybrid_solved_fraction_1024': float(np.where(bad,q,1).mean()),
        'expected_recovered_babai_failures_1024': float(q[bad].sum()),
        'observed_solved_count_1024': sum(v['observed']['1024']['solved'] for v in values),
        'observed_hybrid_solved_count_1024': sum(v['observed']['1024']['hybrid_solved'] for v in values),
        'observed_returned_secret_recovered_1024': sum(v['observed']['1024']['secret_recovered'] for v in values),
        'compiled_cx': summarize_values(cx),
        'compiled_cx_total': int(cx.sum()),
        'compiled_depth': summarize_values([v['compiled_depth'] for v in values]),
        'compiled_two_qubit_depth': summarize_values([v['compiled_two_qubit_depth'] for v in values]),
        'mean_compiled_cx_reduction': float(1-cx.mean()/cx0.mean()),
        'logical_rzz': summarize_values([v['logical_rzz'] for v in values]),
        'per_instance_probability_retention': summarize_values(ratios),
        'instances_retaining_95pct_probability': int((ratios>=.95).sum()),
        'instances_improved_probability': int((ratios>1+1e-10).sum()),
        'worst_retention_instances': [{'id':rows[int(i)]['id'],'retention':float(ratios[i])} for i in pair_order[:5]],
        'phase_ground_set_overlaps_original_optimum_count': sum(v['phase_ground_overlaps_optimum'] for v in values),
        'phase_ground_set_equals_original_optimum_count': sum(v['phase_ground_equals_optimum'] for v in values)
    }


def main():
    config_path = ROOT/'experiments/configs/EXP-004.json'
    config = json.loads(config_path.read_text())
    started = now()
    run = ROOT/'experiments/runs'/f"EXP-004_{started.strftime('%Y%m%dT%H%M%S%f%z')}"
    run.mkdir(parents=True, exist_ok=False)
    save(run/'config.json', config)
    source_run, baseline = ROOT/config['input_run'], ROOT/config['baseline_run']
    manifest = {'run_id':run.name,'status':'running','started_at':started.isoformat(),
        'command':'python experiments/run_sparsification.py','python':platform.python_version(),
        'packages':{k:version(k) for k in ['numpy','scipy','sympy','qiskit','qiskit-aer','threadpoolctl','matplotlib','seaborn']},
        'source_sha256':{},'input_sha256':{},'qpu_seconds':0}
    sources = list((ROOT/'src/lattice_qaoa').glob('*.py'))+list((ROOT/'tests').glob('test_*.py'))+[
        Path(__file__).resolve(),config_path,ROOT/'scripts/plot_sparsification.py',ROOT/'requirements-research.txt']
    inputs = [source_run/'instances.json',baseline/'training.json',baseline/'per_instance.json',baseline/'summary.json']
    for kind, paths in [('source',sources),('input',inputs)]:
        for path in paths:
            rel=path.relative_to(ROOT)
            dest=run/kind/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest)
            manifest[kind+'_sha256'][rel.as_posix()]=sha256(path.read_bytes()).hexdigest()
    save(run/'manifest.json',manifest)
    start_clock=time.perf_counter()
    try:
        instances=json.loads((source_run/'instances.json').read_text())
        training=json.loads((baseline/'training.json').read_text())
        original_rows=json.loads((baseline/'per_instance.json').read_text())
        old_summary=json.loads((baseline/'summary.json').read_text())
        keys={str(x['A_expanded']) for x in instances}
        assert len(keys)==64, 'Bootstrap assumes distinct matrix groups.'
        from lattice_qaoa.instances import expand_module_matrix
        for x in training['instances']:
            assert str(expand_module_matrix(np.array(x['A_polynomials'])).tolist()) not in keys
        frozen=training['fit']['angles']
        fits={}
        # Training is completed before references are used to score any method.
        for fraction in config['fractions']:
            clock=time.perf_counter()
            evaluators=[]
            for x in training['instances']:
                h,J=np.array(x['h']),np.array(x['J'])
                sparse,_=prune_interactions(J,fraction)
                evaluators.append(P1Energy(h,sparse,observable_h=h,observable_J=J))
            objective=lambda a:np.mean([f(a) for f in evaluators])
            settings=config['training']
            fit=differential_evolution(objective,config['parameter_bounds'],seed=settings['optimizer_seed'],
                maxiter=settings['maxiter'],popsize=settings['popsize'],tol=settings['tol'],polish=True)
            angles=fit.x if fit.fun<objective(frozen) else np.array(frozen)
            fits[str(fraction)]={'angles':angles.tolist(),'training_expected_original_energy':float(objective(angles)),
                'frozen_training_expected_original_energy':float(objective(frozen)),
                'objective_evaluations':int(fit.nfev)+4,'instance_energy_evaluations':(int(fit.nfev)+4)*len(evaluators),
                'optimizer_success':bool(fit.success),'optimizer_message':str(fit.message),
                'seconds':time.perf_counter()-clock}
        save(run/'training.json',{'frozen_angles':frozen,'retraining':fits,
            'training_inputs':'16 public instances snapshotted from EXP-003; no hidden answers or secrets used.'})
        method_names=['full_fixed','full_tuned']+[f'weak{int(100*f)}_{policy}' for f in config['fractions'] for policy in ['frozen','retrained']]
        samples={key:np.zeros((64,config['shots']),dtype=np.int64) for key in method_names}
        rows=[]
        checks={'original_binary_costs_checked':0,'compiled_circuits_verified':0,
                'max_probability_discrepancy':0.,'max_analytic_energy_discrepancy':0.,'baseline_reproduced':0}
        simulator=AerSimulator(method='statevector',max_parallel_threads=2,max_parallel_experiments=1)
        for index,(x,prior) in enumerate(zip(instances,original_rows)):
            assert x['id']==prior['id']
            y=np.r_[x['public_target'],np.zeros(4,dtype=int)]
            box=encode_babai_box(np.array(x['reduced_basis']),y)
            good=box.costs==x['reference']['cost']
            assert good.any() and box.nq==16
            checks['original_binary_costs_checked']+=len(box.costs)
            rec={'id':x['id'],'seed':x['seed'],'optimum':x['reference']['cost'],
                 'babai_cost':x['lll_babai_cost'],'babai_optimal':x['lll_babai_optimal'],
                 'normalization':box.normalization,'methods':{},'random25':[]}
            specs=[('full_fixed',0.,frozen),('full_tuned',0.,prior['angles']['qaoa_tuned'])]
            for f in config['fractions']:
                specs += [(f'weak{int(100*f)}_frozen',f,frozen),
                          (f'weak{int(100*f)}_retrained',f,fits[str(f)]['angles'])]
            distributions=[]
            circuits=[]
            for name,fraction,angles in specs:
                sparse,removed=prune_interactions(box.J,fraction)
                phase=diagonal_energy(box.h,sparse)
                prob=state_probabilities(phase,angles)
                np.testing.assert_allclose(prob.sum(),1,atol=1e-10)
                energy_fn=P1Energy(box.h,sparse,observable_h=box.h,observable_J=box.J)
                discrepancy=abs(float(prob@box.energy)-energy_fn(angles))
                assert discrepancy<1e-9
                checks['max_analytic_energy_discrepancy']=max(checks['max_analytic_energy_discrepancy'],discrepancy)
                optimum_probability=float(prob[good].sum())
                if fraction==0:
                    old_key='qaoa_fixed' if name=='full_fixed' else 'qaoa_tuned'
                    np.testing.assert_allclose(optimum_probability,prior['probabilities'][old_key],rtol=1e-10,atol=1e-12)
                    checks['baseline_reproduced']+=1
                qc=circuit(box.h,sparse,angles)
                compiled=transpile(qc,basis_gates=config['compilation']['basis_gates'],optimization_level=3,
                                   seed_transpiler=config['compilation']['seed'])
                ground=np.isclose(phase,phase.min(),atol=1e-10,rtol=0)
                rec['methods'][name]={'fraction':fraction,'angles':list(angles),
                    'removed_edges':removed.tolist(),'removed_absolute_weight':float(np.abs(box.J-sparse).sum()),
                    'optimum_probability':optimum_probability,'refinement_probability':float(prob[box.costs<rec['babai_cost']].sum()),
                    'expected_original_cost':float(prob@box.costs),
                    'expected_best_cost':expected_best_cost(box.costs,prob,config['shots']),
                    'expected_hybrid_best_cost':expected_best_cost(box.costs,prob,config['shots'],rec['babai_cost']),
                    'logical_rzz':int(qc.count_ops().get('rzz',0)),
                    'compiled_cx':int(compiled.count_ops().get('cx',0)),
                    'compiled_depth':compiled.depth(),
                    'compiled_two_qubit_depth':compiled.depth(lambda op:op.operation.num_qubits==2),
                    'compiled_operations':dict(compiled.count_ops()),
                    'phase_ground_overlaps_optimum':bool(np.any(ground&good)),
                    'phase_ground_equals_optimum':bool(np.array_equal(ground,good)),
                    'observed':{}}
                compiled.save_probabilities(label='probabilities')
                compiled.measure_all()
                circuits.append(compiled)
                distributions.append(prob)
            result=simulator.run(circuits,shots=config['shots'],memory=True,
                                seed_simulator=config['measurement_seed']+index).result()
            assert result.success
            for j,(name,_,_) in enumerate(specs):
                qprob=np.asarray(result.data(j)['probabilities'])
                error=float(np.max(np.abs(qprob-distributions[j])))
                assert error<1e-10
                checks['max_probability_discrepancy']=max(checks['max_probability_discrepancy'],error)
                checks['compiled_circuits_verified']+=1
                measured=np.array([int(bits.replace(' ',''),2) for bits in result.get_memory(j)])
                samples[name][index]=measured
                costs=box.costs[measured]
                for n in config['budgets']:
                    best_idx=int(costs[:n].argmin())
                    best=int(costs[best_idx])
                    z=np.array(x['basis_transform'])@box.decode([measured[best_idx]])[0]
                    rec['methods'][name]['observed'][str(n)]={'successful_shots':int(good[measured[:n]].sum()),
                        'best_cost':best,'solved':best==rec['optimum'],
                        'hybrid_solved':min(best,rec['babai_cost'])==rec['optimum'],
                        'secret_recovered':bool(np.array_equal(z[4:],x['planted_secret']))}
            for k in range(config['random_removal_replicates']):
                seed=config['random_removal_seed']+100*index+k
                sparse,removed=prune_interactions(box.J,.25,random_seed=seed)
                prob=state_probabilities(diagonal_energy(box.h,sparse),frozen)
                rec['random25'].append({'seed':seed,'removed_edges':removed.tolist(),
                    'optimum_probability':float(prob[good].sum()),
                    'expected_original_cost':float(prob@box.costs),
                    'logical_rzz':int(np.count_nonzero(sparse))})
            rows.append(rec)
            if (index+1)%8==0:
                print(f'Compared, compiled and sampled {index+1}/64 instances ({8*(index+1)} circuits)',flush=True)
        rng=np.random.default_rng(config['bootstrap_seed'])
        indices=rng.integers(0,64,size=(config['bootstrap_replicates'],64))
        methods={key:summarize(rows,key,config,indices) for key in method_names}
        random_p=np.array([[r['optimum_probability'] for r in row['random25']] for row in rows])
        primary=methods[config['primary_method']]
        thresholds={
            'single_shot_retention_at_least_95pct':primary['single_shot_probability_retention_ratio']>=.95,
            'solved_fraction_retention_at_least_95pct':primary['solved_fraction_retention_ratio_1024']>=.95,
            'compiled_cx_reduction_at_least_20pct':primary['mean_compiled_cx_reduction']>=.2}
        summary={'experiment_id':'EXP-004','run_id':run.name,'count':64,'qubits':16,'depth_p':1,
            'cohort':config['cohort'],'box_optimum_coverage_count':64,'primary_method':config['primary_method'],
            'primary_thresholds':thresholds,'primary_hypothesis_met':all(thresholds.values()),'methods':methods,
            'random25_control':{'replicates_per_instance':config['random_removal_replicates'],
                'mean_single_shot_probability':float(random_p.mean()),
                'mean_expected_solved_fraction_1024':float(repeated_success(random_p,config['shots']).mean()),
                'replicate_cohort_mean_single_shot_probabilities':random_p.mean(axis=0).tolist(),
                'note':'Unchanged full-trained angles, same number of removed interactions; exact distributions, no measured shots.'},
            'classical_controls_from_EXP003':{'babai':old_summary['babai'],'local_refinement':old_summary['classical_refinement'],
                'uniform_box':old_summary['methods']['uniform_box'],'small_secret_sampling':old_summary['methods']['cbd_secret']},
            'training':fits,'validation':checks,'actual_ideal_measurements':64*len(method_names)*config['shots'],
            'timing_seconds':time.perf_counter()-start_clock,'qpu_seconds':0,'limitations':config['limitations']}
        save(run/'per_instance.json',rows)
        save(run/'summary.json',summary)
        np.savez_compressed(run/'samples.npz',**samples)
        manifest.update(status='completed',finished_at=now().isoformat())
        manifest['output_sha256']={n:sha256((run/n).read_bytes()).hexdigest() for n in ['config.json','training.json','per_instance.json','summary.json','samples.npz']}
        save(run/'manifest.json',manifest)
        print('RUN:',run,flush=True)
        for key,m in methods.items():
            print(key,'CX',round(m['compiled_cx']['mean'],2),'retention',round(m['single_shot_probability_retention_ratio'],4),
                  'P1024',round(m['expected_solved_fraction_1024'],4),'observed',m['observed_solved_count_1024'],flush=True)
        print('PRIMARY:',thresholds,'random25 P1024:',summary['random25_control']['mean_expected_solved_fraction_1024'],flush=True)
    except Exception as exc:
        manifest.update(status='failed',finished_at=now().isoformat(),error=repr(exc))
        save(run/'manifest.json',manifest)
        raise


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
