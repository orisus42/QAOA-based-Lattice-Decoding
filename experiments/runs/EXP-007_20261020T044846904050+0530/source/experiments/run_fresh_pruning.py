"""EXP-007: a frozen, prospective evaluation on new Module-LWE matrix groups."""
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from importlib.metadata import version
from itertools import product
from math import comb
import json
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from threadpoolctl import threadpool_limits
from lattice_qaoa.instances import Parameters, generate_instance, expand_module_matrix
from lattice_qaoa.reference import embedding, lll_reduce, exact_cvp, squared_cost
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.classical import local_refine, lift_secrets
from lattice_qaoa.qaoa import P1Energy, circuit, state_probabilities
from lattice_qaoa.sparsify import prune_interactions, diagonal_energy
from lattice_qaoa.sampling import solved_count_interval
from lattice_qaoa.evaluation import matrix_group, distribution_metrics, paired_retention
from analyze_hybrid_refinement import analyze_distribution, check_manifest


def now():
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')


def mean_ci(values, boot):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'ci95': np.quantile(a[boot].mean(axis=1), [.025, .975]).tolist()}


def summary_for(rows, name, budgets, boot):
    out = {'by_budget': {}, 'single_shot_success': mean_ci([r['methods'][name]['p_optimum'] for r in rows], boot)}
    for shots in budgets:
        metrics = [r['methods'][name]['by_budget'][str(shots)] for r in rows]
        values = {}
        for key in ['standalone_success', 'hybrid_success', 'strict_improvement', 'hybrid_expected_cost',
                    'hybrid_expected_distance', 'standalone_expected_distance']:
            values[key] = mean_ci([m[key] for m in metrics], boot)
        failures = [m for r, m in zip(rows, metrics) if r['babai_cost'] > r['optimum_cost']]
        values['babai_failures_recovered_fraction'] = float(np.mean([m['hybrid_success'] for m in failures])) if failures else None
        values['expected_recovered_count'] = float(np.sum([m['hybrid_success'] for m in failures]))
        values['measurement_only_solved_count_interval95'] = solved_count_interval([m['hybrid_success'] for m in metrics])
        values['observed'] = {
            'standalone_solved_count': sum(m['observed_best_cost'] == r['optimum_cost'] for r, m in zip(rows, metrics)),
            'hybrid_solved_count': sum(m['observed_hybrid_cost'] == r['optimum_cost'] for r, m in zip(rows, metrics)),
            'strict_improvement_count': sum(m['observed_hybrid_cost'] < r['babai_cost'] for r, m in zip(rows, metrics)),
            'hybrid_mean_distance': float(np.mean([np.sqrt(m['observed_hybrid_cost']) for m in metrics])),
            'hybrid_planted_secret_recovered_count': sum(m['observed_hybrid_secret_recovered'] for m in metrics),
        }
        out['by_budget'][str(shots)] = values
    if 'compiled_cx' in rows[0]['methods'][name]:
        for key in ['compiled_cx', 'compiled_depth', 'compiled_two_qubit_depth', 'logical_rzz']:
            a = np.array([r['methods'][name][key] for r in rows])
            out[key] = {'mean': float(a.mean()), 'min': int(a.min()), 'median': float(np.median(a)), 'max': int(a.max())}
    return out


def main():
    config_path = ROOT/'experiments/configs/EXP-007.json'
    cfg = json.loads(config_path.read_text())
    start, timer = now(), time.perf_counter()
    run = ROOT/'experiments/runs'/f"EXP-007_{start.strftime('%Y%m%dT%H%M%S%f%z')}"
    run.mkdir(exist_ok=False)
    save(run/'config.json', cfg)
    manifest = {'run_id': run.name, 'status': 'running', 'started_at': start.isoformat(),
        'command': 'python experiments/run_fresh_pruning.py', 'python': platform.python_version(),
        'platform': platform.platform(), 'source_sha256': {}, 'input_sha256': {}, 'qpu_seconds': 0,
        'packages': {k: version(k) for k in ['numpy', 'scipy', 'sympy', 'qiskit', 'qiskit-aer', 'threadpoolctl', 'seaborn', 'matplotlib']}}
    sources = list((ROOT/'src/lattice_qaoa').glob('*.py'))+list((ROOT/'tests').glob('test_*.py'))+[
        Path(__file__), config_path, ROOT/'experiments/analyze_hybrid_refinement.py',
        ROOT/'scripts/plot_fresh_pruning.py', ROOT/'requirements-research.txt']
    inputs = [ROOT/cfg['development_run']/'instances.json', ROOT/cfg['training_run']/'training.json',
              ROOT/cfg['angle_run']/'training.json', ROOT/cfg['angle_run']/'manifest.json']
    for kind, paths in [('source', sources), ('input', inputs)]:
        for path in paths:
            rel = path.relative_to(ROOT)
            dest = run/kind/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            manifest[kind+'_sha256'][rel.as_posix()] = sha256(path.read_bytes()).hexdigest()
    save(run/'manifest.json', manifest)
    print('RUN:', run, flush=True)
    try:
        checks = {'prior_manifest_hashes': check_manifest(ROOT/cfg['angle_run']),
            'unique_fresh_groups': 0, 'overlap_with_training_or_development': 0,
            'binary_costs_checked': 0, 'compiled_circuits_verified': 0, 'order_statistic_checks': 0,
            'returned_candidates_verified': 0, 'max_probability_discrepancy': 0., 'max_analytic_energy_discrepancy': 0.}
        training = json.loads(inputs[1].read_text())
        fits = json.loads(inputs[2].read_text())
        np.testing.assert_array_equal(cfg['methods']['full_fixed']['angles'], training['fit']['angles'])
        np.testing.assert_array_equal(cfg['methods']['weak75_frozen']['angles'], training['fit']['angles'])
        np.testing.assert_array_equal(cfg['methods']['weak90_retrained']['angles'], fits['new_fits']['0.9']['angles'])
        np.testing.assert_array_equal(cfg['methods']['weak75_retrained']['angles'], fits['new_fits']['0.75']['angles'])
        params = Parameters(**cfg['parameters'])
        old = json.loads(inputs[0].read_text())
        excluded = {matrix_group(x['A_expanded'], params.q) for x in old}
        excluded |= {matrix_group(expand_module_matrix(np.array(x['A_polynomials'])), params.q) for x in training['instances']}
        assert len(excluded) == 80
        # Generate the entire fixed cohort and audit public groups before evaluating outcomes.
        cohort = [generate_instance(cfg['first_seed']+i, params) for i in range(cfg['count'])]
        groups = [matrix_group(expand_module_matrix(x.A), params.q) for x in cohort]
        assert len(set(groups)) == cfg['count'] and not (set(groups) & excluded)
        checks['unique_fresh_groups'] = len(set(groups))
        save(run/'cohort.json', {'seeds': [x.seed for x in cohort], 'groups': groups,
             'excluded_groups': sorted(excluded), 'policy': cfg['cohort_policy'], 'audited_at': now().isoformat()})
        names = list(cfg['methods']) + ['uniform_box', 'cbd_secret']
        samples = {name: np.empty((cfg['count'], cfg['shots']), dtype=np.uint16) for name in names}
        secret_support = np.array(list(product(range(-params.eta, params.eta+1), repeat=params.n*params.k)), dtype=np.int64)
        secret_mass = np.array([np.prod([comb(2*params.eta, params.eta+int(v))/2**(2*params.eta) for v in s]) for s in secret_support])
        np.testing.assert_allclose(secret_mass.sum(), 1, atol=1e-15)
        save(run/'sample_codebook.json', {'qaoa_and_uniform': 'uint16 is a little-endian 16-qubit basis-state index',
             'cbd_secret': 'uint16 indexes secret_support; its best modular lift is computed from the public instance',
             'secret_support': secret_support.tolist(), 'secret_mass': secret_mass.tolist(), 'sampling': cfg['sampling']})
        simulator = AerSimulator(method='statevector', max_parallel_threads=2, max_parallel_experiments=1)
        instances, rows = [], []
        for index, x in enumerate(cohort):
            A, t = expand_module_matrix(x.A), x.target.ravel()
            B, y = embedding(A, t, params.q)
            R, U = lll_reduce(B)
            np.testing.assert_array_equal(R, B@U)
            box = encode_babai_box(R, y)
            assert box.nq == cfg['qubits']
            mapped_babai = U@box.center
            babai_cost = squared_cost(B, y, mapped_babai)
            reference = exact_cvp(A, t, params.q, mapped_babai, cfg['exact_reference_candidate_cap'])
            optimum = reference['cost']
            checks['binary_costs_checked'] += len(box.costs)
            s, e = x.secret.ravel(), x.error.ravel()
            np.testing.assert_array_equal((A@s+e) % params.q, t)
            instances.append({'id': f'FRESH-{index:04d}', 'seed': x.seed, 'group': groups[index],
                'A_polynomials': x.A.tolist(), 'A_expanded': A.tolist(), 'public_target': t.tolist(),
                'planted_secret': s.tolist(), 'planted_error': e.tolist(), 'embedding_basis': B.tolist(),
                'reduced_basis': R.tolist(), 'basis_transform': U.tolist(), 'reference': reference})
            row = {'id': f'FRESH-{index:04d}', 'seed': x.seed, 'group': groups[index],
                'babai_cost': babai_cost, 'optimum_cost': optimum,
                'babai_secret_recovered': bool(np.array_equal(mapped_babai[len(t):], s)),
                'box_covers_optimum': bool(np.any(box.costs == optimum)), 'box_minimum': int(box.costs.min()),
                'optimal_box_bitstrings': int(np.count_nonzero(box.costs == optimum)),
                'planted_is_optimum': bool(int(s@s+e@e) == optimum), 'methods': {}, 'classical_refinement': {}}
            for changes in [1, 2]:
                refined = local_refine(R, y, box.center, changes, cfg['classical_refinement_evaluation_cap'])
                coord = U@np.array(refined['coefficients'])
                assert squared_cost(B, y, coord) == refined['cost'] <= babai_cost
                refined['planted_secret_recovered'] = bool(np.array_equal(coord[len(t):], s))
                row['classical_refinement'][f'refine_{changes}'] = refined
            distributions, compiled_circuits = {}, []
            for name, spec in cfg['methods'].items():
                sparse, removed = prune_interactions(box.J, spec['fraction'])
                probability = state_probabilities(diagonal_energy(box.h, sparse), spec['angles'])
                np.testing.assert_allclose(probability.sum(), 1, atol=1e-12)
                discrepancy = abs(float(probability@box.energy) - P1Energy(box.h, sparse, observable_h=box.h, observable_J=box.J)(spec['angles']))
                assert discrepancy < 1e-9
                checks['max_analytic_energy_discrepancy'] = max(checks['max_analytic_energy_discrepancy'], discrepancy)
                qc = circuit(box.h, sparse, spec['angles'])
                compiled = transpile(qc, basis_gates=cfg['compilation']['basis_gates'],
                    optimization_level=cfg['compilation']['optimization_level'], seed_transpiler=cfg['compilation']['seed'])
                row['methods'][name] = {'removed_edges': removed.tolist(), 'logical_rzz': int(qc.count_ops().get('rzz', 0)),
                    'compiled_cx': int(compiled.count_ops().get('cx', 0)), 'compiled_depth': compiled.depth(),
                    'compiled_two_qubit_depth': compiled.depth(lambda op: op.operation.num_qubits == 2),
                    'expected_original_cost': float(probability@box.costs)}
                compiled.save_probabilities(label='probabilities')
                compiled_circuits.append(compiled)
                distributions[name] = probability/probability.sum()
            result = simulator.run(compiled_circuits, shots=1).result()
            assert result.success
            for j, name in enumerate(cfg['methods']):
                discrepancy = float(np.max(np.abs(np.asarray(result.data(j)['probabilities']) - distributions[name])))
                assert discrepancy < 1e-10
                checks['max_probability_discrepancy'] = max(checks['max_probability_discrepancy'], discrepancy)
                checks['compiled_circuits_verified'] += 1
            distributions['uniform_box'] = np.full(len(box.costs), 1/len(box.costs))
            distributions['cbd_secret'] = secret_mass
            secret_coords, secret_costs = lift_secrets(A, t, params.q, secret_support)
            for method_index, name in enumerate(names):
                probability = distributions[name]
                costs = secret_costs if name == 'cbd_secret' else box.costs
                m = row['methods'].setdefault(name, {})
                p_opt, p_improve, metrics = distribution_metrics(costs, probability, babai_cost, optimum, cfg['budgets'])
                m.update(p_optimum=p_opt, p_improve=p_improve, by_budget=metrics)
                # Independent older implementation checks both the replication and large-shot endpoints.
                for shots in [cfg['primary_budget'], cfg['shots']]:
                    check = analyze_distribution(costs, probability, babai_cost, optimum, shots)
                    for key, original in [('hybrid_expected_cost', 'expected_cost'), ('hybrid_expected_distance', 'expected_distance'),
                                          ('hybrid_success', 'optimum_probability'), ('strict_improvement', 'improvement_probability')]:
                        np.testing.assert_allclose(metrics[str(shots)][key], check[original], atol=1e-9, rtol=0)
                    checks['order_statistic_checks'] += 1
                rng = np.random.default_rng(np.random.SeedSequence([cfg['measurement_seed'], index, method_index]))
                measured = rng.choice(len(probability), cfg['shots'], p=probability).astype(np.uint16)
                samples[name][index] = measured
                measured_costs = costs[measured]
                for shots in cfg['budgets']:
                    winner = int(measured_costs[:shots].argmin())
                    best = int(measured_costs[winner])
                    selected = int(measured[winner])
                    candidate = secret_coords[selected] if name == 'cbd_secret' else U@box.decode([selected])[0]
                    assert squared_cost(B, y, candidate) == best
                    returned = candidate if best < babai_cost else mapped_babai
                    assert squared_cost(B, y, returned) == min(best, babai_cost)
                    checks['returned_candidates_verified'] += 1
                    metrics[str(shots)].update(observed_best_cost=best, observed_hybrid_cost=min(best, babai_cost),
                        observed_successful_shots=int(np.count_nonzero(measured_costs[:shots] == optimum)),
                        observed_hybrid_secret_recovered=bool(np.array_equal(returned[len(t):], s)))
            rows.append(row)
            # Append progress rows, so a failed run retains completed work without overwriting any earlier run.
            with (run/'progress.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps({'completed': index+1, 'id': row['id'], 'recorded_at': now().isoformat()})+'\n')
            if (index+1) % 32 == 0:
                print(f'Validated and sampled {index+1}/{cfg["count"]}; elapsed {time.perf_counter()-timer:.1f}s', flush=True)
        save(run/'instances.json', instances)
        save(run/'per_instance.json', rows)
        np.savez_compressed(run/'samples.npz', **samples)
        print('Computing paired instance-bootstrap uncertainty...', flush=True)
        boot = np.random.default_rng(cfg['bootstrap_seed']).integers(0, len(rows), size=(cfg['bootstrap_replicates'], len(rows)))
        methods = {name: summary_for(rows, name, cfg['budgets'], boot) for name in names}
        base_distance = np.sqrt([r['babai_cost'] for r in rows])
        optimum_distance = np.sqrt([r['optimum_cost'] for r in rows])
        comparisons = {}
        for name in list(cfg['methods'])[1:]:
            comp = {'by_budget': {}, 'role': cfg['methods'][name]['role']}
            ratio = np.array([r['methods'][name]['p_optimum']/r['methods']['full_fixed']['p_optimum']
                              if r['methods']['full_fixed']['p_optimum'] > 0 else np.nan for r in rows])
            comp['single_shot_ratio_min_median_max'] = [float(np.nanmin(ratio)), float(np.nanmedian(ratio)), float(np.nanmax(ratio))]
            comp['single_shot_loss_over_5pct_count'] = int(np.count_nonzero(ratio < .95))
            comp['undefined_single_shot_ratios'] = int(np.count_nonzero(np.isnan(ratio)))
            comp['cx_reduction_fraction'] = 1-methods[name]['compiled_cx']['mean']/methods['full_fixed']['compiled_cx']['mean']
            for shots in cfg['budgets']:
                f = np.array([r['methods']['full_fixed']['by_budget'][str(shots)]['hybrid_expected_distance'] for r in rows])
                c = np.array([r['methods'][name]['by_budget'][str(shots)]['hybrid_expected_distance'] for r in rows])
                df = np.array([r['methods'][name]['by_budget'][str(shots)]['hybrid_success']-
                               r['methods']['full_fixed']['by_budget'][str(shots)]['hybrid_success'] for r in rows])
                retained = paired_retention(base_distance, f, c, boot)
                comp['by_budget'][str(shots)] = {'extra_distance_improvement_retention': retained,
                    'paired_hybrid_distance_change': mean_ci(c-f, boot), 'paired_hybrid_success_change': mean_ci(df, boot),
                    'distance_better_tied_worse_count': [int(np.sum(c-f < -1e-12)), int(np.sum(np.abs(c-f) <= 1e-12)), int(np.sum(c-f > 1e-12))]}
            primary = comp['by_budget'][str(cfg['primary_budget'])]['extra_distance_improvement_retention']
            comp['passes_prespecified_primary_retention_criterion'] = bool(primary['ci95'] is not None and primary['ci95'][0] > .95) if comp['role'] == 'primary' else None
            comparisons[name] = comp
        classical = {}
        for name in ['refine_1', 'refine_2']:
            values = [r['classical_refinement'][name] for r in rows]
            classical[name] = {'solved_count': sum(v['cost'] == r['optimum_cost'] for v, r in zip(values, rows)),
                'strict_improved_count': sum(v['cost'] < r['babai_cost'] for v, r in zip(values, rows)),
                'mean_distance': float(np.mean([np.sqrt(v['cost']) for v in values])),
                'secret_recovered_count': sum(v['planted_secret_recovered'] for v in values),
                'candidate_evaluations_min_median_max': [min(v['evaluations'] for v in values), float(np.median([v['evaluations'] for v in values])), max(v['evaluations'] for v in values)],
                'evaluation_cap_reached_count': sum(v['stopping_reason'] == 'evaluation_budget' for v in values)}
        summary = {'experiment_id': cfg['id'], 'run_id': run.name, 'count': len(rows), 'parameters': cfg['parameters'],
            'qubits': cfg['qubits'], 'depth_p': cfg['depth_p'], 'shots': cfg['shots'], 'primary_budget': cfg['primary_budget'],
            'cohort': '512 fresh independent public A groups, same tiny parameter distribution; frozen methods',
            'babai': {'solved_count': sum(r['babai_cost'] == r['optimum_cost'] for r in rows),
                'success': mean_ci([r['babai_cost'] == r['optimum_cost'] for r in rows], boot),
                'mean_distance': float(base_distance.mean()), 'secret_recovered_count': sum(r['babai_secret_recovered'] for r in rows)},
            'reference': {'mean_distance': float(optimum_distance.mean()),
                'secret_vectors_examined': sum(x['reference']['secret_vectors_examined'] for x in instances),
                'planted_is_optimum_count': sum(r['planted_is_optimum'] for r in rows)},
            'box_covers_optimum_count': sum(r['box_covers_optimum'] for r in rows),
            'box_misses_optimum_ids': [r['id'] for r in rows if not r['box_covers_optimum']],
            'methods': methods, 'comparisons_to_full': comparisons, 'classical_refinement': classical,
            'validation': checks, 'new_simulated_quantum_measurements': len(rows)*len(cfg['methods'])*cfg['shots'],
            'classical_candidate_samples': len(rows)*2*cfg['shots'], 'bootstrap_replicates': cfg['bootstrap_replicates'],
            'qpu_seconds': 0, 'timing_seconds': time.perf_counter()-timer, 'limitations': cfg['limitations']}
        save(run/'summary.json', summary)
        manifest.update(status='completed', finished_at=now().isoformat())
        manifest['output_sha256'] = {p.name: sha256(p.read_bytes()).hexdigest() for p in run.iterdir() if p.is_file() and p.name != 'manifest.json'}
        save(run/'manifest.json', manifest)
        print(json.dumps({'run': str(run), 'babai': summary['babai'], 'box_coverage': summary['box_covers_optimum_count'],
            'comparisons': comparisons, 'seconds': summary['timing_seconds']}, indent=2), flush=True)
    except Exception as error:
        manifest.update(status='failed', finished_at=now().isoformat(), error=repr(error))
        save(run/'manifest.json', manifest)
        raise


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
