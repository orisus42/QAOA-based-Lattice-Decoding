"""Audit saved fresh-cohort outputs and rescore every stored sample independently."""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'experiments'))
from lattice_qaoa.instances import Parameters, generate_instance, expand_module_matrix
from lattice_qaoa.reference import babai
from lattice_qaoa.evaluation import matrix_group
from lattice_qaoa.sampling import repeated_success
from analyze_hybrid_refinement import check_manifest


def main():
    run = Path(sys.argv[1]).resolve()
    audit = {'recorded_at': datetime.now().astimezone().isoformat(), 'manifest_hashes_verified': check_manifest(run),
             'fresh_seeds_reproduced': 0, 'sampled_candidates_rescored': 0, 'observed_budget_results_verified': 0,
             'method_budget_aggregates_verified': 0}
    cfg = json.loads((run/'config.json').read_text())
    s = json.loads((run/'summary.json').read_text())
    rows = json.loads((run/'per_instance.json').read_text())
    instances = json.loads((run/'instances.json').read_text())
    cohort = json.loads((run/'cohort.json').read_text())
    codebook = json.loads((run/'sample_codebook.json').read_text())
    manifest = json.loads((run/'manifest.json').read_text())
    decision = next(json.loads(line) for line in (ROOT/'project/decisions.jsonl').read_text().splitlines() if json.loads(line)['id'] == 'D-018')
    frozen_config = run/'source/experiments/configs/EXP-007.json'
    assert sha256(frozen_config.read_bytes()).hexdigest() == decision['config_sha256']
    assert json.loads(frozen_config.read_text()) == cfg
    assert datetime.fromisoformat(decision['recorded_at']) < datetime.fromisoformat(manifest['started_at'])
    assert len(rows) == len(instances) == cfg['count'] == 512
    assert len(set(cohort['groups'])) == 512 and not (set(cohort['groups']) & set(cohort['excluded_groups']))
    stored = np.load(run/'samples.npz')
    support = np.array(codebook['secret_support'])
    names = list(s['methods'])
    for name in names:
        assert stored[name].shape == (512, cfg['shots']) and stored[name].dtype == np.uint16
    # Load each compressed array once; accessing the NPZ key repeatedly would decompress repeatedly.
    samples = {name: stored[name] for name in names}
    for i, (x, row) in enumerate(zip(instances, rows)):
        fresh = generate_instance(cfg['first_seed']+i, Parameters(**cfg['parameters']))
        np.testing.assert_array_equal(fresh.A, x['A_polynomials'])
        np.testing.assert_array_equal(fresh.secret.ravel(), x['planted_secret'])
        np.testing.assert_array_equal(fresh.target.ravel(), x['public_target'])
        assert matrix_group(expand_module_matrix(fresh.A), cfg['parameters']['q']) == row['group'] == cohort['groups'][i]
        audit['fresh_seeds_reproduced'] += 1
        R, U, B = [np.array(x[k]) for k in ['reduced_basis', 'basis_transform', 'embedding_basis']]
        np.testing.assert_array_equal(B@U, R)
        t = np.array(x['public_target']); y = np.r_[t, np.zeros(len(t), dtype=int)]
        center = babai(R, y)
        assert int(np.sum((R@center-y)**2)) == row['babai_cost']
        for name in names:
            measured = samples[name][i].astype(np.int64)
            if name == 'cbd_secret':
                secret = support[measured]
                A = np.array(x['A_expanded']); q = cfg['parameters']['q']
                wraps = (t-secret@A.T+q//2)//q
                original_coords = np.column_stack([wraps, secret])
                residual = original_coords@B.T-y
            else:
                # Independent two-bit decoding, without BinaryBox.decode or its enumerated costs.
                coords = center-2+((measured[:, None] >> (2*np.arange(len(center)))) & 3)
                original_coords = coords@U.T
                residual = original_coords@B.T-y
            costs = np.sum(residual*residual, axis=1)
            audit['sampled_candidates_rescored'] += len(costs)
            assert np.all(costs >= row['optimum_cost'])
            m = row['methods'][name]
            for shots in cfg['budgets']:
                v = m['by_budget'][str(shots)]
                winner = int(costs[:shots].argmin())
                best = int(costs[winner]); hybrid = min(row['babai_cost'], best)
                assert best == v['observed_best_cost'] and hybrid == v['observed_hybrid_cost']
                assert int(np.count_nonzero(costs[:shots] == row['optimum_cost'])) == v['observed_successful_shots']
                returned = original_coords[winner] if best < row['babai_cost'] else U@center
                assert bool(np.array_equal(returned[len(t):], x['planted_secret'])) == v['observed_hybrid_secret_recovered']
                np.testing.assert_allclose(v['standalone_success'], repeated_success(m['p_optimum'], shots), atol=1e-12)
                expected = 1. if row['babai_cost'] == row['optimum_cost'] else v['standalone_success']
                np.testing.assert_allclose(v['hybrid_success'], expected, atol=1e-12)
                audit['observed_budget_results_verified'] += 1
        if (i+1) % 128 == 0:
            print(f'Independently rescored samples for {i+1}/512 instances', flush=True)
    for name in names:
        for shots in cfg['budgets']:
            for key in ['standalone_success', 'hybrid_success', 'strict_improvement', 'hybrid_expected_cost', 'hybrid_expected_distance', 'standalone_expected_distance']:
                expected = np.mean([r['methods'][name]['by_budget'][str(shots)][key] for r in rows])
                np.testing.assert_allclose(s['methods'][name]['by_budget'][str(shots)][key]['mean'], expected, atol=1e-12)
            audit['method_budget_aggregates_verified'] += 1
        if name in cfg['methods']:
            expected = np.mean([r['methods'][name]['compiled_cx'] for r in rows])
            np.testing.assert_allclose(expected, s['methods'][name]['compiled_cx']['mean'])
    out = run/'audit'
    out.mkdir(exist_ok=False)
    shutil.copy2(__file__, out/'audit_source.py')
    (out/'results.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    for path in out.iterdir():
        manifest['output_sha256'][path.relative_to(run).as_posix()] = sha256(path.read_bytes()).hexdigest()
    (run/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
