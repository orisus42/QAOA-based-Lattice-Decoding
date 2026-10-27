"""Independent artifact, probability, circuit and paired-statistic checks."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json
import sys
import shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from qiskit import qpy
from lattice_qaoa.instances import Parameters,generate_instance,expand_module_matrix
from lattice_qaoa.reference import embedding,squared_cost
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import state_probabilities,P1Energy
from lattice_qaoa.sparsify import diagonal_energy
from lattice_qaoa.residual_pruning import mask_from_edges,select_masks
from lattice_qaoa.evaluation import matrix_group


def audit(run):
    cfg=json.loads((run/'config.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text());assert manifest['status']=='completed'
    summary=json.loads((run/'summary.json').read_text())
    rows=json.loads((run/'per_instance.json').read_text());instances=json.loads((run/'instances.json').read_text())
    cohort=json.loads((run/'cohort.json').read_text())
    checks={'source_run':run.relative_to(ROOT).as_posix(),'hashes':0,'public_instances':0,'probabilities':0,
            'shot_formulas':0,'routed_circuits':0,'selected_masks':0,'reference_replays':0,'bootstrap_intervals':0}
    for kind in ['source','input','output']:
        for rel,digest in manifest[kind+'_sha256'].items():
            path=run/(kind if kind!='output' else '')/rel
            assert sha256(path.read_bytes()).hexdigest()==digest,(kind,rel)
            checks['hashes']+=1
    assert len(rows)==len(instances)==cfg['fresh_count']
    assert not set(cohort['fresh_groups'])&set(cohort['excluded_groups'])
    assert len(set(cohort['fresh_groups']))==len(rows)
    p=Parameters(**cfg['parameters']);k=cfg['logical_budget']
    frozen=json.loads((run/('frozen_models.json' if cfg['experiment_id']=='EXP-021' else 'frozen_protocol.json')).read_text())
    assert frozen['config_sha256']==sha256((run/'config.json').read_bytes()).hexdigest()
    replay_indices=set(np.linspace(0,len(rows)-1,6,dtype=int))
    for i,(r,x) in enumerate(zip(rows,instances)):
        assert r['id']==x['id'] and x['seed']==cfg['fresh_first_seed']+i
        generated=generate_instance(x['seed'],p)
        np.testing.assert_array_equal(generated.A,x['A_polynomials'])
        np.testing.assert_array_equal(generated.target.ravel(),x['public_target'])
        A=expand_module_matrix(generated.A);t=np.array(x['public_target'])
        assert matrix_group(A,p.q)==r['group']==cohort['fresh_groups'][i]
        np.testing.assert_array_equal((A@np.array(x['planted_secret'])+np.array(x['planted_error']))%p.q,t)
        B,y=embedding(A,t,p.q);R=np.array(x['reduced_basis']);U=np.array(x['basis_transform'])
        np.testing.assert_array_equal(B@U,R);checks['public_instances']+=1
        box=encode_babai_box(R,y);h,j=box.h,box.J
        assert squared_cost(R,y,box.center)==r['babai_cost']
        assert min(squared_cost(B,y,np.array(z)) for z in x['reference']['minimizers'])==r['optimum']==x['reference']['cost']
        s=json.loads((run/'selection'/f'{r["id"]}.json').read_text())
        if cfg['experiment_id']=='EXP-021':
            edges=s['edges']
            # Reconstruct both cheap rules independently from sorting definitions.
            all_edges=list(zip(*np.nonzero(j)));b=np.tile([1,-1],len(h)//2)
            incumbent=sorted(all_edges,key=lambda e:(j[e]*b[e[0]]*b[e[1]],*e))[:k]
            magnitude=sorted(all_edges,key=lambda e:(abs(j[e]),*e))[len(all_edges)-k:]
            assert set(map(tuple,edges['incumbent']))==set(incumbent)
            assert set(map(tuple,edges['magnitude']))==set(magnitude)
            assert edges['residual']==edges['residual_same'];assert edges['incumbent']==edges['incumbent_same']
            if i in replay_indices:
                chosen,_=select_masks(h,j,[k],cfg['selection_angles'])
                np.testing.assert_array_equal(mask_from_edges(j,edges['residual']),chosen[k]['J']);checks['reference_replays']+=1
            for name,elist in edges.items():
                phase=mask_from_edges(j,elist);m=r['methods'][name]
                assert m['angles']==frozen['angles'][name]
                assert np.count_nonzero(phase)==(np.count_nonzero(j) if name=='full' else k)
                checks['selected_masks']+=1
                prob=state_probabilities(diagonal_energy(h,phase),m['angles']);checks['probabilities']+=1
                np.testing.assert_allclose(prob@box.energy,P1Energy(h,phase,observable_J=j)(m['angles']),atol=1e-12)
                po=prob[box.costs==r['optimum']].sum()
                np.testing.assert_allclose(po,m['p_optimum'],atol=1e-13)
                for shots in cfg['budgets']:
                    expected=1. if r['babai_cost']==r['optimum'] else -np.expm1(shots*np.log1p(-po))
                    np.testing.assert_allclose(expected,m['by_budget'][str(shots)]['hybrid_success'],atol=1e-12)
                    checks['shot_formulas']+=1
            with (run/'circuits'/f'{r["id"]}.qpy').open('rb') as f:circuits=qpy.load(f)
            assert len(circuits)==4
            for name,qc in zip(['full','magnitude','residual','incumbent'],circuits):
                assert qc.count_ops().get('cz',0)==r['methods'][name]['native_cz']
                checks['routed_circuits']+=1
            samples=np.load(run/'samples'/f'{r["id"]}.npz')
            for name in edges:
                for shots in cfg['budgets']:
                    assert min(r['babai_cost'],int(box.costs[samples[name][:shots]].min()))==r['methods'][name]['observed'][str(shots)]['hybrid_cost']
        else:
            assert s['original_edges']==s['cached_edges'] and s['original_fallback']==s['cached_fallback']
            assert [z['removed_edge'] for z in s['original_trace']]==[z['removed_edge'] for z in s['cached_trace']]
            phase=mask_from_edges(j,s['cached_edges']);assert np.count_nonzero(phase)==k
            checks['selected_masks']+=2
            if i in replay_indices:
                chosen,trace=select_masks(h,j,[k],cfg['selection_angles'])
                np.testing.assert_array_equal(chosen[k]['J'],phase)
                assert [z['removed_edge'] for z in trace]==[z['removed_edge'] for z in s['original_trace']]
                checks['reference_replays']+=1
            prob=state_probabilities(diagonal_energy(h,phase),frozen['angles']);checks['probabilities']+=1
            po=prob[box.costs==r['optimum']].sum()
            np.testing.assert_allclose(po,r['quality']['p_optimum'],atol=1e-13)
            assert r['max_probability_difference']==0 and r['identical_path'] and r['identical_mask']
            for shots in cfg['budgets']:
                expected=1. if r['babai_cost']==r['optimum'] else -np.expm1(shots*np.log1p(-po))
                np.testing.assert_allclose(expected,r['quality']['by_budget'][str(shots)]['hybrid_success'],atol=1e-12)
                checks['shot_formulas']+=1
            expected_reps=cfg['timing_repetitions'] if i<cfg['timing_count'] else 1
            assert r['timing_repetitions']==expected_reps
            for name,values in r['timing_seconds'].items():
                assert len(values)==expected_reps and min(values)>0
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    idx=rng.integers(0,len(rows),(cfg['bootstrap_replicates'],len(rows)))
    if cfg['experiment_id']=='EXP-021':
        for pair,reported in summary['paired'].items():
            a,b=pair.split('_minus_')
            for shots in cfg['budgets']:
                dif=np.array([r['methods'][a]['by_budget'][str(shots)]['hybrid_success']-r['methods'][b]['by_budget'][str(shots)]['hybrid_success'] for r in rows])
                expected=np.quantile(dif[idx].mean(axis=1),[.025,.975])
                np.testing.assert_allclose(expected,reported[str(shots)]['hybrid_success']['ci95'],atol=1e-14)
                checks['bootstrap_intervals']+=1
    else:
        timing=rows[:cfg['timing_count']]
        bt=rng.integers(0,len(timing),(cfg['bootstrap_replicates'],len(timing)))
        old=np.array([np.median(r['timing_seconds']['original']) for r in timing])
        new=np.array([np.median(r['timing_seconds']['cached']) for r in timing])
        ratio=old[bt].mean(axis=1)/new[bt].mean(axis=1)
        reported=summary['speedup_ratio_of_mean_medians']
        np.testing.assert_allclose(np.quantile(ratio,[.025,.975]),reported['ci95'],atol=1e-14)
        np.testing.assert_allclose(old.mean()/new.mean(),reported['ratio'],atol=1e-14);checks['bootstrap_intervals']+=1
    checks.update(status='passed',completed_at=datetime.now().astimezone().isoformat())
    out=ROOT/'experiments/runs'/('AUDIT-024_'+datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%f%z'));out.mkdir()
    (out/'validation.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    shutil.copy2(__file__,out/'audit_script.py')
    sources={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/lattice_qaoa').glob('*.py')}
    (out/'manifest.json').write_text(json.dumps({'created_at':checks['completed_at'],'source_sha256':sources,
         'script_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
         'input_manifest_sha256':sha256((run/'manifest.json').read_bytes()).hexdigest(),
         'validation_sha256':sha256((out/'validation.json').read_bytes()).hexdigest()},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks,indent=2));print('AUDIT:',out)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        for argument in sys.argv[1:]:audit(Path(argument).resolve())
