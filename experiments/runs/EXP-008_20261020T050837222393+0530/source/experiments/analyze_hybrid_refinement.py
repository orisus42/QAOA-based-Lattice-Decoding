"""EXP-005: exact and observed Babai-incumbent refinement, without new shots."""
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
from threadpoolctl import threadpool_limits
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import state_probabilities
from lattice_qaoa.sparsify import prune_interactions, diagonal_energy, expected_best_cost
from lattice_qaoa.sampling import repeated_success, solved_count_interval


def now():
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def check_manifest(run):
    manifest = json.loads((run/'manifest.json').read_text())
    checked = 0
    for kind in ['source', 'input', 'output']:
        for rel, expected in manifest[kind+'_sha256'].items():
            path = run/rel if kind=='output' else run/kind/rel
            assert sha256(path.read_bytes()).hexdigest()==expected, str(path)
            checked += 1
    return checked


def analyze_distribution(costs, prob, incumbent, optimum, shots):
    # Independent order-statistic PMF; verify the existing survival-sum method.
    values, inverse = np.unique(np.minimum(costs, incumbent), return_inverse=True)
    masses = np.bincount(inverse, weights=prob, minlength=len(values))
    masses /= masses.sum()
    survival = np.clip(1-np.r_[0, np.cumsum(masses)], 0, 1)
    survival[-1] = 0
    pmf = survival[:-1]**shots-survival[1:]**shots
    assert np.isclose(pmf.sum(), 1) and pmf.min() >= -1e-12
    expected_cost = float(pmf@values)
    expected_distance = float(pmf@np.sqrt(values))
    np.testing.assert_allclose(expected_cost, expected_best_cost(costs, prob, shots, incumbent), atol=1e-10)
    np.testing.assert_allclose(expected_distance, expected_best_cost(np.sqrt(costs), prob, shots, np.sqrt(incumbent)), atol=1e-10)
    improvement = float(repeated_success(prob[costs<incumbent].sum(), shots))
    solved = 1. if incumbent==optimum else float(repeated_success(prob[costs==optimum].sum(), shots))
    np.testing.assert_allclose(improvement, pmf[values<incumbent].sum(), atol=1e-10)
    np.testing.assert_allclose(solved, pmf[values==optimum].sum(), atol=1e-10)
    assert np.sqrt(optimum)-1e-10 <= expected_distance <= np.sqrt(incumbent)+1e-10
    return {'expected_cost': expected_cost, 'expected_distance': expected_distance,
            'improvement_probability': improvement, 'optimum_probability': solved}


def aggregate(rows, method, shots, subset=None):
    selected = rows if subset is None else [r for r in rows if subset(r)]
    vals = [r['methods'][method]['by_budget'][str(shots)] for r in selected]
    babai = np.array([r['babai_distance'] for r in selected])
    optimum = np.array([r['optimum_distance'] for r in selected])
    distances = np.array([v['expected_distance'] for v in vals])
    improvement = [v['improvement_probability'] for v in vals]
    success = [v['optimum_probability'] for v in vals]
    out = {'count':len(selected), 'mean_expected_distance':float(distances.mean()),
           'mean_expected_squared_distance':float(np.mean([v['expected_cost'] for v in vals])),
           'distance_reduction_fraction':float(1-distances.mean()/babai.mean()),
           'available_distance_gap_closed_fraction':float((babai-distances).sum()/(babai-optimum).sum()),
           'expected_improved_count':float(np.sum(improvement)),
           'improved_count_prediction_interval_95pct':solved_count_interval(improvement),
           'expected_solved_count':float(np.sum(success)),
           'expected_solved_fraction':float(np.mean(success)),
           'solved_count_prediction_interval_95pct':solved_count_interval(success)}
    if all('observed_cost' in v for v in vals):
        costs = np.array([v['observed_cost'] for v in vals])
        base = np.array([r['babai_cost'] for r in selected])
        opt = np.array([r['optimum_cost'] for r in selected])
        out['observed']={'mean_distance':float(np.sqrt(costs).mean()), 'mean_squared_distance':float(costs.mean()),
            'improved_count':int((costs<base).sum()), 'tied_count':int((costs==base).sum()),
            'worse_count':int((costs>base).sum()), 'solved_count':int((costs==opt).sum()),
            'improved_ids':[r['id'] for r,c,b in zip(selected,costs,base) if c<b],
            'recovered_optimum_ids':[r['id'] for r,c,b,o in zip(selected,costs,base,opt) if c==o and b>o]}
    return out


def main():
    cfg_path=ROOT/'experiments/configs/EXP-005.json'
    cfg=json.loads(cfg_path.read_text())
    started=now(); clock=time.perf_counter()
    run=ROOT/'experiments/runs'/f"EXP-005_{started.strftime('%Y%m%dT%H%M%S%f%z')}"
    run.mkdir(exist_ok=False)
    save(run/'config.json', cfg)
    source=ROOT/cfg['input_run']; qrun=ROOT/cfg['qaoa_run']; crun=ROOT/cfg['classical_run']
    audited=check_manifest(qrun)
    sources=list((ROOT/'src/lattice_qaoa').glob('*.py'))+list((ROOT/'tests').glob('test_*.py'))+[
        Path(__file__), cfg_path, ROOT/'scripts/plot_hybrid_refinement.py', ROOT/'requirements-research.txt']
    inputs=[source/'instances.json',qrun/'per_instance.json',qrun/'summary.json',qrun/'samples.npz',
            qrun/'manifest.json',crun/'per_instance.json']
    manifest={'run_id':run.name,'status':'running','started_at':started.isoformat(),
        'command':'python experiments/analyze_hybrid_refinement.py','python':platform.python_version(),
        'packages':{k:version(k) for k in ['numpy','sympy','qiskit','threadpoolctl','matplotlib','seaborn']},
        'source_sha256':{},'input_sha256':{},'qpu_seconds':0,'new_shots':0}
    for kind, paths in [('source',sources),('input',inputs)]:
        for path in paths:
            rel=path.relative_to(ROOT); dest=run/kind/rel
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,dest)
            manifest[kind+'_sha256'][rel.as_posix()]=sha256(path.read_bytes()).hexdigest()
    save(run/'manifest.json',manifest)
    try:
        instances=json.loads((source/'instances.json').read_text())
        prior=json.loads((qrun/'per_instance.json').read_text())
        oldsummary=json.loads((qrun/'summary.json').read_text())
        classical=json.loads((crun/'per_instance.json').read_text())
        samples=np.load(qrun/'samples.npz')
        checks={'input_manifest_hashes':audited,'instances':0,'distributions_reproduced':0,
                'observed_bitstrings_rescored':0,'order_statistic_checks':0,'classical_candidates_verified':0}
        rows=[]
        for index,(x,q,c) in enumerate(zip(instances,prior,classical)):
            assert x['id']==q['id']==c['id']
            B=np.array(x['reduced_basis']); U=np.array(x['basis_transform'])
            y=np.r_[x['public_target'],np.zeros(4,dtype=int)]
            box=encode_babai_box(B,y)
            incumbent=x['lll_babai_cost']; optimum=x['reference']['cost']
            assert np.sum((B@box.center-y)**2)==incumbent
            assert box.costs.min()==optimum
            checks['instances']+=1
            row={'id':x['id'],'babai_cost':incumbent,'babai_distance':float(np.sqrt(incumbent)),
                 'optimum_cost':optimum,'optimum_distance':float(np.sqrt(optimum)),
                 'babai_optimal':incumbent==optimum,'methods':{}}
            for name in cfg['methods']+['uniform_box']:
                if name=='uniform_box':
                    prob=np.full(len(box.costs),1/len(box.costs)); observed=None
                else:
                    old=q['methods'][name]
                    sparse,removed=prune_interactions(box.J,old['fraction'])
                    assert removed.tolist()==old['removed_edges']
                    prob=state_probabilities(diagonal_energy(box.h,sparse),old['angles'])
                    np.testing.assert_allclose(prob[box.costs==optimum].sum(),old['optimum_probability'],atol=1e-12)
                    np.testing.assert_allclose(prob[box.costs<incumbent].sum(),old['refinement_probability'],atol=1e-12)
                    checks['distributions_reproduced']+=1
                    measured=samples[name][index]
                    assert len(measured)==cfg['shots']
                    observed=box.costs[measured]
                    checks['observed_bitstrings_rescored']+=len(measured)
                metrics={}
                for shots in cfg['budgets']:
                    values=analyze_distribution(box.costs,prob,incumbent,optimum,shots)
                    checks['order_statistic_checks']+=1
                    if observed is not None:
                        np.testing.assert_equal(observed[:shots].min(),old['observed'][str(shots)]['best_cost'])
                        values['observed_cost']=int(min(incumbent,observed[:shots].min()))
                        if shots==cfg['shots']:
                            best=int(observed.argmin())
                            coeff=box.decode([measured[best]])[0] if observed[best]<incumbent else box.center
                            z=U@coeff
                            residual=np.array(x['embedding_basis'])@z-y
                            assert int(residual@residual)==values['observed_cost']
                            values['observed_returned_coefficients_original_basis']=z.tolist()
                    metrics[str(shots)]=values
                row['methods'][name]={'by_budget':metrics}
            for name in cfg['classical_controls']:
                if name=='uniform_box':
                    continue
                saved=c['classical'][name]
                residual=B@np.array(saved['coefficients'])-y
                cost=int(residual@residual)
                assert cost==saved['cost'] and cost<=incumbent
                checks['classical_candidates_verified']+=1
                values={'expected_cost':float(cost),'expected_distance':float(np.sqrt(cost)),
                        'improvement_probability':float(cost<incumbent),'optimum_probability':float(cost==optimum),
                        'observed_cost':cost}
                row['methods'][name]={'by_budget':{str(cfg['shots']):values},
                                      'candidate_evaluations':saved['evaluations']}
            rows.append(row)
        assert len(rows)==64
        summary={'experiment_id':'EXP-005','run_id':run.name,'count':len(rows),'shots':cfg['shots'],
            'analysis_type':cfg['analysis_type'],'distance_definition':cfg['distance'],
            'babai':{'solved_count':sum(r['babai_optimal'] for r in rows),
                     'mean_distance':float(np.mean([r['babai_distance'] for r in rows])),
                     'mean_squared_distance':float(np.mean([r['babai_cost'] for r in rows]))},
            'reference':{'mean_optimum_distance':float(np.mean([r['optimum_distance'] for r in rows])),
                         'mean_optimum_squared_distance':float(np.mean([r['optimum_cost'] for r in rows]))},
            'methods':{},'validation':checks,'new_shots':0,'saved_measurements_reused':checks['observed_bitstrings_rescored'],
            'qpu_seconds':0,'limitations':cfg['limitations'],'timing_seconds':time.perf_counter()-clock}
        for name in cfg['methods']+cfg['classical_controls']:
            summary['methods'][name]={'all64':aggregate(rows,name,cfg['shots']),
                'babai_failures_only':aggregate(rows,name,cfg['shots'],lambda r:not r['babai_optimal'])}
            if name in oldsummary['methods']:
                result=summary['methods'][name]['all64']; old=oldsummary['methods'][name]
                np.testing.assert_allclose(result['mean_expected_squared_distance'],old['mean_expected_hybrid_best_cost_1024'],atol=1e-10)
                np.testing.assert_allclose(result['expected_solved_fraction'],old['expected_hybrid_solved_fraction_1024'],atol=1e-12)
                assert result['observed']['solved_count']==old['observed_hybrid_solved_count_1024']
                summary['methods'][name]['compiled_cx_mean']=old['compiled_cx']['mean']
                summary['methods'][name]['by_budget']={str(n):aggregate(rows,name,n) for n in cfg['budgets']}
        save(run/'per_instance.json',rows);save(run/'summary.json',summary)
        manifest.update(status='completed',finished_at=now().isoformat())
        manifest['output_sha256']={name:sha256((run/name).read_bytes()).hexdigest() for name in ['config.json','summary.json','per_instance.json']}
        save(run/'manifest.json',manifest)
        print('RUN:',run,flush=True)
        print('BABAI:',summary['babai'],'OPTIMUM:',summary['reference'])
        for name,m in summary['methods'].items():
            a=m['all64'];obs=a.get('observed',{})
            print(name,'distance',round(a['mean_expected_distance'],6),'improved',round(a['expected_improved_count'],4),
                  'solved',round(a['expected_solved_count'],4),'observed improved/solved',obs.get('improved_count'),obs.get('solved_count'))
    except Exception as exc:
        manifest.update(status='failed',finished_at=now().isoformat(),error=repr(exc))
        save(run/'manifest.json',manifest)
        raise


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
