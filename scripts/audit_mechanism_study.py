"""Independent rechecks of saved source hashes, algebra, masks and metrics."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json,sys,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.reference import embedding,squared_cost
from lattice_qaoa.instances import module_product,expand_module_matrix
from lattice_qaoa.qaoa import circuit,P1Energy
from lattice_qaoa.sparsify import prune_interactions
from lattice_qaoa.evaluation import distribution_metrics,matrix_group
from lattice_qaoa.device import hybrid_metrics_from_low_masses


def check_hashes(run):
    m=json.loads((run/'manifest.json').read_text());count=0
    for kind in ['source','input','output']:
        for path,digest in m[kind+'_sha256'].items():
            p=run/path if kind=='output' else run/kind/path
            assert sha256(p.read_bytes()).hexdigest()==digest;count+=1
    return count


def main():
    runs=[Path(s).resolve() for s in sys.argv[1:]]
    out=ROOT/'experiments/runs'/('AUDIT-014_'+datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%f%z'))
    out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'audit_source.py')
    report={'recorded_at':datetime.now().astimezone().isoformat(),'runs':{},'qpu_seconds':0}
    for run in runs:
        cfg=json.loads((run/'config.json').read_text());exp=cfg['experiment_id'];checks={'immutable_hashes':check_hashes(run)}
        if exp=='EXP-010':
            examples=json.loads((run/'nist_coefficient_examples.json').read_text())
            for e in examples:
                A=np.array(e['A_polynomials']);s=np.array(e['secret']);error=np.array(e['error']);t=np.array(e['public_target']);w=np.array(e['wrap'])
                np.testing.assert_array_equal((module_product(A,s)+error)%3329,t)
                np.testing.assert_array_equal(3329*w+module_product(A,s)-t,-error)
            checks['full_size_coefficient_examples_rechecked']=len(examples)
            rows=json.loads((run/'per_instance.json').read_text());cases=json.loads((run/'instances.json').read_text())
            old=json.loads((ROOT/'experiments/runs/EXP-001_20261018T120321801881+0530/instances.json').read_text())
            oldgroups={matrix_group(x['A_expanded'],17) for x in old}
            training=json.loads((ROOT/'experiments/runs/EXP-003_20261020T014441792732+0530/training.json').read_text())
            oldgroups|={matrix_group(expand_module_matrix(np.array(x['A_polynomials'])),17) for x in training['instances']}
            checked=0
            for i,(r,x) in enumerate(zip(rows,cases)):
                A=expand_module_matrix(np.array(x['A_polynomials']));t=np.array(x['public_target']);q=r['parameters']['q'];B,y=embedding(A,t,q)
                U=np.array(x['basis_transform']);R=np.array(x['reduced_basis']);np.testing.assert_array_equal(B@U,R)
                assert r['parameters']['q']!=17 or r['parameters']['n']!=2 or matrix_group(A,q) not in oldgroups
                for v in x['reference']['minimizers']:assert squared_cost(B,y,np.array(v))==r['optimum_cost']
                if i%16==0:
                    box=encode_babai_box(R,y)
                    for fraction in [0,.75,.9]:
                        J,_=prune_interactions(box.J,fraction);p=Statevector.from_instruction(circuit(box.h,J,cfg['angles'])).probabilities()
                        m=distribution_metrics(box.costs,p,r['babai_cost'],r['optimum_cost'],cfg['budgets'])[2]
                        for b in cfg['budgets']:
                            for k in ['hybrid_success','hybrid_expected_cost','hybrid_expected_distance']:
                                np.testing.assert_allclose(m[str(b)][k],r['methods'][str(fraction)]['by_budget'][str(b)][k],atol=1e-10)
                        checked+=1
            checks['parameter_cases_algebra_checked']=len(cases);checks['independent_statevectors']=checked
        elif exp=='EXP-011':
            rows=[json.loads(x) for x in (run/'sweep_all_counts.jsonl').read_text().splitlines()];grid=json.loads((run/'grid.json').read_text())
            source=json.loads((ROOT/cfg['development_run']/'instances.json').read_text())
            checked=0
            for i in range(64):
                selected=[r for r in rows if r['index']==i];M=selected[0]['nonzero_edges'];assert [r['removed_count'] for r in selected]==list(range(M+1))
                for pct in cfg['trained_percentages']:
                    same=next(r for r in grid if r['index']==i and r['policy']=='same_angles' and r['percent']==pct)
                    trained=next(r for r in grid if r['index']==i and r['policy']=='trained_angles' and r['percent']==pct)
                    assert same['logical_rzz']==trained['logical_rzz']
                if i%8==0:
                    x=source[i];box=encode_babai_box(np.array(x['reduced_basis']),np.r_[x['public_target'],np.zeros(4,dtype=int)])
                    for pct in [0,75,81,100]:
                        J,_=prune_interactions(box.J,pct/100);p=Statevector.from_instruction(circuit(box.h,J,cfg['angles'])).probabilities()
                        m=distribution_metrics(box.costs,p,x['lll_babai_cost'],x['reference']['cost'],cfg['budgets'])[2]
                        row=selected[M-int(np.count_nonzero(J))]
                        for b in cfg['budgets']:
                            for k in ['hybrid_success','hybrid_expected_cost','hybrid_expected_distance']:
                                np.testing.assert_allclose(m[str(b)][k],row['by_budget'][str(b)][k],atol=1e-10)
                        checked+=1
            checks['distinct_masks_checked']=len(rows);checks['independent_statevectors']=checked
            training=json.loads((ROOT/cfg['training_run']/'training.json').read_text());fits=json.loads((run/'training.json').read_text())
            for percent,fit in fits.items():
                values=[]
                for x in training['instances']:
                    h=np.array(x['h']);j=np.array(x['J']);a=fit['angles']
                    if percent=='split_full':phaseh=a[0]*h;phasej=a[1]*j;ang=[1,a[2]]
                    else:phaseh=h;phasej=prune_interactions(j,int(percent)/100)[0];ang=a
                    values.append(P1Energy(phaseh,phasej,observable_h=h,observable_J=j)(ang))
                np.testing.assert_allclose(np.mean(values),fit['training_mean_original_energy'],atol=1e-12)
            checks['analytic_training_fits_rechecked']=len(fits)
        elif exp=='EXP-012':
            rows=json.loads((run/'per_instance.json').read_text());summary=json.loads((run/'summary.json').read_text())
            for row in rows:
                a=np.load(run/'noise'/f"{row['id']}_{row['method']}.npz")
                for channel in cfg['channels']:
                    raw=a[channel];assert np.all(raw>=0) and np.all(raw.sum(axis=1)<=1+1e-10) and np.all(raw[:,:row['optimum_cost']]<1e-12)
                    for b in cfg['budgets']:
                        # Older distribution_metrics: add one overflow cost.
                        mass=raw.mean(axis=0);p=np.r_[mass,1-mass.sum()];costs=np.arange(len(p))
                        check=distribution_metrics(costs,p,row['babai_cost'],row['optimum_cost'],[b])[2][str(b)]
                        for k in ['hybrid_success','hybrid_expected_cost','hybrid_expected_distance']:
                            np.testing.assert_allclose(check[k],row['channels'][channel][str(b)][k],atol=1e-9)
            for name in cfg['methods']:
                for channel in cfg['channels']:
                    for b in cfg['budgets']:
                        vals=[r['channels'][channel][str(b)]['hybrid_success'] for r in rows if r['method']==name]
                        expected=(64-len(vals)+sum(vals))/64
                        np.testing.assert_allclose(expected,summary['methods'][name][channel][str(b)]['hybrid_success'],atol=1e-12)
            checks['noise_method_rows_rechecked']=len(rows);checks['independent_minimum_statistic_checks']=len(rows)*len(cfg['channels'])*len(cfg['budgets'])
        report['runs'][str(run.relative_to(ROOT))]=checks
        print(exp,checks,flush=True)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    files={p.name:sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'manifest.json').write_text(json.dumps({'recorded_at':datetime.now().astimezone().isoformat(),'status':'completed','files_sha256':files},indent=2)+'\n')
    print('AUDIT:',out,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
