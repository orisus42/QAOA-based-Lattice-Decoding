"""Independent completed-run checks; writes a new immutable audit folder."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from qiskit import qpy
from threadpoolctl import threadpool_limits
from lattice_qaoa.residual_pruning import public_phase,select_masks,mask_from_edges,edge_list
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import state_probabilities,P1Energy
from lattice_qaoa.sparsify import diagonal_energy
from lattice_qaoa.evaluation import matrix_group


def save(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')


def main():
    run=Path(sys.argv[1]).resolve();manifest=json.loads((run/'manifest.json').read_text());assert manifest['status']=='completed'
    out=ROOT/'experiments/runs'/('AUDIT-018_'+datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%f%z'));out.mkdir()
    checks={'run':str(run.relative_to(ROOT)),'started_at':datetime.now().astimezone().isoformat(),'hashes':0,'analytic_energy_checks':0,'statevector_metric_checks':0,
        'saved_sample_checks':0,'logical_gate_budgets':0,'routed_circuits_reloaded':0,'native_edges_verified':0,'selection_replays':0,'max_statevector_metric_error':0.}
    try:
        for kind in ['source','input','output']:
            for rel,digest in manifest[kind+'_sha256'].items():
                path=run/(kind if kind!='output' else '')/rel
                assert sha256(path.read_bytes()).hexdigest()==digest,(kind,rel)
                checks['hashes']+=1
        rows=json.loads((run/'per_instance.json').read_text());instances=json.loads((run/'instances.json').read_text());cfg=json.loads((run/'config.json').read_text());summary=json.loads((run/'summary.json').read_text())
        groups=json.loads((run/'cohort.json').read_text());assert len(groups['fresh_groups'])==128
        assert len(set(groups['fresh_groups']))==128 and not set(groups['fresh_groups'])&set(groups['excluded_groups'])
        assert len(rows)==len(instances)==192
        snap=json.loads((ROOT/cfg['calibration']).read_text());allowed={tuple(g['qubits']) for g in snap['gates'] if g['name']=='cz'}
        selected={0,31,64,95,127,191}
        for i,(r,x) in enumerate(zip(rows,instances)):
            assert r['id']==x['id'];opt=r['optimum'];inc=r['babai_cost']
            for name,m in r['methods'].items():
                if name.startswith(('magnitude','residual','geometry')):
                    k=int(__import__('re').search(r'(\d+)',name).group());assert m['logical_rzz']==k==len(m['edges']);checks['logical_gate_budgets']+=1
                for n in cfg['budgets']:
                    prob=m['p_optimum'];expected=1. if inc==opt else -np.expm1(n*np.log1p(-prob)) if prob<1 else 1.
                    assert abs(expected-m['by_budget'][str(n)]['hybrid_success'])<1e-10
            with (run/'circuits'/f'{r["id"]}.qpy').open('rb') as f:circuits=qpy.load(f)
            for qc,name in zip(circuits,['full_trained','magnitude24_trained','residual24_trained','geometry24_trained']):
                assert qc.count_ops().get('cz',0)==r['methods'][name]['native_cz'];checks['routed_circuits_reloaded']+=1
                for op in qc.data:
                    if op.operation.name=='cz':
                        edge=tuple(qc.find_bit(q).index for q in op.qubits);assert edge in allowed;checks['native_edges_verified']+=1
            if i not in selected:continue
            R=np.array(x['reduced_basis']);U=np.array(x['basis_transform']);y=np.r_[x['public_target'],np.zeros(4,dtype=int)];box=encode_babai_box(R,y)
            h,j,_,_=public_phase(R,y,box.center);samples=np.load(run/'samples'/f'{r["id"]}.npz')
            for name,m in r['methods'].items():
                if 'edges' not in m:continue
                phase=mask_from_edges(j,m['edges']);p=state_probabilities(diagonal_energy(h,phase),m['angles'])
                delta=abs(float(p[box.costs==opt].sum())-m['p_optimum']);assert delta<1e-12
                checks['max_statevector_metric_error']=max(checks['max_statevector_metric_error'],delta);checks['statevector_metric_checks']+=1
                np.testing.assert_allclose(float(p@box.energy),m['expected_original_energy'],atol=1e-12)
                np.testing.assert_allclose(P1Energy(h,phase,observable_J=j)(m['angles']),m['expected_original_energy'],atol=1e-12);checks['analytic_energy_checks']+=1
                for n in cfg['budgets']:
                    assert min(inc,int(box.costs[samples[name][:n]].min()))==m['observed'][str(n)]['hybrid_cost'];checks['saved_sample_checks']+=1
            if i in [64,191]:
                replay,_=select_masks(h,j,cfg['gate_budgets'],cfg['selection_angles'])
                for k in cfg['gate_budgets']:
                    assert edge_list(replay[k]['J'])==r['methods'][f'residual{k}_same']['edges'];checks['selection_replays']+=1
        # Recompute the prespecified primary paired bootstrap independently.
        fresh=[r for r in rows if r['cohort']=='fresh'];diff=np.array([r['methods']['residual24_trained']['by_budget']['1024']['hybrid_success']-r['methods']['magnitude24_trained']['by_budget']['1024']['hybrid_success'] for r in fresh])
        idx=np.random.default_rng(cfg['bootstrap_seed']+1).integers(0,128,(cfg['bootstrap_replicates'],128));interval=np.quantile(diff[idx].mean(axis=1),[.025,.975])
        reported=summary['cohorts']['fresh']['paired']['residual24_trained_minus_magnitude24_trained']['1024']['hybrid_success']
        np.testing.assert_allclose(interval,reported['ci95'],atol=1e-14);assert abs(diff.mean()-reported['mean'])<1e-14
        checks['primary_bootstrap_reproduced']=True;checks['completed_at']=datetime.now().astimezone().isoformat();checks['status']='passed'
        save(out/'validation.json',checks)
        save(out/'manifest.json',{'script_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'input_manifest_sha256':sha256((run/'manifest.json').read_bytes()).hexdigest(),'validation_sha256':sha256((out/'validation.json').read_bytes()).hexdigest()})
        (out/'audit_script.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps(checks,indent=2));print('AUDIT:',out)
    except Exception as e:
        checks.update(status='failed',error=repr(e));save(out/'validation.json',checks);raise


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
