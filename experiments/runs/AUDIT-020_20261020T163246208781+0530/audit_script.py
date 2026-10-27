"""Independent development-control audit; keeps its own immutable record."""
from datetime import datetime
from pathlib import Path
from hashlib import sha256
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from qiskit import qpy
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import state_probabilities
from lattice_qaoa.sparsify import diagonal_energy


def main():
    run=Path(sys.argv[1]).resolve();manifest=json.loads((run/'manifest.json').read_text());assert manifest['status']=='completed'
    cfg=json.loads((run/'config.json').read_text());summary=json.loads((run/'summary.json').read_text())
    rows=json.loads((run/'per_instance.json').read_text())
    dev=json.loads((run/'input'/cfg['development_run']/'instances.json').read_text())
    old=json.loads((run/'input'/cfg['comparison_checkpoint']/'per_instance.json').read_text());old={r['id']:r for r in old}
    checks={'hashes':0,'phase_masks':0,'probabilities':0,'routed_circuits':0,'bootstrap_intervals':0,'source_run':str(run.relative_to(ROOT))}
    for kind in ['source','input','output']:
        for rel,digest in manifest[kind+'_sha256'].items():
            path=run/(kind if kind!='output' else '')/rel
            assert sha256(path.read_bytes()).hexdigest()==digest
            checks['hashes']+=1
    differences=[]
    for r,x in zip(rows,dev):
        assert r['id']==x['id'];R=np.array(x['reduced_basis']);y=np.r_[x['public_target'],np.zeros(4,dtype=int)]
        box=encode_babai_box(R,y);b=np.tile([1,-1],8)
        edges=list(zip(*np.nonzero(box.J)))
        expected=sorted(edges,key=lambda e:(box.J[e]*b[e[0]]*b[e[1]],*e))[:24]
        assert set(map(tuple,r['edges']))==set(expected);checks['phase_masks']+=1
        phase=np.zeros_like(box.J)
        for edge in expected:phase[edge]=box.J[edge]
        for name,m in r['methods'].items():
            prob=state_probabilities(diagonal_energy(box.h,phase),m['angles'])
            po=float(prob[box.costs==x['reference']['cost']].sum())
            assert abs(po-m['p_optimum'])<1e-13;checks['probabilities']+=1
        with (run/'circuits'/f'{r["id"]}.qpy').open('rb') as f:circuits=qpy.load(f)
        assert len(circuits)==1
        assert circuits[0].count_ops().get('cz',0)==r['methods']['incumbent_trained']['native_cz'];checks['routed_circuits']+=1
        differences.append(old[r['id']]['methods']['residual24_trained']['by_budget']['1024']['hybrid_success']-r['methods']['incumbent_trained']['by_budget']['1024']['hybrid_success'])
    dif=np.array(differences);idx=np.random.default_rng(cfg['bootstrap_seed']).integers(0,64,(cfg['bootstrap_replicates'],64))
    expected=np.quantile(dif[idx].mean(axis=1),[.025,.975])
    reported=summary['paired_development']['residual24_trained_minus_incumbent']['1024']
    np.testing.assert_allclose(expected,reported['ci95'],atol=1e-14);assert abs(dif.mean()-reported['mean'])<1e-14;checks['bootstrap_intervals']+=1
    checks.update(status='passed',completed_at=datetime.now().astimezone().isoformat(),paired_development_gain=reported)
    out=ROOT/'experiments/runs'/('AUDIT-020_'+datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%f%z'));out.mkdir()
    (out/'validation.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    (out/'audit_script.py').write_bytes(Path(__file__).read_bytes())
    (out/'manifest.json').write_text(json.dumps({'script_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_manifest_sha256':sha256((run/'manifest.json').read_bytes()).hexdigest(),
        'validation_sha256':sha256((out/'validation.json').read_bytes()).hexdigest()},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks,indent=2));print('AUDIT:',out)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
