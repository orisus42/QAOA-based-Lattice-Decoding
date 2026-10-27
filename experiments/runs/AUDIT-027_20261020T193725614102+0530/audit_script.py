"""Independent audit of saved native matching, noise masses and primary CI."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from qiskit import qpy
from lattice_qaoa.instances import generate_instance,Parameters,expand_module_matrix
from lattice_qaoa.reference import embedding
from lattice_qaoa.evaluation import matrix_group
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import state_probabilities
from lattice_qaoa.sparsify import diagonal_energy
from lattice_qaoa.residual_pruning import mask_from_edges,edge_list
from lattice_qaoa.cached_pruning import select_masks_cached
from lattice_qaoa.incumbent_pruning import incumbent_mask
from lattice_qaoa.device import logical_to_physical_indices,readout_channel,target_from_snapshot


def audit(path):
    cfg=json.loads((path/'config.json').read_text());manifest=json.loads((path/'manifest.json').read_text())
    assert manifest['status']=='completed'
    summary=json.loads((path/'summary.json').read_text());cohort=json.loads((path/'cohort.json').read_text())
    rows=json.loads((path/'per_instance.json').read_text());instances=json.loads((path/'instances.json').read_text())
    snap=json.loads((path/'input'/cfg['calibration']).read_text());target=target_from_snapshot(snap)
    frozen=json.loads((path/'frozen_protocol.json').read_text())
    checks={'hashes':0,'public_instances':0,'matching_pairs':0,'logical_probabilities':0,
            'native_circuits':0,'native_edges':0,'noise_files':0,'readout_linearity_checks':0,'cached_replays':0}
    for kind in ['source','input','output']:
        for rel,digest in manifest[kind+'_sha256'].items():
            actual=path/(kind if kind!='output' else '')/rel
            assert sha256(actual.read_bytes()).hexdigest()==digest,(kind,rel);checks['hashes']+=1
    assert frozen['training_sha256']==sha256((path/'training.json').read_bytes()).hexdigest()
    assert frozen['config_sha256']==sha256((path/'config.json').read_bytes()).hexdigest()
    assert len(rows)==len(instances)==cfg['fresh_count']==len(set(cohort['fresh_groups']))
    assert not set(cohort['fresh_groups'])&set(cohort['excluded_groups'])
    failures=[i for i,r in enumerate(rows) if r['babai_cost']>r['optimum']]
    assert failures==cohort['babai_failure_indices']
    assert sorted(set(failures)|set(cohort['sentinel_indices']))==cohort['noise_indices']
    p=Parameters(**cfg['parameters']);checks['maximum_native_gap']=0
    replay=set(np.linspace(0,len(rows)-1,6,dtype=int))
    for i,(r,x) in enumerate(zip(rows,instances)):
        assert r['id']==x['id'];generated=generate_instance(cfg['fresh_first_seed']+i,p)
        np.testing.assert_array_equal(generated.A,x['A_polynomials'])
        np.testing.assert_array_equal(generated.target.ravel(),x['public_target'])
        A=expand_module_matrix(generated.A);B,y=embedding(A,np.array(x['public_target']),p.q)
        R=np.array(x['reduced_basis']);U=np.array(x['basis_transform'])
        np.testing.assert_array_equal(B@U,R)
        np.testing.assert_array_equal((A@np.array(x['planted_secret'])+np.array(x['planted_error']))%p.q,x['public_target'])
        assert matrix_group(A,p.q)==r['group']==cohort['fresh_groups'][i];checks['public_instances']+=1
        box=encode_babai_box(R,y);h,j=box.h,box.J
        detail=json.loads((path/'selection'/f'{r["id"]}.json').read_text())
        # Re-derive the matching objective independently over every recorded candidate.
        for cap in cfg['native_caps']:
            options=[]
            for a in detail['resources']['cached']:
                for b in detail['resources']['incumbent']:
                    ca,cb=a['native_cz'],b['native_cz']
                    if max(ca,cb)>cap or abs(ca-cb)>cfg['native_gap_tolerance']:continue
                    key=(-min(ca,cb),abs(ca-cb),-(a['logical_k']+b['logical_k']),
                         -min(a['logical_k'],b['logical_k']),a['duration_unitary_s']+b['duration_unitary_s'],
                         -a['logical_k'],-b['logical_k'],a['routing_seed'],b['routing_seed'])
                    options.append((key,a,b))
            _,a,b=min(options,key=lambda v:v[0])
            for family,choice in [('cached',a),('incumbent',b)]:
                selected=r['methods'][f'{family}_cap{cap}']
                assert selected['native_cz']==choice['native_cz'] and selected['logical_k']==choice['logical_k']
                assert selected['edges']==choice['edges']
            gap=abs(a['native_cz']-b['native_cz'])
            checks['maximum_native_gap']=max(checks['maximum_native_gap'],gap);checks['matching_pairs']+=1
        if i in replay:
            chosen,_=select_masks_cached(h,j,cfg['logical_grid'],cfg['selection_angles'])
            for name in cfg['methods']:
                if name.startswith('cached'):
                    np.testing.assert_array_equal(chosen[r['methods'][name]['logical_k']]['J'],
                                                  mask_from_edges(j,r['methods'][name]['edges']))
            checks['cached_replays']+=1
        with (path/'circuits'/f'{r["id"]}.qpy').open('rb') as f:circuits=qpy.load(f)
        assert len(circuits)==len(cfg['methods'])
        with np.load(path/'noise'/f'{r["id"]}_ideal.npz') as f:ideal={k:f[k].copy() for k in f.files}
        for mi,name in enumerate(cfg['methods']):
            m=r['methods'][name];phase=mask_from_edges(j,m['edges'])
            assert m['angles']==frozen['angles'][name]
            if name.startswith('incumbent'):
                np.testing.assert_array_equal(phase,incumbent_mask(j,np.tile([1,-1],8),m['logical_k']))
            prob=state_probabilities(diagonal_energy(h,phase),m['angles'])
            low=np.array([prob[box.costs==cost].sum() for cost in range(r['babai_cost']+1)])
            np.testing.assert_allclose(low,ideal[name+'_ideal'][0],atol=1e-12);checks['logical_probabilities']+=1
            qc=circuits[mi];assert not qc.parameters
            assert qc.count_ops().get('cz',0)==m['native_cz'];checks['native_circuits']+=1
            for ins in qc.data:
                if ins.operation.num_qubits==2:
                    pair=tuple(qc.find_bit(q).index for q in ins.qubits)
                    assert ins.operation.name=='cz' and pair in target['cz'];checks['native_edges']+=1
            if i not in cohort['noise_indices']:continue
            with np.load(path/'noise'/f'{r["id"]}_{name}.npz') as f:data={k:f[k].copy() for k in f.files}
            assert int(data['seed'])==cfg['noise_seed']+100*i+mi
            for channel,meanname in [('gate_idle','mean_before_readout'),('gate_idle_readout','mean_after_readout')]:
                raw=data[channel];mean=data[meanname]
                assert raw.shape==(cfg['trajectories_per_circuit'],r['babai_cost']+1)
                assert raw.min()>=0 and np.all(raw.sum(axis=1)<=1+cfg['normalization_tolerance'])
                np.testing.assert_allclose(mean.sum(),1,atol=cfg['normalization_tolerance'],rtol=0)
                np.testing.assert_allclose(raw.mean(axis=0),[mean[box.costs==cost].sum() for cost in range(r['babai_cost']+1)],atol=1e-12)
            order=logical_to_physical_indices(m['permutation']);physical=np.empty(65536)
            physical[order]=data['mean_before_readout']
            transformed=readout_channel(physical,snap)[order]
            np.testing.assert_allclose(transformed,data['mean_after_readout'],atol=1e-12)
            checks['readout_linearity_checks']+=1;checks['noise_files']+=1
    progress=[json.loads(line) for line in (path/'noise_progress.jsonl').read_text().splitlines()]
    assert len(progress)==len(cohort['noise_indices'])*len(cfg['methods'])
    assert max(x['max_norm_deviation'] for x in progress)<=cfg['normalization_tolerance']
    # Reconstruct the primary bootstrap directly from optimum probability columns.
    n=len(rows);reps=cfg['bootstrap_replicates'];rng=np.random.default_rng(cfg['bootstrap_seed'])
    weights=rng.multinomial(n,np.full(n,1/n),size=reps)
    good=np.array([r['babai_cost']==r['optimum'] for r in rows],dtype=float)
    pop={};fixed={};point={};halves={};shots=cfg['primary_budget']
    hit=lambda prob:-np.expm1(shots*np.log1p(-np.clip(prob,0,1-1e-16)))
    for name in cfg['methods']:
        group=weights@good/n;mc=np.full(reps,good.mean());value=float(good.mean());half=np.full(2,good.mean())
        for i in failures:
            r=rows[i]
            with np.load(path/'noise'/f'{r["id"]}_{name}.npz') as f:
                a=f['gate_idle_readout'][:,r['optimum']].copy()
            T=len(a);w=rng.multinomial(T,np.full(T,1/T),size=reps)
            draw=hit(w@a/T);group+=weights[:,i]*draw/n;mc+=draw/n;value+=float(hit(a.mean()))/n
            half+=np.array([hit(a[:T//2].mean()),hit(a[T//2:].mean())])/n
        pop[name]=group;fixed[name]=mc;point[name]=value;halves[name]=half.tolist()
    a,b='cached_cap160','incumbent_cap160';reported=summary['paired']['cap160']['gate_idle_readout']['1024']
    np.testing.assert_allclose(point[a]-point[b],reported['success_difference'],atol=1e-12)
    np.testing.assert_allclose(np.quantile(pop[a]-pop[b],[.025,.975]),reported['group_plus_mc_ci95'],atol=1e-12)
    np.testing.assert_allclose(np.quantile(fixed[a]-fixed[b],[.025,.975]),reported['fixed_cohort_mc_ci95'],atol=1e-12)
    checks.update(primary_bootstrap_reproduced=True,half_trajectory_success_sensitivity=halves,
        primary_half_trajectory_differences=(np.array(halves[a])-np.array(halves[b])).tolist(),
        status='passed',completed_at=datetime.now().astimezone().isoformat(),source_run=path.relative_to(ROOT).as_posix())
    out=ROOT/'experiments/runs'/('AUDIT-027_'+datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%f%z'));out.mkdir()
    (out/'validation.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    shutil.copy2(__file__,out/'audit_script.py')
    (out/'manifest.json').write_text(json.dumps({'input_manifest_sha256':sha256((path/'manifest.json').read_bytes()).hexdigest(),
        'script_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256':{p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/lattice_qaoa').glob('*.py')},
        'validation_sha256':sha256((out/'validation.json').read_bytes()).hexdigest()},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks,indent=2));print('AUDIT:',out)


if __name__=='__main__':
    with threadpool_limits(limits=1):audit(Path(sys.argv[1]).resolve())
