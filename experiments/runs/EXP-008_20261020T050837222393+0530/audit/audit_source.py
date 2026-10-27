"""Audit calibration replay, circuits, saved noise distributions and aggregation."""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
from qiskit import qpy
from threadpoolctl import threadpool_limits
from lattice_qaoa.device import target_from_snapshot,readout_channel,logical_to_physical_indices
from lattice_qaoa.encoding import encode_babai_box
from analyze_hybrid_refinement import check_manifest,analyze_distribution


def main():
    run=Path(sys.argv[1]).resolve();cfg=json.loads((run/'config.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text());summary=json.loads((run/'summary.json').read_text())
    snap=json.loads((run/'input'/cfg['calibration']/'snapshot.json').read_text());target=target_from_snapshot(snap)
    rows=json.loads((run/'routed_instances.json').read_text())
    instances=json.loads((run/'input'/cfg['input_run']/'instances.json').read_text())
    reports={r['source_index']:r for r in json.loads((run/'noisy_per_instance.json').read_text())}
    cohort=json.loads((run/'cohort.json').read_text())
    decision=next(json.loads(line) for line in (ROOT/'project/decisions.jsonl').read_text().splitlines() if json.loads(line)['id']=='D-020')
    frozen=run/'source/experiments/configs/EXP-008.json'
    assert sha256(frozen.read_bytes()).hexdigest()==decision['config_sha256'] and json.loads(frozen.read_text())==cfg
    assert datetime.fromisoformat(decision['recorded_at'])<datetime.fromisoformat(manifest['started_at'])
    expected_random=sorted(np.random.default_rng(cfg['random_subset_seed']).choice(512,32,replace=False).tolist())
    assert expected_random==cohort['random_subset_indices']
    expected_failures=[i for i,r in enumerate(rows) if r['babai_cost']>r['optimum_cost']]
    assert expected_failures==cohort['babai_failure_indices']
    assert sorted(set(expected_random)|set(expected_failures))==cohort['noise_union_indices']
    audit={'recorded_at':datetime.now().astimezone().isoformat(),'manifest_hashes_verified':check_manifest(run),
        'qpy_circuits_reloaded':0,'noise_outputs_checked':0,'readout_reapplication_checks':0,
        'independent_order_statistic_checks':0,'aggregate_method_budget_checks':0,'max_readout_difference':0.}
    checkpoints=[json.loads(line) for line in (run/'noise_progress.jsonl').read_text().splitlines()]
    assert len(checkpoints)==101*3 and len({r['seed'] for r in checkpoints})==len(checkpoints)
    for p in checkpoints:
        assert sha256((run/p['file']).read_bytes()).hexdigest()==p['sha256']
        assert p['trajectories']==64
    for i,row in enumerate(rows):
        with (run/'circuits'/f'{row["id"]}.qpy').open('rb') as f:circuits=qpy.load(f)
        assert len(circuits)==3
        for name,qc in zip(cfg['methods'],circuits):
            m=row['methods'][name]
            assert qc.count_ops().get('cz',0)==m['native_cz'] and qc.num_qubits==16
            assert qc.layout.final_index_layout()==m['final_logical_to_patch_permutation']
            np.testing.assert_allclose(qc.estimate_duration(target),m['duration_unitary_s'],atol=1e-15,rtol=0)
            for op in qc.data:
                if op.operation.num_qubits==2:
                    assert op.operation.name=='cz'
                    assert tuple(qc.find_bit(q).index for q in op.qubits) in target['cz']
            audit['qpy_circuits_reloaded']+=1
        if i not in reports:continue
        x=instances[i];box=encode_babai_box(np.array(x['reduced_basis']),np.r_[x['public_target'],np.zeros(4,dtype=int)])
        for name in cfg['methods']:
            with np.load(run/'noise'/f'{row["id"]}_{name}.npz') as data:
                low=data['low_cost_masses'];mean=data['mean_logical_probability'];before=data['mean_before_readout_logical_probability']
            assert low.shape==(64,row['babai_cost']+1)
            np.testing.assert_allclose(mean.sum(),1,atol=1e-10)
            np.testing.assert_allclose(low.mean(axis=0),[mean[box.costs==c].sum() for c in range(row['babai_cost']+1)],atol=1e-12)
            order=logical_to_physical_indices(row['methods'][name]['final_logical_to_patch_permutation'])
            physical=np.empty_like(before);physical[order]=before
            replay=readout_channel(physical,snap)[order]
            delta=float(np.max(np.abs(replay-mean)))
            assert delta<1e-12
            audit['max_readout_difference']=max(audit['max_readout_difference'],delta)
            audit['readout_reapplication_checks']+=1
            for shots in [1024,16384]:
                value=analyze_distribution(box.costs,mean,row['babai_cost'],row['optimum_cost'],shots)
                reported=reports[i]['methods'][name]['by_budget'][str(shots)]
                for original,new in [('expected_distance','hybrid_expected_distance'),('expected_cost','hybrid_expected_cost'),('optimum_probability','hybrid_success'),('improvement_probability','strict_improvement')]:
                    np.testing.assert_allclose(value[original],reported[new],atol=1e-9,rtol=0)
                audit['independent_order_statistic_checks']+=1
            audit['noise_outputs_checked']+=1
    for name in cfg['methods']:
        for shots in cfg['budgets']:
            distances=[];successes=[]
            for i,row in enumerate(rows):
                if row['babai_cost']==row['optimum_cost']:
                    distances.append(np.sqrt(row['babai_cost']));successes.append(1.)
                else:
                    value=reports[i]['methods'][name]['by_budget'][str(shots)]
                    distances.append(value['hybrid_expected_distance']);successes.append(value['hybrid_success'])
            aggregate=summary['methods'][name]['by_budget'][str(shots)]
            np.testing.assert_allclose(np.mean(distances),aggregate['hybrid_distance']['estimate'],atol=1e-12)
            np.testing.assert_allclose(np.mean(successes),aggregate['hybrid_success']['estimate'],atol=1e-12)
            audit['aggregate_method_budget_checks']+=1
    out=run/'audit';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'audit_source.py')
    (out/'results.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    for path in out.iterdir():manifest['output_sha256'][path.relative_to(run).as_posix()]=sha256(path.read_bytes()).hexdigest()
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
