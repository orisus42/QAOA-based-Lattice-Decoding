"""EXP-019: development-only incumbent-informed pruning control."""
from pathlib import Path
import json
import sys
import time
from hashlib import sha256
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from qiskit import transpile,qpy
from qiskit.quantum_info import Statevector
from lattice_qaoa.incumbent_pruning import incumbent_mask
from lattice_qaoa.residual_pruning import public_phase,edge_list
from lattice_qaoa.reference import babai,squared_cost
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import state_probabilities,circuit,P1Energy
from lattice_qaoa.sparsify import diagonal_energy
from lattice_qaoa.mechanisms import CachedMetrics
from lattice_qaoa.evaluation import matrix_group
from lattice_qaoa.instances import expand_module_matrix
from lattice_qaoa.device import target_from_snapshot
from run_residual_pruning import fit_global
from research_run import ResearchRun,save


def main():
    cfg=json.loads((ROOT/'experiments/configs/EXP-019.json').read_text())
    paths=[ROOT/cfg['training_run']/'training.json',ROOT/cfg['development_run']/'instances.json',
        ROOT/cfg['comparison_checkpoint']/'per_instance.json',ROOT/cfg['calibration'],ROOT/'experiments/run_residual_pruning.py']
    run=ResearchRun('EXP-019',Path(__file__),paths)
    try:
        training=json.loads(paths[0].read_text())['instances'];dev=json.loads(paths[1].read_text())
        existing={x['id']:x for x in json.loads(paths[2].read_text()) if x['cohort']=='development'}
        train_groups={matrix_group(expand_module_matrix(np.array(x['A_polynomials'])),17) for x in training}
        assert not train_groups&{matrix_group(x['A_expanded'],17) for x in dev}
        prepared=[]
        for x in training:
            R=np.array(x['reduced_basis']);y=np.r_[x['public_target'],np.zeros(4,dtype=int)]
            h,j,_,_=public_phase(R,y)
            phase=incumbent_mask(j,np.tile([1,-1],len(h)//2),cfg['logical_budget'])
            prepared.append((h,j,phase))
        fit=fit_global([x[0] for x in prepared],[x[1] for x in prepared],[x[2] for x in prepared],cfg)
        save(run.path/'training.json',fit)
        save(run.path/'frozen_model.json',{'training_sha256':sha256((run.path/'training.json').read_bytes()).hexdigest(),'evaluation_scope':'development only'})
        target=target_from_snapshot(json.loads(paths[3].read_text()));rows=[]
        checks={'statevectors':0,'independent_qiskit_checks':0,'max_probability_error':0.,'incumbent_codes_checked':0,'train_dev_overlap':0}
        (run.path/'circuits').mkdir()
        for ci,x in enumerate(dev):
            R=np.array(x['reduced_basis']);y=np.r_[x['public_target'],np.zeros(4,dtype=int)]
            t=time.perf_counter();h,j,z,_=public_phase(R,y)
            phase=incumbent_mask(j,np.tile([1,-1],len(h)//2),cfg['logical_budget']);selection_seconds=time.perf_counter()-t
            box=encode_babai_box(R,y)
            index=sum(1<<(2*i+1) for i in range(len(z)))
            np.testing.assert_array_equal(box.decode([index])[0],z);checks['incumbent_codes_checked']+=1
            np.testing.assert_allclose(box.h,h,atol=1e-13);np.testing.assert_allclose(box.J,j,atol=1e-13)
            inc=squared_cost(R,y,z);opt=x['reference']['cost'];metric=CachedMetrics(box.costs,inc,opt,cfg['budgets'])
            row={'id':x['id'],'seed':x['seed'],'group':matrix_group(x['A_expanded'],17),'selection_seconds':selection_seconds,'edges':edge_list(phase),'methods':{}}
            for name,angles in [('incumbent_same',cfg['selection_angles']),('incumbent_trained',fit['angles'])]:
                prob=state_probabilities(diagonal_energy(h,phase),angles);checks['statevectors']+=1
                np.testing.assert_allclose(prob@box.energy,P1Energy(h,phase,observable_J=j)(angles),atol=1e-12)
                row['methods'][name]={'angles':angles,'logical_rzz':int(np.count_nonzero(phase)),**metric(prob)}
                if ci<2:
                    direct=Statevector.from_instruction(circuit(h,phase,angles)).probabilities()
                    error=float(np.max(abs(prob-direct)));assert error<1e-12
                    checks['max_probability_error']=max(checks['max_probability_error'],error);checks['independent_qiskit_checks']+=1
                if name=='incumbent_trained':
                    qc=circuit(h,phase,angles);qc.barrier()
                    routed=transpile(qc,target=target,initial_layout=list(range(16)),optimization_level=3,seed_transpiler=cfg['routing_seed'],scheduling_method='alap')
                    row['methods'][name].update(native_cz=int(routed.count_ops().get('cz',0)),duration_unitary_s=routed.estimate_duration(target))
                    with (run.path/'circuits'/f'{x["id"]}.qpy').open('wb') as f:qpy.dump([routed],f)
            rows.append(row)
            if (ci+1)%16==0:print('development',ci+1,'/64',flush=True)
        idx=np.random.default_rng(cfg['bootstrap_seed']).integers(0,64,(cfg['bootstrap_replicates'],64))
        def stat(values):
            a=np.array(values);return {'mean':float(a.mean()),'ci95':np.quantile(a[idx].mean(axis=1),[.025,.975]).tolist()}
        summary={'experiment_id':'EXP-019','count':64,'training_count':16,'checks':checks,'methods':{},'paired_development':{},'qpu_seconds':0,'limitations':cfg['limitations']}
        for name in ['incumbent_same','incumbent_trained']:
            summary['methods'][name]={'by_budget':{str(b):stat([r['methods'][name]['by_budget'][str(b)]['hybrid_success'] for r in rows]) for b in cfg['budgets']}}
        summary['methods']['incumbent_trained']['native_cz']=stat([r['methods']['incumbent_trained']['native_cz'] for r in rows])
        for name in ['full_trained','magnitude24_trained','residual24_trained','geometry24_trained','magnitude24_effort']:
            summary['methods'][name]={'by_budget':{str(b):stat([existing[r['id']]['methods'][name]['by_budget'][str(b)]['hybrid_success'] for r in rows]) for b in cfg['budgets']}}
            summary['paired_development'][name+'_minus_incumbent']={str(b):stat([existing[r['id']]['methods'][name]['by_budget'][str(b)]['hybrid_success']-r['methods']['incumbent_trained']['by_budget'][str(b)]['hybrid_success'] for r in rows]) for b in cfg['budgets']}
        save(run.path/'per_instance.json',rows);save(run.path/'validation.json',checks);run.finish(summary)
        print(json.dumps({name:values['by_budget']['1024'] for name,values in summary['methods'].items()},indent=2))
    except Exception as exc:run.finish(error=exc);raise


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
