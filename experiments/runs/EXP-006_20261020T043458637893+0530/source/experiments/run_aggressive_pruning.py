"""EXP-006: extend the saved pruning frontier without modifying earlier runs."""
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from importlib.metadata import version
import copy
import json
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.optimize import differential_evolution
from qiskit import transpile
from qiskit_aer import AerSimulator
from threadpoolctl import threadpool_limits
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.instances import expand_module_matrix
from lattice_qaoa.qaoa import P1Energy, circuit, state_probabilities
from lattice_qaoa.sparsify import prune_interactions, diagonal_energy, expected_best_cost
from run_sparsification import summarize
from analyze_hybrid_refinement import analyze_distribution, aggregate, check_manifest


def now():
    return datetime.now(timezone(timedelta(hours=5,minutes=30)))


def save(path,data):
    path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


def product_distribution(h,angles):
    """Closed-form independent-bit probabilities for the J=0 endpoint."""
    gamma,beta=angles
    z=np.sin(2*beta)*np.sin(2*gamma*np.asarray(h))
    p_one=(1-z)/2
    bits=(np.arange(2**len(h))[:,None]>>np.arange(len(h)))&1
    return np.where(bits,p_one,1-p_one).prod(axis=1)


def main():
    cfg_path=ROOT/'experiments/configs/EXP-006.json';cfg=json.loads(cfg_path.read_text())
    start=now();timer=time.perf_counter()
    run=ROOT/'experiments/runs'/f"EXP-006_{start.strftime('%Y%m%dT%H%M%S%f%z')}"
    run.mkdir(exist_ok=False);save(run/'config.json',cfg)
    source=ROOT/cfg['input_run'];trainrun=ROOT/cfg['training_run']
    baseline=ROOT/cfg['baseline_run'];hybrid=ROOT/cfg['hybrid_run']
    sources=list((ROOT/'src/lattice_qaoa').glob('*.py'))+list((ROOT/'tests').glob('test_*.py'))+[
        Path(__file__),cfg_path,ROOT/'experiments/run_sparsification.py',ROOT/'experiments/analyze_hybrid_refinement.py',
        ROOT/'scripts/plot_aggressive_pruning.py',ROOT/'requirements-research.txt']
    inputs=[source/'instances.json',trainrun/'training.json',baseline/'per_instance.json',baseline/'summary.json',
            baseline/'samples.npz',baseline/'manifest.json',hybrid/'per_instance.json',hybrid/'summary.json',hybrid/'manifest.json']
    manifest={'run_id':run.name,'status':'running','started_at':start.isoformat(),
        'command':'python experiments/run_aggressive_pruning.py','python':platform.python_version(),
        'packages':{k:version(k) for k in ['numpy','scipy','sympy','qiskit','qiskit-aer','threadpoolctl','seaborn','matplotlib']},
        'source_sha256':{},'input_sha256':{},'qpu_seconds':0}
    for kind,paths in [('source',sources),('input',inputs)]:
        for path in paths:
            rel=path.relative_to(ROOT); dest=run/kind/rel
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
            manifest[kind+'_sha256'][rel.as_posix()]=sha256(path.read_bytes()).hexdigest()
    save(run/'manifest.json',manifest)
    try:
        checks={'input_manifest_hashes':check_manifest(baseline)+check_manifest(hybrid),
            'original_binary_costs_checked':0,'new_compiled_circuits_verified':0,'new_order_statistic_checks':0,
            'zero_interaction_product_distributions_verified':0,'max_probability_discrepancy':0.,
            'max_analytic_energy_discrepancy':0.,'max_product_probability_discrepancy':0.}
        instances=json.loads((source/'instances.json').read_text())
        training=json.loads((trainrun/'training.json').read_text())
        rows=copy.deepcopy(json.loads((baseline/'per_instance.json').read_text()))
        hrows=copy.deepcopy(json.loads((hybrid/'per_instance.json').read_text()))
        previous=json.loads((baseline/'summary.json').read_text())
        hprevious=json.loads((hybrid/'summary.json').read_text())
        test_matrices={str(x['A_expanded']) for x in instances}
        assert len(test_matrices)==64
        for x in training['instances']:
            assert str(expand_module_matrix(np.array(x['A_polynomials'])).tolist()) not in test_matrices
        frozen=training['fit']['angles'];fits={}
        # Fit only on the original separate training cases; do not inspect evaluation labels.
        for fraction in cfg['new_fractions']:
            clock=time.perf_counter();evaluators=[]
            for x in training['instances']:
                h,J=np.array(x['h']),np.array(x['J']);sparse,_=prune_interactions(J,fraction)
                evaluators.append(P1Energy(h,sparse,observable_h=h,observable_J=J))
            calls=0
            def objective(a):
                nonlocal calls
                calls+=1
                return float(np.mean([f(a) for f in evaluators]))
            settings=cfg['training']
            fit=differential_evolution(objective,cfg['parameter_bounds'],seed=settings['optimizer_seed'],
                maxiter=settings['maxiter'],popsize=settings['popsize'],tol=settings['tol'],polish=True)
            frozen_value=objective(frozen)
            angles=fit.x if fit.fun<frozen_value else np.array(frozen)
            selected_value=objective(angles)
            fits[str(fraction)]={'angles':angles.tolist(),'training_expected_original_energy':selected_value,
                'frozen_training_expected_original_energy':frozen_value,'objective_evaluations':calls,
                'instance_energy_evaluations':calls*len(evaluators),'optimizer_success':bool(fit.success),
                'optimizer_message':str(fit.message),'seconds':time.perf_counter()-clock}
        save(run/'training.json',{'frozen_angles':frozen,'new_fits':fits,'training_instances':16,
            'note':'Same separate public training cohort as EXP-003; no evaluation secrets or optimum labels used.'})
        new_names=[f'weak{round(100*f)}_{policy}' for f in cfg['new_fractions'] for policy in ['frozen','retrained']]
        all_names=list(previous['methods'])+new_names
        with np.load(baseline/'samples.npz') as original_samples:
            samples={key:original_samples[key].copy() for key in original_samples.files}
        for name in new_names:samples[name]=np.zeros((64,cfg['shots']),dtype=np.int64)
        simulator=AerSimulator(method='statevector',max_parallel_threads=2,max_parallel_experiments=1)
        for index,(x,row,hrow) in enumerate(zip(instances,rows,hrows)):
            assert x['id']==row['id']==hrow['id']
            y=np.r_[x['public_target'],np.zeros(4,dtype=int)]
            box=encode_babai_box(np.array(x['reduced_basis']),y)
            optimum=x['reference']['cost'];incumbent=x['lll_babai_cost'];good=box.costs==optimum
            assert good.any() and box.nq==16
            checks['original_binary_costs_checked']+=len(box.costs)
            specs=[]
            for fraction in cfg['new_fractions']:
                specs.extend([(f'weak{round(100*fraction)}_frozen',fraction,frozen),
                    (f'weak{round(100*fraction)}_retrained',fraction,fits[str(fraction)]['angles'])])
            circuits=[];probabilities=[]
            for name,fraction,angles in specs:
                sparse,removed=prune_interactions(box.J,fraction)
                phase=diagonal_energy(box.h,sparse);prob=state_probabilities(phase,angles)
                np.testing.assert_allclose(prob.sum(),1,atol=1e-12)
                delta=abs(float(prob@box.energy)-P1Energy(box.h,sparse,observable_h=box.h,observable_J=box.J)(angles))
                assert delta<1e-9
                checks['max_analytic_energy_discrepancy']=max(checks['max_analytic_energy_discrepancy'],delta)
                qc=circuit(box.h,sparse,angles)
                compiled=transpile(qc,basis_gates=cfg['compilation']['basis_gates'],optimization_level=3,
                    seed_transpiler=cfg['compilation']['seed'])
                if fraction==1:
                    assert not np.any(sparse) and compiled.count_ops().get('cx',0)==0
                    discrepancy=float(np.max(np.abs(product_distribution(box.h,angles)-prob)))
                    assert discrepancy<1e-12
                    checks['zero_interaction_product_distributions_verified']+=1
                    checks['max_product_probability_discrepancy']=max(checks['max_product_probability_discrepancy'],discrepancy)
                ground=np.isclose(phase,phase.min(),atol=1e-10,rtol=0)
                row['methods'][name]={'fraction':fraction,'angles':list(angles),'removed_edges':removed.tolist(),
                    'removed_absolute_weight':float(np.abs(box.J-sparse).sum()),
                    'optimum_probability':float(prob[good].sum()),'refinement_probability':float(prob[box.costs<incumbent].sum()),
                    'expected_original_cost':float(prob@box.costs),
                    'expected_best_cost':expected_best_cost(box.costs,prob,cfg['shots']),
                    'expected_hybrid_best_cost':expected_best_cost(box.costs,prob,cfg['shots'],incumbent),
                    'logical_rzz':int(qc.count_ops().get('rzz',0)),
                    'compiled_cx':int(compiled.count_ops().get('cx',0)),'compiled_depth':compiled.depth(),
                    'compiled_two_qubit_depth':compiled.depth(lambda op:op.operation.num_qubits==2),
                    'compiled_operations':dict(compiled.count_ops()),
                    'phase_ground_overlaps_optimum':bool(np.any(ground&good)),
                    'phase_ground_equals_optimum':bool(np.array_equal(ground,good)),
                    'observed':{},'measurement_source_run':run.name}
                hrow['methods'][name]={'by_budget':{}}
                for shots in cfg['budgets']:
                    hrow['methods'][name]['by_budget'][str(shots)]=analyze_distribution(box.costs,prob,incumbent,optimum,shots)
                    checks['new_order_statistic_checks']+=1
                compiled.save_probabilities(label='probabilities');compiled.measure_all()
                circuits.append(compiled);probabilities.append(prob)
            result=simulator.run(circuits,shots=cfg['shots'],memory=True,seed_simulator=cfg['measurement_seed']+index).result()
            assert result.success
            for j,(name,_,_) in enumerate(specs):
                qp=np.asarray(result.data(j)['probabilities']);delta=float(np.max(np.abs(qp-probabilities[j])))
                assert delta<1e-10
                checks['max_probability_discrepancy']=max(checks['max_probability_discrepancy'],delta)
                checks['new_compiled_circuits_verified']+=1
                measured=np.array([int(bits.replace(' ',''),2) for bits in result.get_memory(j)])
                samples[name][index]=measured;costs=box.costs[measured]
                for shots in cfg['budgets']:
                    winner=int(costs[:shots].argmin());best=int(costs[winner])
                    z=np.array(x['basis_transform'])@box.decode([measured[winner]])[0]
                    assert int(np.sum((np.array(x['embedding_basis'])@z-y)**2))==best
                    row['methods'][name]['observed'][str(shots)]={'successful_shots':int(good[measured[:shots]].sum()),
                        'best_cost':best,'solved':best==optimum,'hybrid_solved':min(best,incumbent)==optimum,
                        'secret_recovered':bool(np.array_equal(z[4:],x['planted_secret']))}
                    hrow['methods'][name]['by_budget'][str(shots)]['observed_cost']=min(best,incumbent)
            if (index+1)%16==0:print(f'Validated, compiled and sampled {index+1}/64 examples',flush=True)
        boot=np.random.default_rng(cfg['bootstrap_seed']).integers(0,64,size=(cfg['bootstrap_replicates'],64))
        methods={key:summarize(rows,key,cfg,boot) for key in all_names}
        babai=hprevious['babai']['mean_distance']
        full_distance=hprevious['methods']['full_fixed']['all64']['mean_expected_distance']
        half_distance=hprevious['methods']['weak50_frozen']['all64']['mean_expected_distance']
        for key,m in methods.items():
            m['hybrid_all64']=aggregate(hrows,key,cfg['shots'])
            m['hybrid_babai_failures_only']=aggregate(hrows,key,cfg['shots'],lambda r:not r['babai_optimal'])
            d=m['hybrid_all64']['mean_expected_distance']
            m['hybrid_distance_improvement_retention_vs_full']=(babai-d)/(babai-full_distance)
            m['hybrid_distance_improvement_retention_vs_half']=(babai-d)/(babai-half_distance)
            m['standalone_success_retention_vs_half']=m['expected_solved_fraction_1024']/previous['methods']['weak50_frozen']['expected_solved_fraction_1024']
            m['retains_95pct_of_full_standalone_success']=m['solved_fraction_retention_ratio_1024']>=.95
            m['retains_95pct_of_full_hybrid_distance_improvement']=m['hybrid_distance_improvement_retention_vs_full']>=.95
            diff=np.array([r['methods'][key]['by_budget']['1024']['expected_distance']-
                           r['methods']['full_fixed']['by_budget']['1024']['expected_distance'] for r in hrows])
            m['hybrid_distance_better_equal_worse_vs_full']=[int((diff<-1e-12).sum()),int((np.abs(diff)<=1e-12).sum()),int((diff>1e-12).sum())]
            np.testing.assert_allclose(m['hybrid_all64']['expected_solved_fraction'],m['expected_hybrid_solved_fraction_1024'],atol=1e-12)
            if key in previous['methods']:
                np.testing.assert_allclose(m['mean_single_shot_optimum_probability'],previous['methods'][key]['mean_single_shot_optimum_probability'],atol=1e-15)
                np.testing.assert_allclose(d,hprevious['methods'][key]['all64']['mean_expected_distance'],atol=1e-12)
        summary={'experiment_id':'EXP-006','run_id':run.name,'count':64,'qubits':16,'depth_p':1,'shots':cfg['shots'],
            'cohort':cfg['cohort'],'new_methods':new_names,'methods':methods,'training':fits,'validation':checks,
            'babai':hprevious['babai'],'reference':hprevious['reference'],
            'classical_hybrid_controls':{k:hprevious['methods'][k] for k in ['refine_1','refine_2','uniform_box']},
            'classical_controls_from_EXP003':previous['classical_controls_from_EXP003'],
            'new_ideal_measurements':64*len(new_names)*cfg['shots'],
            'reused_ideal_measurements':64*len(previous['methods'])*cfg['shots'],
            'timing_seconds':time.perf_counter()-timer,'qpu_seconds':0,'limitations':cfg['limitations']}
        save(run/'per_instance.json',rows);save(run/'hybrid_per_instance.json',hrows);save(run/'summary.json',summary)
        np.savez_compressed(run/'samples.npz',**samples)
        manifest.update(status='completed',finished_at=now().isoformat())
        manifest['output_sha256']={n:sha256((run/n).read_bytes()).hexdigest() for n in ['config.json','training.json','per_instance.json','hybrid_per_instance.json','summary.json','samples.npz']}
        save(run/'manifest.json',manifest)
        print('RUN:',run)
        for key in ['full_fixed','weak25_frozen','weak50_frozen']+new_names:
            m=methods[key]
            print(key,'CX',round(m['compiled_cx']['mean'],2),'QAOA success',round(100*m['expected_solved_fraction_1024'],3),
                'hybrid distance',round(m['hybrid_all64']['mean_expected_distance'],6),
                'refinement retained',round(100*m['hybrid_distance_improvement_retention_vs_full'],2))
    except Exception as exc:
        manifest.update(status='failed',finished_at=now().isoformat(),error=repr(exc));save(run/'manifest.json',manifest)
        raise


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
