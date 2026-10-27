"""Exploratory coefficient/phase diagnostics; no optimization or QPU calls."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import shutil
import sys
import importlib.metadata
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.sparsify import prune_interactions, diagonal_energy


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def main():
    now = datetime.now().astimezone()
    run = ROOT/'experiments/runs'/('DIAG-009_'+now.strftime('%Y%m%dT%H%M%S%f%z'))
    run.mkdir(exist_ok=False)
    config = {'id':'DIAG-009','purpose':'Explore lattice coefficient geometry and pruning granularity; no performance tuning',
        'development_input':'experiments/runs/EXP-001_20261018T120321801881+0530/instances.json',
        'fresh_results':'experiments/runs/EXP-007_20261020T044846904050+0530/per_instance.json',
        'noise_summary':'experiments/runs/EXP-008_20261020T050837222393+0530/summary.json',
        'gamma':1.3393885543560986,'prune_fraction':0.75,'cohort':'all 64 old development cases',
        'seed':None,'randomness':'none; deterministic reanalysis','qpu_seconds':0}
    write(run/'config.json', config)
    sources=[Path(__file__),ROOT/'src/lattice_qaoa/encoding.py',ROOT/'src/lattice_qaoa/sparsify.py',ROOT/'src/lattice_qaoa/reference.py']
    inputs=[ROOT/config[k] for k in ['development_input','fresh_results','noise_summary']]
    for folder, paths in [('source',sources),('input',inputs)]:
        for path in paths:
            dst=run/folder/path.relative_to(ROOT)
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
    records=[]
    for x in json.loads(inputs[0].read_text()):
        B=np.array(x['reduced_basis']); y=np.r_[x['public_target'],np.zeros(4,dtype=int)]
        box=encode_babai_box(B,y)
        J,removed=prune_interactions(box.J,config['prune_fraction'])
        within=np.array([(2*i,2*i+1) for i in range(8)])
        delta=diagonal_energy(np.zeros(16),box.J-J)
        e2=float(np.mean(delta**2));j2=float(np.sum((box.J-J)**2))
        np.testing.assert_allclose(e2,j2,atol=1e-13)
        phase=np.exp(-1j*config['gamma']*delta)
        distance2=float(np.mean(np.abs(phase-1)**2))
        assert distance2<=config['gamma']**2*j2+1e-12
        records.append({'id':x['id'],'nonzero_edges':int(np.count_nonzero(box.J)),
            'removed_edges':len(removed),'within_coordinate_retained':int(np.count_nonzero(J[within[:,0],within[:,1]])),
            'removed_pair_squared_weight_fraction':j2/float(np.sum(box.J**2)),
            'removed_total_squared_weight_fraction':j2/float(np.sum(box.J**2)+box.h@box.h),
            'p1_distribution_tv_upper_bound':min(1.,abs(config['gamma'])*np.sqrt(j2)),
            'same_angle_p1_state_fidelity':float(abs(np.mean(phase))**2),
            'pure_state_l2_distance_squared':distance2})
    fresh=json.loads(inputs[1].read_text());edges=np.array([x['methods']['full_fixed']['logical_rzz'] for x in fresh])
    noise=json.loads(inputs[2].read_text());full=noise['methods']['full_fixed']['by_budget']['1024']
    decomposition={}
    for name in ['weak75_frozen','weak75_retrained']:
        method=noise['methods'][name]['by_budget']['1024']
        ideal=method['ideal_hybrid_success']-full['ideal_hybrid_success']
        noisy=method['hybrid_success']['estimate']-full['hybrid_success']['estimate']
        decomposition[name]={'ideal_gain_pp':100*ideal,'noisy_gain_pp':100*noisy,
            'difference_in_noise_losses_pp':100*(noisy-ideal)}
    summary={'experiment_id':'DIAG-009','count':64,'recorded_at':now.isoformat(),
        'geometry_statistics':{},'fresh_edge_count_min_median_max':[int(edges.min()),float(np.median(edges)),int(edges.max())],
        'fresh_same_mask_count_74_75':int((np.floor(.74*edges)==np.floor(.75*edges)).sum()),
        'fresh_same_mask_count_75_76':int((np.floor(.75*edges)==np.floor(.76*edges)).sum()),
        'success_gain_decomposition':decomposition,
        'limits':['Exploratory descriptive diagnostics on old development data; no causality or new holdout performance claim.',
            'The state-distance bound is a standard elementary consequence of diagonal phase evolution and orthogonality, not a novelty claim.',
            'The p=1 bound assumes uniform initial state, identical angles/mixer, and unchanged fields/normalization; it does not guarantee improved success.',
            'Difference in noise losses is an algebraic decomposition, not an isolated causal noise-channel experiment.'],
        'qpu_seconds':0}
    for key in records[0]:
        if key=='id':continue
        values=np.array([r[key] for r in records])
        summary['geometry_statistics'][key]={'min':float(values.min()),'median':float(np.median(values)),'mean':float(values.mean()),'max':float(values.max())}
    write(run/'per_instance.json',records);write(run/'summary.json',summary)
    manifest={'run_id':run.name,'status':'completed','started_at':now.isoformat(),'completed_at':datetime.now().astimezone().isoformat(),
        'python':sys.version,'packages':{p:importlib.metadata.version(p) for p in ['numpy','sympy','threadpoolctl']},
        'sha256':{p.relative_to(run).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}}
    write(run/'manifest.json',manifest)
    print(run);print(json.dumps(summary,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
