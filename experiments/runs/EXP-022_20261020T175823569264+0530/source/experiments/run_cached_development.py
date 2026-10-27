"""EXP022: old development inputs only; no fresh outcomes used."""
from pathlib import Path
from datetime import datetime
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from lattice_qaoa.residual_pruning import public_phase,select_masks,edge_list
from lattice_qaoa.cached_pruning import select_masks_cached
from research_run import ResearchRun,save


def main():
    cfg=json.loads((ROOT/'experiments/configs/EXP-022.json').read_text())
    path=ROOT/cfg['development_run']/'instances.json'
    run=ResearchRun('EXP-022',Path(__file__),[path])
    rows=[]
    try:
        for i,x in enumerate(json.loads(path.read_text())):
            R=np.array(x['reduced_basis']);y=np.r_[x['public_target'],np.zeros(4,dtype=int)]
            h,j,_,_=public_phase(R,y);k=cfg['logical_budget'];a=cfg['selection_angles']
            t=time.perf_counter();old,ot=select_masks(h,j,[k],a);old_t=time.perf_counter()-t
            t=time.perf_counter();new,nt=select_masks_cached(h,j,[k],a);new_t=time.perf_counter()-t
            np.testing.assert_array_equal(old[k]['J'],new[k]['J'])
            assert [v['removed_edge'] for v in ot]==[v['removed_edge'] for v in nt]
            np.testing.assert_allclose([v['original_energy'] for v in ot],[v['original_energy'] for v in nt],atol=1e-12)
            rows.append({'id':x['id'],'seed':x['seed'],'identical_mask':True,'identical_path':True,
                         'reference_seconds':old_t,'cached_seconds':new_t,'edges':edge_list(new[k]['J']),
                         'reference_calls':old[k]['analytic_expectation_evaluations'],
                         'cached_calls':new[k]['analytic_expectation_evaluations'],
                         'tie_recomputations':new[k]['tie_recomputations']})
            if (i+1)%8==0:
                save(run.path/'per_instance.json',rows)
                print('Development',i+1,'/64',flush=True)
        save(run.path/'per_instance.json',rows)
        summary={'experiment_id':'EXP-022','count':len(rows),'identical_mask_count':sum(r['identical_mask'] for r in rows),
                 'identical_path_count':sum(r['identical_path'] for r in rows),
                 'diagnostic_mean_seconds':{m:float(np.mean([r[m+'_seconds'] for r in rows])) for m in ['reference','cached']},
                 'limitations':cfg['limitations'],'qpu_seconds':0}
        run.finish(summary);print(json.dumps(summary,indent=2))
    except Exception as exc:
        save(run.path/'per_instance.json',rows);run.finish(error=exc);raise


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
