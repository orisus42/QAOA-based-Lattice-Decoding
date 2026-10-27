"""DIAG-015: exact finite-budget diagnostic using saved probabilities."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from research_run import ResearchRun,save


def success(p,n):
    p=np.asarray(p,dtype=float)
    with np.errstate(divide='ignore',invalid='ignore'):
        result=-np.expm1(n*np.log1p(-p))
    return np.where(p>=1,1.,result) if n else np.zeros_like(p)


def min_shots(p,solved,target):
    p=np.asarray(p);solved=np.asarray(solved)
    value=lambda n:float(np.where(solved,1,success(p,n)).mean())
    if value(0)>=target:return 0
    if np.mean(solved|(p>0))<target:return None
    hi=1
    while value(hi)<target:
        hi*=2
        if hi>2**60:return None
    lo=0
    while lo+1<hi:
        mid=(lo+hi)//2
        if value(mid)>=target:hi=mid
        else:lo=mid
    assert value(hi)>=target and value(hi-1)<target
    return hi


def main():
    cfg=json.loads((ROOT/'experiments/configs/DIAG-015.json').read_text())
    paths=[ROOT/cfg['instances_run']/'instances.json',ROOT/cfg['baseline_run']/'per_instance.json',ROOT/cfg['sweep_run']/'grid.json']
    run=ResearchRun('DIAG-015',Path(__file__),paths)
    try:
        cases,base,grid=[json.loads(p.read_text()) for p in paths]
        assert [x['id'] for x in cases]==[x['id'] for x in base]
        solved=np.array([x['lll_babai_optimal'] for x in cases]);r1=np.array([x['classical']['refine_1']['optimal'] for x in base]);r2=np.array([x['classical']['refine_2']['optimal'] for x in base])
        identifiable=np.array([bool(x['unique_cvp'] and x['planted_secret_in_optimum_set']) for x in cases])
        strata={'all':np.ones(64,dtype=bool),'babai_failures':~solved,'two_coordinate_failures':~r2,
            'unique_planted_optimum':identifiable,'unique_planted_babai_failures':identifiable&~solved,
            'unique_planted_two_coordinate_failures':identifiable&~r2}
        probs={};gates={}
        for name,pct,policy in cfg['methods']:
            rr=sorted([x for x in grid if x['percent']==pct and x['policy']==policy],key=lambda r:r['index'])
            assert len(rr)==64
            probs[name]=np.array([x['p_optimum'] for x in rr]);gates[name]=float(np.mean([x['logical_rzz'] for x in rr]))
        for name in ['uniform_box','cbd_secret']:
            probs[name]=np.array([x['probabilities'][name] for x in base]);gates[name]=0.
        summary={'experiment_id':'DIAG-015','count':64,'babai_solved':int(solved.sum()),'refine_one_solved':int(r1.sum()),'refine_two_solved':int(r2.sum()),
            'strata':{name:{'count':int(m.sum()),'ids':[x['id'] for x,v in zip(cases,m) if v]} for name,m in strata.items()},
            'methods':{},'qpu_seconds':0,'limitations':cfg['limits']}
        detail=[]
        for name,p in probs.items():
            targets={}
            for target in cfg['targets']:
                n=min_shots(p,solved,target)
                targets[str(target)]={'common_shots_per_instance':n,'logical_rzz_shot_volume_per_instance':None if n is None else n*gates[name]}
            data={'mean_logical_rzz':gates[name],'cohort_targets':targets,'by_budget':{},'strata_at_1024':{}}
            for n in cfg['budgets']:
                v=np.where(solved,1,success(p,n));after_r2=np.where(r2,1,success(p,n))
                data['by_budget'][str(n)]={'babai_hybrid':float(v.mean()),'after_two_coordinate_refinement':float(after_r2.mean()),'additional_expected_optima_over_babai':float(v.sum()-solved.sum())}
            for stratum,m in strata.items():
                data['strata_at_1024'][stratum]={'mean_standalone_optimum_probability':float(success(p[m],1024).mean()) if m.any() else None,
                    'mean_babai_hybrid_success':float(np.where(solved[m],1,success(p[m],1024)).mean()) if m.any() else None}
            for i,x in enumerate(cases):
                detail.append({'id':x['id'],'method':name,'p_optimum':float(p[i]),'babai_optimal':bool(solved[i]),'two_coordinate_optimal':bool(r2[i]),
                    'unique_planted_optimum':bool(identifiable[i]),'shots_for_95pct_on_this_instance':min_shots(p[i:i+1],solved[i:i+1],.95)})
            summary['methods'][name]=data
        # Directly audit why the cohort ranking changes; instance rankings cannot reverse.
        a=probs['weak81'];b=probs['weak75'];m=~solved
        summary['ranking_diagnostic']={'weak81_higher_p_on_babai_failures':int(np.sum((a>b+1e-15)&m)),
            'weak75_higher_p_on_babai_failures':int(np.sum((b>a+1e-15)&m)),
            'per_instance_ranking_reversal_possible_at_fixed_p':False,
            'formula':'1-(1-p_i)^S; increasing in p_i for every S>0; cohort rankings may reverse through heterogeneous saturation.'}
        # Reproduction checks against stored finite-budget expectations.
        for name,pct,policy in cfg['methods']:
            rr=sorted([x for x in grid if x['percent']==pct and x['policy']==policy],key=lambda r:r['index'])
            for n in [1024,16384]:
                expected=[x['by_budget'][str(n)]['hybrid_success'] for x in rr]
                np.testing.assert_allclose(np.where(solved,1,success(probs[name],n)),expected,atol=1e-12)
        save(run.path/'per_instance.json',detail)
        sns.set_theme(style='whitegrid',context='talk',font_scale=.83)
        fig,axes=plt.subplots(1,2,figsize=(13,5.3),layout='constrained')
        palette={'full':'#246785','weak75':'#DA702C','weak81':'#9D4774','product100':'#7A878D','uniform_box':'#658E69'}
        ns=np.unique(np.geomspace(64,65536,140).astype(int))
        for name,color in palette.items():
            curve=np.array([np.where(solved,1,success(probs[name],int(n))).mean() for n in ns])
            axes[0].plot(ns,100*curve,label=name,color=color,lw=2.5)
            # Conditional labels are evaluator-only; they are not used by a solver.
            conditional=np.array([success(probs[name][~r2],int(n)).mean() for n in ns])
            axes[1].plot(ns,100*conditional,label=name,color=color,lw=2.5)
        axes[0].axhline(100*r2.mean(),color='#282C33',ls='--',lw=1.5,label='classical 2-coordinate')
        for ax in axes:
            ax.set_xscale('log',base=2);ax.set_xlabel('Samples per instance');ax.set_ylim(0 if ax==axes[1] else 80,100);sns.despine(ax=ax)
        axes[0].set_ylabel('Mean hybrid optimum success (%)');axes[0].set_title('All 64 development cases')
        axes[1].set_ylabel('Optimum sampled at least once (%)');axes[1].set_title('3 cases unresolved by classical refinement')
        axes[0].legend(fontsize=9,loc='lower right');axes[1].text(.04,.94,'Ideal predictions; not measured shots',transform=axes[1].transAxes,fontsize=10,va='top')
        fig.savefig(run.path/'shot_curves.svg');fig.savefig(run.path/'shot_curves.png',dpi=220);plt.close(fig)
        run.finish(summary)
    except Exception as e:run.finish(error=e);raise


if __name__=='__main__':main()
