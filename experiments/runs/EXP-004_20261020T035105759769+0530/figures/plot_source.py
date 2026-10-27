"""EXP-004: reproducible quality/resource frontier and per-instance retention."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    run=Path(sys.argv[1]).resolve()
    s=json.loads((run/'summary.json').read_text())
    rows=json.loads((run/'per_instance.json').read_text())
    assert s['experiment_id']=='EXP-004'
    out=run/'figures';version=1
    while out.exists():
        version+=1;out=run/f'figures-v{version}'
    out.mkdir()
    shutil.copy2(__file__,out/'plot_source.py')
    sns.set_theme(style='whitegrid',font='DejaVu Sans',context='notebook')
    plt.rcParams.update({'svg.fonttype':'none','text.color':'#173042','axes.labelcolor':'#173042',
        'axes.edgecolor':'#d7dede','grid.color':'#e5e9e7','grid.linewidth':.6})
    teal,orange,ink='#087f8c','#ce6a43','#173042'
    fig,axs=plt.subplots(1,2,figsize=(14.5,7.5),gridspec_kw={'width_ratios':[1.05,1]})
    fig.patch.set_facecolor('#fbfaf7')
    fig.subplots_adjust(left=.075,right=.96,bottom=.25,top=.69,wspace=.3)
    fig.text(.075,.94,'MODULE-LWE / QAOA CIRCUIT REDUCTION',fontsize=11,weight='bold',color=teal)
    fig.text(.075,.865,'How much circuit can we remove?',fontsize=26,weight='bold')
    fig.text(.075,.805,'64 toy examples  /  16 qubits  /  one layer  /  unchanged decoding objective',fontsize=12,color='#536877')
    base=s['methods']['full_fixed']
    for policy,color,label in [('frozen',teal,'Reuse existing angles'),('retrained',orange,'Retrain global angles')]:
        keys=['full_fixed']+[f'weak{rate}_{policy}' for rate in [10,25,50]]
        x=[s['methods'][k]['compiled_cx']['mean'] for k in keys]
        y=[100*s['methods'][k]['expected_solved_fraction_1024'] for k in keys]
        sns.lineplot(x=x,y=y,sort=False,marker='o',color=color,label=label,linewidth=2.2,ax=axs[0])
        if policy=='frozen':
            for rate,xx,yy in zip([0,10,25,50],x,y):
                axs[0].annotate(f'{rate}% removed',(xx,yy),xytext=(4,10),textcoords='offset points',fontsize=9)
    tuned=s['methods']['full_tuned']
    axs[0].scatter([tuned['compiled_cx']['mean']],[100*tuned['expected_solved_fraction_1024']],marker='D',s=50,color=ink,label='Full, tuned per instance')
    primary=s['methods']['weak25_frozen']
    axs[0].scatter([primary['compiled_cx']['mean']],[100*primary['expected_solved_fraction_1024']],s=170,facecolor='none',edgecolor=ink,linewidth=1.5,zorder=5)
    axs[0].axhline(95*base['expected_solved_fraction_1024'],color='#85939b',ls='--',lw=1,label='95% of full expected success')
    axs[0].set(xlabel='Mean compiled CX gates per circuit (all-to-all)',ylabel='Expected examples solved at 1,024 shots (%)')
    axs[0].set_title('A   Resource / solution-quality trade-off',loc='left',weight='bold',fontsize=12,pad=18)
    axs[0].legend(loc='upper left',bbox_to_anchor=(-.02,-.20),frameon=False,fontsize=9)
    ratios=np.array([r['methods']['weak25_frozen']['optimum_probability']/r['methods']['full_fixed']['optimum_probability'] for r in rows])
    order=np.argsort(ratios);r=ratios[order]*100
    colors=np.where(r>=95,teal,orange)
    axs[1].vlines(np.arange(64),100,r,color=colors,alpha=.38,lw=1)
    axs[1].scatter(np.arange(64),r,c=colors,s=21,edgecolor='white',linewidth=.4,zorder=3)
    axs[1].axhline(100,color=ink,lw=1)
    axs[1].axhline(95,color='#85939b',ls='--',lw=1)
    axs[1].set(xlabel='All 64 examples, ordered by retention',ylabel='Retained single-shot optimum probability (%)',xlim=(-2,65))
    axs[1].set_title('B   Primary test: remove weakest 25%',loc='left',weight='bold',fontsize=12,pad=18)
    text=(f"CX gates: {100*primary['mean_compiled_cx_reduction']:.1f}% fewer\n"
          f"Mean probability retained: {100*primary['single_shot_probability_retention_ratio']:.1f}%\n"
          f"Examples retaining at least 95%: {primary['instances_retaining_95pct_probability']}/64")
    axs[1].text(.03,.97,text,transform=axs[1].transAxes,va='top',fontsize=10,
                bbox={'boxstyle':'round,pad=.6','facecolor':'white','edgecolor':'#d7dede','alpha':.92})
    fig.text(.075,.105,'Exact ideal probabilities; aggregate retention is a ratio of means. All candidates are scored with the original lattice objective.',fontsize=9,color='#536877')
    fig.text(.075,.055,'Exploratory development cohort. All-to-all compilation excludes device routing and noise. No physical hardware used.',fontsize=9,color='#536877')
    for suffix in ['png','svg']:
        fig.savefig(out/f'sparsification_tradeoff.{suffix}',dpi=300,facecolor=fig.get_facecolor())
    plt.close(fig)
    manifest=json.loads((run/'manifest.json').read_text())
    for path in out.iterdir():
        manifest.setdefault('output_sha256',{})[path.relative_to(run).as_posix()]=sha256(path.read_bytes()).hexdigest()
    manifest['latest_figure']=(out/'sparsification_tradeoff.png').relative_to(run).as_posix()
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out)


if __name__=='__main__':
    main()
