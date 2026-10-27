"""Show complete-cohort outcomes and explicit conditional distance refinements."""
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
    s=json.loads((run/'summary.json').read_text());rows=json.loads((run/'per_instance.json').read_text())
    assert s['experiment_id']=='EXP-005'
    out=run/'figures';i=1
    while out.exists():
        i+=1;out=run/f'figures-v{i}'
    out.mkdir();shutil.copy2(__file__,out/'plot_source.py')
    sns.set_theme(style='whitegrid',font='DejaVu Sans')
    plt.rcParams.update({'svg.fonttype':'none','text.color':'#173042','axes.labelcolor':'#173042',
        'grid.color':'#e5e9e7','axes.edgecolor':'#d7dede'})
    ink,teal,orange,purple='#173042','#087f8c','#ce6a43','#71639e'
    fig,axs=plt.subplots(1,2,figsize=(15,8),gridspec_kw={'width_ratios':[1.1,1]})
    fig.patch.set_facecolor('#fbfaf7')
    fig.subplots_adjust(left=.09,right=.94,bottom=.23,top=.71,wspace=.65)
    fig.text(.07,.94,'MODULE-LWE / BABAI + QAOA',fontsize=11,weight='bold',color=teal)
    fig.text(.07,.865,'Can QAOA refine Babai’s answer?',fontsize=25,weight='bold')
    fig.text(.07,.805,'All 64 examples retained  /  53 already optimal  /  11 have room to improve',fontsize=12,color='#536877')
    failed=[r for r in rows if not r['babai_optimal']]
    x=np.arange(len(failed))
    for key,color,label,offset in [('full_fixed',teal,'Full QAOA',-.18),('weak25_frozen',orange,'25% pruned',0),('weak50_frozen',purple,'50% pruned',.18)]:
        gain=[100*(1-r['methods'][key]['by_budget']['1024']['expected_distance']/r['babai_distance']) for r in failed]
        axs[0].scatter(gain,x+offset,label=label,color=color,s=42,zorder=3)
        axs[0].hlines(x+offset,0,gain,color=color,alpha=.25,lw=2)
    axs[0].set_yticks(x,[r['id'].replace('MLWE-','') for r in failed])
    axs[0].invert_yaxis()
    axs[0].set(xlabel='Expected reduction in returned distance (%)',ylabel='Instance (the 11 nonoptimal Babai cases)',xlim=(0,None))
    axs[0].set_title('A   Where the distance improves',loc='left',fontsize=12,weight='bold',pad=17)
    axs[0].legend(loc='upper left',bbox_to_anchor=(-.03,-.17),ncol=3,frameon=False,fontsize=9,columnspacing=1)
    keys=['babai','full_fixed','weak25_frozen','weak50_frozen','refine_2']
    labels=['Babai alone','+ full QAOA','+ 25% pruned','+ 50% pruned','+ classical refinement']
    colors=[ink,teal,orange,purple,'#4b7962']
    means=[];low=[];high=[]
    for key in keys:
        if key=='babai':
            means.append(53);low.append(53);high.append(53)
        else:
            m=s['methods'][key]['all64'];means.append(m['expected_solved_count'])
            lo,hi=m['solved_count_prediction_interval_95pct'];low.append(lo);high.append(hi)
    yy=np.arange(len(keys))
    for j,(mu,lo,hi,color) in enumerate(zip(means,low,high,colors)):
        axs[1].hlines(j,lo,hi,color=color,lw=2,alpha=.65)
        axs[1].scatter(mu,j,color=color,s=70,zorder=3)
        axs[1].annotate(f'{mu:.2f}',(mu,j),xytext=(0,12),textcoords='offset points',ha='center',color=color,fontsize=10)
    axs[1].set_yticks(yy,labels);axs[1].invert_yaxis()
    axs[1].set(xlabel='Examples with an exact optimum (out of 64)',xlim=(50.5,64.5),ylim=(4.6,-.7))
    axs[1].set_title('B   Full-cohort optimum recovery',loc='left',fontsize=12,weight='bold',pad=17)
    axs[1].axvline(53,color=ink,ls=':',lw=1)
    axs[1].axvline(64,color='#85939b',ls=':',lw=1)
    fig.text(.07,.085,'QAOA: exact ideal expectations for 1,024 shots per example, with existing shared angles. Babai is always retained.',fontsize=9,color='#536877')
    fig.text(.07,.045,'Intervals: 95% shot-variability predictions on these 64 cases. Augmented lattice distance; no hardware or speedup claim.',fontsize=9,color='#536877')
    for suffix in ['png','svg']:
        fig.savefig(out/f'hybrid_refinement.{suffix}',dpi=300,facecolor=fig.get_facecolor())
    plt.close(fig)
    manifest=json.loads((run/'manifest.json').read_text())
    for path in out.iterdir():
        manifest['output_sha256'][path.relative_to(run).as_posix()]=sha256(path.read_bytes()).hexdigest()
    manifest['latest_figure']=(out/'hybrid_refinement.png').relative_to(run).as_posix()
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out)


if __name__=='__main__':
    main()
