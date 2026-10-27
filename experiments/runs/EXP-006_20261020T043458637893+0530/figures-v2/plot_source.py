"""Seaborn resource/quality frontier, including a non-entangling endpoint."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    run=Path(sys.argv[1]).resolve();s=json.loads((run/'summary.json').read_text())
    assert s['experiment_id']=='EXP-006'
    out=run/'figures';i=1
    while out.exists():i+=1;out=run/f'figures-v{i}'
    out.mkdir();shutil.copy2(__file__,out/'plot_source.py')
    sns.set_theme(style='whitegrid',font='DejaVu Sans')
    plt.rcParams.update({'svg.fonttype':'none','text.color':'#173042','axes.labelcolor':'#173042',
        'grid.color':'#e5e9e7','axes.edgecolor':'#d7dede'})
    fig,axs=plt.subplots(1,2,figsize=(15,8))
    fig.patch.set_facecolor('#fbfaf7');fig.subplots_adjust(left=.08,right=.96,bottom=.25,top=.70,wspace=.3)
    fig.text(.08,.94,'MODULE-LWE / AGGRESSIVE QAOA PRUNING',fontsize=11,weight='bold',color='#087f8c')
    fig.text(.08,.865,'How far can the circuit shrink?',fontsize=26,weight='bold')
    fig.text(.08,.805,'64 toy examples  /  16 qubits  /  one layer  /  1,024 shots per example',fontsize=12,color='#536877')
    rates=[0,10,25,50,75,90,100]
    for policy,color,label in [('frozen','#087f8c','Reuse existing angles'),('retrained','#ce6a43','Retrain global angles')]:
        keys=['full_fixed']+[f'weak{r}_{policy}' for r in rates[1:]]
        for ax,metric in zip(axs,['expected_solved_fraction_1024','hybrid_distance_improvement_retention_vs_full']):
            y=[100*s['methods'][k][metric] for k in keys]
            sns.lineplot(x=rates,y=y,sort=False,marker='o',color=color,label=label,linewidth=2.2,ax=ax)
            ax.set_xticks([0,25,50,75,90,100]);ax.set_xlabel('Requested removal of two-qubit interactions (%)')
            ax.axvspan(74,91,color='#087f8c',alpha=.045)
            ax.axvline(100,color='#85939b',ls=':',lw=1)
    axs[0].set_title('A   QAOA-only exact-optimum sampling',loc='left',weight='bold',fontsize=12,pad=18)
    axs[0].set_ylabel('Expected examples solved (%)')
    axs[0].axhline(100*s['methods']['full_fixed']['expected_solved_fraction_1024'],ls='--',color='#85939b',lw=1)
    axs[1].set_title('B   Added refinement beyond Babai',loc='left',weight='bold',fontsize=12,pad=18)
    axs[1].set_ylabel('Retained mean distance improvement vs full QAOA (%)')
    axs[1].axhline(100,color='#85939b',ls='--',lw=1)
    axs[1].axhline(95,color='#85939b',ls=':',lw=1)
    handles,labels=axs[0].get_legend_handles_labels()
    for ax in axs:ax.get_legend().remove()
    fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.071,.14),ncol=2,frameon=False,fontsize=10)
    fig.text(.08,.105,'Exact ideal expectations. Panel B removes the 53-case Babai baseline effect; 100% means retaining the full circuit’s added benefit.',fontsize=9,color='#536877')
    fig.text(.08,.065,'At 100% pruning, no two-qubit interactions remain: the circuit is a classically samplable product-state control.',fontsize=9,color='#536877')
    fig.text(.08,.025,'Exploratory development cohort; interaction removals rounded down per instance. No routing, noise, or hardware execution.',fontsize=9,color='#536877')
    for ext in ['svg','png']:fig.savefig(out/f'aggressive_pruning.{ext}',dpi=300,facecolor=fig.get_facecolor())
    plt.close(fig)
    m=json.loads((run/'manifest.json').read_text())
    for path in out.iterdir():m['output_sha256'][path.relative_to(run).as_posix()]=sha256(path.read_bytes()).hexdigest()
    m['latest_figure']=(out/'aggressive_pruning.png').relative_to(run).as_posix()
    (run/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
