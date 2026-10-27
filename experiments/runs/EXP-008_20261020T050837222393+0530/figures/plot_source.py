"""Seaborn figures for the calibrated local IBM simulation; not poster prose."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

NAMES=['full_fixed','weak75_frozen','weak75_retrained']
COLORS=['#42576d','#00877f','#9064a5']
LABELS=['Full QAOA','75% / same angles','75% / trained angles']


def main():
    run=Path(sys.argv[1]).resolve();s=json.loads((run/'summary.json').read_text())
    rows=json.loads((run/'routed_instances.json').read_text());cfg=json.loads((run/'config.json').read_text())
    out=run/'figures';v=1
    while out.exists():v+=1;out=run/f'figures-v{v}'
    out.mkdir();shutil.copy2(__file__,out/'plot_source.py')
    sns.set_theme(style='whitegrid',font='DejaVu Sans',font_scale=1.03)
    plt.rcParams.update({'svg.fonttype':'none','text.color':'#173042','axes.labelcolor':'#173042',
                         'axes.edgecolor':'#d5dddc','grid.color':'#e5e9e7'})
    fig,axes=plt.subplots(2,2,figsize=(16,12))
    fig.patch.set_facecolor('#fbfaf7');fig.subplots_adjust(left=.08,right=.96,top=.79,bottom=.17,hspace=.55,wspace=.29)
    fig.text(.08,.955,'MODULE-LWE / IBM FEZ CALIBRATION STUDY',fontsize=12,weight='bold',color='#00877f')
    fig.text(.08,.90,'Smaller circuits meet device constraints',fontsize=27,weight='bold')
    fig.text(.08,.851,'512 routed examples / 16 physical qubits / fixed angles / local noisy simulation',fontsize=13,color='#536877')
    x=[r['methods']['full_fixed']['native_cz'] for r in rows]
    y=[r['methods']['weak75_frozen']['native_cz'] for r in rows]
    sns.scatterplot(x=x,y=y,color=COLORS[1],s=28,alpha=.32,edgecolor=None,ax=axes[0,0])
    maximum=max(max(x),max(y))*1.04
    axes[0,0].plot([0,maximum],[0,maximum],ls='--',color='#a2abad',lw=1)
    axes[0,0].set(xlabel='Full QAOA: native CZ gates',ylabel='75% pruning: native CZ gates',xlim=(0,maximum),ylim=(0,maximum))
    axes[0,0].set_title('A   Routing overhead still favors pruning',loc='left',weight='bold',fontsize=12,pad=15)
    full=s['methods']['full_fixed']['resources']['native_cz']['mean'];pruned=s['methods']['weak75_frozen']['resources']['native_cz']['mean']
    axes[0,0].text(.05,.90,f'Mean CZ: {full:.1f} → {pruned:.1f}\n{100*(1-pruned/full):.1f}% fewer after routing',transform=axes[0,0].transAxes,fontsize=12,weight='bold')
    values=[];cats=[]
    for name,label in zip(NAMES,LABELS):
        values += [r['methods'][name]['duration_unitary_s']*1e6 for r in rows];cats += [label]*len(rows)
    sns.boxplot(x=cats,y=values,hue=cats,palette=COLORS,legend=False,width=.5,showfliers=True,fliersize=2,ax=axes[0,1])
    axes[0,1].set(ylabel='Scheduled unitary duration (µs)',xlabel='')
    axes[0,1].set_xticks(range(3),['Full','75%\nsame angles','75%\ntrained angles'])
    axes[0,1].set_title('B   Less time exposed to gate and idle noise',loc='left',weight='bold',fontsize=12,pad=15)
    for name,label,color,shift in zip(NAMES,LABELS,COLORS,[-.035,0,.035]):
        m=s['methods'][name]['by_budget']['1024'];ideal=100*m['ideal_hybrid_success'];noisy=100*m['hybrid_success']['estimate']
        lo,hi=np.array(m['hybrid_success']['fixed_cohort_mc_ci95'])*100
        axes[1,0].plot(np.array([0,1])+shift,[ideal,noisy],marker='o',color=color,lw=2.2,label=label)
        axes[1,0].errorbar(1+shift,noisy,yerr=[[max(0,noisy-lo)],[max(0,hi-noisy)]],color=color,capsize=5,fmt='none')
    axes[1,0].axhline(100*s['babai']['success']['mean'],color='#a2abad',ls='--',lw=1,label='Babai alone')
    axes[1,0].set_xticks([0,1],['Ideal','Calibrated noise'])
    axes[1,0].set(xlim=(-.25,1.25),ylabel='Babai + QAOA exact success (%)')
    axes[1,0].set_title('C   Estimated solution quality at 1,024 shots',loc='left',weight='bold',fontsize=12,pad=15)
    for position,(name,color) in enumerate(zip(NAMES[1:],COLORS[1:])):
        m=s['comparisons_to_full'][name]['1024']['extra_distance_benefit_ratio']
        mean=m['estimate'];lo,hi=m['group_and_mc_ci95'];mlo,mhi=m['fixed_cohort_mc_ci95']
        axes[1,1].errorbar(mean,position,xerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o',color=color,capsize=7,ms=9,lw=2)
        axes[1,1].plot([mlo,mhi],[position,position],color=color,lw=6,solid_capstyle='round')
    axes[1,1].axvline(1,ls='--',color='#42576d',lw=1)
    axes[1,1].set_yticks([0,1],['75% / same','75% / trained'])
    axes[1,1].set(ylim=(-.6,1.6),xlabel='Additional distance benefit / full noisy QAOA')
    axes[1,1].set_title('D   Paired benefit beyond Babai, with uncertainty',loc='left',weight='bold',fontsize=11,pad=15)
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.071,.098),ncol=4,frameon=False,fontsize=10)
    fig.text(.08,.073,'Noise estimates use 64 trajectories per circuit. C: simulation uncertainty. D: thick = simulation; thin = paired groups + simulation (95%).',fontsize=9.8,color='#536877')
    fig.text(.08,.045,'All 512 contribute to hybrid quality; noise is simulated for all 71 improvable cases plus a random subset. Babai preserves the other 441.',fontsize=9.8,color='#536877')
    fig.text(.08,.017,'Fez: 20 Oct 2026; one patch/seed, no crosstalk or drift. Classical refinement solves 97.07%. ZERO QPU execution; no quantum advantage.',fontsize=9.8,color='#536877')
    for ext in ['svg','png']:fig.savefig(out/f'ibm_noise.{ext}',dpi=300,facecolor=fig.get_facecolor())
    plt.close(fig)
    manifest=json.loads((run/'manifest.json').read_text())
    for path in out.iterdir():manifest['output_sha256'][path.relative_to(run).as_posix()]=sha256(path.read_bytes()).hexdigest()
    manifest['latest_figure']=(out/'ibm_noise.png').relative_to(run).as_posix()
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(out)


if __name__=='__main__':main()
