"""Reproducible development figures; working labels, not final poster captions."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json,sys,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parents[1]
BLUE='#235789';ORANGE='#C8622A';TEAL='#148477';GRAY='#7A8793';INK='#263645'
sns.set_theme(style='whitegrid',context='talk',font='DejaVu Sans',rc={'axes.titleweight':'bold','axes.labelcolor':INK,'text.color':INK,'grid.alpha':.25,'svg.fonttype':'none'})


def destination(run):
    out=run/('analysis_'+datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%f%z'))
    out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'plot_source.py');return out


def export(fig,out,name,footer):
    fig.text(.035,.025,footer,fontsize=10,color=INK)
    fig.savefig(out/f'{name}.svg',bbox_inches='tight')
    fig.savefig(out/f'{name}.png',bbox_inches='tight',dpi=250)
    plt.close(fig)


def manifest(out,inputs):
    data={'recorded_at':datetime.now().astimezone().isoformat(),'input_sha256':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in inputs},
        'output_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},'note':'Working diagnostic labels; poster captions and final prose belong to the user.'}
    (out/'analysis_manifest.json').write_text(json.dumps(data,indent=2)+'\n')


def sweep_figures(run):
    s=json.loads((run/'summary.json').read_text());out=destination(run)
    fig,axes=plt.subplots(1,2,figsize=(15,5.8));fig.subplots_adjust(bottom=.22,wspace=.27)
    for policy,label,color in [('same_angles','Unchanged angles',BLUE),('trained_angles','Equally budgeted angle fits',ORANGE)]:
        data=s['grid'][policy];x=np.array([int(k) for k in data]);a=np.argsort(x);x=x[a]
        y=np.array([data[str(k)]['by_budget']['1024']['hybrid_success']['mean']*100 for k in x])
        axes[0].plot(x,y,'o-',ms=3,lw=2.3,color=color,label=label)
        z=np.array([data[str(k)]['by_budget']['1024']['hybrid_success']['difference_from_old_full']*100 for k in x])
        ci=np.array([data[str(k)]['by_budget']['1024']['hybrid_success']['paired_pointwise_ci95'] for k in x])*100
        axes[1].plot(x,z,'o-',ms=3,lw=2.3,color=color,label=label)
        axes[1].fill_between(x,ci[:,0],ci[:,1],color=color,alpha=.10)
    axes[0].axhline(100*s['babai_optimum_count']/64,ls='--',color=GRAY,lw=1.4,label='Babai alone')
    axes[0].set(title='A  Full pruning range',xlabel='Interactions removed (%)',ylabel='Hybrid optimum success (%)',xlim=(-1,101))
    axes[0].legend(fontsize=11,loc='lower left')
    axes[1].axhline(0,ls='--',color=GRAY,lw=1)
    axes[1].set(title='B  Fine sweep with uncertainty',xlabel='Interactions removed (%)',ylabel='Gain over full QAOA (pp)',xlim=(59,96),ylim=(-3.7,3.7))
    sns.despine(fig=fig)
    export(fig,out,'pruning_curve','64 development instances | p=1, 16 qubits | 1,024 samples; Babai retained\nExact ideal probabilities. Shading: paired pointwise 95% bootstrap intervals; many comparisons, not an optimum claim.')
    fig,axes=plt.subplots(1,2,figsize=(15,5.8));fig.subplots_adjust(bottom=.22,wspace=.27)
    for ax,b,color in zip(axes,['1024','16384'],[BLUE,TEAL]):
        for policy,style,label in [('same_angles','--','Unchanged angles'),('trained_angles','-','Trained angles')]:
            data=s['grid'][policy];x=sorted(int(k) for k in data)
            y=[100*data[str(k)]['by_budget'][b]['hybrid_success']['mean'] for k in x]
            ax.plot(x,y,style,color=color,lw=2.3,label=label)
        ax.set(title=f'{int(b):,} samples',xlabel='Interactions removed (%)',ylabel='Hybrid optimum success (%)',xlim=(60,95))
        # Rescale to the displayed region, explicitly using different y scales.
        values=[100*s['grid'][p][str(k)]['by_budget'][b]['hybrid_success']['mean'] for p in ['same_angles','trained_angles'] for k in range(60,91)]
        ax.set_ylim(min(values)-.2,max(values)+.2);ax.legend(fontsize=11)
    sns.despine(fig=fig)
    export(fig,out,'budget_sensitivity','64 development instances | exact ideal probabilities | panels use different vertical scales\nThe best observed grid point changes with sample budget. Percentages selected from this cohort require fresh validation.')
    names=['weak75','protect_coordinate75','ties75_replicate_mean','weight90','full_scaled','split_full','random75_replicate_mean','drop_coordinate75','strong75','no_pairs100']
    labels=['Weakest 75% removed','Protect coordinate pairs, then weak','Weakest 75%, random tie order','Retain 90% of pair squared weight','Scale all couplings; retain all edges','Full circuit, separate phase angles','Random 75% removed','Coordinate pairs removed first','Strongest 75% removed','All pair interactions removed']
    fig,ax=plt.subplots(figsize=(13,7.2));fig.subplots_adjust(left=.35,bottom=.20,right=.91,top=.90)
    for i,(name,label) in enumerate(zip(names,labels)):
        r=s['mechanisms'][name];v=r['by_budget']['1024']['hybrid_success'];mean=100*v['difference_from_old_full'];lo,hi=np.array(v['paired_pointwise_ci95'])*100
        color=TEAL if mean>=0 else ORANGE
        ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=color,ms=7,capsize=4,lw=1.8)
        ax.text(1.015,i,f"{r['mean_logical_rzz']:.1f}",transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=11,color=GRAY)
    ax.text(1.015,1.03,'Mean RZZ',transform=ax.transAxes,fontsize=10,color=GRAY)
    ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Change in hybrid optimum success (percentage points)',title='Which changes reproduce the pruning benefit?')
    ax.invert_yaxis();ax.axvline(0,color=GRAY,ls='--',lw=1);ax.tick_params(axis='y',labelsize=12)
    sns.despine(fig=fig)
    export(fig,out,'mechanism_controls','64 development instances | 1,024 samples | comparator: full QAOA with 101.4 mean logical RZZ gates\nPointwise paired 95% intervals. Same original angles except the separately trained full-circuit phase-angle control.\nCoordinate-first removal also removes strong couplings: geometry and magnitude are not independently isolated.')
    manifest(out,[run/'summary.json']);print('SWEEP FIGURES:',out,flush=True)


def provenance_figures(run):
    s=json.loads((run/'summary.json').read_text());out=destination(run);keys=list(s['families']);data=s['families']
    matrix=np.array([[100*data[k]['babai_optimum_count']/32,100*data[k]['methods']['0']['1024']['hybrid_success'],100*data[k]['methods']['0.75']['1024']['hybrid_success'],100*data[k]['classical_refine2_optimum_count']/32] for k in keys])
    labels=['n=2, k=2, q=17, eta=2','n=2, k=2, q=17, eta=3','n=2, k=2, q=97, eta=2','n=2, k=2, q=3329, eta=3','n=4, k=1, q=17, eta=2','n=4, k=1, q=3329, eta=3']
    fig,ax=plt.subplots(figsize=(13,6.2));fig.subplots_adjust(left=.29,bottom=.24,right=.95,top=.88)
    sns.heatmap(matrix,annot=True,fmt='.1f',vmin=75,vmax=100,cmap=sns.light_palette(TEAL,as_cmap=True),linewidths=1.5,
        xticklabels=['Babai','+ Full QAOA','+ 75% pruned','Classical\n2-coordinate'],yticklabels=labels,cbar_kws={'label':'Optimum success (%)'},ax=ax)
    ax.set_title('Parameter transfer on tiny structured lattices',pad=18)
    ax.tick_params(axis='both',labelsize=12,rotation=0)
    export(fig,out,'parameter_sensitivity','32 new exploratory cases per family | all use 16 qubits | QAOA: 1,024 samples and unchanged prior angles\nQAOA entries are expected hybrid success; classical entries are exact counts/32. Each family has 32/32 box coverage.\nFull-size ML-KEM key-generation algebra separately verified against 75 NIST cases; no full-size decoding performed.')
    manifest(out,[run/'summary.json']);print('PROVENANCE FIGURE:',out,flush=True)


def noise_figures(run):
    s=json.loads((run/'summary.json').read_text());out=destination(run)
    keys=['ideal','readout_only','gate_channels','idle_only','gate_and_idle','gate_idle_readout']
    labels=['Ideal','Readout\nonly','Gate\nchannels','Idle\nonly','Gate +\nidle','Gate + idle\n+ readout']
    fig,axes=plt.subplots(1,2,figsize=(15,5.8));fig.subplots_adjust(bottom=.25,wspace=.25)
    for name,label,color in [('full','Full QAOA',BLUE),('weak75','75% pruned',ORANGE)]:
        y=[100*s['methods'][name][k]['1024']['hybrid_success'] for k in keys]
        ci=np.array([s['methods'][name][k]['1024']['fixed_cohort_mc_ci95'] for k in keys])*100
        x=np.arange(len(keys))+(-.1 if name=='full' else .1)
        axes[0].errorbar(x,y,yerr=[np.maximum(0,np.array(y)-ci[:,0]),np.maximum(0,ci[:,1]-np.array(y))],fmt='o-',color=color,label=label,capsize=3,lw=2)
    axes[0].set(xticks=range(6),xticklabels=labels,ylabel='Hybrid optimum success (%)',title='A  Controlled noise components')
    axes[0].legend(fontsize=11);axes[0].tick_params(axis='x',labelsize=10)
    means=np.array([s['paired_gains'][k]['1024']['gain_percentage_points'] for k in keys]);ci=np.array([s['paired_gains'][k]['1024']['pointwise_ci95_pp'] for k in keys])
    axes[1].errorbar(range(6),means,yerr=[means-ci[:,0],ci[:,1]-means],fmt='o',color=TEAL,capsize=4,lw=2)
    axes[1].axhline(0,color=GRAY,ls='--',lw=1);axes[1].set(xticks=range(6),xticklabels=labels,ylabel='Pruning gain (pp)',title='B  Benefit under each noise model')
    axes[1].tick_params(axis='x',labelsize=10);sns.despine(fig=fig)
    export(fig,out,'noise_channels','64 development cases | same angles | 1,024 samples | 128 noise trajectories/circuit; simulated IBM Fez calibration\nA: fixed-cohort Monte Carlo intervals. B: paired group + Monte Carlo intervals (95%, pointwise).\nGate channels include gate-duration relaxation; idle means scheduled delays. Noise effects need not add. No QPU execution.')
    manifest(out,[run/'summary.json']);print('NOISE FIGURE:',out,flush=True)


if __name__=='__main__':
    mode=sys.argv[1];run=Path(sys.argv[2]).resolve()
    {'sweep':sweep_figures,'provenance':provenance_figures,'noise':noise_figures}[mode](run)
