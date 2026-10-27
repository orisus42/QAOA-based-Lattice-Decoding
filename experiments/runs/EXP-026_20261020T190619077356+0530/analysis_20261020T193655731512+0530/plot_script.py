"""Reproducible native-budget/noise research figure in a new analysis folder."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

run=Path(sys.argv[1]).resolve();s=json.loads((run/'summary.json').read_text())
out=run/('analysis_'+datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%f%z'));out.mkdir()
sns.set_theme(style='whitegrid',context='talk',font='DejaVu Sans',
    rc={'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.edgecolor':'#B7C5CF','grid.color':'#DCE5EA','grid.linewidth':.65})
colours={'cached':'#804A86','incumbent':'#24857C'}
fig,axes=plt.subplots(1,3,figsize=(19,7))
fig.subplots_adjust(left=.13,right=.97,bottom=.2,top=.77,wspace=.45)
spec=[('cap128','CZ cap 128'),('cap160','CZ cap 160*'),('24','24 logical interactions')]
for condition,(suffix,label) in enumerate(spec):
    for offset,family in [(-.18,'cached'),(.18,'incumbent')]:
        name=family+('_' if suffix!='24' else '')+suffix
        m=s['methods'][name]['channels'];noisy=m['gate_idle_readout']['1024'];ideal=m['ideal']['1024']
        yy=condition+offset;colour=colours[family]
        x0,x1=100*ideal['hybrid_success'],100*noisy['hybrid_success']
        lo,hi=np.array(noisy['fixed_cohort_mc_ci95'])*100
        axes[0].plot([x0,x1],[yy,yy],lw=2,color=colour,alpha=.75)
        axes[0].plot([lo,hi],[yy,yy],lw=4,color=colour,alpha=.65)
        axes[0].scatter(x0,yy,s=65,facecolors='white',edgecolors=colour,lw=1.6,zorder=3)
        axes[0].scatter(x1,yy,s=65,color=colour,zorder=4)
    pair=s['paired'][suffix]
    for offset,channel in [(-.16,'ideal'),(.16,'gate_idle_readout')]:
        v=pair[channel]['1024'];m=100*v['success_difference'];lo,hi=np.array(v['group_plus_mc_ci95'])*100
        colour='#94A4AF' if channel=='ideal' else colours['cached']
        axes[1].plot([lo,hi],[condition+offset]*2,lw=2.7,color=colour,solid_capstyle='round')
        axes[1].scatter(m,condition+offset,s=65,facecolors='white' if channel=='ideal' else colour,
                        edgecolors=colour,marker='D' if suffix=='cap160' else 'o',zorder=3)
        if channel=='gate_idle_readout':
            axes[1].annotate(f'{m:+.2f} [{lo:+.2f}, {hi:+.2f}]',(m,condition+offset),
                xytext=(0,-23),textcoords='offset points',ha='center',fontsize=10,color=colour)
    coords=[]
    for family in ['cached','incumbent']:
        name=family+('_' if suffix!='24' else '')+suffix;r=s['methods'][name]['resources']
        xx=r['native_cz']['mean'];yy=1e6*r['duration_unitary_s']['mean'];coords.append((xx,yy))
        axes[2].scatter(xx,yy,s=95,color=colours[family],marker='o' if family=='cached' else 'D',zorder=3)
    axes[2].plot([v[0] for v in coords],[v[1] for v in coords],color='#B7C5CF',lw=2,zorder=1)
    midpoint=np.mean(coords,axis=0)
    axes[2].annotate(label.replace('*',''),midpoint,xytext=(-4,18),textcoords='offset points',ha='center',fontsize=10,color='#536877')
axes[0].set_yticks(range(3),[v[1] for v in spec],fontsize=11)
axes[0].set_ylim(2.65,-.6);axes[0].grid(axis='y',visible=False)
axes[0].set_xlabel('Expected optimum success (%)',fontsize=12)
axes[0].set_title('A  Ideal → calibrated noise',loc='left',fontsize=14,fontweight='bold',pad=16)
axes[0].legend(handles=[Line2D([0],[0],marker='o',color=colours[f],label=l,lw=2) for f,l in
    [('cached','Cached objective-guided'),('incumbent','Incumbent-informed')]],fontsize=9,loc='lower left',framealpha=.95)
axes[1].axvline(0,ls='--',color='#91A1AA',lw=1);axes[1].grid(axis='y',visible=False)
axes[1].set_yticks(range(3),[]);axes[1].set_ylim(2.65,-.6)
axes[1].set_xlabel('Cached − incumbent (percentage points)',fontsize=12)
axes[1].set_title('B  Paired quality difference',loc='left',fontsize=14,fontweight='bold',pad=16)
axes[1].legend(handles=[Line2D([0],[0],marker='o',color='#94A4AF',markerfacecolor='white',label='Ideal'),
    Line2D([0],[0],marker='o',color=colours['cached'],label='Full noise')],fontsize=9,loc='lower left')
axes[2].set(xlabel='Mean native CZ gates',ylabel='Mean scheduled duration (µs)')
axes[2].xaxis.label.set_fontsize(12);axes[2].yaxis.label.set_fontsize(12)
axes[2].set_title('C  Gate count ≠ duration',loc='left',fontsize=14,fontweight='bold',pad=16)
axes[2].margins(x=.22,y=.24)
for ax in axes:sns.despine(ax=ax);ax.tick_params(axis='x',labelsize=11)
fig.text(.04,.94,'Near-matched resources: QAOA decoding under calibrated noise',fontsize=22,fontweight='bold',color='#173042')
fig.text(.04,.88,f'{s["count"]} fresh groups  |  16 qubits, p = 1  |  1,024 samples  |  saved IBM Fez calibration  |  local simulation',fontsize=13,color='#536877')
fig.text(.04,.10,'Near matching: both circuits below the cap, at most 2 CZ apart. *Prespecified primary comparison.',fontsize=11,color='#536877')
fig.text(.04,.055,'A: open = ideal, filled = noisy; bars show fixed-cohort Monte Carlo intervals. B: paired 95% group + Monte Carlo intervals.',fontsize=10,color='#536877')
fig.text(.04,.018,f'Noise model: native gate errors, gate/idle relaxation and readout; {s["trajectories_per_circuit"]} trajectories per simulated circuit. No hardware execution.',fontsize=10,color='#536877')
for ext in ['pdf','svg','png']:fig.savefig(out/f'native_noise.{ext}',dpi=300,facecolor='white')
plt.close(fig)
(out/'plot_script.py').write_bytes(Path(__file__).read_bytes())
(out/'data.json').write_text(json.dumps(s,indent=2)+'\n',encoding='utf-8')
(out/'manifest.json').write_text(json.dumps({'created_at':datetime.now().astimezone().isoformat(),
    'input_summary_sha256':sha256((run/'summary.json').read_bytes()).hexdigest(),
    'outputs':{p.name:sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}},indent=2)+'\n',encoding='utf-8')
print(out)
