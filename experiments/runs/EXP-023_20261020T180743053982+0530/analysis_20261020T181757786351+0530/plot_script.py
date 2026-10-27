"""Seaborn-styled research figures; immutable supplemental analysis folder."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

first=Path(sys.argv[1]).resolve();second=Path(sys.argv[2]).resolve()
s=json.loads((first/'summary.json').read_text());c=json.loads((second/'summary.json').read_text())
out=second/('analysis_'+datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%f%z'));out.mkdir()
sns.set_theme(style='whitegrid',context='talk',font='DejaVu Sans',
    rc={'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.edgecolor':'#B7C5CF','grid.color':'#DCE5EA','grid.linewidth':.65})
colors={'residual':'#804A86','incumbent':'#24857C','magnitude':'#D77B36','full':'#3978A1',
        'cbd_secret':'#647448','uniform_box':'#8D99A1'}
fig,axs=plt.subplots(1,3,figsize=(18,6.7))
fig.subplots_adjust(left=.06,right=.98,bottom=.2,top=.76,wspace=.39)
shots=[256,1024,4096,16384]
for name,label in [('full','Full QAOA'),('magnitude','Magnitude pruning'),('incumbent','Incumbent-informed'),
                   ('residual','Objective-guided'),('cbd_secret','Classical prior sampling')]:
    axs[0].plot(shots,[100*s['methods'][name]['by_budget'][str(b)]['hybrid_success']['mean'] for b in shots],
                marker='o',lw=2,color=colors[name],label=label,ms=5)
axs[0].axhline(100*s['classical_refine2_optimum_count']/s['count'],ls='--',color='#273D4A',lw=1.2,label='Classical local refinement')
axs[0].axhline(100*s['babai_optimum_count']/s['count'],ls=':',color='#778D9A',lw=1.3,label='Babai')
axs[0].set(xscale='log',xlabel='Samples per instance',ylabel='Expected optimum success (%)')
axs[0].set_xticks(shots,['256','1,024','4,096','16,384']);axs[0].tick_params(axis='x',labelsize=10)
axs[0].legend(fontsize=9,loc='lower right',framealpha=.9)
axs[0].set_title('A  Quality across sample budgets',loc='left',fontweight='bold',fontsize=14,pad=17)
axs[1].axvline(0,ls='--',color='#91A1AA',lw=1)
for i,b in enumerate(shots):
    v=s['paired']['residual_minus_incumbent'][str(b)]['hybrid_success']
    lo,hi=np.array(v['ci95'])*100;m=v['mean']*100
    col=colors['residual'] if b==1024 else '#7C8F9E'
    axs[1].plot([lo,hi],[i,i],color=col,lw=3,solid_capstyle='round')
    axs[1].scatter(m,i,color=col,s=75,zorder=3,marker='D' if b==1024 else 'o')
    axs[1].annotate(f'{m:+.2f} [{lo:+.2f}, {hi:+.2f}]',(m,i),xytext=(0,-23),textcoords='offset points',
                    ha='center',fontsize=10,color=col)
axs[1].set_yticks(range(4),['256','1,024*','4,096','16,384'],fontsize=11)
axs[1].set_ylim(3.65,-.6);axs[1].grid(axis='y',visible=False)
axs[1].set(xlabel='Paired gain (percentage points)',ylabel='Samples per instance')
axs[1].set_title('B  Objective-guided minus incumbent',loc='left',fontweight='bold',fontsize=14,pad=17)
names=['original','cached','incumbent','magnitude']
labels=['Original selector','Cached selector','Incumbent-informed','Magnitude']
for i,(name,label) in enumerate(zip(names,labels)):
    v=c['timing'][name];lo,hi=np.array(v['ci95'])*1000;m=v['mean']*1000
    col=colors['residual'] if name in ['original','cached'] else colors[name]
    axs[2].plot([lo,hi],[i,i],color=col,lw=4)
    axs[2].scatter(m,i,s=90,color=col,marker='D' if name=='cached' else 'o',zorder=3)
    axs[2].annotate(f'{m:,.3f} ms',(m,i),xytext=(0,-24),textcoords='offset points',
                    ha='center',fontsize=10,color=col)
axs[2].set(xscale='log',xlabel='Selection time (ms; log scale)')
axs[2].set_yticks(range(4),labels,fontsize=10);axs[2].set_ylim(3.65,-.6);axs[2].grid(axis='y',visible=False)
axs[2].set_title('C  Classical selection cost',loc='left',fontweight='bold',fontsize=14,pad=17)
for ax in axs:
    sns.despine(ax=ax);ax.tick_params(axis='y',labelsize=11);ax.xaxis.label.set_fontsize(12);ax.yaxis.label.set_fontsize(12)
fig.text(.055,.94,'Fresh validation: solution quality and selection overhead',fontsize=23,fontweight='bold',color='#173042')
fig.text(.055,.875,f'A–B: {s["count"]} fresh groups, ideal p = 1 QAOA.  C: 32 separate fresh groups × 5 timing repetitions.',fontsize=13,color='#536877')
fig.text(.055,.075,f'Cached implementation: {c["identical_mask_count"]}/{c["count"]} fresh masks and paths unchanged; '
         f'{c["speedup_ratio_of_mean_medians"]["ratio"]:.1f}× faster selection in this benchmark.',fontsize=13,color='#173042',fontweight='bold')
fig.text(.055,.026,'Paired 95% bootstrap intervals; *predeclared primary sample budget. Timing excludes shared preprocessing and compilation. No hardware execution.',fontsize=10,color='#536877')
for ext in ['pdf','svg','png']:fig.savefig(out/f'fresh_selection.{ext}',dpi=300,facecolor='white')
plt.close(fig)
(out/'plot_script.py').write_bytes(Path(__file__).read_bytes())
(out/'data.json').write_text(json.dumps({'comparison':s,'cached':c},indent=2)+'\n',encoding='utf-8')
manifest={'created_at':datetime.now().astimezone().isoformat(),
          'input_summaries':{str(p):sha256((p/'summary.json').read_bytes()).hexdigest() for p in [first,second]},
          'output_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(out)
