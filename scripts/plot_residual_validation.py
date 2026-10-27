"""Supplemental paired-comparison figure; writes a new analysis folder."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

run=Path(sys.argv[1]).resolve()
summary=json.loads((run/'summary.json').read_text())
fresh=summary['cohorts']['fresh']
out=run/('analysis_'+datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%f%z'))
out.mkdir()
sns.set_theme(style='whitegrid',context='talk',font='DejaVu Sans',rc={'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
spec=[('full_trained','Full QAOA'),('magnitude24_trained','Magnitude pruning\n(primary comparison)'),
      ('magnitude24_effort','Magnitude pruning\n+ per-instance angle tuning'),
      ('geometry24_trained','Geometry-only selection')]
fig,ax=plt.subplots(figsize=(11,6.5))
fig.subplots_adjust(left=.34,right=.96,bottom=.17,top=.81)
ax.axvline(0,color='#475569',lw=1.2,ls='--',zorder=0)
record=[]
for i,(key,label) in enumerate(spec):
    stat=fresh['paired']['residual24_trained_minus_'+key]['1024']['hybrid_success']
    m=100*stat['mean'];lo,hi=[100*x for x in stat['ci95']]
    color='#14766F' if i==1 else '#77558D' if i!=3 else '#5B7280'
    ax.plot([lo,hi],[i,i],lw=3,color=color,solid_capstyle='round')
    ax.plot(m,i,marker='D' if i==1 else 'o',ms=10,color=color)
    ax.annotate(f'{m:+.2f} [{lo:+.2f}, {hi:+.2f}]',(m,i),xytext=(0,-24),textcoords='offset points',ha='center',fontsize=12,color=color)
    record.append({'comparator':key,'mean_percentage_points':m,'ci95_percentage_points':[lo,hi]})
ax.set_yticks(range(len(spec)),[x[1] for x in spec],fontsize=14)
ax.set_ylim(3.6,-.65);ax.set_xlim(-.6,2.85)
ax.set_xlabel('Success probability difference (percentage points)',labelpad=13,fontsize=14)
ax.tick_params(axis='x',labelsize=12);ax.grid(axis='y',visible=False);ax.grid(axis='x',alpha=.25)
ax.spines[['left','bottom']].set_visible(False)
fig.text(.04,.94,'Residual-guided candidate versus comparison methods',fontsize=19,fontweight='bold',color='#173042')
fig.text(.04,.885,'128 fresh groups  |  ideal QAOA, p = 1  |  1,024 samples per instance',fontsize=13,color='#536877')
fig.text(.04,.035,'Paired 95% bootstrap intervals; positive values favour the candidate. Other contrasts are secondary.',fontsize=10.5,color='#536877')
for ext in ['pdf','svg','png']:
    fig.savefig(out/f'paired_validation.{ext}',dpi=300,facecolor='white')
plt.close(fig)
(out/'data.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
(out/'plot_script.py').write_bytes(Path(__file__).read_bytes())
manifest={'created_at':datetime.now().astimezone().isoformat(),'input_summary_sha256':sha256((run/'summary.json').read_bytes()).hexdigest(),
          'source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'outputs':{p.name:sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(out)
