"""Source-linked scientific figures; captions and poster prose remain user-authored."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json
import sys
import csv
import html
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
import seaborn as sns
from threadpoolctl import threadpool_limits
from lattice_qaoa.qaoa import P1Energy

COL={"full_fixed":"#263E62","weak75_frozen":"#E28B32","weak75_retrained":"#B75B32",
     "cached24":"#8754A1","incumbent24":"#18887F"}
LABEL={"full_fixed":"Full","weak75_frozen":"75% pruned","weak75_retrained":"75% retrained",
       "cached24":"Cached","incumbent24":"Incumbent"}
RUN8=ROOT/"experiments/runs/EXP-008_20261020T050837222393+0530"
RUN23=ROOT/"experiments/runs/EXP-023_20261020T180743053982+0530"
RUN26=ROOT/"experiments/runs/EXP-026_20261020T190619077356+0530"


def read(p):return json.loads(p.read_text(encoding="utf-8"))


def setup():
    sns.set_theme(style="ticks",context="talk",font="DejaVu Sans",
       rc={"font.size":20,"axes.labelsize":23,"xtick.labelsize":19,"ytick.labelsize":19,
           "axes.titlesize":24,"legend.fontsize":18,"axes.spines.top":False,"axes.spines.right":False,
           "axes.edgecolor":"#80919C","axes.labelcolor":"#263E62","text.color":"#263E62",
           "pdf.fonttype":42,"svg.fonttype":"none","figure.facecolor":"white","savefig.facecolor":"white"})


def export(fig,out,name,data,description):
    fig.savefig(out/(name+".pdf"),bbox_inches="tight")
    fig.savefig(out/(name+".svg"),bbox_inches="tight")
    fig.savefig(out/(name+".png"),dpi=300,bbox_inches="tight")
    plt.close(fig)
    (out/(name+"_data.json")).write_text(json.dumps({"description":description,"data":data},indent=2,default=str))
    print("FIGURE",name,flush=True)


def base_figures(out,run):
    routed=read(RUN8/"routed_instances.json");noise=read(RUN8/"noisy_per_instance.json")
    nlookup={r["id"]:r for r in noise}
    methods=["full_fixed","weak75_frozen","weak75_retrained"]
    p={}
    for m in methods:
        p[m]=np.array([1. if r["babai_cost"]==r["optimum_cost"] else nlookup[r["id"]]["methods"][m]["single_shot_success_estimate"] for r in routed])
    x=np.array([100*(1-r["methods"]["weak75_frozen"]["native_cz"]/r["methods"]["full_fixed"]["native_cz"]) for r in routed])
    q={m:-np.expm1(1024*np.log1p(-np.minimum(p[m],1-1e-15))) for m in methods}
    y=100*(q["weak75_frozen"]-q["full_fixed"])
    fig,ax=plt.subplots(figsize=(16.3,7.6),layout="constrained")
    hb=ax.hexbin(x,y,gridsize=(34,23),mincnt=1,cmap=sns.color_palette("mako_r",as_cmap=True),norm=LogNorm(),linewidths=.2)
    ax.axhline(0,color="#DA9B5E",lw=1.5,ls="--",zorder=0)
    ax.set(xlabel="Native two-qubit gates removed (%)",ylabel="Change in expected success (pp)")
    cb=fig.colorbar(hb,ax=ax,pad=.02);cb.set_label("Instances")
    export(fig,out,"F01_gate_quality_atlas",{"gate_reduction_percent":x.tolist(),"success_difference_pp":y.tolist(),"n":512},
           "512 paired cases;75% unchanged-angle versus full at1024 shots, saved Fez model. Hexagonal bins show actual observations, not a fitted advantage boundary. Most Babai-optimal cases have zero hybrid difference.")
    shots=np.geomspace(1,16384,400)
    fig,ax=plt.subplots(figsize=(16.3,7.6),layout="constrained")
    curves={}
    for m in methods:
        z=100*(-np.expm1(np.outer(shots,np.log1p(-np.minimum(p[m],1-1e-15))))).mean(axis=1)
        curves[m]=z.tolist();ax.plot(shots,z,color=COL[m],lw=4,label=LABEL[m])
    babai=100*sum(r["babai_cost"]==r["optimum_cost"] for r in routed)/512
    ax.axhline(babai,color="#A0AAB2",lw=2,ls=":",label="Babai")
    ax.set(xscale="log",xlabel="Samples",ylabel="Expected hybrid optimum success (%)")
    ax.legend(loc="upper left",frameon=False)
    export(fig,out,"F02_dense_shot_response",{"shots":shots.tolist(),"curves_percent":curves},
           "400 analytical evaluations of mean[1-(1-p_i)^N], retaining Babai per instance. Conditional on saved finite-trajectory probabilities; curves are model predictions, not400 independent experiments. No interpolation between a few shot outcomes.")
    timings=read(RUN23/"per_instance.json")[:32]
    names=["original","cached","incumbent","magnitude"];colors=["#263E62","#8754A1","#18887F","#A1ABB3"]
    values=[np.array([np.median(r["timing_seconds"][m]) for r in timings]) for m in names]
    logs=[np.log10(v) for v in values]
    fig,ax=plt.subplots(figsize=(16.3,7.6),layout="constrained")
    parts=ax.violinplot(logs,positions=range(4),widths=.68,showextrema=False,points=180)
    for b,c in zip(parts["bodies"],colors):b.set_facecolor(c);b.set_edgecolor(c);b.set_alpha(.28)
    rng=np.random.default_rng(20261020292)
    for i,(v,c) in enumerate(zip(logs,colors)):
        ax.scatter(i+rng.uniform(-.13,.13,len(v)),v,s=43,color=c,alpha=.8,zorder=3)
        ax.plot([i-.18,i+.18],[np.median(v)]*2,color=c,lw=4)
    ax.set_xticks(range(4),["Original","Cached","Incumbent","Magnitude"])
    ax.set_yticks([-4,-3,-2,-1,0,1],["0.1 ms","1 ms","10 ms","100 ms","1 s","10 s"])
    ax.set(ylabel="Classical selection time",ylim=(-4.5,.7))
    export(fig,out,"F03_selection_raincloud",{m:v.tolist() for m,v in zip(names,values)},
           "32 cases; each point is the median of5 interleaved repetitions. Violin density is descriptive in log time; all raw points shown. Cheap methods remain much faster. Cached/original circuits validated separately.")
    # Exact primary uncertainty from the stronger native-budget control.
    s=read(RUN26/"summary.json")
    fig,ax=plt.subplots(figsize=(16.3,6.3),layout="constrained")
    specs=[("cap128","128 CZ cap"),("cap160","160 CZ cap"),("24","24 interactions")]
    vals=[]
    for i,(key,label) in enumerate(specs):
        v=s["paired"][key]["gate_idle_readout"]["1024"]
        mid=100*v["success_difference"];lo,hi=np.array(v["group_plus_mc_ci95"])*100
        ax.plot([lo,hi],[i,i],color=COL["cached24"],lw=7,solid_capstyle="round",alpha=.65)
        ax.scatter(mid,i,s=200,color=COL["cached24"],marker="D" if key=="cap160" else "o",zorder=3)
        vals.append({"condition":label,"estimate_pp":mid,"ci95_pp":[lo,hi]})
    ax.axvline(0,color="#80919C",lw=1.5,ls="--")
    ax.set_yticks(range(3),[s[1] for s in specs]);ax.invert_yaxis()
    ax.set(xlabel="Cached − incumbent success (percentage points)",ylim=(2.6,-.6))
    export(fig,out,"F04_strong_control",vals,"128 fresh groups; paired95% group+Monte-Carlo intervals. Diamond is prespecified primary. Preserve this null/unfavorable comparison. Connecting observations into a smooth curve would be invalid.")
    rows=read(run/"rows.json");ident=rows[0]["id"]
    selected=["full_fixed","weak75_frozen","cached24"]
    datasets={m:np.load(run/"circuits"/(ident+"_"+m+".npz")) for m in selected}
    angles=np.linspace(0,2*np.pi,16,endpoint=False)+np.pi/2
    xy=np.c_[np.cos(angles),np.sin(angles)]
    fig,axes=plt.subplots(1,3,figsize=(16.3,6.4),layout="constrained")
    maxj=max(np.max(abs(d["J"])) for d in datasets.values())
    for ax,m in zip(axes,selected):
        d=datasets[m];J=d["J"]
        for i,j in zip(*np.nonzero(J)):
            path=MPath([xy[i],np.zeros(2),xy[j]],[MPath.MOVETO,MPath.CURVE3,MPath.CURVE3])
            ax.add_patch(PathPatch(path,fill=False,color="#8754A1" if J[i,j]>0 else "#18887F",
                                  alpha=.22+.5*abs(J[i,j])/maxj,lw=.5+3*abs(J[i,j])/maxj))
        ax.scatter(xy[:,0],xy[:,1],s=100,c=d["h"],cmap=sns.diverging_palette(165,285,as_cmap=True),
                   edgecolors="white",linewidths=1.3,zorder=4)
        for i,(xx,yy) in enumerate(xy):ax.text(1.13*xx,1.13*yy,str(i),ha="center",va="center",fontsize=14)
        ax.set_title(LABEL[m]+f"\n{np.count_nonzero(J)} interactions",fontsize=22,pad=8)
        ax.set(xlim=(-1.38,1.38),ylim=(-1.38,1.38),aspect="equal");ax.axis("off")
    export(fig,out,"F05_quantum_interaction_orbits",{"case":ident,"methods":selected},
           "Actual submitted logical Ising couplings. Same16-qubit circular layout in all panels; edge width reflects|J|,edge color sign,node color local field. This is a logical interaction graph, not physical hardware topology.")
    gamma=np.linspace(0,3.2,121);beta=np.linspace(-.8,0,101)
    grids={};minimum=np.inf;maximum=-np.inf
    for m in selected:
        d=datasets[m];full=datasets["full_fixed"]
        fun=P1Energy(d["h"],d["J"],observable_h=full["h"],observable_J=full["J"])
        z=np.array([[fun([g,b]) for g in gamma] for b in beta])
        grids[m]=z;minimum=min(minimum,z.min());maximum=max(maximum,z.max())
    fig,axes=plt.subplots(1,3,figsize=(16.3,6.4),layout="constrained")
    for ax,m in zip(axes,selected):
        im=ax.pcolormesh(gamma,beta,grids[m],cmap=sns.color_palette("mako",as_cmap=True),vmin=minimum,vmax=maximum,shading="auto",rasterized=True)
        ax.contour(gamma,beta,grids[m],levels=8,colors="white",linewidths=.6,alpha=.35)
        a=datasets[m]["angles"];ax.scatter(a[0],a[1],marker="*",s=260,color="#F2B44B",edgecolor="white",lw=.7)
        ax.set(xlabel=r"$\gamma$",title=LABEL[m])
    axes[0].set_ylabel(r"$\beta$")
    cb=fig.colorbar(im,ax=axes,pad=.02,shrink=.85);cb.set_label(r"$\langle H_{\rm full}\rangle$",fontsize=22)
    export(fig,out,"F06_qaoa_phase_landscape",{"case":ident,"gamma":gamma.tolist(),"beta":beta.tolist(),"energy":{k:v.tolist() for k,v in grids.items()}},
           "Exact p=1 analytic expectation of the ORIGINAL normalized full Hamiltonian on a121x101 grid. Stars show frozen submitted angles, not newly optimized ones. Same color scale across panels; diagnostic on one preselected case.")
    export_tables(out,run)


def export_tables(out,run):
    evidence={"simulation512":read(RUN8/"summary.json"),"cache128":read(RUN23/"summary.json"),
              "matched_noise128":read(RUN26/"summary.json"),"hardware_run":str(run.relative_to(ROOT))}
    (out/"verified_evidence.json").write_text(json.dumps(evidence,indent=2))
    rows=read(run/"rows.json")
    with (out/"hardware_circuit_resources.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["id","stratum","method","native_cz","logical_zz","depth","unitary_duration_seconds"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:r[k] for k in fields})


def hardware_figures(out,run):
    analysis=Path((run/"latest_analysis_path.txt").read_text())
    rows=read(analysis/"per_circuit.json");s=read(analysis/"summary.json")
    methods=list(COL);ids=s["failure_case_ids"];lookup={(r["id"],r["method"]):r for r in rows}
    # Measured low-energy populations: each cell retains the rare event count.
    matrix=np.array([[lookup[(i,m)]["improve"]["count"] for i in ids] for m in methods])
    fig,ax=plt.subplots(figsize=(16.3,7.2),layout="constrained")
    sns.heatmap(matrix,annot=True,fmt="d",cmap=sns.color_palette("mako_r",as_cmap=True),linewidths=3,linecolor="white",
       xticklabels=[i.split("-")[-1] for i in ids],yticklabels=[LABEL[m] for m in methods],
       cbar_kws={"label":"Improving measured bitstrings"},annot_kws={"fontsize":24},ax=ax)
    ax.set(xlabel="Fixed difficult instance",ylabel="")
    ax.tick_params(axis="y",rotation=0)
    export(fig,out,"F07_hardware_event_map",{"counts":matrix.tolist(),"cases":ids,"methods":methods,"shots_per_cell":rows[0]["shots"]},
           "Actual hardware strict improvements over Babai, equal shots per cell. Six conditional failure cases, not a population accuracy estimate. Rare events require intervals in supporting table.")
    # Energy probability carpet, using normalized excess cost and shared bins.
    bins=np.linspace(0,90,91);carpet=[];raw=[]
    for m in methods:
        dist=np.zeros(90)
        for ident in ids:
            r=lookup[(ident,m)]
            cost=np.array([float(c) for c in r["cost_histogram"]]);count=np.array(list(r["cost_histogram"].values()))
            excess=(cost-r["optimum_cost"])/max(1,r["babai_cost"]-r["optimum_cost"])
            # Explicit overflow in the final bin, not silently omitted.
            excess=np.minimum(excess,89.999)
            dist+=np.histogram(excess,bins,weights=count)[0]/r["shots"]/6
        carpet.append(dist);raw.append(dist.tolist())
    fig,ax=plt.subplots(figsize=(16.3,6.2),layout="constrained")
    im=ax.imshow(np.maximum(carpet,1e-7),aspect="auto",cmap=sns.color_palette("mako",as_cmap=True),norm=LogNorm(vmin=1e-5,vmax=.12),extent=[0,90,4.5,-.5],interpolation="nearest")
    ax.set_yticks(range(5),[LABEL[m] for m in methods]);ax.axvline(1,color="#F3B33D",lw=2,ls=":")
    ax.set(xlabel="Energy above optimum / Babai gap",ylabel="")
    cb=fig.colorbar(im,ax=ax,pad=.02);cb.set_label("Measured probability")
    export(fig,out,"F08_quantum_energy_carpet",{"probabilities":raw,"bin_edges":bins.tolist(),"methods":methods},
           "Measured cost populations, averaged equally over six difficult cases. Original full objective used for every method. Gold line marks Babai cost; final bin includes overflow. Log color floor is only display, raw probabilities preserved.")
    # Case-level hardware probability versus quantum circuit footprint.
    fig,ax=plt.subplots(figsize=(16.3,7.6),layout="constrained")
    for m in methods:
        rr=[lookup[(i,m)] for i in ids]
        xx=[r["native_cz"] for r in rr];yy=[1e4*r["improve"]["probability"] for r in rr]
        ci=np.array([r["improve"]["clopper_pearson_ci95"] for r in rr])*1e4
        ax.errorbar(xx,yy,yerr=[np.array(yy)-ci[:,0],ci[:,1]-np.array(yy)],fmt="o",ms=9,
                    color=COL[m],alpha=.8,elinewidth=1.2,capsize=2,label=LABEL[m])
    ax.set(xlabel="Native two-qubit gates",ylabel="Improving samples per 10,000 shots")
    ax.legend(frameon=False,ncol=2,loc="upper right",fontsize=16)
    export(fig,out,"F09_hardware_resource_quality",s["methods"],
           "Actual hardware per-case strict-improvement frequencies with exact95% binomial intervals. Gate counts paired by case; curves would be inappropriate for these discrete policies. Conditional6-case cohort.")
    # A directly measured quantum-observable view, without an entanglement claim.
    ident=ids[0];chosen=["full_fixed","weak75_frozen","incumbent24"]
    inputs=[run,Path((ROOT/"project/repeat_hardware_run.txt").read_text())]
    correlations={}
    for m in chosen:
        samples=[]
        for rr in inputs:
            rrrows=read(rr/"rows.json")
            ri=next(i for i,r in enumerate(rrrows) if r["id"]==ident and r["method"]==m)
            for pi,pub in enumerate(read(rr/"pub_order.json")):
                if pub["row_index"]==ri:samples.append(np.load(rr/"counts"/f"pub_{pi:03d}_ordered.npy"))
        a=np.concatenate(samples);z=1.-2*((a[:,None].astype(np.int64)>>np.arange(16))&1)
        mean=z.mean(axis=0);c=z.T@z/len(z)-np.outer(mean,mean)
        np.fill_diagonal(c,np.nan);correlations[m]=c
    bound=max(np.nanmax(abs(c)) for c in correlations.values())
    fig,axes=plt.subplots(1,3,figsize=(16.3,6.2),layout="constrained")
    cmap=sns.diverging_palette(270,35,s=85,l=48,as_cmap=True)
    for ax,m in zip(axes,chosen):
        im=ax.imshow(correlations[m],vmin=-bound,vmax=bound,cmap=cmap,interpolation="nearest")
        ax.set_xticks([0,5,10,15]);ax.set_yticks([0,5,10,15]);ax.set(xlabel="Qubit",title=LABEL[m])
    axes[0].set_ylabel("Qubit")
    cb=fig.colorbar(im,ax=axes,pad=.02,shrink=.8);cb.set_label(r"$\langle Z_iZ_j\rangle-\langle Z_i\rangle\langle Z_j\rangle$",fontsize=18)
    export(fig,out,"F10_measured_Z_correlations",{"case":ident,"matrices":{k:np.nan_to_num(v).tolist() for k,v in correlations.items()}},
           "Measured connected Z-basis correlations on the first prespecified difficult case. Diagonal masked. Same color scale. Classical correlations and noise can produce this pattern; it is not an entanglement witness or proof of quantum advantage.")
    if (run/"completed_noise_path.txt").exists():
        nr=Path((run/"completed_noise_path.txt").read_text());noise={(r["id"],r["method"]):r for r in read(nr/"summary.json")}
        panels={"Ideal":np.array([[lookup[(i,m)]["ideal_p_at_most_babai"] for i in ids] for m in methods]),
                "Noise model":np.array([[noise[(i,m)]["p_at_most_babai"] for i in ids] for m in methods]),
                "Hardware":np.array([[lookup[(i,m)]["at_most_babai"]["probability"] for i in ids] for m in methods])}
        low=max(1e-5,min(v[v>0].min() for v in panels.values()));high=max(v.max() for v in panels.values())
        fig,axes=plt.subplots(1,3,figsize=(16.3,6.5),layout="constrained")
        for ax,(label,values) in zip(axes,panels.items()):
            im=ax.imshow(np.maximum(values,low),cmap=sns.color_palette("mako",as_cmap=True),norm=LogNorm(vmin=low,vmax=high),aspect="auto",interpolation="nearest")
            ax.set_xticks(range(6),[i.split("-")[-1] for i in ids],rotation=90,fontsize=14)
            ax.set_yticks(range(5),[LABEL[m] for m in methods] if ax==axes[0] else [""]*5,fontsize=17)
            ax.set(xlabel="Instance",title=label)
        cb=fig.colorbar(im,ax=axes,pad=.02,shrink=.85);cb.set_label(r"$P(C\leq C_{\rm Babai})$",fontsize=22)
        export(fig,out,"F11_prediction_to_hardware",{"panels":{k:v.tolist() for k,v in panels.items()},"cases":ids,"methods":methods},
               "Exact ideal probabilities,64-trajectory fresh Marrakesh calibration model, and actual pooled hardware frequencies for the same frozen circuits. Shared log color scale; zero values use the display floor. Model mismatch is retained, not recalibrated to results. Uncertainty is in supporting per-circuit records.")
    (out/"hardware_summary.json").write_text(json.dumps(s,indent=2))
    (out/"hardware_per_circuit.json").write_text(json.dumps(rows,indent=2))


def index(out):
    figures=sorted(out.glob("F*.png"))
    body="".join(f'<article><h2>{html.escape(p.stem.replace("_"," "))}</h2><img src="{p.name}"><p><a href="{p.stem}.pdf">Vector PDF</a> · <a href="{p.stem}.svg">SVG</a> · <a href="{p.stem}_data.json">Data and interpretation notes</a></p></article>' for p in figures)
    text='<!doctype html><meta charset="utf-8"><title>Research figure review</title><style>body{max-width:1100px;margin:40px auto;background:#eef3f6;color:#263e62;font:18px system-ui}article{background:white;padding:24px;margin:24px 0;border-radius:16px}img{width:100%}a{color:#18887f}h1{font-size:36px}</style><h1>Research figure review</h1><p>Figure assets and research notes. Final poster captions and prose are written by the presenter. Choose a few main figures; retain the rest as supporting material.</p>'+body
    (out/"index.html").write_text(text,encoding="utf-8")
    (out/"plot_source.py").write_bytes(Path(__file__).read_bytes())
    (out/"manifest.json").write_text(json.dumps({"at":datetime.now().astimezone().isoformat(),"files_sha256":{p.name:sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!="manifest.json"}},indent=2))


def main():
    setup();run=Path((ROOT/"project/current_hardware_run.txt").read_text())
    if len(sys.argv)>1 and sys.argv[1]=="hardware":
        out=Path((ROOT/"project/current_poster_assets.txt").read_text());hardware_figures(out,run)
    else:
        out=ROOT/"deliverables"/("poster_assets_"+datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z"));out.mkdir(parents=True)
        (ROOT/"project/current_poster_assets.txt").write_text(str(out))
        print("OUTPUT",out,flush=True);base_figures(out,run)
    index(out);print("READY",out,flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1,user_api="blas"):main()
