"""Paired group/trajectory uncertainty for the native-budget noise study."""
import numpy as np
from .device import hybrid_metrics_from_low_masses


def analyze_noise(rows,noise_loader,cfg):
    n=len(rows);B=cfg['bootstrap_replicates'];rng=np.random.default_rng(cfg['bootstrap_seed'])
    counts=rng.multinomial(n,np.full(n,1/n),size=B)
    good=np.array([r['babai_cost']==r['optimum'] for r in rows],dtype=float)
    distance=np.sqrt([r['babai_cost'] for r in rows])
    channels=['ideal','readout_only','gate_idle','gate_idle_readout']
    points={};mc={};pop={};outputs={};distances={}
    for name in cfg['methods']:
        outputs[name]={'channels':{},'resources':{}}
        for key in ['logical_k','native_cz','duration_unitary_s','depth']:
            a=np.array([r['methods'][name][key] for r in rows],dtype=float)
            outputs[name]['resources'][key]={'mean':float(a.mean()),'min':float(a.min()),'max':float(a.max()),
                'ci95':np.quantile(counts@a/n,[.025,.975]).tolist()}
        stores={}
        for i,r in enumerate(rows):
            if good[i]:continue
            raw=noise_loader(r['id'],name)
            T=len(raw['gate_idle'])
            weights=rng.multinomial(T,np.full(T,1/T),size=B)
            stores[i]=(raw,weights)
        for channel in channels:
            outputs[name]['channels'][channel]={}
            for shots in cfg['budgets']:
                key=(name,channel,shots)
                v=good.copy();d=distance.copy();fixed=np.full(B,good.mean());group=counts@good/n
                dfixed=np.full(B,distance.mean());dgroup=counts@distance/n;bias=0.
                for i,(raw,weights) in stores.items():
                    r=rows[i];a=raw[channel];mean=a.mean(axis=0)
                    resampled=weights@a/len(a) if len(a)>1 else np.repeat(mean[None,:],B,axis=0)
                    point=hybrid_metrics_from_low_masses(mean,r['babai_cost'],r['optimum'],shots)
                    boot=hybrid_metrics_from_low_masses(resampled,r['babai_cost'],r['optimum'],shots)
                    v[i]=point['hybrid_success'];d[i]=point['hybrid_expected_distance']
                    fixed+=boot['hybrid_success']/n
                    group+=counts[:,i]*boot['hybrid_success']/n
                    dfixed+=(boot['hybrid_expected_distance']-distance[i])/n
                    dgroup+=counts[:,i]*(boot['hybrid_expected_distance']-distance[i])/n
                    if len(a)>1:
                        leave=(a.sum(axis=0)-a)/(len(a)-1)
                        leave_values=hybrid_metrics_from_low_masses(leave,r['babai_cost'],r['optimum'],shots)['hybrid_success']
                        bias+=(len(a)-1)*(v[i]-float(leave_values.mean()))/n
                points[key]=v;mc[key]=fixed;pop[key]=group;distances[key]=(d,dfixed,dgroup)
                outputs[name]['channels'][channel][str(shots)]={
                    'hybrid_success':float(v.mean()),'hybrid_mean_distance':float(d.mean()),
                    'group_plus_mc_ci95':np.quantile(group,[.025,.975]).tolist(),
                    'fixed_cohort_mc_ci95':np.quantile(fixed,[.025,.975]).tolist(),
                    'jackknife_success_correction':float(bias)}
    paired={}
    for suffix in ['cap128','cap160','24']:
        a='cached'+('_' if suffix!='24' else '')+suffix
        b='incumbent'+('_' if suffix!='24' else '')+suffix
        paired[suffix]={}
        for channel in channels:
            paired[suffix][channel]={}
            for shots in cfg['budgets']:
                ka,kb=(a,channel,shots),(b,channel,shots)
                difference=float((points[ka]-points[kb]).mean())
                boot=pop[ka]-pop[kb];mcdif=mc[ka]-mc[kb]
                da,_,dbt=distances[ka];db,_,ddbt=distances[kb]
                paired[suffix][channel][str(shots)]={
                    'success_difference':difference,
                    'group_plus_mc_ci95':np.quantile(boot,[.025,.975]).tolist(),
                    'fixed_cohort_mc_ci95':np.quantile(mcdif,[.025,.975]).tolist(),
                    'mean_distance_difference':float((da-db).mean()),
                    'distance_group_plus_mc_ci95':np.quantile(dbt-ddbt,[.025,.975]).tolist()}
        # Differential noise effect uses the same bootstrap draws for each channel.
        paired[suffix]['differential_noise_effect']={}
        for shots in cfg['budgets']:
            an,bn=(a,'gate_idle_readout',shots),(b,'gate_idle_readout',shots)
            ai,bi=(a,'ideal',shots),(b,'ideal',shots)
            difference=(points[an]-points[ai])-(points[bn]-points[bi])
            boot=(pop[an]-pop[ai])-(pop[bn]-pop[bi])
            paired[suffix]['differential_noise_effect'][str(shots)]={'success_difference_change':float(difference.mean()),
                'group_plus_mc_ci95':np.quantile(boot,[.025,.975]).tolist()}
    primary=paired['cap160']['gate_idle_readout'][str(cfg['primary_budget'])]
    return {'methods':outputs,'paired':paired,'primary_improvement_supported':primary['group_plus_mc_ci95'][0]>0}
