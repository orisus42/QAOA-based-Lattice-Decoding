"""Exploratory stronger controls and finite-shot planning, after the initial pilot."""
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from feasibility import OUT, SEED, scores, fit_schedule, serial
from hardware_readiness import grouped_delta


def restore(rows):
    for x in rows:
        for k in ['features','energy_norm','good','h','J']:
            x[k]=np.array(x[k])
    return rows


def main():
    original=restore(json.loads((OUT/'instances.json').read_text()))
    fresh=restore(json.loads((OUT/'replication_instances.json').read_text()))
    r=json.loads((OUT/'pilot_results.json').read_text())
    a=np.load(OUT/'noise_arrays.npz')
    byid={x['id']:x for x in original}
    train=[byid[i] for i in r['split_ids']['train']]
    cluster=make_pipeline(StandardScaler(),KMeans(n_clusters=3,n_init=10,random_state=SEED))
    cluster.fit(np.stack([x['features'] for x in train]))
    cluster_choices=cluster.predict(np.stack([x['features'] for x in fresh]))+1
    learned=a['ideal'][np.arange(200),a['choices']]
    clustered=a['ideal'][np.arange(200),cluster_choices]
    size_angles={nq:fit_schedule([x for x in train if x['nq']==nq],SEED+nq) for nq in [6,8]}
    separately_trained=np.array([scores([x],size_angles[x['nq']])[0] for x in fresh])
    result={'status':'Exploratory controls added after inspecting initial pilot; all fitting still uses original training groups only.',
        'replication_cluster_success':float(clustered.mean()),
        'replication_learned_minus_cluster':grouped_delta(learned,clustered,fresh),
        'separately_trained_size_angles':size_angles,
        'replication_separately_trained_fixed':float(separately_trained.mean()),
        'replication_learned_minus_separately_trained_fixed':grouped_delta(learned,separately_trained,fresh),
        'finite_shot_design':{}}
    rng=np.random.default_rng(SEED+101)
    ids=a['ids']; choice=a['choices'][ids]; fixed=a['fixedchoices'][ids]
    for name in ['light','moderate']:
        pn=a[name]
        for shots in [512,1536,3072]:
            lower_bounds=[]
            means=[]
            for repeat in range(400):
                counts=rng.binomial(shots,pn)
                delta=(counts[np.arange(24),choice]-counts[np.arange(24),fixed])/shots
                boots=delta[rng.integers(24,size=(1000,24))].mean(axis=1)
                lower_bounds.append(np.quantile(boots,.025))
                means.append(delta.mean())
            result['finite_shot_design'][f'{name}_{shots}_shots_per_circuit']={
                'mock_experiments':400,'fraction_positive_paired_bootstrap_lower_bound':float(np.mean(np.array(lower_bounds)>0)),
                'mean_observed_difference':float(np.mean(means)),
                'scope':'Conditional mock-experiment sensitivity on these 24 synthetic-noise instances, not actual hardware power or a new independent result.'}
    (OUT/'controls.json').write_text(json.dumps(result,indent=2,default=serial))
    print(json.dumps(result,indent=2,default=serial))


if __name__=='__main__':main()
