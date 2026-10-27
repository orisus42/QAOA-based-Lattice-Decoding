"""Local, seeded feasibility pilot. No cloud connection or hardware submission.

Column-basis q-ary lattices; exact answers from all q**n residue codewords.
The bounded search box is centered only on the public Babai estimate.
"""
from pathlib import Path
import itertools
import json
import platform
import time

import numpy as np
from scipy.optimize import differential_evolution
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
SEED = 20261016


def babai(B, t):
    Q, R = np.linalg.qr(B.astype(float))
    v = Q.T @ t
    z = np.zeros(len(t), dtype=int)
    for k in range(len(t)-1, -1, -1):
        z[k] = np.rint((v[k] - R[k, k+1:] @ z[k+1:]) / R[k, k])
    return z


def make_instance(rng, index):
    m = 3 + index % 2
    n = m - 2
    q = int(rng.choice([11, 17, 23]))
    secrets = np.array(list(itertools.product(range(q), repeat=n)))
    while True:
        A = rng.integers(0, q, size=(m, n))
        residues = (secrets @ A.T) % q
        if len(np.unique(residues, axis=0)) == q**n:
            break
    # Every lattice point is one of these residues plus q times an integer vector.
    centered = (residues + q//2) % q - q//2
    norms2 = np.sum(centered**2, axis=1)
    lam2 = int(min(q*q, norms2[norms2 > 0].min()))
    H = hermite_normal_form(Matrix(np.column_stack([A, q*np.eye(m, dtype=int)])))
    lattice_key = str(H.tolist())
    B = np.array(H.T.lll().T).astype(int)
    assert abs(round(np.linalg.det(B))) == q**(m-n)
    chosen_secret = secrets[int(rng.integers(len(secrets)))]
    # This is a disclosed rounded-normal toy generator, not standardized ML-KEM.
    sigma_factor = float(rng.choice([0.20, 0.40, 0.70, 1.00]))
    e = np.rint(rng.normal(size=m)*sigma_factor*np.sqrt(lam2/m)).astype(int)
    t = (A @ chosen_secret + e) % q
    candidates = residues + q*np.rint((t-residues)/q).astype(int)
    distances2 = np.sum((candidates-t)**2, axis=1)
    optimum = int(distances2.min())
    global_minima = candidates[distances2 == optimum]
    z0 = babai(B, t)
    babai_cost = int(np.sum((B@z0-t)**2))
    nq = 2*m
    bits = ((np.arange(2**nq)[:, None] >> np.arange(nq)) & 1)
    W = np.zeros((m, nq), dtype=int)
    for k in range(m):
        W[k, 2*k:2*k+2] = [1, 2]
    offset = z0 - 2
    D = B @ W
    r = B @ offset - t
    # x=(1-Z)/2, hence C=constant + sum(h Z)+sum(J ZZ).
    a = r + D.sum(axis=1)/2
    h = -(a @ D)
    J = D.T @ D / 2
    J = np.triu(J, 1)
    constant = float(a@a + np.sum(D*D)/4)
    energy = np.sum((bits @ D.T + r)**2, axis=1).astype(float)
    spin = 1-2*bits
    recovered = constant + spin@h + np.einsum('bi,ij,bj->b', spin, J, spin)
    assert np.max(abs(energy-recovered)) < 1e-8
    norm = float(np.sqrt(np.sum(h*h) + np.sum(J*J)))
    energy_norm = (energy-constant)/norm
    good = np.isclose(energy, optimum)
    G = B.T @ B
    diag = np.diag(G)
    gram_corr = G / np.sqrt(diag[:,None]*diag[None,:])
    residual = t-B@z0
    # All inference features use public B,t; no error, secret, lambda1, or optimum.
    features = [m, q, np.linalg.cond(B), diag.min()/diag.mean(), diag.max()/diag.mean(),
                np.mean(abs(gram_corr-np.eye(m))), np.linalg.norm(residual)/np.sqrt(diag.mean()),
                np.max(abs(h))/norm, np.mean(abs(h))/norm, np.linalg.norm(h)/norm,
                np.max(abs(J))/norm, np.count_nonzero(J)/(nq*(nq-1)/2)]
    features += sorted(abs(np.linalg.solve(B, residual))) + [0.]*(4-m)
    return {'id': index, 'lattice_key':lattice_key, 'm':m, 'n':n, 'q':q, 'A':A, 'B':B, 't':t, 'error':e,
            'sigma_factor':sigma_factor, 'lambda1_squared':lam2, 'optimum':optimum,
            'unique_cvp':len(global_minima)==1, 'bdd_certified':bool(4*(e@e)<lam2),
            'nonzero_error':bool(e@e>0), 'babai_optimal':babai_cost==optimum,
            'babai_cost':babai_cost, 'box_covers_optimum':bool(good.any()),
            'planted_is_closest':bool(e@e==optimum), 'nq':nq,
            'features':np.array(features), 'energy_norm':energy_norm, 'good':good,
            'h':h/norm, 'J':J/norm, 'normalization':norm, 'constant':constant}


def probabilities(energy, angles):
    """Vectorized statevector QAOA, little-endian qubits, exp(-i beta X)."""
    if energy.ndim == 1:
        energy = energy[None,:]
    nq = int(np.log2(energy.shape[-1]))
    psi = np.ones(energy.shape, dtype=complex) / np.sqrt(2**nq)
    angles = np.array(angles).reshape(-1,2)
    for gamma,beta in angles:
        psi *= np.exp(-1j*gamma*energy)
        c,s = np.cos(beta), -1j*np.sin(beta)
        for k in range(nq):
            v = psi.reshape(len(psi), -1, 2, 2**k)
            u0,u1 = v[:,:,0,:].copy(), v[:,:,1,:].copy()
            v[:,:,0,:] = c*u0+s*u1
            v[:,:,1,:] = s*u0+c*u1
    return abs(psi)**2


def scores(instances, angles):
    result = np.zeros(len(instances))
    for nq in [6,8]:
        ids = [i for i,x in enumerate(instances) if x['nq']==nq]
        if ids:
            E = np.stack([instances[i]['energy_norm'] for i in ids])
            good = np.stack([instances[i]['good'] for i in ids])
            result[ids] = np.sum(probabilities(E,angles)*good, axis=1)
    return result


def fit_schedule(instances, seed):
    fit = differential_evolution(lambda a:-scores(instances,a).mean(),
            [(0,4*np.pi),(-np.pi/2,0)], seed=seed, popsize=10, maxiter=65,
            tol=1e-6, polish=True)
    return fit.x.tolist()


def bootstrap_delta(a,b, seed=SEED):
    if len(a)==0:
        return None
    delta=np.asarray(a)-np.asarray(b)
    rng=np.random.default_rng(seed)
    means=delta[rng.integers(len(delta),size=(4000,len(delta)))].mean(axis=1)
    return {'mean':float(delta.mean()),'ci95':[float(v) for v in np.quantile(means,[.025,.975])]}


def summarize(instances):
    return {'count':len(instances), **{key:sum(x[key] for x in instances) for key in
          ['bdd_certified','nonzero_error','unique_cvp','babai_optimal','box_covers_optimum','planted_is_closest']}}


def serial(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)


def main():
    start=time.time()
    rng=np.random.default_rng(SEED)
    data=[make_instance(rng,i) for i in range(600)]
    groups=np.array([x['lattice_key'] for x in data])
    train_idx,rest_idx=next(GroupShuffleSplit(n_splits=1,test_size=.4,random_state=SEED).split(data,groups=groups))
    v,t=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=SEED+1).split(rest_idx,groups=groups[rest_idx]))
    train=[data[i] for i in train_idx]
    valid=[data[i] for i in rest_idx[v]]
    test=[data[i] for i in rest_idx[t]]
    assert not (set(x['lattice_key'] for x in train)&set(x['lattice_key'] for x in test))
    assert not (set(x['lattice_key'] for x in train)&set(x['lattice_key'] for x in valid))
    assert not (set(x['lattice_key'] for x in valid)&set(x['lattice_key'] for x in test))
    print('Instance audit:',json.dumps(summarize(data)),flush=True)
    print('Nonzero certified BDD:',json.dumps(summarize([x for x in data if x['bdd_certified'] and x['nonzero_error']])),flush=True)
    X=np.stack([x['features'] for x in train])
    cluster=make_pipeline(StandardScaler(),KMeans(n_clusters=3,n_init=10,random_state=SEED))
    labels=cluster.fit_predict(X)
    schedules=[fit_schedule(train,SEED)]
    print('Fitted global schedule',schedules[0],flush=True)
    for k in range(3):
        schedules.append(fit_schedule([x for i,x in enumerate(train) if labels[i]==k],SEED+k+1))
        print('Fitted portfolio member',k+1,schedules[-1],flush=True)
    Ptrain=np.column_stack([scores(train,a) for a in schedules])
    Pvalid=np.column_stack([scores(valid,a) for a in schedules])
    Ptest=np.column_stack([scores(test,a) for a in schedules])
    Xvalid=np.stack([x['features'] for x in valid])
    Xtest=np.stack([x['features'] for x in test])
    models={
        'random_forest':RandomForestRegressor(n_estimators=250,min_samples_leaf=10,random_state=SEED,n_jobs=2),
        'extra_trees':ExtraTreesRegressor(n_estimators=250,min_samples_leaf=10,random_state=SEED,n_jobs=2),
        'nearest_neighbors':make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=20,weights='distance')),
        'decision_stump':DecisionTreeRegressor(max_depth=1,min_samples_leaf=10,random_state=SEED)}
    validation={}
    for name,model in models.items():
        model.fit(X,Ptrain)
        choices=model.predict(Xvalid).argmax(axis=1)
        validation[name]=float(Pvalid[np.arange(len(valid)),choices].mean())
    selected=max(validation,key=validation.get)
    model=models[selected]
    choices=model.predict(Xtest).argmax(axis=1)
    chosen=Ptest[np.arange(len(test)),choices]
    fixed=int(Ptrain.mean(axis=0).argmax())
    # Strong control: one fixed schedule per qubit count, chosen on training only.
    bysize={nq:int(Ptrain[[x['nq']==nq for x in train]].mean(axis=0).argmax()) for nq in [6,8]}
    sizechoices=np.array([bysize[x['nq']] for x in test])
    sizefixed=Ptest[np.arange(len(test)),sizechoices]
    bysizeq={(nq,q):int(Ptrain[[x['nq']==nq and x['q']==q for x in train]].mean(axis=0).argmax()) for nq in [6,8] for q in [11,17,23]}
    sizeqchoices=np.array([bysizeq[(x['nq'],x['q'])] for x in test])
    sizeqfixed=Ptest[np.arange(len(test)),sizeqchoices]
    clusterchoices=cluster.predict(Xtest)+1
    clusterchosen=Ptest[np.arange(len(test)),clusterchoices]
    stumpchoices=models['decision_stump'].predict(Xtest).argmax(axis=1)
    stumpchosen=Ptest[np.arange(len(test)),stumpchoices]
    uniform=np.array([x['good'].sum()/len(x['good']) for x in test])
    result={'seed':SEED,'total':summarize(data),'train':summarize(train), 'validation':summarize(valid),'test':summarize(test),
        'nonzero_bdd':summarize([x for x in data if x['bdd_certified'] and x['nonzero_error']]),
        'split_ids':{'train':[x['id'] for x in train],'validation':[x['id'] for x in valid],'test':[x['id'] for x in test]},
        'schedules':schedules,'validation_models':validation,'selected_model':selected,
        'fixed_schedule':fixed,'size_fixed_schedules':bysize,'unique_lattices':len(set(groups)),
        'split_rule':'GroupShuffleSplit using canonical Hermite-normal-form lattice basis; zero lattice overlap',
        'test_success':{'uniform':float(uniform.mean()),'fixed':float(Ptest[:,fixed].mean()),
                        'fixed_by_size':float(sizefixed.mean()),'learned':float(chosen.mean()),
                        'fixed_by_size_and_modulus':float(sizeqfixed.mean()),
                        'cluster_rule':float(clusterchosen.mean()), 'decision_stump':float(stumpchosen.mean()),
                        'portfolio_oracle':float(Ptest.max(axis=1).mean())},
        'learned_minus_fixed':bootstrap_delta(chosen,Ptest[:,fixed]),
        'learned_minus_size_fixed':bootstrap_delta(chosen,sizefixed),
        'learned_minus_size_modulus_fixed':bootstrap_delta(chosen,sizeqfixed),
        'learned_minus_cluster_rule':bootstrap_delta(chosen,clusterchosen),
        'learned_minus_stump':bootstrap_delta(chosen,stumpchosen),
        'oracle_minus_size_fixed':bootstrap_delta(Ptest.max(axis=1),sizefixed),
        'test_winner_counts':np.bincount(Ptest.argmax(axis=1),minlength=4),
        'subgroups':{}, 'elapsed_seconds':None,
        'interpretation':'Exact noiseless simulation, one exploratory seed. No hardware or quantum advantage. Test labels only used for final evaluation.'}
    for name,mask in {
        'certified_nonzero_bdd':np.array([x['bdd_certified'] and x['nonzero_error'] for x in test]),
        'outside_bdd_promise':np.array([not x['bdd_certified'] for x in test]),
        'babai_failed':np.array([not x['babai_optimal'] for x in test]),
        'six_qubits':np.array([x['nq']==6 for x in test]),
        'eight_qubits':np.array([x['nq']==8 for x in test])}.items():
        result['subgroups'][name]={'count':int(mask.sum()),'learned':float(chosen[mask].mean()) if mask.any() else None,
           'fixed_by_size':float(sizefixed[mask].mean()) if mask.any() else None,
           'delta':bootstrap_delta(chosen[mask],sizefixed[mask])}
    result['elapsed_seconds']=time.time()-start
    (OUT/'pilot_results.json').write_text(json.dumps(result,indent=2,default=serial))
    (OUT/'instances.json').write_text(json.dumps(data,default=serial))
    np.savez(OUT/'pilot_arrays.npz',Ptrain=Ptrain,Pvalid=Pvalid,Ptest=Ptest,choices=choices,sizechoices=sizechoices,sizeqchoices=sizeqchoices,stumpchoices=stumpchoices,clusterchoices=clusterchoices)
    import joblib
    joblib.dump(model,OUT/'selector.joblib')
    print(json.dumps({k:v for k,v in result.items() if k not in ['split_ids','schedules']},indent=2,default=serial),flush=True)


if __name__=='__main__': main()
