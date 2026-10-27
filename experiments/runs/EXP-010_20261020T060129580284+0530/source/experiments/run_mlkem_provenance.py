"""EXP-010: public FIPS203 KAT verification and tiny parameter sensitivity."""
from pathlib import Path
from hashlib import sha256
from itertools import product
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from lattice_qaoa.instances import Parameters,generate_instance,module_product,expand_module_matrix
from lattice_qaoa.reference import embedding,lll_reduce,exact_cvp,squared_cost
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.classical import lift_secrets,local_refine
from lattice_qaoa.qaoa import state_probabilities,P1Energy
from lattice_qaoa.sparsify import prune_interactions,diagonal_energy
from lattice_qaoa.mechanisms import CachedMetrics
from lattice_qaoa.evaluation import matrix_group
from research_run import ResearchRun,save


def main():
    cfg=json.loads((ROOT/'experiments/configs/EXP-010.json').read_text())
    vend=ROOT/cfg['kyber_source'];nist=ROOT/cfg['acvp_source']
    run=ResearchRun('EXP-010',Path(__file__),list(vend.rglob('*.py'))+[vend/'LICENSE',vend/'source_manifest.json']+list(nist.rglob('*.json')))
    try:
        sys.path.insert(0,str(vend/'src'))
        from kyber_py.ml_kem.default_parameters import ML_KEM_512,ML_KEM_768,ML_KEM_1024
        models={'ML-KEM-512':ML_KEM_512,'ML-KEM-768':ML_KEM_768,'ML-KEM-1024':ML_KEM_1024}
        vectors=json.loads((nist/'gen-val/json-files/ML-KEM-keyGen-FIPS203/internalProjection.json').read_text())
        kat=[];examples=[]
        for group in vectors['testGroups']:
            model=models[group['parameterSet']];k=model.k
            for ti,test in enumerate(group['tests']):
                d,z=bytes.fromhex(test['d']),bytes.fromhex(test['z'])
                ek,dk=model._keygen_internal(d,z)
                assert ek==bytes.fromhex(test['ek']) and dk==bytes.fromhex(test['dk'])
                rho,sigma=model._G(d+bytes([k]));assert ek[-32:]==rho
                ah=model._generate_matrix_from_seed(rho).from_ntt()
                sh=model.M.decode_vector(dk[:384*k],k,12,is_ntt=True).from_ntt()
                th=model.M.decode_vector(ek[:-32],k,12,is_ntt=True).from_ntt()
                sv,counter=model._generate_error_vector(sigma,model.eta_1,0)
                ev,_=model._generate_error_vector(sigma,model.eta_1,counter)
                A=np.array([[ah[i,j].coeffs for j in range(k)] for i in range(k)],dtype=np.int64)
                center=lambda a:(np.array(a,dtype=np.int64)+1664)%3329-1664
                s=center([sh[i,0].coeffs for i in range(k)])
                e=center([ev[i,0].coeffs for i in range(k)])
                t=np.array([th[i,0].coeffs for i in range(k)],dtype=np.int64)
                np.testing.assert_array_equal(s,center([sv[i,0].coeffs for i in range(k)]))
                assert abs(s).max()<=model.eta_1 and abs(e).max()<=model.eta_1
                # Independent coefficient convolution and signed-shift integer map.
                direct=module_product(A,s)
                expanded=expand_module_matrix(A)
                np.testing.assert_array_equal(direct.ravel(),expanded@s.ravel())
                np.testing.assert_array_equal((direct+e)%3329,t)
                w=(t-direct-e)//3329
                np.testing.assert_array_equal(3329*w+direct-t,-e)
                kat.append({'parameter_set':group['parameterSet'],'tcId':test['tcId'],'ek_sha256':sha256(ek).hexdigest(),
                    'key_bytes_match':True,'coefficient_equation_match':True,'expanded_equation_match':True,
                    'n':256,'k':k,'q':3329,'eta1':model.eta_1,'augmented_planted_cost':int(np.sum(s*s)+np.sum(e*e))})
                if ti==0:examples.append({'parameter_set':group['parameterSet'],'tcId':test['tcId'],'A_polynomials':A.tolist(),'secret':s.tolist(),'error':e.tolist(),'public_target':t.tolist(),'wrap':w.tolist(),'source':'PUBLIC NIST test vector; verification data only'})
            print('KAT verified:',group['parameterSet'],len(group['tests']),flush=True)
        save(run.path/'nist_kat_checks.json',kat);save(run.path/'nist_coefficient_examples.json',examples)
        rows=[];instances=[];groups_seen=set();maxdiff=0.
        for fi,spec in enumerate(cfg['families']):
            params=Parameters(**spec);family='_'.join(f'{k}{v}' for k,v in spec.items())
            support=np.array(list(product(range(-params.eta,params.eta+1),repeat=params.n*params.k)),dtype=np.int64)
            for i in range(cfg['count_per_family']):
                x=generate_instance(cfg['first_seed']+1000*fi+i,params)
                A=expand_module_matrix(x.A);t=x.target.ravel();B,y=embedding(A,t,params.q);R,U=lll_reduce(B)
                box=encode_babai_box(R,y);z=U@box.center;inc=squared_cost(B,y,z)
                group=(params.q,params.n,params.k,matrix_group(A,params.q));assert group not in groups_seen;groups_seen.add(group)
                coords,costs=lift_secrets(A,t,params.q,support);bound=coords[costs.argmin()]
                if inc<int(costs.min()):bound=z
                reference=exact_cvp(A,t,params.q,bound,cfg['exact_reference_candidate_cap'])
                optimum=reference['cost'];secret=x.secret.ravel();error=x.error.ravel()
                decoded=box.decode(np.arange(len(box.costs)))@U.T
                secret_mask=np.all(decoded[:,len(t):]==secret,axis=1)
                babai_secret=bool(np.array_equal(z[len(t):],secret))
                refsecret=any(np.array_equal(v[len(t):],secret) for v in reference['minimizers'])
                row={'id':f'{family}-{i:03d}','family':family,'parameters':spec,'seed':x.seed,'group':group[-1],
                    'babai_cost':inc,'optimum_cost':optimum,'babai_secret_recovered':babai_secret,
                    'planted_cost':int(secret@secret+error@error),'planted_secret_in_optimum_set':refsecret,
                    'box_covers_optimum':bool(np.any(box.costs==optimum)),
                    'box_contains_planted_secret':bool(np.any(secret_mask)),'methods':{}}
                metrics=CachedMetrics(box.costs,inc,optimum,cfg['budgets'],secret_mask,babai_secret)
                for fraction in cfg['fractions']:
                    J,_=prune_interactions(box.J,fraction)
                    p=state_probabilities(diagonal_energy(box.h,J),cfg['angles'])
                    diff=abs(p@box.energy-P1Energy(box.h,J,observable_J=box.J)(cfg['angles']));assert diff<1e-9;maxdiff=max(maxdiff,float(diff))
                    row['methods'][str(fraction)]=metrics(p)
                local=local_refine(R,y,box.center,2,1024)
                mapped=U@np.array(local['coefficients'])
                row['classical_refine2']={**local,'secret_recovered':bool(np.array_equal(mapped[len(t):],secret))}
                rows.append(row)
                instances.append({'id':row['id'],'A_polynomials':x.A.tolist(),'public_target':t.tolist(),'secret':secret.tolist(),'error':error.tolist(),
                    'reduced_basis':R.tolist(),'basis_transform':U.tolist(),'reference':reference})
            save(run.path/'per_instance.json',rows);save(run.path/'instances.json',instances)
            print('Family complete:',family,len(rows),flush=True)
        summary={'experiment_id':'EXP-010','nist_keygen_vectors_verified':len(kat),'families':{},'max_analytic_discrepancy':maxdiff,
            'qpu_seconds':0,'limits':['KAT agreement is not a certification or secure decoding experiment.','Tiny lower-degree analogues preserve negacyclic module algebra but do not inherit ML-KEM security.','Euclidean objective is a surrogate for CBD likelihood.','Parameter transfer is exploratory; all QAOA angles remain fixed from prior training.','Secret output retains Babai on cost ties and otherwise uses the first sample attaining the minimum cost.']}
        for family in dict.fromkeys(r['family'] for r in rows):
            selected=[r for r in rows if r['family']==family]
            summary['families'][family]={'count':len(selected),'babai_optimum_count':sum(r['babai_cost']==r['optimum_cost'] for r in selected),
                'babai_secret_count':sum(r['babai_secret_recovered'] for r in selected),
                'planted_secret_in_optimum_count':sum(r['planted_secret_in_optimum_set'] for r in selected),
                'box_coverage_count':sum(r['box_covers_optimum'] for r in selected),
                'box_secret_coverage_count':sum(r['box_contains_planted_secret'] for r in selected),
                'classical_refine2_optimum_count':sum(r['classical_refine2']['cost']==r['optimum_cost'] for r in selected),
                'methods':{method:{str(b):{metric:float(np.mean([r['methods'][method]['by_budget'][str(b)][metric] for r in selected]))
                    for metric in ['hybrid_success','hybrid_expected_distance','hybrid_returned_secret_probability','any_sample_secret_probability']}
                    for b in cfg['budgets']} for method in ['0','0.75','0.9']}}
        run.finish(summary)
    except Exception as error:run.finish(error=error);raise


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
