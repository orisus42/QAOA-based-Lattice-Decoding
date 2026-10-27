import numpy as np
from qiskit.quantum_info import Statevector
from lattice_qaoa.qaoa import P1Energy,circuit,state_probabilities
from lattice_qaoa.sparsify import diagonal_energy
from lattice_qaoa.evaluation import distribution_metrics
from lattice_qaoa.mechanisms import BatchP1Energy,CachedMetrics,masked,weight90,ordered_edges


def test_batch_and_split_against_independent_circuit():
    rng=np.random.default_rng(9345);hs=rng.normal(size=(3,5));js=np.triu(rng.normal(size=(3,5,5)),1)
    phase=[masked(j,7)[0] for j in js];batch=BatchP1Energy(hs,phase,js)
    for angles in [[.7,-.3],[1.1,.55,-.21]]:
        gh,gj,beta=(angles[0],angles[0],angles[1]) if len(angles)==2 else angles
        for i,(h,j,s) in enumerate(zip(hs,phase,js)):
            p=Statevector.from_instruction(circuit(gh*h,gj*j,[1,beta])).probabilities()
            np.testing.assert_allclose(batch.values(angles)[i],p@diagonal_energy(h,s),atol=1e-12)
            if len(angles)==2:np.testing.assert_allclose(batch.values(angles)[i],P1Energy(h,j,observable_J=s)(angles),atol=1e-12)


def test_cached_minima_and_secret_tie_rule():
    costs=np.array([1,1,2,3]);p=np.array([.1,.2,.3,.4]);mask=np.array([True,False,True,False])
    metrics=CachedMetrics(costs,2,1,[1,2,1024],mask,True)(p)
    old=distribution_metrics(costs,p,2,1,[1,2,1024])[2]
    for n in [1,2,1024]:
        for key in ['hybrid_success','hybrid_expected_distance','hybrid_expected_cost','strict_improvement']:
            np.testing.assert_allclose(metrics['by_budget'][str(n)][key],old[str(n)][key])
    # Keep Babai on cost ties; otherwise the first cost-1 sample is uniform
    # over the conditional mass at cost 1 (secret probability 1/3).
    np.testing.assert_allclose(metrics['by_budget']['2']['hybrid_returned_secret_probability'],.7**2+(1-.7**2)/3)


def test_masks_cardinality_structure_ties_and_weight():
    j=np.triu(np.ones((6,6)),1)
    for policy in ['weak','strong','random','ties','protect_coordinate','drop_coordinate']:
        out,removed=masked(j,8,policy,42);assert len(removed)==8 and np.count_nonzero(out)==7
    assert all(e[0]//2!=e[1]//2 for e in ordered_edges(j,'protect_coordinate')[:12])
    assert all(e[0]//2==e[1]//2 for e in ordered_edges(j,'drop_coordinate')[:3])
    out=weight90(j);assert np.sum(out*out)>=.9*np.sum(j*j)
    assert np.sum(out*out)-min(out[out!=0])**2 < .9*np.sum(j*j)
