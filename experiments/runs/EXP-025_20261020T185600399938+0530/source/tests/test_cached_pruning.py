import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from lattice_qaoa.cached_pruning import FixedAngleMaskEnergy,select_masks_cached
from lattice_qaoa.residual_pruning import select_masks
from lattice_qaoa.mechanisms import BatchP1Energy
from lattice_qaoa.qaoa import circuit
from lattice_qaoa.sparsify import diagonal_energy


@pytest.mark.parametrize('n',[2,4,8])
def test_cached_masks_match_original_analytic_and_quantum_formulas(n):
    rng=np.random.default_rng(721+n)
    h=rng.normal(size=n);J=np.triu(rng.normal(size=(n,n)),1);a=[1.31,-.46]
    masks=np.triu(rng.random((21,n,n))>.5,1);masks[0]=False;masks[1]=np.triu(np.ones((n,n),bool),1)
    phases=np.where(masks,J,0.)
    actual=FixedAngleMaskEnergy(h,J,a).values(masks)
    expected=BatchP1Energy([h]*len(masks),phases,[J]*len(masks)).values(a)
    np.testing.assert_allclose(actual,expected,atol=1e-13,rtol=1e-13)
    for index in [0,1,2]:
        probabilities=Statevector.from_instruction(circuit(h,phases[index],a)).probabilities()
        np.testing.assert_allclose(actual[index],probabilities@diagonal_energy(h,J),atol=1e-12)


@pytest.mark.parametrize('symmetric',[False,True])
def test_cached_path_matches_reference_including_ties_and_endpoints(symmetric):
    rng=np.random.default_rng(122)
    h=np.zeros(6) if symmetric else rng.normal(size=6)
    J=np.triu(np.ones((6,6)) if symmetric else rng.normal(size=(6,6)),1)
    budgets=[15,7,0];a=[.87,-.35]
    expected,path=select_masks(h,J,budgets,a)
    actual,new_path=select_masks_cached(h,J,budgets,a)
    assert [x['removed_edge'] for x in path]==[x['removed_edge'] for x in new_path]
    for k in budgets:
        np.testing.assert_array_equal(actual[k]['J'],expected[k]['J'])
        np.testing.assert_array_equal(actual[k]['greedy_J'],expected[k]['greedy_J'])
        assert actual[k]['used_magnitude_fallback']==expected[k]['used_magnitude_fallback']
    if symmetric:assert actual[0]['tie_recomputations']>0


def test_cache_handles_zero_cosine_factors_without_division():
    h=np.array([.1,.2,-.1]);J=np.triu(np.full((3,3),np.pi/4),1);a=[1.,-.3]
    masks=np.array([np.triu(np.ones((3,3),bool),1),np.zeros((3,3),bool)])
    actual=FixedAngleMaskEnergy(h,J,a).values(masks)
    expected=BatchP1Energy([h,h],np.where(masks,J,0.),[J,J]).values(a)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual,expected,atol=1e-13)
