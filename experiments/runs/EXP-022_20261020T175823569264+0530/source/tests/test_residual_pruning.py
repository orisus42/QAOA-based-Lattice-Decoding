import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.residual_pruning import public_phase,select_masks,edge_list,mask_from_edges
from lattice_qaoa.sparsify import diagonal_energy
from lattice_qaoa.qaoa import circuit


def test_residual_formula_matches_full_lattice_cost_codebook():
    B=np.array([[3,1],[1,2]]);y=np.array([5,-1]);box=encode_babai_box(B,y)
    h,J,scale,r=public_phase(B,y,box.center)
    np.testing.assert_allclose(h,box.h,atol=1e-14);np.testing.assert_allclose(J,box.J,atol=1e-14)
    assert scale==box.normalization;np.testing.assert_array_equal(r,B@box.center-y)
    # Geometry-only selection is independent of the target and supplied center.
    g=public_phase(B,y,box.center,True);g2=public_phase(B,y+9,box.center-5,True)
    np.testing.assert_array_equal(g[0],g2[0]);np.testing.assert_array_equal(g[1],g2[1])


def test_greedy_decisions_match_independent_statevectors_and_fallback():
    rng=np.random.default_rng(531);h=rng.normal(size=4);J=np.triu(rng.normal(size=(4,4)),1);a=[.71,-.29]
    masks,trace=select_masks(h,J,[2,4],a);remaining=J.copy()
    for step in trace:
        energies=[];edges=edge_list(remaining)
        for i,j in edges:
            candidate=remaining.copy();candidate[i,j]=0
            p=Statevector.from_instruction(circuit(h,candidate,a)).probabilities()
            energies.append(float(p@diagonal_energy(h,J)))
        assert step['removed_edge']==edges[int(np.argmin(energies))]
        np.testing.assert_allclose(step['original_energy'],min(energies),atol=1e-12)
        remaining[tuple(step['removed_edge'])]=0
    for k,r in masks.items():
        assert np.count_nonzero(r['J'])==k
        assert r['selection_energy']<=r['magnitude_energy']+1e-12
        p=Statevector.from_instruction(circuit(h,r['J'],a)).probabilities()
        np.testing.assert_allclose(p@diagonal_energy(h,J),r['selection_energy'],atol=1e-12)
        np.testing.assert_array_equal(mask_from_edges(J,edge_list(r['J'])),r['J'])


def test_selection_is_deterministic_and_checks_budget():
    h=np.array([.2,-.3,.1,.4]);J=np.triu(np.ones((4,4)),1)
    a,_=select_masks(h,J,[0,6],[.7,-.3]);b,_=select_masks(h,J,[0,6],[.7,-.3])
    assert np.count_nonzero(a[0]['J'])==0
    np.testing.assert_array_equal(a[6]['J'],b[6]['J'])
    with pytest.raises(ValueError):select_masks(h,J,[7],[.7,-.3])
