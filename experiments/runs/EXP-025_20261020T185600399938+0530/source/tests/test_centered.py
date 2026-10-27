import numpy as np
from qiskit.quantum_info import Statevector
from lattice_qaoa.centered import coordinate_mask,centered_fields
from lattice_qaoa.mechanisms import BatchP1Energy
from lattice_qaoa.qaoa import circuit
from lattice_qaoa.sparsify import diagonal_energy


def test_centered_identity_and_all_single_coordinate_moves():
    rng=np.random.default_rng(3491);n=8
    h=rng.normal(size=n);J=np.triu(rng.normal(size=(n,n)),1)
    b=rng.choice([-1,1],n);kept=coordinate_mask(J,12)
    sh,constant=centered_fields(h,J,kept,b)
    spins=1-2*((np.arange(2**n)[:,None]>>np.arange(n))&1)
    full=diagonal_energy(h,J);surrogate=diagonal_energy(sh,kept)+constant
    error=np.einsum('bi,ij,bj->b',spins-b,J-kept,spins-b)
    np.testing.assert_allclose(full-surrogate,error,atol=1e-12)
    changed=np.any((spins!=b).reshape(-1,n//2,2),axis=2).sum(axis=1)
    np.testing.assert_allclose(full[changed<=1],surrogate[changed<=1],atol=1e-12)
    assert np.count_nonzero(kept)<=12
    assert all(kept[i,i+1]==J[i,i+1] for i in range(0,n,2))


def test_original_field_scoring_matches_independent_circuit():
    rng=np.random.default_rng(813);n=6
    h=rng.normal(size=n);J=np.triu(rng.normal(size=(n,n)),1)
    kept=coordinate_mask(J,7);sh,_=centered_fields(h,J,kept,np.tile([1,-1],3))
    objective=BatchP1Energy([sh],[kept],[J],score_hs=[h])
    angles=[.73,-.28]
    p=Statevector.from_instruction(circuit(sh,kept,angles)).probabilities()
    np.testing.assert_allclose(objective(angles),p@diagonal_energy(h,J),atol=1e-12)
