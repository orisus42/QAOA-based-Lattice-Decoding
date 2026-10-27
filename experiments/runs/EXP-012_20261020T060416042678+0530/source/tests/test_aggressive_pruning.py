import sys
from pathlib import Path
import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from lattice_qaoa.qaoa import circuit
from lattice_qaoa.sparsify import prune_interactions

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from run_aggressive_pruning import product_distribution


@pytest.mark.parametrize('n',[3,6])
def test_zero_zz_endpoint_matches_independent_single_qubit_product(n):
    h=np.random.default_rng(6200+n).normal(size=n)
    for angles in [[1.339,-.3516],[3.1,.43]]:
        qc=circuit(h,np.zeros((n,n)),angles)
        p=product_distribution(h,angles)
        np.testing.assert_allclose(p,Statevector.from_instruction(qc).probabilities(),atol=1e-12)
        assert np.isclose(p.sum(),1)
        assert 'rzz' not in qc.count_ops()


def test_aggressive_masks_are_nested_and_keep_strongest_terms():
    J=np.triu(np.random.default_rng(6206).normal(size=(16,16)),1)
    masks=[]
    for f in [.25,.5,.75,.9,1.]:
        sparse,removed=prune_interactions(J,f)
        assert len(removed)==int(np.floor(f*120))
        assert np.count_nonzero(sparse)==120-len(removed)
        masks.append({tuple(x) for x in removed})
        if np.any(sparse) and len(removed):
            assert np.abs(J[removed[:,0],removed[:,1]]).max()<=np.abs(sparse[sparse!=0]).min()
    assert all(a<=b for a,b in zip(masks,masks[1:]))
