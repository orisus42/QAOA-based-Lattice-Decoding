import numpy as np
from lattice_qaoa.incumbent_pruning import incumbent_mask


def test_maxcut_limit_retains_cut_edges_before_uncut_edges():
    J=np.triu(np.ones((4,4)),1)
    b=np.array([1,1,-1,-1])
    kept=incumbent_mask(J,b,4)
    assert set(zip(*np.nonzero(kept)))=={(0,2),(0,3),(1,2),(1,3)}
    assert np.array_equal(incumbent_mask(J,b,0),np.zeros_like(J))


def test_weighted_signed_priority_and_reference_gauge_equivalence():
    J=np.array([[0,3,-2,1],[0,0,4,-5],[0,0,0,2],[0,0,0,0]],dtype=float)
    b=np.array([1,-1,1,-1]);D=np.diag(b)
    # A spin-gauge transformation preserves the chosen physical edge set.
    kept=incumbent_mask(J,b,3)
    gauged=incumbent_mask(D@J@D,np.ones(4),3)
    np.testing.assert_array_equal(kept!=0,gauged!=0)
    assert np.count_nonzero(kept)==3
