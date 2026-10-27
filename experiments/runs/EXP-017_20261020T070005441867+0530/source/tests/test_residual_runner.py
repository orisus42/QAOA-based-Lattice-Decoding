from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from run_residual_pruning import effort_fit


def test_large_seed_reproducibility_and_optimizer_budget():
    h=np.array([.1,-.2,.3,.4]);J=np.triu(np.ones((4,4))*.1,1)
    cfg={'optimizer':{'population':24,'gamma_bounds':[0,4.],'beta_bounds':[-1.5,0]},'selection_angles':[1.,-.3]}
    a=effort_fit(h,J,J,[1.,-.3],49,cfg,20261020173)
    b=effort_fit(h,J,J,[1.,-.3],49,cfg,20261020173)
    assert a['expectation_calls']==b['expectation_calls']==48
    np.testing.assert_array_equal(a['angles'],b['angles'])
