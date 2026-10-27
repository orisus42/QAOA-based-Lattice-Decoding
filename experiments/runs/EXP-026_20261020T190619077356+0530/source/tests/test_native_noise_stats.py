import numpy as np
from lattice_qaoa.native_noise_stats import analyze_noise


def test_hybrid_population_retains_optimal_incumbents_and_paired_zero_difference():
    names=['cached_cap128','incumbent_cap128','cached_cap160','incumbent_cap160','cached24','incumbent24']
    resources={'logical_k':20,'native_cz':140,'duration_unitary_s':1e-5,'depth':200}
    rows=[{'id':'good','babai_cost':1,'optimum':1,'methods':{n:resources for n in names}},
          {'id':'bad','babai_cost':2,'optimum':1,'methods':{n:resources for n in names}}]
    cfg={'methods':names,'budgets':[1024],'primary_budget':1024,'bootstrap_replicates':200,'bootstrap_seed':154}
    exact=np.array([[0.,.001,.05]])
    noisy=np.repeat(np.array([[0.,.0005,.05]]),8,axis=0)
    def load(ident,name):
        assert ident=='bad'  # No reference shortcut is supplied to a solver.
        return {'ideal':exact,'readout_only':exact,'gate_idle':noisy,'gate_idle_readout':noisy}
    result=analyze_noise(rows,load,cfg)
    expected=.5+.5*(1-(1-.0005)**1024)
    for name in names:
        reported=result['methods'][name]['channels']['gate_idle_readout']['1024']
        np.testing.assert_allclose(reported['hybrid_success'],expected,atol=1e-13)
        np.testing.assert_allclose(reported['fixed_cohort_mc_ci95'],[expected,expected],atol=1e-13)
        assert reported['group_plus_mc_ci95'][0]<reported['group_plus_mc_ci95'][1]
    for suffix in ['cap128','cap160','24']:
        paired=result['paired'][suffix]['gate_idle_readout']['1024']
        np.testing.assert_allclose(paired['success_difference'],0,atol=1e-13)
        np.testing.assert_allclose(paired['group_plus_mc_ci95'],[0,0],atol=1e-13)
