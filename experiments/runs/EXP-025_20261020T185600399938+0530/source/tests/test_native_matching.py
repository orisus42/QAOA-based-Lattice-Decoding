import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qiskit import transpile
from qiskit.circuit import Parameter
from lattice_qaoa.native_matching import match_counts,bind_angles
from lattice_qaoa.qaoa import circuit,state_probabilities
from lattice_qaoa.sparsify import diagonal_energy


def test_match_uses_exact_intersection_and_never_assumes_monotone_routing():
    r={'cached':[{'logical_k':k,'native_cz':n} for k,n in [(0,0),(1,5),(2,11),(3,8),(4,11)]],
       'incumbent':[{'logical_k':k,'native_cz':n} for k,n in [(0,0),(1,5),(2,8),(3,12),(4,11)]]}
    match=match_counts(r,[10,11])
    assert match[10]=={'native_cz':8,'logical_k':{'cached':3,'incumbent':2},'slack':2}
    assert match[11]['logical_k']=={'cached':4,'incumbent':4}
    assert match_counts(r,[0])[0]['native_cz']==0
    with pytest.raises(ValueError):match_counts({'cached':[{'logical_k':1,'native_cz':2}],
                                               'incumbent':[{'logical_k':1,'native_cz':3}]},[5])


def test_symbolic_compilation_and_binding_preserve_distribution_and_native_count():
    rng=np.random.default_rng(67);h=rng.normal(size=4);j=np.triu(rng.normal(size=(4,4)),1)
    g,b=Parameter('gamma'),Parameter('beta')
    template=transpile(circuit(h,j,[g,b]),basis_gates=['rz','sx','x','cz'],optimization_level=3)
    for angles in [[.73,-.29],[1.54,-.40]]:
        bound=bind_angles(template,angles)
        p=Statevector.from_instruction(bound).probabilities()
        np.testing.assert_allclose(p,state_probabilities(diagonal_energy(h,j),angles),atol=1e-12)
        assert bound.count_ops().get('cz',0)==template.count_ops().get('cz',0)
