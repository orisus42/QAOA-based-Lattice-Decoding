from itertools import product
import numpy as np
import pytest
from qiskit import transpile
from qiskit.quantum_info import Statevector
from lattice_qaoa.qaoa import circuit, P1Energy, state_probabilities
from lattice_qaoa.sparsify import prune_interactions, diagonal_energy, expected_best_cost


@pytest.mark.parametrize('n', [2, 3, 4, 5])
def test_sparse_phase_full_observable_matches_independent_circuits(n):
    rng = np.random.default_rng(4500+n)
    h = rng.normal(size=n)
    J = np.triu(rng.normal(size=(n, n)), 1)
    original = diagonal_energy(h, J)
    for fraction in [0, .25, .5, 1]:
        sparse, _ = prune_interactions(J, fraction)
        score = P1Energy(h, sparse, observable_h=h, observable_J=J)
        energy = diagonal_energy(h, sparse)
        for angles in [[.8, -.3], [2.4, .6]]:
            p = state_probabilities(energy, angles)
            qc = circuit(h, sparse, angles)
            p2 = Statevector.from_instruction(qc).probabilities()
            np.testing.assert_allclose(p, p2, atol=1e-12, rtol=0)
            assert score(angles) == pytest.approx(float(p@original), abs=1e-11)
            compiled = transpile(qc, basis_gates=['rz','sx','x','cx'], optimization_level=3, seed_transpiler=77)
            np.testing.assert_allclose(Statevector.from_instruction(compiled).probabilities(), p, atol=1e-12, rtol=0)


def test_pruning_is_public_deterministic_and_preserves_input():
    J = np.array([[0, -1., 2], [0, 0, 1], [0, 0, 0]])
    sparse, removed = prune_interactions(J, 1/3)
    assert removed.tolist() == [[0, 1]]
    assert J[0, 1] == -1 and sparse[0, 1] == 0
    a, _ = prune_interactions(J, .5, random_seed=12)
    b, _ = prune_interactions(J, .5, random_seed=12)
    np.testing.assert_array_equal(a, b)
    for value in [-.1, 1.1]:
        with pytest.raises(ValueError):
            prune_interactions(J, value)


def test_expected_best_cost_matches_enumerated_sample_sequences():
    costs, p = np.array([1, 1, 4, 7]), np.array([.1, .2, .3, .4])
    for shots in [1, 2, 3]:
        for incumbent in [None, 3]:
            brute = 0
            for indices in product(range(4), repeat=shots):
                best = costs[list(indices)].min()
                if incumbent is not None:
                    best = min(best, incumbent)
                brute += np.prod(p[list(indices)])*best
            assert expected_best_cost(costs, p, shots, incumbent) == pytest.approx(brute)
