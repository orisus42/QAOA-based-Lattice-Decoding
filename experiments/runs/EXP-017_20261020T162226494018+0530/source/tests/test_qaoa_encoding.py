from itertools import product
import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import P1Energy, circuit, state_probabilities
from lattice_qaoa.classical import lift_secrets, local_refine


@pytest.mark.parametrize('n', [2, 3, 4, 5])
def test_p1_formula_matches_two_independent_statevectors(n):
    rng = np.random.default_rng(221+n)
    h = rng.normal(size=n)
    J = np.triu(rng.normal(size=(n, n)), 1)
    spins = 1-2*((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
    energy = spins@h+np.sum((spins@J)*spins, axis=1)
    evaluator = P1Energy(h, J)
    for angles in [[0, 0], [0.3, -0.4], [1.2, 0.7], [np.pi/4, -np.pi/4]]:
        a = state_probabilities(energy, angles)
        b = Statevector.from_instruction(circuit(h, J, angles)).probabilities()
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
        assert evaluator(angles) == pytest.approx(float(a@energy), abs=1e-12)


def test_binary_box_round_trip_and_basis_costs():
    B, target = np.array([[3, 1], [0, 2]]), np.array([4, 2])
    box = encode_babai_box(B, target)
    points = box.decode(np.arange(2**box.nq))
    expected = set(product(*[range(x-2, x+2) for x in box.center]))
    assert {tuple(x) for x in points} == expected
    np.testing.assert_array_equal(box.costs, np.sum((points@B.T-target)**2, axis=1))
    np.testing.assert_allclose(box.energy*box.normalization+box.constant, box.costs)
    with pytest.raises(ValueError):
        box.decode([-1])


def test_refinement_retains_incumbent_and_can_improve_it():
    B, target = np.array([[3, 2], [0, 1]]), np.array([2, 1])
    start = np.zeros(2, dtype=int)
    stalled = local_refine(B, target, start, budget=1)
    assert stalled['cost'] == 5 and stalled['evaluations'] == 1
    refined = local_refine(B, target, start, max_changed=1, budget=20)
    assert refined['cost'] == 0 and refined['coefficients'] == [0, 1]
    assert refined['evaluations'] <= 20


def test_secret_lift_is_best_wrap_and_matches_embedding_cost():
    A, t, q = np.array([[1, -2], [2, 1]]), np.array([2, 4]), 5
    secrets = np.array([[0, 0], [1, -1], [-2, 1]])
    coordinates, costs = lift_secrets(A, t, q, secrets)
    for s, c, z in zip(secrets, costs, coordinates):
        all_w = np.array(list(product(range(-3, 4), repeat=2)))
        brute = np.sum((q*all_w+s@A.T-t)**2, axis=1)+s@s
        assert c == int(brute.min())
        assert c == int(np.sum((q*z[:2]+A@s-t)**2)+s@s)
