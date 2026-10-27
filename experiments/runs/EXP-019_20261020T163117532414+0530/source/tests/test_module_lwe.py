from itertools import product
from math import prod

import numpy as np
import pytest
from sympy import Matrix

from lattice_qaoa.instances import (
    Parameters, generate_instance, negacyclic_product, expand_module_matrix,
    module_product, sample_cbd,
)
from lattice_qaoa.reference import (
    embedding, lll_reduce, babai, squared_cost, exact_cvp, integer_ball,
)


def test_hand_calculated_polynomial_product():
    # (1+2x)(3+4x) = 3+10x+8x^2 = -5+10x when x^2=-1.
    np.testing.assert_array_equal(negacyclic_product([1, 2], [3, 4]), [-5, 10])
    np.testing.assert_array_equal(expand_module_matrix(np.array([[[1, 2]]])), [[1, -2], [2, 1]])


@pytest.mark.parametrize("n", [2, 4, 8])
def test_polynomial_product_against_independent_definition(n):
    rng = np.random.default_rng(18+n)
    for _ in range(10):
        a, b = rng.integers(-5, 6, size=(2, n))
        expected = [0]*n
        for i, j in product(range(n), repeat=2):
            expected[(i+j) % n] += int(a[i]*b[j]) * (1 if i+j < n else -1)
        np.testing.assert_array_equal(negacyclic_product(a, b), expected)


@pytest.mark.parametrize("n,k", [(2, 1), (2, 2), (4, 2)])
def test_expansion_matches_polynomial_arithmetic(n, k):
    instance = generate_instance(20261018, Parameters(n=n, k=k))
    expanded = expand_module_matrix(instance.A)
    np.testing.assert_array_equal(expanded @ instance.secret.ravel(), module_product(instance.A, instance.secret).ravel())
    np.testing.assert_array_equal((expanded @ instance.secret.ravel()+instance.error.ravel()) % 17, instance.target.ravel())


def test_cbd_eta_two_exact_distribution():
    class AllBitPatterns:
        def integers(self, low, high, size, dtype):
            assert (low, high, size) == (0, 2, (16, 4))
            return np.array(list(product([0, 1], repeat=4)), dtype=dtype)

    values, counts = np.unique(sample_cbd(AllBitPatterns(), 2, (16,)), return_counts=True)
    np.testing.assert_array_equal(values, [-2, -1, 0, 1, 2])
    np.testing.assert_array_equal(counts, [1, 4, 6, 4, 1])


def test_seed_reproducibility():
    a, b = generate_instance(42), generate_instance(42)
    for name in ["A", "secret", "error", "target"]:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_embedding_and_lll_preserve_the_lattice(seed):
    instance = generate_instance(seed)
    A, t = expand_module_matrix(instance.A), instance.target.ravel()
    s, e, q = instance.secret.ravel(), instance.error.ravel(), instance.parameters.q
    B, y = embedding(A, t, q)
    numerator = t-e-A@s
    assert np.all(numerator % q == 0)
    planted = np.concatenate([numerator//q, s])
    np.testing.assert_array_equal(B @ planted-y, np.concatenate([-e, s]))
    assert squared_cost(B, y, planted) == int(e@e+s@s)
    assert abs(Matrix(B).det()) == q**len(t)
    reduced, U = lll_reduce(B)
    np.testing.assert_array_equal(reduced, B @ U)
    assert abs(Matrix(U).det()) == 1
    assert abs(Matrix(reduced).det()) == abs(Matrix(B).det())


def test_babai_orthogonal_basis():
    B, target = np.diag([4, 6]), np.array([5, 10])
    np.testing.assert_array_equal(babai(B, target), [1, 2])


def test_rounding_real_solution_can_fail():
    B, target = np.array([[3, 2], [0, 1]]), np.array([2.0, 0.4])
    rounded = np.rint(np.linalg.solve(B, target)).astype(int)
    assert np.sum((B@rounded-target)**2) == pytest.approx(4.16)
    assert np.sum((B@np.array([0, 1])-target)**2) == pytest.approx(0.36)


def test_exact_cvp_against_independent_full_coefficient_enumeration():
    A, target, q = np.array([[1, 1], [0, 2]]), np.array([3, 1]), 5
    B, y = embedding(A, target, q)
    incumbent = np.array([1, 0, 0, 0])
    answer = exact_cvp(A, target, q, incumbent)
    # With incumbent cost 5, |s_i|<=2 and any improving w has |w_i|<=2.
    # This independently searches all four lattice coefficients, not q-lifts.
    candidates = np.array(list(product(range(-3, 4), repeat=4)))
    costs = np.sum((candidates @ B.T-y)**2, axis=1)
    optimum = int(costs.min())
    expected = {tuple(x) for x in candidates[costs == optimum]}
    assert answer["cost"] == optimum
    assert {tuple(x) for x in answer["minimizers"]} == expected


def test_closest_vector_need_not_be_planted_secret():
    answer = exact_cvp(np.array([[1]]), np.array([2]), 5)
    assert answer["cost"] == 2
    assert answer["minimizers"] == [[0, 1]]
    # Planted secret 2, error 0 has cost 4, so exact CVP chooses secret 1.


def test_zero_cost_and_global_ties():
    zero = exact_cvp(np.array([[1]]), np.array([0]), 5)
    assert zero["cost"] == 0 and zero["minimizers"] == [[0, 0]]
    tied = exact_cvp(np.array([[1]]), np.array([1]), 5)
    assert tied["cost"] == 1
    assert {tuple(x) for x in tied["minimizers"]} == {(0, 0), (0, 1)}


def test_exact_solver_fails_explicitly_if_enumeration_budget_is_exceeded():
    with pytest.raises(RuntimeError, match="budget exceeded"):
        exact_cvp(np.array([[1]]), np.array([2]), 5, max_candidates=1)


def test_integer_ball_contains_exactly_the_bounded_integer_vectors():
    expected = {v for v in product(range(-2, 3), repeat=3) if sum(x*x for x in v) <= 5}
    assert set(integer_ball(3, 5)) == expected


@pytest.mark.parametrize("parameters", [{"q": 15}, {"n": 3}, {"eta": 0}, {"k": 0}])
def test_invalid_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        Parameters(**parameters)
