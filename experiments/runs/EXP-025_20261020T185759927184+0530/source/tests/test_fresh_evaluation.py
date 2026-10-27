from itertools import product
import numpy as np
import pytest
from lattice_qaoa.evaluation import distribution_metrics, matrix_group, paired_retention


@pytest.mark.parametrize('incumbent,optimum', [(4, 1), (1, 1), (4, 0)])
def test_new_budget_metrics_match_all_sample_sequences_and_missing_optimum(incumbent, optimum):
    costs = np.array([1, 4, 4, 9])
    probabilities = np.array([.1, .2, .3, .4])
    _, _, output = distribution_metrics(costs, probabilities, incumbent, optimum, [1, 2, 3])
    for shots in [1, 2, 3]:
        expected = {k: 0. for k in output[str(shots)]}
        for indices in product(range(4), repeat=shots):
            probability = np.prod(probabilities[list(indices)])
            best = costs[list(indices)].min()
            hybrid = min(best, incumbent)
            outcomes = [best == optimum, hybrid == optimum, hybrid < incumbent,
                        best, np.sqrt(best), hybrid, np.sqrt(hybrid)]
            for key, value in zip(expected, outcomes):
                expected[key] += probability * value
        for key in expected:
            assert output[str(shots)][key] == pytest.approx(expected[key], abs=1e-12)


def test_large_budget_and_exact_zero_and_one_probability_endpoints():
    for p in [0., .00003, 1.]:
        _, _, r = distribution_metrics(np.array([0, 9]), np.array([p, 1-p]), 9, 0, [16384])
        expected_distance = 3 * (1-p)**16384
        assert r['16384']['hybrid_expected_distance'] == pytest.approx(expected_distance, abs=1e-11)


def test_group_fingerprint_is_modular_and_distinguishes_lattices():
    A = np.array([[1, -2], [2, 1]])
    assert matrix_group(A, 17) == matrix_group(A+34, 17)
    assert matrix_group(A, 17) != matrix_group(A+1, 17)


def test_retention_bootstrap_uses_matched_instances():
    b, f = np.array([3., 4., 5.]), np.array([3., 3., 3.])
    c = b-1.2*(b-f)
    boot = np.array([[0, 1, 2], [2, 2, 1], [0, 0, 0]])
    result = paired_retention(b, f, c, boot)
    assert result['estimate'] == pytest.approx(1.2)
    np.testing.assert_allclose(result['ci95'], [1.2, 1.2])
    assert result['undefined_bootstraps'] == 1
