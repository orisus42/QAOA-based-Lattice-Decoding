from itertools import product
import numpy as np
import pytest
from lattice_qaoa.sampling import repeated_success, solved_count_distribution


def test_repeated_success_handles_boundaries_and_known_example():
    np.testing.assert_array_equal(repeated_success([0, 0.3, 1], 0), [0, 0, 0])
    np.testing.assert_allclose(repeated_success([0, 0.3, 1], 1), [0, 0.3, 1])
    assert repeated_success(0.03, 100) == pytest.approx(0.9524474920745943)
    np.testing.assert_array_equal(repeated_success([0, 1], 100), [0, 1])


def test_aggregate_does_not_apply_formula_to_the_mean_probability():
    # Half the instances are impossible, half are certain: repeats cannot fix it.
    assert repeated_success([0, 1], 100).mean() == 0.5
    assert repeated_success(np.mean([0, 1]), 100) > 0.99


def test_count_distribution_matches_independent_outcome_enumeration():
    probabilities = np.array([0.0, 0.2, 0.6, 1.0])
    expected = np.zeros(5)
    for bits in product([0, 1], repeat=4):
        expected[sum(bits)] += np.prod([p if b else 1-p for b, p in zip(bits, probabilities)])
    np.testing.assert_allclose(solved_count_distribution(probabilities), expected)
    assert np.arange(5) @ expected == pytest.approx(probabilities.sum())


@pytest.mark.parametrize("probabilities, shots", [([1.1], 1), ([-0.1], 1), ([float('nan')], 2), ([0.3], -1), ([0.3], 0.5)])
def test_invalid_sampling_arguments_fail(probabilities, shots):
    with pytest.raises(ValueError):
        repeated_success(probabilities, shots)
