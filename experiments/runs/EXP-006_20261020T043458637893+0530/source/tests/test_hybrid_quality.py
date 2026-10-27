from itertools import product
import numpy as np
import pytest
from lattice_qaoa.sparsify import expected_best_cost
from lattice_qaoa.sampling import repeated_success


@pytest.mark.parametrize('incumbent', [1, 5, 16])
def test_hybrid_distance_and_refinement_match_exhaustive_shot_sequences(incumbent):
    costs = np.array([1, 4, 4, 9, 16])
    p = np.array([.05, .15, .2, .25, .35])
    for shots in [1, 2, 3]:
        distance, improvement, squared = 0., 0., 0.
        for indices in product(range(len(costs)), repeat=shots):
            prob = np.prod(p[list(indices)])
            returned = min(incumbent, costs[list(indices)].min())
            distance += prob*np.sqrt(returned)
            squared += prob*returned
            improvement += prob*(returned < incumbent)
        assert expected_best_cost(np.sqrt(costs), p, shots, np.sqrt(incumbent)) == pytest.approx(distance)
        assert expected_best_cost(costs, p, shots, incumbent) == pytest.approx(squared)
        assert repeated_success(p[costs < incumbent].sum(), shots) == pytest.approx(improvement)
        assert distance <= np.sqrt(squared)+1e-12
