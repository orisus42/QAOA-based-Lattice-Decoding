"""Small, deterministic local example. No network, hidden-answer solver inputs, or QPU."""
from pathlib import Path
import json
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lattice_qaoa.instances import Parameters, generate_instance, expand_module_matrix
from lattice_qaoa.reference import embedding, lll_reduce, squared_cost, exact_cvp
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import state_probabilities
from lattice_qaoa.sparsify import prune_interactions, diagonal_energy
from lattice_qaoa.cached_pruning import select_masks_cached


def main():
    p = Parameters(n=2, k=2, q=17, eta=2)
    instance = generate_instance(2026101800, p)
    A, target = expand_module_matrix(instance.A), instance.target.ravel()
    B, y = embedding(A, target, p.q)
    R, U = lll_reduce(B)
    box = encode_babai_box(R, y)
    angles = (1.3393885543560986, -0.3516024709404892)
    sparse, _ = prune_interactions(box.J, .75)
    budget = min(24, np.count_nonzero(box.J))
    selections, _ = select_masks_cached(box.h, box.J, [budget], angles)
    babai_cost = squared_cost(R, y, box.center)
    # Reference is evaluation-only and is computed after every mask was selected.
    reference = exact_cvp(A, target, p.q, U @ box.center, 1_000_000)
    rows = []
    for name, J in [('full', box.J), ('magnitude_75', sparse), ('cached_K24', selections[budget]['J'])]:
        probabilities = state_probabilities(diagonal_energy(box.h, J), angles)
        assert np.isclose(probabilities.sum(), 1)
        sampled = np.random.default_rng(2026102700).choice(len(probabilities), size=1024, p=probabilities)
        optimal_probability = float(probabilities[box.costs == reference['cost']].sum())
        rows.append({'method': name, 'logical_ZZ': int(np.count_nonzero(J)),
                     'single_shot_optimum_probability': optimal_probability,
                     'expected_hybrid_optimum_success_1024': 1.0 if babai_cost == reference['cost'] else float(-np.expm1(1024*np.log1p(-optimal_probability))) if optimal_probability < 1 else 1.0,
                     'observed_retained_cost_1024': int(min(babai_cost, box.costs[sampled].min()))})
    print(json.dumps({'scope': 'One illustrative tiny instance; not a performance benchmark or hardware result.',
        'qubits': box.nq, 'babai_cost': int(babai_cost), 'reference_cost': reference['cost'],
        'reference_in_box': bool(np.any(box.costs == reference['cost'])), 'results': rows}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='blas'):
        main()
