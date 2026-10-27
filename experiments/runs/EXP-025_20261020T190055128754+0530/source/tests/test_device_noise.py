from itertools import product
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from lattice_qaoa.device import readout_channel, logical_to_physical_indices, idle_noise_pass, hybrid_metrics_from_low_masses


def test_asymmetric_readout_matches_direct_transition_enumeration():
    snapshot = {'num_qubits': 3, 'qubits': [
        {'readout_p1_given0': .02*(k+1), 'readout_p0_given1': .04*(k+1)} for k in range(3)]}
    rng = np.random.default_rng(71)
    p = rng.random((2, 8)); p /= p.sum(axis=1)[:, None]
    expected = np.zeros_like(p)
    for before, after in product(range(8), repeat=2):
        probability = 1.
        for k, q in enumerate(snapshot['qubits']):
            a, b = (before >> k)&1, (after >> k)&1
            flip = q['readout_p0_given1'] if a else q['readout_p1_given0']
            probability *= flip if a != b else 1-flip
        expected[:, after] += probability*p[:, before]
    np.testing.assert_allclose(readout_channel(p, snapshot), expected, atol=1e-15)
    np.testing.assert_allclose(readout_channel(p[0], snapshot), expected[0], atol=1e-15)


def test_routing_permutation_has_correct_little_endian_mapping():
    permutation = [2, 0, 1]
    indices = logical_to_physical_indices(permutation)
    for logical in range(8):
        expected = int(f'{(logical>>0)&1}{(logical>>2)&1}{(logical>>1)&1}', 2)
        assert indices[logical] == expected


@pytest.mark.parametrize('duration', [0., 10e-6, 100e-6])
def test_idle_relaxation_matches_exact_t1_decay(duration):
    snap = {'dt': 1e-9, 'qubits': [{'t1': 100e-6, 't2': 80e-6}]}
    qc = QuantumCircuit(1); qc.x(0); qc.delay(round(duration/snap['dt']), 0)
    noisy = PassManager(idle_noise_pass(snap)).run(qc)
    noisy.save_probabilities()
    result = AerSimulator(method='density_matrix').run(noisy, shots=1).result()
    assert result.data(0)['probabilities'][1] == pytest.approx(np.exp(-duration/100e-6), abs=1e-12)


@pytest.mark.parametrize('incumbent,optimum', [(0, 0), (4, 1), (1, 1)])
def test_low_cost_masses_handle_overflow_and_incumbent_ties(incumbent, optimum):
    from lattice_qaoa.evaluation import distribution_metrics
    costs = np.array([0 if optimum == 0 else 1, 4, 9])
    p = np.array([.1, .2, .7])
    masses = np.array([p[costs == c].sum() for c in range(incumbent+1)])
    for shots in [1, 3, 1024]:
        _, _, expected = distribution_metrics(costs, p, incumbent, optimum, [shots])
        actual = hybrid_metrics_from_low_masses(masses, incumbent, optimum, shots)
        for key in actual:
            assert actual[key] == pytest.approx(expected[str(shots)][key], abs=1e-12)
