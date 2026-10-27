"""Replay public IBM calibration snapshots on a fixed physical subgraph."""
import numpy as np
from qiskit.circuit import Delay, Measure, Parameter, Reset
from qiskit.circuit.library import IGate, RZGate, SXGate, XGate, CZGate
from qiskit.providers import QubitProperties
from qiskit.transpiler import Target, InstructionProperties
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.device import basic_device_gate_errors
from qiskit_aer.noise.passes import RelaxationNoisePass


def target_from_snapshot(snapshot):
    props = [QubitProperties(t1=q['t1'], t2=q['t2'], frequency=q['frequency']) for q in snapshot['qubits']]
    target = Target(num_qubits=snapshot['num_qubits'], dt=snapshot['dt'], qubit_properties=props,
                    **snapshot['timing_constraints'])
    operations = {'id': IGate(), 'rz': RZGate(Parameter('theta')), 'sx': SXGate(), 'x': XGate(),
                  'cz': CZGate(), 'measure': Measure(), 'reset': Reset()}
    for name, operation in operations.items():
        data = {tuple(g['qubits']): InstructionProperties(duration=g['duration'], error=g['error'])
                for g in snapshot['gates'] if g['name'] == name}
        if data:
            target.add_instruction(operation, data)
    target.add_instruction(Delay(Parameter('t')), {(q,): None for q in range(snapshot['num_qubits'])})
    return target


def gate_noise(target):
    """IBM/Aer calibrated depolarization plus gate-time T1/T2; readout separate."""
    model = NoiseModel(basis_gates=list(target.operation_names))
    for name, pair, error in basic_device_gate_errors(target=target):
        model.add_quantum_error(error, name, pair)
    return model


def idle_noise_pass(snapshot):
    return RelaxationNoisePass(t1s=[q['t1'] for q in snapshot['qubits']],
        t2s=[min(q['t2'], 2*q['t1']) for q in snapshot['qubits']],
        dt=snapshot['dt'], op_types=Delay)


def readout_channel(probabilities, snapshot):
    """Apply asymmetric measurement confusion exactly to each physical bit."""
    p = np.array(probabilities, dtype=float, copy=True)
    single = p.ndim == 1
    p = p.reshape(-1, 2**snapshot['num_qubits'])
    for k, q in enumerate(snapshot['qubits']):
        v = p.reshape(len(p), -1, 2, 2**k)
        a, b = v[:, :, 0, :].copy(), v[:, :, 1, :].copy()
        p01, p10 = q['readout_p1_given0'], q['readout_p0_given1']
        v[:, :, 0, :] = (1-p01)*a + p10*b
        v[:, :, 1, :] = p01*a + (1-p10)*b
    return p[0] if single else p


def logical_to_physical_indices(permutation):
    n = len(permutation)
    assert sorted(permutation) == list(range(n))
    logical = np.arange(2**n, dtype=np.int64)
    return sum(((logical >> j) & 1) << physical for j, physical in enumerate(permutation))


def hybrid_metrics_from_low_masses(masses, incumbent, optimum, shots):
    """Metrics from probabilities of exact costs 0..incumbent (overflow omitted).

    Leading dimensions may index trajectories or bootstrap replicates. Probability
    mass above incumbent is retained as failure via the survival formula.
    """
    from .sampling import repeated_success
    a = np.asarray(masses, dtype=float)
    if a.shape[-1] != incumbent+1 or np.min(a) < -1e-12:
        raise ValueError('expected low-cost masses 0..incumbent')
    cdf = np.clip(np.cumsum(a, axis=-1)[..., :incumbent], 0, 1)
    survival = (1-cdf)**shots
    distance = survival @ np.diff(np.sqrt(np.arange(incumbent+1)))
    cost = survival.sum(axis=-1)
    success = repeated_success(np.clip(a[..., optimum], 0, 1), shots)
    improvement = repeated_success(np.clip(a[..., :incumbent].sum(axis=-1), 0, 1), shots)
    return {'hybrid_expected_distance': distance, 'hybrid_expected_cost': cost,
            'standalone_success': success,
            'hybrid_success': np.ones_like(success) if incumbent == optimum else success,
            'strict_improvement': improvement}
