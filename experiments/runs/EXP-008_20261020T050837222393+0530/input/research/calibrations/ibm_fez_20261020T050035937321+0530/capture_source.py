"""Read public IBM calibration metadata only; never submit jobs or save accounts."""
from datetime import datetime
from hashlib import sha256
import json
import logging
from pathlib import Path
import shutil
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

ROOT = Path(__file__).resolve().parents[1]


def main():
    logging.getLogger('qiskit_ibm_runtime').setLevel(logging.ERROR)
    start = datetime.now().astimezone()
    folder = ROOT/'research/calibrations'/f"ibm_fez_{start.strftime('%Y%m%dT%H%M%S%f%z')}"
    folder.mkdir(parents=True, exist_ok=False)
    service = QiskitRuntimeService()
    backend = service.backend('ibm_fez')
    target = backend.target
    props = backend.properties()
    raw = props.to_dict()
    fault = set(props.faulty_qubits())
    good = {i for i in range(target.num_qubits) if i not in fault and
        target.qubit_properties[i].t1 and target.qubit_properties[i].t2 and
        target['measure'][(i,)].error is not None and target['measure'][(i,)].error < .15}
    edges = {tuple(sorted(pair)): value.error for pair, value in target['cz'].items()
             if value is not None and value.duration and value.error is not None and value.error < .1 and set(pair) <= good}
    neighbors = {q: set() for q in good}
    for a, b in edges:
        neighbors[a].add(b); neighbors[b].add(a)
    candidates = []
    for start_node in sorted(good):
        chosen = {start_node}
        while len(chosen) < 16:
            frontier = set().union(*(neighbors[q] for q in chosen))-chosen
            if not frontier:
                break
            def key(q):
                adjacent = neighbors[q] & chosen
                return (-len(adjacent), np.mean([edges[tuple(sorted((q, n)))] for n in adjacent]), q)
            chosen.add(min(frontier, key=key))
        if len(chosen) == 16:
            internal = [e for pair, e in edges.items() if set(pair) <= chosen]
            candidates.append({'nodes': sorted(chosen), 'edge_count': len(internal), 'mean_cz_error': float(np.mean(internal)),
                               'mean_readout_error': float(np.mean([target['measure'][(q,)].error for q in chosen]))})
    selected = min(candidates, key=lambda x: (-x['edge_count'], x['mean_cz_error'], x['mean_readout_error'], x['nodes']))
    nodes = selected['nodes']; inverse = {p: i for i, p in enumerate(nodes)}
    gates = []
    for name in ['id', 'rz', 'sx', 'x', 'cz', 'measure', 'reset']:
        if name not in target.operation_names:
            continue
        for pair, prop in target[name].items():
            if pair is not None and set(pair) <= set(nodes) and prop is not None:
                gates.append({'name': name, 'qubits': [inverse[q] for q in pair], 'physical_qubits': list(pair),
                              'duration': prop.duration, 'error': prop.error})
    qubits = []
    for q in nodes:
        values = {v['name']: v['value'] for v in raw['qubits'][q]}
        ro = target['measure'][(q,)].error
        qubits.append({'physical': q, 't1': target.qubit_properties[q].t1,
            't2': target.qubit_properties[q].t2, 'frequency': target.qubit_properties[q].frequency,
            'readout_p1_given0': values.get('prob_meas1_prep0', ro),
            'readout_p0_given1': values.get('prob_meas0_prep1', ro)})
    config = backend.configuration()
    snapshot = {'backend': backend.name, 'fetched_at': datetime.now().astimezone().isoformat(),
        'calibration_last_update': props.last_update_date.isoformat(), 'live_read_only_fetch': True,
        'physical_qubits': nodes, 'num_qubits': 16, 'dt': backend.dt,
        'timing_constraints': {k: getattr(target, k) for k in ['granularity', 'min_length', 'pulse_alignment', 'acquire_alignment']},
        'default_rep_delay': getattr(config, 'default_rep_delay', None),
        'rep_delay_range': getattr(config, 'rep_delay_range', None), 'gates': gates, 'qubits': qubits,
        'selection': selected, 'selection_rule': 'ibm_fez chosen lexicographically from accessible devices. Grow 16-node connected patches using edge count then calibration error. Choose most internal edges, then lowest mean CZ error, then mean readout error, then node list. No lattice outcomes used.',
        'qpu_jobs_submitted': 0, 'qpu_seconds': 0}
    for name, value in [('snapshot.json', snapshot), ('backend_properties.json', raw), ('patch_candidates.json', candidates)]:
        (folder/name).write_text(json.dumps(value, indent=2, default=str)+'\n', encoding='utf-8')
    shutil.copy2(__file__, folder/'capture_source.py')
    manifest = {'created_at': snapshot['fetched_at'], 'purpose': 'Public calibration data; no credentials or account identifiers',
        'files_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}}
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'folder': str(folder), 'backend': snapshot['backend'], 'calibration': snapshot['calibration_last_update'],
        'selected_patch': selected, 'rep_delay': snapshot['default_rep_delay'], 'qpu_seconds': 0}, indent=2))


if __name__ == '__main__':
    main()
