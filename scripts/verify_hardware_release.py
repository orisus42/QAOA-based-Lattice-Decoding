"""Offline audit of measured counts, integer costs, maps and exported provenance.

Read-only: no updates to historical runs and no cloud access.
"""
from pathlib import Path
from hashlib import sha256
from collections import defaultdict
import json
import sys
import numpy as np
from qiskit import qpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lattice_qaoa.reference import babai


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    runs = [ROOT / 'experiments/runs' / name for name in [
        'EXP-028_20261020T221336422019+0530', 'EXP-029_20261020T221914491556+0530']]
    provenance = {r['path']:r for r in read(ROOT / 'reproducibility/export_provenance.json')['files']}
    omitted = {r['source_path']:r for r in read(ROOT / 'reproducibility/omitted_intermediates.json')['files']}
    source = read(ROOT / 'experiments/runs/EXP-007_20261020T044846904050+0530/instances.json')
    measured = defaultdict(lambda: {'shots':0, 'improving':0, 'optimum':0, 'best':None})
    shots = maps = hashes = usage = omitted_metadata = 0
    for run in runs:
        manifest = read(run / 'manifest.json')
        for name, original_digest in manifest['frozen_files_sha256'].items():
            path = run / name.replace('\\', '/')
            relative = path.relative_to(ROOT).as_posix()
            if not path.exists():
                assert path.name in {'backend_selection.json', 'usage_before.json'}
                assert omitted[relative]['sha256'] == original_digest
                omitted_metadata += 1
                continue
            record = provenance[relative]
            assert record['source_sha256'] == original_digest, relative
            assert sha256(path.read_bytes()).hexdigest() == record['sha256'], relative
            hashes += 1
        rows, order = read(run / 'rows.json'), read(run / 'pub_order.json')
        snapshot = read(run / 'snapshot.json')
        for i, row in enumerate(rows):
            instance = source[row['source_index']]
            R = np.asarray(instance['reduced_basis'], dtype=np.int64)
            y = np.r_[instance['public_target'], np.zeros(4, dtype=np.int64)]
            center = babai(R, y)
            stem = row['id'] + '_' + row['method']
            with np.load(run / 'circuits' / (stem + '.npz')) as data:
                with (run / row['qpy'].replace('\\', '/')).open('rb') as f:
                    circuit = qpy.load(f)[0]
                actual_map = {circuit.find_bit(x.clbits[0]).index: circuit.find_bit(x.qubits[0]).index
                              for x in circuit.data if x.operation.name == 'measure'}
                expected_map = {j: snapshot['physical_qubits'][int(v)] for j,v in enumerate(data['final_permutation'])}
                assert actual_map == expected_map
                chunks = []
                for j, pub in enumerate(order):
                    if pub['row_index'] != i:
                        continue
                    values = np.load(run / 'counts' / f'pub_{j:03d}_ordered.npy')
                    counts = read(run / 'counts' / f'pub_{j:03d}.json')['counts']
                    unique, count = np.unique(values, return_counts=True)
                    assert {int(k,2):v for k,v in counts.items()} == dict(zip(unique.tolist(), count.tolist()))
                    assert len(values) == pub['shots']
                    chunks.append(values)
                values = np.concatenate(chunks)
                bits = (values.astype(np.int64)[:,None] >> np.arange(16)) & 1
                z = center - 2 + bits[:,0::2] + 2*bits[:,1::2]
                residual = z @ R.T - y
                costs = np.sum(residual*residual, axis=1)
                np.testing.assert_array_equal(costs, data['costs'][values])
                assert int(costs.min()) >= row['optimum_cost']
                record = measured[(row['id'], row['method'])]
                record['shots'] += len(values)
                record['improving'] += int(np.sum(costs < row['babai_cost']))
                record['optimum'] += int(np.sum(costs == row['optimum_cost']))
                record['best'] = int(costs.min()) if record['best'] is None else min(record['best'], int(costs.min()))
                shots += len(values)
                maps += 1
        usage += read(run / 'job_metrics.json')['usage']['quantum_seconds']
    pooled = read(ROOT / 'results/hardware_per_circuit.json')
    for row in pooled:
        got = measured[(row['id'], row['method'])]
        assert got['shots'] == row['shots']
        assert got['improving'] == row['improve']['count']
        assert got['optimum'] == row['optimum']['count']
        assert got['best'] == row['best_observed_cost']
    assert (shots, maps, usage) == (983040, 80, 280)
    print(json.dumps({'status':'passed', 'shots_independently_rescored':shots,
                     'measurement_maps_checked':maps, 'historical_frozen_hashes_checked_via_export_provenance':hashes,
                     'omitted_account_metadata_digests_recorded_not_content_verified':omitted_metadata,
                     'pooled_rows_checked':len(pooled), 'qpu_seconds_recorded':usage,
                     'qpu_seconds_used_by_this_check':0}, indent=2))


if __name__ == '__main__':
    main()
