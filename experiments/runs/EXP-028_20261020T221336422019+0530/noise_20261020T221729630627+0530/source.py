"""Local noise predictions for frozen EXP028 circuits; no cloud calls."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from threadpoolctl import threadpool_limits
from qiskit import qpy
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from lattice_qaoa.device import target_from_snapshot, gate_noise, idle_noise_pass, readout_channel, logical_to_physical_indices


def main():
    run = Path((ROOT / "project/current_hardware_run.txt").read_text())
    out = run / ("noise_" + datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z"))
    out.mkdir()
    snap = json.loads((run / "snapshot.json").read_text())
    rows = json.loads((run / "rows.json").read_text())
    settings = {"method": "statevector", "max_parallel_threads": 4, "max_parallel_shots": 4,
                "max_parallel_experiments": 1, "max_memory_mb": 2048, "zero_threshold": 0., "fusion_max_qubit": 1}
    sim = AerSimulator(noise_model=gate_noise(target_from_snapshot(snap)), **settings)
    idle = PassManager(idle_noise_pass(snap))
    summary = []
    (out / "config.json").write_text(json.dumps({"trajectories": 64, "seed": 20261020285, "settings": settings,
        "snapshot_sha256": sha256((run / "snapshot.json").read_bytes()).hexdigest(),
        "limitations": "Local calibrated Markovian model; excludes drift, crosstalk, leakage and coherent errors. Finite trajectory uncertainty."}, indent=2))
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    clock = time.perf_counter()
    for i, row in enumerate(rows):
        stem = row["id"] + "_" + row["method"]
        with (run / "circuits" / (stem + "_compact.qpy")).open("rb") as f:
            qc = qpy.load(f)[0].remove_final_measurements(inplace=False)
        data = np.load(run / "circuits" / (stem + ".npz"))
        noisy = idle.run(qc); noisy.save_probabilities(pershot=True)
        result = sim.run(noisy, shots=64, seed_simulator=20261020285+i).result()
        assert result.success
        probs = np.asarray(result.data()["probabilities"])
        assert probs.shape == (64, 65536)
        norm_error = float(np.max(abs(probs.sum(axis=1)-1)))
        assert norm_error < 1e-10
        probs = readout_channel(probs, snap)[:, logical_to_physical_indices(data["final_permutation"])]
        assert np.max(abs(probs.sum(axis=1)-1)) < 1e-10
        costs = data["costs"]
        low = np.stack([probs[:, costs == c].sum(axis=1) for c in range(row["babai_cost"]+1)], axis=1)
        np.savez_compressed(out / (stem + ".npz"), mean_probability=probs.mean(axis=0), low_cost_masses=low)
        summary.append({"id": row["id"], "method": row["method"], "p_improve": float(low[:, :row["babai_cost"]].sum(axis=1).mean()),
                        "p_at_most_babai": float(low.sum(axis=1).mean()),
                        "p_optimum": float(low[:, row["optimum_cost"]].mean()), "max_norm_error": norm_error})
        (out / "partial.json").write_text(json.dumps(summary, indent=2))
        print("NOISE", i+1, "/40", round(time.perf_counter()-clock, 1), "s", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "manifest.json").write_text(json.dumps({"finished_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": time.perf_counter()-clock, "files_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}}, indent=2))
    (run / "completed_noise_path.txt").write_text(str(out))
    print("COMPLETED", out, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1, user_api="blas"):
        main()
