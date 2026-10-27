"""EXP-002: audit the old QAOA probabilities and actually sample ideal circuits.

No training, cloud connection, or overwrite of earlier pilot artifacts.
"""
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"pilot")]
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from feasibility import babai, probabilities
from hardware_readiness import qaoa_circuit
from lattice_qaoa.sampling import repeated_success, solved_count_interval


def now():
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def audit_instance(row):
    """Recompute every candidate's cost and the full q-ary CVP reference."""
    A, B, t = [np.array(row[k], dtype=np.int64) for k in ["A", "B", "t"]]
    q, nq = row["q"], row["nq"]
    all_secrets = np.array(list(product(range(q), repeat=A.shape[1])), dtype=np.int64)
    residues = (all_secrets @ A.T) % q
    points = residues + q*((t-residues+q//2)//q)
    optimum = int(np.sum((points-t)**2, axis=1).min())
    z0 = babai(B, t)
    bits = (np.arange(2**nq)[:, None] >> np.arange(nq)) & 1
    coefficients = z0-2+bits[:, ::2]+2*bits[:, 1::2]
    costs = np.sum((coefficients @ B.T-t)**2, axis=1)
    good = costs == optimum
    assert optimum == row["optimum"]
    np.testing.assert_array_equal(good, row["good"])
    np.testing.assert_allclose((costs-row["constant"])/row["normalization"], row["energy_norm"], atol=1e-12)
    babai_cost = int(np.sum((B@z0-t)**2))
    assert babai_cost == row["babai_cost"]
    return costs, good, babai_cost, optimum


def main():
    config_path = ROOT/"experiments/configs/EXP-002.json"
    config = json.loads(config_path.read_text())
    started = now()
    output = ROOT/"experiments/runs"/f"EXP-002_{started.strftime('%Y%m%dT%H%M%S%f%z')}"
    output.mkdir(parents=True, exist_ok=False)
    save(output/"config.json", config)
    sources = [Path(__file__).resolve(), ROOT/"src/lattice_qaoa/sampling.py", ROOT/"tests/test_sampling.py",
               ROOT/"pilot/feasibility.py", ROOT/"pilot/hardware_readiness.py", ROOT/"scripts/plot_ideal_sampling.py", config_path]
    inputs = [ROOT/config["instances_path"], ROOT/"outputs/pilot_results.json", ROOT/"outputs/controls.json", ROOT/"outputs/noise_arrays.npz"]
    manifest = {"run_id": output.name, "status": "running", "started_at": started.isoformat(),
                "command": "python experiments/run_ideal_sampling.py", "python": platform.python_version(),
                "packages": {p: version(p) for p in ["numpy", "qiskit", "qiskit-aer", "scipy", "sympy", "scikit-learn", "matplotlib", "seaborn"]},
                "source_sha256": {}, "input_sha256": {}, "qpu_seconds": 0}
    for kind, paths in [("source", sources), ("input", inputs)]:
        for source in paths:
            relative = source.relative_to(ROOT)
            snapshot = output/kind/relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, snapshot)
            manifest[kind+"_sha256"][relative.as_posix()] = sha256(source.read_bytes()).hexdigest()
    save(output/"manifest.json", manifest)
    before = time.perf_counter()
    try:
        rows = json.loads(inputs[0].read_text())
        fitted = json.loads(inputs[1].read_text())
        controls = json.loads(inputs[2].read_text())
        old = np.load(inputs[3])
        choices = old["choices"]
        N, shots = len(rows), config["shots_per_circuit"]
        assert N == 200 and choices.shape == (N,)
        circuits, exact, per_instance = [], {"fixed": [], "learned": [], "uniform": []}, []
        costs_all, good_all, max_difference = [], [], 0.0
        for i, row in enumerate(rows):
            costs, good, babai_cost, optimum = audit_instance(row)
            costs_all.append(costs)
            good_all.append(good)
            data = dict(row)
            for key in ["energy_norm", "h", "J", "good"]:
                data[key] = np.array(data[key])
            policies = {"fixed": controls["separately_trained_size_angles"][str(row["nq"])],
                        "learned": fitted["schedules"][int(choices[i])]}
            record = {"id": row["id"], "qubits": row["nq"], "babai_cost": babai_cost, "optimum": optimum,
                      "babai_optimal": babai_cost == optimum, "box_size": len(costs), "optimal_bitstrings": int(good.sum()),
                      "angles": policies, "probabilities": {}, "observed": {}}
            for method, angles in policies.items():
                circuit = qaoa_circuit(data, angles)
                state_probability = Statevector.from_instruction(circuit).probabilities()
                independent = probabilities(data["energy_norm"], angles)[0]
                difference = float(np.max(np.abs(state_probability-independent)))
                max_difference = max(max_difference, difference)
                assert difference < 1e-10 and abs(state_probability.sum()-1) < 1e-10
                p = float(state_probability[good].sum())
                exact[method].append(p)
                record["probabilities"][method] = p
                circuit.name = f"{method}_{row['id']}"
                circuit.measure_all()
                circuits.append(circuit)
            uniform = float(good.mean())
            exact["uniform"].append(uniform)
            record["probabilities"]["uniform"] = uniform
            per_instance.append(record)
        exact = {k: np.array(v) for k, v in exact.items()}
        np.testing.assert_allclose(exact["learned"], old["ideal"][np.arange(N), choices], atol=1e-12)
        assert abs(exact["fixed"].mean()-controls["replication_separately_trained_fixed"]) < 1e-12
        print(f"Verified {len(circuits)} ideal circuits; maximum independent probability difference {max_difference:.3g}", flush=True)
        simulator = AerSimulator(method="statevector", max_parallel_threads=2, max_parallel_experiments=1)
        result = simulator.run(circuits, shots=shots, memory=True, seed_simulator=config["measurement_seed"]).result()
        assert result.success
        samples = {key: np.zeros((N, shots), dtype=np.int64) for key in exact}
        rng = np.random.default_rng(config["uniform_seed"])
        for i, row in enumerate(rows):
            for j, method in enumerate(["fixed", "learned"]):
                memory = result.get_memory(2*i+j)
                samples[method][i] = [int(bits.replace(" ", ""), 2) for bits in memory]
            samples["uniform"][i] = rng.integers(0, 2**row["nq"], size=shots)
        babai_costs = np.array([r["babai_cost"] for r in per_instance])
        optima = np.array([r["optimum"] for r in per_instance])
        babai_success = babai_costs == optima
        failures = ~babai_success
        methods = {}
        for method, p in exact.items():
            sampled_costs = np.stack([costs_all[i][samples[method][i]] for i in range(N)])
            hits = sampled_costs == optima[:, None]
            cumulative_best = np.minimum.accumulate(sampled_costs, axis=1)
            rows_by_budget = []
            for budget in config["report_budgets"]:
                expected = repeated_success(p, budget)
                observed = cumulative_best[:, budget-1] == optima
                hybrid = np.minimum(cumulative_best[:, budget-1], babai_costs) == optima
                rows_by_budget.append({"shots": budget,
                    "exact_expected_solved_fraction": float(expected.mean()),
                    "exact_expected_solved_count": float(expected.sum()),
                    "measurement_only_95pct_count_interval": solved_count_interval(expected),
                    "observed_solved_count": int(observed.sum()), "observed_solved_fraction": float(observed.mean()),
                    "exact_expected_hybrid_solved_fraction": float(np.where(babai_success, 1, expected).mean()),
                    "observed_hybrid_solved_count": int(hybrid.sum()),
                    "babai_failures_recovered_count": int(observed[failures].sum()),
                    "babai_failure_denominator": int(failures.sum())})
                for i, record in enumerate(per_instance):
                    record["observed"].setdefault(method, {})[str(budget)] = {
                        "successful_measurements": int(hits[i, :budget].sum()),
                        "best_cost": int(cumulative_best[i, budget-1]),
                        "solved": bool(observed[i]), "hybrid_solved": bool(hybrid[i])}
            methods[method] = {"exact_mean_single_shot_probability": float(p.mean()),
                "exact_single_shot_quantiles": dict(zip(["min", "q25", "median", "q75", "max"], np.quantile(p, [0, .25, .5, .75, 1]).tolist())),
                "observed_successful_measurements": int(hits.sum()), "total_measurements": int(hits.size),
                "observed_single_shot_success_fraction": float(hits.mean()),
                "by_shot_budget": rows_by_budget,
                "by_qubits": {str(nq): {"count": sum(r["qubits"] == nq for r in per_instance),
                    "mean_single_shot_probability": float(p[[r["qubits"] == nq for r in per_instance]].mean())} for nq in [6, 8]}}
        summary = {"run_id": output.name, "experiment_id": "EXP-002", "count": N, "cohort": config["cohort"],
            "depth_p": 1, "qubits": [6, 8], "methods": methods,
            "babai": {"solved_count": int(babai_success.sum()), "solved_fraction": float(babai_success.mean())},
            "exhaustive_box": {"solved_count": sum(bool(g.any()) for g in good_all),
                               "candidate_evaluations_per_instance": [64, 256],
                               "note": "Direct classical enumeration of these tiny boxes reaches an optimum on every instance; this is not a quantum advantage experiment."},
            "validation": {"independent_circuits_checked": len(circuits), "maximum_statevector_probability_difference": max_difference,
                           "global_optima_and_encoded_costs_rechecked": N, "matched_saved_ideal_results": True},
            "actual_circuit_shots_sampled": len(circuits)*shots,
            "qpu_seconds": 0, "limitations": config["limitations"], "elapsed_seconds": time.perf_counter()-before}
        save(output/"summary.json", summary)
        save(output/"per_instance.json", per_instance)
        np.savez_compressed(output/"measurement_samples.npz", **samples)
        manifest.update(status="completed", finished_at=now().isoformat())
        manifest["output_sha256"] = {name: sha256((output/name).read_bytes()).hexdigest() for name in
            ["config.json", "summary.json", "per_instance.json", "measurement_samples.npz"]}
        save(output/"manifest.json", manifest)
        print(json.dumps({"output": str(output), "babai": summary["babai"], "methods": methods}, indent=2), flush=True)
    except Exception as error:
        manifest.update(status="failed", finished_at=now().isoformat(), error=f"{type(error).__name__}: {error}")
        save(output/"manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
