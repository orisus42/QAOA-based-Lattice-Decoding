"""EXP-003: a reproducible, ideal p=1 comparison with strong classical controls."""
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from importlib.metadata import version
from itertools import product
from math import comb
import json
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from scipy.optimize import differential_evolution, minimize
from qiskit_aer import AerSimulator
from threadpoolctl import threadpool_limits
from lattice_qaoa.instances import Parameters, generate_instance, expand_module_matrix, sample_cbd
from lattice_qaoa.reference import embedding, lll_reduce, exact_cvp
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import circuit, state_probabilities, P1Energy
from lattice_qaoa.classical import local_refine, lift_secrets
from lattice_qaoa.sampling import repeated_success, solved_count_interval


def now():
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def main():
    config_path = ROOT/"experiments/configs/EXP-003.json"
    config = json.loads(config_path.read_text())
    started = now()
    run = ROOT/"experiments/runs"/f"EXP-003_{started.strftime('%Y%m%dT%H%M%S%f%z')}"
    run.mkdir(parents=True, exist_ok=False)
    save(run/"config.json", config)
    input_run = ROOT/config["input_run"]
    manifest = {"run_id": run.name, "status": "running", "started_at": started.isoformat(),
                "command": "python experiments/run_module_lwe_qaoa.py", "python": platform.python_version(),
                "packages": {k: version(k) for k in ["numpy", "scipy", "sympy", "qiskit", "qiskit-aer", "threadpoolctl", "matplotlib", "seaborn"]},
                "source_sha256": {}, "input_sha256": {}, "qpu_seconds": 0}
    sources = list((ROOT/"src/lattice_qaoa").glob("*.py"))+list((ROOT/"tests").glob("test_*.py"))+[
        Path(__file__).resolve(), config_path, ROOT/"scripts/plot_module_lwe_qaoa.py", ROOT/"requirements-research.txt"]
    for kind, paths in [("source", sources), ("input", [input_run/"instances.json", input_run/"config.json", input_run/"summary.json"])]:
        for path in paths:
            relative = path.relative_to(ROOT)
            snapshot = run/kind/relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, snapshot)
            manifest[kind+"_sha256"][relative.as_posix()] = sha256(path.read_bytes()).hexdigest()
    save(run/"manifest.json", manifest)
    clock = time.perf_counter()
    try:
        source_rows = json.loads((input_run/"instances.json").read_text())
        params = Parameters(**json.loads((input_run/"config.json").read_text())["parameters"])
        target_matrices = {str(np.array(x["A_expanded"]).tolist()) for x in source_rows}
        training, train_evaluators, train_matrices = [], [], set()
        settings = config["global_training"]
        for seed in range(settings["first_seed"], settings["first_seed"]+settings["count"]):
            x = generate_instance(seed, params)
            A = expand_module_matrix(x.A)
            key = str(A.tolist())
            assert key not in target_matrices and key not in train_matrices
            train_matrices.add(key)
            B, y = embedding(A, x.target.ravel(), params.q)
            reduced, U = lll_reduce(B)
            box = encode_babai_box(reduced, y)
            train_evaluators.append(P1Energy(box.h, box.J))
            training.append({"seed": seed, "A_polynomials": x.A.tolist(), "public_target": x.target.ravel().tolist(),
                             "reduced_basis": reduced.tolist(), "h": box.h.tolist(), "J": box.J.tolist(),
                             "note": "Only public problem data stored/used for fitting; no exact optimum required."})
        fit_clock = time.perf_counter()
        fixed_fit = differential_evolution(lambda a: np.mean([f(a) for f in train_evaluators]),
            config["parameter_bounds"], seed=settings["optimizer_seed"], maxiter=settings["maxiter"],
            popsize=settings["popsize"], tol=1e-7, polish=True)
        fixed_angles = fixed_fit.x.tolist()
        training_fit = {"angles": fixed_angles, "normalized_mean_expected_cost": float(fixed_fit.fun),
                       "objective_evaluations": fixed_fit.nfev, "training_instance_energy_evaluations": fixed_fit.nfev*len(training),
                       "optimizer_success": bool(fixed_fit.success), "optimizer_message": str(fixed_fit.message),
                       "elapsed_seconds": time.perf_counter()-fit_clock}
        save(run/"training.json", {"instances": training, "fit": training_fit})
        print("Global p=1 angles trained on 16 separate public instances:", fixed_angles, flush=True)
        boxes, records, circuits, expected_vectors = [], [], [], []
        exact_probs = {m: [] for m in ["qaoa_fixed", "qaoa_tuned", "uniform_box", "cbd_secret"]}
        largest_energy_error = 0.0
        prepare_clock = time.perf_counter()
        for index, x in enumerate(source_rows):
            A, B, R, U = [np.array(x[k], dtype=np.int64) for k in ["A_expanded", "embedding_basis", "reduced_basis", "basis_transform"]]
            y = np.concatenate([x["public_target"], np.zeros(params.n*params.k, dtype=np.int64)])
            assert np.array_equal(B@U, R)
            box = encode_babai_box(R, y)
            np.testing.assert_array_equal(U@box.center, x["lll_babai_coefficients"])
            boxes.append(box)
            reference = exact_cvp(A, np.array(x["public_target"]), params.q, U@box.center)
            assert reference["cost"] == x["reference"]["cost"]
            assert {tuple(z) for z in reference["minimizers"]} == {tuple(z) for z in x["reference"]["minimizers"]}
            good = box.costs == reference["cost"]
            energy_fn = P1Energy(box.h, box.J)
            tuning_clock = time.perf_counter()
            starts = [fixed_angles]+config["instance_tuning"]["extra_starts"]
            fits = [minimize(energy_fn, start, method="L-BFGS-B", bounds=config["parameter_bounds"],
                            options={"maxiter": config["instance_tuning"]["maxiter"], "ftol": 1e-12}) for start in starts]
            winner = min(fits, key=lambda result: result.fun)
            tuned_angles = winner.x.tolist()
            # Retain the trained parameters if numerical optimization made them worse.
            if energy_fn(fixed_angles) < energy_fn(tuned_angles):
                tuned_angles = fixed_angles.copy()
            tuning_seconds = time.perf_counter()-tuning_clock
            assert energy_fn(tuned_angles) <= energy_fn(fixed_angles)+1e-10
            rec = {"id": x["id"], "seed": x["seed"], "qubits": box.nq, "optimum": reference["cost"],
                   "babai_cost": x["lll_babai_cost"], "babai_optimal": x["lll_babai_optimal"],
                   "babai_secret_recovered": x["lll_babai_recovers_planted_secret"],
                   "box_covers_optimum": bool(good.any()), "optimal_box_bitstrings": int(good.sum()),
                   "box_minimum": int(box.costs.min()), "box_candidates": len(box.costs),
                   "center": box.center.tolist(), "offset": box.offset.tolist(),
                   "angles": {"qaoa_fixed": fixed_angles, "qaoa_tuned": tuned_angles},
                   "tuning": {"energy_evaluations": sum(f.nfev for f in fits)+2, "seconds": tuning_seconds,
                              "local_optimizer_success": [bool(f.success) for f in fits]},
                   "logical_rzz_gates": int(np.count_nonzero(np.abs(box.J)>1e-12)),
                   "probabilities": {}, "expected_squared_costs": {}, "observed": {}, "classical": {}}
            for method, angles in rec["angles"].items():
                distribution = state_probabilities(box.energy, angles)
                error = abs(float(distribution@box.energy)-energy_fn(angles))
                largest_energy_error = max(largest_energy_error, error)
                assert error < 1e-9 and abs(distribution.sum()-1) < 1e-10
                p = float(distribution[good].sum())
                exact_probs[method].append(p)
                rec["probabilities"][method] = p
                rec["expected_squared_costs"][method] = float(distribution@box.costs)
                expected_vectors.append(distribution)
                qc = circuit(box.h, box.J, angles)
                qc.name = f"{method}_{x['id']}"
                qc.save_probabilities(label="ideal_probabilities")
                qc.measure_all()
                circuits.append(qc)
            uniform = float(good.mean())
            opt_secrets = {tuple(z[params.n*params.k:]) for z in reference["minimizers"]}
            cbd_p = sum(np.prod([comb(2*params.eta, params.eta+v)/2**(2*params.eta) for v in secret])
                        for secret in opt_secrets if all(abs(v) <= params.eta for v in secret))
            for method, value in [("uniform_box", uniform), ("cbd_secret", float(cbd_p))]:
                exact_probs[method].append(value)
                rec["probabilities"][method] = value
            for changes in [1, 2]:
                result = local_refine(R, y, box.center, max_changed=changes,
                                      budget=config["classical_refinement_evaluation_cap"])
                result["optimal"] = result["cost"] == reference["cost"]
                original = U@np.array(result["coefficients"])
                result["planted_secret_recovered"] = bool(np.array_equal(original[params.n*params.k:], x["planted_secret"]))
                rec["classical"][f"refine_{changes}"] = result
            records.append(rec)
            if (index+1)%16 == 0:
                print(f"Prepared and cost-checked {index+1}/64 instances", flush=True)
        save(run/"prepared_instances.json", records)
        print("Running 128 ideal 16-qubit circuits, 1024 measurements each...", flush=True)
        sample_clock = time.perf_counter()
        simulator = AerSimulator(method="statevector", max_parallel_threads=2, max_parallel_experiments=1)
        result = simulator.run(circuits, shots=config["shots_per_circuit"], memory=True,
                               seed_simulator=config["measurement_seed"]).result()
        assert result.success
        largest_probability_error = 0.0
        N, shots, d = len(records), config["shots_per_circuit"], params.n*params.k
        samples = {m: np.zeros((N, shots), dtype=np.int64) for m in ["qaoa_fixed", "qaoa_tuned", "uniform_box"]}
        cbd_secrets = np.zeros((N, shots, d), dtype=np.int64)
        rng_uniform = np.random.default_rng(config["uniform_seed"])
        rng_cbd = np.random.default_rng(config["secret_sampling_seed"])
        for i in range(N):
            for j, method in enumerate(["qaoa_fixed", "qaoa_tuned"]):
                k = 2*i+j
                actual = np.asarray(result.data(k)["ideal_probabilities"])
                discrepancy = float(np.max(np.abs(actual-expected_vectors[k])))
                largest_probability_error = max(largest_probability_error, discrepancy)
                assert discrepancy < 1e-10
                samples[method][i] = [int(bits.replace(" ", ""), 2) for bits in result.get_memory(k)]
            samples["uniform_box"][i] = rng_uniform.integers(0, 2**boxes[i].nq, size=shots)
            cbd_secrets[i] = sample_cbd(rng_cbd, params.eta, (shots, d))
        sampling_seconds = time.perf_counter()-sample_clock
        methods = {}
        babai_costs = np.array([r["babai_cost"] for r in records])
        optima = np.array([r["optimum"] for r in records])
        babai_good = babai_costs == optima
        for method in exact_probs:
            costs, candidate_secrets = [], []
            for i, (rec, x, box) in enumerate(zip(records, source_rows, boxes)):
                if method == "cbd_secret":
                    coords, values = lift_secrets(np.array(x["A_expanded"]), np.array(x["public_target"]), params.q, cbd_secrets[i])
                else:
                    values = box.costs[samples[method][i]]
                    coords = box.decode(samples[method][i])@np.array(x["basis_transform"]).T
                costs.append(values)
                candidate_secrets.append(coords[:, d:])
            costs, candidate_secrets = np.stack(costs), np.stack(candidate_secrets)
            hits = costs == optima[:, None]
            p = np.array(exact_probs[method])
            budget_rows = []
            for budget in config["report_budgets"]:
                best_indices = costs[:, :budget].argmin(axis=1)
                best_costs = costs[np.arange(N), best_indices]
                selected_secret = candidate_secrets[np.arange(N), best_indices]
                secret_good = np.all(selected_secret == np.array([x["planted_secret"] for x in source_rows]), axis=1)
                hybrid_costs = np.minimum(best_costs, babai_costs)
                expected = repeated_success(p, budget)
                actual = best_costs == optima
                hybrid_secret = np.where((best_costs < babai_costs)[:, None], selected_secret,
                                         np.array([x["lll_babai_coefficients"][d:] for x in source_rows]))
                hybrid_secret_good = np.all(hybrid_secret == np.array([x["planted_secret"] for x in source_rows]), axis=1)
                budget_rows.append({"shots_or_samples": budget, "expected_solved_fraction": float(expected.mean()),
                    "observed_solved_count": int(actual.sum()), "observed_solved_fraction": float(actual.mean()),
                    "measurement_only_95pct_count_interval": solved_count_interval(expected),
                    "expected_hybrid_solved_fraction": float(np.where(babai_good, 1, expected).mean()),
                    "observed_hybrid_solved_count": int((hybrid_costs == optima).sum()),
                    "babai_failures_recovered": int(actual[~babai_good].sum()), "babai_failures_total": int((~babai_good).sum()),
                    "returned_candidate_secret_recovered_count": int(secret_good.sum()),
                    "hybrid_returned_secret_recovered_count": int(hybrid_secret_good.sum())})
                for i, rec in enumerate(records):
                    rec["observed"].setdefault(method, {})[str(budget)] = {"successful_samples": int(hits[i, :budget].sum()),
                        "best_cost": int(best_costs[i]), "solved": bool(actual[i]), "hybrid_solved": bool(hybrid_costs[i] == optima[i]),
                        "selected_secret": selected_secret[i].tolist(), "selected_secret_is_planted": bool(secret_good[i])}
            methods[method] = {"mean_single_shot_probability": float(p.mean()), "single_shot_probability_min_max": [float(p.min()), float(p.max())],
                "observed_successful_samples": int(hits.sum()), "samples_total": int(hits.size),
                "observed_single_shot_fraction": float(hits.mean()), "by_budget": budget_rows}
        classical = {}
        for key in ["refine_1", "refine_2"]:
            values = [r["classical"][key] for r in records]
            classical[key] = {"solved_count": sum(v["optimal"] for v in values),
                "secret_recovered_count": sum(v["planted_secret_recovered"] for v in values),
                "evaluations_min_median_max": [min(v["evaluations"] for v in values), float(np.median([v["evaluations"] for v in values])), max(v["evaluations"] for v in values)],
                "evaluation_budget_reached_count": sum(v["stopping_reason"] == "evaluation_budget" for v in values)}
        summary = {"experiment_id": "EXP-003", "run_id": run.name, "count": N, "cohort": config["cohort"],
            "parameters": vars(params), "qubits": 16, "depth_p": 1, "binary_candidates_per_instance": 65536,
            "box_covers_optimum_count": sum(r["box_covers_optimum"] for r in records),
            "cbd_support_covers_optimum_count": int(np.count_nonzero(exact_probs["cbd_secret"])),
            "babai": {"solved_count": int(babai_good.sum()), "solved_fraction": float(babai_good.mean()),
                      "planted_secret_recovered_count": sum(r["babai_secret_recovered"] for r in records)},
            "classical_refinement": classical, "methods": methods,
            "exact_reference": {"solved_count": N, "secret_candidates_examined_total": sum(x["reference"]["secret_vectors_examined"] for x in source_rows),
                                "candidate_count_max": max(x["reference"]["secret_vectors_examined"] for x in source_rows)},
            "training": training_fit,
            "tuning_energy_evaluations_min_median_max": [min(r["tuning"]["energy_evaluations"] for r in records),
                float(np.median([r["tuning"]["energy_evaluations"] for r in records])), max(r["tuning"]["energy_evaluations"] for r in records)],
            "logical_rzz_gates_min_max": [min(r["logical_rzz_gates"] for r in records), max(r["logical_rzz_gates"] for r in records)],
            "resources_note": "RZZ counts are logical. A usual decomposition uses two CNOTs per RZZ before routing; no device compilation or end-to-end runtime comparison performed.",
            "validation": {"all_binary_costs_checked": N*65536, "global_cvp_references_rechecked": N,
                "ideal_circuits_independently_checked": len(circuits), "max_probability_discrepancy": largest_probability_error,
                "max_analytic_energy_discrepancy": largest_energy_error},
            "actual_circuit_measurements": len(circuits)*shots,
            "classical_candidate_samples": 2*N*shots, "qpu_seconds": 0,
            "timing": {"preparation_and_sampling_seconds": time.perf_counter()-prepare_clock,
                       "circuit_sampling_and_probability_validation_seconds": sampling_seconds,
                       "total_seconds": time.perf_counter()-clock},
            "limitations": config["limitations"]}
        save(run/"per_instance.json", records)
        save(run/"summary.json", summary)
        np.savez_compressed(run/"samples.npz", **samples, cbd_secrets=cbd_secrets)
        np.savez_compressed(run/"encoded_costs.npz", costs=np.stack([box.costs for box in boxes]))
        manifest.update(status="completed", finished_at=now().isoformat())
        manifest["output_sha256"] = {name: sha256((run/name).read_bytes()).hexdigest() for name in
            ["config.json", "training.json", "prepared_instances.json", "per_instance.json", "summary.json", "samples.npz", "encoded_costs.npz"]}
        save(run/"manifest.json", manifest)
        print(json.dumps({"output": str(run), "babai": summary["babai"], "classical_refinement": classical,
            "methods": {m: {"single_shot": v["mean_single_shot_probability"], "budgets": v["by_budget"]} for m, v in methods.items()},
            "validation": summary["validation"], "timing": summary["timing"]}, indent=2), flush=True)
    except Exception as error:
        manifest.update(status="failed", finished_at=now().isoformat(), error=f"{type(error).__name__}: {error}")
        save(run/"manifest.json", manifest)
        raise


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
