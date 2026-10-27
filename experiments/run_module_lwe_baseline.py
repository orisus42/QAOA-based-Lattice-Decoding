"""Run EXP-001 locally; preserve a separate manifest and snapshots for each run."""

from collections import Counter
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from lattice_qaoa.instances import Parameters, generate_instance, expand_module_matrix
from lattice_qaoa.reference import embedding, lll_reduce, babai, squared_cost, exact_cvp


def save_json(path, content):
    path.write_text(json.dumps(content, indent=2)+"\n", encoding="utf-8")


def main():
    config_path = ROOT / "experiments/configs/EXP-001.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    started = datetime.now().astimezone()
    run_id = f"EXP-001_{started.strftime('%Y%m%dT%H%M%S%f%z')}"
    output = ROOT / "experiments/runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", config)
    source_files = sorted((ROOT/"src/lattice_qaoa").glob("*.py")) + [
        Path(__file__).resolve(), config_path, ROOT/"tests/test_module_lwe.py",
        ROOT/"requirements-research.txt", ROOT/"pytest.ini",
    ]
    source_manifest = {}
    for source in source_files:
        relative = source.relative_to(ROOT)
        snapshot = output / "source" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, snapshot)
        source_manifest[relative.as_posix()] = sha256(source.read_bytes()).hexdigest()
    manifest = {
        "run_id": run_id,
        "status": "running",
        "started_at": started.isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ["numpy", "sympy"]},
        "command": "python experiments/run_module_lwe_baseline.py",
        "source_sha256": source_manifest,
        "qpu_seconds": 0,
    }
    save_json(output / "manifest.json", manifest)
    before = time.perf_counter()
    records = []
    p = Parameters(**config["parameters"])
    try:
        for index in range(config["count"]):
            instance = generate_instance(config["first_seed"]+index, p)
            A, t = expand_module_matrix(instance.A), instance.target.ravel()
            s, e = instance.secret.ravel(), instance.error.ravel()
            B, y = embedding(A, t, p.q)
            reduced, U = lll_reduce(B)
            original_z = babai(B, y)
            reduced_z = babai(reduced, y)
            mapped_z = U @ reduced_z
            np.testing.assert_array_equal(B @ mapped_z, reduced @ reduced_z)
            original_cost = squared_cost(B, y, original_z)
            reduced_cost = squared_cost(B, y, mapped_z)
            incumbent = mapped_z if reduced_cost <= original_cost else original_z
            reference = exact_cvp(A, t, p.q, incumbent, config["exact_reference_candidate_cap"])
            numerator = t-e-A@s
            assert np.all(numerator % p.q == 0)
            planted_z = np.concatenate([numerator//p.q, s])
            np.testing.assert_array_equal(B @ planted_z-y, np.concatenate([-e, s]))
            assert np.array_equal((A@s+e) % p.q, t)
            planted_cost = squared_cost(B, y, planted_z)
            d = len(t)
            records.append({
                "id": f"MLWE-{index:03d}", "seed": instance.seed,
                "A_polynomials": instance.A.tolist(), "A_expanded": A.tolist(),
                "public_target": t.tolist(), "planted_secret": s.tolist(), "planted_error": e.tolist(),
                "embedding_basis": B.tolist(), "reduced_basis": reduced.tolist(), "basis_transform": U.tolist(),
                "original_babai_coefficients": original_z.tolist(), "lll_babai_coefficients": mapped_z.tolist(),
                "original_babai_cost": original_cost, "lll_babai_cost": reduced_cost,
                "planted_cost": planted_cost, "reference": reference,
                "original_babai_optimal": original_cost == reference["cost"],
                "lll_babai_optimal": reduced_cost == reference["cost"],
                "original_babai_recovers_planted_secret": bool(np.array_equal(original_z[d:], s)),
                "lll_babai_recovers_planted_secret": bool(np.array_equal(mapped_z[d:], s)),
                "planted_is_an_optimum": planted_cost == reference["cost"],
                "planted_secret_in_optimum_set": any(x[d:] == s.tolist() for x in reference["minimizers"]),
                "unique_cvp": len(reference["minimizers"]) == 1,
                "nonzero_error": bool(np.any(e)),
            })
        save_json(output/"instances.json", records)
        counters = ["original_babai_optimal", "lll_babai_optimal", "original_babai_recovers_planted_secret",
                    "lll_babai_recovers_planted_secret", "planted_is_an_optimum", "planted_secret_in_optimum_set",
                    "unique_cvp", "nonzero_error"]
        summary = {
            "run_id": run_id, "count": len(records), "parameters": config["parameters"],
            "expanded_secret_dimension": p.n*p.k, "embedding_dimension": 2*p.n*p.k,
            "counts": {key: sum(int(r[key]) for r in records) for key in counters},
            "mean_squared_cost": {key: float(np.mean([r[key] for r in records])) for key in
                                  ["original_babai_cost", "lll_babai_cost", "planted_cost"]},
            "mean_exact_squared_cost": float(np.mean([r["reference"]["cost"] for r in records])),
            "optimum_multiplicity_histogram": dict(sorted(Counter(len(r["reference"]["minimizers"]) for r in records).items())),
            "candidate_count_total": sum(r["reference"]["secret_vectors_examined"] for r in records),
            "candidate_count_max": max(r["reference"]["secret_vectors_examined"] for r in records),
            "naive_two_bits_per_embedding_coefficient_qubits": 4*p.n*p.k,
            "encoding_status": "Only a resource illustration; no bit bounds or coverage established yet.",
            "limitations": config["notes"] + ["One tiny parameter set and 64 development instances; no hardware or QAOA experiment."],
            "elapsed_seconds": time.perf_counter()-before,
        }
        save_json(output/"summary.json", summary)
        manifest.update(status="completed", finished_at=datetime.now().astimezone().isoformat())
        manifest["output_sha256"] = {name: sha256((output/name).read_bytes()).hexdigest()
                                     for name in ["config.json", "instances.json", "summary.json"]}
        save_json(output/"manifest.json", manifest)
        print(json.dumps({"output": str(output), "summary": summary}, indent=2))
    except Exception as error:
        manifest.update(status="failed", finished_at=datetime.now().astimezone().isoformat(), error=f"{type(error).__name__}: {error}")
        save_json(output/"manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
