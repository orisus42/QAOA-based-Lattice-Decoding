"""Frozen, budget-limited IBM hardware replication. Never prints credentials."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
import json
import logging
import sys
import platform
import shutil
import argparse
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from threadpoolctl import threadpool_limits
from qiskit import QuantumCircuit, ClassicalRegister, transpile, qpy
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from lattice_qaoa.encoding import encode_babai_box
from lattice_qaoa.qaoa import circuit, state_probabilities
from lattice_qaoa.sparsify import prune_interactions, diagonal_energy
from lattice_qaoa.cached_pruning import select_masks_cached
from lattice_qaoa.incumbent_pruning import incumbent_mask
from lattice_qaoa.device import target_from_snapshot, logical_to_physical_indices

logging.getLogger("qiskit_ibm_runtime").setLevel(logging.ERROR)
SOURCE = ROOT / "experiments/runs/EXP-007_20261020T044846904050+0530"
MODELS = ROOT / "experiments/runs/EXP-021_20261020T175520818890+0530/frozen_models.json"
METHODS = ["full_fixed", "weak75_frozen", "weak75_retrained", "cached24", "incumbent24"]


def save(p, obj):
    p.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def now():
    return datetime.now().astimezone().isoformat()


def safe_usage(service):
    u = service.usage()
    return {k: u[k] for k in ["usage_consumed_seconds", "usage_limit_seconds",
                              "usage_remaining_seconds", "usage_limit_reached", "usage_period"] if k in u}


def assert_free(service):
    instances = service.instances()
    active = service.active_instance()
    matches = [x for x in instances if active in [x.get("crn"), x.get("id")]]
    # If there is only one instance, the current service can only use that one.
    if not matches and len(instances) == 1:
        matches = instances
    assert len(matches) == 1 and matches[0]["plan"] == "open" and matches[0]["pricing_type"] == "free", "Free Open instance not verified"


def capture(backend, run):
    target = backend.target
    props = backend.properties()
    raw = props.to_dict()
    fault = set(props.faulty_qubits())
    good = {q for q in range(target.num_qubits) if q not in fault and target.qubit_properties[q].t1
            and target.qubit_properties[q].t2 and target["measure"][(q,)].error is not None
            and target["measure"][(q,)].error < .15}
    edges = {tuple(sorted(pair)): v.error for pair, v in target["cz"].items()
             if v is not None and v.duration and v.error is not None and v.error < .1 and set(pair) <= good}
    neighbors = {q: set() for q in good}
    for a, b in edges:
        neighbors[a].add(b); neighbors[b].add(a)
    candidates = []
    for start in sorted(good):
        chosen = {start}
        while len(chosen) < 16:
            frontier = set().union(*(neighbors[q] for q in chosen)) - chosen
            if not frontier:
                break
            def key(q):
                adjacent = neighbors[q] & chosen
                return (-len(adjacent), np.mean([edges[tuple(sorted((q, n)))] for n in adjacent]), q)
            chosen.add(min(frontier, key=key))
        if len(chosen) == 16:
            internal = [v for pair, v in edges.items() if set(pair) <= chosen]
            candidates.append({"nodes": sorted(chosen), "edge_count": len(internal),
                               "mean_cz_error": float(np.mean(internal)),
                               "mean_readout_error": float(np.mean([target["measure"][(q,)].error for q in chosen]))})
    selected = min(candidates, key=lambda c: (-c["edge_count"], c["mean_cz_error"], c["mean_readout_error"], c["nodes"]))
    nodes = selected["nodes"]; inverse = {p: i for i, p in enumerate(nodes)}
    gates = []
    for name in ["id", "rz", "sx", "x", "cz", "measure", "reset"]:
        if name not in target.operation_names:
            continue
        for pair, prop in target[name].items():
            if pair is not None and set(pair) <= set(nodes) and prop is not None:
                gates.append({"name": name, "qubits": [inverse[q] for q in pair], "physical_qubits": list(pair),
                              "duration": prop.duration, "error": prop.error})
    qs = []
    for q in nodes:
        values = {v["name"]: v["value"] for v in raw["qubits"][q]}
        ro = target["measure"][(q,)].error
        qs.append({"physical": q, "t1": target.qubit_properties[q].t1, "t2": target.qubit_properties[q].t2,
                   "frequency": target.qubit_properties[q].frequency,
                   "readout_p1_given0": values.get("prob_meas1_prep0", ro),
                   "readout_p0_given1": values.get("prob_meas0_prep1", ro)})
    cfg = backend.configuration()
    snapshot = {"backend": backend.name, "fetched_at": now(), "calibration_last_update": props.last_update_date.isoformat(),
                "physical_qubits": nodes, "num_qubits": 16, "dt": backend.dt, "qubits": qs, "gates": gates,
                "timing_constraints": {k: getattr(target, k) for k in ["granularity", "min_length", "pulse_alignment", "acquire_alignment"]},
                "default_rep_delay": cfg.default_rep_delay, "rep_delay_range": cfg.rep_delay_range, "selection": selected}
    save(run / "snapshot.json", snapshot)
    save(run / "backend_properties.json", raw)
    save(run / "patch_candidates.json", candidates)
    return snapshot


def prepare():
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
    run = ROOT / "experiments/runs" / ("EXP-028_" + stamp)
    run.mkdir(exist_ok=False)
    for d in ["circuits", "input", "source", "counts"]:
        (run / d).mkdir()
    save(run / "status.json", {"status": "preflight", "at": now(), "qpu_jobs_submitted": 0})
    print("RUN", run, flush=True)
    service = QiskitRuntimeService()
    assert_free(service)
    usage = safe_usage(service)
    assert usage["usage_remaining_seconds"] >= 480, "Less than required conservative free budget"
    save(run / "usage_before.json", usage)
    available = []
    for b in service.backends(operational=True, simulator=False):
        st = b.status()
        available.append({"name": b.name, "pending_jobs": st.pending_jobs, "status": st.status_msg})
    active = [v for v in available if v["status"] == "active"]
    chosen = min(active, key=lambda v: (v["pending_jobs"], v["name"]))
    backend = service.backend(chosen["name"])
    save(run / "backend_selection.json", {"at": now(), "candidates": available,
         "rule": "Fewest pending jobs among active accessible devices; name breaks ties. No quality outcomes used.", "selected": chosen})
    print("BACKEND", chosen, "FREE", usage["usage_remaining_seconds"], flush=True)
    snap = capture(backend, run)
    target = target_from_snapshot(snap)
    cfg = json.loads((SOURCE / "config.json").read_text())
    instances = json.loads((SOURCE / "instances.json").read_text())
    earlier = json.loads((SOURCE / "per_instance.json").read_text())
    angles = json.loads(MODELS.read_text())["angles"]
    rng = np.random.default_rng(20261020280)
    failures = [i for i, v in enumerate(earlier) if v["babai_cost"] > v["optimum_cost"]]
    selected_failures = sorted(rng.choice(failures, 6, replace=False).tolist())
    sentinels = sorted(rng.choice([i for i in range(len(instances)) if i not in selected_failures], 2, replace=False).tolist())
    ids = selected_failures + sentinels
    protocol = {"experiment_id": "EXP-028", "frozen_at": now(), "source_run": str(SOURCE.relative_to(ROOT)),
                "selection_seed": 20261020280, "order_seed": 20261020281, "routing_seed": 20261020282,
                "source_indices": ids, "conditional_failure_indices": selected_failures, "sentinel_indices": sentinels,
                "cohort": "Six seeded random Babai-failure examples plus two random remaining sentinels from the earlier 512-case cohort. Hardware replication, not a new population validation. No selection by QAOA/pruning quality.",
                "methods": METHODS, "blocks": 2, "shots_per_block": 4096, "shots_per_case_method": 8192,
                "primary": "75% unchanged-angle pruning minus full QAOA: mean per-shot probability of strict improvement over Babai across the six fixed failure cases. Report individual Wilson intervals and fixed-cohort shot-bootstrap difference interval; no population generalization from this selected subset.",
                "secondary": "Retrained75%, cached24 versus incumbent24, optimum and at-most-Babai masses, original cost distribution, observed best-of8192, circuit resources. Cached/cheap hardware comparison matches24 logical interactions, NOT native gates.",
                "mitigation": "None; no dynamical decoupling, no gate or measurement twirling; standard initialization; default backend repetition delay.",
                "project_budget_seconds": 600, "reserve_seconds": 120, "max_job_seconds": 360,
                "limitations": ["Small same-family conditional replication; no secure ML-KEM attack or quantum advantage.",
                                "Two ordered blocks permit a drift check but do not establish independent hardware replications.",
                                "All methods and angles fixed before hardware data; no adaptive favorable-case filtering.",
                                "Finite-shot estimates at rare optimum events may be imprecise; zero counts do not mean zero probability.",
                                "Classical retained Babai remains available; QAOA-only probability is distinct from hybrid success.",
                                "Cached selection changes implementation time, not quantum behavior relative to its uncached equivalent."]}
    save(run / "frozen_protocol.json", protocol)
    source_paths = list((ROOT / "src/lattice_qaoa").glob("*.py")) + [Path(__file__)]
    manifest = {"created_at": now(), "python": platform.python_version(),
                "packages": {k: version(k) for k in ["qiskit", "qiskit-ibm-runtime", "qiskit-aer", "numpy", "scipy", "seaborn", "matplotlib"]},
                "source_sha256": {}, "input_sha256": {}}
    for p in source_paths:
        dest = run / "source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
        manifest["source_sha256"][str(p.relative_to(ROOT))] = sha256(p.read_bytes()).hexdigest()
    for p in [SOURCE / n for n in ["config.json", "instances.json", "per_instance.json", "manifest.json"]] + [MODELS]:
        dest = run / "input" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
        manifest["input_sha256"][str(p.relative_to(ROOT))] = sha256(p.read_bytes()).hexdigest()
    rows = []
    ideal_sim = AerSimulator(method="statevector", max_parallel_threads=2)
    maximum_error = 0.
    for source_index in ids:
        x, old = instances[source_index], earlier[source_index]
        box = encode_babai_box(np.array(x["reduced_basis"]), np.r_[x["public_target"], np.zeros(4, dtype=int)])
        cache, _ = select_masks_cached(box.h, box.J, [24], cfg["methods"]["full_fixed"]["angles"])
        for method in METHODS:
            if method == "cached24":
                J, ang = cache[24]["J"], angles["residual"]
            elif method == "incumbent24":
                value = incumbent_mask(box.J, np.tile([1, -1], 8), 24)
                J = value[0] if isinstance(value, tuple) else value
                ang = angles["incumbent"]
            else:
                spec = cfg["methods"][method]
                J, _ = prune_interactions(box.J, spec["fraction"]); ang = spec["angles"]
            logical = circuit(box.h, J, ang); logical.barrier()
            routed = transpile(logical, target=target, initial_layout=list(range(16)),
                               optimization_level=3, seed_transpiler=protocol["routing_seed"], scheduling_method="alap")
            permutation = routed.layout.final_index_layout()
            expected = state_probabilities(diagonal_energy(box.h, J), ang)
            check = routed.copy(); check.save_probabilities()
            physical = np.asarray(ideal_sim.run(check).result().data()["probabilities"])
            reordered = physical[logical_to_physical_indices(permutation)]
            err = float(np.max(abs(expected - reordered)))
            assert err < 1e-10
            maximum_error = max(maximum_error, err)
            # Explicitly measure logical bit j into classical bit j after routing.
            measured = routed.copy(); measured.add_register(ClassicalRegister(16, "meas"))
            for j, position in enumerate(permutation):
                measured.measure(position, j)
            physical_qc = QuantumCircuit(backend.num_qubits, name=x["id"] + "_" + method)
            physical_qc.add_register(ClassicalRegister(16, "meas"))
            physical_qc.global_phase = measured.global_phase
            for inst in measured.data:
                qs = [snap["physical_qubits"][measured.find_bit(q).index] for q in inst.qubits]
                cs = [measured.find_bit(c).index for c in inst.clbits]
                if inst.operation.name not in ["barrier", "delay"]:
                    assert backend.target.instruction_supported(operation_name=inst.operation.name, qargs=tuple(qs))
                physical_qc.append(inst.operation, qs, cs)
            for filename, qc in [(x["id"] + "_" + method + "_compact.qpy", measured),
                                 (x["id"] + "_" + method + ".qpy", physical_qc)]:
                with (run / "circuits" / filename).open("wb") as f: qpy.dump(qc, f)
            np.savez_compressed(run / "circuits" / (x["id"] + "_" + method + ".npz"),
                                ideal_probability=expected, costs=box.costs, h=box.h, J=J,
                                angles=ang, final_permutation=permutation)
            rows.append({"id": x["id"], "source_index": source_index, "stratum": "failure" if source_index in selected_failures else "sentinel",
                         "method": method, "angles": ang, "babai_cost": old["babai_cost"], "optimum_cost": old["optimum_cost"],
                         "classical_refinement": old["classical_refinement"], "logical_zz": int(np.count_nonzero(J)),
                         "native_cz": routed.count_ops().get("cz", 0), "depth": routed.depth(),
                         "unitary_duration_seconds": routed.estimate_duration(target),
                         "ideal_p_improve": float(expected[box.costs < old["babai_cost"]].sum()),
                         "ideal_p_at_most_babai": float(expected[box.costs <= old["babai_cost"]].sum()),
                         "ideal_p_optimum": float(expected[box.costs == old["optimum_cost"]].sum()),
                         "mapping_max_error": err, "qpy": "circuits/" + x["id"] + "_" + method + ".qpy"})
        print("PREPARED", x["id"], len(rows), "circuits", flush=True)
    # Independent measurement-order check using sampled ideal compact circuits.
    for row in rows[:5]:
        with (run / "circuits" / (row["id"] + "_" + row["method"] + "_compact.qpy")).open("rb") as f: qc = qpy.load(f)[0]
        counts = ideal_sim.run(qc, shots=4096, seed_simulator=20261020284).result().get_counts()
        data = np.load(run / "circuits" / (row["id"] + "_" + row["method"] + ".npz"))
        observed = sum(n for bits, n in counts.items() if data["costs"][int(bits, 2)] <= row["babai_cost"]) / 4096
        expected = row["ideal_p_at_most_babai"]
        assert abs(observed - expected) < max(.005, 6 * np.sqrt(expected*(1-expected)/4096))
    order_rng = np.random.default_rng(protocol["order_seed"])
    pubs = []
    for block in range(2):
        for i in order_rng.permutation(len(rows)):
            pubs.append({"row_index": int(i), "block": block, "shots": 4096})
    rep = snap["default_rep_delay"]
    readout = max(g["duration"] or 0 for g in snap["gates"] if g["name"] == "measure")
    reset = max([g["duration"] or 0 for g in snap["gates"] if g["name"] == "reset"] + [0])
    execution = sum(p["shots"]*(rep + readout + reset + rows[p["row_index"]]["unitary_duration_seconds"]) for p in pubs)
    conservative = execution + 2 * len(pubs) + 10
    assert conservative < protocol["max_job_seconds"] <= usage["usage_remaining_seconds"]-120
    save(run / "resource_estimate.json", {"scheduled_shot_seconds": execution, "conservative_seconds_with_2s_per_pub_plus10": conservative,
         "pubs": len(pubs), "shots": sum(p["shots"] for p in pubs), "rep_delay": rep, "readout_seconds": readout,
         "reset_seconds": reset, "job_cap_seconds": protocol["max_job_seconds"], "reserve_seconds": 120,
         "note": "Local estimate, not guaranteed usage; hard job cap and actual metrics also recorded."})
    save(run / "rows.json", rows); save(run / "pub_order.json", pubs)
    save(run / "validation.json", {"ideal_circuits_verified": len(rows), "max_probability_error": maximum_error,
         "sampled_measurement_order_checks": 5, "full_backend_isa_verified": True, "free_open_plan_verified": True})
    manifest["frozen_files_sha256"] = {str(p.relative_to(run)): sha256(p.read_bytes()).hexdigest()
         for p in run.rglob("*") if p.is_file() and p.name not in ["status.json", "manifest.json"]}
    save(run / "manifest.json", manifest)
    save(run / "status.json", {"status": "ready_to_submit", "at": now(), "qpu_jobs_submitted": 0})
    (ROOT / "project/current_hardware_run.txt").write_text(str(run), encoding="utf-8")
    print("READY", run, "estimate", conservative, flush=True)


def submit(run):
    assert not (run / "submission_attempt.json").exists(), "Submission already attempted: inspect jobs; never blind retry"
    assert json.loads((run / "status.json").read_text())["status"] == "ready_to_submit"
    manifest = json.loads((run / "manifest.json").read_text())
    for name, digest in manifest["frozen_files_sha256"].items():
        assert sha256((run / name).read_bytes()).hexdigest() == digest
    service = QiskitRuntimeService(); assert_free(service)
    usage = safe_usage(service); save(run / "usage_at_submission.json", usage)
    protocol = json.loads((run / "frozen_protocol.json").read_text())
    assert usage["usage_remaining_seconds"] >= protocol["max_job_seconds"] + 120
    snap = json.loads((run / "snapshot.json").read_text())
    backend = service.backend(snap["backend"])
    rows = json.loads((run / "rows.json").read_text())
    order = json.loads((run / "pub_order.json").read_text())
    circuits = []
    for row in rows:
        with (run / row["qpy"]).open("rb") as f: circuits.append(qpy.load(f)[0])
    sampler = SamplerV2(mode=backend, options={"max_execution_time": protocol["max_job_seconds"],
        "default_shots": 4096, "execution": {"init_qubits": True, "rep_delay": snap["default_rep_delay"]},
        "dynamical_decoupling": {"enable": False}, "twirling": {"enable_gates": False, "enable_measure": False},
        "environment": {"job_tags": ["cse-techday", "EXP-028", run.name]}})
    save(run / "submission_attempt.json", {"at": now(), "tag": run.name, "max_execution_time": protocol["max_job_seconds"],
        "note": "If outcome unknown, query service.jobs(job_tags=[tag]) before any retry."})
    job = sampler.run([(circuits[p["row_index"]], None, p["shots"]) for p in order])
    record = {"job_id": job.job_id(), "backend": backend.name, "submitted_at": now(), "status": job.status(),
              "max_execution_time": protocol["max_job_seconds"]}
    save(run / "job.json", record)
    save(run / "status.json", {"status": "submitted", "at": now(), "qpu_jobs_submitted": 1, "job_id": job.job_id()})
    print(json.dumps(record), flush=True)


def collect(run):
    info = json.loads((run / "job.json").read_text())
    service = QiskitRuntimeService()
    job = service.job(info["job_id"])
    st = job.status()
    try: metrics = job.metrics()
    except Exception: metrics = {}
    save(run / ("poll_" + datetime.now().astimezone().strftime("%H%M%S") + ".json"), {"at": now(), "status": st, "metrics": metrics})
    print("STATUS", st, flush=True)
    if st != "DONE":
        return
    if (run / "hardware_complete.json").exists():
        print("Already collected"); return
    result = job.result()
    order = json.loads((run / "pub_order.json").read_text())
    assert len(result) == len(order)
    for i, pub in enumerate(result):
        bitarray = pub.data.meas
        counts = bitarray.get_counts()
        assert sum(counts.values()) == order[i]["shots"]
        save(run / "counts" / f"pub_{i:03d}.json", {"pub_index": i, **order[i], "counts": counts, "metadata": pub.metadata})
        np.save(run / "counts" / f"pub_{i:03d}_ordered.npy",
                np.array([int(b, 2) for b in bitarray.get_bitstrings()], dtype=np.uint16))
    save(run / "result_metadata.json", result.metadata)
    save(run / "job_metrics.json", job.metrics())
    save(run / "usage_after.json", safe_usage(service))
    save(run / "hardware_complete.json", {"at": now(), "pubs": len(result),
         "shots": sum(p["shots"] for p in order), "job_id": info["job_id"], "backend": info["backend"]})
    save(run / "status.json", {"status": "hardware_collected", "at": now(), "qpu_jobs_submitted": 1})
    print("COLLECTED", len(result), "pubs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "submit", "collect"])
    parser.add_argument("run", nargs="?")
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        if args.action == "prepare": prepare()
        else:
            run = Path(args.run) if args.run else Path((ROOT / "project/current_hardware_run.txt").read_text())
            {"submit": submit, "collect": collect}[args.action](run)
